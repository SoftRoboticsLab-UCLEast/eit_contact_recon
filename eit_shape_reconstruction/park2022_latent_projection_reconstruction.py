"""
park2022_latent_projection_reconstruction.py

Adaptation of the latent-projection EIT reconstruction
strategy used by Park et al., Science Robotics (2022), "A biomimetic
elastomeric robot skin using electrical impedance and acoustic tomography
for tactile sensing."

Core pipeline:
    voltage vector
        -> voltage denoising autoencoder encoder
        -> voltage latent z_v
        -> supervised latent projection P(z_v)
        -> shape/deformation latent z_s
        -> convolutional autoencoder decoder
        -> reconstructed contact/deformation map

Training is performed in three stages:
    1) Train voltage denoising autoencoder.
    2) Train shape/deformation-map denoising convolutional autoencoder.
    3) Freeze both autoencoders and train the latent projection network
       from voltage latent space to shape latent space.

At inference:
    shape_hat = ShapeDecoder(Projector(VoltageEncoder(voltage)))

Paper-supported details:
    - ELU activations.
    - Gaussian noise during autoencoder training.
    - Voltage AE is fully connected.
    - Shape AE is convolutional.
    - Shape decoder uses nearest-neighbour upsampling + convolution
      rather than transposed convolution.
    - Projection network has two hidden layers.
    - Original paper: 94-D voltage -> 84-D latent;
      48x48 map -> 128-D latent;
      projector: 84 -> 256 -> 256 -> 128;
      dropout = 0.03 in voltage AE.

Adaptation defaults for the present EIT dataset:
    - voltage dimension = 208
    - output map = 64 x 64
    - voltage latent = 84
    - shape latent = 128

Unknown implementation details (exact layer widths, noise std, etc.) are
reasonable assumptions and are exposed as command-line parameters.

Expected .npz dataset:
    voltages : float array [N, 208]
    shapes   : float array [N, 64, 64] or [N, 1, 64, 64]

Example:
    python park2022_latent_projection_reconstruction.py \
        --data data/contact_shapes.npz \
        --output-dir runs/park2022

The script saves:
    best_voltage_ae.pt
    best_shape_ae.pt
    best_projector.pt
    test_metrics.txt
    test_predictions.npz
"""

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class EITShapeDataset(Dataset):
    def __init__(self, voltages: np.ndarray, shapes: np.ndarray):
        voltages = np.asarray(voltages, dtype=np.float32)
        shapes = np.asarray(shapes, dtype=np.float32)

        if voltages.ndim != 2:
            raise ValueError(f"voltages must have shape [N,D], got {voltages.shape}")

        if shapes.ndim == 3:
            shapes = shapes[:, None, :, :]
        if shapes.ndim != 4 or shapes.shape[1] != 1:
            raise ValueError(
                f"shapes must be [N,H,W] or [N,1,H,W], got {shapes.shape}"
            )

        if len(voltages) != len(shapes):
            raise ValueError("voltages and shapes must contain the same number of samples")

        self.voltages = torch.from_numpy(voltages)
        self.shapes = torch.from_numpy(shapes)

    def __len__(self):
        return len(self.voltages)

    def __getitem__(self, idx):
        return self.voltages[idx], self.shapes[idx]


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "voltages" not in data or "shapes" not in data:
        raise KeyError(
            "Dataset must contain arrays named 'voltages' and 'shapes'. "
            f"Found keys: {list(data.keys())}"
        )
    return data["voltages"], data["shapes"]


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

@dataclass
class VoltageNormalizer:
    mean: np.ndarray = None
    std: np.ndarray = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = x.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray):
        return ((x - self.mean) / self.std).astype(np.float32)

    def save(self, path: Path):
        np.savez(path, mean=self.mean, std=self.std)


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class VoltageAutoencoder(nn.Module):
    """
    Fully-connected denoising autoencoder for EIT voltage vectors.

    The exact widths are not specified in the paper, so 256/128 are used
    here as reasonable assumptions. Latent dimension defaults to 84, as
    in Park et al.
    """

    def __init__(self, input_dim=208, latent_dim=84, dropout=0.03):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(inplace=True),

            nn.Linear(128, 256),
            nn.ELU(inplace=True),

            nn.Linear(256, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class ShapeAutoencoder(nn.Module):
    """
    Convolutional denoising autoencoder for contact/deformation maps.

    Decoder follows the paper's stated design choice:
        nearest-neighbour upsampling + convolution
    instead of ConvTranspose2d.

    For 64x64 input:
        64 -> 32 -> 16 -> 8
    then flattened into a 128-D latent vector.
    """

    def __init__(self, image_size=64, latent_dim=128):
        super().__init__()

        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8")

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.base_size = image_size // 8

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2),
        )

        flat_dim = 64 * self.base_size * self.base_size
        self.to_latent = nn.Linear(flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat_dim)

        self.conv_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),

            # Use sigmoid for normalized binary/continuous contact maps.
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.conv_encoder(x)
        h = h.flatten(1)
        return self.to_latent(h)

    def decode(self, z):
        h = self.from_latent(z)
        h = h.view(-1, 64, self.base_size, self.base_size)
        return self.conv_decoder(h)

    def forward(self, x):
        return self.decode(self.encode(x))


class LatentProjector(nn.Module):
    """
    Paper-supported latent projection:
        84 -> 256 -> 256 -> 128
    with ELU activation.
    """

    def __init__(self, voltage_latent=84, shape_latent=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(voltage_latent, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, shape_latent),
        )

    def forward(self, z):
        return self.net(z)


class ParkStyleReconstructor(nn.Module):
    """Inference graph: voltage encoder -> latent projector -> shape decoder."""

    def __init__(self, voltage_ae, shape_ae, projector):
        super().__init__()
        self.voltage_encoder = voltage_ae.encoder
        self.projector = projector
        self.shape_decoder = nn.Sequential(
            shape_ae.from_latent,
            _ReshapeForShapeDecoder(shape_ae.base_size),
            shape_ae.conv_decoder,
        )

    def forward(self, voltage):
        z_v = self.voltage_encoder(voltage)
        z_shape_hat = self.projector(z_v)
        return self.shape_decoder(z_shape_hat)


class _ReshapeForShapeDecoder(nn.Module):
    def __init__(self, base_size):
        super().__init__()
        self.base_size = base_size

    def forward(self, h):
        return h.view(-1, 64, self.base_size, self.base_size)


# ---------------------------------------------------------------------
# Noise augmentation
# ---------------------------------------------------------------------

def add_gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x
    return x + std * torch.randn_like(x)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

@torch.no_grad()
def binary_metrics(pred, target, threshold=0.5, eps=1e-8):
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()

    dims = tuple(range(1, pred.ndim))
    intersection = (pred_bin * target_bin).sum(dim=dims)
    pred_sum = pred_bin.sum(dim=dims)
    target_sum = target_bin.sum(dim=dims)
    union = pred_sum + target_sum - intersection

    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)

    mse = ((pred - target) ** 2).mean(dim=dims)
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))

    return iou, dice, psnr


# ---------------------------------------------------------------------
# Generic training helpers
# ---------------------------------------------------------------------

def run_autoencoder_epoch(
    model,
    loader,
    optimizer,
    device,
    noise_std,
    mode: str,
):
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.MSELoss()
    total_loss = 0.0
    n = 0

    for voltages, shapes in loader:
        voltages = voltages.to(device)
        shapes = shapes.to(device)

        clean = voltages if mode == "voltage" else shapes
        noisy = add_gaussian_noise(clean, noise_std)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            recon = model(noisy)
            loss = criterion(recon, clean)

            if is_train:
                loss.backward()
                optimizer.step()

        batch = clean.size(0)
        total_loss += loss.item() * batch
        n += batch

    return total_loss / max(n, 1)


def train_autoencoder(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    noise_std,
    mode,
    checkpoint_path,
    patience=25,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_autoencoder_epoch(
            model, train_loader, optimizer, device, noise_std, mode
        )
        val_loss = run_autoencoder_epoch(
            model, val_loader, None, device, noise_std=0.0, mode=mode
        )

        print(
            f"[{mode} AE] epoch {epoch:03d}/{epochs} "
            f"train={train_loss:.6f} val={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"[{mode} AE] early stopping")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def run_projector_epoch(
    voltage_ae,
    shape_ae,
    projector,
    loader,
    optimizer,
    device,
):
    is_train = optimizer is not None

    voltage_ae.eval()
    shape_ae.eval()
    projector.train(is_train)

    criterion = nn.MSELoss()
    total_loss = 0.0
    n = 0

    for voltages, shapes in loader:
        voltages = voltages.to(device)
        shapes = shapes.to(device)

        # Encoders remain frozen during projection training.
        with torch.no_grad():
            z_v = voltage_ae.encode(voltages)
            z_shape = shape_ae.encode(shapes)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            z_shape_hat = projector(z_v)
            loss = criterion(z_shape_hat, z_shape)

            if is_train:
                loss.backward()
                optimizer.step()

        batch = voltages.size(0)
        total_loss += loss.item() * batch
        n += batch

    return total_loss / max(n, 1)


def train_projector(
    voltage_ae,
    shape_ae,
    projector,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    checkpoint_path,
    patience=25,
):
    for p in voltage_ae.parameters():
        p.requires_grad_(False)
    for p in shape_ae.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(projector.parameters(), lr=lr)

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_projector_epoch(
            voltage_ae, shape_ae, projector, train_loader, optimizer, device
        )
        val_loss = run_projector_epoch(
            voltage_ae, shape_ae, projector, val_loader, None, device
        )

        print(
            f"[projector] epoch {epoch:03d}/{epochs} "
            f"train={train_loss:.6f} val={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(projector.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("[projector] early stopping")
            break

    projector.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return projector


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(voltage_ae, shape_ae, projector, loader, device, threshold=0.5):
    voltage_ae.eval()
    shape_ae.eval()
    projector.eval()

    all_iou, all_dice, all_psnr = [], [], []
    all_pred, all_gt = [], []

    for voltages, shapes in loader:
        voltages = voltages.to(device)
        shapes = shapes.to(device)

        z_v = voltage_ae.encode(voltages)
        z_shape_hat = projector(z_v)
        pred = shape_ae.decode(z_shape_hat)

        iou, dice, psnr = binary_metrics(pred, shapes, threshold)

        all_iou.append(iou.cpu())
        all_dice.append(dice.cpu())
        all_psnr.append(psnr.cpu())
        all_pred.append(pred.cpu())
        all_gt.append(shapes.cpu())

    iou = torch.cat(all_iou)
    dice = torch.cat(all_dice)
    psnr = torch.cat(all_psnr)
    preds = torch.cat(all_pred).numpy()
    gts = torch.cat(all_gt).numpy()

    metrics = {
        "IoU_mean": float(iou.mean()),
        "IoU_std": float(iou.std(unbiased=False)),
        "Dice_mean": float(dice.mean()),
        "Dice_std": float(dice.std(unbiased=False)),
        "PSNR_mean_dB": float(psnr.mean()),
        "PSNR_std_dB": float(psnr.std(unbiased=False)),
    }
    return metrics, preds, gts


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data", type=str, required=True,
                   help="NPZ with arrays 'voltages' and 'shapes'")
    p.add_argument("--output-dir", type=str, default="runs/park2022")

    p.add_argument("--voltage-dim", type=int, default=208)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--voltage-latent", type=int, default=84)
    p.add_argument("--shape-latent", type=int, default=128)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs-voltage", type=int, default=150)
    p.add_argument("--epochs-shape", type=int, default=150)
    p.add_argument("--epochs-projector", type=int, default=150)

    p.add_argument("--lr-voltage", type=float, default=2e-4)
    p.add_argument("--lr-shape", type=float, default=2e-4)
    p.add_argument("--lr-projector", type=float, default=2e-4)

    # Assumed values; tune if needed.
    p.add_argument("--voltage-noise-std", type=float, default=0.02,
                   help="Gaussian std after voltage standardization")
    p.add_argument("--shape-noise-std", type=float, default=0.03,
                   help="Gaussian std for normalized shape maps")
    p.add_argument("--dropout", type=float, default=0.03)

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    voltages_raw, shapes = load_npz(args.data)

    if voltages_raw.shape[1] != args.voltage_dim:
        raise ValueError(
            f"Dataset voltage dimension is {voltages_raw.shape[1]}, "
            f"but --voltage-dim={args.voltage_dim}"
        )

    if shapes.ndim == 3:
        H, W = shapes.shape[1:]
    else:
        H, W = shapes.shape[-2:]

    if H != args.image_size or W != args.image_size:
        raise ValueError(
            f"Dataset shape maps are {H}x{W}, "
            f"but --image-size={args.image_size}"
        )

    # ---------------------------------------------------------------
    # Split indices first, then fit normalization on TRAIN only.
    # ---------------------------------------------------------------
    N = len(voltages_raw)
    n_train = int(args.train_frac * N)
    n_val = int(args.val_frac * N)
    n_test = N - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("train/val/test fractions yield an empty split")

    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=generator).numpy()

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    normalizer = VoltageNormalizer().fit(voltages_raw[train_idx])
    voltages = normalizer.transform(voltages_raw)
    normalizer.save(out / "voltage_normalization.npz")

    # Normalize shape maps to [0,1] if needed.
    shapes = np.asarray(shapes, dtype=np.float32)
    smin, smax = float(shapes.min()), float(shapes.max())
    if smin < 0.0 or smax > 1.0:
        print(
            f"Shape values range from {smin:.4g} to {smax:.4g}; "
            "applying global min-max normalization to [0,1]."
        )
        denom = max(smax - smin, 1e-8)
        shapes = (shapes - smin) / denom

    full_dataset = EITShapeDataset(voltages, shapes)

    train_set = torch.utils.data.Subset(full_dataset, train_idx.tolist())
    val_set = torch.utils.data.Subset(full_dataset, val_idx.tolist())
    test_set = torch.utils.data.Subset(full_dataset, test_idx.tolist())

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    print(f"Dataset: train={n_train}, val={n_val}, test={n_test}")

    # ---------------------------------------------------------------
    # Build models
    # ---------------------------------------------------------------
    voltage_ae = VoltageAutoencoder(
        input_dim=args.voltage_dim,
        latent_dim=args.voltage_latent,
        dropout=args.dropout,
    ).to(device)

    shape_ae = ShapeAutoencoder(
        image_size=args.image_size,
        latent_dim=args.shape_latent,
    ).to(device)

    projector = LatentProjector(
        voltage_latent=args.voltage_latent,
        shape_latent=args.shape_latent,
    ).to(device)

    # ---------------------------------------------------------------
    # Stage 1: voltage denoising autoencoder
    # ---------------------------------------------------------------
    print("\n=== Stage 1: voltage denoising autoencoder ===")
    voltage_ae = train_autoencoder(
        voltage_ae,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs_voltage,
        lr=args.lr_voltage,
        noise_std=args.voltage_noise_std,
        mode="voltage",
        checkpoint_path=out / "best_voltage_ae.pt",
        patience=args.patience,
    )

    # ---------------------------------------------------------------
    # Stage 2: shape/deformation-map denoising autoencoder
    # ---------------------------------------------------------------
    print("\n=== Stage 2: shape-map denoising convolutional autoencoder ===")
    shape_ae = train_autoencoder(
        shape_ae,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs_shape,
        lr=args.lr_shape,
        noise_std=args.shape_noise_std,
        mode="shape",
        checkpoint_path=out / "best_shape_ae.pt",
        patience=args.patience,
    )

    # ---------------------------------------------------------------
    # Stage 3: latent-space projection
    # ---------------------------------------------------------------
    print("\n=== Stage 3: supervised latent projection ===")
    projector = train_projector(
        voltage_ae,
        shape_ae,
        projector,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs_projector,
        lr=args.lr_projector,
        checkpoint_path=out / "best_projector.pt",
        patience=args.patience,
    )

    # ---------------------------------------------------------------
    # Test
    # ---------------------------------------------------------------
    print("\n=== Test evaluation ===")
    metrics, predictions, targets = evaluate(
        voltage_ae, shape_ae, projector, test_loader,
        device=device, threshold=args.threshold
    )

    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    with open(out / "test_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.8f}\n")

    np.savez_compressed(
        out / "test_predictions.npz",
        predictions=predictions,
        targets=targets,
        test_indices=test_idx,
    )

    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nSaved outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()
