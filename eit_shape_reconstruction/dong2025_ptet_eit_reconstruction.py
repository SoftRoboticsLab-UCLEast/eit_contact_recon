"""
dong2025_ptet_eit_reconstruction.py

Standalone PyTorch adaptation of PTET from:

    H. Dong et al.,
    "Efficient Tactile Perception with Soft Electrical Impedance Tomography
    and Pre-trained Transformer," IEEE Transactions on Industrial Electronics,
    2025/2026.

The original PTET framework uses:
    (1) self-supervised masked-autoencoder pretraining of EIT voltage maps,
    (2) self-supervised ResNet autoencoder pretraining of tactile maps,
    (3) supervised alignment of the pretrained voltage encoder and tactile
        decoder using limited paired voltage/tactile samples.

Paper-supported PTET details implemented here
---------------------------------------------
Voltage branch:
    - EIT voltage represented as a 16x16 EIM and bicubically enlarged to a
      64x64 Enhanced Electrical Impedance Map (E2IM).
    - 4x4 patches.
    - 75% random masking during self-supervised MAE pretraining.
    - Transformer embedding dimension = 256.
    - 12 Transformer encoder layers.
    - 2 Transformer decoder layers.
    - 4 attention heads.
    - MSE reconstruction objective.

Tactile branch:
    - ResNet-style autoencoder.
    - Feature channels [32, 64, 128].
    - Residual self-attention in the tactile decoder.
    - MSE reconstruction objective.

Paired reconstruction stage:
    - pretrained voltage encoder is frozen;
    - voltage/tactile latent spaces are connected using a direct LINEAR
      projection (no hidden layers);
    - projection and tactile decoder are optimized on paired data.

Adaptations for the present EIT shape-reconstruction problem
------------------------------------------------------------
The original PTET paper uses 104 independent voltage values (from 208
directional measurements under reciprocity) and 48x48 tactile maps.
The present project uses:
    - 208-dimensional Delta-V measurements;
    - 64x64 binary/continuous contact-shape maps.

Because the exact reciprocity-index mapping of a user's 208 channels depends
on acquisition ordering, this script supports two EIM construction modes:

    --eim-mode learned208   [DEFAULT]
        A learnable linear adapter maps the standardized 208-D vector into a
        16x16 EIM. This is an adaptation for datasets where reciprocity channel
        indices are not supplied.

    --eim-mode paper104
        Uses the paper-style EIM construction from a 104-D independent voltage
        vector. If your stored input contains 208 values, provide
        --reciprocity-pairs pairs.npy, an integer [104,2] array identifying
        reciprocal channel pairs. Each pair is averaged to produce one
        independent measurement.

For maximum fidelity to dong2025, use:
    --eim-mode paper104 --reciprocity-pairs path/to/pairs.npy

Expected NPZ dataset
--------------------
Required arrays:
    voltages : [N,208] (or [N,104] with --eim-mode paper104)
    shapes   : [N,64,64] or [N,1,64,64]

Example
-------
python dong2025_ptet_eit_reconstruction.py \
    --data data/contact_shapes.npz \
    --output-dir runs/ptet

Outputs
-------
    best_voltage_mae.pt
    best_tactile_ae.pt
    best_ptet_finetuned.pt
    voltage_normalization.npz
    test_predictions.npz
    test_metrics.txt
    config.json

Notes
-----
This is an architecture-level reimplementation/adaptation, not the authors'
original released code. Hyperparameters not explicitly reported or altered to
fit the 208-D / 64x64 dataset are exposed as command-line options.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Dataset + normalization
# =============================================================================

class EITShapeDataset(Dataset):
    def __init__(self, voltages: np.ndarray, shapes: np.ndarray):
        v = np.asarray(voltages, dtype=np.float32)
        y = np.asarray(shapes, dtype=np.float32)

        if v.ndim != 2:
            raise ValueError(f"voltages must be [N,D], got {v.shape}")

        if y.ndim == 3:
            y = y[:, None, :, :]
        if y.ndim != 4 or y.shape[1] != 1:
            raise ValueError(f"shapes must be [N,H,W] or [N,1,H,W], got {y.shape}")
        if len(v) != len(y):
            raise ValueError("voltages and shapes must contain the same N")

        self.voltages = torch.from_numpy(v)
        self.shapes = torch.from_numpy(y)

    def __len__(self):
        return len(self.voltages)

    def __getitem__(self, i):
        return self.voltages[i], self.shapes[i]


class VoltageNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = x.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)


# =============================================================================
# EIM / E2IM construction
# =============================================================================

def paper104_to_eim(v104: torch.Tensor) -> torch.Tensor:
    """
    Paper-style 104 -> 16x16 symmetric EIM.

    Implements the published algorithm concept:
      for i,j:
          if j > i+1:
              EIM[i,j] = next independent voltage
              EIM[j,i] = same value

    There are more geometrically eligible entries than 104, so filling stops
    after 104 values, matching the algorithm's index <= 104 condition.

    Input : [B,104]
    Output: [B,1,16,16]
    """
    if v104.shape[1] != 104:
        raise ValueError(f"paper104_to_eim expects 104 values, got {v104.shape[1]}")

    B = v104.shape[0]
    eim = v104.new_zeros((B, 16, 16))

    positions = []
    for i in range(16):
        for j in range(16):
            # Python zero-indexed equivalent of paper condition j > i + 1
            if j > i + 1:
                positions.append((i, j))
                if len(positions) == 104:
                    break
        if len(positions) == 104:
            break

    if len(positions) != 104:
        raise RuntimeError(f"Internal EIM position count is {len(positions)}, expected 104")

    for k, (i, j) in enumerate(positions):
        eim[:, i, j] = v104[:, k]
        eim[:, j, i] = v104[:, k]

    return eim.unsqueeze(1)


def reduce_208_by_reciprocity(v208: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
    """
    Average known reciprocal channel pairs.
    pair_idx: [104,2], integer channel indices.
    """
    if v208.shape[1] != 208:
        raise ValueError("Reciprocity reduction expects 208 input channels")
    if pair_idx.shape != (104, 2):
        raise ValueError(f"reciprocity pairs must be [104,2], got {tuple(pair_idx.shape)}")

    a = v208[:, pair_idx[:, 0]]
    b = v208[:, pair_idx[:, 1]]
    return 0.5 * (a + b)


class E2IMAdapter(nn.Module):
    """
    Convert raw standardized voltage vector to 64x64 E2IM.

    learned208:
        208 -> Linear -> 256 -> reshape 16x16 -> bicubic 64x64

    paper104:
        104 independent values -> paper EIM -> bicubic 64x64
        If raw D=208, reciprocal pair indices are required.
    """
    def __init__(
        self,
        input_dim=208,
        mode="learned208",
        reciprocity_pairs: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mode = mode

        if mode == "learned208":
            self.adapter = nn.Linear(input_dim, 16 * 16)
            self.register_buffer("pair_idx", torch.empty(0, 2, dtype=torch.long))

        elif mode == "paper104":
            self.adapter = None

            if input_dim == 208:
                if reciprocity_pairs is None:
                    raise ValueError(
                        "--eim-mode paper104 with 208-D data requires "
                        "--reciprocity-pairs pairs.npy of shape [104,2]"
                    )
                pairs = np.asarray(reciprocity_pairs, dtype=np.int64)
                if pairs.shape != (104, 2):
                    raise ValueError("reciprocity-pairs must have shape [104,2]")
                self.register_buffer("pair_idx", torch.from_numpy(pairs))
            elif input_dim == 104:
                self.register_buffer("pair_idx", torch.empty(0, 2, dtype=torch.long))
            else:
                raise ValueError("paper104 mode requires input dimension 104 or 208")
        else:
            raise ValueError(f"Unknown EIM mode: {mode}")

    def forward(self, x):
        if self.mode == "learned208":
            eim = self.adapter(x).view(-1, 1, 16, 16)
        else:
            if x.shape[1] == 208:
                x = reduce_208_by_reciprocity(x, self.pair_idx)
            eim = paper104_to_eim(x)

        # Published E2IM construction: bicubic 16x16 -> 64x64
        e2im = F.interpolate(
            eim, size=(64, 64), mode="bicubic", align_corners=False
        )
        return e2im


# =============================================================================
# Transformer blocks
# =============================================================================

class TransformerStack(nn.Module):
    def __init__(self, dim=256, depth=12, heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Voltage masked autoencoder
# =============================================================================

class VoltageMAE(nn.Module):
    """
    PTET-style voltage MAE.

    E2IM: 64x64
    Patch: 4x4
    Tokens: 16x16 = 256
    Embed: 256
    Encoder: 12 transformer layers / 4 heads
    Decoder: 2 transformer layers / 4 heads
    Masking: 75% by default
    """

    def __init__(
        self,
        input_dim=208,
        eim_mode="learned208",
        reciprocity_pairs=None,
        image_size=64,
        patch_size=4,
        embed_dim=256,
        encoder_depth=12,
        decoder_depth=2,
        heads=4,
        mask_ratio=0.75,
        dropout=0.0,
    ):
        super().__init__()
        assert image_size == 64, "PTET E2IM is 64x64"
        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid ** 2
        self.patch_dim = patch_size ** 2
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        self.e2im = E2IMAdapter(
            input_dim=input_dim,
            mode=eim_mode,
            reciprocity_pairs=reciprocity_pairs,
        )

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.encoder = TransformerStack(
            dim=embed_dim,
            depth=encoder_depth,
            heads=heads,
            dropout=dropout,
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.decoder = TransformerStack(
            dim=embed_dim,
            depth=decoder_depth,
            heads=heads,
            dropout=dropout,
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.patch_predictor = nn.Linear(embed_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

    def patchify(self, img):
        B, C, H, W = img.shape
        p = self.patch_size
        x = img.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, self.num_patches, p * p * C)
        return x

    def unpatchify(self, patches):
        B = patches.shape[0]
        p = self.patch_size
        g = self.grid
        x = patches.reshape(B, g, g, p, p, 1)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, 1, g * p, g * p)

    def random_mask(self, tokens):
        """
        MAE-style random masking.
        Returns visible tokens, mask, restore indices.
        mask: 0 visible, 1 masked
        """
        B, N, D = tokens.shape
        n_keep = int(N * (1.0 - self.mask_ratio))

        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :n_keep]

        visible = torch.gather(
            tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        mask = torch.ones(B, N, device=tokens.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return visible, mask, ids_restore, ids_keep

    def encode(self, voltage, masked=False):
        """
        For downstream reconstruction use masked=False, so every token is seen.

        Returns:
            cls latent [B,256]
            patch latents [B,N,256] for unmasked mode
        """
        e2im = self.e2im(voltage)
        patches = self.patchify(e2im)
        tok = self.patch_embed(patches)

        B = tok.shape[0]

        if masked:
            visible, mask, ids_restore, ids_keep = self.random_mask(tok)
            pos_patch = self.pos_embed[:, 1:].expand(B, -1, -1)
            pos_visible = torch.gather(
                pos_patch, 1,
                ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )
            visible = visible + pos_visible
            cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1]
            enc = self.encoder(torch.cat([cls, visible], dim=1))
            enc = self.encoder_norm(enc)
            return enc, mask, ids_restore

        tok = tok + self.pos_embed[:, 1:]
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1]
        enc = self.encoder(torch.cat([cls, tok], dim=1))
        enc = self.encoder_norm(enc)
        return enc[:, 0], enc[:, 1:], e2im

    def forward_mae(self, voltage):
        e2im = self.e2im(voltage)
        target = self.patchify(e2im)

        enc, mask, ids_restore = self.encode(voltage, masked=True)
        B = voltage.shape[0]

        cls = enc[:, :1]
        visible = enc[:, 1:]

        n_mask = self.num_patches - visible.shape[1]
        mask_tokens = self.mask_token.expand(B, n_mask, -1)

        full_patch_tokens = torch.cat([visible, mask_tokens], dim=1)
        full_patch_tokens = torch.gather(
            full_patch_tokens,
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )

        dec_tokens = torch.cat([cls, full_patch_tokens], dim=1)
        dec_tokens = dec_tokens + self.decoder_pos_embed
        dec_tokens = self.decoder(dec_tokens)
        dec_tokens = self.decoder_norm(dec_tokens)

        pred_patches = self.patch_predictor(dec_tokens[:, 1:])

        # MAE loss on masked patches only.
        per_patch_mse = ((pred_patches - target) ** 2).mean(dim=-1)
        loss = (per_patch_mse * mask).sum() / torch.clamp(mask.sum(), min=1.0)

        recon = self.unpatchify(pred_patches)
        return loss, recon, e2im


# =============================================================================
# Tactile ResNet autoencoder + decoder self-attention
# =============================================================================

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class SpatialSelfAttention(nn.Module):
    """
    Residual spatial self-attention in the decoder.
    Paper states Q/K/V dimensionality 128.
    """
    def __init__(self, channels=128):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2)          # B,C,L
        q = self.q(h).transpose(1, 2)        # B,L,C
        k = self.k(h)                        # B,C,L
        v = self.v(h).transpose(1, 2)        # B,L,C

        attn = torch.softmax((q @ k) * self.scale, dim=-1)  # B,L,L
        out = attn @ v                                      # B,L,C
        out = self.proj(out.transpose(1, 2)).reshape(B, C, H, W)
        return x + out


class TactileResNetAE(nn.Module):
    """
    ResNet-style tactile/contact-map autoencoder.

    Adapted to 64x64 maps for the current paper.
    Channels follow Ref. [13]: [32,64,128].
    """

    def __init__(self, image_size=64, latent_dim=256):
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image size must be divisible by 8")

        self.image_size = image_size
        self.base = image_size // 8
        self.latent_dim = latent_dim

        self.stem = nn.Conv2d(1, 32, 3, padding=1)

        self.enc1 = nn.Sequential(
            ResBlock(32), ResBlock(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            ResBlock(64), ResBlock(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            ResBlock(128), ResBlock(128),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        flat_dim = 128 * self.base * self.base
        self.to_latent = nn.Linear(flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat_dim)

        # Decoder: self-attention is deliberately located near the latent
        # representation, consistent with PTET's decoder-side attention.
        self.dec_attn = SpatialSelfAttention(128)

        self.dec1 = nn.Sequential(
            ResBlock(128), ResBlock(128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            ResBlock(128), ResBlock(128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            ResBlock(64), ResBlock(64),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.stem(x)
        h = self.enc1(h)
        h = self.enc2(h)
        h = self.enc3(h)
        return self.to_latent(h.flatten(1))

    def decode(self, z):
        h = self.from_latent(z)
        h = h.view(-1, 128, self.base, self.base)
        h = self.dec_attn(h)
        h = self.dec1(h)
        h = self.dec2(h)
        h = self.dec3(h)
        return self.out(h)

    def forward(self, x):
        return self.decode(self.encode(x))


# =============================================================================
# Final PTET reconstruction model
# =============================================================================

class PTETReconstructor(nn.Module):
    """
    E_vol(v) -> direct linear latent projection -> D_tact(z_t).

    Voltage encoder is frozen in fine-tuning by default.
    """

    def __init__(self, voltage_mae: VoltageMAE, tactile_ae: TactileResNetAE):
        super().__init__()
        self.voltage_mae = voltage_mae
        self.tactile_ae = tactile_ae

        # Paper: direct linear projection without hidden layers.
        self.projection = nn.Linear(
            voltage_mae.embed_dim,
            tactile_ae.latent_dim
        )

    def freeze_voltage_encoder(self):
        # Freeze voltage-side parameters, including E2IM adapter.
        for p in self.voltage_mae.parameters():
            p.requires_grad_(False)

    def forward(self, voltage):
        cls_latent, _, _ = self.voltage_mae.encode(voltage, masked=False)
        z_tact = self.projection(cls_latent)
        return self.tactile_ae.decode(z_tact)


# =============================================================================
# Metrics
# =============================================================================

@torch.no_grad()
def reconstruction_metrics(pred, target, threshold=0.5, eps=1e-8):
    pred_b = (pred >= threshold).float()
    gt_b = (target >= threshold).float()

    dims = (1, 2, 3)
    inter = (pred_b * gt_b).sum(dim=dims)
    pred_sum = pred_b.sum(dim=dims)
    gt_sum = gt_b.sum(dim=dims)
    union = pred_sum + gt_sum - inter

    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (pred_sum + gt_sum + eps)

    mse = ((pred - target) ** 2).mean(dim=dims)
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))
    return iou, dice, psnr


# =============================================================================
# Training: Stage 1 - voltage MAE
# =============================================================================

def voltage_mae_epoch(model, loader, optimizer, device):
    train = optimizer is not None
    model.train(train)

    total, n = 0.0, 0

    for v, _ in loader:
        v = v.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            loss, _, _ = model.forward_mae(v)
            if train:
                loss.backward()
                optimizer.step()

        total += loss.item() * v.size(0)
        n += v.size(0)

    return total / max(n, 1)


def train_voltage_mae(
    model, train_loader, val_loader, device,
    epochs, lr, checkpoint, patience
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = float("inf")
    bad = 0

    for ep in range(1, epochs + 1):
        tr = voltage_mae_epoch(model, train_loader, optimizer, device)
        va = voltage_mae_epoch(model, val_loader, None, device)

        print(f"[Voltage MAE] {ep:03d}/{epochs} train={tr:.7f} val={va:.7f}")

        if va < best:
            best = va
            bad = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad += 1

        if bad >= patience:
            print("[Voltage MAE] early stopping")
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


# =============================================================================
# Training: Stage 2 - tactile AE
# =============================================================================

def tactile_ae_epoch(model, loader, optimizer, device):
    train = optimizer is not None
    model.train(train)
    criterion = nn.MSELoss()

    total, n = 0.0, 0

    for _, shape in loader:
        shape = shape.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred = model(shape)
            loss = criterion(pred, shape)
            if train:
                loss.backward()
                optimizer.step()

        total += loss.item() * shape.size(0)
        n += shape.size(0)

    return total / max(n, 1)


def train_tactile_ae(
    model, train_loader, val_loader, device,
    epochs, lr, checkpoint, patience
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = float("inf")
    bad = 0

    for ep in range(1, epochs + 1):
        tr = tactile_ae_epoch(model, train_loader, optimizer, device)
        va = tactile_ae_epoch(model, val_loader, None, device)

        print(f"[Tactile AE] {ep:03d}/{epochs} train={tr:.7f} val={va:.7f}")

        if va < best:
            best = va
            bad = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad += 1

        if bad >= patience:
            print("[Tactile AE] early stopping")
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


# =============================================================================
# Training: Stage 3 - paired PTET alignment/fine-tuning
# =============================================================================

def paired_epoch(model, loader, optimizer, device):
    train = optimizer is not None

    # Keep frozen voltage encoder in eval mode.
    model.voltage_mae.eval()
    model.tactile_ae.train(train)
    model.projection.train(train)

    criterion = nn.MSELoss()
    total, n = 0.0, 0

    for v, shape in loader:
        v = v.to(device)
        shape = shape.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred = model(v)
            loss = criterion(pred, shape)
            if train:
                loss.backward()
                optimizer.step()

        total += loss.item() * v.size(0)
        n += v.size(0)

    return total / max(n, 1)


def train_paired_ptet(
    model, train_loader, val_loader, device,
    epochs, lr, checkpoint, patience
):
    model.freeze_voltage_encoder()

    # Fine-tune the tactile decoder and the newly introduced latent projection.
    # The pretrained tactile encoder is not used in the final v -> t graph.
    trainable = list(model.projection.parameters())
    trainable += list(model.tactile_ae.from_latent.parameters())
    trainable += list(model.tactile_ae.dec_attn.parameters())
    trainable += list(model.tactile_ae.dec1.parameters())
    trainable += list(model.tactile_ae.dec2.parameters())
    trainable += list(model.tactile_ae.dec3.parameters())
    trainable += list(model.tactile_ae.out.parameters())

    # Freeze tactile encoder explicitly.
    for module in [
        model.tactile_ae.stem,
        model.tactile_ae.enc1,
        model.tactile_ae.enc2,
        model.tactile_ae.enc3,
        model.tactile_ae.to_latent,
    ]:
        for p in module.parameters():
            p.requires_grad_(False)

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    best = float("inf")
    bad = 0

    for ep in range(1, epochs + 1):
        tr = paired_epoch(model, train_loader, optimizer, device)
        va = paired_epoch(model, val_loader, None, device)

        print(f"[PTET paired] {ep:03d}/{epochs} train={tr:.7f} val={va:.7f}")

        if va < best:
            best = va
            bad = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad += 1

        if bad >= patience:
            print("[PTET paired] early stopping")
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()

    all_iou, all_dice, all_psnr = [], [], []
    preds, gts = [], []

    for v, y in loader:
        v = v.to(device)
        y = y.to(device)

        pred = model(v)
        iou, dice, psnr = reconstruction_metrics(pred, y, threshold)

        all_iou.append(iou.cpu())
        all_dice.append(dice.cpu())
        all_psnr.append(psnr.cpu())
        preds.append(pred.cpu())
        gts.append(y.cpu())

    iou = torch.cat(all_iou)
    dice = torch.cat(all_dice)
    psnr = torch.cat(all_psnr)

    metrics = {
        "IoU_mean": float(iou.mean()),
        "IoU_std": float(iou.std(unbiased=False)),
        "Dice_mean": float(dice.mean()),
        "Dice_std": float(dice.std(unbiased=False)),
        "PSNR_mean_dB": float(psnr.mean()),
        "PSNR_std_dB": float(psnr.std(unbiased=False)),
    }

    return (
        metrics,
        torch.cat(preds).numpy(),
        torch.cat(gts).numpy(),
    )


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="PTET-style transformer EIT contact-shape reconstruction"
    )

    p.add_argument("--data", required=True,
                   help="NPZ containing voltages and shapes")
    p.add_argument("--output-dir", default="runs/ptet")

    p.add_argument("--voltage-dim", type=int, default=208)
    p.add_argument("--shape-size", type=int, default=64)

    p.add_argument(
        "--eim-mode",
        choices=["learned208", "paper104"],
        default="learned208",
        help=(
            "learned208 adapts 208-D directly to a 16x16 EIM; "
            "paper104 follows the published 104-value EIM construction"
        ),
    )
    p.add_argument(
        "--reciprocity-pairs",
        default=None,
        help="Optional .npy [104,2] channel-pair indices for paper104 mode",
    )

    # Paper-supported transformer settings
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--encoder-depth", type=int, default=12)
    p.add_argument("--decoder-depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)

    # Adapted tactile latent size
    p.add_argument("--tactile-latent", type=int, default=256)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-voltage", type=int, default=150)
    p.add_argument("--epochs-tactile", type=int, default=150)
    p.add_argument("--epochs-finetune", type=int, default=150)

    p.add_argument("--lr-voltage", type=float, default=1e-4)
    p.add_argument("--lr-tactile", type=float, default=1e-4)
    p.add_argument("--lr-finetune", type=float, default=1e-4)

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=25)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    print("Device:", device)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    data = np.load(args.data)
    if "voltages" not in data or "shapes" not in data:
        raise KeyError("NPZ must contain arrays named 'voltages' and 'shapes'")

    voltages_raw = np.asarray(data["voltages"], dtype=np.float32)
    shapes = np.asarray(data["shapes"], dtype=np.float32)

    if voltages_raw.ndim != 2:
        raise ValueError("voltages must be [N,D]")
    if voltages_raw.shape[1] != args.voltage_dim:
        raise ValueError(
            f"Data has D={voltages_raw.shape[1]}, "
            f"but --voltage-dim={args.voltage_dim}"
        )

    if shapes.ndim == 3:
        H, W = shapes.shape[1:]
    elif shapes.ndim == 4:
        H, W = shapes.shape[-2:]
    else:
        raise ValueError("shapes must be [N,H,W] or [N,1,H,W]")

    if (H, W) != (args.shape_size, args.shape_size):
        raise ValueError(
            f"Shape maps are {H}x{W}; expected "
            f"{args.shape_size}x{args.shape_size}"
        )

    # Contact maps expected in [0,1].
    smin, smax = float(shapes.min()), float(shapes.max())
    if smin < 0 or smax > 1:
        print(
            f"Shape values [{smin:.5g},{smax:.5g}] are outside [0,1]; "
            "applying min-max normalization."
        )
        shapes = (shapes - smin) / max(smax - smin, 1e-8)

    # -------------------------------------------------------------------------
    # Split BEFORE normalization
    # -------------------------------------------------------------------------
    N = len(voltages_raw)
    n_train = int(args.train_frac * N)
    n_val = int(args.val_frac * N)
    n_test = N - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Invalid train/validation/test fractions")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    normalizer = VoltageNormalizer().fit(voltages_raw[train_idx])
    voltages = normalizer.transform(voltages_raw)
    normalizer.save(out / "voltage_normalization.npz")

    ds = EITShapeDataset(voltages, shapes)
    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx.tolist())
    test_ds = Subset(ds, test_idx.tolist())

    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    print(f"Samples: train={n_train}, val={n_val}, test={n_test}")

    # -------------------------------------------------------------------------
    # Optional reciprocity mapping
    # -------------------------------------------------------------------------
    reciprocity_pairs = None
    if args.reciprocity_pairs is not None:
        reciprocity_pairs = np.load(args.reciprocity_pairs)

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------
    voltage_mae = VoltageMAE(
        input_dim=args.voltage_dim,
        eim_mode=args.eim_mode,
        reciprocity_pairs=reciprocity_pairs,
        image_size=64,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        heads=args.heads,
        mask_ratio=args.mask_ratio,
    ).to(device)

    tactile_ae = TactileResNetAE(
        image_size=args.shape_size,
        latent_dim=args.tactile_latent,
    ).to(device)

    # -------------------------------------------------------------------------
    # 1. Self-supervised voltage MAE pretraining
    # -------------------------------------------------------------------------
    print("\n=== Stage 1: E2IM masked-autoencoder pretraining ===")
    voltage_mae = train_voltage_mae(
        voltage_mae,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs_voltage,
        lr=args.lr_voltage,
        checkpoint=out / "best_voltage_mae.pt",
        patience=args.patience,
    )

    # -------------------------------------------------------------------------
    # 2. Self-supervised tactile-map AE pretraining
    # -------------------------------------------------------------------------
    print("\n=== Stage 2: tactile ResNet autoencoder pretraining ===")
    tactile_ae = train_tactile_ae(
        tactile_ae,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs_tactile,
        lr=args.lr_tactile,
        checkpoint=out / "best_tactile_ae.pt",
        patience=args.patience,
    )

    # -------------------------------------------------------------------------
    # 3. Limited-pair alignment/fine-tuning
    # -------------------------------------------------------------------------
    print("\n=== Stage 3: PTET latent alignment + decoder fine-tuning ===")
    model = PTETReconstructor(voltage_mae, tactile_ae).to(device)

    model = train_paired_ptet(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs_finetune,
        lr=args.lr_finetune,
        checkpoint=out / "best_ptet_finetuned.pt",
        patience=args.patience,
    )

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------
    print("\n=== Test evaluation ===")
    metrics, preds, targets = evaluate(
        model, test_loader, device, threshold=args.threshold
    )

    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    with open(out / "test_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.8f}\n")

    np.savez_compressed(
        out / "test_predictions.npz",
        predictions=preds,
        targets=targets,
        test_indices=test_idx,
    )

    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\nFinal inference path:")
    print("  DeltaV -> EIM/E2IM -> pretrained Transformer voltage encoder")
    print("         -> direct linear latent projection")
    print("         -> pretrained/fine-tuned ResNet tactile decoder")
    print("         -> contact-shape map")
    print(f"\nSaved outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()