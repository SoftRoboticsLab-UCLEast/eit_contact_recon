#!/usr/bin/env python3
"""
Hybrid EIT tactile shape reconstruction training script (SIM).

Update (requested):
- If checkpoint exists: load it, skip training, and STILL compute metrics.
- Metrics include:
  (A) Seen test split metrics (IoU/Dice, overall + per-shape) with mean±std
  (B) Unseen shapes metrics (C, Z, plus) via forward simulation with mean±std
- Also selects best threshold on seen test set.

Models supported:
    INPUT_MODE in {"hybrid", "volt_only", "bp_only"}
"""

from pathlib import Path
import csv
import numpy as np
from PIL import Image
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol
from pyeit.eit.fem import EITForward
import pyeit.eit.bp as bp


# =========================
# CONFIG
# =========================
DATA_ROOT = Path("eit_dataset")
CSV_PATH = DATA_ROOT / "voltages.csv"

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
BP_DROPOUT_PROB = 0.5
GRID_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 1

# "hybrid" | "volt_only" | "bp_only"
INPUT_MODE = "hybrid"

CKPT_PATH = DATA_ROOT / f"best_hybrid_model_{INPUT_MODE}.pt"

# Threshold search for binary masks
THRESHOLDS = np.linspace(0.05, 0.95, 19)  # 0.05,0.10,...,0.95

# Unseen-shape eval config
RUN_UNSEEN_EVAL = True
UNSEEN_SHAPES = ["C", "Z", "plus"]
N_SAMPLES_PER_UNSEEN_SHAPE = 300
CONTRAST_LEVELS = [3.0, 5.0, 10.0, 15.0, 20.0]
RANDOM_SEED = 123

# Optional: save a few qualitative examples for unseen shapes
SAVE_UNSEEN_VIZ = False
VIZ_PER_SHAPE = 6
VIZ_DIR = DATA_ROOT / f"unseen_shapes_viz_{INPUT_MODE}"


# =========================
# EIT / GEOMETRY HELPERS
# =========================
def setup_eit(n_el=16, h0=0.04):
    mesh_obj = mesh.create(n_el, h0=h0)
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
    fwd = EITForward(mesh_obj, protocol_obj)

    n_elems = mesh_obj.element.shape[0]
    perm_ref = np.ones(n_elems, dtype=float)
    v0 = fwd.solve_eit(perm_ref)

    eit_bp = bp.BP(mesh_obj, protocol_obj)
    eit_bp.setup(weight="none")
    return mesh_obj, protocol_obj, fwd, eit_bp, v0


def make_grid(mesh_obj, grid_size=GRID_SIZE):
    pts = mesh_obj.node
    tri = mesh_obj.element
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)
    lin = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(lin, lin)
    return triang, xx, yy


def rasterize_to_grid(triang, xx, yy, nodal_vals, fill_value=0.0):
    interp = mtri.LinearTriInterpolator(triang, nodal_vals)
    grid = interp(xx, yy)
    return np.ma.filled(grid, fill_value=fill_value)


def domain_mask_grid(grid_size=GRID_SIZE):
    lin = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(lin, lin)
    return ((xx**2 + yy**2) <= 1.0).astype(np.float32)


# =========================
# DATASET (seen shapes: from CSV)
# =========================
class EITHybridDataset(Dataset):
    def __init__(self, csv_path, data_root, split="train"):
        self.data_root = Path(data_root)
        self.split = split

        self.rows = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.rows.append(row)

        if len(self.rows) == 0:
            raise RuntimeError(f"No rows found for split={split} in {csv_path}")

        with open(csv_path, "r", newline="") as f2:
            reader2 = csv.reader(f2)
            header = next(reader2)

        self.volt_col_start = 5
        self.n_meas = len(header) - self.volt_col_start

        self.mesh_obj, self.protocol_obj, self.fwd, self.eit_bp, self.v0 = setup_eit()
        self.v0 = self.v0.astype(np.float32)

        self.triang, self.xx, self.yy = make_grid(self.mesh_obj, grid_size=GRID_SIZE)
        self.domain_grid = domain_mask_grid(GRID_SIZE)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        delta_v = np.empty(self.n_meas, dtype=np.float32)
        for i in range(self.n_meas):
            delta_v[i] = float(row[f"v_{i}"])

        v1 = self.v0 + delta_v

        sum_abs_diff = np.sum(np.abs(delta_v))
        if sum_abs_diff > 0.3:
            nodal_bp = 192.0 * self.eit_bp.solve(v1, self.v0, normalize=True, log_scale=False)
        else:
            nodal_bp = 192.0 * self.eit_bp.solve(self.v0, self.v0, normalize=True, log_scale=False)
        nodal_bp = np.real(nodal_bp).astype(np.float32)

        bp_grid = rasterize_to_grid(self.triang, self.xx, self.yy, nodal_bp, fill_value=0.0).astype(np.float32)

        mask_path = self.data_root / row["mask_path"]
        mask_img = Image.open(mask_path).convert("L")
        mask_arr = (np.array(mask_img, dtype=np.uint8) > 127).astype(np.float32)

        return {
            "volt": torch.from_numpy(delta_v),                         # [K]
            "bp": torch.from_numpy(bp_grid).unsqueeze(0),              # [1,H,W]
            "domain": torch.from_numpy(self.domain_grid).unsqueeze(0), # [1,H,W]
            "mask": torch.from_numpy(mask_arr).unsqueeze(0),           # [1,H,W]
            "sample_id": row["sample_id"],
            "shape_type": row["shape_type"],
            "contrast": float(row["contrast"]),
        }


# =========================
# MODEL
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch_in = in_channels
        for feat in features:
            self.downs.append(DoubleConv(ch_in, feat))
            ch_in = feat

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        ch_in = features[-1] * 2
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(ch_in, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(ch_in, feat))
            ch_in = feat

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final_conv(x)


class HybridEITNet(nn.Module):
    def __init__(self, volt_dim, img_size, feat_channels=16):
        super().__init__()
        self.volt_to_feat = nn.Linear(volt_dim, feat_channels)
        in_channels = 1 + 1 + feat_channels
        self.unet = UNet(in_channels=in_channels, out_channels=1)

    def forward(self, volt, bp_img, domain, bp_dropout_prob=0.0):
        B, _, H, W = bp_img.shape

        if self.training and bp_dropout_prob > 0.0:
            keep = (torch.rand(B, 1, 1, 1, device=bp_img.device) > bp_dropout_prob).float()
            bp_img = bp_img * keep

        feat = self.volt_to_feat(volt)[:, :, None, None].expand(-1, -1, H, W)
        x = torch.cat([bp_img, domain, feat], dim=1)
        return self.unet(x)


# =========================
# METRICS (torch)
# =========================
@torch.no_grad()
def iou_from_logits(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    targets = targets.float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    return inter / (union + eps)  # [B]


@torch.no_grad()
def dice_from_logits(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    targets = targets.float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 2.0 * inter / (denom + eps)  # [B]


# =========================
# TRAIN/EVAL LOOPS
# =========================
def train_one_epoch(model, loader, optimizer, device, bp_dropout_prob, input_mode):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total_loss, total_iou, n = 0.0, 0.0, 0

    for batch in loader:
        volt = batch["volt"].to(device)
        bp_ = batch["bp"].to(device)
        dom = batch["domain"].to(device)
        mask = batch["mask"].to(device)

        if input_mode == "bp_only":
            volt_in = torch.zeros_like(volt)
            bp_in = bp_
            eff_drop = 0.0
        elif input_mode == "volt_only":
            volt_in = volt
            bp_in = torch.zeros_like(bp_)
            eff_drop = 0.0
        else:
            volt_in = volt
            bp_in = bp_
            eff_drop = bp_dropout_prob

        optimizer.zero_grad()
        logits = model(volt_in, bp_in, dom, bp_dropout_prob=eff_drop)
        loss = 0.5 * bce(logits, mask) + 0.5 * (1.0 - dice_from_logits(logits, mask, thresh=0.5).mean())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += iou_from_logits(logits.detach(), mask, thresh=0.5).mean().item()
        n += 1

    return total_loss / max(n, 1), total_iou / max(n, 1)


@torch.no_grad()
def eval_model_seen(model, loader, device, input_mode, thresh=0.5):
    """
    Returns mean±std across test samples:
      - overall IoU/Dice
      - per-shape IoU/Dice
    """
    model.eval()

    all_iou = []
    all_dice = []

    per_shape = {}  # shape -> {"iou":[], "dice":[]}

    for batch in loader:
        volt = batch["volt"].to(device)
        bp_ = batch["bp"].to(device)
        dom = batch["domain"].to(device)
        mask = batch["mask"].to(device)

        if input_mode == "bp_only":
            volt_in = torch.zeros_like(volt)
            bp_in = bp_
        elif input_mode == "volt_only":
            volt_in = volt
            bp_in = torch.zeros_like(bp_)
        else:
            volt_in = volt
            bp_in = bp_

        logits = model(volt_in, bp_in, dom, bp_dropout_prob=0.0)

        iou_b = iou_from_logits(logits, mask, thresh=thresh)   # [B]
        dice_b = dice_from_logits(logits, mask, thresh=thresh) # [B]

        all_iou.append(iou_b.cpu())
        all_dice.append(dice_b.cpu())

        shapes = batch["shape_type"]
        for i, s in enumerate(shapes):
            per_shape.setdefault(s, {"iou": [], "dice": []})
            per_shape[s]["iou"].append(float(iou_b[i].item()))
            per_shape[s]["dice"].append(float(dice_b[i].item()))

    all_iou = torch.cat(all_iou).numpy()
    all_dice = torch.cat(all_dice).numpy()

    per_shape_stats = {}
    for s, d in per_shape.items():
        ious = np.asarray(d["iou"], dtype=np.float32)
        dices = np.asarray(d["dice"], dtype=np.float32)
        per_shape_stats[s] = {
            "iou_mean": float(np.mean(ious)) if len(ious) else float("nan"),
            "iou_std":  float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0,
            "dice_mean": float(np.mean(dices)) if len(dices) else float("nan"),
            "dice_std":  float(np.std(dices, ddof=1)) if len(dices) > 1 else 0.0,
            "n": int(len(ious)),
        }

    return {
        "iou_mean": float(np.mean(all_iou)),
        "iou_std":  float(np.std(all_iou, ddof=1)) if len(all_iou) > 1 else 0.0,
        "dice_mean": float(np.mean(all_dice)),
        "dice_std":  float(np.std(all_dice, ddof=1)) if len(all_dice) > 1 else 0.0,
        "per_shape": per_shape_stats,
        "n": int(len(all_iou)),
    }


@torch.no_grad()
def find_best_threshold_on_seen(model, loader, device, input_mode, thresholds):
    """
    Pick threshold that maximizes mean IoU on seen test split.
    Returns: best_t, (iou_mean, iou_std), (dice_mean, dice_std)
    """
    best_t = None
    best_iou_mean = -1.0
    best_stats = None

    for t in thresholds:
        res = eval_model_seen(model, loader, device, input_mode, thresh=float(t))
        if res["iou_mean"] > best_iou_mean:
            best_iou_mean = res["iou_mean"]
            best_t = float(t)
            best_stats = res

    return best_t, (best_stats["iou_mean"], best_stats["iou_std"]), (best_stats["dice_mean"], best_stats["dice_std"])


def load_checkpoint_if_exists(model, ckpt_path: Path) -> bool:
    if not ckpt_path.exists():
        return False
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[info] Found checkpoint: {ckpt_path}")
    print(f"[info] Loaded epoch={ckpt.get('epoch','?')} test_iou={ckpt.get('test_iou','N/A')}")
    return True


# =========================
# Unseen shapes evaluation (integrated)
# =========================
def element_centroids(mesh_obj):
    pts = mesh_obj.node
    tri = mesh_obj.element
    return pts[tri].mean(axis=1)


def rotate_points(x, y, angle_rad):
    ca = np.cos(angle_rad)
    sa = np.sin(angle_rad)
    return ca * x - sa * y, sa * x + ca * y


def element_mask_to_nodal(mesh_obj, elem_mask):
    tri = mesh_obj.element
    n_nodes = mesh_obj.node.shape[0]
    nodal = np.zeros(n_nodes, dtype=float)
    for e_idx, nodes in enumerate(tri):
        if elem_mask[e_idx]:
            nodal[nodes] = 1.0
    return nodal


def make_perm_for_mask(mesh_obj, elem_mask, contrast=5.0, base_perm=1.0):
    n_elems = mesh_obj.element.shape[0]
    perm = np.ones(n_elems, dtype=float) * base_perm
    perm[elem_mask] = base_perm * contrast
    return perm


def C_mask(mesh_obj, offset=(0.0, 0.0), angle=0.0, r_outer=0.35, r_inner=0.20, gap_angle=np.pi/3):
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    x_local, y_local = rotate_points(x, y, -angle)
    r2 = x_local**2 + y_local**2
    theta = np.arctan2(y_local, x_local)
    annulus = (r_inner**2 <= r2) & (r2 <= r_outer**2)
    gap = (np.abs(theta) < gap_angle)
    return annulus & (~gap)


def plus_mask_rotated(mesh_obj, offset=(0.0, 0.0), angle=0.0, arm_width=0.10, arm_length=0.45):
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    x_local, y_local = rotate_points(x, y, -angle)
    vert = (np.abs(x_local) < arm_width/2) & (np.abs(y_local) < arm_length/2)
    horiz = (np.abs(y_local) < arm_width/2) & (np.abs(x_local) < arm_length/2)
    return vert | horiz


def Z_mask_rotated(mesh_obj, offset=(0.0, 0.0), angle=0.0, width=0.10, length=0.60):
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    x_local, y_local = rotate_points(x, y, -angle)

    half_L = length / 2.0
    top = (y_local > 0.25) & (y_local < 0.35) & (x_local > -half_L) & (x_local < half_L)
    bottom = (y_local > -0.35) & (y_local < -0.25) & (x_local > -half_L) & (x_local < half_L)

    diag_band = (y_local > -0.25) & (y_local < 0.25)
    dist_to_diag = np.abs(y_local + x_local) / np.sqrt(2.0)
    diag = diag_band & (dist_to_diag < width/2.0)
    return top | bottom | diag


def random_unseen_shape_mask(mesh_obj, rng, shape_type):
    contrast = float(rng.choice(CONTRAST_LEVELS))
    ox = rng.uniform(-0.2, 0.2)
    oy = rng.uniform(-0.2, 0.2)
    angle = rng.uniform(0.0, 2.0 * np.pi)

    if shape_type == "C":
        elem_mask = C_mask(mesh_obj, offset=(ox, oy), angle=angle, r_outer=0.35, r_inner=0.18, gap_angle=np.pi/4)
    elif shape_type == "plus":
        elem_mask = plus_mask_rotated(mesh_obj, offset=(ox, oy), angle=angle, arm_width=0.12, arm_length=0.50)
    elif shape_type == "Z":
        elem_mask = Z_mask_rotated(mesh_obj, offset=(ox, oy), angle=angle, width=0.10, length=0.60)
    else:
        raise ValueError(f"Unknown unseen shape type: {shape_type}")

    return elem_mask, contrast


def visualize_unseen(bp_grid, gt_mask, pred_mask, shape_type, idx, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    ax0, ax1, ax2 = axes

    im0 = ax0.imshow(bp_grid, cmap="viridis")
    ax0.set_title("BP recon")
    ax0.axis("off")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    ax1.imshow(gt_mask, cmap="gray")
    ax1.set_title("Ground truth")
    ax1.axis("off")

    ax2.imshow(pred_mask, cmap="gray")
    ax2.set_title("Prediction")
    ax2.axis("off")

    fig.suptitle(f"Unseen shape: {shape_type}", fontsize=12)
    plt.tight_layout()
    out_path = out_dir / f"{shape_type}_example_{idx:03d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def eval_unseen_shapes(model, device, input_mode, thresh=0.5):
    rng = np.random.default_rng(RANDOM_SEED)

    mesh_obj, protocol_obj, fwd, eit_bp, v0 = setup_eit()
    v0 = v0.astype(np.float32)

    triang, xx, yy = make_grid(mesh_obj, grid_size=GRID_SIZE)
    domain_grid = domain_mask_grid(GRID_SIZE)

    per_shape = {}
    all_iou = []
    all_dice = []

    if SAVE_UNSEEN_VIZ:
        VIZ_DIR.mkdir(parents=True, exist_ok=True)

    for shape_type in UNSEEN_SHAPES:
        ious = []
        dices = []
        viz_count = 0

        for k in range(N_SAMPLES_PER_UNSEEN_SHAPE):
            elem_mask, contrast = random_unseen_shape_mask(mesh_obj, rng, shape_type)
            perm = make_perm_for_mask(mesh_obj, elem_mask, contrast=contrast)
            v1 = fwd.solve_eit(perm).astype(np.float32)
            delta_v = v1 - v0

            sum_abs_diff = float(np.sum(np.abs(delta_v)))
            if sum_abs_diff > 0.3:
                nodal_bp = 192.0 * eit_bp.solve(v1, v0, normalize=True, log_scale=False)
            else:
                nodal_bp = 192.0 * eit_bp.solve(v0, v0, normalize=True, log_scale=False)

            nodal_bp = np.real(nodal_bp).astype(np.float32)
            bp_grid = rasterize_to_grid(triang, xx, yy, nodal_bp, fill_value=0.0).astype(np.float32)

            nodal_mask = element_mask_to_nodal(mesh_obj, elem_mask)
            mask_grid_float = rasterize_to_grid(triang, xx, yy, nodal_mask, fill_value=0.0)
            gt_mask = (mask_grid_float > 0.5).astype(np.float32)

            # tensors
            volt_t = torch.from_numpy(delta_v).unsqueeze(0).to(device)                  # [1,K]
            bp_t = torch.from_numpy(bp_grid).unsqueeze(0).unsqueeze(0).to(device)       # [1,1,H,W]
            dom_t = torch.from_numpy(domain_grid).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

            # apply mode
            if input_mode == "bp_only":
                volt_in = torch.zeros_like(volt_t)
                bp_in = bp_t
            elif input_mode == "volt_only":
                volt_in = volt_t
                bp_in = torch.zeros_like(bp_t)
            else:
                volt_in = volt_t
                bp_in = bp_t

            logits = model(volt_in, bp_in, dom_t, bp_dropout_prob=0.0)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            pred = (prob > thresh).astype(np.float32)

            inter = float(np.sum(pred * gt_mask))
            union = float(np.sum(pred) + np.sum(gt_mask) - inter)
            iou = inter / (union + 1e-6)

            denom = float(np.sum(pred) + np.sum(gt_mask))
            dice = (2.0 * inter) / (denom + 1e-6)

            ious.append(iou)
            dices.append(dice)
            all_iou.append(iou)
            all_dice.append(dice)

            if SAVE_UNSEEN_VIZ and viz_count < VIZ_PER_SHAPE:
                visualize_unseen(bp_grid, gt_mask, pred, shape_type, viz_count, VIZ_DIR)
                viz_count += 1

        ious_np = np.asarray(ious, dtype=np.float32)
        dices_np = np.asarray(dices, dtype=np.float32)
        per_shape[shape_type] = {
            "iou_mean": float(np.mean(ious_np)) if len(ious_np) else float("nan"),
            "iou_std":  float(np.std(ious_np, ddof=1)) if len(ious_np) > 1 else 0.0,
            "dice_mean": float(np.mean(dices_np)) if len(dices_np) else float("nan"),
            "dice_std":  float(np.std(dices_np, ddof=1)) if len(dices_np) > 1 else 0.0,
            "n": int(len(ious_np)),
        }

    all_iou_np = np.asarray(all_iou, dtype=np.float32)
    all_dice_np = np.asarray(all_dice, dtype=np.float32)

    return {
        "iou_mean": float(np.mean(all_iou_np)) if len(all_iou_np) else float("nan"),
        "iou_std":  float(np.std(all_iou_np, ddof=1)) if len(all_iou_np) > 1 else 0.0,
        "dice_mean": float(np.mean(all_dice_np)) if len(all_dice_np) else float("nan"),
        "dice_std":  float(np.std(all_dice_np, ddof=1)) if len(all_dice_np) > 1 else 0.0,
        "per_shape": per_shape,
        "n": int(len(all_iou_np)),
    }


# =========================
# MAIN
# =========================
def main():
    print(f"Using device: {DEVICE}")
    print(f"Input mode: {INPUT_MODE}")
    print(f"Checkpoint: {CKPT_PATH}")

    train_dataset = EITHybridDataset(CSV_PATH, DATA_ROOT, split="train")
    test_dataset = EITHybridDataset(CSV_PATH, DATA_ROOT, split="test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    volt_dim = train_dataset.n_meas
    model = HybridEITNet(volt_dim=volt_dim, img_size=(GRID_SIZE, GRID_SIZE), feat_channels=16).to(DEVICE)

    # -------- load or train --------
    loaded = load_checkpoint_if_exists(model, CKPT_PATH)

    if not loaded:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        print(f"Voltage dim: {volt_dim}, grid: ({GRID_SIZE},{GRID_SIZE}), BP dropout: {BP_DROPOUT_PROB}")

        best_test_iou = -1.0
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_iou = train_one_epoch(
                model, train_loader, optimizer, DEVICE,
                bp_dropout_prob=BP_DROPOUT_PROB,
                input_mode=INPUT_MODE,
            )

            # quick eval at thresh=0.5 for checkpointing
            seen_res_05 = eval_model_seen(model, test_loader, DEVICE, INPUT_MODE, thresh=0.5)
            test_iou_05 = seen_res_05["iou_mean"]
            test_dice_05 = seen_res_05["dice_mean"]

            if epoch % PRINT_EVERY == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train loss: {train_loss:.4f}, IoU@0.5: {train_iou:.4f} | "
                    f"Test IoU@0.5: {test_iou_05:.4f}, Dice@0.5: {test_dice_05:.4f}"
                )

            if test_iou_05 > best_test_iou:
                best_test_iou = test_iou_05
                CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "test_iou": float(test_iou_05),
                        "test_dice": float(test_dice_05),
                        "input_mode": INPUT_MODE,
                    },
                    CKPT_PATH,
                )
                print(f"  -> New best test IoU@0.5: {test_iou_05:.4f}, saved to {CKPT_PATH}")

        print("Training finished.")
    else:
        print("[info] Skipping training (checkpoint already exists).")

    # -------- metrics (always run) --------
    print("\n==============================")
    print("EVALUATION ON SEEN TEST SPLIT")
    print("==============================")

    best_t, (best_iou_m, best_iou_s), (best_dice_m, best_dice_s) = find_best_threshold_on_seen(
        model, test_loader, DEVICE, INPUT_MODE, THRESHOLDS
    )
    print(
        f"[info] Best threshold on seen test: t={best_t:.2f} | "
        f"IoU={best_iou_m:.4f}±{best_iou_s:.4f} | Dice={best_dice_m:.4f}±{best_dice_s:.4f}"
    )

    seen_res = eval_model_seen(model, test_loader, DEVICE, INPUT_MODE, thresh=best_t)
    print(
        f"Seen test (overall): "
        f"IoU={seen_res['iou_mean']:.4f}±{seen_res['iou_std']:.4f}, "
        f"Dice={seen_res['dice_mean']:.4f}±{seen_res['dice_std']:.4f} "
        f"(N={seen_res['n']})"
    )

    print("\nSeen test (per-shape):")
    for s in sorted(seen_res["per_shape"].keys()):
        d = seen_res["per_shape"][s]
        print(
            f"  {s:>10s} | N={d['n']:4d} | "
            f"IoU={d['iou_mean']:.4f}±{d['iou_std']:.4f} | "
            f"Dice={d['dice_mean']:.4f}±{d['dice_std']:.4f}"
        )

    if RUN_UNSEEN_EVAL:
        print("\n==============================")
        print("EVALUATION ON UNSEEN SHAPES")
        print("==============================")
        unseen_res = eval_unseen_shapes(model, DEVICE, INPUT_MODE, thresh=best_t)
        print(
            f"Unseen (overall): "
            f"IoU={unseen_res['iou_mean']:.4f}±{unseen_res['iou_std']:.4f}, "
            f"Dice={unseen_res['dice_mean']:.4f}±{unseen_res['dice_std']:.4f} "
            f"(N={unseen_res['n']})"
        )
        print("\nUnseen (per-shape):")
        for s in UNSEEN_SHAPES:
            d = unseen_res["per_shape"][s]
            print(
                f"  {s:>10s} | N={d['n']:4d} | "
                f"IoU={d['iou_mean']:.4f}±{d['iou_std']:.4f} | "
                f"Dice={d['dice_mean']:.4f}±{d['dice_std']:.4f}"
            )

        if SAVE_UNSEEN_VIZ:
            print(f"\n[info] Unseen visualizations saved to: {VIZ_DIR}")


if __name__ == "__main__":
    main()
