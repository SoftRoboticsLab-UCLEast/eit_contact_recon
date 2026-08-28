#!/usr/bin/env python3
"""
Real EIT tactile shape reconstruction training script (REAL) with SIM-style metrics.

What this script does (mirrors sim pipeline):
- Trains on REAL train split (seen shapes).
- Creates an internal hold-out split from train data -> "seen-test".
- Evaluates:
    (A) Seen-test (held-out from train split): overall + per-shape mean/std IoU & Dice
    (B) Unseen-test (separate real test split): overall + per-shape mean/std IoU & Dice
- Finds best threshold on seen-test (grid search), then uses it for both seen/unseen eval.
- Checkpoint behavior:
    - If checkpoint exists: load and skip training, BUT still run all evaluations.

Dataset assumptions:
- You already produced:
    real_train_with_masks.csv  (5 seen shapes)
    real_test_with_masks.csv   (3 unseen shapes)
- mask_path is relative to DATA_ROOT (same idea as sim).

Inputs:
- Voltages are stored in columns eit_0 ... eit_255 (or similar)
- Baseline is saved in BASELINE_PATH (npy or csv) of length = number of eit_* cols.

Model:
- Same HybridEITNet as before (MLP volt -> feature planes + UNet on [BP, domain, planes])

Notes:
- We keep your "valid_channel_mask" logic (drop channels that are always-zero AND baseline=0),
  and we use THE SAME mask for seen and unseen so formatting is consistent.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.tri as mtri

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol
import pyeit.eit.bp as bp


# =========================
# CONFIG
# =========================
CONFIG = {
    # These are the consolidated split CSVs you generated
    "TRAIN_CSV": "./real_dataset/real_train_with_masks.csv",
    "TEST_CSV": "./real_dataset/real_test_with_masks.csv",
    "DATA_ROOT": "./real_dataset/gt_masks",   # used to resolve mask_path

    # Baseline file saved earlier
    "BASELINE_PATH": "./real_dataset/eit_baseline.npy",

    "BATCH_SIZE": 16,
    "NUM_EPOCHS": 100,
    "LEARNING_RATE": 1e-3,
    "BP_DROPOUT_PROB": 0.5,
    "GRID_SIZE": 64,

    # internal holdout on TRAIN_CSV (seen-test)
    "SEEN_HOLDOUT_FRACTION": 0.2,
    "RANDOM_SEED": 123,

    "PRINT_EVERY": 1,

    # EIT geometry
    "N_EL": 16,
    "MESH_H0": 0.04,

    # "hybrid" | "volt_only" | "bp_only"
    "INPUT_MODE": "bp_only",

    # Threshold search
    "THRESHOLDS": np.linspace(0.05, 0.95, 19),

    # checkpoint
    "CKPT_PATH": None,  # auto-filled below
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# EIT / GEOMETRY HELPERS
# =========================
def setup_eit_for_bp(n_el=16, h0=0.04):
    mesh_obj = mesh.create(n_el, h0=h0)
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
    eit_bp = bp.BP(mesh_obj, protocol_obj)
    eit_bp.setup(weight="none")
    return mesh_obj, protocol_obj, eit_bp


def make_grid(mesh_obj, grid_size):
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


def domain_mask_grid(grid_size):
    lin = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(lin, lin)
    return ((xx**2 + yy**2) <= 1.0).astype(np.float32)


def load_baseline(baseline_path: str) -> np.ndarray:
    p = Path(baseline_path)
    if p.suffix.lower() == ".npy":
        v0 = np.load(p)
    else:
        v0 = np.loadtxt(p, delimiter=",")
    v0 = np.asarray(v0, dtype=float)
    if v0.ndim != 1:
        raise ValueError("Baseline must be 1D.")
    return v0.astype(np.float32)


# =========================
# DATASET (REAL)
# =========================
class RealEITHybridDataset(Dataset):
    """
    IMPORTANT: This dataset uses a precomputed valid_channel_mask so that
    seen/unseen formatting is consistent.

    BP is computed using the same reduced vector (valid channels) to match what
    you already had working.
    """
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        baseline_full: np.ndarray,
        valid_channel_mask: np.ndarray,
        grid_size=64,
        n_el=16,
        mesh_h0=0.04,
    ):
        self.data_root = Path(data_root)
        self.grid_size = int(grid_size)

        self.df = pd.read_csv(csv_path)

        # EIT columns
        self.eit_cols = [c for c in self.df.columns if c.startswith("eit_")]
        self.eit_cols.sort(key=lambda s: int(s.split("_")[1]))
        if len(self.eit_cols) == 0:
            raise RuntimeError(f"No eit_* columns found in {csv_path}")

        if baseline_full.size != len(self.eit_cols):
            raise ValueError(
                f"Baseline length {baseline_full.size} != number of EIT cols {len(self.eit_cols)}"
            )

        self.v0_full = baseline_full.astype(np.float32)
        self.valid_channel_mask = valid_channel_mask.astype(bool)

        self.v0 = self.v0_full[self.valid_channel_mask].astype(np.float32)
        self.raw_channel_count = len(self.eit_cols)
        self.mlp_channel_count = int(self.valid_channel_mask.sum())

        # BP objects
        self.mesh_obj, self.protocol_obj, self.eit_bp = setup_eit_for_bp(n_el=n_el, h0=mesh_h0)
        self.triang, self.xx, self.yy = make_grid(self.mesh_obj, self.grid_size)
        self.domain_grid = domain_mask_grid(self.grid_size)

        # drop rows that are entirely NaN across EIT
        eit_vals = self.df[self.eit_cols].to_numpy(dtype=float)
        valid_rows = ~np.all(np.isnan(eit_vals), axis=1)
        self.df = self.df.loc[valid_rows].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # full vector [256], replace NaNs with baseline (stabilizes BP & delta)
        vals_full = row[self.eit_cols].to_numpy(dtype=float)
        vals_full = np.where(np.isnan(vals_full), self.v0_full, vals_full).astype(np.float32)

        # reduced vector [K]
        vals = vals_full[self.valid_channel_mask]
        delta = vals - self.v0

        # BP reconstruction (reduced)
        sum_abs_diff = float(np.sum(np.abs(delta)))
        if sum_abs_diff > 1e-6:
            v1 = self.v0 + delta
            nodal_bp = 192.0 * self.eit_bp.solve(v1, self.v0, normalize=False, log_scale=False)
        else:
            nodal_bp = 192.0 * self.eit_bp.solve(self.v0, self.v0, normalize=False, log_scale=False)

        nodal_bp = np.real(nodal_bp).astype(np.float32)
        bp_grid = rasterize_to_grid(self.triang, self.xx, self.yy, nodal_bp, fill_value=0.0).astype(np.float32)

        # GT mask
        mask_rel = row["mask_path"]
        mask_path = self.data_root / mask_rel
        mask_img = Image.open(mask_path).convert("L")
        mask_arr = (np.array(mask_img, dtype=np.uint8) > 127).astype(np.float32)

        sample_id = row.get("sample_id", row.get("t", idx))
        shape_type = row.get("shape_type", "unknown")

        return {
            "volt": torch.from_numpy(delta),                         # [K]
            "bp": torch.from_numpy(bp_grid).unsqueeze(0),            # [1,H,W]
            "domain": torch.from_numpy(self.domain_grid).unsqueeze(0),# [1,H,W]
            "mask": torch.from_numpy(mask_arr).unsqueeze(0),         # [1,H,W]
            "sample_id": str(sample_id),
            "shape_type": str(shape_type),
        }


def compute_valid_channel_mask_from_train(train_csv: str, baseline_full: np.ndarray) -> np.ndarray:
    """
    Match your previous logic:
    drop channel if (always zero across train data) AND (baseline is zero).
    """
    df = pd.read_csv(train_csv)
    eit_cols = [c for c in df.columns if c.startswith("eit_")]
    eit_cols.sort(key=lambda s: int(s.split("_")[1]))
    if len(eit_cols) == 0:
        raise RuntimeError("No eit_* columns found in TRAIN_CSV")

    if baseline_full.size != len(eit_cols):
        raise ValueError(
            f"Baseline length {baseline_full.size} != EIT cols {len(eit_cols)}"
        )

    vals = df[eit_cols].to_numpy(dtype=float)
    vals = np.where(np.isnan(vals), 0.0, vals)

    zero_across_train = np.all(vals == 0.0, axis=0)
    zero_in_baseline = (baseline_full == 0.0)

    drop = zero_across_train & zero_in_baseline
    valid = ~drop
    return valid.astype(bool)


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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

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
    def __init__(self, volt_dim, feat_channels=16):
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
# METRICS (per-sample tensors)
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


def mean_std(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    mu = float(np.mean(x))
    # use sample std (ddof=1) when N>1, else 0
    if x.size > 1:
        sd = float(np.std(x, ddof=1))
    else:
        sd = 0.0
    return mu, sd


# =========================
# TRAINING
# =========================
def train_one_epoch(model, loader, optimizer, device, bp_dropout_prob, input_mode):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n = 0

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

        # combine BCE + Dice loss (dice computed at thresh=0.5 just for loss proxy)
        loss = 0.5 * bce(logits, mask) + 0.5 * (1.0 - dice_from_logits(logits, mask, thresh=0.5).mean())
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_dataset(model, loader, device, input_mode, thresh=0.5):
    """
    Returns:
      overall mean/std for IoU and Dice (across samples)
      per-shape mean/std for IoU and Dice (across samples)
    """
    model.eval()

    all_iou = []
    all_dice = []
    per_shape = {}  # s -> {"iou":[], "dice":[]}

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

        iou_b = iou_from_logits(logits, mask, thresh=thresh).detach().cpu().numpy()
        dice_b = dice_from_logits(logits, mask, thresh=thresh).detach().cpu().numpy()

        all_iou.extend(list(iou_b))
        all_dice.extend(list(dice_b))

        shapes = batch["shape_type"]
        for i, s in enumerate(shapes):
            s = str(s)
            per_shape.setdefault(s, {"iou": [], "dice": []})
            per_shape[s]["iou"].append(float(iou_b[i]))
            per_shape[s]["dice"].append(float(dice_b[i]))

    all_iou = np.asarray(all_iou, dtype=float)
    all_dice = np.asarray(all_dice, dtype=float)

    out = {
        "overall": {
            "n": int(all_iou.size),
            "iou_mean": mean_std(all_iou)[0],
            "iou_std": mean_std(all_iou)[1],
            "dice_mean": mean_std(all_dice)[0],
            "dice_std": mean_std(all_dice)[1],
        },
        "per_shape": {}
    }

    for s, d in per_shape.items():
        iou_arr = np.asarray(d["iou"], dtype=float)
        dice_arr = np.asarray(d["dice"], dtype=float)
        mi, si = mean_std(iou_arr)
        md, sd = mean_std(dice_arr)
        out["per_shape"][s] = {
            "n": int(iou_arr.size),
            "iou_mean": mi,
            "iou_std": si,
            "dice_mean": md,
            "dice_std": sd,
        }

    return out


def find_best_threshold(model, loader, device, input_mode, thresholds):
    best_t = None
    best_iou = -1.0
    best_dice = -1.0

    for t in thresholds:
        res = eval_dataset(model, loader, device, input_mode, thresh=float(t))
        miou = res["overall"]["iou_mean"]
        mdice = res["overall"]["dice_mean"]
        if miou > best_iou:
            best_iou = miou
            best_dice = mdice
            best_t = float(t)

    return best_t, best_iou, best_dice


def load_checkpoint_if_exists(model, ckpt_path: Path) -> bool:
    if not ckpt_path.exists():
        return False
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[info] Found checkpoint: {ckpt_path}")
    print(f"[info] Loaded epoch={ckpt.get('epoch','?')} seen_iou@0.5={ckpt.get('seen_iou_05','N/A')}")
    return True


# =========================
# MAIN
# =========================
def main():
    cfg = dict(CONFIG)

    # auto ckpt name
    if cfg["CKPT_PATH"] is None:
        cfg["CKPT_PATH"] = str(Path(cfg["DATA_ROOT"]) / f"best_real_hybrid_{cfg['INPUT_MODE']}.pt")
    ckpt_path = Path(cfg["CKPT_PATH"])

    print(f"Using device: {DEVICE}")
    print(f"INPUT_MODE: {cfg['INPUT_MODE']}")
    print(f"TRAIN_CSV:  {cfg['TRAIN_CSV']}")
    print(f"TEST_CSV:   {cfg['TEST_CSV']}")
    print(f"CKPT_PATH:  {ckpt_path}")

    # baseline + channel mask computed from TRAIN split only
    baseline_full = load_baseline(cfg["BASELINE_PATH"])
    valid_mask = compute_valid_channel_mask_from_train(cfg["TRAIN_CSV"], baseline_full)

    print(f"[info] Raw channels: {baseline_full.size}")
    print(f"[info] Valid channels (MLP & BP): {int(valid_mask.sum())}")

    # Build datasets
    full_train_ds = RealEITHybridDataset(
        csv_path=cfg["TRAIN_CSV"],
        data_root=cfg["DATA_ROOT"],
        baseline_full=baseline_full,
        valid_channel_mask=valid_mask,
        grid_size=cfg["GRID_SIZE"],
        n_el=cfg["N_EL"],
        mesh_h0=cfg["MESH_H0"],
    )

    unseen_test_ds = RealEITHybridDataset(
        csv_path=cfg["TEST_CSV"],
        data_root=cfg["DATA_ROOT"],
        baseline_full=baseline_full,
        valid_channel_mask=valid_mask,
        grid_size=cfg["GRID_SIZE"],
        n_el=cfg["N_EL"],
        mesh_h0=cfg["MESH_H0"],
    )

    # Internal holdout split for SEEN evaluation (from train split)
    rng = np.random.default_rng(cfg["RANDOM_SEED"])
    idx = np.arange(len(full_train_ds))
    rng.shuffle(idx)
    n_hold = int(round(cfg["SEEN_HOLDOUT_FRACTION"] * len(idx)))
    seen_test_idx = idx[:n_hold]
    seen_train_idx = idx[n_hold:]

    seen_train_ds = Subset(full_train_ds, seen_train_idx)
    seen_test_ds = Subset(full_train_ds, seen_test_idx)

    seen_train_loader = DataLoader(seen_train_ds, batch_size=cfg["BATCH_SIZE"], shuffle=True, num_workers=0)
    seen_test_loader  = DataLoader(seen_test_ds,  batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=0)
    unseen_loader     = DataLoader(unseen_test_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=0)

    volt_dim = int(valid_mask.sum())
    model = HybridEITNet(volt_dim=volt_dim, feat_channels=16).to(DEVICE)

    # -------- load or train --------
    loaded = load_checkpoint_if_exists(model, ckpt_path)
    if not loaded:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LEARNING_RATE"])

        print(f"Total TRAIN samples (seen shapes): {len(full_train_ds)}")
        print(f"  seen-train: {len(seen_train_ds)} | seen-test: {len(seen_test_ds)}")
        print(f"Total UNSEEN test samples (unseen shapes): {len(unseen_test_ds)}")
        print(f"Voltage dim: {volt_dim} | Grid: {cfg['GRID_SIZE']}x{cfg['GRID_SIZE']} | BP dropout: {cfg['BP_DROPOUT_PROB']}")

        best_seen_iou_05 = -1.0

        for epoch in range(1, cfg["NUM_EPOCHS"] + 1):
            train_loss = train_one_epoch(
                model, seen_train_loader, optimizer, DEVICE,
                bp_dropout_prob=cfg["BP_DROPOUT_PROB"],
                input_mode=cfg["INPUT_MODE"],
            )

            # checkpointing criterion: seen-test IoU at thresh=0.5 (like sim)
            seen_res_05 = eval_dataset(model, seen_test_loader, DEVICE, cfg["INPUT_MODE"], thresh=0.5)
            seen_iou_05 = seen_res_05["overall"]["iou_mean"]
            seen_dice_05 = seen_res_05["overall"]["dice_mean"]

            if epoch % cfg["PRINT_EVERY"] == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Seen-test IoU@0.5: {seen_iou_05:.4f} Dice@0.5: {seen_dice_05:.4f}"
                )

            if seen_iou_05 > best_seen_iou_05:
                best_seen_iou_05 = seen_iou_05
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "seen_iou_05": float(seen_iou_05),
                        "seen_dice_05": float(seen_dice_05),
                        "input_mode": cfg["INPUT_MODE"],
                        "volt_dim": volt_dim,
                    },
                    ckpt_path,
                )
                print(f"  -> New best seen-test IoU@0.5: {seen_iou_05:.4f} saved to {ckpt_path}")

        print("[info] Training finished.")
    else:
        print("[info] Skipping training (checkpoint already exists).")

    # -------- evaluation (always run) --------
    print("\n==============================")
    print("REAL: SEEN TEST (HOLD-OUT FROM TRAIN SPLIT)")
    print("==============================")

    best_t, best_iou, best_dice = find_best_threshold(
        model, seen_test_loader, DEVICE, cfg["INPUT_MODE"], cfg["THRESHOLDS"]
    )
    print(f"[info] Best threshold on seen-test: t={best_t:.2f}")
    print(f"[info] Seen-test mean IoU={best_iou:.4f} | mean Dice={best_dice:.4f}")

    seen_res = eval_dataset(model, seen_test_loader, DEVICE, cfg["INPUT_MODE"], thresh=best_t)
    o = seen_res["overall"]
    print(f"\nSeen-test overall (N={o['n']}):")
    print(f"  IoU  = {o['iou_mean']:.4f} ± {o['iou_std']:.4f}")
    print(f"  Dice = {o['dice_mean']:.4f} ± {o['dice_std']:.4f}")

    print("\nSeen-test per-shape:")
    for s in sorted(seen_res["per_shape"].keys()):
        d = seen_res["per_shape"][s]
        print(f"  {s:>12s} | N={d['n']:4d} | IoU={d['iou_mean']:.4f}±{d['iou_std']:.4f} | Dice={d['dice_mean']:.4f}±{d['dice_std']:.4f}")

    print("\n==============================")
    print("REAL: UNSEEN TEST (SEPARATE TEST SPLIT)")
    print("==============================")

    unseen_res = eval_dataset(model, unseen_loader, DEVICE, cfg["INPUT_MODE"], thresh=best_t)
    o = unseen_res["overall"]
    print(f"\nUnseen-test overall (N={o['n']}):")
    print(f"  IoU  = {o['iou_mean']:.4f} ± {o['iou_std']:.4f}")
    print(f"  Dice = {o['dice_mean']:.4f} ± {o['dice_std']:.4f}")

    print("\nUnseen-test per-shape:")
    for s in sorted(unseen_res["per_shape"].keys()):
        d = unseen_res["per_shape"][s]
        print(f"  {s:>12s} | N={d['n']:4d} | IoU={d['iou_mean']:.4f}±{d['iou_std']:.4f} | Dice={d['dice_mean']:.4f}±{d['dice_std']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
