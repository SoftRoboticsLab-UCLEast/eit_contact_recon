#!/usr/bin/env python3
"""
Evaluation & visualization for REAL Hybrid EIT shape reconstruction.

Matches train_real_eit_hybrid.py:

    - "hybrid"    : voltages + BP + domain
    - "volt_only" : voltages + domain, BP is zeroed
    - "bp_only"   : BP + domain, voltages are zeroed

- Loads best_real_hybrid_{INPUT_MODE}.pt
- Evaluates on the full real dataset (all rows in DATA_CSV)
- Saves visualizations:
    BP reconstruction | ground truth | prediction
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Import dataset & model from the real training script
from train_real_eit_hybrid import (
    RealEITHybridDataset,
    HybridEITNet,
    CONFIG as TRAIN_CONFIG,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MAX_VIZ_IMAGES = 100      # how many samples to visualize

# --- must match train_real_eit_hybrid.py ---
# "hybrid"    : voltages + BP + domain
# "volt_only" : voltages + domain, BP is zeroed
# "bp_only"   : BP + domain, voltages are zeroed
INPUT_MODE = TRAIN_CONFIG.get("INPUT_MODE", "hybrid")

CKPT_NAME = f"best_real_hybrid_{INPUT_MODE}.pt"


# ============== metrics (same definitions as training) ==============
def dice_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()

    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    dice = 2.0 * intersection / union
    return dice


def iou_from_logits(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection + eps
    iou = intersection / union
    return iou


# ============== evaluation ==============
def evaluate_and_visualize():
    print(f"Using device: {DEVICE}")
    print(f"Input mode: {INPUT_MODE}")

    data_root = TRAIN_CONFIG["DATA_ROOT"]
    csv_path = TRAIN_CONFIG["DATA_CSV"]
    baseline_path = TRAIN_CONFIG["BASELINE_PATH"]
    grid_size = TRAIN_CONFIG["GRID_SIZE"]
    n_el = TRAIN_CONFIG["N_EL"]
    mesh_h0 = TRAIN_CONFIG["MESH_H0"]

    # ---- dataset & loader (full real dataset) ----
    dataset = RealEITHybridDataset(
        csv_path=csv_path,
        data_root=data_root,
        baseline_path=baseline_path,
        grid_size=grid_size,
        n_el=n_el,
        mesh_h0=mesh_h0,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    volt_dim = dataset.mlp_channel_count
    img_size = (grid_size, grid_size)

    # ---- load model & checkpoint ----
    model = HybridEITNet(volt_dim=volt_dim, img_size=img_size, feat_channels=16)
    ckpt_path = Path(data_root) / CKPT_NAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(
        f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
        f"with test IoU {ckpt.get('test_iou', 'N/A'):.4f}"
    )

    # ---- accumulate metrics ----
    all_iou = []
    all_dice = []

    # optional: per-shape metrics
    per_shape_iou = {}
    per_shape_dice = {}
    per_shape_count = {}

    # ---- visualization setup ----
    viz_dir = Path(data_root) / f"real_test_viz_{INPUT_MODE}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            volt = batch["volt"].to(DEVICE)       # [B,K]
            bp = batch["bp"].to(DEVICE)           # [B,1,H,W]
            domain = batch["domain"].to(DEVICE)   # [B,1,H,W]
            mask = batch["mask"].to(DEVICE)       # [B,1,H,W]

            # --- apply same ablation logic as training ---
            if INPUT_MODE == "bp_only":
                volt_in = torch.zeros_like(volt)
                bp_in = bp
                effective_bp_dropout = 0.0
            elif INPUT_MODE == "volt_only":
                volt_in = volt
                bp_in = torch.zeros_like(bp)
                effective_bp_dropout = 0.0
            else:  # "hybrid"
                volt_in = volt
                bp_in = bp
                effective_bp_dropout = 0.0  # no dropout at eval

            logits = model(volt_in, bp_in, domain, bp_dropout_prob=effective_bp_dropout)

            # metrics (per-sample)
            iou = iou_from_logits(logits, mask)   # [B]
            dice = dice_from_logits(logits, mask) # [B]

            all_iou.append(iou.cpu())
            all_dice.append(dice.cpu())

            # per-shape aggregation
            shape_types = batch["shape_type"]  # list of strings
            for i, shape in enumerate(shape_types):
                iou_i = iou[i].item()
                dice_i = dice[i].item()
                per_shape_iou.setdefault(shape, 0.0)
                per_shape_dice.setdefault(shape, 0.0)
                per_shape_count.setdefault(shape, 0)
                per_shape_iou[shape] += iou_i
                per_shape_dice[shape] += dice_i
                per_shape_count[shape] += 1

            # visualization for a subset
            if saved < MAX_VIZ_IMAGES:
                probs = torch.sigmoid(logits).cpu().numpy()  # [B,1,H,W]
                bp_np = bp.cpu().numpy()
                mask_np = mask.cpu().numpy()
                for j in range(volt.shape[0]):
                    if saved >= MAX_VIZ_IMAGES:
                        break

                    sample_id = batch["sample_id"][j]
                    shape_type = batch["shape_type"][j]

                    bp_img = bp_np[j, 0]        # [H,W]
                    gt_mask = mask_np[j, 0]     # [H,W]
                    pred_mask = (probs[j, 0] > 0.5).astype(np.float32)

                    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                    ax0, ax1, ax2 = axes

                    im0 = ax0.imshow(bp_img, cmap="viridis")
                    ax0.set_title("BP recon")
                    ax0.axis("off")
                    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

                    ax1.imshow(gt_mask, cmap="gray")
                    ax1.set_title("Ground truth")
                    ax1.axis("off")

                    ax2.imshow(pred_mask, cmap="gray")
                    ax2.set_title("Prediction")
                    ax2.axis("off")

                    fig.suptitle(
                        f"{sample_id} | shape={shape_type} | mode={INPUT_MODE}",
                        fontsize=10,
                    )
                    plt.tight_layout()

                    out_path = viz_dir / f"{sample_id}_{INPUT_MODE}.png"
                    fig.savefig(out_path, dpi=120)
                    plt.close(fig)

                    saved += 1

    # ---- summarize metrics ----
    all_iou = torch.cat(all_iou).numpy()
    all_dice = torch.cat(all_dice).numpy()

    mean_iou = float(all_iou.mean())
    mean_dice = float(all_dice.mean())

    print("\n=== Overall REAL-data metrics (full dataset) ===")
    print(f"Mean IoU:   {mean_iou:.4f}")
    print(f"Mean Dice:  {mean_dice:.4f}")

    print("\n=== Per-shape metrics ===")
    for shape in sorted(per_shape_count.keys()):
        cnt = per_shape_count[shape]
        miou = per_shape_iou[shape] / cnt
        mdice = per_shape_dice[shape] / cnt
        print(f"{shape:>13s} | N={cnt:4d} | IoU={miou:.4f} | Dice={mdice:.4f}")

    print(f"\nSaved {saved} visualizations in: {viz_dir}")


if __name__ == "__main__":
    evaluate_and_visualize()
