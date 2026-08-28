#!/usr/bin/env python3
"""
Generate tripcolor-style plots for REAL EIT data with:

    BP reconstruction | Ground truth | Model prediction

- Loads real-trained HybridEITNet (best_real_hybrid_{INPUT_MODE}.pt)
- Uses RealEITHybridDataset to compute:
    - delta_v (reduced channels)
    - BP reconstruction grid
    - GT masks
- Runs the model to obtain predicted contact probability maps.
- For each selected sample:
    - Maps BP, GT mask, and prediction from [H,W] grids to FEM nodes.
    - Plots tripcolor on the FEM triangulation side-by-side.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from train_real_eit_hybrid import (
    RealEITHybridDataset,
    HybridEITNet,
    CONFIG as TRAIN_CONFIG,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_IMAGES = 100  # how many samples to export

# Match train_real_eit_hybrid.py
INPUT_MODE = TRAIN_CONFIG.get("INPUT_MODE", "hybrid")
CKPT_NAME = f"best_real_hybrid_{INPUT_MODE}.pt"


def grid_value_at_xy(grid, x, y, grid_size):
    """
    Sample the [H,W] grid at continuous coordinates (x, y) in [-1, 1]^2.

    grid: 2D array [H,W]
    x, y: coordinates in [-1, 1]
    returns: scalar grid value at nearest neighbor.
    """
    H = W = grid_size
    # Map x from [-1,1] -> [0, W-1], y from [-1,1] -> [0, H-1]
    j = int(np.round((x + 1.0) * 0.5 * (W - 1)))
    i = int(np.round((y + 1.0) * 0.5 * (H - 1)))
    i = np.clip(i, 0, H - 1)
    j = np.clip(j, 0, W - 1)
    return grid[i, j]


def make_tripcolor_bp_pred_gt_real():
    print(f"Using device: {DEVICE}")
    print(f"Input mode: {INPUT_MODE}")

    data_root = TRAIN_CONFIG["DATA_ROOT"]
    csv_path = TRAIN_CONFIG["DATA_CSV"]
    baseline_path = TRAIN_CONFIG["BASELINE_PATH"]
    grid_size = TRAIN_CONFIG["GRID_SIZE"]
    n_el = TRAIN_CONFIG["N_EL"]
    mesh_h0 = TRAIN_CONFIG["MESH_H0"]

    # --- dataset (full real dataset) ---
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

    # --- model & checkpoint ---
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

    # --- FEM geometry for tripcolor ---
    mesh_obj = dataset.mesh_obj          # same mesh used for BP
    tri = mesh_obj.element               # (n_elems, 3)
    pts = mesh_obj.node                  # (n_nodes, 2) -> x,y
    x_nodes = pts[:, 0]
    y_nodes = pts[:, 1]

    # --- output directory ---
    out_dir = Path(data_root) / f"real_tripcolor_bp_gt_pred_{INPUT_MODE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving tripcolor images to: {out_dir}")

    saved = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if saved >= MAX_IMAGES:
                break

            volt = batch["volt"].to(DEVICE)       # [B,K]
            bp = batch["bp"].to(DEVICE)           # [B,1,H,W]  BP grid
            domain = batch["domain"].to(DEVICE)   # [B,1,H,W]
            mask = batch["mask"].to(DEVICE)       # [B,1,H,W]

            # match training input modes
            if INPUT_MODE == "bp_only":
                volt_in = torch.zeros_like(volt)
                bp_in = bp
                eff_bp_dropout = 0.0
            elif INPUT_MODE == "volt_only":
                volt_in = volt
                bp_in = torch.zeros_like(bp)
                eff_bp_dropout = 0.0
            else:  # "hybrid"
                volt_in = volt
                bp_in = bp
                eff_bp_dropout = 0.0

            logits = model(volt_in, bp_in, domain, bp_dropout_prob=eff_bp_dropout)
            probs = torch.sigmoid(logits).cpu().numpy()  # [B,1,H,W]
            mask_np = mask.cpu().numpy()                 # [B,1,H,W]
            bp_np = bp.cpu().numpy()                     # [B,1,H,W]

            B = volt.shape[0]
            for j in range(B):
                if saved >= MAX_IMAGES:
                    break

                sample_id = batch["sample_id"][j]
                shape_type = batch["shape_type"][j]

                bp_grid = bp_np[j, 0]        # [H,W]
                gt_grid = mask_np[j, 0]      # [H,W]
                pred_grid = probs[j, 0]      # [H,W]

                # --- map grids to node values ---
                nodal_bp = np.zeros_like(x_nodes)
                nodal_gt = np.zeros_like(x_nodes)
                nodal_pred = np.zeros_like(x_nodes)

                for n in range(len(x_nodes)):
                    x = x_nodes[n]
                    y = y_nodes[n]
                    nodal_bp[n] = grid_value_at_xy(bp_grid, x, y, grid_size)
                    nodal_gt[n] = grid_value_at_xy(gt_grid, x, y, grid_size)
                    nodal_pred[n] = grid_value_at_xy(pred_grid, x, y, grid_size)

                # Force GT to be exactly 0 or 1
                nodal_gt = (nodal_gt > 0.5).astype(float)


                # --- tripcolor plot: 3 panels BP | GT | Pred ---
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                ax0, ax1, ax2 = axes

                # BP reconstruction
                t0 = ax1.tripcolor(
                    x_nodes, y_nodes, tri, nodal_bp,
                    shading="gouraud", cmap="viridis"
                )
                ax1.set_aspect("equal")
                ax1.set_title("BP reconstruction")
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                fig.colorbar(t0, ax=ax1, fraction=0.046, pad=0.04)

                # Ground truth mask
                t1 = ax0.tripcolor(
                    x_nodes, y_nodes, tri, nodal_gt,
                    shading="flat", cmap="gray"
                )
                ax0.set_aspect("equal")
                ax0.set_title("Ground truth")
                ax0.set_xlabel("x")
                ax0.set_ylabel("y")
                fig.colorbar(t1, ax=ax0, fraction=0.046, pad=0.04)

                # Model prediction (probability)
                t2 = ax2.tripcolor(
                    x_nodes, y_nodes, tri, nodal_pred,
                    shading="gouraud", cmap="viridis"
                )
                ax2.set_aspect("equal")
                ax2.set_title("Prediction")
                ax2.set_xlabel("x")
                ax2.set_ylabel("y")
                fig.colorbar(t2, ax=ax2, fraction=0.046, pad=0.04)

                fig.suptitle(
                    f"{sample_id} | shape={shape_type} | mode={INPUT_MODE}",
                    fontsize=10,
                )
                plt.tight_layout()

                out_path = out_dir / f"{sample_id}_tripcolor_bp_gt_pred_{INPUT_MODE}.png"
                fig.savefig(out_path, dpi=150)
                plt.close(fig)

                saved += 1

    print(f"Done. Saved {saved} tripcolor BP/GT/pred images in {out_dir}")


if __name__ == "__main__":
    make_tripcolor_bp_pred_gt_real()
