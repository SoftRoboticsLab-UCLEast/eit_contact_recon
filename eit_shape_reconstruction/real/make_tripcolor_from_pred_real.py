#!/usr/bin/env python3
"""
Generate tripcolor-style plots from REAL EIT shape reconstruction predictions.

- Loads real-trained HybridEITNet (best_real_hybrid_{INPUT_MODE}.pt)
- Uses RealEITHybridDataset to compute:
    - delta_v (reduced channels)
    - BP reconstruction
    - GT masks
- Runs the model to obtain predicted contact probability maps.
- For each selected sample:
    - Maps the predicted grid [H,W] to the mesh nodes.
    - Plots tripcolor on the FEM triangulation (similar look to PyEIT).
    - Optionally also tripcolor of the GT mask.
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


def make_tripcolor_from_pred_real():
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
    out_dir = Path(data_root) / f"real_tripcolor_pred_{INPUT_MODE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving tripcolor images to: {out_dir}")

    saved = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if saved >= MAX_IMAGES:
                break

            volt = batch["volt"].to(DEVICE)       # [B,K]
            bp = batch["bp"].to(DEVICE)           # [B,1,H,W]
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

            B = volt.shape[0]
            for j in range(B):
                if saved >= MAX_IMAGES:
                    break

                sample_id = batch["sample_id"][j]
                shape_type = batch["shape_type"][j]

                pred_grid = probs[j, 0]      # [H,W]
                gt_grid = mask_np[j, 0]      # [H,W]

                # --- map predicted grid to node values ---
                nodal_pred = np.zeros_like(x_nodes)
                nodal_gt = np.zeros_like(x_nodes)

                for n in range(len(x_nodes)):
                    x = x_nodes[n]
                    y = y_nodes[n]
                    nodal_pred[n] = grid_value_at_xy(pred_grid, x, y, grid_size)
                    nodal_gt[n] = grid_value_at_xy(gt_grid, x, y, grid_size)

                # --- tripcolor plot ---
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                ax0, ax1 = axes

                # Prediction
                t0 = ax0.tripcolor(
                    x_nodes, y_nodes, tri, nodal_pred,
                    shading="gouraud", cmap="viridis"
                )
                ax0.set_aspect("equal")
                ax0.set_title("Prediction (probability)")
                ax0.set_xlabel("x")
                ax0.set_ylabel("y")
                fig.colorbar(t0, ax=ax0, fraction=0.046, pad=0.04)

                # Ground truth
                t1 = ax1.tripcolor(
                    x_nodes, y_nodes, tri, nodal_gt,
                    shading="flat", cmap="gray"
                )
                ax1.set_aspect("equal")
                ax1.set_title("Ground truth (mask)")
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                fig.colorbar(t1, ax=ax1, fraction=0.046, pad=0.04)

                fig.suptitle(
                    f"{sample_id} | shape={shape_type} | mode={INPUT_MODE}",
                    fontsize=10,
                )
                plt.tight_layout()

                out_path = out_dir / f"{sample_id}_tripcolor_{INPUT_MODE}.png"
                fig.savefig(out_path, dpi=150)
                plt.close(fig)

                saved += 1

    print(f"Done. Saved {saved} tripcolor prediction images in {out_dir}")


if __name__ == "__main__":
    make_tripcolor_from_pred_real()
