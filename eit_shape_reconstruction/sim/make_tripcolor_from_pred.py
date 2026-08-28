#!/usr/bin/env python3
"""
Generate EIT-style circular tripcolor plots from hybrid model predictions.

- Loads best_hybrid_model.pt
- Runs on the test set
- For each sample:
    - predicted prob map (sigmoid)
    - multiply by contrast (to mimic conductivity level)
    - Gaussian blur to soften edges
    - interpolate grid values back to mesh nodes
    - plot with ax.tripcolor (true circular EIT-style plot)

Output folder: eit_dataset/pred_tripcolor/
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

import torch
from torch.utils.data import DataLoader

import matplotlib.tri as mtri

from sim.train_sim_eit_hybrid import (
    EITHybridDataset,
    HybridEITNet,
    GRID_SIZE,
    DATA_ROOT,
    CSV_PATH,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
CKPT_NAME = "best_hybrid_model.pt"
MAX_IMAGES = None        # set to an int to limit (e.g. 100)
GAUSSIAN_SIGMA = 1.0     # blur strength in pixels
VMIN = 0.0               # min for colormap (0 contrast)
VMAX = 20.0              # max for colormap (max contrast in sim)


def grid_to_nodal(field_grid, mesh_obj):
    """
    Interpolate a field defined on a regular [-1,1]x[-1,1] grid
    to the EIT mesh nodes.

    field_grid: [H,W]
    mesh_obj:   PyEIT mesh object (mesh_obj.node is [N_nodes,2] in [-1,1]^2)
    """
    H, W = field_grid.shape

    # Grid coordinates (must match dataset generation)
    x_lin = np.linspace(-1.0, 1.0, W)
    y_lin = np.linspace(-1.0, 1.0, H)

    # RegularGridInterpolator expects (y, x) order for a [H,W] array
    interp = RegularGridInterpolator(
        (y_lin, x_lin),
        field_grid,
        bounds_error=False,
        fill_value=0.0,
    )

    pts = mesh_obj.node   # [N_nodes, 2], columns: x, y
    x = pts[:, 0]
    y = pts[:, 1]
    # Interpolator expects points as (y, x)
    query_points = np.stack([y, x], axis=-1)
    nodal_vals = interp(query_points)  # [N_nodes]

    return nodal_vals.astype(np.float32)


def main():
    print(f"Using device: {DEVICE}")

    data_root = Path(DATA_ROOT)
    ckpt_path = data_root / CKPT_NAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ---------- Dataset & loader (test only) ----------
    test_dataset = EITHybridDataset(CSV_PATH, DATA_ROOT, split="test")
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    volt_dim = test_dataset.n_meas
    img_size = (GRID_SIZE, GRID_SIZE)

    # Mesh / triangulation for tripcolor
    mesh_obj = test_dataset.mesh_obj
    pts = mesh_obj.node       # [N_nodes,2]
    tri = mesh_obj.element    # [N_elems,3]
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)

    # ---------- Model ----------
    model = HybridEITNet(volt_dim=volt_dim, img_size=img_size, feat_channels=16)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(
        f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
        f"with test IoU {ckpt.get('test_iou', 'N/A'):.4f}"
    )

    # ---------- Output dir ----------
    out_dir = data_root / "pred_tripcolor"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving tripcolor predictions to: {out_dir}")

    saved = 0

    with torch.no_grad():
        for batch in test_loader:
            volt = batch["volt"].to(DEVICE)       # [B,K]
            bp = batch["bp"].to(DEVICE)           # [B,1,H,W]
            domain = batch["domain"].to(DEVICE)   # [B,1,H,W] (not strictly needed here)
            contrasts = batch["contrast"]         # list of floats
            sample_ids = batch["sample_id"]
            shape_types = batch["shape_type"]

            logits = model(volt, bp, domain, bp_dropout_prob=0.0)
            probs = torch.sigmoid(logits).cpu().numpy()  # [B,1,H,W]

            B = probs.shape[0]
            for i in range(B):
                if MAX_IMAGES is not None and saved >= MAX_IMAGES:
                    break

                sample_id = sample_ids[i]
                shape_type = shape_types[i]
                contrast = float(contrasts[i])

                prob_map = probs[i, 0]  # [H,W]

                # Continuous field: prob * contrast
                field_grid = prob_map * contrast

                # Smooth edges slightly
                if GAUSSIAN_SIGMA > 0.0:
                    field_grid = gaussian_filter(field_grid, sigma=GAUSSIAN_SIGMA)

                # Interpolate grid → nodal values
                nodal_vals = grid_to_nodal(field_grid, mesh_obj)

                # Tripcolor plot
                fig, ax = plt.subplots(figsize=(4, 4))
                tpc = ax.tripcolor(
                    triang,
                    nodal_vals,
                    shading="flat",
                    cmap="viridis",
                    vmin=VMIN,
                    vmax=VMAX,
                )
                ax.set_aspect("equal")
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{sample_id}\nshape={shape_type}, c={contrast:g}")
                fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)

                out_path = out_dir / f"{sample_id}_tripcolor.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                saved += 1

            if MAX_IMAGES is not None and saved >= MAX_IMAGES:
                break

    print(f"Done. Saved {saved} tripcolor-style images in {out_dir}")


if __name__ == "__main__":
    main()
