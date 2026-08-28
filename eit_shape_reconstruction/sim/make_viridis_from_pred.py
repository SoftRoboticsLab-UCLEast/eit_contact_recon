#!/usr/bin/env python3
"""
Generate EIT-like 'viridis' images from the trained hybrid model predictions,
without retraining.

- Loads best_hybrid_model.pt
- Runs on test set
- For each sample:
    - predicted prob map (sigmoid)
    - multiply by contrast (to mimic conductivity level)
    - Gaussian blur to soften edges
    - apply circular domain mask
    - save as viridis-colored PNG

Output folder: eit_dataset/pred_viridis/
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter  # pip install scipy if needed

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
MAX_IMAGES = None    # set to an int if you want to limit (e.g. 200)
GAUSSIAN_SIGMA = 1.0 # blur strength in pixels
VMIN = 0.0           # min for colormap (0 contrast)
VMAX = 20.0          # max for colormap (should match max contrast)


def main():
    print(f"Using device: {DEVICE}")
    data_root = Path(DATA_ROOT)
    ckpt_path = data_root / CKPT_NAME

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ----- dataset & loader (test only) -----
    test_dataset = EITHybridDataset(CSV_PATH, DATA_ROOT, split="test")
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    volt_dim = test_dataset.n_meas
    img_size = (GRID_SIZE, GRID_SIZE)

    # ----- load model -----
    model = HybridEITNet(volt_dim=volt_dim, img_size=img_size, feat_channels=16)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} with test IoU {ckpt.get('test_iou', 'N/A'):.4f}")

    # ----- output folder -----
    out_dir = data_root / "pred_viridis"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving viridis predictions to: {out_dir}")

    saved = 0

    with torch.no_grad():
        for batch in test_loader:
            volt = batch["volt"].to(DEVICE)       # [B,K]
            bp = batch["bp"].to(DEVICE)           # [B,1,H,W]
            domain = batch["domain"].to(DEVICE)   # [B,1,H,W]
            mask = batch["mask"]                  # just to know shape, not needed
            contrasts = batch["contrast"]         # list of floats
            sample_ids = batch["sample_id"]
            shape_types = batch["shape_type"]

            # forward pass
            logits = model(volt, bp, domain, bp_dropout_prob=0.0)
            probs = torch.sigmoid(logits).cpu().numpy()  # [B,1,H,W]
            domain_np = domain.cpu().numpy()             # [B,1,H,W]

            B = probs.shape[0]
            for i in range(B):
                if MAX_IMAGES is not None and saved >= MAX_IMAGES:
                    break

                sample_id = sample_ids[i]
                shape_type = shape_types[i]
                contrast = float(contrasts[i])

                prob_map = probs[i, 0]      # [H,W]
                dom = domain_np[i, 0]       # [H,W]

                # scale by contrast to mimic conductivity level
                field = prob_map * contrast

                # apply Gaussian blur to soften the boundary
                if GAUSSIAN_SIGMA > 0.0:
                    field = gaussian_filter(field, sigma=GAUSSIAN_SIGMA)

                # apply domain mask (zero outside sensing circle)
                field = field * dom

                # save as viridis-colored image
                fig, ax = plt.subplots(figsize=(3, 3))
                im = ax.imshow(field, cmap="viridis", vmin=VMIN, vmax=VMAX)
                ax.set_title(f"{sample_id}\nshape={shape_type}, c={contrast:g}")
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                out_path = out_dir / f"{sample_id}_viridis.png"
                fig.savefig(out_path, dpi=120, bbox_inches="tight")
                plt.close(fig)

                saved += 1

            if MAX_IMAGES is not None and saved >= MAX_IMAGES:
                break

    print(f"Done. Saved {saved} viridis-style prediction images in {out_dir}")


if __name__ == "__main__":
    main()
