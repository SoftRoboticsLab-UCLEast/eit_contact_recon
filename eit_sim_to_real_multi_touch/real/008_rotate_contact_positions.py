#!/usr/bin/env python3
"""
Apply -65° rotation to Robot1 and Robot2 positions in training_dataset.csv
to align real sensor data with PyEIT's electrode coordinate frame.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# === Configuration ===
INPUT_CSV = "/home/kiyanoush/Projects/eit_sim_to_real_multi_touch/training_dataset/training_dataset.csv"   # path to your dataset
ROTATION_DEG = -65.0                 # rotation angle (clockwise offset correction)

# === Helper function ===
def rotate_coords(x, y, angle_deg):
    """Rotate coordinates (x, y) by +angle_deg (CCW)."""
    theta = np.deg2rad(angle_deg)
    xr = np.cos(theta) * x - np.sin(theta) * y
    yr = np.sin(theta) * x + np.cos(theta) * y
    return xr, yr

def main():
    csv_path = Path(INPUT_CSV)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"[INFO] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    # detect robot columns
    r1_cols = ["R1_x_norm", "R1_y_norm"]
    r2_cols = ["R2_x_norm", "R2_y_norm"]

    if not all(c in df.columns for c in r1_cols + r2_cols):
        raise RuntimeError("Expected columns x1_norm, y1_norm, x2_norm, y2_norm in dataset.")

    # apply rotation
    print(f"[INFO] Applying rotation of {ROTATION_DEG}° to Robot1 and Robot2 coordinates...")
    df["R1_x_norm"], df["R1_y_norm"] = rotate_coords(df["R1_x_norm"], df["R1_y_norm"], ROTATION_DEG)
    df["R2_x_norm"], df["R2_y_norm"] = rotate_coords(df["R2_x_norm"], df["R2_y_norm"], ROTATION_DEG)

    # save back to same file (overwrite)
    backup_path = csv_path.with_suffix(".bak.csv")
    csv_path.rename(backup_path)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Rotation applied. Original file backed up to {backup_path}")
    print(f"[INFO] Updated dataset saved as {csv_path}")

if __name__ == "__main__":
    main()
