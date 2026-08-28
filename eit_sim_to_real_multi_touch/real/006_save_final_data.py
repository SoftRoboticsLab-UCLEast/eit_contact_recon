#!/usr/bin/env python3
"""
Create the final training-ready CSV for sim-to-real EIT model training.

Input:
  - eit_double_touch_data.csv     (EIT signals, one row per sample)
  - sensor_frame_points_translated_per_robot.csv (2x rows: R1 + R2 positions)

Output:
  - training_dataset.csv with columns:
        R1_x_norm, R1_y_norm, R2_x_norm, R2_y_norm, <EIT signal columns...>
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-eit-csv", required=True, type=Path,
                    help="Path to raw EIT data (eit_double_touch_data.csv)")
    ap.add_argument("--translated-csv", required=True, type=Path,
                    help="Path to translated per-robot positions CSV")
    ap.add_argument("--radius-m", type=float, default=0.09,
                    help="Sensor radius in meters for normalization")
    ap.add_argument("--outdir", required=True, type=Path,
                    help="Directory to save final CSV")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    df_eit = pd.read_csv(args.raw_eit_csv)
    df_pos = pd.read_csv(args.translated_csv)

    if "robot_id" not in df_pos.columns or \
       "x_trans_m" not in df_pos.columns or \
       "y_trans_m" not in df_pos.columns:
        raise ValueError("Translated CSV must have columns: robot_id, x_trans_m, y_trans_m")

    n_eit = len(df_eit)
    n_pos = len(df_pos)
    if n_pos != 2 * n_eit:
        print(f"[WARN] Translated CSV has {n_pos} rows, expected 2×{n_eit}={2*n_eit}")

    # --- Split into robot halves ---
    half = n_pos // 2
    df_r1 = df_pos.iloc[:half].copy()
    df_r2 = df_pos.iloc[half:half + n_eit].copy()

    if len(df_r1) != n_eit or len(df_r2) != n_eit:
        print("[WARN] Length mismatch, truncating to smallest common length.")
        n_min = min(len(df_eit), len(df_r1), len(df_r2))
        df_eit = df_eit.iloc[:n_min].reset_index(drop=True)
        df_r1 = df_r1.iloc[:n_min].reset_index(drop=True)
        df_r2 = df_r2.iloc[:n_min].reset_index(drop=True)
    else:
        df_r1.reset_index(drop=True, inplace=True)
        df_r2.reset_index(drop=True, inplace=True)

    # --- Normalize coordinates ---
    df_r1["R1_x_norm"] = df_r1["x_trans_m"] / args.radius_m
    df_r1["R1_y_norm"] = df_r1["y_trans_m"] / args.radius_m
    df_r2["R2_x_norm"] = df_r2["x_trans_m"] / args.radius_m
    df_r2["R2_y_norm"] = df_r2["y_trans_m"] / args.radius_m

    # --- Combine ---
    df_train = pd.concat(
        [
            df_r1[["R1_x_norm", "R1_y_norm"]].reset_index(drop=True),
            df_r2[["R2_x_norm", "R2_y_norm"]].reset_index(drop=True),
            df_eit.reset_index(drop=True)
        ],
        axis=1
    )

    # --- Save final ---
    out_csv = outdir / "training_dataset.csv"
    df_train.to_csv(out_csv, index=False)

    # --- Report ---
    n_cols = df_train.shape[1]
    print(f"✅ Saved training dataset: {out_csv}")
    print(f"   Samples: {len(df_train)}, Columns: {n_cols}")
    print("   Columns:", ", ".join(df_train.columns[:6]), "...")


if __name__ == "__main__":
    main()
