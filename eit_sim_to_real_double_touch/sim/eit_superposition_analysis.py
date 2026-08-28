"""
Superposition Analysis for Paired Single–Double Touch Dataset
=============================================================

Given the CSV produced by `eit_paired_single_double_gen.py`, this script:

1) Detects whether delta columns (dv_*) exist; if not, uses absolute voltages
   and (optionally) subtracts a provided baseline `v0` to compute deltas.
2) Computes per-sample metrics comparing the double-touch voltage vector to:
      - SUM = v_s1 + v_s2  (or dv_s1 + dv_s2)
      - AVG = 0.5 * (v_s1 + v_s2)  (or 0.5*(dv_s1 + dv_s2))
   Metrics: Mean Absolute Error (MAE) and R^2.
3) Saves summary statistics and histograms of errors.
4) Creates qualitative plots for a handful of samples showing:
      - Overlaid traces: double vs sum vs avg
      - Scatter: double vs sum (with y=x line)
      - Scatter: double vs avg (with y=x line)

Usage
-----
python eit_superposition_analysis.py \
  --csv eit_paired_data/paired_single_double.csv \
  --out-dir superposition_report \
  --baseline eit_paired_data/baseline_v0.npy \
  --num-examples 8

If your CSV already contains dv_* columns (using --save-delta), --baseline is optional.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

def find_columns(prefix: str, df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    # ensure numeric order suffix
    def idx(c):
        try:
            return int(c.split('_')[-1])
        except:
            return 10**9
    return sorted(cols, key=idx)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Paired dataset CSV")
    ap.add_argument("--out-dir", type=str, default="superposition_report")
    ap.add_argument("--baseline", type=str, default=None, help="Path to baseline v0 .npy (used if dv_* missing)")
    ap.add_argument("--num-examples", type=int, default=8, help="Number of qualitative examples to save")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    has_dv = any(c.startswith("dv_double_") for c in df.columns)

    # Identify voltage columns
    if has_dv:
        cols_double = find_columns("dv_double_", df)
        cols_s1     = find_columns("dv_s1_", df)
        cols_s2     = find_columns("dv_s2_", df)
        label = "ΔV (vs baseline)"
    else:
        cols_double = find_columns("v_double_", df)
        cols_s1     = find_columns("v_s1_", df)
        cols_s2     = find_columns("v_s2_", df)
        label = "Absolute V"
        if args.baseline is not None and Path(args.baseline).exists():
            v0 = np.load(args.baseline).astype(np.float32)
            # Convert to ΔV for analysis consistency
            Vd = df[cols_double].to_numpy(np.float32)
            Vs1 = df[cols_s1].to_numpy(np.float32)
            Vs2 = df[cols_s2].to_numpy(np.float32)
            dv_double = Vd - v0[None, :]
            dv_s1 = Vs1 - v0[None, :]
            dv_s2 = Vs2 - v0[None, :]
            X_double = dv_double
            X_s1 = dv_s1
            X_s2 = dv_s2
            label = "ΔV (computed from absolute using baseline)"
        else:
            # Proceed with absolute voltages
            X_double = df[cols_double].to_numpy(np.float32)
            X_s1     = df[cols_s1].to_numpy(np.float32)
            X_s2     = df[cols_s2].to_numpy(np.float32)
    if has_dv:
        X_double = df[cols_double].to_numpy(np.float32)
        X_s1     = df[cols_s1].to_numpy(np.float32)
        X_s2     = df[cols_s2].to_numpy(np.float32)

    n, m = X_double.shape

    # Compute predictions
    X_sum = X_s1 + X_s2
    X_avg = 0.5 * X_sum

    # Metrics per sample
    mae_sum   = np.mean(np.abs(X_double - X_sum), axis=1)
    mae_avg   = np.mean(np.abs(X_double - X_avg), axis=1)
    r2_sum    = np.array([r2_score(X_double[i], X_sum[i]) for i in range(n)], dtype=np.float32)
    r2_avg    = np.array([r2_score(X_double[i], X_avg[i]) for i in range(n)], dtype=np.float32)

    # Summary CSV
    meta_cols = [c for c in df.columns if c in ["x1","y1","force1","x2","y2","force2","probe_diam_mm","r_norm"]]
    summary = pd.DataFrame({
        "mae_sum": mae_sum,
        "mae_avg": mae_avg,
        "r2_sum": r2_sum,
        "r2_avg": r2_avg,
    })
    if meta_cols:
        summary = pd.concat([summary, df[meta_cols].reset_index(drop=True)], axis=1)
    summary_path = out_dir / "superposition_metrics.csv"
    summary.to_csv(summary_path, index=False)

    # Print brief stats
    with open(out_dir / "summary.txt", "w") as f:
        def wln(s): 
            print(s); f.write(s + "\n")
        wln(f"Samples: {n}, Meas length: {m}")
        wln(f"Using: {label}")
        wln(f"MAE(sum)  mean={np.nanmean(mae_sum):.6f}  median={np.nanmedian(mae_sum):.6f}")
        wln(f"MAE(avg)  mean={np.nanmean(mae_avg):.6f}  median={np.nanmedian(mae_avg):.6f}")
        wln(f"R2(sum)   mean={np.nanmean(r2_sum):.4f}   median={np.nanmedian(r2_sum):.4f}")
        wln(f"R2(avg)   mean={np.nanmean(r2_avg):.4f}   median={np.nanmedian(r2_avg):.4f}")

    # Histograms
    plt.figure(figsize=(6,4))
    plt.hist(mae_sum, bins=40, alpha=0.7, label="MAE vs SUM")
    plt.hist(mae_avg, bins=40, alpha=0.7, label="MAE vs AVG")
    plt.legend()
    plt.xlabel("MAE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_mae.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(r2_sum[~np.isnan(r2_sum)], bins=40, alpha=0.7, label="R² vs SUM")
    plt.hist(r2_avg[~np.isnan(r2_avg)], bins=40, alpha=0.7, label="R² vs AVG")
    plt.legend()
    plt.xlabel("R²")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_r2.png", dpi=200)
    plt.close()

    # Qualitative examples
    rng = np.random.default_rng(args.seed)
    # pick a mix: best, worst, random
    idx_best = np.argsort(mae_sum)[:max(1, args.num_examples//3)]
    idx_worst = np.argsort(-mae_sum)[:max(1, args.num_examples//3)]
    remaining = np.setdiff1d(np.arange(n), np.concatenate([idx_best, idx_worst]), assume_unique=False)
    idx_rand = rng.choice(remaining, size=max(0, args.num_examples - len(idx_best) - len(idx_worst)), replace=False) if len(remaining) else np.array([], dtype=int)
    idx_vis = np.concatenate([idx_best, idx_worst, idx_rand]).astype(int)

    for j, i in enumerate(idx_vis):
        v_double = X_double[i]
        v_sum = X_sum[i]
        v_avg = X_avg[i]

        # Overlaid traces
        fig, ax = plt.subplots(figsize=(7,3))
        ax.plot(v_double, label="Double")
        ax.plot(v_sum, label="Sum(s1+s2)")
        # ax.plot(v_avg, label="Avg(0.5·(s1+s2))")
        ax.set_title(f"Sample {i}  |  MAE(sum)={mae_sum[i]:.4f}, R²(sum)={r2_sum[i]:.3f}")
        ax.set_xlabel("Measurement index")
        ax.set_ylabel("Voltage")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"example_{j:02d}_overlay.png", dpi=220)
        plt.close()

        # Scatter plots
        fig, ax = plt.subplots(figsize=(3.2,3.2))
        ax.scatter(v_double, v_sum, s=8, alpha=0.7)
        lims = [min(v_double.min(), v_sum.min()), max(v_double.max(), v_sum.max())]
        ax.plot(lims, lims, linestyle="--")
        ax.set_xlabel("Double")
        ax.set_ylabel("Sum")
        ax.set_title(f"R²={r2_sum[i]:.3f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"example_{j:02d}_scatter_sum.png", dpi=220)
        plt.close()

        fig, ax = plt.subplots(figsize=(3.2,3.2))
        ax.scatter(v_double, v_avg, s=8, alpha=0.7)
        lims = [min(v_double.min(), v_avg.min()), max(v_double.max(), v_avg.max())]
        ax.plot(lims, lims, linestyle="--")
        ax.set_xlabel("Double")
        ax.set_ylabel("Avg")
        ax.set_title(f"R²={r2_avg[i]:.3f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"example_{j:02d}_scatter_avg.png", dpi=220)
        plt.close()

    print("✅ Analysis complete.")
    print(f"- Metrics CSV: {summary_path}")
    print(f"- Plots saved under: {out_dir}/")

if __name__ == "__main__":
    main()
