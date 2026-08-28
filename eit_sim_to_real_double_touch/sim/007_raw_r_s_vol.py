#!/usr/bin/env python3
"""
Minimal raw-voltage peek for SIM vs REAL EIT data.
-------------------------------------------------
- Loads two CSVs (simulated + real)
- Finds EIT columns ('eit_<int>' or 'v<int>'), ignores 'eit_t'
- Converts to numeric
- For real data: drops any columns that are exactly zero for all samples
- Plots N random samples from each dataset (no baseline subtraction)
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM_SUFFIX = re.compile(r"(\d+)$")

def find_eit_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        lc = c.lower()
        if lc == "eit_t":
            continue
        if lc.startswith("eit_") or lc.startswith("v"):
            if NUM_SUFFIX.search(c) is not None:
                cols.append(c)
    if not cols:
        raise RuntimeError("No EIT columns found (expected eit_<index> or v<index>).")
    # Sort numerically
    cols = sorted(cols, key=lambda name: int(NUM_SUFFIX.search(name).group(1)))
    return cols

def to_numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Convert selected columns to numeric array, NaNs → 0"""
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)

def drop_zero_columns(X: np.ndarray, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Drop columns that are exactly zero for all rows."""
    keep_mask = np.any(X != 0.0, axis=0)
    X_new = X[:, keep_mask]
    cols_new = [c for c, keep in zip(cols, keep_mask) if keep]
    dropped = np.sum(~keep_mask)
    print(f"[INFO] Dropped {dropped} all-zero columns from REAL data → kept {len(cols_new)}.")
    return X_new, cols_new

def plot_samples(X: np.ndarray, title: str, outpath: Path, n_show: int = 5, seed: int = 0):
    n = len(X)
    take = min(n_show, n)
    idx = np.random.default_rng(seed).choice(n, size=take, replace=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, i in enumerate(idx):
        ax.plot(X[i], lw=1.2, alpha=0.8, label=f"row {i}" if k == 0 else None)
    ax.set_title(title)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Voltage (raw units)")
    if take > 0:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-csv", required=True, help="Path to simulated CSV")
    ap.add_argument("--real-csv", required=True, help="Path to real CSV")
    ap.add_argument("--out", default="peek_raw_voltages", help="Output directory")
    ap.add_argument("--n", type=int, default=5, help="Number of sample rows to plot from each dataset")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for row sampling")

    ap.add_argument("--sim-is-delta", action="store_true",
                    help="Treat SIM CSV as deltas and add baseline back to display raw.")
    ap.add_argument("--n-el", type=int, default=16)
    ap.add_argument("--dist-exc", type=int, default=1)
    ap.add_argument("--step-meas", type=int, default=1)

    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    df_sim = pd.read_csv(args.sim_csv)
    df_real = pd.read_csv(args.real_csv)

    # Find EIT columns
    sim_cols = find_eit_cols(df_sim)
    real_cols = find_eit_cols(df_real)

    # Convert to numeric matrices
    X_sim = to_numeric_matrix(df_sim, sim_cols)
    X_real = to_numeric_matrix(df_real, real_cols)

    # after X_sim is loaded:
    if args.sim_is_delta:
        import pyeit.mesh as mesh
        from pyeit.eit.protocol import create as create_protocol
        from pyeit.eit.fem import EITForward
        m = mesh.create(n_el=args.n_el)
        proto = create_protocol(args.n_el, dist_exc=args.dist_exc, step_meas=args.step_meas)
        fwd = EITForward(m, proto)
        v0 = fwd.solve_eit(m.perm).astype(np.float32)          # baseline
        if X_sim.shape[1] != v0.size:
            raise RuntimeError(f"SIM cols={X_sim.shape[1]} but v0 has {v0.size}")
        X_sim = X_sim + v0[None, :]  # reconstruct raw for plotting
        print("addeddd .....")

    # Drop all-zero columns from REAL
    X_real, real_cols = drop_zero_columns(X_real, real_cols)

    print(f"[SIM]  rows={X_sim.shape[0]}, channels={X_sim.shape[1]} (example cols: {sim_cols[:5]})")
    print(f"[REAL] rows={X_real.shape[0]}, channels={X_real.shape[1]} (after drop, example cols: {real_cols[:5]})")

    # Quick stats
    def pct(label, A):
        q = np.percentile(np.abs(A), [1, 50, 99])
        print(f"  {label} |v| pct [1,50,99] = {q}")
    pct("SIM ", X_sim)
    pct("REAL", X_real)

    # Plot sample traces
    plot_samples(X_real, f"REAL raw voltages (random {args.n} rows)", outdir / "real_raw_samples.png", n_show=args.n, seed=args.seed)
    plot_samples(X_sim,  f"SIM raw voltages (random {args.n} rows)",  outdir / "sim_raw_samples.png",  n_show=args.n, seed=args.seed+1)

    # Export one row for detailed look
    pd.DataFrame({"real_trace0": X_real[0]}).to_csv(outdir / "real_trace0.csv", index=False)
    pd.DataFrame({"sim_trace0":  X_sim[0]}).to_csv(outdir / "sim_trace0.csv",  index=False)

    print(f"✅ Saved plots to {outdir}/ (real_raw_samples.png, sim_raw_samples.png)")

if __name__ == "__main__":
    main()
