#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.protocol import create as create_protocol
from pyeit.eit.bp import BP

def node2elem(values, mesh_obj):
    # If ds is node-wise, average to elements
    if values.shape[0] == mesh_obj.node.shape[0]:
        return values[mesh_obj.element].mean(axis=1)
    return values

def rotate_coords(x, y, angle_deg):
    """Rotate (x,y) by +angle_deg (in degrees, CCW)."""
    theta = np.deg2rad(angle_deg)
    xr =  np.cos(theta)*x - np.sin(theta)*y
    yr =  np.sin(theta)*x + np.cos(theta)*y
    return xr, yr

def find_eit_cols(df, prefix="eit"):
    """
    Find EIT columns automatically:
    - accepts 'eit_0', 'eit_001', 'eit1', etc.
    - ignores non-numeric suffixes (like 'eit_t')
    - assigns sequential indices if parsing fails
    """
    cols = [c for c in df.columns if c.lower().startswith(prefix.lower())]
    if not cols:
        raise RuntimeError(f"No EIT columns found with prefix '{prefix}'. Got: {list(df.columns)[:10]} ...")

    idx = []
    for i, c in enumerate(cols):
        # try to parse numeric part
        parts = c.split("_")
        num = None
        if len(parts) > 1:
            try:
                num = int(parts[1])
            except ValueError:
                num = None
        if num is None:
            # fallback: sequential index
            num = i
        idx.append(num)

    idx = np.array(idx, dtype=int)
    order = np.argsort(idx)
    cols = [cols[i] for i in order]
    idx = idx[order]
    return cols, idx

def get_coords(df, idx):
    """
    Returns normalized coords (unit circle) for R1 and R2 at row idx.
    Tries R1_x_norm/R1_y_norm etc. Falls back to R1_x/R1_y if needed.
    """
    if {"R1_x_norm","R1_y_norm","R2_x_norm","R2_y_norm"}.issubset(df.columns):
        x1, y1 = df.loc[idx, "R1_x_norm"], df.loc[idx, "R1_y_norm"]
        x2, y2 = df.loc[idx, "R2_x_norm"], df.loc[idx, "R2_y_norm"]
    elif {"R1_x","R1_y","R2_x","R2_y"}.issubset(df.columns):
        # assume already normalized; if not, you can add a --radius-m arg and divide
        x1, y1 = df.loc[idx, "R1_x"], df.loc[idx, "R1_y"]
        x2, y2 = df.loc[idx, "R2_x"], df.loc[idx, "R2_y"]
    else:
        raise RuntimeError("Could not find normalized coordinate columns.")
    return float(x1), float(y1), float(x2), float(y2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to training CSV (with R1_x_norm/R2_x_norm and eit_*)")
    ap.add_argument("--v0",  required=True, help="Path to baseline .npy (no-touch)")
    ap.add_argument("--out", default="recon_check_rt_match", help="Output dir for plots")
    ap.add_argument("--n-el", type=int, default=16)
    ap.add_argument("--dist-exc", type=int, default=1)
    ap.add_argument("--step-meas", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--clip-pct", type=float, default=99.0)
    ap.add_argument("--gate-thresh", type=float, default=0.3, help="Sum |v0 - v1| gating")
    ap.add_argument("--scale", type=float, default=192.0, help="BP output scaling factor")
    ap.add_argument("--rows", type=str, default="", help="Comma-separated row indices to plot (overrides random)")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load CSV and baseline
    df = pd.read_csv(args.csv)
    # --- Load baseline (supports .csv or .npy) ---
        # 1) Load CSV and baseline
    df = pd.read_csv(args.csv)

    # --- Load baseline (supports .csv or .npy) ---
    v0_path = Path(args.v0)
    if v0_path.suffix.lower() == ".npy":
        v0_full = np.load(v0_path).astype(np.float32)
    elif v0_path.suffix.lower() == ".csv":
        # read the CSV, skip header row if present, and take only the first numeric row
        df_base = pd.read_csv(v0_path, header=0)
        # extract only numeric columns
        df_base = df_base.select_dtypes(include=[np.number])
        # if still empty, try forcing conversion
        if df_base.empty:
            df_base = pd.read_csv(v0_path, skiprows=1, header=None)
        v0_full = df_base.iloc[0].to_numpy(dtype=np.float32).flatten()
    else:
        raise RuntimeError(f"Unsupported baseline file format: {v0_path.suffix}")

    eit_cols, eit_idx = find_eit_cols(df)
    if v0_full.shape[0] <= eit_idx.max():
        raise RuntimeError("Baseline v0 length is smaller than CSV EIT index range. Check files.")


    # Align baseline to CSV column order and drop zeros
    v0_unmasked = v0_full[eit_idx]
    nonzero_mask = (v0_unmasked != 0.0)
    kept_cols = [c for c, keep in zip(eit_cols, nonzero_mask) if keep]
    kept_idx  = eit_idx[nonzero_mask]
    v0 = v0_unmasked[nonzero_mask].astype(np.float32)

    M = v0.shape[0]
    print(f"[INFO] CSV rows: {len(df)} | EIT cols total: {len(eit_cols)} | kept (nonzero in v0): {M}")

    # 2) PyEIT setup (unit disk mesh)
    mesh_obj = mesh.create(n_el=args.n_el, h0=0.1)
    protocol = create_protocol(n_el=args.n_el, dist_exc=args.dist_exc, step_meas=args.step_meas, parser_meas="std")
    bp = BP(mesh_obj, protocol)
    bp.setup(weight="none")

    # 3) Which rows to visualize
    n = len(df)
    if args.rows.strip():
        idx_samples = np.array([int(s) for s in args.rows.split(",") if s.strip() != ""], dtype=int)
        idx_samples = idx_samples[(idx_samples >= 0) & (idx_samples < n)]
    else:
        rng = np.random.default_rng(0)
        take = min(args.n_samples, n)
        idx_samples = rng.choice(n, size=take, replace=False)

    # 4) Reconstruct & plot (real-time compatible)
    elem_maps = []
    titles = []
    coords = []

    for idx in idx_samples:
        # 4.1 Load and clean the EIT sample
        v_row_full = df.loc[idx, kept_cols].to_numpy(np.float32)
        v1 = np.array([v for v in v_row_full if v != 0.0], dtype=np.float32)
        v0_filt = np.array([v for v in v0 if v != 0.0], dtype=np.float32)

        # Make sure both arrays have same length
        m = min(len(v1), len(v0_filt))
        v1, v0_filt = v1[:m], v0_filt[:m]

        # 4.2 Compute gating condition (as in real-time code)
        diff_sum = float(np.abs(v0_filt - v1).sum())
        is_active = diff_sum > args.gate_thresh

        # 4.3 Apply same solver and scaling as real-time script
        if is_active:
            ds_node = 192.0 * bp.solve(v1, v0_filt, normalize=True, log_scale=False)
            state = "active"
        else:
            ds_node = 192.0 * bp.solve(v0_filt, v0_filt, normalize=True)
            state = "baseline"

        # 4.4 Convert node-wise to element-wise
        ds_elem = node2elem(ds_node, mesh_obj)
        elem_maps.append(ds_elem)
        titles.append(f"Row {idx} | {state} (Σ|Δ|={diff_sum:.3f})")
        coords.append(get_coords(df, idx))

    elem_maps = np.stack(elem_maps)
    clip = np.percentile(np.abs(elem_maps), args.clip_pct)
    vmin, vmax = -clip, clip

    # 5) Plot per sample with contact markers (rotation etc.)
    for i, idx in enumerate(idx_samples):
        x1, y1, x2, y2 = coords[i]
        # Optional rotation correction (as before)
        x1, y1 = rotate_coords(x1, y1, angle_deg=-65)
        x2, y2 = rotate_coords(x2, y2, angle_deg=-65)

        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        tpc = ax.tripcolor(
            mesh_obj.node[:, 0], mesh_obj.node[:, 1], mesh_obj.element,
            elem_maps[i], shading="flat", cmap=args.cmap,
            vmin=vmin, vmax=vmax
        )

        circ = plt.Circle((0, 0), 1.0, fill=False, color="k", linewidth=0.8, alpha=0.7)
        ax.add_patch(circ)
        # ax.plot(x1, y1, marker="x", ms=8, mew=1.8, color="k")
        # ax.plot(x2, y2, marker="+", ms=8, mew=1.8, color="r")
        ax.set_aspect("equal"); ax.axis("off")
        # plt.title(titles[i], fontsize=11)
        cb = plt.colorbar(tpc, ax=ax, shrink=0.82); cb.ax.tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"recon_rtstyle_{i:02d}_row_{idx}.png", dpi=300)
        plt.close(fig)

    print(f"✅ Saved {len(idx_samples)} recon plots → {out_dir}/")
    print("    (Protocol: adj dist_exc=%d, step_meas=%d; BP normalize=True, weight='none')" % (args.dist_exc, args.step_meas))

if __name__ == "__main__":
    main()
