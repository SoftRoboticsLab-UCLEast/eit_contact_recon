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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to real CSV (e.g., human_grid_R12C12_...csv)")
    ap.add_argument("--v0",  required=True, help="Path to baseline .npy saved by the session")
    ap.add_argument("--out", default="recon_check_rt_match", help="Output dir for plots")
    ap.add_argument("--n-el", type=int, default=16)
    ap.add_argument("--dist-exc", type=int, default=1)
    ap.add_argument("--step-meas", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--clip-pct", type=float, default=99.0)
    ap.add_argument("--gate-thresh", type=float, default=0.3, help="Sum |v0 - v1| gating (same logic as RT)")
    ap.add_argument("--scale", type=float, default=192.0, help="BP output scaling factor (matches RT)")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load CSV and baseline
    df = pd.read_csv(args.csv)
    v0_full = np.load(args.v0).astype(np.float32)

    eit_cols = [c for c in df.columns if c.startswith("eit_")]
    if not eit_cols:
        raise RuntimeError("No EIT columns found (eit_0, eit_1, ...).")

    print(f"[INFO] CSV rows: {len(df)} | EIT cols: {len(eit_cols)} | v0 shape: {v0_full.shape}")

    # 2) Build the **exact-zero mask** from baseline, like split_eit_data (drop zeros)
    #    Map column names -> indices in v0
    try:
        col_idx = np.array([int(c.split("_")[1]) for c in eit_cols], dtype=int)
    except Exception:
        raise RuntimeError("EIT column naming must be 'eit_<index>' integers.")

    if v0_full.shape[0] <= col_idx.max():
        raise RuntimeError("Baseline v0.npy length is smaller than CSV EIT index range. Check files.")

    v0_unmasked = v0_full[col_idx]  # align baseline to CSV column order
    nonzero_mask = (v0_unmasked != 0.0)

    # If baseline has zeros everywhere (unlikely), fall back to per-row nonzeros
    if not np.any(nonzero_mask):
        raise RuntimeError("All baseline entries are zero; cannot build zero-drop mask.")

    kept_cols = [c for c, keep in zip(eit_cols, nonzero_mask) if keep]
    kept_idx  = col_idx[nonzero_mask]
    v0 = v0_unmasked[nonzero_mask].astype(np.float32)
    M = v0.shape[0]
    print(f"[INFO] Kept channels (nonzero in v0): {M} | dropped: {len(eit_cols) - M}")

    # 3) Build PyEIT objects — same protocol as RT, no forward-vector surgery
    mesh_obj = mesh.create(n_el=args.n_el, h0=0.1)
    protocol = create_protocol(n_el=args.n_el, dist_exc=args.dist_exc, step_meas=args.step_meas, parser_meas="std")
    bp = BP(mesh_obj, protocol)
    bp.setup(weight="none")

    # Sanity: RT pipeline typically ends up with 208 channels (adjacent pattern)
    print(f"[INFO] (Sanity) protocol usually expects ~208 channels; we have {M} after zero-drop.")

    # 4) Choose rows to visualize
    n = len(df)
    take = min(args.n_samples, n)
    rng = np.random.default_rng(0)
    idx_samples = rng.choice(n, size=take, replace=False)

    # 5) Reconstruct each sample with the **same logic** as RT
    elem_maps = []
    titles = []
    for idx in idx_samples:
        v_row_full = df.loc[idx, kept_cols].to_numpy(np.float32)  # already filtered to same indices
        # RT gating: sum |v0 - v1|
        diff_sum = float(np.abs(v0 - v_row_full).sum())
        if diff_sum > args.gate_thresh:
            ds_node = args.scale * bp.solve(v_row_full, v0, normalize=True, log_scale=False)
            title = f"Row {idx} | active (Σ|Δ|={diff_sum:.3f})"
        else:
            ds_node = args.scale * bp.solve(v0, v0, normalize=True, log_scale=False)
            title = f"Row {idx} | baseline (Σ|Δ|={diff_sum:.3f})"

        ds_elem = node2elem(ds_node, mesh_obj)
        elem_maps.append(ds_elem)
        titles.append(title)

    elem_maps = np.stack(elem_maps)
    clip = np.percentile(np.abs(elem_maps), args.clip_pct)
    vmin, vmax = -clip, clip

    # 6) Plot, optionally annotate grid targets if available
    for i, idx in enumerate(idx_samples):
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        tpc = ax.tripcolor(
            mesh_obj.node[:, 0], mesh_obj.node[:, 1], mesh_obj.element,
            elem_maps[i], shading="flat", cmap=args.cmap, vmin=vmin, vmax=vmax
        )
        circ = plt.Circle((0,0), 1.0, fill=False, color="k", linewidth=0.8, alpha=0.6)
        ax.add_patch(circ)
        ax.set_aspect("equal"); ax.axis("off")

        # Mark grid centers if present
        # if {"x1_norm","y1_norm"}.issubset(df.columns):
        #     x1, y1 = df.loc[idx, ["x1_norm","y1_norm"]].astype(float).values
        #     ax.plot(x1, y1, marker="x", ms=8, mew=1.5, color="k")
        # if {"x2_norm","y2_norm"}.issubset(df.columns) and pd.notna(df.loc[idx,"x2_norm"]):
        #     x2, y2 = df.loc[idx, ["x2_norm","y2_norm"]].astype(float).values
        #     ax.plot(x2, y2, marker="+", ms=8, mew=1.5, color="k")

        plt.title(titles[i], fontsize=11)
        cb = plt.colorbar(tpc, ax=ax, shrink=0.82); cb.ax.tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(Path(args.out) / f"recon_rt_match_{i:02d}.png", dpi=300)
        plt.close(fig)

    print(f"✅ Saved {take} recon plots → {args.out}/")
    print(f"    (Used baseline-zero mask, BP weight='none', normalize=True, log_scale=False, scale={args.scale})")

if __name__ == "__main__":
    main()
