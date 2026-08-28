#!/usr/bin/env python3
"""
Brute-force diagnose electrode rotation / direction and baseline handling.

It grid-searches:
  - rot_shift in [0..15] electrodes (80° is a shift of round(16*80/360)=4)
  - direction in {"cw", "ccw"}

For each hypothesis, it reconstructs a few rows and scores alignment:
  score = average( mean(ds inside small disks at true touches) - mean(ds in whole disk) )
Higher is better (stronger, localized peaks at true touches).

Outputs:
  - score_matrix.png (rot_shift x direction)
  - best_k_recon_*.png for visualization
  - stdout summary including channel-count checks and best hypothesis
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.protocol import create as create_protocol
from pyeit.eit.fem import EITForward
import pyeit.eit.bp as bp


# -------------------- helpers --------------------
def find_eit_cols(df):
    cols = [c for c in df.columns if c.startswith("eit_") or c.startswith("v")]
    if not cols:
        raise RuntimeError("No EIT columns found (expected 'eit_0' or 'v0' style).")
    return cols

def get_targets(df):
    names = ["x1_norm","y1_norm","x2_norm","y2_norm"]
    if not all(n in df.columns for n in names):
        names = ["x1","y1","x2","y2"]
        if not all(n in df.columns for n in names):
            raise RuntimeError("Need target cols: x1_norm,y1_norm,x2_norm,y2_norm (or x1,y1,x2,y2).")
    Y = df[names].to_numpy(np.float32)
    return Y

def build_mesh_and_protocol(n_el, rot_shift, direction, dist_exc, step_meas):
    m = mesh.create(n_el=n_el)
    el = m.el_pos
    # base indexing is CCW from +X; we want a hypothesis permutation
    if direction == "cw":
        perm = el[::-1]
    else:
        perm = el.copy()
    perm = np.roll(perm, shift=int(rot_shift), axis=0)
    m.el_pos = perm
    prot = create_protocol(n_el, dist_exc=dist_exc, step_meas=step_meas)
    return m, prot

def compute_baseline(mesh_obj, protocol):
    fwd = EITForward(mesh_obj, protocol)
    return fwd.solve_eit(mesh_obj.perm).astype(np.float32)

def bp_recon(mesh_obj, protocol, v1, v0, scale=192.0):
    recon = bp.BP(mesh_obj, protocol)
    recon.setup(weight="none")
    ds = scale * recon.solve(v1, v0, normalize=True)
    return ds

def score_map(ds, mesh_obj, touches, r_disk=0.18):
    """Average (mean in small disks at touches) - (global mean)."""
    pts = mesh_obj.node
    tri = mesh_obj.element
    x = pts[:, 0]; y = pts[:, 1]
    disk_mask = (x**2 + y**2) <= 1.0 + 1e-6
    global_mean = np.mean(ds[disk_mask])
    local_means = []
    for (tx, ty) in touches:
        loc_mask = ((x - tx)**2 + (y - ty)**2) <= (r_disk**2)
        # intersect with unit disk
        loc_mask = loc_mask & disk_mask
        if np.any(loc_mask):
            local_means.append(np.mean(ds[loc_mask]))
        else:
            local_means.append(global_mean)
    return float(np.mean(local_means) - global_mean)

def draw_recon(mesh_obj, ds, touches, path, cmap="viridis"):
    pts = mesh_obj.node
    tri = mesh_obj.element
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, ds, shading="flat", cmap=cmap)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color='k', lw=0.9))
    ax.scatter([t[0] for t in touches], [t[1] for t in touches],
               c=["tab:blue","tab:orange"], s=80, marker="x", lw=2, label="True touches")
    ax.set_aspect("equal"); ax.legend()
    plt.colorbar(im, ax=ax, shrink=0.88)
    plt.tight_layout(); plt.savefig(path, dpi=220); plt.close(fig)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Sim CSV with EIT + x1,y1,x2,y2")
    ap.add_argument("--out", default="diag_rotation", help="Output dir")
    ap.add_argument("--assume-delta", action="store_true",
                    help="CSV contains delta voltages (v_touch - v0). If not set, treat columns as raw and subtract baseline.")
    ap.add_argument("--n-el", type=int, default=16)
    ap.add_argument("--dist-exc", type=int, default=1)
    ap.add_argument("--step-meas", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=24, help="Subset size used for scoring.")
    ap.add_argument("--scale", type=float, default=192.0)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.csv)
    eit_cols = find_eit_cols(df)
    X = df[eit_cols].to_numpy(np.float32)
    Y = get_targets(df)
    N, M = X.shape
    print(f"[INFO] Rows={N}, EIT channels={M}")

    # Quick protocol sanity: what does PyEIT expect for 16 adjacent?
    prot_check = create_protocol(args.n_el, dist_exc=args.dist_exc, step_meas=args.step_meas)
    M_expected = prot_check.n_meas
    print(f"[INFO] Protocol n_meas={M_expected}")
    if M != M_expected:
        print(" [WARN] CSV channel count != protocol n_meas. Still proceeding (may be different mask/order).")

    # Pick subset to evaluate quickly
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(args.n_samples, N), replace=False)
    Xs = X[idx]
    Ys = Y[idx]

    # Grid search rotations x directions
    shifts = list(range(args.n_el))  # 0..15
    dirs = ["cw", "ccw"]
    scores = np.zeros((len(shifts), len(dirs)), dtype=float)

    best = (-1e9, None, None)  # score, shift, dir

    for si, s in enumerate(shifts):
        for di, d in enumerate(dirs):
            mesh_obj, protocol = build_mesh_and_protocol(args.n_el, s, d, args.dist_exc, args.step_meas)
            v0 = compute_baseline(mesh_obj, protocol)

            # channel count guard
            if v0.size != M:
                # Different ordering/measurement parser than CSV -> skip score (very unlikely with std parser)
                scores[si, di] = -1e9
                continue

            # score across subset
            sc = 0.0
            recon_obj = bp.BP(mesh_obj, protocol); recon_obj.setup(weight="none")
            pts = mesh_obj.node; tri = mesh_obj.element
            for row_v, tgt in zip(Xs, Ys):
                if args.assume_delta:
                    v1 = row_v + v0
                else:
                    v1 = row_v
                ds = args.scale * recon_obj.solve(v1, v0, normalize=True)
                sc += score_map(ds, mesh_obj, touches=[(tgt[0],tgt[1]), (tgt[2],tgt[3])])
            sc /= len(Xs)
            scores[si, di] = sc
            if sc > best[0]:
                best = (sc, s, d)

    # Report & plot score matrix
    best_score, best_shift, best_dir = best
    print(f"[RESULT] Best: shift={best_shift} (≈ {best_shift*360/args.n_el:.1f}°), dir={best_dir}, score={best_score:.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    im = ax.imshow(scores, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("direction (0=cw, 1=ccw)")
    ax.set_ylabel("rot_shift (electrodes)")
    ax.set_xticks([0,1]); ax.set_xticklabels(dirs)
    ax.set_yticks(range(len(shifts))); ax.set_yticklabels(shifts)
    plt.colorbar(im, ax=ax, label="alignment score ↑")
    plt.title("Rotation/Direction search score")
    plt.tight_layout(); plt.savefig(out_dir/"score_matrix.png", dpi=220); plt.close(fig)

    # Visualize a few reconstructions with the best hypothesis
    mesh_best, prot_best = build_mesh_and_protocol(args.n_el, best_shift, best_dir, args.dist_exc, args.step_meas)
    v0_best = compute_baseline(mesh_best, prot_best)
    recon_best = bp.BP(mesh_best, prot_best); recon_best.setup(weight="none")

    take = min(6, len(Xs))
    for i in range(take):
        vrow = Xs[i]
        tgt = Ys[i]
        if args.assume_delta:
            v1 = vrow + v0_best
        else:
            v1 = vrow
        ds = args.scale * recon_best.solve(v1, v0_best, normalize=True)
        draw_recon(mesh_best, ds, touches=[(tgt[0],tgt[1]), (tgt[2],tgt[3])],
                   path=out_dir/f"best_recon_{i+1:02d}.png")

    print(f"[INFO] Saved → {out_dir}/score_matrix.png and best_recon_*.png")
    print("     Use the reported (shift, dir) in your training/eval scripts.")
    

if __name__ == "__main__":
    main()
