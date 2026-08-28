#!/usr/bin/env python3
"""
Create presentation-ready BP failure plots for BOTH simulated and real datasets.

Outputs:
- A single PNG figure with 2 rows x 4 columns:
    Row 1: SIM  (L, T, edge, ring)
    Row 2: REAL (L, T, edge, ring)
- Each subplot shows:
    - BP reconstruction heatmap
    - GT contact area as a white contour
    - IoU printed in the title (optional but useful)

Notes:
- SIM uses forward EIT simulation (PyEIT EITForward) + BP.
- REAL uses your logged voltages + baseline + BP, and GT from mask PNG.
- REAL often has inactive channels in CSV (e.g., 256 columns but only 208 valid).
  We apply the same "valid_channel_mask" logic you used before, to avoid shape mismatch.

Edit CONFIG paths to match your repo.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol
from pyeit.eit.fem import EITForward
import pyeit.eit.bp as bp


# =========================
# CONFIG (EDIT ME)
# =========================
CFG = {
    # --- SIM ---
    "SIM_N_EL": 16,
    "SIM_MESH_H0": 0.04,
    "SIM_CONTRAST": 20.0,
    "SIM_BP_SCALE": 192.0,
    "SIM_BP_NORMALIZE": True,

    # --- REAL ---
    "REAL_TRAIN_WITH_MASKS_CSV": "./real_dataset/real_train_with_masks.csv",
    "REAL_DATA_ROOT": "/home/kiyanoush/Projects/eit_shape_reconstruction/real_dataset",
    "REAL_BASELINE_PATH": "./real_dataset/eit_baseline.npy",
    "REAL_N_EL": 16,
    "REAL_MESH_H0": 0.04,
    "REAL_BP_SCALE": 192.0,
    "REAL_BP_NORMALIZE": False,  # matches your real learning dataset logic

    # --- Common ---
    "GRID_SIZE": 64,
    "OUT_FIG": "bp_failure_sim_vs_real.png",

    # Which shapes to show (consistent ordering)
    "SHAPES": ["L", "T", "edge", "ring"],

    # SIM: offsets chosen to make BP struggle a bit (feel free to tweak)
    "SIM_OFFSETS": {
        "L":   (-0.30,  0.00),
        "T":   ( 0.30,  0.00),
        "edge":(-0.30,  0.00),
        "ring":( 0.30,  0.00),
    },

    # REAL: choose the first sample per shape (set to "first").
    # You can later change to "random" or "worst_bp" etc.
    "REAL_PICK_MODE": "first",  # "first" | "random"
    "RANDOM_SEED": 0,
}


# =========================
# Geometry helpers (SIM)
# =========================
def element_centroids(mesh_obj):
    pts = mesh_obj.node
    tri = mesh_obj.element
    return pts[tri].mean(axis=1)

def make_perm_for_mask(mesh_obj, mask, contrast=5.0, base_perm=1.0):
    n_elems = mesh_obj.element.shape[0]
    perm = np.ones(n_elems, dtype=float) * base_perm
    perm[mask] = base_perm * contrast
    return perm

def element_mask_to_nodal(mesh_obj, mask):
    tri = mesh_obj.element
    n_nodes = mesh_obj.node.shape[0]
    nodal = np.zeros(n_nodes, dtype=float)
    for e_idx, nodes in enumerate(tri):
        if mask[e_idx]:
            nodal[nodes] = 1.0
    return nodal

def nodal_to_element_values(mesh_obj, nodal_vals):
    tri = mesh_obj.element
    return nodal_vals[tri].mean(axis=1)

def compute_iou_elem(mask_true_elem, recon_elem_vals, top_frac=0.3):
    """
    SIM-only quick IoU:
    - mask_true_elem is boolean per-element GT
    - recon_elem_vals is per-element reconstruction
    - reconstruction mask = top_frac elements by positive value
    """
    vals = np.clip(recon_elem_vals.astype(np.float32), 0.0, None)
    n = vals.size
    k = max(1, int(top_frac * n))
    idx_sorted = np.argsort(vals)
    recon_mask = np.zeros_like(mask_true_elem, dtype=bool)
    recon_mask[idx_sorted[-k:]] = True

    inter = np.logical_and(mask_true_elem, recon_mask).sum()
    union = np.logical_or(mask_true_elem, recon_mask).sum()
    return float(inter / (union + 1e-6))

# --- SIM shape masks (per-element) ---
def edge_mask(mesh_obj, offset=(0.0, 0.0), x_min=-0.05, x_max=0.05, y_min=-0.6, y_max=0.6):
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    return (x_min <= x) & (x <= x_max) & (y_min <= y) & (y <= y_max)

def T_mask(mesh_obj, offset=(0.0, 0.0)):
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    stem = (np.abs(x) < 0.06) & (y > -0.5) & (y < 0.4)
    bar  = (y > 0.4) & (y < 0.52) & (np.abs(x) < 0.45)
    return stem | bar

def L_mask(mesh_obj, offset=(0.0, 0.0)):
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    vert  = (x > -0.06) & (x < 0.10) & (y > -0.35) & (y < 0.45)
    horiz = (y > -0.45) & (y < -0.30) & (x > -0.06) & (x < 0.50)
    return vert | horiz

def ring_mask(mesh_obj, center=(0.0, 0.0), r_outer=0.35, r_inner=0.18):
    c = element_centroids(mesh_obj)
    dx = c[:, 0] - center[0]
    dy = c[:, 1] - center[1]
    r2 = dx**2 + dy**2
    return (r_inner**2 <= r2) & (r2 <= r_outer**2)

SIM_SHAPE_FN = {
    "L":   lambda m, off: L_mask(m, offset=off),
    "T":   lambda m, off: T_mask(m, offset=off),
    "edge":lambda m, off: edge_mask(m, offset=off),
    "ring":lambda m, off: ring_mask(m, center=off),
}


# =========================
# Grid helpers (REAL plotting)
# =========================
def make_grid(mesh_obj, grid_size):
    pts = mesh_obj.node
    tri = mesh_obj.element
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)
    lin = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(lin, lin)
    return triang, xx, yy

def rasterize_to_grid(triang, xx, yy, nodal_vals, fill_value=0.0):
    interp = mtri.LinearTriInterpolator(triang, nodal_vals)
    grid = interp(xx, yy)
    return np.ma.filled(grid, fill_value=fill_value).astype(np.float32)

def iou_dice_grid(pred01, gt01, eps=1e-6):
    pred = (pred01 > 0.5).astype(np.uint8)
    gt = (gt01 > 0.5).astype(np.uint8)
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum() - inter
    iou = inter / (union + eps)
    dice = (2.0 * inter) / (pred.sum() + gt.sum() + eps)
    return float(iou), float(dice)


# =========================
# REAL: channel masking logic
# =========================
def get_eit_columns(df):
    cols = [c for c in df.columns if c.startswith("eit_")]
    if not cols:
        raise RuntimeError("No eit_* columns found in real CSV.")
    cols.sort(key=lambda s: int(s.split("_")[1]))
    return cols

def load_baseline(path: Path):
    if path.suffix.lower() == ".npy":
        v0 = np.load(path)
    else:
        v0 = np.loadtxt(path, delimiter=",")
    return np.asarray(v0, dtype=np.float32).reshape(-1)

def compute_valid_channel_mask(df, eit_cols, v0_full):
    eit_values = df[eit_cols].to_numpy(dtype=float)
    eit_clean = np.where(np.isnan(eit_values), 0.0, eit_values)
    zero_across_data = np.all(eit_clean == 0.0, axis=0)
    zero_in_baseline = (v0_full == 0.0)
    drop_mask = zero_across_data & zero_in_baseline
    valid_mask = ~drop_mask
    return valid_mask


# =========================
# SIM pipeline
# =========================
def sim_bp_for_shape(shape, offset):
    n_el = CFG["SIM_N_EL"]
    mesh_obj = mesh.create(n_el, h0=CFG["SIM_MESH_H0"])
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")

    fwd = EITForward(mesh_obj, protocol_obj)
    eit_bp = bp.BP(mesh_obj, protocol_obj)
    eit_bp.setup(weight="none")

    # reference
    n_elems = mesh_obj.element.shape[0]
    perm_ref = np.ones(n_elems, dtype=float)
    v0 = fwd.solve_eit(perm_ref).astype(np.float32)

    # shape anomaly
    mask = SIM_SHAPE_FN[shape](mesh_obj, offset)
    perm = make_perm_for_mask(mesh_obj, mask, contrast=CFG["SIM_CONTRAST"])
    v1 = fwd.solve_eit(perm).astype(np.float32)

    dv = v1 - v0
    sum_abs_diff = float(np.sum(np.abs(dv)))

    if sum_abs_diff > 0.3:
        nodal_bp = CFG["SIM_BP_SCALE"] * eit_bp.solve(v1, v0, normalize=CFG["SIM_BP_NORMALIZE"], log_scale=False)
    else:
        nodal_bp = CFG["SIM_BP_SCALE"] * eit_bp.solve(v0, v0, normalize=CFG["SIM_BP_NORMALIZE"], log_scale=False)

    nodal_bp = np.real(nodal_bp).astype(np.float32)
    elem_bp = nodal_to_element_values(mesh_obj, nodal_bp)
    iou = compute_iou_elem(mask, elem_bp, top_frac=0.3)

    return mesh_obj, nodal_bp, mask, iou


# =========================
# REAL pipeline
# =========================
def real_pick_one_row_per_shape(df, shapes):
    rng = np.random.default_rng(CFG["RANDOM_SEED"])
    picked = {}
    for s in shapes:
        sub = df[df["shape_type"] == s]
        if len(sub) == 0:
            raise RuntimeError(f"No real samples found for shape '{s}' in CSV.")
        if CFG["REAL_PICK_MODE"] == "random":
            row = sub.iloc[int(rng.integers(0, len(sub)))]
        else:
            row = sub.iloc[0]
        picked[s] = row
    return picked

def real_bp_from_row(row, eit_cols, v0_full, valid_mask, eit_bp, triang, xx, yy):
    vals_full = row[eit_cols].to_numpy(dtype=float)
    vals_full = np.where(np.isnan(vals_full), v0_full, vals_full).astype(np.float32)

    v0 = v0_full[valid_mask]
    v1 = vals_full[valid_mask]

    dv = v1 - v0
    sum_abs_diff = float(np.sum(np.abs(dv)))

    if sum_abs_diff > 1e-6:
        nodal_bp = CFG["REAL_BP_SCALE"] * eit_bp.solve(v1, v0, normalize=CFG["REAL_BP_NORMALIZE"], log_scale=False)
    else:
        nodal_bp = CFG["REAL_BP_SCALE"] * eit_bp.solve(v0, v0, normalize=CFG["REAL_BP_NORMALIZE"], log_scale=False)

    nodal_bp = np.real(nodal_bp).astype(np.float32)
    bp_grid = rasterize_to_grid(triang, xx, yy, nodal_bp, fill_value=0.0)

    print("Hiii")
    print(str(row["mask_path"])[18:])
    print(str(row["mask_path"]))
    print("----")
    print(CFG["REAL_DATA_ROOT"])
    # load GT mask grid
    cut_path = str(row["mask_path"])[18:]
    mask_path = Path(CFG["REAL_DATA_ROOT"]) / cut_path
    # mask_rel = Path(row["mask_path"])

    # if mask_rel.is_absolute():
    #     mask_path = mask_rel
    # else:
    #     mask_path = (Path(CFG["REAL_DATA_ROOT"]) / mask_rel).resolve()

    # if not mask_path.exists():
    #     raise FileNotFoundError(f"GT mask not found: {mask_path}")
    
    print("Example mask_path from CSV:", row["mask_path"])
    print("Resolved mask_path:", mask_path)

    gt = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    gt = (gt > 127).astype(np.float32)

    # IoU/Dice on grid after simple minmax + threshold(0.5) just for display
    nm = (bp_grid - bp_grid.min()) / (bp_grid.max() - bp_grid.min() + 1e-6)
    pred = (nm > 0.5).astype(np.float32)
    iou, dice = iou_dice_grid(pred, gt)

    return bp_grid, gt, iou, dice


# =========================
# Plotting
# =========================
def main():
    shapes = CFG["SHAPES"]

    # ---------- SIM: generate 4 examples ----------
    sim_data = {}
    for s in shapes:
        off = CFG["SIM_OFFSETS"][s]
        sim_data[s] = sim_bp_for_shape(s, off)

    # ---------- REAL: load csv, baseline, setup solver ----------
    real_csv = Path(CFG["REAL_TRAIN_WITH_MASKS_CSV"])
    df = pd.read_csv(real_csv)
    df["shape_type"] = df["shape_type"].astype(str).str.strip()
    # normalize common variants
    df["shape_type"] = df["shape_type"].replace({"plus": "+", "double circle": "double_circle"})

    eit_cols = get_eit_columns(df)

    v0_full = load_baseline(Path(CFG["REAL_BASELINE_PATH"]))
    if v0_full.size != len(eit_cols):
        raise ValueError(f"Baseline length {v0_full.size} != EIT columns {len(eit_cols)}")

    valid_mask = compute_valid_channel_mask(df, eit_cols, v0_full)
    print(f"[real] raw_channels={len(eit_cols)} | valid_channels={int(valid_mask.sum())}")

    mesh_obj_r = mesh.create(CFG["REAL_N_EL"], h0=CFG["REAL_MESH_H0"])
    protocol_obj_r = protocol.create(CFG["REAL_N_EL"], dist_exc=1, step_meas=1, parser_meas="std")
    eit_bp_r = bp.BP(mesh_obj_r, protocol_obj_r)
    eit_bp_r.setup(weight="none")
    triang_r, xx_r, yy_r = make_grid(mesh_obj_r, CFG["GRID_SIZE"])

    picked_rows = real_pick_one_row_per_shape(df, shapes)

    real_data = {}
    for s in shapes:
        bp_grid, gt_grid, iou, dice = real_bp_from_row(
            picked_rows[s], eit_cols, v0_full, valid_mask, eit_bp_r, triang_r, xx_r, yy_r
        )
        real_data[s] = (bp_grid, gt_grid, iou, dice)

    # ---------- Plot ----------
    fig, axes = plt.subplots(2, len(shapes), figsize=(4.2 * len(shapes), 7.5))

    # SIM colormap range (shared)
    sim_all = np.concatenate([sim_data[s][1].ravel() for s in shapes])  # nodal_bp
    sim_vmin = np.percentile(sim_all, 5)
    sim_vmax = np.percentile(sim_all, 95)

    # REAL colormap range (shared)
    real_all = np.concatenate([real_data[s][0].ravel() for s in shapes])  # bp_grid
    real_vmin = np.percentile(real_all, 5)
    real_vmax = np.percentile(real_all, 95)

    # --- Row 1: SIM ---
    for j, s in enumerate(shapes):
        ax = axes[0, j]
        mesh_obj, nodal_bp, elem_mask, iou = sim_data[s]

        pts = mesh_obj.node
        tri = mesh_obj.element
        triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)

        im = ax.tripcolor(triang, nodal_bp, shading="flat", vmin=sim_vmin, vmax=sim_vmax, cmap="viridis")
        nodal_mask = element_mask_to_nodal(mesh_obj, elem_mask)
        ax.tricontour(triang, nodal_mask, levels=[0.5], colors="white", linewidths=1.2)

        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"SIM {s}\nIoU≈{iou:.2f}")

    # --- Row 2: REAL ---
    for j, s in enumerate(shapes):
        ax = axes[1, j]
        bp_grid, gt_grid, iou, dice = real_data[s]

        im = ax.imshow(bp_grid, vmin=real_vmin, vmax=real_vmax, cmap="viridis", origin="lower")
        ax.contour(gt_grid, levels=[0.5], colors="white", linewidths=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"REAL {s}\nIoU={iou:.2f}, Dice={dice:.2f}")

    # Labels
    axes[0, 0].set_ylabel("Simulated (BP)", fontsize=13)
    axes[1, 0].set_ylabel("Real (BP)", fontsize=13)

    # One colorbar for each row
    cbar0 = fig.colorbar(axes[0, 0].collections[0], ax=axes[0, :].tolist(), shrink=0.85, pad=0.02)
    cbar0.set_label("BP reconstruction (SIM)", rotation=90)

    cbar1 = fig.colorbar(axes[1, 0].images[0], ax=axes[1, :].tolist(), shrink=0.85, pad=0.02)
    cbar1.set_label("BP reconstruction (REAL)", rotation=90)

    fig.suptitle(
        "Model-based BP Contact Reconstruction: Failure Examples\n"
        "Heatmap = BP reconstruction, White contour = Ground Truth contact area",
        fontsize=15
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = Path(CFG["OUT_FIG"])
    fig.savefig(out_path, dpi=200)
    print(f"[done] Saved figure to: {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
