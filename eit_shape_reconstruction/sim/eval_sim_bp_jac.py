#!/usr/bin/env python3
"""
Evaluate BP and JAC on:
1) SEEN shapes: from existing dataset CSV split=test
2) UNSEEN shapes: generated on-the-fly (C, Z, +), like your separate script

Also:
- Automatically chooses the best binarization threshold for BP and JAC
  using TRAIN split (SEEN shapes only), maximizing mean IoU.

Outputs (for BP and for JAC):
- Threshold selected on train/seen
- Seen(test CSV): mean±std IoU & Dice
- Unseen(generated): mean±std IoU & Dice
- Overall (seen+unseen): mean±std IoU & Dice
- Per-shape means for seen and unseen
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.tri as mtri
from PIL import Image

import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol
from pyeit.eit.fem import EITForward
import pyeit.eit.bp as bp
import pyeit.eit.jac as jac

# =========================
# CONFIG
# =========================
DATA_ROOT = Path("eit_dataset")
CSV_PATH = DATA_ROOT / "voltages.csv"
GRID_SIZE = 64

N_EL = 16
MESH_H0 = 0.04

SEEN_SHAPES = {"T", "L", "ring", "edge", "double_circle"}
# in your dataset CSV, your plus might be stored as "+"
UNSEEN_SHAPES = {"C", "Z", "+"}

# For generated unseen evaluation (on-the-fly)
N_SAMPLES_PER_UNSEEN_SHAPE = 300
CONTRAST_LEVELS = [3.0, 5.0, 10.0, 15.0, 20.0]
RANDOM_SEED = 123

# Threshold search space (on normalized [0,1] maps)
THRESH_MIN = 0.05
THRESH_MAX = 0.95
THRESH_STEP = 0.01


# =========================
# Mesh / EIT setup
# =========================
def setup_eit(n_el=N_EL, h0=MESH_H0):
    mesh_obj = mesh.create(n_el, h0=h0)

    protocol_obj = protocol.create(
        n_el, dist_exc=1, step_meas=1, parser_meas="std"
    )

    fwd = EITForward(mesh_obj, protocol_obj)

    n_elems = mesh_obj.element.shape[0]
    perm_ref = np.ones(n_elems, dtype=float)
    v0 = fwd.solve_eit(perm_ref).astype(np.float32)

    eit_bp = bp.BP(mesh_obj, protocol_obj)
    eit_bp.setup(weight="none")

    eit_jac = jac.JAC(mesh_obj, protocol_obj)
    eit_jac.setup(p=0.5, lamb=0.01, method="kotre")  # baseline; tune later if needed

    return mesh_obj, protocol_obj, fwd, eit_bp, eit_jac, v0


def make_grid(mesh_obj, grid_size=GRID_SIZE):
    pts = mesh_obj.node
    tri = mesh_obj.element
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)

    lin = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(lin, lin)
    return triang, xx, yy


def rasterize_to_grid(triang, xx, yy, nodal_vals, fill_value=0.0):
    interp = mtri.LinearTriInterpolator(triang, nodal_vals)
    grid = interp(xx, yy)
    grid = np.ma.filled(grid, fill_value=fill_value)
    return grid


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


def rotate_points(x, y, angle_rad):
    ca = np.cos(angle_rad)
    sa = np.sin(angle_rad)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    return xr, yr


def element_to_node_average(mesh_obj, elem_vals):
    tri_e = mesh_obj.element
    n_nodes = mesh_obj.node.shape[0]
    accum = np.zeros(n_nodes, dtype=np.float32)
    counts = np.zeros(n_nodes, dtype=np.float32)
    for e_idx, nodes in enumerate(tri_e):
        accum[nodes] += elem_vals[e_idx]
        counts[nodes] += 1.0
    return accum / np.maximum(counts, 1.0)


# =========================
# CSV loading (seen data)
# =========================
def load_rows(csv_path, split):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                rows.append(row)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise RuntimeError("Could not read CSV header.")

    volt_col_start = 5  # after [sample_id, split, shape_type, contrast, mask_path]
    n_meas = len(fieldnames) - volt_col_start
    return rows, n_meas


def load_delta_v_from_csv_row(row, n_meas):
    dv = np.zeros(n_meas, dtype=np.float32)
    for i in range(n_meas):
        dv[i] = float(row[f"v_{i}"])
    return dv


def load_gt_mask_from_csv_row(row):
    mask_path = DATA_ROOT / row["mask_path"]
    gt = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    gt = (gt > 127).astype(np.uint8)
    return gt


# =========================
# Normalization + metrics
# =========================
def minmax01(x):
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    return (x - mn) / (mx - mn + 1e-6)


def iou_dice_from_normmap(norm_map01, gt01, thresh, eps=1e-6):
    pred = (norm_map01 > thresh).astype(np.uint8)
    gt = (gt01 > 0.5).astype(np.uint8)

    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum() - inter
    iou = inter / (union + eps)

    dice = (2.0 * inter) / (pred.sum() + gt.sum() + eps)
    return float(iou), float(dice)


def mean_std(arr):
    arr = np.array(arr, dtype=np.float32)
    if arr.size == 0:
        return (np.nan, np.nan)
    return (float(arr.mean()), float(arr.std()))


# =========================
# Solver-to-normmap (CSV rows)
# =========================
def bp_normmap_from_csv_row(row, n_meas, v0, eit_bp, triang, xx, yy):
    dv = load_delta_v_from_csv_row(row, n_meas)
    v1 = v0 + dv
    nodal_bp = 192.0 * eit_bp.solve(v1, v0, normalize=True, log_scale=False)
    nodal_bp = np.real(nodal_bp).astype(np.float32)
    grid = rasterize_to_grid(triang, xx, yy, nodal_bp, fill_value=0.0)
    return minmax01(grid)


def jac_normmap_from_csv_row(row, n_meas, v0, eit_jac, mesh_obj, triang, xx, yy):
    dv = load_delta_v_from_csv_row(row, n_meas)
    v1 = v0 + dv
    ds_elem = eit_jac.solve(v1, v0, normalize=True)
    ds_elem = np.real(ds_elem).astype(np.float32)
    nodal = element_to_node_average(mesh_obj, ds_elem)
    grid = rasterize_to_grid(triang, xx, yy, nodal, fill_value=0.0)
    return minmax01(grid)


# =========================
# UNSEEN shape generation (on-the-fly)
# =========================
def C_mask(mesh_obj, offset=(0.0, 0.0), angle=0.0,
           r_outer=0.35, r_inner=0.18, gap_angle=np.pi/4):
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    x_local, y_local = rotate_points(x, y, -angle)

    r2 = x_local**2 + y_local**2
    theta = np.arctan2(y_local, x_local)
    annulus = (r_inner**2 <= r2) & (r2 <= r_outer**2)
    gap = (np.abs(theta) < gap_angle)  # opening around +x
    return annulus & (~gap)


def plus_mask_rotated(mesh_obj, offset=(0.0, 0.0), angle=0.0,
                      arm_width=0.12, arm_length=0.50):
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    x_local, y_local = rotate_points(x, y, -angle)

    vert = (np.abs(x_local) < arm_width/2) & (np.abs(y_local) < arm_length/2)
    horiz = (np.abs(y_local) < arm_width/2) & (np.abs(x_local) < arm_length/2)
    return vert | horiz


def Z_mask_rotated(mesh_obj, offset=(0.0, 0.0), angle=0.0,
                   width=0.10, length=0.60):
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    x_local, y_local = rotate_points(x, y, -angle)

    half_L = length / 2.0
    top = (y_local > 0.25) & (y_local < 0.35) & (x_local > -half_L) & (x_local < half_L)
    bottom = (y_local > -0.35) & (y_local < -0.25) & (x_local > -half_L) & (x_local < half_L)

    diag_band = (y_local > -0.25) & (y_local < 0.25)
    dist_to_diag = np.abs(y_local + x_local) / np.sqrt(2.0)
    diag = diag_band & (dist_to_diag < width/2.0)

    return top | bottom | diag


def random_unseen_shape_mask(mesh_obj, rng, shape_type):
    contrast = float(rng.choice(CONTRAST_LEVELS))
    ox = rng.uniform(-0.2, 0.2)
    oy = rng.uniform(-0.2, 0.2)
    angle = rng.uniform(0.0, 2.0 * np.pi)

    if shape_type == "C":
        mask = C_mask(mesh_obj, offset=(ox, oy), angle=angle)
    elif shape_type == "+":
        mask = plus_mask_rotated(mesh_obj, offset=(ox, oy), angle=angle)
    elif shape_type == "Z":
        mask = Z_mask_rotated(mesh_obj, offset=(ox, oy), angle=angle)
    else:
        raise ValueError(f"Unknown unseen shape type: {shape_type}")

    return mask, contrast


def generate_unseen_samples(mesh_obj, fwd, v0, eit_bp, eit_jac, triang, xx, yy,
                            n_samples_per_shape=N_SAMPLES_PER_UNSEEN_SHAPE,
                            seed=RANDOM_SEED):
    """
    Generate unseen samples and return per-sample normmaps and GT masks for evaluation.

    Returns:
        dict[solver_name]["samples"] -> list of (normmap01, gt01, shape_type)
            solver_name in {"BP","JAC"}
    """
    rng = np.random.default_rng(seed)

    out = {"BP": [], "JAC": []}

    for shape in ["C", "Z", "+"]:
        for _ in range(n_samples_per_shape):
            elem_mask, contrast = random_unseen_shape_mask(mesh_obj, rng, shape)

            perm = make_perm_for_mask(mesh_obj, elem_mask, contrast=contrast)
            v1 = fwd.solve_eit(perm).astype(np.float32)
            dv = v1 - v0

            # GT grid from mask
            nodal_mask = element_mask_to_nodal(mesh_obj, elem_mask)
            gt_grid_f = rasterize_to_grid(triang, xx, yy, nodal_mask, fill_value=0.0)
            gt = (gt_grid_f > 0.5).astype(np.uint8)

            # BP normmap
            nodal_bp = 192.0 * eit_bp.solve(v1, v0, normalize=True, log_scale=False)
            nodal_bp = np.real(nodal_bp).astype(np.float32)
            bp_grid = rasterize_to_grid(triang, xx, yy, nodal_bp, fill_value=0.0)
            bp_nm = minmax01(bp_grid)

            # JAC normmap
            ds_elem = eit_jac.solve(v1, v0, normalize=True)
            ds_elem = np.real(ds_elem).astype(np.float32)
            nodal_j = element_to_node_average(mesh_obj, ds_elem)
            jac_grid = rasterize_to_grid(triang, xx, yy, nodal_j, fill_value=0.0)
            jac_nm = minmax01(jac_grid)

            out["BP"].append((bp_nm, gt, shape))
            out["JAC"].append((jac_nm, gt, shape))

    return out


# =========================
# Threshold selection on TRAIN/SEEN (CSV)
# =========================
def choose_best_threshold_on_train_seen(rows_train, n_meas, solver_name, nm_fn):
    """
    Choose threshold maximizing mean IoU on TRAIN split, SEEN shapes only.
    nm_fn(row)->norm_map01
    """
    thresholds = np.arange(THRESH_MIN, THRESH_MAX + 1e-9, THRESH_STEP, dtype=np.float32)

    pairs = []
    for row in rows_train:
        shape = row["shape_type"]
        if shape not in SEEN_SHAPES:
            continue
        gt = load_gt_mask_from_csv_row(row)
        nm = nm_fn(row)
        pairs.append((nm, gt))

    if len(pairs) == 0:
        raise RuntimeError("No TRAIN/SEEN samples found for threshold selection.")

    best_t = None
    best_iou = -1.0

    for t in thresholds:
        ious = []
        for nm, gt in pairs:
            iou, _ = iou_dice_from_normmap(nm, gt, float(t))
            ious.append(iou)
        miou = float(np.mean(ious))
        if miou > best_iou:
            best_iou = miou
            best_t = float(t)

    print(f"[thr] {solver_name}: best threshold={best_t:.2f} (train/seen mean IoU={best_iou:.4f})")
    return best_t


# =========================
# Evaluation helpers
# =========================
def eval_seen_from_csv(rows_test, n_meas, solver_name, nm_fn, thresh):
    """
    Evaluate only SEEN shapes from test CSV.
    Returns metrics dict with per-shape and overall.
    """
    ious, dices = [], []
    per_shape = {}

    for row in rows_test:
        shape = row["shape_type"]
        if shape not in SEEN_SHAPES:
            continue
        gt = load_gt_mask_from_csv_row(row)
        nm = nm_fn(row)
        iou, dice = iou_dice_from_normmap(nm, gt, thresh)

        ious.append(iou)
        dices.append(dice)
        per_shape.setdefault(shape, {"iou": [], "dice": []})
        per_shape[shape]["iou"].append(iou)
        per_shape[shape]["dice"].append(dice)

    return {"ious": ious, "dices": dices, "per_shape": per_shape}


def eval_unseen_generated(unseen_samples, solver_name, thresh):
    """
    unseen_samples: list of (normmap01, gt01, shape_type)
    """
    ious, dices = [], []
    per_shape = {}

    for nm, gt, shape in unseen_samples:
        iou, dice = iou_dice_from_normmap(nm, gt, thresh)
        ious.append(iou)
        dices.append(dice)
        per_shape.setdefault(shape, {"iou": [], "dice": []})
        per_shape[shape]["iou"].append(iou)
        per_shape[shape]["dice"].append(dice)

    return {"ious": ious, "dices": dices, "per_shape": per_shape}


def print_summary_block(solver_name, tag, ious, dices):
    miou, siou = mean_std(ious)
    md, sd = mean_std(dices)
    print(f"{solver_name} | {tag:6s} | IoU: {miou:.4f} ± {siou:.4f} | Dice: {md:.4f} ± {sd:.4f}")


def print_per_shape(per_shape_dict):
    for shape in sorted(per_shape_dict.keys()):
        miou, _ = mean_std(per_shape_dict[shape]["iou"])
        md, _ = mean_std(per_shape_dict[shape]["dice"])
        n = len(per_shape_dict[shape]["iou"])
        print(f"  {shape:>12s} | N={n:4d} | IoU={miou:.4f} | Dice={md:.4f}")


# =========================
# Main
# =========================
def main():
    rows_train, n_meas_train = load_rows(CSV_PATH, split="train")
    rows_test, n_meas_test = load_rows(CSV_PATH, split="test")
    if n_meas_train != n_meas_test:
        raise RuntimeError("n_meas mismatch between train and test CSV parsing.")
    n_meas = n_meas_train

    print(f"[info] Train rows: {len(rows_train)} | Test rows: {len(rows_test)} | n_meas={n_meas}")

    mesh_obj, protocol_obj, fwd, eit_bp, eit_jac, v0 = setup_eit()
    triang, xx, yy = make_grid(mesh_obj, grid_size=GRID_SIZE)

    # normmap functions for CSV rows
    bp_nm_fn = lambda row: bp_normmap_from_csv_row(row, n_meas, v0, eit_bp, triang, xx, yy)
    jac_nm_fn = lambda row: jac_normmap_from_csv_row(row, n_meas, v0, eit_jac, mesh_obj, triang, xx, yy)

    # Choose thresholds on TRAIN/SEEN only
    bp_thresh = choose_best_threshold_on_train_seen(rows_train, n_meas, "BP", bp_nm_fn)
    jac_thresh = choose_best_threshold_on_train_seen(rows_train, n_meas, "JAC", jac_nm_fn)

    # Generate unseen samples once (reused for both thresholds)
    print(f"\n[gen] Generating unseen samples: {N_SAMPLES_PER_UNSEEN_SHAPE} per shape (C, Z, +)")
    unseen_data = generate_unseen_samples(
        mesh_obj, fwd, v0, eit_bp, eit_jac, triang, xx, yy,
        n_samples_per_shape=N_SAMPLES_PER_UNSEEN_SHAPE,
        seed=RANDOM_SEED
    )

    # --- Evaluate BP ---
    bp_seen = eval_seen_from_csv(rows_test, n_meas, "BP", bp_nm_fn, bp_thresh)
    bp_unseen = eval_unseen_generated(unseen_data["BP"], "BP", bp_thresh)
    bp_all_ious = bp_seen["ious"] + bp_unseen["ious"]
    bp_all_dices = bp_seen["dices"] + bp_unseen["dices"]

    print(f"\n=== BP (threshold={bp_thresh:.2f}) ===")
    print_summary_block("BP", "seen", bp_seen["ious"], bp_seen["dices"])
    print_summary_block("BP", "unseen", bp_unseen["ious"], bp_unseen["dices"])
    print_summary_block("BP", "overall", bp_all_ious, bp_all_dices)

    print("\nBP per-shape (seen, from CSV test):")
    print_per_shape(bp_seen["per_shape"])
    print("\nBP per-shape (unseen, generated):")
    print_per_shape(bp_unseen["per_shape"])

    # --- Evaluate JAC ---
    jac_seen = eval_seen_from_csv(rows_test, n_meas, "JAC", jac_nm_fn, jac_thresh)
    jac_unseen = eval_unseen_generated(unseen_data["JAC"], "JAC", jac_thresh)
    jac_all_ious = jac_seen["ious"] + jac_unseen["ious"]
    jac_all_dices = jac_seen["dices"] + jac_unseen["dices"]

    print(f"\n=== JAC (threshold={jac_thresh:.2f}) ===")
    print_summary_block("JAC", "seen", jac_seen["ious"], jac_seen["dices"])
    print_summary_block("JAC", "unseen", jac_unseen["ious"], jac_unseen["dices"])
    print_summary_block("JAC", "overall", jac_all_ious, jac_all_dices)

    print("\nJAC per-shape (seen, from CSV test):")
    print_per_shape(jac_seen["per_shape"])
    print("\nJAC per-shape (unseen, generated):")
    print_per_shape(jac_unseen["per_shape"])

    print("\n[done] You can paste the SEEN/UNSEEN mean±std numbers into the table.")


if __name__ == "__main__":
    main()
