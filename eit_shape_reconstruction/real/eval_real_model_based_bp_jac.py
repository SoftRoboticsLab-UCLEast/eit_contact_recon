#!/usr/bin/env python3
"""
Evaluate BP and JAC on REAL dataset.

We treat:
- SEEN shapes  = {T, L, ring, edge, double_circle}
- UNSEEN shapes = {C, Z, plus}  (plus may appear as "+" or "plus" depending on your labeling)

Evaluation:
1) Choose best binarization threshold for BP and JAC using TRAIN split, SEEN shapes only
   (maximize mean IoU).
2) Evaluate on TEST split:
   - Seen(test): mean±std IoU & Dice
   - Unseen(test): mean±std IoU & Dice
   - Overall(test): mean±std IoU & Dice
   - Per-shape means for seen/unseen

Notes:
- For real data, "unseen" should come from your real test CSV (not generated).
- BP and JAC maps are min-max normalized to [0,1] per sample before thresholding.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.tri as mtri

import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol
import pyeit.eit.bp as bp
import pyeit.eit.jac as jac


# =========================
# CONFIG (EDIT)
# =========================
CFG = {
    # These are the consolidated split CSVs you generated
    "TRAIN_CSV": "./real_dataset/real_train_with_masks.csv",
    "TEST_CSV": "./real_dataset/real_test_with_masks.csv",
    "DATA_ROOT": "./real_dataset/gt_masks",   # used to resolve mask_path

    # Baseline file saved earlier
    "BASELINE_PATH": "./real_dataset/eit_baseline.npy",

    "GRID_SIZE": 64,
    "N_EL": 16,
    "MESH_H0": 0.04,

    # Your shape labels (adjust if your CSV uses different strings)
    "SEEN_SHAPES": {"T", "L", "ring", "edge", "double_circle"},
    "UNSEEN_SHAPES": {"C", "Z", "+", "plus"},  # accept both "+" and "plus"
    # If your real CSV uses "double_circle_touch" or similar, update above.

    # Threshold search space
    "THRESH_MIN": 0.05,
    "THRESH_MAX": 0.95,
    "THRESH_STEP": 0.01,

    # BP/JAC configuration
    "JAC_P": 0.5,
    "JAC_LAMB": 0.01,
    "JAC_METHOD": "kotre",
}


# =========================
# Helpers
# =========================
def load_baseline(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        v0 = np.load(path)
    else:
        v0 = np.loadtxt(path, delimiter=",")
    v0 = np.asarray(v0, dtype=np.float32).reshape(-1)
    return v0


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
    grid = np.ma.filled(grid, fill_value=fill_value)
    return grid


def element_to_node_average(mesh_obj, elem_vals):
    tri_e = mesh_obj.element
    n_nodes = mesh_obj.node.shape[0]
    accum = np.zeros(n_nodes, dtype=np.float32)
    counts = np.zeros(n_nodes, dtype=np.float32)
    for e_idx, nodes in enumerate(tri_e):
        accum[nodes] += elem_vals[e_idx]
        counts[nodes] += 1.0
    return accum / np.maximum(counts, 1.0)


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
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return (np.nan, np.nan)
    return (float(arr.mean()), float(arr.std()))


def load_gt_mask(data_root: Path, mask_rel_path: str) -> np.ndarray:
    p = data_root / mask_rel_path
    gt = np.array(Image.open(p).convert("L"), dtype=np.uint8)
    # force binary
    gt = (gt > 127).astype(np.uint8)
    return gt


def get_eit_cols(df: pd.DataFrame):
    eit_cols = [c for c in df.columns if c.startswith("eit_")]
    if not eit_cols:
        raise RuntimeError("No eit_* columns found in CSV.")
    eit_cols.sort(key=lambda s: int(s.split("_")[1]))
    return eit_cols


def sanitize_full_sample(vals_full: np.ndarray, baseline_full: np.ndarray) -> np.ndarray:
    """
    Replace NaNs in sample with baseline values (so delta_v=0 there).
    """
    vals_full = np.asarray(vals_full, dtype=np.float32)
    baseline_full = np.asarray(baseline_full, dtype=np.float32)
    vals_full = np.where(np.isnan(vals_full), baseline_full, vals_full)
    return vals_full.astype(np.float32)


def compute_valid_channel_mask(df_train: pd.DataFrame, eit_cols, baseline_full: np.ndarray) -> np.ndarray:
    """
    Drop channels that are always zero in the training data AND zero in baseline.
    (Matches what you did for learning-based models.)
    """
    eit_values = df_train[eit_cols].to_numpy(dtype=np.float32)
    eit_clean = np.where(np.isnan(eit_values), 0.0, eit_values)
    zero_across_data = np.all(eit_clean == 0.0, axis=0)
    zero_in_baseline = (baseline_full == 0.0)
    drop_mask = zero_across_data & zero_in_baseline
    return ~drop_mask  # valid_channel_mask


# =========================
# EIT setup (BP + JAC)
# =========================
def setup_eit(n_el, h0, jac_p, jac_lamb, jac_method):
    mesh_obj = mesh.create(n_el, h0=h0)
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")

    eit_bp = bp.BP(mesh_obj, protocol_obj)
    eit_bp.setup(weight="none")

    eit_jac = jac.JAC(mesh_obj, protocol_obj)
    eit_jac.setup(p=jac_p, lamb=jac_lamb, method=jac_method)

    return mesh_obj, protocol_obj, eit_bp, eit_jac


# =========================
# Normmap builders (REAL)
# =========================
def bp_normmap_from_row(vals_full, v0_full, valid_mask, eit_bp, triang, xx, yy):
    vals_full = sanitize_full_sample(vals_full, v0_full)

    v1 = vals_full[valid_mask]
    v0 = v0_full[valid_mask]

    # PyEIT BP expects v1,v0 arrays aligned with protocol.
    nodal_bp = 192.0 * eit_bp.solve(v1, v0, normalize=False, log_scale=False)
    nodal_bp = np.real(nodal_bp).astype(np.float32)

    grid = rasterize_to_grid(triang, xx, yy, nodal_bp, fill_value=0.0)
    return minmax01(grid)


def jac_normmap_from_row(vals_full, v0_full, valid_mask, eit_jac, mesh_obj, triang, xx, yy):
    vals_full = sanitize_full_sample(vals_full, v0_full)

    v1 = vals_full[valid_mask]
    v0 = v0_full[valid_mask]

    ds_elem = eit_jac.solve(v1, v0, normalize=False)
    ds_elem = np.real(ds_elem).astype(np.float32)

    nodal = element_to_node_average(mesh_obj, ds_elem)
    grid = rasterize_to_grid(triang, xx, yy, nodal, fill_value=0.0)
    return minmax01(grid)


# =========================
# Threshold selection on TRAIN/SEEN
# =========================
def choose_best_threshold_on_train_seen(df_train, data_root, eit_cols, v0_full, valid_mask,
                                       solver_name, nm_fn, seen_shapes,
                                       tmin, tmax, tstep):
    thresholds = np.arange(tmin, tmax + 1e-9, tstep, dtype=np.float32)

    pairs = []
    for _, row in df_train.iterrows():
        shape = str(row.get("shape_type", ""))
        if shape not in seen_shapes:
            continue
        gt = load_gt_mask(data_root, row["mask_path"])
        vals_full = row[eit_cols].to_numpy(dtype=np.float32)
        nm = nm_fn(vals_full, v0_full, valid_mask)
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
# Evaluation on TEST (seen/unseen)
# =========================
def eval_split(df, data_root, eit_cols, v0_full, valid_mask,
               nm_fn, thresh, seen_shapes, unseen_shapes):
    """
    Returns dict with:
      seen: ious,dices, per_shape
      unseen: ious,dices, per_shape
      overall: ious,dices
    """
    out = {
        "seen": {"ious": [], "dices": [], "per_shape": {}},
        "unseen": {"ious": [], "dices": [], "per_shape": {}},
        "overall": {"ious": [], "dices": []},
    }

    for _, row in df.iterrows():
        shape = str(row.get("shape_type", ""))
        vals_full = row[eit_cols].to_numpy(dtype=np.float32)

        gt = load_gt_mask(data_root, row["mask_path"])
        nm = nm_fn(vals_full, v0_full, valid_mask)
        iou, dice = iou_dice_from_normmap(nm, gt, thresh)

        # overall
        out["overall"]["ious"].append(iou)
        out["overall"]["dices"].append(dice)

        # seen/unseen
        if shape in seen_shapes:
            bucket = out["seen"]
        elif shape in unseen_shapes:
            bucket = out["unseen"]
        else:
            # ignore unknown shapes (or you can track them)
            continue

        bucket["ious"].append(iou)
        bucket["dices"].append(dice)
        bucket["per_shape"].setdefault(shape, {"iou": [], "dice": []})
        bucket["per_shape"][shape]["iou"].append(iou)
        bucket["per_shape"][shape]["dice"].append(dice)

    return out


def print_summary_block(solver_name, tag, ious, dices):
    miou, siou = mean_std(ious)
    md, sd = mean_std(dices)
    print(f"{solver_name} | {tag:7s} | IoU: {miou:.4f} ± {siou:.4f} | Dice: {md:.4f} ± {sd:.4f}")


def print_per_shape(per_shape_dict):
    for shape in sorted(per_shape_dict.keys()):
        miou, _ = mean_std(per_shape_dict[shape]["iou"])
        md, _ = mean_std(per_shape_dict[shape]["dice"])
        n = len(per_shape_dict[shape]["iou"])
        print(f"  {shape:>14s} | N={n:4d} | IoU={miou:.4f} | Dice={md:.4f}")


# =========================
# Main
# =========================
def main():
    cfg = CFG
    data_root = Path(cfg["DATA_ROOT"])

    df_train = pd.read_csv(cfg["TRAIN_CSV"])
    df_test = pd.read_csv(cfg["TEST_CSV"])
    eit_cols = get_eit_cols(df_train)

    v0_full = load_baseline(Path(cfg["BASELINE_PATH"]))
    if v0_full.size != len(eit_cols):
        raise ValueError(f"Baseline length {v0_full.size} != #eit cols {len(eit_cols)}")

    # Valid channel mask computed from TRAIN (matches learning-based)
    valid_mask = compute_valid_channel_mask(df_train, eit_cols, v0_full)
    print(f"[info] Raw channels: {len(eit_cols)} | Valid channels: {int(valid_mask.sum())}")

    # Setup mesh+solvers and grid (use same as learning scripts)
    mesh_obj, protocol_obj, eit_bp, eit_jac = setup_eit(
        n_el=cfg["N_EL"],
        h0=cfg["MESH_H0"],
        jac_p=cfg["JAC_P"],
        jac_lamb=cfg["JAC_LAMB"],
        jac_method=cfg["JAC_METHOD"],
    )
    triang, xx, yy = make_grid(mesh_obj, grid_size=cfg["GRID_SIZE"])

    # Build nm functions
    bp_nm_fn = lambda vals_full, v0, vm: bp_normmap_from_row(vals_full, v0, vm, eit_bp, triang, xx, yy)
    jac_nm_fn = lambda vals_full, v0, vm: jac_normmap_from_row(vals_full, v0, vm, eit_jac, mesh_obj, triang, xx, yy)

    seen_shapes = set(cfg["SEEN_SHAPES"])
    unseen_shapes = set(cfg["UNSEEN_SHAPES"])

    # -------- threshold selection (TRAIN, SEEN only) --------
    bp_t = choose_best_threshold_on_train_seen(
        df_train, data_root, eit_cols, v0_full, valid_mask,
        "BP", bp_nm_fn, seen_shapes,
        cfg["THRESH_MIN"], cfg["THRESH_MAX"], cfg["THRESH_STEP"]
    )
    jac_t = choose_best_threshold_on_train_seen(
        df_train, data_root, eit_cols, v0_full, valid_mask,
        "JAC", jac_nm_fn, seen_shapes,
        cfg["THRESH_MIN"], cfg["THRESH_MAX"], cfg["THRESH_STEP"]
    )


    shape_series = df_test["shape_type"].astype(str).str.strip()
    seen_count = shape_series.isin(CFG["SEEN_SHAPES"]).sum()
    unseen_count = shape_series.isin(CFG["UNSEEN_SHAPES"]).sum()

    print("[debug] TEST total:", len(df_test))
    print("[debug] TEST seen_count:", int(seen_count))
    print("[debug] TEST unseen_count:", int(unseen_count))
    print("[debug] TEST unknown_count:", int(len(df_test) - seen_count - unseen_count))



    # -------- evaluation on TEST --------
    bp_res = eval_split(df_test, data_root, eit_cols, v0_full, valid_mask,
                        bp_nm_fn, bp_t, seen_shapes, unseen_shapes)
    jac_res = eval_split(df_test, data_root, eit_cols, v0_full, valid_mask,
                         jac_nm_fn, jac_t, seen_shapes, unseen_shapes)

    # -------- print results --------
    print(f"\n=== BP (threshold={bp_t:.2f}) ===")
    print_summary_block("BP", "seen", bp_res["seen"]["ious"], bp_res["seen"]["dices"])
    print_summary_block("BP", "unseen", bp_res["unseen"]["ious"], bp_res["unseen"]["dices"])
    print_summary_block("BP", "overall", bp_res["overall"]["ious"], bp_res["overall"]["dices"])

    print("\nBP per-shape (seen, TEST):")
    print_per_shape(bp_res["seen"]["per_shape"])
    print("\nBP per-shape (unseen, TEST):")
    print_per_shape(bp_res["unseen"]["per_shape"])

    print(f"\n=== JAC (threshold={jac_t:.2f}) ===")
    print_summary_block("JAC", "seen", jac_res["seen"]["ious"], jac_res["seen"]["dices"])
    print_summary_block("JAC", "unseen", jac_res["unseen"]["ious"], jac_res["unseen"]["dices"])
    print_summary_block("JAC", "overall", jac_res["overall"]["ious"], jac_res["overall"]["dices"])

    print("\nJAC per-shape (seen, TEST):")
    print_per_shape(jac_res["seen"]["per_shape"])
    print("\nJAC per-shape (unseen, TEST):")
    print_per_shape(jac_res["unseen"]["per_shape"])

    print("\n[done] Paste the mean±std values into your table.")


if __name__ == "__main__":
    main()
