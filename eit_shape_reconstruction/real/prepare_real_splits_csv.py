#!/usr/bin/env python3
"""
REAL DATA - Model-based evaluation (BP & JAC) on SEEN shapes only.

SIM-style protocol:
1) Load ONLY the real TRAIN dataset (5 seen shapes) from a train_with_masks.csv
2) Split into 80% train / 20% test (stratified by shape_type)
3) Choose best threshold for BP and JAC on the 80% TRAIN split (maximize mean IoU)
4) Evaluate on the 20% TEST split:
   - mean±std IoU & Dice (per-sample)
   - per-shape mean IoU & Dice

Important REAL-DATA detail:
- Real data often contains "inactive" channels (all zeros in data and baseline).
- PyEIT solvers may be configured to operate on the reduced set (e.g., 208),
  so we must consistently mask v0 and each sample to those valid channels.
"""

from pathlib import Path
import numpy as np
import pandas as pd
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
CFG = {
    "TRAIN_WITH_MASKS_CSV": "./real_dataset/real_train_with_masks.csv",
    "DATA_ROOT": "./real_dataset/gt_masks",
    "BASELINE_PATH": "./real_dataset/eit_baseline.npy",

    "N_EL": 16,
    "MESH_H0": 0.04,
    "GRID_SIZE": 64,

    "TRAIN_FRACTION": 0.8,
    "RANDOM_SEED": 123,

    "THRESH_MIN": 0.05,
    "THRESH_MAX": 0.95,
    "THRESH_STEP": 0.01,

    "BP_SCALE": 192.0,
    "BP_NORMALIZE": False,

    "JAC_P": 0.5,
    "JAC_LAMB": 0.01,
    "JAC_METHOD": "kotre",
}

# =========================
# Helpers
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
    return np.ma.filled(grid, fill_value=fill_value)

def minmax01(x):
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    return (x - mn) / (mx - mn + 1e-6)

def mean_std(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return (np.nan, np.nan)
    return (float(arr.mean()), float(arr.std()))

def iou_dice_from_normmap(nm01, gt01, thresh, eps=1e-6):
    pred = (nm01 > thresh).astype(np.uint8)
    gt = (gt01 > 0.5).astype(np.uint8)
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum() - inter
    iou = inter / (union + eps)
    dice = (2.0 * inter) / (pred.sum() + gt.sum() + eps)
    return float(iou), float(dice)

def normalize_shape_name(s: str) -> str:
    if s is None:
        return "unknown"
    s = str(s).strip()
    s_low = s.lower()
    if s_low in ["+", "plus", "plus_shape", "plus-shape"]:
        return "+"
    if s_low in ["doublecircle", "double_circle", "double-circle", "double circle"]:
        return "double_circle"
    if s_low in ["edge", "edge_shape", "edge-shape"]:
        return "edge"
    if s_low in ["ring", "annulus"]:
        return "ring"
    if s_low == "l":
        return "L"
    if s_low == "t":
        return "T"
    return s

def load_baseline(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        v0 = np.load(path)
    else:
        v0 = np.loadtxt(path, delimiter=",")
    return np.asarray(v0, dtype=np.float32).reshape(-1)

def load_gt_mask(mask_path: Path) -> np.ndarray:
    gt = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return (gt > 127).astype(np.uint8)

def setup_solvers():
    mesh_obj = mesh.create(CFG["N_EL"], h0=CFG["MESH_H0"])
    protocol_obj = protocol.create(CFG["N_EL"], dist_exc=1, step_meas=1, parser_meas="std")
    fwd = EITForward(mesh_obj, protocol_obj)

    eit_bp = bp.BP(mesh_obj, protocol_obj)
    eit_bp.setup(weight="none")

    eit_jac = jac.JAC(mesh_obj, protocol_obj)
    eit_jac.setup(p=CFG["JAC_P"], lamb=CFG["JAC_LAMB"], method=CFG["JAC_METHOD"])

    triang, xx, yy = make_grid(mesh_obj, CFG["GRID_SIZE"])
    return mesh_obj, protocol_obj, fwd, eit_bp, eit_jac, triang, xx, yy

def get_eit_columns(df):
    eit_cols = [c for c in df.columns if c.startswith("eit_")]
    if not eit_cols:
        raise RuntimeError("No eit_* columns found in CSV.")
    eit_cols.sort(key=lambda s: int(s.split("_")[1]))
    return eit_cols

# =========================
# ### NEW/CHANGED: channel masking (256 -> 208 etc.)
# =========================
def compute_valid_channel_mask(df, eit_cols, v0_full):
    """
    Replicates your real learning-dataset logic:
    - replace NaNs with 0 for the test
    - find channels that are all-zero across data AND zero in baseline
    - drop those channels
    """
    eit_values = df[eit_cols].to_numpy(dtype=float)
    eit_clean = np.where(np.isnan(eit_values), 0.0, eit_values)

    zero_across_data = np.all(eit_clean == 0.0, axis=0)
    zero_in_baseline = (v0_full == 0.0)

    drop_mask = zero_across_data & zero_in_baseline
    valid_mask = ~drop_mask

    print(f"[info] Raw channels: {len(eit_cols)}")
    print(f"[info] Valid channels after masking: {int(valid_mask.sum())}")
    return valid_mask

def element_to_node_average(mesh_obj, elem_vals):
    tri_e = mesh_obj.element
    n_nodes = mesh_obj.node.shape[0]
    accum = np.zeros(n_nodes, dtype=np.float32)
    counts = np.zeros(n_nodes, dtype=np.float32)
    for e_idx, nodes in enumerate(tri_e):
        accum[nodes] += elem_vals[e_idx]
        counts[nodes] += 1.0
    return accum / np.maximum(counts, 1.0)

def real_normmaps_from_row(row, eit_cols, v0_full, valid_mask,
                           eit_bp, eit_jac, mesh_obj, triang, xx, yy):
    """
    Returns:
      bp_normmap01, jac_normmap01
    Uses masked channels so solver dim matches (e.g., 208).
    """
    vals_full = row[eit_cols].to_numpy(dtype=float)
    vals_full = np.where(np.isnan(vals_full), v0_full, vals_full).astype(np.float32)

    # mask down
    v0 = v0_full[valid_mask]
    vals = vals_full[valid_mask]

    dv = vals - v0
    v1 = v0 + dv

    # BP
    nodal_bp = CFG["BP_SCALE"] * eit_bp.solve(
        v1, v0, normalize=CFG["BP_NORMALIZE"], log_scale=False
    )
    nodal_bp = np.real(nodal_bp).astype(np.float32)
    bp_grid = rasterize_to_grid(triang, xx, yy, nodal_bp, fill_value=0.0)
    bp_nm = minmax01(bp_grid)

    # JAC
    ds_elem = eit_jac.solve(v1, v0, normalize=True)
    ds_elem = np.real(ds_elem).astype(np.float32)
    nodal_j = element_to_node_average(mesh_obj, ds_elem)
    jac_grid = rasterize_to_grid(triang, xx, yy, nodal_j, fill_value=0.0)
    jac_nm = minmax01(jac_grid)

    return bp_nm, jac_nm

def stratified_split_indices(shapes, train_fraction, seed):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    shapes = np.asarray(shapes)

    for s in sorted(set(shapes.tolist())):
        idx = np.where(shapes == s)[0]
        rng.shuffle(idx)
        n_train = int(np.round(train_fraction * idx.size))
        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())

    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)

def choose_best_threshold(df_subset, eit_cols, v0_full, valid_mask, solver_name, nm_getter):
    thresholds = np.arange(CFG["THRESH_MIN"], CFG["THRESH_MAX"] + 1e-9, CFG["THRESH_STEP"], dtype=np.float32)
    best_t, best_iou = None, -1.0

    cache = []
    for _, row in df_subset.iterrows():
        mask_path = Path(CFG["DATA_ROOT"]) / str(row["mask_path"])
        gt = load_gt_mask(mask_path)
        nm = nm_getter(row, eit_cols, v0_full, valid_mask)
        cache.append((nm, gt))

    if len(cache) == 0:
        raise RuntimeError("No samples available for threshold selection.")

    for t in thresholds:
        ious = [iou_dice_from_normmap(nm, gt, float(t))[0] for nm, gt in cache]
        miou = float(np.mean(ious))
        if miou > best_iou:
            best_iou = miou
            best_t = float(t)

    print(f"[thr] {solver_name}: best threshold={best_t:.2f} (train mean IoU={best_iou:.4f})")
    return best_t

def eval_split(df_subset, eit_cols, v0_full, valid_mask, solver_name, nm_getter, thresh):
    ious, dices = [], []
    per_shape = {}

    for _, row in df_subset.iterrows():
        shape = normalize_shape_name(row.get("shape_type", "unknown"))
        mask_path = Path(CFG["DATA_ROOT"]) / str(row["mask_path"])
        gt = load_gt_mask(mask_path)

        nm = nm_getter(row, eit_cols, v0_full, valid_mask)
        iou, dice = iou_dice_from_normmap(nm, gt, thresh)

        ious.append(iou)
        dices.append(dice)
        per_shape.setdefault(shape, {"iou": [], "dice": []})
        per_shape[shape]["iou"].append(iou)
        per_shape[shape]["dice"].append(dice)

    miou, siou = mean_std(ious)
    mdice, sdice = mean_std(dices)

    return {
        "miou": miou, "siou": siou,
        "mdice": mdice, "sdice": sdice,
        "per_shape": per_shape
    }

def print_per_shape(per_shape):
    for s in sorted(per_shape.keys()):
        mi, si = mean_std(per_shape[s]["iou"])
        md, sd = mean_std(per_shape[s]["dice"])
        n = len(per_shape[s]["iou"])
        print(f"  {s:>12s} | N={n:4d} | IoU={mi:.4f} ± {si:.4f} | Dice={md:.4f} ± {sd:.4f}")

# =========================
# MAIN
# =========================
def main():
    train_csv = Path(CFG["TRAIN_WITH_MASKS_CSV"])
    if not train_csv.exists():
        raise FileNotFoundError(f"CSV not found: {train_csv}")

    df = pd.read_csv(train_csv)
    df["shape_type"] = df["shape_type"].apply(normalize_shape_name)

    eit_cols = get_eit_columns(df)

    v0_full = load_baseline(Path(CFG["BASELINE_PATH"]))
    if v0_full.size != len(eit_cols):
        raise ValueError(f"Baseline length {v0_full.size} != EIT columns {len(eit_cols)}")

    mesh_obj, protocol_obj, fwd, eit_bp, eit_jac, triang, xx, yy = setup_solvers()

    # --- NEW: compute valid_channel_mask once ---
    valid_mask = compute_valid_channel_mask(df, eit_cols, v0_full)

    # closures
    def bp_nm_getter(row, eit_cols_, v0_, valid_mask_):
        bp_nm, _ = real_normmaps_from_row(row, eit_cols_, v0_, valid_mask_,
                                          eit_bp, eit_jac, mesh_obj, triang, xx, yy)
        return bp_nm

    def jac_nm_getter(row, eit_cols_, v0_, valid_mask_):
        _, jac_nm = real_normmaps_from_row(row, eit_cols_, v0_, valid_mask_,
                                           eit_bp, eit_jac, mesh_obj, triang, xx, yy)
        return jac_nm

    # split 80/20 stratified
    train_idx, test_idx = stratified_split_indices(
        df["shape_type"].values, CFG["TRAIN_FRACTION"], CFG["RANDOM_SEED"]
    )
    df_tr = df.iloc[train_idx].reset_index(drop=True)
    df_te = df.iloc[test_idx].reset_index(drop=True)

    print(f"[info] Loaded seen-only CSV: {train_csv}")
    print(f"[info] Total={len(df)} | train={len(df_tr)} | test={len(df_te)}")
    print("[info] Test per-shape counts:")
    for s in sorted(df_te["shape_type"].unique()):
        print(f"  {s:>12s}: {int((df_te['shape_type']==s).sum())}")

    # ---- BP ----
    bp_t = choose_best_threshold(df_tr, eit_cols, v0_full, valid_mask, "BP", bp_nm_getter)
    bp_res = eval_split(df_te, eit_cols, v0_full, valid_mask, "BP", bp_nm_getter, bp_t)

    print(f"\n=== BP (seen-only 80/20) | threshold={bp_t:.2f} ===")
    print(f"BP | seen_test | IoU: {bp_res['miou']:.4f} ± {bp_res['siou']:.4f} | Dice: {bp_res['mdice']:.4f} ± {bp_res['sdice']:.4f}")
    print("\nBP per-shape (seen test):")
    print_per_shape(bp_res["per_shape"])

    # ---- JAC ----
    jac_t = choose_best_threshold(df_tr, eit_cols, v0_full, valid_mask, "JAC", jac_nm_getter)
    jac_res = eval_split(df_te, eit_cols, v0_full, valid_mask, "JAC", jac_nm_getter, jac_t)

    print(f"\n=== JAC (seen-only 80/20) | threshold={jac_t:.2f} ===")
    print(f"JAC | seen_test | IoU: {jac_res['miou']:.4f} ± {jac_res['siou']:.4f} | Dice: {jac_res['mdice']:.4f} ± {jac_res['sdice']:.4f}")
    print("\nJAC per-shape (seen test):")
    print_per_shape(jac_res["per_shape"])

    print("\n[done] This fixes the 208 vs 256 mismatch and outputs mean±std like sim.")

if __name__ == "__main__":
    main()
