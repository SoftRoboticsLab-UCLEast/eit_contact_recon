#!/usr/bin/env python3
"""
Generate ground-truth binary contact masks for REAL EIT dataset
organized as:

real_dataset/
├── train/
│   ├── L/
│   │   ├── *.csv / *.parquet
│   ├── T/
│   └── ...
├── test/
│   ├── C/
│   ├── Z/
│   └── plus/

Outputs:
├── gt_masks/
│   ├── train/*.png
│   └── test/*.png
├── real_train_with_masks.csv
├── real_test_with_masks.csv
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# =========================
# CONFIG
# =========================
CONFIG = {
    "DATA_ROOT": "./data/dataset",   # <-- root containing train/ and test/
    "SENSOR_RADIUS_M": 0.045,
    "GRID_SIZE": 64,
    "OUT_MASK_ROOT": "./real_dataset/gt_masks",

    "COL_SHAPE_TYPE": "shape_type",
    "COL_U_M": "contact_u_m",
    "COL_V_M": "contact_v_m",
    "COL_YAW": "yaw_rad",
}

# =========================
# GRID HELPERS
# =========================
def make_normalized_grid(grid_size):
    lin = np.linspace(-1.0, 1.0, grid_size)
    return np.meshgrid(lin, lin)


def world_to_local(xx, yy, cx, cy, yaw_rad):
    x = xx - cx
    y = yy - cy
    c = np.cos(-yaw_rad)
    s = np.sin(-yaw_rad)
    return c * x - s * y, s * x + c * y

# =========================
# SHAPE DEFINITIONS (LOCAL FRAME)
# =========================
# All shapes below are defined in a canonical local frame centered at (0,0),
# in normalized units (sensor radius = 1). You should adjust these numbers
# to match your CAD geometry.

def shape_mask_L(x_local, y_local):
    """
    L shape: union of vertical and horizontal bars in local coordinates.
    Values chosen to roughly match your simulator's L_mask_rotated, which was
    defined on a circular domain ~[-1,1] x [-1,1].

    vert bar: x in [-0.06, 0.10], y in [-0.35, 0.45]
    horiz bar: y in [-0.45, -0.30], x in [-0.06, 0.50]
    """
    # dimension in raw coordinates in meters
    # vert = (x_local > -0.004) & (x_local < 0.002) & (y_local > -0.0175) & (y_local < 0.0175)
    # horiz = (y_local > -0.0175) & (y_local < -0.0115) & (x_local > 0.002) & (x_local < 0.0175)

    vert = (x_local > -0.0889) & (x_local < 0.0444) & (y_local > -0.3889) & (y_local < 0.3889)
    horiz = (y_local > -0.3889) & (y_local < -0.2556) & (x_local > 0.0444) & (x_local < 0.3889)
    return vert | horiz


def shape_mask_T(x_local, y_local):
    """
    T shape: stem + bar, similar to T_mask_rotated in your simulator.
    """
    stem = (np.abs(x_local) < (0.003/0.045)) & (y_local > -(0.015/0.045)) & (y_local < (0.015/0.045))
    bar = (y_local > (0.015/0.045)) & (y_local < (0.021/0.045)) & (np.abs(x_local) < (0.013/0.045))
    return stem | bar


def shape_mask_edge(x_local, y_local):
    """
    Edge-like vertical bar.
    """
    x_min, x_max = -0.003 / 0.045, 0.005 / 0.045
    y_min, y_max = -0.015 / 0.045, 0.015 / 0.045
    return (x_min <= x_local) & (x_local <= x_max) & (y_min <= y_local) & (y_local <= y_max)


def shape_mask_ring(x_local, y_local):
    """
    Ring / annulus: center at origin, inner/outer radii in normalized units.
    Adjust r_inner, r_outer to match your physical ring geometry.
    """
    r2 = x_local**2 + y_local**2
    r_outer = 0.0306 / 0.045
    r_inner = 0.024 / 0.045
    return (r_inner**2 <= r2) & (r2 <= r_outer**2)


def shape_mask_double_circle(x_local, y_local):
    """
    Double round touch: two disks roughly opposite each other.
    Again, adjust radii and centers to match CAD.
    """
    # parameters in normalized coordinates
    r = 0.009 / 0.045
    cx1, cy1 = 0.0, 0.012 / 0.045
    cx2, cy2 =  0.0, -0.012 / 0.045

    r2_1 = (x_local - cx1)**2 + (y_local - cy1)**2
    r2_2 = (x_local - cx2)**2 + (y_local - cy2)**2
    return (r2_1 <= r**2) | (r2_2 <= r**2)


def shape_mask_plus(x_local, y_local):
    """
    '+' shape: union of horizontal and vertical bars.

    Adjust thickness and lengths as needed.
    """
    # vertical bar
    v = (np.abs(x_local) < (0.003 / 0.045)) & (np.abs(y_local) < (0.012 / 0.045))
    # horizontal bar
    h = (np.abs(x_local) < (0.0105 / 0.045)) & ((-0.0025 / 0.045) < y_local) & (y_local <= (0.0035 / 0.045))
    return v | h


# def shape_mask_C(x_local, y_local):
#     """
#     'C' shape: approximated as a ring minus a vertical bar on the right side.

#     Adjust radii and bar width as needed.
#     """
#     # base ring
#     r2 = x_local**2 + y_local**2
#     r_outer = 0.0153 / 0.045
#     r_inner = 0.011 / 0.045
#     ring = (r_inner**2 <= r2) & (r2 <= r_outer**2)

#     # "open" side: remove a vertical bar on +x side to create C
#     bar = (x_local > (0.00657 / 0.045)) & (x_local < (0.00916 / 0.045)) & (np.abs(y_local) < (0.0884 / 0.045))

#     return ring & (~bar)

def shape_mask_C(x_local, y_local):
    r2 = x_local**2 + y_local**2
    r_outer = 0.0153 / 0.045
    r_inner = 0.011 / 0.045
    ring = (r_inner**2 <= r2) & (r2 <= r_outer**2)

    # remove right side: choose cut near the opening location
    x_cut = 0.00916 / 0.045   # or tune this
    bar = (x_local > x_cut)   # <-- no upper bound

    return ring & (~bar)


def shape_mask_Z(x_local, y_local):
    """
    'Z' shape approximation: union of 3 bars (top, bottom, diagonal).

    This is a rough approximation; tweak for your CAD.
    """
    # Top horizontal bar
    top = (y_local > (0.015 / 0.045)) & (y_local < (0.021 / 0.045)) & (np.abs(x_local) < (0.014 / 0.045))
    # Bottom horizontal bar
    bottom = (y_local > (-0.021 / 0.045)) & (y_local < (-0.015 / 0.045)) & (np.abs(x_local) < (0.014 / 0.045))
    # Diagonal bar (approx Z middle)
    ## -------------------------
    # DIAGONAL BAR (from CAD)
    # -------------------------
    m = -1.26          # = tan(theta)
    half_w = 0.00484 / 0.045

    # Distance to diagonal centerline
    dist = np.abs(m * x_local - y_local) / np.sqrt(m**2 + 1)

    diag_band = dist < half_w

    # Limit diagonal length
    # diag_limit = (y_local > (-0.015 / 0.045)) & (y_local < (0.015 / 0.045))
    diag_limit = (y_local > (-0.021 / 0.045)) & (y_local < (0.021 / 0.045))

    middle = diag_band & diag_limit
    return top | bottom | middle


# Map shape_type string → local-frame mask function
SHAPE_MASK_FUNCS = {
    "L": shape_mask_L,
    "T": shape_mask_T,
    "edge": shape_mask_edge,
    "ring": shape_mask_ring,
    "double_circle": shape_mask_double_circle,
    "+": shape_mask_plus,
    "C": shape_mask_C,
    "Z": shape_mask_Z,
}



# =========================
# CORE MASK GENERATION
# =========================
def generate_mask_for_row(row, xx, yy, cfg):
    shape_type = row[cfg["COL_SHAPE_TYPE"]]
    u_m = row[cfg["COL_U_M"]]
    v_m = row[cfg["COL_V_M"]]
    yaw = row[cfg["COL_YAW"]]

    if shape_type not in SHAPE_MASK_FUNCS:
        raise ValueError(f"Unknown shape_type '{shape_type}'")

    cx = u_m / cfg["SENSOR_RADIUS_M"]
    cy = v_m / cfg["SENSOR_RADIUS_M"]

    domain = (xx**2 + yy**2) <= 1.0
    x_local, y_local = world_to_local(xx, yy, cx, cy, yaw)
    mask = SHAPE_MASK_FUNCS[shape_type](x_local, y_local)

    return (mask & domain).astype(np.uint8)


# =========================
# PROCESS ONE SPLIT
# =========================
def process_split(split_name, cfg):
    data_root = Path(cfg["DATA_ROOT"]) / split_name
    out_mask_dir = Path(cfg["OUT_MASK_ROOT"]) / split_name
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    xx, yy = make_normalized_grid(cfg["GRID_SIZE"])
    all_rows = []
    global_idx = 0

    for shape_dir in sorted(data_root.iterdir()):
        if not shape_dir.is_dir():
            continue

        shape_type = shape_dir.name
        print(f"[info] {split_name} | shape: {shape_type}")

        files = list(shape_dir.glob("*.csv")) + list(shape_dir.glob("*.parquet"))
        if not files:
            print(f"  [warn] no data files found in {shape_dir}")
            continue

        for fpath in files:
            if fpath.suffix == ".csv":
                df = pd.read_csv(fpath)
            else:
                df = pd.read_parquet(fpath)

            df[cfg["COL_SHAPE_TYPE"]] = shape_type
            df["split"] = split_name

            for _, row in df.iterrows():
                mask = generate_mask_for_row(row, xx, yy, cfg)
                img = Image.fromarray((mask * 255).astype(np.uint8))

                fname = f"{split_name}_{global_idx:06d}_{shape_type}.png"
                fpath_mask = out_mask_dir / fname
                img.save(fpath_mask)

                row_out = row.copy()
                row_out["mask_path"] = os.path.relpath(
                    fpath_mask, start=Path(cfg["DATA_ROOT"])
                )
                all_rows.append(row_out)

                global_idx += 1

    return pd.DataFrame(all_rows)


# =========================
# MAIN
# =========================
def main():
    cfg = CONFIG
    out_root = Path(cfg["OUT_MASK_ROOT"])
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n=== Processing TRAIN split ===")
    df_train = process_split("train", cfg)
    train_csv = Path(cfg["DATA_ROOT"]) / "real_train_with_masks.csv"
    df_train.to_csv(train_csv, index=False)
    print(f"[info] Saved {len(df_train)} train samples → {train_csv}")

    print("\n=== Processing TEST split ===")
    df_test = process_split("test", cfg)
    test_csv = Path(cfg["DATA_ROOT"]) / "real_test_with_masks.csv"
    df_test.to_csv(test_csv, index=False)
    print(f"[info] Saved {len(df_test)} test samples → {test_csv}")

    print("\nDone.")
    print(f"Masks saved under: {cfg['OUT_MASK_ROOT']}")


if __name__ == "__main__":
    main()