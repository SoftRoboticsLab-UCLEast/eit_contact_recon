#!/usr/bin/env python3
"""
Generate simulated EIT double-touch data using the *exact* touch locations
(from your real training CSV). The output stores DELTA voltages (v_touch - v0)
and the corresponding (x1,y1,x2,y2) labels.

Usage:
  python sim/103_sim_from_training_locations.py \
      --train-csv training_dataset/training_dataset.csv \
      --out sim/sim_from_training_locs.csv \
      --contact-radius 0.15
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd

# PyEIT
import pyeit.mesh as mesh
from pyeit.mesh import set_perm
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.eit.protocol import create as create_protocol
from pyeit.eit.fem import EITForward


def load_touch_coords(train_csv: Path):
    df = pd.read_csv(train_csv)
    # Prefer normalized columns; fallback to plain
    cols_norm = ["R1_x_norm", "R1_y_norm", "R2_x_norm", "R2_y_norm"]
    cols_plain = ["x1", "y1", "x2", "y2"]
    if all(c in df.columns for c in cols_norm):
        arr = df[cols_norm].to_numpy(dtype=np.float32)
    elif all(c in df.columns for c in cols_plain):
        arr = df[cols_plain].to_numpy(dtype=np.float32)
    else:
        raise RuntimeError(
            f"Could not find touch columns in {train_csv}. "
            f"Need x1_norm,y1_norm,x2_norm,y2_norm (or x1,y1,x2,y2)."
        )
    return arr  # shape (N, 4)


def main():
    ap = argparse.ArgumentParser(description="Simulate EIT deltas from real training touch locations.")
    ap.add_argument("--train-csv", required=True, help="Path to your training_dataset CSV (with x1/y1/x2/y2).")
    ap.add_argument("--out", default="sim/sim_from_training_locs.csv", help="Output CSV path.")
    # PyEIT geometry (keep simple; no alignment to real sensor for now)
    ap.add_argument("--n-el", type=int, default=16, help="Number of electrodes.")
    ap.add_argument("--dist-exc", type=int, default=1, help="Adjacent protocol current distance.")
    ap.add_argument("--step-meas", type=int, default=1, help="Adjacent protocol measurement step.")
    # Touch rendering
    ap.add_argument("--contact-radius", type=float, default=0.15, help="Normalized touch radius (unit disk).")
    ap.add_argument("--force-min", type=float, default=0.6, help="Min normalized force (for perm mapping).")
    ap.add_argument("--force-max", type=float, default=1.0, help="Max normalized force (for perm mapping).")
    ap.add_argument("--perm-a", type=float, default=1.0, help="Permittivity offset in perm = a + b*force.")
    ap.add_argument("--perm-b", type=float, default=9.0, help="Permittivity scale  in perm = a + b*force.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for force sampling.")
    args = ap.parse_args()

    train_csv = Path(args.train_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load real (normalized) touch locations
    Y = load_touch_coords(train_csv)  # columns: [x1,y1,x2,y2]
    N = Y.shape[0]
    print(f"[INFO] Loaded {N} double-touch rows from {train_csv}")

    # 2) Build simple PyEIT model (default disk; no rotation alignment)
    mesh_obj = mesh.create(n_el=args.n_el)
    protocol = create_protocol(args.n_el, dist_exc=args.dist_exc, step_meas=args.step_meas)
    fwd = EITForward(mesh_obj, protocol)

    # Baseline
    v0 = fwd.solve_eit(mesh_obj.perm).astype(np.float32)
    n_meas = v0.size
    print(f"[INFO] Baseline computed with {n_meas} measurements.")

    # 3) Simulate deltas for each real pair of touches
    rng = np.random.default_rng(args.seed)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = [f"eit_{i}" for i in range(n_meas)] + ["x1", "y1", "x2", "y2"]
        w.writerow(header)

        for i in range(N):
            x1, y1, x2, y2 = map(float, Y[i])

            # (Optional) safety clamp to unit disk margin
            r1 = (x1**2 + y1**2) ** 0.5
            r2 = (x2**2 + y2**2) ** 0.5
            if r1 > 1.0 or r2 > 1.0:
                # Skip extreme outliers silently; or clamp
                # Here we clamp slightly inside the unit circle to avoid FEM issues
                if r1 > 1.0:
                    x1, y1 = (x1 / r1) * 0.995, (y1 / r1) * 0.995
                if r2 > 1.0:
                    x2, y2 = (x2 / r2) * 0.995, (y2 / r2) * 0.995

            # Randomize "force" → perm (kept fairly strong by default)
            f1 = float(rng.uniform(args.force_min, args.force_max))
            f2 = float(rng.uniform(args.force_min, args.force_max))
            perm1 = args.perm_a + args.perm_b * f1
            perm2 = args.perm_a + args.perm_b * f2

            anomalies = [
                PyEITAnomaly_Circle(center=(x1, y1), r=args.contact_radius, perm=perm1),
                PyEITAnomaly_Circle(center=(x2, y2), r=args.contact_radius, perm=perm2),
            ]
            mesh_mod = set_perm(mesh_obj, anomaly=anomalies)

            v1 = fwd.solve_eit(mesh_mod.perm).astype(np.float32)
            delta = (v1 - v0).astype(np.float32)

            w.writerow(list(delta) + [x1, y1, x2, y2])

            if (i + 1) % 200 == 0:
                print(f"  simulated {i+1}/{N}")

    print(f"✅ Saved {N} samples → {out_path}")
    print("   (Columns are baseline-subtracted deltas + the exact real touch locations.)")


if __name__ == "__main__":
    main()
