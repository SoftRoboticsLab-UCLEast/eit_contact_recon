"""
Paired Single–Double Touch Dataset Generator (PyEIT)
====================================================
Goal
----
For each randomly sampled *double-touch* configuration (positions, forces, probe size),
also generate the two corresponding *single-touch* samples at the exact same contacts.
This enables testing (approximate) linear superposition on measured voltages.

Outputs
-------
- CSV file with one row per paired sample, including:
  * v_double_* : absolute voltage vector for double-touch
  * v_s1_*     : absolute voltage vector for single-touch contact #1
  * v_s2_*     : absolute voltage vector for single-touch contact #2
  * dv_double_*, dv_s1_*, dv_s2_* : delta voltages vs baseline (optional, enabled by default)
  * metadata   : x1,y1,force1, x2,y2,force2, probe_diam_mm, r_norm, seed
- Baseline vector saved to a separate .npy for reference.

Usage
-----
python eit_paired_single_double_gen.py \
  --out-dir eit_paired_data \
  --n-samples 4000 \
  --probe-diams-mm 7 10 15 \
  --sensor-diam-mm 60 \
  --n-el 16 --dist-exc 1 --step-meas 1 \
  --force-dist normal --force-min 0.0 --force-max 1.0

Notes
-----
- Positions are sampled uniformly-in-area within the unit disk with edge clearance.
- Contacts are non-overlapping with a configurable separation.
- Forces for the two contacts are sampled independently from the specified distribution.
"""

import argparse
import csv
import math
import os
import random
from typing import List, Tuple, Optional

import numpy as np

# PyEIT
import pyeit.mesh as mesh
from pyeit.mesh import set_perm
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.eit.protocol import create as create_protocol
from pyeit.eit.fem import EITForward


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def diam_mm_to_norm_radius(diam_mm: float, sensor_diam_mm: Optional[float], fallback_norm_r: float) -> Tuple[float, float]:
    if sensor_diam_mm is None or sensor_diam_mm <= 0:
        return float(fallback_norm_r), float("nan")
    # diameter scaled by sensor diameter; unit disk radius=1 ↔ sensor radius=1
    norm_r = (diam_mm / sensor_diam_mm)
    return float(norm_r), float(diam_mm)


def sample_forces(k: int, force_min: float, force_max: float, dist: str) -> np.ndarray:
    if dist == "uniform":
        return np.random.uniform(force_min, force_max, size=k).astype(np.float32)
    elif dist == "normal":
        mu = 0.5 * (force_min + force_max)
        sigma = 0.25 * (force_max - force_min)
        vals = np.random.normal(mu, sigma, size=k)
        return np.clip(vals, force_min, force_max).astype(np.float32)
    else:
        raise ValueError(f"Unknown force_dist: {dist}")


def force_to_perm(force_vals: np.ndarray, a: float, b: float) -> np.ndarray:
    return (a + b * force_vals).astype(np.float32)


def valid_non_overlapping_positions(k: int, radii: List[float], edge_clearance: float, sep_scale: float, max_retries: int) -> Optional[List[Tuple[float, float]]]:
    r_list = list(radii)
    bounds = [1.0 - r - edge_clearance for r in r_list]

    for _ in range(max_retries):
        positions = []
        ok = True
        for i in range(k):
            ri = r_list[i]
            bound = max(bounds[i], 1e-6)
            placed = False
            for _attempt in range(300):
                rr = (np.random.rand() ** 0.5) * bound   # uniform in area
                theta = 2 * np.pi * np.random.rand()
                xi = rr * np.cos(theta)
                yi = rr * np.sin(theta)

                # separation
                sep_ok = True
                for (xj, yj), rj in zip(positions, r_list[:len(positions)]):
                    dist = math.hypot(xi - xj, yi - yj)
                    if dist < sep_scale * (ri + rj):
                        sep_ok = False
                        break
                if sep_ok:
                    positions.append((xi, yi))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return positions
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="eit_paired_data")
    ap.add_argument("--filename", type=str, default="paired_single_double.csv")
    ap.add_argument("--n-samples", type=int, default=4000, help="Number of paired samples to generate (per probe size).")
    ap.add_argument("--probe-diams-mm", type=float, nargs="+", default=[7.0, 10.0, 15.0])
    ap.add_argument("--sensor-diam-mm", type=float, default=None, help="Sensor diameter in mm; if omitted, uses fallbacks.")
    ap.add_argument("--fallback-norm-radii", type=float, nargs="+", default=[0.07, 0.10, 0.15])
    ap.add_argument("--n-el", type=int, default=16)
    ap.add_argument("--dist-exc", type=int, default=1)
    ap.add_argument("--step-meas", type=int, default=1)
    ap.add_argument("--force-min", type=float, default=0.0)
    ap.add_argument("--force-max", type=float, default=1.0)
    ap.add_argument("--force-dist", type=str, default="normal", choices=["uniform", "normal"])
    ap.add_argument("--perm-a", type=float, default=1.0)
    ap.add_argument("--perm-b", type=float, default=9.0)
    ap.add_argument("--edge-clearance", type=float, default=0.01)
    ap.add_argument("--sep-scale", type=float, default=1.2)
    ap.add_argument("--max-retries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-delta", action="store_true", help="Also save delta voltages (dv=V-V0).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Mesh & protocol
    mesh_obj = mesh.create(n_el=args.n_el)
    protocol = create_protocol(n_el=args.n_el, dist_exc=args.dist_exc, step_meas=args.step_meas)
    fwd = EITForward(mesh_obj, protocol)
    v0 = fwd.solve_eit(mesh_obj.perm).astype(np.float32)
    mlen = len(v0)

    base_path = os.path.join(args.out_dir, args.filename)
    with open(base_path, "w", newline="") as f:
        w = csv.writer(f)

        # Header
        cols = []
        cols += [f"v_double_{i}" for i in range(mlen)]
        cols += [f"v_s1_{i}" for i in range(mlen)]
        cols += [f"v_s2_{i}" for i in range(mlen)]
        if args.save_delta:
            cols += [f"dv_double_{i}" for i in range(mlen)]
            cols += [f"dv_s1_{i}" for i in range(mlen)]
            cols += [f"dv_s2_{i}" for i in range(mlen)]
        cols += ["x1","y1","force1","x2","y2","force2","probe_diam_mm","r_norm","seed"]
        w.writerow(cols)

        # Loop over probe sizes
        for idx, diam_mm in enumerate(args.probe_diams_mm):
            fallback_r = args.fallback_norm_radii[min(idx, len(args.fallback_norm_radii)-1)]
            r_norm, diam_tag = diam_mm_to_norm_radius(diam_mm, args.sensor_diam_mm, fallback_r)

            ns = args.n_samples
            for _ in range(ns):
                # positions for double touch
                positions = valid_non_overlapping_positions(
                    k=2, radii=[r_norm, r_norm],
                    edge_clearance=args.edge_clearance, sep_scale=args.sep_scale,
                    max_retries=args.max_retries
                )
                if positions is None:
                    continue

                # independent forces
                forces = sample_forces(2, args.force_min, args.force_max, args.force_dist)
                perms = force_to_perm(forces, args.perm_a, args.perm_b)

                # Build anomalies
                anom_double = [
                    PyEITAnomaly_Circle(center=list(positions[0]), r=r_norm, perm=float(perms[0])),
                    PyEITAnomaly_Circle(center=list(positions[1]), r=r_norm, perm=float(perms[1])),
                ]
                anom_s1 = [PyEITAnomaly_Circle(center=list(positions[0]), r=r_norm, perm=float(perms[0]))]
                anom_s2 = [PyEITAnomaly_Circle(center=list(positions[1]), r=r_norm, perm=float(perms[1]))]

                # Simulate
                v_double = fwd.solve_eit(set_perm(mesh_obj, anomaly=anom_double).perm).astype(np.float32)
                v_s1     = fwd.solve_eit(set_perm(mesh_obj, anomaly=anom_s1).perm).astype(np.float32)
                v_s2     = fwd.solve_eit(set_perm(mesh_obj, anomaly=anom_s2).perm).astype(np.float32)

                row = list(v_double) + list(v_s1) + list(v_s2)

                if args.save_delta:
                    dv_double = (v_double - v0).astype(np.float32)
                    dv_s1 = (v_s1 - v0).astype(np.float32)
                    dv_s2 = (v_s2 - v0).astype(np.float32)
                    row += list(dv_double) + list(dv_s1) + list(dv_s2)

                (x1, y1), (x2, y2) = positions
                row += [float(x1), float(y1), float(forces[0]), float(x2), float(y2), float(forces[1]), float(diam_tag), float(r_norm), int(args.seed)]
                w.writerow(row)

            print(f"Saved {ns} paired rows for probe {diam_mm} mm (norm_r={r_norm:.3f})")

    # Save the baseline too (handy for later analysis)
    npy_path = os.path.join(args.out_dir, "baseline_v0.npy")
    np.save(npy_path, v0)
    print("✅ Done.")
    print(f"CSV: {base_path}")
    print(f"Baseline: {npy_path}")


if __name__ == "__main__":
    main()
