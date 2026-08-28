"""
EIT Sim Data Generator (single & multi-touch) for PyEIT
-------------------------------------------------------
- Produces separate CSVs per touch-count: 1, 2, 3, 5 (configurable).
- Each row contains delta voltages (v_meas - v_baseline), followed by metadata:
  x_i, y_i, force_i for each contact i, plus: contact_count, anomaly_diam_mm (or normalized radius).
- Supports three probe sizes (default: 7, 10, 15 mm). If --sensor-diam-mm is provided,
  these are converted to normalized radii; else uses normalized radii [0.07, 0.10, 0.15].
- Enforces non-overlap between contacts.
- Uses PyEIT FEM forward model with adjacent protocol by default.

NOTE: This script expects PyEIT to be installed. It only *generates* data; it does not visualize by default.
"""

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# PyEIT imports
import pyeit.mesh as mesh
from pyeit.mesh import set_perm
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.eit.protocol import create as create_protocol
from pyeit.eit.fem import EITForward

# ----------------------------
# Helpers
# ----------------------------

@dataclass
class Config:
    out_dir: str
    n_el: int
    dist_exc: int
    step_meas: int
    seed: int
    samples_per_combo: int
    contact_counts: List[int]
    probe_diams_mm: List[float]
    sensor_diam_mm: Optional[float]
    normalized_radii_if_unknown_sensor: List[float]
    force_min: float
    force_max: float
    force_dist: str
    perm_a: float
    perm_b: float
    min_edge_clearance: float
    min_contact_separation_scale: float
    max_retries: int


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def diam_mm_to_norm_radius(diam_mm: float, sensor_diam_mm: Optional[float], fallback_norm_r: float) -> Tuple[float, float]:
    """
    Returns (normalized_radius, effective_diam_mm_or_nan).
    If sensor_diam_mm is None, we can't convert; use fallback normalized radius and output NaN for mm.
    """
    if sensor_diam_mm is None or sensor_diam_mm <= 0:
        return float(fallback_norm_r), float("nan")
    norm_r = (diam_mm / sensor_diam_mm)  # diameter-to-diameter; unit-circle radius=1 means sensor radius=1, so r_norm = D_probe/D_sensor
    return float(norm_r), float(diam_mm)


def sample_forces(k: int, cfg: Config) -> np.ndarray:
    if cfg.force_dist == "uniform":
        return np.random.uniform(cfg.force_min, cfg.force_max, size=k).astype(np.float32)
    elif cfg.force_dist == "normal":
        mu = 0.5 * (cfg.force_min + cfg.force_max)
        sigma = 0.25 * (cfg.force_max - cfg.force_min)
        vals = np.random.normal(mu, sigma, size=k)
        return np.clip(vals, cfg.force_min, cfg.force_max).astype(np.float32)
    else:
        raise ValueError(f"Unknown force_dist: {cfg.force_dist}")


def force_to_perm(force_vals: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Map force in [0,1] to relative permittivity scaling: perm = a + b * force
    Defaults reproduce user's earlier mapping with a=1.0, b=9.0.
    """
    return (a + b * force_vals).astype(np.float32)


def valid_non_overlapping_positions(k: int, radii: List[float], cfg: Config) -> Optional[List[Tuple[float, float]]]:
    """
    Rejection sample k positions inside unit circle (radius=1) such that:
    - Each circle of radius r_i is fully inside the unit disk with margin cfg.min_edge_clearance
    - Minimum center-to-center separation is s = cfg.min_contact_separation_scale * (r_i + r_j)
    """
    max_trials = cfg.max_retries
    r_list = list(radii)
    positions: List[Tuple[float, float]] = []

    # Precompute allowable radial bound to stay inside (1 - r - clearance)
    bounds = [1.0 - r - cfg.min_edge_clearance for r in r_list]

    for _ in range(max_trials):
        positions.clear()
        ok = True
        for i in range(k):
            ri = r_list[i]
            bound = bounds[i]
            placed = False
            # Try several attempts for this contact
            for _attempt in range(200):
                # Sample radius ~ sqrt(u) * bound to be uniform in area
                rr = math.sqrt(np.random.rand()) * max(bound, 1e-6)
                theta = 2 * math.pi * np.random.rand()
                xi = rr * math.cos(theta)
                yi = rr * math.sin(theta)

                # Check separation
                sep_ok = True
                for j in range(len(positions)):
                    rj = r_list[j]
                    xj, yj = positions[j]
                    dist = math.hypot(xi - xj, yi - yj)
                    need = cfg.min_contact_separation_scale * (ri + rj)
                    if dist < need:
                        sep_ok = False
                        break

                if sep_ok:
                    positions.append((xi, yi))
                    placed = True
                    break

            if not placed:
                ok = False
                break

        if ok and len(positions) == k:
            return positions

    return None


def build_header(n_meas: int, k: int) -> List[str]:
    header = [f"v{i}" for i in range(n_meas)]
    for j in range(k):
        header += [f"x{j+1}", f"y{j+1}", f"force{j+1}"]
    header += ["contact_count", "anomaly_diam_mm", "anomaly_r_norm"]
    return header


def main():
    p = argparse.ArgumentParser(description="Generate EIT single & multi-touch datasets (PyEIT).")
    p.add_argument("--out-dir", type=str, default="eit_sim_data", help="Output directory.")
    p.add_argument("--n-el", type=int, default=16, help="Number of electrodes.")
    p.add_argument("--dist-exc", type=int, default=1, help="Protocol: dist_exc for current injection.")
    p.add_argument("--step-meas", type=int, default=1, help="Protocol: step_meas for measurements.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--samples-per-combo", type=int, default=2000, help="Samples per (touch-count x probe-size) combo.")
    p.add_argument("--contact-counts", type=int, nargs="+", default=[1, 2, 3, 5], help="Touch counts to simulate.")
    p.add_argument("--probe-diams-mm", type=float, nargs="+", default=[7.0, 10.0, 15.0], help="Probe diameters in mm.")
    p.add_argument("--sensor-diam-mm", type=float, default=None, help="Sensor diameter in mm for scaling. If omitted, uses normalized radii fallback.")
    p.add_argument("--fallback-norm-radii", type=float, nargs="+", default=[0.07, 0.10, 0.15], help="Used when --sensor-diam-mm is not provided.")
    p.add_argument("--force-min", type=float, default=0.0, help="Min force (normalized).")
    p.add_argument("--force-max", type=float, default=1.0, help="Max force (normalized).")
    p.add_argument("--force-dist", type=str, default="normal", choices=["uniform", "normal"], help="Sampling distribution for forces.")
    p.add_argument("--perm-a", type=float, default=1.0, help="Permittivity map: perm = a + b*force (offset).")
    p.add_argument("--perm-b", type=float, default=9.0, help="Permittivity map: perm = a + b*force (scale).")
    p.add_argument("--edge-clearance", type=float, default=0.01, help="Min clearance to boundary of unit disk.")
    p.add_argument("--sep-scale", type=float, default=1.2, help="Min separation scale times (r_i + r_j).")
    p.add_argument("--max-retries", type=int, default=200, help="Max placement retries for a sample.")
    args = p.parse_args()

    cfg = Config(
        out_dir=args.out_dir,
        n_el=args.n_el,
        dist_exc=args.dist_exc,
        step_meas=args.step_meas,
        seed=args.seed,
        samples_per_combo=args.samples_per_combo,
        contact_counts=args.contact_counts,
        probe_diams_mm=args.probe_diams_mm,
        sensor_diam_mm=args.sensor_diam_mm,
        normalized_radii_if_unknown_sensor=args.fallback_norm_radii,
        force_min=args.force_min,
        force_max=args.force_max,
        force_dist=args.force_dist,
        perm_a=args.perm_a,
        perm_b=args.perm_b,
        min_edge_clearance=args.edge_clearance,
        min_contact_separation_scale=args.sep_scale,
        max_retries=args.max_retries,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    # --- Mesh & protocol ---
    mesh_obj = mesh.create(n_el=cfg.n_el)  # unit disk by default
    protocol = create_protocol(n_el=cfg.n_el, dist_exc=cfg.dist_exc, step_meas=cfg.step_meas)
    fwd = EITForward(mesh_obj, protocol)

    # Baseline
    v_baseline = fwd.solve_eit(mesh_obj.perm)
    n_meas = len(v_baseline)

    # Prepare outputs: one CSV per contact_count
    writers = {}
    files = {}
    for k in cfg.contact_counts:
        path = os.path.join(cfg.out_dir, f"sim_k{k}.csv")
        f = open(path, "w", newline="")
        files[k] = f
        w = csv.writer(f)
        w.writerow(build_header(n_meas, k))
        writers[k] = w

    try:
        # Loop over probe sizes
        for idx, diam_mm in enumerate(cfg.probe_diams_mm):
            # Determine radius in normalized coordinates
            fallback_r = cfg.normalized_radii_if_unknown_sensor[min(idx, len(cfg.normalized_radii_if_unknown_sensor)-1)]
            r_norm, diam_tag = diam_mm_to_norm_radius(diam_mm, cfg.sensor_diam_mm, fallback_r)

            # For each contact-count (k)
            for k in cfg.contact_counts:
                rows_written = 0
                w = writers[k]

                while rows_written < cfg.samples_per_combo:
                    # radii list for k contacts (all same probe size here)
                    radii = [r_norm for _ in range(k)]

                    # Sample non-overlapping positions
                    positions = valid_non_overlapping_positions(k, radii, cfg)
                    if positions is None:
                        # couldn't place; try again
                        continue

                    # Forces and permittivities
                    forces = sample_forces(k, cfg)
                    perms = force_to_perm(forces, cfg.perm_a, cfg.perm_b)

                    # Build anomaly list
                    anomaly = []
                    for (x, y), r, perm in zip(positions, radii, perms):
                        anomaly.append(PyEITAnomaly_Circle(center=[x, y], r=r, perm=perm))

                    # Simulate
                    mesh_mod = set_perm(mesh_obj, anomaly=anomaly)
                    v_touch = fwd.solve_eit(mesh_mod.perm)
                    delta_v = (v_touch - v_baseline).astype(np.float32)

                    # Assemble row
                    row = list(delta_v)
                    for (x, y), f in zip(positions, forces):
                        row += [float(x), float(y), float(f)]
                    row += [int(k), float(diam_tag), float(r_norm)]
                    w.writerow(row)
                    rows_written += 1

                print(f"Saved {rows_written} rows for k={k}, probe_diam={diam_mm}mm (norm_r={r_norm:.3f})")

    finally:
        for f in files.values():
            f.close()

    print("✅ Done. Files written to:", cfg.out_dir)


if __name__ == "__main__":
    main()
