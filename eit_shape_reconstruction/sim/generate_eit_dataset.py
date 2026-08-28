#!/usr/bin/env python3
"""
Synthetic EIT tactile dataset generator.

- Circular mesh with 16 electrodes.
- Random contact shapes: ring (annulus), T, L, edge, double circle.
- Randomized position and rotation (for non-radial shapes).
- 5 conductivity contrast levels.
- Outputs:
    - voltages.csv   (metadata + flattened voltage differences)
    - masks/train/*.png  (binary masks for training samples)
    - masks/test/*.png   (binary masks for test samples)

Each row in voltages.csv corresponds to one image mask file.
"""

import os
import csv
import numpy as np
from pathlib import Path

import matplotlib.tri as mtri
from PIL import Image

import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol
from pyeit.eit.fem import EITForward
import pyeit.eit.bp as bp


# =========================
# CONFIGURATION
# =========================
OUTPUT_DIR = "eit_dataset"
N_TRAIN = 8000
N_TEST = 2000
GRID_SIZE = 64           # mask image resolution (GRID_SIZE x GRID_SIZE)
N_EL = 16                # number of electrodes
MESH_H0 = 0.04           # mesh characteristic length
CONTRAST_LEVELS = [3.0, 5.0, 10.0, 15.0, 20.0]
NOISE_STD = 0.0          # standard deviation for Gaussian noise on v1
RANDOM_SEED = 42


# =========================
# GEOMETRY HELPERS
# =========================
def element_centroids(mesh_obj):
    """Return element centroids as (n_elems, 2) array."""
    pts = mesh_obj.node       # (n_nodes, 2)
    tri = mesh_obj.element    # (n_elems, 3) node indices
    centroids = pts[tri].mean(axis=1)
    return centroids


def make_perm_for_mask(mesh_obj, mask, contrast=5.0, base_perm=1.0):
    """Given a boolean mask over elements, return permittivity array."""
    n_elems = mesh_obj.element.shape[0]
    perm = np.ones(n_elems, dtype=float) * base_perm
    perm[mask] = base_perm * contrast
    return perm


def element_mask_to_nodal(mesh_obj, mask):
    """
    Convert a per-element boolean mask to a nodal mask (0/1),
    marking a node as 1 if it belongs to any 'True' element.
    """
    tri = mesh_obj.element
    n_nodes = mesh_obj.node.shape[0]
    nodal = np.zeros(n_nodes, dtype=float)
    for e_idx, nodes in enumerate(tri):
        if mask[e_idx]:
            nodal[nodes] = 1.0
    return nodal


def rotate_points(x, y, angle_rad):
    """
    Rotate (x, y) by angle_rad around origin.
    angle_rad > 0 => counter-clockwise rotation.
    """
    ca = np.cos(angle_rad)
    sa = np.sin(angle_rad)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    return xr, yr


# =========================
# SHAPE MASKS
# (in "local" coords)
# =========================
def O_mask(mesh_obj, center=(0.0, 0.0), r_outer=0.35, r_inner=0.18):
    """
    Ring shape: annulus between r_inner and r_outer.
    """
    c = element_centroids(mesh_obj)
    dx = c[:, 0] - center[0]
    dy = c[:, 1] - center[1]
    r2 = dx**2 + dy**2
    return (r_inner**2 <= r2) & (r2 <= r_outer**2)


def double_circle_mask(mesh_obj, center1=(-0.3, 0.0), r1=0.15,
                       center2=(0.3, 0.0), r2=0.15):
    """
    Two separate circular contacts (double round touch).
    """
    c = element_centroids(mesh_obj)
    dx1 = c[:, 0] - center1[0]
    dy1 = c[:, 1] - center1[1]
    dx2 = c[:, 0] - center2[0]
    dy2 = c[:, 1] - center2[1]
    m1 = dx1**2 + dy1**2 <= r1**2
    m2 = dx2**2 + dy2**2 <= r2**2
    return m1 | m2


def T_mask_rotated(mesh_obj, offset=(0.0, 0.0), angle=0.0):
    """
    T-shape with position and rotation:
        - offset: translation of the T center
        - angle: (rad) rotation around the origin.

    The canonical T is defined in a local axis-aligned frame.
    We rotate global coords into that frame before applying conditions.
    """
    c = element_centroids(mesh_obj)
    # translate to shape-centered frame
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    # rotate by -angle to align with canonical T
    x_local, y_local = rotate_points(x, y, -angle)

    # Canonical T (similar to your original implementation)
    stem = (np.abs(x_local) < 0.06) & (y_local > -0.5) & (y_local < 0.4)
    bar = (y_local > 0.4) & (y_local < 0.52) & (np.abs(x_local) < 0.45)
    return stem | bar


def L_mask_rotated(mesh_obj, offset=(0.0, 0.0), angle=0.0):
    """
    L-shape with position and rotation.
    """
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    x_local, y_local = rotate_points(x, y, -angle)

    vert = (x_local > -0.06) & (x_local < 0.10) & (y_local > -0.35) & (y_local < 0.45)
    horiz = (y_local > -0.45) & (y_local < -0.30) & (x_local > -0.06) & (x_local < 0.50)
    return vert | horiz


def edge_mask_rotated(mesh_obj, offset=(0.0, 0.0), angle=0.0,
                      x_min=-0.05, x_max=0.05,
                      y_min=-0.6, y_max=0.6):
    """
    Vertical edge-like bar with position and rotation.
    """
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    x_local, y_local = rotate_points(x, y, -angle)

    return (x_min <= x_local) & (x_local <= x_max) & \
           (y_min <= y_local) & (y_local <= y_max)


# =========================
# RANDOM SHAPE SAMPLER
# =========================
def random_shape_mask(mesh_obj, rng):
    """
    Return (shape_name, element_mask, contrast) for one random shape instance.
    Shapes:
        - ring
        - T
        - L
        - edge
        - double_circle
    """
    shape_type = rng.choice(["ring", "T", "L", "edge", "double_circle"])

    if shape_type == "ring":
        cx = rng.uniform(-0.1, 0.1)
        cy = rng.uniform(-0.1, 0.1)
        # Fixed size, just small variation in outer radius
        r_outer = rng.uniform(0.28, 0.35)
        r_inner = r_outer * 0.5
        mask = O_mask(mesh_obj, center=(cx, cy),
                      r_outer=r_outer, r_inner=r_inner)

    elif shape_type == "T":
        ox = rng.uniform(-0.2, 0.2)
        oy = rng.uniform(-0.2, 0.2)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        mask = T_mask_rotated(mesh_obj, offset=(ox, oy), angle=angle)

    elif shape_type == "L":
        ox = rng.uniform(-0.2, 0.2)
        oy = rng.uniform(-0.2, 0.2)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        mask = L_mask_rotated(mesh_obj, offset=(ox, oy), angle=angle)

    elif shape_type == "edge":
        # place near left or right side
        side = rng.choice(["left", "right"])
        x_off = -0.3 if side == "left" else 0.3
        y_off = rng.uniform(-0.1, 0.1)
        angle = rng.uniform(-0.5 * np.pi, 0.5 * np.pi)  # not too crazy tilt
        mask = edge_mask_rotated(mesh_obj, offset=(x_off, y_off), angle=angle)

    elif shape_type == "double_circle":
        # Two disks roughly opposite
        r = 0.15
        cx1 = rng.uniform(-0.45, -0.25)
        cy1 = rng.uniform(-0.15, 0.15)
        cx2 = rng.uniform(0.25, 0.45)
        cy2 = rng.uniform(-0.15, 0.15)
        mask = double_circle_mask(mesh_obj,
                                  center1=(cx1, cy1), r1=r,
                                  center2=(cx2, cy2), r2=r)
    else:
        raise ValueError("Unknown shape_type")

    # Random conductivity contrast
    contrast = float(rng.choice(CONTRAST_LEVELS))
    return shape_type, mask, contrast


# =========================
# EIT SETUP & RASTERIZATION
# =========================
def setup_eit(n_el=N_EL, h0=MESH_H0):
    """
    Create mesh, protocol, forward and BP solvers, plus reference voltages.
    """
    mesh_obj = mesh.create(n_el, h0=h0)

    protocol_obj = protocol.create(
        n_el, dist_exc=1, step_meas=1, parser_meas="std"
    )

    fwd = EITForward(mesh_obj, protocol_obj)

    n_elems = mesh_obj.element.shape[0]
    perm_ref = np.ones(n_elems, dtype=float)
    v0 = fwd.solve_eit(perm_ref)  # reference voltages

    eit_bp = bp.BP(mesh_obj, protocol_obj)
    eit_bp.setup(weight="none")

    return mesh_obj, protocol_obj, fwd, eit_bp, v0


def make_grid(mesh_obj, grid_size=GRID_SIZE):
    """
    Prepare triangulation and grid coordinates for rasterization.
    """
    pts = mesh_obj.node
    tri = mesh_obj.element
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)

    # Grid in [-1, 1] x [-1, 1]
    lin = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(lin, lin)
    return triang, xx, yy


def rasterize_to_grid(triang, xx, yy, nodal_vals, fill_value=0.0):
    """
    Interpolate nodal_vals defined on triangulation to regular grid.
    """
    interp = mtri.LinearTriInterpolator(triang, nodal_vals)
    grid = interp(xx, yy)
    grid = np.ma.filled(grid, fill_value=fill_value)
    return grid


def domain_mask_grid(xx, yy, radius=1.0):
    """Circular domain mask."""
    return (xx**2 + yy**2) <= radius**2


# =========================
# SINGLE SAMPLE SIMULATION
# =========================
def simulate_one_sample(mesh_obj, fwd, v0,
                        triang, xx, yy,
                        rng, noise_std=NOISE_STD):
    """
    Simulate one random shape sample:
      - volt_diff: v1 - v0  (measurement vector)
      - mask_grid: binary contact mask on grid (0/1)
      - shape_name: string label
      - contrast: conductivity contrast used
    """
    # 1. Draw a random shape and build element-level permittivity
    shape_name, elem_mask, contrast = random_shape_mask(mesh_obj, rng)
    perm_shape = make_perm_for_mask(mesh_obj, elem_mask, contrast=contrast)

    # 2. Forward solve: new voltages
    v1 = fwd.solve_eit(perm_shape)

    # Optional measurement noise
    if noise_std > 0.0:
        v1 = v1 + rng.normal(scale=noise_std, size=v1.shape)

    # 3. Build "input" measurement feature: difference vs baseline
    volt_diff = v1 - v0

    # 4. Create nodal version of mask and rasterize
    nodal_mask = element_mask_to_nodal(mesh_obj, elem_mask)
    mask_grid_float = rasterize_to_grid(triang, xx, yy, nodal_mask,
                                        fill_value=0.0)
    mask_grid = (mask_grid_float > 0.5).astype(np.uint8)

    return {
        "volt_diff": volt_diff.astype(np.float32),
        "mask_grid": mask_grid,
        "shape_name": shape_name,
        "contrast": contrast,
    }


# =========================
# MAIN DATA GENERATION
# =========================
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # Create output directories
    out_dir = Path(OUTPUT_DIR)
    masks_train_dir = out_dir / "masks" / "train"
    masks_test_dir = out_dir / "masks" / "test"
    masks_train_dir.mkdir(parents=True, exist_ok=True)
    masks_test_dir.mkdir(parents=True, exist_ok=True)

    # Setup EIT and grid
    mesh_obj, protocol_obj, fwd, eit_bp, v0 = setup_eit()
    triang, xx, yy = make_grid(mesh_obj, grid_size=GRID_SIZE)

    # CSV file for voltages
    csv_path = out_dir / "voltages.csv"

    # We don't know n_meas before solving; get it from a dummy solve
    n_meas = v0.size

    with csv_path.open("w", newline="") as f_csv:
        writer = csv.writer(f_csv)

        # Header: metadata + voltages
        header = ["sample_id", "split", "shape_type", "contrast", "mask_path"]
        header += [f"v_{i}" for i in range(n_meas)]
        writer.writerow(header)

        total_samples = N_TRAIN + N_TEST
        print(f"Generating dataset: {N_TRAIN} train, {N_TEST} test (total {total_samples})")

        for idx in range(total_samples):
            split = "train" if idx < N_TRAIN else "test"
            split_idx = idx if split == "train" else (idx - N_TRAIN)

            sample_id = f"{split}_{split_idx:06d}"
            # Simulate one sample
            sample = simulate_one_sample(
                mesh_obj, fwd, v0,
                triang, xx, yy,
                rng, noise_std=NOISE_STD
            )

            volt = sample["volt_diff"]
            mask_grid = sample["mask_grid"]
            shape_type = sample["shape_name"]
            contrast = sample["contrast"]

            # Save mask image
            if split == "train":
                mask_dir = masks_train_dir
            else:
                mask_dir = masks_test_dir

            mask_filename = f"{sample_id}.png"
            mask_path = mask_dir / mask_filename

            # Convert mask to 0/255 uint8 image
            img = Image.fromarray(mask_grid * 255)
            img.save(mask_path)

            # Write CSV row
            row = [
                sample_id,
                split,
                shape_type,
                contrast,
                str(mask_path.relative_to(out_dir)),  # relative path
            ]
            row += [f"{v:.8e}" for v in volt]
            writer.writerow(row)

            if (idx + 1) % 500 == 0 or (idx + 1) == total_samples:
                print(f"  Generated {idx + 1}/{total_samples} samples")

    print("Done.")
    print(f"- Voltages CSV: {csv_path}")
    print(f"- Train masks: {masks_train_dir}")
    print(f"- Test masks:  {masks_test_dir}")


if __name__ == "__main__":
    main()
