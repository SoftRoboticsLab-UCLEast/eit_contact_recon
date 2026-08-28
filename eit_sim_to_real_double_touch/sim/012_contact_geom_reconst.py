#!/usr/bin/env python3
"""
EIT shape reconstruction demo with PyEIT (BP version)

Shapes: circle, edge, T, L, O
- Circular mesh with 16 electrodes
- Shape-specific conductivity anomalies
- BP reconstruction (same style as your real-time script)
- IoU between ground-truth anomaly and reconstruction
- Each subplot: reconstruction + ground-truth mask contour
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol
import pyeit.eit.bp as bp
from pyeit.eit.fem import EITForward

from pyeit.eit.jac import JAC
from pyeit.eit.greit import GREIT

from functools import partial


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def element_centroids(mesh_obj):
    """Return element centroids as (n_elems, 2) array."""
    pts = mesh_obj.node       # (n_nodes, 2)
    tri = mesh_obj.element    # (n_elems, 3) node indices
    centroids = pts[tri].mean(axis=1)
    return centroids


# ----------------------- shape masks ----------------------------------
def circle_mask(mesh_obj, center=(0.0, 0.0), radius=0.25):
    c = element_centroids(mesh_obj)
    dx = c[:, 0] - center[0]
    dy = c[:, 1] - center[1]
    return dx**2 + dy**2 <= radius**2

def edge_mask(mesh_obj, offset=(0.0, 0.0),
              x_min=-0.05, x_max=0.05,
              y_min=-0.6, y_max=0.6):
    """
    Vertical edge; offset moves it around the disk.
    """
    c = element_centroids(mesh_obj)
    x_local = c[:, 0] - offset[0]
    y_local = c[:, 1] - offset[1]
    return (x_min <= x_local) & (x_local <= x_max) & (y_min <= y_local) & (y_local <= y_max)


def T_mask(mesh_obj, offset=(0.0, 0.0)):
    """
    T-shape: vertical stem + horizontal bar.
    offset shifts the whole T around.
    """
    c = element_centroids(mesh_obj)
    x_local = c[:, 0] - offset[0]
    y_local = c[:, 1] - offset[1]

    stem = (np.abs(x_local) < 0.06) & (y_local > -0.5) & (y_local < 0.4)
    bar  = (y_local > 0.4) & (y_local < 0.52) & (np.abs(x_local) < 0.45)

    return stem | bar

def L_mask(mesh_obj, offset=(0.0, 0.0)):
    """
    L-shape; offset shifts the whole L.
    """
    c = element_centroids(mesh_obj)
    x_local = c[:, 0] - offset[0]
    y_local = c[:, 1] - offset[1]

    vert  = (x_local > -0.06) & (x_local < 0.10) & (y_local > -0.35) & (y_local < 0.45)
    horiz = (y_local > -0.45) & (y_local < -0.30) & (x_local > -0.06) & (x_local < 0.50)

    return vert | horiz


def O_mask(mesh_obj, center=(0.0, 0.0),
           r_outer=0.35, r_inner=0.18):
    """
    O-shape: annulus between r_inner and r_outer.
    """
    c = element_centroids(mesh_obj)
    dx = c[:, 0] - center[0]
    dy = c[:, 1] - center[1]
    r2 = dx**2 + dy**2
    return (r_inner**2 <= r2) & (r2 <= r_outer**2)

# ----------------------------------------------------------------------
# EIT helpers
# ----------------------------------------------------------------------
def make_perm_for_mask(mesh_obj, mask, contrast=5.0, base_perm=1.0):
    """
    Given a boolean mask over elements, return a permittivity array
    (one value per element).
    """
    n_elems = mesh_obj.element.shape[0]
    perm = np.ones(n_elems, dtype=float) * base_perm
    perm[mask] = base_perm * contrast
    return perm


def nodal_to_element_values(mesh_obj, nodal_vals):
    """
    Convert nodal values (len = n_nodes) to element values (len = n_elems)
    by averaging the 3 node values of each triangle.
    """
    tri = mesh_obj.element  # (n_elems, 3)
    return nodal_vals[tri].mean(axis=1)


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


def compute_iou(mask_true, recon_elem_values, top_frac=0.3):
    """
    Simple IoU between:
      - mask_true: boolean mask for the anomaly (per element)
      - recon_elem_values: reconstruction per element
    recon region = top_frac of elements by value (on positive part).
    """
    assert mask_true.shape == recon_elem_values.shape

    vals = np.clip(recon_elem_values, a_min=0.0, a_max=None)

    n_elems = vals.size
    k = max(1, int(top_frac * n_elems))

    idx_sorted = np.argsort(vals)
    recon_mask = np.zeros_like(mask_true, dtype=bool)
    recon_mask[idx_sorted[-k:]] = True

    inter = np.logical_and(mask_true, recon_mask).sum()
    union = np.logical_or(mask_true, recon_mask).sum()
    if union == 0:
        return 0.0
    return inter / union


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------
def run():
    np.random.seed(0)

    # --------------------------------------------------------------
    # 1. Build mesh and protocol (same API as your BP script)
    # --------------------------------------------------------------
    n_el = 128#16
    mesh_obj = mesh.create(n_el, h0=0.04)

    protocol_obj = protocol.create(
        n_el, dist_exc=1, step_meas=1, parser_meas="std"
    )

    # Forward solver
    fwd = EITForward(mesh_obj, protocol_obj)

    # Reference homogeneous permittivity (one value per element)
    n_elems = mesh_obj.element.shape[0]
    perm_ref = np.ones(n_elems, dtype=float)
    v0 = fwd.solve_eit(perm_ref)  # (n_meas,)

    # BP inverse solver (same as your real-time code)
    eit_bp = bp.BP(mesh_obj, protocol_obj)
    eit_bp.setup(weight="none")


    # --------------------------------------------------------------
    # 2. Define shapes
    # --------------------------------------------------------------
    # shape_funcs = {
    #     "Circle": circle_mask,
    #     "Edge": edge_mask,
    #     "T": T_mask,
    #     "L": L_mask,
    # }
    shape_funcs = {
    # -----------------------
    # CIRCLE
    # -----------------------
    # "Circle_center": lambda m: circle_mask(m, center=(0.0, 0.0)),
    # "Circle_top":    lambda m: circle_mask(m, center=(0.0,  0.30)),
    # "Circle_bottom": lambda m: circle_mask(m, center=(0.0, -0.40)),
    # "Circle_left":   lambda m: circle_mask(m, center=(-0.40, 0.0)),
    "Circle_right":  lambda m: circle_mask(m, center=( 0.30, 0.0)),

    # -----------------------
    # T SHAPE
    # -----------------------
    # "T_center": lambda m: T_mask(m, offset=(0.0, 0.0)),
    # "T_top":    lambda m: T_mask(m, offset=(0.0,  0.30)),
    # "T_bottom": lambda m: T_mask(m, offset=(0.0, -0.30)),
    # "T_left":   lambda m: T_mask(m, offset=(-0.30, 0.0)),
    "T_right":  lambda m: T_mask(m, offset=( 0.30, 0.0)),

    # -----------------------
    # L SHAPE
    # -----------------------
    # "L_center": lambda m: L_mask(m, offset=(0.0, 0.0)),
    # "L_top":    lambda m: L_mask(m, offset=(0.0,  0.30)),
    # "L_bottom": lambda m: L_mask(m, offset=(0.0, -0.30)),
    "L_left":   lambda m: L_mask(m, offset=(-0.30, 0.0)),
    # "L_right":  lambda m: L_mask(m, offset=( 0.30, 0.0)),

    # -----------------------
    # EDGE SHAPE (vertical bar)
    # -----------------------
    # "Edge_center": lambda m: edge_mask(m, offset=(0.0, 0.0)),
    # "Edge_top":    lambda m: edge_mask(m, offset=(0.0,  0.30)),
    # "Edge_bottom": lambda m: edge_mask(m, offset=(0.0, -0.30)),
    "Edge_left":   lambda m: edge_mask(m, offset=(-0.30, 0.0)),
    # "Edge_right":  lambda m: edge_mask(m, offset=( 0.30, 0.0)),
    }

    all_ds = []
    results = {}  # name -> dict with nodal_bp, elem_ds, mask, iou

    # --------------------------------------------------------------
    # 3. Simulate each shape and reconstruct with BP
    # --------------------------------------------------------------
    for name, mask_func in shape_funcs.items():
        mask = mask_func(mesh_obj)

        perm_shape = make_perm_for_mask(mesh_obj, mask, contrast=20.0)
        v1 = fwd.solve_eit(perm_shape)

        # mimic your real-time logic: compute difference, then BP
        subtraction = v0 - v1
        sum_abs_diff = np.sum(np.abs(subtraction))

               # ---------- BP ----------
        if sum_abs_diff > 0.3:
            nodal_bp = 192.0 * eit_bp.solve(
                v1, v0, normalize=True, log_scale=False
            )
        else:
            nodal_bp = 192.0 * eit_bp.solve(
                v0, v0, normalize=True, log_scale=False
            )
        nodal_bp = np.real(nodal_bp)
        elem_bp = nodal_to_element_values(mesh_obj, nodal_bp)
        iou_bp = compute_iou(mask, elem_bp, top_frac=0.3)  # if you switched IoU to alpha-based



        print(
            f"{name}: IoU_BP={iou_bp:.3f}, "
            f"sum|v0-v1|={sum_abs_diff:.3f}"
        )

        results[name] = {
            "mask": mask,
            "nodal_bp": nodal_bp,
            "elem_bp": elem_bp,
            "iou_bp": iou_bp,
        }
        all_ds.append(nodal_bp)  # still use BP for global vmin/vmax if you like

    all_ds = np.concatenate(all_ds)
    vmin = np.percentile(all_ds, 5)
    vmax = np.percentile(all_ds, 95)

    # --------------------------------------------------------------
    # 4. Plot reconstructions + ground-truth contour (same subplot)
    # --------------------------------------------------------------
    n_shapes = len(shape_funcs)
    # n_cols = 3
    # n_rows = int(np.ceil(n_shapes / n_cols))
    n_cols = 2
    n_rows = 2

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows)
    )
    axes = np.atleast_1d(axes).flatten()

    pts = mesh_obj.node
    tri = mesh_obj.element
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)

    for ax, (name, data) in zip(axes, results.items()):
        nodal_bp = data["nodal_bp"]
        mask = data["mask"]
        iou_val = data["iou_bp"]

        # Reconstruction
        tpc = ax.tripcolor(
            triang, nodal_bp, shading="flat",
            vmin=vmin, vmax=vmax, cmap="viridis"
        )

        # Ground-truth mask as white contour
        nodal_mask = element_mask_to_nodal(mesh_obj, mask)
        ax.tricontour(
            triang, nodal_mask, levels=[0.5],
            colors="white", linewidths=1.0
        )

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name} (IoU={iou_val:.2f})")

    # hide any unused subplots
    for extra_ax in axes[n_shapes:]:
        extra_ax.axis("off")

    # fig.colorbar(tpc, ax=axes.tolist(), shrink=0.6)
    fig.suptitle(
        "EIT Reconstructions of Different Contact Geometries (BP)\n"
        "Filled: reconstruction, white contour: ground-truth shape",
        fontsize=14,
    )
    plt.tight_layout()
    
    plt.show()



if __name__ == "__main__":
    run()
