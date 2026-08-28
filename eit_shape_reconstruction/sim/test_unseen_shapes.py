#!/usr/bin/env python3
"""
Test generalization of the hybrid EIT model to unseen shapes:
    - C shape
    - Z shape
    - plus (+) shape

For each shape:
    - Random pose (offset + rotation)
    - Random contrast (same range as training)
    - Forward EIT simulation
    - BP reconstruction
    - Hybrid model prediction
    - IoU & Dice vs ground truth

Also saves a few visualization examples per shape:
    BP recon | Ground truth | Prediction
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch

# Import model + some constants from training script
from sim.train_sim_eit_hybrid import (
    HybridEITNet,
    GRID_SIZE,
    DATA_ROOT,
)

import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol
from pyeit.eit.fem import EITForward
import pyeit.eit.bp as bp


# ------------------ CONFIG ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_SAMPLES_PER_SHAPE = 300   # how many test cases per unseen shape
UNSEEN_SHAPES = ["C", "Z", "plus"]

CONTRAST_LEVELS = [3.0, 5.0, 10.0, 15.0, 20.0]
RANDOM_SEED = 123

CKPT_NAME = "best_hybrid_model.pt"
VIZ_PER_SHAPE = 6           # how many examples to save per shape
VIZ_DIR = Path(DATA_ROOT) / "unseen_shapes_viz"


# ------------- BASIC HELPERS (mesh & geometry) -------------
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


# ------------- UNSEEN SHAPE MASKS (per-element) -------------
def C_mask(mesh_obj, offset=(0.0, 0.0), angle=0.0,
           r_outer=0.35, r_inner=0.20, gap_angle=np.pi/3):
    """
    'C' shape = annulus with a missing angular segment.
    gap_angle: half-width of the missing sector (opening of the C)
               we open around +x direction in local frame.
    """
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]

    # rotate global -> local
    x_local, y_local = rotate_points(x, y, -angle)

    r2 = x_local**2 + y_local**2
    theta = np.arctan2(y_local, x_local)  # [-pi, pi]

    # annulus
    annulus = (r_inner**2 <= r2) & (r2 <= r_outer**2)

    # keep everything EXCEPT an angular gap around theta=0 (the opening)
    gap = (np.abs(theta) < gap_angle)
    return annulus & (~gap)


def plus_mask_rotated(mesh_obj, offset=(0.0, 0.0), angle=0.0,
                      arm_width=0.10, arm_length=0.45):
    """
    '+' shape: union of horizontal and vertical bars in a local frame,
    then rotated and translated.
    """
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    x_local, y_local = rotate_points(x, y, -angle)

    vert = (np.abs(x_local) < arm_width/2) & (np.abs(y_local) < arm_length/2)
    horiz = (np.abs(y_local) < arm_width/2) & (np.abs(x_local) < arm_length/2)
    return vert | horiz


def Z_mask_rotated(mesh_obj, offset=(0.0, 0.0), angle=0.0,
                   width=0.10, length=0.60):
    """
    'Z' shape in local coords:
        - top horizontal bar
        - bottom horizontal bar
        - diagonal bar connecting them (approx)
    """
    c = element_centroids(mesh_obj)
    x = c[:, 0] - offset[0]
    y = c[:, 1] - offset[1]
    x_local, y_local = rotate_points(x, y, -angle)

    # local coordinate frame:
    # x in [-L/2, L/2], y in [-0.5, 0.5] (approx)
    half_L = length / 2.0

    top = (y_local > 0.25) & (y_local < 0.35) & (x_local > -half_L) & (x_local < half_L)
    bottom = (y_local > -0.35) & (y_local < -0.25) & (x_local > -half_L) & (x_local < half_L)

    # diagonal piece: around line y = -x in local frame
    # restrict to central band in y
    diag_band = (y_local > -0.25) & (y_local < 0.25)
    # distance to line y = -x is |y + x| / sqrt(2)
    dist_to_diag = np.abs(y_local + x_local) / np.sqrt(2.0)
    diag = diag_band & (dist_to_diag < width/2.0)

    return top | bottom | diag


def random_unseen_shape_mask(mesh_obj, rng, shape_type):
    """
    Sample a random instance of an unseen shape (C, Z, plus).
    Returns (mask, shape_type, contrast).
    """
    contrast = float(rng.choice(CONTRAST_LEVELS))

    # random translation (keep inside unit disk-ish)
    ox = rng.uniform(-0.2, 0.2)
    oy = rng.uniform(-0.2, 0.2)
    angle = rng.uniform(0.0, 2.0 * np.pi)

    if shape_type == "C":
        mask = C_mask(mesh_obj, offset=(ox, oy), angle=angle,
                      r_outer=0.35, r_inner=0.18,
                      gap_angle=np.pi / 4)
    elif shape_type == "plus":
        mask = plus_mask_rotated(mesh_obj, offset=(ox, oy), angle=angle,
                                 arm_width=0.12, arm_length=0.50)
    elif shape_type == "Z":
        mask = Z_mask_rotated(mesh_obj, offset=(ox, oy), angle=angle,
                              width=0.10, length=0.60)
    else:
        raise ValueError(f"Unknown unseen shape type: {shape_type}")

    return mask, contrast


# ------------- EIT & GRID HELPERS -------------
def setup_eit(n_el=16, h0=0.04):
    mesh_obj = mesh.create(n_el, h0=h0)

    protocol_obj = protocol.create(
        n_el, dist_exc=1, step_meas=1, parser_meas="std"
    )
    fwd = EITForward(mesh_obj, protocol_obj)

    n_elems = mesh_obj.element.shape[0]
    perm_ref = np.ones(n_elems, dtype=float)
    v0 = fwd.solve_eit(perm_ref)

    eit_bp = bp.BP(mesh_obj, protocol_obj)
    eit_bp.setup(weight="none")

    return mesh_obj, protocol_obj, fwd, eit_bp, v0


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


def domain_mask_grid(grid_size=GRID_SIZE):
    lin = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(lin, lin)
    return ((xx ** 2 + yy ** 2) <= 1.0).astype(np.float32)


# ------------- METRICS -------------
def iou_numpy(pred, target, eps=1e-6):
    """
    pred, target: [H,W] in {0,1}
    """
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - inter
    return float(inter / (union + eps))


def dice_numpy(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = np.sum(pred * target)
    denom = np.sum(pred) + np.sum(target)
    return float(2.0 * inter / (denom + eps))


# ------------- VISUALIZATION -------------
def visualize_example(bp_grid, gt_mask, pred_mask, shape_type, idx, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    ax0, ax1, ax2 = axes

    im0 = ax0.imshow(bp_grid, cmap="viridis")
    ax0.set_title("BP recon")
    ax0.axis("off")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    ax1.imshow(gt_mask, cmap="gray")
    ax1.set_title("Ground truth")
    ax1.axis("off")

    ax2.imshow(pred_mask, cmap="gray")
    ax2.set_title("Prediction")
    ax2.axis("off")

    fig.suptitle(f"Unseen shape: {shape_type}", fontsize=12)
    plt.tight_layout()

    out_path = out_dir / f"{shape_type}_example_{idx:03d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------- MAIN -------------
def main():
    print(f"Using device: {DEVICE}")
    rng = np.random.default_rng(RANDOM_SEED)

    # Prepare output dir
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Visualization dir: {VIZ_DIR}")

    # EIT setup (same style as training)
    mesh_obj, protocol_obj, fwd, eit_bp, v0 = setup_eit()
    v0 = v0.astype(np.float32)

    triang, xx, yy = make_grid(mesh_obj, grid_size=GRID_SIZE)
    domain_grid = domain_mask_grid(GRID_SIZE)

    # Load model
    ckpt_path = Path(DATA_ROOT) / CKPT_NAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    volt_dim = v0.size  # number of measurements
    model = HybridEITNet(volt_dim=volt_dim, img_size=(GRID_SIZE, GRID_SIZE), feat_channels=16)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(
        f"Loaded model from epoch {ckpt.get('epoch', '?')} "
        f"with training/test IoU {ckpt.get('test_iou', 'N/A'):.4f}"
    )

    overall_iou = []
    overall_dice = []

    per_shape_iou = {s: [] for s in UNSEEN_SHAPES}
    per_shape_dice = {s: [] for s in UNSEEN_SHAPES}

    for shape_type in UNSEEN_SHAPES:
        print(f"\n=== Testing unseen shape: {shape_type} ===")

        viz_count = 0

        for idx in range(N_SAMPLES_PER_SHAPE):
            # 1) generate shape mask + contrast
            elem_mask, contrast = random_unseen_shape_mask(mesh_obj, rng, shape_type)

            # 2) build perm and simulate EIT
            perm = make_perm_for_mask(mesh_obj, elem_mask, contrast=contrast)
            v1 = fwd.solve_eit(perm).astype(np.float32)
            delta_v = v1 - v0  # same as training

            # 3) BP reconstruction (nodal)
            sum_abs_diff = np.sum(np.abs(delta_v))
            if sum_abs_diff > 0.3:
                nodal_bp = 192.0 * eit_bp.solve(
                    v1, v0, normalize=True, log_scale=False
                )
            else:
                nodal_bp = 192.0 * eit_bp.solve(
                    v0, v0, normalize=True, log_scale=False
                )
            nodal_bp = np.real(nodal_bp).astype(np.float32)

            bp_grid = rasterize_to_grid(triang, xx, yy, nodal_bp, fill_value=0.0)
            bp_grid = bp_grid.astype(np.float32)

            # 4) ground-truth mask on grid
            nodal_mask = element_mask_to_nodal(mesh_obj, elem_mask)
            mask_grid_float = rasterize_to_grid(
                triang, xx, yy, nodal_mask, fill_value=0.0
            )
            gt_mask_grid = (mask_grid_float > 0.5).astype(np.float32)

            # 5) prepare tensors & run model
            volt_t = torch.from_numpy(delta_v).unsqueeze(0).to(DEVICE)        # [1,K]
            bp_t = torch.from_numpy(bp_grid).unsqueeze(0).unsqueeze(0).to(DEVICE)   # [1,1,H,W]
            dom_t = torch.from_numpy(domain_grid).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,H,W]

            with torch.no_grad():
                logits = model(volt_t, bp_t, dom_t, bp_dropout_prob=0.0)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]  # [H,W]

            pred_mask_grid = (probs > 0.5).astype(np.float32)

            # 6) metrics
            iou_val = iou_numpy(pred_mask_grid, gt_mask_grid)
            dice_val = dice_numpy(pred_mask_grid, gt_mask_grid)

            per_shape_iou[shape_type].append(iou_val)
            per_shape_dice[shape_type].append(dice_val)
            overall_iou.append(iou_val)
            overall_dice.append(dice_val)

            # 7) visualization (few examples)
            if viz_count < VIZ_PER_SHAPE:
                visualize_example(bp_grid, gt_mask_grid, pred_mask_grid,
                                  shape_type, viz_count, VIZ_DIR)
                viz_count += 1

        # per-shape summary
        miou = float(np.mean(per_shape_iou[shape_type]))
        mdice = float(np.mean(per_shape_dice[shape_type]))
        print(f"{shape_type}: mean IoU={miou:.4f}, mean Dice={mdice:.4f}")

    print("\n=== Overall unseen-shape performance ===")
    print(f"Mean IoU:  {np.mean(overall_iou):.4f}")
    print(f"Mean Dice: {np.mean(overall_dice):.4f}")
    print(f"Visualizations saved in: {VIZ_DIR}")


if __name__ == "__main__":
    main()
