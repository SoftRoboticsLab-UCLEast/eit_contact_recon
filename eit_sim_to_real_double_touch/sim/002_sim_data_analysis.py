#!/usr/bin/env python3
"""
Analysis and visualization for simulated double-touch EIT dataset.
Checks:
  1. Scatter plot of all contact locations
  2. Electrode geometry visualization
  3. Reconstructed maps overlayed with contact points
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path

import pyeit.mesh as mesh
from pyeit.eit.protocol import create as create_protocol
from pyeit.eit.fem import EITForward
import pyeit.eit.bp as bp

plt.rcParams['pdf.fonttype'] = 42   # use TrueType fonts, not Type 3
plt.rcParams['ps.fonttype'] = 42
# =====================================================
# CONFIGURATION
# =====================================================
DATA_FILE = "/home/kiyanoush/Projects/eit_sim_to_real_multi_touch/data/eit_sim_data_double_touch/sim_double_touch.csv"  # update if needed
N_EL = 16
ROT_DEG = 80          # +30° offset (1.5 o'clock)
DIST_EXC = 1
STEP_MEAS = 1
N_RECONS = 300          # how many reconstructions to visualize
SCALE = 192.0
CMAP = "viridis"


# =====================================================
# LOAD DATA
# =====================================================
print(f"[INFO] Loading data: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
print(f"[INFO] Loaded {len(df)} rows")

# Detect column naming scheme (some generators use x1, some use x1_norm)
def find_coord_cols(df):
    if "x1_norm" in df.columns:
        return "x1_norm", "y1_norm", "x2_norm", "y2_norm"
    elif "x1" in df.columns:
        return "x1", "y1", "x2", "y2"
    else:
        raise RuntimeError("Could not find x1/x2 columns in dataset.")

x1_col, y1_col, x2_col, y2_col = find_coord_cols(df)
x1, y1 = df[x1_col], df[y1_col]
x2, y2 = df[x2_col], df[y2_col]


# =====================================================
# 1. SCATTER DISTRIBUTION OF TOUCH POINTS
# =====================================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x1, y1, c='tab:blue', s=10, alpha=0.5, label='Touch 1')
ax.scatter(x2, y2, c='tab:orange', s=10, alpha=0.5, label='Touch 2')
circle = plt.Circle((0, 0), 1.0, fill=False, color='k', lw=1.2)
ax.add_patch(circle)
ax.set_aspect('equal')
ax.legend()
ax.set_title("Distribution of Double-Touch Points")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()
plt.savefig("contact_distribution.png", dpi=200)
plt.close(fig)
print("[INFO] Saved contact distribution → contact_distribution.png")


# =====================================================
# 2. ELECTRODE LAYOUT VISUALIZATION (robust version)
# =====================================================
mesh_obj = mesh.create(n_el=N_EL)
# Apply same rotation and CW numbering as generator
rot_offset = int(round(N_EL * ROT_DEG / 360))
mesh_obj.el_pos = np.roll(mesh_obj.el_pos[::-1], shift=rot_offset, axis=0)

pts = mesh_obj.node
els = mesh_obj.element

# --- Robust electrode coordinate extraction ---
# Analytic electrode positions (exactly on unit circle)
angles_deg = 80.0 - np.arange(N_EL) * (360.0 / N_EL)  # +30°, CW numbering
angles = np.deg2rad(angles_deg)
el_coords = np.c_[np.cos(angles), np.sin(angles)]

fig, ax = plt.subplots(figsize=(6, 6))
ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color='k', lw=1))
ax.scatter(el_coords[:, 0], el_coords[:, 1], color='red', zorder=5)
for i, (x, y) in enumerate(el_coords):
    ax.text(x * 1.08, y * 1.08, str(i + 1), ha='center', va='center', fontsize=9, color='red')
ax.set_aspect('equal')
ax.set_title("Electrode Positions (Rotated +30°, CW numbering)")
plt.tight_layout()
plt.savefig("electrode_layout.png", dpi=200)
plt.close(fig)
print("[INFO] Saved electrode layout → electrode_layout.png")


# =====================================================
# 3. RECONSTRUCTION SANITY CHECK
# =====================================================
protocol_obj = create_protocol(N_EL, dist_exc=DIST_EXC, step_meas=STEP_MEAS)
fwd = EITForward(mesh_obj, protocol_obj)
v0 = fwd.solve_eit(mesh_obj.perm)

eit_bp = bp.BP(mesh_obj, protocol_obj)
eit_bp.setup(weight="none")

pts = mesh_obj.node
tri = mesh_obj.element

# Find EIT columns (some datasets use v0... others eit_0...)
eit_cols = [c for c in df.columns if c.startswith("eit_") or c.startswith("v")]
if not eit_cols:
    raise RuntimeError("No EIT voltage columns found (expected 'eit_0' or 'v0').")

print(f"[INFO] Found {len(eit_cols)} EIT columns.")

# Choose random samples
sample_indices = random.sample(range(len(df)), min(N_RECONS, len(df)))
print(f"[INFO] Reconstructing {len(sample_indices)} random samples...")

for idx, i in enumerate(sample_indices, start=1):
    row = df.iloc[i]
    v_touch = row[eit_cols].to_numpy(np.float32)
    ds = SCALE * eit_bp.solve(v_touch + v0, v0, normalize=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, ds, shading="flat", cmap=CMAP)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color='k', lw=0.8))
    # ax.scatter([row[x1_col], row[x2_col]], [row[y1_col], row[y2_col]],
    #            c=['tab:blue', 'tab:orange'], s=80, marker='x', lw=2, label='True touches')
    ax.set_aspect('equal')
    # ax.set_title(f"Reconstruction #{idx}")
    # ax.legend()
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"reconstruction_{idx}.png", dpi=300)
    plt.close(fig)

print(f"[INFO] Saved {len(sample_indices)} reconstruction images.")
print("✅ Analysis complete.")
