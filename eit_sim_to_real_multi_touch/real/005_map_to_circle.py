#!/usr/bin/env python3
"""
Per-robot XY translation for EIT contact points, with optional boundary clamp
ONLY for a selected robot (e.g., R1). No scaling, no warping.

- Left plot: BEFORE (original sensor-frame points)
- Right plot: AFTER (with per-robot translation + optional clamp for selected robot)

Outputs:
  - sensor_frame_points_translated_per_robot.csv
  - scatter_before_after_translation_per_robot.png
  - translation_summary.txt

run command:
    python real/005_map_to_circle.py   --in-csv translation_test/sensor_frame_points_translated_per_robot.csv   --outdir ./translation_test   --radius-m 0.09   --r1-id R1 --r1-shift-x 0.005 --r1-shift-y 0.01   --r2-id R2 --r2-shift-x -0.04 --r2-shift-y -0.11   --clamp-robot R1 --clamp-eps 0.0005
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_sensor_disk(ax, R):
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(R*np.cos(th), R*np.sin(th), "k-", lw=1.0)


def scatter_before_after(df_before, df_after, R, out_png):
    fig = plt.figure(figsize=(12, 5))

    # BEFORE
    ax1 = fig.add_subplot(1, 2, 1)
    plot_sensor_disk(ax1, R)
    for rid, sub in df_before.groupby("robot_id"):
        ax1.scatter(sub["x_sens_m"], sub["y_sens_m"], s=8, alpha=0.7, label=str(rid))
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Before translation")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.legend(loc="upper right")

    # AFTER
    ax2 = fig.add_subplot(1, 2, 2)
    plot_sensor_disk(ax2, R)
    for rid, sub in df_after.groupby("robot_id"):
        ax2.scatter(sub["x_trans_m"], sub["y_trans_m"], s=8, alpha=0.7, label=str(rid))
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("After per-robot translation (+ optional clamp)")
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Per-robot translation with optional clamp for one robot")
    ap.add_argument("--in-csv", required=True, type=Path, help="Input CSV with columns: robot_id,x_sens_m,y_sens_m")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--radius-m", type=float, default=0.09, help="Sensor radius (m)")

    # Robot IDs (as they appear in 'robot_id' column)
    ap.add_argument("--r1-id", type=str, default="R1")
    ap.add_argument("--r2-id", type=str, default="R2")

    # Per-robot translations (meters)
    ap.add_argument("--r1-shift-x", type=float, default=0.0)
    ap.add_argument("--r1-shift-y", type=float, default=0.0)
    ap.add_argument("--r2-shift-x", type=float, default=0.0)
    ap.add_argument("--r2-shift-y", type=float, default=0.0)

    # Default translation for any other labels (optional)
    ap.add_argument("--default-shift-x", type=float, default=0.0)
    ap.add_argument("--default-shift-y", type=float, default=0.0)

    # Clamp settings: ONLY this robot gets its out-of-disk points projected to boundary
    ap.add_argument("--clamp-robot", type=str, default="R1",
                    help="Robot label whose out-of-disk points will be clamped to the boundary")
    ap.add_argument("--clamp-eps", type=float, default=5e-4,
                    help="Tiny inward margin when clamping (m), default 0.0005 m")

    args = ap.parse_args()
    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df_before = pd.read_csv(args.in_csv)
    for c in ["robot_id", "x_sens_m", "y_sens_m"]:
        if c not in df_before.columns:
            raise ValueError(f"Missing required column '{c}' in {args.in_csv}")

    # Build per-row shifts
    df_after = df_before.copy()
    rid_str = df_after["robot_id"].astype(str)

    dx = np.full(len(df_after), args.default_shift_x, dtype=float)
    dy = np.full(len(df_after), args.default_shift_y, dtype=float)

    m1 = (rid_str == str(args.r1_id))
    m2 = (rid_str == str(args.r2_id))
    dx[m1] = args.r1_shift_x; dy[m1] = args.r1_shift_y
    dx[m2] = args.r2_shift_x; dy[m2] = args.r2_shift_y

    # Apply translations
    df_after["x_trans_m"] = df_after["x_sens_m"] + dx
    df_after["y_trans_m"] = df_after["y_sens_m"] + dy

    # Clamp ONLY the selected robot's out-of-disk points (project to boundary - eps)
    clamp_mask = (rid_str == str(args.clamp_robot))
    x = df_after.loc[clamp_mask, "x_trans_m"].to_numpy(copy=True)
    y = df_after.loc[clamp_mask, "y_trans_m"].to_numpy(copy=True)
    r = np.sqrt(x**2 + y**2)
    too_big = r > args.radius_m
    if np.any(too_big):
        # scale factor per point
        scale = (args.radius_m - args.clamp_eps) / r[too_big]
        x[too_big] *= scale
        y[too_big] *= scale
        # write back
        df_after.loc[clamp_mask, "x_trans_m"] = x
        df_after.loc[clamp_mask, "y_trans_m"] = y

    # Recompute polar + inside flag
    df_after["r_trans_m"] = np.sqrt(df_after["x_trans_m"]**2 + df_after["y_trans_m"]**2)
    df_after["theta_trans_rad"] = np.arctan2(df_after["y_trans_m"], df_after["x_trans_m"])
    df_after["on_disk_after"] = df_after["r_trans_m"] <= (args.radius_m + 1e-9)

    # Save CSV
    out_csv = outdir / "sensor_frame_points_translated_per_robot.csv"
    df_after.to_csv(out_csv, index=False)

    # Save plot (fixed: proper BEFORE/AFTER datasets)
    out_png = outdir / "scatter_before_after_translation_per_robot.png"
    scatter_before_after(df_before, df_after, args.radius_m, out_png)

    # Summary
    summary_path = outdir / "translation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Translation + selective clamp summary ===\n")
        f.write(f"Radius: {args.radius_m} m\n")
        f.write(f"R1 ({args.r1_id}) shift: dx={args.r1_shift_x:+.6f}, dy={args.r1_shift_y:+.6f} m\n")
        f.write(f"R2 ({args.r2_id}) shift: dx={args.r2_shift_x:+.6f}, dy={args.r2_shift_y:+.6f} m\n")
        f.write(f"Default shift      : dx={args.default_shift_x:+.6f}, dy={args.default_shift_y:+.6f} m\n")
        f.write(f"Clamp robot: {args.clamp_robot}, epsilon={args.clamp_eps} m\n\n")
        for rid, sub in df_after.groupby("robot_id"):
            on = int(sub["on_disk_after"].sum())
            tot = int(len(sub))
            max_r = float(sub["r_trans_m"].max())
            f.write(f"[{rid}] inside: {on}/{tot} ({on/max(tot,1):.1%}), max r={max_r:.5f} m\n")

    print("Done.")
    print(f"- CSV  -> {out_csv}")
    print(f"- Plot -> {out_png}")
    print(f"- Info -> {summary_path}")


if __name__ == "__main__":
    main()
