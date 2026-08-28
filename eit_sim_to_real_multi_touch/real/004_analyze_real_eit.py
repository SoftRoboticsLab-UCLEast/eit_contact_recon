#!/usr/bin/env python3
# Analyze real EIT double-touch dataset:
# - Load per-robot calibration JSONs (2D similarity transforms)
# - Load fused CSV (R1_tx,R1_ty,R2_tx,R2_ty, segment_id)
# - Map robot-frame XY -> sensor disk frame (meters)
# - Save: tidy per-contact CSV, text summary, and plots

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Helpers
# --------------------------
def load_calib(path: Path):
    with open(path, "r") as f:
        c = json.load(f)
    scale = float(c["forward_robot_to_sensor"]["scale"])
    R = np.array(c["forward_robot_to_sensor"]["R"], dtype=float)  # 2x2
    t = np.array(c["forward_robot_to_sensor"]["t"], dtype=float)  # (2,)
    rms = float(c.get("meta", {}).get("rms_error_mm", np.nan))
    return {"scale": scale, "R": R, "t": t, "rms_mm": rms, "raw": c}

def apply_similarity(p_xy: np.ndarray, calib):
    # p_xy: (..., 2)
    p_xy = np.asarray(p_xy, dtype=float)
    return calib["scale"] * (p_xy @ calib["R"].T) + calib["t"]

def compute_polar(xy: np.ndarray):
    r = np.sqrt((xy**2).sum(axis=1))
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    return r, theta

def plot_sensor_disk(ax, radius_m):
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(radius_m*np.cos(th), radius_m*np.sin(th), lw=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

def ensure_columns(df: pd.DataFrame, names):
    for n in names:
        if n not in df.columns:
            df[n] = np.nan
    return df

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="EIT real dataset contact distribution analysis")
    ap.add_argument("--calib-r1", required=True, type=Path, help="Robot1 calibration JSON")
    ap.add_argument("--calib-r2", required=True, type=Path, help="Robot2 calibration JSON")
    ap.add_argument("--csv", required=True, type=Path, help="Fused CSV with R1/R2 positions and EIT")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory for plots/CSV/summary")
    ap.add_argument("--radius-m", default=0.09, type=float, help="Sensor radius in meters (default 0.09)")
    ap.add_argument("--edge-eps-m", default=0.002, type=float, help="On-disk tolerance (default 2 mm)")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Load calibrations
    cal_r1 = load_calib(args.calib_r1)
    cal_r2 = load_calib(args.calib_r2)

    # Read CSV
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    needed_cols = ["segment_id", "R1_tx", "R1_ty", "R2_tx", "R2_ty"]
    df = ensure_columns(df, needed_cols)

    # Extract and transform
    r1_xy = df[["R1_tx", "R1_ty"]].to_numpy(dtype=float, copy=True)
    r2_xy = df[["R2_tx", "R2_ty"]].to_numpy(dtype=float, copy=True)

    r1_sens = apply_similarity(r1_xy, cal_r1)
    r2_sens = apply_similarity(r2_xy, cal_r2)

    # Build tidy per-contact table
    rows = []
    def append_rows(sensor_xy, robot_id, seg_ids):
        r, th = compute_polar(sensor_xy)
        for i in range(sensor_xy.shape[0]):
            x, y = sensor_xy[i]
            seg = seg_ids[i] if seg_ids is not None else np.nan
            rows.append({
                "segment_id": seg,
                "robot_id": robot_id,
                "x_sens_m": x,
                "y_sens_m": y,
                "r_m": r[i],
                "theta_rad": th[i],
            })

    seg = df["segment_id"].to_numpy() if "segment_id" in df.columns else None
    append_rows(r1_sens, "R1", seg)
    append_rows(r2_sens, "R2", seg)

    tidy = pd.DataFrame(rows)
    tidy["on_disk"] = tidy["r_m"] <= (args.radius_m + args.edge_eps_m)
    tidy["near_edge"] = (tidy["r_m"] >= (0.85 * args.radius_m)) & tidy["on_disk"]

    # Save tidy CSV
    tidy_csv = outdir / "sensor_frame_points.csv"
    tidy.to_csv(tidy_csv, index=False)

    # Pairwise inter-contact distance (same-row pairs)
    valid_pair = np.isfinite(r1_sens).all(axis=1) & np.isfinite(r2_sens).all(axis=1)
    pair_dist = np.linalg.norm(r1_sens[valid_pair] - r2_sens[valid_pair], axis=1)

    # Summary
    n_total = len(tidy)
    n_on = int(tidy["on_disk"].sum())
    n_off = int((~tidy["on_disk"]).sum())
    frac_off = n_off / max(n_total, 1)
    radial_on = tidy.loc[tidy["on_disk"], "r_m"]
    radial_mean = float(radial_on.mean()) if len(radial_on) else float("nan")
    radial_median = float(radial_on.median()) if len(radial_on) else float("nan")
    radial_p95 = float(radial_on.quantile(0.95)) if len(radial_on) else float("nan")

    summary_txt = outdir / "real_data_analysis_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("=== Real Dataset Distribution Summary ===\n")
        f.write(f"Sensor radius (m): {args.radius_m}\n")
        f.write(f"Calibration RMS (mm): R1={cal_r1['rms_mm']:.2f}, R2={cal_r2['rms_mm']:.2f}\n")
        f.write(f"Tidy contacts total: {n_total}\n")
        f.write(f"On-disk count: {n_on}\n")
        f.write(f"Off-disk count: {n_off}  (fraction {frac_off:.3%})\n")
        f.write(f"Radial stats on-disk (m): mean={radial_mean:.4f}, median={radial_median:.4f}, p95={radial_p95:.4f}\n")
        if pair_dist.size > 0:
            f.write(
                "Pairwise distance (double-touch) — "
                f"count={pair_dist.size}, mean={pair_dist.mean():.4f} m, "
                f"median={np.median(pair_dist):.4f} m, "
                f"p5={np.percentile(pair_dist,5):.4f} m, "
                f"p95={np.percentile(pair_dist,95):.4f} m\n"
            )
        else:
            f.write("Pairwise distance: no valid simultaneous pairs detected.\n")

    # -------- Plots (each in its own figure) --------
    # 1) Scatter of both robots on disk
    fig1 = plt.figure(figsize=(6, 6))
    ax1 = fig1.add_subplot(111)
    plot_sensor_disk(ax1, args.radius_m)
    ax1.scatter(r1_sens[:, 0], r1_sens[:, 1], s=6, label="R1", alpha=0.7)
    ax1.scatter(r2_sens[:, 0], r2_sens[:, 1], s=6, label="R2", alpha=0.7)
    ax1.set_title("Contact point distribution (sensor frame)")
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    fig1.savefig(outdir / "scatter_sensor_frame.png", dpi=180)

    # 2) Radial histogram (on-disk only)
    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(111)
    ax2.hist(radial_on.to_numpy(), bins=40)
    ax2.set_xlabel("radius r (m)")
    ax2.set_ylabel("count")
    ax2.set_title("Radial distribution (on-disk)")
    fig2.tight_layout()
    fig2.savefig(outdir / "hist_radial.png", dpi=180)

    # 3) Angular histogram (on-disk only)
    fig3 = plt.figure(figsize=(6, 4))
    ax3 = fig3.add_subplot(111)
    theta_deg = np.degrees(tidy.loc[tidy["on_disk"], "theta_rad"].to_numpy())
    ax3.hist(theta_deg, bins=36, range=(-180, 180))
    ax3.set_xlabel("angle θ (deg)")
    ax3.set_ylabel("count")
    ax3.set_title("Angular distribution (on-disk)")
    fig3.tight_layout()
    fig3.savefig(outdir / "hist_angular.png", dpi=180)

    # 4) Inter-contact distance histogram (valid pairs only)
    if pair_dist.size > 0:
        fig4 = plt.figure(figsize=(6, 4))
        ax4 = fig4.add_subplot(111)
        ax4.hist(pair_dist, bins=40)
        ax4.set_xlabel("inter-contact distance (m)")
        ax4.set_ylabel("count")
        ax4.set_title("Double-touch inter-contact distances")
        fig4.tight_layout()
        fig4.savefig(outdir / "hist_intercontact.png", dpi=180)

    print("Done.")
    print(f"- Tidy CSV -> {tidy_csv}")
    print(f"- Summary  -> {summary_txt}")
    print(f"- Plots    -> {outdir / 'scatter_sensor_frame.png'}, "
          f"{outdir / 'hist_radial.png'}, {outdir / 'hist_angular.png'}, "
          f"{outdir / 'hist_intercontact.png'} (if pairs exist)")

if __name__ == "__main__":
    main()
