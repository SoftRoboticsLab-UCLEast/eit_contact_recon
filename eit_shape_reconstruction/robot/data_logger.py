#!/usr/bin/env python3
"""
UR5 + EIT data collection script for EIT-based tactile contact-shape experiments.

Key features:
- Robot starts from SENSOR_CENTER_POSE (center of circular sensor, zero yaw).
- For each sample:
    - (First 4 samples): calibration contacts at sensor center (u=v=0) with yaw = 0°, 90°, 180°, 270°.
    - Remaining samples:
        - Randomly sample:
            - shape_type (e.g. "L", "T", "ring", "double_circle", ...)
            - contact center (u_m, v_m) inside sensor disc, ensuring FULL shape stays inside
            - yaw rotation (theta_z) around sensor normal
    - Move in-plane to that (x, y, yaw) above sensor (safe z, no contact).
    - Move vertically down by FIXED_PRESS_DEPTH_M.
    - Wait SETTLE_S seconds.
    - Sample one EIT frame (latest readings).
    - Log: timestamp, TCP pose, shape_type, u_m, v_m, r_norm, theta, yaw, depth, EIT channels.
    - Return to safe height at same (x, y, yaw).
- At the end, return to SENSOR_CENTER_POSE and save all rows to Parquet (full run snapshot).

NOTE: This script does NOT generate ground-truth images directly.
      It logs everything needed to reconstruct 2D contact masks offline:
      (shape_type, center (u_m, v_m), yaw, sensor radius, and CAD-defined shape geometry).
"""

import time
from datetime import datetime
from pathlib import Path
import threading
import queue
import json

import numpy as np
import pandas as pd

import URBasic
try:
    import serial
except ImportError:
    serial = None

# If you want to actually rotate orientation around z, you'll need scipy:
try:
    from scipy.spatial.transform import Rotation as R
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("[warn] scipy not installed; orientation yaw will NOT be applied to the UR5 pose (only logged).")


# =====================
# CONFIG — EDIT ME
# =====================
CONFIG = {
    # Robot
    "ROBOT_IP": "169.254.76.5",

    # Sensor center pose [x, y, z, rx, ry, rz] in base frame.
    # This should place the EE directly above the center of the circular sensor at SAFE height
    # with the desired "zero yaw" orientation.
    "SENSOR_CENTER_POSE": [-0.5072, -0.0026, 0.16, 0.0, 3.142, 0.0],

    # Sensor physical radius (meters)
    "SENSOR_RADIUS_M": 0.045,

    # Per-shape maximum radial extent (meters) of contact geometry
    # i.e., max distance from shape center to any contact point, in the contact plane.
    # >>> Fill these from your CAD files <<<
    "SHAPE_MAX_RADIUS_M": {
        "L": 0.024,              # placeholder; set from CAD
        "T": 0.028,
        "edge": 0.025,
        "ring": 0.025,           # for ring centered on EE, use its outer radius
        "double_circle": 0.024, # distance from center to farthest circle edge
        "C": 0.024,
        "Z": 0.029,
        "+": 0.021,
    },

    # Which shapes to collect data for
    # "SHAPE_TYPES": ["L", "T", "edge", "ring", "double_circle", "C", "Z", "+"],
    "SHAPE_TYPES": ["Z"],

    # Sampling parameters
    "N_SAMPLES": 500,          # total number of contact samples in this run
    "R_MIN_FRAC": 0.0,          # minimum normalized radius for contact center (0 = include center)
    "EDGE_MARGIN_FRAC": 0.02,   # small safety margin so shapes do not touch exact boundary

    # Yaw (rotation around sensor normal) range, radians
    "YAW_MIN": 0.0,
    "YAW_MAX": 0.5 * np.pi,

    # Motion: press depth (contact) and speeds
    "FIXED_PRESS_DEPTH_M": 0.059,   # 10 mm downwards from safe height
    "MOVE_V": 0.5,                  # general move velocity
    "MOVE_A": 0.5,                  # general move acceleration
    "PRESS_V": 0.02,                # vertical press velocity
    "PRESS_A": 0.02,                # vertical press acceleration

    # Settling time at contact depth before sampling EIT (seconds)
    "SETTLE_S": 3.0,

    # Serial ports (EIT only)
    "EIT_PORT": "/dev/ttyACM0",     # update to your EIT board
    "EIT_BAUD": 115200,

    # Output
    "OUT_DIR": "./data",
    "SESSION_TAG": "eit_random_shapes",

    # Live logging
    "CSV_LIVE": True,
    "PARQUET_EVERY_N": 0,           # >0 to write chunked parquet
}


# =====================
# Robot helpers
# =====================

def connect_robot(ip: str):
    model = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=ip, robotModel=model)
    try:
        robot.reset_error()
        time.sleep(0.5)
    except Exception:
        pass
    return robot


def movel(robot, pose, a=0.05, v=0.05):
    robot.movel(pose, a=a, v=v)


def set_z(pose, new_z):
    p = pose.copy()
    p[2] = new_z
    return p


def shift_xy(pose, du_m, dv_m):
    p = pose.copy()
    p[0] += du_m
    p[1] += dv_m
    return p


def apply_yaw_to_pose(pose, yaw_rad, base_pose=None):
    """
    Apply an additional yaw (rotation about the sensor normal) to the pose.
    We assume 'zero yaw' is the orientation of SENSOR_CENTER_POSE.
    This function uses scipy to:
        - interpret pose orientation as axis-angle (rx, ry, rz)
        - convert to rotation matrix
        - multiply by Rz(yaw)
        - convert back to axis-angle

    If scipy is not available, we just return the original pose unchanged
    and only log 'yaw_rad' as metadata.
    """
    if not HAVE_SCIPY:
        return pose

    # Use base_pose orientation as reference for zero-yaw (if provided).
    # Otherwise, use pose's own orientation as baseline.
    if base_pose is None:
        base_pose = pose

    rx0, ry0, rz0 = base_pose[3], base_pose[4], base_pose[5]
    angle0 = np.linalg.norm([rx0, ry0, rz0])
    if angle0 < 1e-9:
        # Pure identity orientation; just apply Rz(yaw)
        R0 = np.eye(3)
    else:
        axis0 = np.array([rx0, ry0, rz0]) / angle0
        R0 = R.from_rotvec(axis0 * angle0).as_matrix()

    # Yaw about z-axis in sensor frame
    Rz = R.from_euler('z', yaw_rad).as_matrix()

    # New rotation
    R_new = Rz @ R0
    rvec = R.from_matrix(R_new).as_rotvec()
    new_pose = pose.copy()
    new_pose[3] = float(rvec[0])
    new_pose[4] = float(rvec[1])
    new_pose[5] = float(rvec[2])
    return new_pose


# =====================
# Serial / EIT helpers
# =====================

class SerialLineReader(threading.Thread):
    """Generic serial line reader: reads lines, parses them, pushes dicts to a queue."""
    def __init__(self, port, baud, parse_fn, out_q, name, init_fn=None):
        super().__init__(daemon=True, name=name)
        if serial is None:
            raise RuntimeError("pyserial not installed. pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.parse_fn = parse_fn
        self.q = out_q
        self.stop_evt = threading.Event()
        self.init_fn = init_fn

    def run(self):
        if self.init_fn is not None:
            try:
                self.init_fn(self.ser)
            except Exception as e:
                print(f"[{self.name}] init_fn error: {e}")

        while not self.stop_evt.is_set():
            try:
                # If the port gets closed while we're here, this may raise
                line = self.ser.readline()
            except Exception:
                # Port likely closed during shutdown; exit thread quietly
                break

            if not line:
                continue

            t_now = time.monotonic()
            sample = self.parse_fn(line, t_now)
            if sample is not None:
                self.q.put(sample)

    def stop(self):
        self.stop_evt.set()
        try:
            self.ser.close()
        except Exception:
            pass


def parse_eit(line: bytes, t_now: float):
    """Parse EIT CSV lines; strip 'magnitudes:' prefix if present."""
    s = line.decode(errors='ignore').strip()
    if not s:
        return None
    if s.startswith("magnitudes:"):
        s = s[len("magnitudes:"):].strip()
    try:
        vals = [float(x) for x in s.split(',') if x.strip() != ""]
        if len(vals) == 0:
            return None
        return {"t": t_now, "readings": vals}
    except Exception:
        return None


def init_eit_port(ser):
    """Example: send 'y' to trigger streaming and discard first line."""
    time.sleep(0.2)
    ser.write(b"y")
    ser.flush()
    _ = ser.readline()


def sample_eit_now(q_eit):
    """Return the most recent EIT sample (no averaging)."""
    last = None
    while True:
        try:
            last = q_eit.get_nowait()
        except queue.Empty:
            break
    return last


# =====================
# Random + Calibration contact sampling
# =====================

def sample_random_contact(cfg, rng):
    """
    Sample a random contact configuration:

    Returns dict with:
        - shape_type
        - r_norm (center radius normalized by sensor radius)
        - theta (polar angle of center)
        - u_m, v_m (contact center in meters, in sensor plane)
        - yaw_rad (rotation of shape in 2D about its center)
    """

    sensor_R = cfg["SENSOR_RADIUS_M"]
    shape_max_radius_m = cfg["SHAPE_MAX_RADIUS_M"]
    shape_types = cfg["SHAPE_TYPES"]

    shape_type = rng.choice(shape_types)

    # Normalized max radius for this shape
    r_shape_frac = shape_max_radius_m[shape_type] / sensor_R

    r_min_frac = cfg.get("R_MIN_FRAC", 0.0)
    edge_margin = cfg.get("EDGE_MARGIN_FRAC", 0.0)
    # Max allowed center radius so shape remains fully inside disc:
    r_max_frac = 1.0 - r_shape_frac - edge_margin
    if r_max_frac <= 0.0:
        raise ValueError(f"Shape '{shape_type}' too large for sensor radius with given margins.")

    # Sample center radius & angle
    r_norm = rng.uniform(r_min_frac, r_max_frac)
    theta = rng.uniform(0.0, 2.0 * np.pi)

    # Convert to meters
    u_m = r_norm * sensor_R * np.cos(theta)
    v_m = r_norm * sensor_R * np.sin(theta)

    # Random yaw about sensor normal
    yaw_min = cfg.get("YAW_MIN", 0.0)
    yaw_max = cfg.get("YAW_MAX", 2.0 * np.pi)
    yaw_rad = rng.uniform(yaw_min, yaw_max)

    return {
        "shape_type": shape_type,
        "r_norm": float(r_norm),
        "theta": float(theta),
        "u_m": float(u_m),
        "v_m": float(v_m),
        "yaw_rad": float(yaw_rad),
    }


def sample_contact_with_calibration(cfg, rng, sample_idx):
    """
    Wrapper to handle the first few samples as calibration poses.

    For sample_idx = 0,1,2,3:
        - contact at sensor center (u=v=0)
        - yaw = 0, 90, 180, 270 degrees
        - shape_type = first element of SHAPE_TYPES
        - r_norm = 0, theta = 0 (center)

    For sample_idx >= 4:
        - use fully random sampling (sample_random_contact).
    """
    # Number of calibration samples to reserve at the beginning
    N_CALIB = 4

    if sample_idx < N_CALIB:
        shape_type = cfg["SHAPE_TYPES"][0]  # use first shape type for calibration
        yaw_list = [0.0, -0.5 * np.pi, -np.pi, 0.5 * np.pi]
        yaw_rad = yaw_list[sample_idx]

        sensor_R = cfg["SENSOR_RADIUS_M"]
        u_m = 0.0  # center of sensor
        v_m = 0.0
        r_norm = 0.0
        theta = 0.0  # undefined at center, but 0 is fine for bookkeeping

        return {
            "shape_type": shape_type,
            "r_norm": float(r_norm),
            "theta": float(theta),
            "u_m": float(u_m),
            "v_m": float(v_m),
            "yaw_rad": float(yaw_rad),
        }
    else:
        return sample_random_contact(cfg, rng)


# =====================
# Main
# =====================

def main():
    cfg = CONFIG
    Path(cfg["OUT_DIR"]).mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_base = Path(cfg["OUT_DIR"]) / f"{cfg['SESSION_TAG']}_{ts}"
    csv_path = run_base.with_suffix('.csv')

    rng = np.random.default_rng()

    # Base CSV header (without EIT columns)
    base_header_cols = [
        "t",
        "tcp_x", "tcp_y", "tcp_z",
        "tcp_rx", "tcp_ry", "tcp_rz",
        "shape_type",
        "contact_u_m", "contact_v_m",
        "contact_r_norm", "contact_theta",
        "yaw_rad",
        "depth_m",
        "t_eit",
    ]
    eit_channel_count = 0  # after first EIT sample

    # This will be updated after we know channel count
    full_header_cols = base_header_cols.copy()

    def write_row_csv(rec: dict):
        """Append one row to CSV immediately with stable header (including EIT columns)."""
        nonlocal full_header_cols, eit_channel_count
        if not cfg.get("CSV_LIVE", True):
            return

        # Ensure EIT keys exist if we know the channel count
        if eit_channel_count > 0:
            for i in range(eit_channel_count):
                rec.setdefault(f"eit_{i}", None)

        df1 = pd.DataFrame([rec], columns=full_header_cols)
        df1.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    parquet_buf = []
    parquet_chunk_id = 0

    def maybe_write_parquet_chunk(rec: dict):
        nonlocal parquet_buf, parquet_chunk_id
        N = int(cfg.get("PARQUET_EVERY_N", 0))
        if N <= 0:
            return
        parquet_buf.append(rec)
        if len(parquet_buf) >= N:
            chunk_dir = Path(cfg["OUT_DIR"]) / f"{cfg['SESSION_TAG']}_{ts}_chunks"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = chunk_dir / f"part_{parquet_chunk_id:05d}.parquet"
            pd.DataFrame(parquet_buf).to_parquet(chunk_path, index=False)
            parquet_buf.clear()
            parquet_chunk_id += 1

    # --- Robot connection ---
    robot = connect_robot(cfg["ROBOT_IP"])
    center_pose = cfg["SENSOR_CENTER_POSE"]
    movel(robot, center_pose, a=cfg["MOVE_A"], v=cfg["MOVE_V"])
    time.sleep(0.5)

    # --- EIT serial thread ---
    q_eit = queue.Queue(maxsize=4000)
    eit_reader = SerialLineReader(cfg["EIT_PORT"], cfg["EIT_BAUD"], parse_eit, q_eit, "eit", init_fn=init_eit_port)
    eit_reader.start()

    # --- Determine EIT channel count before motions ---
    t0 = time.time()
    while True:
        try:
            samp = q_eit.get(timeout=2.0)
            if samp and "readings" in samp and len(samp["readings"]) > 0:
                eit_channel_count = len(samp["readings"])
                break
        except queue.Empty:
            pass
        if time.time() - t0 > 5.0:
            print("[warn] No EIT data seen within 5s; proceeding without fixed EIT columns.")
            break

    if eit_channel_count > 0:
        # Extend full header ONCE with eit_0..eit_{N-1}
        full_header_cols = base_header_cols + [f"eit_{i}" for i in range(eit_channel_count)]
        print(f"[info] Detected {eit_channel_count} EIT channels.")

    rows = []
    press_depth = cfg["FIXED_PRESS_DEPTH_M"]

    try:
        for sample_idx in range(cfg["N_SAMPLES"]):
            # 1) Get contact (calibration for first 4, then random)
            contact = sample_contact_with_calibration(cfg, rng, sample_idx)

            # 2) Construct approach pose at safe height
            #    - Start from SENSOR_CENTER_POSE
            #    - Shift x,y by (u_m, v_m)
            #    - Apply yaw around sensor normal
            appr_pose = shift_xy(center_pose, contact["u_m"], contact["v_m"])
            appr_pose = apply_yaw_to_pose(appr_pose, contact["yaw_rad"], base_pose=center_pose)

            # Ensure we are at safe height (the z of center_pose)
            appr_pose = set_z(appr_pose, center_pose[2])

            # Move to approach (no contact)
            movel(robot, appr_pose, a=cfg["MOVE_A"], v=cfg["MOVE_V"])
            time.sleep(0.2)

            # 3) Press straight down to fixed depth
            press_pose = set_z(appr_pose, center_pose[2] - press_depth)
            movel(robot, press_pose, a=cfg["PRESS_A"], v=cfg["PRESS_V"])

            # Clear EIT queue to avoid stale samples
            while not q_eit.empty():
                try:
                    q_eit.get_nowait()
                except queue.Empty:
                    break

            # 4) Wait at contact and sample EIT
            time.sleep(cfg["SETTLE_S"])
            e_now = sample_eit_now(q_eit)

            # 5) Log one row
            tcp = press_pose
            rec = {
                "t": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                "tcp_x": tcp[0], "tcp_y": tcp[1], "tcp_z": tcp[2],
                "tcp_rx": tcp[3], "tcp_ry": tcp[4], "tcp_rz": tcp[5],
                "shape_type": contact["shape_type"],
                "contact_u_m": contact["u_m"],
                "contact_v_m": contact["v_m"],
                "contact_r_norm": contact["r_norm"],
                "contact_theta": contact["theta"],
                "yaw_rad": contact["yaw_rad"],
                "depth_m": press_depth,
                "t_eit": e_now["t"] if e_now is not None and "t" in e_now else None,
            }

            if eit_channel_count > 0:
                for i in range(eit_channel_count):
                    rec[f"eit_{i}"] = None
            if e_now is not None and "readings" in e_now:
                for i, v in enumerate(e_now["readings"]):
                    rec[f"eit_{i}"] = v

            rows.append(rec)
            write_row_csv(rec)
            maybe_write_parquet_chunk(rec)

            print(f"[info] Sample {sample_idx+1}/{cfg['N_SAMPLES']}: "
                  f"shape={contact['shape_type']}, r_norm={contact['r_norm']:.3f}, "
                  f"theta={contact['theta']:.2f}, yaw={contact['yaw_rad']:.2f}")

            # 6) Return to approach height (no contact) before next sample
            movel(robot, appr_pose, a=cfg["MOVE_A"], v=cfg["MOVE_V"])
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt — returning to center pose...")
    finally:
        try:
            movel(robot, cfg["SENSOR_CENTER_POSE"], a=cfg["MOVE_A"], v=cfg["MOVE_V"])
            robot.close()
        except Exception:
            pass

        eit_reader.stop()
        eit_reader.join(timeout=1.0)

        # Save final full Parquet snapshot
        if rows:
            df = pd.DataFrame(rows)
            pq_path = str(run_base.with_suffix('.parquet'))
            df.to_parquet(pq_path, index=False)
            print(f"Saved Parquet → {pq_path}")

            # Flush any remaining chunk buffer
            if cfg.get("PARQUET_EVERY_N", 0) > 0 and len(parquet_buf) > 0:
                chunk_dir = Path(cfg["OUT_DIR"]) / f"{cfg['SESSION_TAG']}_{ts}_chunks"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                chunk_path = chunk_dir / f"part_{parquet_chunk_id:05d}.parquet"
                pd.DataFrame(parquet_buf).to_parquet(chunk_path, index=False)
                print(f"Saved last chunk → {chunk_path}")


if __name__ == "__main__":
    main()
