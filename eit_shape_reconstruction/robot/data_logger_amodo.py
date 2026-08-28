#!/usr/bin/env python3
"""
UR5 + Amodo EIT data collection script
(Updated from Teensy → Amodo board)

Changes:
- Uses amodo_eit instead of serial
- Uses device.latest_frame instead of queue
- Logs clipping flag
- Everything else unchanged
"""

import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from pathlib import Path
import sys


import URBasic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AMODO_PY_DIR = PROJECT_ROOT / "ARIA_EITSYS_FW_copy" / "Python"

if str(AMODO_PY_DIR) not in sys.path:
    sys.path.insert(0, str(AMODO_PY_DIR))

import amodo_eit as eit

import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol

try:
    from scipy.spatial.transform import Rotation as R
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# =====================
# CONFIG
# =====================
CONFIG = {
    "ROBOT_IP": "169.254.76.5",
    "SENSOR_CENTER_POSE": [-0.5072, -0.0026, 0.16, 0.0, 3.142, 0.0],
    "SENSOR_RADIUS_M": 0.045,

    "SHAPE_MAX_RADIUS_M": {
        "L": 0.024,
        "T": 0.028,
        "edge": 0.025,
        "ring": 0.025,
        "double_circle": 0.024,
        "C": 0.024,
        "Z": 0.029,
        "+": 0.021,
    },

    "SHAPE_TYPES": ["Z"],

    "N_SAMPLES": 500,
    "EDGE_MARGIN_FRAC": 0.02,

    "YAW_MIN": 0.0,
    "YAW_MAX": 0.5 * np.pi,

    "FIXED_PRESS_DEPTH_M": 0.059,

    "MOVE_V": 0.5,
    "MOVE_A": 0.5,
    "PRESS_V": 0.02,
    "PRESS_A": 0.02,

    "SETTLE_S": 3.0,

    "OUT_DIR": "./data",
    "SESSION_TAG": "eit_amodo",

    # Amodo config
    "NUM_ELECTRODES": 16,
    "MESH_H0": 0.04,
    "STIM_FREQ_KHZ": 10,
    "PERIODS": 20,
    "TX_GAIN": 8,
    "RX_GAIN": 2,
}


# =====================
# ROBOT HELPERS
# =====================
def connect_robot(ip):
    model = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=ip, robotModel=model)
    robot.reset_error()
    time.sleep(0.5)
    return robot


def movel(robot, pose, a, v):
    robot.movel(pose, a=a, v=v)


def set_z(pose, z):
    p = pose.copy()
    p[2] = z
    return p


def shift_xy(pose, u, v):
    p = pose.copy()
    p[0] += u
    p[1] += v
    return p


def apply_yaw_to_pose(pose, yaw_rad, base_pose):
    if not HAVE_SCIPY:
        return pose

    rx, ry, rz = base_pose[3:6]
    angle = np.linalg.norm([rx, ry, rz])
    axis = np.array([rx, ry, rz]) / (angle + 1e-9)
    R0 = R.from_rotvec(axis * angle).as_matrix()

    Rz = R.from_euler("z", yaw_rad).as_matrix()
    R_new = Rz @ R0

    rvec = R.from_matrix(R_new).as_rotvec()

    p = pose.copy()
    p[3:] = rvec
    return p


# =====================
# RANDOM SAMPLING
# =====================
def sample_contact(cfg, rng):
    shape = rng.choice(cfg["SHAPE_TYPES"])
    R_sensor = cfg["SENSOR_RADIUS_M"]
    r_shape = cfg["SHAPE_MAX_RADIUS_M"][shape]

    r_max = 1.0 - (r_shape / R_sensor) - cfg["EDGE_MARGIN_FRAC"]

    r = rng.uniform(0, r_max)
    theta = rng.uniform(0, 2*np.pi)

    u = r * R_sensor * np.cos(theta)
    v = r * R_sensor * np.sin(theta)

    yaw = rng.uniform(cfg["YAW_MIN"], cfg["YAW_MAX"])

    return dict(
        shape_type=shape,
        u_m=u,
        v_m=v,
        r_norm=r,
        theta=theta,
        yaw_rad=yaw,
    )


# =====================
# MAIN
# =====================
def main():
    cfg = CONFIG
    Path(cfg["OUT_DIR"]).mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(cfg["OUT_DIR"]) / f"{cfg['SESSION_TAG']}_{ts}.csv"

    rng = np.random.default_rng()

    # ---------- ROBOT ----------
    robot = connect_robot(cfg["ROBOT_IP"])
    center_pose = cfg["SENSOR_CENTER_POSE"]
    movel(robot, center_pose, cfg["MOVE_A"], cfg["MOVE_V"])

    # ---------- AMODO ----------
    devices = eit.get_connected_devices()
    if not devices:
        raise RuntimeError("No Amodo devices found")

    device = devices[0]
    print(f"[info] Using Amodo: {device.port}")

    mesh_obj = mesh.create(cfg["NUM_ELECTRODES"], h0=cfg["MESH_H0"])
    protocol_obj = protocol.create(
        cfg["NUM_ELECTRODES"],
        dist_exc=1,
        step_meas=1,
        parser_meas="rotate_meas",
    )

    electrode_configurations = []
    for i_exc, exc_pair in enumerate(protocol_obj.ex_mat):
        A, B = exc_pair
        for M, N in protocol_obj.meas_mat[i_exc]:
            electrode_configurations.append(
                (A+1, B+1, M+1, N+1, cfg["TX_GAIN"], cfg["RX_GAIN"])
            )

    with device:
        device.set_stimulation_frequency(cfg["STIM_FREQ_KHZ"])
        device.set_electrode_configurations(electrode_configurations)
        device.set_num_periods_to_sample_per_measurement(cfg["PERIODS"])

        device.start_streaming(lambda x: None)

        while device.latest_frame is None:
            time.sleep(0.01)

        baseline, _ = device.latest_frame
        baseline = np.array(baseline)

        n_channels = len(baseline)
        print(f"[info] Channels: {n_channels}")

        header = [
            "t", "tcp_x","tcp_y","tcp_z",
            "tcp_rx","tcp_ry","tcp_rz",
            "shape_type",
            "contact_u_m","contact_v_m",
            "contact_r_norm","contact_theta",
            "yaw_rad","depth_m",
            "t_eit","clipping"
        ] + [f"eit_{i}" for i in range(n_channels)]

        rows = []

        for i in range(cfg["N_SAMPLES"]):

            contact = sample_contact(cfg, rng)

            pose = shift_xy(center_pose, contact["u_m"], contact["v_m"])
            pose = apply_yaw_to_pose(pose, contact["yaw_rad"], center_pose)

            movel(robot, pose, cfg["MOVE_A"], cfg["MOVE_V"])

            press_pose = set_z(pose, center_pose[2] - cfg["FIXED_PRESS_DEPTH_M"])
            movel(robot, press_pose, cfg["PRESS_A"], cfg["PRESS_V"])

            time.sleep(cfg["SETTLE_S"])

            frame, clipping = device.latest_frame
            frame = np.array(frame)

            rec = {
                "t": datetime.now().isoformat(),
                "tcp_x": press_pose[0],
                "tcp_y": press_pose[1],
                "tcp_z": press_pose[2],
                "tcp_rx": press_pose[3],
                "tcp_ry": press_pose[4],
                "tcp_rz": press_pose[5],
                "shape_type": contact["shape_type"],
                "contact_u_m": contact["u_m"],
                "contact_v_m": contact["v_m"],
                "contact_r_norm": contact["r_norm"],
                "contact_theta": contact["theta"],
                "yaw_rad": contact["yaw_rad"],
                "depth_m": cfg["FIXED_PRESS_DEPTH_M"],
                "t_eit": time.time(),
                "clipping": clipping,
            }

            for j in range(n_channels):
                rec[f"eit_{j}"] = frame[j]

            rows.append(rec)

            pd.DataFrame([rec], columns=header).to_csv(
                csv_path, mode="a", header=not csv_path.exists(), index=False
            )

            print(f"[{i+1}/{cfg['N_SAMPLES']}] shape={contact['shape_type']}")

            movel(robot, pose, cfg["MOVE_A"], cfg["MOVE_V"])

    print(f"[done] Saved → {csv_path}")


if __name__ == "__main__":
    main()