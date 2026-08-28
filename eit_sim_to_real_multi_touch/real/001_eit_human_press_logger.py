"""
Guided Human EIT Collection — Uniform Sampling of Pairs (9x9)
=============================================================
Flow per pair:
  1) SINGLE at P1  → capture
  2) SINGLE at P2  → capture
  3) DOUBLE P1+P2  → capture

- Grid: 9×9 over a 19 cm disk (editable).
- Uniform sampling: random permutation of valid cells inside disk; cells are paired in order
  with a minimum separation (configurable) to avoid nearly overlapping touches.
- Minimal operator interaction: just press ENTER to start the next capture.
  - You may also type 's' to skip a step, or 'q' to quit.
- Safe logging: each row is appended to CSV immediately with flush+fsync.

Usage
-----
pip install pyserial pandas numpy

python collect_eit_guided_pairs.py \
  --eit-port /dev/ttyACM0 --baud 115200 \
  --rows 9 --cols 9 --sensor-diam-cm 19 \
  --baseline-sec 10 --pre-hold-sec 1.0 --hold-sec 3.0 --avg-frames 10 \
  --min-sep 0.15 \
  --n-pairs 60 \
  --probe-id mix \
  --out-dir real_data --session-tag guided_R9C9
"""
import sys
import os
import time
import csv
import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np

try:
    import serial
except ImportError:
    serial = None
    print("ERROR: pyserial not installed. Run: pip install pyserial", file=sys.stderr)


@dataclass
class Cell:
    row: int
    col: int
    x_norm: float   # x in [-1,1] normalized by sensor radius
    y_norm: float   # y in [-1,1] normalized by sensor radius
    valid: bool


def build_circular_grid(rows: int, cols: int) -> Tuple[List[Cell], np.ndarray]:
    ys = np.linspace(-1.0, 1.0, rows, endpoint=True)   # row index -> y (top to bottom)
    xs = np.linspace(-1.0, 1.0, cols, endpoint=True)   # col index -> x (left to right)
    id_map = -np.ones((rows, cols), dtype=int)
    cells: List[Cell] = []
    for r in range(rows):
        for c in range(cols):
            x = float(xs[c])
            y = float(ys[rows-1-r])  # row 0 at top visually; y increases upward
            valid = (x*x + y*y) <= 1.0
            id_map[r, c] = 1 if valid else -1
            cells.append(Cell(row=r, col=c, x_norm=x, y_norm=y, valid=valid))
    return cells, id_map


def open_serial(eit_port: str, baud: int):
    ser = serial.Serial(eit_port, baudrate=baud, timeout=0.1)
    time.sleep(0.2)
    try:
        ser.write(b"y"); ser.flush()
    except Exception:
        pass
    try:
        _ = ser.readline()
    except Exception:
        pass
    return ser


def read_eit_line(ser) -> Optional[np.ndarray]:
    line = ser.readline()
    if not line:
        return None
    s = line.decode(errors='ignore').strip()
    if not s:
        return None
    if s.startswith("magnitudes:"):
        s = s[len("magnitudes:"):].strip()
    try:
        vals = [float(x) for x in s.split(',') if x.strip() != ""]
        if len(vals) == 0:
            return None
        return np.asarray(vals, dtype=np.float32)
    except Exception:
        return None


def capture_baseline(ser, seconds: float, avg_frames: int) -> np.ndarray:
    buf = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        acc = []
        for _ in range(max(1, avg_frames)):
            v = read_eit_line(ser)
            if v is not None:
                acc.append(v)
        if acc:
            buf.append(np.mean(np.stack(acc, axis=0), axis=0))
    if not buf:
        raise RuntimeError("No EIT data captured during baseline. Check serial / board.")
    v0 = np.mean(np.stack(buf, axis=0), axis=0).astype(np.float32)
    return v0


def capture_sample(ser, pre_hold_sec: float, hold_sec: float, avg_frames: int) -> Tuple[float, np.ndarray]:
    time.sleep(max(0.0, pre_hold_sec))
    t0 = time.time()
    buf = []
    while time.time() - t0 < hold_sec:
        acc = []
        for _ in range(max(1, avg_frames)):
            v = read_eit_line(ser)
            if v is not None:
                acc.append(v)
        if acc:
            buf.append(np.mean(np.stack(acc, axis=0), axis=0))
    if not buf:
        raise RuntimeError("No EIT frames captured during sample hold. Increase --hold-sec or --avg-frames.")
    v = np.mean(np.stack(buf, axis=0), axis=0).astype(np.float32)
    t_mid = t0 + hold_sec/2.0
    return t_mid, v


def pair_cells(valid_cells: List[Cell], n_pairs: int, min_sep: float, seed: int=42) -> List[Tuple[Cell, Cell]]:
    rng = np.random.default_rng(seed)
    cells = [c for c in valid_cells]
    rng.shuffle(cells)
    pairs = []
    i = 0
    while i < len(cells) - 1 and len(pairs) < n_pairs:
        c1 = cells[i]
        # find next with separation
        j = i + 1
        found = False
        while j < len(cells):
            c2 = cells[j]
            d = np.sqrt((c1.x_norm - c2.x_norm)**2 + (c1.y_norm - c2.y_norm)**2)
            if d >= min_sep:
                pairs.append((c1, c2))
                # remove chosen indices from pool in order (mark)
                cells.pop(j)
                cells.pop(i)
                found = True
                break
            j += 1
        if not found:
            i += 1
    return pairs


def prompt(msg: str) -> str:
    try:
        return input(msg).strip().lower()
    except EOFError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eit-port", type=str, required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--sensor-diam-cm", type=float, default=19.0)
    ap.add_argument("--baseline-sec", type=float, default=10.0)
    ap.add_argument("--pre-hold-sec", type=float, default=1.0)
    ap.add_argument("--hold-sec", type=float, default=3.0)
    ap.add_argument("--avg-frames", type=int, default=10)
    ap.add_argument("--min-sep", type=float, default=0.15, help="Minimum normalized distance between paired cells")
    ap.add_argument("--n-pairs", type=int, default=60)
    ap.add_argument("--probe-id", type=str, default="mix")
    ap.add_argument("--out-dir", type=str, default="real_data")
    ap.add_argument("--session-tag", type=str, default="guided_R9C9")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if serial is None:
        print("pyserial is required. Install with: pip install pyserial", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{args.session_tag}_{ts}"
    out_csv = out_dir / f"{session_id}.csv"
    out_v0 = out_dir / f"{session_id}_v0.npy"

    # Grid
    cells, id_map = build_circular_grid(args.rows, args.cols)
    valid_cells = [c for c in cells if c.valid]
    print(f"Valid cells inside disk: {len(valid_cells)} (of {args.rows*args.cols})")
    print("Orientation: row 0 at TOP, col 0 at LEFT.")

    # Serial open
    print(f"Opening EIT serial at {args.eit_port} @ {args.baud} ...")
    ser = open_serial(args.eit_port, args.baud)
    # Channel length
    print("Reading one frame to determine channel count...")
    first = None
    while first is None:
        first = read_eit_line(ser)
    M = first.size
    print(f"Detected {M} EIT channels.")

    # Baseline
    print(f"\n=== Baseline: keep hands off sensor for {args.baseline_sec:.1f} s ===")
    v0 = capture_baseline(ser, args.baseline_sec, args.avg_frames)
    np.save(out_v0, v0)
    print(f"Saved baseline → {out_v0}")

    # CSV header
    base_cols = [
        "session_id","datetime","mode","plan_idx","step","sample_id","probe_id",
        "grid_rows","grid_cols","sensor_diam_cm",
        "r1","c1","x1_norm","y1_norm",
        "r2","c2","x2_norm","y2_norm",
        "n_avg","v0_source","v0_len"
    ]
    eit_cols = [f"eit_{i}" for i in range(M)]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(base_cols + eit_cols)
        f.flush(); os.fsync(f.fileno())

    # Pair plan
    pairs = pair_cells(valid_cells, n_pairs=args.n_pairs, min_sep=args.min_sep, seed=args.seed)
    if len(pairs) == 0:
        print("No valid pairs could be formed. Try lowering --min-sep.")
        sys.exit(1)
    print(f"Planned {len(pairs)} pairs with min separation {args.min_sep}.")

    sample_id = 0
    try:
        for pidx, (c1, c2) in enumerate(pairs):
            print("\n===========================================")
            print(f"PAIR {pidx+1}/{len(pairs)}: P1=(r={c1.row},c={c1.col})  P2=(r={c2.row},c={c2.col})")
            print("===========================================")

            # Step A: SINGLE at P1
            ans = prompt(f"[A] SINGLE at P1 (r={c1.row}, c={c1.col}). Press ENTER to start, 's' to skip, 'q' to quit: ")
            if ans == 'q': break
            if ans != 's':
                _, v = capture_sample(ser, args.pre_hold_sec, args.hold_sec, args.avg_frames)
                row = [
                    session_id, datetime.now().isoformat(timespec="milliseconds"),
                    "single", pidx, "A", sample_id, args.probe_id,
                    args.rows, args.cols, args.sensor_diam_cm/100.0,
                    c1.row, c1.col, c1.x_norm, c1.y_norm,
                    "", "", "", "",
                    args.avg_frames, "session_v0", len(v0)
                ] + list(v.astype(float))
                with open(out_csv, "a", newline="") as f:
                    csv.writer(f).writerow(row); f.flush(); os.fsync(f.fileno())
                print(f"[saved] sample_id={sample_id} SINGLE P1")
                sample_id += 1

            # Step B: SINGLE at P2
            ans = prompt(f"[B] SINGLE at P2 (r={c2.row}, c={c2.col}). Press ENTER to start, 's' to skip, 'q' to quit: ")
            if ans == 'q': break
            if ans != 's':
                _, v = capture_sample(ser, args.pre_hold_sec, args.hold_sec, args.avg_frames)
                row = [
                    session_id, datetime.now().isoformat(timespec="milliseconds"),
                    "single", pidx, "B", sample_id, args.probe_id,
                    args.rows, args.cols, args.sensor_diam_cm/100.0,
                    c2.row, c2.col, c2.x_norm, c2.y_norm,
                    "", "", "", "",
                    args.avg_frames, "session_v0", len(v0)
                ] + list(v.astype(float))
                with open(out_csv, "a", newline="") as f:
                    csv.writer(f).writerow(row); f.flush(); os.fsync(f.fileno())
                print(f"[saved] sample_id={sample_id} SINGLE P2")
                sample_id += 1

            # Step C: DOUBLE at P1+P2
            ans = prompt(f"[C] DOUBLE at P1+P2. Place BOTH probes at (r={c1.row},c={c1.col}) and (r={c2.row},c={c2.col}). ENTER to start, 's' to skip, 'q' to quit: ")
            if ans == 'q': break
            if ans != 's':
                _, v = capture_sample(ser, args.pre_hold_sec, args.hold_sec, args.avg_frames)
                row = [
                    session_id, datetime.now().isoformat(timespec="milliseconds"),
                    "double", pidx, "C", sample_id, args.probe_id,
                    args.rows, args.cols, args.sensor_diam_cm/100.0,
                    c1.row, c1.col, c1.x_norm, c1.y_norm,
                    c2.row, c2.col, c2.x_norm, c2.y_norm,
                    args.avg_frames, "session_v0", len(v0)
                ] + list(v.astype(float))
                with open(out_csv, "a", newline="") as f:
                    csv.writer(f).writerow(row); f.flush(); os.fsync(f.fileno())
                print(f"[saved] sample_id={sample_id} DOUBLE P1+P2")
                sample_id += 1

    except KeyboardInterrupt:
        print("\n[info] Interrupted. Finalizing...")
    finally:
        try:
            ser.close()
        except Exception:
            pass

    print(f"\nDone. CSV saved → {out_csv}\nBaseline saved → {out_v0}")
    print("You can baseline-subtract using the saved v0.npy.")
if __name__ == "__main__":
    main()