#!/usr/bin/env python3
"""
Collect baseline (no-touch) EIT readings and save the average vector.

Make sure:
1. No object touches the sensor.
2. The Teensy EIT firmware is running and streaming.
3. The correct serial port and baudrate are set below.
"""

import time
import numpy as np
import serial

# -------------------------------------------------------
# USER CONFIG
# -------------------------------------------------------
EIT_PORT = "/dev/ttyACM0"      # update if needed
EIT_BAUD = 115200
N_SAMPLES = 50                 # number of frames to average
WAIT_BETWEEN_READS = 0.02      # seconds

OUT_NPY = "eit_baseline.npy"
OUT_CSV = "eit_baseline.csv"

# -------------------------------------------------------
# Helper: parse one EIT line
# -------------------------------------------------------
def parse_eit_line(line: bytes):
    """Parse EIT CSV line (strip prefix if present)."""
    s = line.decode(errors="ignore").strip()
    if not s:
        return None
    if s.startswith("magnitudes:"):
        s = s[len("magnitudes:"):].strip()

    try:
        vals = [float(x) for x in s.split(",") if x.strip() != ""]
        return vals if len(vals) > 0 else None
    except Exception:
        return None

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    print("Opening serial port:", EIT_PORT)
    ser = serial.Serial(EIT_PORT, EIT_BAUD, timeout=0.1)
    time.sleep(0.3)

    # Trigger streaming if your firmware requires it
    print("Triggering EIT board...")
    try:
        ser.write(b"y")
        ser.flush()
    except:
        pass

    # Discard first line
    _ = ser.readline()

    readings = []
    print(f"Collecting {N_SAMPLES} baseline samples...")

    while len(readings) < N_SAMPLES:
        line = ser.readline()
        vals = parse_eit_line(line)
        if vals is None:
            continue
        readings.append(vals)

        if len(readings) == 1:
            print(f"Detected {len(vals)} channels.")
        print(f"  Sample {len(readings)}/{N_SAMPLES}", end="\r")

        time.sleep(WAIT_BETWEEN_READS)

    ser.close()
    print("\nDone collecting.")

    # Convert to numpy
    arr = np.array(readings)   # shape (N_SAMPLES, N_channels)
    baseline = arr.mean(axis=0)

    print("Baseline shape:", baseline.shape)
    print("Saving to:")
    print(" ", OUT_NPY)
    print(" ", OUT_CSV)

    # Save
    np.save(OUT_NPY, baseline)
    np.savetxt(OUT_CSV, baseline, delimiter=",")

    print("Finished.")


if __name__ == "__main__":
    main()
