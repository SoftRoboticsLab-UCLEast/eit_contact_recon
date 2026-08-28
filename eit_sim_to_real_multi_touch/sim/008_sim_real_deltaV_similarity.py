#!/usr/bin/env python3
"""
Compare ΔV (delta voltages) between SIM and REAL for the SAME double-touch locations.

Key robustness:
- REAL: ignore `eit_t`, drop exact-zero channels, keep numeric indices.
- SIM: align by intersection of numeric indices with REAL (no strict equality).
- Baseline (REAL) aligned to the same indices by header suffix if available, else by length.
- Matches rows by (x1,y1,x2,y2) rounded keys (uses *_norm if present else raw).
- Plots ΔV_sim vs ΔV_real and their difference; writes corr & L2 per sample.

Usage (SIM already stores deltas):
python 011_compare_delta_for_matching_samples.py \
  --sim-csv sim/sim_from_training_locs.csv \
  --real-csv training_dataset/training_dataset.csv \
  --real-baseline-csv data/eit_windows/eit_baseline.csv \
  --out compare_delta_pairs \
  --n-samples 8 \
  --sim-is-delta
"""

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM_SUFFIX = re.compile(r"(\d+)$")

def parse_idx(name: str):
    m = NUM_SUFFIX.search(name)
    return int(m.group(1)) if m else None

def find_eit_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        lc = c.lower()
        if lc == "eit_t":  # ignore time column if present
            continue
        if lc.startswith("eit_") or lc.startswith("v"):
            if parse_idx(c) is not None:
                cols.append(c)
    if not cols:
        raise RuntimeError("No EIT columns found (expected eit_<int> or v<int>).")
    # sort by numeric suffix
    cols = sorted(cols, key=lambda n: parse_idx(n))
    return cols

def to_numeric(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)

def pick_label_cols(df: pd.DataFrame):
    norm = ["R1_x_norm","R1_y_norm","R2_x_norm","R2_y_norm"]
    plain = ["x1","y1","x2","y2"]
    if all(c in df.columns for c in norm):
        return norm
    if all(c in df.columns for c in plain):
        return plain
    raise RuntimeError("Could not find any of (x1_norm,y1_norm,x2_norm,y2_norm) or (x1,y1,x2,y2).")

def make_key_series(df: pd.DataFrame, cols: list[str], prec: int = 4) -> pd.Series:
    r = df[cols].astype(float).round(prec)
    return r[cols[0]].astype(str) + "|" + r[cols[1]].astype(str) + "|" + r[cols[2]].astype(str) + "|" + r[cols[3]].astype(str)

def load_baseline_csv(path: Path) -> tuple[np.ndarray, list[int]]:
    """
    Load baseline CSV (one row). Returns (values, numeric_indices_or_empty).
    If headers have numeric suffixes, we return those indices; else empty list.
    """
    try:
        df = pd.read_csv(path, header=0)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        vals = df.to_numpy(np.float32).flatten()
        idxs = [parse_idx(c) for c in df.columns]
        if all(i is not None for i in idxs):
            return vals, idxs
        return vals, []
    except Exception:
        df = pd.read_csv(path, header=None)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return df.to_numpy(np.float32).flatten(), []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-csv", required=True)
    ap.add_argument("--real-csv", required=True)
    ap.add_argument("--real-baseline-csv", required=True, help="Baseline CSV for REAL (one row).")
    ap.add_argument("--sim-is-delta", action="store_true", help="SIM CSV already stores (v_touch - v0).")
    ap.add_argument("--sim-baseline-csv", default=None, help="If SIM is raw, pass a baseline CSV to compute deltas.")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--out", default="compare_delta_pairs")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_sim  = pd.read_csv(args.sim_csv)
    df_real = pd.read_csv(args.real_csv)

    # Labels and keys
    sim_lbl = pick_label_cols(df_sim)
    real_lbl = pick_label_cols(df_real)
    df_sim  = df_sim.copy();  df_sim["__key__"]  = make_key_series(df_sim,  sim_lbl).values
    df_real = df_real.copy(); df_real["__key__"] = make_key_series(df_real, real_lbl).values

    # EIT columns
    sim_cols_all  = find_eit_cols(df_sim)
    real_cols_all = find_eit_cols(df_real)

    # REAL: numeric + zero-drop
    X_real_all = to_numeric(df_real, real_cols_all)
    keep_mask = np.any(X_real_all != 0.0, axis=0)
    real_cols_kept = [c for c,k in zip(real_cols_all, keep_mask) if k]
    X_real_kept = X_real_all[:, keep_mask]
    real_idx_kept = [parse_idx(c) for c in real_cols_kept]
    print(f"[INFO] REAL: {len(real_cols_all)} → kept {len(real_cols_kept)} after zero-drop.")

    # SIM: align by intersection of indices
    sim_idx_all = [parse_idx(c) for c in sim_cols_all]
    common_idx = sorted(set(real_idx_kept).intersection(sim_idx_all))
    if len(common_idx) < 100:
        missing_in_sim  = sorted(set(real_idx_kept) - set(sim_idx_all))
        missing_in_real = sorted(set(sim_idx_all) - set(real_idx_kept))
        with open(out_dir/"alignment_report.txt","w") as fh:
            fh.write(f"common indices = {len(common_idx)}\n")
            fh.write(f"missing in SIM (first 50): {missing_in_sim[:50]}\n")
            fh.write(f"missing in REAL (first 50): {missing_in_real[:50]}\n")
        raise RuntimeError(f"Too few common channels ({len(common_idx)}). See alignment_report.txt.")

    # Build column lists in the SAME order for both matrices
    real_cols_sel = []
    sim_cols_sel  = []
    sim_name_map_eit = {parse_idx(c): c for c in sim_cols_all if c.startswith("eit_")}
    sim_name_map_v   = {parse_idx(c): c for c in sim_cols_all if c.startswith("v")}
    real_name_map    = {parse_idx(c): c for c in real_cols_kept}

    for i in common_idx:
        real_cols_sel.append(real_name_map[i])
        # prefer same 'eit_' naming; fallback to 'v'
        if i in sim_name_map_eit:
            sim_cols_sel.append(sim_name_map_eit[i])
        elif i in sim_name_map_v:
            sim_cols_sel.append(sim_name_map_v[i])
        else:
            # shouldn't happen because we used intersection
            pass

    X_real_aligned = to_numeric(df_real, real_cols_sel)  # still raw (not delta)
    X_sim_aligned  = to_numeric(df_sim,  sim_cols_sel)   # raw or delta depending on flag

    # REAL baseline (align to common_idx order)
    v0_vals, v0_idx = load_baseline_csv(Path(args.real_baseline_csv))
    if v0_idx:  # we have headers we can map by index
        idx2pos = {idx: j for j, idx in enumerate(v0_idx)}
        try:
            v0_real = np.array([v0_vals[idx2pos[i]] if i in idx2pos else 0.0 for i in common_idx], dtype=np.float32)
        except Exception:
            raise RuntimeError("REAL baseline headers did not cover all common indices.")
    else:
        # fallback: assume baseline already in the same order/length as aligned real
        if len(v0_vals) != X_real_aligned.shape[1]:
            raise RuntimeError(f"REAL baseline length {len(v0_vals)} does not match aligned channels {X_real_aligned.shape[1]}.")
        v0_real = v0_vals.astype(np.float32)

    # ΔV for REAL
    dV_real = X_real_aligned - v0_real[None, :]

    # ΔV for SIM
    if args.sim_is_delta:
        dV_sim = X_sim_aligned
    else:
        if args.sim_baseline_csv is None:
            raise RuntimeError("SIM looks raw. Provide --sim-baseline-csv to compute deltas.")
        v0_sim_vals, v0_sim_idx = load_baseline_csv(Path(args.sim_baseline_csv))
        # align SIM baseline to common_idx
        if v0_sim_idx:
            idx2pos = {idx: j for j, idx in enumerate(v0_sim_idx)}
            v0_sim = np.array([v0_sim_vals[idx2pos[i]] if i in idx2pos else 0.0 for i in common_idx], dtype=np.float32)
        else:
            if len(v0_sim_vals) != X_sim_aligned.shape[1]:
                raise RuntimeError("SIM baseline length does not match aligned SIM columns.")
            v0_sim = v0_sim_vals.astype(np.float32)
        dV_sim = X_sim_aligned - v0_sim[None, :]

    # Match keys (touch locations)
    common_keys = pd.Index(df_sim["__key__"]).intersection(df_real["__key__"])
    if len(common_keys) == 0:
        raise RuntimeError("No matching touch-location keys between SIM and REAL.")
    print(f"[INFO] Matched rows available: {len(common_keys)}")

    rng = np.random.default_rng(0)
    K = min(args.n_samples, len(common_keys))
    sel_keys = list(rng.choice(common_keys.to_numpy(), size=K, replace=False))

    with open(out_dir/"pair_metrics.txt", "w") as fh:
        for k_i, key in enumerate(sel_keys):
            si = df_sim.index[df_sim["__key__"] == key][0]
            ri = df_real.index[df_real["__key__"] == key][0]

            dv_s = dV_sim[si]
            dv_r = dV_real[ri]

            # metrics
            if np.std(dv_s) > 1e-9 and np.std(dv_r) > 1e-9:
                corr = float(np.corrcoef(dv_s, dv_r)[0,1])
            else:
                corr = float("nan")
            l2 = float(np.linalg.norm(dv_r - dv_s))
            fh.write(f"key={key}  corr={corr:.4f}  L2={l2:.6f}\n")

            # plot
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axes[0].plot(dv_s, lw=1.2, label="SIM ΔV")
            axes[0].plot(dv_r, lw=1.2, label="REAL ΔV", alpha=0.85)
            axes[0].set_title(f"ΔV comparison (sample {k_i+1}/{K})  corr={corr:.3f}  L2={l2:.4f}")
            axes[0].set_ylabel("ΔV")
            axes[0].legend()

            axes[1].plot(dv_r - dv_s, lw=1.0, label="REAL − SIM")
            axes[1].set_xlabel("Aligned channel index (common)")
            axes[1].set_ylabel("difference")
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(out_dir / f"pair_{k_i+1:02d}.png", dpi=220)
            plt.close(fig)

    print(f"✅ Saved {K} pair plots + metrics → {out_dir}/")
    print("   - pair_*.png, pair_metrics.txt, and alignment_report.txt if intersection was small.")

if __name__ == "__main__":
    main()
