#!/usr/bin/env python3
# (full code was provided in the chat; saving minimal wrapper that imports it is not possible offline)
# Rewriting full content now:

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def load_and_align(csv_path: Path, v0_path: Path):
    df = pd.read_csv(csv_path)
    eit_cols = [c for c in df.columns if c.startswith("eit_")]
    if not eit_cols:
        raise RuntimeError("No EIT columns found.")
    v0_full = np.load(v0_path).astype(np.float32)
    col_idx = np.array([int(c.split("_")[1]) for c in eit_cols], dtype=int)
    if v0_full.shape[0] <= col_idx.max():
        raise RuntimeError("Baseline length smaller than CSV EIT indices.")
    v0_unmasked = v0_full[col_idx]
    nonzero_mask = (v0_unmasked != 0.0)
    kept_cols = [c for c, keep in zip(eit_cols, nonzero_mask) if keep]
    v0 = v0_unmasked[nonzero_mask].astype(np.float32)
    df_eit = df[kept_cols].astype(np.float32).copy()
    df_meta = df.drop(columns=eit_cols, errors='ignore').copy()
    df = pd.concat([df_meta, df_eit], axis=1)
    return df, v0, kept_cols

def pair_by_plan_idx(df: pd.DataFrame):
    if not {"plan_idx","step","mode"}.issubset(df.columns):
        return []
    pairs = []
    for pidx, g in df.groupby("plan_idx"):
        try:
            ia = g.index[(g["step"]=="A") & (g["mode"]=="single")][0]
            ib = g.index[(g["step"]=="B") & (g["mode"]=="single")][0]
            ic = g.index[(g["step"]=="C") & (g["mode"]=="double")][0]
            pairs.append((ia, ib, ic))
        except Exception:
            continue
    return pairs

def pair_by_coords(df: pd.DataFrame):
    needed_cols = {"mode","r1","c1","r2","c2"}
    if not needed_cols.issubset(df.columns):
        return []
    singles = df[df["mode"]=="single"].copy()
    doubles = df[df["mode"]=="double"].copy()
    pairs = []
    used_single_idx = set()
    for ic, row in doubles.iterrows():
        r1, c1 = int(row["r1"]), int(row["c1"])
        r2, c2 = row.get("r2", np.nan), row.get("c2", np.nan)
        if pd.isna(r2) or pd.isna(c2):
            continue
        r2, c2 = int(r2), int(c2)
        matches = []
        for isg, sg in singles.iterrows():
            if isg in used_single_idx:
                continue
            rr, cc = int(sg["r1"]), int(sg["c1"])
            if (rr==r1 and cc==c1) or (rr==r2 and cc==c2):
                matches.append(isg)
        if len(matches) >= 2:
            ia, ib = matches[:2]
            pairs.append((ia, ib, ic))
            used_single_idx.add(ia); used_single_idx.add(ib)
    return pairs

def plot_vector_overlay(y_true, y_pred, title, out_png):
    plt.figure(figsize=(6,3.1))
    plt.plot(y_true)
    plt.plot(y_pred, alpha=0.9)
    plt.xlabel("Channel index")
    plt.ylabel("Voltage (a.u.)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_scatter(y_true, y_pred, title, out_png):
    plt.figure(figsize=(3.6,3.6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.7)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Measured double")
    plt.ylabel("Predicted sum")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--v0", required=True)
    ap.add_argument("--out", default="superposition_real")
    ap.add_argument("--max-plots", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    df, v0, kept_cols = load_and_align(Path(args.csv), Path(args.v0))
    print(f"[INFO] Using {len(kept_cols)} channels after zero-drop.")

    pairs = pair_by_plan_idx(df)
    if not pairs:
        pairs = pair_by_coords(df)
    if not pairs:
        raise RuntimeError("Could not pair singles and doubles.")

    print(f"[INFO] Found {len(pairs)} paired examples.")

    records = []
    for k, (ia, ib, ic) in enumerate(pairs):
        v_a = df.loc[ia, kept_cols].to_numpy(np.float32)
        v_b = df.loc[ib, kept_cols].to_numpy(np.float32)
        v_c = df.loc[ic, kept_cols].to_numpy(np.float32)

        v_pred = v_a + v_b - v0

        mae_abs = float(np.mean(np.abs(v_c - v_pred)))
        r2_abs  = r2_score(v_c, v_pred)

        dv_c   = v_c - v0
        dv_sum = (v_a - v0) + (v_b - v0)
        mae_dv = float(np.mean(np.abs(dv_c - dv_sum)))
        r2_dv  = r2_score(dv_c, dv_sum)

        records.append({
            "pair_idx": k, "idx_single_a": int(ia), "idx_single_b": int(ib), "idx_double": int(ic),
            "mae_abs": mae_abs, "r2_abs": r2_abs, "mae_dv": mae_dv, "r2_dv": r2_dv
        })

        if k < args.max_plots:
            ttl = f"Pair {k} | MAE={mae_abs:.4f}, R2={r2_abs:.3f}"
            plot_vector_overlay(v_c, v_pred, ttl, out_dir / f"pair_{k:02d}_overlay_abs.png")
            plot_scatter(v_c, v_pred, f"Abs domain (Pair {k})", out_dir / f"pair_{k:02d}_scatter_abs.png")

            ttl2 = f"ΔV Pair {k} | MAE={mae_dv:.4f}, R2={r2_dv:.3f}"
            plot_vector_overlay(dv_c, dv_sum, ttl2, out_dir / f"pair_{k:02d}_overlay_dv.png")
            plot_scatter(dv_c, dv_sum, f"ΔV domain (Pair {k})", out_dir / f"pair_{k:02d}_scatter_dv.png")

    df_sum = pd.DataFrame.from_records(records)
    df_sum.to_csv(out_dir / "summary.csv", index=False)
    print(f"Saved plots & summary to: {out_dir}/")

if __name__ == "__main__":
    main()