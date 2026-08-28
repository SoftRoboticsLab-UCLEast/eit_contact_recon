#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ----------------- helpers -----------------
NUM_SUFFIX = re.compile(r"(\d+)$")

def parse_idx(name: str) -> int | None:
    m = NUM_SUFFIX.search(name)
    return int(m.group(1)) if m else None

def find_eit_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        lc = c.lower()
        if lc == "eit_t":
            continue
        if lc.startswith("eit_") or lc.startswith("v"):
            if parse_idx(c) is not None:
                cols.append(c)
    if not cols:
        raise RuntimeError("No EIT columns found (after excluding 'eit_t').")
    return cols

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)

def drop_real_zero_cols(X_real: np.ndarray, names_real: list[str]) -> tuple[np.ndarray, list[str], list[int]]:
    keep = np.any(X_real != 0.0, axis=0)
    kept_names = [n for n, k in zip(names_real, keep) if k]
    kept_idx = [parse_idx(n) for n in kept_names]
    return coerce_numeric(pd.DataFrame(X_real, columns=names_real), kept_names), kept_names, kept_idx

def index_map(names: list[str]) -> dict[int, int]:
    """Map channel_index -> column_position in the matrix."""
    mp = {}
    for j, n in enumerate(names):
        idx = parse_idx(n)
        if idx is not None and idx not in mp:
            mp[idx] = j
    return mp

def center_per_dataset(Xs: np.ndarray, Xr: np.ndarray):
    ms = Xs.mean(axis=0, keepdims=True)
    mr = Xr.mean(axis=0, keepdims=True)
    return (Xs - ms), (Xr - mr), ms.squeeze(), mr.squeeze()

def plot_sample_traces(Xs: np.ndarray, Xr: np.ndarray, outdir: Path, n=10):
    rng = np.random.default_rng(0)
    isamp = rng.choice(len(Xs), size=min(n, len(Xs)), replace=False)
    jsamp = rng.choice(len(Xr), size=min(n, len(Xr)), replace=False)
    fig, ax = plt.subplots(figsize=(10,4))
    for i in isamp: ax.plot(Xs[i], alpha=0.35, lw=1.1, label="Sim" if i==isamp[0] else None, color="r")
    for j in jsamp: ax.plot(Xr[j], alpha=0.35, lw=1.1, label="Real" if j==jsamp[0] else None, color="b")
    ax.set_title("Sample EIT traces (aligned by common indices)")
    ax.set_xlabel("Aligned channel order"); ax.set_ylabel("Voltage (arb.)")
    ax.legend(); plt.tight_layout(); plt.savefig(outdir/"01_sample_traces.png", dpi=220); plt.close()

def plot_mean_std(Xs: np.ndarray, Xr: np.ndarray, outdir: Path):
    ms, ss = Xs.mean(0), Xs.std(0); mr, sr = Xr.mean(0), Xr.std(0)
    x = np.arange(Xs.shape[1]); fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(x, ms, lw=1.6, label="Sim mean"); ax.fill_between(x, ms-ss, ms+ss, alpha=0.2)
    ax.plot(x, mr, lw=1.6, label="Real mean"); ax.fill_between(x, mr-sr, mr+sr, alpha=0.2)
    ax.set_title("Channel-wise mean ± std (aligned)"); ax.set_xlabel("Aligned channel order"); ax.set_ylabel("Voltage (arb.)")
    ax.legend(); plt.tight_layout(); plt.savefig(outdir/"02_mean_std.png", dpi=220); plt.close()

def plot_hist_abs(Xs: np.ndarray, Xr: np.ndarray, outdir: Path, bins=60):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(np.abs(Xs).ravel(), bins=bins, density=True, alpha=0.6, label="Sim |v|")
    ax.hist(np.abs(Xr).ravel(), bins=bins, density=True, alpha=0.6, label="Real |v|")
    ax.set_title("|voltage| distribution (aligned channels)"); ax.set_xlabel("|v|"); ax.set_ylabel("density"); ax.legend()
    plt.tight_layout(); plt.savefig(outdir/"03_abs_hist.png", dpi=220); plt.close()

def plot_pca2(Xs: np.ndarray, Xr: np.ndarray, outdir: Path):
    n = min(len(Xs), len(Xr), 6000)
    Xs_sub, Xr_sub = Xs[:n], Xr[:n]
    X = np.vstack([Xs_sub, Xr_sub])
    p = PCA(n_components=2, random_state=0)
    Z = p.fit_transform(X); Zs = Z[:len(Xs_sub)]; Zr = Z[len(Xs_sub):]
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(Zs[:,0], Zs[:,1], s=8, alpha=0.5, label="Sim")
    ax.scatter(Zr[:,0], Zr[:,1], s=8, alpha=0.5, label="Real")
    ax.set_title("PCA (2D) of EIT vectors (aligned by common indices)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
    plt.tight_layout(); plt.savefig(outdir/"04_pca_scatter.png", dpi=220); plt.close()

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-csv", required=True, help="Sim CSV (usually DELTAs).")
    ap.add_argument("--real-csv", required=True, help="Real CSV (may include 256 columns + 'eit_t').")
    ap.add_argument("--out", default="compare_sim_real_plots")
    ap.add_argument("--no-center", action="store_true", help="Disable per-dataset mean centering.")
    ap.add_argument("--min-common", type=int, default=150, help="Minimum #common channels required to proceed.")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df_sim  = pd.read_csv(args.sim_csv)
    df_real = pd.read_csv(args.real_csv)

    # Find EIT columns (ignore eit_t)
    sim_cols_all  = find_eit_cols(df_sim)
    real_cols_all = find_eit_cols(df_real)

    # Coerce numeric
    X_real_full = coerce_numeric(df_real, real_cols_all)

    # Drop exact-zero columns from REAL; obtain their numeric indices
    X_real_nz, real_kept_names, real_kept_idx = drop_real_zero_cols(X_real_full, real_cols_all)
    print(f"[INFO] Real: {len(real_cols_all)} cols → dropped {len(real_cols_all)-len(real_kept_names)} zeros → kept {len(real_kept_names)}")

    # Build index maps for SIM and REAL-kept
    sim_idx_map  = index_map(sim_cols_all)
    real_idx_map = index_map(real_kept_names)  # positions in X_real_nz

    # Intersection of numeric indices
    common_idx = sorted(set(sim_idx_map.keys()).intersection(real_idx_map.keys()))
    if len(common_idx) < args.min_common:
        # Report missing details and stop
        missing_in_sim  = sorted(set(real_idx_map.keys()) - set(sim_idx_map.keys()))
        missing_in_real = sorted(set(sim_idx_map.keys())  - set(real_idx_map.keys()))
        with open(outdir/"alignment_report.txt", "w") as fh:
            fh.write(f"Common indices: {len(common_idx)}\n")
            fh.write(f"Missing in SIM (present in REAL): {missing_in_sim[:50]}{' ...' if len(missing_in_sim)>50 else ''}\n")
            fh.write(f"Missing in REAL (present in SIM): {missing_in_real[:50]}{' ...' if len(missing_in_real)>50 else ''}\n")
        raise RuntimeError(f"Only {len(common_idx)} common channel indices found (min required {args.min_common}). See alignment_report.txt.")

    # Align matrices by the intersection (same order for both)
    sim_cols_sel  = [sim_idx_map[i]  for i in common_idx]
    real_cols_sel = [real_idx_map[i] for i in common_idx]
    X_sim_aligned  = coerce_numeric(df_sim, [sim_cols_all[j] for j in sim_cols_sel])
    X_real_aligned = X_real_nz[:, real_cols_sel]

    print(f"[INFO] Aligned channels by intersection: {len(common_idx)}")
    with open(outdir/"alignment_report.txt", "w") as fh:
        fh.write(f"Aligned common indices ({len(common_idx)}): {common_idx[:50]}{' ...' if len(common_idx)>50 else ''}\n")

    # Optional centering
    if args.no_center:
        Xs, Xr = X_sim_aligned.copy(), X_real_aligned.copy()
    else:
        Xs, Xr, ms, mr = center_per_dataset(X_sim_aligned, X_real_aligned)
        print("[INFO] Centered each dataset by its own mean vector.")

    # Quick stats
    def pct(lbl, A):
        q = np.percentile(np.abs(A), [1,50,99])
        print(f"  {lbl}: |v| pct [1,50,99] = {q}")
    print("[STATS] after alignment" + ("" if args.no_center else " + centering"))
    pct("SIM", Xs); pct("REAL", Xr)

    # Plots
    plot_sample_traces(Xs, Xr, outdir, n=10)
    plot_mean_std(Xs, Xr, outdir)
    plot_hist_abs(Xs, Xr, outdir)
    plot_pca2(Xs, Xr, outdir)

    # Save a couple of previews
    pd.DataFrame({"sim_trace0": Xs[0]}).to_csv(outdir/"preview_sim_trace0.csv", index=False)
    pd.DataFrame({"real_trace0": Xr[0]}).to_csv(outdir/"preview_real_trace0.csv", index=False)

    print(f"✅ Saved plots + alignment_report → {outdir}/")

if __name__ == "__main__":
    main()
