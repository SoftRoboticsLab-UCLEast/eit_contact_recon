#!/usr/bin/env python3
"""
Residual Sim→Real mapper for EIT ΔV vectors.

• Loads SIM and REAL CSVs, finds EIT columns (eit_<int> or v<int>, ignores 'eit_t').
• REAL: drops exact-zero columns.
• Aligns SIM/REAL by intersection of channel indices.
• Builds ΔV:
    - If --*_is_delta, uses CSV values as ΔV.
    - Else subtracts a provided baseline CSV (one row, headers optional).
• Pairs rows by identical touch locations (x1,y1,x2,y2 or *_norm rounded to 4 decimals).
• Trains y = x + MLP([x, c?]) with L1 + (1 - corr) + optional physics penalty.
• Saves model, metrics, and sample comparison plots.
• Optionally writes a CSV with SIM rows but EIT columns replaced by translated ΔV.

Author: you + ChatGPT
"""

from __future__ import annotations
import argparse, re, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Tuple

# ------------- utilities ----------------
NUM_SUFFIX = re.compile(r"(\d+)$")

def parse_idx(name: str) -> int | None:
    m = NUM_SUFFIX.search(name)
    return int(m.group(1)) if m else None

def find_eit_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        lc = c.lower()
        if lc == "eit_t":  # ignore timestamp/aux
            continue
        if lc.startswith("eit_") or lc.startswith("v"):
            if parse_idx(c) is not None:
                cols.append(c)
    if not cols:
        raise RuntimeError("No EIT columns found (eit_<int> or v<int>).")
    return sorted(cols, key=lambda n: parse_idx(n))

def to_numeric(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)

def drop_real_zero_cols(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str], list[int]]:
    keep = np.any(X != 0.0, axis=0)
    kept_names = [n for n,k in zip(names, keep) if k]
    kept_idx = [parse_idx(n) for n in kept_names]
    return X[:, keep], kept_names, kept_idx

def load_baseline_csv(path: Path) -> tuple[np.ndarray, list[int]]:
    """
    Returns (values, numeric_indices). If headers don’t end with digits,
    numeric_indices is empty and we’ll align by length.
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
        df = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return df.to_numpy(np.float32).flatten(), []

def choose_label_cols(df: pd.DataFrame) -> list[str]:
    cand1 = ["R1_x_norm","R1_y_norm","R2_x_norm","R2_y_norm"]
    cand2 = ["x1","y1","x2","y2"]
    if all(c in df.columns for c in cand1): return cand1
    if all(c in df.columns for c in cand2): return cand2
    raise RuntimeError("Touch label columns not found (need x1,y1,x2,y2 or *_norm).")

def make_keys(df: pd.DataFrame, cols: list[str], prec: int=4) -> pd.Series:
    r = df[cols].astype(float).round(prec)
    return r[cols[0]].astype(str)+"|"+r[cols[1]].astype(str)+"|"+r[cols[2]].astype(str)+"|"+r[cols[3]].astype(str)

def align_by_intersection(sim_cols: list[str], real_cols: list[str]) -> tuple[list[str], list[str], list[int]]:
    sim_idx = [parse_idx(c) for c in sim_cols]
    real_idx = [parse_idx(c) for c in real_cols]
    common = sorted(set(sim_idx).intersection(real_idx))
    if len(common) < 100:
        raise RuntimeError(f"Too few common channels ({len(common)}).")
    # Name maps
    sim_map_eit = {parse_idx(c): c for c in sim_cols if c.startswith("eit_")}
    sim_map_v   = {parse_idx(c): c for c in sim_cols if c.startswith("v")}
    real_map    = {parse_idx(c): c for c in real_cols}
    sim_sel, real_sel = [], []
    for i in common:
        real_sel.append(real_map[i])
        if i in sim_map_eit: sim_sel.append(sim_map_eit[i])
        else: sim_sel.append(sim_map_v[i])
    return sim_sel, real_sel, common

def corr_loss(y: torch.Tensor, t: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    1 - Pearson correlation per-sample (across channels), then mean over batch.
    y,t: [B,C]
    """
    y0 = y - y.mean(dim=1, keepdim=True)
    t0 = t - t.mean(dim=1, keepdim=True)
    num = (y0*t0).sum(dim=1)
    den = torch.sqrt((y0*y0).sum(dim=1)* (t0*t0).sum(dim=1) + eps)
    r = num / (den + eps)
    return (1.0 - r).mean()

# ------------- dataset ----------------
class PairDataset(Dataset):
    def __init__(self, Xs: np.ndarray, Xr: np.ndarray, keys: np.ndarray):
        self.Xs = torch.from_numpy(Xs)  # ΔV_sim aligned
        self.Xr = torch.from_numpy(Xr)  # ΔV_real aligned
        self.keys = list(keys)

    def __len__(self): return len(self.keys)

    def __getitem__(self, i):
        return self.Xs[i], self.Xr[i]

# ------------- model ------------------
class ResidualSim2Real(nn.Module):
    def __init__(self, c_in: int, ctx_dim: int=0, width: int=512, depth: int=5):
        super().__init__()
        layers = []
        d = c_in + ctx_dim
        for k in range(depth):
            layers += [nn.Linear(d if k==0 else width, width), nn.SiLU()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, c_in)
        self.ctx_dim = ctx_dim
        self.c_in = c_in

    def forward(self, x, ctx=None):
        if ctx is not None:
            h = torch.cat([x, ctx], dim=1)
        else:
            h = x
        r = self.head(self.backbone(h))
        return x + r  # residual add

# ------------- training & eval ----------
def train_epoch(model, dl, opt, l1_w=1.0, corr_w=0.2, A=None, phys_w=0.0, device="cuda"):
    model.train(); l1m = nn.L1Loss()
    tot=0.0
    for xs, xr in dl:
        xs = xs.to(device); xr = xr.to(device)
        y = model(xs)
        loss = l1_w*l1m(y, xr) + corr_w*corr_loss(y, xr)
        if A is not None and phys_w>0:
            # simple physics: ||A y||^2
            Ay = torch.matmul(A, y.T).T   # [B,M]
            loss = loss + phys_w*(Ay.pow(2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*xs.size(0)
    return tot/len(dl.dataset)

@torch.no_grad()
def eval_epoch(model, dl, device="cuda"):
    model.eval()
    mae_list, corr_list = [], []
    for xs, xr in dl:
        xs = xs.to(device); xr = xr.to(device)
        y = model(xs)
        # MAE over channels
        mae = (y - xr).abs().mean(dim=1).cpu().numpy()
        # corr per sample
        y0 = y - y.mean(1, keepdim=True); r0 = xr - xr.mean(1, keepdim=True)
        num = (y0*r0).sum(1).cpu().numpy()
        den = torch.sqrt((y0*y0).sum(1)* (r0*r0).sum(1)+1e-6).cpu().numpy()
        corr = num/(den+1e-6)
        mae_list.append(mae); corr_list.append(corr)
    mae = np.concatenate(mae_list).mean()
    corr = np.concatenate(corr_list).mean()
    return mae, corr

def plot_sample_pairs(Xs, Xr, Yhat, outdir: Path, n_show=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs), size=min(n_show, len(Xs)), replace=False)
    outdir.mkdir(parents=True, exist_ok=True)
    for k,i in enumerate(idx):
        xs = Xs[i]; xr = Xr[i]; y = Yhat[i]
        fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
        ax[0].plot(xs, lw=1.0, label="SIM ΔV"); ax[0].plot(xr, lw=1.0, label="REAL ΔV")
        ax[0].plot(y,  lw=1.0, label="F(SIM) ΔV", alpha=0.9)
        ax[0].legend(); ax[0].set_ylabel("ΔV")
        ax[0].set_title(f"Pair {k+1}")
        ax[1].plot(xr - xs, lw=1.0, label="REAL - SIM", alpha=0.8)
        ax[1].plot(xr - y,  lw=1.0, label="REAL - F(SIM)", alpha=0.8)
        ax[1].legend(); ax[1].set_xlabel("channel"); ax[1].set_ylabel("Δ")
        plt.tight_layout(); plt.savefig(outdir/f"pair_{k+1:02d}.png", dpi=220); plt.close(fig)

# ------------- main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-csv", required=True)
    ap.add_argument("--real-csv", required=True)
    # ΔV construction
    ap.add_argument("--sim-is-delta", action="store_true",
                    help="SIM CSV already stores (v_touch - v0).")
    ap.add_argument("--real-is-delta", action="store_true",
                    help="REAL CSV already stores (v_touch - v0).")
    ap.add_argument("--sim-baseline-csv", default=None,
                    help="If SIM is raw, provide baseline CSV to compute deltas.")
    ap.add_argument("--real-baseline-csv", default=None,
                    help="If REAL is raw, provide baseline CSV to compute deltas.")
    # training
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1-w", type=float, default=1.0)
    ap.add_argument("--corr-w", type=float, default=0.2)
    ap.add_argument("--phys-A-npy", default=None, help="Optional constraint matrix A.npy; applies ||A y||^2.")
    ap.add_argument("--phys-w", type=float, default=0.0)
    # splits & outputs
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--out", default="sim2real_residual_out")
    ap.add_argument("--export-translated-sim", action="store_true",
                    help="Write a CSV mirroring the SIM CSV but with EIT columns replaced by translated ΔV.")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    dfS = pd.read_csv(args.sim_csv)
    dfR = pd.read_csv(args.real_csv)

    # Touch-location keys to pair rows
    lblS = choose_label_cols(dfS)
    lblR = choose_label_cols(dfR)
    keyS = make_keys(dfS, lblS).values
    keyR = make_keys(dfR, lblR).values
    dfS = dfS.copy(); dfS["__key__"] = keyS
    dfR = dfR.copy(); dfR["__key__"] = keyR

    # EIT columns
    colsS_all = find_eit_cols(dfS)
    colsR_all = find_eit_cols(dfR)

    # REAL zero-drop → align lists
    XR_all = to_numeric(dfR, colsR_all)
    XR_nz, colsR_kept, idxR_kept = drop_real_zero_cols(XR_all, colsR_all)

    # SIM align to REAL-kept by intersection of indices
    sim_sel, real_sel, common_idx = align_by_intersection(colsS_all, colsR_kept)
    XS_aligned = to_numeric(dfS, sim_sel)
    XR_aligned = to_numeric(dfR, real_sel)  # still raw; will become ΔV depending on flags

    # Build ΔV for SIM
    if args.sim_is_delta:
        dVS = XS_aligned
    else:
        if not args.sim_baseline_csv:
            raise RuntimeError("SIM is raw → provide --sim-baseline-csv.")
        v0S_vals, v0S_idx = load_baseline_csv(Path(args.sim_baseline_csv))
        # align baseline to common_idx
        if v0S_idx:
            pos = {i:j for j,i in enumerate(v0S_idx)}
            v0S = np.array([v0S_vals[pos[i]] if i in pos else 0.0 for i in common_idx], dtype=np.float32)
        else:
            if len(v0S_vals) != XS_aligned.shape[1]:
                raise RuntimeError("SIM baseline length mismatch.")
            v0S = v0S_vals
        dVS = XS_aligned - v0S[None, :]

    # Build ΔV for REAL
    if args.real_is_delta:
        dVR = XR_aligned
    else:
        if not args.real_baseline_csv:
            raise RuntimeError("REAL is raw → provide --real-baseline-csv.")
        v0R_vals, v0R_idx = load_baseline_csv(Path(args.real_baseline_csv))
        if v0R_idx:
            pos = {i:j for j,i in enumerate(v0R_idx)}
            v0R = np.array([v0R_vals[pos[i]] if i in pos else 0.0 for i in common_idx], dtype=np.float32)
        else:
            if len(v0R_vals) != XR_aligned.shape[1]:
                raise RuntimeError("REAL baseline length mismatch.")
            v0R = v0R_vals
        dVR = XR_aligned - v0R[None, :]

    # Pair rows by key intersection
    keys_common = pd.Index(dfS["__key__"]).intersection(dfR["__key__"])
    if len(keys_common) == 0:
        raise RuntimeError("No matching (x1,y1,x2,y2) keys between SIM and REAL.")
    # Build paired arrays in the same order
    rowsS, rowsR = [], []
    for k in keys_common:
        i = dfS.index[dfS["__key__"]==k][0]
        j = dfR.index[dfR["__key__"]==k][0]
        rowsS.append(i); rowsR.append(j)
    Xs = dVS[rowsS]
    Xr = dVR[rowsR]

    # Train/val/test split
    N = len(Xs)
    idx = np.arange(N)
    rng = np.random.default_rng(0); rng.shuffle(idx)
    n_test = int(args.test_frac*N)
    n_val  = int(args.val_frac*N)
    test_idx = idx[:n_test]
    val_idx  = idx[n_test:n_test+n_val]
    train_idx = idx[n_test+n_val:]

    ds_tr = PairDataset(Xs[train_idx], Xr[train_idx], keys_common.to_numpy()[train_idx])
    ds_va = PairDataset(Xs[val_idx],   Xr[val_idx],   keys_common.to_numpy()[val_idx])
    ds_te = PairDataset(Xs[test_idx],  Xr[test_idx],  keys_common.to_numpy()[test_idx])
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResidualSim2Real(c_in=Xs.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    A = None
    if args.phys_A_npy:
        A = torch.from_numpy(np.load(args.phys_A_npy).astype(np.float32)).to(device)  # [M,C]

    # Train
    best_val = 1e9; best_path = outdir/"sim2real_residual.pt"
    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, dl_tr, opt, l1_w=args.l1_w, corr_w=args.corr_w, A=A, phys_w=args.phys_w, device=device)
        va_mae, va_corr = eval_epoch(model, dl_va, device=device)
        print(f"[{ep:03d}] train_loss={tr_loss:.4f} | val_MAE={va_mae:.5f} | val_corr={va_corr:.4f}")
        # save by MAE
        if va_mae < best_val:
            best_val = va_mae
            torch.save({"state_dict": model.state_dict(),
                        "channels": len(common_idx)}, best_path)

    # Test with best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    te_mae, te_corr = eval_epoch(model, dl_te, device=device)
    with open(outdir/"metrics.txt","w") as f:
        f.write(f"val_best_MAE={best_val:.6f}\n")
        f.write(f"test_MAE={te_mae:.6f}\n")
        f.write(f"test_corr={te_corr:.6f}\n")
    print(f"[TEST] MAE={te_mae:.6f} corr={te_corr:.4f}")
    # Sample plots
    model.eval()
    with torch.no_grad():
        Yhat = []
        for xs,_ in dl_te:
            xs = xs.to(device); Yhat.append(model(xs).cpu().numpy())
        Yhat = np.concatenate(Yhat, axis=0)
    plot_sample_pairs(Xs[test_idx], Xr[test_idx], Yhat, outdir/ "sample_pairs", n_show=8)

    # Optional: export translated SIM CSV (ΔV columns replaced by F(ΔV_sim))
    if args.export_translated_sim:
        # predict for ALL sim rows (aligned subset order)
        with torch.no_grad():
            y_all = []
            B = 1024
            for i in range(0, len(dVS), B):
                xs = torch.from_numpy(dVS[i:i+B]).to(device)
                y = model(xs).cpu().numpy()
                y_all.append(y)
            y_all = np.vstack(y_all)
        # write CSV mirroring SIM but with aligned columns replaced
        out_csv = outdir/"sim_translated_deltaV.csv"
        df_out = dfS.copy()
        for col in find_eit_cols(df_out):
            if col in sim_sel:  # only those in intersection / aligned
                j = sim_sel.index(col)      # position in aligned order
                df_out[col] = y_all[:, j]
            else:
                # keep other columns as-is (e.g., extra channels not in intersection)
                pass
        df_out.to_csv(out_csv, index=False)
        print(f"Translated SIM ΔV written → {out_csv}")

if __name__ == "__main__":
    main()
