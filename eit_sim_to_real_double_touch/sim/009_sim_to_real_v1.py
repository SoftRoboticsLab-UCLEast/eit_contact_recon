#!/usr/bin/env python3
"""
Train on diffusion-translated ΔV (SIM→REAL-like) and evaluate on REAL.

- Uses channel alignment saved by the diffusion step (preproc_stats.npz).
- REAL baseline supported when REAL CSV contains raw voltages.
- Pairs rows by identical (x1,y1,x2,y2) or (R1_x_norm,...R2_y_norm) keys.
- Vectorized pairing to avoid per-row shape drift.

Example:
  python 010_train_on_translated_eval_on_real.py \
    --translated-csv runs/diffusion_v2/translated_deltaV.csv \
    --real-csv training_dataset/training_dataset.csv \
    --preproc-stats runs/diffusion_v2/preproc_stats.npz \
    --real-is-delta \
    --out runs/sim2real_train_eval_v1
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------ helpers ------------------

NUM_SUFFIX = re.compile(r"(\d+)$")
def parse_idx(name: str):
    m = NUM_SUFFIX.search(name); return int(m.group(1)) if m else None

def to_numeric(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    return df[list(cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)

def choose_label_cols(df: pd.DataFrame) -> List[str]:
    norm = ["R1_x_norm","R1_y_norm","R2_x_norm","R2_y_norm"]
    raw  = ["x1","y1","x2","y2"]
    if all(c in df.columns for c in norm): return norm
    if all(c in df.columns for c in raw):  return raw
    raise RuntimeError("Label columns not found (need R1_x_norm.. or x1..).")

def row_keys(df: pd.DataFrame, cols: List[str], prec=4) -> pd.Series:
    r = df[cols].astype(float).round(prec)
    return (r[cols[0]].astype(str)+"|"+r[cols[1]].astype(str)+"|"+
            r[cols[2]].astype(str)+"|"+r[cols[3]].astype(str))

def load_baseline_csv(path: Path) -> Tuple[np.ndarray, List[int]]:
    """Load baseline CSV:
       - headered: uses column name numeric suffixes for indexing
       - headerless: just a flat vector
    """
    try:
        bdf = pd.read_csv(path, header=0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        vals = bdf.to_numpy(np.float32).flatten()
        idxs = [parse_idx(c) for c in bdf.columns]
        if all(i is not None for i in idxs):
            return vals, idxs
        return vals, []
    except Exception:
        bdf = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return bdf.to_numpy(np.float32).flatten(), []

# ------------------ model ------------------

class EITNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
            nn.Linear(128, 64),     nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translated-csv", required=True, help="Diffusion-translated SIM CSV (features source)")
    ap.add_argument("--real-csv", required=True, help="REAL dataset CSV (labels + eval features)")
    ap.add_argument("--preproc-stats", required=True, help="preproc_stats.npz from diffusion run")
    ap.add_argument("--real-is-delta", action="store_true", help="REAL CSV voltages already ΔV (no baseline subtraction)")
    ap.add_argument("--real-baseline-csv", default=None, help="REAL baseline CSV if REAL is raw")
    ap.add_argument("--out", default="runs/sim2real_train_eval", help="Output directory")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-split", type=float, default=0.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # ----- load stats from diffusion step (channel alignment!) -----
    stats = np.load(args.preproc_stats, allow_pickle=True)
    sim_cols  = [str(x) for x in stats["sim_cols"].tolist()]
    real_cols = [str(x) for x in stats["real_cols"].tolist()]
    print(f"[INFO] Loaded alignment: sim_cols={len(sim_cols)}, real_cols={len(real_cols)}")

    # ----- load CSVs -----
    dfT = pd.read_csv(args.translated_csv)  # translated features live here (already ΔV in SIM naming)
    dfR = pd.read_csv(args.real_csv)        # real labels + real features (ΔV or raw)

   # ----- build row-keys to pair rows (same touch positions) -----
    lblT = choose_label_cols(dfT)
    lblR = choose_label_cols(dfR)
    dfT = dfT.copy(); dfT["__key__"] = row_keys(dfT, lblT).values
    dfR = dfR.copy(); dfR["__key__"] = row_keys(dfR, lblR).values

    # Report duplicates
    dupT = dfT["__key__"].duplicated(keep=False).sum()
    dupR = dfR["__key__"].duplicated(keep=False).sum()
    print(f"[KEYS] duplicates → translated={dupT}, real={dupR}")

    # De-duplicate to enforce 1–1 mapping by key (keep first occurrence)
    # (If you prefer averaging duplicates on REAL, see the commented block below.)
    dfT_u = dfT.drop_duplicates(subset="__key__", keep="first").copy()
    dfR_u = dfR.drop_duplicates(subset="__key__", keep="first").copy()

    # Intersect keys after de-dup
    keys_common = pd.Index(dfT_u["__key__"]).intersection(dfR_u["__key__"])
    if len(keys_common) == 0:
        raise RuntimeError("No paired rows found between translated and REAL after de-duplication.")

    # Keep only paired rows (and align REAL rows to TRANSLATED order)
    dfT_p = dfT_u[dfT_u["__key__"].isin(keys_common)].copy()
    dfR_p = dfR_u[dfR_u["__key__"].isin(keys_common)].copy().set_index("__key__")
    dfR_aligned = dfR_p.loc[dfT_p["__key__"]].reset_index()  # one REAL row per TRANSLATED row

    # ---- build X_trans (training), X_real (eval), Y (labels) in one shot ----
    X_trans = to_numeric(dfT_p, sim_cols)
    XR_raw  = to_numeric(dfR_aligned, real_cols)

    # If REAL is raw → ΔV
    if not args.real_is_delta:
        if not args.real_baseline_csv:
            raise RuntimeError("REAL is raw ⇒ provide --real-baseline-csv.")
        v0_vals, v0_idx = load_baseline_csv(Path(args.real_baseline_csv))
        if v0_idx:
            pos = {i: j for j, i in enumerate(v0_idx)}
            idxs = [parse_idx(c) for c in real_cols]
            v0 = np.array([v0_vals[pos[i]] if i in pos else 0.0 for i in idxs], dtype=np.float32)
        else:
            if len(v0_vals) != len(real_cols):
                raise RuntimeError("Baseline length mismatch with REAL channel count.")
            v0 = v0_vals.astype(np.float32)
        X_real = XR_raw - v0[None, :]
    else:
        X_real = XR_raw

    Y = to_numeric(dfR_aligned, lblR)

    # Final sanity checks
    if X_trans.shape[0] != X_real.shape[0] or X_real.shape[0] != Y.shape[0]:
        raise RuntimeError(f"Row mismatch after pairing: X_trans={X_trans.shape}, X_real={X_real.shape}, Y={Y.shape}")
    if X_trans.shape[1] != X_real.shape[1]:
        raise RuntimeError(f"Feature dim mismatch: translated has {X_trans.shape[1]}, REAL has {X_real.shape[1]}.")
    print(f"[PAIR] paired={X_trans.shape[0]} rows | feat_dim={X_trans.shape[1]}")

    N, C = X_trans.shape

    # ----- split (same indices for features+labels) -----
    tr_idx, te_idx = train_test_split(np.arange(N), test_size=args.test_split, random_state=42, shuffle=True)
    Xtr = X_trans[tr_idx]
    Xte_real = X_real[te_idx]   # evaluation on REAL features
    Ytr = Y[tr_idx]
    Yte = Y[te_idx]

    # ----- scale using TRAIN (translated) stats; apply to REAL test too -----
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte_real)

    # ----- torch datasets -----
    device = torch.device(args.device)
    train_ds = torch.utils.data.TensorDataset(torch.tensor(Xtr_s), torch.tensor(Ytr))
    test_ds  = torch.utils.data.TensorDataset(torch.tensor(Xte_s), torch.tensor(Yte))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl  = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ----- model -----
    model = EITNet(in_dim=C).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    # ----- train -----
    train_losses = []
    for ep in range(1, args.epochs+1):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.float32)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()*len(xb)
        ep_loss /= len(train_dl.dataset)
        train_losses.append(ep_loss)
        if ep % 10 == 0 or ep == 1:
            print(f"[{ep:03d}] train_mse={ep_loss:.6f}")

    # save checkpoint
    ckpt = {
        "state": model.state_dict(),
        "in_dim": C,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "sim_cols": sim_cols,
        "real_cols": real_cols,
    }
    torch.save(ckpt, out_dir / "model.pt")
    print(f"✅ saved → {out_dir/'model.pt'}")

    # ----- eval -----
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for xb, yb in test_dl:
            pr = model(xb.to(device, dtype=torch.float32)).cpu().numpy()
            preds.append(pr)
            gts.append(yb.numpy())
    preds = np.vstack(preds)
    gts = np.vstack(gts)
    mse = float(np.mean((preds - gts)**2))
    print(f"[RESULTS] Test MSE={mse:.6f}")

    # ----- plots -----
    # loss
    plt.figure()
    plt.plot(train_losses, label="Train MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Training Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "training_loss_curve.png", dpi=200); plt.close()

    # joint disk plot (both touches together)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.add_patch(plt.Circle((0,0),1.0,fill=False,color='k',lw=1.0))
    ax.scatter(gts[:,0], gts[:,1], s=10, c="tab:blue", alpha=0.5, label="True")
    ax.scatter(gts[:,2], gts[:,3], s=10, c="tab:blue", alpha=0.5)
    ax.scatter(preds[:,0], preds[:,1], s=10, c="tab:red",  alpha=0.5, label="Pred")
    ax.scatter(preds[:,2], preds[:,3], s=10, c="tab:red",  alpha=0.5)
    ax.set_aspect("equal"); ax.set_xlim([-1.1,1.1]); ax.set_ylim([-1.1,1.1])
    ax.legend(); ax.set_title("Pred vs True (REAL test)")
    plt.tight_layout(); plt.savefig(out_dir / "pred_vs_true_joint.png", dpi=220); plt.close()

    # separate R1/R2 panels
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    for a in axs:
        a.add_patch(plt.Circle((0,0),1.0,fill=False,color='k',lw=1.0))
        a.set_aspect("equal"); a.set_xlim([-1.1,1.1]); a.set_ylim([-1.1,1.1])
    axs[0].scatter(gts[:,0], gts[:,1], s=10, c="tab:blue", alpha=0.5, label="True")
    axs[0].scatter(preds[:,0], preds[:,1], s=10, c="tab:red",  alpha=0.5, label="Pred")
    axs[0].set_title("Robot 1"); axs[0].legend()
    axs[1].scatter(gts[:,2], gts[:,3], s=10, c="tab:blue", alpha=0.5, label="True")
    axs[1].scatter(preds[:,2], preds[:,3], s=10, c="tab:red",  alpha=0.5, label="Pred")
    axs[1].set_title("Robot 2"); axs[1].legend()
    plt.tight_layout(); plt.savefig(out_dir / "pred_vs_true_separate.png", dpi=220); plt.close()

    # numeric summary
    mae = np.mean(np.abs(preds-gts), axis=0)
    rmse = np.sqrt(np.mean((preds-gts)**2, axis=0))
    with open(out_dir / "metrics.txt","w") as f:
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"MAE: {mae.tolist()}\n")
        f.write(f"RMSE:{rmse.tolist()}\n")
    print(f"✅ results saved in {out_dir}")

if __name__ == "__main__":
    main()
