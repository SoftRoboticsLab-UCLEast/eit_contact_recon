#!/usr/bin/env python3
"""
Evaluate a model trained on transformed (sim→real) data on REAL data
using domain-aware normalization (real→sim z-score mapping).

Usage:
  python eval_domain_align.py \
    --model training_results_sim2real/eit_contact_model_simtrained.pt \
    --train-csv /path/to/transformed_dataset.csv \
    --test-csv  /path/to/training_dataset.csv \
    --out-dir results/simtrained_on_real_eval
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------ helpers ------------------
def parse_idx(name):
    import re
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None

def find_eit_cols(df: pd.DataFrame):
    cols=[]
    for c in df.columns:
        lc=c.lower()
        if lc=="eit_t": continue
        if lc.startswith("eit_") or lc.startswith("v"):
            if parse_idx(c) is not None:
                cols.append(c)
    if not cols:
        raise RuntimeError("No EIT columns found.")
    return cols

def to_numeric(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)

def align_by_intersection(cols_train, cols_test):
    # map by numeric suffix, intersect, sort by index
    def build_map(cols):
        out={}
        for c in cols:
            i=parse_idx(c)
            if i is not None:
                out[i]=c
        return out
    mt = build_map(cols_train)
    ms = build_map(cols_test)
    common = sorted(set(mt.keys()) & set(ms.keys()))
    if len(common) < 50:
        raise RuntimeError(f"Too few common channels ({len(common)}).")
    train_sel = [mt[i] for i in common]
    test_sel  = [ms[i] for i in common]
    return train_sel, test_sel, common

class EITNet(nn.Module):
    def __init__(self, in_dim, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
            nn.Linear(128, 64),     nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-csv", required=True, help="transformed (sim→real) CSV used for training")
    ap.add_argument("--test-csv",  required=True, help="real CSV for evaluation")
    ap.add_argument("--out-dir",   default="results/simtrained_on_real_eval")
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1) load model checkpoint (no scaler assumed inside)
    ckpt = torch.load(args.model, map_location=args.device)
    in_dim = ckpt["in_dim"]
    state  = ckpt["model_state"]

    # 2) load CSVs
    df_tr = pd.read_csv(args.train_csv)  # transformed training domain
    df_te = pd.read_csv(args.test_csv)   # real domain

    # columns (align!)
    cols_tr_all = find_eit_cols(df_tr)
    cols_te_all = find_eit_cols(df_te)
    tr_sel, te_sel, _ = align_by_intersection(cols_tr_all, cols_te_all)

    X_tr = to_numeric(df_tr, tr_sel)
    X_te = to_numeric(df_te, te_sel)

    # optional: drop exact-zero cols by REAL to avoid dead channels
    keep = np.any(X_te != 0.0, axis=0)
    X_tr = X_tr[:, keep]
    X_te = X_te[:, keep]
    sel_names = [c for c,k in zip(tr_sel, keep) if k]

    if X_tr.shape[1] != in_dim:
        # If model was trained with a different channel mask, warn clearly
        print(f"[WARN] Model expects in_dim={in_dim}, but aligned features={X_tr.shape[1]}. "
              f"Proceeding only if equal…")
    assert X_tr.shape[1] == X_te.shape[1], "Train/Test feature mismatch after alignment."
    assert X_tr.shape[1] == in_dim, "Input dim mismatch for loaded model."

    # labels from REAL csv
    y_cols = ["R1_x_norm","R1_y_norm","R2_x_norm","R2_y_norm"]
    if not all(c in df_te.columns for c in y_cols):
        raise RuntimeError("Missing label columns in REAL test CSV.")
    Y_te = df_te[y_cols].to_numpy(np.float32)

    # 3) compute stats: sim-domain (transformed) and real-domain (test)
    mu_s = X_tr.mean(axis=0);  sig_s = X_tr.std(axis=0) + 1e-6
    mu_r = X_te.mean(axis=0);  sig_r = X_te.std(axis=0) + 1e-6

    # domain-aware mapping: z_r → z_s_expected = a*z_r + b
    a = (sig_r / sig_s)             # shape [C]
    b = (mu_r - mu_s) / sig_s

    # 4) Build model
    device = args.device
    model = EITNet(in_dim=X_tr.shape[1]).to(device)
    model.load_state_dict(state)
    model.eval()

    # 5) Prepare test inputs:
    #    z_r = (x - mu_r)/sig_r;   z_in = a*z_r + b
    Z_r  = (X_te - mu_r[None, :]) / sig_r[None, :]
    Z_in = (a[None, :] * Z_r) + b[None, :]
    Z_in = np.nan_to_num(Z_in, 0.0, 0.0, 0.0)

    with torch.no_grad():
        pred = model(torch.from_numpy(Z_in).to(device)).cpu().numpy()

    # 6) metrics (per-dimension + grouped)
    mae_xy = mean_absolute_error(Y_te, pred, multioutput="raw_values")  # [4]
    rmse_xy = np.sqrt(mean_squared_error(Y_te, pred, multioutput="raw_values"))

    # aggregate per touch
    mae_r1 = float(np.mean(mae_xy[:2])); rmse_r1 = float(np.mean(rmse_xy[:2]))
    mae_r2 = float(np.mean(mae_xy[2:])); rmse_r2 = float(np.mean(rmse_xy[2:]))

    print(f"[RESULTS] MAE(x1,y1,x2,y2) = {mae_xy}")
    print(f"[RESULTS] RMSE(x1,y1,x2,y2) = {rmse_xy}")
    print(f"[RESULTS] R1  MAE={mae_r1:.4f}  RMSE={rmse_r1:.4f}")
    print(f"[RESULTS] R2  MAE={mae_r2:.4f}  RMSE={rmse_r2:.4f}")

    # save
    with open(out/"metrics.txt","w") as f:
        f.write(f"MAE_x1={mae_xy[0]:.6f}\nMAE_y1={mae_xy[1]:.6f}\n"
                f"MAE_x2={mae_xy[2]:.6f}\nMAE_y2={mae_xy[3]:.6f}\n")
        f.write(f"RMSE_x1={rmse_xy[0]:.6f}\nRMSE_y1={rmse_xy[1]:.6f}\n"
                f"RMSE_x2={rmse_xy[2]:.6f}\nRMSE_y2={rmse_xy[3]:.6f}\n")
        f.write(f"R1_mean_MAE={mae_r1:.6f}\nR1_mean_RMSE={rmse_r1:.6f}\n")
        f.write(f"R2_mean_MAE={mae_r2:.6f}\nR2_mean_RMSE={rmse_r2:.6f}\n")

if __name__ == "__main__":
    main()
