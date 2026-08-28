#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import re

# ------------ helpers ------------
NUM_SUFFIX = re.compile(r"(\d+)$")
def parse_idx(name: str):
    m = NUM_SUFFIX.search(name)
    return int(m.group(1)) if m else None

def find_eit_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        lc = c.lower()
        if lc == "eit_t":  # ignore time column
            continue
        if lc.startswith("eit_") or lc.startswith("v"):
            if parse_idx(c) is not None:
                cols.append(c)
    # keep file order here; we'll reorder to model's list later
    return cols

def to_numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)

def load_baseline_csv(path: Path) -> tuple[np.ndarray, list[int]]:
    """Return (values_flat, numeric_indices_or_empty)."""
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

def drop_real_zero_cols(X: np.ndarray, names: list[str]):
    keep = np.any(X != 0.0, axis=0)
    return X[:, keep], [n for n,k in zip(names, keep) if k]

def reorder_to_model_order(df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Select only model_cols and **in exactly that order**. Warn on missing cols."""
    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing {len(missing)} EIT cols in real CSV; first few: {missing[:5]}")
    present = [c for c in model_cols if c in df.columns]
    return df[present], present, missing

def maybe_roll_channels(X: np.ndarray, k: int) -> np.ndarray:
    if k == 0: return X
    return np.roll(X, shift=k, axis=1)

def perm_invariant_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_* shape [N,4]: (x1,y1,x2,y2). Compute MAE/RMSE per coord with best permutation per-row.
    """
    t1 = y_true[:, :2]; t2 = y_true[:, 2:4]
    p1 = y_pred[:, :2]; p2 = y_pred[:, 2:4]
    # two permutations: (p1->t1,p2->t2) or (p1->t2,p2->t1)
    e_a = np.concatenate([p1 - t1, p2 - t2], axis=1)
    e_b = np.concatenate([p1 - t2, p2 - t1], axis=1)
    # choose per-row smaller L2
    l2_a = np.sqrt((e_a[:,0]**2+e_a[:,1]**2) + (e_a[:,2]**2+e_a[:,3]**2))
    l2_b = np.sqrt((e_b[:,0]**2+e_b[:,1]**2) + (e_b[:,2]**2+e_b[:,3]**2))
    choose_a = l2_a <= l2_b
    E = np.where(choose_a[:,None], e_a, e_b)
    mae = np.mean(np.abs(E), axis=0)
    rmse = np.sqrt(np.mean(E**2, axis=0))
    return mae, rmse, choose_a

# ------------ model ------------
class Net(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(),
            nn.Linear(256, 128),  nn.ReLU(),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64, 4)
        )
    def forward(self, x): return self.net(x)

# ------------ main ------------
def main():
    ap = argparse.ArgumentParser(description="Test sim-trained model on real EIT dataset (fixed alignment + ΔV + PI metrics)")
    ap.add_argument("--model", required=True, help="Path to model.pt (trained on sim or translated sim)")
    ap.add_argument("--real-csv", required=True, help="Path to real dataset CSV")
    ap.add_argument("--real-baseline-csv", default=None, help="If provided, compute ΔV_real = V - v0")
    ap.add_argument("--drop-real-zero-cols", action="store_true", help="Drop exact-zero columns in REAL before alignment (recommended).")
    ap.add_argument("--roll", type=int, default=0, help="Optional cyclic roll to apply to REAL channels before scaling (to test electrode rotation).")
    ap.add_argument("--out", default="results/sim_to_real_eval_fixed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Load model + training metadata
    ckpt = torch.load(args.model, map_location=args.device)
    in_dim = ckpt["in_dim"]
    scaler_mean = ckpt["scaler_mean"]
    scaler_scale = ckpt["scaler_scale"]
    model_eit_cols = ckpt["eit_cols"]          # **expected order**
    model_is_delta = ckpt.get("is_delta", True) # whether the model was trained on ΔV
    print(f"[INFO] Model expects {in_dim} features; ΔV trained = {model_is_delta}")

    model = Net(d_in=in_dim).to(args.device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    # --- 2) Load REAL CSV and build feature matrix in the **model's column order**
    df = pd.read_csv(args.real_csv)
    all_real_cols = find_eit_cols(df)

    # Optionally drop zero-only columns BEFORE reordering
    if args.drop_real_zero_cols:
        X_all = to_numeric_matrix(df, all_real_cols)
        X_all, kept_names = drop_real_zero_cols(X_all, all_real_cols)
        df = df.drop(columns=[c for c in all_real_cols if c not in kept_names])
        all_real_cols = kept_names
        print(f"[INFO] REAL zero-drop: kept {len(all_real_cols)} cols")

    # Reorder to the **exact model order**
    df_eit, present_cols, missing_cols = reorder_to_model_order(df, model_eit_cols)
    if len(present_cols) != len(model_eit_cols):
        print(f"[WARN] Only {len(present_cols)}/{len(model_eit_cols)} EIT cols present; filling missing with zeros.")
        # Fill missing expected cols with zeros (kept at the end to preserve expected order length)
        for c in model_eit_cols:
            if c not in df_eit.columns:
                df_eit[c] = 0.0
        # reorder again to exact model order
        df_eit = df_eit[model_eit_cols]

    X_real = to_numeric_matrix(df_eit, list(df_eit.columns))  # numeric, in model order

    # --- 3) Compute ΔV_real if needed
    if model_is_delta:
        if args.real_baseline_csv is None:
            print("[WARN] Model trained on ΔV but no REAL baseline given; assuming CSV already stores ΔV (risky).")
        else:
            v0_vals, v0_idx = load_baseline_csv(Path(args.real_baseline_csv))
            if v0_idx:
                # Map baseline by numeric suffix to the model's columns
                idx_map = {parse_idx(c): j for j, c in enumerate(model_eit_cols)}
                base = np.zeros((len(model_eit_cols),), dtype=np.float32)
                bmap = {i:j for j,i in enumerate(v0_idx)}
                for i, j in idx_map.items():
                    if i in bmap:
                        base[j] = v0_vals[bmap[i]]
                v0 = base
            else:
                if len(v0_vals) != X_real.shape[1]:
                    raise RuntimeError("REAL baseline length doesn't match model feature count.")
                v0 = v0_vals.astype(np.float32)
            X_real = X_real - v0[None, :]

    # --- 4) Optional cyclic roll to mimic electrode rotation mismatch
    if args.roll != 0:
        X_real = maybe_roll_channels(X_real, args.roll)
        print(f"[INFO] Applied cyclic roll k={args.roll} to REAL channels.")

    # --- 5) Scale using the **sim-trained** scaler
    X_scaled = (X_real - scaler_mean) / scaler_scale
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 6) Labels
    label_cols_norm = ["R1_x_norm","R1_y_norm","R2_x_norm","R2_y_norm"]
    label_cols_plain = ["x1","y1","x2","y2"]
    if all(c in df.columns for c in label_cols_norm): label_cols = label_cols_norm
    elif all(c in df.columns for c in label_cols_plain): label_cols = label_cols_plain
    else:
        raise RuntimeError("Could not find label columns in real dataset")
    y_true = df[label_cols].to_numpy(np.float32)

    # --- 7) Predict
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=args.device)
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    # --- 8) Permutation-invariant metrics (accounts for swapped touch order)
    pi_mae, pi_rmse, chose_A = perm_invariant_mae_rmse(y_true, y_pred)
    print(f"[RESULTS: PI] MAE = {pi_mae}, RMSE = {pi_rmse}")

    with open(out_dir / "metrics_permutation_invariant.txt", "w") as f:
        f.write(f"PI_MAE:  {pi_mae.tolist()}\n")
        f.write(f"PI_RMSE: {pi_rmse.tolist()}\n")

    # --- 9) Plot (true vs pred on one circle)
    fig, ax = plt.subplots(figsize=(6,6))
    circ = plt.Circle((0,0), 1.0, fill=False, color="k", lw=1)
    ax.add_patch(circ)
    ax.scatter(y_true[:,0], y_true[:,1], s=10, c="tab:blue", alpha=0.5, label="True")
    ax.scatter(y_true[:,2], y_true[:,3], s=10, c="tab:blue", alpha=0.5)
    ax.scatter(y_pred[:,0], y_pred[:,1], s=10, c="tab:red",  alpha=0.5, label="Pred")
    ax.scatter(y_pred[:,2], y_pred[:,3], s=10, c="tab:red",  alpha=0.5)
    ax.set_aspect("equal"); ax.set_xlim([-1.1,1.1]); ax.set_ylim([-1.1,1.1])
    ax.legend(); ax.set_title("Sim-trained model on REAL (raw vs PI order not enforced)")
    plt.tight_layout(); plt.savefig(out_dir/"pred_vs_true_realdata_raw_order.png", dpi=220); plt.close()

    # Plot with permuted predictions (best matching per row)
    y_perm = y_pred.copy()
    swap = ~chose_A  # rows where best matching is the swapped assignment
    y_perm[swap, :2], y_perm[swap, 2:4] = y_perm[swap, 2:4].copy(), y_perm[swap, :2].copy()

    fig, ax = plt.subplots(figsize=(6,6))
    ax.add_patch(plt.Circle((0,0), 1.0, fill=False, color="k", lw=1))
    ax.scatter(y_true[:,0], y_true[:,1], s=10, c="tab:blue", alpha=0.5, label="True")
    ax.scatter(y_true[:,2], y_true[:,3], s=10, c="tab:blue", alpha=0.5)
    ax.scatter(y_perm[:,0], y_perm[:,1], s=10, c="tab:orange", alpha=0.5, label="Pred (PI)")
    ax.scatter(y_perm[:,2], y_perm[:,3], s=10, c="tab:orange", alpha=0.5)
    ax.set_aspect("equal"); ax.set_xlim([-1.1,1.1]); ax.set_ylim([-1.1,1.1])
    ax.legend(); ax.set_title("Sim-trained model on REAL (permutation-invariant assignment)")
    plt.tight_layout(); plt.savefig(out_dir/"pred_vs_true_realdata_perm_invariant.png", dpi=220); plt.close()

    print(f"✅ Results saved to {out_dir}")

if __name__ == "__main__":
    main()
