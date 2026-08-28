#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def pi_pair_errors(pred, target):
    """
    pred, target: [B,4] = (x1,y1,x2,y2)
    returns (eA, eB):
      eA = errors for assignment A: p1->t1, p2->t2
      eB = errors for assignment B: p1->t2, p2->t1
    """
    p1, p2 = pred[:, :2], pred[:, 2:4]
    t1, t2 = target[:, :2], target[:, 2:4]
    eA = torch.abs(torch.cat([p1 - t1, p2 - t2], dim=1))  # [B,4]
    eB = torch.abs(torch.cat([p1 - t2, p2 - t1], dim=1))  # [B,4]
    return eA, eB

def pi_l1_loss(pred, target, reduce="mean"):
    eA, eB = pi_pair_errors(pred, target)
    LA = eA.mean(dim=1)  # L1 per-sample
    LB = eB.mean(dim=1)
    L  = torch.minimum(LA, LB)  # pick the better assignment per row
    return L.mean() if reduce=="mean" else L

def pi_l2_loss(pred, target, reduce="mean"):
    eA, eB = pi_pair_errors(pred, target)
    LA = (eA**2).mean(dim=1)
    LB = (eB**2).mean(dim=1)
    L  = torch.minimum(LA, LB)
    return L.mean() if reduce=="mean" else L

# ---------- tiny MLP ----------
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

def find_eit_cols(df):
    cols = [c for c in df.columns if c.startswith("eit_") or c.startswith("v")]
    if not cols: raise RuntimeError("No EIT channels found.")
    return cols

def main():
    ap = argparse.ArgumentParser(description="Train regressor on BASIC simulated double-touch deltas.")
    ap.add_argument("--csv", default="sim/basic_sim.csv")
    ap.add_argument("--out", default="sim_basic_training")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--test-split", type=float, default=0.2)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load CSV (already delta voltages!)
    df = pd.read_csv(args.csv)
    eit_cols = find_eit_cols(df)
    X = df[eit_cols].to_numpy(np.float32)
    y = df[["x1","y1","x2","y2"]].to_numpy(np.float32)

    # 2) Drop all-zero channels (if any)
    nonzero = np.any(X != 0.0, axis=0)
    X = X[:, nonzero]
    kept = [c for c, m in zip(eit_cols, nonzero) if m]
    print(f"[INFO] using {X.shape[1]} channels")

    # 3) Split + scale
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_split, random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    # 4) Torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr_ds = torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    te_ds = torch.utils.data.TensorDataset(torch.tensor(Xte), torch.tensor(yte))
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=args.batch, shuffle=False)

    model = Net(d_in=X.shape[1]).to(dev)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # 5) Train
    losses = []
    for ep in range(1, args.epochs+1):
        model.train(); tot = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            pred = model(xb)
            # loss = loss_fn(pred, yb)
            loss = pi_l1_loss(pred, yb)
            loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        losses.append(tot / len(tr_ds))
        if ep % 20 == 0 or ep == 1:
            print(f"[{ep:03d}/{args.epochs}] loss={losses[-1]:.6f}")

    # 6) Save model + scaler
    torch.save({
        "state": model.state_dict(),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "in_dim": X.shape[1], "eit_cols": kept
    }, out_dir / "model.pt")

    # 7) Loss curve
    plt.figure(); plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.title("Training loss (basic sim)"); plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=200); plt.close()

    # 8) Eval
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in te_dl:
            xb = xb.to(dev)
            yp = model(xb).cpu().numpy()
            preds.append(yp)
    preds = np.vstack(preds)

    # MAE/RMSE
    err = preds - yte
    mae = np.mean(np.abs(err), axis=0); rmse = np.sqrt(np.mean(err**2, axis=0))
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"MAE:  {mae.tolist()}\nRMSE: {rmse.tolist()}\n")
    print(f"[INFO] MAE {mae} | RMSE {rmse}")

    # 9) Joint plot (both touches on one disk)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.add_patch(plt.Circle((0,0),1.0,fill=False,color="k",lw=1.0))
    ax.scatter(yte[:,0], yte[:,1], s=10, c="tab:blue",  alpha=0.5, label="True")
    ax.scatter(yte[:,2], yte[:,3], s=10, c="tab:blue",  alpha=0.5)
    ax.scatter(preds[:,0], preds[:,1], s=10, c="tab:red", alpha=0.5, label="Pred")
    ax.scatter(preds[:,2], preds[:,3], s=10, c="tab:red", alpha=0.5)
    ax.set_aspect("equal"); ax.set_xlim([-1.1,1.1]); ax.set_ylim([-1.1,1.1])
    ax.legend(); ax.set_title("True vs Predicted Double-Touch (joint)")
    plt.tight_layout(); plt.savefig(out_dir / "pred_vs_true_joint.png", dpi=220); plt.close()

    print(f"✅ Results saved in {out_dir}/")

if __name__ == "__main__":
    main()
