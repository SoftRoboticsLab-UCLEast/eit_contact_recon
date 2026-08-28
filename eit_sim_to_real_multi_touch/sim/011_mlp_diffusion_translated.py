#!/usr/bin/env python3
"""
Train a deep model to predict double-touch contact locations
from EIT voltage data, and report per-coordinate MAE/RMSE on the test set.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim


# === Config ===
DATA_CSV = "/home/kiyanoush/Projects/eit_sim_to_real_multi_touch/runs/diffusion_final/translated_deltaV.csv"
OUT_DIR = Path("diff_mlp_training_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
TEST_SPLIT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# === Model Definition ===
class EITNet(nn.Module):
    def __init__(self, in_dim, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# === Data Loading ===
print(f"[INFO] Loading dataset: {DATA_CSV}")
df = pd.read_csv(DATA_CSV)

# Identify EIT columns (drop absolute-zero columns)
eit_cols = [c for c in df.columns if c.startswith("eit_")]
X = df[eit_cols].to_numpy(np.float32)
nonzero_mask = np.any(X != 0, axis=0)
X = X[:, nonzero_mask]
print(f"[INFO] Input features: {X.shape[1]} (non-zero channels)")

# Extract target coordinates (rotated and normalized)
y_cols = ["x1", "y1", "x2", "y2"]
y = df[y_cols].to_numpy(np.float32)
print(f"[INFO] Target output dims: {y.shape[1]}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=42
)

# Normalize inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds  = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl  = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# === Model Setup ===
model = EITNet(in_dim=X.shape[1]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# === Training Loop ===
train_losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    epoch_loss /= len(train_dl.dataset)
    train_losses.append(epoch_loss)
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Loss: {epoch_loss:.6f}")

# Save model
torch.save({
    "model_state": model.state_dict(),
    "scaler": scaler,
    "in_dim": X.shape[1],
    "eit_cols_kept": [c for c,m in zip(eit_cols, nonzero_mask) if m],
}, OUT_DIR / "eit_contact_model.pt")
print(f"[INFO] Model saved → {OUT_DIR/'eit_contact_model.pt'}")

# Plot loss curve
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "training_loss_curve.png", dpi=200)
plt.close()

# === Evaluation ===
model.eval()
with torch.no_grad():
    preds = []
    gts = []
    for xb, yb in test_dl:
        xb = xb.to(DEVICE)
        pred = model(xb).cpu().numpy()
        preds.append(pred)
        gts.append(yb.numpy())
preds = np.vstack(preds)  # [N,4]
gts   = np.vstack(gts)    # [N,4]

# Overall MSE
mse = float(np.mean((preds - gts) ** 2))
print(f"[INFO] Test MSE: {mse:.6f}")

# --- Per-coordinate MAE & RMSE ---
abs_err = np.abs(preds - gts)           # [N,4]
sq_err  = (preds - gts) ** 2            # [N,4]
mae_vec = abs_err.mean(axis=0)          # [4]
rmse_vec = np.sqrt(sq_err.mean(axis=0)) # [4]

labels = ["R1_x", "R1_y", "R2_x", "R2_y"]
for name, m, r in zip(labels, mae_vec, rmse_vec):
    print(f"[TEST] {name}: MAE={m:.6f} | RMSE={r:.6f}")

# Optionally, also report per-touch (avg over x,y for each touch)
mae_touch = np.array([mae_vec[0:2].mean(), mae_vec[2:4].mean()])
rmse_touch = np.array([rmse_vec[0:2].mean(), rmse_vec[2:4].mean()])
print(f"[TEST] Touch1 avg: MAE={mae_touch[0]:.6f} | RMSE={rmse_touch[0]:.6f}")
print(f"[TEST] Touch2 avg: MAE={mae_touch[1]:.6f} | RMSE={rmse_touch[1]:.6f}")

# Save metrics to disk
with open(OUT_DIR / "test_metrics.txt", "w") as f:
    f.write(f"Overall MSE: {mse:.8f}\n")
    for name, m, r in zip(labels, mae_vec, rmse_vec):
        f.write(f"{name} MAE: {m:.8f} | RMSE: {r:.8f}\n")
    f.write(f"Touch1 (avg x,y) MAE: {mae_touch[0]:.8f} | RMSE: {rmse_touch[0]:.8f}\n")
    f.write(f"Touch2 (avg x,y) MAE: {mae_touch[1]:.8f} | RMSE: {rmse_touch[1]:.8f}\n")

pd.DataFrame({
    "coord": labels + ["Touch1_avg", "Touch2_avg"],
    "MAE":   list(mae_vec) + list(mae_touch),
    "RMSE":  list(rmse_vec) + list(rmse_touch),
}).to_csv(OUT_DIR / "test_metrics.csv", index=False)

# === Plot predicted vs true contact points ===
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(gts[:, 0], gts[:, 1], c='b', alpha=0.5, label='True')
axs[0].scatter(preds[:, 0], preds[:, 1], c='r', alpha=0.5, label='Pred')
axs[0].set_title("Robot 1 contact")
axs[0].set_aspect('equal')
axs[0].legend()

axs[1].scatter(gts[:, 2], gts[:, 3], c='b', alpha=0.5, label='True')
axs[1].scatter(preds[:, 2], preds[:, 3], c='r', alpha=0.5, label='Pred')
axs[1].set_title("Robot 2 contact")
axs[1].set_aspect('equal')
axs[1].legend()

for ax in axs:
    circ = plt.Circle((0, 0), 1.0, fill=False, color='k', lw=0.8)
    ax.add_patch(circ)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

plt.tight_layout()
plt.savefig(OUT_DIR / "pred_vs_true_contacts.png", dpi=200)
plt.close()

# Joint disk plot
fig, ax = plt.subplots(figsize=(6,6))
ax.add_patch(plt.Circle((0,0),1.0,fill=False,color="k",lw=1.0))
ax.scatter(gts[:,0], gts[:,1], s=10, c="tab:blue",  alpha=0.5, label="True")
ax.scatter(gts[:,2], gts[:,3], s=10, c="tab:blue",  alpha=0.5)
ax.scatter(preds[:,0], preds[:,1], s=10, c="tab:red", alpha=0.5, label="Pred")
ax.scatter(preds[:,2], preds[:,3], s=10, c="tab:red", alpha=0.5)
ax.set_aspect("equal"); ax.set_xlim([-1.1,1.1]); ax.set_ylim([-1.1,1.1])
ax.legend(); ax.set_title("True vs Predicted Double-Touch (joint)")
plt.tight_layout(); plt.savefig(OUT_DIR / "pred_vs_true_joint.png", dpi=220); plt.close()

print(f"[INFO] Saved plots & metrics → {OUT_DIR}/")
print("✅ Training complete.")
