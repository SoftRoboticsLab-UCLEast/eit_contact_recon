#!/usr/bin/env python3
"""
Fine-tune a SIM-trained HybridEITNet on REAL EIT data.

Pipeline:
- Load SIM-trained model checkpoint.
- Load REAL dataset (RealEITHybridDataset).
- Randomly split REAL dataset:
    - 50% for fine-tuning
    - 50% for held-out evaluation
- Evaluate SIM model on REAL eval split (baseline sim->real performance).
- Fine-tune on REAL train split.
- Evaluate fine-tuned model on REAL eval split.
- Save fine-tuned checkpoint.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import random

from train_real_eit_hybrid import (
    RealEITHybridDataset,
    HybridEITNet,
    CONFIG as REAL_CONFIG,
)

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4         # smaller LR for fine-tuning
BP_DROPOUT_PROB = 0.5        # match your sim training if using "hybrid"
INPUT_MODE = "hybrid"        # MUST match how the SIM model was trained

# Path to the SIM-trained checkpoint (adjust this!)
SIM_CKPT_PATH = Path("eit_dataset") / f"best_hybrid_model_{INPUT_MODE}.pt"

# Name for the fine-tuned checkpoint
FINETUNE_CKPT_NAME = f"finetuned_from_sim_{INPUT_MODE}.pt"

# For reproducible split
RANDOM_SEED = 1234


# =========================
# LOSSES & METRICS
# (same as your train scripts)
# =========================
def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()

    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    dice = 2.0 * intersection / union
    loss = 1.0 - dice
    return loss.mean()


def iou_metric(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection + eps
    iou = intersection / union
    return iou.mean().item()


def train_one_epoch(model, loader, optimizer, device, bp_dropout_prob, input_mode):
    model.train()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    running_iou = 0.0
    n_batches = 0

    for batch in loader:
        volt = batch["volt"].to(device)           # [B,K]
        bp = batch["bp"].to(device)               # [B,1,H,W]
        domain = batch["domain"].to(device)       # [B,1,H,W]
        mask = batch["mask"].to(device)           # [B,1,H,W]

        # Input mode logic
        if input_mode == "bp_only":
            volt_in = torch.zeros_like(volt)
            bp_in = bp
            effective_bp_dropout = 0.0
        elif input_mode == "volt_only":
            volt_in = volt
            bp_in = torch.zeros_like(bp)
            effective_bp_dropout = 0.0
        else:  # "hybrid"
            volt_in = volt
            bp_in = bp
            effective_bp_dropout = bp_dropout_prob

        optimizer.zero_grad()
        logits = model(volt_in, bp_in, domain, bp_dropout_prob=effective_bp_dropout)

        loss_bce = bce_loss_fn(logits, mask)
        loss_dice = dice_loss_with_logits(logits, mask)
        loss = 0.5 * loss_bce + 0.5 * loss_dice

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_iou += iou_metric(logits.detach(), mask)
        n_batches += 1

    return running_loss / n_batches, running_iou / n_batches


def eval_one_epoch(model, loader, device, input_mode):
    model.eval()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    running_iou = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            volt = batch["volt"].to(device)
            bp = batch["bp"].to(device)
            domain = batch["domain"].to(device)
            mask = batch["mask"].to(device)

            if input_mode == "bp_only":
                volt_in = torch.zeros_like(volt)
                bp_in = bp
                effective_bp_dropout = 0.0
            elif input_mode == "volt_only":
                volt_in = volt
                bp_in = torch.zeros_like(bp)
                effective_bp_dropout = 0.0
            else:
                volt_in = volt
                bp_in = bp
                effective_bp_dropout = 0.0

            logits = model(volt_in, bp_in, domain, bp_dropout_prob=effective_bp_dropout)

            loss_bce = bce_loss_fn(logits, mask)
            loss_dice = dice_loss_with_logits(logits, mask)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            running_loss += loss.item()
            running_iou += iou_metric(logits, mask)
            n_batches += 1

    return running_loss / n_batches, running_iou / n_batches


def load_baseline(baseline_path: str) -> np.ndarray:
    p = Path(baseline_path)
    if p.suffix.lower() == ".npy":
        v0 = np.load(p)
    else:
        v0 = np.loadtxt(p, delimiter=",")
    v0 = np.asarray(v0, dtype=float)
    if v0.ndim != 1:
        raise ValueError("Baseline must be 1D.")
    return v0.astype(np.float32)


def compute_valid_channel_mask_from_train(train_csv: str, baseline_full: np.ndarray) -> np.ndarray:
    """
    Match your previous logic:
    drop channel if (always zero across train data) AND (baseline is zero).
    """
    df = pd.read_csv(train_csv)
    eit_cols = [c for c in df.columns if c.startswith("eit_")]
    eit_cols.sort(key=lambda s: int(s.split("_")[1]))
    if len(eit_cols) == 0:
        raise RuntimeError("No eit_* columns found in TRAIN_CSV")

    if baseline_full.size != len(eit_cols):
        raise ValueError(
            f"Baseline length {baseline_full.size} != EIT cols {len(eit_cols)}"
        )

    vals = df[eit_cols].to_numpy(dtype=float)
    vals = np.where(np.isnan(vals), 0.0, vals)

    zero_across_train = np.all(vals == 0.0, axis=0)
    zero_in_baseline = (baseline_full == 0.0)

    drop = zero_across_train & zero_in_baseline
    valid = ~drop
    return valid.astype(bool)

# =========================
# MAIN
# =========================
def main():
    print(f"Using device: {DEVICE}")
    print(f"Input mode (SIM model): {INPUT_MODE}")
    print(f"SIM checkpoint: {SIM_CKPT_PATH}")

    if not SIM_CKPT_PATH.exists():
        raise FileNotFoundError(f"SIM checkpoint not found at {SIM_CKPT_PATH}")

    # --- REAL config ---
    data_root = REAL_CONFIG["DATA_ROOT"]
    csv_path = REAL_CONFIG["TRAIN_CSV"]
    baseline_path = REAL_CONFIG["BASELINE_PATH"]
    grid_size = REAL_CONFIG["GRID_SIZE"]
    n_el = REAL_CONFIG["N_EL"]
    mesh_h0 = REAL_CONFIG["MESH_H0"]

     # baseline + channel mask computed from TRAIN split only
    baseline_full = load_baseline(REAL_CONFIG["BASELINE_PATH"])
    valid_mask = compute_valid_channel_mask_from_train(REAL_CONFIG["TRAIN_CSV"], baseline_full)

    # --- REAL dataset (full) ---
    full_dataset = RealEITHybridDataset(
        csv_path=csv_path,
        data_root=data_root,
        baseline_full=baseline_full,
        valid_channel_mask=valid_mask,
        grid_size=grid_size,
        n_el=n_el,
        mesh_h0=mesh_h0,
    )
    n_total = len(full_dataset)
    print(f"Total REAL samples: {n_total}")

    # --- 50/50 split: fine-tune vs eval ---
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(n_total)
    split = n_total // 2

    train_indices = indices[:split]
    eval_indices = indices[split:]

    train_dataset = Subset(full_dataset, train_indices)
    eval_dataset = Subset(full_dataset, eval_indices)

    print(f"Fine-tune set: {len(train_dataset)} samples")
    print(f"Eval set:      {len(eval_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model: create with REAL volt_dim but load SIM weights ---
    volt_dim = full_dataset.mlp_channel_count  # e.g. 208
    img_size = (grid_size, grid_size)

    print(f"REAL volt_dim: {volt_dim}, grid: {img_size}")

    model = HybridEITNet(volt_dim=volt_dim, feat_channels=16)

    ckpt = torch.load(SIM_CKPT_PATH, map_location="cpu")
    print(
        f"Loaded SIM checkpoint from epoch {ckpt.get('epoch', '?')} "
        f"(SIM test IoU {ckpt.get('test_iou', 'N/A'):.4f})"
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE)

    # --- Optimizer for fine-tuning ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Baseline sim->real performance BEFORE fine-tune ---
    base_loss, base_iou = eval_one_epoch(model, eval_loader, DEVICE, INPUT_MODE)
    print("\n=== Baseline SIM->REAL performance (before fine-tune) ===")
    print(f"Loss: {base_loss:.4f}, IoU: {base_iou:.4f}")

    best_iou = base_iou
    ckpt_dir = Path(data_root)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    finetune_ckpt_path = ckpt_dir / FINETUNE_CKPT_NAME

    # --- Fine-tuning loop ---
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, DEVICE,
            bp_dropout_prob=BP_DROPOUT_PROB,
            input_mode=INPUT_MODE,
        )
        eval_loss, eval_iou = eval_one_epoch(
            model, eval_loader, DEVICE,
            input_mode=INPUT_MODE,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f}, IoU: {train_iou:.4f} | "
            f"Eval loss: {eval_loss:.4f}, IoU: {eval_iou:.4f}"
        )

        # Save best fine-tuned model on REAL eval set
        if eval_iou > best_iou:
            best_iou = eval_iou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_iou": eval_iou,
                    "base_sim2real_iou": base_iou,
                },
                finetune_ckpt_path,
            )
            print(f"  -> New best eval IoU: {eval_iou:.4f}, saved to {finetune_ckpt_path}")

    print("\nFine-tuning finished.")
    print(f"Baseline SIM->REAL IoU: {base_iou:.4f}")
    print(f"Best fine-tuned REAL eval IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
