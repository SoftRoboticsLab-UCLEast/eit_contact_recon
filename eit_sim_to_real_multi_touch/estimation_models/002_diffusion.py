#!/usr/bin/env python3
"""
Conditional diffusion 'realism layer' for EIT: ΔV_sim → ΔV_real-like.

Usage (example):
  python estimation_models/002_diffusion.py \
      --sim-csv sim/from_real_positions.csv \
      --real-csv training_dataset/training_dataset.csv \
      --sim-is-delta \
      --real-is-delta \
      --drop-real-zero-cols \
      --epochs 150 \
      --steps 400 \
      --out runs/diffusion_v2 \
      --translate-limit 100    # fast debug

Notes
- Trains in standardized space. Sampler standardizes x_s with SIM stats and
  returns standardized REAL; we then de-standardize with REAL stats for plots/CSV.
- If your CSVs are raw voltages, pass baselines with --sim-baseline-csv / --real-baseline-csv.
"""

from __future__ import annotations
import argparse, re, math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ------------------ IO helpers ------------------
NUM_SUFFIX = re.compile(r"(\d+)$")
def parse_idx(name: str):
    m = NUM_SUFFIX.search(name); return int(m.group(1)) if m else None

def find_eit_cols(df: pd.DataFrame) -> List[str]:
    cols=[]
    for c in df.columns:
        lc=c.lower()
        if lc=="eit_t": continue
        if lc.startswith("eit_") or lc.startswith("v"):
            if parse_idx(c) is not None:
                cols.append(c)
    if not cols: raise RuntimeError("No EIT columns found (eit_<int> or v<int>).")
    return cols

def to_numeric(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)

def drop_zero_cols_real(X: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
    keep = np.any(X != 0.0, axis=0)
    return X[:, keep], [n for n,k in zip(names, keep) if k]

def choose_label_cols(df: pd.DataFrame) -> List[str]:
    norm=["R1_x_norm","R1_y_norm","R2_x_norm","R2_y_norm"]
    raw =["x1","y1","x2","y2"]
    if all(c in df.columns for c in norm): return norm
    if all(c in df.columns for c in raw):  return raw
    raise RuntimeError("Labels not found: need (x1,y1,x2,y2) or *_norm")

def row_keys(df: pd.DataFrame, cols: List[str], prec=4) -> pd.Series:
    r = df[cols].astype(float).round(prec)
    return (r[cols[0]].astype(str)+"|"+r[cols[1]].astype(str)+"|"+
            r[cols[2]].astype(str)+"|"+r[cols[3]].astype(str))

def load_baseline_csv(path: Path) -> Tuple[np.ndarray, List[int]]:
    try:
        bdf = pd.read_csv(path, header=0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        vals = bdf.to_numpy(np.float32).flatten()
        idxs = [parse_idx(c) for c in bdf.columns]
        if all(i is not None for i in idxs): return vals, idxs
        return vals, []
    except Exception:
        bdf = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return bdf.to_numpy(np.float32).flatten(), []

def align_by_intersection(sim_cols: list[str], real_cols: list[str], min_common: int = 100):
    """Try small integer shifts on SIM indices; return aligned name lists and numeric index list."""
    def build_map(cols):
        out = {}
        for c in cols:
            i = parse_idx(c)
            if i is not None:
                out[i] = c
        return out
    sim_map_raw  = build_map(sim_cols)
    real_map_raw = build_map(real_cols)

    def try_shift(shift: int):
        sm = sim_map_raw if shift==0 else {i+shift:n for i,n in sim_map_raw.items()}
        com = sorted(set(sm.keys()).intersection(real_map_raw.keys()))
        return [sm[i] for i in com], [real_map_raw[i] for i in com], com

    cands = []
    for k in (-2,-1,0,1,2):
        ssel, rsel, com = try_shift(k)
        cands.append((len(com), k, ssel, rsel, com))
    cands.sort(reverse=True, key=lambda x: x[0])
    best_n, best_shift, sim_sel, real_sel, common_idx = cands[0]

    sim_idx_list  = sorted(sim_map_raw.keys())[:10]
    real_idx_list = sorted(real_map_raw.keys())[:10]
    print(f"[ALIGN] SIM idx sample:  {sim_idx_list} … (total {len(sim_map_raw)})")
    print(f"[ALIGN] REAL idx sample: {real_idx_list} … (total {len(real_map_raw)})")
    print(f"[ALIGN] Best overlap={best_n} with SIM index shift={best_shift}")

    if best_n < min_common:
        raise RuntimeError(f"Too few common channels after alignment: {best_n}/{min_common}")
    return sim_sel, real_sel, common_idx

# ------------------ Diffusion utilities ------------------
def cosine_beta_schedule(T, s=0.008):
    # https://arxiv.org/abs/2102.09672
    t = torch.linspace(0, T, T+1)
    f = torch.cos(((t/T + s)/(1+s)) * math.pi/2)**2
    alphas_bar = f/f[0]
    betas = 1 - (alphas_bar[1:]/alphas_bar[:-1])
    return betas.clamp(1e-5, 0.999)

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim//2
    freqs = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0)/half))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim%2==1: emb = F.pad(emb, (0,1))
    return emb

def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    L = x.size(-1)
    Lp = int(math.ceil(L / multiple) * multiple)
    pad = Lp - L
    if pad > 0: x = F.pad(x, (0, pad))
    return x, pad

def crop_right(x: torch.Tensor, orig_len: int) -> torch.Tensor:
    return x[..., :orig_len]

def align_len_for_concat(h: torch.Tensor, skip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    Lh, Ls = h.size(-1), skip.size(-1)
    if Lh == Ls: return h, skip
    if Lh < Ls:  return h, skip[..., :Lh]
    return h[..., :Ls], skip

# ------------------ 1D Conditional U-Net ------------------
class FiLM(nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super().__init__()
        self.to_scale = nn.Linear(emb_dim, feat_dim)
        self.to_shift = nn.Linear(emb_dim, feat_dim)
    def forward(self, h, emb):
        s = self.to_scale(emb).unsqueeze(-1)
        b = self.to_shift(emb).unsqueeze(-1)
        return h * (1+s) + b

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.film = FiLM(out_ch, emb_dim)
        self.down = nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1)
    def forward(self, x, emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.act(self.norm2(self.conv2(h)))
        h = self.film(h, emb)
        d = self.down(h)
        return h, d

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1)
        self.conv1 = nn.Conv1d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.film = FiLM(out_ch, emb_dim)
    def forward(self, x, skip, emb):
        h = self.up(x)
        h, skip = align_len_for_concat(h, skip)
        h = torch.cat([h, skip], dim=1)
        h = self.act(self.norm1(self.conv1(h)))
        h = self.act(self.norm2(self.conv2(h)))
        h = self.film(h, emb)
        return h

class UNet1D(nn.Module):
    def __init__(self, base=128, emb_dim=128, cond_channels=1):
        super().__init__()
        self.emb_dim = emb_dim
        in_ch = 1 + cond_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*2), nn.SiLU(),
            nn.Linear(emb_dim*2, emb_dim)
        )
        self.in_conv = nn.Conv1d(in_ch, base, 3, padding=1)
        self.down1 = DownBlock(base, base*2, emb_dim)
        self.down2 = DownBlock(base*2, base*4, emb_dim)

        self.mid1 = nn.Conv1d(base*4, base*4, 3, padding=1)
        self.mid2 = nn.Conv1d(base*4, base*4, 3, padding=1)
        self.mid_norm1 = nn.GroupNorm(8, base*4)
        self.mid_norm2 = nn.GroupNorm(8, base*4)
        self.mid_act = nn.SiLU()
        self.mid_film = FiLM(base*4, emb_dim)

        self.up2 = UpBlock(base*4, base*2, skip_ch=base*4, emb_dim=emb_dim)
        self.up1 = UpBlock(base*2, base,   skip_ch=base*2, emb_dim=emb_dim)
        self.out = nn.Conv1d(base, 1, 3, padding=1)

    def forward(self, x_t, x_s, t):
        # x_t, x_s: [B, C] in STANDARDIZED space
        B, C = x_t.shape
        xt = x_t.unsqueeze(1)
        xs = x_s.unsqueeze(1)
        xt, _ = pad_to_multiple(xt, 4)
        xs, _ = pad_to_multiple(xs, 4)
        h = torch.cat([xt, xs], dim=1)

        temb = timestep_embedding(t, self.emb_dim)
        temb = self.time_mlp(temb)

        h = self.in_conv(h)
        s1, d1 = self.down1(h, temb)
        s2, d2 = self.down2(d1, temb)

        m = self.mid_act(self.mid_norm1(self.mid1(d2)))
        m = self.mid_act(self.mid_norm2(self.mid2(m)))
        m = self.mid_film(m, temb)

        u2 = self.up2(m, s2, temb)
        u1 = self.up1(u2, s1, temb)
        out = self.out(u1).squeeze(1)
        out = crop_right(out, C)
        return out

# ------------------ physics helpers ------------------
def physics_penalty(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if A is None: return torch.tensor(0.0, device=x.device)
    Ax = A @ x.T
    return (Ax**2).mean()

# ------------------ training / sampling ------------------
@torch.no_grad()
def predict_x0_from_eps(x_t, eps_hat, alpha_bar_t):
    return (x_t - torch.sqrt(1 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)

@torch.no_grad()
def ddpm_sample(model, x_s_raw, betas, sim_mean, sim_std, real_mean, real_std,
                phys_A: Optional[torch.Tensor]=None, phys_w: float=0.0):
    """
    x_s_raw: numpy 1D array [C] (ΔV_sim RAW). Returns RAW ΔV_real-like [C].
    """
    device = next(model.parameters()).device
    # --- standardize SIM condition
    xs_std = (x_s_raw - sim_mean) / sim_std
    xs_std = np.nan_to_num(xs_std, 0.0, 0.0, 0.0)

    xs = torch.from_numpy(xs_std).to(device).unsqueeze(0)  # [1,C]
    T = len(betas)
    alphas = 1.0 - betas
    a_bar = torch.cumprod(alphas, dim=0)

    x = torch.randn(1, xs.size(1), device=device)  # start at noise (std REAL space)

    for t in reversed(range(T)):
        tt = torch.full((1,), t, device=device, dtype=torch.long)
        eps_hat = model(x, xs, tt)
        ab_t = a_bar[t].unsqueeze(0)
        x0_hat = predict_x0_from_eps(x, eps_hat, ab_t)
        if t > 0:
            mean = torch.sqrt(alphas[t-1]) * x0_hat + torch.sqrt(1 - alphas[t-1]) * eps_hat
            z = torch.randn_like(x)
            x = mean + torch.sqrt(betas[t]) * z
        else:
            x = x0_hat

        if phys_A is not None and phys_w > 0:
            Ax = phys_A @ x.T
            grad = (phys_A.T @ Ax).T
            x = x - 0.1 * phys_w * grad

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    x_std = x.squeeze(0).cpu().numpy()                 # standardized REAL
    x_raw = (x_std * real_std) + real_mean             # back to RAW
    x_raw = np.nan_to_num(x_raw, 0.0, 0.0, 0.0)
    return x_raw

@torch.no_grad()
def ddpm_sample_batch(model,
                      Xs_raw: np.ndarray,                 # [N,C] RAW ΔV_sim
                      betas: torch.Tensor,
                      sim_mean: np.ndarray, sim_std: np.ndarray,
                      real_mean: np.ndarray, real_std: np.ndarray,
                      phys_A: Optional[torch.Tensor] = None,
                      phys_w: float = 0.0,
                      batch_size: int = 512) -> np.ndarray:
    """
    Vectorized DDPM sampling for speed. Returns RAW ΔV_real-like with shape [N,C].
    """
    device = next(model.parameters()).device
    N, C = Xs_raw.shape

    # standardize SIM → Xs_std [N,C]
    Xs_std = (Xs_raw - sim_mean[None, :]) / sim_std[None, :]
    Xs_std = np.nan_to_num(Xs_std, 0.0, 0.0, 0.0)

    alphas = 1.0 - betas
    a_bar = torch.cumprod(alphas, dim=0)

    out_raw_chunks = []
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        xs = torch.from_numpy(Xs_std[s:e]).to(device)  # [B,C]
        B = xs.shape[0]

        x = torch.randn(B, xs.size(1), device=device)  # standardized REAL

        for t in reversed(range(len(betas))):
            tt = torch.full((B,), t, device=device, dtype=torch.long)
            eps_hat = model(x, xs, tt)
            ab_t = a_bar[t].view(1, 1)
            x0_hat = (x - torch.sqrt(1 - ab_t) * eps_hat) / torch.sqrt(ab_t)

            if t > 0:
                mean = torch.sqrt(alphas[t-1]) * x0_hat + torch.sqrt(1 - alphas[t-1]) * eps_hat
                z = torch.randn_like(x)
                x = mean + torch.sqrt(betas[t]) * z
            else:
                x = x0_hat

            if phys_A is not None and phys_w > 0.0:
                Ax = phys_A @ x.T
                grad = (phys_A.T @ Ax).T
                x = x - 0.1 * phys_w * grad

            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # de-standardize to RAW REAL units
        x_std = x.cpu().numpy()                                  # [B,C]
        x_raw = x_std * real_std[None, :] + real_mean[None, :]   # [B,C]
        x_raw = np.nan_to_num(x_raw, 0.0, 0.0, 0.0)
        out_raw_chunks.append(x_raw)

    return np.concatenate(out_raw_chunks, axis=0)

# ------------------ main pipeline ------------------
def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--sim-csv", required=True)
    ap.add_argument("--real-csv", required=True)
    ap.add_argument("--sim-is-delta", action="store_true")
    ap.add_argument("--real-is-delta", action="store_true")
    ap.add_argument("--sim-baseline-csv", default=None)
    ap.add_argument("--real-baseline-csv", default=None)
    ap.add_argument("--drop-real-zero-cols", action="store_true")
    # Train/eval
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    # Diffusion
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--base-ch", type=int, default=128)
    # Physics
    ap.add_argument("--phys-A-npy", default=None)
    ap.add_argument("--phys-w", type=float, default=0.0)
    # Output
    ap.add_argument("--out", default="runs/diffusion_v2")
    ap.add_argument("--n-sample-plots", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--translate-limit", type=int, default=0,
                    help="If >0, only translate the first N SIM rows (debug).")
    ap.add_argument("--sample-batch", type=int, default=512,
                    help="Batch size for batched DDPM sampling (debug speed-up).")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Load CSVs
    dfS = pd.read_csv(args.sim_csv)
    dfR = pd.read_csv(args.real_csv)

    # Keys from labels (to pair rows)
    lblS = choose_label_cols(dfS)
    lblR = choose_label_cols(dfR)
    keyS = row_keys(dfS, lblS).values
    keyR = row_keys(dfR, lblR).values
    dfS = dfS.copy(); dfS["__key__"]=keyS
    dfR = dfR.copy(); dfR["__key__"]=keyR

    # EIT columns & REAL zero-drop
    colsS_all = find_eit_cols(dfS)
    colsR_all = find_eit_cols(dfR)
    XR_all = to_numeric(dfR, colsR_all)
    if args.drop_real_zero_cols:
        XR_all, colsR_all = drop_zero_cols_real(XR_all, colsR_all)

    # Align by intersection
    sim_sel, real_sel, common_idx = align_by_intersection(colsS_all, colsR_all)
    XS = to_numeric(dfS, sim_sel)
    XR = to_numeric(dfR, real_sel)

    # Build ΔV
    if args.sim_is_delta:
        dVS = XS
    else:
        if not args.sim_baseline_csv: raise RuntimeError("SIM raw → need --sim-baseline-csv")
        v0S_vals, v0S_idx = load_baseline_csv(Path(args.sim_baseline_csv))
        if v0S_idx:
            pos = {i:j for j,i in enumerate(v0S_idx)}
            v0S = np.array([v0S_vals[pos[i]] if i in pos else 0.0 for i in common_idx], np.float32)
        else:
            if len(v0S_vals) != XS.shape[1]: raise RuntimeError("SIM baseline length mismatch")
            v0S = v0S_vals.astype(np.float32)
        dVS = XS - v0S[None,:]

    if args.real_is_delta:
        dVR = XR
    else:
        if not args.real_baseline_csv: raise RuntimeError("REAL raw → need --real-baseline-csv")
        v0R_vals, v0R_idx = load_baseline_csv(Path(args.real_baseline_csv))
        if v0R_idx:
            pos = {i:j for j,i in enumerate(v0R_idx)}
            v0R = np.array([v0R_vals[pos[i]] if i in pos else 0.0 for i in common_idx], np.float32)
        else:
            if len(v0R_vals) != XR.shape[1]: raise RuntimeError("REAL baseline length mismatch")
            v0R = v0R_vals.astype(np.float32)
        dVR = XR - v0R[None,:]

    # Pair rows by identical keys
    keys_common = pd.Index(dfS["__key__"]).intersection(dfR["__key__"])
    if len(keys_common)==0:
        raise RuntimeError("No paired rows found. Use sim generated from real positions.")
    sim_idx = np.array([int(dfS.index[dfS["__key__"]==k][0]) for k in keys_common], int)
    real_idx= np.array([int(dfR.index[dfR["__key__"]==k][0]) for k in keys_common], int)

    Xs = dVS[sim_idx]  # RAW ΔV_sim aligned
    Xr = dVR[real_idx] # RAW ΔV_real aligned

    # Splits
    N = len(Xs)
    idx = np.arange(N); rng.shuffle(idx)
    n_te = int(args.test_frac*N)
    n_va = int(args.val_frac*N)
    te_ids = idx[:n_te]; va_ids = idx[n_te:n_te+n_va]; tr_ids = idx[n_te+n_va:]
    Xs_tr, Xr_tr = Xs[tr_ids], Xr[tr_ids]
    Xs_va, Xr_va = Xs[va_ids], Xr[va_ids]
    Xs_te, Xr_te = Xs[te_ids], Xr[te_ids]

    # Preprocessing stats (SAVE!)
    sim_mean  = (Xs_tr.mean(axis=0)).astype(np.float32)
    sim_std   = (Xs_tr.std(axis=0) + 1e-6).astype(np.float32)
    real_mean = (Xr_tr.mean(axis=0)).astype(np.float32)
    real_std  = (Xr_tr.std(axis=0) + 1e-6).astype(np.float32)
    np.savez(str(outdir / "preproc_stats.npz"),
             sim_mean=sim_mean, sim_std=sim_std,
             real_mean=real_mean, real_std=real_std,
             sim_cols=np.array(sim_sel, dtype=object),
             real_cols=np.array(real_sel, dtype=object),
             common_idx=np.array(common_idx, dtype=int))

    # Standardize for TRAIN/VAL only
    Xs_tr_std = np.nan_to_num((Xs_tr - sim_mean)/sim_std, 0.0, 0.0, 0.0)
    Xr_tr_std = np.nan_to_num((Xr_tr - real_mean)/real_std, 0.0, 0.0, 0.0)
    Xs_va_std = np.nan_to_num((Xs_va - sim_mean)/sim_std, 0.0, 0.0, 0.0)
    Xr_va_std = np.nan_to_num((Xr_va - real_mean)/real_std, 0.0, 0.0, 0.0)

    device = torch.device(args.device)
    betas = cosine_beta_schedule(args.steps).to(device)
    alphas = 1.0 - betas
    a_bar = torch.cumprod(alphas, dim=0)

    model = UNet1D(base=args.base_ch, emb_dim=args.emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    phys_A = None
    if args.phys_A_npy:
        phys_A = torch.from_numpy(np.load(args.phys_A_npy).astype(np.float32)).to(device)

    # Training loop
    bs = args.batch
    rng_np = np.random.default_rng(args.seed)

    def iterate_batches(A, B):
        n = len(A); order = np.arange(n); rng_np.shuffle(order)
        for s in range(0, n, bs):
            idxb = order[s:s+bs]
            yield torch.from_numpy(A[idxb]).to(device), torch.from_numpy(B[idxb]).to(device)

    for ep in range(1, args.epochs+1):
        model.train(); losses=[]
        for x_s, x_r in iterate_batches(Xs_tr_std, Xr_tr_std):
            B = x_s.shape[0]
            t = torch.randint(0, len(betas), (B,), device=device)
            ab_t = a_bar[t].unsqueeze(1)
            eps = torch.randn_like(x_r)
            x_t = torch.sqrt(ab_t)*x_r + torch.sqrt(1-ab_t)*eps

            eps_hat = model(x_t, x_s, t)
            loss = (eps_hat - eps).pow(2).mean()

            if phys_A is not None and args.phys_w>0:
                x0_hat = predict_x0_from_eps(x_t, eps_hat, ab_t)
                loss = loss + args.phys_w * physics_penalty(phys_A, x0_hat)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        # simple validation correlation in standardized space
        model.eval(); corr_list=[]
        with torch.no_grad():
            for x_s, x_r in iterate_batches(Xs_va_std, Xr_va_std):
                B = x_s.shape[0]
                t = torch.randint(0, len(betas), (B,), device=device)
                ab_t = a_bar[t].unsqueeze(1)
                eps = torch.randn_like(x_r)
                x_t = torch.sqrt(ab_t)*x_r + torch.sqrt(1-ab_t)*eps
                eps_hat = model(x_t, x_s, t)
                x0_hat = predict_x0_from_eps(x_t, eps_hat, ab_t)
                y0 = x0_hat - x0_hat.mean(1, keepdim=True)
                r0 = x_r    - x_r.mean(1, keepdim=True)
                num = (y0*r0).sum(1)
                den = torch.sqrt((y0*y0).sum(1)*(r0*r0).sum(1)+1e-8)
                corr = (num/(den+1e-8)).mean().item()
                corr_list.append(corr)
        print(f"[{ep:03d}] train_mse={np.mean(losses):.5f} | val_corr≈{np.mean(corr_list):.4f}")

    # Save checkpoint
    ckpt = {
        "state": model.state_dict(),
        "betas": betas.detach().cpu().numpy(),
        "base": args.base_ch,
        "emb_dim": args.emb_dim,
    }
    torch.save(ckpt, outdir/"diffusion_mapper.pt")
    print(f"✅ Saved model → {outdir/'diffusion_mapper.pt'}")

    # --- Translate SIM rows (debug-friendly: limit + batch) ---
    model.eval()

    # indices to translate
    if args.translate_limit and args.translate_limit > 0:
        idx_all = np.arange(len(dVS))[:args.translate_limit]
        print(f"[INFO] Translating only first {len(idx_all)} rows for debug.")
    else:
        idx_all = np.arange(len(dVS))
        print(f"[INFO] Translating ALL {len(idx_all)} rows.")

    Xs_raw_to_translate = dVS[idx_all]  # RAW ΔV_sim (aligned channels) [N,C]

    translated = ddpm_sample_batch(
        model,
        Xs_raw_to_translate,
        betas,
        sim_mean=sim_mean, sim_std=sim_std,
        real_mean=real_mean, real_std=real_std,
        phys_A=phys_A, phys_w=args.phys_w,
        batch_size=args.sample_batch
    )  # [N,C]

    # Write translated CSV (subset or full)
    df_out = dfS.drop(columns=["__key__"]).iloc[idx_all].copy()
    for j, col in enumerate(sim_sel):
        df_out[col] = translated[:, j]
    out_csv = outdir / ("translated_deltaV_subset.csv" if (args.translate_limit and args.translate_limit > 0)
                        else "translated_deltaV.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Wrote translated ΔV → {out_csv}")

    # --- Quick debug plots from subset ---
    n_plot = min(args.n_sample_plots, len(idx_all))
    # build map: sim row index -> real row index (from earlier pairing)
    sim_to_real = {s: r for s, r in zip(sim_idx, real_idx)}

    for k in range(n_plot):
        sim_row = int(idx_all[k])
        if sim_row not in sim_to_real:
            continue  # skip if no paired real
        real_row = sim_to_real[sim_row]

        sim_v  = dVS[sim_row]
        real_v = dVR[real_row]
        pred_v = translated[k]  # same order as idx_all

        plt.figure(figsize=(10, 6))
        plt.plot(sim_v,  lw=1.0, label="ΔV_sim")
        plt.plot(real_v, lw=1.0, label="ΔV_real")
        plt.plot(pred_v, lw=1.2, label="Diffusion(ΔV_sim)")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / f"pair_subset_{k+1:02d}.png", dpi=220)
        plt.close()

if __name__ == "__main__":
    main()
