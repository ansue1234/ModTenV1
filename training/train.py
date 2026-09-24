#!/usr/bin/env python3
"""
train.py -- train the magnetic self-pose MLP (unit-direction regression).

Trains an MLP that maps a unit-normalised, de-biased magnetometer reading
(mx, my, mz) of segment i+1 -- taken while the coil of segment i is energised --
to the UNIT DIRECTION VECTOR d = (dx, dy, dz) of the joint between the two
segments.  Hyper-parameters are read from an Optuna best-params JSON
(configs/mlp_dir_optuna_best_params.json is the set used in the paper).

Input CSV (produced by data_collection/prepare_dataset.py) must contain
    X : mx_uT_debias, my_uT_debias, mz_uT_debias      (unit-normalised)
    y : Theta_deg  = azimuth   (deg, -180..180, about +Z)
        lambda_deg = colatitude (deg, 0 = straight/upright, measured from +Z)
Angles are converted to a unit direction with the COLATITUDE convention
    dx = sin(lambda) * cos(Theta)
    dy = sin(lambda) * sin(Theta)
    dz = cos(lambda)
which is exactly what ros_ws/.../pose_estimator_node.py inverts at run time.

Pipeline
- X is used RAW (no standardisation -- the CSV rows are already unit vectors)
- a holdout split is carved off first and never used for model selection
- the model is trained on the remainder with an internal validation split
  (loss = 1 - cos(pred, target); AdamW)
- optional inverse-frequency 2-D histogram weighting on (Theta, lambda),
  applied only to the training loss (off in the paper configuration)
- per-epoch direction / angle metrics on train + val, optional TensorBoard / W&B
- checkpoints best.pt, latest.pt, epoch_XXXX.pt and the final artifact
  final_model.pt  (consumed by evaluate.py and export_onnx.py)

Typical use (from the training/ folder):
    python train.py --csv data/train.csv \
        --optuna_json configs/mlp_dir_optuna_best_params.json --epochs 600

Requires:  pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# make `arch` importable when the script is launched from another cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch.mlp import MLP  # noqa: E402

ANGLE_CONVENTION = "colatitude"  # stored in final_model.pt; evaluate.py / export_onnx.py read it


# ----------------------------
# Repro + device
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------
# Geometry helpers + metrics
# ----------------------------
def wrap_to_180(deg: np.ndarray) -> np.ndarray:
    """Wrap azimuth differences to [-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def angles_deg_to_unit_dir(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    """
    Convert spherical angles to unit direction vector.

    theta : azimuth (degrees), any range, measured from +X toward +Y
    phi   : colatitude (degrees), 0 = +Z pole, 180 = -Z pole

    Cartesian convention:
        dx = sin(phi) * cos(theta)
        dy = sin(phi) * sin(theta)
        dz = cos(phi)
    """
    th = np.deg2rad(theta_deg.astype(np.float64))
    ph = np.deg2rad(phi_deg.astype(np.float64))
    sin_ph = np.sin(ph)
    x = sin_ph * np.cos(th)
    y = sin_ph * np.sin(th)
    z = np.cos(ph)
    d = np.stack([x, y, z], axis=-1)
    n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    return (d / n).astype(np.float32)


def unit_dir_to_angles_deg(d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert unit direction vectors back to (theta, phi) in degrees.

    Returns
    -------
    theta : azimuth in degrees (range -180..180)
    phi   : colatitude in degrees (range 0..180)
    """
    d = d.astype(np.float64)
    n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    d = d / n
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    theta = np.degrees(np.arctan2(dy, dx))                        # azimuth: -180..180
    phi   = np.degrees(np.arccos(np.clip(dz, -1.0, 1.0)))         # colatitude: 0..180
    return theta.astype(np.float64), phi.astype(np.float64)


def direction_angle_metrics(y_true_dir: np.ndarray, y_pred_dir: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    yt = y_true_dir.astype(np.float64)
    yp = y_pred_dir.astype(np.float64)

    yt_u = yt / (np.linalg.norm(yt, axis=1, keepdims=True) + eps)
    yp_u = yp / (np.linalg.norm(yp, axis=1, keepdims=True) + eps)

    dot = np.sum(yt_u * yp_u, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    ang_err_deg = np.degrees(np.arccos(dot))

    th_t, ph_t = unit_dir_to_angles_deg(yt_u.astype(np.float32))
    th_p, ph_p = unit_dir_to_angles_deg(yp_u.astype(np.float32))

    # Azimuth difference is circular; colatitude difference is linear (0..180)
    th_diff = wrap_to_180(th_p - th_t)
    ph_diff = ph_p - ph_t          # simple difference; no wrapping needed for colatitude

    # R2 on direction components
    ss_res = np.sum((yt_u - yp_u) ** 2, axis=0)
    ss_tot = np.sum((yt_u - np.mean(yt_u, axis=0)) ** 2, axis=0) + eps
    r2_xyz = 1.0 - ss_res / ss_tot

    # R2 for colatitude (non-circular, linear range 0..180)
    ss_res_ph = float(np.sum((ph_t - ph_p) ** 2))
    ss_tot_ph = float(np.sum((ph_t - np.mean(ph_t)) ** 2) + eps)
    r2_phi = float(1.0 - ss_res_ph / ss_tot_ph)

    return {
        "dir_mae_deg":   float(np.mean(np.abs(ang_err_deg))),
        "dir_rmse_deg":  float(np.sqrt(np.mean(ang_err_deg**2))),
        "dir_p95_deg":   float(np.quantile(ang_err_deg, 0.95)),
        "rmse_theta":    float(np.sqrt(np.mean(th_diff**2))),
        "rmse_phi":      float(np.sqrt(np.mean(ph_diff**2))),
        "mae_theta":     float(np.mean(np.abs(th_diff))),
        "mae_phi":       float(np.mean(np.abs(ph_diff))),
        "rmse_mean":     float(0.5 * (np.sqrt(np.mean(th_diff**2)) + np.sqrt(np.mean(ph_diff**2)))),
        "mae_mean":      float(0.5 * (np.mean(np.abs(th_diff)) + np.mean(np.abs(ph_diff)))),
        "r2_dx":         float(r2_xyz[0]),
        "r2_dy":         float(r2_xyz[1]),
        "r2_dz":         float(r2_xyz[2]),
        "r2_dir_mean":   float(np.mean(r2_xyz)),
        "r2_phi":        r2_phi,
    }


# ----------------------------
# 2D weighting in (Theta, phi)
# ----------------------------
def hist2d_inverse_freq_weights(
    y_train_angles: np.ndarray,    # (N, 2): [:, 0]=theta, [:, 1]=phi
    y_query_angles: np.ndarray,
    bins_theta: int,
    bins_lambda: int,              # parameter name kept for JSON compatibility; controls phi bins
    eps: float,
    clip: float,
    power: float,
) -> np.ndarray:
    th_tr = np.asarray(y_train_angles[:, 0], dtype=np.float64)
    ph_tr = np.asarray(y_train_angles[:, 1], dtype=np.float64)   # colatitude
    th_q  = np.asarray(y_query_angles[:, 0], dtype=np.float64)
    ph_q  = np.asarray(y_query_angles[:, 1], dtype=np.float64)

    th_min, th_max = np.nanmin(th_tr), np.nanmax(th_tr)
    ph_min, ph_max = np.nanmin(ph_tr), np.nanmax(ph_tr)

    if not (np.isfinite(th_min) and np.isfinite(th_max) and th_max > th_min):
        return np.ones((len(y_query_angles),), dtype=np.float32)
    if not (np.isfinite(ph_min) and np.isfinite(ph_max) and ph_max > ph_min):
        return np.ones((len(y_query_angles),), dtype=np.float32)

    th_edges = np.linspace(th_min, th_max, int(bins_theta) + 1)
    ph_edges = np.linspace(ph_min, ph_max, int(bins_lambda) + 1)

    counts2d, _, _ = np.histogram2d(th_tr, ph_tr, bins=[th_edges, ph_edges])

    th_idx = np.digitize(th_q, th_edges, right=False) - 1
    ph_idx = np.digitize(ph_q, ph_edges, right=False) - 1
    th_idx = np.clip(th_idx, 0, counts2d.shape[0] - 1)
    ph_idx = np.clip(ph_idx, 0, counts2d.shape[1] - 1)

    c = counts2d[th_idx, ph_idx].astype(np.float64)
    w = 1.0 / (np.power(c + eps, power))
    w = w / (np.mean(w) + 1e-12)
    w = np.clip(w, 1.0 / clip, clip)
    return w.astype(np.float32)


# ----------------------------
# Data loaders
# ----------------------------
def make_loader(
    X: np.ndarray,
    Y_dir: np.ndarray,
    W: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> torch.utils.data.DataLoader:
    X_t = torch.from_numpy(X).float().to(device)
    Y_t = torch.from_numpy(Y_dir).float().to(device)

    if W is None:
        ds = torch.utils.data.TensorDataset(X_t, Y_t)
    else:
        W_t = torch.from_numpy(W).float().to(device)
        ds = torch.utils.data.TensorDataset(X_t, Y_t, W_t)

    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def predict_numpy(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).float().to(device)
        outs.append(model(xb).detach().cpu().numpy())
    return np.vstack(outs) if outs else np.zeros((0, 3), dtype=np.float32)


# ----------------------------
# Loss
# ----------------------------
def cosine_or_hybrid_loss(
    pred_raw: torch.Tensor,   # (B,3)
    yb_dir: torch.Tensor,     # (B,3) unit
    loss_type: str,
    alpha: float,
    huber_beta: float,
) -> torch.Tensor:
    pred_unit = pred_raw / pred_raw.norm(dim=1, keepdim=True).clamp_min(1e-9)
    y_unit = yb_dir / yb_dir.norm(dim=1, keepdim=True).clamp_min(1e-9)

    dot = (pred_unit * y_unit).sum(dim=1).clamp(-1.0, 1.0)
    cos_per = 1.0 - dot

    if loss_type == "cosine":
        return cos_per

    huber_el = F.smooth_l1_loss(pred_unit, y_unit, reduction="none", beta=huber_beta)
    huber_per = huber_el.mean(dim=1)
    return cos_per + float(alpha) * huber_per


# ----------------------------
# Logging (W&B)
# ----------------------------
def maybe_import_wandb(use_wandb: bool):
    if not use_wandb:
        return None
    import wandb  # type: ignore
    return wandb


# ----------------------------
# Checkpoints
# ----------------------------
def save_checkpoint(path: str, model: nn.Module, opt: torch.optim.Optimizer, epoch: int, best_val: float, meta: Dict[str, object]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optim_state": opt.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "meta": meta,
        },
        path,
    )


# ----------------------------
# Parse Optuna JSON -> params
# ----------------------------
def model_params_from_optuna(best: Dict[str, object]) -> Dict[str, object]:
    n_layers = int(best["n_layers"])
    hidden_dims: List[int] = [int(best[f"width_{i}"]) for i in range(n_layers)]

    activation = str(best.get("activation", "gelu"))
    dropout = float(best.get("dropout", 0.0))
    use_batchnorm = bool(best.get("use_batchnorm", False))
    leaky_slope = float(best.get("leaky_slope", 0.01))

    return {
        "hidden_dims": hidden_dims,
        "activation": activation,
        "dropout": dropout,
        "use_batchnorm": use_batchnorm,
        "leaky_slope": leaky_slope,
        "out_dim": 3,
    }


def training_params_from_optuna(best: Dict[str, object]) -> Dict[str, object]:
    lr = float(best.get("lr", 1e-3))
    weight_decay = float(best.get("weight_decay", 1e-4))
    batch_size = int(best.get("batch_size", 1024))
    return {"lr": lr, "weight_decay": weight_decay, "batch_size": batch_size}


def weight_params_from_optuna(best: Dict[str, object]) -> Dict[str, object]:
    use_weight_2d = bool(best.get("use_weight_2d", False))
    return {
        "use_weight_2d": use_weight_2d,
        "weight_bins_theta": int(best.get("weight_bins_theta", 30)),
        "weight_bins_lambda": int(best.get("weight_bins_lambda", 30)),  # reused for phi bins
        "weight_power": float(best.get("weight_power", 1.0)),
        "weight_clip": float(best.get("weight_clip", 20.0)),
        "weight_eps": 1e-6,
    }


def loss_params_from_optuna(best: Dict[str, object]) -> Dict[str, object]:
    loss_type = str(best.get("loss_type", "hybrid"))
    alpha = float(best.get("alpha", 0.1))
    huber_beta = float(best.get("huber_beta", 0.1))
    if loss_type == "cosine":
        alpha = 0.0
    return {"loss_type": loss_type, "alpha": alpha, "huber_beta": huber_beta}


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainConfig:
    # splits
    holdout_frac: float = 0.15
    internal_val_frac: float = 0.10
    seed: int = 42

    # training budget
    max_epochs: int = 600
    early_stop_patience: int = 10**9
    min_delta: float = 1e-6

    # checkpoints
    out_dir: str = "runs"
    save_checkpoints: bool = True
    save_every_epochs: int = 25
    save_latest: bool = True

    # logging
    use_tensorboard: bool = False
    tb_root: str = "tb_logs"
    use_wandb: bool = False
    wandb_project: str = "mlp-final-dir"
    wandb_entity: Optional[str] = None
    wandb_tags: Tuple[str, ...] = ()

    # weighting override
    force_use_weight_2d: Optional[bool] = None  # None => follow JSON


# ----------------------------
# Train final model
# ----------------------------
def train_from_csv_and_optuna_json(csv_path: str, optuna_json_path: str, cfg: TrainConfig) -> Dict[str, object]:
    set_seed(cfg.seed)
    device = device_auto()
    wandb = maybe_import_wandb(cfg.use_wandb)

    df = pd.read_csv(csv_path)

    x_cols = ["mx_uT_debias", "my_uT_debias", "mz_uT_debias"]
    y_cols_angles = ["Theta_deg", "lambda_deg"]   # azimuth, colatitude (0..180 deg)

    missing = [c for c in (x_cols + y_cols_angles) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_use = df[x_cols + y_cols_angles].dropna().copy()
    X_all = df_use[x_cols].to_numpy(dtype=np.float32)
    y_angles_all = df_use[y_cols_angles].to_numpy(dtype=np.float32)
    # angles_deg_to_unit_dir now expects (theta, phi=colatitude)
    y_dir_all = angles_deg_to_unit_dir(y_angles_all[:, 0], y_angles_all[:, 1])  # (N,3)

    with open(optuna_json_path, "r", encoding="utf-8") as f:
        best = json.load(f)

    mparams = model_params_from_optuna(best)
    tparams = training_params_from_optuna(best)
    wparams = weight_params_from_optuna(best)
    lparams = loss_params_from_optuna(best)

    if cfg.force_use_weight_2d is not None:
        wparams["use_weight_2d"] = bool(cfg.force_use_weight_2d)

    os.makedirs(cfg.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(cfg.out_dir, f"run_{stamp}")
    os.makedirs(run_root, exist_ok=True)

    # save snapshots
    with open(os.path.join(run_root, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)
    with open(os.path.join(run_root, "optuna_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    # W&B single run
    wandb_run = None
    if wandb is not None:
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"final_{stamp}",
            tags=list(cfg.wandb_tags),
            config={
                "train_cfg": cfg.__dict__,
                "optuna_best": best,
                "model_params": mparams,
                "train_params": tparams,
                "weight_params": wparams,
                "loss_params": lparams,
            },
        )
        wandb_run.log({"run_root": run_root})

    # splits
    idx = np.arange(len(df_use))
    idx_trainval, idx_holdout = train_test_split(
        idx, test_size=cfg.holdout_frac, random_state=cfg.seed, shuffle=True
    )
    X_trainval = X_all[idx_trainval]
    y_trainval_dir = y_dir_all[idx_trainval]
    y_trainval_angles = y_angles_all[idx_trainval]
    X_holdout = X_all[idx_holdout]
    y_holdout_dir = y_dir_all[idx_holdout]
    y_holdout_angles = y_angles_all[idx_holdout]

    X_tr, X_va, y_tr_dir, y_va_dir, y_tr_angles, y_va_angles = train_test_split(
        X_trainval, y_trainval_dir, y_trainval_angles,
        test_size=cfg.internal_val_frac, random_state=cfg.seed, shuffle=True
    )

    # weights (train only); y_tr_angles[:, 1] is now phi (colatitude)
    w_tr = None
    if wparams["use_weight_2d"]:
        w_tr = hist2d_inverse_freq_weights(
            y_train_angles=y_tr_angles,
            y_query_angles=y_tr_angles,
            bins_theta=int(wparams["weight_bins_theta"]),
            bins_lambda=int(wparams["weight_bins_lambda"]),  # controls phi bins
            eps=float(wparams["weight_eps"]),
            clip=float(wparams["weight_clip"]),
            power=float(wparams["weight_power"]),
        )

    batch_size = int(tparams["batch_size"])
    train_loader = make_loader(X_tr, y_tr_dir, w_tr, batch_size=batch_size, shuffle=True, device=device)
    val_loader   = make_loader(X_va, y_va_dir, None, batch_size=batch_size, shuffle=False, device=device)

    # model
    model = MLP(
        in_dim=3,
        hidden_dims=list(mparams["hidden_dims"]),
        out_dim=3,
        activation=str(mparams["activation"]),
        dropout=float(mparams["dropout"]),
        use_batchnorm=bool(mparams["use_batchnorm"]),
        leaky_slope=float(mparams.get("leaky_slope", 0.01)),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(tparams["lr"]), weight_decay=float(tparams["weight_decay"]))

    # TB
    writer = None
    if cfg.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter  # optional dependency
        tb_dir = os.path.join(cfg.tb_root, os.path.basename(run_root), "final")
        writer = SummaryWriter(log_dir=tb_dir)

    best_ckpt  = os.path.join(run_root, "best.pt")
    latest_ckpt = os.path.join(run_root, "latest.pt")

    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = cfg.early_stop_patience

    t0 = time.time()
    epoch_bar = tqdm(range(1, cfg.max_epochs + 1), desc="final", leave=True)

    for epoch in epoch_bar:
        model.train()
        train_losses = []

        for batch in train_loader:
            if w_tr is None:
                xb, yb = batch
                wb = None
            else:
                xb, yb, wb = batch
                wb = wb.view(-1)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)  # (B,3)

            per = cosine_or_hybrid_loss(
                pred_raw=pred,
                yb_dir=yb,
                loss_type=str(lparams["loss_type"]),
                alpha=float(lparams["alpha"]),
                huber_beta=float(lparams["huber_beta"]),
            )
            loss = per.mean() if wb is None else (wb * per).mean()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().item()))

        # val loss
        model.eval()
        vlosses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                per = cosine_or_hybrid_loss(
                    pred_raw=pred,
                    yb_dir=yb,
                    loss_type=str(lparams["loss_type"]),
                    alpha=float(lparams["alpha"]),
                    huber_beta=float(lparams["huber_beta"]),
                )
                vlosses.append(per.mean().item())
        val_loss = float(np.mean(vlosses)) if vlosses else float("nan")

        # metrics using normalised predictions
        ytr_pred_raw = predict_numpy(model, X_tr, device=device)
        yva_pred_raw = predict_numpy(model, X_va, device=device)
        ytr_pred = (ytr_pred_raw / (np.linalg.norm(ytr_pred_raw, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
        yva_pred = (yva_pred_raw / (np.linalg.norm(yva_pred_raw, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

        train_m = direction_angle_metrics(y_tr_dir, ytr_pred)
        val_m   = direction_angle_metrics(y_va_dir, yva_pred)

        elapsed = time.time() - t0
        epoch_bar.set_postfix(val_loss=val_loss, dir_rmse=val_m["dir_rmse_deg"], best=best_val, patience=patience_left)

        # log TB
        if writer is not None:
            writer.add_scalar("train/loss_epoch", float(np.mean(train_losses)) if train_losses else float("nan"), epoch)
            writer.add_scalar("val/loss_epoch", val_loss, epoch)
            for k, v in train_m.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_m.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        # log W&B
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss_epoch": float(np.mean(train_losses)) if train_losses else float("nan"),
                    "val/loss_epoch": val_loss,
                    "time/elapsed_s": elapsed,
                    "epoch": epoch,
                    **{f"train/{k}": v for k, v in train_m.items()},
                    **{f"val/{k}": v for k, v in val_m.items()},
                }
            )

        # checkpoints
        meta = {
            "cfg": cfg.__dict__,
            "optuna_best": best,
            "model_params": mparams,
            "train_params": tparams,
            "weight_params": wparams,
            "loss_params": lparams,
        }

        if cfg.save_checkpoints and cfg.save_latest:
            save_checkpoint(latest_ckpt, model, opt, epoch, best_val, meta)
        if cfg.save_checkpoints and cfg.save_every_epochs > 0 and epoch % cfg.save_every_epochs == 0:
            save_checkpoint(os.path.join(run_root, f"epoch_{epoch:04d}.pt"), model, opt, epoch, best_val, meta)

        # early stopping on val_loss
        improved = (best_val - val_loss) > cfg.min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.early_stop_patience
            if cfg.save_checkpoints:
                save_checkpoint(best_ckpt, model, opt, epoch, best_val, meta)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # restore best
    model.load_state_dict(best_state)

    # holdout eval
    y_hold_pred_raw = predict_numpy(model, X_holdout, device=device)
    y_hold_pred = (y_hold_pred_raw / (np.linalg.norm(y_hold_pred_raw, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    holdout_m = direction_angle_metrics(y_holdout_dir, y_hold_pred)

    # save final artifact
    final_model_path = os.path.join(run_root, "final_model.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                **cfg.__dict__,
                **mparams,
                **tparams,
                **wparams,
                **lparams,
                "out_dim": 3,
                "angle_convention": ANGLE_CONVENTION,
            },
            "x_cols": x_cols,
            "y_cols_angles": y_cols_angles,   # ["Theta_deg", "lambda_deg"]
            "holdout_metrics": holdout_m,
        },
        final_model_path,
    )

    if writer is not None:
        writer.close()

    if wandb_run is not None:
        for k, v in holdout_m.items():
            wandb_run.summary[f"holdout_{k}"] = v
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.summary["final_model_path"] = final_model_path
        wandb_run.finish()

    return {
        "run_root": run_root,
        "final_model_path": final_model_path,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "holdout_metrics": holdout_m,
    }


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--optuna_json", type=str, required=True)

    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--patience", type=int, default=10**9)
    p.add_argument("--min_delta", type=float, default=1e-6)
    p.add_argument("--holdout", type=float, default=0.15)
    p.add_argument("--internal_val", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--no_latest", action="store_true")
    p.add_argument("--no_ckpt", action="store_true")

    p.add_argument("--force_weight_2d", choices=["on", "off"], default="",
                   help="Override JSON: 'on' forces weighting, 'off' disables weighting.")

    p.add_argument("--use_tb", action="store_true")
    p.add_argument("--tb_root", type=str, default="tb_logs")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="mlp-final-dir")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default="", help="Comma-separated tags.")

    args = p.parse_args()

    force_use_weight_2d = None
    if args.force_weight_2d == "on":
        force_use_weight_2d = True
    elif args.force_weight_2d == "off":
        force_use_weight_2d = False

    cfg = TrainConfig(
        holdout_frac=float(args.holdout),
        internal_val_frac=float(args.internal_val),
        seed=int(args.seed),
        max_epochs=int(args.epochs),
        early_stop_patience=int(args.patience),
        min_delta=float(args.min_delta),
        out_dir=str(args.out_dir),
        save_checkpoints=not bool(args.no_ckpt),
        save_every_epochs=int(args.save_every),
        save_latest=not bool(args.no_latest),
        use_tensorboard=bool(args.use_tb),
        tb_root=str(args.tb_root),
        use_wandb=bool(args.use_wandb),
        wandb_project=str(args.wandb_project),
        wandb_entity=args.wandb_entity,
        wandb_tags=tuple([t.strip() for t in args.wandb_tags.split(",") if t.strip()]),
        force_use_weight_2d=force_use_weight_2d,
    )

    results = train_from_csv_and_optuna_json(args.csv, args.optuna_json, cfg)

    print("\n=== Holdout Metrics (degrees) ===")
    for k, v in results["holdout_metrics"].items():
        print(f"{k:>14s}: {v:.6f}")

    print("\nSaved artifacts:")
    print(f"  run_root:       {results['run_root']}")
    print(f"  final_model.pt: {results['final_model_path']}")


if __name__ == "__main__":
    main()