#!/usr/bin/env python3
"""
evaluate.py -- evaluate a trained final_model.pt on a held-out CSV.

Reports direction-space (great-circle) and angle-space metrics for the
unit-direction MLP saved by train.py.  The CSV must have the same columns as
the training CSV (see data_collection/prepare_dataset.py); if the label
columns are missing only predictions are written.

Angle convention
  lambda_deg in the dataset CSVs is a COLATITUDE (tilt from the +Z axis,
  0 deg = upright).  train.py encodes the target as
      d = (sin(lam)cos(th), sin(lam)sin(th), cos(lam))          -> "colatitude"
  and stores "angle_convention" in final_model.pt, which is picked up here
  automatically.  "elevation" (d = (cos(lam)cos(th), cos(lam)sin(th), sin(lam)))
  is kept only for checkpoints produced by older experiments; the convention
  used at EVAL time must match the one used at TRAIN time.

Usage:
  python evaluate.py --model runs/run_XXXX/final_model.pt --csv data/test.csv
  # outputs (next to the model unless --out/--metrics_json given):
  #   <run_dir>/test_preds.csv      per-row predictions + errors
  #   <run_dir>/test_metrics.json   all metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch.mlp import MLP  # noqa: E402


# ----------------------------
# Device
# ----------------------------
def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------
# Geometry
# ----------------------------
def wrap_to_180(deg: np.ndarray) -> np.ndarray:
    return (deg + 180.0) % 360.0 - 180.0


def angles_to_dir(theta_deg: np.ndarray, lam_deg: np.ndarray, convention: str) -> np.ndarray:
    th = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    la = np.deg2rad(np.asarray(lam_deg, dtype=np.float64))
    if convention == "colatitude":
        s = np.sin(la)
        d = np.stack([s * np.cos(th), s * np.sin(th), np.cos(la)], axis=-1)
    elif convention == "elevation":
        c = np.cos(la)
        d = np.stack([c * np.cos(th), c * np.sin(th), np.sin(la)], axis=-1)
    else:
        raise ValueError(f"unknown convention {convention}")
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    return d


def dir_to_angles(d: np.ndarray, convention: str) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(d, dtype=np.float64)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    theta = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    dz = np.clip(d[:, 2], -1.0, 1.0)
    if convention == "colatitude":
        lam = np.degrees(np.arccos(dz))
    elif convention == "elevation":
        lam = np.degrees(np.arcsin(dz))
    else:
        raise ValueError(f"unknown convention {convention}")
    return theta, lam


# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(
    th_true: np.ndarray, la_true: np.ndarray,
    th_pred: np.ndarray, la_pred: np.ndarray,
    convention: str,
    theta_min_lambda: float = 10.0,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    th_true = np.asarray(th_true, np.float64); la_true = np.asarray(la_true, np.float64)
    th_pred = np.asarray(th_pred, np.float64); la_pred = np.asarray(la_pred, np.float64)

    th_err = wrap_to_180(th_pred - th_true)          # circular azimuth error
    la_err = la_pred - la_true                        # tilt error

    d_true = angles_to_dir(th_true, la_true, convention)
    d_pred = angles_to_dir(th_pred, la_pred, convention)
    dot = np.clip(np.sum(d_true * d_pred, axis=1), -1.0, 1.0)
    dir_err = np.degrees(np.arccos(dot))              # great-circle error

    eps = 1e-12
    r2_lambda = 1.0 - np.sum(la_err ** 2) / (np.sum((la_true - la_true.mean()) ** 2) + eps)
    ss_res = np.sum((d_true - d_pred) ** 2, axis=0)
    ss_tot = np.sum((d_true - d_true.mean(axis=0)) ** 2, axis=0) + eps
    r2_xyz = 1.0 - ss_res / ss_tot

    m: Dict[str, float] = {
        "n_rows": int(len(th_true)),
        "convention": convention,
        # direction (great-circle) error - the headline metric
        "dir_rmse_deg": float(np.sqrt(np.mean(dir_err ** 2))),
        "dir_mae_deg": float(np.mean(dir_err)),
        "dir_median_deg": float(np.median(dir_err)),
        "dir_p95_deg": float(np.quantile(dir_err, 0.95)),
        # tilt (lambda)
        "rmse_lambda": float(np.sqrt(np.mean(la_err ** 2))),
        "mae_lambda": float(np.mean(np.abs(la_err))),
        "bias_lambda": float(np.mean(la_err)),
        "r2_lambda": float(r2_lambda),
        # azimuth (theta), circular
        "rmse_theta": float(np.sqrt(np.mean(th_err ** 2))),
        "mae_theta": float(np.mean(np.abs(th_err))),
        "median_abs_theta": float(np.median(np.abs(th_err))),
        # component R2 of unit vectors
        "r2_dx": float(r2_xyz[0]), "r2_dy": float(r2_xyz[1]), "r2_dz": float(r2_xyz[2]),
        "r2_dir_mean": float(np.mean(r2_xyz)),
    }
    # azimuth is ill-defined when the tilt is ~0, so also report theta error on tilted rows only
    mask = la_true >= theta_min_lambda
    if mask.any():
        m[f"rmse_theta_lambda_ge_{theta_min_lambda:g}"] = float(np.sqrt(np.mean(th_err[mask] ** 2)))
        m[f"mae_theta_lambda_ge_{theta_min_lambda:g}"] = float(np.mean(np.abs(th_err[mask])))
        m[f"n_rows_lambda_ge_{theta_min_lambda:g}"] = int(mask.sum())

    per_row = {"theta_err_deg": th_err, "lambda_err_deg": la_err, "dir_err_deg": dir_err}
    return m, per_row


# ----------------------------
# Model loading
# ----------------------------
def load_model(model_path: str, device: torch.device):
    """
    Load a direction-MLP artifact written by train.py.

    Accepts the final artifact (final_model.pt: "config" + "model_state") as well
    as the intermediate checkpoints (best.pt / latest.pt / epoch_XXXX.pt), which
    keep the same information under "meta".
    Returns (model, x_cols, y_cols, info).
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model_state" not in ckpt:
        raise ValueError("artifact has no 'model_state'")

    if "config" in ckpt:
        cfg = dict(ckpt["config"])
    elif isinstance(ckpt.get("meta"), dict):  # checkpoint: rebuild a flat config from meta
        meta = ckpt["meta"]
        cfg = {**meta.get("model_params", {}), **meta.get("train_params", {}), **meta.get("loss_params", {})}
    else:
        raise ValueError("Unrecognised artifact format (need 'config' or 'meta').")

    if int(cfg.get("out_dim", 3)) != 3:
        raise ValueError(f"expected a 3-output direction model, got out_dim={cfg.get('out_dim')}")

    x_cols = list(ckpt.get("x_cols", ["mx_uT_debias", "my_uT_debias", "mz_uT_debias"]))
    y_cols = list(ckpt.get("y_cols_angles", ["Theta_deg", "lambda_deg"]))
    model = MLP(
        in_dim=len(x_cols),
        hidden_dims=list(cfg["hidden_dims"]),
        out_dim=3,
        activation=cfg.get("activation", "gelu"),
        dropout=float(cfg.get("dropout", 0.0)),
        use_batchnorm=bool(cfg.get("use_batchnorm", False)),
        leaky_slope=float(cfg.get("leaky_slope", 0.01)),
    )
    model.load_state_dict(ckpt["model_state"])
    info = {"hidden_dims": list(cfg["hidden_dims"]), "activation": cfg.get("activation"),
            "dropout": cfg.get("dropout"), "use_batchnorm": cfg.get("use_batchnorm"),
            "lr": cfg.get("lr"), "weight_decay": cfg.get("weight_decay"), "batch_size": cfg.get("batch_size"),
            "loss_type": cfg.get("loss_type"), "stored_convention": cfg.get("angle_convention"),
            "holdout_metrics": ckpt.get("holdout_metrics")}
    return model.to(device).eval(), x_cols, y_cols, info


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).float().to(device)
        outs.append(model(xb).detach().cpu().numpy())
    return np.vstack(outs)


# ----------------------------
# Main
# ----------------------------
def evaluate(model_path: str, csv_path: str, convention: str = "auto", out_csv: str = "",
             metrics_json: str = "", device: str = "auto", theta_min_lambda: float = 10.0,
             quiet: bool = False) -> Dict[str, float]:
    dev = device_auto() if device == "auto" else torch.device(device)
    model, x_cols, y_cols, info = load_model(model_path, dev)

    if convention == "auto":
        convention = info.get("stored_convention") or "colatitude"
    info["eval_convention"] = convention

    df = pd.read_csv(csv_path)
    missing = [c for c in x_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing feature columns {missing}")
    has_labels = all(c in df.columns for c in y_cols)
    need = list(x_cols) + (list(y_cols) if has_labels else [])
    n0 = len(df)
    df = df.dropna(subset=need).reset_index(drop=True)
    if len(df) < n0 and not quiet:
        print(f"[warn] dropped {n0 - len(df)} rows with NaNs")

    X = df[x_cols].to_numpy(dtype=np.float32)
    raw = predict(model, X, dev)

    d_pred = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12)
    th_pred, la_pred = dir_to_angles(d_pred, convention)
    df["pred_dx"], df["pred_dy"], df["pred_dz"] = d_pred[:, 0], d_pred[:, 1], d_pred[:, 2]
    df["pred_Theta_deg"] = th_pred
    df["pred_lambda_deg"] = la_pred

    metrics: Dict[str, float] = {"model": model_path, "csv": csv_path,
                                 "convention": convention, "n_rows": int(len(df))}
    if has_labels:
        m, per_row = compute_metrics(df[y_cols[0]].to_numpy(), df[y_cols[1]].to_numpy(),
                                     th_pred, la_pred, convention, theta_min_lambda)
        metrics.update(m)
        for k, v in per_row.items():
            df[k] = v

    if not quiet:
        print("\n=== Model ===")
        for k in ["hidden_dims", "activation", "dropout", "use_batchnorm", "lr", "weight_decay",
                  "batch_size", "loss_type", "stored_convention", "eval_convention"]:
            if info.get(k) is not None:
                print(f"  {k:>18s}: {info[k]}")
        if info.get("holdout_metrics"):
            hm = info["holdout_metrics"]
            keys = [k for k in ("dir_rmse_deg", "rmse_theta", "rmse_phi", "rmse_lambda") if k in hm]
            print("  internal holdout (from training run): " + ", ".join(f"{k}={hm[k]:.3f}" for k in keys))
        if has_labels:
            print(f"\n=== Test metrics on {csv_path}  ({len(df)} rows, convention={convention}) ===")
            order = ["dir_rmse_deg", "dir_mae_deg", "dir_median_deg", "dir_p95_deg",
                     "rmse_lambda", "mae_lambda", "bias_lambda", "r2_lambda",
                     "rmse_theta", "mae_theta", "median_abs_theta",
                     f"rmse_theta_lambda_ge_{theta_min_lambda:g}", f"mae_theta_lambda_ge_{theta_min_lambda:g}",
                     "r2_dir_mean", "r2_dx", "r2_dy", "r2_dz"]
            for k in order:
                if k in metrics:
                    print(f"  {k:>28s}: {metrics[k]:10.4f}")
        else:
            print("\nNo label columns found -> predictions only.")

    run_dir = os.path.dirname(os.path.abspath(model_path))
    if out_csv == "":
        out_csv = os.path.join(run_dir, "test_preds.csv")
    if metrics_json == "":
        metrics_json = os.path.join(run_dir, "test_metrics.json")
    if out_csv.lower() != "none":
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        df.to_csv(out_csv, index=False)
        if not quiet:
            print(f"\nSaved predictions to: {out_csv}")
    if metrics_json.lower() != "none":
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        if not quiet:
            print(f"Saved metrics to:     {metrics_json}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path to final_model.pt")
    p.add_argument("--csv", required=True, help="Held-out CSV (same columns as training CSV)")
    p.add_argument("--convention", choices=["auto", "colatitude", "elevation"], default="auto",
                   help="How the model encodes lambda. auto -> value stored in the checkpoint, else 'colatitude' "
                        "(what train.py uses). 'elevation' only for checkpoints from older experiments.")
    p.add_argument("--out", default="", help="Predictions CSV path (default: <run_dir>/test_preds.csv, 'none' to skip)")
    p.add_argument("--metrics_json", default="", help="Metrics JSON path (default: <run_dir>/test_metrics.json, 'none' to skip)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--theta_min_lambda", type=float, default=10.0,
                   help="Also report azimuth error restricted to rows with true lambda >= this (deg)")
    args = p.parse_args()
    evaluate(args.model, args.csv, args.convention, args.out, args.metrics_json, args.device, args.theta_min_lambda)


if __name__ == "__main__":
    main()
