#!/usr/bin/env python3
"""
run_train_eval.py -- one-shot train + held-out evaluation.

  1) train the direction MLP with train.py's routine (hyper-parameters read
     from --optuna_json; configs/mlp_dir_optuna_best_params.json = paper setting)
  2) evaluate the resulting final_model.pt on the held-out test CSV with
     evaluate.py (colatitude convention, i.e. the encoding train.py uses)
  3) write <run_dir>/test_metrics.json, <run_dir>/test_preds.csv and
     <run_dir>/summary.json

Typical use (from the training/ folder):
  python run_train_eval.py --train_csv data/train.csv --test_csv data/test.csv \
      --optuna_json configs/mlp_dir_optuna_best_params.json --epochs 600

Add --use_wandb --wandb_project <name> for Weights & Biases logging.
The held-out test CSV should come from a SEPARATE collection session
(session-to-session shift is larger than random-row validation error).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# make sure imports resolve when the script is launched from another cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import TrainConfig, train_from_csv_and_optuna_json  # noqa: E402
from evaluate import evaluate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_csv", required=True, help="training CSV from prepare_dataset.py")
    p.add_argument("--test_csv", required=True, help="held-out CSV from a separate session")
    p.add_argument("--optuna_json", default="configs/mlp_dir_optuna_best_params.json")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--patience", type=int, default=10**9)
    p.add_argument("--holdout", type=float, default=0.15, help="internal holdout split carved from the TRAIN csv")
    p.add_argument("--internal_val", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--no_ckpt", action="store_true")
    p.add_argument("--force_weight_2d", choices=["on", "off", ""], default="")
    p.add_argument("--use_tb", action="store_true")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="mlp-final-dir")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_tags", default="")
    args = p.parse_args()

    force = None if args.force_weight_2d == "" else (args.force_weight_2d == "on")
    cfg = TrainConfig(
        holdout_frac=args.holdout,
        internal_val_frac=args.internal_val,
        seed=args.seed,
        max_epochs=args.epochs,
        early_stop_patience=args.patience,
        out_dir=args.out_dir,
        save_checkpoints=not args.no_ckpt,
        save_every_epochs=args.save_every,
        save_latest=not args.no_ckpt,
        use_tensorboard=args.use_tb,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_tags=tuple(t.strip() for t in args.wandb_tags.split(",") if t.strip()),
        force_use_weight_2d=force,
    )

    print(f"[1/2] training on {args.train_csv} with params from {args.optuna_json} ...")
    res = train_from_csv_and_optuna_json(args.train_csv, args.optuna_json, cfg)
    print("\n=== Internal holdout metrics (15% split of the TRAIN csv) ===")
    for k, v in res["holdout_metrics"].items():
        print(f"{k:>14s}: {v:.4f}")
    print(f"run_root: {res['run_root']}  (best epoch {res['best_epoch']})")

    print(f"\n[2/2] evaluating {res['final_model_path']} on {args.test_csv} ...")
    metrics = evaluate(res["final_model_path"], args.test_csv, convention="colatitude")

    summary_path = os.path.join(res["run_root"], "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"train_csv": args.train_csv, "test_csv": args.test_csv, "optuna_json": args.optuna_json,
                   "epochs": args.epochs, "best_epoch": res["best_epoch"], "best_val_loss": res["best_val_loss"],
                   "holdout_metrics": res["holdout_metrics"], "test_metrics": metrics}, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
