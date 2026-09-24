#!/usr/bin/env bash
# End-to-end: sessions -> dataset -> train -> evaluate -> ONNX -> convention check.
#
#   ./run_pipeline.sh "<train session csv glob>" "<test session csv glob>" [epochs]
#   e.g. ./run_pipeline.sh "../data/2026021*/tendon_data_*.csv" "../data/holdout/*.csv" 600
#
# Quote the globs so this script (not your shell) expands them.  Outputs:
#   data/train.csv, data/test.csv
#   runs/run_<stamp>/final_model.pt, test_metrics.json, test_preds.csv, summary.json
#   runs/run_<stamp>/final_model.onnx (+ .json contract)
set -euo pipefail
cd "$(dirname "$0")"

TRAIN_GLOB="${1:?train session csv glob}"
TEST_GLOB="${2:?test session csv glob}"
EPOCHS="${3:-600}"
PARAMS="configs/mlp_dir_optuna_best_params.json"

echo "== [1/5] dataset"
python ../data_collection/prepare_dataset.py $TRAIN_GLOB --out data/train.csv
python ../data_collection/prepare_dataset.py $TEST_GLOB  --out data/test.csv

echo "== [2/5] train + evaluate ($EPOCHS epochs, $PARAMS)"
python run_train_eval.py --train_csv data/train.csv --test_csv data/test.csv \
    --optuna_json "$PARAMS" --epochs "$EPOCHS" --out_dir runs
RUN_DIR="$(ls -td runs/run_* | head -n 1)"
echo "   run dir: $RUN_DIR"

echo "== [3/5] export ONNX"
python export_onnx.py "$RUN_DIR/final_model.pt" --out "$RUN_DIR/final_model.onnx"

echo "== [4/5] convention check (trainer <-> ROS node <-> ONNX <-> data)"
python check_convention.py --onnx "$RUN_DIR/final_model.onnx" --csv data/test.csv

echo "== [5/5] done"
echo "copy $RUN_DIR/final_model.onnx and final_model.json to ../ros_ws/src/pose_estimator/model/ to deploy it"
