# Training, evaluation and ONNX export

```
prepare_dataset.py ──► data/train.csv ──► train.py ──► runs/run_<stamp>/final_model.pt ──► export_onnx.py ──► final_model.onnx
                       data/test.csv  ──► evaluate.py ─────────────────┘                        check_convention.py ──┘
```

```bash
pip install -r requirements.txt
bash run_pipeline.sh "../data/train_sessions/*.csv" "../data/holdout/*.csv" 600   # everything below in one go
```

| script | purpose |
|---|---|
| `arch/mlp.py` | the network: fully connected MLP, He init, GELU/ReLU/…, optional dropout / batch-norm, 3 raw outputs |
| `train.py` | trains the MLP from a CSV with the hyper-parameters of an Optuna best-params JSON; writes `final_model.pt` |
| `evaluate.py` | metrics of a `final_model.pt` on a held-out CSV (great-circle, colatitude and azimuth errors) |
| `run_train_eval.py` | `train.py` + `evaluate.py` in one call, writes `summary.json` |
| `export_onnx.py` | `final_model.pt` → `final_model.onnx` (+ `.json` contract), verified with onnxruntime |
| `check_convention.py` | asserts the angle convention is the same in the trainer, the ROS node, the ONNX graph and the data |
| `configs/mlp_dir_optuna_best_params.json` | the paper's hyper-parameters (Optuna TPE, 840 trials, objective = validation direction RMSE) |
| `configs/train_config_paper_model.json`, `configs/test_metrics_paper_model.json` | training configuration and held-out metrics of the released model |

## Model

Input: unit-normalised de-biased magnetometer vector of segment *i+1* while coil *i* is on.
Output: raw 3-vector, normalised to the unit joint direction `d = (sin λ cos θ, sin λ sin θ, cos λ)`.
Loss `1 − cos(d_pred, d_true)`; AdamW. Best Optuna configuration:

```
hidden [1078, 222, 212, 346]   activation gelu   dropout 3.9e-4   batchnorm off
lr 5.63e-4   weight_decay 2.31e-3   batch 256   loss cosine   no histogram re-weighting
```

`train.py` first splits off a 15 % holdout (never used for selection), then trains on the rest
with a 10 % internal validation split for 600 epochs, keeping the epoch with the lowest
validation loss. Checkpoints (`best.pt`, `latest.pt`, `epoch_XXXX.pt`) and `final_model.pt` land
in `runs/run_<stamp>/`. Add `--use_wandb --wandb_project <name>` or `--use_tb` for logging.

```bash
python train.py --csv data/train.csv --optuna_json configs/mlp_dir_optuna_best_params.json --epochs 600
python evaluate.py --model runs/run_XXXX/final_model.pt --csv data/test.csv
python export_onnx.py runs/run_XXXX/final_model.pt --out ../ros_ws/src/pose_estimator/model/final_model.onnx
python check_convention.py --onnx ../ros_ws/src/pose_estimator/model/final_model.onnx --csv data/test.csv
```

Random-row validation inside one session underestimates the error on a new session
(≈3.9° vs ≈5.9° direction RMSE for the released model): mounting and calibration differences
between sessions dominate, so always report numbers on whole held-out sessions.

## ONNX contract

`export_onnx.py` bakes the input normalisation and the output normalisation into the graph:

```
input  "input"  float32 (N, 3)   de-biased magnetometer vector, any scale
output "output" float32 (N, 3)   unit joint direction (dx, dy, dz)
theta_deg  = degrees(atan2(dy, dx))      azimuth
lambda_deg = degrees(acos(dz))           colatitude, 0 = straight
```

`--raw` exports the bare MLP instead (unit-vector input expected, raw logits out). The ROS node
normalises input and output itself, so it runs either variant. A `.json` sidecar records the
contract, the source checkpoint, the architecture and the metrics.
