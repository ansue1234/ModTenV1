# ModTenV1 — Tendon-Driven Continuum Robot with Modular Stiffness and In-Situ Self Pose Estimation

Code release for

> G. N. Sue, Z. Cao, J. Hu, X. Bu, D. Quinn, T. Wu, Z. Erickson, C. Majidi,
> *Tendon-Driven Continuum Robot with Modular Stiffness and In-Situ Self Pose Estimation*,
> IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2026.
> arXiv:2609.16256 <https://arxiv.org/abs/2609.16256>

Project website: <https://ansue1234.github.io/ModTenV1/>

The robot is a chain of interchangeable segments. Every segment carries a small PCB with an
**ATtiny3224**, an **ICM-20948** 9-DoF IMU and an **electromagnetic coil**. Energising the coil of
segment *i* and reading the magnetometer of segment *i+1* gives a 3-vector that depends only on
the bending of the joint between them; a small MLP maps that vector to the joint direction. An
Arduino MEGA sequences the coils over I²C and streams the readings to a Raspberry Pi 5, where a
ROS 2 node runs the network (ONNX) and publishes the pose of every segment as a TF tree.

This repository contains everything needed to build and run that pipeline: segment firmware, the
data-collection rig software, dataset preparation, training / evaluation / ONNX export, the
pretrained model, and the ROS 2 deployment package. Github Pages and code are refactored by Claude. Contact gsue@andrew.cmu.edu for any questions.

Mechanical Design files could be accessed here: https://tinyurl.com/ModTenV1

```
                     segment 0 (base)   segment 1        segment 2   ...   segment N
                    ┌────────────┐   ┌────────────┐   ┌────────────┐
   Arduino MEGA ──I²C──│ ATtiny3224 │───│ ATtiny3224 │───│ ATtiny3224 │─── ...   (daisy chain)
   (coil sequencer)    │ ICM-20948  │   │ ICM-20948  │   │ ICM-20948  │
        │              │ coil+NMOS  │   │ coil+NMOS  │   │ coil+NMOS  │
        │ USB serial   └────────────┘   └────────────┘   └────────────┘
        ▼
   Raspberry Pi 5 ── ROS 2 Humble (Docker)
     imu_serial_node  ──► /imu/sequential_mag_array ──► pose_estimator_node (ONNX MLP) ──► /tf, /coil_poses
     tendon_controller ◄─ /tendon_cmd_vel ◄─ keyboard_publisher            (4× Dynamixel via U2D2)
```

## Repository layout

| Path | What it is |
|---|---|
| `firmware/segment/Bridge/` | ATtiny3224 firmware that runs on **every segment**: SPI to the ICM-20948, I²C register map towards the MEGA, coil switch. |
| `firmware/base_controller/MultiBridgeCollectionFast/` | Arduino MEGA sketch used for deployment: energises one coil at a time, reads the next segment's magnetometer, streams `SEQMAG` rows over serial (≈1.2 Hz for a 10-segment robot). |
| `firmware/data_collection_rig/DataCollectDynamixel/` | Arduino sketch for the two-segment training rig (`BIAS` / `SAMPLE` serial commands). |
| `data_collection/tendon_data_collection.py` | Drives the rig through random waypoints with four Dynamixel servos and logs one CSV per session. |
| `data_collection/prepare_dataset.py` | Session CSVs → training CSV (outlier rejection, per-waypoint means, accelerometer ground truth, unit-normalised magnetometer). |
| `training/` | `train.py`, `evaluate.py`, `run_train_eval.py`, `export_onnx.py`, `check_convention.py`, the MLP definition and the Optuna best hyper-parameters used in the paper. `run_pipeline.sh` runs the whole thing. |
| `ros_ws/` | ROS 2 Humble workspace (`pose_estimator` package), Dockerfile/compose for the Pi, and the **pretrained model** `src/pose_estimator/model/final_model.onnx`. |

## 1. Build the electronics and flash the firmware

See [`firmware/README.md`](firmware/README.md) for wiring, I²C addresses and the flashing
procedure (megaTinyCore + UPDI for the segments, Arduino IDE for the MEGA / rig Arduino).

## 2. Collect training data (two-segment rig)

See [`data_collection/README.md`](data_collection/README.md). In short:

```bash
pip install -r data_collection/requirements.txt
# edit the CONFIGURATION block (ports, servo centre positions), then
python data_collection/tendon_data_collection.py           # -> data/<date>/tendon_data_<date>_<nnn>.csv
```

Each waypoint yields an ambient reading (coil off), then 10 samples with the coil on; the de-biased
field and the accelerometer of the moving segment are logged.

## 3. Prepare the dataset, train, evaluate, export

```bash
pip install -r training/requirements.txt
cd training
bash run_pipeline.sh "../data/2026*/tendon_data_*.csv" "../data/holdout/tendon_data_*.csv"
```

`run_pipeline.sh` calls, in order, `prepare_dataset.py` (train and held-out sets — keep whole
sessions apart), `run_train_eval.py` (600 epochs with `configs/mlp_dir_optuna_best_params.json`),
`export_onnx.py` and `check_convention.py`, and leaves `runs/run_<stamp>/final_model.{pt,onnx,json}`
plus `test_metrics.json`. Every step can also be run on its own; see
[`training/README.md`](training/README.md).

## 4. Deploy on the Raspberry Pi (ROS 2 Humble)

```bash
cd ros_ws
docker compose build && docker compose run --rm ros      # or a native ROS 2 Humble install
colcon build --packages-select pose_estimator && source install/setup.bash
ros2 launch pose_estimator imu_serial.launch.py serial_port:=/dev/ttyACM0     # terminal 1
ros2 launch pose_estimator pose_estimator.launch.py                           # terminal 2 (pretrained model by default)
ros2 launch pose_estimator tendon_system.launch.py u2d2_port:=/dev/ttyUSB0    # optional: keyboard teleop of the tendons
```

Details, topics and parameters: [`ros_ws/README.md`](ros_ws/README.md).

## The angle convention (read this before changing anything)

Every stage of the pipeline uses the same encoding, and `training/check_convention.py` verifies it:

* dataset labels: `Theta_deg` = azimuth about the segment's +Z axis (−180…180°),
  `lambda_deg` = **colatitude**, 0° when the joint is straight;
* network target / output: unit direction `d = (sin λ cos θ, sin λ sin θ, cos λ)`;
* ROS node decoding: `θ = atan2(dy, dx)`, `λ = acos(dz)`.

The network input is the de-biased magnetometer vector normalised to unit length; the exported
ONNX graph performs that normalisation itself, so it accepts raw µT values.

## Pretrained model

`ros_ws/src/pose_estimator/model/final_model.onnx` (1.5 MB) is the model reported in the paper:
MLP 3 → 1078 → 222 → 212 → 346 → 3, GELU, dropout 3.9·10⁻⁴, trained with AdamW and a cosine loss on
≈64 k samples from 13 sessions, hyper-parameters found with Optuna (840 trials). On the held-out
session (5 154 samples, a different pair of segments) it reaches **5.9° great-circle RMSE**
(4.3° on colatitude, 10.8° on azimuth for tilts ≥ 10°). `final_model.json` next to it records the
contract, training configuration and metrics.

## Citation

```bibtex
@inproceedings{sue2026tendon,
  title     = {Tendon-Driven Continuum Robot with Modular Stiffness and In-Situ Self Pose Estimation},
  author    = {Sue, Guo Ning and Cao, Zheng and Hu, Junzhe and Bu, Xiangyun and Quinn, David and
               Wu, Tiancheng and Erickson, Zackory and Majidi, Carmel},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
  note      = {arXiv:2609.16256}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
