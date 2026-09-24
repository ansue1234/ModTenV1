#!/usr/bin/env python3
"""
check_convention.py -- guard against the azimuth/colatitude convention drifting
between the dataset, the trainer, the exported ONNX graph and the ROS node.

    python check_convention.py                                   # encoder <-> decoder round trip only
    python check_convention.py --onnx ../ros_ws/src/pose_estimator/model/final_model.onnx
    python check_convention.py --onnx <model.onnx> --csv data/test.csv   # + real-data sanity check

Checks
  1. train.py's angle -> direction encoder and pose_estimator_node.py's
     direction -> angle decoder (imported from the node file itself) are exact
     inverses under the COLATITUDE convention
        d = (sin(lambda) cos(theta), sin(lambda) sin(theta), cos(lambda))
        theta = atan2(dy, dx),  lambda = acos(dz)
  2. (--onnx) the graph maps (N,3) -> (N,3) and its .json sidecar says colatitude
  3. (--onnx --csv) on a labelled CSV from prepare_dataset.py the ONNX output
     decoded as colatitude fits the labels far better than decoded as elevation
     (lambda = asin(dz)); an elevation-era checkpoint fails this loudly.

Exit code 0 when everything is consistent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = os.path.join(HERE, "..", "ros_ws", "src", "pose_estimator", "scripts", "pose_estimator_node.py")
sys.path.insert(0, HERE)


def load_node_decoder():
    """Import unit_dir_to_angles_deg from the ROS node without importing rclpy."""
    src = open(NODE, encoding="utf-8").read()
    start = src.index("def normalize_vectors")
    end = src.index("def angles_deg_to_cartesian")
    ns: dict = {"np": np}
    exec(compile(src[start:end], NODE, "exec"), ns)
    return ns["unit_dir_to_angles_deg"]


def check_roundtrip() -> bool:
    from train import angles_deg_to_unit_dir, ANGLE_CONVENTION
    decode = load_node_decoder()
    rng = np.random.default_rng(0)
    theta = rng.uniform(-180, 180, 20000)
    lam = rng.uniform(0, 90, 20000)
    d = angles_deg_to_unit_dir(theta, lam)
    th2, la2 = decode(d)
    dth = (th2 - theta + 180) % 360 - 180
    ok = ANGLE_CONVENTION == "colatitude" and np.abs(dth).max() < 1e-3 and np.abs(la2 - lam).max() < 1e-3
    # explicit formula check so the test does not depend on both sides sharing a bug
    ok &= np.allclose(d[:, 2], np.cos(np.deg2rad(lam)), atol=1e-6)
    print(f"[1] trainer encoder <-> node decoder round trip: max |dtheta| {np.abs(dth).max():.2e} deg, "
          f"max |dlambda| {np.abs(la2 - lam).max():.2e} deg, dz == cos(lambda): "
          f"{np.allclose(d[:, 2], np.cos(np.deg2rad(lam)), atol=1e-6)}  -> {'OK' if ok else 'FAIL'}")
    return bool(ok)


def check_onnx(path: str, csv: str | None) -> bool:
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    shape_ok = list(inp.shape)[-1] == 3 and list(out.shape)[-1] == 3
    side = os.path.splitext(path)[0] + ".json"
    conv = json.load(open(side)).get("angle_convention") if os.path.exists(side) else None
    ok = shape_ok and conv in (None, "colatitude")
    print(f"[2] onnx {os.path.basename(path)}: input {inp.name}{inp.shape} output {out.name}{out.shape}, "
          f"sidecar convention={conv}  -> {'OK' if ok else 'FAIL'}")
    if not csv:
        return ok
    import pandas as pd
    df = pd.read_csv(csv).dropna(subset=["mx_uT_debias", "my_uT_debias", "mz_uT_debias", "Theta_deg", "lambda_deg"])
    X = df[["mx_uT_debias", "my_uT_debias", "mz_uT_debias"]].to_numpy(np.float32)
    y = sess.run(None, {inp.name: X})[0]
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    lam = df["lambda_deg"].to_numpy()
    lam_col = np.degrees(np.arccos(np.clip(y[:, 2], -1, 1)))
    lam_ele = np.degrees(np.arcsin(np.clip(y[:, 2], -1, 1)))
    rmse_col = float(np.sqrt(np.mean((lam_col - lam) ** 2)))
    rmse_ele = float(np.sqrt(np.mean((lam_ele - lam) ** 2)))
    th = np.deg2rad(df["Theta_deg"].to_numpy()); la = np.deg2rad(lam)
    d_true = np.stack([np.sin(la) * np.cos(th), np.sin(la) * np.sin(th), np.cos(la)], 1)
    gc = np.degrees(np.arctan2(np.linalg.norm(np.cross(y, d_true), axis=1), (y * d_true).sum(1)))
    data_ok = rmse_col < rmse_ele
    print(f"[3] {len(df)} labelled rows: dir-RMSE {np.sqrt(np.mean(gc ** 2)):.2f} deg | lambda-RMSE decoded as "
          f"colatitude {rmse_col:.2f} deg vs as elevation {rmse_ele:.2f} deg  -> "
          f"{'OK (colatitude model)' if data_ok else 'FAIL: this looks like an elevation-era model'}")
    return ok and data_ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", default=None, help="exported model to check")
    p.add_argument("--csv", default=None, help="labelled CSV from prepare_dataset.py (needs --onnx)")
    a = p.parse_args()
    ok = check_roundtrip()
    if a.onnx:
        ok &= check_onnx(a.onnx, a.csv)
    print("ALL CONSISTENT" if ok else "INCONSISTENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
