#!/usr/bin/env python3
"""
export_onnx.py -- export a trained final_model.pt to ONNX for the Raspberry Pi.

    python export_onnx.py runs/<run>/final_model.pt                      # -> runs/<run>/final_model.onnx
    python export_onnx.py runs/<run>/final_model.pt --out ../ros_ws/src/pose_estimator/model/final_model.onnx

Graph contract (default, "baked" normalisation)
    input  "input"  : float32 (N, 3)  de-biased magnetometer reading (mx, my, mz) of
                      segment i+1 while the coil of segment i is ON.  Any scale
                      (uT or raw counts) -- the graph divides by the norm first.
    output "output" : float32 (N, 3)  unit direction vector d = (dx, dy, dz) of the
                      joint, colatitude convention:
                          azimuth    theta  = atan2(dy, dx)
                          colatitude lambda = acos(dz)        (0 = straight)

With --raw the bare MLP is exported instead (unit-vector input expected, raw
logits out).  ros_ws/.../pose_estimator_node.py normalises its input and output
itself, so it works with either variant.

The export is verified against the PyTorch model with onnxruntime, and a
<name>.json sidecar with the contract + source checkpoint + metrics is written.

Requires:  pip install torch onnx onnxruntime   (see requirements.txt)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_model  # noqa: E402


class UnitInUnitOut(nn.Module):
    """raw magnetometer vector -> unit input -> MLP -> unit direction."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)
        y = self.net(x)
        return y / y.norm(dim=1, keepdim=True).clamp_min(1e-9)


def export(bundle: str, out: str | None, opset: int, raw: bool, n_check: int = 512) -> int:
    model, x_cols, y_cols, info = load_model(bundle, torch.device("cpu"))
    convention = info.get("stored_convention") or "colatitude"
    if convention != "colatitude":
        print(f"[warn] checkpoint stores angle_convention={convention!r}; the ROS node assumes colatitude")

    wrapped = (model if raw else UnitInUnitOut(model)).cpu().eval()
    out = out or os.path.join(os.path.dirname(os.path.abspath(bundle)), "final_model.onnx")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    dummy = torch.tensor([[12.0, -3.5, 40.0]], dtype=torch.float32)
    kwargs = dict(input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                  opset_version=opset)
    try:  # TorchScript-based exporter (torch <= 2.x); the dynamo exporter is the fallback
        torch.onnx.export(wrapped, (dummy,), out, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapped, (dummy,), out, **kwargs)

    # ---- verify with onnxruntime ------------------------------------------------
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 30.0, (n_check, 3)).astype(np.float32)          # ~uT-scale de-biased fields
    if raw:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    with torch.no_grad():
        ref = wrapped(torch.from_numpy(X)).numpy()
    got = sess.run(None, {"input": X})[0]
    ref_u = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    got_u = got / np.linalg.norm(got, axis=1, keepdims=True)
    ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(ref_u, got_u), axis=1), (ref_u * got_u).sum(1)))
    max_abs = float(np.abs(ref - got).max())

    meta = {
        "source_checkpoint": os.path.abspath(bundle),
        "input": "input: float32 (N,3) de-biased magnetometer vector (mx,my,mz)" +
                 (" -- must already be unit length" if raw else " -- any scale, normalised inside the graph"),
        "output": "output: float32 (N,3) direction vector d=(dx,dy,dz)" +
                  (" -- raw logits, normalise before use" if raw else " -- unit length"),
        "angle_convention": convention,
        "theta_deg": "degrees(arctan2(dy, dx))   azimuth about +Z",
        "lambda_deg": "degrees(arccos(dz))       colatitude, 0 = straight",
        "x_cols": x_cols, "y_cols_angles": y_cols,
        "hidden_dims": info.get("hidden_dims"), "activation": info.get("activation"),
        "dropout": info.get("dropout"), "use_batchnorm": info.get("use_batchnorm"),
        "loss_type": info.get("loss_type"),
        "holdout_metrics": info.get("holdout_metrics"),
        "opset": opset, "normalisation_baked": not raw,
        "verification": {"n_inputs": n_check, "max_angle_diff_deg": float(ang.max()), "max_abs_diff": max_abs},
    }
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB), opset {opset}, "
          f"{'raw MLP' if raw else 'unit-in/unit-out'} graph")
    print(f"onnxruntime vs torch on {n_check} inputs: max angle diff {ang.max():.2e} deg, max abs diff {max_abs:.2e}")
    ok = ang.max() < 1e-3
    print("verification", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bundle", help="final_model.pt written by train.py")
    p.add_argument("--out", default=None, help="output .onnx path (default: next to the checkpoint)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--raw", action="store_true", help="export the bare MLP (no input/output normalisation)")
    a = p.parse_args()
    return export(a.bundle, a.out, a.opset, a.raw)


if __name__ == "__main__":
    sys.exit(main())
