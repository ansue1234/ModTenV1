#!/usr/bin/env python3
"""
prepare_dataset.py -- turn raw collection sessions into a training CSV.

    python prepare_dataset.py data/20260215/*.csv --out data/train.csv
    python prepare_dataset.py data/holdout_session.csv --out data/test.csv --no-avg

Input : one or more session CSVs written by tendon_data_collection.py
        (columns waypoint, ax, ay, az, mx_debias, my_debias, mz_debias, ...)
Output: one CSV with the columns the trainer expects
        mx_uT_debias, my_uT_debias, mz_uT_debias   unit-normalised de-biased field
        Theta_deg                                   azimuth   (deg, -180..180)
        lambda_deg                                  colatitude (deg, 0 = straight)
        plus 'session' and 'waypoint' bookkeeping columns (ignored by train.py).

Per session the steps are (same as the paper's processing):
  1. drop rows with missing values (failed IMU reads are logged as nan)
  2. per-waypoint outlier rejection: a sample is dropped if any de-biased
     magnetometer component is more than K (default 2) std from the mean of
     its waypoint  (--outlier-k, 0 disables)
  3. optionally append the per-waypoint MEAN of the surviving samples as extra
     rows (--avg, on by default; the paper's training set contains both the
     individual samples and the waypoint means)
  4. ground-truth joint angles from the accelerometer of the moving segment
     (gravity direction -> tilt of the segment):
        Theta_deg  = azimuth of the segment axis
        lambda_deg = colatitude, 0 when the segment is straight
  5. unit-normalise the de-biased magnetometer vector

A held-out TEST set should be a separate session (or sessions) passed in a
second call -- never split the rows of one session between train and test.

Requires: numpy, pandas
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List

import numpy as np
import pandas as pd

MAG_COLS = ["mx_debias", "my_debias", "mz_debias"]
ACC_COLS = ["ax", "ay", "az"]
KEEP_COLS = ["waypoint", "target_x", "target_y", "t_ms"] + ACC_COLS + MAG_COLS
OUT_COLS = ["mx_uT_debias", "my_uT_debias", "mz_uT_debias", "Theta_deg", "lambda_deg", "session", "waypoint"]


# ----------------------------------------------------------------------------
# Ground truth from the accelerometer
# ----------------------------------------------------------------------------
def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def angles_from_accel(ax_b: np.ndarray, ay_b: np.ndarray, az_b: np.ndarray, psi0_rad: float = 0.0):
    """
    Joint angles of the moving segment from its accelerometer (gravity) reading.

    Frames (right-handed):
        world : +X West, +Y North, +Z out of the dome (up when the base is upright)
        sensor: +x West, +y South, +z into the dome   ->  ax_w = ax, ay_w = -ay, az_w = -az

    Returns
        lambda : colatitude of the segment axis, 0 = straight (rad, 0..pi)
        theta  : azimuth of the segment axis, 0 = +Y(North), increasing towards +X(West)
                 (rad, -pi..pi).  psi0_rad adds a constant yaw offset if the sensor
                 is mounted rotated about the segment axis.
    """
    ax, ay, az = ax_b, -ay_b, -az_b
    g = np.sqrt(ax**2 + ay**2 + az**2)
    g = np.where(g == 0.0, np.nan, g)
    ax, ay, az = ax / g, ay / g, az / g

    roll = np.arctan2(ay, az)                          # about +X
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))    # about +Y

    c, s = np.cos(psi0_rad), np.sin(psi0_rad)
    rx = np.sin(pitch) * np.cos(roll) * c - np.sin(roll) * s
    ry = np.sin(pitch) * np.cos(roll) * s + np.sin(roll) * c
    rz = np.cos(pitch) * np.cos(roll)
    n = np.sqrt(rx**2 + ry**2 + rz**2)
    n = np.where(n == 0.0, np.nan, n)
    rx, ry, rz = rx / n, ry / n, rz / n

    lam = np.arccos(np.clip(rz, -1.0, 1.0))            # colatitude, 0 at the top of the dome
    theta = wrap_to_pi(np.arctan2(rx, ry))             # azimuth, 0 = North
    return lam, theta


# ----------------------------------------------------------------------------
# Per-session processing
# ----------------------------------------------------------------------------
def filter_waypoint_outliers(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """Drop samples whose de-biased field deviates > k std from their waypoint mean (any axis)."""
    if k <= 0:
        return df
    g = df.groupby("waypoint")[MAG_COLS]
    mu = g.transform("mean")
    sigma = g.transform("std").replace(0, np.nan)      # groups of size 1 / zero std -> never dropped
    z = (df[MAG_COLS] - mu).div(sigma).fillna(0.0)
    return df.loc[(z.abs() <= k).all(axis=1)].reset_index(drop=True)


def process_session(path: str, outlier_k: float, add_avg: bool, psi0_rad: float, verbose: bool) -> pd.DataFrame:
    raw = pd.read_csv(path)
    missing = [c for c in ["waypoint"] + ACC_COLS + MAG_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    cols = [c for c in KEEP_COLS if c in raw.columns]
    df = raw[cols].apply(pd.to_numeric, errors="coerce").dropna(subset=ACC_COLS + MAG_COLS)
    n_raw, n_nan = len(raw), len(raw) - len(df)

    df = filter_waypoint_outliers(df, outlier_k)
    n_out = n_raw - n_nan - len(df)

    parts = [df.assign(kind="sample")]
    if add_avg:
        avg = df.groupby("waypoint", as_index=False)[ACC_COLS + MAG_COLS].mean()
        parts.append(avg.assign(kind="waypoint_mean"))
    df = pd.concat(parts, ignore_index=True)

    lam, theta = angles_from_accel(df["ax"].to_numpy(float), df["ay"].to_numpy(float),
                                   df["az"].to_numpy(float), psi0_rad)
    mag = df[MAG_COLS].to_numpy(float)
    norm = np.linalg.norm(mag, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    unit = mag / norm

    out = pd.DataFrame({
        "mx_uT_debias": unit[:, 0], "my_uT_debias": unit[:, 1], "mz_uT_debias": unit[:, 2],
        "Theta_deg": np.degrees(theta), "lambda_deg": np.degrees(lam),
        "session": os.path.splitext(os.path.basename(path))[0],
        "waypoint": df["waypoint"].to_numpy(),
        "kind": df["kind"].to_numpy(),
    }).dropna(subset=["Theta_deg", "lambda_deg"]).reset_index(drop=True)

    if verbose:
        n_wp = df["waypoint"].nunique()
        print(f"  {os.path.basename(path):45s} rows {n_raw:6d} -> {len(out):6d} "
              f"(nan {n_nan}, outliers {n_out}, waypoints {n_wp}{', + means' if add_avg else ''})  "
              f"lambda {out['lambda_deg'].min():5.1f}..{out['lambda_deg'].max():5.1f} deg")
    return out


def expand(paths: List[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.csv")))
        else:
            hits = sorted(glob.glob(p))
            files += hits if hits else [p]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="session CSVs, globs or directories")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--outlier-k", type=float, default=2.0, help="per-waypoint z-score cut (0 = off)")
    ap.add_argument("--no-avg", dest="avg", action="store_false", help="do not append per-waypoint mean rows")
    ap.add_argument("--psi0-deg", type=float, default=0.0, help="sensor yaw offset about the segment axis")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    files = expand(a.inputs)
    if not files:
        print("no input files", file=sys.stderr)
        return 1
    if not a.quiet:
        print(f"processing {len(files)} session(s)")
    frames = [process_session(f, a.outlier_k, a.avg, np.deg2rad(a.psi0_deg), not a.quiet) for f in files]
    df = pd.concat(frames, ignore_index=True)[OUT_COLS + ["kind"]]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    if not a.quiet:
        print(f"wrote {a.out}: {len(df)} rows, {df['session'].nunique()} session(s), "
              f"{(df['kind'] == 'waypoint_mean').sum()} waypoint-mean rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
