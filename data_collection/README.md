# Data collection (two-segment training rig)

The training data come from a rig with one fixed segment (its coil is energised) and one
tendon-actuated segment (its magnetometer and accelerometer are read). Four Dynamixel XM430
servos pull four tendons; the accelerometer of the moving segment gives the ground-truth joint
angles, so no motion-capture is needed for training (motion capture was only used in the paper
to validate the labels).

```
 Python (tendon_data_collection.py)
   │ U2D2, Dynamixel Protocol 2.0, 57600 baud            │ USB serial, 115200 baud
   ▼                                                      ▼
 4× Dynamixel XM430 (IDs 10–13)                    Arduino (DataCollectDynamixel.ino)
   T1 +X   T2 +Y   T3 −X   T4 −Y                      │ I²C
                                                       ├─ segment board A  (ATTINY_I2C_ADDRESS)  → ICM-20948 of the moving segment
                                                       └─ segment board B  (COIL_ATTINY_ADDRESS) → coil of the fixed segment
```

## Setup

```bash
pip install -r requirements.txt          # pyserial, dynamixel-sdk, numpy, pandas
```

1. Flash the two segment boards (`firmware/segment/Bridge.ino`, distinct addresses) and the
   Arduino (`firmware/data_collection_rig/DataCollectDynamixel.ino`, same two addresses).
2. Set the servo IDs to 10–13, baud 57600, position-control mode (Dynamixel Wizard).
3. Edit the `CONFIGURATION` block of `tendon_data_collection.py`:
   * `U2D2_PORT`, `ARDUINO_PORT`
   * `CENTER_POSITIONS` — servo ticks with the segment straight (read them in Dynamixel Wizard,
     or set the homing offsets so that the straight pose is 2048 on every servo)
   * `AMPLITUDE_TICKS` — start small (≈400) and increase until the mechanical limit
   * `SAMPLING_SCHEME` — `weighted_uniform` (paper: 20 % straight, 50 % small tilts, 30 % whole
     workspace), `uniform` or `gaussian`
   * `NUM_WAYPOINTS`, `NUM_DATA_SAMPLES` (10), cooldown period (coil heating)

## Run

```bash
python tendon_data_collection.py         # writes data/<YYYYMMDD>/tendon_data_<YYYYMMDD>_<nnn>.csv
```

For every waypoint the script draws a target `(x, y) ∈ [−1, 1]²`, maps it to the four servo
positions (`T1/T3` antagonistic on X, `T2/T4` on Y), waits for the servos, requests `BIAS` (coil
off, 10 readings averaged) and then `SAMPLE` (coil on, 10 rows). Each row of the CSV holds

| column | meaning |
|---|---|
| `waypoint`, `target_x`, `target_y` | waypoint index and normalised target |
| `pos_t1 … pos_t4` | actual servo positions (ticks) |
| `bias_mx … bias_mz` | ambient field with the coil off (µT) |
| `sample_idx`, `t_ms` | sample index within the waypoint, Arduino time |
| `ax ay az`, `gx gy gz` | accelerometer (m/s²) and gyro (dps) of the moving segment |
| `mx_raw … mz_raw` | magnetometer with the coil on (µT) |
| `mx_debias … mz_debias` | magnetometer minus the ambient bias (µT) — the network input |

The script pauses for `COOLDOWN_DURATION_SEC` every `COOLDOWN_AFTER_WAYPOINTS` waypoints so the
coil does not overheat, flushes the CSV every waypoint, and stops (returning the servos to centre)
if the Arduino reports a coil failure or an unresponsive I²C device. Collect the **test session
with a different pair of segments** than the training sessions if you want the numbers to reflect
segment-to-segment generalisation, as in the paper.

## From sessions to a training CSV

```bash
python prepare_dataset.py "data/2026*/tendon_data_*.csv"  --out ../training/data/train.csv
python prepare_dataset.py  data/holdout/tendon_data_x.csv --out ../training/data/test.csv
```

`prepare_dataset.py` reproduces the processing used for the paper:

1. drop rows with failed reads (`nan`);
2. per waypoint, drop samples whose de-biased field is more than 2 σ from the waypoint mean on any
   axis (`--outlier-k`);
3. append the per-waypoint mean of the remaining samples as extra rows (`--no-avg` to skip);
4. ground truth from gravity: the accelerometer of the moving segment is mapped into the world
   frame (`+X` West, `+Y` North, `+Z` out of the dome), roll/pitch give the segment axis, and
   `lambda_deg = acos(r_z)` (colatitude, 0 = straight), `Theta_deg = atan2(r_x, r_y)` (azimuth);
5. unit-normalise `(mx, my, mz)_debias` → `mx_uT_debias, my_uT_debias, mz_uT_debias`.

Output columns: `mx_uT_debias, my_uT_debias, mz_uT_debias, Theta_deg, lambda_deg, session, waypoint, kind`.
