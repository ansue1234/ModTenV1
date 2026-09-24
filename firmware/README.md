# Firmware

Three sketches, three boards:

| Sketch | Runs on | Role |
|---|---|---|
| `segment/Bridge/Bridge.ino` | ATtiny3224 on **every segment PCB** | reads the ICM-20948 over SPI, exposes the readings as an I²C register map, switches the segment's coil |
| `base_controller/MultiBridgeCollectionFast/MultiBridgeCollectionFast.ino` | Arduino MEGA at the base of the robot | deployment: sequences the coils, reads the magnetometers, streams rows to the Raspberry Pi |
| `data_collection_rig/DataCollectDynamixel/DataCollectDynamixel.ino` | Arduino (Uno/Nano/MEGA) on the two-segment training rig | on request from `tendon_data_collection.py`: ambient reading with coil off, N samples with coil on |

## Segment board (`Bridge.ino`)

Hardware on each PCB: ATtiny3224, ICM-20948 (SPI), coil driven by an NMOS from `PA5`,
3.3 V LDO and level shifter, JST-XH 5-pin in/out connectors so the segments daisy-chain the
I²C bus and power.

```
ATtiny3224            ICM-20948            ATtiny3224        I²C bus (to the next segment / MEGA)
PA1 (MOSI)  ───────►  SDA/SDI              PB1 (SDA)  ◄────► SDA
PA2 (MISO)  ◄───────  SDO                  PB0 (SCL)  ◄────► SCL
PA3 (SCK)   ───────►  SCL                  PA5 (Arduino pin 1) ──► NMOS gate ──► coil
PA4 (CS, Arduino pin 0) ─► CS
```

1. Arduino IDE → Boards Manager → install **megaTinyCore**
   (`http://drazzy.com/package_drazzy.com_index.json`); Library Manager → **SparkFun ICM-20948**.
2. Board *ATtiny3224/…*, Chip *ATtiny3224*, Clock *20 MHz internal*, Programmer = your UPDI
   programmer (jtag2updi / SerialUPDI).
3. **Set a unique address per segment** — `#define I2C_ADDRESS 0x45` at the top of the sketch —
   and *Sketch → Upload Using Programmer*. The robot in the paper used `0x42 … 0x54`; any
   7-bit address works as long as it matches `BRIDGE_ADDRESSES[]` in the MEGA sketch.

Register map (all 16-bit values big-endian, auto-incrementing address):

| reg | content | reg | content |
|---|---|---|---|
| `0x00–0x05` | accel X/Y/Z (raw counts, ±2 g default) | `0x12–0x13` | temperature |
| `0x06–0x0B` | gyro X/Y/Z (raw counts, ±250 dps default) | `0x14` | status (1 = IMU ok) |
| `0x0C–0x11` | mag X/Y/Z (raw counts, 0.15 µT/LSB) | `0x15` | ICM WHO_AM_I (expect `0xEA`) |
| | | `0x16` | coil control: write 1 = on, 0 = off |

The loop keeps `sensorData` fresh as fast as the ICM delivers samples (magnetometer ≈100 Hz).

## Base controller (`MultiBridgeCollectionFast.ino`)

Wire the MEGA's I²C (`SDA 20`, `SCL 21`; `A4/A5` on an Uno/Nano) to the first segment and
edit the configuration block:

```cpp
#define NUM_BRIDGES 11                       // segments in the chain, including the base entry
const uint8_t BRIDGE_ADDRESSES[NUM_BRIDGES] = { 0x01, 0x42, 0x43, ... };   // top of the chain first
#define PLACEHOLDER_COIL_PIN 10              // 0x01 = "coil on MEGA pin 10", no IMU (the fixed base segment)
#define COIL_SETTLE_MS 20                    // wait after switching a coil on
```

Each cycle, for every joint *i*: read the ambient field of segment *i+1* (all coils off), energise
coil *i*, read segment *i+1*, switch the coil off. The de-biased vectors are then dumped as

```
SEQMAG,<t_ms>,<active_coil_id>,<source_bridge_id>,<mx_uT>,<my_uT>,<mz_uT>
# Sequential array dump complete
```

(plus verbose `AMBIENT,…` and `DATA,…` rows) at 115200 baud, which `imu_serial_node.py` turns into
one `SequentialMagArray` message per cycle. With 10 joints and the timing above a cycle takes
≈0.8 s (≈1.2 Hz), limited by the ≈10 ms magnetometer period.

Note: in this sketch only the magnetometer conversion (`×0.15 µT/LSB`) matters for pose
estimation; the accelerometer/gyro columns of the `AMBIENT`/`DATA` rows are printed with a
mg-based scale and are informational only.

## Training rig (`DataCollectDynamixel.ino`)

Two segment boards on the rig: one whose **IMU** is read (`ATTINY_I2C_ADDRESS`, the moving
segment) and one whose **coil** is energised (`COIL_ATTINY_ADDRESS`, the fixed segment). Set both
defines to the addresses flashed into those two boards. The sketch answers the serial protocol
used by `data_collection/tendon_data_collection.py`:

```
BIAS,<n>,<delay_ms>    → averages n magnetometer readings with the coil off   → BIAS_DONE,bx,by,bz
SAMPLE,<n>,<delay_ms>  → coil on, n rows DATA,t_ms,ax,ay,az,gx,gy,gz,mx_raw,my_raw,mz_raw,mx_debias,my_debias,mz_debias, coil off → SAMPLE_DONE
```

Accelerometer is converted with 16384 LSB/g, gyro with 131 LSB/dps, magnetometer with
0.15 µT/LSB. If the de-biased field magnitude with the coil on drops below
`COIL_FIELD_THRESHOLD` (5 µT) the sketch reports `COIL_FAILURE_DETECTED` and lights pin 13.
