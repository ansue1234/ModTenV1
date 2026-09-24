# ROS 2 deployment (`pose_estimator` package)

Runs on the Raspberry Pi 5 (ROS 2 Humble, in Docker) — or on any Linux machine with ROS 2 Humble.

```
Arduino MEGA ──USB──► imu_serial_node ──► /imu/sequential_mag_array ──► pose_estimator_node ──► /tf, /coil_poses
                                     ├──► /imu/accel  /imu/mag_raw  /imu/mag_bias  /imu/mag_debias
U2D2 ─────────USB──► tendon_controller ◄── /tendon_cmd_vel ◄── keyboard_publisher
```

## Build

```bash
cd ros_ws
docker compose build                    # ros:humble-ros-base + numpy/scipy/pyserial/onnxruntime/dynamixel-sdk
docker compose run --rm ros             # shell inside the container, this folder mounted at /ws
colcon build --packages-select pose_estimator
source install/setup.bash
```

Without Docker: `source /opt/ros/humble/setup.bash`, `pip3 install -r requirements.txt`, then the
same `colcon build`. Edit the `devices:` section of `docker-compose.yml` to the serial ports of
your MEGA and U2D2 (`ls -l /dev/serial/by-id/`).

## Run

```bash
# 1. serial bridge to the segment chain (MEGA running firmware/base_controller/MultiBridgeCollectionFast.ino)
ros2 launch pose_estimator imu_serial.launch.py serial_port:=/dev/ttyACM0

# 2. pose estimation with the pretrained model installed by colcon (model/final_model.onnx)
ros2 launch pose_estimator pose_estimator.launch.py
#    or your own export:  ros2 launch pose_estimator pose_estimator.launch.py model_path:=/path/to/final_model.onnx

# 3. optional: tendon teleoperation (needs an interactive terminal: docker compose run, not exec -d)
ros2 launch pose_estimator tendon_system.launch.py u2d2_port:=/dev/ttyUSB0

# look at it
ros2 run tf2_tools view_frames   /   rviz2 (fixed frame: world)   /   ros2 topic echo /coil_poses
```

## Nodes

### `imu_serial_node.py`
Parses the MEGA's text stream. `SEQMAG` rows are buffered until the
`# Sequential array dump complete` marker and published as one `SequentialMagArray` per cycle;
`AMBIENT`/`DATA` rows are republished on the `/imu/*` topics for logging. Parameters: `serial_port`
(`/dev/ttyACM0`), `baud_rate` (115200), `timeout`.

### `pose_estimator_node.py`
For each `SequentialMagArray` (one entry per joint, in chain order):

1. normalise each de-biased magnetometer vector to unit length,
2. run the ONNX MLP (`onnxruntime`, CPU) → raw direction, normalised to a unit vector,
3. decode azimuth `θ = atan2(dy, dx)` and **colatitude** `λ = acos(dz)`,
4. place segment *i+1* at `coil_radius` along `(θ, λ)` from `platform_i`, oriented so that its
   Z axis follows the direction, then `platform_{i+1}` a further `platform_z_offset` along that Z.

Published: `/tf` (`world → coil_0 → platform_0 → coil_1 → platform_1 → …`, re-broadcast at
`publish_rate`) and `/coil_poses` (`geometry_msgs/PoseArray` in `coil_0`).
Parameters: `model_path` (default: the packaged model), `coil_radius` (0.012 m), `platform_z_offset`
(0.047 m), `publish_rate` (10 Hz) — set the two lengths to your segment geometry.

The model **must** be a colatitude-convention direction model (what `training/train.py` produces;
`training/check_convention.py` verifies a given `.onnx`).

### `tendon_controller.py` / `keyboard_publisher.py`
Velocity control of the four Dynamixel XM430 tendon servos (IDs 10–13) with antagonistic
reversal (`T1/T3` on X, `T2/T4` on Y); subscribes to `/tendon_cmd_vel` (`geometry_msgs/Twist`,
`linear.x/y`) and `/tendon_emergency_stop`; publishes `motor_velocities`, `servo_positions`,
`torque_enabled`. Torque is always disabled on shutdown. The keyboard node maps
`W/A/S/D` (+ diagonals `Q/E/Z/C`) to velocities, `+/-` change the magnitude, `SPACE` stops,
`ESC` exits.

## Messages

| message | fields |
|---|---|
| `SequentialMagData` | `timestamp_ms`, `active_coil_id`, `source_bridge_id`, `mag_debias` (µT) |
| `SequentialMagArray` | `header`, `cycle_number`, `data[]` |
| `IMUData` | `header`, `timestamp_ms`, `bridge_id`, `bridge_addr`, `accel`, `gyro`, `mag` |
| `MagnetometerData` | `header`, `timestamp_ms`, `active_coil_id/addr`, `reading_bridge_id/addr`, `mag` |
| `MagnetometerBias` | `header`, `timestamp_ms`, `bridge_id`, `bridge_addr`, `bias` |

## Files

```
ros_ws/
├── Dockerfile, docker-compose.yml, requirements.txt
└── src/pose_estimator/
    ├── CMakeLists.txt, package.xml          ament_cmake package with the five custom messages
    ├── msg/                                  message definitions
    ├── launch/                               imu_serial / pose_estimator / tendon_system
    ├── config/                               parameter files (same values as the launch defaults)
    ├── scripts/                              the four nodes
    └── model/final_model.onnx (+ .json)      pretrained model, installed to share/pose_estimator/model
```
