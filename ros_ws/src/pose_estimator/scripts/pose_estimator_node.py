#!/usr/bin/env python3
"""
Pose Estimator Node

Subscribes to sequential magnetometer array data and uses an ONNX neural network
to predict the relative position of each coil in 3D space.

Architecture:
  - Input:  (batch_size, 3) normalised magnetometer unit vectors
  - Output: (batch_size, 3) raw direction vector [dx, dy, dz]
             → normalised to unit length at inference time
             → converted to azimuth (theta) + colatitude (phi)

Coordinate convention (colatitude version):
  dx = sin(phi) * cos(theta)
  dy = sin(phi) * sin(theta)
  dz = cos(phi)

  theta : azimuth     in [-180, 180] deg  (rotation around Z)
  phi   : colatitude  in [   0, 180] deg  (angle down from +Z)

TF Tree:
  world → coil_0 (origin / reference, identity transform)
  coil_0 → platform_0 → coil_1 → platform_1 → coil_2 → ...

Published topics:
  /coil_poses   (geometry_msgs/PoseArray)
  /tf           (via tf2_ros.TransformBroadcaster)
"""

import rclpy
from rclpy.node import Node
from pose_estimator.msg import SequentialMagArray
from geometry_msgs.msg import TransformStamped, PoseArray, Pose, Point, Quaternion
import tf2_ros
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
    """
    Normalize each row of vecs to unit length.
    Rows with near-zero norm are set to [0, 0, 0] rather than exploding.
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-10, 1.0, norms)
    return vecs / safe_norms


def unit_dir_to_angles_deg(d: np.ndarray):
    """
    Convert unit direction vectors to azimuth / colatitude angles.

    Spherical convention used here:
      theta = azimuth     = arctan2(dy, dx)  in [-180, 180] deg
      phi   = colatitude  = arccos(dz)       in [   0, 180] deg

    Parameters
    ----------
    d : (N, 3) array, rows are unit direction vectors [dx, dy, dz]

    Returns
    -------
    theta_deg : (N,) azimuth in degrees
    phi_deg   : (N,) colatitude in degrees
    """
    d = d.astype(np.float64)

    # Re-normalise defensively
    norms = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    d = d / norms

    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]

    theta_deg = np.degrees(np.arctan2(dy, dx))              # [-180, 180]
    phi_deg   = np.degrees(np.arccos(np.clip(dz, -1.0, 1.0)))  # [0, 180]

    return theta_deg.astype(np.float64), phi_deg.astype(np.float64)


def angles_deg_to_cartesian(theta_deg: float, phi_deg: float, radius: float):
    """
    Convert azimuth/colatitude + radius to Cartesian XYZ.

    Colatitude convention:
      x = r * sin(phi) * cos(theta)
      y = r * sin(phi) * sin(theta)
      z = r * cos(phi)

    Parameters
    ----------
    theta_deg : azimuth in degrees
    phi_deg   : colatitude in degrees
    radius    : radial distance in metres

    Returns
    -------
    (x, y, z) as Python floats
    """
    th = np.deg2rad(float(theta_deg))
    ph = np.deg2rad(float(phi_deg))

    x = radius * np.sin(ph) * np.cos(th)
    y = radius * np.sin(ph) * np.sin(th)
    z = radius * np.cos(ph)

    return float(x), float(y), float(z)


def orientation_from_angles(theta_deg: float, phi_deg: float):
    """
    Compute the coil orientation quaternion directly from azimuth and colatitude.

    The coil's local Z-axis points in the direction (theta, phi).
    No yaw rotation is applied — the coil only tilts from world Z.

    Colatitude convention:
      d = [sin(phi)cos(theta), sin(phi)sin(theta), cos(phi)]

    Parameters
    ----------
    theta_deg : azimuth in degrees   [-180, 180]
    phi_deg   : colatitude in degrees [0, 180]

    Returns
    -------
    [qx, qy, qz, qw] as a list of Python floats
    """
    th = np.deg2rad(float(theta_deg))
    ph = np.deg2rad(float(phi_deg))

    z_target = np.array([
        np.sin(ph) * np.cos(th),
        np.sin(ph) * np.sin(th),
        np.cos(ph),
    ])

    z_ref = np.array([0.0, 0.0, 1.0])

    if np.allclose(z_target, z_ref):
        return [0.0, 0.0, 0.0, 1.0]

    if np.allclose(z_target, -z_ref):
        return [1.0, 0.0, 0.0, 0.0]

    rot, _ = R.align_vectors([z_target], [z_ref])
    quat = rot.as_quat()  # [qx, qy, qz, qw]
    return [float(q) for q in quat]


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class PoseEstimatorNode(Node):
    """
    ROS2 Humble node — magnetometer data → ONNX MLP → coil TF tree.

    ROS2 Parameters
    ---------------
    model_path        : str   Path to ONNX model file (required)
    coil_radius       : float Radial distance of each coil from origin, metres
    publish_rate      : float TF re-broadcast rate, Hz
    platform_z_offset : float Platform offset along local coil Z, metres
    """

    def __init__(self):
        super().__init__('pose_estimator_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('model_path', '')
        self.declare_parameter('coil_radius', 1.0)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('platform_z_offset', 0.047)

        model_path = self.get_parameter('model_path').value
        self.radius = float(self.get_parameter('coil_radius').value)
        publish_rate = float(self.get_parameter('publish_rate').value)
        self.platform_z_off = float(self.get_parameter('platform_z_offset').value)

        if not model_path:
            self.get_logger().error('model_path parameter is required!')
            raise ValueError('model_path parameter must be set.')

        # ── Load ONNX model ─────────────────────────────────────────────────
        self.get_logger().info(f'Loading ONNX model: {model_path}')
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            in_shape = self.session.get_inputs()[0].shape
            out_shape = self.session.get_outputs()[0].shape
            self.get_logger().info(
                f'Model loaded  |  input: {in_shape}  output: {out_shape}'
            )

            if out_shape[-1] != 3:
                raise ValueError(
                    f'Expected model output dim=3 ([dx,dy,dz]), got {out_shape[-1]}'
                )

        except Exception as exc:
            self.get_logger().error(f'Failed to load ONNX model: {exc}')
            raise

        # ── ROS2 pub / sub ──────────────────────────────────────────────────
        self.subscription = self.create_subscription(
            SequentialMagArray,
            '/imu/sequential_mag_array',
            self.array_callback,
            10,
        )

        self.pose_array_pub = self.create_publisher(PoseArray, '/coil_poses', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── Periodic TF re-broadcast ────────────────────────────────────────
        self.latest_transforms: list = []
        self.create_timer(1.0 / publish_rate, self._republish_tf)

        self.get_logger().info(
            f'Pose Estimator ready  |  radius={self.radius} m  |  '
            f'platform_z_offset={self.platform_z_off*1000:.1f} mm  |  TF rate={publish_rate} Hz'
        )

    # ── Inference ──────────────────────────────────────────────────────────

    def _run_inference(self, mag_array: np.ndarray) -> np.ndarray:
        """
        Run ONNX model on a (batch, 3) magnetometer array.

        Pipeline:
          1. Normalize each input row to a unit vector
          2. Forward pass → raw (batch, 3) direction logits
          3. Normalize outputs to unit vectors

        Returns
        -------
        unit_dirs : (batch, 3) float32 unit direction vectors
        """
        mag_norm = normalize_vectors(mag_array.astype(np.float32))

        raw_out = self.session.run(
            [self.output_name],
            {self.input_name: mag_norm},
        )[0]

        unit_dirs = normalize_vectors(raw_out.astype(np.float64)).astype(np.float32)
        return unit_dirs

    # ── Main callback ──────────────────────────────────────────────────────

    def array_callback(self, msg: SequentialMagArray) -> None:
        """Receive sequential mag array, run inference, publish TF + PoseArray."""
        batch_size = len(msg.data)
        if batch_size == 0:
            self.get_logger().warning('Received empty sequential mag array — skipping.')
            return

        self.get_logger().info(
            f'Cycle {msg.cycle_number}  |  batch_size={batch_size}'
        )

        # ── Extract magnetometer vectors ────────────────────────────────────
        mag_rows = []
        for i, entry in enumerate(msg.data):
            try:
                mag_rows.append([
                    float(entry.mag_debias.x),
                    float(entry.mag_debias.y),
                    float(entry.mag_debias.z),
                ])
            except Exception as exc:
                self.get_logger().error(
                    f'Failed to read mag data from entry {i}: {exc}  '
                    f'(type={type(entry.mag_debias)})'
                )
                return

        mag_array = np.array(mag_rows, dtype=np.float32)

        # ── ONNX inference ──────────────────────────────────────────────────
        try:
            unit_dirs = self._run_inference(mag_array)
        except Exception as exc:
            self.get_logger().error(f'Inference failed: {exc}')
            return

        # ── Convert unit direction vectors → azimuth / colatitude ──────────
        theta_arr, phi_arr = unit_dir_to_angles_deg(unit_dirs)

        # Log predictions
        for i in range(batch_size):
            self.get_logger().info(
                f'  Coil {i+1:>2d}  |  '
                f'theta={theta_arr[i]:+8.2f}°  phi={phi_arr[i]:+7.2f}°  |  '
                f'dir=({unit_dirs[i,0]:+.4f}, {unit_dirs[i,1]:+.4f}, {unit_dirs[i,2]:+.4f})'
            )

        # ── Build TF transforms + PoseArray ────────────────────────────────
        stamp = self.get_clock().now().to_msg()
        transforms = []
        poses = []

        PLATFORM_Z_OFFSET = self.platform_z_off

        # world -> coil_0
        t0 = TransformStamped()
        t0.header.stamp = stamp
        t0.header.frame_id = 'world'
        t0.child_frame_id = 'coil_0'
        t0.transform.translation.x = 0.0
        t0.transform.translation.y = 0.0
        t0.transform.translation.z = 0.0
        t0.transform.rotation.x = 0.0  # 180 deg rotation around X to flip Z
        t0.transform.rotation.y = -np.sqrt(0.5)
        t0.transform.rotation.z = 0.0
        t0.transform.rotation.w = np.sqrt(0.5)
        transforms.append(t0)

        poses.append(Pose(
            position=Point(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ))

        # coil_0 -> platform_0
        tp0 = TransformStamped()
        tp0.header.stamp = stamp
        tp0.header.frame_id = 'coil_0'
        tp0.child_frame_id = 'platform_0'
        tp0.transform.translation.x = 0.0
        tp0.transform.translation.y = 0.0
        tp0.transform.translation.z = PLATFORM_Z_OFFSET
        tp0.transform.rotation.x = 0.0
        tp0.transform.rotation.y = 0.0
        tp0.transform.rotation.z = 0.0
        tp0.transform.rotation.w = 1.0
        transforms.append(tp0)

        # platform_i -> coil_{i+1} -> platform_{i+1}
        for i in range(batch_size):
            x, y, z = angles_deg_to_cartesian(
                theta_arr[i], phi_arr[i], self.radius
            )
            quat = orientation_from_angles(theta_arr[i], phi_arr[i])

            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = f'platform_{i}'
            t.child_frame_id = f'coil_{i+1}'
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]
            transforms.append(t)

            poses.append(Pose(
                position=Point(x=x, y=y, z=z),
                orientation=Quaternion(
                    x=quat[0], y=quat[1], z=quat[2], w=quat[3]
                ),
            ))

            tp = TransformStamped()
            tp.header.stamp = stamp
            tp.header.frame_id = f'coil_{i+1}'
            tp.child_frame_id = f'platform_{i+1}'
            tp.transform.translation.x = 0.0
            tp.transform.translation.y = 0.0
            tp.transform.translation.z = PLATFORM_Z_OFFSET
            tp.transform.rotation.x = 0.0
            tp.transform.rotation.y = 0.0
            tp.transform.rotation.z = 0.0
            tp.transform.rotation.w = 1.0
            transforms.append(tp)

        # ── Publish ─────────────────────────────────────────────────────────
        for t in transforms:
            self.tf_broadcaster.sendTransform(t)

        self.latest_transforms = transforms

        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = 'coil_0'
        pa.poses = poses
        self.pose_array_pub.publish(pa)

        self.get_logger().info(
            f'Published {len(transforms)} TF frames + PoseArray'
        )

    # ── Periodic TF re-broadcast ────────────────────────────────────────────

    def _republish_tf(self) -> None:
        """Re-send latest TF transforms so the TF tree doesn't expire."""
        if not self.latest_transforms:
            return

        stamp = self.get_clock().now().to_msg()
        for t in self.latest_transforms:
            t.header.stamp = stamp
            self.tf_broadcaster.sendTransform(t)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    try:
        node = PoseEstimatorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'Fatal error: {exc}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()