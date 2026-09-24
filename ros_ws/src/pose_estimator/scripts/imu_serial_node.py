#!/usr/bin/env python3
"""
Multi-Bridge IMU Serial Node

Reads serial data from Arduino running the multi-bridge IMU collection code
and publishes data to ROS2 topics.

Topics published:
- /imu/accel: IMUData messages with accelerometer data
- /imu/mag_raw: MagnetometerData messages with raw magnetometer data
- /imu/mag_bias: MagnetometerBias messages with ambient bias
- /imu/mag_debias: MagnetometerData messages with debiased magnetometer data
- /imu/sequential_mag_array: SequentialMagArray messages with complete array dumps
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header
import serial
import re
from pose_estimator.msg import (
    IMUData,
    MagnetometerData,
    MagnetometerBias,
    SequentialMagData,
    SequentialMagArray
)


class IMUSerialNode(Node):
    """ROS2 node for reading and publishing multi-bridge IMU data from serial."""

    def __init__(self):
        super().__init__('imu_serial_node')
        
        # Declare parameters
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('timeout', 1.0)
        
        # Get parameters
        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.timeout = self.get_parameter('timeout').value
        
        # Create publishers
        self.accel_pub = self.create_publisher(IMUData, '/imu/accel', 10)
        self.mag_raw_pub = self.create_publisher(MagnetometerData, '/imu/mag_raw', 10)
        self.mag_bias_pub = self.create_publisher(MagnetometerBias, '/imu/mag_bias', 10)
        self.mag_debias_pub = self.create_publisher(MagnetometerData, '/imu/mag_debias', 10)
        self.sequential_array_pub = self.create_publisher(
            SequentialMagArray, '/imu/sequential_mag_array', 10
        )
        
        # Sequential array buffer
        self.sequential_buffer = []
        self.current_cycle = 0
        
        # Initialize serial connection
        self.serial_conn = None
        self.init_serial()
        
        # Create timer for reading serial data
        self.timer = self.create_timer(0.001, self.serial_callback)  # 1ms timer
        
        self.get_logger().info(f'IMU Serial Node initialized on {self.serial_port} at {self.baud_rate} baud')

    def init_serial(self):
        """Initialize serial connection to Arduino."""
        try:
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            self.get_logger().info(f'Serial connection opened: {self.serial_port}')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            self.serial_conn = None

    def parse_hex_addr(self, addr_str):
        """Parse hex address string like '0x42' to integer."""
        try:
            return int(addr_str, 16)
        except ValueError:
            return 0

    def parse_ambient_line(self, parts):
        """
        Parse AMBIENT line.
        Format: AMBIENT,timestamp,bridge_id,bridge_addr,ax,ay,az,gx,gy,gz,mx,my,mz
        """
        try:
            if len(parts) < 13:
                return
            
            timestamp_ms = int(parts[1])
            bridge_id = int(parts[2])
            bridge_addr = self.parse_hex_addr(parts[3])
            
            # Check for nan values
            if parts[4] == 'nan':
                return
            
            # Parse accelerometer data
            ax = float(parts[4])
            ay = float(parts[5])
            az = float(parts[6])
            
            # Parse gyroscope data
            gx = float(parts[7])
            gy = float(parts[8])
            gz = float(parts[9])
            
            # Parse magnetometer data (bias)
            mx = float(parts[10])
            my = float(parts[11])
            mz = float(parts[12])
            
            # Publish accelerometer data
            accel_msg = IMUData()
            accel_msg.header = Header()
            accel_msg.header.stamp = self.get_clock().now().to_msg()
            accel_msg.header.frame_id = f'imu_bridge_{bridge_id}'
            accel_msg.timestamp_ms = timestamp_ms
            accel_msg.bridge_id = bridge_id
            accel_msg.bridge_addr = bridge_addr
            accel_msg.accel = Vector3(x=ax, y=ay, z=az)
            accel_msg.gyro = Vector3(x=gx, y=gy, z=gz)
            accel_msg.mag = Vector3(x=mx, y=my, z=mz)
            self.accel_pub.publish(accel_msg)
            
            # Publish magnetometer bias
            bias_msg = MagnetometerBias()
            bias_msg.header = Header()
            bias_msg.header.stamp = self.get_clock().now().to_msg()
            bias_msg.header.frame_id = f'imu_bridge_{bridge_id}'
            bias_msg.timestamp_ms = timestamp_ms
            bias_msg.bridge_id = bridge_id
            bias_msg.bridge_addr = bridge_addr
            bias_msg.bias = Vector3(x=mx, y=my, z=mz)
            self.mag_bias_pub.publish(bias_msg)
            
        except (ValueError, IndexError) as e:
            self.get_logger().warning(f'Failed to parse AMBIENT line: {e}')

    def parse_data_line(self, parts):
        """
        Parse DATA line.
        Format: DATA,timestamp,active_coil_id,active_coil_addr,reading_bridge_id,
                reading_bridge_addr,ax,ay,az,gx,gy,gz,mx_raw,my_raw,mz_raw,
                mx_debias,my_debias,mz_debias
        """
        try:
            if len(parts) < 18:
                return
            
            timestamp_ms = int(parts[1])
            active_coil_id = int(parts[2])
            active_coil_addr = self.parse_hex_addr(parts[3])
            reading_bridge_id = int(parts[4])
            reading_bridge_addr = self.parse_hex_addr(parts[5])
            
            # Check for nan values
            if parts[6] == 'nan':
                return
            
            # Parse accelerometer data
            ax = float(parts[6])
            ay = float(parts[7])
            az = float(parts[8])
            
            # Parse gyroscope data
            gx = float(parts[9])
            gy = float(parts[10])
            gz = float(parts[11])
            
            # Parse raw magnetometer data
            mx_raw = float(parts[12])
            my_raw = float(parts[13])
            mz_raw = float(parts[14])
            
            # Parse debiased magnetometer data
            mx_debias = float(parts[15])
            my_debias = float(parts[16])
            mz_debias = float(parts[17])
            
            # Publish accelerometer data
            accel_msg = IMUData()
            accel_msg.header = Header()
            accel_msg.header.stamp = self.get_clock().now().to_msg()
            accel_msg.header.frame_id = f'imu_bridge_{reading_bridge_id}'
            accel_msg.timestamp_ms = timestamp_ms
            accel_msg.bridge_id = reading_bridge_id
            accel_msg.bridge_addr = reading_bridge_addr
            accel_msg.accel = Vector3(x=ax, y=ay, z=az)
            accel_msg.gyro = Vector3(x=gx, y=gy, z=gz)
            accel_msg.mag = Vector3(x=mx_raw, y=my_raw, z=mz_raw)
            self.accel_pub.publish(accel_msg)
            
            # Publish raw magnetometer data
            mag_raw_msg = MagnetometerData()
            mag_raw_msg.header = Header()
            mag_raw_msg.header.stamp = self.get_clock().now().to_msg()
            mag_raw_msg.header.frame_id = f'imu_bridge_{reading_bridge_id}'
            mag_raw_msg.timestamp_ms = timestamp_ms
            mag_raw_msg.active_coil_id = active_coil_id
            mag_raw_msg.active_coil_addr = active_coil_addr
            mag_raw_msg.reading_bridge_id = reading_bridge_id
            mag_raw_msg.reading_bridge_addr = reading_bridge_addr
            mag_raw_msg.mag = Vector3(x=mx_raw, y=my_raw, z=mz_raw)
            self.mag_raw_pub.publish(mag_raw_msg)
            
            # Publish debiased magnetometer data
            mag_debias_msg = MagnetometerData()
            mag_debias_msg.header = Header()
            mag_debias_msg.header.stamp = self.get_clock().now().to_msg()
            mag_debias_msg.header.frame_id = f'imu_bridge_{reading_bridge_id}'
            mag_debias_msg.timestamp_ms = timestamp_ms
            mag_debias_msg.active_coil_id = active_coil_id
            mag_debias_msg.active_coil_addr = active_coil_addr
            mag_debias_msg.reading_bridge_id = reading_bridge_id
            mag_debias_msg.reading_bridge_addr = reading_bridge_addr
            mag_debias_msg.mag = Vector3(x=mx_debias, y=my_debias, z=mz_debias)
            self.mag_debias_pub.publish(mag_debias_msg)
            
        except (ValueError, IndexError) as e:
            self.get_logger().warning(f'Failed to parse DATA line: {e}')

    def parse_seqmag_line(self, parts):
        """
        Parse SEQMAG line.
        Format: SEQMAG,timestamp,active_coil_id,source_bridge_id,mx_debias,my_debias,mz_debias
        """
        try:
            if len(parts) < 7:
                return
            
            timestamp_ms = int(parts[1])
            active_coil_id = int(parts[2])
            source_bridge_id = int(parts[3])
            
            # Check for nan values
            if parts[4] == 'nan':
                return
            
            mx_debias = float(parts[4])
            my_debias = float(parts[5])
            mz_debias = float(parts[6])
            
            # Create sequential mag data entry
            seq_data = SequentialMagData()
            seq_data.timestamp_ms = timestamp_ms
            seq_data.active_coil_id = active_coil_id
            seq_data.source_bridge_id = source_bridge_id
            seq_data.mag_debias = Vector3(x=mx_debias, y=my_debias, z=mz_debias)
            
            # Add to buffer
            self.sequential_buffer.append(seq_data)
            
        except (ValueError, IndexError) as e:
            self.get_logger().warning(f'Failed to parse SEQMAG line: {e}')

    def parse_cycle_marker(self, line):
        """Parse cycle completion marker and publish sequential array."""
        # Look for "# Sequential array dump complete"
        if "Sequential array dump complete" in line:
            if self.sequential_buffer:
                # Publish the complete array
                array_msg = SequentialMagArray()
                array_msg.header = Header()
                array_msg.header.stamp = self.get_clock().now().to_msg()
                array_msg.header.frame_id = 'imu_sequential'
                array_msg.cycle_number = self.current_cycle
                array_msg.data = self.sequential_buffer
                
                self.sequential_array_pub.publish(array_msg)
                self.get_logger().info(
                    f'Published sequential array with {len(self.sequential_buffer)} elements'
                )
                
                # Clear buffer for next cycle
                self.sequential_buffer = []
        
        # Look for cycle number
        cycle_match = re.search(r'Cycle (\d+)', line)
        if cycle_match:
            self.current_cycle = int(cycle_match.group(1))

    def serial_callback(self):
        """Read and parse serial data from Arduino."""
        if self.serial_conn is None or not self.serial_conn.is_open:
            return
        
        try:
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                
                # Skip empty lines and comments (except cycle markers)
                if not line:
                    return
                
                if line.startswith('#'):
                    self.parse_cycle_marker(line)
                    return
                
                # Parse CSV data lines
                parts = line.split(',')
                if len(parts) < 2:
                    return
                
                data_type = parts[0]
                
                if data_type == 'AMBIENT':
                    self.parse_ambient_line(parts)
                elif data_type == 'DATA':
                    self.parse_data_line(parts)
                elif data_type == 'SEQMAG':
                    self.parse_seqmag_line(parts)
                
        except serial.SerialException as e:
            self.get_logger().error(f'Serial error: {e}')
        except UnicodeDecodeError as e:
            self.get_logger().warning(f'Unicode decode error: {e}')

    def destroy_node(self):
        """Clean up serial connection on shutdown."""
        if self.serial_conn is not None and self.serial_conn.is_open:
            self.serial_conn.close()
            self.get_logger().info('Serial connection closed')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IMUSerialNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()