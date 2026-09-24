#!/usr/bin/env python3
"""Launch file for multi-bridge IMU serial node."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyACM0',
        description='Serial port of the Arduino MEGA running MultiBridgeCollectionFast.ino '
                    '(prefer the stable /dev/serial/by-id/usb-Arduino_... path)'
    )
    
    baud_rate_arg = DeclareLaunchArgument(
        'baud_rate',
        default_value='115200',
        description='Baud rate for serial communication'
    )
    
    timeout_arg = DeclareLaunchArgument(
        'timeout',
        default_value='1.0',
        description='Serial read timeout in seconds'
    )
    
    # Create node
    imu_serial_node = Node(
        package='pose_estimator',
        executable='imu_serial_node.py',
        name='imu_serial_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': LaunchConfiguration('baud_rate'),
            'timeout': LaunchConfiguration('timeout'),
        }]
    )
    
    return LaunchDescription([
        serial_port_arg,
        baud_rate_arg,
        timeout_arg,
        imu_serial_node,
    ])