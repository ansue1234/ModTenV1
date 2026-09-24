#!/usr/bin/env python3
"""Launch file for pose estimator node."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=PathJoinSubstitution([FindPackageShare('pose_estimator'), 'model', 'final_model.onnx']),
        description='Path to the ONNX model (default: the pretrained model installed with the package)'
    )
    
    coil_radius_arg = DeclareLaunchArgument(
        'coil_radius',
        default_value='0.012',
        description='Radius of coil positions in meters'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='10.0',
        description='TF publish rate in Hz'
    )

    platform_z_offset_arg = DeclareLaunchArgument(
        'platform_z_offset',
        default_value='0.047',
        description='Z offset from coil frame to platform frame in metres (default 47 mm)'
    )
    
    # Create node
    pose_estimator_node = Node(
        package='pose_estimator',
        executable='pose_estimator_node.py',
        name='pose_estimator_node',
        output='screen',
        parameters=[{
            'model_path':        LaunchConfiguration('model_path'),
            'coil_radius':       LaunchConfiguration('coil_radius'),
            'publish_rate':      LaunchConfiguration('publish_rate'),
            'platform_z_offset': LaunchConfiguration('platform_z_offset'),
        }]
    )
    
    return LaunchDescription([
        model_path_arg,
        coil_radius_arg,
        publish_rate_arg,
        platform_z_offset_arg,
        pose_estimator_node,
    ])