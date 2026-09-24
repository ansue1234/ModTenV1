from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """
    Tendon teleoperation (ROS 2 Humble): keyboard velocity publisher +
    Dynamixel tendon velocity controller.

    Usage (needs an interactive terminal for the keyboard node):
        ros2 launch pose_estimator tendon_system.launch.py

    Custom parameters:
        ros2 launch pose_estimator tendon_system.launch.py \
            u2d2_port:=/dev/ttyUSB0 default_velocity:=30.0
    """
    
    # Declare launch arguments
    u2d2_port_arg = DeclareLaunchArgument(
        'u2d2_port',
        default_value='/dev/ttyUSB0',
        description='U2D2 serial port (prefer the stable /dev/serial/by-id/usb-FTDI_... path)'
    )
    
    u2d2_baud_arg = DeclareLaunchArgument(
        'u2d2_baud',
        default_value='57600',
        description='U2D2 baudrate'
    )
    
    default_velocity_arg = DeclareLaunchArgument(
        'default_velocity',
        default_value='20.0',
        description='Default velocity magnitude for keyboard control'
    )
    
    velocity_scale_arg = DeclareLaunchArgument(
        'velocity_scale',
        default_value='1.0',
        description='Scale factor for velocity commands (controller side)'
    )
    
    control_rate_arg = DeclareLaunchArgument(
        'control_rate',
        default_value='50.0',
        description='Control loop rate in Hz'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='20.0',
        description='Keyboard publish rate in Hz'
    )
    
    # Keyboard Velocity Publisher Node
    keyboard_node = Node(
        package='pose_estimator',
        executable='keyboard_publisher.py',
        name='keyboard_velocity_publisher',
        output='screen',
        parameters=[{
            'default_velocity': LaunchConfiguration('default_velocity'),
            'publish_rate': LaunchConfiguration('publish_rate'),
        }],
    )
    
    # Tendon Velocity Controller Node
    controller_node = Node(
        package='pose_estimator',
        executable='tendon_controller.py',
        name='tendon_controller',
        output='screen',
        parameters=[{
            'u2d2_port': LaunchConfiguration('u2d2_port'),
            'u2d2_baud': LaunchConfiguration('u2d2_baud'),
            'velocity_scale': LaunchConfiguration('velocity_scale'),
            'control_rate': LaunchConfiguration('control_rate'),
        }]
    )
    
    return LaunchDescription([
        # Arguments
        u2d2_port_arg,
        u2d2_baud_arg,
        default_velocity_arg,
        velocity_scale_arg,
        control_rate_arg,
        publish_rate_arg,
        
        # Nodes
        controller_node,  # Start controller first (initializes hardware)
        keyboard_node,    # Then start keyboard interface
    ])