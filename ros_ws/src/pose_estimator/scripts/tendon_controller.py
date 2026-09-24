#!/usr/bin/env python3
"""
ROS2 Node: Tendon Velocity Controller with Antagonistic Reversal
Controls Dynamixel servos in VELOCITY mode based on velocity commands

ANTAGONISTIC MOTOR BEHAVIOR:
Instead of blocking, antagonistic motors rotate in OPPOSITE directions
- Motor 1 CW → Motor 3 CCW (balances X-axis tension)
- Motor 3 CW → Motor 1 CCW
- Motor 2 CW → Motor 4 CCW (balances Y-axis tension)
- Motor 4 CW → Motor 2 CCW

Motor Mapping:
    Motor 1 (ID 10): +X direction
    Motor 2 (ID 11): +Y direction
    Motor 3 (ID 12): -X direction (antagonistic to Motor 1)
    Motor 4 (ID 13): -Y direction (antagonistic to Motor 2)

Hardware:
    - 4x Dynamixel XM servos via U2D2
    - Operating in VELOCITY CONTROL mode
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Float32MultiArray, Bool, Int32MultiArray
import sys
import signal
from dynamixel_sdk import *

# ==================== CONFIGURATION ====================
# Dynamixel Settings
PROTOCOL_VERSION = 2.0
DXL_IDS = [10, 11, 12, 13]  # Motor 1, 2, 3, 4

# Control Table Addresses (XM series)
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_VELOCITY = 104
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

# Operating Modes
VELOCITY_CONTROL_MODE = 1

# Velocity limits (Dynamixel units)
# For XM430-W350: 1 unit = 0.229 rpm
# Max velocity ~= 57 rpm (manufacturer spec)
MAX_VELOCITY_UNITS = 250  # ~57 rpm, adjust based on your needs


# ==================== DYNAMIXEL CONTROLLER CLASS ====================
class DynamixelVelocityController:
    def __init__(self, port, baudrate):
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.connected = False
        self.baudrate = baudrate
        
    def connect(self):
        """Connect to Dynamixel servos"""
        if not self.port_handler.openPort():
            return False
        if not self.port_handler.setBaudRate(self.baudrate):
            return False
        self.connected = True
        return True
    
    def set_operating_mode(self, dxl_id, mode):
        """Set operating mode (must disable torque first)"""
        # Disable torque
        self.packet_handler.write1ByteTxRx(
            self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, 0)
        
        # Set mode
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, dxl_id, ADDR_OPERATING_MODE, mode)
        
        # Re-enable torque
        self.packet_handler.write1ByteTxRx(
            self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, 1)
        
        return result == COMM_SUCCESS
    
    def enable_torque(self, dxl_id, enable=True):
        """Enable or disable torque for a servo"""
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, 1 if enable else 0)
        return result == COMM_SUCCESS
    
    def set_goal_velocity(self, dxl_id, velocity):
        """
        Set goal velocity for a servo
        velocity: signed integer in Dynamixel units
                  positive = CCW, negative = CW
        """
        # Clamp velocity
        velocity = max(-MAX_VELOCITY_UNITS, min(MAX_VELOCITY_UNITS, int(velocity)))
        
        # Dynamixel velocity is 4-byte signed integer
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, dxl_id, ADDR_GOAL_VELOCITY, velocity)
        
        return result == COMM_SUCCESS
    
    def get_present_velocity(self, dxl_id):
        """Read current velocity"""
        velocity, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, dxl_id, ADDR_PRESENT_VELOCITY)
        if result != COMM_SUCCESS:
            return None
        # Convert unsigned to signed
        if velocity > 0x7FFFFFFF:
            velocity = velocity - 0x100000000
        return velocity
    
    def get_present_position(self, dxl_id):
        """Read current position"""
        position, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, dxl_id, ADDR_PRESENT_POSITION)
        if result != COMM_SUCCESS:
            return None
        return position
    
    def set_all_velocities(self, velocities):
        """Set velocities for all servos simultaneously"""
        for servo_id, velocity in velocities.items():
            self.set_goal_velocity(servo_id, velocity)
    
    def stop_all(self):
        """Stop all motors"""
        for servo_id in DXL_IDS:
            self.set_goal_velocity(servo_id, 0)
    
    def close(self):
        """Close connection"""
        self.port_handler.closePort()


# ==================== ROS2 NODE CLASS ====================
class TendonController(Node):
    def __init__(self):
        super().__init__('tendon_controller')
        
        # Declare parameters
        self.declare_parameter('u2d2_port', '/dev/ttyUSB0')
        self.declare_parameter('u2d2_baud', 57600)
        self.declare_parameter('control_rate', 50.0)  # Hz
        self.declare_parameter('velocity_scale', 1.0)  # Scale factor for velocity commands
        
        # Get parameters
        u2d2_port = self.get_parameter('u2d2_port').value
        u2d2_baud = self.get_parameter('u2d2_baud').value
        control_rate = self.get_parameter('control_rate').value
        self.velocity_scale = self.get_parameter('velocity_scale').value
        
        # Initialize Dynamixel controller
        self.dxl = DynamixelVelocityController(u2d2_port, u2d2_baud)
        if not self.dxl.connect():
            self.get_logger().error(f"Failed to connect to Dynamixel controller on {u2d2_port}")
            sys.exit(1)
        
        # Current commanded velocities (for publishing feedback)
        self.commanded_velocities = {10: 0, 11: 0, 12: 0, 13: 0}
        
        # Torque enabled state
        self.torque_enabled = False
        
        # Publishers
        self.velocity_feedback_pub = self.create_publisher(
            Int32MultiArray, 'motor_velocities', 10)
        self.position_pub = self.create_publisher(
            Float32MultiArray, 'servo_positions', 10)
        self.torque_state_pub = self.create_publisher(Bool, 'torque_enabled', 10)
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'tendon_cmd_vel', self.cmd_vel_callback, 10)
        self.emergency_stop_sub = self.create_subscription(
            Bool, 'tendon_emergency_stop', self.emergency_stop_callback, 10)
        
        # Timer for control loop and publishing state
        self.timer = self.create_timer(1.0 / control_rate, self.control_loop)
        
        # Initialize servos to velocity mode
        self.initialize_servos()
        
        self.get_logger().info("Tendon Velocity Controller Started")
        self.get_logger().info(f"Port: {u2d2_port} @ {u2d2_baud}")
        self.get_logger().info(f"Operating Mode: VELOCITY CONTROL with ANTAGONISTIC REVERSAL")
        self.get_logger().info(f"Motor Mapping:")
        self.get_logger().info(f"  Motor 1 (ID 10): +X direction")
        self.get_logger().info(f"  Motor 2 (ID 11): +Y direction")
        self.get_logger().info(f"  Motor 3 (ID 12): -X direction (antagonistic to Motor 1)")
        self.get_logger().info(f"  Motor 4 (ID 13): -Y direction (antagonistic to Motor 2)")
        self.get_logger().info(f"Antagonistic Behavior: REVERSE direction when both active")
        self.get_logger().info(f"Subscribed to: {self.cmd_vel_sub.topic_name}")
        
    def initialize_servos(self):
        """Initialize servos to velocity control mode"""
        self.get_logger().info("Initializing servos to velocity control mode...")
        
        for servo_id in DXL_IDS:
            # Set to velocity control mode
            success = self.dxl.set_operating_mode(servo_id, VELOCITY_CONTROL_MODE)
            if success:
                self.get_logger().info(f"  Motor {DXL_IDS.index(servo_id)+1} (ID {servo_id}): Velocity mode ✓")
            else:
                self.get_logger().error(f"  Motor {DXL_IDS.index(servo_id)+1} (ID {servo_id}): Failed to set mode")
        
        self.torque_enabled = True
        
        # Stop all motors initially
        self.dxl.stop_all()
        
        self.get_logger().info("All servos initialized and stopped")
        self.publish_torque_state()
    
    def velocity_to_motor_commands(self, vx, vy):
        """
        Convert velocity command to individual motor velocities
        WITH ANTAGONISTIC MOTOR REVERSAL
        
        vx, vy: velocity commands from /tendon_cmd_vel
        
        Returns: dict of {motor_id: velocity}
        
        ANTAGONISTIC REVERSAL BEHAVIOR:
        When a motor is commanded to rotate in one direction,
        its antagonistic partner rotates in the OPPOSITE direction.
        
        Example:
        - Motor 1 CW (+velocity) → Motor 3 CCW (-velocity)
        - Motor 1 CCW (-velocity) → Motor 3 CW (+velocity)
        """
        velocities = {10: 0, 11: 0, 12: 0, 13: 0}
        
        # Apply velocity scale
        vx = vx * self.velocity_scale
        vy = vy * self.velocity_scale
        
        # X-axis control - Motor 1 and Motor 3 rotate in OPPOSITE directions
        # Motor 1: primary control
        # Motor 3: follows with reversed direction
        velocities[10] = int(vx)   # Motor 1: command velocity
        velocities[12] = int(-vx)    # Motor 3: OPPOSITE direction (antagonistic reversal)
        
        # Y-axis control - Motor 2 and Motor 4 rotate in OPPOSITE directions
        # Motor 2: primary control
        # Motor 4: follows with reversed direction
        velocities[11] = int(-vy)   # Motor 2: command velocity
        velocities[13] = int(vy)    # Motor 4: OPPOSITE direction (antagonistic reversal)
        
        return velocities
    
    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        vx = msg.linear.x
        vy = msg.linear.y
        
        # Convert to motor velocities with antagonistic reversal
        velocities = self.velocity_to_motor_commands(vx, vy)
        
        # Send to motors (only if torque is enabled)
        if self.torque_enabled:
            self.dxl.set_all_velocities(velocities)
            self.commanded_velocities = velocities
            
            # Log active motors with their directions
            active_motors = []
            if velocities[10] != 0:
                direction = "CW" if velocities[10] < 0 else "CCW"
                active_motors.append(f"M1({direction}):{velocities[10]}")
            if velocities[11] != 0:
                direction = "CW" if velocities[11] < 0 else "CCW"
                active_motors.append(f"M2({direction}):{velocities[11]}")
            if velocities[12] != 0:
                direction = "CW" if velocities[12] < 0 else "CCW"
                active_motors.append(f"M3({direction}):{velocities[12]}")
            if velocities[13] != 0:
                direction = "CW" if velocities[13] < 0 else "CCW"
                active_motors.append(f"M4({direction}):{velocities[13]}")
            
            if active_motors:
                self.get_logger().debug(f"Active: {', '.join(active_motors)}")
    
    def emergency_stop_callback(self, msg):
        """Handle emergency stop command"""
        if msg.data:
            self.emergency_stop()
    
    def emergency_stop(self):
        """Stop all motors and disable torque"""
        self.get_logger().warn("EMERGENCY STOP - Stopping all motors and disabling torque")
        
        # Stop all motors
        self.dxl.stop_all()
        
        # Disable torque
        for servo_id in DXL_IDS:
            self.dxl.enable_torque(servo_id, False)
        
        self.torque_enabled = False
        self.commanded_velocities = {10: 0, 11: 0, 12: 0, 13: 0}
        self.publish_torque_state()
    
    def stop_all_motors(self):
        """Stop all motors but keep torque enabled"""
        self.get_logger().info("Stopping all motors")
        self.dxl.stop_all()
        self.commanded_velocities = {10: 0, 11: 0, 12: 0, 13: 0}
    
    def control_loop(self):
        """Main control loop - publishes state"""
        self.publish_velocities()
        self.publish_positions()
    
    def publish_velocities(self):
        """Publish current motor velocities"""
        msg = Int32MultiArray()
        # Order: Motor 1, 2, 3, 4 (IDs 10, 11, 12, 13)
        msg.data = [
            self.commanded_velocities[10],
            self.commanded_velocities[11],
            self.commanded_velocities[12],
            self.commanded_velocities[13]
        ]
        self.velocity_feedback_pub.publish(msg)
    
    def publish_positions(self):
        """Publish current servo positions (for monitoring)"""
        positions = []
        for servo_id in DXL_IDS:
            pos = self.dxl.get_present_position(servo_id)
            positions.append(float(pos) if pos is not None else 0.0)
        
        msg = Float32MultiArray()
        msg.data = positions
        self.position_pub.publish(msg)
    
    def publish_torque_state(self):
        """Publish torque state"""
        msg = Bool()
        msg.data = self.torque_enabled
        self.torque_state_pub.publish(msg)
    
    def cleanup(self):
        """Cleanup before shutdown - ALWAYS disables torque"""
        self.get_logger().warn("="*60)
        self.get_logger().warn("SHUTTING DOWN - DISABLING ALL TORQUE")
        self.get_logger().warn("="*60)
        
        # Stop all motors FIRST
        try:
            self.get_logger().info("Step 1: Stopping all motors...")
            self.dxl.stop_all()
            import time
            time.sleep(0.1)  # Brief delay to ensure stop command is sent
            self.get_logger().info("  ✓ All motors stopped")
        except Exception as e:
            self.get_logger().error(f"  ✗ Error stopping motors: {e}")
        
        # Disable torque on ALL motors
        try:
            self.get_logger().info("Step 2: Disabling torque on all motors...")
            for servo_id in DXL_IDS:
                success = self.dxl.enable_torque(servo_id, False)
                if success:
                    self.get_logger().info(f"  ✓ Motor {servo_id}: Torque disabled")
                else:
                    self.get_logger().warn(f"  ✗ Motor {servo_id}: Failed to disable torque")
        except Exception as e:
            self.get_logger().error(f"  ✗ Error disabling torque: {e}")
        
        # Update state
        self.torque_enabled = False
        self.commanded_velocities = {10: 0, 11: 0, 12: 0, 13: 0}
        
        # Close connection
        try:
            self.get_logger().info("Step 3: Closing serial connection...")
            self.dxl.close()
            self.get_logger().info("  ✓ Connection closed")
        except Exception as e:
            self.get_logger().error(f"  ✗ Error closing connection: {e}")
        
        self.get_logger().warn("="*60)
        self.get_logger().warn("SHUTDOWN COMPLETE - ALL TORQUE DISABLED")
        self.get_logger().warn("="*60)


# Global controller reference for signal handler
_controller_instance = None

def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    global _controller_instance
    print("\n")
    print("="*60)
    print("!!! CTRL+C DETECTED - EMERGENCY SHUTDOWN !!!")
    print("="*60)
    
    if _controller_instance:
        try:
            _controller_instance.cleanup()
        except Exception as e:
            print(f"Error during emergency cleanup: {e}")
    
    print("Exiting...")
    sys.exit(0)


# ==================== MAIN ====================
def main(args=None):
    global _controller_instance
    
    rclpy.init(args=args)
    
    controller = None
    try:
        controller = TendonController()
        _controller_instance = controller
        
        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)
        
        rclpy.spin(controller)
        
    except KeyboardInterrupt:
        # This should be caught by signal handler, but keep as backup
        print("\n")
        print("="*60)
        print("!!! KEYBOARD INTERRUPT DETECTED !!!")
        print("="*60)
        if controller:
            controller.get_logger().warn("Keyboard interrupt - disabling torque...")
            
    except Exception as e:
        if controller:
            controller.get_logger().error(f"Exception occurred: {e}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # ALWAYS cleanup, even on Ctrl+C or exceptions
        if controller:
            try:
                controller.cleanup()
            except Exception as e:
                print(f"Error during cleanup: {e}")
        
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()