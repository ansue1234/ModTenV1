#!/usr/bin/env python3
"""
ROS2 Node: Keyboard Velocity Command Publisher
Publishes velocity commands for direct motor control

Motor Mapping:
    Motor 1 (ID 10): +X direction
    Motor 2 (ID 11): +Y direction
    Motor 3 (ID 12): -X direction (antagonistic to Motor 1)
    Motor 4 (ID 13): -Y direction (antagonistic to Motor 2)

Controls:
    W/S: +Y / -Y velocity (Motors 2/4)
    A/D: -X / +X velocity (Motors 3/1)
    Q/E/Z/C: Diagonal velocities
    +/-: Increase/decrease velocity magnitude
    SPACE: Stop all motors
    R: Reset/Stop
    ESC/X: Exit
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import sys
import select
import tty
import termios

# ==================== CONFIGURATION ====================
MIN_VELOCITY = 0.0
MAX_VELOCITY = 100.0  # Maximum velocity (adjust based on your system)
VELOCITY_INCREMENT = 5.0
DEFAULT_VELOCITY = 20.0


# ==================== ROS2 NODE CLASS ====================
class KeyboardPublisher(Node):
    def __init__(self):
        super().__init__('keyboard_publisher')
        
        # Declare parameters
        self.declare_parameter('default_velocity', DEFAULT_VELOCITY)
        
        # Get parameters
        self.velocity_magnitude = self.get_parameter('default_velocity').value
        
        # Check if running in interactive terminal
        self.is_interactive = self.check_interactive_terminal()
        
        if not self.is_interactive:
            self.get_logger().error("="*60)
            self.get_logger().error("NOT RUNNING IN INTERACTIVE TERMINAL")
            self.get_logger().error("="*60)
            self.get_logger().error("Run with: docker run -it ...")
            self.get_logger().error("Or run on host machine")
            self.get_logger().error("="*60)
            sys.exit(1)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'tendon_cmd_vel', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, 'tendon_emergency_stop', 10)
        
        # Current velocity command
        self.current_vel = Twist()
        
        # Terminal settings
        try:
            self.settings = termios.tcgetattr(sys.stdin)
        except termios.error as e:
            self.get_logger().error(f"Cannot get terminal settings: {e}")
            sys.exit(1)
        
        self.get_logger().info("Keyboard Velocity Publisher Started")
        self.get_logger().info(f"Publishing to: {self.cmd_vel_pub.topic_name}")
        self.get_logger().info("Mode: VELOCITY CONTROL (publish on keypress only)")
        self.print_instructions()
    
    def check_interactive_terminal(self):
        """Check if running in an interactive terminal"""
        try:
            if not sys.stdin.isatty():
                return False
            termios.tcgetattr(sys.stdin)
            return True
        except (AttributeError, termios.error):
            return False
    
    def get_key(self, timeout=0.1):
        """Get keyboard input with timeout"""
        try:
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                key = sys.stdin.read(1)
            else:
                key = ''
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            return key
        except Exception as e:
            self.get_logger().error(f"Error reading keyboard: {e}")
            return ''
    
    def print_instructions(self):
        """Print control instructions"""
        print("\n" + "="*70)
        print("  KEYBOARD VELOCITY CONTROL (Publish on Keypress)")
        print("="*70)
        print("  Motor Mapping:")
        print("    Motor 1 (ID 10): +X direction")
        print("    Motor 2 (ID 11): +Y direction")
        print("    Motor 3 (ID 12): -X direction (antagonistic to Motor 1)")
        print("    Motor 4 (ID 13): -Y direction (antagonistic to Motor 2)")
        print("")
        print("  Movement (publishes once per keypress):")
        print("    W : +Y velocity (Motor 2 only)")
        print("    S : -Y velocity (Motor 4 only)")
        print("    D : +X velocity (Motor 1 only)")
        print("    A : -X velocity (Motor 3 only)")
        print("")
        print("  Diagonal Movement:")
        print("    Q : (-X, +Y) - Motors 3 & 2")
        print("    E : (+X, +Y) - Motors 1 & 2")
        print("    Z : (-X, -Y) - Motors 3 & 4")
        print("    C : (+X, -Y) - Motors 1 & 4")
        print("")
        print("  Velocity Control:")
        print("    + / = : Increase velocity magnitude")
        print("    - / _ : Decrease velocity magnitude")
        print("")
        print("  Commands:")
        print("    R / SPACE : Stop all motors (zero velocity)")
        print("    T         : Emergency stop")
        print("    H         : Show this help")
        print("    ESC / X   : Exit")
        print("="*70)
        print(f"  Current velocity magnitude: {self.velocity_magnitude:.1f}")
        print("="*70 + "\n")
    
    def update_display(self):
        """Update status display"""
        motors_active = []
        if self.current_vel.linear.x > 0:
            motors_active.append("M1(+X)")
        elif self.current_vel.linear.x < 0:
            motors_active.append("M3(-X)")
        
        if self.current_vel.linear.y > 0:
            motors_active.append("M2(+Y)")
        elif self.current_vel.linear.y < 0:
            motors_active.append("M4(-Y)")
        
        motors_str = ", ".join(motors_active) if motors_active else "STOPPED"
        
        sys.stdout.write(
            f"\r  Vel: ({self.current_vel.linear.x:+6.1f}, {self.current_vel.linear.y:+6.1f}) | "
            f"Mag: {self.velocity_magnitude:5.1f} | "
            f"Active: {motors_str:20s} | Press 'h' for help   "
        )
        sys.stdout.flush()
    
    def publish_velocity(self):
        """Publish the current velocity command once"""
        self.cmd_vel_pub.publish(self.current_vel)

    def set_velocity(self, vx, vy):
        """
        Set velocity with antagonistic motor safety check and publish immediately.
        Ensures antagonistic motors are never active simultaneously.
        """
        if vx > 0:
            self.current_vel.linear.x = vx   # Motor 1 active
        elif vx < 0:
            self.current_vel.linear.x = vx   # Motor 3 active
        else:
            self.current_vel.linear.x = 0.0  # Both motors stopped

        if vy > 0:
            self.current_vel.linear.y = vy   # Motor 2 active
        elif vy < 0:
            self.current_vel.linear.y = vy   # Motor 4 active
        else:
            self.current_vel.linear.y = 0.0  # Both motors stopped

        self.publish_velocity()
        self.update_display()
    
    def stop_all(self):
        """Stop all motors and publish immediately"""
        self.current_vel.linear.x = 0.0
        self.current_vel.linear.y = 0.0
        self.publish_velocity()
        self.update_display()
    
    def process_key(self, key):
        """Process keyboard input, update velocity, and publish on keypress"""
        
        # Movement commands - single axis
        if key.lower() == 'w':
            self.set_velocity(0.0, self.velocity_magnitude)
        
        elif key.lower() == 's':
            self.set_velocity(0.0, -self.velocity_magnitude)
        
        elif key.lower() == 'd':
            self.set_velocity(self.velocity_magnitude, 0.0)
        
        elif key.lower() == 'a':
            self.set_velocity(-self.velocity_magnitude, 0.0)
        
        # Diagonal movements
        elif key.lower() == 'q':
            self.set_velocity(-self.velocity_magnitude, self.velocity_magnitude)
        
        elif key.lower() == 'e':
            self.set_velocity(self.velocity_magnitude, self.velocity_magnitude)
        
        elif key.lower() == 'z':
            self.set_velocity(-self.velocity_magnitude, -self.velocity_magnitude)
        
        elif key.lower() == 'c':
            self.set_velocity(self.velocity_magnitude, -self.velocity_magnitude)
        
        # Stop commands
        elif key.lower() == 'r' or key == ' ':
            self.stop_all()
            self.get_logger().info("All motors stopped")
        
        # Emergency stop
        elif key.lower() == 't':
            self.stop_all()
            msg = Bool()
            msg.data = True
            self.emergency_stop_pub.publish(msg)
            self.get_logger().warn("EMERGENCY STOP sent")
        
        # Velocity magnitude adjustment
        elif key in ['+', '=']:
            self.velocity_magnitude = min(MAX_VELOCITY, self.velocity_magnitude + VELOCITY_INCREMENT)
            self.get_logger().info(f"Velocity magnitude: {self.velocity_magnitude:.1f}")
            # Update magnitude if motors already active
            if self.current_vel.linear.x != 0 or self.current_vel.linear.y != 0:
                sign_x = 1 if self.current_vel.linear.x > 0 else (-1 if self.current_vel.linear.x < 0 else 0)
                sign_y = 1 if self.current_vel.linear.y > 0 else (-1 if self.current_vel.linear.y < 0 else 0)
                self.set_velocity(sign_x * self.velocity_magnitude, sign_y * self.velocity_magnitude)
        
        elif key in ['-', '_']:
            self.velocity_magnitude = max(MIN_VELOCITY, self.velocity_magnitude - VELOCITY_INCREMENT)
            self.get_logger().info(f"Velocity magnitude: {self.velocity_magnitude:.1f}")
            if self.current_vel.linear.x != 0 or self.current_vel.linear.y != 0:
                sign_x = 1 if self.current_vel.linear.x > 0 else (-1 if self.current_vel.linear.x < 0 else 0)
                sign_y = 1 if self.current_vel.linear.y > 0 else (-1 if self.current_vel.linear.y < 0 else 0)
                self.set_velocity(sign_x * self.velocity_magnitude, sign_y * self.velocity_magnitude)
        
        # Help
        elif key.lower() == 'h':
            self.print_instructions()
        
        # Exit
        elif key == '\x1b' or key.lower() == 'x':
            self.get_logger().info("Exiting...")
            return False
        
        return True
    
    def run(self):
        """Main control loop"""
        try:
            while rclpy.ok():
                key = self.get_key(timeout=0.1)
                
                if key and not self.process_key(key):
                    break
                
                # Spin once to process any pending ROS callbacks
                rclpy.spin_once(self, timeout_sec=0)
        
        except KeyboardInterrupt:
            self.get_logger().info("Keyboard interrupt received")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup before exit"""
        self.get_logger().info("Shutting down keyboard velocity publisher...")
        self.stop_all()
        
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        except:
            pass
        
        self.get_logger().info("Shutdown complete")


# ==================== MAIN ====================
def main(args=None):
    rclpy.init(args=args)
    
    try:
        publisher = KeyboardPublisher()
        publisher.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()