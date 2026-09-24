"""
tendon_data_collection.py -- training-data collection on the two-segment rig.

Drives the tendon-actuated segment to random waypoints and, at each one, asks
the Arduino (firmware/data_collection_rig/DataCollectDynamixel.ino) for an
ambient-field reading with the coil OFF followed by N magnetometer/accelerometer
samples with the coil ON.  Everything is logged to one CSV per session, which
prepare_dataset.py turns into a training set.

Hardware:
- 4x Dynamixel XM430 servos (IDs 10-13) on a U2D2, one tendon each
- Arduino (Uno/Nano/MEGA) talking I2C to two ATTiny3224 segment boards
  (firmware/segment/Bridge.ino): one reads the ICM-20948 of the moving
  segment, the other switches the coil of the fixed segment
- Electromagnetic coil switched through the NMOS on the segment board

Communication:
- U2D2 serial port for Dynamixel control (Protocol 2.0, 57600 baud)
- Arduino serial port (115200 baud), text protocol:
    BIAS,<n>,<delay_ms>    -> BIAS_DONE,bx,by,bz
    SAMPLE,<n>,<delay_ms>  -> N x DATA,... lines, then SAMPLE_DONE

Edit the CONFIGURATION block below for your ports and servo centre positions.
"""

import serial
import time
import csv
import random
from datetime import datetime
from dynamixel_sdk import *
import os

# ==================== EXCEPTIONS ====================
class CoilFailureException(Exception):
    """Raised when coil failure is detected"""
    pass

class I2CDeviceOfflineException(Exception):
    """Raised when an I2C device stops responding"""
    pass

# ==================== CONFIGURATION ====================
# Serial Ports
U2D2_PORT = 'COM4'          # U2D2 for Dynamixel control
ARDUINO_PORT = 'COM5'        # Arduino for IMU + coil
U2D2_BAUD = 57600
ARDUINO_BAUD = 115200        # Match Arduino Serial.begin()

# Dynamixel Settings
PROTOCOL_VERSION = 2.0
DXL_IDS = [10, 11, 12, 13]   # T1, T2, T3, T4

# Control Table Addresses (XM series)
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_MOVING = 122
ADDR_MOVING_STATUS = 123

# Calibration - MEASURE THESE FOR YOUR RIG (servo ticks, 0-4095, with the
# platform level / segment straight).  2048 assumes the servo homing offsets
# were set so that the straight pose reads mid-range.
CENTER_POSITIONS = {
    10: 2048,  # T1 (+X)
    11: 2048,  # T2 (+Y)
    12: 2048,  # T3 (-X)
    13: 2048,  # T4 (-Y)
}

# Movement Parameters
AMPLITUDE_TICKS = 800        # +/- range from center (adjust as needed)
POSITION_TOLERANCE = 20      # ticks - consider "arrived" when within this
MAX_WAIT_TIME = 3.0          # seconds to wait for servos to reach target

# Data Collection Parameters
NUM_BIAS_SAMPLES = 10        # Samples for ambient field measurement
NUM_DATA_SAMPLES = 10        # Samples with coil ON
SAMPLE_DELAY_MS = 10         # Delay between samples on Arduino side

# Coil Failure Detection
COIL_FIELD_THRESHOLD = 5.0   # Minimum expected coil field magnitude (µT)
ENABLE_COIL_CHECK = True     # Enable automatic coil failure detection

# Experiment Settings
NUM_WAYPOINTS = 500          # Total number of random positions to sample
DATA_ROOT = 'data'           # sessions are written to <DATA_ROOT>/<YYYYMMDD>/tendon_data_<YYYYMMDD>_<NNN>.csv

# Sampling Scheme Selection
#   'uniform'          : uniform over the whole workspace [-1, 1]^2
#   'weighted_uniform' : 20 % exactly straight, 50 % small tilts (|x|,|y| <= 0.25), 30 % uniform
#                        (used for the final training sessions in the paper)
#   'gaussian'         : normal around the straight pose, std = GAUSSIAN_STD, clipped to [-1, 1]
SAMPLING_SCHEME = 'weighted_uniform'
GAUSSIAN_STD = 0.3           # Standard deviation for gaussian sampling (0.3 means ~95% within ±0.6)

# Cooldown Settings (to prevent coil overheating)
COOLDOWN_AFTER_WAYPOINTS = 500  # Cooldown after this many waypoints
COOLDOWN_DURATION_SEC = 600     # Cooldown for 10 minutes (600 seconds)
SHOW_IMU_STREAM = True          # Display IMU data in terminal during collection

# ==================== DYNAMIXEL CONTROLLER ====================
class DynamixelController:
    def __init__(self, port, baudrate):
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.connected = False
        
    def connect(self):
        if not self.port_handler.openPort():
            print(f"[Dynamixel] Failed to open port {self.port_handler.port_name}")
            return False
        if not self.port_handler.setBaudRate(U2D2_BAUD):
            print(f"[Dynamixel] Failed to set baudrate {U2D2_BAUD}")
            return False
        print(f"[Dynamixel] Connected: {self.port_handler.port_name} @ {U2D2_BAUD}")
        self.connected = True
        return True
    
    def enable_torque(self, dxl_id, enable=True):
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, 1 if enable else 0)
        if result != COMM_SUCCESS:
            print(f"[Dynamixel] Torque enable failed for ID {dxl_id}")
        return result == COMM_SUCCESS
    
    def set_goal_position(self, dxl_id, position):
        """Set goal position for a single servo"""
        position = max(0, min(4095, int(position)))
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, dxl_id, ADDR_GOAL_POSITION, position)
        return result == COMM_SUCCESS
    
    def get_present_position(self, dxl_id):
        """Read current position"""
        position, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, dxl_id, ADDR_PRESENT_POSITION)
        if result != COMM_SUCCESS:
            return None
        return position
    
    def is_moving(self, dxl_id):
        """Check if servo is still moving"""
        moving, result, error = self.packet_handler.read1ByteTxRx(
            self.port_handler, dxl_id, ADDR_MOVING)
        if result != COMM_SUCCESS:
            return False
        return moving == 1
    
    def move_to_targets(self, target_positions):
        """
        Move all servos to target positions simultaneously
        target_positions: dict {servo_id: position}
        """
        # Send goal positions
        for servo_id, position in target_positions.items():
            self.set_goal_position(servo_id, position)
        
        # Wait for all servos to reach targets
        start_time = time.time()
        while time.time() - start_time < MAX_WAIT_TIME:
            all_arrived = True
            for servo_id, target_pos in target_positions.items():
                current_pos = self.get_present_position(servo_id)
                if current_pos is None:
                    continue
                if abs(current_pos - target_pos) > POSITION_TOLERANCE:
                    all_arrived = False
                    break
            
            if all_arrived:
                return True
            time.sleep(0.01)
        
        print("[Warning] Timeout waiting for servos to reach target")
        return False
    
    def close(self):
        self.port_handler.closePort()


# ==================== ARDUINO COMMUNICATION ====================
class ArduinoIMU:
    def __init__(self, port, baudrate):
        self.serial = None
        self.port = port
        self.baudrate = baudrate
    
    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)  # Wait for Arduino reset
            # Flush any startup messages
            self.serial.reset_input_buffer()
            print(f"[Arduino] Connected: {self.port} @ {self.baudrate}")
            return True
        except serial.SerialException as e:
            print(f"[Arduino] Connection failed: {e}")
            return False
    
    def send_command(self, command):
        """Send command to Arduino"""
        self.serial.write(f"{command}\n".encode())
        self.serial.flush()
    
    def wait_for_response(self, expected_prefix, timeout=5.0):
        """Wait for a response line starting with expected_prefix"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.serial.in_waiting:
                line = self.serial.readline().decode('utf-8').strip()
                if line.startswith(expected_prefix):
                    return line
        return None
    
    def collect_bias_samples(self):
        """Request ambient field bias measurement"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.send_command(f"BIAS,{NUM_BIAS_SAMPLES},{SAMPLE_DELAY_MS}")
                response = self.wait_for_response("BIAS_DONE", timeout=3.0)
                
                # Check for I2C device errors
                if self.serial.in_waiting:
                    # Check for error messages before BIAS_DONE
                    self.serial.reset_input_buffer()
                
                # Read any pending messages
                start_time = time.time()
                while time.time() - start_time < 1.0:
                    if self.serial.in_waiting:
                        line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                        
                        if line.startswith("ERROR_IMU_BRIDGE_OFFLINE"):
                            raise I2CDeviceOfflineException("IMU bridge ATTiny (ATTINY_I2C_ADDRESS in the sketch) not responding on I2C bus")
                        
                        if line.startswith("BIAS_DONE"):
                            response = line
                            break
                
                if response:
                    # Parse: BIAS_DONE,bx,by,bz
                    parts = response.split(',')
                    if len(parts) == 4:
                        try:
                            bias = {
                                'bx': float(parts[1]),
                                'by': float(parts[2]),
                                'bz': float(parts[3])
                            }
                            return bias
                        except ValueError:
                            pass
                print("[Warning] Failed to get bias samples")
                return None
                
            except (serial.SerialException, OSError) as e:
                retry_count += 1
                print(f"\n[Error] Serial communication failed during bias: {e}")
                
                if retry_count < max_retries:
                    print(f"[Recovery] Attempting to reconnect ({retry_count}/{max_retries})...")
                    try:
                        self.serial.close()
                        time.sleep(1)
                        self.serial = serial.Serial(self.port, self.baudrate, timeout=2)
                        time.sleep(2)
                        self.serial.reset_input_buffer()
                        print("[Recovery] Reconnected successfully")
                    except Exception as reconnect_error:
                        print(f"[Recovery] Reconnect failed: {reconnect_error}")
                        if retry_count >= max_retries:
                            return None
                else:
                    print(f"[Error] Max retries ({max_retries}) exceeded")
                    return None
        
        return None
    
    def collect_data_samples(self, num_samples):
        """
        Request data samples with coil ON
        Returns list of sample dictionaries
        Raises CoilFailureException if coil field is too weak
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.send_command(f"SAMPLE,{num_samples},{SAMPLE_DELAY_MS}")
                
                samples = []
                coil_failure_warned = False
                
                for i in range(num_samples):
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Check for I2C device errors
                    if line.startswith("ERROR_IMU_BRIDGE_OFFLINE"):
                        raise I2CDeviceOfflineException("IMU bridge ATTiny (ATTINY_I2C_ADDRESS in the sketch) not responding on I2C bus")
                    
                    if line.startswith("ERROR_COIL_CONTROLLER_OFFLINE"):
                        raise I2CDeviceOfflineException("coil controller ATTiny (COIL_ATTINY_ADDRESS in the sketch) not responding on I2C bus")
                    
                    # Check for coil failure alert from Arduino
                    if line.startswith("COIL_FAILURE_DETECTED"):
                        if ENABLE_COIL_CHECK:
                            raise CoilFailureException("Arduino detected coil failure - field too weak: " + line)
                        else:
                            if not coil_failure_warned:
                                print("\n[WARNING] Coil failure detected but checks disabled")
                                coil_failure_warned = True
                    
                    # Check for field magnitude report
                    if line.startswith("COIL_FIELD_MAG,"):
                        mag = float(line.split(',')[1])
                        if ENABLE_COIL_CHECK and mag < COIL_FIELD_THRESHOLD:
                            raise CoilFailureException(
                                f"Coil field too weak: {mag:.2f} µT < {COIL_FIELD_THRESHOLD} µT threshold")
                    
                    if line.startswith("DATA,"):
                        # Parse: DATA,t_ms,ax,ay,az,gx,gy,gz,mx_raw,my_raw,mz_raw,mx_debias,my_debias,mz_debias
                        parts = line.split(',')
                        if len(parts) == 14:
                            try:
                                sample = {
                                    't_ms': int(parts[1]),
                                    'ax': float(parts[2]),
                                    'ay': float(parts[3]),
                                    'az': float(parts[4]),
                                    'gx': float(parts[5]),
                                    'gy': float(parts[6]),
                                    'gz': float(parts[7]),
                                    'mx_raw': float(parts[8]),
                                    'my_raw': float(parts[9]),
                                    'mz_raw': float(parts[10]),
                                    'mx_debias': float(parts[11]),
                                    'my_debias': float(parts[12]),
                                    'mz_debias': float(parts[13])
                                }
                                samples.append(sample)
                                
                                # Python-side coil field check
                                if ENABLE_COIL_CHECK and i == 0:  # Check first sample
                                    mx = sample['mx_debias']
                                    my = sample['my_debias']
                                    mz = sample['mz_debias']
                                    mag = (mx**2 + my**2 + mz**2)**0.5
                                    
                                    if mag < COIL_FIELD_THRESHOLD:
                                        raise CoilFailureException(
                                            f"Coil field too weak: {mag:.2f} µT < {COIL_FIELD_THRESHOLD} µT threshold")
                                
                                # Display in terminal if enabled
                                display_imu_sample(sample, i)
                                
                            except ValueError as e:
                                print(f"[Warning] Failed to parse sample: {e}")
                
                # Wait for completion confirmation
                response = self.wait_for_response("SAMPLE_DONE", timeout=2.0)
                
                return samples
                
            except CoilFailureException:
                # Don't retry coil failures - re-raise immediately
                raise
                
            except (serial.SerialException, OSError) as e:
                retry_count += 1
                print(f"\n[Error] Serial communication failed: {e}")
                
                if retry_count < max_retries:
                    print(f"[Recovery] Attempting to reconnect ({retry_count}/{max_retries})...")
                    try:
                        # Close and reopen serial connection
                        self.serial.close()
                        time.sleep(1)
                        self.serial = serial.Serial(self.port, self.baudrate, timeout=2)
                        time.sleep(2)
                        self.serial.reset_input_buffer()
                        print("[Recovery] Reconnected successfully")
                    except Exception as reconnect_error:
                        print(f"[Recovery] Reconnect failed: {reconnect_error}")
                        if retry_count >= max_retries:
                            raise
                else:
                    print(f"[Error] Max retries ({max_retries}) exceeded")
                    raise
        
        return []
    
    def close(self):
        if self.serial:
            self.serial.close()


# ==================== COORDINATE MAPPING ====================
def xy_to_servo_positions(x, y):
    """
    Map joystick coordinates (x, y) to 4 servo positions
    x, y: normalized values in range [-1.0, 1.0]
    
    Mapping:
    T1 (ID 10): X-axis positive direction
    T2 (ID 11): Y-axis positive direction  
    T3 (ID 12): X-axis negative direction
    T4 (ID 13): Y-axis negative direction
    """
    x_ticks = x * AMPLITUDE_TICKS
    y_ticks = y * AMPLITUDE_TICKS
    
    positions = {
        10: int(CENTER_POSITIONS[10] + x_ticks),   # T1: +X
        11: int(CENTER_POSITIONS[11] + y_ticks),   # T2: +Y
        12: int(CENTER_POSITIONS[12] - x_ticks),   # T3: -X
        13: int(CENTER_POSITIONS[13] - y_ticks),   # T4: -Y
    }
    
    # Clamp to valid range
    for servo_id in positions:
        positions[servo_id] = max(0, min(4095, positions[servo_id]))
    
    return positions


def generate_random_target():
    """
    Generate random normalized (x, y) coordinates based on SAMPLING_SCHEME
    
    Schemes:
    - 'uniform': Uniform distribution across [-1, 1] x [-1, 1]
    - 'gaussian': Gaussian distribution centered at (0, 0) with std=GAUSSIAN_STD
    """
    if SAMPLING_SCHEME.lower() == 'uniform':
        # Uniform random sampling across the entire workspace
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)

    elif SAMPLING_SCHEME.lower() == 'weighted_uniform':
        # Over-sample the straight pose and small tilts, where azimuth is hardest to learn
        c = random.random()
        if c < 0.2:
            x, y = 0.0, 0.0                              # 20 %: exactly straight
        elif c < 0.7:
            x = random.uniform(-0.25, 0.25)              # 50 %: small tilts
            y = random.uniform(-0.25, 0.25)
        else:
            x = random.uniform(-1.0, 1.0)                # 30 %: whole workspace
            y = random.uniform(-1.0, 1.0)

    elif SAMPLING_SCHEME.lower() == 'gaussian':
        # Gaussian sampling centered at origin
        # Using Box-Muller transform for normal distribution
        x = random.gauss(0.0, GAUSSIAN_STD)
        y = random.gauss(0.0, GAUSSIAN_STD)
        
        # Clamp to [-1, 1] range (tail truncation)
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
    
    else:
        # Default to uniform if unknown scheme
        print(f"[Warning] Unknown sampling scheme '{SAMPLING_SCHEME}', using uniform")
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
    
    return x, y


def display_cooldown(duration_sec, arduino=None):
    """Display cooldown timer with progress bar"""
    print("\n" + "="*70)
    print("  🌡️  COOLDOWN MODE - Coil Thermal Protection")
    print("="*70)
    print(f"  Cooling down for {duration_sec} seconds ({duration_sec/60:.1f} minutes)")
    print("  This prevents coil overheating during extended data collection")
    print("-"*70)
    
    start_time = time.time()
    end_time = start_time + duration_sec
    last_flush = start_time
    
    while time.time() < end_time:
        elapsed = time.time() - start_time
        remaining = duration_sec - elapsed
        progress = elapsed / duration_sec
        
        # Progress bar
        bar_length = 50
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Time display
        mins_remaining = int(remaining // 60)
        secs_remaining = int(remaining % 60)
        
        print(f"\r  [{bar}] {progress*100:.1f}% | "
              f"Time remaining: {mins_remaining:02d}:{secs_remaining:02d}  ", 
              end='', flush=True)
        
        # Flush serial buffer every 10 seconds to prevent buildup
        if arduino and (time.time() - last_flush > 10):
            try:
                if arduino.serial and arduino.serial.is_open:
                    arduino.serial.reset_input_buffer()
                last_flush = time.time()
            except:
                pass
        
        time.sleep(1)
    
    print("\n" + "-"*70)
    print("  ✓ Cooldown complete! Resuming data collection...")
    print("="*70 + "\n")


def display_imu_sample(sample, sample_idx):
    """Display a single IMU sample in terminal"""
    if not SHOW_IMU_STREAM:
        return
    
    # Format: compact single-line display
    ax = sample['ax']
    ay = sample['ay']
    az = sample['az']
    a_mag = (ax**2 + ay**2 + az**2)**0.5
    
    mx = sample['mx_debias']
    my = sample['my_debias']
    mz = sample['mz_debias']
    m_mag = (mx**2 + my**2 + mz**2)**0.5
    
    print(f"    [{sample_idx}] "
          f"a=({ax:+7.3f},{ay:+7.3f},{az:+7.3f})|{a_mag:6.3f} m/s² | "
          f"m=({mx:+6.1f},{my:+6.1f},{mz:+6.1f})|{m_mag:6.1f} µT")


# ==================== MAIN DATA COLLECTION LOOP ====================
def make_output_path(data_root=DATA_ROOT):
    """<data_root>/<YYYYMMDD>/tendon_data_<YYYYMMDD>_<NNN>.csv with NNN = next free index."""
    current_date = datetime.now().strftime("%Y%m%d")
    data_folder = os.path.join(data_root, current_date)
    os.makedirs(data_folder, exist_ok=True)
    existing = [f for f in os.listdir(data_folder)
                if os.path.isfile(os.path.join(data_folder, f))
                and f.startswith('tendon_data') and f.endswith('.csv')]
    return os.path.join(data_folder, f'tendon_data_{current_date}_{len(existing) + 1:03d}.csv')


def main():
    print("=" * 70)
    print("  Tendon-Driven Platform Data Collection System")
    print("  Dynamixel Control (U2D2) + IMU Sampling (Arduino/ATTiny)")
    print("=" * 70)
    OUTPUT_CSV = make_output_path()
    print(f"[Output] {OUTPUT_CSV}")
    
    # Initialize Dynamixel controller
    dxl = DynamixelController(U2D2_PORT, U2D2_BAUD)
    if not dxl.connect():
        print("[Error] Failed to connect to Dynamixel controller")
        return
    
    # Initialize Arduino IMU
    arduino = ArduinoIMU(ARDUINO_PORT, ARDUINO_BAUD)
    if not arduino.connect():
        print("[Error] Failed to connect to Arduino")
        dxl.close()
        return
    
    # Enable torque and move to center position
    print("\n[Init] Enabling torque and moving to center positions...")
    for servo_id in DXL_IDS:
        dxl.enable_torque(servo_id, True)
    
    center_targets = {sid: CENTER_POSITIONS[sid] for sid in DXL_IDS}
    dxl.move_to_targets(center_targets)
    print("[Init] Servos at center position")
    
    time.sleep(1.0)
    
    # Prepare CSV file
    csv_file = open(OUTPUT_CSV, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Write header
    header = [
        'waypoint', 'target_x', 'target_y',
        'pos_t1', 'pos_t2', 'pos_t3', 'pos_t4',
        'bias_mx', 'bias_my', 'bias_mz',
        'sample_idx', 't_ms',
        'ax', 'ay', 'az',
        'gx', 'gy', 'gz',
        'mx_raw', 'my_raw', 'mz_raw',
        'mx_debias', 'my_debias', 'mz_debias'
    ]
    csv_writer.writerow(header)
    
    print(f"\n[Running] Collecting data for {NUM_WAYPOINTS} waypoints...")
    print(f"[Output] {OUTPUT_CSV}")
    print(f"[Sampling] Scheme: {SAMPLING_SCHEME.upper()}" + 
          (f" (σ={GAUSSIAN_STD})" if SAMPLING_SCHEME.lower() == 'gaussian' else ""))
    print(f"[Cooldown] Every {COOLDOWN_AFTER_WAYPOINTS} waypoints, system will cool down for {COOLDOWN_DURATION_SEC/60:.1f} minutes")
    print(f"[Display] IMU stream: {'ENABLED' if SHOW_IMU_STREAM else 'DISABLED'}")
    print(f"[Safety] Auto-save every 10 waypoints, auto-reconnect on serial errors")
    print(f"[Coil Check] {'ENABLED' if ENABLE_COIL_CHECK else 'DISABLED'} (threshold: {COIL_FIELD_THRESHOLD} µT)")
    print("-" * 70)
    
    try:
        for waypoint_idx in range(NUM_WAYPOINTS):
            # Check if cooldown is needed
            if waypoint_idx > 0 and waypoint_idx % COOLDOWN_AFTER_WAYPOINTS == 0:
                # Flush CSV before cooldown
                csv_file.flush()
                print(f"\n[Checkpoint] CSV saved ({waypoint_idx} waypoints)")
                display_cooldown(COOLDOWN_DURATION_SEC, arduino)
            
            # 1. Generate random target
            target_x, target_y = generate_random_target()
            target_positions = xy_to_servo_positions(target_x, target_y)
            
            print(f"\n[Waypoint {waypoint_idx + 1}/{NUM_WAYPOINTS}] "
                  f"Target: ({target_x:+.3f}, {target_y:+.3f})")
            
            # 2. Move servos to target
            print("  Moving servos...", end='', flush=True)
            success = dxl.move_to_targets(target_positions)
            if success:
                print(" ✓ Arrived")
            else:
                print(" ⚠ Timeout (continuing anyway)")
            
            # Read actual positions
            actual_positions = {sid: dxl.get_present_position(sid) for sid in DXL_IDS}
            
            # 3. Collect ambient field bias (coil OFF)
            print("  Collecting bias samples...", end='', flush=True)
            bias = arduino.collect_bias_samples()
            if bias:
                print(f" ✓ Bias: ({bias['bx']:.2f}, {bias['by']:.2f}, {bias['bz']:.2f}) µT")
            else:
                print(" ⚠ Failed")
                bias = {'bx': 0.0, 'by': 0.0, 'bz': 0.0}
            
            # 4. Collect data samples (coil ON) - with coil failure detection
            print(f"  Collecting {NUM_DATA_SAMPLES} samples (coil ON)...")
            if SHOW_IMU_STREAM:
                print("  IMU Data Stream:")
            
            try:
                samples = arduino.collect_data_samples(NUM_DATA_SAMPLES)
                
                # Display collected samples count
                if not SHOW_IMU_STREAM:
                    print(f"  ✓ Got {len(samples)} samples")
                else:
                    print(f"  ✓ Completed {len(samples)} samples")
            
            except CoilFailureException as e:
                print(f"\n{'='*70}")
                print(f"  ⚠️  COIL FAILURE DETECTED!")
                print(f"{'='*70}")
                print(f"  Error: {e}")
                print(f"  Waypoint: {waypoint_idx + 1}/{NUM_WAYPOINTS}")
                print(f"  Emergency LED should be ON (Arduino Pin 13)")
                print(f"  Data collection STOPPED for safety")
                print(f"{'='*70}")
                
                # Save what we have
                csv_file.flush()
                print(f"\n[Emergency Save] Data up to waypoint {waypoint_idx} saved")
                
                # Return servos to center
                print("[Cleanup] Returning servos to center position...")
                dxl.move_to_targets(center_targets)
                time.sleep(0.5)
                
                # Raise exception to exit
                raise
            
            except I2CDeviceOfflineException as e:
                print(f"\n{'='*70}")
                print(f"  ⚠️  I2C DEVICE COMMUNICATION FAILURE!")
                print(f"{'='*70}")
                print(f"  Error: {e}")
                print(f"  Waypoint: {waypoint_idx + 1}/{NUM_WAYPOINTS}")
                print(f"  One or more ATTiny devices stopped responding")
                print(f"  Data collection STOPPED")
                print(f"{'='*70}")
                
                # Save what we have
                csv_file.flush()
                print(f"\n[Emergency Save] Data up to waypoint {waypoint_idx} saved")
                
                # Return servos to center
                print("[Cleanup] Returning servos to center position...")
                dxl.move_to_targets(center_targets)
                time.sleep(0.5)
                
                # Raise exception to exit
                raise
            
            # 5. Write data to CSV
            for sample_idx, sample in enumerate(samples):
                row = [
                    waypoint_idx,
                    f"{target_x:.6f}",
                    f"{target_y:.6f}",
                    *[actual_positions.get(sid, 0) for sid in DXL_IDS],
                    f"{bias['bx']:.3f}",
                    f"{bias['by']:.3f}",
                    f"{bias['bz']:.3f}",
                    sample_idx,
                    sample['t_ms'],
                    f"{sample['ax']:.6f}",
                    f"{sample['ay']:.6f}",
                    f"{sample['az']:.6f}",
                    f"{sample['gx']:.3f}",
                    f"{sample['gy']:.3f}",
                    f"{sample['gz']:.3f}",
                    f"{sample['mx_raw']:.3f}",
                    f"{sample['my_raw']:.3f}",
                    f"{sample['mz_raw']:.3f}",
                    f"{sample['mx_debias']:.3f}",
                    f"{sample['my_debias']:.3f}",
                    f"{sample['mz_debias']:.3f}"
                ]
                csv_writer.writerow(row)
            
            csv_file.flush()
            
            # Periodic checkpoint save every 10 waypoints
            if (waypoint_idx + 1) % 10 == 0:
                print(f"  [Checkpoint] Progress saved ({waypoint_idx + 1}/{NUM_WAYPOINTS})")
            
            # Optional: small delay between waypoints
            time.sleep(0.1)
    
    except I2CDeviceOfflineException as e:
        print("\n\n[EMERGENCY STOP] I2C Device Communication Failure")
        print(f"Reason: {e}")
        print("Check I2C wiring and ATTiny power before resuming")
        print("Possible causes:")
        print("  - Loose I2C wiring (SDA/SCL)")
        print("  - ATTiny power failure")
        print("  - I2C bus collision or noise")
    
    except CoilFailureException as e:
        print("\n\n[EMERGENCY STOP] Coil failure detected")
        print(f"Reason: {e}")
        print("Check Arduino Pin 13 LED - should be ON")
        print("Inspect coil wiring and power supply before resuming")
    
    except KeyboardInterrupt:
        print("\n\n[Interrupted] User stopped collection")
    
    finally:
        # Cleanup
        print("\n[Cleanup] Returning to center and disabling torque...")
        dxl.move_to_targets(center_targets)
        time.sleep(0.5)
        
        for servo_id in DXL_IDS:
            dxl.enable_torque(servo_id, False)
        
        csv_file.close()
        arduino.close()
        dxl.close()
        
        print(f"[Done] Data saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()