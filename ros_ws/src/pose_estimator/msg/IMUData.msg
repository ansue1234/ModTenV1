# IMU data from a single bridge
std_msgs/Header header
uint32 timestamp_ms          # Arduino timestamp in milliseconds
uint8 bridge_id              # Bridge ID (0 to NUM_BRIDGES-1)
uint8 bridge_addr            # I2C address of the bridge
geometry_msgs/Vector3 accel  # Acceleration in m/s^2
geometry_msgs/Vector3 gyro   # Angular velocity in deg/s
geometry_msgs/Vector3 mag    # Magnetic field in µT