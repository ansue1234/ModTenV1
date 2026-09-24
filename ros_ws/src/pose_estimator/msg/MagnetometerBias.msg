# Ambient magnetometer bias for a bridge
std_msgs/Header header
uint32 timestamp_ms          # Arduino timestamp in milliseconds
uint8 bridge_id              # Bridge ID
uint8 bridge_addr            # I2C address of the bridge
geometry_msgs/Vector3 bias   # Magnetic field bias in µT