# Raw or processed magnetometer data
std_msgs/Header header
uint32 timestamp_ms          # Arduino timestamp in milliseconds
uint8 active_coil_id         # ID of the coil that was active (-1 for ambient)
uint8 active_coil_addr       # I2C address of active coil (0 for ambient)
uint8 reading_bridge_id      # ID of bridge that provided this reading
uint8 reading_bridge_addr    # I2C address of reading bridge
geometry_msgs/Vector3 mag    # Magnetic field in µT