# Single entry in the sequential magnetometer array
uint32 timestamp_ms          # Arduino timestamp in milliseconds
uint8 active_coil_id         # ID of the coil that was active
uint8 source_bridge_id       # ID of bridge that provided this reading
geometry_msgs/Vector3 mag_debias  # Debiased magnetic field in µT