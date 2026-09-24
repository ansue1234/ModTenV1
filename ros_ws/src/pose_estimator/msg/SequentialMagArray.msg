# Complete sequential magnetometer array (dumped at end of cycle)
std_msgs/Header header
uint32 cycle_number          # Cycle number from Arduino
SequentialMagData[] data     # Array of sequential magnetometer readings