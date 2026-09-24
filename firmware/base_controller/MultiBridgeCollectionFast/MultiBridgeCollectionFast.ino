/*
 * Multi-Bridge IMU Data Collection
 * 
 * Sequentially activates each coil (one at a time) while all bridges read IMU data.
 * For each coil activation:
 * 1. Read ambient bias with all coils OFF
 * 2. Turn ON one specific coil
 * 3. Wait for field to settle
 * 4. Read data from all bridges
 * 5. Turn OFF the coil
 * 6. Move to next coil
 * 
 * Hardware:
 * - Nx ATTiny3224 I2C bridges @ various addresses
 * - Each bridge connected to an ICM-20948 IMU via SPI
 * - Each bridge controls one coil via GPIO
 * - Special: Address 0x01 is a placeholder for direct Arduino pin control
 * 
 * I2C Connection:
 * - SDA: A4 (Arduino Uno/Nano)
 * - SCL: A5 (Arduino Uno/Nano)
 */

#include <Wire.h>

// ==================== CONFIGURATION ====================
#define NUM_BRIDGES 11  // <<< CHANGE THIS to match your number of bridges (2-16 supported)
#define SERIAL_BAUD 115200

// Special placeholder address for direct pin control
#define PLACEHOLDER_ADDRESS 0x01
#define PLACEHOLDER_COIL_PIN 10  // Digital pin to control when placeholder coil is activated

// I2C addresses for the bridges
// IMPORTANT: Update this array to match your NUM_BRIDGES setting
// Special: Use 0x01 as placeholder for direct pin control (not an I2C device)
// Example: {0x01, 0x42, 0x43, 0x44} - first is placeholder, rest are I2C
const uint8_t BRIDGE_ADDRESSES[NUM_BRIDGES] = {
  0x01, 0x42, 0x43, 0x44, 0x46, 0x48, 0x50, 0x51, 0x52, 0x54, 0x53
//  0x01, 0x44, 0x46, 0x48, 0x50, 0x51, 0x52, 0x54, 0x53
};

// Timing parameters
#define COIL_SETTLE_MS 20        // Time to wait after turning coil ON for field to settle
#define AMBIENT_SETTLE_MS 20    // Time to wait after turning all coils OFF for ambient reading
#define INTER_SAMPLE_DELAY_MS 1 // Delay between consecutive bridge readings

// ATTiny Register Map (from Bridge.ino)
#define REG_ACCEL_X_H 0x00
#define REG_ACCEL_X_L 0x01
#define REG_ACCEL_Y_H 0x02
#define REG_ACCEL_Y_L 0x03
#define REG_ACCEL_Z_H 0x04
#define REG_ACCEL_Z_L 0x05
#define REG_GYRO_X_H 0x06
#define REG_GYRO_X_L 0x07
#define REG_GYRO_Y_H 0x08
#define REG_GYRO_Y_L 0x09
#define REG_GYRO_Z_H 0x0A
#define REG_GYRO_Z_L 0x0B
#define REG_MAG_X_H 0x0C
#define REG_MAG_X_L 0x0D
#define REG_MAG_Y_H 0x0E
#define REG_MAG_Y_L 0x0F
#define REG_MAG_Z_H 0x10
#define REG_MAG_Z_L 0x11
#define REG_TEMP_H 0x12
#define REG_TEMP_L 0x13
#define REG_STATUS 0x14
#define REG_WHO_AM_I 0x15
#define REG_COIL_CTRL 0x16

// Conversion constants
#define G0 9.80665f  // Gravity constant (m/s^2)

// ==================== DATA STRUCTURES ====================
struct IMUData {
  int16_t accel_x;
  int16_t accel_y;
  int16_t accel_z;
  int16_t gyro_x;
  int16_t gyro_y;
  int16_t gyro_z;
  int16_t mag_x;
  int16_t mag_y;
  int16_t mag_z;
  int16_t temp;
  bool valid;
};

struct BiasData {
  float mx;
  float my;
  float mz;
};

// Sequential magnetometer data array (N-1 elements where N = NUM_BRIDGES)
// When bridge i's coil is ON, we store bridge i+1's debiased mag data here
// Maximum 15 elements (for 16 bridges max)
#define MAX_SEQUENTIAL_ARRAY 15
#define SEQUENTIAL_ARRAY_SIZE (NUM_BRIDGES - 1)

struct SequentialMagData {
  uint32_t timestamp;
  uint8_t source_bridge_id;     // Bridge that was read (i+1)
  uint8_t active_coil_id;        // Coil that was active (i)
  float mx_debias;
  float my_debias;
  float mz_debias;
  bool valid;
};

SequentialMagData sequentialMagArray[MAX_SEQUENTIAL_ARRAY];  // Maximum size
uint8_t sequentialArrayIndex = 0;          // Current position in the array (0 to SEQUENTIAL_ARRAY_SIZE-1)

// Store ambient bias for all bridges
BiasData ambientBias[NUM_BRIDGES];

// ==================== HELPER FUNCTIONS ====================
// Check if an address is the placeholder
inline bool isPlaceholder(uint8_t addr) {
  return (addr == PLACEHOLDER_ADDRESS);
}

// ==================== I2C COMMUNICATION ====================
void writeRegister(uint8_t i2c_addr, uint8_t reg) {
  // Skip I2C communication for placeholder
  if (isPlaceholder(i2c_addr)) return;
  
  Wire.beginTransmission(i2c_addr);
  Wire.write(reg);
  Wire.endTransmission();
}

uint8_t readByte(uint8_t i2c_addr, uint8_t reg) {
  // Return 0 for placeholder address
  if (isPlaceholder(i2c_addr)) return 0;
  
  writeRegister(i2c_addr, reg);
  Wire.requestFrom(i2c_addr, (uint8_t)1);
  if (Wire.available()) {
    return Wire.read();
  }
  return 0;
}

int16_t readInt16(uint8_t i2c_addr, uint8_t reg_h, uint8_t reg_l) {
  // Return 0 for placeholder address
  if (isPlaceholder(i2c_addr)) return 0;
  
  uint8_t high = readByte(i2c_addr, reg_h);
  uint8_t low = readByte(i2c_addr, reg_l);
  return (int16_t)((high << 8) | low);
}

void setCoilState(uint8_t i2c_addr, bool on) {
  // Handle placeholder address with digital pin control
  if (isPlaceholder(i2c_addr)) {
    digitalWrite(PLACEHOLDER_COIL_PIN, on ? HIGH : LOW);
    return;
  }
  
  // Normal I2C coil control
  Wire.beginTransmission(i2c_addr);
  Wire.write(REG_COIL_CTRL);
  Wire.write(on ? 1 : 0);
  Wire.endTransmission();
}

void turnOffAllCoils() {
  for (int i = 0; i < NUM_BRIDGES; i++) {
    setCoilState(BRIDGE_ADDRESSES[i], false);
  }
}

void turnOnSetCoil(int index) {
  for (int i = 0; i < NUM_BRIDGES; i++) {
    if (index == i) {
      setCoilState(BRIDGE_ADDRESSES[index], true);
    } else {
      setCoilState(BRIDGE_ADDRESSES[i], false);
    }
  }
}
// ==================== IMU DATA READING ====================
bool readIMUData(uint8_t i2c_addr, IMUData &data) {
  // Handle placeholder address - return all zeros
  if (isPlaceholder(i2c_addr)) {
    data.accel_x = 0;
    data.accel_y = 0;
    data.accel_z = 0;
    data.gyro_x = 0;
    data.gyro_y = 0;
    data.gyro_z = 0;
    data.mag_x = 0;
    data.mag_y = 0;
    data.mag_z = 0;
    data.temp = 0;
    data.valid = true;  // Mark as valid so it doesn't trigger error messages
    return true;
  }
  
  // Normal I2C read for real devices
  data.accel_x = readInt16(i2c_addr, REG_ACCEL_X_H, REG_ACCEL_X_L);
  data.accel_y = readInt16(i2c_addr, REG_ACCEL_Y_H, REG_ACCEL_Y_L);
  data.accel_z = readInt16(i2c_addr, REG_ACCEL_Z_H, REG_ACCEL_Z_L);
  
  data.gyro_x = readInt16(i2c_addr, REG_GYRO_X_H, REG_GYRO_X_L);
  data.gyro_y = readInt16(i2c_addr, REG_GYRO_Y_H, REG_GYRO_Y_L);
  data.gyro_z = readInt16(i2c_addr, REG_GYRO_Z_H, REG_GYRO_Z_L);
  
  data.mag_x = readInt16(i2c_addr, REG_MAG_X_H, REG_MAG_X_L);
  data.mag_y = readInt16(i2c_addr, REG_MAG_Y_H, REG_MAG_Y_L);
  data.mag_z = readInt16(i2c_addr, REG_MAG_Z_H, REG_MAG_Z_L);
  
  data.temp = readInt16(i2c_addr, REG_TEMP_H, REG_TEMP_L);
  
  // Check if data is valid (simple check for all-zeros or all-FFs)
  if ((data.accel_x == 0 && data.accel_y == 0 && data.accel_z == 0) ||
      (data.accel_x == -1 && data.accel_y == -1 && data.accel_z == -1)) {
    data.valid = false;
    return false;
  }
  
  data.valid = true;
  return true;
}

// ==================== UNIT CONVERSIONS ====================
float accelToMS2(int16_t raw) {
  // Library returns mg, convert to m/s^2
  float mg = (float)raw;
  return (mg * 1e-3f) * G0;
}

float gyroToDPS(int16_t raw) {
  // Library returns dps directly
  return (float)raw;
}

float magToUT(int16_t raw) {
  // SparkFun ICM-20948 library returns raw values that need scaling
  // Multiply by 0.15 to get µT
  return (float)raw * 0.15f;
}

// ==================== SEQUENTIAL MAG ARRAY DUMP ====================
void dumpSequentialMagArray() {
  Serial.print("# Sequential Magnetometer Array Dump (");
  Serial.print(SEQUENTIAL_ARRAY_SIZE);
  Serial.println(" elements)");
  
  for (int i = 0; i < SEQUENTIAL_ARRAY_SIZE; i++) {
    if (sequentialMagArray[i].valid) {
      // Format: SEQMAG,timestamp,active_coil_id,source_bridge_id,mx_debias,my_debias,mz_debias
      Serial.print("SEQMAG,");
      Serial.print(sequentialMagArray[i].timestamp);
      Serial.print(",");
      Serial.print(sequentialMagArray[i].active_coil_id);
      Serial.print(",");
      Serial.print(sequentialMagArray[i].source_bridge_id);
      Serial.print(",");
      Serial.print(sequentialMagArray[i].mx_debias, 3);
      Serial.print(",");
      Serial.print(sequentialMagArray[i].my_debias, 3);
      Serial.print(",");
      Serial.println(sequentialMagArray[i].mz_debias, 3);
    } else {
      Serial.print("SEQMAG,0,");
      Serial.print(i);
      Serial.print(",");
      Serial.print(i + 1);
      Serial.println(",nan,nan,nan");
    }
  }
  
  Serial.println("# Sequential array dump complete");
  
  // Reset array validity flags for next cycle
  for (int i = 0; i < SEQUENTIAL_ARRAY_SIZE; i++) {
    sequentialMagArray[i].valid = false;
  }
}

// ==================== DATA COLLECTION ====================
void collectAmbientBias(int i) {
  // Ensure all coils are OFF
  delay(AMBIENT_SETTLE_MS);
  
  Serial.println("# Collecting ambient bias (all coils OFF)");
  IMUData data;
  uint32_t timestamp = millis();
  
  if (readIMUData(BRIDGE_ADDRESSES[i], data)) {
    // Store bias values
    ambientBias[i].mx = magToUT(data.mag_x);
    ambientBias[i].my = magToUT(data.mag_y);
    ambientBias[i].mz = magToUT(data.mag_z);
    
    // Print ambient data
    // Format: AMBIENT,timestamp,bridge_id,bridge_addr,ax,ay,az,gx,gy,gz,mx,my,mz
    Serial.print("AMBIENT,");
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(i);
    Serial.print(",0x");
    Serial.print(BRIDGE_ADDRESSES[i], HEX);
    Serial.print(",");
    Serial.print(accelToMS2(data.accel_x), 6);
    Serial.print(",");
    Serial.print(accelToMS2(data.accel_y), 6);
    Serial.print(",");
    Serial.print(accelToMS2(data.accel_z), 6);
    Serial.print(",");
    Serial.print(gyroToDPS(data.gyro_x), 3);
    Serial.print(",");
    Serial.print(gyroToDPS(data.gyro_y), 3);
    Serial.print(",");
    Serial.print(gyroToDPS(data.gyro_z), 3);
    Serial.print(",");
    Serial.print(ambientBias[i].mx, 3);
    Serial.print(",");
    Serial.print(ambientBias[i].my, 3);
    Serial.print(",");
    Serial.println(ambientBias[i].mz, 3);
  } else {
    // Set bias to zero if read failed
    ambientBias[i].mx = 0.0;
    ambientBias[i].my = 0.0;
    ambientBias[i].mz = 0.0;
    
    Serial.print("AMBIENT,");
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(i);
    Serial.print(",0x");
    Serial.print(BRIDGE_ADDRESSES[i], HEX);
    Serial.println(",nan,nan,nan,nan,nan,nan,nan,nan,nan");
  }
  
}

void collectCoilData(int active_coil_idx) {
  // Turn on only the active coil, turn off all others
  turnOnSetCoil(active_coil_idx);
  delay(COIL_SETTLE_MS);
  
  Serial.print("# Coil ");
  Serial.print(active_coil_idx);
  Serial.print(" (0x");
  Serial.print(BRIDGE_ADDRESSES[active_coil_idx], HEX);
  
  // Special note if this is the placeholder
  if (isPlaceholder(BRIDGE_ADDRESSES[active_coil_idx])) {
    Serial.print(" - PLACEHOLDER, Pin ");
    Serial.print(PLACEHOLDER_COIL_PIN);
  }
  
  Serial.println(") ON - collecting data from active_coil_id");
  
  // Read data from one bridges
  IMUData data;
  uint32_t timestamp = millis();
  int i = active_coil_idx + 1;
  if (readIMUData(BRIDGE_ADDRESSES[i], data)) {
    // Convert to engineering units
    float ax = accelToMS2(data.accel_x);
    float ay = accelToMS2(data.accel_y);
    float az = accelToMS2(data.accel_z);
    
    float gx = gyroToDPS(data.gyro_x);
    float gy = gyroToDPS(data.gyro_y);
    float gz = gyroToDPS(data.gyro_z);
    
    float mx_raw = magToUT(data.mag_x);
    float my_raw = magToUT(data.mag_y);
    float mz_raw = magToUT(data.mag_z);
    
    // Apply bias correction
    float mx_debias = mx_raw - ambientBias[i].mx;
    float my_debias = my_raw - ambientBias[i].my;
    float mz_debias = mz_raw - ambientBias[i].mz;
    
  // Check if this is the sequential reading we want to store
  // When coil i is ON, store data from bridge i+1
  // Only store if i < NUM_BRIDGES-1 (so i+1 is valid) and reading from bridge i+1

    sequentialMagArray[active_coil_idx].timestamp = timestamp;
    sequentialMagArray[active_coil_idx].source_bridge_id = i;
    sequentialMagArray[active_coil_idx].active_coil_id = active_coil_idx;
    sequentialMagArray[active_coil_idx].mx_debias = mx_debias;
    sequentialMagArray[active_coil_idx].my_debias = my_debias;
    sequentialMagArray[active_coil_idx].mz_debias = mz_debias;
    sequentialMagArray[active_coil_idx].valid = true;
    
//    sequentialArrayIndex++;
    
    // When array is full (all SEQUENTIAL_ARRAY_SIZE elements filled), dump it
//    if (sequentialArrayIndex >= SEQUENTIAL_ARRAY_SIZE) {
//      dumpSequentialMagArray();
//      sequentialArrayIndex = 0;  // Reset for next cycle
//    }
  
    
    // Format: DATA,timestamp,active_coil_id,active_coil_addr,reading_bridge_id,reading_bridge_addr,
    //         ax,ay,az,gx,gy,gz,mx_raw,my_raw,mz_raw,mx_debias,my_debias,mz_debias
    Serial.print("DATA,");
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(active_coil_idx);
    Serial.print(",0x");
    Serial.print(BRIDGE_ADDRESSES[active_coil_idx], HEX);
    Serial.print(",");
    Serial.print(i);
    Serial.print(",0x");
    Serial.print(BRIDGE_ADDRESSES[i], HEX);
    Serial.print(",");
    Serial.print(ax, 6);
    Serial.print(",");
    Serial.print(ay, 6);
    Serial.print(",");
    Serial.print(az, 6);
    Serial.print(",");
    Serial.print(gx, 3);
    Serial.print(",");
    Serial.print(gy, 3);
    Serial.print(",");
    Serial.print(gz, 3);
    Serial.print(",");
    Serial.print(mx_raw, 3);
    Serial.print(",");
    Serial.print(my_raw, 3);
    Serial.print(",");
    Serial.print(mz_raw, 3);
    Serial.print(",");
    Serial.print(mx_debias, 3);
    Serial.print(",");
    Serial.print(my_debias, 3);
    Serial.print(",");
    Serial.println(mz_debias, 3);
  } else {
    // Send NaN data if read failed
    Serial.print("DATA,");
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(active_coil_idx);
    Serial.print(",0x");
    Serial.print(BRIDGE_ADDRESSES[active_coil_idx], HEX);
    Serial.print(",");
    Serial.print(i);
    Serial.print(",0x");
    Serial.print(BRIDGE_ADDRESSES[i], HEX);
    Serial.println(",nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan");
  }
  
  // Turn OFF the active coil
  setCoilState(BRIDGE_ADDRESSES[active_coil_idx], false);
}

// ==================== SETUP ====================
void setup() {
  Serial.begin(SERIAL_BAUD);
  Wire.begin();
  
  // Setup placeholder coil pin as output
  pinMode(PLACEHOLDER_COIL_PIN, OUTPUT);
  digitalWrite(PLACEHOLDER_COIL_PIN, LOW);
  
  // Validate configuration
  if (NUM_BRIDGES < 2 || NUM_BRIDGES > 16) {
    Serial.println("# ERROR: NUM_BRIDGES must be between 2 and 16!");
    while(1) delay(1000);
  }
  
  if (SEQUENTIAL_ARRAY_SIZE > MAX_SEQUENTIAL_ARRAY) {
    Serial.println("# ERROR: Sequential array overflow! Reduce NUM_BRIDGES or increase MAX_SEQUENTIAL_ARRAY");
    while(1) delay(1000);
  }
  
  // Initialize sequential mag array
  for (int i = 0; i < SEQUENTIAL_ARRAY_SIZE; i++) {
    sequentialMagArray[i].valid = false;
    sequentialMagArray[i].timestamp = 0;
    sequentialMagArray[i].source_bridge_id = i + 1;
    sequentialMagArray[i].active_coil_id = i;
    sequentialMagArray[i].mx_debias = 0.0;
    sequentialMagArray[i].my_debias = 0.0;
    sequentialMagArray[i].mz_debias = 0.0;
  }
  sequentialArrayIndex = 0;
  
  // Ensure all coils are OFF
  turnOffAllCoils();
  
  delay(500);
  
  Serial.println("# ==================== Multi-Bridge Data Collection ====================");
  Serial.println("# Configuration:");
  Serial.print("# Number of bridges: ");
  Serial.println(NUM_BRIDGES);
  Serial.print("# Sequential array size: ");
  Serial.println(SEQUENTIAL_ARRAY_SIZE);
  Serial.print("# Placeholder coil pin: ");
  Serial.println(PLACEHOLDER_COIL_PIN);
  Serial.print("# Bridge addresses: ");
  for (int i = 0; i < NUM_BRIDGES; i++) {
    Serial.print("0x");
    Serial.print(BRIDGE_ADDRESSES[i], HEX);
    if (isPlaceholder(BRIDGE_ADDRESSES[i])) {
      Serial.print("(PLACEHOLDER)");
    }
    if (i < NUM_BRIDGES - 1) Serial.print(", ");
  }
  Serial.println();
  
  // Test I2C connections (skip placeholder)
  Serial.println("# Testing I2C connections...");
  for (int i = 0; i < NUM_BRIDGES; i++) {
    Serial.print("# Bridge ");
    Serial.print(i);
    Serial.print(" (0x");
    Serial.print(BRIDGE_ADDRESSES[i], HEX);
    Serial.print(") ");
    
    if (isPlaceholder(BRIDGE_ADDRESSES[i])) {
      Serial.print("PLACEHOLDER - Pin ");
      Serial.println(PLACEHOLDER_COIL_PIN);
    } else {
      uint8_t whoami = readByte(BRIDGE_ADDRESSES[i], REG_WHO_AM_I);
      Serial.print("WHO_AM_I: 0x");
      Serial.println(whoami, HEX);
    }
  }
  
  Serial.println("# ========================================================================");
  Serial.println("# CSV Headers:");
  Serial.println("# AMBIENT: timestamp,bridge_id,bridge_addr,ax_ms2,ay_ms2,az_ms2,gx_dps,gy_dps,gz_dps,mx_uT,my_uT,mz_uT");
  Serial.println("# DATA: timestamp,active_coil_id,active_coil_addr,reading_bridge_id,reading_bridge_addr,ax_ms2,ay_ms2,az_ms2,gx_dps,gy_dps,gz_dps,mx_raw_uT,my_raw_uT,mz_raw_uT,mx_debias_uT,my_debias_uT,mz_debias_uT");
  Serial.println("# SEQMAG: timestamp,active_coil_id,source_bridge_id,mx_debias_uT,my_debias_uT,mz_debias_uT");
  Serial.println("# ========================================================================");
  
  delay(1000);
}

// ==================== LOOP ====================
void loop() {
  static uint32_t cycle_count = 0;
  
  Serial.println();
  Serial.print("# ==================== Cycle ");
  Serial.print(cycle_count);
  Serial.println(" ====================");
  
  // Step 1: Collect ambient bias with all coils OFF
  
  // Step 2: Sequentially activate each coil and collect data
  for (int coil_idx = 0; coil_idx < SEQUENTIAL_ARRAY_SIZE; coil_idx++) {
//    turnOffAllCoils();
    collectAmbientBias(coil_idx + 1);
    collectCoilData(coil_idx);
  }
  dumpSequentialMagArray();
  
  Serial.print("# Cycle ");
  Serial.print(cycle_count);
  Serial.println(" complete");
  
  cycle_count++;
}
