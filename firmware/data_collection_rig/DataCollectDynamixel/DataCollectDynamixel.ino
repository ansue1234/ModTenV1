/*
 * Arduino IMU Data Collection via ATTiny3224 I2C Bridge

 */

#include <Wire.h>

// ==================== CONFIGURATION ====================
#define ATTINY_I2C_ADDRESS 0x51  // Must match Bridge.ino I2C_ADDRESS for IMU!
#define COIL_ATTINY_ADDRESS 0x50 // ATTiny controlling the coil (different device)
#define SERIAL_BAUD 115200
#define EMERGENCY_LED_PIN 13     // Built-in LED for coil failure warning

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

// Conversion constants (ICM-20948 datasheet values)
#define G0 9.80665f                  // Gravity constant (m/s^2)
#define ACCEL_SENSITIVITY 16384.0f   // LSB/g for ±2g range
#define GYRO_SENSITIVITY 131.0f      // LSB/dps for ±250 dps range  
#define MAG_SENSITIVITY 0.15f        // µT/LSB for AK09916

// ==================== GLOBAL STATE ====================
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
};

// Ambient field bias (computed during BIAS command)
float bias_mx = 0.0;
float bias_my = 0.0;
float bias_mz = 0.0;
bool bias_valid = false;

// Coil failure detection
#define COIL_FIELD_THRESHOLD 5.0  // Minimum expected coil field magnitude (µT)
bool coil_failure_detected = false;

// ==================== I2C COMMUNICATION ====================
void writeRegister(uint8_t reg) {
  Wire.beginTransmission(ATTINY_I2C_ADDRESS);
  Wire.write(reg);
  Wire.endTransmission();
}

uint8_t readByte(uint8_t reg) {
  writeRegister(reg);
  Wire.requestFrom(ATTINY_I2C_ADDRESS, 1);
  if (Wire.available()) {
    return Wire.read();
  }
  return 0;
}

int16_t readInt16(uint8_t reg_h, uint8_t reg_l) {
  uint8_t high = readByte(reg_h);
  uint8_t low = readByte(reg_l);
  return (int16_t)((high << 8) | low);
}

void setCoilState(bool on) {
  // Send coil control command to separate ATTiny controlling the coil
  Wire.beginTransmission(COIL_ATTINY_ADDRESS);
  Wire.write(REG_COIL_CTRL);
  Wire.write(on ? 1 : 0);
  Wire.endTransmission();
}

// ==================== I2C DEVICE SCANNING ====================
bool scanI2CDevice(uint8_t address, const char* deviceName) {
  Wire.beginTransmission(address);
  byte error = Wire.endTransmission();
  
  if (error == 0) {
    Serial.print("  ✓ ");
    Serial.print(deviceName);
    Serial.print(" found at 0x");
    if (address < 16) Serial.print("0");
    Serial.println(address, HEX);
    return true;
  } else {
    Serial.print("  ✗ ");
    Serial.print(deviceName);
    Serial.print(" NOT found at 0x");
    if (address < 16) Serial.print("0");
    Serial.println(address, HEX);
    return false;
  }
}

void scanAllI2CDevices() {
  Serial.println("\n=== I2C Device Scan ===");
  
  bool imu_ok = scanI2CDevice(ATTINY_I2C_ADDRESS, "IMU Bridge ATTiny");
  bool coil_ok = scanI2CDevice(COIL_ATTINY_ADDRESS, "Coil Controller ATTiny");
  
  Serial.println("=======================\n");
  
  if (!imu_ok) {
    Serial.println("WARNING: IMU Bridge not responding!");
    Serial.println("  Check: I2C wiring, ATTiny #1 power, I2C address (should be 0x43)");
  }
  
  if (!coil_ok) {
    Serial.println("WARNING: Coil Controller not responding!");
    Serial.println("  Check: I2C wiring, ATTiny #2 power, I2C address (should be 0x44)");
  }
  
  if (imu_ok && coil_ok) {
    Serial.println("✓ All I2C devices verified and ready!");
  } else {
    Serial.println("⚠ System may not function correctly with missing I2C devices");
  }
  Serial.println();
}

// ==================== IMU DATA READING ====================
bool readIMUData(IMUData &data) {
  // Read all sensor data via I2C burst read
  data.accel_x = readInt16(REG_ACCEL_X_H, REG_ACCEL_X_L);
  data.accel_y = readInt16(REG_ACCEL_Y_H, REG_ACCEL_Y_L);
  data.accel_z = readInt16(REG_ACCEL_Z_H, REG_ACCEL_Z_L);
  
  data.gyro_x = readInt16(REG_GYRO_X_H, REG_GYRO_X_L);
  data.gyro_y = readInt16(REG_GYRO_Y_H, REG_GYRO_Y_L);
  data.gyro_z = readInt16(REG_GYRO_Z_H, REG_GYRO_Z_L);
  
  data.mag_x = readInt16(REG_MAG_X_H, REG_MAG_X_L);
  data.mag_y = readInt16(REG_MAG_Y_H, REG_MAG_Y_L);
  data.mag_z = readInt16(REG_MAG_Z_H, REG_MAG_Z_L);
  
  data.temp = readInt16(REG_TEMP_H, REG_TEMP_L);
  
  // Check if data is valid (simple check for all-zeros or all-FFs)
  if ((data.accel_x == 0 && data.accel_y == 0 && data.accel_z == 0) ||
      (data.accel_x == -1 && data.accel_y == -1 && data.accel_z == -1)) {
    return false;
  }
  
  return true;
}


float accelToMS2(int16_t raw) {
  // Convert raw ADC counts to m/s²
  // Sensitivity: 16384 LSB/g for ±2g range
  // Formula: (raw / 16384) * g₀ where g₀ = 9.80665 m/s²
  // Example: 16384 counts = 1g = 9.81 m/s²
  // Expected at rest: Z ≈ ±16384 counts = ±9.8 m/s², X/Y ≈ 0-800 counts = 0-0.5 m/s²
  return ((float)raw / 16384.0f) * G0;
}

float gyroToDPS(int16_t raw) {
  // Convert raw ADC counts to degrees per second
  // Sensitivity: 131 LSB/dps for ±250 dps range
  // Expected at rest: ≈ 0-500 counts = 0-4 dps
  return (float)raw / 131.0f;
}

float magToUT(int16_t raw) {
  // Convert raw ADC counts to µT
  // Sensitivity: 0.15 µT/LSB for AK09916 magnetometer
  // Expected without coil: ±167-433 counts = ±25-65 µT (Earth's field)
  // Expected with coil: ±67-1333 counts = ±10-200 µT
  return (float)raw * 0.15f;
}

// ==================== COMMAND HANDLERS ====================
void handleBiasCommand(int num_samples, int delay_ms) {
  // Quick I2C health check before sampling
  Wire.beginTransmission(ATTINY_I2C_ADDRESS);
  byte imu_status = Wire.endTransmission();
  
  if (imu_status != 0) {
    Serial.println("ERROR_IMU_BRIDGE_OFFLINE");
    Serial.println("BIAS_DONE,0.000,0.000,0.000");
    return;
  }
  
  // Ensure coil is OFF
  setCoilState(false);
  delay(50);  // Let field settle
  
  // Collect samples and average
  float sum_mx = 0.0;
  float sum_my = 0.0;
  float sum_mz = 0.0;
  int valid_samples = 0;
  
  for (int i = 0; i < num_samples; i++) {
    IMUData data;
    if (readIMUData(data)) {
      sum_mx += magToUT(data.mag_x);
      sum_my += magToUT(data.mag_y);
      sum_mz += magToUT(data.mag_z);
      valid_samples++;
    }
    if (delay_ms > 0) {
      delay(delay_ms);
    }
  }
  
  if (valid_samples > 0) {
    bias_mx = sum_mx / valid_samples;
    bias_my = sum_my / valid_samples;
    bias_mz = sum_mz / valid_samples;
    bias_valid = true;
    
    // Send response: BIAS_DONE,bx,by,bz
    Serial.print("BIAS_DONE,");
    Serial.print(bias_mx, 3);
    Serial.print(",");
    Serial.print(bias_my, 3);
    Serial.print(",");
    Serial.println(bias_mz, 3);
  } else {
    // Failed to get samples
    bias_valid = false;
    Serial.println("BIAS_DONE,0.000,0.000,0.000");
  }
}

void handleSampleCommand(int num_samples, int delay_ms) {
  // Quick I2C health check before sampling
  Wire.beginTransmission(ATTINY_I2C_ADDRESS);
  byte imu_status = Wire.endTransmission();
  
  Wire.beginTransmission(COIL_ATTINY_ADDRESS);
  byte coil_status = Wire.endTransmission();
  
  if (imu_status != 0) {
    Serial.println("ERROR_IMU_BRIDGE_OFFLINE");
    Serial.println("SAMPLE_DONE");
    return;
  }
  
  if (coil_status != 0) {
    Serial.println("ERROR_COIL_CONTROLLER_OFFLINE");
    Serial.println("SAMPLE_DONE");
    return;
  }
  
  // Turn coil ON
  setCoilState(true);
  delay(50);  // Let coil field stabilize
  
  // Collect samples
  for (int i = 0; i < num_samples; i++) {
    IMUData data;
    uint32_t timestamp = millis();
    
    if (readIMUData(data)) {
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
      float mx_debias = mx_raw - (bias_valid ? bias_mx : 0.0);
      float my_debias = my_raw - (bias_valid ? bias_my : 0.0);
      float mz_debias = mz_raw - (bias_valid ? bias_mz : 0.0);
      
      // Check coil field strength (debiased magnitude should be significant)
      float mag_field_magnitude = sqrt(mx_debias*mx_debias + my_debias*my_debias + mz_debias*mz_debias);
      
      if (mag_field_magnitude < COIL_FIELD_THRESHOLD && !coil_failure_detected) {
        // Coil failure detected!
        coil_failure_detected = true;
        digitalWrite(EMERGENCY_LED_PIN, HIGH);
        
        // Send emergency alert
        Serial.print("COIL_FAILURE_DETECTED, ");
        Serial.print("COIL_FIELD_MAG,");
        Serial.println(mag_field_magnitude, 3);
      }
      
      // Send data: DATA,t_ms,ax,ay,az,gx,gy,gz,mx_raw,my_raw,mz_raw,mx_debias,my_debias,mz_debias
      Serial.print("DATA,");
      Serial.print(timestamp);
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
      Serial.println(",nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan");
    }
    
    if (delay_ms > 0 && i < num_samples - 1) {
      delay(delay_ms);
    }
  }
  
  // Turn coil OFF
  setCoilState(false);
  
  // Send completion
  Serial.println("SAMPLE_DONE");
}

// ==================== SERIAL COMMAND PARSER ====================
void processCommand(String cmd) {
  cmd.trim();
  
  if (cmd.startsWith("BIAS,")) {
    // Parse: BIAS,<num_samples>,<delay_ms>
    int idx1 = cmd.indexOf(',');
    int idx2 = cmd.indexOf(',', idx1 + 1);
    
    if (idx1 > 0 && idx2 > 0) {
      int num_samples = cmd.substring(idx1 + 1, idx2).toInt();
      int delay_ms = cmd.substring(idx2 + 1).toInt();
      
      handleBiasCommand(num_samples, delay_ms);
    }
  }
  else if (cmd.startsWith("SAMPLE,")) {
    // Parse: SAMPLE,<num_samples>,<delay_ms>
    int idx1 = cmd.indexOf(',');
    int idx2 = cmd.indexOf(',', idx1 + 1);
    
    if (idx1 > 0 && idx2 > 0) {
      int num_samples = cmd.substring(idx1 + 1, idx2).toInt();
      int delay_ms = cmd.substring(idx2 + 1).toInt();
      
      handleSampleCommand(num_samples, delay_ms);
    }
  }
}

// ==================== SETUP ====================
void setup() {
  Serial.begin(SERIAL_BAUD);
  Wire.begin();
  
  // Setup emergency LED
  pinMode(EMERGENCY_LED_PIN, OUTPUT);
  digitalWrite(EMERGENCY_LED_PIN, LOW);
  
  delay(500);
  
  Serial.println("\n==================================================");
  Serial.println("  Arduino IMU Bridge - Dual ATTiny Setup");
  Serial.println("==================================================");
  
  // Scan for both I2C devices
  scanAllI2CDevices();
  
  // Test connection to IMU bridge and read WHO_AM_I
  Serial.println("Testing IMU Bridge...");
  uint8_t whoami = readByte(REG_WHO_AM_I);
  Serial.print("  ATTiny IMU Bridge (0x");
  Serial.print(ATTINY_I2C_ADDRESS, HEX);
  Serial.print(") WHO_AM_I: 0x");
  Serial.println(whoami, HEX);
  
  if (whoami == 0xEA) {
    Serial.println("  ✓ ICM-20948 detected and communicating correctly");
  } else if (whoami == 0x00 || whoami == 0xFF) {
    Serial.println("  ⚠ WARNING: No valid response from ICM-20948");
    Serial.println("    Check: ICM-20948 power, SPI wiring to ATTiny #1");
  } else {
    Serial.println("  ⚠ WARNING: Unexpected WHO_AM_I value");
    Serial.println("    Expected 0xEA for ICM-20948");
  }
  
  Serial.println();
  
  // Test coil controller by ensuring coil is OFF
  Serial.println("Testing Coil Controller...");
  Serial.print("  Coil Controller ATTiny Address: 0x");
  Serial.println(COIL_ATTINY_ADDRESS, HEX);
  Serial.println("  Sending coil OFF command...");
  setCoilState(false);
  delay(100);
  Serial.println("  ✓ Coil control command sent successfully");
  
  Serial.println();
  Serial.println("==================================================");
  Serial.println("Arduino IMU Bridge Ready");
  Serial.println("Commands: BIAS,<n>,<delay> | SAMPLE,<n>,<delay>");
  Serial.println("==================================================\n");
}

// ==================== LOOP ====================
void loop() {
  // Wait for serial commands from Python
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    processCommand(cmd);
  }
}