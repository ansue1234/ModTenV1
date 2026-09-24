/*
 * ATTiny3224 - Intelligent I2C Bridge for ICM-20948
 * Uses SparkFun ICM-20948 library
 * Upload via UPDI
 * 
 * ============ CONFIGURATION ============
 * CHANGE THIS ADDRESS FOR EACH ATTINY:
 * Slave 1: 0x42
 * Slave 2: 0x43
 * Slave 3: 0x44, etc.
 * =======================================
 */

#define I2C_ADDRESS 0x45  // <<< CHANGE THIS FOR EACH SLAVE!

/*
 * Required Library: SparkFun ICM-20948 Arduino Library
 * Install via Arduino Library Manager
 * 
 * Hardware connections:
 * SPI (to ICM-20948):
 *   MOSI: PA1 (SPI0 MOSI)
 *   MISO: PA2 (SPI0 MISO)
 *   SCK:  PA3 (SPI0 SCK)
 *   CS:   PA4 (Pin 0 in Arduino)
 * 
 * I2C (to Arduino UNO):
 *   SDA: PB1 (TWI0 SDA)
 *   SCL: PB0 (TWI0 SCL)
 */

#include <Wire.h>
#include <SPI.h>
#include "ICM_20948.h"

#define CS_PIN 0  // PA4 = Arduino pin 0
#define COIL_PIN 1  // PA5 = Arduino pin 1 (controls NMOS)

// IMU object
ICM_20948_SPI myICM;

// Data structure to hold sensor readings
struct SensorData {
  int16_t accelX;
  int16_t accelY;
  int16_t accelZ;
  int16_t gyroX;
  int16_t gyroY;
  int16_t gyroZ;
  int16_t magX;
  int16_t magY;
  int16_t magZ;
  int16_t temp;
} sensorData;

// I2C register map
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
#define REG_WHO_AM_I 0x15  // For debugging
#define REG_COIL_CTRL 0x16 // Coil control register (0=OFF, 1=ON)

volatile uint8_t registerAddress = 0;
uint8_t whoAmI = 0;
volatile uint8_t coilState = 0;  // Coil state (0=OFF, 1=ON)

void setup() {
  // Initialize coil control pin
  pinMode(COIL_PIN, OUTPUT);
  digitalWrite(COIL_PIN, LOW);  // Start with coil OFF
  
  // Initialize SPI
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);
  SPI.begin();
  
  delay(250);  // Longer delay for ICM-20948 power-up
  
  // Initialize ICM-20948 with lower SPI speed first
  myICM.begin(CS_PIN, SPI, 1000000);  // Start at 1 MHz for reliability
  
  if (myICM.status != ICM_20948_Stat_Ok) {
    // IMU initialization failed - set all data to error values
    memset(&sensorData, 0xFF, sizeof(sensorData));
  } else {
    // Try to wake up the magnetometer
    delay(100);
    myICM.startupMagnetometer();
    delay(100);
    
    // Read WHO_AM_I for debugging
    whoAmI = myICM.getWhoAmI();
  }
  
  // Initialize I2C as slave
  Wire.begin(I2C_ADDRESS);
  Wire.onReceive(receiveEvent);
  Wire.onRequest(requestEvent);
}

void loop() {
  // Read sensor data if IMU is working
  if (myICM.status == ICM_20948_Stat_Ok) {
    if (myICM.dataReady()) {
      myICM.getAGMT();
      
      // Store accel data (library returns in mg, stored as raw int16)
      // The library's .acc.axes values are already int16_t
      sensorData.accelX = myICM.agmt.acc.axes.x;
      sensorData.accelY = myICM.agmt.acc.axes.y;
      sensorData.accelZ = myICM.agmt.acc.axes.z;
      
      // Store gyro data (library returns in dps, stored as raw int16)
      sensorData.gyroX = myICM.agmt.gyr.axes.x;
      sensorData.gyroY = myICM.agmt.gyr.axes.y;
      sensorData.gyroZ = myICM.agmt.gyr.axes.z;
      
      // Store mag data (library returns in uT, stored as raw int16)
      sensorData.magX = myICM.agmt.mag.axes.x;
      sensorData.magY = myICM.agmt.mag.axes.y;
      sensorData.magZ = myICM.agmt.mag.axes.z;
      
      // Store temperature
      sensorData.temp = myICM.agmt.tmp.val;
    }
  }
  
  // No delay - let the loop run as fast as possible to keep data fresh
}

// Called when I2C master sends data (sets register address)
void receiveEvent(int numBytes) {
  if (numBytes > 0) {
    registerAddress = Wire.read();
    
    // If there's a second byte, it's a write operation
    if (numBytes > 1) {
      uint8_t value = Wire.read();
      
      // Handle write to coil control register
      if (registerAddress == REG_COIL_CTRL) {
        coilState = (value != 0) ? 1 : 0;
        digitalWrite(COIL_PIN, coilState);
      }
    }
  }
}

// Called when I2C master requests data
void requestEvent() {
  uint8_t value = 0;
  
  switch (registerAddress) {
    case REG_ACCEL_X_H: value = (sensorData.accelX >> 8) & 0xFF; break;
    case REG_ACCEL_X_L: value = sensorData.accelX & 0xFF; break;
    case REG_ACCEL_Y_H: value = (sensorData.accelY >> 8) & 0xFF; break;
    case REG_ACCEL_Y_L: value = sensorData.accelY & 0xFF; break;
    case REG_ACCEL_Z_H: value = (sensorData.accelZ >> 8) & 0xFF; break;
    case REG_ACCEL_Z_L: value = sensorData.accelZ & 0xFF; break;
    
    case REG_GYRO_X_H: value = (sensorData.gyroX >> 8) & 0xFF; break;
    case REG_GYRO_X_L: value = sensorData.gyroX & 0xFF; break;
    case REG_GYRO_Y_H: value = (sensorData.gyroY >> 8) & 0xFF; break;
    case REG_GYRO_Y_L: value = sensorData.gyroY & 0xFF; break;
    case REG_GYRO_Z_H: value = (sensorData.gyroZ >> 8) & 0xFF; break;
    case REG_GYRO_Z_L: value = sensorData.gyroZ & 0xFF; break;
    
    case REG_MAG_X_H: value = (sensorData.magX >> 8) & 0xFF; break;
    case REG_MAG_X_L: value = sensorData.magX & 0xFF; break;
    case REG_MAG_Y_H: value = (sensorData.magY >> 8) & 0xFF; break;
    case REG_MAG_Y_L: value = sensorData.magY & 0xFF; break;
    case REG_MAG_Z_H: value = (sensorData.magZ >> 8) & 0xFF; break;
    case REG_MAG_Z_L: value = sensorData.magZ & 0xFF; break;
    
    case REG_TEMP_H: value = (sensorData.temp >> 8) & 0xFF; break;
    case REG_TEMP_L: value = sensorData.temp & 0xFF; break;
    
    case REG_STATUS: 
      value = (myICM.status == ICM_20948_Stat_Ok) ? 0x01 : 0x00;
      break;
    
    case REG_WHO_AM_I:
      value = whoAmI;
      break;
    
    case REG_COIL_CTRL:
      value = coilState;
      break;
      
    default: value = 0x00; break;
  }
  
  Wire.write(value);
  
  // Auto-increment register address for burst reads
  registerAddress++;
}