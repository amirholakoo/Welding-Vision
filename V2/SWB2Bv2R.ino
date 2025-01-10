#include <Wire.h>
#include "Adafruit_VL53L0X.h"

// Pins (same as before)
int DC_PIN   = 12;
int EN_PIN   = 8;
int X_STEP   = 2;
int X_DIR    = 5;
int Y_STEP   = 3;
int Y_DIR    = 6;
int Z_STEP   = 4;
int Z_DIR    = 7;
int LASER_PIN = 13;

// Motion parameters
int X_CENTER_STEPS = 1000; 
int Y_TARGET_STEPS = 500;  
int X_TRAVEL_STEPS = 400; 
int X_CYCLES = 26; 
int STEP_DELAY_US = 500;  
int Z_STEP_DELAY_US = 800; 
int Z_step_interval = 10; 

// Distance setpoint (3cm = 30mm)
int DISTANCE_SETPOINT = 30;
// Tolerance +/- 1mm
int DISTANCE_TOL_LOW = 29;
int DISTANCE_TOL_HIGH = 31;

// VL53L0X object
Adafruit_VL53L0X lox = Adafruit_VL53L0X();

void setup() {
  Serial.begin(9600);
  pinMode(DC_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);
  pinMode(X_STEP, OUTPUT);
  pinMode(X_DIR, OUTPUT);
  pinMode(Y_STEP, OUTPUT);
  pinMode(Y_DIR, OUTPUT);
  pinMode(Z_STEP, OUTPUT);
  pinMode(Z_DIR, OUTPUT);
  pinMode(LASER_PIN, OUTPUT);

  // Start I2C for sensor
  Wire.begin();
  if(!lox.begin()) {
    Serial.println("Failed to find VL53L0X sensor!");
    while(1);
  }

  // Enable steppers
  digitalWrite(EN_PIN, LOW);

  // Turn on DC motor (LOW = ON)
  digitalWrite(DC_PIN, LOW);

  // Turn on laser
  digitalWrite(LASER_PIN, HIGH);

  // Move X to center
  moveStepper(X_DIR, X_STEP, true, X_CENTER_STEPS, STEP_DELAY_US);

  // Move Y to target
  moveStepper(Y_DIR, Y_STEP, true, Y_TARGET_STEPS, STEP_DELAY_US);

  // Set Z direction
  digitalWrite(Z_DIR, HIGH);

  // Perform X back-and-forth with Z spinning and distance control
  for (int cycle = 0; cycle < X_CYCLES; cycle++) {
    moveXwithZandDistance(true, X_TRAVEL_STEPS);
    moveXwithZandDistance(false, X_TRAVEL_STEPS);
  }

  // Turn off DC motor
  digitalWrite(DC_PIN, HIGH);

  // Turn off laser
  digitalWrite(LASER_PIN, LOW);

  // Return Y home
  moveStepper(Y_DIR, Y_STEP, false, Y_TARGET_STEPS, STEP_DELAY_US);

  // Return X home
  moveStepper(X_DIR, X_STEP, false, X_CENTER_STEPS, STEP_DELAY_US);
}

void loop() {
  // nothing here
}

void moveStepper(int dirPin, int stepPin, bool forward, int steps, int stepDelay) {
  digitalWrite(dirPin, forward ? HIGH : LOW);
  for (int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(stepDelay);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(stepDelay);
  }
}

// Similar to previous moveXwithZ, but we now also read distance and adjust Y
void moveXwithZandDistance(bool forward, int steps) {
  digitalWrite(X_DIR, forward ? HIGH : LOW);

  int Z_counter = 0;

  for (int i = 0; i < steps; i++) {
    // Step X
    digitalWrite(X_STEP, HIGH);
    delayMicroseconds(STEP_DELAY_US);
    digitalWrite(X_STEP, LOW);
    delayMicroseconds(STEP_DELAY_US);

    // Step Z intermittently
    Z_counter++;
    if (Z_counter >= Z_step_interval) {
      digitalWrite(Z_STEP, HIGH);
      delayMicroseconds(Z_STEP_DELAY_US);
      digitalWrite(Z_STEP, LOW);
      delayMicroseconds(Z_STEP_DELAY_US);
      Z_counter = 0;
    }

    // Read distance and adjust Y
    adjustYForDistance();
  }
}

void adjustYForDistance() {
  VL53L0X_RangingMeasurementData_t measure;
  lox.rangingTest(&measure, false);

  if (measure.RangeStatus != 4) { 
    int dist = measure.RangeMilliMeter;
    // Check if we are out of desired range
    if (dist > DISTANCE_TOL_HIGH) {
      // too far, move Y forward few steps
      moveStepper(Y_DIR, Y_STEP, true, 10, STEP_DELAY_US); 
    } else if (dist < DISTANCE_TOL_LOW) {
      // too close, move Y backward few steps
      moveStepper(Y_DIR, Y_STEP, false, 10, STEP_DELAY_US);
    }
    // If within 29-31mm, do nothing.
  }
}
