VL53L0X Sensor Integration
--------------------------

**What it does:**\
The VL53L0X is a time-of-flight distance sensor communicating over I2C. It gives distance readings in millimeters. We'll use this reading to control the Y axis so that it maintains a 3 cm (30 mm) distance from the object in front of it.

**I2C Connection Notes (Arduino Uno):**

-   SDA = A4
-   SCL = A5
-   VIN = 5V (or 3.3V depending on your breakout board; check your VL53L0X module's recommended voltage)
-   GND = GND

**On the CNC Shield:**\
The CNC shield typically does not break out SDA and SCL in a convenient location for sensors. You can use jumper wires to connect directly from the Arduino pins (A4 and A5) to the sensor. Just plug them into the Arduino headers that remain exposed when the CNC shield is mounted on top. If your CNC Shield V3 board has the Arduino pins accessible on top, you can reach them from there.

**Wiring the VL53L0X:**

-   VL53L0X VIN → Arduino 5V (Check if your module is 5V tolerant. Many are.)
-   VL53L0X GND → Arduino GND
-   VL53L0X SDA → Arduino SDA (A4)
-   VL53L0X SCL → Arduino SCL (A5)

**Testing the Sensor Alone:** Before integrating, let's just read distance from the VL53L0X:

cpp

Copy code

`#include <Wire.h>
#include "Adafruit_VL53L0X.h"  // Install Adafruit VL53L0X library

Adafruit_VL53L0X lox = Adafruit_VL53L0X();

void setup() {
  Serial.begin(9600);
  Wire.begin();

  if(!lox.begin()) {
    Serial.println("Failed to find VL53L0X sensor!");
    while(1);
  }
}

void loop() {
  VL53L0X_RangingMeasurementData_t measure;
  lox.rangingTest(&measure, false);

  if (measure.RangeStatus != 4) { // If not out of range
    Serial.print("Distance (mm): "); Serial.println(measure.RangeMilliMeter);
  } else {
    Serial.println("Out of range");
  }
  delay(100);
}`

**What to expect:**\
Open Serial Monitor, you should see distance measurements. If it works, we're good to integrate.

* * * * *

Controlling Y Based on Distance
-------------------------------

**Goal:**\
While X and Z are operating, we monitor the sensor's distance continuously. If the distance is not 3 cm (30 mm), we adjust Y accordingly:

-   If distance > 30 mm, move Y forward (towards the object) a few steps until we get closer to 30 mm.
-   If distance < 30 mm, move Y backward (away from the object).

This will require a feedback loop. Y movement must be slow and controlled, possibly with some tolerance (e.g., ±1 mm). You don't want Y jittering back and forth constantly. We'll implement a simple logic with a small deadband around 30 mm, say 29-31 mm no movement, else move Y.

**Important:**

-   The VL53L0X can read distances continuously.
-   We must integrate this into the main loop where X and Z are moving.
-   We can run a non-blocking style loop or periodically read the sensor and adjust Y by a small number of steps at a time.

**Example Approach:**

1.  After starting the DC and laser, and as soon as we start X back-and-forth (and thus start Z), we also start reading the sensor.
2.  On each iteration (or every few iterations) of the loop, read the distance.
3.  Adjust Y by a small number of steps (e.g., 10 steps) if needed.
4.  Keep doing this until X finishes its cycles.
5.  When X is done, stop measuring and move Y back home.

* * * * *

Laser Module Integration
------------------------

**The Laser Module (SYD1230) at 3-5 V:**

-   You likely need a digital pin to turn it ON/OFF.
-   The CNC shield doesn't dedicate a pin specifically for a laser, but it does bring out the Arduino pins.
-   You can use a spare digital pin for the laser. For example, you could use the "Coolant" enable pin on the CNC shield if you're not using it for anything else. On many CNC shields, the coolant enable line corresponds to Arduino pin D13 or A3.
-   Check your CNC shield documentation. Often "Coolant" or "Mist" output is available. If you're not using it, that's a convenient pin to control the laser.

**Wiring Laser:**

-   Laser + (V+) → 5V from CNC shield (check if shield has a 5V output terminal; it usually does)
-   Laser - (GND) → GND
-   Control pin: If the laser module has a built-in driver and you just need to supply power, you can connect its ground side through a transistor or MOSFET controlled by a digital pin. If it's just a simple diode module with no driver, you must ensure it won't draw more current than Arduino pin can supply. A transistor or MOSFET is safer.

**Simple case (assuming small current laser, has built-in driver):**

-   Laser V+ → 5V
-   Laser GND → Arduino GND
-   Arduino pin (e.g., D13) → Laser EN input (if it has one) or control a transistor for on/off.

If no enable pin on the laser:\
Use a simple NPN transistor:

-   Arduino pin D13 → Base of NPN (through a 220-1k resistor)
-   Emitter of NPN → GND
-   Laser GND → Collector of NPN
-   Laser V+ → 5V

This way, setting D13 HIGH turns transistor on, pulling laser GND low and powering it.

**Turn Laser On After DC Starts and Off Before Going Home:** We'll just do:

cpp

Copy code

`// Initialize laser pin as output
int LASER_PIN = 13;

void setup() {
  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW); // Laser off initially
}

// After DC motor on:
digitalWrite(LASER_PIN, HIGH); // Laser ON

// Before going home at the end:
digitalWrite(LASER_PIN, LOW); // Laser OFF`

(If your laser's logic is reversed, just invert HIGH/LOW.)

* * * * *

Integrated Code Sketch (Conceptual)
-----------------------------------

Below is a conceptual snippet integrating the distance control. We assume:

-   You have the same pin assignments as before for X, Y, Z, DC.
-   We add the laser pin (LASER_PIN = 13).
-   We monitor the sensor only while X moves back-and-forth.
-   We adjust Y continuously to maintain ~30 mm distance.
-   This code will be more complex, so test step-by-step.

**Key Changes:**

-   Initialize sensor in `setup()`.
-   In the loop where X and Z move, also read sensor and adjust Y.
-   Add a small function `adjustYForDistance()` that checks the distance and moves Y a few steps if needed.

cpp

Copy code

`#include <Wire.h>
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
}`

**What to Expect:**

-   DC and laser turn on.
-   X moves to center, Y moves to starting position.
-   As X begins back-and-forth and Z runs, the sensor checks distance every X step.
-   If the distance drifts above 31mm, Y moves a little closer. If below 29mm, Y moves away.
-   After all cycles, DC and laser off, Y and X home.

**Notes:**

-   The above code uses a very simplistic approach to distance control. For smoother control, you might:
    -   Reduce the step size (maybe 1-2 steps at a time).
    -   Add some delays before adjusting Y to avoid jitter.
-   Make sure the sensor is aligned and stable. Sudden changes in reading might cause Y to move frequently.
-   Adjust `DISTANCE_TOL_LOW` and `DISTANCE_TOL_HIGH` to get stable behavior.
-   Adjust step sizes and delays as needed.

**Testing Steps:**

1.  Test sensor alone: ensure you get stable readings.
2.  Test laser pin alone: turn it on/off with a simple blink code.
3.  Test integrated Y control at a standstill (not moving X/Z) to see if it tries to maintain 30mm by moving Y when you move the object closer/farther.
4.  Integrate into full sequence and confirm that while X and Z run, Y adjusts distance correctly.