/************************************************************
 *  Non-Blocking Multi-Stepper and DC Motor Control Example
 *  
 *  Author: ChatGPT (modified based on your requests)
 *  Description:
 *    - Demonstrates how to move X, Y, and F (filler) stepper
 *      motors in a non-blocking manner using micros() timing.
 *    - Controls a DC motor for pipe rotation (on/off).
 *    - Reads a basic encoder for DC motor (polling method, no ISR).
 *    - Provides a simple "back-and-forth" cycle on X axis.
 *
 *  NOTE: You must adjust step intervals, directions, and
 *        mechanical setup to match your actual rig.
 *        Also consider adding limit switch logic and 
 *        advanced safety features (emergency stops, etc.).
 ************************************************************/

/*****************************************************
 * User Configuration Area
 * 
 * --> THESE ARE THINGS YOU NEED TO SET/CONFIRM <--
 * 
 * 1) Pin assignments that match your hardware wiring.
 * 2) Step intervals or microstepping settings.
 * 3) The direction (HIGH/LOW) that moves each motor 
 *    forward/backward in your physical setup.
 * 4) RPM or speed control logic for the DC motor 
 *    if needed (currently just ON/OFF).
 * 5) Possibly incorporate sensor feedback or 
 *    advanced logic as needed.
 ****************************************************/

// --- Pin Assignments ---

// DC motor (pipe rotation) and encoder
int DC_PIN     = 12;   // Pin controlling DC motor relay/driver (LOW = ON, HIGH = OFF)
int ENCODER_A  = 9;    // Encoder A pin
int ENCODER_B  = 10;   // Encoder B pin

// Enable pin for stepper drivers (most CNC shields use one pin to enable/disable all steppers)
int EN_PIN     = 8;    // LOW = enabled, HIGH = disabled

// X axis stepper
int X_STEP     = 2;
int X_DIR      = 5;

// Y axis stepper
int Y_STEP     = 3;
int Y_DIR      = 6;

// F axis stepper (filler) - previously "Z"
int F_STEP     = 4;
int F_DIR      = 7;

/****************************************************
 * Motion Parameters
 *  -> Tweak these values to suit your machine.
 ****************************************************/

// X axis travel
int X_CENTER_STEPS = 1000;  // Steps for X to reach "center" position from home
int X_TRAVEL_STEPS = 400;   // Steps for back-and-forth travel from center
int X_CYCLES       = 26;    // How many back-and-forth cycles to do

// Y axis target position (from home)
int Y_TARGET_STEPS = 500;

// Non-blocking step intervals (in microseconds) 
// -> Lower intervals = faster speed
unsigned long X_STEP_INTERVAL_US = 500;  
unsigned long Y_STEP_INTERVAL_US = 500;  
unsigned long F_STEP_INTERVAL_US = 800;  // For filler feed motor

// DC motor on/off control
// - This code only toggles ON/OFF, no PID speed control here.
// - If you need adjustable RPM, add separate logic for PWM or driver with speed feedback.
bool dcMotorRunning = false;

// If you'd like the filler motor to step in ratio to X moves, set something like:
int F_STEP_RATIO = 10;   // e.g. every 10 X steps we do 1 F step (used in more advanced logic if needed)

// Simple encoder reading (polling) variables
// -> If you need high-speed encoder reading, consider using interrupts or a library
volatile long encoderCount = 0;  
int lastEncAState = 0;   
int lastEncBState = 0;   

/****************************************************
 * Variables for Non-Blocking Motion
 ****************************************************/
// X axis
unsigned long lastXStepTime   = 0; // last time (micros) we stepped X
long currentXPosition         = 0; // our running "position" for X in steps
long targetXPosition          = 0; // where we want X to go (in steps)
bool  movingForward           = true; // direction state for X axis cycles
int   xCycleCount             = 0;   // how many forward-back cycles completed

// Y axis
unsigned long lastYStepTime   = 0;
long currentYPosition         = 0;
long targetYPosition          = 0;

// F axis (filler)
unsigned long lastFStepTime   = 0;
long currentFPosition         = 0;
long targetFPosition          = 0;

// System states
bool systemRunning = false;   // Whether the system is active (X cycles, etc.)

/********************************************************
 *                    setup()
 *  - Initializes all pins
 *  - Enables stepper drivers (EN_PIN)
 *  - Turns on DC motor (if needed)
 *  - Sets initial target for X and Y
 *  - Preps for encoder read
 ********************************************************/
void setup() {
  Serial.begin(115200);

  // --- Pin Modes ---
  pinMode(DC_PIN,     OUTPUT);
  pinMode(ENCODER_A,  INPUT_PULLUP);
  pinMode(ENCODER_B,  INPUT_PULLUP);
  pinMode(EN_PIN,     OUTPUT);
  pinMode(X_STEP,     OUTPUT);
  pinMode(X_DIR,      OUTPUT);
  pinMode(Y_STEP,     OUTPUT);
  pinMode(Y_DIR,      OUTPUT);
  pinMode(F_STEP,     OUTPUT);
  pinMode(F_DIR,      OUTPUT);

  // Enable all steppers (active LOW)
  digitalWrite(EN_PIN, LOW);

  // Turn ON DC motor (for pipe rotation) -> LOW = ON (depends on your relay/driver)
  stopDCMotor(false);
  dcMotorRunning = true;

  // Move X to center
  // -> Set a target for X, we do not block/wait here,
  //    actual motion is handled in loop() by updateX()
  setXTarget(X_CENTER_STEPS, true);

  // Move Y to target
  setYTarget(Y_TARGET_STEPS, true);

  // Mark system as running so we can start doing X cycles once it arrives
  systemRunning = true;

  // Initialize encoder variables
  lastEncAState = digitalRead(ENCODER_A);
  lastEncBState = digitalRead(ENCODER_B);
  encoderCount   = 0;
}

/********************************************************
 *                     loop()
 *  - Continuously updates the motion of X, Y, F 
 *    in a non-blocking manner using micros() timing
 *  - Polls the encoder for DC motor
 *  - Manages the X-axis back-and-forth cycles
 ********************************************************/
void loop() {
  // 1) Simple polling for the DC motor encoder
  updateEncoder();

  // 2) Update each axis in a non-blocking manner
  updateX();
  updateY();
  updateF();

  // 3) Manage the cycle for X (back-and-forth motion)
  if (systemRunning) {
    manageXCycle();
  }

  // Additional logic can be placed here if needed,
  // e.g., adjusting filler feed speed based on gap measurement,
  // checking limit switches, or reading distance sensors, etc.
}

/********************************************************
 *                DC Motor (ON/OFF) Control
 *   - stopDCMotor(true)  => turns DC motor OFF
 *   - stopDCMotor(false) => turns DC motor ON
 *   NOTE: This is a simple relay-based control, no speed control.
 ********************************************************/
void stopDCMotor(bool stop) {
  // If your relay logic is reversed, adjust HIGH/LOW
  digitalWrite(DC_PIN, stop ? HIGH : LOW);
  dcMotorRunning = !stop;
}

/********************************************************
 *                  setXTarget()
 *   - Sets the new target position for the X axis.
 *   - 'steps' is how many steps from the currentXPosition,
 *     not from absolute zero.
 *   - 'forward' sets the DIR pin (HIGH or LOW).
 ********************************************************/
void setXTarget(long steps, bool forward) {
  digitalWrite(X_DIR, forward ? HIGH : LOW);
  // If forward is true, we want currentXPosition + steps
  // If forward is false, we want currentXPosition - steps
  targetXPosition = (forward) 
                      ? (currentXPosition + steps)
                      : (currentXPosition - steps);
}

/********************************************************
 *                  setYTarget()
 *   - Same logic as X but for Y axis
 ********************************************************/
void setYTarget(long steps, bool forward) {
  digitalWrite(Y_DIR, forward ? HIGH : LOW);
  targetYPosition = (forward) 
                      ? (currentYPosition + steps)
                      : (currentYPosition - steps);
}

/********************************************************
 *                  setFTarget()
 *   - Same logic as X but for filler axis (F)
 ********************************************************/
void setFTarget(long steps, bool forward) {
  digitalWrite(F_DIR, forward ? HIGH : LOW);
  targetFPosition = (forward)
                      ? (currentFPosition + steps)
                      : (currentFPosition - steps);
}

/********************************************************
 *                 updateX(), updateY(), updateF()
 *   - Non-blocking step functions. 
 *   - Each checks if enough time (micros) has passed 
 *     since the last step to issue the next one.
 *   - Increments or decrements 'currentPosition'.
 ********************************************************/
void updateX() {
  unsigned long now = micros();
  bool forward = (targetXPosition > currentXPosition);

  // If X still hasn't reached its target
  if (currentXPosition != targetXPosition) {
    if (now - lastXStepTime >= X_STEP_INTERVAL_US) {
      lastXStepTime = now;

      // Step pin HIGH -> delay -> LOW
      digitalWrite(X_STEP, HIGH);
      delayMicroseconds(5); // short pulse
      digitalWrite(X_STEP, LOW);

      // Update currentXPosition
      currentXPosition += (forward ? 1 : -1);
    }
  }
}

void updateY() {
  unsigned long now = micros();
  bool forward = (targetYPosition > currentYPosition);

  if (currentYPosition != targetYPosition) {
    if (now - lastYStepTime >= Y_STEP_INTERVAL_US) {
      lastYStepTime = now;

      digitalWrite(Y_STEP, HIGH);
      delayMicroseconds(5);
      digitalWrite(Y_STEP, LOW);

      currentYPosition += (forward ? 1 : -1);
    }
  }
}

void updateF() {
  unsigned long now = micros();
  bool forward = (targetFPosition > currentFPosition);

  if (currentFPosition != targetFPosition) {
    if (now - lastFStepTime >= F_STEP_INTERVAL_US) {
      lastFStepTime = now;

      digitalWrite(F_STEP, HIGH);
      delayMicroseconds(5);
      digitalWrite(F_STEP, LOW);

      currentFPosition += (forward ? 1 : -1);
    }
  }
}

/********************************************************
 *               manageXCycle()
 *   - Checks if X has reached its target. 
 *   - If so, decides whether to reverse direction or 
 *     if we've finished all cycles.
 *   - When all cycles done, it may stop the DC motor 
 *     and return X/Y to home, etc.
 *   - This logic is an example. Adjust to your needs.
 ********************************************************/
void manageXCycle() {
  // Only proceed if X reached its target
  if (currentXPosition == targetXPosition) {
    if (movingForward) {
      // We just arrived at the forward limit
      setXTarget(X_TRAVEL_STEPS, false); // go back
      movingForward = false;
    } else {
      // We just arrived back at the backward limit
      xCycleCount++;
      if (xCycleCount < X_CYCLES) {
        // Start next forward move
        setXTarget(X_TRAVEL_STEPS, true);
        movingForward = true;
      } else {
        // All cycles completed
        // Example: turn off DC motor, bring X & Y home, etc.
        stopDCMotor(true); // Off
        setYTarget(Y_TARGET_STEPS, false); // Return Y to home
        setXTarget(X_CENTER_STEPS, false); // Return X to home
      }
    }
  }
}

/********************************************************
 *                updateEncoder()
 *   - Simple polling approach to detect encoder changes 
 *     on DC motor. 
 *   - If your motor rotates quickly, consider using 
 *     external interrupt or dedicated library.
 ********************************************************/
void updateEncoder() {
  int encA = digitalRead(ENCODER_A);
  int encB = digitalRead(ENCODER_B);

  // If A changed, that indicates a possible "tick"
  if (encA != lastEncAState) {
    // Determine direction by reading B
    if (encB == encA) {
      encoderCount++;
    } else {
      encoderCount--;
    }
  }

  // Save state
  lastEncAState = encA;
  lastEncBState = encB;
}
