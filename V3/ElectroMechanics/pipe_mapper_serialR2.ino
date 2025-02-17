#include <Encoder.h>

// Motor pins
#define STEP_PIN 2
#define DIR_PIN 5
#define ENABLE_PIN 38

// Encoder setup
Encoder myEnc(20, 21);
const long COUNTS_PER_REV = 1440;  // Encoder counts per revolution

// Variables
long lastEncoderPosition = 0;
bool isRotating = false;
int currentStep = 0;
const int STEPS_PER_REV = 200;
const int STEPS_PER_MEASUREMENT = 5;  // 5 steps = 9 degrees (40 measurements per revolution)

void setup() {
  Serial.begin(115200);
  
  // Configure pins
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);  // Enable stepper
  digitalWrite(DIR_PIN, HIGH);    // Set initial direction
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    processCommand(command);
  }
}

void processCommand(String command) {
  command.trim();
  
  if (command == "MOVE") {
    // Move one increment and report position
    moveIncrement();
    reportPosition();
  }
  else if (command == "POSITION") {
    // Just report current position
    reportPosition();
  }
  else if (command == "RESET") {
    // Reset encoder position
    myEnc.write(0);
    Serial.println("RESET_OK");
  }
  else if (command == "STOP") {
    isRotating = false;
    Serial.println("STOP_OK");
  }
}

void moveIncrement() {
  // Move stepper motor one increment
  for (int i = 0; i < STEPS_PER_MEASUREMENT; i++) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(700);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(700);
  }
  
  // Small delay for stability
  delay(100);
}

void reportPosition() {
  long encoderPos = myEnc.read();
  float degrees = (float)encoderPos * 360.0 / COUNTS_PER_REV;
  
  // Send position in JSON format
  Serial.print("{\"encoder\":");
  Serial.print(encoderPos);
  Serial.print(",\"degrees\":");
  Serial.print(degrees);
  Serial.println("}");
}
