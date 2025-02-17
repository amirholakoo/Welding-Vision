### **1\. High-Level Control Commands (Raspberry Pi → Arduino Mega)**

These commands will manage motor control, sensor monitoring, safety checks, and system flow.

| Command | Description |
| ----- | ----- |
| **START** | Starts the welding process.|
| **STOP** | Stops all motors and halts the process.|
| **RESET** | Resets all motors and encoders to home positions.|
| **POSITION** | Reports the current angle and encoder positions for all axes (R, X, Y, F).|
| **MOVE\_X:x** | Moves the X-axis to position `x` (steps). |
| **MOVE\_Y:y** | Moves the Y-axis to position `y` (steps). |
| **ROTATE\_R:a** | Rotates the pipe to angle `a` degrees. |
| **SET\_SPEED\_X:speed** | Sets the speed for the X-axis stepper motor. |
| **SET\_SPEED\_Y:speed** | Sets the speed for the Y-axis stepper motor. |
| **SET\_SPEED\_F:speed** | Sets the speed for the filler wire feeder motor. |
| **FILLER\_ON** | Starts the filler wire feeder motor. |
| **FILLER\_OFF** | Stops the filler wire feeder motor. |
| **ADJUST\_Y:distance** | Adjusts the Y-axis (height) to maintain a specific distance using the VL53L0X sensor. |
| **CALIBRATE\_X** | Calibrates and sets the home position for the X-axis. |
| **CALIBRATE\_Y** | Calibrates and sets the home position for the Y-axis. |
| **CALIBRATE\_R** | Calibrates and sets the zero position for pipe rotation (R-axis). |
| **LIMIT\_STATUS** | Returns the status of all limit switches (0 for open, 1 for triggered). |
| **ERROR\_STATUS** | Returns the status of the system errors (e.g., overcurrent, motor stall). |

---

### **2\. Sensor and Feedback Commands (Arduino Mega → Raspberry Pi)**

These commands send feedback to the Raspberry Pi about the status of the system.

| Command | Description |
| ----- | ----- |
| **ENCODER\_R** | Reports the current angle of the pipe rotation. |
| **ENCODER\_X** | Reports the current position of the X-axis. |
| **ENCODER\_Y** | Reports the current position of the Y-axis. |
| **ENCODER\_F** | Reports the current speed/position of the filler wire feeder. |
| **VL53L0X\_DIST** | Reports the current distance reading from the VL53L0X sensor. |
| **TEMP\_STATUS** | Reports the current system temperature (if applicable). |
| **SYSTEM\_READY** | Confirms the system is ready to start. |
| **ERROR:e** | Reports an error code (e.g., `ERROR:1` for overcurrent, `ERROR:2` for motor stall). |

---

### **3\. User Interface Commands (Arduino Uno with TFT LCD)**

These commands adjust configurations, start/stop operations, and set calibration values via the user interface.

| Command | Description |
| ----- | ----- |
| **SET\_X\_TRAVEL:distance** | Sets the travel distance for the X-axis (in mm or steps). |
| **SET\_Y\_TRAVEL:distance** | Sets the travel distance for the Y-axis (in mm or steps). |
| **SET\_X\_SPEED:speed** | Sets the speed of the X-axis motor. |
| **SET\_Y\_SPEED:speed** | Sets the speed of the Y-axis motor. |
| **SET\_R\_SPEED:speed** | Sets the rotation speed of the pipe rotator (R-axis). |
| **SET\_F\_SPEED:speed** | Sets the speed of the filler wire feeder motor. |
| **START\_CALIBRATION** | Starts the calibration process for all axes. |
| **SAVE\_CONFIG** | Saves the current configuration to EEPROM. |
| **LOAD\_CONFIG** | Loads the saved configuration from EEPROM. |
| **SHOW\_STATUS** | Displays the current system status (position, speed, errors). |
| **RESET\_UI** | Resets the UI to the main menu screen. |

---

### **4\. Safety and Emergency Commands**

These commands ensure safety and provide immediate control over the system.

| Command | Description |
| ----- | ----- |
| **EMERGENCY\_STOP** | Immediately stops all motors and activates the safety relay. |
| **CHECK\_LIMITS** | Checks the status of all limit switches and stops if any are triggered. |
| **OVERLOAD\_CHECK** | Checks for motor overload and stops the process if detected. |
| **STOP\_IF\_TEMP\_HIGH** | Stops the process if the system temperature exceeds the safety threshold. |

---

### **5\. Calibration Commands**

These commands help with calibration and fine-tuning of the system for accurate operation.

| Command | Description |
| ----- | ----- |
| **CALIBRATE\_CAMERA** | Calibrates the camera to calculate pixels per mm. |
| **CALIBRATE\_ENCODER** | Calibrates the encoders for all axes to set zero positions. |
| **CALIBRATE\_VL53L0X** | Calibrates the VL53L0X distance sensor for accurate readings. |
| **ADJUST\_MM\_TO\_PIXELS** | Adjusts the mm-to-pixel ratio for image processing. |

---

### **6\. Maintenance and Diagnostics Commands**

For diagnostics, maintenance, and monitoring.

| Command | Description |
| ----- | ----- |
| **DIAG\_MOTORS** | Runs a diagnostic test for all motors. |
| **DIAG\_ENCODERS** | Runs a diagnostic test for all encoders. |
| **DIAG\_SENSORS** | Runs a diagnostic test for all sensors (distance, temperature, etc.). |
| **FIRMWARE\_VERSION** | Returns the firmware version running on the Arduino. |
| **SYSTEM\_CHECK** | Performs a full system check and reports status. |

---

### **7\. Reporting and Data Management**

For managing data logging and generating reports.

| Command | Description |
| ----- | ----- |
| **LOG\_START** | Starts logging data to a file. |
| **LOG\_STOP** | Stops logging data. |
| **GENERATE\_REPORT** | Generates a summary report of the gap mapping or welding process. |
| **SEND\_LOGS** | Sends logs and reports to the Raspberry Pi for storage. |

---

### **8\. Communication and Network Commands (if applicable)**

If your system involves network communication, these commands can be useful.

| Command | Description |
| ----- | ----- |
| **CHECK\_NETWORK** | Verifies if the network is available (for Raspberry Pi). |
| **SEND\_DATA** | Sends data to a remote server or database. |
| **RECEIVE\_UPDATE** | Receives firmware or configuration updates from a remote server. |

