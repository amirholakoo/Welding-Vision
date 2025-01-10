**Autonomous Pipe Welding and Gap Mapping Robot**
=================================================

An advanced robotic system designed for precise **butt welding of pipes** with **gap mapping**, **dynamic filler control**, and **torch positioning** using Raspberry Pi, Arduino Mega, and multiple sensors and motors.

* * * * *

**Table of Contents** 📋
------------------------

1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [Hardware Components](#hardware-components)
4.  [Wiring Diagram](#wiring-diagram)
5.  [System Architecture](#system-architecture)
6.  [Installation](#installation)
7.  [How It Works](#how-it-works)
8.  [Contributing](#contributing)
9.  [License](#license)

* * * * *

**Introduction** 🛠️
--------------------

This project automates the process of **butt welding two pipes** by:

-   Mapping and analyzing gaps in the seam.
-   Dynamically adjusting filler feed and torch positioning based on real-time gap measurements.
-   Coordinating multiple motors and sensors for precise operation.

The system is designed for **30 cm diameter pipes** (~30 kg each) and includes features for **error detection**, **job rejection**, and **process logging**.

* * * * *

**Features** ✨
--------------

-   **360° Pipe Rotation**: Stepper motor with encoder feedback for accurate positioning.
-   **Torch Movement**: Horizontal (X-axis) and vertical (Y-axis) adjustments for precise welding.
-   **Gap Mapping**: Raspberry Pi camera with OpenCV analyzes seam gaps.
-   **Dynamic Filler Control**: Adjusts filler speed based on gap measurements.
-   **Error Handling**: Rejects jobs with gaps >6 mm.
-   **Data Logging**: Stores seam profiles for analysis and traceability.

* * * * *

**Hardware Components** 🧩
--------------------------

### 1\. **Microcontrollers**

-   **Raspberry Pi 5**: Vision processing, decision-making, and system coordination.
-   **Arduino Mega 2560**: Real-time motor and sensor control.

### 2\. **Motors**

-   **Stepper R**: Rotates the pipe at a controlled speed (e.g., 60 RPM).
-   **Stepper X**: Moves the welding torch horizontally.
-   **Stepper Y**: Adjusts torch height dynamically using a distance sensor.
-   **Stepper F**: Feeds filler material based on gap size.

### 3\. **Sensors**

-   **VL53L0X**: Distance sensor for maintaining torch-to-surface gap (~1-2 mm).
-   **E6A2 Encoder**: Tracks pipe rotation angle.

### 4\. **Other Components**

-   **Raspberry Pi Camera V2**: Captures seam images for gap analysis.
-   **Laser Module**: Provides alignment guidance.
-   **Relays**: Controls the welding torch and laser.
-   **Power Supply**: 12V/24V DC for motors and sensors.

* * * * *

**Wiring Diagram** 🔌
---------------------

Below is an overview of the connections:

| Component | Pin Assignments |
| --- | --- |
| **DC Motor (Pipe Rotation)** | Pin 12 (Relay) |
| **Encoder A/B** | Pins 9, 10 |
| **X Stepper** | STEP: Pin 2, DIR: Pin 5 |
| **Y Stepper** | STEP: Pin 3, DIR: Pin 6 |
| **F Stepper** | STEP: Pin 4, DIR: Pin 7 |
| **VL53L0X** | SDA: A4, SCL: A5 |
| **Laser Module** | Pin 11 (Relay) |

* * * * *

**System Architecture** 🧠
--------------------------

### **Subsystems Overview**

1.  **Pipe Rotation Subsystem (Stepper R)**:
    -   Rotates the pipe and tracks position with the encoder.
2.  **Torch Positioning Subsystem (Stepper X/Y)**:
    -   Moves the torch horizontally and vertically.
3.  **Gap Mapping Subsystem**:
    -   Uses Raspberry Pi camera and OpenCV for seam analysis.
4.  **Filler Control Subsystem (Stepper F)**:
    -   Feeds filler material dynamically based on gap measurements.

* * * * *

**Installation** ⚙️
-------------------

### **1\. Hardware Setup**

-   Assemble pipe holders and align motors.
-   Mount Raspberry Pi and camera for a clear view of the seam.
-   Wire all motors, sensors, and relays as per the diagram.

### **2\. Software Installation**

#### On Raspberry Pi:

1.  Install **Raspberry Pi OS**.
2.  Install Python libraries:

    bash

    Copy code

    `sudo apt update
    sudo apt install python3-opencv
    pip3 install vl53l0x smbus`

3.  Clone this repository:

    bash

    Copy code

    `git clone <repository-url>
    cd <repository-folder>`

#### On Arduino:

1.  Install the Arduino IDE.
2.  Add libraries for VL53L0X and stepper motor control.
3.  Upload the provided firmware to the Arduino Mega.

* * * * *

**How It Works** 🛠️
--------------------

### **1\. Mapping Phase**

-   The pipe rotates incrementally.
-   Raspberry Pi captures images and calculates gaps using OpenCV.
-   A gap map is created and stored (angle vs. gap size).

### **2\. Welding Phase**

-   Torch positions itself based on the gap map.
-   Pipe rotates continuously.
-   Filler feed adjusts dynamically.
-   Welding completes after a full 360° rotation.

### **3\. Shutdown Phase**

-   Motors stop.
-   Torch and laser guide return to their home positions.
-   Logs are saved for analysis.

* * * * *

**Contributing** 🤝
-------------------

Feel free to contribute by:

-   Reporting issues.
-   Suggesting new features.
-   Submitting pull requests.

* * * * *

**License** 📜
--------------

This project is licensed under the MIT License. See the `LICENSE` file for details.

* * * * *

**Acknowledgments** 🙏
----------------------

-   Special thanks to **ChatGPT** for detailed planning and implementation support.
-   Kudos to the community for their feedback and testing!