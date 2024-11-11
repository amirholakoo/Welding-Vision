🤖 Automated Pipe Welding System with Computer Vision
=====================================================

An intelligent welding assistant powered by Raspberry Pi and Computer Vision

🌟 Overview
-----------

This project implements an automated pipe welding system that uses computer vision to detect gaps and control welding parameters in real-time. The system controls torch oscillation and filler wire feed based on detected gap sizes, making welding more precise and consistent.

🛠️ Hardware Requirements
-------------------------

-   Raspberry Pi 4 (2GB+ RAM recommended)

-   Raspberry Pi NoIR Camera V2 (or compatible camera)

-   2x NEMA 17 Stepper Motors

-   One for torch oscillation

-   One for filler wire feed

-   2x A4988 Stepper Motor Drivers

-   Power Supply (12V-24V depending on motors)

-   GPIO Breakout Board (optional but recommended)

-   LED lighting for consistent illumination

-   🔧 Mechanical Components:

-   Linear rails/guides for torch movement

-   Wire feed mechanism

-   Mounting brackets and hardware

📋 Software Dependencies

# System packages

```
sudo apt-get update

sudo apt-get install -y python3-pip python3-opencv libcamera-dev

pip3 install numpy

pip3 install picamera2

pip3 install scipy

pip3 install RPi.GPIO
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔌 Wiring Diagram

Raspberry Pi GPIO    ->    Component

---------------------------------

GPIO20 (Pin 38)     ->    Torch Motor Direction

GPIO21 (Pin 40)     ->    Torch Motor Step

GPIO16 (Pin 36)     ->    Torch Motor Enable

GPIO23 (Pin 16)     ->    Filler Motor Direction

GPIO24 (Pin 18)     ->    Filler Motor Step

GPIO25 (Pin 22)     ->    Filler Motor Enable

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 Installation & Setup
-----------------------

-   Clone the repository:

    git clone https://github.com/yourusername/pipe-welding-vision.git

    cd pipe-welding-vision

-   Enable camera interface:

    sudo raspi-config

    # Navigate to Interface Options -> Camera -> Enable

3\. Set up permissions:
```
sudo usermod -a -G video $USER

sudo usermod -a -G gpio $USER
```
-   Run the test program:
```
    python3 welding_control.py
```

🎯 Features
-----------

-   Real-time gap detection and measurement

-   Adaptive torch oscillation control

-   Automated filler wire feed based on gap size

-   Safety limits and emergency stops

-   Debug visualization and logging

-   Gap mapping and position verification

-   📊 Data logging and analysis capabilities

🔍 Vision System Capabilities
-----------------------------

-   Gap detection range: 0.5mm - 9mm

-   Processing rate: 30 FPS

-   ROI-based processing for improved performance

-   Adaptive thresholding for varying lighting conditions

-   Debug image storage for quality control

⚙️ Configuration
----------------

Key parameters can be adjusted in the code:

OSCILLATION_WIDTH = 8  # mm

STEPS_PER_MM = 25     # Calibrate for your setup

BASE_SPEED_DELAY = 0.001  # Adjust for motor speed

🛡️ Safety Features
-------------------

-   Emergency stop on large gaps (>9mm)

-   Motor disable on program exit

-   Gap verification against initial scan

-   Continuous system state monitoring

-   Error logging and handling

📝 Usage Tips
-------------

-   Ensure consistent lighting for best vision results

-   Calibrate STEPS_PER_MM for your mechanical setup

-   Adjust ROI parameters based on your camera position

-   Monitor debug images for system verification

-   Regular maintenance of mechanical components


🙏 Acknowledgments
------------------

Special thanks to the amazing Claude AI assistant for helping develop this system! Your expertise in computer vision and control systems made this project possible.

📄 License
----------

This project is licensed under the MIT License - see the LICENSE file for details.


🎥 Demo Video
-------------

![DEMO_fast](https://github.com/user-attachments/assets/045cf7c6-b78c-4608-a0a3-584151ad5a27)

-------------


Made with ❤️ by Cursor + amirholakoo

Remember to weld responsibly! 🌟

#ComputerVision #Robotics #Automation #Welding #RaspberryPi #Python
