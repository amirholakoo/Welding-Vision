#!/usr/bin/env python3
import serial
import json
import time
import cv2
import numpy as np
from datetime import datetime
import os
import subprocess
import matplotlib


class PipeGapMapper:
    def __init__(self, serial_port='/dev/ttyACM0', baud_rate=115200):
        self.serial = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        
        # ROI dimensions in mm (height: 50mm, width: 20mm)
        self.roi_height_mm = 50  # height of ROI in mm (vertical)
        self.roi_width_mm = 20   # width of ROI in mm (horizontal)
        # Assuming a calibration factor (pixels per mm) - this should be calibrated for your setup
        self.pixels_per_mm = 10  # This value needs to be calibrated for your camera setup
        
        # Create output directory
        self.output_dir = f"gap_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create directory to store captured images
        self.image_dir = f"{self.output_dir}/captured_images"
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize results storage
        self.measurements = []

    def send_command(self, command):
        self.serial.write(f"{command}\n".encode())
        response = self.serial.readline().decode().strip()
        return response

    def capture_image(self, angle):
        # Define the filename for each image
        filename = f"{self.image_dir}/image_{angle:.1f}.jpg"
        
        # Command to capture the image using rpicam-still
        command = f"rpicam-still -o {filename}"
        
        # Run the command to capture the image
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Captured {filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error capturing image {filename}: {e}")

    def analyze_image(self, angle):
        # Load captured image
        filename = f"{self.image_dir}/image_{angle:.1f}.jpg"
        frame = cv2.imread(filename)
        
        if frame is None:
            print(f"Failed to load image {filename}")
            return None
        
        # Calculate ROI dimensions in pixels
        roi_width_px = int(self.roi_width_mm * self.pixels_per_mm)
        roi_height_px = int(self.roi_height_mm * self.pixels_per_mm)
        
        # Calculate ROI position (center of the image)
        height, width = frame.shape[:2]
        roi_x = (width - roi_width_px) // 2
        roi_y = (height - roi_height_px) // 2
        
        # Extract ROI
        roi = frame[roi_y:roi_y + roi_height_px, roi_x:roi_x + roi_width_px]
        
        # Convert ROI to grayscale for processing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to highlight gaps
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours in ROI
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw ROI rectangle on original image
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width_px, roi_y + roi_height_px), (255, 0, 0), 2)
        
        # Analyze gap size within ROI
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            gap_area = cv2.contourArea(max_contour)
            
            # Draw contour on ROI and copy back to original image
            cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)
            frame[roi_y:roi_y + roi_height_px, roi_x:roi_x + roi_width_px] = roi
            
            # Save analyzed image
            cv2.imwrite(f"{self.image_dir}/analyzed_{angle:.1f}.jpg", frame)
            
            return gap_area
        return 0

    def map_pipe_gaps(self):
        # Reset position
        self.send_command("RESET")
        print("Starting pipe gap mapping...")
        
        # Open results file
        with open(f"{self.output_dir}/measurements.csv", 'w') as f:
            f.write("Angle,EncoderPosition,GapArea\n")
            
            # Perform full rotation with measurements
            for _ in range(40):  # 40 measurements (9 degrees each)
                # Get current position
                response = self.send_command("POSITION")
                try:
                    position_data = json.loads(response)
                    angle = position_data['degrees']
                    
                    # Capture image
                    self.capture_image(angle)
                    
                    # Analyze image
                    gap_area = self.analyze_image(angle)
                    
                    # Save measurement
                    measurement = {
                        'angle': angle,
                        'encoder_position': position_data['encoder'],
                        'gap_area': gap_area
                    }
                    self.measurements.append(measurement)
                    
                    # Write to CSV
                    f.write(f"{angle},{position_data['encoder']},{gap_area}\n")
                    f.flush()  # Ensure data is written immediately
                    
                    print(f"Measured angle: {angle:.1f}°, Gap area: {gap_area}")
                    
                    # Move to next position
                    self.send_command("MOVE")
                    time.sleep(0.5)  # Wait for movement to complete
                    
                except json.JSONDecodeError:
                    print(f"Error parsing response: {response}")
                except Exception as e:
                    print(f"Error during measurement: {e}")

        print("Mapping complete!")
        self.generate_report()

    def generate_report(self):
        # Create visualization of gap measurements
        angles = [m['angle'] for m in self.measurements]
        gaps = [m['gap_area'] for m in self.measurements]
        
        # Plot using matplotlib
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.polar(np.deg2rad(angles), gaps)
        plt.title('Pipe Gap Map')
        plt.savefig(f"{self.output_dir}/gap_map.png")
        
        # Generate HTML report
        html_content = """
        <html>
        <head>
            <title>Pipe Gap Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .image-container { margin: 20px 0; }
                img { max-width: 800px; }
            </style>
        </head>
        <body>
            <h1>Pipe Gap Analysis Report</h1>
            <h2>Gap Map</h2>
            <img src="gap_map.png">
            <h2>Measurements</h2>
            <table border="1">
                <tr><th>Angle</th><th>Gap Area</th></tr>
        """
        
        for m in self.measurements:
            html_content += f"<tr><td>{m['angle']:.1f}°</td><td>{m['gap_area']}</td></tr>"
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(f"{self.output_dir}/report.html", 'w') as f:
            f.write(html_content)

    def cleanup(self):
        self.serial.close()

if __name__ == "__main__":
    try:
        mapper = PipeGapMapper()
        mapper.map_pipe_gaps()
    except KeyboardInterrupt:
        print("\nMapping interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        mapper.cleanup()
