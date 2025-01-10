import RPi.GPIO as GPIO
from time import sleep, time
from picamera2 import Picamera2
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import os

# GPIO Pin definitions
TORCH_DIR_PIN = 20    # Direction pin for torch oscillation motor
TORCH_STEP_PIN = 21   # Step pin for torch oscillation motor
TORCH_EN_PIN = 16     # Enable pin for torch oscillation motor

FILLER_DIR_PIN = 23   # Direction pin for filler motor
FILLER_STEP_PIN = 24  # Step pin for filler motor
FILLER_EN_PIN = 25    # Enable pin for filler motor

# Motor and motion parameters
STEPS_PER_MM = 25     # Calibrated steps per mm for NEMA 17
OSCILLATION_WIDTH = 8 # mm
BASE_SPEED_DELAY = 0.001  # Base delay between steps (adjust for your setup)

class WeldingSystem:
    def __init__(self, camera=None):
        # Disable GPIO warnings
        GPIO.setwarnings(False)
        
        # GPIO Setup
        GPIO.setmode(GPIO.BCM)
        self._setup_gpio()
        
        # Use existing camera instance or create new one
        self.camera = camera if camera else initialize_camera()
        
        # System state
        self.is_running = True
        self.current_position = 0
        
        # Vision processing parameters
        self.roi_y_offset = 400    # ROI offset from top of image
        self.roi_height = 200      # Height of ROI
        
        # Create debug directory if it doesn't exist
        if not os.path.exists('debug'):
            os.makedirs('debug')
    
    def _setup_gpio(self):
        # Setup torch motor pins
        for pin in [TORCH_DIR_PIN, TORCH_STEP_PIN, TORCH_EN_PIN,
                   FILLER_DIR_PIN, FILLER_STEP_PIN, FILLER_EN_PIN]:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
    
    def move_torch(self, direction, speed_delay):
        GPIO.output(TORCH_DIR_PIN, direction)
        steps = int(OSCILLATION_WIDTH * STEPS_PER_MM)
        
        for _ in range(steps):
            if not self.is_running:
                break
            GPIO.output(TORCH_STEP_PIN, GPIO.HIGH)
            sleep(speed_delay)
            GPIO.output(TORCH_STEP_PIN, GPIO.LOW)
            sleep(speed_delay)
    
    def feed_filler(self, gap_size):
        # Calculate filler feed rate based on gap size
        feed_steps = int(gap_size * STEPS_PER_MM * 1.2)  # 20% extra for good penetration
        
        GPIO.output(FILLER_DIR_PIN, GPIO.HIGH)
        for _ in range(feed_steps):
            GPIO.output(FILLER_STEP_PIN, GPIO.HIGH)
            sleep(BASE_SPEED_DELAY)
            GPIO.output(FILLER_STEP_PIN, GPIO.LOW)
            sleep(BASE_SPEED_DELAY)
    
    def detect_gap(self):
        """Enhanced gap detection with improved image processing and visualization"""
        try:
            # Capture frame and extract ROI
            frame = self.camera.capture_array()
            height, width = frame.shape[:2]
            
            # Ensure ROI parameters are within frame bounds
            self.roi_y_offset = min(self.roi_y_offset, height - self.roi_height)
            roi = frame[self.roi_y_offset:self.roi_y_offset + self.roi_height, :]
            
            # Convert to grayscale and apply preprocessing
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = gaussian_filter(gray, sigma=2)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                21,  # Block size
                5    # C constant
            )
            
            # Morphological operations to clean up the image
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find and filter contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Debug visualization
            debug_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            # Draw ROI on original frame for visualization
            frame_debug = frame.copy()
            cv2.rectangle(frame_debug, 
                         (0, self.roi_y_offset), 
                         (width, self.roi_y_offset + self.roi_height), 
                         (0, 255, 0), 2)
            
            max_gap = 0
            gap_location = None
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Minimum area threshold
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # Calculate gap width (use the smaller dimension)
                    width = min(rect[1])
                    gap_size = width * 0.1  # Convert pixels to mm (adjust calibration)
                    
                    if gap_size > max_gap:
                        max_gap = gap_size
                        gap_location = rect[0]
                    
                    # Draw rotated rectangle for visualization
                    cv2.drawContours(debug_image, [box], 0, (0, 255, 0), 2)
                    
                    # Add gap size text
                    cv2.putText(debug_image, 
                               f'{gap_size:.1f}mm',
                               (int(rect[0][0]), int(rect[0][1])),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.5,
                               (0, 0, 255),
                               2)
            
            # Display both original frame with ROI and processed image
            cv2.imshow('Original with ROI', frame_debug)
            cv2.imshow('Gap Detection', debug_image)
            cv2.waitKey(1)
            
            # Save debug images periodically
            timestamp = int(time())
            if timestamp % 5 == 0:  # Save every 5 seconds
                cv2.imwrite(f'debug/original_{timestamp}.jpg', frame_debug)
                cv2.imwrite(f'debug/processed_{timestamp}.jpg', debug_image)
            
            return max_gap, gap_location
            
        except Exception as e:
            print(f"Error in gap detection: {str(e)}")
            return 0, None
    
    def run_welding_cycle(self):
        try:
            while self.is_running:
                # Detect gap with enhanced visualization
                gap_size, gap_location = self.detect_gap()
                
                # Log gap measurements
                print(f"Detected gap size: {gap_size:.1f}mm")
                if gap_location:
                    print(f"Gap location (x,y): ({gap_location[0]:.1f}, {gap_location[1]:.1f})")
                
                # Stop if gap is too large
                if gap_size > 9:
                    print("Gap too large (>9mm). Stopping system.")
                    self.is_running = False
                    break
                
                # Adjust speed based on gap size
                if gap_size > 5:
                    # Feed filler wire
                    self.feed_filler(gap_size)
                    speed_delay = BASE_SPEED_DELAY * 1.5  # Slower for larger gaps
                else:
                    speed_delay = BASE_SPEED_DELAY
                
                # Oscillate torch
                self.move_torch(GPIO.HIGH, speed_delay)  # Move right
                sleep(0.1)  # Dwell time at extremes
                self.move_torch(GPIO.LOW, speed_delay)   # Move left
                sleep(0.1)  # Dwell time at extremes
                
        except KeyboardInterrupt:
            print("Program stopped by user")
        finally:
            cv2.destroyAllWindows()
            self.cleanup()
    
    def cleanup(self):
        GPIO.output(TORCH_EN_PIN, GPIO.HIGH)  # Disable motors
        GPIO.output(FILLER_EN_PIN, GPIO.HIGH)
        GPIO.cleanup()
        if self.camera:
            self.camera.stop()
    
    def calibrate_camera(self):
        """Calibrate pixel to mm conversion using a reference object"""
        frame = self.camera.capture_array()
        # Display calibration grid
        cv2.imshow('Calibration', frame)
        cv2.waitKey(0)
        # Let user input known distance
        known_distance_mm = float(input("Enter known distance in mm: "))
        # Calculate conversion factor
        self.px_to_mm = known_distance_mm / measured_pixels

def initialize_camera():
    try:
        # First, check if any existing camera instances are running
        os.system('sudo pkill -f "libcamera"')
        sleep(2)  # Wait for camera to be released
        
        # Initialize camera
        camera = Picamera2()
        
        # Configure camera with explicit settings
        preview_config = camera.create_preview_configuration(
            main={"format": 'RGB888', "size": (1640, 1232)},
            controls={
                "FrameDurationLimits": (33333, 33333),  # 30fps
                "ExposureTime": 20000,  # 20ms exposure
                "AnalogueGain": 2.0,
                "Brightness": 0.5,
                "Contrast": 1.2
            }
        )
        
        # Apply configuration
        camera.configure(preview_config)
        
        # Start camera
        camera.start()
        print("Camera started successfully")
        
        # Wait for camera to warm up
        sleep(2)
        
        return camera
        
    except Exception as e:
        print(f"Camera initialization failed: {str(e)}")
        raise

# Test the camera
if __name__ == "__main__":
    camera = None
    try:
        # Test camera first
        camera = initialize_camera()
        
        # Test capture and display
        frame = camera.capture_array()
        if frame is None:
            raise Exception("Failed to capture frame")
            
        # Display test image
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(1000)
        
        # Continuous preview test
        for _ in range(10):
            frame = camera.capture_array()
            cv2.imshow("Live Preview", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        print("Camera test successful, initializing welding system...")
        # Pass the existing camera instance to WeldingSystem
        welder = WeldingSystem(camera=camera)
        welder.run_welding_cycle()
        
    except KeyboardInterrupt:
        print("Test stopped by user")
    except Exception as e:
        print(f"Error during test: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        if camera:
            camera.stop()
        GPIO.cleanup()
