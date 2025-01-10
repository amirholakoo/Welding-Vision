import RPi.GPIO as GPIO
from time import sleep, time
from picamera2 import Picamera2
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import os
from collections import defaultdict
import json

# GPIO Pin definitions
TORCH_DIR_PIN = 20    # Direction pin for torch oscillation motor
TORCH_STEP_PIN = 21   # Step pin for torch oscillation motor
TORCH_EN_PIN = 16     # Enable pin for torch oscillation motor

FILLER_DIR_PIN = 23   # Direction pin for filler motor
FILLER_STEP_PIN = 24  # Step pin for filler motor
FILLER_EN_PIN = 25    # Enable pin for filler motor

# Additional GPIO Pin definitions
PIPE_DIR_PIN = 17    # Direction pin for pipe rotation motor
PIPE_STEP_PIN = 27   # Step pin for pipe rotation motor
PIPE_EN_PIN = 22     # Enable pin for pipe rotation motor
ENCODER_A_PIN = 5    # Encoder channel A
ENCODER_B_PIN = 6    # Encoder channel B

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
        
        # Pipe rotation parameters
        self.steps_per_revolution = 200 * 16  # 200 steps * 16 microsteps
        self.encoder_count = 0
        self.encoder_resolution = 1024  # Pulses per revolution
        self.gap_map = defaultdict(list)  # Store gaps by angle
        self.current_angle = 0
        
        # Setup additional GPIO pins
        self._setup_pipe_control()
    
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
    
    def _setup_pipe_control(self):
        """Setup GPIO pins for pipe rotation and encoder"""
        # Setup pipe motor pins
        for pin in [PIPE_DIR_PIN, PIPE_STEP_PIN, PIPE_EN_PIN]:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        
        # Setup encoder pins
        GPIO.setup(ENCODER_A_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(ENCODER_B_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Add encoder interrupt handlers
        GPIO.add_event_detect(ENCODER_A_PIN, GPIO.BOTH, callback=self._encoder_callback)
    
    def _encoder_callback(self, channel):
        """Handle encoder pulses"""
        a_state = GPIO.input(ENCODER_A_PIN)
        b_state = GPIO.input(ENCODER_B_PIN)
        
        if a_state == b_state:
            self.encoder_count += 1
        else:
            self.encoder_count -= 1
        
        # Calculate current angle
        self.current_angle = (self.encoder_count % self.encoder_resolution) * 360 / self.encoder_resolution
    
    def rotate_pipe(self, angle):
        """Rotate pipe by specified angle"""
        steps = int((angle / 360) * self.steps_per_revolution)
        direction = GPIO.HIGH if steps > 0 else GPIO.LOW
        steps = abs(steps)
        
        GPIO.output(PIPE_DIR_PIN, direction)
        for _ in range(steps):
            if not self.is_running:
                break
            GPIO.output(PIPE_STEP_PIN, GPIO.HIGH)
            sleep(0.0002)  # Adjust speed as needed
            GPIO.output(PIPE_STEP_PIN, GPIO.LOW)
            sleep(0.0002)
    
    def scan_pipe_circumference(self):
        """Scan entire pipe circumference for gaps"""
        print("Starting pipe scan...")
        self.gap_map.clear()
        
        for angle in range(0, 360, 2):  # Scan every 2 degrees
            self.rotate_pipe(2)
            sleep(0.1)  # Allow camera to stabilize
            
            gap_size, gap_location = self.detect_gap()
            
            if gap_size > 0:
                # Store gap information
                self.gap_map[angle] = {
                    'size': gap_size,
                    'location': gap_location if gap_location is not None else None
                }
                
                # Check for critically large gaps
                if gap_size > 7:
                    print(f"WARNING: Critical gap detected at {angle}° - Size: {gap_size:.1f}mm")
                    cv2.imwrite(f'debug/critical_gap_{angle}deg_{time()}.jpg', 
                              self.camera.capture_array())
                    return False
                
            # Update progress
            if angle % 30 == 0:
                print(f"Scan progress: {angle/360*100:.1f}%")
        
        # Save gap map
        self._save_gap_map()
        return True
        
    def _save_gap_map(self):
        """Save gap map to file"""
        gap_data = {
            'timestamp': time(),
            'gaps': dict(self.gap_map)  # Convert defaultdict to regular dict
        }
        
        with open(f'debug/gap_map_{int(time())}.json', 'w') as f:
            json.dump(gap_data, f, indent=2)
            
    def verify_gap_position(self, current_gap):
        """Verify current gap against gap map"""
        # Find closest mapped gap
        closest_angle = min(self.gap_map.keys(), 
                          key=lambda x: abs(x - self.current_angle))
        
        mapped_gap = self.gap_map[closest_angle]
        
        # Check if current gap matches mapped gap within tolerance
        angle_tolerance = 5  # degrees
        size_tolerance = 1   # mm
        
        if (abs(closest_angle - self.current_angle) <= angle_tolerance and
            abs(mapped_gap['size'] - current_gap) <= size_tolerance):
            return True
        return False
    
    def run_welding_cycle(self):
        try:
            # First scan the pipe
            print("Performing initial pipe scan...")
            if not self.scan_pipe_circumference():
                print("Critical gap detected. Stopping system.")
                return
            
            print("Starting welding cycle...")
            while self.is_running:
                # Detect gap with enhanced visualization
                gap_size, gap_location = self.detect_gap()
                
                # Log gap measurements
                print(f"Current angle: {self.current_angle:.1f}°")
                print(f"Detected gap size: {gap_size:.1f}mm")
                
                # Verify gap position against map
                if gap_size > 0 and not self.verify_gap_position(gap_size):
                    print("Warning: Gap position mismatch with initial scan")
                
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
