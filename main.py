import sys
import platform
import cv2
import mediapipe as mp
import pyautogui
import json # Keep for potential future use, but QSettings is preferred now
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSystemTrayIcon, QMenu, QSlider, QComboBox, QCheckBox, QMessageBox, QSizePolicy, QGroupBox, QFormLayout, QStyle, QSpinBox, QTextEdit, QTabWidget, QFrame, QScrollArea, QStyleFactory
from PyQt6.QtGui import QIcon, QAction, QImage, QPixmap, QPainter, QColor, QFont, QPen, QBrush, QLinearGradient, QRadialGradient, QPalette # Import QPen and QPalette
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings, QPoint, QPointF, pyqtSlot, QSize, QPropertyAnimation, QEasingCurve, QRect, QParallelAnimationGroup, QSequentialAnimationGroup
from pynput import keyboard
import time # For visual feedback timer
import math # For distance calculation
import pygame # For sound effects
import os
import requests
import threading

# --- Constants ---
APP_NAME = "Swoosh"
ORG_NAME = "SwooshApp" # Use a specific org name for QSettings
VERSION = "2.0"

# Sound effect URLs (free sounds from freesound.org or similar)
DEFAULT_SOUNDS = {
    'swoosh_left': 'https://www.soundjay.com/misc/sounds/swoosh-1.wav',
    'swoosh_right': 'https://www.soundjay.com/misc/sounds/swoosh-2.wav',
    'page_flip': 'https://www.soundjay.com/misc/sounds/page-flip-01.wav'
}

# --- Global Variables ---
overlay_visible = False
listener = None
sound_manager = None

# --- Windows Native Styling ---
def apply_windows_native_style(app):
    """Apply Windows-native styling and themes"""
    if platform.system() == "Windows":
        # Use Windows native style
        app.setStyle('windowsvista')
        
        # Apply Windows 10/11 color scheme
        palette = QPalette()
        
        # Windows 10/11 colors
        window_color = QColor(240, 240, 240)  # Light gray background
        button_color = QColor(225, 225, 225)  # Light button color
        text_color = QColor(0, 0, 0)          # Black text
        highlight_color = QColor(0, 120, 215) # Windows blue
        disabled_color = QColor(109, 109, 109) # Gray for disabled
        
        # Set palette colors
        palette.setColor(QPalette.ColorRole.Window, window_color)
        palette.setColor(QPalette.ColorRole.WindowText, text_color)
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.ColorRole.Button, button_color)
        palette.setColor(QPalette.ColorRole.ButtonText, text_color)
        palette.setColor(QPalette.ColorRole.Highlight, highlight_color)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        # Set disabled colors using ColorGroup
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_color)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_color)
        
        app.setPalette(palette)
        
        # Set Windows-style fonts
        font = QFont("Segoe UI", 9)  # Windows 10/11 default font
        app.setFont(font)
    
    return app
# --- Utility Functions ---
def get_available_cameras(max_to_test=5):
    """Detects available camera indices."""
    available_cameras = []
    for i in range(max_to_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # Use CAP_DSHOW on Windows for better compatibility
        if cap is not None and cap.isOpened():
            available_cameras.append(i)
            cap.release()
    print(f"Available cameras found: {available_cameras}")
    return available_cameras

# --- Sound Manager ---
class SoundManager:
    def __init__(self):
        self.enabled = True
        self.volume = 0.7
        self.sounds = {}
        self.sound_dir = os.path.join(os.path.dirname(__file__), "sounds")
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.mixer_initialized = True
        except Exception as e:
            print(f"Failed to initialize sound mixer: {e}")
            self.mixer_initialized = False
            self.enabled = False
        
        # Create sounds directory if it doesn't exist
        if not os.path.exists(self.sound_dir):
            os.makedirs(self.sound_dir)
            
        self.load_default_sounds()
    
    def load_default_sounds(self):
        """Load or download default sound effects"""
        sound_files = {
            'swoosh_left': 'swoosh_left.wav',
            'swoosh_right': 'swoosh_right.wav', 
            'page_flip': 'page_flip.wav'
        }
        
        # Create simple sound effects programmatically since downloading may not work
        self.create_default_sounds()
    
    def create_default_sounds(self):
        """Create simple sound effects programmatically"""
        if not self.mixer_initialized:
            return
            
        try:
            # Try to create sounds with numpy if available
            try:
                import numpy as np
                self.create_numpy_sounds(np)
                print("Default sound effects created with numpy")
                return
            except ImportError:
                print("NumPy not available, using simple beep sounds")
                self.create_simple_beeps()
                
        except Exception as e:
            print(f"Failed to create sound effects: {e}")
            self.enabled = False
    
    def create_numpy_sounds(self, np):
        """Create sound effects using numpy"""
        sample_rate = 22050
        duration = 0.3  # seconds
        
        # Generate swoosh sounds (sine wave with frequency modulation)
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Left swoosh - frequency decreasing
        freq_left = 800 * np.exp(-t * 3)  # Exponentially decreasing frequency
        left_wave = np.sin(2 * np.pi * freq_left * t) * np.exp(-t * 2)
        
        # Right swoosh - frequency increasing then decreasing
        freq_right = 400 + 400 * np.sin(t * 10) * np.exp(-t * 2)
        right_wave = np.sin(2 * np.pi * freq_right * t) * np.exp(-t * 1.5)
        
        # Page flip - quick burst
        freq_flip = 1200 * np.exp(-t * 15)
        flip_wave = np.sin(2 * np.pi * freq_flip * t) * np.exp(-t * 10)
        
        # Convert to pygame sound format
        left_sound = pygame.sndarray.make_sound((left_wave * 32767).astype(np.int16))
        right_sound = pygame.sndarray.make_sound((right_wave * 32767).astype(np.int16))
        flip_sound = pygame.sndarray.make_sound((flip_wave * 32767).astype(np.int16))
        
        self.sounds['swoosh_left'] = left_sound
        self.sounds['swoosh_right'] = right_sound
        self.sounds['page_flip'] = flip_sound
    
    def create_simple_beeps(self):
        """Create simple beep sounds as fallback"""
        try:
            # Create simple square wave beeps
            sample_rate = 22050
            duration = 0.2
            
            def create_beep(frequency):
                frames = int(duration * sample_rate)
                arr = []
                for i in range(frames):
                    wave = 4096 * ((i // (sample_rate // frequency)) % 2)
                    arr.append([wave, wave])
                return pygame.sndarray.make_sound(arr)
            
            self.sounds['swoosh_left'] = create_beep(440)  # A note
            self.sounds['swoosh_right'] = create_beep(554)  # C# note  
            self.sounds['page_flip'] = create_beep(660)  # E note
            
        except Exception as e:
            print(f"Failed to create simple beeps: {e}")
            self.enabled = False
    
    def play_sound(self, sound_name):
        """Play a sound effect"""
        if not self.enabled or not self.mixer_initialized:
            return
            
        if sound_name in self.sounds:
            try:
                sound = self.sounds[sound_name]
                sound.set_volume(self.volume)
                sound.play()
            except Exception as e:
                print(f"Failed to play sound {sound_name}: {e}")
    
    def set_volume(self, volume):
        """Set volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
    
    def set_enabled(self, enabled):
        """Enable or disable sound effects"""
        self.enabled = enabled

# --- Hand Tracking Thread ---
class HandTrackingThread(QThread):
    changePixmap = pyqtSignal(QImage)
    # Update signal: gesture direction (str), is_fist_closed (bool)
    gestureDetected = pyqtSignal(str, bool, int)  # Added intensity parameter
    cameraError = pyqtSignal(str)
    # New signals for mouse mode
    mouseModeChanged = pyqtSignal(bool)  # True when entering mouse mode
    mouseAction = pyqtSignal(str, dict)  # Action type and parameters

    def __init__(self, camera_index=0, sensitivity=50, sound_manager=None):
        super().__init__()
        self.running = True
        self.camera_index = camera_index
        self.sensitivity = sensitivity
        self.sound_manager = sound_manager
        
        # Mode switching variables
        self.current_mode = "swoosh"  # "swoosh" or "mouse"
        self.mode_switch_timer = 0
        self.MODE_SWITCH_FRAMES = 60  # Frames to hold pose to switch modes (increased from 30 to 60)
        self.palm_open_frames = 0
        self.pointing_frames = 0
        
        # Mouse control variables
        self.prev_cursor_pos = (0, 0)
        self.mouse_down = False
        self.last_mouse_gesture_time = 0
        self.mouse_gesture_cooldown = 0.3
        self.smoothing_factor = 0.3
        self.finger_status = [False] * 5  # Thumb, Index, Middle, Ring, Pinky
        self.prev_hand_pos_y = None
        self.scroll_sensitivity = 0.5
        self.is_click_gesture = False
        self.is_drag_gesture = False
        self.is_scroll_gesture = False
        
        # Get screen dimensions for mouse mode
        import pyautogui
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.01
        
        # Try to initialize MediaPipe with error handling
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.6, max_num_hands=1)
            self.mp_draw = mp.solutions.drawing_utils
            self.mediapipe_available = True
            print("✅ MediaPipe initialized successfully")
        except Exception as e:
            print(f"⚠️ MediaPipe initialization failed: {e}")
            print("🔄 Hand tracking will be disabled")
            self.mediapipe_available = False
            self.mp_hands = None
            self.hands = None
            self.mp_draw = None
        
        self.prev_x = None
        self.gesture_cooldown = 0
        self.COOLDOWN_FRAMES = 15
        self.last_known_fist_state = False # Track last known state

    # Helper function to check if fist is closed
    def is_fist_closed(self, hand_landmarks, frame_shape):
        # Use landmarks relative to wrist or palm center
        # Heuristic: Check if fingertips are close to the palm center
        try:
            palm_center_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x # Use wrist as reference for now
            palm_center_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y

            # Fingertip landmarks indices: 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
            fingertip_indices = [8, 12, 16, 20]
            max_distance_threshold = 0.15 # Adjust this threshold based on testing
            closed_fingers = 0

            for idx in fingertip_indices:
                tip = hand_landmarks.landmark[idx]
                distance = math.sqrt((tip.x - palm_center_x)**2 + (tip.y - palm_center_y)**2)
                if distance < max_distance_threshold:
                    closed_fingers += 1

            # Consider fist closed if at least 3 fingertips are close to the palm/wrist
            return closed_fingers >= 3
        except Exception as e:
            print(f"Error checking fist state: {e}")
            return False # Default to not closed if error occurs

    def update_finger_statuses(self, hand_landmarks):
        """Update the status of each finger (extended/closed)"""
        try:
            # Thumb: Check if thumb tip (4) is to the right of thumb IP (3) for right hand
            self.finger_status[0] = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x
            
            # Other fingers: Check if finger tips are above their respective MCP joints
            for i in range(1, 5):
                tip_id = i * 4 + 4
                pip_id = i * 4 + 2
                self.finger_status[i] = hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y
        except Exception as e:
            print(f"Error updating finger status: {e}")

    def detect_mode_switch_gesture(self, hand_landmarks):
        """Detect gestures to switch between swoosh and mouse modes"""
        try:
            # Update finger statuses
            self.update_finger_statuses(hand_landmarks)
            
            # Mode switch logic using fist detection for better reliability:
            # - Closed fist for 2 seconds -> Switch modes (toggle between swoosh and mouse)
            
            is_fist = self.is_fist_closed(hand_landmarks, (640, 480))  # Use standard frame size
            
            if is_fist:
                self.pointing_frames += 1
                self.palm_open_frames = 0  # Reset other counter
                
                # Debug print every 10 frames to show progress
                if self.pointing_frames % 10 == 0:
                    progress = (self.pointing_frames / self.MODE_SWITCH_FRAMES) * 100
                    print(f"🤛 Fist detected - Mode switch progress: {progress:.0f}% ({self.pointing_frames}/{self.MODE_SWITCH_FRAMES})")
                
                if self.pointing_frames >= self.MODE_SWITCH_FRAMES:
                    # Switch modes
                    if self.current_mode == "swoosh":
                        self.current_mode = "mouse"
                        self.mouseModeChanged.emit(True)
                        print("🖱️ Switched to Mouse Mode (Fist detected)")
                    else:
                        self.current_mode = "swoosh"
                        self.mouseModeChanged.emit(False)
                        print("🖐️ Switched to Swoosh Mode (Fist detected)")
                    
                    # Reset counters after switch
                    self.pointing_frames = 0
                    self.palm_open_frames = 0
                    return True
            else:
                # Reset fist counter when hand is open
                if self.pointing_frames > 0:
                    print(f"✋ Hand opened - Resetting mode switch counter (was at {self.pointing_frames})")
                self.pointing_frames = 0
            
            return False
        except Exception as e:
            print(f"Error in mode switch detection: {e}")
            return False

    def detect_mouse_gestures(self, hand_landmarks, frame_width, frame_height):
        """Detect mouse control gestures"""
        try:
            # Get landmark positions
            landmarks = {}
            for i, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                landmarks[i] = (x, y)
            
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            
            # Distance between thumb and index finger for click detection
            thumb_index_distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + 
                                            (thumb_tip[1] - index_tip[1])**2)
            
            # Gesture detection
            is_pinching = thumb_index_distance < 30
            is_scrolling = (self.finger_status[1] and self.finger_status[2] and 
                           not self.finger_status[0] and not self.finger_status[3] and 
                           not self.finger_status[4])
            is_dragging = (self.finger_status[1] and not self.finger_status[0] and 
                          not self.finger_status[2] and not self.finger_status[3] and 
                          not self.finger_status[4])
            
            # Handle cursor movement (always move cursor in mouse mode)
            self.handle_cursor_movement(index_tip, frame_width, frame_height)
            
            # Handle scroll gesture
            if is_scrolling:
                current_y = index_tip[1]
                if self.prev_hand_pos_y is not None:
                    y_diff = current_y - self.prev_hand_pos_y
                    if abs(y_diff) > 5:  # Minimum movement threshold
                        scroll_amount = int(y_diff * self.scroll_sensitivity)
                        self.mouseAction.emit("scroll", {"amount": -scroll_amount})
                self.prev_hand_pos_y = current_y
                return "scroll"
            
            # Handle click gesture
            if is_pinching:
                if not self.is_click_gesture and time.time() - self.last_mouse_gesture_time > self.mouse_gesture_cooldown:
                    self.is_click_gesture = True
                    self.last_mouse_gesture_time = time.time()
                    self.mouseAction.emit("click", {})
                    return "click"
            else:
                self.is_click_gesture = False
            
            # Handle drag gesture
            if is_dragging:
                if not self.is_drag_gesture:
                    self.is_drag_gesture = True
                    self.mouseAction.emit("drag_start", {})
                return "drag"
            else:
                if self.is_drag_gesture:
                    self.is_drag_gesture = False
                    self.mouseAction.emit("drag_end", {})
            
            return "move"
            
        except Exception as e:
            print(f"Error in mouse gesture detection: {e}")
            return None

    def handle_cursor_movement(self, index_tip_pos, frame_width, frame_height):
        """Handle cursor movement in mouse mode"""
        try:
            if index_tip_pos:
                # Convert finger position to screen coordinates
                cursor_x = 1.5 * (index_tip_pos[0] * self.screen_width / frame_width)
                cursor_y = 1.5 * (index_tip_pos[1] * self.screen_height / frame_height)
                
                # Apply smoothing
                if self.prev_cursor_pos != (0, 0):
                    cursor_x = self.prev_cursor_pos[0] * self.smoothing_factor + cursor_x * (1 - self.smoothing_factor)
                    cursor_y = self.prev_cursor_pos[1] * self.smoothing_factor + cursor_y * (1 - self.smoothing_factor)
                
                # Limit coordinates to screen bounds
                cursor_x = max(0, min(self.screen_width - 1, cursor_x))
                cursor_y = max(0, min(self.screen_height - 1, cursor_y))
                
                # Update previous position
                self.prev_cursor_pos = (cursor_x, cursor_y)
                
                # Send mouse movement command
                self.mouseAction.emit("move", {"x": cursor_x, "y": cursor_y})
        except Exception as e:
            print(f"Error in cursor movement: {e}")

    def run(self):
        if not self.mediapipe_available:
            print("❌ Cannot start hand tracking: MediaPipe not available")
            self.cameraError.emit("MediaPipe initialization failed. Hand tracking disabled.")
            return
            
        try:
            # Use CAP_DSHOW on Windows
            api_preference = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.camera_index, api_preference)
            if not cap.isOpened():
                error_msg = f"Error: Could not open camera {self.camera_index}. It might be in use or unavailable."
                print(error_msg)
                self.cameraError.emit(error_msg)
                return
        except Exception as e:
             error_msg = f"Exception opening camera {self.camera_index}: {e}"
             print(error_msg)
             self.cameraError.emit(error_msg)
             return

        while self.running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Flip the frame horizontally for a later selfie-view display
            # and convert the BGR image to RGB.
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            
            # Only process with MediaPipe if available
            results = None
            if self.mediapipe_available and self.hands:
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                frame.flags.writeable = False
                results = self.hands.process(frame)
                # Draw the hand annotations on the image.
                frame.flags.writeable = True

            gesture = 'none'
            is_closed = self.last_known_fist_state # Default to last known state if no hand detected this frame
            gesture_intensity = 0  # How strong the gesture is (0-100)

            if results and results.multi_hand_landmarks and self.mediapipe_available:
                for hand_landmarks in results.multi_hand_landmarks:
                    try:
                        # --- Fist State Detection ---
                        is_closed = self.is_fist_closed(hand_landmarks, frame.shape)
                        self.last_known_fist_state = is_closed # Update last known state

                        # Check for mode switching gesture first
                        mode_switched = self.detect_mode_switch_gesture(hand_landmarks)
                        
                        # Skip normal gesture processing if we just switched modes
                        if mode_switched:
                            continue
                        
                        if self.current_mode == "mouse":
                            # Mouse control mode
                            mouse_gesture = self.detect_mouse_gestures(hand_landmarks, frame.shape[1], frame.shape[0])
                            # Set drawing color to blue for mouse mode
                            landmark_color = (255, 0, 0)  # Blue for mouse mode
                            
                        elif self.current_mode == "swoosh":
                            # Swoosh gesture mode (original logic)
                            # --- Gesture Detection Logic ---
                            # Using the wrist landmark (index 0) as a reference point
                            wrist_landmark = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                            # Get landmark relative to the center for potentially more stable detection
                            center_x = sum([lm.x for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
                            # cx = int(wrist_landmark.x * frame.shape[1])
                            cx = int(center_x * frame.shape[1]) # Use average X

                            if self.gesture_cooldown == 0:
                                if self.prev_x is not None:
                                    delta_x = cx - self.prev_x
                                    # Adjust sensitivity threshold based on slider (10-100) -> map to pixel threshold
                                    # Lower sensitivity value means HIGHER threshold (less sensitive)
                                    # Higher sensitivity value means LOWER threshold (more sensitive)
                                    sensitivity_factor = (110 - self.sensitivity) / 100.0 # Inverted scale (1.0 to 0.1)
                                    threshold = frame.shape[1] * 0.05 * sensitivity_factor # Base threshold * factor

                                    if delta_x > threshold:
                                        gesture = 'right'
                                        # Calculate intensity based on how much the threshold was exceeded
                                        gesture_intensity = min(100, int((delta_x / threshold - 1) * 50) + 50)
                                        print(f"Swipe Right Detected (Fist Closed: {is_closed}, Intensity: {gesture_intensity})")
                                        # Play sound effect
                                        if self.sound_manager:
                                            if is_closed:
                                                self.sound_manager.play_sound('page_flip')
                                            else:
                                                self.sound_manager.play_sound('swoosh_right')
                                        self.gesture_cooldown = self.COOLDOWN_FRAMES
                                    elif delta_x < -threshold:
                                        gesture = 'left'
                                        # Calculate intensity for left swipes too
                                        gesture_intensity = min(100, int((-delta_x / threshold - 1) * 50) + 50)
                                        print(f"Swipe Left Detected (Fist Closed: {is_closed}, Intensity: {gesture_intensity})")
                                        # Play sound effect
                                        if self.sound_manager:
                                            if is_closed:
                                                self.sound_manager.play_sound('page_flip')
                                            else:
                                                self.sound_manager.play_sound('swoosh_left')
                                        self.gesture_cooldown = self.COOLDOWN_FRAMES

                            self.prev_x = cx
                            # Set drawing color based on fist state
                            landmark_color = (0, 255, 0) if not is_closed else (0, 0, 255) # Green for open, Red for closed
                            # --- End Gesture Detection ---

                        # --- Drawing --- #
                        if self.mp_draw:
                            self.mp_draw.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                self.mp_draw.DrawingSpec(color=landmark_color, thickness=2, circle_radius=4),
                                self.mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                            )
                    except Exception as e:
                        print(f"Error processing hand landmarks: {e}")
                        continue
            else: # No hand detected or MediaPipe not available
                 self.prev_x = None # Reset prev_x if no hand is visible
                 # Keep last_known_fist_state as is

            if self.gesture_cooldown > 0:
                self.gesture_cooldown -= 1
            else:
                # Reset prev_x only when cooldown finishes AND no hand is detected
                 if not (results and results.multi_hand_landmarks):
                    self.prev_x = None

            # Emit gesture signal with fist state and intensity
            if gesture != 'none':
                self.gestureDetected.emit(gesture, is_closed, gesture_intensity)

            # Convert frame for Qt display (already RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            p = convert_to_Qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
            self.changePixmap.emit(p)

            # Use QThread's sleep for better integration
            self.msleep(10) # ~100 FPS theoretical max, adjust if needed

        cap.release()
        if self.hands:
            self.hands.close()
        print("Hand tracking thread stopped.")

    def stop(self):
        self.running = False
        self.wait() # Wait for the thread to finish

# --- Settings Window ---
class SettingsWindow(QWidget):
    settingsChanged = pyqtSignal()

    def __init__(self, sound_manager=None):
        super().__init__()
        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.sound_manager = sound_manager
        self.setMinimumSize(600, 500)
        self.setWindowTitle(f"{APP_NAME} Settings v{VERSION}")
        
        # Apply Windows native styling
        if platform.system() == "Windows":
            self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint | Qt.WindowType.WindowMinimizeButtonHint)
            # Use Windows 10/11 style window
            self.setStyleSheet("""
                QWidget {
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                    background-color: #f0f0f0;
                }
                QTabWidget::pane {
                    border: 1px solid #c0c0c0;
                    background-color: white;
                }
                QTabBar::tab {
                    background: #e1e1e1;
                    border: 1px solid #c0c0c0;
                    padding: 8px 16px;
                    margin-right: 2px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                QTabBar::tab:selected {
                    background: white;
                    border-bottom-color: white;
                }
                QTabBar::tab:hover {
                    background: #e5f3ff;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #c0c0c0;
                    border-radius: 8px;
                    margin-top: 1ex;
                    padding-top: 10px;
                    background-color: white;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 8px 0 8px;
                    background-color: white;
                }
                QPushButton {
                    background-color: #e1e1e1;
                    border: 1px solid #adadad;
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-weight: normal;
                }
                QPushButton:hover {
                    background-color: #e5f3ff;
                    border-color: #0078d4;
                }
                QPushButton:pressed {
                    background-color: #cce4f7;
                }
                QCheckBox {
                    spacing: 8px;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border: 1px solid #c0c0c0;
                    border-radius: 2px;
                    background-color: white;
                }
                QCheckBox::indicator:checked {
                    background-color: #0078d4;
                    border-color: #0078d4;
                }
                QSlider::groove:horizontal {
                    border: 1px solid #c0c0c0;
                    height: 8px;
                    background: #f0f0f0;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #0078d4;
                    border: 1px solid #005a9e;
                    width: 18px;
                    margin: -5px 0;
                    border-radius: 9px;
                }
                QSlider::handle:horizontal:hover {
                    background: #106ebe;
                }
                QComboBox, QSpinBox {
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    padding: 4px 8px;
                    background-color: white;
                }
                QComboBox:focus, QSpinBox:focus {
                    border-color: #0078d4;
                }
                QTextEdit {
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    background-color: white;
                    padding: 8px;
                }
            """)
        
        # self.setWindowIcon(QIcon("path/to/your/icon.png")) # TODO: Add an icon
        
        # Create tabbed interface
        self.tab_widget = QTabWidget()
        
        # Input Settings Tab
        self.input_tab = self.create_input_tab()
        self.tab_widget.addTab(self.input_tab, "📹 Input")
        
        # Actions Tab  
        self.actions_tab = self.create_actions_tab()
        self.tab_widget.addTab(self.actions_tab, "⚡ Actions")
        
        # Audio Tab
        self.audio_tab = self.create_audio_tab()
        self.tab_widget.addTab(self.audio_tab, "🔊 Audio")
        
        # Display Tab
        self.display_tab = self.create_display_tab()
        self.tab_widget.addTab(self.display_tab, "🎨 Display")
        
        # About Tab
        self.about_tab = self.create_about_tab()
        self.tab_widget.addTab(self.about_tab, "ℹ️ About")
        
        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)
        
        # Buttons layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        reset_button = QPushButton("🔄 Reset to Defaults")
        reset_button.setToolTip("Restore all settings to the defaults")
        reset_button.clicked.connect(self.reset_settings)
        reset_button.setMinimumHeight(32)
        
        test_button = QPushButton("🧪 Test Sounds")
        test_button.setToolTip("Test the sound effects")
        test_button.clicked.connect(self.test_sounds)
        test_button.setMinimumHeight(32)
        
        close_button = QPushButton("✅ Close")
        close_button.clicked.connect(self.close_settings)
        close_button.setMinimumHeight(32)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                border: 1px solid #005a9e;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        
        button_layout.addWidget(reset_button)
        button_layout.addWidget(test_button)
        button_layout.addStretch(1)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.load_settings() # Load settings when window is created
    
    def create_input_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Camera Settings group
        camera_group = QGroupBox("Camera Settings")
        camera_form = QFormLayout()
        
        self.camera_combo = QComboBox()
        self.populate_camera_list()
        self.camera_combo.currentIndexChanged.connect(self.update_camera_setting)
        camera_form.addRow(QLabel("Select Camera:"), self.camera_combo)
        
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(10, 100)
        self.sensitivity_slider.setToolTip("Adjust how much hand movement triggers a swipe.")
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity_label)
        self.sensitivity_slider.valueChanged.connect(self.settingsChanged.emit)
        self.sensitivity_label = QLabel()
        camera_form.addRow(QLabel("Sensitivity:"), self.sensitivity_slider)
        camera_form.addRow(QLabel(""), self.sensitivity_label)
        
        # Cooldown setting
        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(5, 50)
        self.cooldown_spin.setSuffix(" frames")
        self.cooldown_spin.setToolTip("Delay between gesture detections")
        self.cooldown_spin.valueChanged.connect(lambda v: self.settings.setValue("gesture_cooldown", v))
        camera_form.addRow(QLabel("Gesture Cooldown:"), self.cooldown_spin)
        
        camera_group.setLayout(camera_form)
        layout.addWidget(camera_group)
        
        # Detection Settings group
        detection_group = QGroupBox("Detection Settings")
        detection_form = QFormLayout()
        
        self.min_detection_spin = QSpinBox()
        self.min_detection_spin.setRange(10, 100)
        self.min_detection_spin.setSuffix("%")
        self.min_detection_spin.setToolTip("Minimum confidence for hand detection")
        self.min_detection_spin.valueChanged.connect(lambda v: self.settings.setValue("min_detection_confidence", v/100.0))
        detection_form.addRow(QLabel("Detection Confidence:"), self.min_detection_spin)
        
        self.min_tracking_spin = QSpinBox()
        self.min_tracking_spin.setRange(10, 100)
        self.min_tracking_spin.setSuffix("%")
        self.min_tracking_spin.setToolTip("Minimum confidence for hand tracking")
        self.min_tracking_spin.valueChanged.connect(lambda v: self.settings.setValue("min_tracking_confidence", v/100.0))
        detection_form.addRow(QLabel("Tracking Confidence:"), self.min_tracking_spin)
        
        detection_group.setLayout(detection_form)
        layout.addWidget(detection_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_actions_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Action Toggles group
        action_group = QGroupBox("Gesture Actions")
        action_layout = QVBoxLayout()
        
        self.window_switch_cb = QCheckBox("Enable Window Switching (Alt+Tab / Cmd+Tab)")
        self.window_switch_cb.setToolTip("Switch between open applications")
        self.window_switch_cb.toggled.connect(lambda checked: (self.settings.setValue("enable_window_switch", checked), self.settingsChanged.emit()))
        action_layout.addWidget(self.window_switch_cb)
        
        self.desktop_switch_cb = QCheckBox("Enable Virtual Desktop Switching (Ctrl+Win+Arrows / Ctrl+Arrows)")
        self.desktop_switch_cb.setToolTip("Switch between virtual desktops/spaces")
        self.desktop_switch_cb.toggled.connect(lambda checked: (self.settings.setValue("enable_desktop_switch", checked), self.settingsChanged.emit()))
        action_layout.addWidget(self.desktop_switch_cb)
        
        # Mode selection
        mode_group = QGroupBox("Control Mode")
        mode_layout = QVBoxLayout()
        
        self.fist_mode_cb = QCheckBox("Fist = Window Switch, Open Hand = Desktop Switch")
        self.fist_mode_cb.setToolTip("Closed fist switches windows, open hand switches desktops")
        self.fist_mode_cb.toggled.connect(lambda checked: self.settings.setValue("fist_mode", checked))
        mode_layout.addWidget(self.fist_mode_cb)
        
        mode_group.setLayout(mode_layout)
        action_layout.addWidget(mode_group)
        
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # Custom hotkeys section
        hotkey_group = QGroupBox("Keyboard Shortcuts")
        hotkey_layout = QFormLayout()
        
        self.toggle_hotkey_label = QLabel("Ctrl+Shift+A")
        hotkey_layout.addRow(QLabel("Toggle Overlay:"), self.toggle_hotkey_label)
        
        hotkey_group.setLayout(hotkey_layout)
        layout.addWidget(hotkey_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_audio_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Audio Settings group
        audio_group = QGroupBox("Sound Effects")
        audio_form = QFormLayout()
        
        self.sound_enabled_cb = QCheckBox("Enable Sound Effects")
        self.sound_enabled_cb.setToolTip("Play sound effects when gestures are detected")
        self.sound_enabled_cb.toggled.connect(self.toggle_sound_effects)
        audio_form.addRow(self.sound_enabled_cb)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setToolTip("Adjust volume of sound effects")
        self.volume_slider.valueChanged.connect(self.update_volume)
        self.volume_label = QLabel()
        audio_form.addRow(QLabel("Volume:"), self.volume_slider)
        audio_form.addRow(QLabel(""), self.volume_label)
        
        audio_group.setLayout(audio_form)
        layout.addWidget(audio_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_display_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Visual Settings group
        visual_group = QGroupBox("Visual Effects")
        visual_form = QFormLayout()
        
        self.show_trail_cb = QCheckBox("Show Gesture Trail")
        self.show_trail_cb.setToolTip("Show visual trail when gestures are detected")
        self.show_trail_cb.toggled.connect(lambda checked: self.settings.setValue("show_trail", checked))
        visual_form.addRow(self.show_trail_cb)
        
        self.show_feedback_cb = QCheckBox("Show Text Feedback")
        self.show_feedback_cb.setToolTip("Show text feedback for actions")
        self.show_feedback_cb.toggled.connect(lambda checked: self.settings.setValue("show_feedback", checked))
        visual_form.addRow(self.show_feedback_cb)
        
        self.overlay_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_opacity_slider.setRange(20, 100)
        self.overlay_opacity_slider.setToolTip("Adjust overlay transparency")
        self.overlay_opacity_slider.valueChanged.connect(self.update_opacity_label)
        self.opacity_label = QLabel()
        visual_form.addRow(QLabel("Overlay Opacity:"), self.overlay_opacity_slider)
        visual_form.addRow(QLabel(""), self.opacity_label)
        
        visual_group.setLayout(visual_form)
        layout.addWidget(visual_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_about_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # About information
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setMaximumHeight(300)
        about_content = f"""
<h2>🚀 {APP_NAME} v{VERSION}</h2>
<p><b>The Ultimate Hand Gesture Desktop Control</b></p>

<h3>📖 How to Use:</h3>
<ul>
<li><b>Open Hand + Swipe:</b> Switch virtual desktops</li>
<li><b>Closed Fist + Swipe:</b> Switch between applications</li>
<li><b>Left Swipe:</b> Go to previous desktop/window</li>
<li><b>Right Swipe:</b> Go to next desktop/window</li>
</ul>

<h3>⌨️ Keyboard Shortcuts:</h3>
<ul>
<li><b>Ctrl+Shift+A:</b> Toggle camera overlay</li>
</ul>

<h3>🎯 Features:</h3>
<ul>
<li>Real-time hand gesture recognition</li>
<li>Customizable sensitivity settings</li>
<li>Sound effects and visual feedback</li>
<li>System tray integration</li>
<li>Cross-platform support</li>
</ul>

<h3>🔧 Technical:</h3>
<p>Built with PyQt6, OpenCV, and MediaPipe<br>
Supports Windows, macOS, and Linux</p>

<h3>💡 Tips:</h3>
<ul>
<li>Ensure good lighting for best hand detection</li>
<li>Adjust sensitivity if gestures are too sensitive/insensitive</li>
<li>Use the test sounds feature to check audio</li>
</ul>
        """
        about_text.setHtml(about_content)
        layout.addWidget(about_text)
        
        widget.setLayout(layout)
        return widget

    def populate_camera_list(self):
        self.camera_combo.clear()
        cameras = get_available_cameras()
        if not cameras:
            self.camera_combo.addItem("No cameras found", -1)
            self.camera_combo.setEnabled(False)
        else:
            for idx in cameras:
                self.camera_combo.addItem(f"Camera {idx}", idx)
            self.camera_combo.setEnabled(True)

    def update_camera_setting(self, index):
         if index != -1:
            camera_index = self.camera_combo.itemData(index)
            if camera_index is not None:
                self.settings.setValue("camera_index", camera_index)
                print(f"Camera setting updated to index: {camera_index}")
                self.settingsChanged.emit() # Emit signal to restart tracking

    def update_sensitivity_label(self, value):
        self.sensitivity_label.setText(f"Current Sensitivity: {value}")
        self.settings.setValue("sensitivity", value)
    
    def update_volume(self, value):
        self.volume_label.setText(f"Volume: {value}%")
        volume = value / 100.0
        self.settings.setValue("sound_volume", volume)
        if self.sound_manager:
            self.sound_manager.set_volume(volume)
    
    def update_opacity_label(self, value):
        self.opacity_label.setText(f"Opacity: {value}%")
        self.settings.setValue("overlay_opacity", value / 100.0)
    
    def toggle_sound_effects(self, enabled):
        self.settings.setValue("sound_enabled", enabled)
        if self.sound_manager:
            self.sound_manager.set_enabled(enabled)
        # Enable/disable volume slider
        self.volume_slider.setEnabled(enabled)
    
    def test_sounds(self):
        if self.sound_manager and self.sound_manager.enabled:
            # Play all three sounds in sequence
            QTimer.singleShot(0, lambda: self.sound_manager.play_sound('swoosh_left'))
            QTimer.singleShot(500, lambda: self.sound_manager.play_sound('swoosh_right'))
            QTimer.singleShot(1000, lambda: self.sound_manager.play_sound('page_flip'))
        else:
            QMessageBox.information(self, "Sound Test", "Sound effects are disabled or not available.")

    def load_settings(self):
        # Load from QSettings, providing defaults
        camera_index = self.settings.value("camera_index", 0, type=int)
        sensitivity = self.settings.value("sensitivity", 50, type=int)
        enable_window_switch = self.settings.value("enable_window_switch", True, type=bool)
        enable_desktop_switch = self.settings.value("enable_desktop_switch", True, type=bool)
        sound_enabled = self.settings.value("sound_enabled", True, type=bool)
        sound_volume = self.settings.value("sound_volume", 0.7, type=float)
        show_trail = self.settings.value("show_trail", True, type=bool)
        show_feedback = self.settings.value("show_feedback", True, type=bool)
        overlay_opacity = self.settings.value("overlay_opacity", 0.8, type=float)
        gesture_cooldown = self.settings.value("gesture_cooldown", 15, type=int)
        min_detection = self.settings.value("min_detection_confidence", 0.75, type=float)
        min_tracking = self.settings.value("min_tracking_confidence", 0.6, type=float)
        fist_mode = self.settings.value("fist_mode", True, type=bool)

        # Update UI elements
        index = self.camera_combo.findData(camera_index)
        if index != -1:
            self.camera_combo.setCurrentIndex(index)
        else: # If saved camera not found, default to first available
             if self.camera_combo.count() > 0:
                 self.camera_combo.setCurrentIndex(0)
                 self.settings.setValue("camera_index", self.camera_combo.itemData(0)) # Save the default

        self.sensitivity_slider.setValue(sensitivity)
        self.update_sensitivity_label(sensitivity) # Update label too
        self.window_switch_cb.setChecked(enable_window_switch)
        self.desktop_switch_cb.setChecked(enable_desktop_switch)
        
        # Audio settings
        self.sound_enabled_cb.setChecked(sound_enabled)
        self.volume_slider.setValue(int(sound_volume * 100))
        self.update_volume(int(sound_volume * 100))
        self.volume_slider.setEnabled(sound_enabled)
        
        # Display settings
        self.show_trail_cb.setChecked(show_trail)
        self.show_feedback_cb.setChecked(show_feedback)
        self.overlay_opacity_slider.setValue(int(overlay_opacity * 100))
        self.update_opacity_label(int(overlay_opacity * 100))
        
        # Detection settings
        self.cooldown_spin.setValue(gesture_cooldown)
        self.min_detection_spin.setValue(int(min_detection * 100))
        self.min_tracking_spin.setValue(int(min_tracking * 100))
        
        # Mode settings
        self.fist_mode_cb.setChecked(fist_mode)
        
        print("Settings loaded from QSettings")

    def close_settings(self):
        # Settings are saved automatically on UI interaction via signals
        # We might want an explicit save on close if not using signals
        # self.settings.sync() # Ensure settings are written to disk
        print("Settings window closed.")
        self.settingsChanged.emit() # Emit signal to ensure tracking restarts with latest settings
        self.hide()
    
    def reset_settings(self):
        """Clear all stored settings and reload defaults."""
        self.settings.clear()
        self.load_settings()
        self.settingsChanged.emit()
        QMessageBox.information(self, "Settings Reset", "Settings have been reset to defaults.")


# --- Overlay Window ---
class OverlayWindow(QWidget):
    # Signal to request showing settings
    requestShowSettings = pyqtSignal()

    def __init__(self, sound_manager=None):
        super().__init__()
        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.sound_manager = sound_manager
        
        # Apply Windows-native window styling
        if platform.system() == "Windows":
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.WindowStaysOnTopHint |
                Qt.WindowType.Tool
            )
            # Use modern Windows styling for overlay
            self.setStyleSheet("""
                QWidget {
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                }
                QLabel {
                    color: white;
                    font-weight: 500;
                }
                QPushButton {
                    background-color: rgba(32, 32, 32, 0.8);
                    color: white;
                    border: 1px solid rgba(128, 128, 128, 0.3);
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: rgba(64, 64, 64, 0.9);
                    border-color: rgba(0, 120, 215, 0.8);
                }
                QPushButton:pressed {
                    background-color: rgba(0, 84, 153, 0.9);
                }
            """)
        else:
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.WindowStaysOnTopHint |
                Qt.WindowType.Tool
            )
        
        # Remove transparency attributes that can cause issues on Windows
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Main layout (vertical)
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5) # Add small margins

        # Top bar layout (horizontal) for buttons
        self.top_bar_layout = QHBoxLayout()
        self.top_bar_layout.setContentsMargins(0, 0, 0, 0)

        # Mode indicator with Windows styling
        self.mode_label = QLabel("👋 Ready")
        if platform.system() == "Windows":
            self.mode_label.setStyleSheet("""
                QLabel { 
                    color: white; 
                    font-size: 11px; 
                    font-weight: 600;
                    font-family: 'Segoe UI';
                    padding: 2px 6px;
                }
            """)
        else:
            self.mode_label.setStyleSheet("color: white; font-size: 12px; font-weight: bold;")
        self.mode_label.setToolTip("Current detection mode")
        
        # Settings Button with Windows-native styling
        self.settings_button = QPushButton("⚙️") # Gear icon
        self.settings_button.setFixedSize(QSize(28, 28))
        self.settings_button.setToolTip("Open Settings")
        self.settings_button.clicked.connect(self.requestShowSettings.emit)

        # Minimize button with Windows-native styling
        self.minimize_button = QPushButton("─")  # Use proper minimize symbol
        self.minimize_button.setFixedSize(QSize(28, 28))
        self.minimize_button.setToolTip("Hide Overlay")
        self.minimize_button.clicked.connect(self.hide)

        self.top_bar_layout.addWidget(self.mode_label)
        self.top_bar_layout.addStretch(1) # Push buttons to the right
        self.top_bar_layout.addWidget(self.settings_button)
        self.top_bar_layout.addWidget(self.minimize_button)
        self.main_layout.addLayout(self.top_bar_layout)

        # Image Label (takes remaining space) with Windows styling
        self.image_label = QLabel("🎥 Initializing Camera...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        if platform.system() == "Windows":
            self.image_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(32, 32, 32, 0.9);
                    color: white;
                    font-size: 16px;
                    font-family: 'Segoe UI';
                    font-weight: 500;
                    border-radius: 8px;
                    border: 1px solid rgba(128, 128, 128, 0.3);
                }
            """)
        else:
            self.image_label.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: white; font-size: 18px; border-radius: 10px;")
        
        self.main_layout.addWidget(self.image_label)

        self.setLayout(self.main_layout)

        # Define the fixed size for the camera feed display
        self.fixed_camera_feed_size = QSize(480, 360) # Example fixed size

        # Adjust window size based on fixed feed size + button bar height
        screen_geometry = QApplication.primaryScreen().geometry()
        width = self.fixed_camera_feed_size.width() + 10 # Add padding
        height = self.fixed_camera_feed_size.height() + 30 + 10 # Add button height and padding
        self.default_pos = QPoint((screen_geometry.width() - width) // 2, (screen_geometry.height() - height) // 2)
        self.setGeometry(self.default_pos.x(), self.default_pos.y(), width, height)
        # Set minimum size to prevent making it smaller than the feed
        self.setMinimumSize(width, height)

        self.tracking_thread = None
        self.feedback_timer = QTimer(self)
        self.feedback_timer.setSingleShot(True)
        self.feedback_timer.timeout.connect(self.hide_feedback)
        self.feedback_text = ""  # Text for feedback
        self.feedback_color = QColor("transparent")
        
        # Enhanced visual feedback elements
        self.gesture_animation = QPropertyAnimation(self, b"geometry")
        self.gesture_animation.setDuration(300)
        self.gesture_animation.setEasingCurve(QEasingCurve.Type.OutBack)
        
        # Text animation
        self.text_scale = 1.0
        self.text_animation = QPropertyAnimation()
        self.text_animation.setDuration(400)
        
        # Track gesture trail (last few positions)
        self.gesture_trail = []
        self.gesture_trail_max = 8  # Maximum number of points to remember
        self.gesture_direction = None  # Current gesture direction
        self.gesture_intensity = 0  # Current gesture intensity
        
        # Particle system for swoosh effect
        self.particles = []
        self.particle_timer = QTimer()
        self.particle_timer.timeout.connect(self.update_particles)
        
        # Prepare fade-in animation
        self.fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self.fade_anim.setDuration(300)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Status indicators
        self.last_gesture_time = 0
        self.fps_counter = 0
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)  # Update every second

    # Fade in on show
    def showEvent(self, event):
        super().showEvent(event)
        self.setWindowOpacity(0)
        self.fade_anim.stop()
        self.fade_anim.setStartValue(0)
        self.fade_anim.setEndValue(self.settings.value("overlay_opacity", 0.8, type=float))
        self.fade_anim.start()
    
    def update_fps(self):
        """Update FPS counter and mode indicator"""
        self.fps_counter = 0
        # Update mode indicator based on hand state
        current_time = time.time()
        if current_time - self.last_gesture_time < 2:  # Show gesture info for 2 seconds
            return  # Keep current gesture info
        
        # Default mode display
        self.mode_label.setText("👋 Ready")
    
    def create_particles(self, direction, intensity):
        """Create particle effects for swoosh"""
        center = self.rect().center()
        particle_count = min(20, max(5, intensity // 5))
        
        for i in range(particle_count):
            particle = {
                'x': center.x() + (i - particle_count//2) * 10,
                'y': center.y() + (i - particle_count//2) * 5,
                'vx': (50 if direction == 'right' else -50) + (i - particle_count//2) * 2,
                'vy': (i - particle_count//2) * 3,
                'life': 1.0,
                'size': 3 + (intensity // 20),
                'color': QColor(255, 165, 0) if direction == 'right' else QColor(100, 200, 255)
            }
            self.particles.append(particle)
        
        # Start particle animation if not already running
        if not self.particle_timer.isActive():
            self.particle_timer.start(16)  # ~60 FPS
    
    def update_particles(self):
        """Update particle animation"""
        if not self.particles:
            self.particle_timer.stop()
            return
        
        # Update each particle
        for particle in self.particles[:]:  # Copy list to allow removal during iteration
            particle['x'] += particle['vx'] * 0.1
            particle['y'] += particle['vy'] * 0.1
            particle['life'] -= 0.05
            particle['vy'] += 0.5  # Gravity
            particle['vx'] *= 0.98  # Air resistance
            
            if particle['life'] <= 0:
                self.particles.remove(particle)
        
        self.update()  # Trigger repaint
        
        if not self.particles:
            self.particle_timer.stop()
    
    # Override paintEvent to draw feedback with Windows-native styling
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get settings for visual effects
        show_trail = self.settings.value("show_trail", True, type=bool)
        show_feedback = self.settings.value("show_feedback", True, type=bool)

        # Draw background with Windows 10/11 acrylic-like effect
        center = self.rect().center()
        
        if platform.system() == "Windows":
            # Modern Windows acrylic background
            bg_gradient = QRadialGradient(float(center.x()), float(center.y()), min(self.width(), self.height()) / 2)
            bg_gradient.setColorAt(0, QColor(45, 45, 48, int(0.85 * 255)))    # Dark center
            bg_gradient.setColorAt(1, QColor(32, 32, 35, int(0.95 * 255)))    # Darker edges
            
            border_color = self.feedback_color
            border_width = 2 if border_color != QColor("transparent") else 1

            painter.setBrush(QBrush(bg_gradient))
            painter.setPen(QPen(border_color if border_width > 1 else QColor(64, 64, 64, 150), border_width))
            rect = self.rect().adjusted(1, 1, -1, -1)
            painter.drawRoundedRect(rect, 8, 8)  # Smaller radius for Windows style
        else:
            # Original styling for non-Windows
            bg_gradient = QRadialGradient(float(center.x()), float(center.y()), min(self.width(), self.height()) / 2)
            bg_gradient.setColorAt(0, QColor(0, 0, 0, int(0.4 * 255)))
            bg_gradient.setColorAt(1, QColor(0, 0, 0, int(0.7 * 255)))
            
            border_color = self.feedback_color
            border_width = 3 if border_color != QColor("transparent") else 1

            painter.setBrush(QBrush(bg_gradient))
            painter.setPen(QPen(border_color if border_width > 1 else QColor(100, 100, 100, 100), border_width))
            rect = self.rect().adjusted(1, 1, -1, -1)
            painter.drawRoundedRect(rect, 10, 10)

        # Draw particles
        for particle in self.particles:
            alpha = int(particle['life'] * 255)
            color = QColor(particle['color'])
            color.setAlpha(alpha)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            size = particle['size'] * particle['life']
            painter.drawEllipse(QPoint(int(particle['x']), int(particle['y'])), int(size), int(size))
        
        # Draw gesture trail if enabled and active
        if show_trail and self.gesture_trail and self.gesture_direction:
            # Create gradient for trail
            trail_gradient = QLinearGradient()
            if len(self.gesture_trail) > 1:
                start_point = self.gesture_trail[0]
                end_point = self.gesture_trail[-1]
                # Convert QPoint to QPointF for Qt6 compatibility
                trail_gradient.setStart(QPointF(start_point))
                trail_gradient.setFinalStop(QPointF(end_point))
            
            trail_color = QColor(255, 165, 0, 180)  # Orange with transparency
            if self.gesture_direction == "left":
                trail_color = QColor(255, 100, 100, 180)  # Red-ish
                trail_gradient.setColorAt(0, QColor(255, 100, 100, 50))
                trail_gradient.setColorAt(1, QColor(255, 100, 100, 200))
            elif self.gesture_direction == "right":
                trail_color = QColor(100, 200, 255, 180)  # Blue-ish
                trail_gradient.setColorAt(0, QColor(100, 200, 255, 50))
                trail_gradient.setColorAt(1, QColor(100, 200, 255, 200))
                
            # Draw connecting lines between trail points with thickness variation
            if len(self.gesture_trail) > 1:
                for i in range(1, len(self.gesture_trail)):
                    thickness = 2 + (i * 3 / len(self.gesture_trail))  # Thicker towards end
                    alpha = 100 + (155 * i / len(self.gesture_trail))
                    line_color = QColor(trail_color)
                    line_color.setAlpha(int(alpha))
                    
                    pen = QPen(line_color, thickness)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(pen)
                    
                    p1 = self.gesture_trail[i-1]
                    p2 = self.gesture_trail[i]
                    painter.drawLine(p1, p2)
                    
            # Draw trail points (dots with glow effect)
            for i, point in enumerate(self.gesture_trail):
                # Gradually increase size and opacity for newest points
                alpha = 100 + (155 * i / len(self.gesture_trail))
                size = 4 + (i * 3)
                
                # Draw glow effect
                glow_color = QColor(trail_color)
                glow_color.setAlpha(int(alpha * 0.3))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(glow_color)
                painter.drawEllipse(point, size + 4, size + 4)
                
                # Draw main dot
                point_color = QColor(trail_color)
                point_color.setAlpha(int(alpha))
                painter.setBrush(point_color)
                painter.drawEllipse(point, size, size)
                    
            # Draw enhanced arrow at the end of trail
            if self.gesture_direction and self.gesture_trail:
                last_point = self.gesture_trail[-1]
                arrow_size = 15 + (self.gesture_intensity // 8)
                arrow_color = QColor(trail_color)
                arrow_color.setAlpha(240)
                
                # Draw arrow glow
                glow_color = QColor(arrow_color)
                glow_color.setAlpha(80)
                painter.setBrush(glow_color)
                painter.setPen(Qt.PenStyle.NoPen)
                
                # Main arrow
                painter.setBrush(arrow_color)
                
                # Draw arrow shape based on direction with better proportions
                if self.gesture_direction == "left":
                    points = [
                        QPoint(last_point.x() - arrow_size, last_point.y()),
                        QPoint(last_point.x() + arrow_size//3, last_point.y() - arrow_size//2),
                        QPoint(last_point.x() + arrow_size//6, last_point.y()),
                        QPoint(last_point.x() + arrow_size//3, last_point.y() + arrow_size//2)
                    ]
                else:  # right
                    points = [
                        QPoint(last_point.x() + arrow_size, last_point.y()),
                        QPoint(last_point.x() - arrow_size//3, last_point.y() - arrow_size//2),
                        QPoint(last_point.x() - arrow_size//6, last_point.y()),
                        QPoint(last_point.x() - arrow_size//3, last_point.y() + arrow_size//2)
                    ]
                painter.drawPolygon(points)

        # Draw feedback text with enhanced styling if enabled
        if show_feedback and self.feedback_text:
            # Text shadow effect
            shadow_color = QColor(0, 0, 0, 180)
            painter.setPen(shadow_color)
            
            if platform.system() == "Windows":
                font = QFont("Segoe UI", int(14 * self.text_scale))
                font.setWeight(QFont.Weight.DemiBold)
            else:
                font = QFont()
                font.setPointSize(int(16 * self.text_scale))
                font.setBold(True)
                font.setFamily("Segoe UI")
            
            painter.setFont(font)
            
            shadow_rect = self.rect().adjusted(2, 2, 2, 2)
            painter.drawText(shadow_rect, Qt.AlignmentFlag.AlignCenter, self.feedback_text)
            
            # Main text with Windows-appropriate color
            if platform.system() == "Windows":
                painter.setPen(QColor(255, 255, 255))
            else:
                # Original gradient for non-Windows
                text_gradient = QLinearGradient(0, 0, 0, font.pointSize() * 2)
                text_gradient.setColorAt(0, QColor(255, 255, 255))
                text_gradient.setColorAt(1, QColor(200, 200, 200))
                painter.setPen(QColor("white"))
            
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.feedback_text)

        # Important: Call superclass paintEvent if needed, though maybe not for fully custom painting
        # super().paintEvent(event)

    @pyqtSlot(QImage)
    def setImage(self, image):
        # Scale the incoming image to the fixed size first
        scaled_image = image.scaled(self.fixed_camera_feed_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        pixmap = QPixmap.fromImage(scaled_image)
        # Set the fixed-size pixmap on the label
        self.image_label.setPixmap(pixmap)
        # The label's alignment property will handle centering if the label is larger
        self.update() # Trigger repaint for feedback overlay

    # Update slot signature
    @pyqtSlot(str, bool, int)
    def handleGesture(self, gesture, is_fist_closed, intensity):
        print(f"Overlay received gesture: {gesture}, Fist Closed: {is_fist_closed}, Intensity: {intensity}")

        # Update timing for mode indicator
        self.last_gesture_time = time.time()
        
        # Record gesture information for visualization
        self.gesture_direction = gesture
        self.gesture_intensity = intensity
        
        # Update mode indicator
        hand_icon = "✊" if is_fist_closed else "🖐️"
        direction_icon = "←" if gesture == "left" else "→"
        self.mode_label.setText(f"{hand_icon} {direction_icon}")
        
        # Create particle effects
        self.create_particles(gesture, intensity)
        
        # Add enhanced trail points
        center = self.rect().center()
        if gesture == "left":
            # Create trail points from right to left with curve
            self.gesture_trail = []
            for i in range(self.gesture_trail_max):
                progress = i / (self.gesture_trail_max - 1)
                x_offset = 150 - (progress * 300)  # Right to left
                y_offset = math.sin(progress * math.pi) * 30  # Curved motion
                point = QPoint(
                    center.x() + int(x_offset), 
                    center.y() + int(y_offset)
                )
                self.gesture_trail.append(point)
        elif gesture == "right":
            # Create trail points from left to right with curve
            self.gesture_trail = []
            for i in range(self.gesture_trail_max):
                progress = i / (self.gesture_trail_max - 1)
                x_offset = -150 + (progress * 300)  # Left to right
                y_offset = math.sin(progress * math.pi) * 30  # Curved motion
                point = QPoint(
                    center.x() + int(x_offset), 
                    center.y() + int(y_offset)
                )
                self.gesture_trail.append(point)

        # Play enhanced animation
        self.animate_gesture_response(gesture, intensity)
                
        current_os = platform.system().lower()
        action_taken = False
        action_type = ""

        enable_window_switch = self.settings.value("enable_window_switch", True, type=bool)
        enable_desktop_switch = self.settings.value("enable_desktop_switch", True, type=bool)

        if gesture == 'left':
            if is_fist_closed and enable_window_switch:
                action_type = "🪟 Window Switch (Prev)"
                if current_os == "windows": pyautogui.hotkey('alt', 'shift', 'tab')
                elif current_os == "darwin": pyautogui.hotkey('command', 'shift', 'tab')
                else: pyautogui.hotkey('alt', 'shift', 'tab')
                action_taken = True
            elif not is_fist_closed and enable_desktop_switch:
                action_type = "🖥️ Desktop Switch (Prev)"
                if current_os == "windows": pyautogui.hotkey('ctrl', 'win', 'left')
                elif current_os == "darwin": pyautogui.hotkey('ctrl', 'left')
                # Add Linux desktop switching if needed
                action_taken = True

        elif gesture == 'right':
            if is_fist_closed and enable_window_switch:
                action_type = "🪟 Window Switch (Next)"
                if current_os == "windows": pyautogui.hotkey('alt', 'tab')
                elif current_os == "darwin": pyautogui.hotkey('command', 'tab')
                else: pyautogui.hotkey('alt', 'tab')
                action_taken = True
            elif not is_fist_closed and enable_desktop_switch:
                action_type = "🖥️ Desktop Switch (Next)"
                if current_os == "windows": pyautogui.hotkey('ctrl', 'win', 'right')
                elif current_os == "darwin": pyautogui.hotkey('ctrl', 'right')
                # Add Linux desktop switching if needed
                action_taken = True

        if action_taken:
            print(f"Action taken: {action_type}")
            self.show_feedback(gesture, is_fist_closed, action_type, intensity)
        else:
            print("Gesture detected but no action configured/enabled for this state.")
            # Show feedback even if no action taken
            self.show_feedback(gesture, is_fist_closed, "No Action", intensity)
            
    def animate_gesture_response(self, direction, intensity):
        """Create a slight push animation in the direction of the gesture"""
        # Scale the animation based on intensity (subtle even at max intensity)
        scale_factor = intensity / 400  # 100 intensity = 25% movement
        current_geometry = self.geometry()
        
        if direction == "left":
            # Move slightly left then back
            offset = int(self.width() * scale_factor)
            end_geometry = QRect(
                current_geometry.x() - offset,
                current_geometry.y(),
                current_geometry.width(),
                current_geometry.height()
            )
        else:  # right
            # Move slightly right then back
            offset = int(self.width() * scale_factor)
            end_geometry = QRect(
                current_geometry.x() + offset,
                current_geometry.y(),
                current_geometry.width(),
                current_geometry.height()
            )
            
        # Setup the animation
        self.gesture_animation.stop()
        self.gesture_animation.setStartValue(current_geometry)
        self.gesture_animation.setEndValue(end_geometry)
        self.gesture_animation.start()
        
        # Setup return animation after a short delay
        QTimer.singleShot(200, lambda: self.animate_return_to_position(current_geometry))
        
    def animate_return_to_position(self, original_geometry):
        """Return window to original position"""
        self.gesture_animation.stop()
        self.gesture_animation.setStartValue(self.geometry())
        self.gesture_animation.setEndValue(original_geometry)
        self.gesture_animation.setEasingCurve(QEasingCurve.Type.OutBounce)
        self.gesture_animation.start()
        # Reset easing curve for next gesture
        QTimer.singleShot(300, lambda: self.gesture_animation.setEasingCurve(QEasingCurve.Type.OutBack))

    # Update feedback method with enhanced text effects
    def show_feedback(self, gesture_direction, is_fist_closed, action_type, intensity):
        show_feedback = self.settings.value("show_feedback", True, type=bool)
        if not show_feedback:
            return
            
        # Color based on fist state: Green for open, Blue for closed
        self.feedback_color = QColor(0, 255, 0, 200) if not is_fist_closed else QColor(0, 150, 255, 200)
        arrow = "←" if gesture_direction == 'left' else "→"
        fist_icon = "✊" if is_fist_closed else "🖐️"
        
        # Enhanced feedback text with emojis and better formatting
        intensity_bar = "▓" * (intensity // 20) + "░" * (5 - intensity // 20)
        self.feedback_text = f"{arrow} {fist_icon}\n{action_type}\n⚡ {intensity_bar} {intensity}%"
        
        # Animate text scale
        self.text_scale = 1.2
        self.text_animation.stop()
        self.text_animation = QPropertyAnimation()
        self.text_animation.finished.connect(lambda: setattr(self, 'text_scale', 1.0))
        
        # Custom animation for text scale
        self.animate_text_scale()

        # Use update() to trigger paintEvent instead of setStyleSheet
        self.update()
        self.feedback_timer.start(800) # Show feedback for longer
    
    def animate_text_scale(self):
        """Animate text scaling effect"""
        def update_scale(value):
            self.text_scale = value
            self.update()
        
        # Create scaling animation
        scale_values = [1.0, 1.2, 1.0]
        duration = 200
        
        for i, scale in enumerate(scale_values[1:], 1):
            QTimer.singleShot(i * duration // len(scale_values), 
                            lambda s=scale: update_scale(s))

    def hide_feedback(self):
        self.feedback_color = QColor("transparent")
        self.feedback_text = ""
        self.gesture_trail = []  # Clear the trail
        self.gesture_direction = None
        self.particles = []  # Clear particles
        self.text_scale = 1.0
        self.update() # Trigger repaint

    def restart_tracking(self):
        if self.tracking_thread and self.tracking_thread.isRunning():
            print("Stopping existing tracking thread...")
            self.tracking_thread.stop()
            print("Existing tracking thread stopped.")

        # Ensure settings are fresh before starting
        self.settings.sync()
        camera_index = self.settings.value("camera_index", 0, type=int)
        sensitivity = self.settings.value("sensitivity", 50, type=int)

        print(f"Restarting tracking with Camera: {camera_index}, Sensitivity: {sensitivity}")
        self.tracking_thread = HandTrackingThread(
            camera_index=camera_index,
            sensitivity=sensitivity,
            sound_manager=self.sound_manager
        )
        self.tracking_thread.changePixmap.connect(self.setImage)
        self.tracking_thread.gestureDetected.connect(self.handleGesture)
        self.tracking_thread.cameraError.connect(self.handle_camera_error) # Connect error signal
        self.tracking_thread.mouseModeChanged.connect(self.handleMouseModeChange)
        self.tracking_thread.mouseAction.connect(self.handleMouseAction)
        self.tracking_thread.start()
        print("Hand tracking thread started/restarted.")

    @pyqtSlot(str)
    def handle_camera_error(self, message):
         # Show error message to the user
         QMessageBox.warning(self, "Camera Error", message)
         # Optionally hide overlay or show error state
         self.image_label.setText("Camera Error!")
         self.image_label.setStyleSheet("background-color: rgba(150, 0, 0, 0.7); color: white; font-size: 18px; border-radius: 10px; padding: 10px;")

    @pyqtSlot(bool)
    def handleMouseModeChange(self, is_mouse_mode):
        """Handle switching between swoosh and mouse modes"""
        try:
            if is_mouse_mode:
                self.mode_label.setText("🖱️ Mouse Mode")
                self.mode_label.setStyleSheet("""
                    background-color: rgba(0, 100, 255, 200);
                    color: white;
                    border-radius: 15px;
                    padding: 8px 16px;
                    font-size: 14px;
                    font-weight: bold;
                """)
                self.show_temporary_feedback("🖱️ Mouse Control Active", 2000)
            else:
                self.mode_label.setText("🖐️ Gesture Mode")
                self.mode_label.setStyleSheet("""
                    background-color: rgba(0, 200, 100, 200);
                    color: white;
                    border-radius: 15px;
                    padding: 8px 16px;
                    font-size: 14px;
                    font-weight: bold;
                """)
                self.show_temporary_feedback("🖐️ Gesture Control Active", 2000)
        except Exception as e:
            print(f"Error handling mouse mode change: {e}")

    @pyqtSlot(str, dict)
    def handleMouseAction(self, action_type, params):
        """Handle mouse actions from gesture detection"""
        try:
            if action_type == "move":
                # Move cursor to specified position
                x, y = params.get("x", 0), params.get("y", 0)
                pyautogui.moveTo(x, y)
                
            elif action_type == "click":
                # Perform mouse click
                pyautogui.click()
                self.show_temporary_feedback("🖱️ Click", 500)
                
            elif action_type == "drag_start":
                # Start dragging
                pyautogui.mouseDown()
                self.show_temporary_feedback("🔄 Drag Start", 500)
                
            elif action_type == "drag_end":
                # End dragging
                pyautogui.mouseUp()
                self.show_temporary_feedback("🔄 Drag End", 500)
                
            elif action_type == "scroll":
                # Scroll action
                scroll_amount = params.get("amount", 0)
                pyautogui.scroll(scroll_amount)
                direction = "↑" if scroll_amount > 0 else "↓"
                self.show_temporary_feedback(f"📜 Scroll {direction}", 300)
                
        except Exception as e:
            print(f"Error handling mouse action {action_type}: {e}")

    def show_temporary_feedback(self, message, duration_ms):
        """Show temporary feedback message"""
        try:
            self.feedback_text = message
            self.feedback_color = QColor(100, 150, 255, 180)
            self.text_scale = 1.1
            self.update()
            QTimer.singleShot(duration_ms, self.hide_feedback)
        except Exception as e:
            print(f"Error showing temporary feedback: {e}")


    def closeEvent(self, event):
        print("Closing overlay window...")
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
        super().closeEvent(event)

    # Allow moving the overlay
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_start_position)
            event.accept()


# --- Main Application Logic ---
class SwooshApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setApplicationName(APP_NAME)
        self.setOrganizationName(ORG_NAME)
        self.setApplicationVersion(VERSION)
        self.setQuitOnLastWindowClosed(False)
        
        # Apply Windows-native styling
        apply_windows_native_style(self)

        # Initialize sound manager first
        global sound_manager
        sound_manager = SoundManager()
        
        # Load settings before creating windows that depend on them
        self.settings = QSettings() # Access global settings

        self.overlay = OverlayWindow(sound_manager)
        self.settings_window = SettingsWindow(sound_manager)
        
        # Connect signal from settings window to overlay's restart method
        self.settings_window.settingsChanged.connect(self.handle_settings_change)

        # Connect overlay's request signal to show settings
        self.overlay.requestShowSettings.connect(self.show_settings)

        # Create enhanced system tray icon with Windows-native styling
        self.tray_icon = QSystemTrayIcon(self)
        
        # Try to load custom icon, fallback to system icon
        icon_path = os.path.join(os.path.dirname(__file__), 'swoosh_icon.ico')
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            print(f"✅ Loaded custom icon: {icon_path}")
        else:
            # Create icon if it doesn't exist
            try:
                import subprocess
                subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'create_icon.py')], 
                             capture_output=True, timeout=10)
                if os.path.exists(icon_path):
                    icon = QIcon(icon_path)
                    print(f"✅ Created and loaded custom icon: {icon_path}")
                else:
                    raise FileNotFoundError("Icon creation failed")
            except:
                # Fallback to system icon
                if platform.system() == "Windows":
                    icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
                else:
                    icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
                print("⚠️ Using system default icon")
        
        self.tray_icon.setIcon(icon)
        self.settings_window.setWindowIcon(icon)
        self.overlay.setWindowIcon(icon)
        self.tray_icon.setToolTip(f"{APP_NAME} v{VERSION} - Hand Gesture Control")

        # Create Windows-style context menu
        tray_menu = QMenu()
        
        if platform.system() == "Windows":
            tray_menu.setStyleSheet("""
                QMenu {
                    background-color: #f0f0f0;
                    border: 1px solid #c0c0c0;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                }
                QMenu::item {
                    padding: 8px 24px 8px 32px;
                    color: black;
                }
                QMenu::item:selected {
                    background-color: #e5f3ff;
                    color: black;
                }
                QMenu::item:disabled {
                    color: #6d6d6d;
                }
                QMenu::separator {
                    height: 1px;
                    background-color: #d0d0d0;
                    margin: 4px 0px;
                }
            """)
        
        # Main actions
        show_action = QAction("🎥 Show/Hide Overlay", self)
        show_action.triggered.connect(self.toggle_overlay)
        tray_menu.addAction(show_action)

        settings_action = QAction("⚙️ Settings", self)
        settings_action.triggered.connect(self.show_settings)
        tray_menu.addAction(settings_action)
        
        tray_menu.addSeparator()
        
        # Quick toggles
        sound_action = QAction("🔊 Toggle Sound", self)
        sound_action.triggered.connect(self.toggle_sound)
        tray_menu.addAction(sound_action)
        
        test_action = QAction("🧪 Test Sounds", self)
        test_action.triggered.connect(self.test_sounds)
        tray_menu.addAction(test_action)

        tray_menu.addSeparator()

        # About and quit
        about_action = QAction(f"ℹ️ About {APP_NAME}", self)
        about_action.triggered.connect(self.show_about)
        tray_menu.addAction(about_action)
        
        quit_action = QAction("❌ Quit", self)
        quit_action.triggered.connect(self.quit_app)
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        self.tray_icon.activated.connect(self.on_tray_icon_activated)
        
        # Show startup notification with Windows styling
        if self.tray_icon.supportsMessages():
            self.tray_icon.showMessage(
                f"{APP_NAME} Started",
                "Hand gesture control is ready! Use Ctrl+Shift+A to toggle overlay.",
                QSystemTrayIcon.MessageIcon.Information,
                4000  # Show for 4 seconds
            )

    def handle_settings_change(self):
        # Restart tracking only if overlay is currently visible
        if self.overlay.isVisible():
            print("Settings changed, restarting tracking...")
            self.overlay.restart_tracking()
        else:
            print("Settings changed, will apply when overlay is shown.")

    def toggle_overlay(self):
        global overlay_visible
        if self.overlay.isVisible():
            self.overlay.hide()
            overlay_visible = False
            # Stop camera when hidden to save resources
            if self.overlay.tracking_thread and self.overlay.tracking_thread.isRunning():
                 print("Stopping tracking thread as overlay is hidden.")
                 self.overlay.tracking_thread.stop()
            print("Overlay hidden.")
            
            # Show tray notification
            if self.tray_icon.supportsMessages():
                self.tray_icon.showMessage(
                    "Overlay Hidden",
                    "Hand gesture detection paused. Click tray icon to reactivate.",
                    QSystemTrayIcon.MessageIcon.Information,
                    2000
                )
        else:
            # Restart camera/tracking when shown
            print("Showing overlay and starting tracking...")
            # Ensure overlay is positioned correctly before showing
            self.overlay.setGeometry(self.overlay.default_pos.x(), self.overlay.default_pos.y(), self.overlay.width(), self.overlay.height())
            self.overlay.show()
            self.overlay.restart_tracking() # Start/Restart tracking
            overlay_visible = True
            print("Overlay shown.")

    def toggle_sound(self):
        """Toggle sound effects on/off"""
        if sound_manager:
            current_state = sound_manager.enabled
            sound_manager.set_enabled(not current_state)
            self.settings.setValue("sound_enabled", not current_state)
            
            status = "enabled" if not current_state else "disabled"
            if self.tray_icon.supportsMessages():
                self.tray_icon.showMessage(
                    "Sound Effects",
                    f"Sound effects {status}",
                    QSystemTrayIcon.MessageIcon.Information,
                    1500
                )
    
    def test_sounds(self):
        """Test sound effects"""
        if sound_manager and sound_manager.enabled:
            # Play sounds in sequence
            QTimer.singleShot(0, lambda: sound_manager.play_sound('swoosh_left'))
            QTimer.singleShot(400, lambda: sound_manager.play_sound('swoosh_right'))
            QTimer.singleShot(800, lambda: sound_manager.play_sound('page_flip'))
        else:
            if self.tray_icon.supportsMessages():
                self.tray_icon.showMessage(
                    "Sound Test",
                    "Sound effects are disabled",
                    QSystemTrayIcon.MessageIcon.Warning,
                    2000
                )

    def show_settings(self):
        # Refresh camera list in case hardware changed since app start
        self.settings_window.populate_camera_list()
        self.settings_window.load_settings() # Reload settings into UI elements
        self.settings_window.show()
        self.settings_window.activateWindow() # Bring to front
    
    def show_about(self):
        """Show about dialog with Windows-native styling"""
        about_text = f"""
        <h2>🚀 {APP_NAME} v{VERSION}</h2>
        <p><b>The Ultimate Hand Gesture Desktop Control</b></p>
        <p>Control your desktop with simple hand gestures!</p>
        <ul>
        <li>✊ Fist = Window switching</li>
        <li>🖐️ Open hand = Desktop switching</li>
        <li>⬅️ Left swipe = Previous</li>
        <li>➡️ Right swipe = Next</li>
        </ul>
        <p><small>Built with PyQt6, OpenCV, and MediaPipe</small></p>
        """
        
        # Create a custom message box for Windows
        msg_box = QMessageBox()
        msg_box.setWindowTitle(f"About {APP_NAME}")
        msg_box.setText(about_text)
        msg_box.setIconPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon).pixmap(64, 64))
        
        if platform.system() == "Windows":
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                }
                QMessageBox QLabel {
                    color: black;
                    padding: 16px;
                }
                QMessageBox QPushButton {
                    background-color: #e1e1e1;
                    border: 1px solid #adadad;
                    border-radius: 4px;
                    padding: 6px 20px;
                    font-weight: normal;
                    min-width: 75px;
                }
                QMessageBox QPushButton:hover {
                    background-color: #e5f3ff;
                    border-color: #0078d4;
                }
                QMessageBox QPushButton:pressed {
                    background-color: #cce4f7;
                }
            """)
        
        msg_box.exec()

    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger: # Left click
             self.toggle_overlay()

    def quit_app(self):
        global listener
        print("Quitting application...")
        if listener:
            print("Stopping hotkey listener...")
            listener.stop()
        if self.overlay.tracking_thread and self.overlay.tracking_thread.isRunning():
            print("Stopping tracking thread...")
            self.overlay.tracking_thread.stop()
        
        # Cleanup sound manager
        if sound_manager:
            try:
                pygame.mixer.quit()
            except:
                pass
                
        self.tray_icon.hide()
        print("Exiting.")
        self.quit()


# --- Keyboard Shortcut Handling ---
def on_activate_shortcut():
    print("Shortcut activated!")
    # Use QTimer.singleShot for thread safety with Qt GUI
    # Check if 'app' exists and is initialized before calling toggle_overlay
    if 'app' in globals() and isinstance(app, SwooshApp):
        QTimer.singleShot(0, app.toggle_overlay)
    else:
        print("Error: App object not ready for shortcut.")


def setup_hotkey_listener():
    global listener
    shortcut_str = '<ctrl>+<shift>+a'
    try:
        listener = keyboard.GlobalHotKeys({
            shortcut_str: on_activate_shortcut
        })
        listener.start()
        print(f"Hotkey listener started for {shortcut_str}.")
    except Exception as e:
        # More specific error handling might be needed depending on the exception
        # e.g., permissions errors on Linux/macOS, conflicts on Windows
        error_msg = f"Error setting up hotkey ({shortcut_str}): {e}\n"
        error_msg += "Ensure the key combination isn't already globally registered by another application.\n"
        if platform.system() != "Windows":
             error_msg += "On Linux/macOS, you might need specific permissions to monitor global hotkeys."
        print(error_msg)
        # Optionally show a message box to the user
        QTimer.singleShot(0, lambda: QMessageBox.critical(None, "Hotkey Error", error_msg))
        listener = None # Ensure listener is None if setup failed


# --- Entry Point ---
if __name__ == '__main__':
    print(f"🚀 Starting {APP_NAME} v{VERSION}...")
    
    # Set Windows-specific attributes before creating app
    if platform.system() == "Windows":
        # Enable high DPI support for Windows 10/11 (PyQt6 syntax)
        try:
            QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        except AttributeError:
            pass  # Not available in all PyQt6 versions
    
    app = SwooshApp(sys.argv)

    # Start the hotkey listener
    setup_hotkey_listener()

    # Show welcome message
    print(f"✅ {APP_NAME} started successfully!")
    print("💡 Use Ctrl+Shift+A to toggle the camera overlay")
    print("💡 Right-click the system tray icon for more options")

    # Initial check: Show overlay immediately on start? Or wait for hotkey/tray?
    # Let's wait for user action by default, but can be configured
    settings = QSettings(ORG_NAME, APP_NAME)
    show_on_startup = settings.value("show_on_startup", False, type=bool)
    if show_on_startup:
        app.toggle_overlay()

    sys.exit(app.exec())
