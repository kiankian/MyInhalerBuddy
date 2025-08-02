import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time
from abc import ABC, abstractmethod

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Drawing utility
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Correct MediaPipe lip landmark indices
# Upper lip (outer boundary)
UPPER_LIP = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
# Lower lip (outer boundary) 
LOWER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
# Complete lip outline (both upper and lower)
LIP_OUTLINE = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415]

class InhalerTracker:
    """Tracks inhaler position over time for persistence during motion"""
    
    def __init__(self, max_history=10, max_gap_frames=15):
        self.history = deque(maxlen=max_history)  # Store recent detections
        self.last_detection_time = 0
        self.max_gap_frames = max_gap_frames
        self.frame_count = 0
        self.prediction_center = None
        self.prediction_radius = 150  # Search radius around predicted position
        
    def update(self, detections):
        """Update tracker with new detections"""
        self.frame_count += 1
        current_time = time.time()
        
        if detections:
            # Find the best detection (closest to prediction if we have one)
            best_detection = self._select_best_detection(detections)
            
            if best_detection is not None:
                # Add to history
                x, y, w, h = cv2.boundingRect(best_detection)
                center = (x + w//2, y + h//2)
                self.history.append({
                    'center': center,
                    'bbox': (x, y, w, h),
                    'contour': best_detection,
                    'frame': self.frame_count,
                    'time': current_time
                })
                self.last_detection_time = current_time
                
                # Update prediction for next frame
                self._update_prediction()
                
                return [best_detection]  # Return single best detection
        
        # No detections this frame
        # Check if we should keep searching based on recent history
        if self._should_keep_searching():
            self._update_prediction()  # Update prediction even without detection
            return []  # Return empty but keep tracking
        else:
            # Lost track completely
            self.prediction_center = None
            return []
    
    def _select_best_detection(self, detections):
        """Select best detection based on prediction and history"""
        if not detections:
            return None
            
        if self.prediction_center is None:
            # No prediction, return largest detection
            return max(detections, key=cv2.contourArea)
        
        # Find detection closest to predicted position
        best_detection = None
        min_distance = float('inf')
        
        for detection in detections:
            x, y, w, h = cv2.boundingRect(detection)
            center = (x + w//2, y + h//2)
            
            distance = math.sqrt(
                (center[0] - self.prediction_center[0])**2 + 
                (center[1] - self.prediction_center[1])**2
            )
            
            if distance < min_distance and distance < self.prediction_radius:
                min_distance = distance
                best_detection = detection
        
        # If no detection within prediction radius, take closest one
        if best_detection is None and detections:
            best_detection = min(detections, key=lambda det: self._distance_to_prediction(det))
            
        return best_detection
    
    def _distance_to_prediction(self, detection):
        """Calculate distance from detection to prediction"""
        if self.prediction_center is None:
            return 0
        
        x, y, w, h = cv2.boundingRect(detection)
        center = (x + w//2, y + h//2)
        
        return math.sqrt(
            (center[0] - self.prediction_center[0])**2 + 
            (center[1] - self.prediction_center[1])**2
        )
    
    def _should_keep_searching(self):
        """Determine if we should keep tracking despite no current detection"""
        if not self.history:
            return False
            
        frames_since_last = self.frame_count - self.history[-1]['frame']
        return frames_since_last <= self.max_gap_frames
    
    def _update_prediction(self):
        """Update prediction of where inhaler should be next"""
        if len(self.history) < 2:
            if self.history:
                self.prediction_center = self.history[-1]['center']
            return
        
        # Simple motion prediction based on recent movement
        recent_positions = [h['center'] for h in list(self.history)[-3:]]
        
        if len(recent_positions) >= 2:
            # Calculate velocity
            vel_x = recent_positions[-1][0] - recent_positions[-2][0]
            vel_y = recent_positions[-1][1] - recent_positions[-2][1]
            
            # Predict next position
            next_x = recent_positions[-1][0] + vel_x
            next_y = recent_positions[-1][1] + vel_y
            
            self.prediction_center = (int(next_x), int(next_y))
        else:
            self.prediction_center = recent_positions[-1]
    
    def get_search_region(self):
        """Get region to prioritize in next detection"""
        if self.prediction_center is None:
            return None
        
        x, y = self.prediction_center
        r = self.prediction_radius
        
        return {
            'center': self.prediction_center,
            'radius': r,
            'bbox': (max(0, x-r), max(0, y-r), 2*r, 2*r)
        }
    
    def get_smoothed_position(self):
        """Get smoothed position based on recent history"""
        if not self.history:
            return None
            
        # Simple averaging of recent positions
        recent_centers = [h['center'] for h in list(self.history)[-3:]]
        
        avg_x = sum(c[0] for c in recent_centers) // len(recent_centers)
        avg_y = sum(c[1] for c in recent_centers) // len(recent_centers)
        
        return (avg_x, avg_y)

def detect_red_inhaler(frame, tracker=None):
    """Detect red inhaler using HSV color filtering with improved shape analysis and tracking"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Adjust detection sensitivity based on tracker state
    if tracker and tracker.get_search_region():
        # We're tracking - relax constraints slightly for motion blur
        lower_red1 = np.array([0, 120, 80])   # Slightly more permissive
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 80])
        upper_red2 = np.array([180, 255, 255])
        min_area = 1500  # Lower minimum area for motion blur
        max_area = 60000  # Higher maximum area
    else:
        # Initial detection - stricter constraints
        lower_red1 = np.array([0, 140, 100])
        upper_red1 = np.array([8, 255, 255])
        lower_red2 = np.array([172, 140, 100])
        upper_red2 = np.array([180, 255, 255])
        min_area = 2000
        max_area = 50000
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine masks
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small noise
    kernel_small = np.ones((2,2), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Prioritize contours near predicted position if tracking
    search_region = tracker.get_search_region() if tracker else None
    
    # Filter contours with more specific inhaler characteristics
    inhaler_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Adjust area constraints based on tracking state
        if min_area < area < max_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Slightly more permissive aspect ratio when tracking
            min_ratio = 0.3 if (tracker and tracker.get_search_region()) else 0.4
            max_ratio = 3.0 if (tracker and tracker.get_search_region()) else 2.5
            
            if min_ratio < aspect_ratio < max_ratio:
                
                # Relax shape analysis when tracking (motion blur affects these metrics)
                if tracker and tracker.get_search_region():
                    # During tracking, prioritize based on proximity to prediction
                    center = (x + w//2, y + h//2)
                    if search_region:
                        pred_center = search_region['center']
                        distance = math.sqrt(
                            (center[0] - pred_center[0])**2 + 
                            (center[1] - pred_center[1])**2
                        )
                        # Accept if within search radius
                        if distance <= search_region['radius']:
                            inhaler_contours.append(contour)
                            continue
                
                # Full shape analysis for new detections or when not tracking
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                    
                    # More permissive circularity when tracking
                    max_circularity = 0.9 if (tracker and tracker.get_search_region()) else 0.85
                    
                    if circularity < max_circularity:
                        
                        # Check if contour is reasonably solid/filled
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            solidity = area / hull_area
                            
                            # More permissive solidity when tracking
                            min_solidity = 0.5 if (tracker and tracker.get_search_region()) else 0.6
                            
                            if solidity > min_solidity:
                                
                                # Check extent
                                rect_area = w * h
                                if rect_area > 0:
                                    extent = area / rect_area
                                    
                                    # More permissive extent when tracking
                                    min_extent = 0.3 if (tracker and tracker.get_search_region()) else 0.4
                                    
                                    if extent > min_extent:
                                        inhaler_contours.append(contour)
    
    return inhaler_contours, red_mask

def draw_inhaler_detection(frame, inhaler_contours):
    """Draw bounding boxes and labels for detected inhalers"""
    for contour in inhaler_contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw bounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Draw contour outline
        cv2.drawContours(frame, [contour], -1, (255, 0, 255), 2)
        
        # Add label
        label = "INHALER DETECTED"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), (0, 255, 0), -1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add center point
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

def calculate_inhaler_distance_to_lips(inhaler_contours, face_landmarks, img_width, img_height):
    """Calculate distance between inhaler and lips"""
    if not inhaler_contours or not face_landmarks:
        return None
    
    # Get lip center (approximate)
    lip_points = []
    for idx in UPPER_LIP + LOWER_LIP:
        x = int(face_landmarks.landmark[idx].x * img_width)
        y = int(face_landmarks.landmark[idx].y * img_height)
        lip_points.append((x, y))
    
    if not lip_points:
        return None
    
    # Calculate lip center
    lip_center_x = sum(p[0] for p in lip_points) // len(lip_points)
    lip_center_y = sum(p[1] for p in lip_points) // len(lip_points)
    
    # Find closest inhaler
    min_distance = float('inf')
    closest_inhaler = None
    
    for contour in inhaler_contours:
        x, y, w, h = cv2.boundingRect(contour)
        inhaler_center_x = x + w // 2
        inhaler_center_y = y + h // 2
        
        # Calculate distance
        distance = math.sqrt((inhaler_center_x - lip_center_x)**2 + (inhaler_center_y - lip_center_y)**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_inhaler = (inhaler_center_x, inhaler_center_y)
    
    return min_distance, (lip_center_x, lip_center_y), closest_inhaler

def calculate_head_pitch(landmarks, img_width, img_height):
    """Calculate head pitch using simple geometric approach - 90° = straight ahead"""
    
    # Get key landmark points
    nose_tip = landmarks[1]      # Nose tip
    nose_bridge = landmarks[6]   # Nose bridge (between eyes)
    chin = landmarks[152]        # Chin
    
    # Convert to pixel coordinates
    nose_tip_y = nose_tip.y * img_height
    nose_bridge_y = nose_bridge.y * img_height
    chin_y = chin.y * img_height
    
    # Calculate the angle of the nose line relative to vertical
    # Nose line goes from nose bridge to nose tip
    nose_line_length = abs(nose_tip_y - nose_bridge_y)
    
    # Calculate the deviation from expected vertical alignment
    # When looking straight ahead, nose should point straight down from bridge
    if nose_line_length > 0:
        # Simple angle calculation based on nose direction
        # Use the relationship between nose bridge and chin for reference
        bridge_to_chin = chin_y - nose_bridge_y
        bridge_to_nose = nose_tip_y - nose_bridge_y
        if bridge_to_chin > 0:
            # Calculate ratio - when looking straight, nose tip should be 
            # about 35% of the way from bridge to chin (adjusted for straight ahead)
            expected_ratio = 0.35
            actual_ratio = bridge_to_nose / bridge_to_chin
            
            # Convert ratio difference to angle
            ratio_diff = actual_ratio - expected_ratio
            
            # Convert to degrees (90 = straight ahead)
            # REVERSED: ratio > expected = looking down (lower angle)
            #          ratio < expected = looking up (higher angle)
            pitch = 90 - (ratio_diff * 100)  # Reversed sign and scale by 100
        else:
            pitch = 90
    else:
        pitch = 90
    
    # Clamp to reasonable range
    pitch = max(30, min(150, pitch))
    
    return pitch

def draw_lips(image, face_landmarks, img_width, img_height):
    """Draw highlighted lips using MediaPipe's lip connections"""
    
    # Create an overlay for lip highlighting
    lip_overlay = image.copy()
    
    # Draw lip connections with thick lines for highlighting
    mp_drawing.draw_landmarks(
        image=lip_overlay,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_LIPS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3)
    )
    
    # Blend the lip overlay with the main image (more transparent)
    alpha = 0.3
    cv2.addWeighted(lip_overlay, alpha, image, 1 - alpha, 0, image)
    
    # Draw upper lip landmark points
    for idx in UPPER_LIP:
        x = int(face_landmarks.landmark[idx].x * img_width)
        y = int(face_landmarks.landmark[idx].y * img_height)
        cv2.circle(image, (x, y), 3, (0, 255, 255), -1)  # Yellow dots for upper lip
    
    # Draw lower lip landmark points  
    for idx in LOWER_LIP:
        x = int(face_landmarks.landmark[idx].x * img_width)
        y = int(face_landmarks.landmark[idx].y * img_height)
        cv2.circle(image, (x, y), 3, (255, 255, 0), -1)  # Cyan dots for lower lip

# ===== MODE SYSTEM =====

class AppMode(ABC):
    """Abstract base class for application modes"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_active = False
        
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return the modified frame"""
        pass
    
    @abstractmethod
    def handle_key(self, key: int) -> bool:
        """Handle key press. Return True if key was handled, False otherwise"""
        pass
    
    @abstractmethod
    def get_instructions(self) -> list:
        """Return list of instruction strings to display"""
        pass
    
    def on_activate(self):
        """Called when mode becomes active"""
        self.is_active = True
        
    def on_deactivate(self):
        """Called when mode becomes inactive"""
        self.is_active = False

class BaseProcessingMode(AppMode):
    """Base class with common processing functionality that most modes share"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.inhaler_tracker = InhalerTracker(max_history=10, max_gap_frames=15)
        self.show_mask = False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Common frame processing logic"""
        img_height, img_width = frame.shape[:2]
        
        # Detect red inhaler with tracking
        raw_detections, red_mask = detect_red_inhaler(frame, self.inhaler_tracker)
        
        # Update tracker and get filtered detections
        inhaler_contours = self.inhaler_tracker.update(raw_detections)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Initialize distance info
        distance_info = None
        pitch = None
        
        # Process face landmarks if detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Create transparent overlay for face mesh
                overlay = frame.copy()
                
                # Draw face mesh on overlay
                mp_drawing.draw_landmarks(
                    image=overlay,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec,
                )
                
                # Blend overlay with original frame (30% mesh opacity)
                alpha = 0.3
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                # Draw highlighted lips on top of transparent mesh
                draw_lips(frame, face_landmarks, img_width, img_height)
                
                # Calculate and display head pitch
                pitch = calculate_head_pitch(face_landmarks.landmark, img_width, img_height)
                
                # Allow modes to add custom head pitch processing
                frame = self.process_head_pitch(frame, pitch, img_width, img_height)
                
                # Calculate distance between inhaler and lips
                distance_info = calculate_inhaler_distance_to_lips(inhaler_contours, face_landmarks, img_width, img_height)
                
                # Display pitch information
                cv2.putText(frame, f"Head Pitch: {pitch:.1f}°", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw inhaler detection
        if inhaler_contours:
            draw_inhaler_detection(frame, inhaler_contours)
            
            # Display inhaler count
            cv2.putText(frame, f"Inhalers detected: {len(inhaler_contours)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display distance to lips if available
            if distance_info:
                distance, lip_center, inhaler_center = distance_info
                cv2.putText(frame, f"Distance to lips: {distance:.1f}px", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Draw line between lips and closest inhaler
                cv2.line(frame, lip_center, inhaler_center, (255, 255, 0), 2)
                
                # Provide guidance
                if distance < 100:
                    guidance = "GOOD POSITION!"
                    color = (0, 255, 0)
                elif distance < 200:
                    guidance = "Getting close..."
                    color = (0, 255, 255)
                else:
                    guidance = "Move inhaler closer to lips"
                    color = (0, 0, 255)
                
                cv2.putText(frame, guidance, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw tracking visualization
        search_region = self.inhaler_tracker.get_search_region()
        if search_region:
            # Draw prediction circle
            center = search_region['center']
            radius = min(search_region['radius'], 200)  # Cap visualization radius
            cv2.circle(frame, center, radius, (255, 255, 0), 2)  # Yellow prediction circle
            cv2.circle(frame, center, 5, (255, 255, 0), -1)  # Yellow center dot
            
            # Add tracking status
            tracking_status = f"Tracking (History: {len(self.inhaler_tracker.history)})"
            cv2.putText(frame, tracking_status, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Searching for inhaler...", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Allow modes to add custom processing
        frame = self.process_custom(frame, inhaler_contours, distance_info, pitch, img_width, img_height)

        # Show debug mask if enabled
        if self.show_mask:
            cv2.imshow("Debug - Red Mask", red_mask)
        
        return frame
    
    def process_head_pitch(self, frame: np.ndarray, pitch: float, img_width: int, img_height: int) -> np.ndarray:
        """Override this method to add custom head pitch processing"""
        return frame
    
    def process_custom(self, frame: np.ndarray, inhaler_contours, distance_info, pitch: float, img_width: int, img_height: int) -> np.ndarray:
        """Override this method to add custom processing at the end"""
        return frame
    
    def handle_key(self, key: int) -> bool:
        """Common key handling - can be overridden"""
        if key == ord('m'):
            self.show_mask = not self.show_mask
            if not self.show_mask:
                cv2.destroyWindow("Debug - Red Mask")
            return True
        return False

class FullMode(BaseProcessingMode):
    """Full functionality mode - original behavior"""
    
    def __init__(self):
        super().__init__("Full Mode")
    
    def get_instructions(self):
        return [
            "Full Mode: Complete inhaler detection + face tracking",
            "Press 'm' to toggle mask view, '1-8' to switch modes"
        ]

class HeadPositioningMode(BaseProcessingMode):
    """Head Positioning Mode - Focus on head position alerts"""
    
    def __init__(self):
        super().__init__("Head Positioning Mode")
    
    def process_head_pitch(self, frame: np.ndarray, pitch: float, img_width: int, img_height: int) -> np.ndarray:
        """Add head positioning alerts"""
        # HEAD POSITIONING ALERT - Check if pitch is in optimal range (89-95 degrees)
        if not (89 <= pitch <= 95):
            # Create prominent alert
            alert_text = "HEAD POSITION ALERT!"
            if pitch < 89:
                direction_text = "TILT HEAD UP"
                alert_color = (0, 0, 255)  # Red
            else:  # pitch > 95
                direction_text = "TILT HEAD DOWN"
                alert_color = (0, 0, 255)  # Red
            
            # Draw alert background rectangle
            alert_bg_rect = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            direction_bg_rect = cv2.getTextSize(direction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            max_width = max(alert_bg_rect[0], direction_bg_rect[0])
            
            # Alert background
            cv2.rectangle(frame, (img_width//2 - max_width//2 - 20, 10), 
                        (img_width//2 + max_width//2 + 20, 90), alert_color, -1)
            cv2.rectangle(frame, (img_width//2 - max_width//2 - 20, 10), 
                        (img_width//2 + max_width//2 + 20, 90), (255, 255, 255), 3)
            
            # Alert text
            cv2.putText(frame, alert_text, (img_width//2 - alert_bg_rect[0]//2, 35), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, direction_text, (img_width//2 - direction_bg_rect[0]//2, 65), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Good positioning feedback
            good_text = "HEAD POSITION: GOOD"
            good_bg_rect = cv2.getTextSize(good_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (img_width//2 - good_bg_rect[0]//2 - 15, 15), 
                        (img_width//2 + good_bg_rect[0]//2 + 15, 50), (0, 255, 0), -1)
            cv2.putText(frame, good_text, (img_width//2 - good_bg_rect[0]//2, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def get_instructions(self):
        return [
            "Head Positioning Mode: Alerts when head pitch is outside 89-95°",
            "Press 'm' to toggle mask view, '1-8' to switch modes"
        ]

class InhalerShakingMode(BaseProcessingMode):
    """Inhaler Shaking Mode - Focus on inhaler shaking alerts"""
    
    def __init__(self):
        super().__init__("Inhaler Shaking Mode")
        self.shake_threshold = 15  # Minimum movement speed to consider shaking
        self.direction_change_threshold = 3  # Number of direction changes needed
        self.shake_window_frames = 10  # Number of recent frames to analyze
        self.last_shake_time = 0  # Timestamp of last detected shaking
        self.shake_delay = 1.0  # Seconds to maintain "shaking" state after stopping
    
    def _detect_shaking(self):
        """Detect if inhaler is being shaken based on position history"""
        if len(self.inhaler_tracker.history) < 4:
            return False
        
        # Get recent positions (up to shake_window_frames)
        recent_history = list(self.inhaler_tracker.history)[-self.shake_window_frames:]
        positions = [h['center'] for h in recent_history]
        
        if len(positions) < 4:
            return False
        
        # Calculate velocities and direction changes
        velocities = []
        direction_changes = 0
        
        for i in range(1, len(positions)):
            # Calculate velocity (distance between consecutive positions)
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocity = math.sqrt(dx*dx + dy*dy)
            velocities.append(velocity)
            
            # Check for direction changes (oscillation detection)
            if len(velocities) >= 3:
                # Look for velocity direction changes in x and y
                if i >= 2:
                    prev_dx = positions[i-1][0] - positions[i-2][0]
                    curr_dx = positions[i][0] - positions[i-1][0]
                    prev_dy = positions[i-1][1] - positions[i-2][1]
                    curr_dy = positions[i][1] - positions[i-1][1]
                    
                    # Check if direction changed significantly in either axis
                    if (prev_dx * curr_dx < 0 and abs(curr_dx) > 5) or \
                       (prev_dy * curr_dy < 0 and abs(curr_dy) > 5):
                        direction_changes += 1
        
        # Determine if shaking based on:
        # 1. Sufficient movement speed
        # 2. Multiple direction changes (oscillation)
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0
        max_velocity = max(velocities) if velocities else 0
        
        is_shaking = (avg_velocity > self.shake_threshold or max_velocity > self.shake_threshold * 1.5) and \
                     direction_changes >= self.direction_change_threshold
        
        # Update last shake time if shaking is detected
        if is_shaking:
            self.last_shake_time = time.time()
        
        return is_shaking
    
    def _is_shaking_with_delay(self):
        """Determine if we should show 'shaking' state including delay period"""
        current_time = time.time()
        currently_shaking = self._detect_shaking()
        
        # If currently shaking, show as shaking
        if currently_shaking:
            return True
        
        # If not currently shaking, check if we're still in delay period
        time_since_last_shake = current_time - self.last_shake_time
        return time_since_last_shake < self.shake_delay
    
    def process_custom(self, frame: np.ndarray, inhaler_contours, distance_info, pitch: float, img_width: int, img_height: int) -> np.ndarray:
        """Add shaking detection and alert display"""
        if inhaler_contours:
            # Detect if inhaler is being shaken (with delay)
            is_shaking_display = self._is_shaking_with_delay()
            currently_shaking = self._detect_shaking()  # For debug info
            
            # Prepare alert text
            if is_shaking_display:
                alert_text = "INHALER SHAKING: GOOD"
                alert_color = (0, 255, 255)  # Yellow (BGR format)
                border_color = (255, 255, 255)  # White border
            else:
                alert_text = "SHAKE THE INHALER!"
                alert_color = (0, 0, 255)  # Red (BGR format)
                border_color = (255, 255, 255)  # White border
            
            # Calculate text size for proper box sizing
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]
            
            # Position alert box at top center
            box_width = text_size[0] + 40
            box_height = text_size[1] + 30
            box_x = (img_width - box_width) // 2
            box_y = 100
            
            # Draw alert box background
            cv2.rectangle(frame, (box_x, box_y), 
                         (box_x + box_width, box_y + box_height), 
                         alert_color, -1)
            
            # Draw white border
            cv2.rectangle(frame, (box_x, box_y), 
                         (box_x + box_width, box_y + box_height), 
                         border_color, 3)
            
            # Draw alert text
            text_x = box_x + (box_width - text_size[0]) // 2
            text_y = box_y + (box_height + text_size[1]) // 2
            cv2.putText(frame, alert_text, (text_x, text_y), 
                       font, font_scale, (0, 0, 0), thickness)  # Black text
            
            # Add debug info for development
            if hasattr(self.inhaler_tracker, 'history') and len(self.inhaler_tracker.history) > 1:
                recent_positions = [h['center'] for h in list(self.inhaler_tracker.history)[-5:]]
                if len(recent_positions) >= 2:
                    # Calculate recent movement for debug
                    last_movement = math.sqrt(
                        (recent_positions[-1][0] - recent_positions[-2][0])**2 + 
                        (recent_positions[-1][1] - recent_positions[-2][1])**2
                    )
                    
                    # Display movement debug info
                    debug_text = f"Movement: {last_movement:.1f}px"
                    cv2.putText(frame, debug_text, (10, 220), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    current_shake_status = "YES" if currently_shaking else "NO"
                    display_shake_status = "YES" if is_shaking_display else "NO"
                    cv2.putText(frame, f"Currently Shaking: {current_shake_status}", (10, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Display Status: {display_shake_status}", (10, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No inhaler detected - show instruction to detect inhaler first
            instruction_text = "POINT CAMERA AT RED INHALER"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(instruction_text, font, font_scale, thickness)[0]
            
            # Position instruction box at top center
            box_width = text_size[0] + 40
            box_height = text_size[1] + 30
            box_x = (img_width - box_width) // 2
            box_y = 100
            
            # Draw instruction box (gray)
            cv2.rectangle(frame, (box_x, box_y), 
                         (box_x + box_width, box_y + box_height), 
                         (128, 128, 128), -1)
            
            # Draw white border
            cv2.rectangle(frame, (box_x, box_y), 
                         (box_x + box_width, box_y + box_height), 
                         (255, 255, 255), 3)
            
            # Draw instruction text
            text_x = box_x + (box_width - text_size[0]) // 2
            text_y = box_y + (box_height + text_size[1]) // 2
            cv2.putText(frame, instruction_text, (text_x, text_y), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def get_instructions(self):
        return [
            "Inhaler Shaking Mode: Alerts when inhaler is not shaking",
            "Red alert = Not shaking, Yellow alert = Shaking detected",
            "Press 'm' to toggle mask view, '1-8' to switch modes"
        ]

class PreInhalationExhaleMode(BaseProcessingMode):
    """Pre Inhalation Exhale Mode - Focus on pre inhalation exhale alerts"""
    
    def __init__(self):
        super().__init__("Mode 4")
    
    def get_instructions(self):
        return [
            "Pre Inhalation Exhale Mode: Alerts when pre inhalation exhale is not exhaled",
            "Press 'm' to toggle mask view, '1-8' to switch modes"
        ]

class LipsSealedMode(BaseProcessingMode):
    """Lips Sealed Mode - Focus on lips sealed alerts"""
    
    def __init__(self):
        super().__init__("Mode 5")
    
    def get_instructions(self):
        return [
            "Lips Sealed Mode: Alerts when lips are not sealed",
            "Press 'm' to toggle mask view, '1-8' to switch modes"
        ]

class ActuationTimingMode(BaseProcessingMode):
    """Actuation Timing Mode - Focus on actuation timing alerts"""
    
    def __init__(self):
        super().__init__("Mode 6")
    
    def get_instructions(self):
        return [
            "Actuation Timing Mode: Alerts when actuation timing is not correct",
            "Press 'm' to toggle mask view, '1-8' to switch modes"
        ]

class PostInhalationHoldMode(BaseProcessingMode):
    """Post Inhalation Hold Mode - Focus on post inhalation hold alerts"""
    
    def __init__(self):
        super().__init__("Mode 7")
    
    def get_instructions(self):
        return [
            "Post Inhalation Hold Mode: Alerts when post inhalation hold is not correct",
            "Press 'm' to toggle mask view, '1-8' to switch modes"
        ]
    
class InhalationSpeedMode(BaseProcessingMode):
    """Inhalation Speed Mode - Focus on inhalation speed alerts"""
    
    def __init__(self):
        super().__init__("Mode 8")
    
    def get_instructions(self):
        return [
            "Inhalation Speed Mode: Alerts when inhalation speed is not correct",
            "Press 'm' to toggle mask view, '1-8' to switch modes"
        ]

class InhalerBuddyApp:
    """Main application class managing different modes"""
    
    def __init__(self):
        self.cap = None
        self.modes = {
            '1': FullMode(),
            '2': HeadPositioningMode(),
            '3': InhalerShakingMode(),
            '4': PreInhalationExhaleMode(),
            '5': LipsSealedMode(),
            '6': ActuationTimingMode(),
            '7': PostInhalationHoldMode(),
            '8': InhalationSpeedMode()
        }
        self.current_mode = self.modes['1']  # Start with full mode
        self.running = False
        
    def start(self):
        """Start the application"""
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        self.running = True
        self.current_mode.on_activate()
        
        print("InhalerBuddy started!")
        print(f"Current mode: {self.current_mode.name}")
        self._print_instructions()
        
        # Main loop
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame with current mode
            processed_frame = self.current_mode.process_frame(frame)
            
            # Draw mode info and instructions
            self._draw_ui(processed_frame)
            
            # Display frame
            cv2.imshow("InhalerBuddy - Multiple Testing Modes", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                break
        
        self._cleanup()
    
    def _handle_key(self, key):
        """Handle key presses, return False to quit"""
        # Global keys
        if key == ord('q'):
            return False
        
        # Mode switching
        if chr(key) in self.modes:
            new_mode = self.modes[chr(key)]
            if new_mode != self.current_mode:
                self.current_mode.on_deactivate()
                self.current_mode = new_mode
                self.current_mode.on_activate()
                print(f"Switched to: {self.current_mode.name}")
                self._print_instructions()
        
        # Let current mode handle the key
        elif not self.current_mode.handle_key(key):
            # Mode didn't handle it, check for help
            if key == ord('?') or key == ord('/'):
                self._print_instructions()
        
        return True
    
    def _print_instructions(self):
        """Print current mode instructions to console"""
        print(f"\n=== {self.current_mode.name} ===")
        for instruction in self.current_mode.get_instructions():
            print(f"  {instruction}")
        print("  Press '?' for help, 'q' to quit")
        print()
    
    def _draw_ui(self, frame):
        """Draw UI elements on frame"""
        img_height, img_width = frame.shape[:2]
        
        # Mode indicator
        mode_text = f"Mode: {self.current_mode.name} (Press 1-8 to switch)"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Quick instructions
        instructions = self.current_mode.get_instructions()
        if instructions:
            cv2.putText(frame, instructions[0], (10, img_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if len(instructions) > 1:
                cv2.putText(frame, instructions[1], (10, img_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Quick help
        cv2.putText(frame, "Press '?' for help, 'q' to quit", (10, img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    def _cleanup(self):
        """Clean up resources"""
        if self.current_mode:
            self.current_mode.on_deactivate()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("InhalerBuddy stopped.")

if __name__ == "__main__":
    app = InhalerBuddyApp()
    app.start()