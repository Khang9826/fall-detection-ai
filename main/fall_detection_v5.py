import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set
from collections import deque, defaultdict
from enum import Enum
from dataclasses import dataclass, field
from ultralytics import YOLO
import time


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Model Paths and Detection Configuration
YOLO_MODEL_PATH = r"G:\VLU\ultralytics\yolov8\yolov8m.pt"
# YOLO_MODEL_PATH = r"G:\VLU\NĂM 2\HỌC KÌ 2\Machine Learning\project\experiments\checkpoints\v2\fall_detection_v2\weights\best.pt" 
PERSON_CLASS_ID = 0  # COCO dataset person class identifier

# Detection Parameters
# These thresholds control the trade-off between detection sensitivity and false positives
YOLO_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence for person detection
YOLO_IOU_THRESHOLD = 0.45  # Non-maximum suppression IoU threshold
MIN_PERSON_AREA_RATIO = 0.02  # Minimum person bounding box area (% of frame)
MAX_PERSON_AREA_RATIO = 0.90  # Maximum person bounding box area (% of frame)

# Tracking Parameters
# ByteTrack configuration for maintaining identity across frames
TRACKER_TYPE = "bytetrack"  # Options: bytetrack, botsort
TRACK_BUFFER = 30  # Number of frames to retain lost tracks
MIN_TRACK_AGE = 3  # Minimum frames before considering a track as valid
MIN_TRACK_CONFIDENCE = 0.6  # Minimum detection confidence for tracking

# Bounding Box Processing
# Spatial processing to improve pose estimation accuracy
BBOX_PADDING_RATIO = 0.15  # Padding around detected person for pose estimation
BBOX_SMOOTH_ALPHA = 0.7  # Exponential moving average smoothing factor (0-1)

# Pose Estimation Parameters
# MediaPipe Pose configuration and validation thresholds
POSE_CONFIDENCE_THRESHOLD = 0.6  # Minimum landmark visibility score
MIN_VISIBLE_LANDMARKS = 8  # Minimum number of visible keypoints for valid pose
POSE_RETENTION_FRAMES = 10  # Frames to retain last valid pose if current fails
MAX_POSE_BBOX_DEVIATION = 0.3  # Maximum allowed landmark deviation outside bbox

# Temporal Analysis Configuration
# Sliding window parameters for temporal feature smoothing
TEMPORAL_WINDOW_SIZE = 15  # Maximum history size for feature tracking
FEATURE_SMOOTHING_WINDOW = 5  # Window size for feature averaging
MIN_FRAMES_FOR_STATE_CHANGE = 4  # Frames required to confirm state transition

# State Classification Thresholds
# These parameters define the decision boundaries for each state
# STANDING state: Upright posture with vertical torso
STANDING_TORSO_ANGLE_MAX = 25.0  # Maximum torso deviation from vertical (degrees)
TRANSITION_TORSO_ANGLE_RANGE = (25.0, 55.0)  # Intermediate posture angles
LYING_TORSO_ANGLE_MIN = 65.0  # Minimum angle for lying state (near horizontal)

# Body aspect ratio (height/width) thresholds
STANDING_BODY_RATIO_MIN = 2.0  # Minimum H/W ratio for standing (tall, narrow)
TRANSITION_BODY_RATIO_RANGE = (1.3, 2.0)  # Intermediate body ratios
LYING_BODY_RATIO_MAX = 1.3  # Maximum H/W ratio for lying (wide, flat)

# Vertical distance metrics (normalized by frame height)
STANDING_HIP_ANKLE_MIN = 0.35  # Minimum hip-ankle distance when standing
LYING_HIP_ANKLE_MAX = 0.20  # Maximum hip-ankle distance when lying

# Fall Detection Criteria
# Rapid change detection for identifying fall events
RAPID_ANGLE_CHANGE_THRESHOLD = 20.0  # Degrees per frame for rapid orientation change
RAPID_RATIO_CHANGE_THRESHOLD = 0.4  # Body ratio change per frame
RAPID_VERTICAL_VELOCITY_THRESHOLD = 0.05  # Normalized vertical velocity threshold
MIN_FALL_HEIGHT_CHANGE = 0.10  # Minimum center-of-mass drop (normalized)
SLOW_MOVEMENT_VELOCITY_MAX = 0.02  # Maximum velocity for "slow lying down"

# Finite State Machine Configuration
# Temporal constraints for state transitions and fall confirmation
FALL_COOLDOWN_FRAMES = 45  # Frames to wait before detecting another fall
LYING_MIN_DURATION = 20  # Minimum frames in LYING state to confirm fall
TRANSITION_TIMEOUT_FRAMES = 30  # Maximum frames allowed in TRANSITION state

# Visualization Parameters
BBOX_THICKNESS = 3  # Bounding box line thickness
POSE_THICKNESS = 2  # Skeleton line thickness
FONT_SCALE = 0.7  # Text size scaling
FONT_THICKNESS = 2  # Text line thickness
ALERT_DURATION_FRAMES = 30  # Number of frames to display fall alert

# Color Scheme (BGR format)
COLOR_STANDING = (0, 255, 0)  # Green for normal standing
COLOR_TRANSITION = (0, 255, 255)  # Yellow for transition state
COLOR_FALL = (0, 0, 255)  # Red for fall detection
COLOR_LYING = (255, 0, 255)  # Magenta for lying state
COLOR_TEXT = (255, 255, 255)  # White for text overlays
COLOR_BACKGROUND = (0, 0, 0)  # Black for background


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class FallState(Enum):
    STANDING = "STANDING"
    TRANSITION = "TRANSITION"
    FALL = "FALL"
    LYING = "LYING"


@dataclass
class PoseFeatures:
    torso_angle: float
    body_ratio: float
    hip_ankle_dist: float
    shoulder_y: float
    hip_y: float
    center_of_mass_y: float
    timestamp: int
    confidence: float


@dataclass
class TrackedPerson:
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    age: int
    
    # Pose tracking
    landmarks: Optional[np.ndarray] = None
    last_valid_landmarks: Optional[np.ndarray] = None
    frames_since_valid_pose: int = 0
    
    # Smoothed bbox
    smoothed_bbox: Optional[Tuple[int, int, int, int]] = None
    
    # Feature history
    feature_history: deque = field(default_factory=lambda: deque(maxlen=TEMPORAL_WINDOW_SIZE))
    last_valid_features: Optional[PoseFeatures] = None
    
    # State machine
    current_state: FallState = FallState.STANDING
    previous_state: FallState = FallState.STANDING
    state_frame_count: int = 0
    candidate_state: Optional[FallState] = None
    candidate_frame_count: int = 0
    
    # Fall detection
    fall_detected: bool = False
    fall_cooldown: int = 0
    fall_alert_counter: int = 0
    total_falls: int = 0
    initial_height: Optional[float] = None
    
    # Transition tracking
    transition_start_frame: int = 0
    pre_transition_state: FallState = FallState.STANDING


# ============================================================================
# PERSON DETECTION AND TRACKING MODULE
# ============================================================================

class PersonTracker:
    def __init__(
        self,
        yolo_model_path: str = YOLO_MODEL_PATH,
        conf_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
        iou_threshold: float = YOLO_IOU_THRESHOLD,
        tracker_type: str = TRACKER_TYPE
    ):
        
        print(f"[PersonTracker] Loading YOLO model: {yolo_model_path}")
        self.model = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_type = tracker_type
        
        # Track management structures
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.active_track_ids: Set[int] = set()
        self.frame_count = 0
        
        print(f"[PersonTracker] Initialized with {tracker_type.upper()} tracker")
    
    def detect_and_track(
        self,
        frame: np.ndarray
    ) -> Dict[int, TrackedPerson]:
        
        self.frame_count += 1
        
        # Run YOLO with integrated tracking
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[PERSON_CLASS_ID],
            tracker=f"{self.tracker_type}.yaml",
            persist=True,
            verbose=False
        )
        
        current_tracks = {}
        frame_area = frame.shape[0] * frame.shape[1]
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Check if tracking IDs are available
            if boxes.id is not None:
                for i in range(len(boxes)):
                    # Extract tracking information
                    track_id = int(boxes.id[i])
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i])
                    
                    # Area-based filtering to remove unrealistic detections
                    bbox_area = (x2 - x1) * (y2 - y1)
                    area_ratio = bbox_area / frame_area
                    
                    if not (MIN_PERSON_AREA_RATIO <= area_ratio <= MAX_PERSON_AREA_RATIO):
                        continue
                    
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    
                    # Update existing track or create new one
                    if track_id in self.tracked_persons:
                        person = self.tracked_persons[track_id]
                        person.bbox = bbox
                        person.confidence = confidence
                        person.age += 1
                    else:
                        person = TrackedPerson(
                            track_id=track_id,
                            bbox=bbox,
                            confidence=confidence,
                            age=1
                        )
                        self.tracked_persons[track_id] = person
                    
                    # Apply exponential moving average smoothing
                    person.smoothed_bbox = self._smooth_bbox(person, bbox)
                    
                    current_tracks[track_id] = person
        
        # Update active track set
        self.active_track_ids = set(current_tracks.keys())
        
        # Remove inactive tracks
        self._cleanup_tracks()
        
        return current_tracks
    
    def _smooth_bbox(
        self,
        person: TrackedPerson,
        new_bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        
        if person.smoothed_bbox is None:
            return new_bbox
        
        alpha = BBOX_SMOOTH_ALPHA
        old_bbox = person.smoothed_bbox
        
        smoothed = tuple(
            int(alpha * old + (1 - alpha) * new)
            for old, new in zip(old_bbox, new_bbox)
        )
        
        return smoothed
    
    def _cleanup_tracks(self):
        
        to_remove = []
        for track_id, person in self.tracked_persons.items():
            if track_id not in self.active_track_ids:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracked_persons[track_id]
    
    def expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int],
        padding_ratio: float = BBOX_PADDING_RATIO
    ) -> Tuple[int, int, int, int]:
        
        x1, y1, x2, y2 = bbox
        height, width = frame_shape
        
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        pad_x = int(bbox_width * padding_ratio)
        pad_y = int(bbox_height * padding_ratio)
        
        x1_exp = max(0, x1 - pad_x)
        y1_exp = max(0, y1 - pad_y)
        x2_exp = min(width, x2 + pad_x)
        y2_exp = min(height, y2 + pad_y)
        
        return (x1_exp, y1_exp, x2_exp, y2_exp)
    
    def get_valid_tracks(
        self,
        min_age: int = MIN_TRACK_AGE
    ) -> Dict[int, TrackedPerson]:
       
        return {
            track_id: person
            for track_id, person in self.tracked_persons.items()
            if track_id in self.active_track_ids and person.age >= min_age
        }


# ============================================================================
# POSE ESTIMATION AND STABILIZATION MODULE
# ============================================================================

class PoseStabilizer:
    
    
    # MediaPipe landmark indices for key body points
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    def __init__(
        self,
        confidence_threshold: float = POSE_CONFIDENCE_THRESHOLD,
        min_visible_landmarks: int = MIN_VISIBLE_LANDMARKS
    ):
        
        self.confidence_threshold = confidence_threshold
        self.min_visible_landmarks = min_visible_landmarks
        
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,  # Video mode with temporal smoothing
            model_complexity=1,  # Balance between accuracy and speed
            smooth_landmarks=True,  # Enable temporal smoothing
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
    
    def extract_pose(
        self,
        person_crop: np.ndarray,
        person: TrackedPerson
    ) -> Optional[np.ndarray]:
        
        # Convert to RGB for MediaPipe
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        
        # Run MediaPipe Pose
        results = self.pose.process(rgb_crop)
        
        if results.pose_landmarks is None:
            person.frames_since_valid_pose += 1
            
            # Retain last valid pose for continuity
            if (person.frames_since_valid_pose <= POSE_RETENTION_FRAMES and
                person.last_valid_landmarks is not None):
                return person.last_valid_landmarks
            
            return None
        
        # Convert landmarks to numpy array
        landmarks = self._landmarks_to_array(
            results.pose_landmarks,
            person_crop.shape[1],
            person_crop.shape[0]
        )
        
        # Validate landmark quality
        if not self._validate_landmarks(landmarks):
            person.frames_since_valid_pose += 1
            
            if (person.frames_since_valid_pose <= POSE_RETENTION_FRAMES and
                person.last_valid_landmarks is not None):
                return person.last_valid_landmarks
            
            return None
        
        # Update tracking state
        person.frames_since_valid_pose = 0
        person.last_valid_landmarks = landmarks.copy()
        
        return landmarks
    
    def _landmarks_to_array(
        self,
        landmarks,
        width: int,
        height: int
    ) -> np.ndarray:
        
        points = []
        for landmark in landmarks.landmark:
            points.append([
                landmark.x * width,  # Convert normalized x to pixels
                landmark.y * height,  # Convert normalized y to pixels
                landmark.z,  # Depth (relative, not metric)
                landmark.visibility  # Confidence score [0, 1]
            ])
        return np.array(points)
    
    def _validate_landmarks(self, landmarks: np.ndarray) -> bool:
     
        visible = landmarks[:, 3] >= self.confidence_threshold
        num_visible = np.sum(visible)
        return num_visible >= self.min_visible_landmarks
    
    def map_to_frame_coords(
        self,
        landmarks_crop: np.ndarray,
        crop_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        
        landmarks_frame = landmarks_crop.copy()
        x1, y1, _, _ = crop_bbox
        
        # Add offset to translate from crop to frame
        landmarks_frame[:, 0] += x1
        landmarks_frame[:, 1] += y1
        
        return landmarks_frame
    
    def check_landmarks_in_bbox(
        self,
        landmarks: np.ndarray,
        bbox: Tuple[int, int, int, int],
        max_deviation: float = MAX_POSE_BBOX_DEVIATION
    ) -> bool:
      
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Filter to visible landmarks only
        visible = landmarks[:, 3] >= self.confidence_threshold
        if np.sum(visible) == 0:
            return False
        
        visible_landmarks = landmarks[visible]
        
        # Check x-coordinate bounds
        x_coords = visible_landmarks[:, 0]
        x_margin = bbox_width * max_deviation
        x_valid = np.all((x_coords >= x1 - x_margin) & (x_coords <= x2 + x_margin))
        
        # Check y-coordinate bounds
        y_coords = visible_landmarks[:, 1]
        y_margin = bbox_height * max_deviation
        y_valid = np.all((y_coords >= y1 - y_margin) & (y_coords <= y2 + y_margin))
        
        return x_valid and y_valid
    
    def extract_features(
        self,
        landmarks: np.ndarray,
        frame_height: int,
        timestamp: int
    ) -> Optional[PoseFeatures]:
       
        try:
            # Helper function to safely extract visible keypoints
            def get_point(idx):
                if landmarks[idx, 3] < self.confidence_threshold:
                    return None
                return landmarks[idx, :2]
            
            # Extract key anatomical points
            l_shoulder = get_point(self.LEFT_SHOULDER)
            r_shoulder = get_point(self.RIGHT_SHOULDER)
            l_hip = get_point(self.LEFT_HIP)
            r_hip = get_point(self.RIGHT_HIP)
            l_ankle = get_point(self.LEFT_ANKLE)
            r_ankle = get_point(self.RIGHT_ANKLE)
            
            # Require essential torso landmarks
            if None in [l_shoulder, r_shoulder, l_hip, r_hip]:
                return None
            
            # Compute anatomical centers
            shoulder_center = (l_shoulder + r_shoulder) / 2
            hip_center = (l_hip + r_hip) / 2
            
            # === FEATURE 1: Torso Angle ===
            # Compute angle between torso vector and vertical axis
            torso_vector = hip_center - shoulder_center
            vertical_vector = np.array([0, 1])
            
            cos_angle = np.dot(torso_vector, vertical_vector) / (
                np.linalg.norm(torso_vector) * np.linalg.norm(vertical_vector) + 1e-6
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            # === FEATURE 2: Body Aspect Ratio ===
            # Compute bounding box of all visible keypoints
            visible = landmarks[:, 3] >= self.confidence_threshold
            if np.sum(visible) < 4:
                return None
            
            x_coords = landmarks[visible, 0]
            y_coords = landmarks[visible, 1]
            
            body_width = np.max(x_coords) - np.min(x_coords)
            body_height = np.max(y_coords) - np.min(y_coords)
            body_ratio = body_height / (body_width + 1e-6)
            
            # === FEATURE 3: Hip-Ankle Vertical Distance ===
            if l_ankle is not None and r_ankle is not None:
                ankle_center = (l_ankle + r_ankle) / 2
                hip_ankle_dist = abs(hip_center[1] - ankle_center[1]) / frame_height
            else:
                hip_ankle_dist = 0.0
            
            # === FEATURE 4: Normalized Vertical Positions ===
            shoulder_y_norm = shoulder_center[1] / frame_height
            hip_y_norm = hip_center[1] / frame_height
            
            # Center of mass (simplified as midpoint of shoulder-hip)
            com_y = (shoulder_center[1] + hip_center[1]) / 2
            com_y_norm = com_y / frame_height
            
            # === Confidence Metric ===
            # Average visibility of key torso landmarks
            key_indices = [
                self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
                self.LEFT_HIP, self.RIGHT_HIP
            ]
            confidence = np.mean([landmarks[i, 3] for i in key_indices])
            
            return PoseFeatures(
                torso_angle=float(angle),
                body_ratio=float(body_ratio),
                hip_ankle_dist=float(hip_ankle_dist),
                shoulder_y=float(shoulder_y_norm),
                hip_y=float(hip_y_norm),
                center_of_mass_y=float(com_y_norm),
                timestamp=timestamp,
                confidence=float(confidence)
            )
        
        except Exception:
            return None
    
    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()


# ============================================================================
# TEMPORAL FINITE STATE MACHINE FOR FALL DETECTION
# ============================================================================

class FallStateMachine:
    @staticmethod
    def update_person_state(
        person: TrackedPerson,
        features: Optional[PoseFeatures],
        frame_number: int
    ) -> Tuple[FallState, bool]:
        # Update temporal counters
        person.state_frame_count += 1
        if person.fall_cooldown > 0:
            person.fall_cooldown -= 1
        if person.fall_alert_counter > 0:
            person.fall_alert_counter -= 1
        
        # Add features to temporal history
        if features is not None:
            person.feature_history.append(features)
            person.last_valid_features = features
        
        # Get temporally smoothed features
        smoothed = FallStateMachine._get_smoothed_features(person)
        if smoothed is None:
            return person.current_state, False
        
        # Evaluate suggested state from smoothed features
        suggested_state = FallStateMachine._evaluate_state(smoothed)
        
        # Handle state transitions with confirmation logic
        if suggested_state != person.current_state:
            if suggested_state != person.candidate_state:
                # New candidate state proposed
                person.candidate_state = suggested_state
                person.candidate_frame_count = 1
            else:
                # Existing candidate state persists
                person.candidate_frame_count += 1
            
            # Confirm transition after sufficient frames
            if person.candidate_frame_count >= MIN_FRAMES_FOR_STATE_CHANGE:
                FallStateMachine._transition_state(person, suggested_state, smoothed)
        else:
            # State matches current - reset candidate
            person.candidate_state = None
            person.candidate_frame_count = 0
        
        # Check for fall event detection
        fall_detected = FallStateMachine._check_fall(person, smoothed)
        
        # Transition state timeout (prevent getting stuck in TRANSITION)
        if person.current_state == FallState.TRANSITION:
            if person.state_frame_count > TRANSITION_TIMEOUT_FRAMES:
                FallStateMachine._transition_state(
                    person, person.pre_transition_state, smoothed
                )
        
        return person.current_state, fall_detected
    
    @staticmethod
    def _get_smoothed_features(person: TrackedPerson) -> Optional[PoseFeatures]:
        if len(person.feature_history) == 0:
            return person.last_valid_features
        
        window_size = min(FEATURE_SMOOTHING_WINDOW, len(person.feature_history))
        recent = list(person.feature_history)[-window_size:]
        
        return PoseFeatures(
            torso_angle=np.mean([f.torso_angle for f in recent]),
            body_ratio=np.mean([f.body_ratio for f in recent]),
            hip_ankle_dist=np.mean([f.hip_ankle_dist for f in recent]),
            shoulder_y=np.mean([f.shoulder_y for f in recent]),
            hip_y=np.mean([f.hip_y for f in recent]),
            center_of_mass_y=np.mean([f.center_of_mass_y for f in recent]),
            timestamp=person.feature_history[-1].timestamp,
            confidence=np.mean([f.confidence for f in recent])
        )
    
    @staticmethod
    def _evaluate_state(features: PoseFeatures) -> FallState:
        angle = features.torso_angle
        ratio = features.body_ratio
        hip_ankle = features.hip_ankle_dist
        
        # STANDING: Upright posture
        if (angle < STANDING_TORSO_ANGLE_MAX and
            ratio > STANDING_BODY_RATIO_MIN and
            hip_ankle > STANDING_HIP_ANKLE_MIN):
            return FallState.STANDING
        
        # LYING: Horizontal posture
        if (angle > LYING_TORSO_ANGLE_MIN and
            ratio < LYING_BODY_RATIO_MAX and
            hip_ankle < LYING_HIP_ANKLE_MAX):
            return FallState.LYING
        
        # TRANSITION: Intermediate state
        return FallState.TRANSITION
    
    @staticmethod
    def _transition_state(
        person: TrackedPerson,
        new_state: FallState,
        features: PoseFeatures
    ):
        person.previous_state = person.current_state
        person.current_state = new_state
        person.state_frame_count = 0
        
        # Track pre-transition state for timeout recovery
        if new_state == FallState.TRANSITION:
            person.pre_transition_state = person.previous_state
        
        # Initialize height baseline when entering STANDING
        if new_state == FallState.STANDING:
            person.initial_height = features.center_of_mass_y
        
        # Reset candidate state
        person.candidate_state = None
        person.candidate_frame_count = 0
    
    @staticmethod
    def _check_fall(person: TrackedPerson, features: PoseFeatures) -> bool:
        if person.fall_cooldown > 0:
            return False
        
        # Primary condition: TRANSITION → LYING
        if (person.current_state == FallState.LYING and
            person.previous_state == FallState.TRANSITION and
            person.state_frame_count >= LYING_MIN_DURATION):
            
            # Criterion A: Rapid posture change
            rapid = FallStateMachine._detect_rapid_change(person)
            
            # Criterion B: Significant height drop
            height_drop = FallStateMachine._detect_height_drop(person, features)
            
            # Negative Rule: Slow controlled movement (e.g., lying down intentionally)
            slow = FallStateMachine._is_slow_movement(person)
            
            # Fall detected if (rapid OR height_drop) AND NOT slow
            if (rapid or height_drop) and not slow:
                person.total_falls += 1
                person.fall_detected = True
                person.fall_cooldown = FALL_COOLDOWN_FRAMES
                person.fall_alert_counter = ALERT_DURATION_FRAMES
                return True
        
        return False
    
    @staticmethod
    def _detect_rapid_change(person: TrackedPerson) -> bool:
        if len(person.feature_history) < 2:
            return False
        
        current = person.feature_history[-1]
        previous = person.feature_history[-2]
        
        # Compute frame-to-frame changes
        angle_vel = abs(current.torso_angle - previous.torso_angle)
        ratio_vel = abs(current.body_ratio - previous.body_ratio)
        vertical_vel = abs(current.center_of_mass_y - previous.center_of_mass_y)
        
        # Check against thresholds
        rapid_angle = angle_vel > RAPID_ANGLE_CHANGE_THRESHOLD
        rapid_ratio = ratio_vel > RAPID_RATIO_CHANGE_THRESHOLD
        rapid_vertical = vertical_vel > RAPID_VERTICAL_VELOCITY_THRESHOLD
        
        # Rapid change if both posture AND vertical motion are rapid
        return (rapid_angle or rapid_ratio) and rapid_vertical
    
    @staticmethod
    def _detect_height_drop(person: TrackedPerson, features: PoseFeatures) -> bool:
        if person.initial_height is None:
            return False
        
        height_change = features.center_of_mass_y - person.initial_height
        return height_change > MIN_FALL_HEIGHT_CHANGE
    
    @staticmethod
    def _is_slow_movement(person: TrackedPerson) -> bool:
        if len(person.feature_history) < 2:
            return True
        
        current = person.feature_history[-1]
        previous = person.feature_history[-2]
        
        vertical_vel = abs(current.center_of_mass_y - previous.center_of_mass_y)
        return vertical_vel < SLOW_MOVEMENT_VELOCITY_MAX


# ============================================================================
# INTEGRATED FALL DETECTION SYSTEM
# ============================================================================

class TrackingFallDetector:
    
    
    def __init__(
        self,
        yolo_model_path: str = YOLO_MODEL_PATH,
        tracker_type: str = TRACKER_TYPE
    ):
        
        print("\n" + "=" * 80)
        print("TRACKING-BASED FALL DETECTION SYSTEM")
        print("=" * 80)
        
        # Initialize detection and tracking
        self.tracker = PersonTracker(
            yolo_model_path=yolo_model_path,
            tracker_type=tracker_type
        )
        
        # Initialize pose estimation
        self.pose_stabilizer = PoseStabilizer()
        
        # Statistics tracking
        self.frame_count = 0
        self.total_falls = 0
        self.processing_times = deque(maxlen=30)
        
        print("[System] Initialization complete!")
        print("=" * 80 + "\n")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        
        start_time = time.time()
        self.frame_count += 1
        
        annotated_frame = frame.copy()
        num_falls = 0
        
        # Stage 1: Detect and track all persons
        tracked_persons = self.tracker.detect_and_track(frame)
        
        # Stage 2: Filter to valid tracks (sufficient age)
        valid_tracks = self.tracker.get_valid_tracks()
        
        # Stage 3: Process each tracked person
        for track_id, person in valid_tracks.items():
            # Expand bbox for better pose estimation coverage
            expanded_bbox = self.tracker.expand_bbox(
                person.smoothed_bbox if person.smoothed_bbox else person.bbox,
                (frame.shape[0], frame.shape[1])
            )
            
            x1, y1, x2, y2 = expanded_bbox
            
            # Crop person region
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                continue
            
            # Extract pose from crop
            landmarks_crop = self.pose_stabilizer.extract_pose(person_crop, person)
            
            if landmarks_crop is not None:
                # Transform landmarks to global frame coordinates
                landmarks_frame = self.pose_stabilizer.map_to_frame_coords(
                    landmarks_crop, expanded_bbox
                )
                
                # Validate spatial consistency
                bbox_for_check = person.smoothed_bbox if person.smoothed_bbox else person.bbox
                if self.pose_stabilizer.check_landmarks_in_bbox(
                    landmarks_frame, bbox_for_check
                ):
                    # Extract geometric features
                    features = self.pose_stabilizer.extract_features(
                        landmarks_frame,
                        frame.shape[0],
                        self.frame_count
                    )
                    
                    # Update state machine and check for fall
                    current_state, fall_detected = FallStateMachine.update_person_state(
                        person, features, self.frame_count
                    )
                    
                    if fall_detected:
                        num_falls += 1
                        self.total_falls += 1
                    
                    # Visualize with pose skeleton
                    person.landmarks = landmarks_frame
                    annotated_frame = self._visualize_person(
                        annotated_frame, person, features
                    )
                else:
                    # Landmarks spatially inconsistent - visualize bbox only
                    annotated_frame = self._visualize_person(
                        annotated_frame, person, None
                    )
            else:
                # No valid pose - visualize bbox only
                annotated_frame = self._visualize_person(
                    annotated_frame, person, None
                )
        
        # Stage 4: Overlay global information
        elapsed = time.time() - start_time
        self.processing_times.append(elapsed)
        avg_fps = 1.0 / (np.mean(self.processing_times) + 1e-6)
        
        annotated_frame = self._draw_global_info(
            annotated_frame, len(valid_tracks), avg_fps
        )
        
        return annotated_frame, num_falls
    
    def _visualize_person(
        self,
        frame: np.ndarray,
        person: TrackedPerson,
        features: Optional[PoseFeatures]
    ) -> np.ndarray:
        
        # Select color based on current state
        color_map = {
            FallState.STANDING: COLOR_STANDING,
            FallState.TRANSITION: COLOR_TRANSITION,
            FallState.FALL: COLOR_FALL,
            FallState.LYING: COLOR_LYING
        }
        color = color_map[person.current_state]
        
        # Get bounding box (prefer smoothed)
        bbox = person.smoothed_bbox if person.smoothed_bbox else person.bbox
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box (thicker if fall alert active)
        thickness = BBOX_THICKNESS * 2 if person.fall_alert_counter > 0 else BBOX_THICKNESS
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw track ID and state label
        label = f"ID:{person.track_id} {person.current_state.value}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
        label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
        
        cv2.rectangle(
            frame,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0] + 5, label_y + 5),
            color, -1
        )
        cv2.putText(
            frame, label, (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_TEXT, FONT_THICKNESS
        )
        
        # Draw pose skeleton if available
        if person.landmarks is not None:
            frame = self._draw_skeleton(frame, person.landmarks, color)
        
        # Draw flashing fall alert
        if person.fall_alert_counter > 0:
            alert_text = f"FALL DETECTED - ID:{person.track_id}"
            text_size = cv2.getTextSize(
                alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
            )[0]
            
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = 60
            
            # Flashing effect
            if (self.frame_count // 5) % 2 == 0:
                cv2.rectangle(
                    frame,
                    (text_x - 10, text_y - text_size[1] - 10),
                    (text_x + text_size[0] + 10, text_y + 10),
                    COLOR_FALL, -1
                )
                cv2.putText(
                    frame, alert_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 3
                )
        
        # Draw debug feature information
        if features is not None:
            info_y = y2 + 25
            info_lines = [
                f"Angle: {features.torso_angle:.1f}°",
                f"Ratio: {features.body_ratio:.2f}",
                f"Age: {person.age}f"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(
                    frame, line, (x1, info_y + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
        
        return frame
    
    def _draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
       
        # Define anatomical connections (MediaPipe standard)
        connections = [
            # Torso
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            # Legs
            (23, 25), (25, 27), (24, 26), (26, 28),
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8)
        ]
        
        # Draw connections
        for conn in connections:
            idx1, idx2 = conn
            
            if (landmarks[idx1, 3] >= POSE_CONFIDENCE_THRESHOLD and
                landmarks[idx2, 3] >= POSE_CONFIDENCE_THRESHOLD):
                
                pt1 = (int(landmarks[idx1, 0]), int(landmarks[idx1, 1]))
                pt2 = (int(landmarks[idx2, 0]), int(landmarks[idx2, 1]))
                
                cv2.line(frame, pt1, pt2, color, POSE_THICKNESS)
        
        # Draw keypoints
        for i in range(len(landmarks)):
            if landmarks[i, 3] >= POSE_CONFIDENCE_THRESHOLD:
                pt = (int(landmarks[i, 0]), int(landmarks[i, 1]))
                cv2.circle(frame, pt, 3, color, -1)
        
        return frame
    
    def _draw_global_info(
        self,
        frame: np.ndarray,
        num_tracks: int,
        fps: float
    ) -> np.ndarray:
        
        # Draw semi-transparent background bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), COLOR_BACKGROUND, -1)
        
        # Overlay statistics
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Tracks: {num_tracks}",
            f"FPS: {fps:.1f}",
            f"Falls: {self.total_falls}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(
                frame, line, (10, 25 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2
            )
        
        return frame
    
    def reset(self):
        """Reset detector state."""
        self.tracker.tracked_persons.clear()
        self.tracker.active_track_ids.clear()
        self.frame_count = 0
        self.total_falls = 0
    
    def close(self):
        """Release system resources."""
        self.pose_stabilizer.close()


# ============================================================================
# VIDEO PROCESSING INTERFACE
# ============================================================================

def process_video(
    detector: TrackingFallDetector,
    input_source: str = "0",
    output_path: Optional[str] = None,
    display: bool = True
):

    # Open video source
    if input_source.isdigit():
        cap = cv2.VideoCapture(int(input_source))
        source_name = f"Webcam {input_source}"
    else:
        cap = cv2.VideoCapture(input_source)
        source_name = Path(input_source).name
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {input_source}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("\n" + "=" * 80)
    print("VIDEO PROCESSING")
    print("=" * 80)
    print(f"Source: {source_name}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    print("=" * 80)
    print("Controls: 'q' = quit, 'r' = reset")
    print("=" * 80 + "\n")
    
    # Initialize video writer if output requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, num_falls = detector.process_frame(frame)
            
            # Write to output video
            if writer:
                writer.write(annotated_frame)
            
            # Display with interactive controls
            if display:
                cv2.imshow("Tracking-Based Fall Detection", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n[INFO] Stopped by user")
                    break
                elif key == ord('r'):
                    print("\n[INFO] Resetting detector...")
                    detector.reset()
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total falls detected: {detector.total_falls}")
        print("=" * 80 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    
    detector = TrackingFallDetector(
        yolo_model_path=YOLO_MODEL_PATH,
        tracker_type=TRACKER_TYPE
    )
    
    try:
        process_video(
            detector=detector,
            input_source="0",  # Webcam 0
            display=True
        )
    finally:
        detector.close()


if __name__ == "__main__":
    main()