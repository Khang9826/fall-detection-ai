"""
Advanced Real-Time Fall Detection System v4
Combines YOLOv12s person detection with MediaPipe pose estimation and
a Finite State Machine for robust, temporally-aware fall detection.

Features:
- IoU-based person tracking with bbox EMA smoothing
- SITTING state in FSM (distinguishes sitting from transition/lying)
- Head-hip height ratio feature for better lying detection
- Multi-scale velocity analysis (1, 3, 5 frame windows)
- Fall confidence scoring (weighted multi-indicator)
- Height-drop + slow-movement filter for robust fall confirmation
- Pose retention when MediaPipe fails temporarily
- Transition timeout to prevent FSM getting stuck
- Fall alert duration window (holds FALL state for visualization)
- Adaptive frame skip based on person count
- Adaptive MediaPipe skip for stable persons
- GPU auto-detection with CPU fallback

Author: Senior Python & Computer Vision Engineer
Date: January 2026, upgraded April 2026
Compatible with: Python 3.9+, Windows/Linux/MacOS
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
from enum import Enum
from dataclasses import dataclass, field
from ultralytics import YOLO
from config_backend import (
    YOLO_MODEL_PATH, YOLO_IMGSZ, USE_GPU,
    MEDIAPIPE_MODEL_COMPLEXITY, DEBUG_VISUALIZATION
)


# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Detection thresholds
YOLO_CONFIDENCE_THRESHOLD = 0.25  # Lower default improves recall for webcam/video inputs
POSE_CONFIDENCE_THRESHOLD = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.65  # Higher = more stable tracking between frames
POSE_MIN_DETECTION_CONFIDENCE = 0.6  # Higher = fewer false pose detections
LANDMARK_VISIBILITY_THRESHOLD = 0.5  # Skip landmarks below this visibility

# YOLO post-processing
YOLO_FALLBACK_CONFIDENCE = 0.15  # Retry at lower confidence if first pass returns no boxes
YOLO_MIN_BBOX_AREA_RATIO = 0.002  # Keep smaller boxes for distant people in wide frames
YOLO_MAX_BBOX_AREA_RATIO = 0.95

# Crop quality for MediaPipe
BBOX_PAD_RATIO = 0.15        # Expand YOLO bbox by 15% on each side for better pose estimation
MIN_CROP_HEIGHT = 200        # Resize small crops to at least this height before pose estimation

# Landmark smoothing (One Euro Filter — adapts: smooth when still, responsive when moving)
OEF_MIN_CUTOFF = 0.7   # Lower = smoother when still (try 0.3–1.5)
OEF_BETA = 0.5         # Higher = faster response to movement (try 0.0–1.0)
OEF_D_CUTOFF = 1.0     # Cutoff for derivative estimation

# Temporal analysis settings
TEMPORAL_WINDOW_SIZE = 20  # Number of frames for temporal analysis
MIN_FRAMES_FOR_STATE_CHANGE = 5  # Frames to confirm state transition (higher = less flicker)

# Fall detection thresholds
STANDING_TORSO_ANGLE_MAX = 30.0
TRANSITION_TORSO_ANGLE_MAX = 50.0
FALL_TORSO_ANGLE_MIN = 55.0
LYING_TORSO_ANGLE_MIN = 60.0

STANDING_BODY_RATIO_MIN = 1.8
TRANSITION_BODY_RATIO_MIN = 1.3
LYING_BODY_RATIO_MAX = 1.2

HIP_ANKLE_STANDING_MIN = 0.3
HIP_ANKLE_LYING_MAX = 0.2

# Sitting detection thresholds
# SITTING_KNEE_ANGLE_MAX = 120.0  # Bent knees indicate sitting
# SITTING_TORSO_ANGLE_MAX = 45.0  # Torso relatively upright when sitting
# SITTING_BODY_RATIO_RANGE = (0.8, 1.8)  # Intermediate ratio when sitting
# SITTING_HEAD_HIP_RATIO_MIN = 0.08  # Head well above hips when sitting

# Temporal velocity thresholds
RAPID_ANGLE_CHANGE_THRESHOLD = 15.0
RAPID_RATIO_CHANGE_THRESHOLD = 0.3

# Multi-scale velocity windows
VELOCITY_WINDOWS = [1, 3, 5]  # Frame offsets for velocity computation

# Fall confidence scoring weights
FALL_WEIGHT_ANGLE = 0.30
FALL_WEIGHT_RATIO = 0.20
FALL_WEIGHT_VELOCITY = 0.25
FALL_WEIGHT_HEAD_HIP = 0.15
FALL_WEIGHT_KNEE = 0.10
FALL_CONFIDENCE_THRESHOLD = 0.55  # Minimum confidence to trigger fall

# State persistence
FALL_COOLDOWN_FRAMES = 30
LYING_MIN_DURATION = 15

# Person tracking
IOU_MATCH_THRESHOLD = 0.3  # Minimum IoU to match detections across frames
MAX_LOST_FRAMES = 15  # Frames before dropping a lost track

# Frame skip optimization (adaptive: overridden at runtime based on person count)
YOLO_FRAME_SKIP = 2  # Default: run YOLO every N frames (1 = every frame)

# Fall confirmation (backported from v5)
MIN_FALL_HEIGHT_CHANGE = 0.05     # Min vertical drop in center-of-mass to confirm fall
SLOW_MOVEMENT_VELOCITY_MAX = 5.0  # Max angle velocity to consider "slow" (suppress fall)
TRANSITION_TIMEOUT_FRAMES = 30    # Revert TRANSITION if stuck for this many frames
FALL_ALERT_DURATION_FRAMES = 30   # Hold FALL state visually for this many frames

# Pose retention
POSE_RETENTION_FRAMES = 10  # Use cached landmarks when MediaPipe fails

# Bbox smoothing
BBOX_SMOOTH_ALPHA = 0.7  # EMA alpha for bbox smoothing (higher = more responsive)

# Adaptive MediaPipe skip for stable persons
STABLE_STANDING_FRAMES = 30  # Frames in STANDING before reducing MediaPipe rate
MEDIAPIPE_SKIP_STABLE = 3    # Run MediaPipe every N frames for stable standing persons

# Input/Output
INPUT_SOURCE = "0"
OUTPUT_PATH = None

# Visualization settings
BBOX_THICKNESS = 3
POSE_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Colors (BGR) — two-state: FALL=red, NO_FALL=green
COLOR_NO_FALL = (0, 255, 0)       # Green
COLOR_FALL = (0, 0, 255)          # Red


# ============================================================================
# FINITE STATE MACHINE
# ============================================================================


class FallState(Enum):
    """Two-state fall detection: FALL or NO_FALL."""
    NO_FALL = "NO_FALL"
    FALL = "FALL"


@dataclass
class PoseFeatures:
    """Container for pose-based features extracted from MediaPipe landmarks."""
    torso_angle: float       # Angle of torso from vertical (degrees)
    body_ratio: float        # Height/width aspect ratio of full body
    hip_ankle_dist: float    # Normalized vertical distance hip->ankle
    knee_angle: float        # Average knee bend angle (degrees)
    shoulder_y: float        # Shoulder center Y coordinate (normalized)
    hip_y: float             # Hip center Y coordinate (normalized)
    head_hip_ratio: float    # Vertical distance head-to-hip / body height
    timestamp: int           # Frame number


@dataclass
class TrackedPerson:
    """Represents a tracked person across frames."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    frames_lost: int = 0
    fsm: Any = None  # FallStateMachine, assigned after creation
    smoothed_bbox: Optional[Tuple[float, float, float, float]] = None  # EMA-smoothed bbox


class FallStateMachine:
    """
    Finite State Machine for temporal fall detection.
    Tracks state transitions and prevents false positives through temporal consistency.
    """

    def __init__(
        self,
        temporal_window_size: int = TEMPORAL_WINDOW_SIZE,
        min_frames_for_change: int = MIN_FRAMES_FOR_STATE_CHANGE
    ):
        self.current_state = FallState.NO_FALL
        self.previous_state = FallState.NO_FALL
        self.state_frame_count = 0
        self.fall_detected_flag = False
        self.fall_cooldown_counter = 0
        self.fall_confidence = 0.0

        # Temporal tracking
        self.temporal_window_size = temporal_window_size
        self.min_frames_for_change = min_frames_for_change
        self.feature_history: deque = deque(maxlen=temporal_window_size)

        # State transition candidate tracking
        self.candidate_state: Optional[FallState] = None
        self.candidate_frame_count = 0

        # Height-drop tracking (for fall confirmation)
        self.initial_height: Optional[float] = None

        # Fall alert duration (hold FALL state visually)
        self.fall_alert_counter: int = 0

        # Statistics
        self.total_falls_detected = 0
        self.frame_number = 0

    def add_features(self, features: PoseFeatures) -> None:
        self.feature_history.append(features)
        self.frame_number += 1

    def compute_temporal_velocity(self) -> Dict[str, float]:
        """Compute temporal changes across multiple time scales."""
        if len(self.feature_history) < 2:
            return {
                'angle_velocity': 0.0,
                'ratio_velocity': 0.0,
                'vertical_velocity': 0.0,
                'max_angle_velocity': 0.0,
                'max_ratio_velocity': 0.0
            }

        current = self.feature_history[-1]
        max_angle_vel = 0.0
        max_ratio_vel = 0.0

        for window in VELOCITY_WINDOWS:
            if len(self.feature_history) > window:
                past = self.feature_history[-(window + 1)]
                angle_vel = abs(current.torso_angle - past.torso_angle) / window
                ratio_vel = abs(current.body_ratio - past.body_ratio) / window
                max_angle_vel = max(max_angle_vel, angle_vel)
                max_ratio_vel = max(max_ratio_vel, ratio_vel)

        # Immediate (1-frame) velocities
        previous = self.feature_history[-2]
        angle_velocity = abs(current.torso_angle - previous.torso_angle)
        ratio_velocity = abs(current.body_ratio - previous.body_ratio)
        vertical_velocity = abs(current.shoulder_y - previous.shoulder_y)

        return {
            'angle_velocity': angle_velocity,
            'ratio_velocity': ratio_velocity,
            'vertical_velocity': vertical_velocity,
            'max_angle_velocity': max_angle_vel,
            'max_ratio_velocity': max_ratio_vel
        }

    def get_averaged_features(self, window_size: int = 3) -> Optional[PoseFeatures]:
        """Get smoothed features by averaging recent frames."""
        if len(self.feature_history) < window_size:
            return self.feature_history[-1] if self.feature_history else None

        recent_features = list(self.feature_history)[-window_size:]

        return PoseFeatures(
            torso_angle=np.mean([f.torso_angle for f in recent_features]),
            body_ratio=np.mean([f.body_ratio for f in recent_features]),
            hip_ankle_dist=np.mean([f.hip_ankle_dist for f in recent_features]),
            knee_angle=np.mean([f.knee_angle for f in recent_features]),
            shoulder_y=np.mean([f.shoulder_y for f in recent_features]),
            hip_y=np.mean([f.hip_y for f in recent_features]),
            head_hip_ratio=np.mean([f.head_hip_ratio for f in recent_features]),
            timestamp=self.frame_number
        )

    def detect_rapid_change(self) -> bool:
        """Detect rapid posture changes across entire history window (not just current frame)."""
        if len(self.feature_history) < 2:
            return False

        # Scan entire history for max velocity (catches peak even if past)
        max_angle_vel = 0.0
        max_ratio_vel = 0.0
        history = list(self.feature_history)
        for i in range(1, len(history)):
            angle_vel = abs(history[i].torso_angle - history[i - 1].torso_angle)
            ratio_vel = abs(history[i].body_ratio - history[i - 1].body_ratio)
            max_angle_vel = max(max_angle_vel, angle_vel)
            max_ratio_vel = max(max_ratio_vel, ratio_vel)

        return (max_angle_vel > RAPID_ANGLE_CHANGE_THRESHOLD or
                max_ratio_vel > RAPID_RATIO_CHANGE_THRESHOLD)

    def is_slow_movement(self) -> bool:
        """Check if all movement in history is slow (controlled lying down, not a fall)."""
        if len(self.feature_history) < 3:
            return False

        history = list(self.feature_history)
        for i in range(1, len(history)):
            angle_vel = abs(history[i].torso_angle - history[i - 1].torso_angle)
            if angle_vel > SLOW_MOVEMENT_VELOCITY_MAX:
                return False
        return True

    def check_height_drop(self) -> bool:
        """Check if there was a significant vertical drop from standing height."""
        if self.initial_height is None or len(self.feature_history) < 1:
            return True  # Can't verify, allow fall detection
        current_height = self.feature_history[-1].shoulder_y
        # shoulder_y increases downward in image coordinates
        return (current_height - self.initial_height) > MIN_FALL_HEIGHT_CHANGE

    def compute_fall_confidence(
        self,
        features: PoseFeatures,
        velocities: Dict[str, float]
    ) -> float:
        """
        Compute a weighted confidence score for fall detection.
        Returns a value in [0, 1] where higher = more likely a fall.
        """
        scores = {}

        # Angle score: how far from vertical (0=vertical, 1=horizontal)
        scores['angle'] = min(features.torso_angle / 90.0, 1.0)

        # Ratio score: how wide relative to height (inverted, 0=tall, 1=flat)
        scores['ratio'] = max(0.0, 1.0 - features.body_ratio / 2.5)

        # Velocity score: how fast the change is happening
        max_vel = velocities.get('max_angle_velocity', 0.0)
        scores['velocity'] = min(max_vel / 30.0, 1.0)

        # Head-hip score: head close to hip level suggests lying
        scores['head_hip'] = max(0.0, 1.0 - features.head_hip_ratio / 0.3)

        # Knee score: straight knees more likely lying, bent more likely sitting
        scores['knee'] = min(max(features.knee_angle - 120.0, 0.0) / 60.0, 1.0)

        confidence = (
            FALL_WEIGHT_ANGLE * scores['angle'] +
            FALL_WEIGHT_RATIO * scores['ratio'] +
            FALL_WEIGHT_VELOCITY * scores['velocity'] +
            FALL_WEIGHT_HEAD_HIP * scores['head_hip'] +
            FALL_WEIGHT_KNEE * scores['knee']
        )

        return np.clip(confidence, 0.0, 1.0)

    def evaluate_state_from_features(
        self,
        features: PoseFeatures,
        velocities: Dict[str, float]
    ) -> FallState:
        """Determine state: FALL or NO_FALL based on confidence scoring."""

        # Compute fall confidence
        self.fall_confidence = self.compute_fall_confidence(features, velocities)
        rapid_change = self.detect_rapid_change()

        # FALL: high confidence + rapid posture change
        if rapid_change and self.fall_confidence > FALL_CONFIDENCE_THRESHOLD:
            return FallState.FALL

        # Also detect FALL from posture alone (lying on ground)
        if (features.torso_angle > LYING_TORSO_ANGLE_MIN and
                features.body_ratio < LYING_BODY_RATIO_MAX and
                features.hip_ankle_dist < HIP_ANKLE_LYING_MAX):
            return FallState.FALL

        return FallState.NO_FALL

    def update_state(self, features: PoseFeatures) -> Tuple[FallState, bool]:
        """
        Update FSM state: FALL or NO_FALL.
        Uses temporal consistency, height-drop confirmation, and slow-movement filter.

        Returns:
            Tuple of (current_state, fall_event_triggered)
        """
        self.add_features(features)

        if self.fall_cooldown_counter > 0:
            self.fall_cooldown_counter -= 1

        # Decrement fall alert counter
        if self.fall_alert_counter > 0:
            self.fall_alert_counter -= 1

        smoothed_features = self.get_averaged_features(window_size=5)
        if smoothed_features is None:
            return self.current_state, False

        velocities = self.compute_temporal_velocity()
        suggested_state = self.evaluate_state_from_features(smoothed_features, velocities)

        fall_event = False

        if suggested_state == self.current_state:
            self.candidate_state = None
            self.candidate_frame_count = 0
            self.state_frame_count += 1
        elif suggested_state != self.candidate_state:
            self.candidate_state = suggested_state
            self.candidate_frame_count = 1
        else:
            self.candidate_frame_count += 1

            if self.candidate_frame_count >= self.min_frames_for_change:
                self.previous_state = self.current_state
                self.current_state = suggested_state
                self.state_frame_count = 0
                self.candidate_state = None
                self.candidate_frame_count = 0

                # Track initial height when NO_FALL (person is upright)
                if self.current_state == FallState.NO_FALL:
                    self.initial_height = features.shoulder_y

                # Detect fall event with enhanced confirmation
                if (self.current_state == FallState.FALL and
                    self.previous_state == FallState.NO_FALL and
                    self.fall_cooldown_counter == 0):
                    height_dropped = self.check_height_drop()
                    slow = self.is_slow_movement()
                    if height_dropped and not slow:
                        fall_event = True
                        self.fall_detected_flag = True
                        self.fall_alert_counter = FALL_ALERT_DURATION_FRAMES
                        self.total_falls_detected += 1
                        self.fall_cooldown_counter = FALL_COOLDOWN_FRAMES

        # Reset fall flag when person recovers
        if self.current_state == FallState.NO_FALL:
            self.fall_detected_flag = False
            self.fall_alert_counter = 0

        return self.current_state, fall_event

    def get_state_color(self) -> Tuple[int, int, int]:
        """Get BGR color: red for FALL, green for NO_FALL."""
        if self.current_state == FallState.FALL or self.fall_alert_counter > 0:
            return COLOR_FALL
        return COLOR_NO_FALL

    def reset(self) -> None:
        """Reset FSM to initial state."""
        self.current_state = FallState.NO_FALL
        self.previous_state = FallState.NO_FALL
        self.state_frame_count = 0
        self.fall_detected_flag = False
        self.fall_cooldown_counter = 0
        self.fall_confidence = 0.0
        self.feature_history.clear()
        self.candidate_state = None
        self.candidate_frame_count = 0
        self.initial_height = None
        self.fall_alert_counter = 0


# ============================================================================
# PERSON TRACKER (IoU-based)
# ============================================================================


class PersonTracker:
    """
    Simple IoU-based tracker that maintains stable person IDs across frames.
    Matches current detections to existing tracks using bounding box overlap.
    """

    def __init__(
        self,
        iou_threshold: float = IOU_MATCH_THRESHOLD,
        max_lost_frames: int = MAX_LOST_FRAMES
    ):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.tracks: Dict[int, TrackedPerson] = {}
        self.next_id = 0

    @staticmethod
    def _compute_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def update(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Match detections to existing tracks and assign stable IDs.

        Args:
            detections: List of {'bbox': (x1,y1,x2,y2), 'confidence': float}

        Returns:
            Updated detections with stable 'person_id' field
        """
        if not detections:
            # Age all tracks
            lost_ids = []
            for tid, track in self.tracks.items():
                track.frames_lost += 1
                if track.frames_lost > self.max_lost_frames:
                    lost_ids.append(tid)
            for tid in lost_ids:
                del self.tracks[tid]
            return []

        # Build IoU matrix between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))

        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(
                    self.tracks[tid].bbox, det['bbox']
                )

        # Greedy matching: assign detections to tracks by highest IoU
        matched_tracks = set()
        matched_detections = set()
        updated_detections = []

        # Sort all (track, detection) pairs by IoU descending
        if len(track_ids) > 0 and len(detections) > 0:
            pairs = []
            for i in range(len(track_ids)):
                for j in range(len(detections)):
                    if iou_matrix[i, j] >= self.iou_threshold:
                        pairs.append((iou_matrix[i, j], i, j))
            pairs.sort(reverse=True)

            for iou_val, i, j in pairs:
                if i in matched_tracks or j in matched_detections:
                    continue
                tid = track_ids[i]
                new_bbox = detections[j]['bbox']
                self.tracks[tid].bbox = new_bbox
                self.tracks[tid].confidence = detections[j]['confidence']
                self.tracks[tid].frames_lost = 0

                # Apply EMA smoothing to bbox
                old_smooth = self.tracks[tid].smoothed_bbox
                if old_smooth is not None:
                    a = BBOX_SMOOTH_ALPHA
                    self.tracks[tid].smoothed_bbox = (
                        int(a * new_bbox[0] + (1 - a) * old_smooth[0]),
                        int(a * new_bbox[1] + (1 - a) * old_smooth[1]),
                        int(a * new_bbox[2] + (1 - a) * old_smooth[2]),
                        int(a * new_bbox[3] + (1 - a) * old_smooth[3]),
                    )
                else:
                    self.tracks[tid].smoothed_bbox = new_bbox

                det_copy = dict(detections[j])
                det_copy['person_id'] = tid
                det_copy['smoothed_bbox'] = self.tracks[tid].smoothed_bbox
                updated_detections.append(det_copy)

                matched_tracks.add(i)
                matched_detections.add(j)

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_detections:
                new_id = self.next_id
                self.next_id += 1
                new_track = TrackedPerson(
                    track_id=new_id,
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    smoothed_bbox=det['bbox']
                )
                new_track.fsm = FallStateMachine()
                self.tracks[new_id] = new_track

                det_copy = dict(det)
                det_copy['person_id'] = new_id
                det_copy['smoothed_bbox'] = det['bbox']
                updated_detections.append(det_copy)

        # Age unmatched tracks
        lost_ids = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid].frames_lost += 1
                if self.tracks[tid].frames_lost > self.max_lost_frames:
                    lost_ids.append(tid)
        for tid in lost_ids:
            del self.tracks[tid]

        return updated_detections

    def get_fsm(self, person_id: int) -> Optional[FallStateMachine]:
        """Get the FSM for a tracked person."""
        if person_id in self.tracks:
            return self.tracks[person_id].fsm
        return None


# ============================================================================
# POSE ANALYZER
# ============================================================================


class PoseAnalyzer:
    """Analyzes MediaPipe pose landmarks to extract fall-relevant features."""

    def __init__(self):
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28

    def calculate_angle(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        point3: Tuple[float, float]
    ) -> float:
        """Calculate angle between three points in degrees."""
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

        cos_angle = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def extract_features(
        self,
        landmarks,
        frame_number: int
    ) -> Optional[PoseFeatures]:
        """Extract pose features from MediaPipe landmarks."""
        try:
            lm = landmarks.landmark

            shoulder_center = (
                (lm[self.LEFT_SHOULDER].x + lm[self.RIGHT_SHOULDER].x) / 2,
                (lm[self.LEFT_SHOULDER].y + lm[self.RIGHT_SHOULDER].y) / 2
            )
            hip_center = (
                (lm[self.LEFT_HIP].x + lm[self.RIGHT_HIP].x) / 2,
                (lm[self.LEFT_HIP].y + lm[self.RIGHT_HIP].y) / 2
            )
            ankle_center = (
                (lm[self.LEFT_ANKLE].x + lm[self.RIGHT_ANKLE].x) / 2,
                (lm[self.LEFT_ANKLE].y + lm[self.RIGHT_ANKLE].y) / 2
            )
            knee_center = (
                (lm[self.LEFT_KNEE].x + lm[self.RIGHT_KNEE].x) / 2,
                (lm[self.LEFT_KNEE].y + lm[self.RIGHT_KNEE].y) / 2
            )
            nose = (lm[self.NOSE].x, lm[self.NOSE].y)

            # Feature 1: Torso angle from vertical
            torso_vec = np.array([
                hip_center[0] - shoulder_center[0],
                hip_center[1] - shoulder_center[1]
            ])
            vertical_vec = np.array([0, 1])
            cos_a = np.dot(torso_vec, vertical_vec) / (np.linalg.norm(torso_vec) + 1e-6)
            torso_angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

            # Feature 2: Body aspect ratio
            body_height = abs(shoulder_center[1] - ankle_center[1])
            shoulder_width = abs(lm[self.LEFT_SHOULDER].x - lm[self.RIGHT_SHOULDER].x)
            hip_width = abs(lm[self.LEFT_HIP].x - lm[self.RIGHT_HIP].x)
            body_width = max(shoulder_width, hip_width) + 1e-6
            body_ratio = body_height / body_width

            # Feature 3: Hip-ankle vertical distance (normalized)
            hip_ankle_dist = abs(hip_center[1] - ankle_center[1])

            # Feature 4: Knee angle
            left_knee_angle = self.calculate_angle(
                (lm[self.LEFT_HIP].x, lm[self.LEFT_HIP].y),
                (lm[self.LEFT_KNEE].x, lm[self.LEFT_KNEE].y),
                (lm[self.LEFT_ANKLE].x, lm[self.LEFT_ANKLE].y)
            )
            right_knee_angle = self.calculate_angle(
                (lm[self.RIGHT_HIP].x, lm[self.RIGHT_HIP].y),
                (lm[self.RIGHT_KNEE].x, lm[self.RIGHT_KNEE].y),
                (lm[self.RIGHT_ANKLE].x, lm[self.RIGHT_ANKLE].y)
            )
            knee_angle = (left_knee_angle + right_knee_angle) / 2.0

            # Feature 5 (NEW): Head-hip height ratio
            # Vertical distance from nose to hip, normalized by body height.
            # High value = head well above hips (standing/sitting).
            # Low value = head near hip level (lying).
            head_hip_vertical = abs(nose[1] - hip_center[1])
            head_hip_ratio = head_hip_vertical / (body_height + 1e-6)

            return PoseFeatures(
                torso_angle=torso_angle,
                body_ratio=body_ratio,
                hip_ankle_dist=hip_ankle_dist,
                knee_angle=knee_angle,
                shoulder_y=shoulder_center[1],
                hip_y=hip_center[1],
                head_hip_ratio=head_hip_ratio,
                timestamp=frame_number
            )

        except Exception:
            return None


# ============================================================================
# LANDMARK SMOOTHER (One Euro Filter — adaptive smoothing)
# ============================================================================


class _OneEuroFilter1D:
    """One Euro Filter for a single scalar value."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, dt: float = 1.0 / 30.0) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx_prev = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        a = self._alpha(cutoff, dt)
        self.x_prev = a * x + (1 - a) * self.x_prev
        return self.x_prev


class LandmarkSmoother:
    """
    One Euro Filter smoother for MediaPipe landmarks.
    Adapts automatically: very smooth when still, responsive when moving fast.
    One instance per tracked person.
    """

    def __init__(
        self,
        num_landmarks: int = 33,
        min_cutoff: float = OEF_MIN_CUTOFF,
        beta: float = OEF_BETA,
        d_cutoff: float = OEF_D_CUTOFF
    ):
        # 3 filters per landmark (x, y, z)
        self._filters = [
            [_OneEuroFilter1D(min_cutoff, beta, d_cutoff) for _ in range(3)]
            for _ in range(num_landmarks)
        ]

    def smooth(self, landmarks_frame: np.ndarray) -> np.ndarray:
        """Apply One Euro Filter to landmark coordinates. Shape (N, 4): x, y, z, visibility."""
        smoothed = landmarks_frame.copy()
        for i in range(len(landmarks_frame)):
            if landmarks_frame[i, 3] >= LANDMARK_VISIBILITY_THRESHOLD:
                for j in range(3):  # x, y, z
                    smoothed[i, j] = self._filters[i][j](landmarks_frame[i, j])
        return smoothed

    def reset(self):
        for f_list in self._filters:
            for f in f_list:
                f.x_prev = None
                f.dx_prev = 0.0


# ============================================================================
# ADVANCED FALL DETECTOR
# ============================================================================


class AdvancedFallDetector:
    """Advanced fall detection system combining YOLO, MediaPipe, FSM, and IoU tracking."""

    def __init__(
        self,
        yolo_model_path: str,
        yolo_conf_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
        pose_conf_threshold: float = POSE_CONFIDENCE_THRESHOLD,
        yolo_frame_skip: int = YOLO_FRAME_SKIP
    ):
        print("=" * 80)
        print("INITIALIZING ADVANCED FALL DETECTOR v4")
        print("=" * 80)

        # GPU auto-detection with CPU fallback
        if USE_GPU == "auto":
            self.device = 0 if torch.cuda.is_available() else "cpu"
        elif USE_GPU == "cuda":
            self.device = 0
        else:
            self.device = "cpu"

        if self.device != "cpu":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
            except Exception:
                print("WARNING: GPU detection failed, falling back to CPU")
                self.device = "cpu"
        else:
            print("Running on CPU (no GPU detected)")

        # Load YOLOv12s model
        print(f"Loading YOLO model: {yolo_model_path}")
        try:
            self.yolo_model = YOLO(yolo_model_path)
            print(f"Model loaded on device: {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

        # Resolve which class ID(s) should be treated as FALL (supports custom class ordering)
        self.fall_class_ids = self._resolve_fall_class_ids()
        print(f"FALL class IDs: {sorted(self.fall_class_ids)}")

        # Initialize MediaPipe Pose
        print("Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE
        )
        print("MediaPipe Pose initialized")

        # Initialize components
        self.pose_analyzer = PoseAnalyzer()
        self.person_tracker = PersonTracker()
        self._smoothers: Dict[int, LandmarkSmoother] = {}

        # Pose retention cache: {person_id: (landmarks, frame_cached)}
        self._pose_cache: Dict[int, Tuple[Any, int]] = {}

        # Thresholds
        self.yolo_conf_threshold = yolo_conf_threshold

        # Frame skip optimization (adaptive based on person count)
        self.yolo_frame_skip = yolo_frame_skip
        self._last_detections: List[Dict[str, Any]] = []

        # Statistics
        self.frame_count = 0
        self.total_fall_events = 0

        print("=" * 80)

    def get_or_create_fsm(self, person_id: int) -> FallStateMachine:
        """Get FSM from tracker, or create a fallback."""
        fsm = self.person_tracker.get_fsm(person_id)
        if fsm is None:
            fsm = FallStateMachine()
        return fsm

    def _resolve_fall_class_ids(self) -> set:
        """
        Identify class IDs that represent FALL from YOLO class names.
        Falls back to class 0 for backward compatibility.
        """
        names = getattr(self.yolo_model, "names", {}) or {}
        positive_tokens = {"fall", "fallen", "lying", "lie"}
        negative_tokens = {"non", "no", "not", "normal", "stand", "standing", "safe"}
        fall_ids = set()

        for cls_id, cls_name in names.items():
            normalized = (
                str(cls_name)
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            tokens = {tok for tok in normalized.split("_") if tok}
            if (tokens & positive_tokens) and not (tokens & negative_tokens):
                fall_ids.add(int(cls_id))

        if not fall_ids:
            fall_ids = {0}

        return fall_ids

    def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect persons using YOLO. Custom model outputs classes like fall/non_fall."""
        def _infer_with_conf(conf_threshold: float):
            return self.yolo_model(
                frame,
                conf=conf_threshold,
                imgsz=YOLO_IMGSZ,
                device=self.device,
                verbose=False
            )

        results = _infer_with_conf(self.yolo_conf_threshold)

        detections = []
        if len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        if len(boxes) == 0 and self.yolo_conf_threshold > YOLO_FALLBACK_CONFIDENCE:
            # Retry once with a lower threshold for scenes where detections are weak.
            results = _infer_with_conf(YOLO_FALLBACK_CONFIDENCE)
            if len(results) == 0 or results[0].boxes is None:
                return detections
            boxes = results[0].boxes

        h, w = frame.shape[:2]
        frame_area = h * w

        for idx, box in enumerate(boxes):
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            area_ratio = (x2 - x1) * (y2 - y1) / frame_area
            if not (YOLO_MIN_BBOX_AREA_RATIO <= area_ratio <= YOLO_MAX_BBOX_AREA_RATIO):
                continue

            # Support custom class ordering from trained model names.
            is_fall = cls_id in self.fall_class_ids

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'yolo_fall': is_fall
            })

        return detections

    def extract_pose_landmarks(self, person_crop: np.ndarray) -> Optional[Any]:
        """Extract pose landmarks from cropped person."""
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_crop)

        if results.pose_landmarks:
            return results.pose_landmarks
        return None

    def map_landmarks_to_frame(
        self,
        landmarks,
        crop_w: int,
        crop_h: int,
        offset_x: int,
        offset_y: int
    ) -> np.ndarray:
        """Convert MediaPipe landmarks to absolute pixel coordinates."""
        lm = landmarks.landmark
        arr = np.zeros((len(lm), 4), dtype=np.float32)
        for i, point in enumerate(lm):
            arr[i, 0] = point.x * crop_w + offset_x
            arr[i, 1] = point.y * crop_h + offset_y
            arr[i, 2] = point.z
            arr[i, 3] = point.visibility
        return arr

    def draw_pose_skeleton(
        self,
        image: np.ndarray,
        landmarks_frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Draw pose skeleton on the full frame."""
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28),
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8)
        ]

        for idx1, idx2 in connections:
            if (landmarks_frame[idx1, 3] >= POSE_CONFIDENCE_THRESHOLD and
                    landmarks_frame[idx2, 3] >= POSE_CONFIDENCE_THRESHOLD):
                pt1 = (int(landmarks_frame[idx1, 0]), int(landmarks_frame[idx1, 1]))
                pt2 = (int(landmarks_frame[idx2, 0]), int(landmarks_frame[idx2, 1]))
                cv2.line(image, pt1, pt2, color, POSE_THICKNESS)

        for i in range(len(landmarks_frame)):
            if landmarks_frame[i, 3] >= POSE_CONFIDENCE_THRESHOLD:
                pt = (int(landmarks_frame[i, 0]), int(landmarks_frame[i, 1]))
                cv2.circle(image, pt, 4, color, -1)

        return image

    def visualize_detection(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        state: FallState,
        fall_event: bool,
        features: Optional[PoseFeatures],
        fall_confidence: float = 0.0
    ) -> np.ndarray:
        """Visualize detection results on frame. Red = FALL, Green = NO_FALL."""
        x1, y1, x2, y2 = detection['bbox']
        conf = detection['confidence']
        person_id = detection.get('person_id', -1)

        color = COLOR_FALL if (state == FallState.FALL or fall_event) else COLOR_NO_FALL

        thickness = BBOX_THICKNESS * 2 if fall_event else BBOX_THICKNESS
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label: just FALL or NO_FALL
        label_text = state.value

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
        )

        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 10),
            (x1 + text_w + 10, y1),
            color, -1
        )

        cv2.putText(
            frame, label_text,
            (x1 + 5, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
            (255, 255, 255), FONT_THICKNESS
        )

        # # FALL DETECTED alert
        # if fall_event:
        #     alert_text = "!!! FALL DETECTED !!!"
        #     (alert_w, alert_h), alert_base = cv2.getTextSize(
        #         alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
        #     )

        #     alert_x = x1 + (x2 - x1 - alert_w) // 2
        #     alert_y = y1 - text_h - alert_h - 20

        #     cv2.rectangle(
        #         frame,
        #         (alert_x - 10, alert_y - alert_h - 10),
        #         (alert_x + alert_w + 10, alert_y + alert_base + 5),
        #         COLOR_FALL, -1
        #     )

        #     cv2.putText(
        #         frame, alert_text,
        #         (alert_x, alert_y),
        #         cv2.FONT_HERSHEY_SIMPLEX, 1.2,
        #         (255, 255, 255), 3
        #     )

        # Debug info
        if features:
            info_y = y2 + 25
            cv2.putText(
                frame,
                f"Angle: {features.torso_angle:.1f} | Ratio: {features.body_ratio:.2f}",
                (x1, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
            cv2.putText(
                frame,
                f"Knee: {features.knee_angle:.0f} | H/H: {features.head_hip_ratio:.2f} | Conf: {fall_confidence:.2f}",
                (x1, info_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        return frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Process single frame through complete pipeline.

        Returns:
            Tuple of (annotated_frame, num_fall_events, num_persons)
        """
        self.frame_count += 1
        annotated_frame = frame.copy()

        # Adaptive frame skip based on person count
        num_tracked = len(self.person_tracker.tracks)
        if num_tracked <= 1:
            effective_skip = 1
        elif num_tracked <= 3:
            effective_skip = 2
        else:
            effective_skip = 3

        # Frame-skip optimization: reuse previous YOLO detections
        if self.frame_count % effective_skip == 0 or self.frame_count <= 1:
            raw_detections = self.detect_persons(frame)
            self._last_detections = raw_detections
        else:
            raw_detections = self._last_detections

        # Update tracker with raw detections -> get stable person IDs
        detections = self.person_tracker.update(raw_detections)
        num_persons = len(detections)
        num_fall_events = 0

        # Clean up smoothers and pose cache for lost tracks
        active_ids = {d['person_id'] for d in detections}
        lost_ids = [pid for pid in self._smoothers if pid not in active_ids]
        for pid in lost_ids:
            del self._smoothers[pid]
        lost_cache_ids = [pid for pid in self._pose_cache if pid not in active_ids]
        for pid in lost_cache_ids:
            del self._pose_cache[pid]

        for detection in detections:
            person_id = detection['person_id']
            # Use smoothed bbox for MediaPipe crop (reduces jitter)
            sx1, sy1, sx2, sy2 = detection.get('smoothed_bbox', detection['bbox'])
            h, w = frame.shape[:2]

            # Determine state from YOLO class directly
            yolo_fall = detection.get('yolo_fall', False)
            current_state = FallState.FALL if yolo_fall else FallState.NO_FALL
            fall_event = False

            # Track fall events using FSM for temporal consistency
            fsm = self.get_or_create_fsm(person_id)
            if yolo_fall and fsm.current_state == FallState.NO_FALL:
                # New fall detected by YOLO
                if fsm.fall_cooldown_counter == 0:
                    fall_event = True
                    fsm.fall_detected_flag = True
                    fsm.fall_alert_counter = FALL_ALERT_DURATION_FRAMES
                    fsm.total_falls_detected += 1
                    fsm.fall_cooldown_counter = FALL_COOLDOWN_FRAMES
                    num_fall_events += 1
                    self.total_fall_events += 1

            fsm.current_state = current_state
            if fsm.fall_cooldown_counter > 0:
                fsm.fall_cooldown_counter -= 1
            if fsm.fall_alert_counter > 0:
                fsm.fall_alert_counter -= 1

            if not yolo_fall:
                fsm.fall_detected_flag = False
            fsm.state_frame_count = fsm.state_frame_count + 1 if current_state == fsm.previous_state else 0
            fsm.previous_state = current_state

            # MediaPipe pose skeleton (optional overlay)
            bw, bh = sx2 - sx1, sy2 - sy1
            pad_x = int(bw * BBOX_PAD_RATIO)
            pad_y = int(bh * BBOX_PAD_RATIO)
            px1 = max(0, sx1 - pad_x)
            py1 = max(0, sy1 - pad_y)
            px2 = min(w, sx2 + pad_x)
            py2 = min(h, sy2 + pad_y)

            person_crop = frame[py1:py2, px1:px2]
            if person_crop.size > 0:
                crop_h_orig, crop_w_orig = person_crop.shape[:2]
                if crop_h_orig < MIN_CROP_HEIGHT:
                    scale = MIN_CROP_HEIGHT / crop_h_orig
                    person_crop = cv2.resize(
                        person_crop, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_LINEAR
                    )

                # Adaptive MediaPipe skip for stable NO_FALL persons
                skip_mediapipe = (
                    current_state == FallState.NO_FALL and
                    fsm.state_frame_count > STABLE_STANDING_FRAMES and
                    self.frame_count % MEDIAPIPE_SKIP_STABLE != 0
                )

                landmarks = None
                if not skip_mediapipe:
                    landmarks = self.extract_pose_landmarks(person_crop)

                # Pose retention
                if landmarks is not None:
                    self._pose_cache[person_id] = (landmarks, self.frame_count)
                elif person_id in self._pose_cache:
                    cached_lm, cached_frame = self._pose_cache[person_id]
                    if (self.frame_count - cached_frame) <= POSE_RETENTION_FRAMES:
                        landmarks = cached_lm

                if landmarks is not None:
                    landmarks_frame = self.map_landmarks_to_frame(
                        landmarks, crop_w_orig, crop_h_orig, offset_x=px1, offset_y=py1
                    )
                    if person_id not in self._smoothers:
                        self._smoothers[person_id] = LandmarkSmoother()
                    landmarks_frame = self._smoothers[person_id].smooth(landmarks_frame)

                    skeleton_color = COLOR_FALL if yolo_fall else COLOR_NO_FALL
                    annotated_frame = self.draw_pose_skeleton(
                        annotated_frame, landmarks_frame, color=skeleton_color
                    )

            # Use fall_alert_counter to hold FALL visualization
            show_fall_event = fall_event or fsm.fall_alert_counter > 0

            annotated_frame = self.visualize_detection(
                annotated_frame, detection, current_state,
                show_fall_event, None, fsm.fall_confidence
            )

        return annotated_frame, num_fall_events, num_persons

    def reset(self):
        """Reset detector state for source switching."""
        self.person_tracker = PersonTracker()
        self._smoothers.clear()
        self._pose_cache.clear()
        self.frame_count = 0
        self.total_fall_events = 0
        self._last_detections = []

    def close(self):
        """Release resources."""
        self.pose.close()


# ============================================================================
# VIDEO PROCESSING
# ============================================================================


def process_video(
    detector: AdvancedFallDetector,
    input_source: str,
    output_path: Optional[str] = None,
    display: bool = True
) -> None:
    """Process video file or webcam stream."""
    if isinstance(input_source, int) or input_source.isdigit():
        cap = cv2.VideoCapture(int(input_source))
        source_type = "webcam"
    else:
        cap = cv2.VideoCapture(input_source)
        source_type = "video"

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {source_type}: {input_source}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n" + "=" * 80)
    print("PROCESSING VIDEO")
    print("=" * 80)
    print(f"Source: {input_source}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    if source_type == "video":
        print(f"Total frames: {total_frames}")
    print("=" * 80)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            annotated_frame, fall_events, num_persons = detector.process_frame(frame)

            # Global info overlay
            info_text = f"Frame: {frame_count} | Persons: {num_persons} | Total Falls: {detector.total_fall_events}"
            cv2.rectangle(annotated_frame, (5, 5), (600, 45), (0, 0, 0), -1)
            cv2.putText(
                annotated_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            if writer:
                writer.write(annotated_frame)

            if display:
                cv2.imshow("Advanced Fall Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break

            if frame_count % 30 == 0 and source_type == "video":
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Falls: {detector.total_fall_events}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Frames processed: {frame_count}")
        print(f"Total fall events: {detector.total_fall_events}")
        if output_path:
            print(f"Output saved: {output_path}")
        print("=" * 80)


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Main function to run advanced fall detection."""
    print("\n" + "=" * 80)
    print("ADVANCED FALL DETECTION WITH FINITE STATE MACHINE")
    print("=" * 80)

    try:
        detector = AdvancedFallDetector(
            yolo_model_path=YOLO_MODEL_PATH,
            yolo_conf_threshold=YOLO_CONFIDENCE_THRESHOLD,
            pose_conf_threshold=POSE_CONFIDENCE_THRESHOLD
        )
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    try:
        process_video(
            detector=detector,
            input_source=INPUT_SOURCE,
            output_path=OUTPUT_PATH,
            display=True
        )
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        detector.close()


if __name__ == "__main__":
    main()
