"""Exercise classifier for FMS test detection."""
import numpy as np
from typing import Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

from app.ml.pose_estimator import JointIdx, landmarks_to_array
from app.ml.angle_calculator import AngleCalculator

logger = logging.getLogger(__name__)


class ExercisePhase(str, Enum):
    """Movement phases within an exercise."""
    SETUP = "setup"
    DESCENT = "descent"
    BOTTOM = "bottom"
    ASCENT = "ascent"
    TOP = "top"
    HOLD = "hold"
    UNKNOWN = "unknown"


@dataclass
class ExerciseDetection:
    """Result of exercise classification."""
    exercise: str
    confidence: float
    phase: ExercisePhase
    side: Optional[str] = None  # "left", "right", or None for bilateral


class RuleBasedClassifier:
    """
    Rule-based classifier for FMS exercises.
    
    Uses geometric features and temporal patterns to classify exercises.
    This is a bootstrapping approach - can be replaced with trained LSTM.
    """
    
    def __init__(self):
        self.angle_calc = AngleCalculator()
        self.frame_buffer = []
        self.buffer_size = 30  # ~1 second at 30fps
        
    def classify_frame(
        self, 
        landmarks: np.ndarray,
        prev_landmarks: Optional[np.ndarray] = None
    ) -> ExerciseDetection:
        """
        Classify exercise from a single frame with optional temporal context.
        
        Args:
            landmarks: Current frame landmarks (33, 4)
            prev_landmarks: Previous frame landmarks for motion detection
            
        Returns:
            ExerciseDetection with predicted exercise and confidence
        """
        metrics = self.angle_calc.calculate_all_metrics(landmarks)
        
        # Calculate key indicators
        is_standing = self._is_standing(landmarks)
        is_squatting = self._is_squatting(landmarks, metrics)
        is_lying = self._is_lying(landmarks)
        is_prone = self._is_prone(landmarks)
        arms_position = self._get_arms_position(landmarks, metrics)
        
        # Rule-based classification
        if is_lying:
            # Could be ASLR
            leg_raise = self._detect_leg_raise(landmarks, metrics)
            if leg_raise:
                return ExerciseDetection(
                    exercise=f"aslr_{leg_raise}",
                    confidence=0.8,
                    phase=ExercisePhase.HOLD,
                    side=leg_raise
                )
            return ExerciseDetection(
                exercise="unknown",
                confidence=0.3,
                phase=ExercisePhase.UNKNOWN
            )
        
        if is_prone:
            # Could be push-up or rotary stability
            if self._is_pushup_position(landmarks):
                return ExerciseDetection(
                    exercise="trunk_stability_pushup",
                    confidence=0.75,
                    phase=self._get_pushup_phase(landmarks, metrics)
                )
            return ExerciseDetection(
                exercise="rotary_stability",
                confidence=0.6,
                phase=ExercisePhase.HOLD
            )
        
        if is_squatting:
            # Deep squat
            squat_metrics = self.angle_calc.calculate_deep_squat_metrics(landmarks)
            phase = self._get_squat_phase(squat_metrics)
            return ExerciseDetection(
                exercise="deep_squat",
                confidence=0.85,
                phase=phase
            )
        
        if is_standing:
            # Check for hurdle step or inline lunge
            if arms_position == "overhead":
                # Could be starting position for deep squat
                return ExerciseDetection(
                    exercise="deep_squat",
                    confidence=0.7,
                    phase=ExercisePhase.SETUP
                )
            
            # Check for single-leg stance (hurdle step)
            single_leg = self._detect_single_leg_stance(landmarks)
            if single_leg:
                return ExerciseDetection(
                    exercise=f"hurdle_step_{single_leg}",
                    confidence=0.75,
                    phase=ExercisePhase.HOLD,
                    side=single_leg
                )
            
            # Check for lunge position
            lunge = self._detect_lunge(landmarks, metrics)
            if lunge:
                return ExerciseDetection(
                    exercise=f"inline_lunge_{lunge}",
                    confidence=0.7,
                    phase=ExercisePhase.BOTTOM,
                    side=lunge
                )
            
            # Check for shoulder mobility (hands behind back)
            if self._is_shoulder_mobility(landmarks):
                side = self._get_shoulder_mobility_side(landmarks)
                return ExerciseDetection(
                    exercise=f"shoulder_mobility_{side}",
                    confidence=0.7,
                    phase=ExercisePhase.HOLD,
                    side=side
                )
        
        return ExerciseDetection(
            exercise="unknown",
            confidence=0.2,
            phase=ExercisePhase.UNKNOWN
        )
    
    def classify_sequence(
        self, 
        landmarks_sequence: list[np.ndarray],
        min_frames: int = 15
    ) -> list[Tuple[str, int, int, float]]:
        """
        Classify exercise segments from a sequence of frames.
        
        Args:
            landmarks_sequence: List of (33, 4) landmark arrays
            min_frames: Minimum frames for a valid segment
            
        Returns:
            List of (exercise_name, start_frame, end_frame, confidence)
        """
        if len(landmarks_sequence) < min_frames:
            return []
        
        segments = []
        current_exercise = None
        start_frame = 0
        confidences = []
        
        for i, landmarks in enumerate(landmarks_sequence):
            detection = self.classify_frame(
                landmarks,
                landmarks_sequence[i-1] if i > 0 else None
            )
            
            if detection.exercise != current_exercise:
                # Save previous segment if long enough
                if current_exercise and i - start_frame >= min_frames:
                    avg_conf = np.mean(confidences) if confidences else 0.5
                    segments.append((
                        current_exercise,
                        start_frame,
                        i - 1,
                        avg_conf
                    ))
                
                # Start new segment
                current_exercise = detection.exercise
                start_frame = i
                confidences = [detection.confidence]
            else:
                confidences.append(detection.confidence)
        
        # Add final segment
        if current_exercise and len(landmarks_sequence) - start_frame >= min_frames:
            avg_conf = np.mean(confidences) if confidences else 0.5
            segments.append((
                current_exercise,
                start_frame,
                len(landmarks_sequence) - 1,
                avg_conf
            ))
        
        return segments
    
    def _is_standing(self, landmarks: np.ndarray) -> bool:
        """Check if person is in standing position."""
        hip_y = (landmarks[JointIdx.LEFT_HIP, 1] + landmarks[JointIdx.RIGHT_HIP, 1]) / 2
        ankle_y = (landmarks[JointIdx.LEFT_ANKLE, 1] + landmarks[JointIdx.RIGHT_ANKLE, 1]) / 2
        shoulder_y = (landmarks[JointIdx.LEFT_SHOULDER, 1] + landmarks[JointIdx.RIGHT_SHOULDER, 1]) / 2
        
        # In image coords, Y increases downward
        # Standing: shoulders above hips above ankles
        return shoulder_y < hip_y < ankle_y and (ankle_y - shoulder_y) > 0.3
    
    def _is_squatting(self, landmarks: np.ndarray, metrics) -> bool:
        """Check if person is in squat position."""
        knee_flexion = (metrics.knee_flexion_left + metrics.knee_flexion_right) / 2
        return knee_flexion > 60  # Significant knee bend
    
    def _is_lying(self, landmarks: np.ndarray) -> bool:
        """Check if person is lying on back."""
        hip_y = (landmarks[JointIdx.LEFT_HIP, 1] + landmarks[JointIdx.RIGHT_HIP, 1]) / 2
        shoulder_y = (landmarks[JointIdx.LEFT_SHOULDER, 1] + landmarks[JointIdx.RIGHT_SHOULDER, 1]) / 2
        
        # Small vertical distance between hip and shoulder = lying down
        return abs(hip_y - shoulder_y) < 0.15
    
    def _is_prone(self, landmarks: np.ndarray) -> bool:
        """Check if person is in prone position (face down)."""
        # Similar to lying but check arm positions for push-up setup
        is_flat = self._is_lying(landmarks)
        wrist_y = (landmarks[JointIdx.LEFT_WRIST, 1] + landmarks[JointIdx.RIGHT_WRIST, 1]) / 2
        shoulder_y = (landmarks[JointIdx.LEFT_SHOULDER, 1] + landmarks[JointIdx.RIGHT_SHOULDER, 1]) / 2
        
        # In prone push-up, wrists are typically near or below shoulders
        return is_flat and wrist_y >= shoulder_y - 0.1
    
    def _get_arms_position(self, landmarks: np.ndarray, metrics) -> str:
        """Classify arm position: overhead, neutral, behind."""
        avg_elevation = (metrics.arm_elevation_left + metrics.arm_elevation_right) / 2
        
        if avg_elevation > 150:
            return "overhead"
        elif avg_elevation > 90:
            return "elevated"
        elif avg_elevation > 45:
            return "neutral"
        else:
            return "behind"
    
    def _detect_single_leg_stance(self, landmarks: np.ndarray) -> Optional[str]:
        """Detect if one leg is raised (hurdle step)."""
        l_knee_y = landmarks[JointIdx.LEFT_KNEE, 1]
        r_knee_y = landmarks[JointIdx.RIGHT_KNEE, 1]
        l_hip_y = landmarks[JointIdx.LEFT_HIP, 1]
        r_hip_y = landmarks[JointIdx.RIGHT_HIP, 1]
        
        # Check if one knee is significantly higher than the other
        knee_diff = abs(l_knee_y - r_knee_y)
        
        if knee_diff > 0.15:  # Significant asymmetry
            if l_knee_y < r_knee_y:
                return "left"  # Left leg raised
            else:
                return "right"
        return None
    
    def _detect_lunge(self, landmarks: np.ndarray, metrics) -> Optional[str]:
        """Detect lunge position and which leg is forward."""
        l_knee_flex = metrics.knee_flexion_left
        r_knee_flex = metrics.knee_flexion_right
        
        l_ankle_z = landmarks[JointIdx.LEFT_ANKLE, 2]
        r_ankle_z = landmarks[JointIdx.RIGHT_ANKLE, 2]
        
        # Lunge: one leg forward (lower Z), significant knee flexion
        if abs(l_ankle_z - r_ankle_z) > 0.1:
            if l_ankle_z < r_ankle_z and l_knee_flex > 60:
                return "left"
            elif r_ankle_z < l_ankle_z and r_knee_flex > 60:
                return "right"
        return None
    
    def _detect_leg_raise(self, landmarks: np.ndarray, metrics) -> Optional[str]:
        """Detect which leg is raised in ASLR."""
        l_hip_flex = metrics.hip_flexion_left
        r_hip_flex = metrics.hip_flexion_right
        
        if l_hip_flex > 45 and l_hip_flex > r_hip_flex + 20:
            return "left"
        elif r_hip_flex > 45 and r_hip_flex > l_hip_flex + 20:
            return "right"
        return None
    
    def _is_pushup_position(self, landmarks: np.ndarray) -> bool:
        """Check if in push-up position."""
        # Hands and feet on ground, body elevated
        wrist_y = (landmarks[JointIdx.LEFT_WRIST, 1] + landmarks[JointIdx.RIGHT_WRIST, 1]) / 2
        ankle_y = (landmarks[JointIdx.LEFT_ANKLE, 1] + landmarks[JointIdx.RIGHT_ANKLE, 1]) / 2
        hip_y = (landmarks[JointIdx.LEFT_HIP, 1] + landmarks[JointIdx.RIGHT_HIP, 1]) / 2
        
        # Hands and feet roughly level, hip slightly above or level
        return abs(wrist_y - ankle_y) < 0.2 and hip_y < max(wrist_y, ankle_y)
    
    def _is_shoulder_mobility(self, landmarks: np.ndarray) -> bool:
        """Check if performing shoulder mobility test."""
        l_wrist = landmarks[JointIdx.LEFT_WRIST]
        r_wrist = landmarks[JointIdx.RIGHT_WRIST]
        l_shoulder = landmarks[JointIdx.LEFT_SHOULDER]
        r_shoulder = landmarks[JointIdx.RIGHT_SHOULDER]
        mid_spine_y = (landmarks[JointIdx.LEFT_HIP, 1] + landmarks[JointIdx.LEFT_SHOULDER, 1]) / 2
        
        # One wrist above shoulder (reaching up), one below mid-spine (reaching down)
        l_above = l_wrist[1] < l_shoulder[1]
        r_above = r_wrist[1] < r_shoulder[1]
        l_below = l_wrist[1] > mid_spine_y
        r_below = r_wrist[1] > mid_spine_y
        
        return (l_above and r_below) or (r_above and l_below)
    
    def _get_shoulder_mobility_side(self, landmarks: np.ndarray) -> str:
        """Determine which side is reaching up in shoulder mobility."""
        l_wrist_y = landmarks[JointIdx.LEFT_WRIST, 1]
        r_wrist_y = landmarks[JointIdx.RIGHT_WRIST, 1]
        
        if l_wrist_y < r_wrist_y:
            return "left"
        return "right"
    
    def _get_squat_phase(self, squat_metrics: dict) -> ExercisePhase:
        """Determine squat phase from metrics."""
        knee_flex = squat_metrics["knee_flexion_avg"]
        
        if knee_flex < 30:
            return ExercisePhase.TOP
        elif knee_flex < 60:
            return ExercisePhase.DESCENT
        elif squat_metrics["hip_below_knee"]:
            return ExercisePhase.BOTTOM
        else:
            return ExercisePhase.DESCENT
    
    def _get_pushup_phase(self, landmarks: np.ndarray, metrics) -> ExercisePhase:
        """Determine push-up phase."""
        elbow_angle = (
            180 - (landmarks[JointIdx.LEFT_SHOULDER, :3] @ 
                   landmarks[JointIdx.LEFT_ELBOW, :3])
        )
        
        # Simplified: check elbow flexion
        l_elbow = landmarks[JointIdx.LEFT_ELBOW]
        r_elbow = landmarks[JointIdx.RIGHT_ELBOW]
        l_shoulder = landmarks[JointIdx.LEFT_SHOULDER]
        r_shoulder = landmarks[JointIdx.RIGHT_SHOULDER]
        
        elbow_below_shoulder = (l_elbow[1] > l_shoulder[1] + 0.05 or 
                                 r_elbow[1] > r_shoulder[1] + 0.05)
        
        if elbow_below_shoulder:
            return ExercisePhase.BOTTOM
        return ExercisePhase.TOP
