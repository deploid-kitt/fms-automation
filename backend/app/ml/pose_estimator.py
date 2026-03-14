"""Pose estimation using MediaPipe BlazePose.

DEPRECATED: This module is maintained for backward compatibility.
Use app.ml.pose_models instead for multi-model support.
"""
import cv2
import numpy as np
from typing import Generator, Optional
from pathlib import Path
import logging
import time

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# MediaPipe landmark indices
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# Key joint indices for FMS analysis
class JointIdx:
    """Landmark indices for key joints."""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class PoseSmoother:
    """Smooth pose landmarks over time to reduce jitter."""
    
    def __init__(self, smoothing_factor: float = 0.5, num_landmarks: int = 33):
        """
        Initialize pose smoother.
        
        Args:
            smoothing_factor: Weight for previous frame (0=no smoothing, 1=max)
            num_landmarks: Number of pose landmarks
        """
        self.smoothing_factor = smoothing_factor
        self.num_landmarks = num_landmarks
        self.prev_landmarks: Optional[np.ndarray] = None
        
    def smooth(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply exponential moving average smoothing to landmarks.
        
        Args:
            landmarks: Current frame landmarks (33, 4) - x, y, z, visibility
            
        Returns:
            Smoothed landmarks
        """
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks.copy()
            return landmarks
        
        # Apply smoothing only to x, y, z (not visibility)
        smoothed = landmarks.copy()
        alpha = 1.0 - self.smoothing_factor
        
        for i in range(self.num_landmarks):
            # Only smooth if both current and previous have good visibility
            if landmarks[i, 3] > 0.5 and self.prev_landmarks[i, 3] > 0.5:
                smoothed[i, :3] = (
                    alpha * landmarks[i, :3] + 
                    self.smoothing_factor * self.prev_landmarks[i, :3]
                )
        
        self.prev_landmarks = smoothed.copy()
        return smoothed
    
    def reset(self):
        """Reset smoother state."""
        self.prev_landmarks = None


class PoseEstimator:
    """MediaPipe BlazePose wrapper for pose estimation.
    
    DEPRECATED: Use pose_models.MediaPipePoseEstimator instead.
    This class is maintained for backward compatibility.
    """
    
    def __init__(self, enable_smoothing: bool = True, for_live: bool = False):
        """
        Initialize pose estimator.
        
        Args:
            enable_smoothing: Enable landmark smoothing
            for_live: Optimize for live analysis (lower complexity)
        """
        if not HAS_MEDIAPIPE:
            raise ImportError("mediapipe is not installed")
        
        settings = get_settings()
        self.settings = settings
        self.mp_pose = mp.solutions.pose
        
        # Use lower model complexity for live analysis
        model_complexity = settings.pose_model_complexity
        if for_live and model_complexity > 1:
            model_complexity = 1  # Use "full" instead of "heavy" for real-time
            
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            smooth_landmarks=True,  # MediaPipe's built-in smoothing
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence
        )
        
        # Additional smoothing
        self.smoother = PoseSmoother(
            smoothing_factor=settings.pose_smoothing_factor
        ) if enable_smoothing and settings.pose_smoothing_enabled else None
        
        # Performance tracking
        self.last_process_time: float = 0
        self.avg_process_time: float = 0
        self.frame_count: int = 0
        
    def process_frame(
        self, 
        frame: np.ndarray,
        downscale: bool = False
    ) -> Optional[dict]:
        """
        Process a single frame and extract pose landmarks.
        
        Args:
            frame: BGR image from OpenCV
            downscale: Whether to downscale frame for faster processing
            
        Returns:
            Dictionary with landmarks data or None if no pose detected
        """
        start_time = time.perf_counter()
        
        # Optionally downscale for faster processing
        process_frame = frame
        if downscale:
            target_w = self.settings.live_process_width
            target_h = self.settings.live_process_height
            h, w = frame.shape[:2]
            if w > target_w or h > target_h:
                process_frame = cv2.resize(
                    frame, 
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR
                )
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        # Track processing time
        self.last_process_time = time.perf_counter() - start_time
        self.frame_count += 1
        # Exponential moving average
        self.avg_process_time = (
            0.9 * self.avg_process_time + 0.1 * self.last_process_time
            if self.avg_process_time > 0 else self.last_process_time
        )
        
        if not results.pose_landmarks:
            return None
            
        # Extract landmarks
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
        
        # Apply smoothing if enabled
        if self.smoother is not None:
            landmarks_array = landmarks_to_array(landmarks)
            smoothed_array = self.smoother.smooth(landmarks_array)
            landmarks = [
                {"x": float(lm[0]), "y": float(lm[1]), "z": float(lm[2]), "visibility": float(lm[3])}
                for lm in smoothed_array
            ]
        
        # Extract world landmarks (3D in meters)
        world_landmarks = None
        if results.pose_world_landmarks:
            world_landmarks = []
            for wlm in results.pose_world_landmarks.landmark:
                world_landmarks.append({
                    "x": wlm.x,
                    "y": wlm.y,
                    "z": wlm.z,
                    "visibility": wlm.visibility
                })
        
        return {
            "landmarks": landmarks,
            "world_landmarks": world_landmarks,
            "process_time_ms": self.last_process_time * 1000
        }
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        return {
            "avg_process_time_ms": self.avg_process_time * 1000,
            "last_process_time_ms": self.last_process_time * 1000,
            "frames_processed": self.frame_count,
            "estimated_max_fps": 1.0 / self.avg_process_time if self.avg_process_time > 0 else 0
        }
    
    def reset_smoother(self):
        """Reset pose smoother state."""
        if self.smoother:
            self.smoother.reset()
    
    def process_video(
        self, 
        video_path: Path,
        target_fps: Optional[int] = None,
        max_frames: Optional[int] = None
    ) -> Generator[dict, None, None]:
        """
        Process video file and yield pose data for each frame.
        
        Args:
            video_path: Path to video file
            target_fps: Downsample to this FPS (None = use original)
            max_frames: Maximum frames to process
            
        Yields:
            Dictionary with frame_number, timestamp, and landmarks
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame skip for target FPS
        frame_skip = 1
        if target_fps and fps > target_fps:
            frame_skip = int(fps / target_fps)
        
        frame_count = 0
        processed_count = 0
        
        logger.info(f"Processing video: {video_path.name}, {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames if downsampling
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Process frame
            pose_data = self.process_frame(frame)
            
            timestamp = frame_count / fps
            
            yield {
                "frame_number": frame_count,
                "timestamp": timestamp,
                "landmarks": pose_data["landmarks"] if pose_data else None,
                "world_landmarks": pose_data["world_landmarks"] if pose_data else None,
                "pose_detected": pose_data is not None
            }
            
            frame_count += 1
            processed_count += 1
            
            if max_frames and processed_count >= max_frames:
                break
        
        cap.release()
        logger.info(f"Processed {processed_count} frames from {video_path.name}")
    
    def close(self):
        """Release resources."""
        self.pose.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def landmarks_to_array(landmarks: list[dict]) -> np.ndarray:
    """Convert landmarks list to numpy array (33, 4) - x, y, z, visibility."""
    return np.array([[lm["x"], lm["y"], lm["z"], lm["visibility"]] for lm in landmarks])


def get_joint_position(landmarks: np.ndarray, joint_idx: int) -> np.ndarray:
    """Get 3D position of a joint."""
    return landmarks[joint_idx, :3]


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate angle at point b formed by points a, b, c.
    
    Returns angle in degrees.
    """
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


def calculate_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(a - b)
