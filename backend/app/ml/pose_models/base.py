"""Base class for pose estimation models."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Generator, Any
import numpy as np
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)


class ModelSize(str, Enum):
    """Model size variants."""
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    FULL = "full"
    HEAVY = "heavy"


class ModelBackend(str, Enum):
    """Supported inference backends."""
    MEDIAPIPE = "mediapipe"
    RTMPOSE = "rtmpose"
    YOLO = "yolo"


@dataclass
class PoseEstimatorConfig:
    """Configuration for pose estimator."""
    # Model identification
    model_id: str
    backend: ModelBackend
    model_size: ModelSize = ModelSize.MEDIUM
    
    # Model paths
    model_path: Optional[Path] = None
    config_path: Optional[Path] = None
    
    # Detection thresholds
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    
    # Processing options
    use_gpu: bool = False
    half_precision: bool = False  # FP16 inference
    
    # Smoothing options
    enable_smoothing: bool = True
    smoothing_factor: float = 0.5
    
    # Performance tuning
    batch_size: int = 1
    input_width: int = 640
    input_height: int = 480
    
    # Model-specific options
    extra_options: dict = field(default_factory=dict)


@dataclass
class PoseLandmarks:
    """Standardized pose landmark output format.
    
    All models output landmarks in this standardized format with 33 landmarks
    for compatibility with MediaPipe's BlazePose convention.
    """
    # Raw landmark data: (33, 4) - x, y, z, visibility
    landmarks: np.ndarray
    
    # World coordinates in meters (if available)
    world_landmarks: Optional[np.ndarray] = None
    
    # Detection confidence
    confidence: float = 0.0
    
    # Processing time in milliseconds
    process_time_ms: float = 0.0
    
    # Model that generated this
    model_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary format for API."""
        landmarks = [
            {
                "x": float(lm[0]),
                "y": float(lm[1]),
                "z": float(lm[2]),
                "visibility": float(lm[3])
            }
            for lm in self.landmarks
        ]
        
        world_landmarks = None
        if self.world_landmarks is not None:
            world_landmarks = [
                {
                    "x": float(wlm[0]),
                    "y": float(wlm[1]),
                    "z": float(wlm[2]),
                    "visibility": float(wlm[3])
                }
                for wlm in self.world_landmarks
            ]
        
        return {
            "landmarks": landmarks,
            "world_landmarks": world_landmarks,
            "confidence": self.confidence,
            "process_time_ms": self.process_time_ms,
            "model_id": self.model_id
        }


# MediaPipe BlazePose landmark indices (33 landmarks)
# This is our standardized output format
MEDIAPIPE_LANDMARK_NAMES = [
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


# COCO keypoint indices (17 keypoints)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Mapping from COCO (17) to MediaPipe (33)
# -1 means no direct mapping (will be interpolated)
COCO_TO_MEDIAPIPE = {
    0: 0,    # nose -> nose
    1: 2,    # left_eye -> left_eye
    2: 5,    # right_eye -> right_eye
    3: 7,    # left_ear -> left_ear
    4: 8,    # right_ear -> right_ear
    5: 11,   # left_shoulder -> left_shoulder
    6: 12,   # right_shoulder -> right_shoulder
    7: 13,   # left_elbow -> left_elbow
    8: 14,   # right_elbow -> right_elbow
    9: 15,   # left_wrist -> left_wrist
    10: 16,  # right_wrist -> right_wrist
    11: 23,  # left_hip -> left_hip
    12: 24,  # right_hip -> right_hip
    13: 25,  # left_knee -> left_knee
    14: 26,  # right_knee -> right_knee
    15: 27,  # left_ankle -> left_ankle
    16: 28,  # right_ankle -> right_ankle
}


def convert_coco_to_mediapipe(
    coco_keypoints: np.ndarray,
    confidences: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert COCO 17-keypoint format to MediaPipe 33-landmark format.
    
    Args:
        coco_keypoints: (17, 2) or (17, 3) array of keypoint coordinates
        confidences: Optional (17,) array of confidence scores
        
    Returns:
        (33, 4) array in MediaPipe format (x, y, z, visibility)
    """
    # Initialize output with zeros
    mp_landmarks = np.zeros((33, 4), dtype=np.float32)
    
    # Copy directly mapped keypoints
    for coco_idx, mp_idx in COCO_TO_MEDIAPIPE.items():
        if coco_keypoints.shape[1] >= 3:
            mp_landmarks[mp_idx, :3] = coco_keypoints[coco_idx, :3]
        else:
            mp_landmarks[mp_idx, :2] = coco_keypoints[coco_idx, :2]
            mp_landmarks[mp_idx, 2] = 0  # z = 0 for 2D
        
        if confidences is not None:
            mp_landmarks[mp_idx, 3] = confidences[coco_idx]
        else:
            mp_landmarks[mp_idx, 3] = 1.0
    
    # Interpolate missing landmarks
    # Face landmarks (eyes inner/outer, mouth)
    left_eye = mp_landmarks[2, :3]
    right_eye = mp_landmarks[5, :3]
    nose = mp_landmarks[0, :3]
    
    # Eye inner/outer (interpolate from eye and nose)
    mp_landmarks[1, :3] = 0.7 * left_eye + 0.3 * nose  # left_eye_inner
    mp_landmarks[3, :3] = 1.3 * left_eye - 0.3 * nose  # left_eye_outer
    mp_landmarks[4, :3] = 0.7 * right_eye + 0.3 * nose  # right_eye_inner
    mp_landmarks[6, :3] = 1.3 * right_eye - 0.3 * nose  # right_eye_outer
    
    # Mouth (between nose and midpoint of shoulders)
    shoulder_mid = (mp_landmarks[11, :3] + mp_landmarks[12, :3]) / 2
    mouth_y = 0.3 * nose + 0.7 * shoulder_mid
    mp_landmarks[9, :3] = mouth_y + (left_eye - right_eye) * 0.1  # mouth_left
    mp_landmarks[10, :3] = mouth_y - (left_eye - right_eye) * 0.1  # mouth_right
    
    # Hand landmarks (wrist extensions)
    left_wrist = mp_landmarks[15, :3]
    left_elbow = mp_landmarks[13, :3]
    right_wrist = mp_landmarks[16, :3]
    right_elbow = mp_landmarks[14, :3]
    
    wrist_extension_left = left_wrist - left_elbow
    wrist_extension_right = right_wrist - right_elbow
    
    mp_landmarks[17, :3] = left_wrist + wrist_extension_left * 0.15  # left_pinky
    mp_landmarks[19, :3] = left_wrist + wrist_extension_left * 0.12  # left_index
    mp_landmarks[21, :3] = left_wrist + wrist_extension_left * 0.08  # left_thumb
    mp_landmarks[18, :3] = right_wrist + wrist_extension_right * 0.15  # right_pinky
    mp_landmarks[20, :3] = right_wrist + wrist_extension_right * 0.12  # right_index
    mp_landmarks[22, :3] = right_wrist + wrist_extension_right * 0.08  # right_thumb
    
    # Foot landmarks (ankle extensions)
    left_ankle = mp_landmarks[27, :3]
    left_knee = mp_landmarks[25, :3]
    right_ankle = mp_landmarks[28, :3]
    right_knee = mp_landmarks[26, :3]
    
    ankle_extension_left = left_ankle - left_knee
    ankle_extension_right = right_ankle - right_knee
    
    mp_landmarks[29, :3] = left_ankle + ankle_extension_left * 0.2  # left_heel
    mp_landmarks[31, :3] = left_ankle + ankle_extension_left * 0.3  # left_foot_index
    mp_landmarks[30, :3] = right_ankle + ankle_extension_right * 0.2  # right_heel
    mp_landmarks[32, :3] = right_ankle + ankle_extension_right * 0.3  # right_foot_index
    
    # Set visibility for interpolated landmarks (lower confidence)
    interpolated_indices = [1, 3, 4, 6, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32]
    for idx in interpolated_indices:
        mp_landmarks[idx, 3] = 0.5  # Lower visibility for interpolated
    
    return mp_landmarks


class BasePoseEstimator(ABC):
    """Abstract base class for pose estimation models."""
    
    # Class-level metadata
    MODEL_NAME: str = "base"
    MODEL_DESCRIPTION: str = "Base pose estimator"
    SUPPORTED_SIZES: list[ModelSize] = []
    RECOMMENDED_FOR: list[str] = []  # "live", "upload", "accuracy", "speed"
    
    def __init__(self, config: PoseEstimatorConfig):
        """Initialize pose estimator."""
        self.config = config
        self.is_loaded = False
        self.is_warming_up = False
        
        # Performance tracking
        self.frame_count = 0
        self.total_process_time = 0.0
        self.avg_process_time = 0.0
        
        # Smoother state
        self._prev_landmarks: Optional[np.ndarray] = None
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the model into memory.
        
        Returns:
            True if loaded successfully
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> bool:
        """
        Unload the model from memory.
        
        Returns:
            True if unloaded successfully
        """
        pass
    
    @abstractmethod
    def _process_frame_impl(self, frame: np.ndarray) -> Optional[PoseLandmarks]:
        """
        Internal frame processing implementation.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            PoseLandmarks or None if no pose detected
        """
        pass
    
    def process_frame(
        self, 
        frame: np.ndarray,
        downscale: bool = False
    ) -> Optional[PoseLandmarks]:
        """
        Process a single frame and extract pose landmarks.
        
        Args:
            frame: BGR image from OpenCV
            downscale: Whether to downscale for faster processing
            
        Returns:
            PoseLandmarks or None if no pose detected
        """
        if not self.is_loaded:
            logger.warning(f"Model {self.config.model_id} not loaded")
            return None
        
        start_time = time.perf_counter()
        
        # Optionally downscale
        process_frame = frame
        if downscale:
            h, w = frame.shape[:2]
            target_w = self.config.input_width
            target_h = self.config.input_height
            if w > target_w or h > target_h:
                import cv2
                process_frame = cv2.resize(
                    frame,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR
                )
        
        # Process frame
        result = self._process_frame_impl(process_frame)
        
        # Track timing
        process_time_ms = (time.perf_counter() - start_time) * 1000
        self.frame_count += 1
        self.total_process_time += process_time_ms
        self.avg_process_time = self.total_process_time / self.frame_count
        
        if result is not None:
            result.process_time_ms = process_time_ms
            result.model_id = self.config.model_id
            
            # Apply smoothing if enabled
            if self.config.enable_smoothing and self._prev_landmarks is not None:
                result.landmarks = self._smooth_landmarks(result.landmarks)
            
            self._prev_landmarks = result.landmarks.copy()
        
        return result
    
    def _smooth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply exponential moving average smoothing."""
        if self._prev_landmarks is None:
            return landmarks
        
        alpha = 1.0 - self.config.smoothing_factor
        smoothed = landmarks.copy()
        
        for i in range(33):
            if landmarks[i, 3] > 0.5 and self._prev_landmarks[i, 3] > 0.5:
                smoothed[i, :3] = (
                    alpha * landmarks[i, :3] +
                    self.config.smoothing_factor * self._prev_landmarks[i, :3]
                )
        
        return smoothed
    
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
            target_fps: Downsample to this FPS
            max_frames: Maximum frames to process
            
        Yields:
            Dictionary with frame_number, timestamp, and landmarks
        """
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame skip
        frame_skip = 1
        if target_fps and fps > target_fps:
            frame_skip = int(fps / target_fps)
        
        frame_count = 0
        processed_count = 0
        
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
                "landmarks": pose_data.to_dict()["landmarks"] if pose_data else None,
                "world_landmarks": pose_data.to_dict()["world_landmarks"] if pose_data else None,
                "pose_detected": pose_data is not None
            }
            
            frame_count += 1
            processed_count += 1
            
            if max_frames and processed_count >= max_frames:
                break
        
        cap.release()
    
    def warm_up(self, num_frames: int = 10) -> float:
        """
        Warm up the model with dummy frames.
        
        Args:
            num_frames: Number of warm-up frames
            
        Returns:
            Average warm-up frame time in ms
        """
        if not self.is_loaded:
            self.load_model()
        
        self.is_warming_up = True
        dummy_frame = np.zeros(
            (self.config.input_height, self.config.input_width, 3),
            dtype=np.uint8
        )
        
        times = []
        for _ in range(num_frames):
            start = time.perf_counter()
            self._process_frame_impl(dummy_frame)
            times.append((time.perf_counter() - start) * 1000)
        
        self.is_warming_up = False
        return sum(times) / len(times) if times else 0
    
    def reset_state(self):
        """Reset smoother and internal state."""
        self._prev_landmarks = None
        self.frame_count = 0
        self.total_process_time = 0.0
        self.avg_process_time = 0.0
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        return {
            "model_id": self.config.model_id,
            "backend": self.config.backend.value,
            "model_size": self.config.model_size.value,
            "is_loaded": self.is_loaded,
            "frames_processed": self.frame_count,
            "avg_process_time_ms": self.avg_process_time,
            "estimated_fps": 1000 / self.avg_process_time if self.avg_process_time > 0 else 0,
            "use_gpu": self.config.use_gpu,
        }
    
    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_id": self.config.model_id,
            "name": self.MODEL_NAME,
            "description": self.MODEL_DESCRIPTION,
            "backend": self.config.backend.value,
            "model_size": self.config.model_size.value,
            "supported_sizes": [s.value for s in self.SUPPORTED_SIZES],
            "recommended_for": self.RECOMMENDED_FOR,
            "is_loaded": self.is_loaded,
        }
    
    def close(self):
        """Clean up resources."""
        if self.is_loaded:
            self.unload_model()
    
    def __enter__(self):
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
