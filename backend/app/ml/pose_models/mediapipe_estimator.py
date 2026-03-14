"""MediaPipe BlazePose implementation."""
import cv2
import numpy as np
from typing import Optional
import logging

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

from .base import (
    BasePoseEstimator, 
    PoseEstimatorConfig, 
    PoseLandmarks,
    ModelSize,
    ModelBackend
)

logger = logging.getLogger(__name__)


class MediaPipePoseEstimator(BasePoseEstimator):
    """MediaPipe BlazePose pose estimator.
    
    The original and most widely used pose estimation model. Provides
    excellent real-time performance with good accuracy for most use cases.
    
    Model complexities:
    - Lite (0): Fastest, lower accuracy
    - Full (1): Balanced speed/accuracy
    - Heavy (2): Best accuracy, slower
    """
    
    MODEL_NAME = "MediaPipe BlazePose"
    MODEL_DESCRIPTION = (
        "Google's real-time pose estimation. Optimized for mobile and web "
        "with excellent speed-accuracy tradeoff. Native 33-landmark output."
    )
    SUPPORTED_SIZES = [ModelSize.NANO, ModelSize.MEDIUM, ModelSize.LARGE]
    RECOMMENDED_FOR = ["live", "speed"]
    
    # Map our sizes to MediaPipe complexity
    SIZE_TO_COMPLEXITY = {
        ModelSize.NANO: 0,   # Lite
        ModelSize.SMALL: 0,  # Lite
        ModelSize.MEDIUM: 1, # Full
        ModelSize.LARGE: 2,  # Heavy
        ModelSize.FULL: 1,   # Full
        ModelSize.HEAVY: 2,  # Heavy
    }
    
    def __init__(self, config: PoseEstimatorConfig):
        """Initialize MediaPipe pose estimator."""
        if not HAS_MEDIAPIPE:
            raise ImportError("mediapipe is not installed")
        
        super().__init__(config)
        self.mp_pose = mp.solutions.pose
        self.pose = None
    
    def load_model(self) -> bool:
        """Load MediaPipe pose model."""
        try:
            complexity = self.SIZE_TO_COMPLEXITY.get(
                self.config.model_size, 
                1  # Default to Full
            )
            
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=complexity,
                enable_segmentation=False,
                smooth_landmarks=True,
                min_detection_confidence=self.config.detection_confidence,
                min_tracking_confidence=self.config.tracking_confidence
            )
            
            self.is_loaded = True
            logger.info(
                f"Loaded MediaPipe BlazePose (complexity={complexity}, "
                f"size={self.config.model_size.value})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MediaPipe: {e}")
            return False
    
    def unload_model(self) -> bool:
        """Unload MediaPipe model."""
        try:
            if self.pose:
                self.pose.close()
                self.pose = None
            self.is_loaded = False
            logger.info("Unloaded MediaPipe BlazePose")
            return True
        except Exception as e:
            logger.error(f"Failed to unload MediaPipe: {e}")
            return False
    
    def _process_frame_impl(self, frame: np.ndarray) -> Optional[PoseLandmarks]:
        """Process frame with MediaPipe."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks (already in 33-landmark format)
        landmarks = np.zeros((33, 4), dtype=np.float32)
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]
        
        # Calculate overall confidence
        confidence = float(np.mean(landmarks[:, 3]))
        
        # Extract world landmarks if available
        world_landmarks = None
        if results.pose_world_landmarks:
            world_landmarks = np.zeros((33, 4), dtype=np.float32)
            for i, wlm in enumerate(results.pose_world_landmarks.landmark):
                world_landmarks[i] = [wlm.x, wlm.y, wlm.z, wlm.visibility]
        
        return PoseLandmarks(
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            confidence=confidence
        )


def get_mediapipe_config(
    model_size: str = "medium",
    for_live: bool = True,
    **kwargs
) -> PoseEstimatorConfig:
    """
    Create MediaPipe configuration.
    
    Args:
        model_size: "nano", "medium", or "large"
        for_live: Optimize for live analysis
        **kwargs: Additional config overrides
        
    Returns:
        PoseEstimatorConfig for MediaPipe
    """
    size = ModelSize(model_size.lower())
    
    # Optimize settings for live analysis
    if for_live and size == ModelSize.LARGE:
        size = ModelSize.MEDIUM  # Don't use Heavy for live
    
    config = PoseEstimatorConfig(
        model_id=f"mediapipe-{size.value}",
        backend=ModelBackend.MEDIAPIPE,
        model_size=size,
        detection_confidence=kwargs.get("detection_confidence", 0.5),
        tracking_confidence=kwargs.get("tracking_confidence", 0.5),
        enable_smoothing=kwargs.get("enable_smoothing", True),
        smoothing_factor=kwargs.get("smoothing_factor", 0.5),
        input_width=kwargs.get("input_width", 640 if for_live else 1280),
        input_height=kwargs.get("input_height", 480 if for_live else 720),
    )
    
    return config
