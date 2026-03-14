"""YOLO11 Pose estimation implementation.

YOLOv8/v11 Pose provides efficient pose estimation with multiple model sizes
for different speed/accuracy tradeoffs.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import logging

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

from .base import (
    BasePoseEstimator,
    PoseEstimatorConfig,
    PoseLandmarks,
    ModelSize,
    ModelBackend,
    convert_coco_to_mediapipe
)

logger = logging.getLogger(__name__)


# YOLO Pose model configurations
YOLO_POSE_MODELS = {
    ModelSize.NANO: {
        "name": "yolo11n-pose",
        "model_file": "yolo11n-pose.pt",
        "description": "Nano - Fastest, good for live analysis",
        "input_size": 640,
    },
    ModelSize.SMALL: {
        "name": "yolo11s-pose",
        "model_file": "yolo11s-pose.pt",
        "description": "Small - Balanced speed/accuracy",
        "input_size": 640,
    },
    ModelSize.MEDIUM: {
        "name": "yolo11m-pose",
        "model_file": "yolo11m-pose.pt",
        "description": "Medium - Good accuracy, moderate speed",
        "input_size": 640,
    },
    ModelSize.LARGE: {
        "name": "yolo11l-pose",
        "model_file": "yolo11l-pose.pt",
        "description": "Large - High accuracy",
        "input_size": 640,
    },
    ModelSize.HEAVY: {
        "name": "yolo11x-pose",
        "model_file": "yolo11x-pose.pt",
        "description": "Extra Large - Best accuracy, slowest",
        "input_size": 640,
    },
}


class YOLOPoseEstimator(BasePoseEstimator):
    """YOLO11 Pose estimator.
    
    YOLOv11 Pose provides state-of-the-art object detection combined with
    pose estimation. Multiple model sizes allow trading off between
    speed and accuracy.
    
    Features:
    - Single-shot detection and pose estimation
    - Multiple model sizes (nano to extra-large)
    - GPU acceleration support
    - Multi-person detection (uses highest confidence)
    - 17 COCO keypoints output (converted to 33 MediaPipe format)
    """
    
    MODEL_NAME = "YOLO11 Pose"
    MODEL_DESCRIPTION = (
        "Ultralytics YOLO pose estimation. Offers multiple model sizes "
        "from nano (fastest) to extra-large (most accurate). Good for "
        "both live and upload modes depending on model size selected."
    )
    SUPPORTED_SIZES = [ModelSize.NANO, ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE, ModelSize.HEAVY]
    RECOMMENDED_FOR = ["live", "speed", "accuracy"]
    
    def __init__(self, config: PoseEstimatorConfig):
        """Initialize YOLO pose estimator."""
        if not HAS_ULTRALYTICS:
            raise ImportError("ultralytics is not installed")
        
        super().__init__(config)
        self.model = None
        self.model_info = YOLO_POSE_MODELS.get(
            config.model_size,
            YOLO_POSE_MODELS[ModelSize.MEDIUM]
        )
    
    def _get_model_path(self) -> str:
        """Get model name/path for YOLO."""
        # YOLO models are downloaded automatically by ultralytics
        # Just return the model name
        return self.model_info["model_file"]
    
    def load_model(self) -> bool:
        """Load YOLO pose model."""
        try:
            model_name = self._get_model_path()
            
            # Check for custom model path
            if self.config.model_path:
                custom_path = self.config.model_path / model_name
                if custom_path.exists():
                    model_name = str(custom_path)
            
            # Load model
            self.model = YOLO(model_name)
            
            # Configure device
            if self.config.use_gpu:
                self.device = "cuda"
            else:
                self.device = "cpu"
            
            # Move model to device
            self.model.to(self.device)
            
            # Enable half precision if requested
            if self.config.half_precision and self.config.use_gpu:
                self.model.model.half()
            
            self.is_loaded = True
            logger.info(
                f"Loaded YOLO11 Pose ({self.config.model_size.value}) "
                f"on {self.device}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            return False
    
    def unload_model(self) -> bool:
        """Unload YOLO model."""
        try:
            if self.model:
                del self.model
                self.model = None
            self.is_loaded = False
            logger.info("Unloaded YOLO11 Pose")
            return True
        except Exception as e:
            logger.error(f"Failed to unload YOLO: {e}")
            return False
    
    def _process_frame_impl(self, frame: np.ndarray) -> Optional[PoseLandmarks]:
        """Process frame with YOLO pose."""
        # Run inference
        results = self.model(
            frame,
            conf=self.config.detection_confidence,
            verbose=False,
            imgsz=self.model_info["input_size"],
        )
        
        # Get pose results
        if not results or len(results) == 0:
            return None
        
        result = results[0]
        
        # Check for keypoints
        if result.keypoints is None or len(result.keypoints) == 0:
            return None
        
        # Get keypoints from best detection (highest confidence)
        keypoints = result.keypoints
        
        if keypoints.conf is None:
            return None
        
        # Find best person detection
        confidences = keypoints.conf.cpu().numpy()
        if len(confidences) == 0:
            return None
        
        # Average confidence across keypoints for each person
        person_confidences = np.mean(confidences, axis=1)
        best_idx = np.argmax(person_confidences)
        
        # Get keypoints for best person
        kpts = keypoints.xyn[best_idx].cpu().numpy()  # Normalized xy
        kpts_conf = confidences[best_idx]
        
        # Check if we have valid keypoints
        avg_conf = float(np.mean(kpts_conf))
        if avg_conf < self.config.detection_confidence:
            return None
        
        # YOLO outputs 17 COCO keypoints
        # Format: (17, 2) normalized coordinates
        keypoints_3d = np.zeros((17, 3), dtype=np.float32)
        keypoints_3d[:, :2] = kpts
        
        # Convert to MediaPipe 33 landmark format
        landmarks = convert_coco_to_mediapipe(keypoints_3d, kpts_conf)
        
        return PoseLandmarks(
            landmarks=landmarks,
            world_landmarks=None,  # YOLO doesn't provide 3D world coordinates
            confidence=avg_conf
        )


def get_yolo_config(
    model_size: str = "small",
    for_live: bool = True,
    **kwargs
) -> PoseEstimatorConfig:
    """
    Create YOLO Pose configuration.
    
    Args:
        model_size: "nano", "small", "medium", "large", or "heavy"
        for_live: Optimize for live analysis
        **kwargs: Additional config overrides
        
    Returns:
        PoseEstimatorConfig for YOLO
    """
    # Map string to enum
    size_map = {
        "nano": ModelSize.NANO,
        "small": ModelSize.SMALL,
        "medium": ModelSize.MEDIUM,
        "large": ModelSize.LARGE,
        "heavy": ModelSize.HEAVY,
        "xlarge": ModelSize.HEAVY,
    }
    size = size_map.get(model_size.lower(), ModelSize.SMALL)
    
    # For live, prefer smaller models
    if for_live and size in [ModelSize.LARGE, ModelSize.HEAVY]:
        size = ModelSize.SMALL
        logger.info("YOLO: Using smaller model for live analysis")
    
    model_info = YOLO_POSE_MODELS.get(size, YOLO_POSE_MODELS[ModelSize.SMALL])
    
    config = PoseEstimatorConfig(
        model_id=f"yolo-pose-{size.value}",
        backend=ModelBackend.YOLO,
        model_size=size,
        model_path=kwargs.get("model_path", Path("/app/models/yolo")),
        detection_confidence=kwargs.get("detection_confidence", 0.5),
        tracking_confidence=kwargs.get("tracking_confidence", 0.5),
        enable_smoothing=kwargs.get("enable_smoothing", True),
        smoothing_factor=kwargs.get("smoothing_factor", 0.3),
        input_width=model_info["input_size"],
        input_height=model_info["input_size"],
        use_gpu=kwargs.get("use_gpu", False),
        half_precision=kwargs.get("half_precision", False),
    )
    
    return config


# Convenience functions for specific YOLO sizes
def get_yolo_nano_config(**kwargs) -> PoseEstimatorConfig:
    """Get YOLO11 Nano Pose config (fastest)."""
    return get_yolo_config("nano", for_live=True, **kwargs)


def get_yolo_small_config(**kwargs) -> PoseEstimatorConfig:
    """Get YOLO11 Small Pose config (balanced)."""
    return get_yolo_config("small", for_live=True, **kwargs)


def get_yolo_medium_config(**kwargs) -> PoseEstimatorConfig:
    """Get YOLO11 Medium Pose config."""
    return get_yolo_config("medium", for_live=False, **kwargs)


def get_yolo_large_config(**kwargs) -> PoseEstimatorConfig:
    """Get YOLO11 Large Pose config (accurate)."""
    return get_yolo_config("large", for_live=False, **kwargs)
