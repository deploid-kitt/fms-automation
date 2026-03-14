"""RTMPose implementation for FMS analysis.

RTMPose (Real-Time Multi-person Pose estimation) from MMPose.
Validated on FMS research as providing accurate pose estimation for
movement assessment applications.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import logging

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from .base import (
    BasePoseEstimator,
    PoseEstimatorConfig,
    PoseLandmarks,
    ModelSize,
    ModelBackend,
    convert_coco_to_mediapipe
)

logger = logging.getLogger(__name__)


# RTMPose model configurations
RTMPOSE_MODELS = {
    ModelSize.NANO: {
        "name": "rtmpose-t",
        "url": "https://github.com/open-mmlab/mmpose/releases/download/v1.1.0/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.onnx",
        "input_size": (192, 256),
        "description": "Tiny model, fastest inference"
    },
    ModelSize.SMALL: {
        "name": "rtmpose-s",
        "url": "https://github.com/open-mmlab/mmpose/releases/download/v1.1.0/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx",
        "input_size": (192, 256),
        "description": "Small model, good speed/accuracy"
    },
    ModelSize.MEDIUM: {
        "name": "rtmpose-m",
        "url": "https://github.com/open-mmlab/mmpose/releases/download/v1.1.0/rtmpose-m_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.onnx",
        "input_size": (192, 256),
        "description": "Medium model, balanced"
    },
    ModelSize.LARGE: {
        "name": "rtmpose-l",
        "url": "https://github.com/open-mmlab/mmpose/releases/download/v1.1.0/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.onnx",
        "input_size": (288, 384),
        "description": "Large model, best accuracy"
    },
}


class RTMPoseEstimator(BasePoseEstimator):
    """RTMPose estimator using ONNX Runtime.
    
    RTMPose is specifically validated for FMS analysis in research papers.
    It provides excellent accuracy for clinical movement assessment while
    maintaining reasonable inference speed.
    
    Features:
    - SimCC (Simple Coordinate Classification) for better keypoint precision
    - Multi-scale feature fusion
    - Validated on FMS movement patterns
    - 17 COCO keypoints output (converted to 33 MediaPipe format)
    """
    
    MODEL_NAME = "RTMPose"
    MODEL_DESCRIPTION = (
        "MMPose's real-time pose estimation. Research-validated for FMS "
        "analysis with excellent accuracy for clinical movement assessment. "
        "Best choice for upload mode where accuracy matters most."
    )
    SUPPORTED_SIZES = [ModelSize.NANO, ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE]
    RECOMMENDED_FOR = ["upload", "accuracy"]
    
    def __init__(self, config: PoseEstimatorConfig):
        """Initialize RTMPose estimator."""
        if not HAS_ONNX:
            raise ImportError("onnxruntime is not installed")
        
        super().__init__(config)
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_size = RTMPOSE_MODELS.get(
            config.model_size, 
            RTMPOSE_MODELS[ModelSize.MEDIUM]
        )["input_size"]
    
    def _get_model_path(self) -> Path:
        """Get or download model file."""
        models_dir = self.config.model_path or Path("/app/models/rtmpose")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_info = RTMPOSE_MODELS.get(
            self.config.model_size,
            RTMPOSE_MODELS[ModelSize.MEDIUM]
        )
        
        model_name = model_info["name"]
        model_path = models_dir / f"{model_name}.onnx"
        
        # Download if not exists
        if not model_path.exists():
            self._download_model(model_info["url"], model_path)
        
        return model_path
    
    def _download_model(self, url: str, path: Path):
        """Download model from URL."""
        import urllib.request
        
        logger.info(f"Downloading RTMPose model from {url}")
        try:
            urllib.request.urlretrieve(url, path)
            logger.info(f"Downloaded model to {path}")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    def load_model(self) -> bool:
        """Load RTMPose ONNX model."""
        try:
            model_path = self._get_model_path()
            
            # Configure ONNX Runtime
            providers = []
            if self.config.use_gpu:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            self.is_loaded = True
            provider_used = self.session.get_providers()[0]
            logger.info(
                f"Loaded RTMPose ({self.config.model_size.value}) "
                f"with {provider_used}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RTMPose: {e}")
            return False
    
    def unload_model(self) -> bool:
        """Unload RTMPose model."""
        try:
            self.session = None
            self.is_loaded = False
            logger.info("Unloaded RTMPose")
            return True
        except Exception as e:
            logger.error(f"Failed to unload RTMPose: {e}")
            return False
    
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Preprocess frame for RTMPose."""
        h, w = frame.shape[:2]
        input_w, input_h = self.input_size
        
        # Center and scale for person detection
        # For simplicity, assume single person centered in frame
        center = np.array([w / 2, h / 2])
        scale = np.array([w, h])
        
        # Resize and normalize
        img = cv2.resize(frame, (input_w, input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # HWC to CHW, add batch dim
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        return img, {"center": center, "scale": scale, "original_size": (w, h)}
    
    def _postprocess(
        self, 
        outputs: list,
        meta: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess RTMPose outputs to keypoints."""
        # RTMPose SimCC outputs: simcc_x, simcc_y
        simcc_x = outputs[0]  # (1, 17, W_x)
        simcc_y = outputs[1]  # (1, 17, W_y)
        
        # Get keypoint predictions from SimCC
        x_locs = np.argmax(simcc_x, axis=2)  # (1, 17)
        y_locs = np.argmax(simcc_y, axis=2)  # (1, 17)
        
        # Get confidence scores
        x_scores = np.max(simcc_x, axis=2)
        y_scores = np.max(simcc_y, axis=2)
        scores = (x_scores + y_scores) / 2
        
        # Convert to normalized coordinates
        input_w, input_h = self.input_size
        orig_w, orig_h = meta["original_size"]
        
        # SimCC uses 2x spatial resolution
        x_coords = x_locs[0] / (simcc_x.shape[2] / 2)  # Normalize to [0, 1]
        y_coords = y_locs[0] / (simcc_y.shape[2] / 2)
        
        # Adjust for aspect ratio
        x_coords = x_coords / (input_w / orig_w) if input_w != orig_w else x_coords
        y_coords = y_coords / (input_h / orig_h) if input_h != orig_h else y_coords
        
        # Clamp to valid range
        x_coords = np.clip(x_coords, 0, 1)
        y_coords = np.clip(y_coords, 0, 1)
        
        # Build keypoints array (17, 2)
        keypoints = np.stack([x_coords, y_coords], axis=1)
        confidences = scores[0]
        
        return keypoints, confidences
    
    def _process_frame_impl(self, frame: np.ndarray) -> Optional[PoseLandmarks]:
        """Process frame with RTMPose."""
        # Preprocess
        input_tensor, meta = self._preprocess(frame)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )
        
        # Postprocess
        keypoints, confidences = self._postprocess(outputs, meta)
        
        # Check if detection is valid
        avg_confidence = np.mean(confidences)
        if avg_confidence < self.config.detection_confidence:
            return None
        
        # Convert COCO 17 keypoints to MediaPipe 33 format
        # Add z-coordinate (set to 0 for 2D)
        keypoints_3d = np.zeros((17, 3), dtype=np.float32)
        keypoints_3d[:, :2] = keypoints
        
        landmarks = convert_coco_to_mediapipe(keypoints_3d, confidences)
        
        return PoseLandmarks(
            landmarks=landmarks,
            world_landmarks=None,  # RTMPose doesn't provide world coordinates
            confidence=float(avg_confidence)
        )


def get_rtmpose_config(
    model_size: str = "medium",
    for_live: bool = False,
    **kwargs
) -> PoseEstimatorConfig:
    """
    Create RTMPose configuration.
    
    Args:
        model_size: "nano", "small", "medium", or "large"
        for_live: Optimize for live analysis (not recommended for RTMPose)
        **kwargs: Additional config overrides
        
    Returns:
        PoseEstimatorConfig for RTMPose
    """
    size = ModelSize(model_size.lower())
    
    # For live, prefer smaller models
    if for_live and size in [ModelSize.LARGE, ModelSize.HEAVY]:
        size = ModelSize.SMALL
        logger.info("RTMPose: Using smaller model for live analysis")
    
    model_info = RTMPOSE_MODELS.get(size, RTMPOSE_MODELS[ModelSize.MEDIUM])
    
    config = PoseEstimatorConfig(
        model_id=f"rtmpose-{size.value}",
        backend=ModelBackend.RTMPOSE,
        model_size=size,
        model_path=kwargs.get("model_path", Path("/app/models/rtmpose")),
        detection_confidence=kwargs.get("detection_confidence", 0.5),
        tracking_confidence=kwargs.get("tracking_confidence", 0.5),
        enable_smoothing=kwargs.get("enable_smoothing", True),
        smoothing_factor=kwargs.get("smoothing_factor", 0.4),
        input_width=model_info["input_size"][0],
        input_height=model_info["input_size"][1],
        use_gpu=kwargs.get("use_gpu", False),
    )
    
    return config
