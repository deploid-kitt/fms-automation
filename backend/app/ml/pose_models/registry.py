"""Pose model registry for managing multiple pose estimation backends."""
import threading
import time
from typing import Optional, Dict, List, Type
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import platform

from .base import (
    BasePoseEstimator,
    PoseEstimatorConfig,
    ModelBackend,
    ModelSize
)

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an available pose model."""
    model_id: str
    name: str
    description: str
    backend: str
    model_size: str
    supported_sizes: List[str]
    recommended_for: List[str]
    is_available: bool = True  # Dependencies installed
    is_loaded: bool = False
    estimated_fps: float = 0.0
    memory_mb: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "backend": self.backend,
            "model_size": self.model_size,
            "supported_sizes": self.supported_sizes,
            "recommended_for": self.recommended_for,
            "is_available": self.is_available,
            "is_loaded": self.is_loaded,
            "estimated_fps": self.estimated_fps,
            "memory_mb": self.memory_mb,
        }


@dataclass
class PoseModelPreferences:
    """User preferences for pose model selection."""
    live_model: str = "mediapipe-medium"
    upload_model: str = "rtmpose-large"
    fallback_model: str = "mediapipe-medium"
    
    # Auto-selection settings
    auto_select: bool = True
    prefer_gpu: bool = False
    prefer_accuracy: bool = False  # False = prefer speed
    
    # Per-model settings
    model_settings: Dict[str, dict] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "live_model": self.live_model,
            "upload_model": self.upload_model,
            "fallback_model": self.fallback_model,
            "auto_select": self.auto_select,
            "prefer_gpu": self.prefer_gpu,
            "prefer_accuracy": self.prefer_accuracy,
            "model_settings": self.model_settings,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PoseModelPreferences":
        return cls(
            live_model=data.get("live_model", "mediapipe-medium"),
            upload_model=data.get("upload_model", "rtmpose-large"),
            fallback_model=data.get("fallback_model", "mediapipe-medium"),
            auto_select=data.get("auto_select", True),
            prefer_gpu=data.get("prefer_gpu", False),
            prefer_accuracy=data.get("prefer_accuracy", False),
            model_settings=data.get("model_settings", {}),
        )


class PoseModelRegistry:
    """Registry for managing pose estimation models.
    
    Features:
    - Model discovery and availability checking
    - Dynamic model loading/unloading
    - Memory management
    - Benchmarking
    - Model caching
    """
    
    # Available model implementations
    MODEL_CLASSES: Dict[ModelBackend, Type[BasePoseEstimator]] = {}
    
    # Default model configs for each backend
    DEFAULT_CONFIGS: Dict[str, dict] = {
        "mediapipe-nano": {
            "backend": ModelBackend.MEDIAPIPE,
            "model_size": ModelSize.NANO,
        },
        "mediapipe-medium": {
            "backend": ModelBackend.MEDIAPIPE,
            "model_size": ModelSize.MEDIUM,
        },
        "mediapipe-large": {
            "backend": ModelBackend.MEDIAPIPE,
            "model_size": ModelSize.LARGE,
        },
        "rtmpose-nano": {
            "backend": ModelBackend.RTMPOSE,
            "model_size": ModelSize.NANO,
        },
        "rtmpose-small": {
            "backend": ModelBackend.RTMPOSE,
            "model_size": ModelSize.SMALL,
        },
        "rtmpose-medium": {
            "backend": ModelBackend.RTMPOSE,
            "model_size": ModelSize.MEDIUM,
        },
        "rtmpose-large": {
            "backend": ModelBackend.RTMPOSE,
            "model_size": ModelSize.LARGE,
        },
        "yolo-pose-nano": {
            "backend": ModelBackend.YOLO,
            "model_size": ModelSize.NANO,
        },
        "yolo-pose-small": {
            "backend": ModelBackend.YOLO,
            "model_size": ModelSize.SMALL,
        },
        "yolo-pose-medium": {
            "backend": ModelBackend.YOLO,
            "model_size": ModelSize.MEDIUM,
        },
        "yolo-pose-large": {
            "backend": ModelBackend.YOLO,
            "model_size": ModelSize.LARGE,
        },
        "yolo-pose-heavy": {
            "backend": ModelBackend.YOLO,
            "model_size": ModelSize.HEAVY,
        },
    }
    
    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize the registry."""
        self.models_dir = models_dir or Path("/app/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Loaded model cache
        self._loaded_models: Dict[str, BasePoseEstimator] = {}
        self._model_lock = threading.Lock()
        
        # Preferences
        self._preferences = PoseModelPreferences()
        self._preferences_path = self.models_dir / "pose_preferences.json"
        self._load_preferences()
        
        # Check available backends
        self._check_backends()
        
        # Benchmark results cache
        self._benchmarks: Dict[str, dict] = {}
    
    def _check_backends(self):
        """Check which backends are available."""
        # Check MediaPipe
        try:
            from .mediapipe_estimator import MediaPipePoseEstimator
            self.MODEL_CLASSES[ModelBackend.MEDIAPIPE] = MediaPipePoseEstimator
            logger.info("MediaPipe backend available")
        except ImportError:
            logger.warning("MediaPipe not available")
        
        # Check RTMPose
        try:
            from .rtmpose_estimator import RTMPoseEstimator
            import onnxruntime
            self.MODEL_CLASSES[ModelBackend.RTMPOSE] = RTMPoseEstimator
            logger.info("RTMPose backend available")
        except ImportError:
            logger.warning("RTMPose not available (missing onnxruntime)")
        
        # Check YOLO
        try:
            from .yolo_estimator import YOLOPoseEstimator
            from ultralytics import YOLO
            self.MODEL_CLASSES[ModelBackend.YOLO] = YOLOPoseEstimator
            logger.info("YOLO backend available")
        except ImportError:
            logger.warning("YOLO not available (missing ultralytics)")
    
    def _load_preferences(self):
        """Load preferences from file."""
        try:
            if self._preferences_path.exists():
                with open(self._preferences_path) as f:
                    data = json.load(f)
                    self._preferences = PoseModelPreferences.from_dict(data)
                    logger.info("Loaded pose model preferences")
        except Exception as e:
            logger.warning(f"Failed to load preferences: {e}")
    
    def _save_preferences(self):
        """Save preferences to file."""
        try:
            with open(self._preferences_path, "w") as f:
                json.dump(self._preferences.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save preferences: {e}")
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of all available pose models."""
        models = []
        
        for model_id, config in self.DEFAULT_CONFIGS.items():
            backend = config["backend"]
            size = config["model_size"]
            
            # Check if backend is available
            is_available = backend in self.MODEL_CLASSES
            
            # Get model class metadata
            if is_available:
                model_class = self.MODEL_CLASSES[backend]
                name = model_class.MODEL_NAME
                description = model_class.MODEL_DESCRIPTION
                supported_sizes = [s.value for s in model_class.SUPPORTED_SIZES]
                recommended_for = model_class.RECOMMENDED_FOR
            else:
                name = f"{backend.value.title()} Pose"
                description = f"{backend.value} backend not installed"
                supported_sizes = []
                recommended_for = []
            
            # Check if loaded
            is_loaded = model_id in self._loaded_models
            
            # Get benchmark data if available
            estimated_fps = self._benchmarks.get(model_id, {}).get("fps", 0)
            memory_mb = self._benchmarks.get(model_id, {}).get("memory_mb", 0)
            
            models.append(ModelInfo(
                model_id=model_id,
                name=name,
                description=description,
                backend=backend.value,
                model_size=size.value,
                supported_sizes=supported_sizes,
                recommended_for=recommended_for,
                is_available=is_available,
                is_loaded=is_loaded,
                estimated_fps=estimated_fps,
                memory_mb=memory_mb,
            ))
        
        return models
    
    def get_model(
        self, 
        model_id: str,
        load_if_needed: bool = True
    ) -> Optional[BasePoseEstimator]:
        """
        Get a pose estimation model by ID.
        
        Args:
            model_id: Model identifier (e.g., "mediapipe-medium")
            load_if_needed: Load model if not already loaded
            
        Returns:
            Pose estimator instance or None if not available
        """
        with self._model_lock:
            # Return cached model if loaded
            if model_id in self._loaded_models:
                return self._loaded_models[model_id]
            
            if not load_if_needed:
                return None
            
            # Load the model
            model = self._load_model(model_id)
            return model
    
    def _load_model(self, model_id: str) -> Optional[BasePoseEstimator]:
        """Load a model by ID."""
        if model_id not in self.DEFAULT_CONFIGS:
            logger.error(f"Unknown model: {model_id}")
            return None
        
        config_data = self.DEFAULT_CONFIGS[model_id]
        backend = config_data["backend"]
        
        if backend not in self.MODEL_CLASSES:
            logger.error(f"Backend not available: {backend}")
            return None
        
        # Create config
        model_settings = self._preferences.model_settings.get(model_id, {})
        config = PoseEstimatorConfig(
            model_id=model_id,
            backend=backend,
            model_size=config_data["model_size"],
            model_path=self.models_dir / backend.value,
            use_gpu=self._preferences.prefer_gpu,
            **model_settings
        )
        
        # Create and load model
        model_class = self.MODEL_CLASSES[backend]
        model = model_class(config)
        
        if model.load_model():
            self._loaded_models[model_id] = model
            logger.info(f"Loaded model: {model_id}")
            return model
        else:
            logger.error(f"Failed to load model: {model_id}")
            return None
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model to free memory."""
        with self._model_lock:
            if model_id not in self._loaded_models:
                return True
            
            model = self._loaded_models[model_id]
            if model.unload_model():
                del self._loaded_models[model_id]
                logger.info(f"Unloaded model: {model_id}")
                return True
            return False
    
    def unload_all(self) -> int:
        """Unload all models."""
        count = 0
        with self._model_lock:
            for model_id in list(self._loaded_models.keys()):
                if self.unload_model(model_id):
                    count += 1
        return count
    
    def get_model_for_mode(self, mode: str = "live") -> Optional[BasePoseEstimator]:
        """
        Get the appropriate model for a use mode.
        
        Args:
            mode: "live" or "upload"
            
        Returns:
            Best available model for the mode
        """
        if mode == "live":
            model_id = self._preferences.live_model
        else:
            model_id = self._preferences.upload_model
        
        model = self.get_model(model_id)
        
        # Fall back if primary not available
        if model is None:
            model = self.get_model(self._preferences.fallback_model)
        
        return model
    
    def get_preferences(self) -> PoseModelPreferences:
        """Get current preferences."""
        return self._preferences
    
    def update_preferences(self, prefs: PoseModelPreferences) -> PoseModelPreferences:
        """Update and save preferences."""
        self._preferences = prefs
        self._save_preferences()
        return self._preferences
    
    def benchmark_model(
        self, 
        model_id: str,
        num_frames: int = 30
    ) -> dict:
        """
        Benchmark a model's performance.
        
        Args:
            model_id: Model to benchmark
            num_frames: Number of frames for benchmark
            
        Returns:
            Benchmark results dict
        """
        import numpy as np
        
        model = self.get_model(model_id)
        if model is None:
            return {"error": f"Model not available: {model_id}"}
        
        # Warm up
        model.warm_up(5)
        
        # Create test frame
        test_frame = np.random.randint(
            0, 255, 
            (480, 640, 3), 
            dtype=np.uint8
        )
        
        # Benchmark
        times = []
        for _ in range(num_frames):
            start = time.perf_counter()
            model._process_frame_impl(test_frame)
            times.append((time.perf_counter() - start) * 1000)
        
        results = {
            "model_id": model_id,
            "num_frames": num_frames,
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "fps": 1000 / (sum(times) / len(times)),
            "memory_mb": 0,  # TODO: measure memory usage
        }
        
        # Cache results
        self._benchmarks[model_id] = results
        
        return results
    
    def benchmark_all(self, num_frames: int = 30) -> List[dict]:
        """Benchmark all available models."""
        results = []
        for model_info in self.get_available_models():
            if model_info.is_available:
                result = self.benchmark_model(model_info.model_id, num_frames)
                results.append(result)
        return results
    
    def recommend_model(
        self,
        mode: str = "live",
        prefer_accuracy: bool = False
    ) -> str:
        """
        Recommend the best model for given requirements.
        
        Args:
            mode: "live" or "upload"
            prefer_accuracy: Prefer accuracy over speed
            
        Returns:
            Recommended model ID
        """
        available = [m for m in self.get_available_models() if m.is_available]
        
        if not available:
            return "mediapipe-medium"  # Default fallback
        
        # Filter by recommended use
        suitable = [m for m in available if mode in m.recommended_for]
        if not suitable:
            suitable = available
        
        # Sort by criteria
        if prefer_accuracy or mode == "upload":
            # Prefer larger models
            suitable.sort(key=lambda m: (
                -["nano", "small", "medium", "large", "heavy"].index(m.model_size) 
                if m.model_size in ["nano", "small", "medium", "large", "heavy"] 
                else 0
            ))
        else:
            # Prefer faster models
            suitable.sort(key=lambda m: (
                ["nano", "small", "medium", "large", "heavy"].index(m.model_size)
                if m.model_size in ["nano", "small", "medium", "large", "heavy"]
                else 10
            ))
        
        return suitable[0].model_id if suitable else "mediapipe-medium"
    
    def get_stats(self) -> dict:
        """Get registry statistics."""
        available = self.get_available_models()
        return {
            "total_models": len(available),
            "available_models": len([m for m in available if m.is_available]),
            "loaded_models": len(self._loaded_models),
            "loaded_model_ids": list(self._loaded_models.keys()),
            "backends": [b.value for b in self.MODEL_CLASSES.keys()],
            "preferences": self._preferences.to_dict(),
        }


# Global registry instance
_registry: Optional[PoseModelRegistry] = None


def get_pose_model_registry(
    models_dir: Optional[Path] = None
) -> PoseModelRegistry:
    """Get or create the global pose model registry."""
    global _registry
    if _registry is None:
        _registry = PoseModelRegistry(models_dir)
    return _registry
