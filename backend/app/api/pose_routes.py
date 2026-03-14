"""API routes for pose model management."""
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.ml.pose_models import (
    get_pose_model_registry,
    PoseModelPreferences as PrefsDataclass
)

router = APIRouter(prefix="/api/pose", tags=["pose-models"])


# Request/Response models
class PoseModelPreferences(BaseModel):
    """Pose model preferences."""
    live_model: str = "mediapipe-medium"
    upload_model: str = "rtmpose-large"
    fallback_model: str = "mediapipe-medium"
    auto_select: bool = True
    prefer_gpu: bool = False
    prefer_accuracy: bool = False
    model_settings: dict = {}


class BenchmarkRequest(BaseModel):
    """Request for benchmarking."""
    model_id: Optional[str] = None
    num_frames: int = 30


# Routes
@router.get("/models")
async def list_models():
    """
    List all available pose estimation models.
    
    Returns model information including:
    - Model ID and name
    - Backend (mediapipe, rtmpose, yolo)
    - Model size (nano, small, medium, large, heavy)
    - Availability status
    - Load status
    - Performance estimates
    """
    registry = get_pose_model_registry()
    models = registry.get_available_models()
    return [m.to_dict() for m in models]


@router.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed information about a specific model."""
    registry = get_pose_model_registry()
    
    models = registry.get_available_models()
    model = next((m for m in models if m.model_id == model_id), None)
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    result = model.to_dict()
    
    # Add stats if loaded
    if model.is_loaded:
        estimator = registry.get_model(model_id, load_if_needed=False)
        if estimator:
            result["stats"] = estimator.get_stats()
    
    return result


@router.post("/models/{model_id}/load")
async def load_model(model_id: str, background_tasks: BackgroundTasks):
    """
    Load a pose estimation model.
    
    This will download model weights if necessary and load the model
    into memory. Use warm_up=true to also warm up the model.
    """
    registry = get_pose_model_registry()
    
    # Check if model exists
    models = registry.get_available_models()
    model_info = next((m for m in models if m.model_id == model_id), None)
    
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    if not model_info.is_available:
        raise HTTPException(
            status_code=400, 
            detail=f"Model backend not available: {model_info.backend}"
        )
    
    # Load model (synchronous for now)
    model = registry.get_model(model_id, load_if_needed=True)
    
    if not model:
        raise HTTPException(status_code=500, detail="Failed to load model")
    
    # Warm up in background
    background_tasks.add_task(model.warm_up, 10)
    
    return {
        "status": "loaded",
        "model_id": model_id,
        "info": model.get_model_info()
    }


@router.post("/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a pose estimation model to free memory."""
    registry = get_pose_model_registry()
    
    if registry.unload_model(model_id):
        return {"status": "unloaded", "model_id": model_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to unload model")


@router.post("/models/{model_id}/benchmark")
async def benchmark_model(model_id: str, request: BenchmarkRequest):
    """
    Benchmark a model's performance.
    
    Runs the model on synthetic frames and measures processing time.
    """
    registry = get_pose_model_registry()
    
    result = registry.benchmark_model(model_id, request.num_frames)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/benchmark/all")
async def benchmark_all_models(request: BenchmarkRequest):
    """Benchmark all available models."""
    registry = get_pose_model_registry()
    results = registry.benchmark_all(request.num_frames)
    return {"benchmarks": results}


@router.get("/preferences")
async def get_preferences():
    """Get current pose model preferences."""
    registry = get_pose_model_registry()
    return registry.get_preferences().to_dict()


@router.put("/preferences")
async def update_preferences(prefs: PoseModelPreferences):
    """
    Update pose model preferences.
    
    Preferences control which models are used for live vs upload analysis,
    GPU usage, and per-model settings.
    """
    registry = get_pose_model_registry()
    
    # Convert Pydantic to dataclass
    prefs_dc = PrefsDataclass(
        live_model=prefs.live_model,
        upload_model=prefs.upload_model,
        fallback_model=prefs.fallback_model,
        auto_select=prefs.auto_select,
        prefer_gpu=prefs.prefer_gpu,
        prefer_accuracy=prefs.prefer_accuracy,
        model_settings=prefs.model_settings,
    )
    
    updated = registry.update_preferences(prefs_dc)
    return updated.to_dict()


@router.get("/recommend")
async def recommend_model(
    mode: str = "live",
    prefer_accuracy: bool = False
):
    """
    Get model recommendation for a use case.
    
    Args:
        mode: "live" for real-time analysis, "upload" for video upload
        prefer_accuracy: If true, prefer accuracy over speed
    
    Returns:
        Recommended model ID
    """
    registry = get_pose_model_registry()
    
    if mode not in ["live", "upload"]:
        raise HTTPException(status_code=400, detail="Mode must be 'live' or 'upload'")
    
    model_id = registry.recommend_model(mode, prefer_accuracy)
    
    # Get model info
    models = registry.get_available_models()
    model = next((m for m in models if m.model_id == model_id), None)
    
    return {
        "recommended_model": model_id,
        "mode": mode,
        "prefer_accuracy": prefer_accuracy,
        "model_info": model.to_dict() if model else None
    }


@router.get("/stats")
async def get_stats():
    """Get pose model registry statistics."""
    registry = get_pose_model_registry()
    return registry.get_stats()


@router.get("/backends")
async def get_backends():
    """Get information about available backends."""
    registry = get_pose_model_registry()
    
    backends = []
    
    # MediaPipe
    try:
        import mediapipe
        backends.append({
            "backend": "mediapipe",
            "available": True,
            "version": mediapipe.__version__,
            "description": "Google MediaPipe BlazePose"
        })
    except ImportError:
        backends.append({
            "backend": "mediapipe",
            "available": False,
            "description": "Not installed"
        })
    
    # ONNX Runtime (RTMPose)
    try:
        import onnxruntime as ort
        backends.append({
            "backend": "rtmpose",
            "available": True,
            "version": ort.__version__,
            "providers": ort.get_available_providers(),
            "description": "MMPose RTMPose via ONNX Runtime"
        })
    except ImportError:
        backends.append({
            "backend": "rtmpose",
            "available": False,
            "description": "onnxruntime not installed"
        })
    
    # Ultralytics YOLO
    try:
        import ultralytics
        import torch
        backends.append({
            "backend": "yolo",
            "available": True,
            "version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "description": "Ultralytics YOLO Pose"
        })
    except ImportError:
        backends.append({
            "backend": "yolo",
            "available": False,
            "description": "ultralytics not installed"
        })
    
    return {"backends": backends}
