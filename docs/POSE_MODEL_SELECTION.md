# Pose Model Selection System

## Overview

The FMS Automation system now supports multiple pose estimation backends, allowing users to choose the best model for their use case (speed vs accuracy).

## Supported Models

### MediaPipe BlazePose
- **Backend:** `mediapipe`
- **Sizes:** nano, medium, large
- **Best for:** Live analysis, real-time feedback
- **Pros:** Fast, well-optimized, native 33-landmark output
- **Cons:** May miss some subtle movements

### RTMPose
- **Backend:** `rtmpose` (via ONNX Runtime)
- **Sizes:** nano, small, medium, large
- **Best for:** Upload analysis, FMS accuracy
- **Pros:** Research-validated for FMS, high accuracy
- **Cons:** Slower than MediaPipe, requires model download

### YOLO11 Pose
- **Backend:** `yolo` (via Ultralytics)
- **Sizes:** nano, small, medium, large, heavy
- **Best for:** Flexible use - multiple size options
- **Pros:** State-of-the-art detection, GPU acceleration
- **Cons:** Larger models are slower

## Model IDs

| Model ID | Backend | Size | Recommended For |
|----------|---------|------|-----------------|
| `mediapipe-nano` | MediaPipe | Nano | Live (fastest) |
| `mediapipe-medium` | MediaPipe | Medium | Live (default) |
| `mediapipe-large` | MediaPipe | Large | Upload |
| `rtmpose-nano` | RTMPose | Nano | Testing |
| `rtmpose-small` | RTMPose | Small | Live (high accuracy) |
| `rtmpose-medium` | RTMPose | Medium | Balanced |
| `rtmpose-large` | RTMPose | Large | Upload (default) |
| `yolo-pose-nano` | YOLO | Nano | Live |
| `yolo-pose-small` | YOLO | Small | Live |
| `yolo-pose-medium` | YOLO | Medium | Upload |
| `yolo-pose-large` | YOLO | Large | Upload |
| `yolo-pose-heavy` | YOLO | Heavy | Maximum accuracy |

## API Endpoints

### List Models
```
GET /api/pose/models
```
Returns all available pose models with their status.

### Get Model Info
```
GET /api/pose/models/{model_id}
```
Returns detailed information about a specific model.

### Load Model
```
POST /api/pose/models/{model_id}/load
```
Loads a model into memory.

### Unload Model
```
POST /api/pose/models/{model_id}/unload
```
Unloads a model from memory.

### Get Preferences
```
GET /api/pose/preferences
```
Returns current pose model preferences.

### Update Preferences
```
PUT /api/pose/preferences
```
Updates pose model preferences.

Example body:
```json
{
  "live_model": "mediapipe-medium",
  "upload_model": "rtmpose-large",
  "fallback_model": "mediapipe-medium",
  "auto_select": true,
  "prefer_gpu": false,
  "prefer_accuracy": false
}
```

### Get Recommendation
```
GET /api/pose/recommend?mode=live&prefer_accuracy=false
```
Returns the recommended model for a given use case.

### Benchmark
```
POST /api/pose/models/{model_id}/benchmark
POST /api/pose/benchmark/all
```
Benchmark model performance.

### Get Backends
```
GET /api/pose/backends
```
Returns information about installed backends.

## WebSocket Protocol

During live analysis, clients can switch models using WebSocket messages:

### Switch Model
```json
{"type": "switch_model", "model_id": "yolo-pose-small"}
```

Response:
```json
{"type": "model_switched", "success": true, "model_id": "yolo-pose-small"}
```

### List Models
```json
{"type": "list_models"}
```

Response:
```json
{"type": "models_list", "models": [...], "current_model": "mediapipe-medium"}
```

## Configuration

Environment variables:
```bash
FMS_POSE_MODEL_LIVE=mediapipe-medium
FMS_POSE_MODEL_UPLOAD=rtmpose-large
FMS_POSE_MODEL_FALLBACK=mediapipe-medium
FMS_POSE_PREFER_GPU=false
```

## Frontend Components

### PoseModelSettings
Full settings modal for configuring pose models.

### PoseModelQuickSelect
Compact dropdown for switching models during analysis.

### PoseModelIndicator
Status indicator showing current model with click to expand.

## Benchmarking

Run benchmarks to find the best model for your hardware:

```bash
python scripts/benchmark_pose_models.py --frames 100 --output results.json
```

## Architecture

```
app/ml/pose_models/
├── __init__.py          # Package exports
├── base.py              # Base classes and utilities
├── mediapipe_estimator.py
├── rtmpose_estimator.py
├── yolo_estimator.py
└── registry.py          # Model registry and management
```

## Landmark Format

All models output landmarks in the standardized MediaPipe 33-landmark format:

```python
landmarks = np.array([[x, y, z, visibility], ...])  # Shape: (33, 4)
```

Models that output fewer landmarks (e.g., COCO 17-keypoint) are automatically converted to the 33-landmark format with interpolated values for missing landmarks.
