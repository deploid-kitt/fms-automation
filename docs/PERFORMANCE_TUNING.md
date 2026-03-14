# FMS Live Analysis Performance Tuning Guide

This document describes the performance optimizations implemented to reduce pose lag in the live analysis system.

## Overview

The live analysis system was experiencing significant lag between the video feed and pose detection overlay. The following optimizations were implemented to achieve smooth real-time performance.

## Optimizations Implemented

### 1. Backend Optimizations

#### MediaPipe Configuration (config.py)
```python
pose_model_complexity: int = 1  # Changed from 2 to 1 for real-time
live_frame_skip: int = 2        # Process every 2nd frame
live_process_width: int = 320   # Downscale for processing
live_process_height: int = 240  # Downscale for processing
pose_smoothing_enabled: bool = True
pose_smoothing_factor: float = 0.5
```

**Key changes:**
- Model complexity reduced from 2 (heavy) to 1 (full) - provides ~2x speedup
- Frame downscaling: Processing at 320x240 instead of 640x480 - ~4x fewer pixels
- Frame skipping: Process every 2nd frame, return cached results for skipped frames

#### Pose Smoother (pose_estimator.py)
New `PoseSmoother` class applies exponential moving average smoothing to reduce jitter:
- Smooths x, y, z coordinates between frames
- Only applies when visibility > 0.5 for both current and previous frame
- Configurable smoothing factor (0.5 = moderate smoothing)

#### Performance Tracking
Added timing metrics throughout the pipeline:
- `process_time_ms` in pose results
- `server_process_ms` in feedback
- `avg_process_time` tracking for FPS estimation

### 2. WebSocket Optimizations (websocket.py)

#### Frame Skipping
- Server processes every Nth frame (configurable via `live_frame_skip`)
- Skipped frames return cached feedback with last known skeleton
- Reduces CPU load significantly

#### Timestamp Synchronization
- Client sends `timestamp` and `sequence` with each frame
- Server tracks processing time and returns `server_timestamp`
- Enables accurate latency calculation

#### Performance Statistics Endpoint
New `stats` message type returns:
- Frames received vs processed
- Average processing time
- Effective FPS

### 3. Frontend Optimizations (LiveAnalysis.jsx)

#### Skeleton Interpolation
Smooth animation between pose updates:
```javascript
function interpolateSkeleton(prevSkeleton, nextSkeleton, t) {
  // Interpolate x, y, z coordinates
  // Uses ease-out curve for natural motion
}
```

- Separate render loop runs at display refresh rate (~60fps)
- Pose updates arrive at ~15fps, interpolation fills the gaps
- Result: Smooth skeleton movement matching video feed

#### Canvas Optimizations
- Reusable capture canvas for frame encoding
- Batched drawing operations
- Single path for skeleton connections
- Efficient joint rendering (normal joints first, then problem joints with glow)

#### Frame Capture Optimizations
- Reduced JPEG quality: 0.7 → 0.6 (smaller payload)
- Reusable canvas element
- Transform reuse for mirror effect

#### Latency Tracking
- Map of `sequence → sendTime` for round-trip measurement
- Exponential moving average smoothing
- Performance monitor displays: Send FPS, Latency, Server time

#### Audio Cue Rate Limiting
- 2-second cooldown between speech cues
- Prevents audio spam during rapid feedback

## Configuration

### Environment Variables
```bash
FMS_POSE_MODEL_COMPLEXITY=1      # 0=lite, 1=full, 2=heavy
FMS_LIVE_FRAME_SKIP=2            # Process every Nth frame
FMS_LIVE_PROCESS_WIDTH=320       # Processing resolution width
FMS_LIVE_PROCESS_HEIGHT=240      # Processing resolution height
FMS_POSE_SMOOTHING_ENABLED=true  # Enable landmark smoothing
FMS_POSE_SMOOTHING_FACTOR=0.5    # Smoothing strength (0-1)
```

### Recommended Settings by Hardware

#### Low-end (Raspberry Pi, old laptop)
```
pose_model_complexity: 0
live_frame_skip: 3
live_process_width: 256
live_process_height: 192
```

#### Mid-range (Modern laptop, desktop)
```
pose_model_complexity: 1
live_frame_skip: 2
live_process_width: 320
live_process_height: 240
```

#### High-end (GPU-accelerated, powerful desktop)
```
pose_model_complexity: 2
live_frame_skip: 1
live_process_width: 640
live_process_height: 480
```

## Performance Metrics

### Target Performance
- **Pose processing**: < 50ms per frame
- **Total round-trip latency**: < 150ms
- **Perceived lag**: < 100ms (with interpolation)
- **Frame send rate**: 15 FPS
- **Render rate**: 60 FPS (display refresh)

### Monitoring
The frontend displays real-time performance metrics:
- **Send FPS**: Frames sent to server per second
- **Latency**: Round-trip time from frame send to feedback received
- **Server**: Server-side processing time

Toggle visibility with "Show/Hide Stats" button.

## Troubleshooting

### High Latency (> 200ms)
1. Check network connection (WebSocket over localhost is best)
2. Reduce `live_process_width/height`
3. Increase `live_frame_skip`
4. Switch to lower `pose_model_complexity`

### Jittery Skeleton
1. Increase `pose_smoothing_factor` (max 0.8)
2. Ensure good lighting for better pose detection
3. Check `min_detection_confidence` (lower = more detections, more noise)

### Low FPS
1. Close other applications
2. Check CPU usage
3. Try a lower resolution camera setting
4. Reduce `pose_model_complexity`

### Skeleton Lagging Behind Video
1. Check server processing time in stats
2. Reduce processing resolution
3. Increase frame skip
4. The interpolation should mask small delays (< 100ms)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Frontend                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Camera (30fps) ──► Frame Capture (15fps) ──► WebSocket Send        │
│                                                                      │
│  WebSocket Receive ──► Skeleton Queue ──► Interpolation ──► Render  │
│                            ↓                    ↓                    │
│                     Feedback State         Canvas (60fps)            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ WebSocket (JSON)
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                           Backend                                    │
├─────────────────────────────────────────────────────────────────────┤
│  Frame Receive ──► Skip Check ──► Decode ──► Downscale              │
│                         │                         │                  │
│                    Cached Result           MediaPipe Pose            │
│                                                   │                  │
│                                             Smooth ──► Analyze       │
│                                                          │           │
│                                                     Feedback         │
└─────────────────────────────────────────────────────────────────────┘
```

## Future Optimizations

1. **WebSocket Binary Protocol**: Send frames as binary instead of base64 JSON (~33% smaller)
2. **Web Workers**: Move frame capture to web worker to avoid main thread blocking
3. **Server-side GPU**: Use CUDA/Metal for MediaPipe when available
4. **Adaptive Quality**: Auto-adjust settings based on measured latency
5. **Delta Encoding**: Only send changed joints instead of full skeleton
