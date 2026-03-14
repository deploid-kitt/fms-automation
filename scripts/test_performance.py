#!/usr/bin/env python3
"""Test script to verify performance optimizations for live pose analysis."""
import sys
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, '/root/.openclaw/workspace/projects/development/fms-automation/backend')

def test_pose_smoother():
    """Test the PoseSmoother class."""
    from app.ml.pose_estimator import PoseSmoother
    
    smoother = PoseSmoother(smoothing_factor=0.5, num_landmarks=33)
    
    # Generate test landmarks
    landmarks1 = np.random.rand(33, 4)
    landmarks1[:, 3] = 0.9  # High visibility
    
    landmarks2 = landmarks1.copy()
    landmarks2[:, :3] += 0.1  # Small movement
    
    # First call should return original
    result1 = smoother.smooth(landmarks1)
    assert np.allclose(result1, landmarks1), "First call should return original"
    
    # Second call should be smoothed
    result2 = smoother.smooth(landmarks2)
    
    # Result should be between landmarks1 and landmarks2
    for i in range(33):
        for j in range(3):  # x, y, z only
            assert result2[i, j] >= min(landmarks1[i, j], landmarks2[i, j]) - 0.01
            assert result2[i, j] <= max(landmarks1[i, j], landmarks2[i, j]) + 0.01
    
    print("✓ PoseSmoother test passed")


def test_config_settings():
    """Test that new config settings are accessible."""
    from app.core.config import get_settings
    
    settings = get_settings()
    
    assert hasattr(settings, 'live_frame_skip'), "Missing live_frame_skip"
    assert hasattr(settings, 'live_process_width'), "Missing live_process_width"
    assert hasattr(settings, 'live_process_height'), "Missing live_process_height"
    assert hasattr(settings, 'pose_smoothing_enabled'), "Missing pose_smoothing_enabled"
    assert hasattr(settings, 'pose_smoothing_factor'), "Missing pose_smoothing_factor"
    
    print(f"✓ Config test passed")
    print(f"  - pose_model_complexity: {settings.pose_model_complexity}")
    print(f"  - live_frame_skip: {settings.live_frame_skip}")
    print(f"  - live_process_width: {settings.live_process_width}")
    print(f"  - live_process_height: {settings.live_process_height}")
    print(f"  - pose_smoothing_enabled: {settings.pose_smoothing_enabled}")
    print(f"  - pose_smoothing_factor: {settings.pose_smoothing_factor}")


def test_pose_estimator_performance():
    """Test PoseEstimator with performance tracking."""
    try:
        from app.ml.pose_estimator import PoseEstimator
        
        # Create estimator optimized for live
        estimator = PoseEstimator(enable_smoothing=True, for_live=True)
        
        # Create a test image (black with some noise)
        test_frame = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        
        # Process multiple frames
        times = []
        for i in range(10):
            start = time.perf_counter()
            result = estimator.process_frame(test_frame, downscale=True)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        
        print(f"✓ PoseEstimator performance test")
        print(f"  - Average process time: {avg_time:.1f}ms")
        print(f"  - Frames processed: {estimator.frame_count}")
        
        stats = estimator.get_performance_stats()
        print(f"  - Estimated max FPS: {stats['estimated_max_fps']:.1f}")
        
        estimator.close()
        
    except ImportError as e:
        print(f"⚠ Skipping PoseEstimator test (MediaPipe not available): {e}")


def test_live_feedback_dataclass():
    """Test LiveFeedback dataclass has new fields."""
    from app.api.websocket import LiveFeedback, ExercisePhase
    
    feedback = LiveFeedback(
        exercise="deep_squat",
        phase=ExercisePhase.ACTIVE,
        frame_count=100,
        elapsed_seconds=5.0,
        pose_detected=True,
        pose_confidence=0.95,
        current_score=3,
        score_confidence=0.8,
        form_quality="excellent",
        primary_cue="Good form",
        secondary_cues=[],
        skeleton=[],
        joint_angles={"knee_flexion": 90},
        problem_joints=[],
        server_process_ms=15.5,
        pose_process_ms=12.3,
        server_timestamp=time.time(),
        frame_sequence=42
    )
    
    assert feedback.server_process_ms == 15.5
    assert feedback.pose_process_ms == 12.3
    assert feedback.frame_sequence == 42
    
    print("✓ LiveFeedback dataclass test passed")


if __name__ == "__main__":
    print("FMS Performance Optimization Tests")
    print("=" * 40)
    
    test_config_settings()
    test_pose_smoother()
    test_live_feedback_dataclass()
    test_pose_estimator_performance()
    
    print("=" * 40)
    print("All tests completed!")
