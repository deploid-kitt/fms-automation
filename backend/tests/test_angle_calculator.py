"""Tests for angle calculator."""
import numpy as np
import pytest

from app.ml.angle_calculator import AngleCalculator, FMSMetrics
from app.ml.pose_estimator import JointIdx


class TestAngleCalculator:
    """Test angle calculation functions."""
    
    @pytest.fixture
    def calculator(self):
        return AngleCalculator()
    
    @pytest.fixture
    def standing_pose(self):
        """Create a standing pose (landmarks)."""
        # 33 landmarks x 4 values (x, y, z, visibility)
        landmarks = np.zeros((33, 4))
        
        # Standing position (Y increases downward in image coords)
        # Head at top
        landmarks[JointIdx.NOSE] = [0.5, 0.1, 0, 1.0]
        
        # Shoulders
        landmarks[JointIdx.LEFT_SHOULDER] = [0.4, 0.2, 0, 1.0]
        landmarks[JointIdx.RIGHT_SHOULDER] = [0.6, 0.2, 0, 1.0]
        
        # Elbows (arms down)
        landmarks[JointIdx.LEFT_ELBOW] = [0.4, 0.35, 0, 1.0]
        landmarks[JointIdx.RIGHT_ELBOW] = [0.6, 0.35, 0, 1.0]
        
        # Wrists
        landmarks[JointIdx.LEFT_WRIST] = [0.4, 0.5, 0, 1.0]
        landmarks[JointIdx.RIGHT_WRIST] = [0.6, 0.5, 0, 1.0]
        
        # Hips
        landmarks[JointIdx.LEFT_HIP] = [0.45, 0.5, 0, 1.0]
        landmarks[JointIdx.RIGHT_HIP] = [0.55, 0.5, 0, 1.0]
        
        # Knees
        landmarks[JointIdx.LEFT_KNEE] = [0.45, 0.7, 0, 1.0]
        landmarks[JointIdx.RIGHT_KNEE] = [0.55, 0.7, 0, 1.0]
        
        # Ankles
        landmarks[JointIdx.LEFT_ANKLE] = [0.45, 0.9, 0, 1.0]
        landmarks[JointIdx.RIGHT_ANKLE] = [0.55, 0.9, 0, 1.0]
        
        # Feet
        landmarks[JointIdx.LEFT_FOOT_INDEX] = [0.43, 0.95, 0, 1.0]
        landmarks[JointIdx.RIGHT_FOOT_INDEX] = [0.57, 0.95, 0, 1.0]
        landmarks[JointIdx.LEFT_HEEL] = [0.47, 0.92, 0, 1.0]
        landmarks[JointIdx.RIGHT_HEEL] = [0.53, 0.92, 0, 1.0]
        
        return landmarks
    
    @pytest.fixture
    def squat_pose(self):
        """Create a squat pose."""
        landmarks = np.zeros((33, 4))
        
        # Squatting - hips lower
        landmarks[JointIdx.NOSE] = [0.5, 0.3, 0, 1.0]
        
        # Shoulders (forward lean)
        landmarks[JointIdx.LEFT_SHOULDER] = [0.35, 0.4, 0.1, 1.0]
        landmarks[JointIdx.RIGHT_SHOULDER] = [0.65, 0.4, 0.1, 1.0]
        
        # Arms overhead
        landmarks[JointIdx.LEFT_ELBOW] = [0.35, 0.25, 0, 1.0]
        landmarks[JointIdx.RIGHT_ELBOW] = [0.65, 0.25, 0, 1.0]
        landmarks[JointIdx.LEFT_WRIST] = [0.35, 0.1, 0, 1.0]
        landmarks[JointIdx.RIGHT_WRIST] = [0.65, 0.1, 0, 1.0]
        
        # Hips low
        landmarks[JointIdx.LEFT_HIP] = [0.4, 0.7, 0, 1.0]
        landmarks[JointIdx.RIGHT_HIP] = [0.6, 0.7, 0, 1.0]
        
        # Knees bent
        landmarks[JointIdx.LEFT_KNEE] = [0.35, 0.75, 0.1, 1.0]
        landmarks[JointIdx.RIGHT_KNEE] = [0.65, 0.75, 0.1, 1.0]
        
        # Ankles
        landmarks[JointIdx.LEFT_ANKLE] = [0.4, 0.9, 0, 1.0]
        landmarks[JointIdx.RIGHT_ANKLE] = [0.6, 0.9, 0, 1.0]
        
        # Feet
        landmarks[JointIdx.LEFT_FOOT_INDEX] = [0.35, 0.95, 0, 1.0]
        landmarks[JointIdx.RIGHT_FOOT_INDEX] = [0.65, 0.95, 0, 1.0]
        landmarks[JointIdx.LEFT_HEEL] = [0.42, 0.92, 0, 1.0]
        landmarks[JointIdx.RIGHT_HEEL] = [0.58, 0.92, 0, 1.0]
        
        return landmarks
    
    def test_calculate_all_metrics_returns_fms_metrics(self, calculator, standing_pose):
        """Test that calculate_all_metrics returns FMSMetrics."""
        metrics = calculator.calculate_all_metrics(standing_pose)
        assert isinstance(metrics, FMSMetrics)
    
    def test_standing_knee_flexion_minimal(self, calculator, standing_pose):
        """Test that standing pose has minimal knee flexion."""
        metrics = calculator.calculate_all_metrics(standing_pose)
        # Standing should have near-zero knee flexion
        assert metrics.knee_flexion_left < 30
        assert metrics.knee_flexion_right < 30
    
    def test_squat_knee_flexion_significant(self, calculator, squat_pose):
        """Test that squat pose has significant knee flexion."""
        metrics = calculator.calculate_all_metrics(squat_pose)
        # Squatting should have significant knee flexion
        assert metrics.knee_flexion_left > 60
        assert metrics.knee_flexion_right > 60
    
    def test_deep_squat_metrics(self, calculator, squat_pose):
        """Test deep squat specific metrics."""
        metrics = calculator.calculate_deep_squat_metrics(squat_pose)
        
        assert "knee_flexion_avg" in metrics
        assert "hip_below_knee" in metrics
        assert "trunk_upright" in metrics
        assert "arms_overhead" in metrics
        assert "knee_valgus" in metrics
    
    def test_aslr_metrics(self, calculator, standing_pose):
        """Test ASLR metrics calculation."""
        metrics = calculator.calculate_aslr_metrics(standing_pose, "left")
        
        assert "hip_flexion" in metrics
        assert "contralateral_stable" in metrics
        assert "pelvis_neutral" in metrics


class TestPoseEstimator:
    """Test pose estimation utilities."""
    
    def test_landmarks_to_array(self):
        """Test landmark conversion."""
        from app.ml.pose_estimator import landmarks_to_array
        
        landmarks = [
            {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0},
            {"x": 0.4, "y": 0.3, "z": 0.1, "visibility": 0.9},
        ]
        
        arr = landmarks_to_array(landmarks)
        
        assert arr.shape == (2, 4)
        assert arr[0, 0] == 0.5
        assert arr[1, 3] == 0.9
    
    def test_calculate_angle(self):
        """Test angle calculation between three points."""
        from app.ml.pose_estimator import calculate_angle
        
        # 90 degree angle
        a = np.array([0, 1, 0])
        b = np.array([0, 0, 0])
        c = np.array([1, 0, 0])
        
        angle = calculate_angle(a, b, c)
        assert abs(angle - 90) < 0.1
    
    def test_calculate_distance(self):
        """Test distance calculation."""
        from app.ml.pose_estimator import calculate_distance
        
        a = np.array([0, 0, 0])
        b = np.array([3, 4, 0])
        
        dist = calculate_distance(a, b)
        assert abs(dist - 5.0) < 0.01
