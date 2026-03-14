"""Joint angle calculation for FMS scoring."""
import numpy as np
from typing import Optional
from dataclasses import dataclass

from app.ml.pose_estimator import JointIdx, calculate_angle, calculate_distance


@dataclass
class FMSMetrics:
    """Metrics extracted from pose for FMS scoring."""
    # Deep Squat metrics
    knee_flexion_left: Optional[float] = None
    knee_flexion_right: Optional[float] = None
    hip_flexion_left: Optional[float] = None
    hip_flexion_right: Optional[float] = None
    ankle_dorsiflexion_left: Optional[float] = None
    ankle_dorsiflexion_right: Optional[float] = None
    trunk_angle: Optional[float] = None
    arm_elevation_left: Optional[float] = None
    arm_elevation_right: Optional[float] = None
    knee_valgus_angle: Optional[float] = None
    hip_depth_ratio: Optional[float] = None  # hip height / knee height
    
    # Hurdle Step metrics
    hip_clearance: Optional[float] = None
    pelvic_tilt: Optional[float] = None
    trunk_lateral_lean: Optional[float] = None
    knee_clearance: Optional[float] = None
    
    # Inline Lunge metrics
    knee_over_ankle: Optional[float] = None
    torso_alignment: Optional[float] = None
    rear_knee_flexion: Optional[float] = None
    
    # Shoulder Mobility metrics
    hand_distance_normalized: Optional[float] = None
    
    # ASLR metrics
    hip_flexion_aslr: Optional[float] = None
    pelvis_tilt_aslr: Optional[float] = None
    contralateral_leg_angle: Optional[float] = None
    
    # Trunk Stability Push-up metrics
    spine_alignment: Optional[float] = None
    hip_sag: Optional[float] = None
    
    # Rotary Stability metrics
    spine_rotation: Optional[float] = None
    hip_stability: Optional[float] = None


class AngleCalculator:
    """Calculate joint angles and metrics for FMS scoring."""
    
    def __init__(self):
        """Initialize calculator."""
        pass
    
    def calculate_all_metrics(self, landmarks: np.ndarray) -> FMSMetrics:
        """
        Calculate all FMS-relevant metrics from pose landmarks.
        
        Args:
            landmarks: (33, 4) array of x, y, z, visibility
            
        Returns:
            FMSMetrics with calculated values
        """
        metrics = FMSMetrics()
        
        # Extract key joints (using normalized coordinates)
        l_shoulder = landmarks[JointIdx.LEFT_SHOULDER, :3]
        r_shoulder = landmarks[JointIdx.RIGHT_SHOULDER, :3]
        l_elbow = landmarks[JointIdx.LEFT_ELBOW, :3]
        r_elbow = landmarks[JointIdx.RIGHT_ELBOW, :3]
        l_wrist = landmarks[JointIdx.LEFT_WRIST, :3]
        r_wrist = landmarks[JointIdx.RIGHT_WRIST, :3]
        l_hip = landmarks[JointIdx.LEFT_HIP, :3]
        r_hip = landmarks[JointIdx.RIGHT_HIP, :3]
        l_knee = landmarks[JointIdx.LEFT_KNEE, :3]
        r_knee = landmarks[JointIdx.RIGHT_KNEE, :3]
        l_ankle = landmarks[JointIdx.LEFT_ANKLE, :3]
        r_ankle = landmarks[JointIdx.RIGHT_ANKLE, :3]
        l_heel = landmarks[JointIdx.LEFT_HEEL, :3]
        r_heel = landmarks[JointIdx.RIGHT_HEEL, :3]
        l_foot = landmarks[JointIdx.LEFT_FOOT_INDEX, :3]
        r_foot = landmarks[JointIdx.RIGHT_FOOT_INDEX, :3]
        nose = landmarks[JointIdx.NOSE, :3]
        
        # Calculate mid-points
        mid_hip = (l_hip + r_hip) / 2
        mid_shoulder = (l_shoulder + r_shoulder) / 2
        mid_knee = (l_knee + r_knee) / 2
        mid_ankle = (l_ankle + r_ankle) / 2
        
        # --- Knee Flexion (hip-knee-ankle angle) ---
        metrics.knee_flexion_left = 180 - calculate_angle(l_hip, l_knee, l_ankle)
        metrics.knee_flexion_right = 180 - calculate_angle(r_hip, r_knee, r_ankle)
        
        # --- Hip Flexion (shoulder-hip-knee angle) ---
        metrics.hip_flexion_left = 180 - calculate_angle(l_shoulder, l_hip, l_knee)
        metrics.hip_flexion_right = 180 - calculate_angle(r_shoulder, r_hip, r_knee)
        
        # --- Ankle Dorsiflexion (knee-ankle-foot angle) ---
        metrics.ankle_dorsiflexion_left = calculate_angle(l_knee, l_ankle, l_foot)
        metrics.ankle_dorsiflexion_right = calculate_angle(r_knee, r_ankle, r_foot)
        
        # --- Trunk Angle (vertical alignment) ---
        # Angle between vertical and hip-shoulder line
        vertical = np.array([0, -1, 0])  # Pointing up
        trunk_vector = mid_shoulder - mid_hip
        trunk_vector_norm = trunk_vector / (np.linalg.norm(trunk_vector) + 1e-8)
        trunk_cos = np.dot(trunk_vector_norm, vertical)
        metrics.trunk_angle = np.degrees(np.arccos(np.clip(trunk_cos, -1, 1)))
        
        # --- Arm Elevation (shoulder-elbow angle from vertical) ---
        l_arm_vector = l_elbow - l_shoulder
        l_arm_norm = l_arm_vector / (np.linalg.norm(l_arm_vector) + 1e-8)
        metrics.arm_elevation_left = np.degrees(np.arccos(np.clip(np.dot(l_arm_norm, -vertical), -1, 1)))
        
        r_arm_vector = r_elbow - r_shoulder
        r_arm_norm = r_arm_vector / (np.linalg.norm(r_arm_vector) + 1e-8)
        metrics.arm_elevation_right = np.degrees(np.arccos(np.clip(np.dot(r_arm_norm, -vertical), -1, 1)))
        
        # --- Knee Valgus (frontal plane knee angle) ---
        # Simplified: angle between hip-knee and knee-ankle in XY plane
        l_hip_knee_xy = (l_knee[:2] - l_hip[:2])
        l_knee_ankle_xy = (l_ankle[:2] - l_knee[:2])
        r_hip_knee_xy = (r_knee[:2] - r_hip[:2])
        r_knee_ankle_xy = (r_ankle[:2] - r_knee[:2])
        
        # Cross product for direction
        l_valgus = np.cross(l_hip_knee_xy, l_knee_ankle_xy)
        r_valgus = np.cross(r_hip_knee_xy, r_knee_ankle_xy)
        
        # Convert to angle approximation
        metrics.knee_valgus_angle = float(np.degrees(np.arcsin(np.clip((l_valgus + r_valgus) / 2, -1, 1))))
        
        # --- Hip Depth Ratio (for squat depth) ---
        hip_height = 1 - mid_hip[1]  # Y is inverted in image coords
        knee_height = 1 - mid_knee[1]
        metrics.hip_depth_ratio = hip_height / (knee_height + 1e-8)
        
        # --- Pelvic Tilt (hip asymmetry) ---
        metrics.pelvic_tilt = np.degrees(np.arctan2(
            l_hip[1] - r_hip[1],  # Vertical difference
            l_hip[0] - r_hip[0]   # Horizontal difference
        ))
        
        # --- Trunk Lateral Lean ---
        metrics.trunk_lateral_lean = np.degrees(np.arctan2(
            l_shoulder[0] - r_shoulder[0] - (l_hip[0] - r_hip[0]),
            l_shoulder[1] - r_shoulder[1]
        ))
        
        # --- Spine Alignment (nose-mid_shoulder-mid_hip angle) ---
        metrics.spine_alignment = calculate_angle(nose, mid_shoulder, mid_hip)
        
        # --- Hip Sag (for push-up) ---
        # Deviation of hip from shoulder-ankle line
        shoulder_ankle = mid_ankle - mid_shoulder
        shoulder_hip = mid_hip - mid_shoulder
        proj_length = np.dot(shoulder_hip, shoulder_ankle) / (np.dot(shoulder_ankle, shoulder_ankle) + 1e-8)
        proj_point = mid_shoulder + proj_length * shoulder_ankle
        metrics.hip_sag = calculate_distance(mid_hip, proj_point)
        
        # --- Hand Distance (for shoulder mobility) ---
        hand_distance = calculate_distance(l_wrist, r_wrist)
        shoulder_width = calculate_distance(l_shoulder, r_shoulder)
        metrics.hand_distance_normalized = hand_distance / (shoulder_width + 1e-8)
        
        return metrics
    
    def calculate_deep_squat_metrics(self, landmarks: np.ndarray) -> dict:
        """
        Calculate specific metrics for deep squat scoring.
        
        Returns dict with:
        - knee_flexion_avg: Average knee flexion angle
        - hip_below_knee: Boolean, is hip below knee level
        - trunk_upright: Boolean, trunk angle < 30 degrees
        - arms_overhead: Boolean, arms elevated > 150 degrees
        - knee_valgus: Valgus angle (should be near 0)
        - heels_down: Estimated heel contact
        """
        metrics = self.calculate_all_metrics(landmarks)
        
        knee_flexion = (metrics.knee_flexion_left + metrics.knee_flexion_right) / 2
        arm_elevation = (metrics.arm_elevation_left + metrics.arm_elevation_right) / 2
        
        return {
            "knee_flexion_avg": knee_flexion,
            "hip_below_knee": metrics.hip_depth_ratio < 1.0,
            "trunk_upright": metrics.trunk_angle < 35,
            "arms_overhead": arm_elevation > 150,
            "knee_valgus": abs(metrics.knee_valgus_angle) if metrics.knee_valgus_angle else 0,
            "ankle_dorsiflexion_avg": (metrics.ankle_dorsiflexion_left + metrics.ankle_dorsiflexion_right) / 2,
        }
    
    def calculate_aslr_metrics(self, landmarks: np.ndarray, side: str = "left") -> dict:
        """
        Calculate metrics for Active Straight Leg Raise.
        
        Args:
            landmarks: Pose landmarks
            side: "left" or "right" - which leg is being raised
            
        Returns dict with:
        - hip_flexion: Angle of raised leg
        - contralateral_stable: Is other leg flat
        - pelvis_neutral: Pelvis tilt angle
        """
        metrics = self.calculate_all_metrics(landmarks)
        
        if side == "left":
            raised_flexion = metrics.hip_flexion_left
            stable_flexion = metrics.hip_flexion_right
        else:
            raised_flexion = metrics.hip_flexion_right
            stable_flexion = metrics.hip_flexion_left
        
        return {
            "hip_flexion": raised_flexion,
            "contralateral_stable": stable_flexion < 20,
            "pelvis_neutral": abs(metrics.pelvic_tilt) < 10,
        }
    
    def calculate_hurdle_step_metrics(self, landmarks: np.ndarray, side: str = "left") -> dict:
        """
        Calculate metrics for Hurdle Step.
        
        Returns dict with:
        - hip_clearance: Hip flexion of stepping leg
        - trunk_stable: Trunk lateral lean angle
        - pelvis_level: Pelvic tilt angle
        """
        metrics = self.calculate_all_metrics(landmarks)
        
        if side == "left":
            hip_flex = metrics.hip_flexion_left
        else:
            hip_flex = metrics.hip_flexion_right
        
        return {
            "hip_clearance": hip_flex,
            "trunk_stable": abs(metrics.trunk_lateral_lean) < 15,
            "pelvis_level": abs(metrics.pelvic_tilt) < 10,
        }
    
    def calculate_pushup_metrics(self, landmarks: np.ndarray) -> dict:
        """
        Calculate metrics for Trunk Stability Push-up.
        
        Returns dict with:
        - spine_aligned: Is spine in neutral
        - hip_sag: Amount of hip sag/pike
        """
        metrics = self.calculate_all_metrics(landmarks)
        
        return {
            "spine_aligned": 160 < metrics.spine_alignment < 200,
            "hip_sag": metrics.hip_sag,
            "body_alignment": metrics.spine_alignment,
        }
