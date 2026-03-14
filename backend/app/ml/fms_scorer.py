"""FMS scoring engine using rule-based criteria."""
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

from app.models.schemas import FMSTest, TestScore, Fault
from app.ml.pose_estimator import landmarks_to_array
from app.ml.angle_calculator import AngleCalculator

logger = logging.getLogger(__name__)


@dataclass
class ScoringCriteria:
    """Criteria thresholds for FMS scoring."""
    # Deep Squat
    SQUAT_DEPTH_PARALLEL = 0.95  # hip/knee ratio for parallel
    SQUAT_DEPTH_BELOW = 1.05     # hip below knee
    SQUAT_TRUNK_UPRIGHT = 35    # degrees from vertical
    SQUAT_ARM_OVERHEAD = 160    # degrees elevation
    SQUAT_KNEE_VALGUS_MAX = 15  # degrees valgus
    SQUAT_KNEE_FLEXION_MIN = 90 # degrees for full squat
    
    # ASLR
    ASLR_HIP_FLEX_SCORE3 = 80   # degrees for score 3
    ASLR_HIP_FLEX_SCORE2 = 70   # degrees for score 2
    ASLR_HIP_FLEX_SCORE1 = 50   # degrees for score 1
    ASLR_PELVIS_NEUTRAL = 10    # max degrees tilt
    
    # Hurdle Step
    HURDLE_HIP_CLEAR_SCORE3 = 90  # degrees hip flexion
    HURDLE_HIP_CLEAR_SCORE2 = 70
    HURDLE_TRUNK_LEAN_MAX = 15
    HURDLE_PELVIS_TILT_MAX = 10
    
    # Inline Lunge
    LUNGE_KNEE_FLEX_MIN = 90
    LUNGE_TRUNK_UPRIGHT = 30
    LUNGE_KNEE_VALGUS_MAX = 15
    
    # Trunk Stability Push-up
    PUSHUP_BODY_LINE = 170      # degrees for straight body
    PUSHUP_HIP_SAG_MAX = 0.05   # normalized deviation
    
    # Shoulder Mobility
    SHOULDER_HAND_DIST_SCORE3 = 1.0  # hand distance / hand length
    SHOULDER_HAND_DIST_SCORE2 = 1.5
    SHOULDER_HAND_DIST_SCORE1 = 2.0


class FMSScorer:
    """Rule-based FMS scoring engine."""
    
    def __init__(self):
        self.angle_calc = AngleCalculator()
        self.criteria = ScoringCriteria()
    
    def score_deep_squat(
        self, 
        landmarks_sequence: list[np.ndarray],
        frame_range: Tuple[int, int] = (0, 0)
    ) -> TestScore:
        """
        Score deep squat test.
        
        Criteria for score 3:
        - Upper torso parallel with tibia or toward vertical
        - Femur below horizontal
        - Knees aligned over feet
        - Dowel aligned over feet
        - Heels on floor
        """
        faults = []
        scores_per_frame = []
        
        best_frame_idx = 0
        best_depth = 999
        
        for i, landmarks in enumerate(landmarks_sequence):
            metrics = self.angle_calc.calculate_deep_squat_metrics(landmarks)
            
            # Track deepest position
            if metrics["knee_flexion_avg"] < best_depth:
                best_depth = metrics["knee_flexion_avg"]
                best_frame_idx = i
            
            frame_score = 3
            frame_faults = []
            
            # Check trunk position
            if not metrics["trunk_upright"]:
                frame_score = min(frame_score, 2)
                frame_faults.append(("TRUNK_FORWARD", "Trunk excessively forward"))
            
            # Check depth
            if not metrics["hip_below_knee"]:
                frame_score = min(frame_score, 2)
                frame_faults.append(("DEPTH_INSUFFICIENT", "Hip not below knee level"))
            
            # Check knee valgus
            if metrics["knee_valgus"] > self.criteria.SQUAT_KNEE_VALGUS_MAX:
                frame_score = min(frame_score, 2)
                frame_faults.append(("KNEE_VALGUS", "Excessive knee valgus"))
            
            # Check arms overhead
            if not metrics["arms_overhead"]:
                frame_score = min(frame_score, 2)
                frame_faults.append(("ARMS_DROPPED", "Arms not maintained overhead"))
            
            # Check knee flexion minimum
            if metrics["knee_flexion_avg"] < 45:
                frame_score = min(frame_score, 1)
                frame_faults.append(("INCOMPLETE", "Unable to achieve squat position"))
            
            scores_per_frame.append((frame_score, frame_faults))
        
        # Determine final score from best attempt
        if not scores_per_frame:
            return TestScore(
                test=FMSTest.DEEP_SQUAT,
                score=0,
                faults=[Fault(code="NO_DATA", description="No pose data available")],
                confidence=0.0,
                frame_range=frame_range
            )
        
        # Use the frame with deepest squat for scoring
        final_score, final_faults = scores_per_frame[best_frame_idx]
        
        # Aggregate faults
        fault_counts = {}
        for score, frame_faults in scores_per_frame:
            for code, desc in frame_faults:
                fault_counts[code] = fault_counts.get(code, 0) + 1
        
        # Create fault objects for persistent faults (>30% of frames)
        threshold = len(scores_per_frame) * 0.3
        faults = []
        for code, count in fault_counts.items():
            if count > threshold:
                severity = "major" if count > len(scores_per_frame) * 0.6 else "minor"
                desc = {
                    "TRUNK_FORWARD": "Trunk leans excessively forward during descent",
                    "DEPTH_INSUFFICIENT": "Unable to achieve hip below knee level",
                    "KNEE_VALGUS": "Knees collapse inward during squat",
                    "ARMS_DROPPED": "Arms drop below overhead position",
                    "INCOMPLETE": "Cannot perform squat pattern",
                }.get(code, code)
                faults.append(Fault(code=code, description=desc, severity=severity))
        
        # Calculate confidence based on pose visibility
        avg_visibility = np.mean([
            np.mean(lm[:, 3]) for lm in landmarks_sequence
        ])
        confidence = min(0.95, avg_visibility * 1.2)
        
        notes = self._generate_squat_notes(final_score, faults)
        
        return TestScore(
            test=FMSTest.DEEP_SQUAT,
            score=final_score,
            faults=faults,
            notes=notes,
            confidence=confidence,
            frame_range=frame_range
        )
    
    def score_aslr(
        self,
        landmarks_sequence: list[np.ndarray],
        side: str,
        frame_range: Tuple[int, int] = (0, 0)
    ) -> TestScore:
        """
        Score Active Straight Leg Raise test.
        
        Criteria:
        - Score 3: Raised leg passes vertical (>80°)
        - Score 2: Raised leg between mid-thigh and knee (70-80°)
        - Score 1: Raised leg below mid-thigh (<70°)
        - Contralateral leg remains flat
        """
        test = (FMSTest.ACTIVE_STRAIGHT_LEG_RAISE_LEFT if side == "left" 
                else FMSTest.ACTIVE_STRAIGHT_LEG_RAISE_RIGHT)
        
        faults = []
        max_hip_flexion = 0
        pelvis_issues = 0
        contra_issues = 0
        
        for landmarks in landmarks_sequence:
            metrics = self.angle_calc.calculate_aslr_metrics(landmarks, side)
            
            max_hip_flexion = max(max_hip_flexion, metrics["hip_flexion"])
            
            if not metrics["pelvis_neutral"]:
                pelvis_issues += 1
            if not metrics["contralateral_stable"]:
                contra_issues += 1
        
        # Determine score based on max ROM achieved
        if max_hip_flexion >= self.criteria.ASLR_HIP_FLEX_SCORE3:
            score = 3
        elif max_hip_flexion >= self.criteria.ASLR_HIP_FLEX_SCORE2:
            score = 2
        elif max_hip_flexion >= self.criteria.ASLR_HIP_FLEX_SCORE1:
            score = 1
        else:
            score = 1
        
        # Check for compensation patterns
        if pelvis_issues > len(landmarks_sequence) * 0.3:
            faults.append(Fault(
                code="PELVIS_ROTATION",
                description="Pelvis rotates during leg raise",
                severity="minor"
            ))
        
        if contra_issues > len(landmarks_sequence) * 0.3:
            faults.append(Fault(
                code="CONTRA_LEG_LIFT",
                description="Contralateral leg lifts off surface",
                severity="minor"
            ))
        
        notes = f"Maximum hip flexion: {max_hip_flexion:.1f}°. "
        if faults:
            notes += "Compensation patterns observed. "
        
        return TestScore(
            test=test,
            score=score,
            faults=faults,
            notes=notes,
            confidence=0.8,
            frame_range=frame_range
        )
    
    def score_hurdle_step(
        self,
        landmarks_sequence: list[np.ndarray],
        side: str,
        frame_range: Tuple[int, int] = (0, 0)
    ) -> TestScore:
        """
        Score Hurdle Step test.
        
        Criteria:
        - Hips, knees, ankles remain aligned in sagittal plane
        - Minimal movement in lumbar spine
        - Dowel and hurdle remain parallel
        """
        test = (FMSTest.HURDLE_STEP_LEFT if side == "left" 
                else FMSTest.HURDLE_STEP_RIGHT)
        
        faults = []
        max_clearance = 0
        trunk_issues = 0
        pelvis_issues = 0
        
        for landmarks in landmarks_sequence:
            metrics = self.angle_calc.calculate_hurdle_step_metrics(landmarks, side)
            
            max_clearance = max(max_clearance, metrics["hip_clearance"])
            
            if not metrics["trunk_stable"]:
                trunk_issues += 1
            if not metrics["pelvis_level"]:
                pelvis_issues += 1
        
        # Score based on clearance and form
        if max_clearance >= self.criteria.HURDLE_HIP_CLEAR_SCORE3:
            score = 3
        elif max_clearance >= self.criteria.HURDLE_HIP_CLEAR_SCORE2:
            score = 2
        else:
            score = 1
        
        # Deduct for compensation
        if trunk_issues > len(landmarks_sequence) * 0.3:
            score = min(score, 2)
            faults.append(Fault(
                code="TRUNK_LEAN",
                description="Trunk leans laterally during step",
                severity="moderate"
            ))
        
        if pelvis_issues > len(landmarks_sequence) * 0.3:
            score = min(score, 2)
            faults.append(Fault(
                code="PELVIS_DROP",
                description="Pelvis drops on stance side",
                severity="moderate"
            ))
        
        notes = f"Hip clearance: {max_clearance:.1f}°. "
        
        return TestScore(
            test=test,
            score=score,
            faults=faults,
            notes=notes,
            confidence=0.75,
            frame_range=frame_range
        )
    
    def score_trunk_stability_pushup(
        self,
        landmarks_sequence: list[np.ndarray],
        frame_range: Tuple[int, int] = (0, 0)
    ) -> TestScore:
        """
        Score Trunk Stability Push-up test.
        
        Criteria:
        - Body lifts as a unit with no lag in spine
        - Men: thumbs at top of head
        - Women: thumbs at chin level
        """
        faults = []
        successful_reps = 0
        hip_sag_issues = 0
        spine_issues = 0
        
        for landmarks in landmarks_sequence:
            metrics = self.angle_calc.calculate_pushup_metrics(landmarks)
            
            if metrics["hip_sag"] > self.criteria.PUSHUP_HIP_SAG_MAX:
                hip_sag_issues += 1
            if not metrics["spine_aligned"]:
                spine_issues += 1
            else:
                successful_reps += 1
        
        # Score based on form
        total_frames = len(landmarks_sequence)
        good_form_ratio = successful_reps / max(total_frames, 1)
        
        if good_form_ratio > 0.7:
            score = 3
        elif good_form_ratio > 0.4:
            score = 2
        else:
            score = 1
        
        if hip_sag_issues > total_frames * 0.3:
            faults.append(Fault(
                code="HIP_SAG",
                description="Hips sag during push-up",
                severity="moderate"
            ))
        
        if spine_issues > total_frames * 0.3:
            faults.append(Fault(
                code="SPINE_FLEX",
                description="Spine not maintained in neutral",
                severity="moderate"
            ))
        
        notes = f"Body alignment maintained in {good_form_ratio*100:.0f}% of frames. "
        
        return TestScore(
            test=FMSTest.TRUNK_STABILITY_PUSHUP,
            score=score,
            faults=faults,
            notes=notes,
            confidence=0.7,
            frame_range=frame_range
        )
    
    def _generate_squat_notes(self, score: int, faults: list[Fault]) -> str:
        """Generate human-readable notes for squat score."""
        if score == 3:
            return "Excellent squat pattern. Full depth achieved with good form."
        elif score == 2:
            fault_desc = ", ".join([f.description.lower() for f in faults[:2]])
            return f"Squat completed with compensation: {fault_desc}."
        else:
            return "Unable to perform squat pattern. Consider mobility assessment."
    
    def score_exercise(
        self,
        exercise: str,
        landmarks_sequence: list[np.ndarray],
        frame_range: Tuple[int, int] = (0, 0)
    ) -> Optional[TestScore]:
        """
        Score an exercise based on its type.
        
        Args:
            exercise: Exercise name (e.g., "deep_squat", "aslr_left")
            landmarks_sequence: Sequence of pose landmarks
            frame_range: Start and end frame numbers
            
        Returns:
            TestScore or None if exercise not recognized
        """
        if exercise == "deep_squat":
            return self.score_deep_squat(landmarks_sequence, frame_range)
        elif exercise.startswith("aslr_"):
            side = exercise.split("_")[1]
            return self.score_aslr(landmarks_sequence, side, frame_range)
        elif exercise.startswith("hurdle_step_"):
            side = exercise.split("_")[2]
            return self.score_hurdle_step(landmarks_sequence, side, frame_range)
        elif exercise == "trunk_stability_pushup":
            return self.score_trunk_stability_pushup(landmarks_sequence, frame_range)
        # Add more exercises as needed
        
        logger.warning(f"No scorer implemented for exercise: {exercise}")
        return None
