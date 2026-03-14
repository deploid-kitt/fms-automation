"""WebSocket endpoints for real-time FMS analysis."""
import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ml.pose_estimator import PoseEstimator, landmarks_to_array, JointIdx
from app.ml.angle_calculator import AngleCalculator
from app.ml.fms_scorer import FMSScorer, ScoringCriteria
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["websocket"])


class ExercisePhase(str, Enum):
    """Phases of an exercise for real-time tracking."""
    IDLE = "idle"
    PREPARING = "preparing"
    ACTIVE = "active"
    HOLD = "hold"
    RETURNING = "returning"
    COMPLETE = "complete"


@dataclass
class LiveFeedback:
    """Real-time feedback for the user."""
    # Current state
    exercise: str
    phase: ExercisePhase
    frame_count: int
    elapsed_seconds: float
    
    # Pose quality
    pose_detected: bool
    pose_confidence: float
    
    # Current score estimate
    current_score: int  # 0-3
    score_confidence: float
    
    # Form feedback
    form_quality: str  # "excellent", "good", "needs_work", "poor"
    primary_cue: str  # Main coaching cue
    secondary_cues: list[str]  # Additional cues
    
    # Visual feedback data
    skeleton: list[dict]  # Landmark positions for overlay
    joint_angles: dict[str, float]  # Key joint angles
    problem_joints: list[int]  # Joint indices with issues
    
    # Audio cue (if any)
    audio_cue: Optional[str] = None
    audio_priority: int = 0  # 0=none, 1=info, 2=warning, 3=critical


class LiveAnalyzer:
    """Real-time FMS analysis engine."""
    
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.angle_calc = AngleCalculator()
        self.scorer = FMSScorer()
        self.criteria = ScoringCriteria()
        
        # Session state
        self.session_id: Optional[str] = None
        self.current_exercise: str = "deep_squat"
        self.phase: ExercisePhase = ExercisePhase.IDLE
        self.frame_count: int = 0
        self.start_time: float = 0
        
        # Analysis buffers
        self.landmark_buffer: list[np.ndarray] = []
        self.score_buffer: list[int] = []
        self.max_buffer_size: int = 90  # ~3 seconds at 30fps
        
        # Feedback state
        self.last_audio_cue_time: float = 0
        self.audio_cue_cooldown: float = 2.0  # seconds between audio cues
        self.cue_history: list[str] = []
        
    def start_session(self, exercise: str = "deep_squat") -> str:
        """Start a new analysis session."""
        self.session_id = str(uuid.uuid4())
        self.current_exercise = exercise
        self.phase = ExercisePhase.PREPARING
        self.frame_count = 0
        self.start_time = time.time()
        self.landmark_buffer.clear()
        self.score_buffer.clear()
        self.cue_history.clear()
        
        logger.info(f"Started live session {self.session_id} for {exercise}")
        return self.session_id
    
    def process_frame(self, frame: np.ndarray) -> LiveFeedback:
        """
        Process a single frame and return real-time feedback.
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            LiveFeedback with all analysis results
        """
        self.frame_count += 1
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Extract pose
        pose_data = self.pose_estimator.process_frame(frame)
        
        if not pose_data or not pose_data.get("landmarks"):
            return self._no_pose_feedback(elapsed)
        
        # Convert to numpy array
        landmarks = landmarks_to_array(pose_data["landmarks"])
        
        # Add to buffer
        self.landmark_buffer.append(landmarks)
        if len(self.landmark_buffer) > self.max_buffer_size:
            self.landmark_buffer.pop(0)
        
        # Calculate pose confidence
        pose_confidence = float(np.mean(landmarks[:, 3]))  # visibility
        
        # Get exercise-specific analysis
        feedback = self._analyze_exercise(landmarks, elapsed, pose_confidence)
        
        return feedback
    
    def _no_pose_feedback(self, elapsed: float) -> LiveFeedback:
        """Return feedback when no pose is detected."""
        return LiveFeedback(
            exercise=self.current_exercise,
            phase=self.phase,
            frame_count=self.frame_count,
            elapsed_seconds=elapsed,
            pose_detected=False,
            pose_confidence=0.0,
            current_score=0,
            score_confidence=0.0,
            form_quality="poor",
            primary_cue="Step back so your full body is visible",
            secondary_cues=["Ensure good lighting", "Face the camera"],
            skeleton=[],
            joint_angles={},
            problem_joints=[],
            audio_cue="Please step back so I can see you" if self.frame_count % 60 == 0 else None,
            audio_priority=2 if self.frame_count % 60 == 0 else 0
        )
    
    def _analyze_exercise(
        self, 
        landmarks: np.ndarray, 
        elapsed: float,
        pose_confidence: float
    ) -> LiveFeedback:
        """Analyze based on current exercise type."""
        
        if self.current_exercise == "deep_squat":
            return self._analyze_deep_squat(landmarks, elapsed, pose_confidence)
        elif self.current_exercise.startswith("aslr_"):
            side = self.current_exercise.split("_")[1]
            return self._analyze_aslr(landmarks, elapsed, pose_confidence, side)
        elif self.current_exercise.startswith("hurdle_step_"):
            side = self.current_exercise.split("_")[2]
            return self._analyze_hurdle_step(landmarks, elapsed, pose_confidence, side)
        elif self.current_exercise == "trunk_stability_pushup":
            return self._analyze_pushup(landmarks, elapsed, pose_confidence)
        else:
            return self._analyze_deep_squat(landmarks, elapsed, pose_confidence)
    
    def _analyze_deep_squat(
        self,
        landmarks: np.ndarray,
        elapsed: float,
        pose_confidence: float
    ) -> LiveFeedback:
        """Real-time deep squat analysis."""
        
        metrics = self.angle_calc.calculate_deep_squat_metrics(landmarks)
        
        # Determine phase based on knee flexion
        knee_flex = metrics["knee_flexion_avg"]
        
        if knee_flex < 20:
            self.phase = ExercisePhase.PREPARING
        elif knee_flex < 60:
            self.phase = ExercisePhase.ACTIVE
        elif knee_flex >= 60:
            self.phase = ExercisePhase.HOLD
        
        # Calculate current score estimate
        score = 3
        cues = []
        secondary_cues = []
        problem_joints = []
        
        # Check trunk position
        if not metrics["trunk_upright"]:
            score = min(score, 2)
            cues.append("Keep your chest up")
            problem_joints.extend([JointIdx.LEFT_SHOULDER, JointIdx.RIGHT_SHOULDER])
        
        # Check depth (only in active/hold phase)
        if self.phase in [ExercisePhase.ACTIVE, ExercisePhase.HOLD]:
            if not metrics["hip_below_knee"]:
                score = min(score, 2)
                secondary_cues.append("Go deeper - hips below knees")
                problem_joints.extend([JointIdx.LEFT_HIP, JointIdx.RIGHT_HIP])
        
        # Check knee valgus
        if metrics["knee_valgus"] > self.criteria.SQUAT_KNEE_VALGUS_MAX:
            score = min(score, 2)
            cues.append("Push your knees out")
            problem_joints.extend([JointIdx.LEFT_KNEE, JointIdx.RIGHT_KNEE])
        
        # Check arms overhead
        if not metrics["arms_overhead"]:
            score = min(score, 2)
            secondary_cues.append("Keep arms overhead")
            problem_joints.extend([JointIdx.LEFT_ELBOW, JointIdx.RIGHT_ELBOW])
        
        # Check ankle mobility
        if metrics["ankle_dorsiflexion_avg"] < 70:
            secondary_cues.append("Keep heels down")
            problem_joints.extend([JointIdx.LEFT_ANKLE, JointIdx.RIGHT_ANKLE])
        
        # Determine primary cue
        primary_cue = cues[0] if cues else "Good form - continue the movement"
        
        # Form quality assessment
        if score == 3 and not secondary_cues:
            form_quality = "excellent"
        elif score == 3:
            form_quality = "good"
        elif score == 2:
            form_quality = "needs_work"
        else:
            form_quality = "poor"
        
        # Audio cue logic
        audio_cue = None
        audio_priority = 0
        current_time = time.time()
        
        if current_time - self.last_audio_cue_time > self.audio_cue_cooldown:
            if cues and cues[0] not in self.cue_history[-3:] if self.cue_history else True:
                audio_cue = cues[0]
                audio_priority = 2 if score <= 1 else 1
                self.last_audio_cue_time = current_time
                self.cue_history.append(cues[0])
        
        # Build skeleton for overlay
        skeleton = self._landmarks_to_skeleton(landmarks)
        
        # Store score for averaging
        self.score_buffer.append(score)
        if len(self.score_buffer) > 30:
            self.score_buffer.pop(0)
        
        avg_score = round(sum(self.score_buffer) / len(self.score_buffer))
        
        return LiveFeedback(
            exercise=self.current_exercise,
            phase=self.phase,
            frame_count=self.frame_count,
            elapsed_seconds=elapsed,
            pose_detected=True,
            pose_confidence=pose_confidence,
            current_score=avg_score,
            score_confidence=min(len(self.score_buffer) / 30, 1.0),
            form_quality=form_quality,
            primary_cue=primary_cue,
            secondary_cues=secondary_cues[:3],
            skeleton=skeleton,
            joint_angles={
                "knee_flexion": round(knee_flex, 1),
                "trunk_angle": round(self.angle_calc.calculate_all_metrics(landmarks).trunk_angle, 1),
                "knee_valgus": round(metrics["knee_valgus"], 1),
            },
            problem_joints=list(set(problem_joints)),
            audio_cue=audio_cue,
            audio_priority=audio_priority
        )
    
    def _analyze_aslr(
        self,
        landmarks: np.ndarray,
        elapsed: float,
        pose_confidence: float,
        side: str
    ) -> LiveFeedback:
        """Real-time Active Straight Leg Raise analysis."""
        
        metrics = self.angle_calc.calculate_aslr_metrics(landmarks, side)
        
        hip_flex = metrics["hip_flexion"]
        
        # Determine phase
        if hip_flex < 20:
            self.phase = ExercisePhase.PREPARING
        elif hip_flex < 60:
            self.phase = ExercisePhase.ACTIVE
        else:
            self.phase = ExercisePhase.HOLD
        
        # Score calculation
        if hip_flex >= self.criteria.ASLR_HIP_FLEX_SCORE3:
            score = 3
        elif hip_flex >= self.criteria.ASLR_HIP_FLEX_SCORE2:
            score = 2
        elif hip_flex >= self.criteria.ASLR_HIP_FLEX_SCORE1:
            score = 1
        else:
            score = 1
        
        cues = []
        secondary_cues = []
        problem_joints = []
        
        # Check pelvis
        if not metrics["pelvis_neutral"]:
            cues.append("Keep your pelvis flat")
            problem_joints.extend([JointIdx.LEFT_HIP, JointIdx.RIGHT_HIP])
        
        # Check contralateral leg
        if not metrics["contralateral_stable"]:
            cues.append(f"Keep your {'right' if side == 'left' else 'left'} leg down")
            idx = JointIdx.RIGHT_KNEE if side == "left" else JointIdx.LEFT_KNEE
            problem_joints.append(idx)
        
        # Encourage more ROM
        if self.phase == ExercisePhase.HOLD and hip_flex < 80:
            secondary_cues.append("Lift your leg higher if you can")
        
        primary_cue = cues[0] if cues else f"Good - hip flexion at {hip_flex:.0f}°"
        
        if score == 3:
            form_quality = "excellent"
        elif score == 2 and not cues:
            form_quality = "good"
        elif score == 2:
            form_quality = "needs_work"
        else:
            form_quality = "poor"
        
        skeleton = self._landmarks_to_skeleton(landmarks)
        
        self.score_buffer.append(score)
        if len(self.score_buffer) > 30:
            self.score_buffer.pop(0)
        
        return LiveFeedback(
            exercise=self.current_exercise,
            phase=self.phase,
            frame_count=self.frame_count,
            elapsed_seconds=elapsed,
            pose_detected=True,
            pose_confidence=pose_confidence,
            current_score=round(sum(self.score_buffer) / len(self.score_buffer)),
            score_confidence=min(len(self.score_buffer) / 30, 1.0),
            form_quality=form_quality,
            primary_cue=primary_cue,
            secondary_cues=secondary_cues,
            skeleton=skeleton,
            joint_angles={
                "hip_flexion": round(hip_flex, 1),
            },
            problem_joints=problem_joints,
        )
    
    def _analyze_hurdle_step(
        self,
        landmarks: np.ndarray,
        elapsed: float,
        pose_confidence: float,
        side: str
    ) -> LiveFeedback:
        """Real-time Hurdle Step analysis."""
        
        metrics = self.angle_calc.calculate_hurdle_step_metrics(landmarks, side)
        
        hip_clear = metrics["hip_clearance"]
        
        # Determine phase
        if hip_clear < 30:
            self.phase = ExercisePhase.PREPARING
        elif hip_clear < 70:
            self.phase = ExercisePhase.ACTIVE
        else:
            self.phase = ExercisePhase.HOLD
        
        # Score calculation
        if hip_clear >= self.criteria.HURDLE_HIP_CLEAR_SCORE3:
            score = 3
        elif hip_clear >= self.criteria.HURDLE_HIP_CLEAR_SCORE2:
            score = 2
        else:
            score = 1
        
        cues = []
        secondary_cues = []
        problem_joints = []
        
        if not metrics["trunk_stable"]:
            score = min(score, 2)
            cues.append("Keep your trunk upright")
            problem_joints.extend([JointIdx.LEFT_SHOULDER, JointIdx.RIGHT_SHOULDER])
        
        if not metrics["pelvis_level"]:
            score = min(score, 2)
            cues.append("Keep your hips level")
            problem_joints.extend([JointIdx.LEFT_HIP, JointIdx.RIGHT_HIP])
        
        primary_cue = cues[0] if cues else "Good form - lift knee higher"
        
        if score == 3:
            form_quality = "excellent"
        elif score == 2 and not cues:
            form_quality = "good"
        else:
            form_quality = "needs_work"
        
        skeleton = self._landmarks_to_skeleton(landmarks)
        
        self.score_buffer.append(score)
        if len(self.score_buffer) > 30:
            self.score_buffer.pop(0)
        
        return LiveFeedback(
            exercise=self.current_exercise,
            phase=self.phase,
            frame_count=self.frame_count,
            elapsed_seconds=elapsed,
            pose_detected=True,
            pose_confidence=pose_confidence,
            current_score=round(sum(self.score_buffer) / len(self.score_buffer)),
            score_confidence=min(len(self.score_buffer) / 30, 1.0),
            form_quality=form_quality,
            primary_cue=primary_cue,
            secondary_cues=secondary_cues,
            skeleton=skeleton,
            joint_angles={
                "hip_clearance": round(hip_clear, 1),
            },
            problem_joints=problem_joints,
        )
    
    def _analyze_pushup(
        self,
        landmarks: np.ndarray,
        elapsed: float,
        pose_confidence: float
    ) -> LiveFeedback:
        """Real-time Trunk Stability Push-up analysis."""
        
        metrics = self.angle_calc.calculate_pushup_metrics(landmarks)
        
        cues = []
        secondary_cues = []
        problem_joints = []
        score = 3
        
        if not metrics["spine_aligned"]:
            score = min(score, 2)
            cues.append("Keep your body in a straight line")
            problem_joints.extend([JointIdx.LEFT_HIP, JointIdx.RIGHT_HIP])
        
        if metrics["hip_sag"] > self.criteria.PUSHUP_HIP_SAG_MAX:
            score = min(score, 2)
            cues.append("Don't let your hips sag")
            problem_joints.extend([JointIdx.LEFT_HIP, JointIdx.RIGHT_HIP])
        
        primary_cue = cues[0] if cues else "Good body alignment"
        
        if score == 3:
            form_quality = "excellent"
        elif score == 2:
            form_quality = "needs_work"
        else:
            form_quality = "poor"
        
        skeleton = self._landmarks_to_skeleton(landmarks)
        
        self.score_buffer.append(score)
        if len(self.score_buffer) > 30:
            self.score_buffer.pop(0)
        
        return LiveFeedback(
            exercise=self.current_exercise,
            phase=self.phase,
            frame_count=self.frame_count,
            elapsed_seconds=elapsed,
            pose_detected=True,
            pose_confidence=pose_confidence,
            current_score=round(sum(self.score_buffer) / len(self.score_buffer)),
            score_confidence=min(len(self.score_buffer) / 30, 1.0),
            form_quality=form_quality,
            primary_cue=primary_cue,
            secondary_cues=secondary_cues,
            skeleton=skeleton,
            joint_angles={
                "body_alignment": round(metrics["body_alignment"], 1),
            },
            problem_joints=problem_joints,
        )
    
    def _landmarks_to_skeleton(self, landmarks: np.ndarray) -> list[dict]:
        """Convert landmarks to skeleton format for frontend overlay."""
        skeleton = []
        for i, lm in enumerate(landmarks):
            skeleton.append({
                "id": i,
                "x": float(lm[0]),
                "y": float(lm[1]),
                "z": float(lm[2]),
                "visibility": float(lm[3])
            })
        return skeleton
    
    def get_final_score(self) -> dict:
        """Get final score and summary when session ends."""
        if not self.score_buffer:
            return {"score": 0, "confidence": 0, "frames_analyzed": 0}
        
        avg_score = round(sum(self.score_buffer) / len(self.score_buffer))
        return {
            "score": avg_score,
            "confidence": min(len(self.score_buffer) / 30, 1.0),
            "frames_analyzed": self.frame_count,
            "exercise": self.current_exercise,
            "session_id": self.session_id
        }
    
    def close(self):
        """Clean up resources."""
        self.pose_estimator.close()


# Connection manager for WebSocket sessions
class ConnectionManager:
    """Manage WebSocket connections for live analysis."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.analyzers: Dict[str, LiveAnalyzer] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.analyzers[session_id] = LiveAnalyzer()
        logger.info(f"WebSocket connected: {session_id}")
    
    def disconnect(self, session_id: str):
        """Handle disconnection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.analyzers:
            self.analyzers[session_id].close()
            del self.analyzers[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_feedback(self, session_id: str, feedback: LiveFeedback):
        """Send feedback to a specific client."""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(asdict(feedback))
    
    def get_analyzer(self, session_id: str) -> Optional[LiveAnalyzer]:
        """Get analyzer for a session."""
        return self.analyzers.get(session_id)


manager = ConnectionManager()


@router.websocket("/live/{session_id}")
async def live_analysis_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time FMS analysis.
    
    Protocol:
    - Client sends: {"type": "start", "exercise": "deep_squat"}
    - Client sends: {"type": "frame", "data": "<base64 JPEG>"}
    - Server sends: LiveFeedback JSON
    - Client sends: {"type": "stop"} to end session
    """
    await manager.connect(websocket, session_id)
    
    try:
        analyzer = manager.get_analyzer(session_id)
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "start":
                exercise = data.get("exercise", "deep_squat")
                analyzer.start_session(exercise)
                await websocket.send_json({
                    "type": "started",
                    "session_id": analyzer.session_id,
                    "exercise": exercise
                })
            
            elif msg_type == "frame":
                # Decode base64 frame
                frame_data = data.get("data", "")
                if frame_data.startswith("data:image"):
                    frame_data = frame_data.split(",")[1]
                
                try:
                    img_bytes = base64.b64decode(frame_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Process frame
                        feedback = analyzer.process_frame(frame)
                        await websocket.send_json({
                            "type": "feedback",
                            **asdict(feedback)
                        })
                except Exception as e:
                    logger.error(f"Frame decode error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to decode frame"
                    })
            
            elif msg_type == "change_exercise":
                exercise = data.get("exercise", "deep_squat")
                analyzer.start_session(exercise)
                await websocket.send_json({
                    "type": "exercise_changed",
                    "exercise": exercise
                })
            
            elif msg_type == "stop":
                final_score = analyzer.get_final_score()
                await websocket.send_json({
                    "type": "stopped",
                    **final_score
                })
                break
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.exception(f"WebSocket error for {session_id}: {e}")
    finally:
        manager.disconnect(session_id)


@router.get("/exercises")
async def get_available_exercises():
    """Get list of exercises available for live analysis."""
    return {
        "exercises": [
            {
                "id": "deep_squat",
                "name": "Deep Squat",
                "description": "Overhead squat assessment",
                "instructions": [
                    "Stand with feet shoulder-width apart",
                    "Raise arms overhead with a dowel or broomstick",
                    "Squat as deep as you can while keeping heels down",
                    "Hold the bottom position briefly, then return"
                ]
            },
            {
                "id": "aslr_left",
                "name": "Active Straight Leg Raise (Left)",
                "description": "Left leg hip flexion assessment",
                "instructions": [
                    "Lie flat on your back",
                    "Keep both legs straight",
                    "Raise your left leg as high as possible",
                    "Keep your right leg flat on the ground"
                ]
            },
            {
                "id": "aslr_right",
                "name": "Active Straight Leg Raise (Right)",
                "description": "Right leg hip flexion assessment",
                "instructions": [
                    "Lie flat on your back",
                    "Keep both legs straight",
                    "Raise your right leg as high as possible",
                    "Keep your left leg flat on the ground"
                ]
            },
            {
                "id": "hurdle_step_left",
                "name": "Hurdle Step (Left)",
                "description": "Left leg stepping pattern assessment",
                "instructions": [
                    "Stand tall with feet together",
                    "Place a string or hurdle at tibial height",
                    "Step over with your left leg",
                    "Touch heel to ground and return"
                ]
            },
            {
                "id": "hurdle_step_right",
                "name": "Hurdle Step (Right)",
                "description": "Right leg stepping pattern assessment",
                "instructions": [
                    "Stand tall with feet together",
                    "Place a string or hurdle at tibial height",
                    "Step over with your right leg",
                    "Touch heel to ground and return"
                ]
            },
            {
                "id": "trunk_stability_pushup",
                "name": "Trunk Stability Push-up",
                "description": "Core stability assessment",
                "instructions": [
                    "Lie face down with hands at appropriate position",
                    "Men: thumbs aligned with top of head",
                    "Women: thumbs aligned with chin",
                    "Perform a push-up while keeping body rigid"
                ]
            }
        ]
    }
