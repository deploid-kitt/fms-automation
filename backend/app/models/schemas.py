"""Pydantic schemas for API models."""
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FMSTest(str, Enum):
    """FMS test types."""
    DEEP_SQUAT = "deep_squat"
    HURDLE_STEP_LEFT = "hurdle_step_left"
    HURDLE_STEP_RIGHT = "hurdle_step_right"
    INLINE_LUNGE_LEFT = "inline_lunge_left"
    INLINE_LUNGE_RIGHT = "inline_lunge_right"
    SHOULDER_MOBILITY_LEFT = "shoulder_mobility_left"
    SHOULDER_MOBILITY_RIGHT = "shoulder_mobility_right"
    ACTIVE_STRAIGHT_LEG_RAISE_LEFT = "aslr_left"
    ACTIVE_STRAIGHT_LEG_RAISE_RIGHT = "aslr_right"
    TRUNK_STABILITY_PUSHUP = "trunk_stability_pushup"
    ROTARY_STABILITY_LEFT = "rotary_stability_left"
    ROTARY_STABILITY_RIGHT = "rotary_stability_right"


class JobStatus(str, Enum):
    """Processing job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    EXTRACTING_POSES = "extracting_poses"
    CLASSIFYING = "classifying"
    SCORING = "scoring"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"


class Fault(BaseModel):
    """Movement fault detected during test."""
    code: str
    description: str
    severity: str = "minor"  # minor, moderate, major
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None


class TestScore(BaseModel):
    """Score for a single FMS test."""
    test: FMSTest
    score: int = Field(ge=0, le=3)
    faults: list[Fault] = []
    notes: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    frame_range: tuple[int, int] = (0, 0)


class FMSReport(BaseModel):
    """Complete FMS assessment report."""
    job_id: str
    created_at: datetime
    video_filename: str
    duration_seconds: float
    total_frames: int
    
    # Composite scores (max of left/right for bilateral tests)
    deep_squat: Optional[TestScore] = None
    hurdle_step: Optional[int] = None  # min(left, right)
    inline_lunge: Optional[int] = None
    shoulder_mobility: Optional[int] = None
    active_straight_leg_raise: Optional[int] = None
    trunk_stability_pushup: Optional[TestScore] = None
    rotary_stability: Optional[int] = None
    
    # Individual test scores
    test_scores: list[TestScore] = []
    
    # Total score (sum of 7 composite scores, max 21)
    total_score: int = 0
    
    # Overall assessment
    summary: str = ""
    recommendations: list[str] = []


class UploadResponse(BaseModel):
    """Response after video upload."""
    job_id: str
    status: JobStatus
    message: str


class StatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    status: JobStatus
    progress: float = 0.0  # 0-100
    message: str = ""
    error: Optional[str] = None


class PoseFrame(BaseModel):
    """Pose data for a single frame."""
    frame_number: int
    timestamp: float
    landmarks: list[dict]  # List of {x, y, z, visibility} for 33 landmarks
    world_landmarks: Optional[list[dict]] = None  # 3D world coordinates


class VideoMetadata(BaseModel):
    """Video metadata."""
    filename: str
    duration: float
    fps: float
    width: int
    height: int
    total_frames: int
