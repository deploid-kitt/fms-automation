"""Pose estimation using MediaPipe BlazePose."""
import cv2
import numpy as np
import mediapipe as mp
from typing import Generator, Optional
from pathlib import Path
import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# MediaPipe landmark indices
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# Key joint indices for FMS analysis
class JointIdx:
    """Landmark indices for key joints."""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class PoseEstimator:
    """MediaPipe BlazePose wrapper for pose estimation."""
    
    def __init__(self):
        """Initialize pose estimator."""
        settings = get_settings()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=settings.pose_model_complexity,
            enable_segmentation=False,
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence
        )
        
    def process_frame(self, frame: np.ndarray) -> Optional[dict]:
        """
        Process a single frame and extract pose landmarks.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Dictionary with landmarks data or None if no pose detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
            
        # Extract landmarks
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
        
        # Extract world landmarks (3D in meters)
        world_landmarks = None
        if results.pose_world_landmarks:
            world_landmarks = []
            for wlm in results.pose_world_landmarks.landmark:
                world_landmarks.append({
                    "x": wlm.x,
                    "y": wlm.y,
                    "z": wlm.z,
                    "visibility": wlm.visibility
                })
        
        return {
            "landmarks": landmarks,
            "world_landmarks": world_landmarks
        }
    
    def process_video(
        self, 
        video_path: Path,
        target_fps: Optional[int] = None,
        max_frames: Optional[int] = None
    ) -> Generator[dict, None, None]:
        """
        Process video file and yield pose data for each frame.
        
        Args:
            video_path: Path to video file
            target_fps: Downsample to this FPS (None = use original)
            max_frames: Maximum frames to process
            
        Yields:
            Dictionary with frame_number, timestamp, and landmarks
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame skip for target FPS
        frame_skip = 1
        if target_fps and fps > target_fps:
            frame_skip = int(fps / target_fps)
        
        frame_count = 0
        processed_count = 0
        
        logger.info(f"Processing video: {video_path.name}, {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames if downsampling
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Process frame
            pose_data = self.process_frame(frame)
            
            timestamp = frame_count / fps
            
            yield {
                "frame_number": frame_count,
                "timestamp": timestamp,
                "landmarks": pose_data["landmarks"] if pose_data else None,
                "world_landmarks": pose_data["world_landmarks"] if pose_data else None,
                "pose_detected": pose_data is not None
            }
            
            frame_count += 1
            processed_count += 1
            
            if max_frames and processed_count >= max_frames:
                break
        
        cap.release()
        logger.info(f"Processed {processed_count} frames from {video_path.name}")
    
    def close(self):
        """Release resources."""
        self.pose.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def landmarks_to_array(landmarks: list[dict]) -> np.ndarray:
    """Convert landmarks list to numpy array (33, 4) - x, y, z, visibility."""
    return np.array([[lm["x"], lm["y"], lm["z"], lm["visibility"]] for lm in landmarks])


def get_joint_position(landmarks: np.ndarray, joint_idx: int) -> np.ndarray:
    """Get 3D position of a joint."""
    return landmarks[joint_idx, :3]


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate angle at point b formed by points a, b, c.
    
    Returns angle in degrees.
    """
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


def calculate_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(a - b)
