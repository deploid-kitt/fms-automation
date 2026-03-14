"""Video processing service for FMS analysis."""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Generator
import logging
import json
from datetime import datetime
import uuid

from app.core.config import get_settings
from app.models.schemas import (
    VideoMetadata, FMSReport, TestScore, JobStatus, FMSTest
)
from app.ml.pose_estimator import landmarks_to_array
from app.ml.pose_models import get_pose_model_registry, BasePoseEstimator
from app.ml.exercise_classifier import RuleBasedClassifier
from app.ml.fms_scorer import FMSScorer

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process videos for FMS analysis."""
    
    def __init__(self, pose_model_id: Optional[str] = None):
        self.settings = get_settings()
        
        # Get pose estimator from registry (prefer upload-optimized model)
        self.registry = get_pose_model_registry()
        if pose_model_id:
            self.pose_estimator = self.registry.get_model(pose_model_id)
        else:
            self.pose_estimator = self.registry.get_model_for_mode("upload")
        
        self.classifier = RuleBasedClassifier()
        self.scorer = FMSScorer()
    
    def get_video_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from video file."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        metadata = VideoMetadata(
            filename=video_path.name,
            duration=cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            fps=cap.get(cv2.CAP_PROP_FPS),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        
        cap.release()
        return metadata
    
    def extract_poses(
        self, 
        video_path: Path,
        progress_callback: Optional[callable] = None
    ) -> list[dict]:
        """
        Extract pose data from all frames in video.
        
        Args:
            video_path: Path to video file
            progress_callback: Optional callback(progress: float, status: str)
            
        Returns:
            List of pose data dictionaries per frame
        """
        poses = []
        metadata = self.get_video_metadata(video_path)
        
        logger.info(f"Extracting poses from {video_path.name}: "
                   f"{metadata.total_frames} frames "
                   f"(model: {self.pose_estimator.config.model_id if self.pose_estimator else 'none'})")
        
        if not self.pose_estimator:
            logger.error("No pose estimator available")
            return poses
        
        for i, pose_data in enumerate(self.pose_estimator.process_video(
            video_path,
            target_fps=self.settings.video_fps_target
        )):
            poses.append(pose_data)
            
            if progress_callback and i % 30 == 0:
                progress = (i / metadata.total_frames) * 100
                progress_callback(progress, f"Extracting poses: frame {i}")
        
        logger.info(f"Extracted {len(poses)} pose frames")
        return poses
    
    def classify_exercises(
        self,
        poses: list[dict],
        progress_callback: Optional[callable] = None
    ) -> list[tuple]:
        """
        Classify which FMS exercises are performed in the video.
        
        Args:
            poses: List of pose data from extract_poses
            progress_callback: Optional callback
            
        Returns:
            List of (exercise_name, start_frame, end_frame, confidence)
        """
        # Filter frames with detected poses
        valid_poses = []
        frame_mapping = []
        
        for i, pose in enumerate(poses):
            if pose["landmarks"]:
                valid_poses.append(landmarks_to_array(pose["landmarks"]))
                frame_mapping.append(pose["frame_number"])
        
        if not valid_poses:
            logger.warning("No valid poses detected in video")
            return []
        
        logger.info(f"Classifying exercises from {len(valid_poses)} pose frames")
        
        # Classify sequence
        segments = self.classifier.classify_sequence(valid_poses, min_frames=15)
        
        # Map back to original frame numbers
        mapped_segments = []
        for exercise, start, end, confidence in segments:
            if exercise != "unknown":
                orig_start = frame_mapping[start] if start < len(frame_mapping) else start
                orig_end = frame_mapping[end] if end < len(frame_mapping) else end
                mapped_segments.append((exercise, orig_start, orig_end, confidence))
        
        logger.info(f"Detected {len(mapped_segments)} exercise segments")
        return mapped_segments
    
    def score_exercises(
        self,
        poses: list[dict],
        exercise_segments: list[tuple],
        progress_callback: Optional[callable] = None
    ) -> list[TestScore]:
        """
        Score each detected exercise segment.
        
        Args:
            poses: List of pose data
            exercise_segments: List of (exercise, start, end, confidence)
            progress_callback: Optional callback
            
        Returns:
            List of TestScore objects
        """
        scores = []
        
        # Build frame index for quick lookup
        frame_to_pose = {p["frame_number"]: p for p in poses if p["landmarks"]}
        
        for exercise, start_frame, end_frame, det_confidence in exercise_segments:
            logger.info(f"Scoring {exercise}: frames {start_frame}-{end_frame}")
            
            # Extract landmarks for this segment
            segment_landmarks = []
            for frame_num in range(start_frame, end_frame + 1):
                if frame_num in frame_to_pose:
                    lm = landmarks_to_array(frame_to_pose[frame_num]["landmarks"])
                    segment_landmarks.append(lm)
            
            if not segment_landmarks:
                continue
            
            # Score the exercise
            test_score = self.scorer.score_exercise(
                exercise,
                segment_landmarks,
                frame_range=(start_frame, end_frame)
            )
            
            if test_score:
                # Adjust confidence based on detection confidence
                test_score.confidence *= det_confidence
                scores.append(test_score)
        
        return scores
    
    def generate_report(
        self,
        job_id: str,
        video_path: Path,
        test_scores: list[TestScore]
    ) -> FMSReport:
        """
        Generate complete FMS report from test scores.
        
        Args:
            job_id: Unique job identifier
            video_path: Path to processed video
            test_scores: List of individual test scores
            
        Returns:
            Complete FMSReport
        """
        metadata = self.get_video_metadata(video_path)
        
        report = FMSReport(
            job_id=job_id,
            created_at=datetime.utcnow(),
            video_filename=video_path.name,
            duration_seconds=metadata.duration,
            total_frames=metadata.total_frames,
            test_scores=test_scores
        )
        
        # Calculate composite scores
        score_map = {}
        for ts in test_scores:
            score_map[ts.test] = ts.score
        
        # Deep squat
        if FMSTest.DEEP_SQUAT in score_map:
            report.deep_squat = next(
                (ts for ts in test_scores if ts.test == FMSTest.DEEP_SQUAT), 
                None
            )
        
        # Bilateral tests: take minimum of left/right
        def get_bilateral_score(left_test: FMSTest, right_test: FMSTest) -> Optional[int]:
            left = score_map.get(left_test)
            right = score_map.get(right_test)
            if left is not None and right is not None:
                return min(left, right)
            return left or right
        
        report.hurdle_step = get_bilateral_score(
            FMSTest.HURDLE_STEP_LEFT, FMSTest.HURDLE_STEP_RIGHT
        )
        report.inline_lunge = get_bilateral_score(
            FMSTest.INLINE_LUNGE_LEFT, FMSTest.INLINE_LUNGE_RIGHT
        )
        report.shoulder_mobility = get_bilateral_score(
            FMSTest.SHOULDER_MOBILITY_LEFT, FMSTest.SHOULDER_MOBILITY_RIGHT
        )
        report.active_straight_leg_raise = get_bilateral_score(
            FMSTest.ACTIVE_STRAIGHT_LEG_RAISE_LEFT, 
            FMSTest.ACTIVE_STRAIGHT_LEG_RAISE_RIGHT
        )
        report.rotary_stability = get_bilateral_score(
            FMSTest.ROTARY_STABILITY_LEFT, FMSTest.ROTARY_STABILITY_RIGHT
        )
        
        # Trunk stability
        if FMSTest.TRUNK_STABILITY_PUSHUP in score_map:
            report.trunk_stability_pushup = next(
                (ts for ts in test_scores if ts.test == FMSTest.TRUNK_STABILITY_PUSHUP),
                None
            )
        
        # Calculate total score
        composite_scores = [
            report.deep_squat.score if report.deep_squat else 0,
            report.hurdle_step or 0,
            report.inline_lunge or 0,
            report.shoulder_mobility or 0,
            report.active_straight_leg_raise or 0,
            report.trunk_stability_pushup.score if report.trunk_stability_pushup else 0,
            report.rotary_stability or 0
        ]
        report.total_score = sum(composite_scores)
        
        # Generate summary and recommendations
        report.summary = self._generate_summary(report)
        report.recommendations = self._generate_recommendations(test_scores)
        
        return report
    
    def _generate_summary(self, report: FMSReport) -> str:
        """Generate human-readable summary."""
        total = report.total_score
        max_score = 21
        
        if total >= 18:
            quality = "excellent movement quality"
        elif total >= 14:
            quality = "good movement quality with minor limitations"
        elif total >= 10:
            quality = "moderate movement dysfunction"
        else:
            quality = "significant movement dysfunction requiring attention"
        
        tests_performed = len([s for s in report.test_scores if s.score is not None])
        
        return (
            f"FMS Total Score: {total}/{max_score}. "
            f"Assessment indicates {quality}. "
            f"{tests_performed} tests analyzed from video."
        )
    
    def _generate_recommendations(self, test_scores: list[TestScore]) -> list[str]:
        """Generate recommendations based on scores and faults."""
        recommendations = []
        
        for ts in test_scores:
            if ts.score <= 1:
                rec = self._get_exercise_recommendation(ts.test, ts.faults)
                if rec:
                    recommendations.append(rec)
        
        # Add general recommendations
        low_scores = [ts for ts in test_scores if ts.score <= 1]
        if len(low_scores) > 2:
            recommendations.append(
                "Multiple movement dysfunctions detected. "
                "Consider comprehensive mobility and stability assessment."
            )
        
        return recommendations[:5]  # Limit to top 5
    
    def _get_exercise_recommendation(
        self, 
        test: FMSTest, 
        faults: list
    ) -> Optional[str]:
        """Get specific recommendation for a test."""
        recommendations = {
            FMSTest.DEEP_SQUAT: 
                "Focus on ankle mobility and hip flexor stretching. "
                "Practice goblet squats with pauses.",
            FMSTest.ACTIVE_STRAIGHT_LEG_RAISE_LEFT:
                "Address left hip flexor tightness and hamstring mobility.",
            FMSTest.ACTIVE_STRAIGHT_LEG_RAISE_RIGHT:
                "Address right hip flexor tightness and hamstring mobility.",
            FMSTest.HURDLE_STEP_LEFT:
                "Improve left hip stability and single-leg balance.",
            FMSTest.HURDLE_STEP_RIGHT:
                "Improve right hip stability and single-leg balance.",
            FMSTest.TRUNK_STABILITY_PUSHUP:
                "Strengthen core with planks and dead bugs before progressing.",
        }
        return recommendations.get(test)
    
    def process_video(
        self,
        video_path: Path,
        job_id: str,
        progress_callback: Optional[callable] = None
    ) -> FMSReport:
        """
        Complete video processing pipeline.
        
        Args:
            video_path: Path to video file
            job_id: Unique job identifier
            progress_callback: Optional callback(progress, status)
            
        Returns:
            Complete FMSReport
        """
        logger.info(f"Processing video: {video_path} (job: {job_id})")
        
        # Stage 1: Extract poses
        if progress_callback:
            progress_callback(0, JobStatus.EXTRACTING_POSES)
        poses = self.extract_poses(video_path, progress_callback)
        
        # Stage 2: Classify exercises
        if progress_callback:
            progress_callback(40, JobStatus.CLASSIFYING)
        segments = self.classify_exercises(poses)
        
        # Stage 3: Score exercises
        if progress_callback:
            progress_callback(60, JobStatus.SCORING)
        test_scores = self.score_exercises(poses, segments)
        
        # Stage 4: Generate report
        if progress_callback:
            progress_callback(80, JobStatus.GENERATING_REPORT)
        report = self.generate_report(job_id, video_path, test_scores)
        
        if progress_callback:
            progress_callback(100, JobStatus.COMPLETED)
        
        logger.info(f"Completed processing. Total score: {report.total_score}/21")
        return report
    
    def close(self):
        """Release resources."""
        # Pose estimator is managed by registry, don't close it here
        pass
    
    def switch_pose_model(self, model_id: str) -> bool:
        """Switch to a different pose estimation model."""
        new_model = self.registry.get_model(model_id)
        if new_model is None:
            logger.warning(f"Failed to load model: {model_id}")
            return False
        self.pose_estimator = new_model
        return True


# Singleton instance
_processor: Optional[VideoProcessor] = None


def get_video_processor() -> VideoProcessor:
    """Get or create video processor instance."""
    global _processor
    if _processor is None:
        _processor = VideoProcessor()
    return _processor
