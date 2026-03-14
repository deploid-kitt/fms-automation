"""FMS-specific LLM services for enhanced analysis."""
import asyncio
import logging
import re
from typing import Optional
from dataclasses import dataclass

from app.llm.manager import get_llm_manager, LLMManager, ModelPreferences
from app.llm.base import ModelCapability, LLMResponse
from app.llm.fms_prompts import (
    COACHING_SYSTEM_PROMPT,
    MOVEMENT_ANALYSIS_SYSTEM_PROMPT,
    REPORT_SYSTEM_PROMPT,
    EXERCISE_CLASSIFICATION_SYSTEM_PROMPT,
    get_coaching_cue_prompt,
    get_movement_analysis_prompt,
    get_report_generation_prompt,
    get_enhanced_scoring_prompt,
    get_exercise_classification_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class CoachingCueResult:
    """Result from coaching cue generation."""
    cue: str
    model_used: str
    latency_ms: float
    from_cache: bool
    success: bool
    error: Optional[str] = None


@dataclass
class MovementAnalysisResult:
    """Result from movement analysis."""
    analysis: str
    model_used: str
    latency_ms: float
    from_cache: bool
    success: bool
    error: Optional[str] = None


@dataclass
class EnhancedScoreResult:
    """Result from enhanced scoring."""
    agreed: bool
    suggested_score: Optional[int]
    reasoning: str
    model_used: str
    latency_ms: float
    success: bool


class FMSLLMService:
    """
    Service for FMS-specific LLM operations.
    
    Provides high-level methods for:
    - Real-time coaching cue generation
    - Movement analysis
    - Report generation
    - Score verification
    - Exercise classification
    """
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        self.manager = llm_manager or get_llm_manager()
        self._cue_history: list[str] = []
        self._max_cue_history = 10
    
    async def generate_coaching_cue(
        self,
        exercise: str,
        score: int,
        form_quality: str,
        faults: list[str],
        metrics: dict,
        timeout: float = 3.0,  # Very short timeout for real-time
    ) -> CoachingCueResult:
        """
        Generate a coaching cue for real-time feedback.
        
        Uses fast models with aggressive timeouts and caching.
        """
        prompt = get_coaching_cue_prompt(
            exercise=exercise,
            score=score,
            form_quality=form_quality,
            faults=faults,
            metrics=metrics,
            previous_cues=self._cue_history,
        )
        
        response = await self.manager.complete_for_realtime(
            prompt=prompt,
            system_prompt=COACHING_SYSTEM_PROMPT,
            timeout=timeout,
        )
        
        if response.success:
            cue = response.content.strip()
            
            # Clean up the cue (remove quotes, etc.)
            cue = cue.strip('"\'')
            
            # Add to history
            self._cue_history.append(cue)
            if len(self._cue_history) > self._max_cue_history:
                self._cue_history.pop(0)
            
            return CoachingCueResult(
                cue=cue,
                model_used=response.model,
                latency_ms=response.latency_ms,
                from_cache=response.from_cache,
                success=True,
            )
        
        # Return a fallback cue on failure
        fallback_cue = self._get_fallback_cue(faults, form_quality)
        return CoachingCueResult(
            cue=fallback_cue,
            model_used="fallback",
            latency_ms=0,
            from_cache=False,
            success=False,
            error=response.error,
        )
    
    def _get_fallback_cue(self, faults: list[str], form_quality: str) -> str:
        """Get a rule-based fallback cue when LLM fails."""
        if form_quality == "excellent":
            return "Great form! Keep it up"
        
        if not faults:
            return "Focus on controlled movement"
        
        # Map common faults to cues
        fault_cues = {
            "trunk_forward": "Keep your chest up",
            "knee_valgus": "Push your knees out",
            "depth": "Go deeper if you can",
            "arms": "Keep arms overhead",
            "pelvis": "Keep your hips level",
            "hip_sag": "Engage your core",
        }
        
        for fault in faults:
            fault_lower = fault.lower()
            for key, cue in fault_cues.items():
                if key in fault_lower:
                    return cue
        
        return "Focus on your form"
    
    async def analyze_movement(
        self,
        exercise: str,
        landmarks_summary: dict,
        metrics: dict,
        faults: list[dict],
        score: int,
        timeout: float = 15.0,
    ) -> MovementAnalysisResult:
        """
        Generate detailed movement analysis.
        
        Uses more capable models for thorough analysis.
        """
        prompt = get_movement_analysis_prompt(
            exercise=exercise,
            landmarks_summary=landmarks_summary,
            metrics=metrics,
            faults=faults,
            score=score,
        )
        
        response = await self.manager.complete_for_analysis(
            prompt=prompt,
            system_prompt=MOVEMENT_ANALYSIS_SYSTEM_PROMPT,
            timeout=timeout,
        )
        
        if response.success:
            return MovementAnalysisResult(
                analysis=response.content,
                model_used=response.model,
                latency_ms=response.latency_ms,
                from_cache=response.from_cache,
                success=True,
            )
        
        return MovementAnalysisResult(
            analysis="Movement analysis unavailable.",
            model_used="none",
            latency_ms=0,
            from_cache=False,
            success=False,
            error=response.error,
        )
    
    async def generate_report_content(
        self,
        test_scores: list[dict],
        total_score: int,
        video_duration: float,
        frames_analyzed: int,
        timeout: float = 45.0,
    ) -> tuple[str, str, list[str]]:
        """
        Generate enhanced report content.
        
        Returns:
            Tuple of (summary, analysis, recommendations)
        """
        prompt = get_report_generation_prompt(
            test_scores=test_scores,
            total_score=total_score,
            video_duration=video_duration,
            frames_analyzed=frames_analyzed,
        )
        
        response = await self.manager.complete_for_report(
            prompt=prompt,
            system_prompt=REPORT_SYSTEM_PROMPT,
            timeout=timeout,
        )
        
        if not response.success:
            # Return basic fallback content
            return (
                f"FMS assessment completed with a total score of {total_score}/21.",
                "Detailed analysis unavailable.",
                ["Continue with current training program."],
            )
        
        # Parse the report content
        content = response.content
        
        # Extract sections (simple parsing)
        summary = self._extract_section(content, "Executive Summary", "Strengths")
        analysis = self._extract_section(content, "Movement Pattern Analysis", "Recommendations")
        
        # Extract recommendations as list
        rec_section = self._extract_section(content, "Recommendations", None)
        recommendations = self._parse_recommendations(rec_section)
        
        if not summary:
            summary = f"FMS assessment completed with a total score of {total_score}/21."
        
        if not analysis:
            analysis = content  # Use full content as analysis
        
        if not recommendations:
            recommendations = ["Review individual test results for specific areas to address."]
        
        return (summary, analysis, recommendations)
    
    def _extract_section(self, content: str, start_header: str, end_header: Optional[str]) -> str:
        """Extract a section from the report content."""
        # Look for section headers with ** markers or # markers
        patterns = [
            rf"\*\*{start_header}\*\*\s*(.*?)(?:\*\*{end_header}\*\*|$)" if end_header else rf"\*\*{start_header}\*\*\s*(.*)",
            rf"#{1,3}\s*{start_header}\s*(.*?)(?:#{1,3}\s*{end_header}|$)" if end_header else rf"#{1,3}\s*{start_header}\s*(.*)",
            rf"{start_header}:?\s*(.*?)(?:{end_header}|$)" if end_header else rf"{start_header}:?\s*(.*)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _parse_recommendations(self, text: str) -> list[str]:
        """Parse recommendations from text."""
        if not text:
            return []
        
        # Split by numbered items or bullet points
        lines = text.split('\n')
        recommendations = []
        
        for line in lines:
            line = line.strip()
            # Remove numbering and bullets
            line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)
            
            if line and len(line) > 10:  # Filter out very short lines
                recommendations.append(line)
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    async def verify_score(
        self,
        exercise: str,
        rule_based_score: int,
        metrics: dict,
        faults: list[dict],
        landmarks_quality: float,
        timeout: float = 10.0,
    ) -> EnhancedScoreResult:
        """
        Use LLM to verify/enhance rule-based scoring.
        
        This can catch edge cases the rule-based system might miss.
        """
        prompt = get_enhanced_scoring_prompt(
            exercise=exercise,
            rule_based_score=rule_based_score,
            metrics=metrics,
            faults=faults,
            landmarks_quality=landmarks_quality,
        )
        
        response = await self.manager.complete(
            prompt=prompt,
            capability=ModelCapability.MOVEMENT_ANALYSIS,
            temperature=0.3,  # Low temperature for consistency
            max_tokens=200,
            timeout=timeout,
        )
        
        if not response.success:
            return EnhancedScoreResult(
                agreed=True,  # Default to agreeing if LLM fails
                suggested_score=None,
                reasoning="LLM verification unavailable",
                model_used="none",
                latency_ms=0,
                success=False,
            )
        
        # Parse the response
        content = response.content.strip()
        
        if content.startswith("AGREE"):
            return EnhancedScoreResult(
                agreed=True,
                suggested_score=None,
                reasoning=content.split("|")[-1] if "|" in content else "Score confirmed",
                model_used=response.model,
                latency_ms=response.latency_ms,
                success=True,
            )
        
        # Parse adjustment suggestion
        match = re.match(r'SUGGEST_ADJUSTMENT:(\d)', content)
        if match:
            suggested = int(match.group(1))
            reasoning = content.split("|")[-1] if "|" in content else "Score adjustment suggested"
            
            return EnhancedScoreResult(
                agreed=False,
                suggested_score=min(3, max(0, suggested)),
                reasoning=reasoning,
                model_used=response.model,
                latency_ms=response.latency_ms,
                success=True,
            )
        
        # Couldn't parse, default to agree
        return EnhancedScoreResult(
            agreed=True,
            suggested_score=None,
            reasoning=content[:100],  # Use first 100 chars as reasoning
            model_used=response.model,
            latency_ms=response.latency_ms,
            success=True,
        )
    
    async def classify_exercise(
        self,
        joint_angles: dict,
        body_position: str,
        movement_phase: str,
        timeout: float = 5.0,
    ) -> Optional[str]:
        """
        Use LLM to classify exercise from movement data.
        
        Useful for edge cases where rule-based classification is uncertain.
        """
        prompt = get_exercise_classification_prompt(
            joint_angles=joint_angles,
            body_position=body_position,
            movement_phase=movement_phase,
        )
        
        response = await self.manager.complete(
            prompt=prompt,
            capability=ModelCapability.EXERCISE_CLASSIFICATION,
            system_prompt=EXERCISE_CLASSIFICATION_SYSTEM_PROMPT,
            temperature=0.1,  # Very low for classification
            max_tokens=50,
            timeout=timeout,
        )
        
        if not response.success:
            return None
        
        # Clean and validate the response
        exercise_id = response.content.strip().lower().replace(" ", "_")
        
        valid_exercises = {
            "deep_squat", "hurdle_step_left", "hurdle_step_right",
            "inline_lunge_left", "inline_lunge_right",
            "shoulder_mobility_left", "shoulder_mobility_right",
            "aslr_left", "aslr_right",
            "trunk_stability_pushup",
            "rotary_stability_left", "rotary_stability_right",
        }
        
        if exercise_id in valid_exercises:
            return exercise_id
        
        # Try partial matching
        for valid in valid_exercises:
            if valid in exercise_id or exercise_id in valid:
                return valid
        
        return None
    
    def clear_cue_history(self):
        """Clear the coaching cue history (e.g., when starting new session)."""
        self._cue_history.clear()


# Singleton instance
_fms_service: Optional[FMSLLMService] = None


def get_fms_llm_service() -> FMSLLMService:
    """Get or create the FMS LLM service instance."""
    global _fms_service
    
    if _fms_service is None:
        _fms_service = FMSLLMService()
    
    return _fms_service
