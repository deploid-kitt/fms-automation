"""LLM-enhanced report generation for FMS assessments."""
import asyncio
import logging
from datetime import datetime
from typing import Optional

from app.models.schemas import FMSReport, TestScore
from app.llm.fms_service import get_fms_llm_service
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMReportEnhancer:
    """
    Enhances FMS reports with LLM-generated content.
    
    Works alongside the existing ReportGenerator to add:
    - Rich narrative summaries
    - Detailed movement analysis
    - Personalized recommendations
    """
    
    def __init__(self):
        self.llm_service = get_fms_llm_service() if settings.enable_llm_reports else None
        
    async def enhance_report(self, report: FMSReport) -> FMSReport:
        """
        Enhance a report with LLM-generated content.
        
        Args:
            report: The base FMS report with scores
            
        Returns:
            Enhanced report with LLM-generated summary and recommendations
        """
        if not self.llm_service or not settings.enable_llm_reports:
            logger.info("LLM reports disabled, returning base report")
            return report
        
        try:
            # Prepare test scores for LLM
            test_data = []
            for ts in report.test_scores:
                test_data.append({
                    "test": ts.test.value,
                    "score": ts.score,
                    "confidence": ts.confidence,
                    "faults": [{"code": f.code, "description": f.description} for f in ts.faults],
                    "notes": ts.notes,
                })
            
            # Generate enhanced content
            summary, analysis, recommendations = await self.llm_service.generate_report_content(
                test_scores=test_data,
                total_score=report.total_score,
                video_duration=report.duration_seconds,
                frames_analyzed=report.total_frames,
                timeout=settings.llm_report_timeout,
            )
            
            # Update report with LLM content
            report.summary = summary
            report.recommendations = recommendations
            
            # Add analysis as extended notes on the report
            # (This could be stored in a new field if needed)
            
            logger.info(f"Report enhanced with LLM content for job {report.job_id}")
            
        except Exception as e:
            logger.error(f"Failed to enhance report with LLM: {e}")
            # Fall back to basic summary
            if not report.summary:
                report.summary = self._generate_basic_summary(report)
            if not report.recommendations:
                report.recommendations = self._generate_basic_recommendations(report)
        
        return report
    
    def _generate_basic_summary(self, report: FMSReport) -> str:
        """Generate a basic summary when LLM is unavailable."""
        quality = self._get_quality_label(report.total_score)
        return (
            f"FMS assessment completed with a total score of {report.total_score}/21. "
            f"Overall movement quality is rated as {quality.lower()}."
        )
    
    def _generate_basic_recommendations(self, report: FMSReport) -> list[str]:
        """Generate basic recommendations when LLM is unavailable."""
        recommendations = []
        
        # Find lowest scoring tests
        low_scores = []
        for ts in report.test_scores:
            if ts.score <= 1:
                low_scores.append(ts.test.value.replace("_", " ").title())
        
        if low_scores:
            recommendations.append(
                f"Focus on mobility and stability exercises for: {', '.join(low_scores[:3])}"
            )
        
        if report.total_score < 14:
            recommendations.append(
                "Consider a comprehensive mobility assessment before high-intensity training"
            )
        
        if report.total_score >= 14:
            recommendations.append(
                "Continue with current training program while addressing identified limitations"
            )
        
        recommendations.append(
            "Re-assess in 4-6 weeks to track progress"
        )
        
        return recommendations
    
    def _get_quality_label(self, total_score: int) -> str:
        """Get movement quality label."""
        if total_score >= 18:
            return "Excellent"
        elif total_score >= 14:
            return "Good"
        elif total_score >= 10:
            return "Moderate"
        else:
            return "Needs Improvement"


# Singleton instance
_enhancer: Optional[LLMReportEnhancer] = None


def get_llm_report_enhancer() -> LLMReportEnhancer:
    """Get or create the LLM report enhancer instance."""
    global _enhancer
    if _enhancer is None:
        _enhancer = LLMReportEnhancer()
    return _enhancer


async def enhance_fms_report(report: FMSReport) -> FMSReport:
    """
    Convenience function to enhance a report.
    
    This is the main entry point for report enhancement.
    """
    enhancer = get_llm_report_enhancer()
    return await enhancer.enhance_report(report)
