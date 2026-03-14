"""PDF report generation for FMS assessments."""
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

from app.models.schemas import FMSReport, TestScore, FMSTest


class ReportGenerator:
    """Generate PDF reports from FMS assessments."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()
    
    def _setup_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='Title',
            parent=self.styles['Heading1'],
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=12*mm
        ))
        
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            textColor=colors.grey,
            spaceAfter=8*mm
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=8*mm,
            spaceAfter=4*mm,
            textColor=colors.HexColor('#2563eb')
        ))
        
        self.styles.add(ParagraphStyle(
            name='Body',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=4*mm
        ))
        
        self.styles.add(ParagraphStyle(
            name='ScoreText',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_LEFT
        ))
    
    def generate_pdf(self, report: FMSReport) -> bytes:
        """
        Generate PDF report from FMS assessment.
        
        Args:
            report: FMSReport object
            
        Returns:
            PDF file contents as bytes
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        elements = []
        
        # Title
        elements.append(Paragraph(
            "Functional Movement Screen Report",
            self.styles['Title']
        ))
        
        # Subtitle with date
        date_str = report.created_at.strftime("%B %d, %Y")
        elements.append(Paragraph(
            f"Assessment Date: {date_str}",
            self.styles['Subtitle']
        ))
        
        elements.append(Spacer(1, 8*mm))
        
        # Summary box
        elements.append(self._create_summary_box(report))
        elements.append(Spacer(1, 8*mm))
        
        # Score breakdown
        elements.append(Paragraph("Score Breakdown", self.styles['SectionHeader']))
        elements.append(self._create_score_table(report))
        elements.append(Spacer(1, 8*mm))
        
        # Detailed results
        elements.append(Paragraph("Detailed Results", self.styles['SectionHeader']))
        for score in report.test_scores:
            elements.extend(self._create_test_section(score))
        
        # Recommendations
        if report.recommendations:
            elements.append(Paragraph("Recommendations", self.styles['SectionHeader']))
            for rec in report.recommendations:
                elements.append(Paragraph(f"• {rec}", self.styles['Body']))
        
        # Footer info
        elements.append(Spacer(1, 12*mm))
        elements.append(Paragraph(
            f"<i>Video analyzed: {report.video_filename} | "
            f"Duration: {report.duration_seconds:.1f}s | "
            f"Job ID: {report.job_id}</i>",
            self.styles['Subtitle']
        ))
        
        doc.build(elements)
        return buffer.getvalue()
    
    def _create_summary_box(self, report: FMSReport) -> Table:
        """Create summary statistics box."""
        score_color = self._get_score_color(report.total_score, 21)
        
        data = [
            ['Total Score', f'{report.total_score}/21'],
            ['Movement Quality', self._get_quality_label(report.total_score)],
            ['Tests Completed', str(len(report.test_scores))],
        ]
        
        table = Table(data, colWidths=[6*cm, 8*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
            ('TEXTCOLOR', (1, 0), (1, 0), score_color),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('FONTSIZE', (1, 0), (1, 0), 18),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ]))
        
        return table
    
    def _create_score_table(self, report: FMSReport) -> Table:
        """Create score breakdown table."""
        data = [
            ['Test', 'Score', 'Status']
        ]
        
        tests = [
            ('Deep Squat', report.deep_squat.score if report.deep_squat else None),
            ('Hurdle Step', report.hurdle_step),
            ('Inline Lunge', report.inline_lunge),
            ('Shoulder Mobility', report.shoulder_mobility),
            ('Active Straight Leg Raise', report.active_straight_leg_raise),
            ('Trunk Stability Push-up', 
             report.trunk_stability_pushup.score if report.trunk_stability_pushup else None),
            ('Rotary Stability', report.rotary_stability),
        ]
        
        for name, score in tests:
            if score is not None:
                status = self._get_status_emoji(score)
                data.append([name, str(score), status])
            else:
                data.append([name, '-', 'Not tested'])
        
        table = Table(data, colWidths=[8*cm, 3*cm, 3*cm])
        
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ]
        
        # Color code scores
        for i, (_, score) in enumerate(tests, start=1):
            if score is not None:
                color = self._get_score_color(score, 3)
                style.append(('TEXTCOLOR', (1, i), (1, i), color))
        
        table.setStyle(TableStyle(style))
        return table
    
    def _create_test_section(self, score: TestScore) -> list:
        """Create detailed section for a single test."""
        elements = []
        
        test_name = self._format_test_name(score.test)
        elements.append(Paragraph(
            f"<b>{test_name}</b> — Score: {score.score}/3",
            self.styles['ScoreText']
        ))
        
        if score.notes:
            elements.append(Paragraph(score.notes, self.styles['Body']))
        
        if score.faults:
            fault_text = "Faults detected: "
            fault_text += "; ".join([f.description for f in score.faults])
            elements.append(Paragraph(
                f"<i>{fault_text}</i>",
                self.styles['Body']
            ))
        
        elements.append(Spacer(1, 4*mm))
        return elements
    
    def _format_test_name(self, test: FMSTest) -> str:
        """Convert test enum to readable name."""
        mapping = {
            FMSTest.DEEP_SQUAT: "Deep Squat",
            FMSTest.HURDLE_STEP_LEFT: "Hurdle Step (Left)",
            FMSTest.HURDLE_STEP_RIGHT: "Hurdle Step (Right)",
            FMSTest.INLINE_LUNGE_LEFT: "Inline Lunge (Left)",
            FMSTest.INLINE_LUNGE_RIGHT: "Inline Lunge (Right)",
            FMSTest.SHOULDER_MOBILITY_LEFT: "Shoulder Mobility (Left)",
            FMSTest.SHOULDER_MOBILITY_RIGHT: "Shoulder Mobility (Right)",
            FMSTest.ACTIVE_STRAIGHT_LEG_RAISE_LEFT: "Active Straight Leg Raise (Left)",
            FMSTest.ACTIVE_STRAIGHT_LEG_RAISE_RIGHT: "Active Straight Leg Raise (Right)",
            FMSTest.TRUNK_STABILITY_PUSHUP: "Trunk Stability Push-up",
            FMSTest.ROTARY_STABILITY_LEFT: "Rotary Stability (Left)",
            FMSTest.ROTARY_STABILITY_RIGHT: "Rotary Stability (Right)",
        }
        return mapping.get(test, str(test))
    
    def _get_score_color(self, score: int, max_score: int) -> colors.Color:
        """Get color based on score percentage."""
        ratio = score / max_score
        if ratio >= 0.85:
            return colors.HexColor('#16a34a')  # Green
        elif ratio >= 0.65:
            return colors.HexColor('#ca8a04')  # Yellow
        elif ratio >= 0.45:
            return colors.HexColor('#ea580c')  # Orange
        else:
            return colors.HexColor('#dc2626')  # Red
    
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
    
    def _get_status_emoji(self, score: int) -> str:
        """Get status text for score."""
        if score == 3:
            return "Excellent"
        elif score == 2:
            return "Fair"
        elif score == 1:
            return "Poor"
        else:
            return "Pain - Refer"


# Singleton
_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """Get or create report generator instance."""
    global _generator
    if _generator is None:
        _generator = ReportGenerator()
    return _generator
