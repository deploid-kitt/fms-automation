"""FMS-specific prompts for LLM integration."""

# System prompts for different FMS tasks
COACHING_SYSTEM_PROMPT = """You are an expert FMS (Functional Movement Screen) coach providing real-time feedback.

Your role:
- Give clear, concise, actionable coaching cues
- Use simple language that's easy to understand during movement
- Focus on the most important correction first
- Be encouraging but direct

Guidelines:
- Keep cues under 10 words when possible
- Use positive language ("push knees out" not "don't let knees cave")
- Focus on body position and movement quality
- Reference specific body parts when needed

You will receive movement data including scores, faults, and metrics. Generate appropriate coaching cues."""

MOVEMENT_ANALYSIS_SYSTEM_PROMPT = """You are an expert FMS (Functional Movement Screen) analyst and movement specialist.

Your role:
- Analyze movement patterns and biomechanics
- Identify compensations and movement dysfunctions
- Explain the clinical significance of findings
- Provide detailed technical analysis

Guidelines:
- Use precise anatomical terminology
- Reference relevant biomechanical principles
- Consider the kinetic chain and its implications
- Identify root causes vs compensations
- Be thorough but organized

You will receive detailed movement data including joint angles, scores, and detected faults."""

REPORT_SYSTEM_PROMPT = """You are an expert FMS (Functional Movement Screen) analyst writing comprehensive assessment reports.

Your role:
- Create professional, detailed assessment reports
- Synthesize data into meaningful insights
- Provide evidence-based recommendations
- Use clear, professional language appropriate for healthcare/fitness professionals

Report structure:
1. Executive Summary
2. Individual Test Analysis
3. Pattern Recognition
4. Clinical Implications
5. Prioritized Recommendations

Guidelines:
- Be thorough and systematic
- Support conclusions with observed data
- Provide specific, actionable recommendations
- Consider the whole-body movement system
- Use professional medical/fitness terminology appropriately"""

EXERCISE_CLASSIFICATION_SYSTEM_PROMPT = """You are an FMS (Functional Movement Screen) expert identifying exercises from movement data.

Your role:
- Identify which FMS test is being performed
- Distinguish between similar movement patterns
- Handle edge cases and ambiguous movements

FMS Tests to identify:
1. Deep Squat
2. Hurdle Step (Left/Right)
3. Inline Lunge (Left/Right)
4. Shoulder Mobility (Left/Right)
5. Active Straight Leg Raise (Left/Right)
6. Trunk Stability Push-up
7. Rotary Stability (Left/Right)

Output format: Return only the exercise ID in lowercase with underscores (e.g., "deep_squat", "hurdle_step_left")."""


def get_coaching_cue_prompt(
    exercise: str,
    score: int,
    form_quality: str,
    faults: list[str],
    metrics: dict,
    previous_cues: list[str] = None,
) -> str:
    """Generate prompt for coaching cue generation."""
    
    faults_text = ", ".join(faults) if faults else "none detected"
    metrics_text = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
    previous_text = ", ".join(previous_cues[-3:]) if previous_cues else "none"
    
    return f"""Current Exercise: {exercise}
Current Score: {score}/3
Form Quality: {form_quality}
Detected Faults: {faults_text}
Key Metrics: {metrics_text}
Recent Cues Given: {previous_text}

Generate a single coaching cue (under 10 words) to help improve form. 
If form is excellent, give an encouraging confirmation.
Avoid repeating recent cues unless they are still the most important issue.

Respond with ONLY the coaching cue, nothing else."""


def get_movement_analysis_prompt(
    exercise: str,
    landmarks_summary: dict,
    metrics: dict,
    faults: list[dict],
    score: int,
) -> str:
    """Generate prompt for detailed movement analysis."""
    
    faults_text = "\n".join([
        f"- {f['code']}: {f['description']} (severity: {f.get('severity', 'unknown')})"
        for f in faults
    ]) if faults else "No significant faults detected"
    
    metrics_text = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
    
    return f"""Analyze the following FMS {exercise} performance:

## Score
{score}/3

## Key Metrics
{metrics_text}

## Detected Faults
{faults_text}

## Landmark Data Summary
{landmarks_summary}

Please provide:
1. Analysis of the movement quality
2. Biomechanical implications of observed patterns
3. Possible underlying movement dysfunctions
4. Specific areas to address in training
5. Brief recommendations for improvement

Keep the analysis focused and clinically relevant."""


def get_report_generation_prompt(
    test_scores: list[dict],
    total_score: int,
    video_duration: float,
    frames_analyzed: int,
) -> str:
    """Generate prompt for comprehensive report generation."""
    
    scores_text = "\n".join([
        f"- {t['test']}: {t['score']}/3 (confidence: {t.get('confidence', 0)*100:.0f}%)"
        + (f"\n  Faults: {', '.join([f['description'] for f in t.get('faults', [])])}" if t.get('faults') else "")
        + (f"\n  Notes: {t.get('notes', '')}" if t.get('notes') else "")
        for t in test_scores
    ])
    
    return f"""Generate a comprehensive FMS Assessment Report based on the following data:

## Overall Score
{total_score}/21

## Video Analysis Details
- Duration: {video_duration:.1f} seconds
- Frames analyzed: {frames_analyzed}

## Individual Test Scores
{scores_text}

Please create a professional report including:

1. **Executive Summary** (2-3 sentences summarizing overall movement quality)

2. **Strengths** (movement patterns that scored well)

3. **Areas for Improvement** (patterns that need work, prioritized by importance)

4. **Movement Pattern Analysis** (discuss any recurring compensations or dysfunctions across tests)

5. **Recommendations** (3-5 specific, actionable recommendations for improvement)

Write in a professional tone suitable for healthcare/fitness professionals while being accessible to the client."""


def get_enhanced_scoring_prompt(
    exercise: str,
    rule_based_score: int,
    metrics: dict,
    faults: list[dict],
    landmarks_quality: float,
) -> str:
    """Generate prompt for LLM-enhanced scoring verification."""
    
    metrics_text = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
    faults_text = "\n".join([
        f"- {f['description']} (severity: {f.get('severity', 'minor')})"
        for f in faults
    ]) if faults else "None detected"
    
    return f"""Review this FMS {exercise} scoring:

## Rule-Based Score: {rule_based_score}/3

## Metrics
{metrics_text}

## Detected Faults
{faults_text}

## Data Quality
Landmark visibility: {landmarks_quality*100:.0f}%

Based on FMS scoring criteria, does the rule-based score of {rule_based_score} seem appropriate?

Consider:
- Score 3: Performs pattern correctly with no compensations
- Score 2: Performs pattern with compensations
- Score 1: Cannot perform pattern
- Score 0: Pain during movement

Respond with:
1. AGREE or SUGGEST_ADJUSTMENT
2. If adjusting, the recommended score (0-3)
3. Brief reasoning (1-2 sentences)

Format: AGREE|SUGGEST_ADJUSTMENT:<score>|<reasoning>"""


def get_exercise_classification_prompt(
    joint_angles: dict,
    body_position: str,
    movement_phase: str,
) -> str:
    """Generate prompt for exercise classification."""
    
    angles_text = ", ".join([f"{k}: {v}°" for k, v in joint_angles.items()])
    
    return f"""Identify the FMS exercise based on this movement data:

Body Position: {body_position}
Movement Phase: {movement_phase}
Key Joint Angles: {angles_text}

Which FMS test is being performed? Respond with ONLY the exercise ID:
- deep_squat
- hurdle_step_left / hurdle_step_right
- inline_lunge_left / inline_lunge_right
- shoulder_mobility_left / shoulder_mobility_right
- aslr_left / aslr_right
- trunk_stability_pushup
- rotary_stability_left / rotary_stability_right

Exercise ID:"""
