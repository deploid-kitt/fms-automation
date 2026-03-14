"""Audio feedback generation for real-time coaching."""
import logging
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AudioCueType(str, Enum):
    """Types of audio cues."""
    INFO = "info"
    WARNING = "warning"
    ENCOURAGEMENT = "encouragement"
    CORRECTION = "correction"
    COUNTDOWN = "countdown"
    COMPLETION = "completion"


# Pre-defined coaching cues for each exercise
COACHING_CUES = {
    "deep_squat": {
        "start": "Let's begin the deep squat. Raise your arms overhead.",
        "preparing": "Get ready. Feet shoulder-width apart.",
        "good_form": "Great form! Keep it up.",
        "excellent": "Excellent squat depth!",
        
        # Corrections
        "chest_up": "Keep your chest up and proud.",
        "knees_out": "Push your knees out over your toes.",
        "heels_down": "Keep your heels pressed to the floor.",
        "arms_overhead": "Raise your arms higher.",
        "go_deeper": "Try to go a little deeper.",
        "trunk_forward": "Your trunk is leaning too far forward.",
        
        # Completion
        "complete": "Great job! You scored {score} out of 3.",
    },
    
    "aslr": {
        "start": "Lie flat on your back for the leg raise.",
        "preparing": "Keep both legs straight.",
        "good_form": "Good range of motion!",
        "excellent": "Excellent hip mobility!",
        
        # Corrections
        "pelvis_flat": "Keep your pelvis flat on the ground.",
        "other_leg_down": "Keep your other leg flat.",
        "leg_straight": "Keep your leg straight as you lift.",
        "lift_higher": "Try to lift your leg higher.",
        
        # Completion
        "complete": "Nice work! You achieved {angle} degrees of hip flexion.",
    },
    
    "hurdle_step": {
        "start": "Stand tall for the hurdle step.",
        "preparing": "Place your hands on your hips.",
        "good_form": "Good hip clearance!",
        "excellent": "Excellent balance and control!",
        
        # Corrections
        "hips_level": "Keep your hips level.",
        "stand_tall": "Stand tall, don't lean.",
        "lift_knee": "Lift your knee higher.",
        "balance": "Focus on your balance.",
        
        # Completion
        "complete": "Well done! You scored {score} out of 3.",
    },
    
    "trunk_stability_pushup": {
        "start": "Get into the push-up position.",
        "preparing": "Place your hands at the correct position.",
        "good_form": "Good body alignment!",
        "excellent": "Excellent core control!",
        
        # Corrections
        "body_straight": "Keep your body in a straight line.",
        "hips_up": "Your hips are sagging. Engage your core.",
        "control": "Move with control, not momentum.",
        
        # Completion
        "complete": "Great job! You maintained good form for {duration} seconds.",
    },
}

# Countdown phrases
COUNTDOWN = {
    3: "Three",
    2: "Two", 
    1: "One",
    0: "Go!"
}

# General encouragements
ENCOURAGEMENTS = [
    "You're doing great!",
    "Keep going!",
    "Almost there!",
    "Looking good!",
    "Nice work!",
    "That's it!",
    "Perfect!",
    "Excellent!",
]


class AudioFeedbackGenerator:
    """Generate audio feedback cues for real-time coaching."""
    
    def __init__(self):
        self.last_cue: Optional[str] = None
        self.cue_count = 0
        self.encouragement_index = 0
    
    def get_start_cue(self, exercise: str) -> str:
        """Get the starting cue for an exercise."""
        base_exercise = exercise.split("_")[0] if "_left" in exercise or "_right" in exercise else exercise
        if base_exercise in ["aslr"]:
            base_exercise = "aslr"
        elif base_exercise in ["hurdle", "hurdle_step"]:
            base_exercise = "hurdle_step"
        
        cues = COACHING_CUES.get(base_exercise, COACHING_CUES["deep_squat"])
        return cues.get("start", "Let's begin.")
    
    def get_preparing_cue(self, exercise: str) -> str:
        """Get the preparing cue for an exercise."""
        base_exercise = self._get_base_exercise(exercise)
        cues = COACHING_CUES.get(base_exercise, COACHING_CUES["deep_squat"])
        return cues.get("preparing", "Get ready.")
    
    def get_correction_cue(self, exercise: str, issue: str) -> Optional[str]:
        """Get a correction cue based on the detected issue."""
        base_exercise = self._get_base_exercise(exercise)
        cues = COACHING_CUES.get(base_exercise, {})
        
        # Map issue codes to cue keys
        issue_map = {
            "TRUNK_FORWARD": "trunk_forward",
            "KNEE_VALGUS": "knees_out",
            "ARMS_DROPPED": "arms_overhead",
            "DEPTH_INSUFFICIENT": "go_deeper",
            "HEELS_LIFTED": "heels_down",
            "PELVIS_ROTATION": "pelvis_flat",
            "CONTRA_LEG_LIFT": "other_leg_down",
            "TRUNK_LEAN": "stand_tall",
            "PELVIS_DROP": "hips_level",
            "HIP_SAG": "hips_up",
            "SPINE_FLEX": "body_straight",
        }
        
        cue_key = issue_map.get(issue)
        if cue_key:
            return cues.get(cue_key)
        
        return None
    
    def get_encouragement(self) -> str:
        """Get an encouragement phrase (cycles through list)."""
        phrase = ENCOURAGEMENTS[self.encouragement_index % len(ENCOURAGEMENTS)]
        self.encouragement_index += 1
        return phrase
    
    def get_completion_cue(self, exercise: str, **kwargs) -> str:
        """Get the completion cue with interpolated values."""
        base_exercise = self._get_base_exercise(exercise)
        cues = COACHING_CUES.get(base_exercise, COACHING_CUES["deep_squat"])
        template = cues.get("complete", "Great job! Test complete.")
        
        try:
            return template.format(**kwargs)
        except KeyError:
            return template
    
    def get_countdown(self, number: int) -> str:
        """Get countdown phrase."""
        return COUNTDOWN.get(number, str(number))
    
    def get_form_cue(self, exercise: str, form_quality: str) -> Optional[str]:
        """Get a cue based on current form quality."""
        base_exercise = self._get_base_exercise(exercise)
        cues = COACHING_CUES.get(base_exercise, COACHING_CUES["deep_squat"])
        
        if form_quality == "excellent":
            return cues.get("excellent")
        elif form_quality == "good":
            return cues.get("good_form")
        
        return None
    
    def _get_base_exercise(self, exercise: str) -> str:
        """Extract base exercise name from full exercise ID."""
        if exercise.startswith("aslr"):
            return "aslr"
        elif exercise.startswith("hurdle_step"):
            return "hurdle_step"
        return exercise


# Text-to-Speech integration placeholder
# In production, this would integrate with a TTS service

class TTSProvider:
    """Text-to-speech provider interface."""
    
    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech audio.
        
        Returns:
            Audio bytes (e.g., MP3 or WAV)
        """
        # Placeholder - in production, integrate with:
        # - Google Cloud TTS
        # - Amazon Polly
        # - Azure Cognitive Services
        # - ElevenLabs
        # - Local TTS (espeak, pyttsx3)
        
        logger.warning("TTS not implemented - returning empty audio")
        return b""
    
    def get_cache_key(self, text: str) -> str:
        """Generate cache key for audio file."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()


# Singleton instances
_feedback_generator: Optional[AudioFeedbackGenerator] = None
_tts_provider: Optional[TTSProvider] = None


def get_feedback_generator() -> AudioFeedbackGenerator:
    """Get or create feedback generator instance."""
    global _feedback_generator
    if _feedback_generator is None:
        _feedback_generator = AudioFeedbackGenerator()
    return _feedback_generator


def get_tts_provider() -> TTSProvider:
    """Get or create TTS provider instance."""
    global _tts_provider
    if _tts_provider is None:
        _tts_provider = TTSProvider()
    return _tts_provider
