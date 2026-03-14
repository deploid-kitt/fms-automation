"""Application configuration."""
from functools import lru_cache
from pathlib import Path
from typing import Optional
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "FMS Automation"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    upload_dir: Path = Path("/app/data/uploads")
    models_dir: Path = Path("/app/models")
    
    # Redis
    redis_url: str = "redis://redis:6379/0"
    
    # Processing
    max_video_size_mb: int = 500
    video_fps_target: int = 30
    max_video_duration_sec: int = 600  # 10 minutes
    
    # Pose Estimation
    pose_model_complexity: int = 1  # 0=lite, 1=full, 2=heavy (use 1 for real-time)
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Real-time Performance Settings
    live_frame_skip: int = 2  # Process every Nth frame (1=all, 2=half, 3=third)
    live_process_width: int = 320  # Downscale width for pose processing
    live_process_height: int = 240  # Downscale height for pose processing
    pose_smoothing_enabled: bool = True  # Enable pose landmark smoothing
    pose_smoothing_factor: float = 0.5  # Smoothing factor (0=no smooth, 1=max smooth)
    
    # LLM Integration
    enable_llm_reports: bool = True
    enable_llm_coaching: bool = True
    enable_llm_analysis: bool = True
    
    # LLM API Keys (optional - can be set via environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    ollama_url: str = "http://localhost:11434"
    
    # LLM Model Preferences
    llm_realtime_model: str = "gpt-4o-mini"
    llm_analysis_model: str = "claude-sonnet-4-20250514"
    llm_report_model: str = "claude-sonnet-4-20250514"
    llm_fallback_model: str = "gpt-3.5-turbo"
    
    # LLM Performance Settings
    llm_cache_enabled: bool = True
    llm_cache_ttl: int = 3600  # 1 hour
    llm_realtime_timeout: float = 3.0  # seconds
    llm_analysis_timeout: float = 30.0
    llm_report_timeout: float = 60.0
    
    # CORS
    cors_origins: list[str] = ["*"]
    
    class Config:
        env_prefix = "FMS_"
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
