"""Application configuration."""
from functools import lru_cache
from pathlib import Path
from typing import Optional

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
    pose_model_complexity: int = 2  # 0, 1, or 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Scoring
    enable_llm_reports: bool = True
    
    # CORS
    cors_origins: list[str] = ["*"]
    
    class Config:
        env_prefix = "FMS_"
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
