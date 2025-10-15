"""Application settings using Pydantic and environment variables."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys (now optional)
    openai_api_key: Optional[str] = None
    pexels_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: Optional[str] = None
    
    # Application settings
    app_name: str = "Script-to-Rough-Cut"
    debug: bool = False
    port: int = 8080
    
    # AI/Generation modes
    script_mode: str = "transformers"  # "template" | "transformers" | "openai"
    tts_mode: str = "gtts"  # "gtts" | "elevenlabs"
    image_mode: str = "pollinations"  # "pollinations" | "pexels" | "pillow"
    
    # File paths
    outputs_dir: str = "outputs"
    logs_dir: str = "logs"
    runs_dir: str = "outputs"  # Alias for outputs_dir for compatibility
    
    # Video settings
    target_duration: int = 30  # seconds
    video_width: int = 1920
    video_height: int = 1080
    video_fps: int = 24
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields instead of raising error


# Global settings instance
settings = Settings()
