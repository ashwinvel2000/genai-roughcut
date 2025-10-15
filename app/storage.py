"""Storage helpers and path management utilities."""

import os
import uuid
from pathlib import Path
from datetime import datetime
import logging
import random
import string

from app.settings import settings

logger = logging.getLogger(__name__)


def ensure_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        settings.outputs_dir,
        settings.logs_dir,
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")


def generate_run_id() -> str:
    """
    Generate a unique run ID for tracking pipeline executions.
    Format: YYYYmmdd_HHMMSS_random4
    
    Returns:
        Unique run ID string (e.g., 20231015_143022_a7b3)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{timestamp}_{random_chars}"


def run_dir(run_id: str) -> str:
    """
    Get the directory path for a specific run.
    
    Args:
        run_id: The run identifier
        
    Returns:
        Full path to the run directory
    """
    return os.path.join(settings.outputs_dir, run_id)


def ensure_run_dirs(run_id: str):
    """
    Create all necessary directories for a run.
    
    Args:
        run_id: The run identifier
    """
    run_path = run_dir(run_id)
    Path(run_path).mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured run directory exists: {run_path}")


def get_script_path(run_id: str) -> str:
    """
    Get the path for the script.json file.
    
    Args:
        run_id: The run identifier
        
    Returns:
        Full path to script.json
    """
    return os.path.join(run_dir(run_id), "script.json")


def get_voice_path(run_id: str) -> str:
    """
    Get the path for the voice.mp3 file.
    
    Args:
        run_id: The run identifier
        
    Returns:
        Full path to voice.mp3
    """
    return os.path.join(run_dir(run_id), "voice.mp3")


def get_output_video_path(run_id: str) -> str:
    """
    Get the path for the output.mp4 file.
    
    Args:
        run_id: The run identifier
        
    Returns:
        Full path to output.mp4
    """
    return os.path.join(run_dir(run_id), "output.mp4")


def get_log_path(run_id: str) -> str:
    """
    Get the path for the log.json file.
    
    Args:
        run_id: The run identifier
        
    Returns:
        Full path to log.json
    """
    return os.path.join(run_dir(run_id), "log.json")


def get_image_path(run_id: str, index: int) -> str:
    """
    Get the path for an image file.
    
    Args:
        run_id: The run identifier
        index: Image index (0-based)
        
    Returns:
        Full path to scene_{index+1}.jpg
    """
    return os.path.join(run_dir(run_id), f"scene_{index + 1}.jpg")


def get_all_image_paths(run_id: str, count: int = 5) -> list[str]:
    """
    Get paths for all image files.
    
    Args:
        run_id: The run identifier
        count: Number of images
        
    Returns:
        List of paths to image files
    """
    return [get_image_path(run_id, i) for i in range(count)]


def get_output_path(run_id: str, extension: str = "mp4") -> str:
    """
    Get the full path for an output file.
    
    Args:
        run_id: The run identifier
        extension: File extension (default: mp4)
        
    Returns:
        Full path to the output file
    """
    return os.path.join(settings.outputs_dir, f"{run_id}.{extension}")


def get_temp_path(run_id: str, filename: str) -> str:
    """
    Get a temporary file path for intermediate files.
    
    Args:
        run_id: The run identifier
        filename: Name of the temporary file
        
    Returns:
        Full path to the temporary file
    """
    temp_dir = os.path.join(settings.outputs_dir, run_id)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(temp_dir, filename)


def cleanup_temp_files(run_id: str):
    """
    Clean up temporary files for a run.
    
    Args:
        run_id: The run identifier
    """
    temp_dir = os.path.join(settings.outputs_dir, run_id)
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary files for run: {run_id}")


# Initialize directories on module import
ensure_directories()
