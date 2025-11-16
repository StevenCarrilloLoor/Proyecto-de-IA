"""
Utilidades generales del proyecto
"""

from .config_loader import load_config
from .video_utils import VideoProcessor
from .logger import setup_logger

__all__ = ['load_config', 'VideoProcessor', 'setup_logger']
