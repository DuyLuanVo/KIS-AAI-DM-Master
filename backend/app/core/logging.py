"""
Logging configuration
"""
import sys

from app.config import settings
from loguru import logger


def setup_logging():
    """Setup logging configuration"""

    # Remove default logger
    logger.remove()

    # Add custom logger
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True
    )

    # Add file logger
    logger.add(
        "logs/app.log",
        level=settings.log_level,
        format=settings.log_format,
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )

    logger.info("Logging setup complete")


def get_logger(name: str = __name__):
    """Get logger instance"""
    return logger.bind(name=name)
