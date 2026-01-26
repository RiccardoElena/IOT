"""
Centralized Logging with Loguru for IoT Financial Data Analytics.

This module configures loguru with settings from config.logging.
All modules should use this logger instead of print() statements.

Features:
- Colored console output (automatic)
- File logging with rotation (optional)
- Better exception formatting
- Zero boilerplate

Usage:
    from utils.logger import logger
    
    logger.info("Data loaded successfully")
    logger.warning("Missing values detected")
    logger.error("Failed to load CSV file")
    logger.success("Processing completed!")
"""

import sys
from pathlib import Path
from loguru import logger

# Import configuration
try:
    from config.logging import (
        LOG_LEVEL,
        ENABLE_FILE_LOGGING,
        LOG_FILE_PATH,
        LOG_FILE_ROTATION,
        LOG_FILE_RETENTION,
        LOG_FILE_COMPRESSION,
        LOG_FORMAT,
    )
except ImportError:
    # Fallback defaults if config not available
    LOG_LEVEL = "INFO"
    ENABLE_FILE_LOGGING = False
    LOG_FILE_PATH = "logs/app.log"
    LOG_FILE_ROTATION = "10 MB"
    LOG_FILE_RETENTION = "1 week"
    LOG_FILE_COMPRESSION = "zip"
    LOG_FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> | "
        "<level>{message}</level>"
    )


# =============================================================================
# LOGGER CONFIGURATION
# =============================================================================

def setup_logger() -> None:
    """
    Configure loguru with console and optional file handlers.
    
    This is called automatically when the module is imported.
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with colors
    logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        colorize=True,
    )
    
    # Add file handler if enabled
    if ENABLE_FILE_LOGGING:
        log_path = Path(LOG_FILE_PATH)
        log_path.parent.mkdir(exist_ok=True)
        
        logger.add(
            LOG_FILE_PATH,
            format=LOG_FORMAT,
            level="DEBUG",  # File gets more detailed logs
            rotation=LOG_FILE_ROTATION,
            retention=LOG_FILE_RETENTION,
            compression=LOG_FILE_COMPRESSION,
        )


# =============================================================================
# INITIALIZATION
# =============================================================================

# Setup logging when module is imported
setup_logger()

# Export logger instance
__all__ = ["logger"]
