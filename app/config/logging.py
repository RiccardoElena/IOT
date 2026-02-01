"""
Logging configuration for IoT Financial Data Analytics.

Centralizes all logging settings using loguru for better UX.
"""

# =============================================================================
# LOG LEVELS
# =============================================================================

# Console log level
# Options: "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
LOG_LEVEL = "INFO"

# Enable file logging
ENABLE_FILE_LOGGING = False

# Log file settings (if file logging enabled)
LOG_FILE_PATH = "logs/app.log"
LOG_FILE_ROTATION = "10 MB"  # Rotate when file reaches 10MB
LOG_FILE_RETENTION = "1 week"  # Keep logs for 1 week
LOG_FILE_COMPRESSION = "zip"  # Compress rotated logs


# =============================================================================
# LOG FORMAT
# =============================================================================

# Loguru format string
# Available tokens: {time}, {level}, {name}, {function}, {line}, {message}
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
    "<level>{message}</level>"
)

# Simplified format for production (less verbose)
LOG_FORMAT_PRODUCTION = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}"
