"""
Configuration file for the IoT Financial Data Analytics project.

All configurable parameters are centralized here.
Change values here to adapt the project to different datasets.
"""

import os

# =============================================================================
# ASSET CONFIGURATION
# =============================================================================

# Asset identifiers (internal keys used throughout the code)
# Map: internal_key -> display_name (shown in UI)
ASSETS = {
    "sp500": "S&P 500",
    "gold": "Gold",
    "oil": "Oil",
    "usd": "USD Index",
    "btc": "Bitcoin"
}

# File names for each asset (without path)
# Map: internal_key -> filename
# Change these to match your actual file names
FILE_NAMES = {
    "sp500": "sp500.csv",
    "gold": "gold.csv",
    "oil": "oil.csv",
    "usd": "usd.csv",
    "btc": "btc.csv"
}

# =============================================================================
# DATA PATHS
# =============================================================================

# Base directory for all data files (relative to config.py location)
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BASE_PATH = os.path.join(_CONFIG_DIR, "..", "data")
DATA_BASE_PATH = os.path.abspath(DATA_BASE_PATH)  # Convert to absolute path

# Subdirectories for each granularity
GRANULARITY_PATHS = {
    "minute": "1-minute",
    "hourly": "2-hourly",
    "daily": "3-daily",
}

# Display names for granularities (shown in UI)
GRANULARITY_DISPLAY = {
    "minute": "Minute",
    "hourly": "Hourly",
    "daily": "Daily",
}

# =============================================================================
# COLUMN NAMES
# =============================================================================

# Expected column names in CSV files
# Change these if your CSV has different column names
COLUMNS = {
    "timestamp": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "vwap": "vw",
    "num_trades": "n"
}

# =============================================================================
# ANOMALY DETECTION PARAMETERS
# =============================================================================

# Z-score thresholds
ZSCORE_WARNING_THRESHOLD = 2.0  # Values above this are "suspicious"
ZSCORE_ANOMALY_THRESHOLD = 3.0  # Values above this are "anomalies"

# Percentile thresholds for anomaly detection
PERCENTILE_LOW = 1    # Below this percentile = anomaly
PERCENTILE_HIGH = 99  # Above this percentile = anomaly

# Percentage change thresholds (in %)
PCT_CHANGE_THRESHOLD_DAILY = 5.0   # Daily: > 5% is anomaly
PCT_CHANGE_THRESHOLD_MINUTE = 1.0  # Minute: > 1% is anomaly

# =============================================================================
# SLIDING WINDOW PARAMETERS (IoT Real-time)
# =============================================================================

# Default window sizes (in number of data points)
WINDOW_SIZE_MINUTE = 60   # 60 minutes = 1 hour
WINDOW_SIZE_DAILY = 20    # 20 days ~ 1 month

# Configurable range for UI slider
WINDOW_SIZE_MIN = 30
WINDOW_SIZE_MAX = 120

# =============================================================================
# CROSS-ASSET PARAMETERS
# =============================================================================

# Rolling correlation window (in days)
CORRELATION_WINDOW = 30

# Correlation thresholds
CORRELATION_STRONG_POSITIVE = 0.7
CORRELATION_STRONG_NEGATIVE = -0.7

# Minimum assets with anomalies to flag as "systemic event"
SYSTEMIC_EVENT_THRESHOLD = 3

# =============================================================================
# PATTERN RECOGNITION PARAMETERS
# =============================================================================

# Doji: body must be less than this fraction of total range
DOJI_BODY_RATIO = 0.1

# Hammer: lower shadow must be at least this multiple of body
HAMMER_SHADOW_RATIO = 2.0

# =============================================================================
# UI CONFIGURATION
# =============================================================================

# Page configuration
PAGE_TITLE = "IoT Financial Analytics"
PAGE_ICON = "📊"
LAYOUT = "wide"

# Color scheme for anomalies
COLOR_NORMAL = "#636EFA"      # Blue
COLOR_WARNING = "#FFA15A"     # Orange
COLOR_ANOMALY = "#EF553B"     # Red
COLOR_BULLISH = "#00CC96"     # Green
COLOR_BEARISH = "#EF553B"     # Red

# Marker sizes
MARKER_SIZE_NORMAL = 6
MARKER_SIZE_ANOMALY = 12
