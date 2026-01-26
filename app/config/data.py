"""
Data path and schema configuration.

Defines where to find data files and how they are structured.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# DATA PATHS
# =============================================================================

# Base directory for all data files (relative to this file's location)
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_BASE_PATH = os.path.join(_CONFIG_DIR, "..", "data")
DATA_BASE_PATH = os.getenv("DATA_BASE_PATH", DEFAULT_DATA_BASE_PATH)
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
