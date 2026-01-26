"""
UI configuration and constants.

Page layout, colors, chart settings, and all visual parameters.
"""

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

# Page identifiers
PAGE_TYPE_SINGLE_ASSET = "single_asset"
PAGE_TYPE_REALTIME = "realtime"
PAGE_TYPE_CROSS_ASSET = "cross_asset"
PAGE_TYPE_PATTERNS = "patterns"

PAGE_TYPES = [
    PAGE_TYPE_SINGLE_ASSET,
    PAGE_TYPE_REALTIME,
    PAGE_TYPE_CROSS_ASSET,
    PAGE_TYPE_PATTERNS
]

# Streamlit page configuration
PAGE_TITLE = "IoT Financial Analytics"
PAGE_ICON = ""
LAYOUT = "wide"

# =============================================================================
# COLOR SCHEME
# =============================================================================

# Color scheme for anomalies and trends
COLOR_NORMAL = "#636EFA"      # Blue
COLOR_WARNING = "#FFA15A"     # Orange
COLOR_ANOMALY = "#EF553B"     # Red
COLOR_BULLISH = "#00CC96"     # Green
COLOR_BEARISH = "#EF553B"     # Red

# =============================================================================
# MARKER SIZES
# =============================================================================

# Marker sizes for charts
MARKER_SIZE_NORMAL = 6
MARKER_SIZE_ANOMALY = 12

# =============================================================================
# CHAT UI CONSTANTS
# =============================================================================

# Chat message container height
CHAT_MESSAGES_HEIGHT = 300

# Maximum number of charts that can be attached to Gemini context
MAX_CHARTS_IN_CONTEXT = 5

# Number of messages to display initially (lazy loading)
INITIAL_MESSAGE_DISPLAY_COUNT = 20

# =============================================================================
# CHART EXPORT SETTINGS
# =============================================================================

# Chart export dimensions
CHART_EXPORT_WIDTH = 1200
CHART_EXPORT_HEIGHT = 800
CHART_EXPORT_SCALE = 2.0

# =============================================================================
# AUTO-SCROLL TIMING
# =============================================================================

# Auto-scroll timing (milliseconds)
SCROLL_DEBOUNCE_MS = 150
SCROLL_INITIAL_WAIT_MS = 200

# =============================================================================
# CHART CONFIGURATION
# =============================================================================

# Default Plotly chart configuration
DEFAULT_CHART_CONFIG = {
    'displayModeBar': False,
}

# Plotly chart configuration with mode bar
CHART_CONFIG_WITH_MODEBAR = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
}
