"""
Pattern recognition configuration.

Parameters for candlestick and chart pattern detection.
"""

# =============================================================================
# CANDLESTICK PATTERN PARAMETERS (from settings.py)
# =============================================================================

# Doji: body must be less than this fraction of total range
DOJI_BODY_RATIO = 0.1

# Hammer: lower shadow must be at least this multiple of body
HAMMER_SHADOW_RATIO = 2.0

# =============================================================================
# PEAK/TROUGH DETECTION (from constants.py)
# =============================================================================

# Peak/trough detection parameters
PEAK_DISTANCE = 5
PEAK_PROMINENCE_PCT = 0.01
SMOOTHING_WINDOW = 5

# =============================================================================
# CANDLESTICK PATTERN DEFAULTS (from constants.py)
# =============================================================================

# Doji pattern
DOJI_BODY_RATIO_DEFAULT = 0.1

# Hammer pattern
HAMMER_BODY_RATIO_DEFAULT = 0.3
HAMMER_SHADOW_RATIO_DEFAULT = 2.0

# =============================================================================
# CHART PATTERN PARAMETERS (from constants.py)
# =============================================================================

# Chart pattern lookback
CHART_PATTERN_LOOKBACK_DEFAULT = 50
CHART_PATTERN_LOOKBACK_MIN = 20
CHART_PATTERN_LOOKBACK_MAX = 100
CUP_RIM_TOLERANCE = 0.15
HANDLE_PULLBACK_RATIO = 0.5
SMOOTH_WINDOW = 5
PEAK_PROMINENCE = 0.01

# =============================================================================
# UI SLIDER RANGES (from constants.py)
# =============================================================================

# Tolerance defaults for UI sliders
PRICE_TOLERANCE_MIN = 2.0
PRICE_TOLERANCE_MAX = 15.0
PRICE_TOLERANCE_DEFAULT = 2.0
PRICE_TOLERANCE_STEP = 1.0

PROMINENCE_MIN = 0.5
PROMINENCE_MAX = 5.0
PROMINENCE_DEFAULT = 5.0
PROMINENCE_STEP = 0.5

# =============================================================================
# DISPLAY LIMITS
# =============================================================================

# Maximum number of patterns to show in context
MAX_PATTERNS_IN_CONTEXT = 10

