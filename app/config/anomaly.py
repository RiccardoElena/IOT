"""
Anomaly detection configuration.

All parameters for Z-score detection, sliding windows, correlations, and thresholds.
"""

# =============================================================================
# Z-SCORE THRESHOLDS
# =============================================================================

# Z-score thresholds for anomaly classification
ZSCORE_WARNING_THRESHOLD = 2.0  # Values above this are "suspicious"
ZSCORE_ANOMALY_THRESHOLD = 3.0  # Values above this are "anomalies"

# =============================================================================
# PERCENTILE-BASED DETECTION
# =============================================================================

# Percentile thresholds for anomaly detection
PERCENTILE_LOW = 1    # Below this percentile = anomaly
PERCENTILE_HIGH = 99  # Above this percentile = anomaly

# =============================================================================
# PERCENTAGE CHANGE DETECTION
# =============================================================================

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
# SIMULATION PARAMETERS
# =============================================================================

# Real-time simulation speed
SIM_SPEED_MIN = 1
SIM_SPEED_MAX = 50
SIM_SPEED_DEFAULT = 10

# =============================================================================
# DISPLAY LIMITS
# =============================================================================

# Maximum number of anomalies to show in various contexts
MAX_ANOMALIES_IN_CONTEXT = 10
MAX_REALTIME_ANOMALIES_DISPLAY = 10
MAX_SYSTEMIC_EVENTS_DISPLAY = 5
