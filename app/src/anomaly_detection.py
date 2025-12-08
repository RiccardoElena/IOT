"""
Anomaly Detection Module for IoT Financial Data Analytics.

This module implements anomaly detection techniques:
- Z-score (batch and rolling)
- Percentile-based detection
- Percentage change detection

All thresholds are configurable via config.py
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# =============================================================================
# Z-SCORE CALCULATIONS
# =============================================================================

def calculate_zscore_batch(series: pd.Series) -> pd.Series:
    """
    Calculate Z-score for entire series (batch mode).
    
    Uses global mean and standard deviation calculated over the 
    entire dataset. Suitable for daily/hourly data analysis.
    
    Formula: Z = (x - mean) / std
    
    Args:
        series: Pandas Series with numeric values
    
    Returns:
        Series with Z-score values
    """
    mean = series.mean()
    std = series.std()
    
    # Avoid division by zero
    if std == 0:
        return pd.Series(0, index=series.index)
    
    return (series - mean) / std


def calculate_zscore_rolling(series: pd.Series, window: int = None) -> pd.Series:
    """
    Calculate Z-score using rolling window (streaming mode).
    
    Uses mean and standard deviation calculated over a sliding window.
    Suitable for real-time IoT-style processing on minute data.
    
    Formula: Z = (x - rolling_mean) / rolling_std
    
    Args:
        series: Pandas Series with numeric values
        window: Window size in number of points (default from config)
    
    Returns:
        Series with rolling Z-score values
    """
    if window is None:
        window = config.WINDOW_SIZE_MINUTE
    
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    zscore = (series - rolling_mean) / rolling_std
    
    # Fill NaN with 0 (first points where we don't have enough data)
    return zscore.fillna(0)


# =============================================================================
# PERCENTILE-BASED DETECTION
# =============================================================================

def calculate_percentile_bounds(
    series: pd.Series, 
    low_pct: float = None, 
    high_pct: float = None
) -> Tuple[float, float]:
    """
    Calculate percentile bounds for anomaly detection.
    
    Args:
        series: Pandas Series with numeric values
        low_pct: Lower percentile threshold (default from config)
        high_pct: Upper percentile threshold (default from config)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if low_pct is None:
        low_pct = config.PERCENTILE_LOW
    if high_pct is None:
        high_pct = config.PERCENTILE_HIGH
    
    lower_bound = np.percentile(series.dropna(), low_pct)
    upper_bound = np.percentile(series.dropna(), high_pct)
    
    return lower_bound, upper_bound


def detect_percentile_anomalies(
    series: pd.Series,
    low_pct: float = None,
    high_pct: float = None
) -> pd.Series:
    """
    Detect anomalies based on percentile thresholds.
    
    Values below low_pct or above high_pct are flagged as anomalies.
    
    Args:
        series: Pandas Series with numeric values
        low_pct: Lower percentile threshold
        high_pct: Upper percentile threshold
    
    Returns:
        Boolean Series (True = anomaly)
    """
    lower_bound, upper_bound = calculate_percentile_bounds(series, low_pct, high_pct)
    
    return (series < lower_bound) | (series > upper_bound)


# =============================================================================
# PERCENTAGE CHANGE DETECTION
# =============================================================================

def calculate_percentage_change(series: pd.Series) -> pd.Series:
    """
    Calculate period-over-period percentage change.
    
    Formula: pct_change = (current - previous) / previous * 100
    
    Args:
        series: Pandas Series with numeric values
    
    Returns:
        Series with percentage change values
    """
    return series.pct_change() * 100


def detect_pct_change_anomalies(
    series: pd.Series, 
    threshold: float = None,
    granularity: str = "daily"
) -> pd.Series:
    """
    Detect anomalies based on percentage change threshold.
    
    Args:
        series: Pandas Series with numeric values
        threshold: Percentage threshold (absolute value)
        granularity: Data granularity to determine default threshold
    
    Returns:
        Boolean Series (True = anomaly)
    """
    if threshold is None:
        if granularity == "minute":
            threshold = config.PCT_CHANGE_THRESHOLD_MINUTE
        else:
            threshold = config.PCT_CHANGE_THRESHOLD_DAILY
    
    pct_change = calculate_percentage_change(series)
    
    return pct_change.abs() > threshold


# =============================================================================
# VOLATILITY CALCULATION
# =============================================================================

def calculate_volatility(df: pd.DataFrame) -> pd.Series:
    """
    Calculate intraperiod volatility (high - low).
    
    Args:
        df: DataFrame with 'high' and 'low' columns
    
    Returns:
        Series with volatility values
    """
    high_col = config.COLUMNS["high"]
    low_col = config.COLUMNS["low"]
    
    return df[high_col] - df[low_col]


# =============================================================================
# ANOMALY CLASSIFICATION
# =============================================================================

def classify_zscore(zscore: float) -> str:
    """
    Classify a Z-score value into categories.
    
    Args:
        zscore: Z-score value
    
    Returns:
        Classification string: 'normal', 'warning', or 'anomaly'
    """
    abs_z = abs(zscore)
    
    if abs_z >= config.ZSCORE_ANOMALY_THRESHOLD:
        return "anomaly"
    elif abs_z >= config.ZSCORE_WARNING_THRESHOLD:
        return "warning"
    else:
        return "normal"


def classify_zscore_series(zscore_series: pd.Series) -> pd.Series:
    """
    Classify all Z-scores in a series.
    
    Args:
        zscore_series: Series with Z-score values
    
    Returns:
        Series with classification strings
    """
    return zscore_series.apply(classify_zscore)


# =============================================================================
# MAIN ANOMALY DETECTION FUNCTION
# =============================================================================

def detect_anomalies(
    df: pd.DataFrame, 
    zscore_threshold: float = None,
    mode: str = "batch"
) -> pd.DataFrame:
    """
    Main function to detect all anomalies in a DataFrame.
    
    Adds the following columns to the DataFrame:
    - zscore_close: Z-score of closing price
    - zscore_volume: Z-score of volume
    - zscore_volatility: Z-score of volatility (high-low)
    - pct_change: Percentage change in close price
    - anomaly_price: Boolean, True if price is anomaly
    - anomaly_volume: Boolean, True if volume is anomaly
    - anomaly_volatility: Boolean, True if volatility is anomaly
    - anomaly_any: Boolean, True if any anomaly detected
    
    Args:
        df: DataFrame with OHLCV data
        zscore_threshold: Z-score threshold for anomalies (default from config)
        mode: 'batch' for global stats, 'rolling' for sliding window
    
    Returns:
        DataFrame with added anomaly columns
    """
    if zscore_threshold is None:
        zscore_threshold = config.ZSCORE_ANOMALY_THRESHOLD
    
    # Create a copy to avoid modifying original
    result = df.copy()
    
    # Get column names from config
    close_col = config.COLUMNS["close"]
    volume_col = config.COLUMNS["volume"]
    
    # Calculate volatility
    result["volatility"] = calculate_volatility(result)
    
    # Calculate percentage change BEFORE any filtering
    result["pct_change"] = calculate_percentage_change(result[close_col])
    
    # Calculate Z-scores based on mode
    if mode == "rolling":
        result["zscore_close"] = calculate_zscore_rolling(result[close_col])
        result["zscore_volume"] = calculate_zscore_rolling(result[volume_col])
        result["zscore_volatility"] = calculate_zscore_rolling(result["volatility"])
    else:  # batch mode
        result["zscore_close"] = calculate_zscore_batch(result[close_col])
        result["zscore_volume"] = calculate_zscore_batch(result[volume_col])
        result["zscore_volatility"] = calculate_zscore_batch(result["volatility"])
    
    # Detect anomalies based on Z-score threshold
    result["anomaly_price"] = result["zscore_close"].abs() >= zscore_threshold
    result["anomaly_volume"] = result["zscore_volume"].abs() >= zscore_threshold
    result["anomaly_volatility"] = result["zscore_volatility"].abs() >= zscore_threshold
    
    # Combined anomaly flag
    result["anomaly_any"] = (
        result["anomaly_price"] | 
        result["anomaly_volume"] | 
        result["anomaly_volatility"]
    )
    
    # Add classification columns
    result["class_price"] = classify_zscore_series(result["zscore_close"])
    result["class_volume"] = classify_zscore_series(result["zscore_volume"])
    result["class_volatility"] = classify_zscore_series(result["zscore_volatility"])
    
    return result


# =============================================================================
# ANOMALY EXTRACTION
# =============================================================================

def get_anomaly_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a summary table of all anomalies.
    
    Creates a human-readable table with one row per anomaly event.
    Includes percentage change for price anomalies.
    
    Args:
        df: DataFrame with anomaly columns (output of detect_anomalies)
    
    Returns:
        DataFrame with anomaly summary (1-based index)
    """
    anomalies = []
    
    close_col = config.COLUMNS["close"]
    volume_col = config.COLUMNS["volume"]
    
    for idx, row in df.iterrows():
        # Check price anomaly - include pct_change
        if row.get("anomaly_price", False):
            pct_val = row.get("pct_change", None)
            anomalies.append({
                "timestamp": idx,
                "type": "Price",
                "value": row[close_col],
                "zscore": row["zscore_close"],
                "pct_change": pct_val if pd.notna(pct_val) else None
            })
        
        # Check volume anomaly
        if row.get("anomaly_volume", False):
            anomalies.append({
                "timestamp": idx,
                "type": "Volume",
                "value": row[volume_col],
                "zscore": row["zscore_volume"],
                "pct_change": None
            })
        
        # Check volatility anomaly
        if row.get("anomaly_volatility", False):
            anomalies.append({
                "timestamp": idx,
                "type": "Volatility",
                "value": row.get("volatility", None),
                "zscore": row["zscore_volatility"],
                "pct_change": None
            })
    
    if not anomalies:
        return pd.DataFrame(columns=["timestamp", "type", "value", "zscore", "pct_change"])
    
    result = pd.DataFrame(anomalies)
    # Set 1-based index
    result.index = range(1, len(result) + 1)
    
    return result


def count_anomalies(df: pd.DataFrame) -> Dict[str, int]:
    """
    Count anomalies by type.
    
    Args:
        df: DataFrame with anomaly columns
    
    Returns:
        Dictionary with anomaly counts
    """
    return {
        "price": int(df["anomaly_price"].sum()) if "anomaly_price" in df.columns else 0,
        "volume": int(df["anomaly_volume"].sum()) if "anomaly_volume" in df.columns else 0,
        "volatility": int(df["anomaly_volatility"].sum()) if "anomaly_volatility" in df.columns else 0,
        "total": int(df["anomaly_any"].sum()) if "anomaly_any" in df.columns else 0
    }


# =============================================================================
# STATISTICS FUNCTIONS
# =============================================================================

def get_zscore_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Get summary statistics for Z-scores.
    
    Args:
        df: DataFrame with Z-score columns
    
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    for col in ["zscore_close", "zscore_volume", "zscore_volatility"]:
        if col in df.columns:
            stats[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std())
            }
    
    return stats


def get_threshold_lines(threshold: float = None) -> Dict[str, float]:
    """
    Get threshold values for plotting.
    
    Args:
        threshold: Z-score threshold (default from config)
    
    Returns:
        Dictionary with threshold values
    """
    if threshold is None:
        threshold = config.ZSCORE_ANOMALY_THRESHOLD
    
    warning = config.ZSCORE_WARNING_THRESHOLD
    
    return {
        "anomaly_upper": threshold,
        "anomaly_lower": -threshold,
        "warning_upper": warning,
        "warning_lower": -warning
    }
