"""
Anomaly Detection Module for IoT Financial Data Analytics.

This module implements anomaly detection techniques:
- Z-score (batch and rolling)
- Percentile-based detection
- Percentage change detection

All thresholds are configurable via config.py
"""

from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd

# Import only what we need from config
from config.anomaly import (
    ZSCORE_WARNING_THRESHOLD,
    ZSCORE_ANOMALY_THRESHOLD,
    WINDOW_SIZE_MINUTE,
)
from config.data import COLUMNS


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


def calculate_zscore_rolling(series: pd.Series, window: Optional[int] = None) -> pd.Series:
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
        window = WINDOW_SIZE_MINUTE
    
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    zscore = (series - rolling_mean) / rolling_std
    
    # Fill NaN with 0 (first points where we don't have enough data)
    return zscore.fillna(0)

# =============================================================================
# VOLATILITY CALCULATION
# =============================================================================

def calculate_volatility(df: pd.DataFrame) -> pd.Series:

    high_col = COLUMNS["high"]
    low_col = COLUMNS["low"]
    
    return df[high_col] - df[low_col]

# =============================================================================
# ANOMALY CLASSIFICATION
# =============================================================================

def classify_zscore(zscore: float) -> str:
    abs_z = abs(zscore)
    
    if abs_z >= ZSCORE_ANOMALY_THRESHOLD:
        return "anomaly"
    elif abs_z >= ZSCORE_WARNING_THRESHOLD:
        return "warning"
    else:
        return "normal"


def classify_zscore_series(zscore_series: pd.Series) -> pd.Series:
    return zscore_series.apply(classify_zscore)

# =============================================================================
# MAIN ANOMALY DETECTION FUNCTION
# =============================================================================

def detect_anomalies(
    df: pd.DataFrame, 
    zscore_threshold: Optional[float] = None,
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
        zscore_threshold = ZSCORE_ANOMALY_THRESHOLD
    
    # Create a copy to avoid modifying original
    result = df.copy()
    
    # Get column names
    close_col = COLUMNS["close"]
    volume_col = COLUMNS["volume"]
    
    # Calculate volatility
    result["volatility"] = calculate_volatility(result)
    
    # Calculate percentage change BEFORE any filtering
    result["pct_change"] = result[close_col].pct_change()*100
    result["pct_change_volume"] = result[volume_col].pct_change()*100
    result["pct_change_volatility"] = result["volatility"].pct_change()*100
    
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
    anomalies = []
    
    close_col = COLUMNS["close"]
    volume_col = COLUMNS["volume"]
    
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
            pct_val = row.get("pct_change_volume", None)
            anomalies.append({
                "timestamp": idx,
                "type": "Volume",
                "value": row[volume_col],
                "zscore": row["zscore_volume"],
                "pct_change": pct_val if pd.notna(pct_val) else None
            })
        
        # Check volatility anomaly
        if row.get("anomaly_volatility", False):
            pct_val = row.get("pct_change_volatility", None)
            anomalies.append({
                "timestamp": idx,
                "type": "Volatility",
                "value": row.get("volatility", None),
                "zscore": row["zscore_volatility"],
                "pct_change": pct_val if pd.notna(pct_val) else None
            })
    
    if not anomalies:
        return pd.DataFrame(columns=["timestamp", "type", "value", "zscore", "pct_change"])
    
    result = pd.DataFrame(anomalies)
    # Set 1-based index
    result.index = list(range(1, len(result) + 1))
    
    return result


def count_anomalies(df: pd.DataFrame) -> Dict[str, int]:
    return {
        "price": int(df["anomaly_price"].sum()) if "anomaly_price" in df.columns else 0,
        "volume": int(df["anomaly_volume"].sum()) if "anomaly_volume" in df.columns else 0,
        "volatility": int(df["anomaly_volatility"].sum()) if "anomaly_volatility" in df.columns else 0,
        "total": int(df["anomaly_any"].sum()) if "anomaly_any" in df.columns else 0
    }


# =============================================================================
# STATISTICS FUNCTIONS
# =============================================================================

def get_threshold_lines(threshold: Optional[float] = None) -> Dict[str, float]:
    if threshold is None:
        threshold = ZSCORE_ANOMALY_THRESHOLD
    
    warning = ZSCORE_WARNING_THRESHOLD
    
    return {
        "anomaly_upper": threshold,
        "anomaly_lower": -threshold,
        "warning_upper": warning,
        "warning_lower": -warning
    }


def get_severity(zscore: float, threshold: float) -> str:
    abs_z = abs(zscore)
    if abs_z >= threshold + 1.0:
        return "🔴 HIGH"
    elif abs_z >= threshold + 0.5:
        return "🟠 MEDIUM"
    else:
        return "🟡 LOW"

def calculate_anomalies_batch(
    prices: np.ndarray,
    window: int,
    threshold: float,
    timestamps: pd.DatetimeIndex
) -> List[Dict[str, Any]]:
    """Calculate all anomalies in batch mode (for Run All)."""
    anomalies = []
    for i in range(window, len(prices)):
        window_data = np.asarray(prices[i - window:i])
        current_price = prices[i]
        mean = window_data.mean()
        std = window_data.std()
        if std > 0:
            zscore = (current_price - mean) / std
            if abs(zscore) >= threshold:
                anomalies.append({
                    "idx": i,
                    "timestamp": timestamps[i],
                    "price": current_price,
                    "zscore": zscore
                })
    return anomalies


def process_batch(
    start_idx: int,
    batch_size: int,
    prices: np.ndarray,
    window: int,
    threshold: float,
    timestamps: pd.DatetimeIndex,
    existing_anomalies: List[Dict[str, Any]]
) -> Tuple[int, List[Dict[str, Any]]]:
    """Process a batch of points and return new anomalies."""
    new_anomalies = []
    end_idx = min(start_idx + batch_size, len(prices))
    
    for i in range(start_idx, end_idx):
        if i >= window:
            window_data = np.asarray(prices[i - window:i])
            current_price = prices[i]
            mean = window_data.mean()
            std = window_data.std()
            
            if std > 0:
                zscore = (current_price - mean) / std
                if abs(zscore) >= threshold:
                    new_anomalies.append({
                        "idx": i,
                        "timestamp": timestamps[i],
                        "price": current_price,
                        "zscore": zscore
                    })
    
    return end_idx, existing_anomalies + new_anomalies

def compute_anomaly_log(anomalies: List[Dict[str, Any]], current_idx: int, zscore_threshold: float) -> Optional[pd.DataFrame]:
    """Render the anomaly log as a dataframe."""
    visible_anomalies = [a for a in anomalies if a["idx"] < current_idx]
    if visible_anomalies:
        log_data = []
        for a in visible_anomalies:
            log_data.append({
                "Time": str(a["timestamp"])[11:19],
                "Price": f"${a['price']:.2f}",
                "Z-Score": f"{a['zscore']:.2f}σ",
                "Severity": get_severity(a["zscore"], zscore_threshold)
            })
        return pd.DataFrame(log_data)
    return None