"""
Cross-Asset Analysis Module for IoT Financial Data Analytics.

This module implements cross-asset analysis techniques:
- Pearson correlation calculation
- Rolling correlation
- Simultaneous anomaly detection
- Price normalization for comparison

Used to identify relationships between different assets.
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
# CORRELATION CALCULATIONS
# =============================================================================

def calculate_correlation_matrix(price_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Pearson correlation matrix for all assets.
    
    Args:
        price_matrix: DataFrame with assets as columns, timestamps as index
    
    Returns:
        Correlation matrix (DataFrame)
    """
    return price_matrix.corr(method="pearson")


def calculate_returns_matrix(price_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentage returns for all assets.
    
    Args:
        price_matrix: DataFrame with assets as columns
    
    Returns:
        DataFrame with percentage returns
    """
    return price_matrix.pct_change() * 100


def calculate_rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = None
) -> pd.Series:
    """
    Calculate rolling correlation between two series.
    
    Args:
        series_a: First price/return series
        series_b: Second price/return series
        window: Rolling window size (default from config)
    
    Returns:
        Series with rolling correlation values
    """
    if window is None:
        window = config.CORRELATION_WINDOW
    
    return series_a.rolling(window=window).corr(series_b)


def calculate_all_rolling_correlations(
    price_matrix: pd.DataFrame,
    window: int = None
) -> Dict[Tuple[str, str], pd.Series]:
    """
    Calculate rolling correlations for all asset pairs.
    
    Args:
        price_matrix: DataFrame with assets as columns
        window: Rolling window size
    
    Returns:
        Dictionary mapping (asset_a, asset_b) to correlation series
    """
    if window is None:
        window = config.CORRELATION_WINDOW
    
    # Use returns for correlation (more stationary)
    returns = calculate_returns_matrix(price_matrix)
    
    correlations = {}
    assets = list(price_matrix.columns)
    
    for i, asset_a in enumerate(assets):
        for asset_b in assets[i+1:]:
            corr = calculate_rolling_correlation(
                returns[asset_a], 
                returns[asset_b], 
                window
            )
            correlations[(asset_a, asset_b)] = corr
    
    return correlations


# =============================================================================
# CORRELATION ANOMALY DETECTION
# =============================================================================

def detect_correlation_anomalies(
    rolling_corr: pd.Series,
    threshold_std: float = 2.0
) -> pd.Series:
    """
    Detect anomalies in rolling correlation.
    
    Flags points where correlation deviates significantly from historical mean.
    
    Args:
        rolling_corr: Series with rolling correlation values
        threshold_std: Number of std deviations for anomaly threshold
    
    Returns:
        Boolean Series (True = correlation anomaly)
    """
    mean_corr = rolling_corr.mean()
    std_corr = rolling_corr.std()
    
    upper_bound = mean_corr + threshold_std * std_corr
    lower_bound = mean_corr - threshold_std * std_corr
    
    return (rolling_corr > upper_bound) | (rolling_corr < lower_bound)


def get_correlation_statistics(rolling_corr: pd.Series) -> Dict[str, float]:
    """
    Get summary statistics for a rolling correlation series.
    
    Args:
        rolling_corr: Series with rolling correlation values
    
    Returns:
        Dictionary with statistics
    """
    return {
        "mean": float(rolling_corr.mean()),
        "std": float(rolling_corr.std()),
        "min": float(rolling_corr.min()),
        "max": float(rolling_corr.max()),
        "current": float(rolling_corr.iloc[-1]) if len(rolling_corr) > 0 else None
    }


# =============================================================================
# SIMULTANEOUS ANOMALIES
# =============================================================================

def count_simultaneous_anomalies(
    anomaly_flags: Dict[str, pd.Series]
) -> pd.Series:
    """
    Count how many assets have anomalies at each timestamp.
    
    Args:
        anomaly_flags: Dictionary mapping asset names to boolean anomaly series
    
    Returns:
        Series with count of simultaneous anomalies per timestamp
    """
    # Combine all anomaly flags into a DataFrame
    anomaly_df = pd.DataFrame(anomaly_flags)
    
    # Fill NaN with False (no anomaly if no data)
    anomaly_df = anomaly_df.infer_objects(copy = False)
    
    # Sum across columns (count True values)
    return anomaly_df.sum(axis=1)


def detect_systemic_events(
    anomaly_counts: pd.Series,
    threshold: int = None
) -> pd.Series:
    """
    Detect systemic events (multiple assets anomalous simultaneously).
    
    Args:
        anomaly_counts: Series with count of anomalies per timestamp
        threshold: Minimum anomalies for systemic event (default from config)
    
    Returns:
        Boolean Series (True = systemic event)
    """
    if threshold is None:
        threshold = config.SYSTEMIC_EVENT_THRESHOLD
    
    return anomaly_counts >= threshold


def get_systemic_event_details(
    anomaly_flags: Dict[str, pd.Series],
    systemic_mask: pd.Series
) -> pd.DataFrame:
    """
    Get details of systemic events (which assets were affected).
    
    Args:
        anomaly_flags: Dictionary mapping asset names to boolean anomaly series
        systemic_mask: Boolean series indicating systemic events
    
    Returns:
        DataFrame with systemic event details
    """
    events = []
    
    anomaly_df = pd.DataFrame(anomaly_flags)
    
    # Fill NaN with False to avoid masking errors
    anomaly_df = anomaly_df.infer_objects(copy=False)
    
    # Ensure boolean dtype
    anomaly_df = anomaly_df.astype(bool)
    
    for timestamp in anomaly_df.index[systemic_mask]:
        row = anomaly_df.loc[timestamp]
        # Now row is guaranteed to be boolean, safe to use as mask
        affected_assets = list(row[row].index)
        events.append({
            "timestamp": timestamp,
            "count": len(affected_assets),
            "assets": ", ".join(affected_assets)
        })
    
    if not events:
        return pd.DataFrame(columns=["timestamp", "count", "assets"])
    
    result = pd.DataFrame(events)
    result.index = range(1, len(result) + 1)
    return result


# =============================================================================
# PRICE NORMALIZATION
# =============================================================================

def normalize_prices(price_matrix: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """
    Normalize all asset prices to a common base value.
    
    Useful for visual comparison of assets with different price scales.
    
    Args:
        price_matrix: DataFrame with assets as columns
        base: Starting value for normalization (default 100)
    
    Returns:
        DataFrame with normalized prices
    """
    first_values = price_matrix.iloc[0]
    return (price_matrix / first_values) * base


# =============================================================================
# CORRELATION INTERPRETATION
# =============================================================================

def interpret_correlation(corr: float) -> str:
    """
    Provide human-readable interpretation of correlation value.
    
    Args:
        corr: Correlation coefficient (-1 to 1)
    
    Returns:
        Interpretation string
    """
    abs_corr = abs(corr)
    
    if abs_corr >= 0.7:
        strength = "Strong"
    elif abs_corr >= 0.4:
        strength = "Moderate"
    elif abs_corr >= 0.2:
        strength = "Weak"
    else:
        strength = "Very weak"
    
    direction = "positive" if corr >= 0 else "negative"
    
    return f"{strength} {direction} ({corr:.3f})"


def get_typical_correlations() -> Dict[Tuple[str, str], str]:
    """
    Get typical expected correlations between asset pairs.
    
    Returns:
        Dictionary with expected correlation descriptions
    """
    return {
        ("gold", "usd"): "Typically negative (gold is USD hedge)",
        ("oil", "usd"): "Typically negative (oil priced in USD)",
        ("sp500", "btc"): "Variable (risk-on correlation varies)",
        ("gold", "sp500"): "Low/negative (gold is safe haven)",
        ("oil", "sp500"): "Moderate positive (economic activity)",
    }


# =============================================================================
# ASSET PAIR ANALYSIS
# =============================================================================

def analyze_asset_pair(
    price_matrix: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    window: int = None
) -> Dict[str, any]:
    """
    Comprehensive analysis of a single asset pair.
    
    Args:
        price_matrix: DataFrame with assets as columns
        asset_a: First asset key
        asset_b: Second asset key
        window: Rolling correlation window
    
    Returns:
        Dictionary with analysis results
    """
    if window is None:
        window = config.CORRELATION_WINDOW
    
    # Get price series
    prices_a = price_matrix[asset_a]
    prices_b = price_matrix[asset_b]
    
    # Calculate returns
    returns_a = prices_a.pct_change() * 100
    returns_b = prices_b.pct_change() * 100
    
    # Static correlation
    static_corr = returns_a.corr(returns_b)
    
    # Rolling correlation
    rolling_corr = calculate_rolling_correlation(returns_a, returns_b, window)
    
    # Correlation statistics
    corr_stats = get_correlation_statistics(rolling_corr)
    
    # Correlation anomalies
    corr_anomalies = detect_correlation_anomalies(rolling_corr)
    
    return {
        "asset_a": asset_a,
        "asset_b": asset_b,
        "static_correlation": static_corr,
        "rolling_correlation": rolling_corr,
        "statistics": corr_stats,
        "anomaly_mask": corr_anomalies,
        "anomaly_count": int(corr_anomalies.sum()),
        "interpretation": interpret_correlation(static_corr),
        "returns_a": returns_a,
        "returns_b": returns_b
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_price_matrix_from_dict(
    data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Create price matrix from dictionary of DataFrames.
    
    Args:
        data: Dictionary mapping asset keys to DataFrames
    
    Returns:
        DataFrame with assets as columns, aligned by timestamp
    """
    close_col = config.COLUMNS["close"]
    
    price_series = {}
    for asset, df in data.items():
        price_series[asset] = df[close_col]
    
    # Combine and align by index
    matrix = pd.DataFrame(price_series)
    
    # Forward fill any gaps
    matrix = matrix.ffill()
    
    # Drop rows with any NaN
    matrix = matrix.dropna()
    
    return matrix


def get_asset_pairs() -> List[Tuple[str, str]]:
    """
    Get all unique asset pairs.
    
    Returns:
        List of (asset_a, asset_b) tuples
    """
    assets = list(config.ASSETS.keys())
    pairs = []
    
    for i, asset_a in enumerate(assets):
        for asset_b in assets[i+1:]:
            pairs.append((asset_a, asset_b))
    
    return pairs


def format_pair_name(asset_a: str, asset_b: str) -> str:
    """
    Format asset pair name for display.
    
    Args:
        asset_a: First asset key
        asset_b: Second asset key
    
    Returns:
        Formatted pair name
    """
    name_a = config.ASSETS.get(asset_a, asset_a)
    name_b = config.ASSETS.get(asset_b, asset_b)
    return f"{name_a} / {name_b}"
