"""
Cross-Asset Analysis Module for IoT Financial Data Analytics.

This module implements cross-asset analysis techniques:
- Pearson correlation calculation
- Rolling correlation
- Simultaneous anomaly detection
- Price normalization for comparison

Used to identify relationships between different assets.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from config.anomaly import CORRELATION_WINDOW
from config.data import COLUMNS
from config.assets import ASSETS
from utils.dates import filter_by_date_range
from .anomaly_detection import detect_anomalies

# =============================================================================
# CORRELATION CALCULATIONS
# =============================================================================


def calculate_rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: Optional[int] = None
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
        window = CORRELATION_WINDOW
    
    return series_a.rolling(window=window).corr(series_b)


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


def get_correlation_statistics(rolling_corr: pd.Series) -> Dict[str, float | None]:
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
    window: Optional[int] = None
) -> Dict[str, Any]:
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
        window = CORRELATION_WINDOW
    
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
    close_col = COLUMNS["close"]
    
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
    assets = list(ASSETS.keys())
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
    name_a = ASSETS.get(asset_a, asset_a)
    name_b = ASSETS.get(asset_b, asset_b)
    return f"{name_a} / {name_b}"

def get_anomaly_details_by_date(anomaly_flags: dict) -> pd.DataFrame:
    """
    Create a DataFrame with anomaly details for each date.
    Ensures consistent handling of NaN values.
    
    Returns DataFrame with columns: date, count, assets_list, assets_str
    """
    # Create DataFrame from flags
    anomaly_df = pd.DataFrame(anomaly_flags)
    
    # CRITICAL: Fill NaN with False BEFORE any operations
    anomaly_df = anomaly_df.fillna(False).astype(bool)
    
    results = []
    for timestamp in anomaly_df.index:
        row = anomaly_df.loc[timestamp]
        # Get list of assets with True
        affected_assets = list(row[row].index)
        
        if len(affected_assets) > 0:
            results.append({
                "timestamp": timestamp,
                "count": len(affected_assets),
                "assets_list": affected_assets,
                "assets_str": ", ".join([str(ASSETS.get(a, a)) for a in affected_assets])
            })
    
    if not results:
        return pd.DataFrame(columns=["timestamp", "count", "assets_list", "assets_str"])
    
    return pd.DataFrame(results).set_index("timestamp")

def count_simultaneous_anomalies(anomaly_flags: dict) -> pd.Series:
    """
    Count simultaneous anomalies with consistent NaN handling.
    """
    anomaly_df = pd.DataFrame(anomaly_flags)
    # CRITICAL: Same fillna as get_anomaly_details_by_date
    anomaly_df = anomaly_df.fillna(False).astype(bool)
    return anomaly_df.sum(axis=1)

def process_cross_asset_data(
    data_dict: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    zscore_threshold: float
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, pd.Series]]:
    """Process all assets for cross-asset analysis."""
    
    # Filter each asset by date range
    filtered_data = {}
    anomaly_flags = {}
    
    for asset, df in data_dict.items():
        df_filtered = filter_by_date_range(df, start_date, end_date)
        df_processed = detect_anomalies(df_filtered, zscore_threshold=zscore_threshold)
        filtered_data[asset] = df_processed
        anomaly_flags[asset] = df_processed["anomaly_any"]
    
    # Create price matrix
    price_matrix = create_price_matrix_from_dict(filtered_data)
    
    return filtered_data, price_matrix, anomaly_flags