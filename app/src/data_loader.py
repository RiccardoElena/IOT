"""
Data Loader Module for IoT Financial Data Analytics.

This module handles all data loading and preprocessing operations.
It reads CSV files, cleans the data, and provides ready-to-use DataFrames.
Supports lazy loading for large datasets (minute-level data).
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
# CORE LOADING FUNCTIONS
# =============================================================================

def get_file_path(asset: str, granularity: str) -> str:
    """
    Build the full file path for a given asset and granularity.
    
    Args:
        asset: Asset key (e.g., 'sp500', 'gold')
        granularity: Time granularity ('minute', 'hourly', 'daily')
    
    Returns:
        Full path to the CSV file
    
    Raises:
        ValueError: If asset or granularity is not recognized
    """
    if asset not in config.FILE_NAMES:
        raise ValueError(
            f"Unknown asset: {asset}. Valid options: {list(config.FILE_NAMES.keys())}"
        )
    
    if granularity not in config.GRANULARITY_PATHS:
        raise ValueError(
            f"Unknown granularity: {granularity}. "
            f"Valid options: {list(config.GRANULARITY_PATHS.keys())}"
        )
    
    return os.path.join(
        config.DATA_BASE_PATH,
        config.GRANULARITY_PATHS[granularity],
        config.FILE_NAMES[asset]
    )


def load_single_asset(asset: str, granularity: str) -> pd.DataFrame:
    """
    Load data for a single asset at a specific granularity.
    
    Performs the following operations:
    1. Reads CSV file
    2. Parses timestamp column as datetime
    3. Sets timestamp as index
    4. Sorts by timestamp
    5. Handles missing values
    
    Args:
        asset: Asset key (e.g., 'sp500', 'gold')
        granularity: Time granularity ('minute', 'hourly', 'daily')
    
    Returns:
        DataFrame with cleaned data, indexed by timestamp
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
    """
    file_path = get_file_path(asset, granularity)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Read CSV with proper parsing
    df = pd.read_csv(
        file_path,
        parse_dates=[config.COLUMNS["timestamp"]],
        index_col=config.COLUMNS["timestamp"]
    )
    
    # Sort by timestamp (ascending)
    df = df.sort_index()
    
    # Handle any missing values by forward-filling
    # This is appropriate for financial data where we carry last known value
    df = df.ffill()
    
    # Add asset identifier column (useful when combining multiple assets)
    df["asset"] = asset
    
    return df


def load_all_assets(granularity: str) -> Dict[str, pd.DataFrame]:
    """
    Load data for all configured assets at a specific granularity.
    
    Args:
        granularity: Time granularity ('minute', 'hourly', 'daily')
    
    Returns:
        Dictionary mapping asset keys to their DataFrames
    """
    data = {}
    
    for asset in config.ASSETS.keys():
        try:
            data[asset] = load_single_asset(asset, granularity)
        except FileNotFoundError as e:
            print(f"Warning: Could not load {asset}: {e}")
            continue
    
    return data


def load_all_assets_combined(granularity: str) -> pd.DataFrame:
    """
    Load all assets and combine into a single DataFrame.
    
    Creates a DataFrame with MultiIndex (timestamp, asset) or 
    with asset as a column, depending on use case.
    
    Args:
        granularity: Time granularity ('minute', 'hourly', 'daily')
    
    Returns:
        Combined DataFrame with all assets
    """
    all_data = load_all_assets(granularity)
    
    if not all_data:
        raise ValueError("No data loaded. Check that CSV files exist.")
    
    # Combine all DataFrames
    combined = pd.concat(all_data.values(), axis=0)
    
    return combined


# =============================================================================
# LAZY LOADING FUNCTIONS (for large datasets)
# =============================================================================

def get_date_range_fast(asset: str, granularity: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the date range of a dataset WITHOUT loading all data.
    
    Reads only the timestamp column to determine available date range.
    Much faster than loading the entire file for minute-level data.
    
    Args:
        asset: Asset key (e.g., 'sp500', 'gold')
        granularity: Time granularity ('minute', 'hourly', 'daily')
    
    Returns:
        Tuple of (min_date, max_date) as pandas Timestamps
    """
    file_path = get_file_path(asset, granularity)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Read only the timestamp column
    df_dates = pd.read_csv(
        file_path,
        usecols=[config.COLUMNS["timestamp"]],
        parse_dates=[config.COLUMNS["timestamp"]]
    )
    
    timestamps = df_dates[config.COLUMNS["timestamp"]]
    
    return timestamps.min(), timestamps.max()


def load_asset_date_range(
    asset: str, 
    granularity: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Load data for a specific date range only (lazy loading).
    
    More memory efficient for large datasets like minute-level data.
    
    Args:
        asset: Asset key (e.g., 'sp500', 'gold')
        granularity: Time granularity ('minute', 'hourly', 'daily')
        start_date: Start date (inclusive), format 'YYYY-MM-DD'
        end_date: End date (inclusive), format 'YYYY-MM-DD'
    
    Returns:
        DataFrame with data for the specified date range
    """
    file_path = get_file_path(asset, granularity)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Read full CSV (we need to filter after reading for CSV format)
    df = pd.read_csv(
        file_path,
        parse_dates=[config.COLUMNS["timestamp"]],
        index_col=config.COLUMNS["timestamp"]
    )
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Filter to date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    
    # Handle missing values
    df = df.ffill()
    
    # Add asset identifier
    df["asset"] = asset
    
    return df


def get_row_count_fast(asset: str, granularity: str) -> int:
    """
    Get the number of rows in a dataset WITHOUT loading all data.
    
    Args:
        asset: Asset key
        granularity: Time granularity
    
    Returns:
        Number of rows in the dataset
    """
    file_path = get_file_path(asset, granularity)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Count lines (subtract 1 for header)
    with open(file_path, 'r') as f:
        return sum(1 for _ in f) - 1


# =============================================================================
# DATA FILTERING FUNCTIONS
# =============================================================================

def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter DataFrame by date range (inclusive on both ends).
    
    Args:
        df: DataFrame with datetime index
        start_date: Start date (inclusive), format 'YYYY-MM-DD' or None
        end_date: End date (inclusive), format 'YYYY-MM-DD' or None
    
    Returns:
        Filtered DataFrame
    """
    result = df.copy()
    
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        result = result[result.index >= start_dt]
    
    if end_date is not None:
        # Include the entire end date (up to 23:59:59)
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        result = result[result.index <= end_dt]
    
    return result


def filter_by_single_day(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Filter minute-level data to a single day.
    
    Args:
        df: DataFrame with datetime index (minute granularity)
        date: Date to filter, format 'YYYY-MM-DD'
    
    Returns:
        DataFrame containing only data from the specified day
    """
    target_date = pd.to_datetime(date).date()
    mask = df.index.date == target_date
    return df[mask].copy()


# =============================================================================
# DATA TRANSFORMATION FUNCTIONS
# =============================================================================

def calculate_returns(df: pd.DataFrame, column: str = None) -> pd.Series:
    """
    Calculate percentage returns (period-over-period change).
    
    Args:
        df: DataFrame with price data
        column: Column to use (defaults to 'close')
    
    Returns:
        Series with percentage returns
    """
    if column is None:
        column = config.COLUMNS["close"]
    
    return df[column].pct_change() * 100


def calculate_volatility(df: pd.DataFrame) -> pd.Series:
    """
    Calculate intraperiod volatility (high - low).
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series with volatility values
    """
    high_col = config.COLUMNS["high"]
    low_col = config.COLUMNS["low"]
    
    return df[high_col] - df[low_col]


def normalize_prices(df: pd.DataFrame, base_value: float = 100.0) -> pd.DataFrame:
    """
    Normalize prices to a base value (useful for comparing assets).
    
    Sets the first value to base_value and scales all subsequent values
    proportionally.
    
    Args:
        df: DataFrame with price data
        base_value: Starting value for normalization (default 100)
    
    Returns:
        DataFrame with normalized prices
    """
    close_col = config.COLUMNS["close"]
    
    result = df.copy()
    first_value = result[close_col].iloc[0]
    result[f"{close_col}_normalized"] = (result[close_col] / first_value) * base_value
    
    return result


def create_price_matrix(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a matrix of closing prices for all assets.
    
    Args:
        data: Dictionary mapping asset keys to DataFrames
    
    Returns:
        DataFrame with timestamps as index and assets as columns
    """
    close_col = config.COLUMNS["close"]
    
    price_series = {}
    for asset, df in data.items():
        price_series[asset] = df[close_col]
    
    return pd.DataFrame(price_series)


# =============================================================================
# DATA VALIDATION FUNCTIONS
# =============================================================================

def validate_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate loaded data and return summary statistics.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results and statistics
    """
    close_col = config.COLUMNS["close"]
    volume_col = config.COLUMNS["volume"]
    
    return {
        "rows": len(df),
        "start_date": df.index.min(),
        "end_date": df.index.max(),
        "missing_values": df.isnull().sum().to_dict(),
        "price_range": (df[close_col].min(), df[close_col].max()),
        "volume_range": (df[volume_col].min(), df[volume_col].max()),
    }


def get_available_dates(df: pd.DataFrame) -> List[str]:
    """
    Get list of unique dates available in the DataFrame.
    
    Useful for populating date selectors in the UI.
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        List of date strings in 'YYYY-MM-DD' format
    """
    unique_dates = df.index.date
    return sorted(list(set([str(d) for d in unique_dates])))


# =============================================================================
# CACHING WRAPPER (for Streamlit)
# =============================================================================

def get_cached_data(asset: str, granularity: str) -> pd.DataFrame:
    """
    Load data with caching support.
    
    This function is designed to be wrapped with @st.cache_data
    in the Streamlit app for performance optimization.
    
    Args:
        asset: Asset key
        granularity: Time granularity
    
    Returns:
        DataFrame with loaded data
    """
    return load_single_asset(asset, granularity)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_asset_display_name(asset: str) -> str:
    """
    Get the display name for an asset.
    
    Args:
        asset: Asset key (e.g., 'sp500')
    
    Returns:
        Display name (e.g., 'S&P 500')
    """
    return config.ASSETS.get(asset, asset)


def get_granularity_display_name(granularity: str) -> str:
    """
    Get the display name for a granularity.
    
    Args:
        granularity: Granularity key (e.g., 'daily')
    
    Returns:
        Display name (e.g., 'Daily')
    """
    return config.GRANULARITY_DISPLAY.get(granularity, granularity)


def list_available_assets() -> List[str]:
    """
    Get list of available asset keys.
    
    Returns:
        List of asset keys
    """
    return list(config.ASSETS.keys())


def list_available_granularities() -> List[str]:
    """
    Get list of available granularity keys.
    
    Returns:
        List of granularity keys
    """
    return list(config.GRANULARITY_PATHS.keys())
