"""
Data Loader Module for IoT Financial Data Analytics.

This module handles all data loading and preprocessing operations.
It reads CSV files, cleans the data, and provides ready-to-use DataFrames.
Supports lazy loading for large datasets (minute-level data).
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Import logger
from utils.logger import logger

# Import only what we need from config
from config.data import (
    DATA_BASE_PATH,
    GRANULARITY_PATHS,
    COLUMNS,
)
from config.assets import ASSETS, FILE_NAMES


# =============================================================================
# CORE LOADING FUNCTIONS
# =============================================================================

def _get_file_path(asset: str, granularity: str) -> str:
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
    if asset not in FILE_NAMES:
        raise ValueError(
            f"Unknown asset: {asset}. Valid options: {list(FILE_NAMES.keys())}"
        )
    
    if granularity not in GRANULARITY_PATHS:
        raise ValueError(
            f"Unknown granularity: {granularity}. "
            f"Valid options: {list(GRANULARITY_PATHS.keys())}"
        )
    
    return os.path.join(
        DATA_BASE_PATH,
        GRANULARITY_PATHS[granularity],
        FILE_NAMES[asset]
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
    file_path = _get_file_path(asset, granularity)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Read CSV with proper parsing
    df = pd.read_csv(
        file_path,
        parse_dates=[COLUMNS["timestamp"]],
        index_col=COLUMNS["timestamp"]
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
    
    for asset in ASSETS.keys():
        try:
            data[asset] = load_single_asset(asset, granularity)
        except FileNotFoundError as e:
            logger.warning(f"Could not load {asset}: {e}")
            continue
    
    return data

def get_date_range(asset: str, granularity: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
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
    file_path = _get_file_path(asset, granularity)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Read only the timestamp column
    df_dates = pd.read_csv(
        file_path,
        usecols=[COLUMNS["timestamp"]],
        parse_dates=[COLUMNS["timestamp"]]
    )
    
    timestamps = df_dates[COLUMNS["timestamp"]]
    
    return timestamps.min(), timestamps.max()