import pandas as pd
from typing import Optional
from config.data import COLUMNS

# =============================================================================
# DATA TRANSFORMATION FUNCTIONS
# =============================================================================

def calculate_returns(df: pd.DataFrame, column: Optional[str] = None) -> pd.Series:
    """
    Calculate percentage returns (period-over-period change).
    
    Args:
        df: DataFrame with price data
        column: Column to use (defaults to 'close')
    
    Returns:
        Series with percentage returns
    """
    if column is None:
        column = COLUMNS["close"]
    
    return df[column].pct_change() * 100


def calculate_volatility(df: pd.DataFrame) -> pd.Series:
    """
    Calculate intraperiod volatility (high - low).
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series with volatility values
    """
    high_col = COLUMNS["high"]
    low_col = COLUMNS["low"]
    
    return df[high_col] - df[low_col]



