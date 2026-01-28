import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from config import COLUMNS, SMOOTHING_WINDOW, PEAK_DISTANCE, PEAK_PROMINENCE_PCT, CUP_RIM_TOLERANCE, HANDLE_PULLBACK_RATIO

def get_candle_body(row: pd.Series) -> float:
    """
    Calculate the body size of a candle.
    
    Args:
        row: DataFrame row with OHLC data
    
    Returns:
        Body size (absolute value)
    """
    return abs(row[COLUMNS["close"]] - row[COLUMNS["open"]])


def get_candle_range(row: pd.Series) -> float:
    """
    Calculate the full range of a candle (high - low).
    
    Args:
        row: DataFrame row with OHLC data
    
    Returns:
        Full range
    """
    return row[COLUMNS["high"]] - row[COLUMNS["low"]]


def get_upper_shadow(row: pd.Series) -> float:
    """
    Calculate the upper shadow length.
    
    Args:
        row: DataFrame row with OHLC data
    
    Returns:
        Upper shadow length
    """
    high = row[COLUMNS["high"]]
    body_top = max(row[COLUMNS["open"]], row[COLUMNS["close"]])
    return high - body_top


def get_lower_shadow(row: pd.Series) -> float:
    """
    Calculate the lower shadow length.
    
    Args:
        row: DataFrame row with OHLC data
    
    Returns:
        Lower shadow length
    """
    low = row[COLUMNS["low"]]
    body_bottom = min(row[COLUMNS["open"]], row[COLUMNS["close"]])
    return body_bottom - low


def is_bullish(row: pd.Series) -> bool:
    """Check if candle is bullish (close > open)."""
    return row[COLUMNS["close"]] > row[COLUMNS["open"]]


def is_bearish(row: pd.Series) -> bool:
    """Check if candle is bearish (close < open)."""
    return row[COLUMNS["close"]] < row[COLUMNS["open"]]

def smooth_prices(prices: pd.Series, window: int = SMOOTHING_WINDOW) -> pd.Series:
    """
    Smooth prices using simple moving average.
    
    Args:
        prices: Price series
        window: Smoothing window
    
    Returns:
        Smoothed price series
    """
    return prices.rolling(window=window, min_periods=1).mean()


def find_local_peaks(
    prices: pd.Series, 
    distance: int = PEAK_DISTANCE, 
    prominence_pct: float = PEAK_PROMINENCE_PCT
) -> np.ndarray:
    """
    Find local maxima (peaks) in price series.
    
    Args:
        prices: Price series
        distance: Minimum distance between peaks
        prominence_pct: Minimum prominence as percentage of price range
    
    Returns:
        Array of peak indices
    """
    price_range = prices.max() - prices.min()
    if price_range == 0:
        return np.array([])
    
    prominence = price_range * prominence_pct
    
    peaks, _ = find_peaks(prices.values, distance=distance, prominence=prominence)
    return peaks


def find_local_troughs(
    prices: pd.Series, 
    distance: int = 5, 
    prominence_pct: float = 0.01
) -> np.ndarray:
    """
    Find local minima (troughs) in price series.
    
    Args:
        prices: Price series
        distance: Minimum distance between troughs
        prominence_pct: Minimum prominence as percentage of price range
    
    Returns:
        Array of trough indices
    """
    inverted = -prices
    
    price_range = prices.max() - prices.min()
    if price_range == 0:
        return np.array([])
    
    prominence = price_range * prominence_pct
    
    troughs, _ = find_peaks(inverted.values, distance=distance, prominence=prominence)
    return troughs

def find_rims_and_bottom(prices, peaks, troughs, lookback, cup_depth_min, cup_depth_max):
    for i in range(len(peaks) - 1):
        left_rim_idx = peaks[i]
        right_rim_candidates = peaks[(peaks > left_rim_idx + lookback // 3)]
        if len(right_rim_candidates) == 0:
            continue
        right_rim_idx = right_rim_candidates[0]
        left_rim_price = prices.iloc[left_rim_idx]
        right_rim_price = prices.iloc[right_rim_idx]
        rim_diff = abs(left_rim_price - right_rim_price) / left_rim_price
        if rim_diff > CUP_RIM_TOLERANCE:
            continue
        cup_troughs = troughs[(troughs > left_rim_idx) & (troughs < right_rim_idx)]
        if len(cup_troughs) == 0:
            continue
        cup_bottom_idx = cup_troughs[np.argmin(prices.iloc[cup_troughs])]
        cup_bottom_price = prices.iloc[cup_bottom_idx]
        avg_rim = (left_rim_price + right_rim_price) / 2
        cup_depth = (avg_rim - cup_bottom_price) / avg_rim
        if cup_depth < cup_depth_min or cup_depth > cup_depth_max:
            continue
        yield left_rim_idx, right_rim_idx, cup_bottom_idx, left_rim_price, right_rim_price, cup_bottom_price, cup_depth

def find_handle(prices, right_rim_idx, lookback, right_rim_price, cup_depth):
    handle_end_idx = min(right_rim_idx + lookback // 4, len(prices) - 1)
    if handle_end_idx <= right_rim_idx:
        return None
    handle_region = prices.iloc[right_rim_idx:handle_end_idx + 1]
    if len(handle_region) < 3:
        return None
    handle_low = handle_region.min()
    handle_pullback = (right_rim_price - handle_low) / right_rim_price
    if handle_pullback > cup_depth * HANDLE_PULLBACK_RATIO:
        return None
    return handle_end_idx, handle_pullback