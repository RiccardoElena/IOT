"""
Pattern Recognition Module for IoT Financial Data Analytics.

This module implements pattern recognition techniques:

Candlestick Patterns (1-2 candles):
- Doji: Indecision pattern
- Hammer: Bullish reversal
- Engulfing Bullish: Bullish reversal
- Engulfing Bearish: Bearish reversal

Chart Patterns (multi-candle):
- Double Top: Bearish reversal (M shape)
- Double Bottom: Bullish reversal (W shape)
- Head and Shoulders: Bearish reversal
- Cup and Handle: Bullish continuation

Default parameters are permissive to detect more patterns.
Use calibration sliders in the UI to fine-tune.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# =============================================================================
# CANDLESTICK HELPER FUNCTIONS
# =============================================================================

def get_candle_body(row: pd.Series) -> float:
    """
    Calculate the body size of a candle.
    
    Args:
        row: DataFrame row with OHLC data
    
    Returns:
        Body size (absolute value)
    """
    return abs(row[config.COLUMNS["close"]] - row[config.COLUMNS["open"]])


def get_candle_range(row: pd.Series) -> float:
    """
    Calculate the full range of a candle (high - low).
    
    Args:
        row: DataFrame row with OHLC data
    
    Returns:
        Full range
    """
    return row[config.COLUMNS["high"]] - row[config.COLUMNS["low"]]


def get_upper_shadow(row: pd.Series) -> float:
    """
    Calculate the upper shadow length.
    
    Args:
        row: DataFrame row with OHLC data
    
    Returns:
        Upper shadow length
    """
    high = row[config.COLUMNS["high"]]
    body_top = max(row[config.COLUMNS["open"]], row[config.COLUMNS["close"]])
    return high - body_top


def get_lower_shadow(row: pd.Series) -> float:
    """
    Calculate the lower shadow length.
    
    Args:
        row: DataFrame row with OHLC data
    
    Returns:
        Lower shadow length
    """
    low = row[config.COLUMNS["low"]]
    body_bottom = min(row[config.COLUMNS["open"]], row[config.COLUMNS["close"]])
    return body_bottom - low


def is_bullish(row: pd.Series) -> bool:
    """Check if candle is bullish (close > open)."""
    return row[config.COLUMNS["close"]] > row[config.COLUMNS["open"]]


def is_bearish(row: pd.Series) -> bool:
    """Check if candle is bearish (close < open)."""
    return row[config.COLUMNS["close"]] < row[config.COLUMNS["open"]]


# =============================================================================
# CANDLESTICK PATTERNS
# =============================================================================

def detect_doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
    """
    Detect Doji pattern.
    
    Doji: Body is very small relative to the range.
    Indicates market indecision.
    
    Args:
        df: DataFrame with OHLC data
        threshold: Maximum body/range ratio for Doji (default 0.1 = 10%)
    
    Returns:
        Boolean Series (True = Doji detected)
    """
    results = []
    
    for idx, row in df.iterrows():
        body = get_candle_body(row)
        range_val = get_candle_range(row)
        
        if range_val == 0:
            results.append(False)
            continue
        
        ratio = body / range_val
        results.append(ratio < threshold)
    
    return pd.Series(results, index=df.index, name="doji")


def detect_hammer(df: pd.DataFrame, body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> pd.Series:
    """
    Detect Hammer pattern.
    
    Hammer: Small body at top, long lower shadow (at least 2x body).
    Bullish reversal pattern at bottom of downtrend.
    
    Args:
        df: DataFrame with OHLC data
        body_ratio: Maximum body/range ratio
        shadow_ratio: Minimum lower_shadow/body ratio
    
    Returns:
        Boolean Series (True = Hammer detected)
    """
    results = []
    
    for idx, row in df.iterrows():
        body = get_candle_body(row)
        range_val = get_candle_range(row)
        lower_shadow = get_lower_shadow(row)
        upper_shadow = get_upper_shadow(row)
        
        if range_val == 0 or body == 0:
            results.append(False)
            continue
        
        body_small = (body / range_val) < body_ratio
        lower_long = lower_shadow >= (body * shadow_ratio)
        upper_small = upper_shadow < body
        
        results.append(body_small and lower_long and upper_small)
    
    return pd.Series(results, index=df.index, name="hammer")


def detect_engulfing_bullish(df: pd.DataFrame) -> pd.Series:
    """
    Detect Bullish Engulfing pattern.
    
    Bullish Engulfing: Current bullish candle completely engulfs 
    the previous bearish candle.
    """
    results = [False]  # First candle can't be engulfing
    
    open_col = config.COLUMNS["open"]
    close_col = config.COLUMNS["close"]
    
    for i in range(1, len(df)):
        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]
        
        prev_bearish = is_bearish(prev_row)
        curr_bullish = is_bullish(curr_row)
        
        curr_open = curr_row[open_col]
        curr_close = curr_row[close_col]
        prev_open = prev_row[open_col]
        prev_close = prev_row[close_col]
        
        engulfs = (curr_open < prev_close) and (curr_close > prev_open)
        
        results.append(prev_bearish and curr_bullish and engulfs)
    
    return pd.Series(results, index=df.index, name="engulfing_bullish")


def detect_engulfing_bearish(df: pd.DataFrame) -> pd.Series:
    """
    Detect Bearish Engulfing pattern.
    
    Bearish Engulfing: Current bearish candle completely engulfs 
    the previous bullish candle.
    """
    results = [False]
    
    open_col = config.COLUMNS["open"]
    close_col = config.COLUMNS["close"]
    
    for i in range(1, len(df)):
        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]
        
        prev_bullish = is_bullish(prev_row)
        curr_bearish = is_bearish(curr_row)
        
        curr_open = curr_row[open_col]
        curr_close = curr_row[close_col]
        prev_open = prev_row[open_col]
        prev_close = prev_row[close_col]
        
        engulfs = (curr_open > prev_close) and (curr_close < prev_open)
        
        results.append(prev_bullish and curr_bearish and engulfs)
    
    return pd.Series(results, index=df.index, name="engulfing_bearish")


# =============================================================================
# CHART PATTERN HELPER FUNCTIONS
# =============================================================================

def smooth_prices(prices: pd.Series, window: int = 5) -> pd.Series:
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
    distance: int = 5, 
    prominence_pct: float = 0.01
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


# =============================================================================
# CHART PATTERNS
# =============================================================================

def detect_double_top(
    df: pd.DataFrame, 
    lookback: int = 50,
    tolerance: float = 0.08,
    min_distance: int = 5,
    prominence_pct: float = 0.01
) -> List[Dict]:
    """
    Detect Double Top pattern (M shape).
    
    Double Top: Two peaks at similar levels with a trough between.
    Bearish reversal pattern.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Window size for pattern detection
        tolerance: Price tolerance for matching peaks (8% default - PERMISSIVE)
        min_distance: Minimum distance between peaks
        prominence_pct: Minimum prominence for peak detection (1% default - LOW)
    
    Returns:
        List of detected patterns with details
    """
    close_col = config.COLUMNS["close"]
    prices = df[close_col]
    smoothed = smooth_prices(prices, window=5)
    
    patterns = []
    
    # Find peaks and troughs with permissive settings
    peaks = find_local_peaks(smoothed, distance=min_distance, prominence_pct=prominence_pct)
    troughs = find_local_troughs(smoothed, distance=min_distance, prominence_pct=prominence_pct)
    
    if len(peaks) < 2:
        return patterns
    
    # Look for pairs of peaks with a trough between
    for i in range(len(peaks) - 1):
        peak1_idx = peaks[i]
        peak2_idx = peaks[i + 1]
        
        peak1_price = prices.iloc[peak1_idx]
        peak2_price = prices.iloc[peak2_idx]
        
        # Check if peaks are at similar levels (within tolerance)
        avg_peak = (peak1_price + peak2_price) / 2
        price_diff = abs(peak1_price - peak2_price) / avg_peak
        
        if price_diff > tolerance:
            continue
        
        # Find trough between peaks
        troughs_between = troughs[(troughs > peak1_idx) & (troughs < peak2_idx)]
        if len(troughs_between) == 0:
            continue
        
        trough_idx = troughs_between[np.argmin(prices.iloc[troughs_between])]
        trough_price = prices.iloc[trough_idx]
        
        # Trough should be lower than peaks (at least 1%)
        neckline_drop = (avg_peak - trough_price) / avg_peak
        if neckline_drop < 0.01:
            continue
        
        patterns.append({
            "type": "Double Top",
            "start_idx": peak1_idx,
            "end_idx": peak2_idx,
            "start_date": df.index[peak1_idx],
            "end_date": df.index[peak2_idx],
            "peak1_price": peak1_price,
            "peak2_price": peak2_price,
            "trough_price": trough_price,
            "neckline": trough_price,
            "signal": "Bearish",
            "confidence": 1.0 - price_diff
        })
    
    return patterns


def detect_double_bottom(
    df: pd.DataFrame, 
    lookback: int = 50,
    tolerance: float = 0.08,
    min_distance: int = 5,
    prominence_pct: float = 0.01
) -> List[Dict]:
    """
    Detect Double Bottom pattern (W shape).
    
    Double Bottom: Two troughs at similar levels with a peak between.
    Bullish reversal pattern.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Window size for pattern detection
        tolerance: Price tolerance for matching troughs (8% default - PERMISSIVE)
        min_distance: Minimum distance between troughs
        prominence_pct: Minimum prominence for trough detection (1% default - LOW)
    
    Returns:
        List of detected patterns with details
    """
    close_col = config.COLUMNS["close"]
    prices = df[close_col]
    smoothed = smooth_prices(prices, window=5)
    
    patterns = []
    
    peaks = find_local_peaks(smoothed, distance=min_distance, prominence_pct=prominence_pct)
    troughs = find_local_troughs(smoothed, distance=min_distance, prominence_pct=prominence_pct)
    
    if len(troughs) < 2:
        return patterns
    
    for i in range(len(troughs) - 1):
        trough1_idx = troughs[i]
        trough2_idx = troughs[i + 1]
        
        trough1_price = prices.iloc[trough1_idx]
        trough2_price = prices.iloc[trough2_idx]
        
        # Check if troughs are at similar levels
        avg_trough = (trough1_price + trough2_price) / 2
        price_diff = abs(trough1_price - trough2_price) / avg_trough
        
        if price_diff > tolerance:
            continue
        
        # Find peak between troughs
        peaks_between = peaks[(peaks > trough1_idx) & (peaks < trough2_idx)]
        if len(peaks_between) == 0:
            continue
        
        peak_idx = peaks_between[np.argmax(prices.iloc[peaks_between])]
        peak_price = prices.iloc[peak_idx]
        
        # Peak should be higher than troughs (at least 1%)
        neckline_rise = (peak_price - avg_trough) / avg_trough
        if neckline_rise < 0.01:
            continue
        
        patterns.append({
            "type": "Double Bottom",
            "start_idx": trough1_idx,
            "end_idx": trough2_idx,
            "start_date": df.index[trough1_idx],
            "end_date": df.index[trough2_idx],
            "trough1_price": trough1_price,
            "trough2_price": trough2_price,
            "peak_price": peak_price,
            "neckline": peak_price,
            "signal": "Bullish",
            "confidence": 1.0 - price_diff
        })
    
    return patterns


def detect_head_and_shoulders(
    df: pd.DataFrame, 
    lookback: int = 60,
    tolerance: float = 0.08,
    min_distance: int = 5,
    prominence_pct: float = 0.01
) -> List[Dict]:
    """
    Detect Head and Shoulders pattern.
    
    H&S: Three peaks - left shoulder, higher head, right shoulder at similar 
    level to left. Bearish reversal pattern.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Window size for pattern detection
        tolerance: Price tolerance for matching shoulders (8% default)
        min_distance: Minimum distance between peaks
        prominence_pct: Minimum prominence for peak detection
    
    Returns:
        List of detected patterns with details
    """
    close_col = config.COLUMNS["close"]
    prices = df[close_col]
    smoothed = smooth_prices(prices, window=5)
    
    patterns = []
    
    peaks = find_local_peaks(smoothed, distance=min_distance, prominence_pct=prominence_pct)
    troughs = find_local_troughs(smoothed, distance=min_distance, prominence_pct=prominence_pct)
    
    if len(peaks) < 3:
        return patterns
    
    for i in range(len(peaks) - 2):
        left_idx = peaks[i]
        head_idx = peaks[i + 1]
        right_idx = peaks[i + 2]
        
        left_price = prices.iloc[left_idx]
        head_price = prices.iloc[head_idx]
        right_price = prices.iloc[right_idx]
        
        # Head must be higher than both shoulders
        if head_price <= left_price or head_price <= right_price:
            continue
        
        # Shoulders should be at similar levels
        avg_shoulder = (left_price + right_price) / 2
        shoulder_diff = abs(left_price - right_price) / avg_shoulder
        
        if shoulder_diff > tolerance:
            continue
        
        # Head should be notably higher (at least 1%)
        head_prominence = (head_price - avg_shoulder) / avg_shoulder
        if head_prominence < 0.01:
            continue
        
        # Calculate neckline
        troughs_left = troughs[(troughs > left_idx) & (troughs < head_idx)]
        troughs_right = troughs[(troughs > head_idx) & (troughs < right_idx)]
        
        if len(troughs_left) == 0 or len(troughs_right) == 0:
            continue
        
        neckline_left = prices.iloc[troughs_left[0]]
        neckline_right = prices.iloc[troughs_right[0]]
        neckline = (neckline_left + neckline_right) / 2
        
        patterns.append({
            "type": "Head & Shoulders",
            "start_idx": left_idx,
            "end_idx": right_idx,
            "start_date": df.index[left_idx],
            "end_date": df.index[right_idx],
            "left_shoulder": left_price,
            "head": head_price,
            "right_shoulder": right_price,
            "neckline": neckline,
            "signal": "Bearish",
            "confidence": 1.0 - shoulder_diff
        })
    
    return patterns


def detect_cup_and_handle(
    df: pd.DataFrame, 
    lookback: int = 60,
    cup_depth_min: float = 0.05,
    cup_depth_max: float = 0.50
) -> List[Dict]:
    """
    Detect Cup and Handle pattern.
    
    Cup and Handle: Rounded bottom (cup) followed by small pullback (handle).
    Bullish continuation pattern.
    
    PERMISSIVE CRITERIA - detects more patterns, user can filter.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Window size for pattern detection
        cup_depth_min: Minimum cup depth (5% default - LOW)
        cup_depth_max: Maximum cup depth (50% default - HIGH)
    
    Returns:
        List of detected patterns with details
    """
    close_col = config.COLUMNS["close"]
    prices = df[close_col]
    smoothed = smooth_prices(prices, window=5)
    
    patterns = []
    
    # Find peaks (potential cup rims)
    peaks = find_local_peaks(smoothed, distance=lookback // 4, prominence_pct=0.01)
    troughs = find_local_troughs(smoothed, distance=lookback // 4, prominence_pct=0.01)
    
    if len(peaks) < 2 or len(troughs) < 1:
        return patterns
    
    for i in range(len(peaks) - 1):
        left_rim_idx = peaks[i]
        
        # Find right rim candidate
        right_rim_candidates = peaks[(peaks > left_rim_idx + lookback // 3)]
        if len(right_rim_candidates) == 0:
            continue
        
        right_rim_idx = right_rim_candidates[0]
        
        left_rim_price = prices.iloc[left_rim_idx]
        right_rim_price = prices.iloc[right_rim_idx]
        
        # Rims should be at similar levels (within 15%)
        rim_diff = abs(left_rim_price - right_rim_price) / left_rim_price
        if rim_diff > 0.15:
            continue
        
        # Find cup bottom between rims
        cup_troughs = troughs[(troughs > left_rim_idx) & (troughs < right_rim_idx)]
        if len(cup_troughs) == 0:
            continue
        
        cup_bottom_idx = cup_troughs[np.argmin(prices.iloc[cup_troughs])]
        cup_bottom_price = prices.iloc[cup_bottom_idx]
        
        # Check cup depth
        avg_rim = (left_rim_price + right_rim_price) / 2
        cup_depth = (avg_rim - cup_bottom_price) / avg_rim
        
        if cup_depth < cup_depth_min or cup_depth > cup_depth_max:
            continue
        
        # Look for handle (small pullback after right rim)
        handle_end_idx = min(right_rim_idx + lookback // 4, len(prices) - 1)
        if handle_end_idx <= right_rim_idx:
            continue
        
        handle_region = prices.iloc[right_rim_idx:handle_end_idx + 1]
        if len(handle_region) < 3:
            continue
        
        handle_low = handle_region.min()
        handle_pullback = (right_rim_price - handle_low) / right_rim_price
        
        # Handle should be a small pullback (less than half of cup depth)
        if handle_pullback > cup_depth * 0.5:
            continue
        
        patterns.append({
            "type": "Cup & Handle",
            "start_idx": left_rim_idx,
            "end_idx": handle_end_idx,
            "start_date": df.index[left_rim_idx],
            "end_date": df.index[handle_end_idx],
            "left_rim": left_rim_price,
            "cup_bottom": cup_bottom_price,
            "right_rim": right_rim_price,
            "cup_depth_pct": cup_depth * 100,
            "handle_depth_pct": handle_pullback * 100,
            "signal": "Bullish",
            "confidence": 0.7  # Fixed confidence for cup & handle
        })
    
    return patterns


# =============================================================================
# MAIN DETECTION FUNCTIONS
# =============================================================================

def detect_all_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect all candlestick patterns in the data.
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame with pattern detection columns
    """
    result = df.copy()
    
    result["pattern_doji"] = detect_doji(df)
    result["pattern_hammer"] = detect_hammer(df)
    result["pattern_engulfing_bullish"] = detect_engulfing_bullish(df)
    result["pattern_engulfing_bearish"] = detect_engulfing_bearish(df)
    
    result["has_pattern"] = (
        result["pattern_doji"] | 
        result["pattern_hammer"] | 
        result["pattern_engulfing_bullish"] | 
        result["pattern_engulfing_bearish"]
    )
    
    return result


def detect_all_chart_patterns(
    df: pd.DataFrame, 
    lookback: int = 50,
    tolerance: float = 0.08,
    prominence_pct: float = 0.01
) -> List[Dict]:
    """
    Detect all chart patterns in the data.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Window size for pattern detection
        tolerance: Price tolerance for matching levels (8% default - PERMISSIVE)
        prominence_pct: Minimum prominence for peak/trough detection (1% default - LOW)
    
    Returns:
        List of all detected patterns
    """
    patterns = []
    
    min_distance = max(3, lookback // 10)
    
    patterns.extend(detect_double_top(
        df, lookback, tolerance, 
        min_distance=min_distance, 
        prominence_pct=prominence_pct
    ))
    
    patterns.extend(detect_double_bottom(
        df, lookback, tolerance,
        min_distance=min_distance,
        prominence_pct=prominence_pct
    ))
    
    patterns.extend(detect_head_and_shoulders(
        df, lookback, tolerance,
        min_distance=min_distance,
        prominence_pct=prominence_pct
    ))
    
    patterns.extend(detect_cup_and_handle(
        df, lookback,
        cup_depth_min=0.05,
        cup_depth_max=0.50
    ))
    
    # Sort by start date
    patterns.sort(key=lambda x: x["start_date"])
    
    return patterns


def get_pattern_summary(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get summary of candlestick pattern counts.
    
    Args:
        df: DataFrame with pattern detection columns
    
    Returns:
        Dictionary with pattern counts
    """
    return {
        "doji": int(df["pattern_doji"].sum()) if "pattern_doji" in df.columns else 0,
        "hammer": int(df["pattern_hammer"].sum()) if "pattern_hammer" in df.columns else 0,
        "engulfing_bullish": int(df["pattern_engulfing_bullish"].sum()) if "pattern_engulfing_bullish" in df.columns else 0,
        "engulfing_bearish": int(df["pattern_engulfing_bearish"].sum()) if "pattern_engulfing_bearish" in df.columns else 0,
    }


def get_pattern_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a table of all detected candlestick patterns.
    
    Args:
        df: DataFrame with pattern detection columns
    
    Returns:
        DataFrame with pattern details
    """
    patterns = []
    close_col = config.COLUMNS["close"]
    
    pattern_cols = {
        "pattern_doji": "Doji",
        "pattern_hammer": "Hammer",
        "pattern_engulfing_bullish": "Engulfing Bullish",
        "pattern_engulfing_bearish": "Engulfing Bearish"
    }
    
    for col, name in pattern_cols.items():
        if col not in df.columns:
            continue
        
        mask = df[col]
        for idx in df[mask].index:
            row = df.loc[idx]
            patterns.append({
                "timestamp": idx,
                "pattern": name,
                "price": row[close_col],
                "signal": "Bullish" if "Bullish" in name or name == "Hammer" else (
                    "Bearish" if "Bearish" in name else "Neutral"
                )
            })
    
    if not patterns:
        return pd.DataFrame(columns=["timestamp", "pattern", "price", "signal"])
    
    result = pd.DataFrame(patterns)
    result = result.sort_values("timestamp")
    result.index = range(1, len(result) + 1)
    
    return result
