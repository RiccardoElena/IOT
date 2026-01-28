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

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Import only what we need from config
from config.patterns import (
    SMOOTH_WINDOW,
    PEAK_PROMINENCE,
    CUP_RIM_TOLERANCE,
    HANDLE_PULLBACK_RATIO,
)
from config.data import COLUMNS
from .helpers import smooth_prices, find_local_peaks, find_local_troughs, find_rims_and_bottom, find_handle


# =============================================================================
# CHART PATTERNS
# =============================================================================

def detect_double_top(
    df: pd.DataFrame,
    min_confidence: float = 0.5,
    min_distance: int = 5,
    prominence_pct: float = 0.01,
) -> List[Dict]:
    """
    Detect Double Top pattern (M shape).
    
    Double Top: Two peaks at similar levels with a trough between.
    Bearish reversal pattern.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Window size for pattern detection
        min_confidence: Minimum confidence for pattern detection (0.5 default)
        min_distance: Minimum distance between peaks
        prominence_pct: Minimum prominence for peak detection (1% default - LOW)
    
    Returns:
        List of detected patterns with details
    """
    close_col = COLUMNS["close"]
    prices = df[close_col]
    smoothed = smooth_prices(prices, window=5)
    
    patterns: List[Dict] = []
    
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
        
        # Check if peaks are at similar levels (within min_con)
        avg_peak = (peak1_price + peak2_price) / 2
        price_diff = abs(peak1_price - peak2_price) / avg_peak
        
        confidence = 1.0 - price_diff
        if confidence < min_confidence:
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
            "signal": "Bearish reversal",
            "confidence": confidence
        })
    
    return patterns


def detect_double_bottom(
    df: pd.DataFrame, 
    min_confidence: float = 0.5,
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
        min_confidence: Minimum confidence for pattern detection (0.5 default)
        min_distance: Minimum distance between troughs
        prominence_pct: Minimum prominence for trough detection (1% default - LOW)
    
    Returns:
        List of detected patterns with details
    """
    close_col = COLUMNS["close"]
    prices = df[close_col]
    smoothed = smooth_prices(prices, window=5)
    
    patterns: List[Dict] =  []
    
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
        confidence = 1.0 - price_diff
        
        if confidence < min_confidence:
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
            "signal": "Bullish reversal",
            "confidence": confidence
        })
    
    return patterns


def detect_head_and_shoulders(
    df: pd.DataFrame, 
    min_confidence: float = 0.5,
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
        min_confidence: Minimum confidence for pattern detection (0.5 default)
        min_distance: Minimum distance between peaks
        prominence_pct: Minimum prominence for peak detection
    
    Returns:
        List of detected patterns with details
    """
    close_col = COLUMNS["close"]
    prices = df[close_col]
    smoothed = smooth_prices(prices, window=5)
    
    patterns: List[Dict] =  []
    
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
        min_shoulder = min(left_price, right_price)
        if min_shoulder > 0:
            shoulder_diff = abs(left_price - right_price) / min_shoulder
        else:
            # fallback: se entrambe sono 0 consideriamo differenza massima
            shoulder_diff = 1.0

        # confidence normalizzata in [0..1]
        confidence = max(0.0, min(1.0, 1.0 - shoulder_diff))
        if confidence < min_confidence:
            continue
        
        # Head should be notably higher (at least 1%)
        max_shoulder = max(left_price, right_price)
        
        head_prominence = (head_price - max_shoulder) / head_price
        if head_prominence < shoulder_diff:
            continue
        print(head_prominence, shoulder_diff)
        
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
            "signal": "Bearish reversal",
            "confidence": confidence
        })
    
    return patterns

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def compute_cup_handle_confidence(
    left_rim_price: float,
    right_rim_price: float,
    cup_depth: float,
    handle_pullback: float,
    cup_depth_min: float,
    cup_depth_max: float,
    rim_tolerance: float = CUP_RIM_TOLERANCE,
    handle_ratio: float = HANDLE_PULLBACK_RATIO,
    weights: tuple = (0.30, 0.50, 0.20)
) -> float:
    """
    Calcola la confidence del pattern Cup & Handle combinando:
      - similarità dei rim (rim_score)
      - profondità della cup (depth_score)
      - pullback dell'handle (handle_score)

    Restituisce valore normalizzato [0..1].
    """
    W_RIM, W_DEPTH, W_HANDLE = weights

    avg_rim = (left_rim_price + right_rim_price) / 2 if (left_rim_price + right_rim_price) != 0 else 0.0
    rim_diff = abs(left_rim_price - right_rim_price) / avg_rim if avg_rim != 0 else 1.0
    rim_score = _clamp01(1.0 - (rim_diff / rim_tolerance)) if rim_tolerance > 0 else _clamp01(1.0 - rim_diff)

    if cup_depth_max > cup_depth_min:
        depth_score = _clamp01((cup_depth - cup_depth_min) / (cup_depth_max - cup_depth_min))
    else:
        depth_score = _clamp01(cup_depth)

    handle_threshold = cup_depth * handle_ratio
    if handle_threshold > 0:
        handle_score = _clamp01(1.0 - (handle_pullback / handle_threshold))
    else:
        handle_score = 1.0 if handle_pullback == 0 else 0.0

    return _clamp01(rim_score * W_RIM + depth_score * W_DEPTH + handle_score * W_HANDLE)


def detect_cup_and_handle(
    df: pd.DataFrame, 
    tolerance: float,
    lookback: int = 60,
    cup_depth_min: float = 0.05,
    cup_depth_max: float = 0.50
) -> List[Dict]:
    """
    Detect Cup and Handle pattern with dynamic confidence score.
    """
    close_col = COLUMNS["close"]
    prices = df[close_col]
    smoothed = smooth_prices(prices, window=SMOOTH_WINDOW)
    patterns: List[Dict] = []
    peaks = find_local_peaks(smoothed, distance=lookback // 4, prominence_pct=PEAK_PROMINENCE)
    troughs = find_local_troughs(smoothed, distance=lookback // 4, prominence_pct=PEAK_PROMINENCE)
    if len(peaks) < 2 or len(troughs) < 1:
        return patterns

    for left_rim_idx, right_rim_idx, _ , left_rim_price, right_rim_price, cup_bottom_price, cup_depth in \
            find_rims_and_bottom(prices, peaks, troughs, lookback, cup_depth_min, cup_depth_max):
        handle = find_handle(prices, right_rim_idx, lookback, right_rim_price, cup_depth)
        if handle is None:
            continue
        handle_end_idx, handle_pullback = handle

        # Calcola la confidence usando la funzione estratta
        confidence = compute_cup_handle_confidence(
            left_rim_price=left_rim_price,
            right_rim_price=right_rim_price,
            cup_depth=cup_depth,
            handle_pullback=handle_pullback,
            cup_depth_min=cup_depth_min,
            cup_depth_max=cup_depth_max
        )

        if 1-confidence < tolerance:
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
            "signal": "Bullish continuation",
            "confidence": confidence
        })
    return patterns