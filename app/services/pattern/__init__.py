import pandas as pd
from typing import Dict, List
from .chart import *
from .candlestick import *


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
    min_confidence: float = 0.5,
    prominence_pct: float = 0.01,
) -> List[Dict]:
    """
    Detect all chart patterns in the data.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Window size for pattern detection
        min_confidence: Minimum confidence for pattern detection (0.5 default)
        prominence_pct: Minimum prominence for peak/trough detection (1% default - LOW)
    Returns:
        List of all detected patterns
    """
    patterns: List[Dict] =  []
    
    min_distance = max(3, lookback // 10)
    
    patterns.extend(detect_double_top(
        df, min_confidence,
        min_distance=min_distance, 
        prominence_pct=prominence_pct
    ))
    
    patterns.extend(detect_double_bottom(
        df, min_confidence,
        min_distance=min_distance,
        prominence_pct=prominence_pct
    ))
    
    patterns.extend(detect_head_and_shoulders(
        df, min_confidence,
        min_distance=min_distance,
        prominence_pct=prominence_pct
    ))
    
    patterns.extend(detect_cup_and_handle(
        df, min_confidence, lookback,
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
    patterns: List[Dict] =  []
    close_col = COLUMNS["close"]
    
    pattern_cols = {
        "pattern_doji": "Doji",
        "pattern_hammer": "Hammer",
        "pattern_engulfing_bullish": "Engulfing Bullish",
        "pattern_engulfing_bearish": "Engulfing Bearish"
    }

    signals = {
        "pattern_doji": "Neutral",
        "pattern_hammer": "Bullish reversal",
        "pattern_engulfing_bullish": "Bullish reversal",
        "pattern_engulfing_bearish": "Bearish reversal"
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
                "signal": signals.get(col, "Neutral")
            })
    
    if not patterns:
        return pd.DataFrame(columns=["timestamp", "pattern", "price", "signal"])
    
    result = pd.DataFrame(patterns)
    result = result.sort_values("timestamp")
    result.index = list(range(1, len(result) + 1))
    
    return result
