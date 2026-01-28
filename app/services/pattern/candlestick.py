import pandas as pd
from config import COLUMNS, DOJI_BODY_RATIO_DEFAULT, HAMMER_BODY_RATIO_DEFAULT, HAMMER_SHADOW_RATIO_DEFAULT
from .helpers import get_candle_body, get_candle_range, get_upper_shadow, get_lower_shadow, is_bullish, is_bearish

def detect_doji(df: pd.DataFrame, threshold: float = DOJI_BODY_RATIO_DEFAULT) -> pd.Series:
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


def detect_hammer(df: pd.DataFrame, body_ratio: float = HAMMER_BODY_RATIO_DEFAULT, shadow_ratio: float = HAMMER_SHADOW_RATIO_DEFAULT) -> pd.Series:
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
    
    open_col = COLUMNS["open"]
    close_col = COLUMNS["close"]
    
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
    
    open_col = COLUMNS["open"]
    close_col = COLUMNS["close"]
    
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