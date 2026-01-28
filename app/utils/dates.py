from typing import List, Dict, Any, Optional
from datetime import date, timedelta
import streamlit as st
import pandas as pd

@st.cache_data
def get_weeks_in_range(min_date: date, max_date: date) -> List[Dict[str, Any]]:
    """Get list of available weeks between min_date and max_date."""
    weeks = []
    current_start = min_date

    while current_start <= max_date:
        current_end = current_start + timedelta(days=6)
        if current_end > max_date:
            current_end = max_date

        weeks.append({
            "start": current_start,
            "end": current_end,
            "label": f"{current_start.strftime('%Y-%m-%d')} → {current_end.strftime('%Y-%m-%d')}"
        })

        current_start = current_end + timedelta(days=1)

    return weeks

@st.cache_data
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

@st.cache_data
def filter_day_data(df: pd.DataFrame, selected_day) -> pd.DataFrame:
    """Filter dataframe to a specific day - cached for performance."""
    # Usa operazioni vettorizzate per massima performance
    start = pd.Timestamp(selected_day)
    end = start + pd.Timedelta(days=1)
    return df[(df.index >= start) & (df.index < end)].copy()