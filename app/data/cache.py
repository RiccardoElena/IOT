import streamlit as st
from typing import Any, Dict, List, Tuple
from datetime import timedelta
import pandas as pd
from .core import load_single_asset, get_date_range, load_all_assets
from services import detect_anomalies


@st.cache_data
def cached_load_single_asset(asset: str, granularity: str) -> pd.DataFrame:
    """Load full dataset for an asset."""
    return load_single_asset(asset, granularity)

@st.cache_data
def cached_load_all_assets(granularity: str) -> Dict[str, pd.DataFrame]:
    """Load all assets for a given granularity."""
    return load_all_assets(granularity)


@st.cache_data
def cached_get_date_range(asset: str, granularity: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get min/max dates for a dataset without loading all data."""
    return get_date_range(asset, granularity)


@st.cache_data
def process_anomalies(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Run anomaly detection on data."""
    return detect_anomalies(df, zscore_threshold=threshold, mode="batch")


@st.cache_data
def get_available_weeks(asset: str) -> List[Dict[str, Any]]:
    """Get list of available weeks for minute data."""
    df = cached_load_single_asset(asset, "minute")
    
    dates = pd.to_datetime(df.index).date
    min_date = dates.min()
    max_date = dates.max()
    
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
def get_available_days(df: pd.DataFrame) -> List:
    # Usa normalize() per ottenere solo le date, evitando l'accesso diretto a .date
    dates = pd.DatetimeIndex(df.index).normalize().unique()
    return sorted([d.date() for d in dates])
