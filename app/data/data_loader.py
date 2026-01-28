import streamlit as st
from .cache import (
    cached_load_single_asset,
    process_anomalies,
    cached_get_date_range,
    get_available_days,
    cached_load_all_assets,
)
from config import FILE_NAMES
from utils.logger import logger
from utils.dates import filter_by_date_range
import pandas as pd
from typing import Tuple, Dict
from datetime import date

def single_asset_analisys_data(
        selected_asset,
        selected_granularity,
        start_date,
        end_date,
        zscore_threshold)-> pd.DataFrame:
    try:
        with st.spinner("Loading data..."):
            df_full = cached_load_single_asset(selected_asset, selected_granularity)
            df = filter_by_date_range(df_full, str(start_date), str(end_date))
        
        if len(df) == 0:
            st.warning("No data available for selected date range.")
            st.stop()
        
        with st.spinner("Detecting anomalies..."):
            df_processed = process_anomalies(df.copy(), zscore_threshold)
            return df_processed
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()

def single_asset_analisys_dates(
        selected_asset: str,
        selected_granularity: str) -> Tuple[date, date]:
    try:
        min_date_ts, max_date_ts = cached_get_date_range(selected_asset, selected_granularity)
        min_date = min_date_ts.date()
        max_date = max_date_ts.date()
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        st.error(f"""
        **Data file not found!**
        
        Please ensure the CSV file exists:
        `data/{selected_granularity}/{FILE_NAMES[selected_asset]}`
        
        Error: {e}
        """)
        st.stop()
    except Exception as e:
        logger.error(f"Error reading data: {e}", exc_info=True)
        st.error(f"Error reading data: {e}")
        st.stop()

    return min_date, max_date

def anomaly_realtime_data(selected_asset) -> Tuple[pd.DataFrame, list]:

    try:
        with st.spinner("Loading minute data..."):
            df_full = cached_load_single_asset(selected_asset, "minute")
            available_days = get_available_days(df_full)
            return df_full, available_days
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        st.error(f"Data file not found: {e}")
        st.stop()
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        st.error(f"Error loading data: {e}")
    st.stop()

def cross_asset_data() -> Dict[str, pd.DataFrame]:
    try:
        all_data = cached_load_all_assets("daily")
    
        if len(all_data) == 0:
            st.error("No data loaded. Please check that CSV files exist.")
            st.stop()
        
        if len(all_data) < 2:
            st.error("Need at least 2 assets for cross-asset analysis.")
            st.stop()
        return all_data
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        st.error(f"Error loading data: {e}")
        st.stop()

def pattern_data(selected_asset) -> Tuple[pd.DataFrame, date, date]:
    try:
        with st.spinner("Loading data..."):
            df = cached_load_single_asset(selected_asset, "daily")
            min_date_ts, max_date_ts = cached_get_date_range(selected_asset, "daily")
            min_date = min_date_ts.date()
            max_date = max_date_ts.date()
            return df, min_date, max_date
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        st.error(f"Data file not found: {e}")
        st.stop()
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        st.error(f"Error loading data: {e}")
        st.stop()