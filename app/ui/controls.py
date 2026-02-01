import streamlit as st
from typing import Tuple
from config import ZSCORE_ANOMALY_THRESHOLD, CORRELATION_WINDOW, SYSTEMIC_EVENT_THRESHOLD
from utils import get_weeks_in_range

def single_asset_controls(reset_zoom, asset_options, granularity_options):
    selected_asset = st.selectbox(
        "Select Asset",
        options=list(asset_options.keys()),
        on_change=reset_zoom,
        format_func=lambda x: asset_options[x]
    )
    
    # Granularity selection
    selected_granularity = st.selectbox(
        "Select Granularity",
        options=list(granularity_options.keys()),
        on_change=reset_zoom,
        format_func=lambda x: granularity_options[x],
        index=2  # Default to daily
    )
    
    st.markdown("---")
    
    # Z-Score threshold
    zscore_threshold = st.slider(
        "Z-Score Threshold",
        min_value=1.0,
        max_value=5.0,
        value=ZSCORE_ANOMALY_THRESHOLD,
        step=0.5,
        help="Values beyond this threshold are classified as anomalies"
    )
    
    # Show anomalies toggle
    show_anomalies = st.checkbox("Highlight Anomalies", value=True)
    return selected_asset,selected_granularity,zscore_threshold,show_anomalies

def realtime_controls(asset_options) -> Tuple[str | None, int, float, int]:
    selected_asset = st.selectbox(
        "Select Asset",
        options=list(asset_options.keys()),
        format_func=lambda x: asset_options[x],
         key="selected_asset_key"
    )
    
    st.markdown("---")
    
    window_size = st.slider(
        "Sliding Window Size",
        min_value=20,
        max_value=120,
        value=60,
        step=10,
        help="Number of points used for rolling statistics"
    )
    
    zscore_threshold = st.slider(
        "Z-Score Threshold",
        min_value=1.5,
        max_value=4.0,
        value=float(ZSCORE_ANOMALY_THRESHOLD),
        step=0.5,
        help="Values beyond this threshold are flagged as anomalies"
    )
    
    sim_speed = st.slider(
        "Simulation Speed",
        min_value=1,
        max_value=50,
        value=10,
        help="Points per batch (higher = faster simulation)"
    )
    
    return selected_asset, window_size, zscore_threshold, sim_speed


def cross_asset_controls():
    correlation_window = st.slider(
        "Correlation Window (days)",
        min_value=10,
        max_value=60,
        value=CORRELATION_WINDOW,
        step=5,
        help="Number of days used to calculate rolling correlation. Smaller = more reactive, larger = more stable."
    )
    
    # Systemic event threshold
    systemic_threshold = st.slider(
        "Systemic Event Threshold",
        min_value=2,
        max_value=5,
        value=SYSTEMIC_EVENT_THRESHOLD,
        help="Minimum number of assets that must show anomalies simultaneously to flag as a systemic event."
    )
    
    return correlation_window,systemic_threshold

def pattern_controls(asset_options):
    selected_asset = st.selectbox(
        "Select Asset",
        options=list(asset_options.keys()),
        format_func=lambda x: asset_options[x]
    )
    
    st.info("Using **Daily** data for pattern recognition")
    
    st.markdown("---")
    st.subheader("Pattern Calibration")
    
    prominence = st.slider(
        "Peak Prominence (%)",
        min_value=0.5,
        max_value=5.0,
        value=5.0,
        step=0.5,
        help="Minimum height of peaks/troughs"
    )
    prominence_decimal = prominence / 100
    
    chart_lookback = st.slider(
        "Chart Pattern Window",
        min_value=20,
        max_value=100,
        value=50,
        step=10,
        help="Time window for detecting multi-candle patterns"
    )

    minimal_confidence = st.slider(
        "Minimal Confidence (%)",
        min_value=50,
        max_value=100,
        value=70,
        step=5,
        help="Minimum confidence for chart pattern detection"
    ) / 100
    
    return selected_asset,prominence,prominence_decimal,chart_lookback,minimal_confidence

def date_selector(selected_granularity, min_date, max_date):
    st.markdown("### 📅 Date Range")

    if selected_granularity == "minute":
        st.info("⚠️ Minute data is limited to **one week at a time** for performance.")
    
        weeks = get_weeks_in_range(min_date, max_date)
    
        selected_week = st.selectbox(
            "Select Week",
            options=range(len(weeks)),
            format_func=lambda i: weeks[i]["label"],
            index=len(weeks) - 1
        )
    
        start_date = weeks[selected_week]["start"]
        end_date = weeks[selected_week]["end"]

    else:
        col1, col2 = st.columns(2)
    
        with col1:
            start_date = st.date_input(
            "Start Date", 
            value=min_date, 
            min_value=min_date, 
            max_value=max_date
        )
        with col2:
            end_date = st.date_input(
            "End Date", 
            value=max_date, 
            min_value=min_date, 
            max_value=max_date
        )

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()
    return start_date,end_date