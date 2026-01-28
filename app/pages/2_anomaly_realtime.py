"""
Real-time Anomaly Detection Page (IoT Simulation)

This page simulates IoT-style streaming data processing with:
- Sliding window anomaly detection using rolling Z-scores
- Real-time chart updates with price, volume, and Z-score visualization
- Anomaly logging with severity classification
- Post-simulation analysis summary

Gemini AI assistant provides contextual help based on simulation state.

Run with: streamlit run app.py (then navigate to this page)
"""

import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import streamlit as st

import config
from config.ui import PageType
from utils.dates import filter_day_data
from data import (
    anomaly_realtime_data
)

# Import UI components including Gemini sidebar
from pages.components import (
    title,
    footer,
    render_chat,
    render_chart_add_button,
)

# Import data loader
from utils.dictionaries import (
    get_asset_display_name,
)

# Import Gemini context builder for this page
from services import context_builder_factory, calculate_anomalies_batch, process_batch, compute_anomaly_log
from ui import create_combined_chart, realtime_controls

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=f"Real-time IoT Simulation | {config.PAGE_TITLE}",
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT
)

title("Real-time Anomaly Detection",
      "Simulating IoT data streaming with sliding window anomaly detection.")


# =============================================================================
# SIDEBAR - CONTROLS AND INFO
# =============================================================================

with st.sidebar:
    st.header("Controls")
    
    asset_options = {key: get_asset_display_name(key) for key in config.ASSETS.keys()}
    selected_asset, window_size, zscore_threshold, sim_speed = realtime_controls(asset_options)
    
    st.markdown("---")
    


# =============================================================================
# DATA LOADING
# =============================================================================

df_full, available_days = anomaly_realtime_data(selected_asset)

# =============================================================================
# DAY SELECTION
# =============================================================================

st.markdown("---")
st.markdown("### 📅 Select Simulation Day")

# Initialize selected_day in session_state if not present
if "selected_day_persist" not in st.session_state:
    st.session_state.selected_day_persist = available_days[-1]

# Check if selected day is available for current asset
if st.session_state.selected_day_persist not in available_days:
    # If date not available, find closest date
    current_date = st.session_state.selected_day_persist
    closest_date = min(available_days, key=lambda x: abs((x - current_date).days))
    st.session_state.selected_day_persist = closest_date
    st.warning(f"Selected date not available for this asset. Using closest date: {closest_date}")

def on_day_change() -> None:
    """Callback when day selection changes"""
    st.session_state.selected_day_persist = st.session_state.day_selector

selected_day = st.selectbox(
    "Choose a day to simulate",
    options=available_days,
    index=available_days.index(st.session_state.selected_day_persist),
    format_func=lambda x: x.strftime("%Y-%m-%d (%A)"),
    key="day_selector",
    on_change=on_day_change
)

df_day = filter_day_data(df_full, selected_day)

if len(df_day) == 0:
    st.warning("No data available for selected day.")
    st.stop()

if len(df_day) < window_size:
    st.error(f"Not enough data points. Day has {len(df_day)} points but window requires {window_size}.")
    st.stop()

# Get data arrays
close_col = config.COLUMNS["close"]
volume_col = config.COLUMNS["volume"]
prices = df_day[close_col].values
volumes = df_day[volume_col].values
timestamps = df_day.index


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

sim_key = f"{selected_asset}_{selected_day}"

if "sim_key" not in st.session_state or st.session_state.sim_key != sim_key:
    st.session_state.sim_key = sim_key
    st.session_state.sim_running = False
    st.session_state.sim_paused = False
    st.session_state.current_idx = 0
    st.session_state.anomalies = []
    st.session_state.sim_complete = False


# =============================================================================
# GEMINI CONTEXT AND SIDEBAR
# =============================================================================

# Calculate current simulation metrics for Gemini context
current_idx = st.session_state.current_idx
anomalies = st.session_state.anomalies
progress_pct = (current_idx / len(df_day)) * 100 if len(df_day) > 0 else 0
current_price = prices[current_idx - 1] if current_idx > 0 else None

# Calculate current Z-score if enough data
current_zscore = None
if current_idx > window_size:
    window_data = np.asarray(prices[current_idx - window_size:current_idx])
    mean = window_data.mean()
    std = window_data.std()
    if std > 0:
        current_zscore = (prices[current_idx - 1] - mean) / std

# Format anomaly list for context (last 10 anomalies)
anomaly_list = []
for a in anomalies[-10:]:
    anomaly_list.append({
        "time": str(a["timestamp"])[11:19],  # HH:MM:SS format
        "price": f"${a['price']:.2f}",
        "zscore": f"{a['zscore']:.2f}σ"
    })

# Optional window statistics
window_stats = None
if current_idx > window_size:
    window_data = np.asarray(prices[current_idx - window_size:current_idx])
    window_stats = {
        "mean": f"${window_data.mean():.2f}",
        "std": f"${window_data.std():.4f}",
        "min": f"${window_data.min():.2f}",
        "max": f"${window_data.max():.2f}"
    }


# Build Gemini context with current simulation state
gemini_context = context_builder_factory(PageType.REALTIME)(
    asset=selected_asset,
    asset_display=get_asset_display_name(selected_asset), # type: ignore
    simulation_day=str(selected_day),
    window_size=window_size,
    zscore_threshold=zscore_threshold,
    progress_pct=progress_pct,
    points_streamed=current_idx,
    total_points=len(df_day),
    anomalies_found=len(anomalies),
    anomaly_list=anomaly_list,
    current_price=current_price,
    current_zscore=current_zscore,
    window_stats=window_stats
)

# Render Gemini sidebar with real-time context
with st.sidebar:
    render_chat(
        page_context=gemini_context,
        page_type=PageType.REALTIME
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Import constants
try:
    from config import DEFAULT_CHART_CONFIG
    chart_config = DEFAULT_CHART_CONFIG
except ImportError:
    chart_config = {'displayModeBar': False}


# =============================================================================
# CONTROL BUTTONS
# =============================================================================

t_col1, t_col2 = st.columns([0.92, 0.08])
with t_col1:
  st.markdown("#### Simulation Controls")

col1, col2, col3, col4 = st.columns(4)

with col1:
    start_btn = st.button(
        "▶️ Start", 
        width='stretch', 
        disabled=st.session_state.sim_running or st.session_state.sim_complete
    )

with col2:
    if st.session_state.sim_paused:
        resume_btn = st.button("▶️ Resume", width='stretch')
        pause_btn = False
    else:
        pause_btn = st.button(
            "⏸️ Pause", 
            width='stretch',
            disabled=not st.session_state.sim_running
        )
        resume_btn = False

with col3:
    reset_btn = st.button(
        "🔄 Reset", 
        width='stretch',
        disabled=st.session_state.current_idx == 0
    )

with col4:
    run_all_btn = st.button(
        "⏭️ Run All", 
        width='stretch', 
        disabled=st.session_state.sim_running or st.session_state.sim_complete
    )

# Handle button clicks
if reset_btn:
    st.session_state.current_idx = 0
    st.session_state.anomalies = []
    st.session_state.sim_running = False
    st.session_state.sim_paused = False
    st.session_state.sim_complete = False
    st.rerun()

if start_btn:
    st.session_state.sim_running = True
    st.session_state.sim_paused = False
    st.session_state.sim_complete = False
    if st.session_state.current_idx == 0:
        st.session_state.anomalies = []
    st.rerun()

if pause_btn:
    st.session_state.sim_paused = True
    st.session_state.sim_running = False
    st.rerun()

if resume_btn:
    st.session_state.sim_paused = False
    st.session_state.sim_running = True
    st.rerun()

if run_all_btn:
    st.session_state.anomalies = calculate_anomalies_batch(
        prices, window_size, zscore_threshold, timestamps # type: ignore
    )
    st.session_state.current_idx = len(df_day)
    st.session_state.sim_complete = True
    st.session_state.sim_running = False
    st.session_state.sim_paused = False
    st.rerun()


# =============================================================================
# PROCESS BATCH IF RUNNING
# =============================================================================

if st.session_state.sim_running and not st.session_state.sim_paused:
    current_idx = st.session_state.current_idx
    
    if current_idx < len(df_day):
        # Process a batch of points
        batch_size = sim_speed
        new_idx, new_anomalies = process_batch(
            current_idx, 
            batch_size, 
            prices, # type: ignore
            window_size, 
            zscore_threshold, 
            timestamps, # type: ignore
            st.session_state.anomalies
        )
        
        st.session_state.current_idx = new_idx
        st.session_state.anomalies = new_anomalies
        
        # Check if complete
        if new_idx >= len(df_day):
            st.session_state.sim_complete = True
            st.session_state.sim_running = False
    else:
        st.session_state.sim_complete = True
        st.session_state.sim_running = False


# =============================================================================
# DISPLAY CHART
# =============================================================================

current_idx = st.session_state.current_idx
anomalies = st.session_state.anomalies

# Create chart
fig_combined = create_combined_chart(current_idx, anomalies, timestamps, prices, volumes, window_size, zscore_threshold) # type: ignore

# Update session state se chart è già stato aggiunto
# (necessario perché il chart si aggiorna dinamicamente)
if "gemini_selected_charts" in st.session_state:
    if "realtime_main" in st.session_state.gemini_selected_charts:
        st.session_state.gemini_selected_charts["realtime_main"]["figure"] = fig_combined

# Chart with add button
with t_col2:
    render_chart_add_button(
        chart_id="realtime_main",
        figure=fig_combined,
        label=f"Real-time Simulation - {get_asset_display_name(selected_asset)}", # type: ignore
        page="realtime",
        position="inline",
        disabled=not st.session_state.sim_complete
    )

st.plotly_chart(
    fig_combined, 
    width='stretch', 
    config=chart_config
)

# =============================================================================
# METRICS ROW
# =============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

progress = (current_idx / len(df_day)) * 100 if len(df_day) > 0 else 0
current_price = prices[current_idx - 1] if current_idx > 0 else 0

with col1:
    st.metric("Progress", f"{progress:.1f}%")
with col2:
    st.metric("Points Streamed", f"{current_idx}/{len(df_day)}")
with col3:
    st.metric("Anomalies Found", len(anomalies))
with col4:
    st.metric("Current Price", f"${current_price:.2f}" if current_price > 0 else "-")
with col5:
    if current_idx > window_size:
        window_data = np.asarray(prices[current_idx - window_size:current_idx])
        mean = window_data.mean()
        std = window_data.std()
        if std > 0:
            current_z = (prices[current_idx - 1] - mean) / std
            st.metric("Current Z-Score", f"{current_z:.2f}σ")
        else:
            st.metric("Current Z-Score", "-")
    else:
        remaining = window_size - current_idx if current_idx > 0 else window_size
        st.metric("Current Z-Score", f"Need {remaining} pts" if current_idx > 0 else "-")


# =============================================================================
# ANOMALY LOG
# =============================================================================

st.markdown("---")
st.markdown("### Anomaly Log")

log_df = compute_anomaly_log(anomalies, current_idx, zscore_threshold)
if log_df is not None:
    st.dataframe(log_df, width='stretch', height=200)
elif st.session_state.sim_complete:
    st.toast("No anomalies detected during this simulation.", icon="ℹ️")


# =============================================================================
# AUTO-RERUN IF SIMULATION IS RUNNING
# =============================================================================

if st.session_state.sim_running and not st.session_state.sim_paused and not st.session_state.sim_complete:
    time.sleep(0.2)
    st.rerun()


# =============================================================================
# POST-SIMULATION ANALYSIS
# =============================================================================

if st.session_state.sim_complete:
    st.markdown("---")
    st.markdown("### Post-Simulation Analysis")
    st.toast("Simulation complete!", icon="✅")
    
    col1, col2, col3 = st.columns(3)
    total_anomalies = len(st.session_state.anomalies)
    anomaly_rate = (total_anomalies / len(df_day)) * 100
    
    with col1:
        st.metric("Total Points", len(df_day))
    with col2:
        st.metric("Total Anomalies", total_anomalies)
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    
    # Price statistics
    st.markdown("#### Price Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Start Price", f"${prices[0]:.2f}")
    with col2:
        st.metric("End Price", f"${prices[-1]:.2f}")
    with col3:
        prices_arr = np.asarray(prices)
        st.metric("High", f"${prices_arr.max():.2f}")
    with col4:
        st.metric("Low", f"${prices_arr.min():.2f}")


# =============================================================================
# FOOTER
# =============================================================================

footer("Real-time Anomaly Detection")
