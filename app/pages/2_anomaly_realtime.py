"""
Real-time Anomaly Detection Page (IoT Simulation)

Version 2: Uses st.rerun() based architecture for smooth updates and responsive controls.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import (
    get_asset_display_name,
    load_single_asset,
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=f"Real-time IoT Simulation | {config.PAGE_TITLE}",
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT
)

st.title("Real-time Anomaly Detection")
st.markdown("Simulating IoT data streaming with sliding window anomaly detection.")


# =============================================================================
# SIDEBAR - CONTROLS AND INFO
# =============================================================================

with st.sidebar:
    st.header("Controls")
    
    asset_options = {key: get_asset_display_name(key) for key in config.ASSETS.keys()}
    selected_asset = st.selectbox(
        "Select Asset",
        options=list(asset_options.keys()),
        format_func=lambda x: asset_options[x]
    )
    
    st.info("Using **Minute** data for real-time simulation")
    
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
        value=float(config.ZSCORE_ANOMALY_THRESHOLD),
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
    
    st.markdown("---")
    
    with st.expander("📡 What is IoT Streaming?", expanded=False):
        st.markdown("""
        **Internet of Things (IoT)** devices continuously generate 
        data streams in real-time.
        
        **Examples in finance:**
        - High-frequency trading data
        - Real-time price feeds
        - Transaction monitoring
        
        **This simulation** mimics how a real IoT system would:
        1. Receive one data point
        2. Update statistics incrementally
        3. Check for anomalies in real-time
        4. Alert if threshold exceeded
        """)
    
    with st.expander("Sliding Window Explained", expanded=False):
        st.markdown(f"""
        **What is a Sliding Window?**
        
        A fixed-size window that moves forward as new data arrives.
        
        ```
        Time:  1  2  3  4  5  6  7  8  9  10
        Data:  •  •  •  •  •  •  •  •  •  •
                  [----window----]
                     [----window----]
        ```
        
        **Current window size: {window_size} points**
        
        **Trade-offs:**
        - **Small window:** More sensitive, more false positives
        - **Large window:** More stable, may miss quick changes
        """)
    
    with st.expander("Rolling Z-Score", expanded=False):
        st.markdown(f"""
        **Rolling Z-Score:**
        ```
        Z = (x - μ_window) / σ_window
        ```
        
        Uses only the last {window_size} points for mean/std,
        adapting to recent market conditions.
        """)
    
    with st.expander("⚠️ Anomaly Severity Levels", expanded=False):
        st.markdown(f"""
        | Level | Condition |
        |-------|-----------|
        | 🟡 LOW | {zscore_threshold}σ ≤ |Z| < {zscore_threshold + 0.5}σ |
        | 🟠 MEDIUM | {zscore_threshold + 0.5}σ ≤ |Z| < {zscore_threshold + 1.0}σ |
        | 🔴 HIGH | |Z| ≥ {zscore_threshold + 1.0}σ |
        """)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_minute_data(asset: str):
    return load_single_asset(asset, "minute")


@st.cache_data
def get_available_days(df: pd.DataFrame):
    dates = pd.Series(df.index.date).unique()
    return sorted(dates)


try:
    with st.spinner("Loading minute data..."):
        df_full = load_minute_data(selected_asset)
        available_days = get_available_days(df_full)
except FileNotFoundError as e:
    st.error(f"Data file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# =============================================================================
# DAY SELECTION
# =============================================================================

st.markdown("---")
st.markdown("### 📅 Select Simulation Day")

selected_day = st.selectbox(
    "Choose a day to simulate",
    options=available_days,
    index=len(available_days) - 1,
    format_func=lambda x: x.strftime("%Y-%m-%d (%A)")
)

df_day = df_full[df_full.index.date == selected_day].copy()

if len(df_day) == 0:
    st.warning("No data available for selected day.")
    st.stop()

if len(df_day) < window_size:
    st.error(f"Not enough data points. Day has {len(df_day)} points but window requires {window_size}.")
    st.stop()

st.info(f"📊 **{len(df_day)}** minute data points available for {selected_day}")

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
# HELPER FUNCTIONS
# =============================================================================

def get_severity(zscore: float, threshold: float) -> str:
    abs_z = abs(zscore)
    if abs_z >= threshold + 1.0:
        return "🔴 HIGH"
    elif abs_z >= threshold + 0.5:
        return "🟠 MEDIUM"
    else:
        return "🟡 LOW"


def calculate_anomalies_batch(prices, window, threshold, timestamps):
    """Calculate all anomalies in batch mode (for Run All)."""
    anomalies = []
    for i in range(window, len(prices)):
        window_data = prices[i - window:i]
        current_price = prices[i]
        mean = np.mean(window_data)
        std = np.std(window_data)
        if std > 0:
            zscore = (current_price - mean) / std
            if abs(zscore) >= threshold:
                anomalies.append({
                    "idx": i,
                    "timestamp": timestamps[i],
                    "price": current_price,
                    "zscore": zscore
                })
    return anomalies


def process_batch(start_idx, batch_size, prices, window, threshold, timestamps, existing_anomalies):
    """Process a batch of points and return new anomalies."""
    new_anomalies = []
    end_idx = min(start_idx + batch_size, len(prices))
    
    for i in range(start_idx, end_idx):
        if i >= window:
            window_data = prices[i - window:i]
            current_price = prices[i]
            mean = np.mean(window_data)
            std = np.std(window_data)
            
            if std > 0:
                zscore = (current_price - mean) / std
                if abs(zscore) >= threshold:
                    new_anomalies.append({
                        "idx": i,
                        "timestamp": timestamps[i],
                        "price": current_price,
                        "zscore": zscore
                    })
    
    return end_idx, existing_anomalies + new_anomalies


def create_price_chart(current_idx, anomalies):
    """Create the price chart with anomaly markers."""
    fig = go.Figure()
    
    if current_idx > 0:
        display_timestamps = timestamps[:current_idx]
        display_prices = prices[:current_idx]
        
        fig.add_trace(
            go.Scatter(
                x=display_timestamps,
                y=display_prices,
                mode="lines",
                name="Price",
                line=dict(color=config.COLOR_NORMAL, width=2),
                hovertemplate="Time: %{x}<br>Price: $%{y:.2f}<extra></extra>"
            )
        )
        
        if current_idx > window_size:
            window_start_idx = current_idx - window_size
            fig.add_vrect(
                x0=timestamps[window_start_idx],
                x1=timestamps[current_idx - 1],
                fillcolor="rgba(100, 149, 237, 0.2)",
                line_width=2,
                line_color="rgba(100, 149, 237, 0.8)",
                annotation_text="Analysis Window",
                annotation_position="top left"
            )
        else:
            fig.add_vrect(
                x0=timestamps[0],
                x1=timestamps[current_idx - 1],
                fillcolor="rgba(255, 193, 7, 0.2)",
                line_width=2,
                line_color="rgba(255, 193, 7, 0.8)",
                annotation_text=f"Building Window ({current_idx}/{window_size})",
                annotation_position="top left"
            )
        
        visible_anomalies = [a for a in anomalies if a["idx"] < current_idx]
        if visible_anomalies:
            fig.add_trace(
                go.Scatter(
                    x=[a["timestamp"] for a in visible_anomalies],
                    y=[a["price"] for a in visible_anomalies],
                    mode="markers",
                    name="Anomaly",
                    marker=dict(size=15, color=config.COLOR_ANOMALY, symbol="x", line=dict(width=2)),
                    customdata=[a["zscore"] for a in visible_anomalies],
                    hovertemplate="<b>⚠️ ANOMALY</b><br>Time: %{x}<br>Price: $%{y:.2f}<br>Z-Score: %{customdata:.2f}σ<extra></extra>"
                )
            )
    else:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="Press ▶️ Start to begin streaming simulation",
            showarrow=False,
            font=dict(size=18, color="gray")
        )
    
    fig.update_layout(
        title="📈 Price (Streaming)",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=350,
        hovermode="x unified",
        showlegend=True
    )
    
    return fig


def create_zscore_chart(current_idx):
    """Create the rolling Z-score chart."""
    fig = go.Figure()
    
    if current_idx > window_size:
        zscores = []
        zscore_times = []
        
        for i in range(window_size, current_idx):
            window_data = prices[i - window_size:i]
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std > 0:
                zscores.append((prices[i] - mean) / std)
                zscore_times.append(timestamps[i])
        
        if zscores:
            fig.add_trace(
                go.Scatter(
                    x=zscore_times,
                    y=zscores,
                    mode="lines",
                    name="Z-Score",
                    line=dict(color=config.COLOR_NORMAL, width=2),
                    fill='tozeroy',
                    fillcolor='rgba(100, 149, 237, 0.1)'
                )
            )
            
            fig.add_hline(y=zscore_threshold, line_dash="dash", line_color=config.COLOR_ANOMALY, 
                         annotation_text=f"+{zscore_threshold}σ", annotation_position="right")
            fig.add_hline(y=-zscore_threshold, line_dash="dash", line_color=config.COLOR_ANOMALY,
                         annotation_text=f"-{zscore_threshold}σ", annotation_position="right")
            fig.add_hline(y=0, line_color="gray", line_width=1)
    else:
        remaining = window_size - current_idx
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"Z-Score available after {window_size} points ({remaining} more needed)" if current_idx > 0 else "Waiting for data...",
            showarrow=False,
            font=dict(size=14, color="gray")
        )
    
    fig.update_layout(
        title="📊 Rolling Z-Score",
        xaxis_title="Time",
        yaxis_title="Z-Score (σ)",
        height=300,
        hovermode="x unified",
        showlegend=True
    )
    
    return fig


def create_volume_chart(current_idx):
    """Create the volume chart."""
    fig = go.Figure()
    
    if current_idx > 0:
        display_timestamps = timestamps[:current_idx]
        display_volumes = volumes[:current_idx]
        
        fig.add_trace(
            go.Bar(
                x=display_timestamps,
                y=display_volumes,
                name="Volume",
                marker_color=config.COLOR_NORMAL,
                opacity=0.7,
                hovertemplate="Time: %{x}<br>Volume: %{y:,.0f}<extra></extra>"
            )
        )
    else:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="Waiting for data...",
            showarrow=False,
            font=dict(size=14, color="gray")
        )
    
    fig.update_layout(
        title="📊 Volume",
        xaxis_title="Time",
        yaxis_title="Volume",
        height=300,
        hovermode="x unified",
        showlegend=True
    )
    
    return fig


def render_anomaly_log(anomalies, current_idx):
    """Render the anomaly log as a dataframe."""
    visible_anomalies = [a for a in anomalies if a["idx"] < current_idx]
    
    if visible_anomalies:
        log_data = []
        for a in visible_anomalies:
            log_data.append({
                "Time": str(a["timestamp"])[11:19],
                "Price": f"${a['price']:.2f}",
                "Z-Score": f"{a['zscore']:.2f}σ",
                "Severity": get_severity(a["zscore"], zscore_threshold)
            })
        return pd.DataFrame(log_data)
    return None


# =============================================================================
# CONTROL BUTTONS
# =============================================================================

st.markdown("---")
st.markdown("### Simulation Controls")

col1, col2, col3, col4 = st.columns(4)

with col1:
    start_btn = st.button(
        "▶️ Start", 
        width='stretch', 
        disabled=st.session_state.sim_running or st.session_state.sim_complete
    )

with col2:
    if st.session_state.sim_paused:
        resume_btn = st.button("▶️ Resume", width='stretch')
        pause_btn = False
    else:
        pause_btn = st.button(
            "⏸️ Pause", 
            width='stretch',
            disabled=not st.session_state.sim_running
        )
        resume_btn = False

with col3:
    reset_btn = st.button("🔄 Reset", width='stretch')

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
        prices, window_size, zscore_threshold, timestamps
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
        batch_size = sim_speed  # Points per batch = speed setting
        new_idx, new_anomalies = process_batch(
            current_idx, 
            batch_size, 
            prices, 
            window_size, 
            zscore_threshold, 
            timestamps,
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
# DISPLAY METRICS
# =============================================================================

st.markdown("---")
st.markdown("### 📈 Real-time Charts")

current_idx = st.session_state.current_idx
anomalies = st.session_state.anomalies

# Metrics row
col1, col2, col3, col4, col5 = st.columns(5)

progress = (current_idx / len(df_day)) * 100 if len(df_day) > 0 else 0
current_price = prices[current_idx - 1] if current_idx > 0 else 0

with col1:
    st.metric("Progress", f"{progress:.1f}%")
with col2:
    st.metric("Points Streamed", f"{current_idx}")
with col3:
    st.metric("Anomalies Found", len(anomalies))
with col4:
    st.metric("Current Price", f"${current_price:.2f}" if current_price > 0 else "-")
with col5:
    if current_idx > window_size:
        window_data = prices[current_idx - window_size:current_idx]
        mean = np.mean(window_data)
        std = np.std(window_data)
        if std > 0:
            current_z = (prices[current_idx - 1] - mean) / std
            st.metric("Current Z-Score", f"{current_z:.2f}σ")
        else:
            st.metric("Current Z-Score", "-")
    else:
        remaining = window_size - current_idx if current_idx > 0 else window_size
        st.metric("Current Z-Score", f"Need {remaining} pts" if current_idx > 0 else "-")


# =============================================================================
# DISPLAY CHARTS
# =============================================================================

st.plotly_chart(create_price_chart(current_idx, anomalies), width='stretch')
st.plotly_chart(create_zscore_chart(current_idx), width='stretch')
st.plotly_chart(create_volume_chart(current_idx), width='stretch')


# =============================================================================
# ANOMALY LOG
# =============================================================================

st.markdown("---")
st.markdown("### 📋 Anomaly Log")

log_df = render_anomaly_log(anomalies, current_idx)
if log_df is not None:
    st.dataframe(log_df, width='stretch', height=200)
else:
    st.info("No anomalies detected yet. Start the simulation to begin streaming.")


# =============================================================================
# AUTO-RERUN IF SIMULATION IS RUNNING
# =============================================================================

if st.session_state.sim_running and not st.session_state.sim_paused and not st.session_state.sim_complete:
    # Small delay for smooth rendering (~5 FPS)
    time.sleep(0.2)
    st.rerun()


# =============================================================================
# POST-SIMULATION ANALYSIS
# =============================================================================

if st.session_state.sim_complete:
    st.markdown("---")
    st.markdown("### 🔍 Post-Simulation Analysis")
    st.success("✅ Simulation complete!")
    
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
        st.metric("High", f"${prices.max():.2f}")
    with col4:
        st.metric("Low", f"${prices.min():.2f}")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Real-time IoT Simulation | IoT & Data Analytics Project
</div>
""", unsafe_allow_html=True)
