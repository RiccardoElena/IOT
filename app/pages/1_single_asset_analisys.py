"""
Single Asset Analysis Page

This page provides comprehensive analysis of a single asset including:
- Interactive candlestick chart with zoom-to-anomaly feature
- Volume chart
- Z-score visualization with thresholds
- Anomaly detection and highlighting
- Detailed anomaly table
- For minute data: simple week selectbox for easy navigation
"""

import os
import sys
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

from components import (
    footer,
    title,
)

from src.anomaly_detection import (
    count_anomalies,
    detect_anomalies,
    get_anomaly_table,
    get_threshold_lines,
)
from src.data_loader import (
    filter_by_date_range,
    get_asset_display_name,
    get_date_range_fast,
    get_granularity_display_name,
    load_single_asset,
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=f"Single Asset Analysis | {config.PAGE_TITLE}",
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT
)

title("Single Asset Analysis", "Explore individual asset data with anomaly detection.")


def reset_zoom():
  st.session_state.selected_zoom_range = None

  st.session_state.anomaly_selector = None

# =============================================================================
# SIDEBAR - CONTROLS AND INFO
# =============================================================================

with st.sidebar:
    st.header("Controls")
    
    # Asset selection
    asset_options = {key: get_asset_display_name(key) for key in config.ASSETS.keys()}
    selected_asset = st.selectbox(
        "Select Asset",
        options=list(asset_options.keys()),
        on_change=reset_zoom,
        format_func=lambda x: asset_options[x]
    )
    
    # Granularity selection (minute, hourly, daily)
    granularity_options = {
        key: get_granularity_display_name(key) 
        for key in config.GRANULARITY_PATHS.keys()
    }
    selected_granularity = st.selectbox(
        "Select Granularity",
        options=list(granularity_options.keys()),
        on_change=reset_zoom,
        format_func=lambda x: granularity_options[x],
        index=2  # Default to daily
    )
    
    st.markdown("---")
    
    # Z-score threshold slider
    zscore_threshold = st.slider(
        "Z-Score Threshold",
        min_value=1.0,
        max_value=5.0,
        value=config.ZSCORE_ANOMALY_THRESHOLD,
        step=0.5,
        help="Values beyond this threshold are classified as anomalies"
    )
    
    # Show anomalies toggle
    show_anomalies = st.checkbox("Highlight Anomalies", value=True)
    
    st.markdown("---")
    
    # INFO BOX: About the Asset
    with st.expander(" Asset Information", expanded=False):
        asset_info = {
            "sp500": """
            **S&P 500 Index**
            
            The Standard & Poor's 500 is a stock market index tracking 
            500 large companies listed on US stock exchanges.
            
            **Key characteristics:**
            - Benchmark for US equity market
            - Market-cap weighted
            - Represents ~80% of US market cap
            - Sectors: Tech, Healthcare, Finance, etc.
            
            **Trading hours:** 9:30 AM - 4:00 PM ET (weekdays)
            
            **What moves it:**
            - Corporate earnings
            - Economic indicators (GDP, employment)
            - Federal Reserve policy
            - Geopolitical events
            """,
            "gold": """
            **Gold (XAU)**
            
            Gold is a precious metal traded globally as a commodity 
            and store of value.
            
            **Key characteristics:**
            - Safe-haven asset
            - Hedge against inflation
            - Negatively correlated with USD
            - Central bank reserves
            
            **Trading:** 24 hours, 5 days/week (global markets)
            
            **What moves it:**
            - Interest rates (inverse relationship)
            - US Dollar strength
            - Inflation expectations
            - Geopolitical uncertainty
            - Central bank buying/selling
            """,
            "oil": """
            **Crude Oil (WTI)**
            
            West Texas Intermediate is a grade of crude oil used 
            as a benchmark in oil pricing.
            
            **Key characteristics:**
            - Primary energy commodity
            - Highly volatile
            - Influenced by OPEC decisions
            - Economic growth indicator
            
            **Trading:** Nearly 24 hours (NYMEX, ICE)
            
            **What moves it:**
            - Supply/demand balance
            - OPEC+ production decisions
            - Geopolitical tensions (Middle East)
            - Inventory reports (EIA weekly)
            - Economic growth outlook
            """,
            "usd": """
            **US Dollar Index (DXY)**
            
            Measures the value of USD against a basket of 6 major 
            currencies (EUR, JPY, GBP, CAD, SEK, CHF).
            
            **Key characteristics:**
            - Global reserve currency
            - Safe-haven in crises
            - Euro has largest weight (~58%)
            
            **Trading:** 24 hours, 5 days/week (forex markets)
            
            **What moves it:**
            - Federal Reserve policy
            - Interest rate differentials
            - US economic data
            - Risk sentiment (flight to safety)
            - Trade balance
            """,
            "btc": """
            **Bitcoin (BTC)**
            
            The first and largest cryptocurrency by market cap.
            
            **Key characteristics:**
            - Decentralized digital currency
            - Fixed supply (21 million max)
            - Highly volatile
            - 24/7 trading
            
            **Trading:** 24/7/365 (crypto exchanges)
            
            **What moves it:**
            - Regulatory news
            - Institutional adoption
            - Halving events (every ~4 years)
            - Macro risk sentiment
            - Technical factors (on-chain metrics)
            """
        }
        st.markdown(asset_info.get(selected_asset, "Information not available."))
    
    # INFO BOX: About Z-Score
    with st.expander(" Understanding Z-Score", expanded=False):
        st.markdown(f"""
        **What is Z-Score?**
        
        Z-Score measures how many standard deviations a value is 
        from the mean (average).
        
        **Formula:**
        ```
        Z = (x - μ) / σ
        ```
        Where:
        - x = observed value
        - μ (mu) = mean
        - σ (sigma) = standard deviation
        
        ---
        
        **Interpretation:**
        
        | Z-Score | Meaning | Probability |
        |---------|---------|-------------|
        | 0 | At the mean | 50% |
        | ±1σ | Slightly unusual | ~32% outside |
        | ±2σ | Unusual | ~5% outside |
        | ±3σ | Very unusual | ~0.3% outside |
        | ±4σ | Extremely rare | ~0.006% outside |
        
        ---
        
        **Current threshold: ±{zscore_threshold}σ**
        
        Values beyond this threshold are marked as anomalies.
        
        ---
        
        **Why use Z-Score for anomalies?**
        
        1. **Scale-independent:** Works for any price level
        2. **Statistically grounded:** Based on probability
        3. **Adaptable:** Threshold can be tuned
        4. **Interpretable:** Easy to understand severity
        """)
    
    # INFO BOX: About Candlesticks
    with st.expander("Reading Candlestick Charts", expanded=False):
        st.markdown("""
        **Candlestick Anatomy:**
        
        ```
            │  ← Upper shadow (wick)
           ─┴─
           │ │ ← Body
           │ │
           ─┬─
            │  ← Lower shadow (wick)
        ```
        
        **Components:**
        - **Body:** Distance between Open and Close
        - **Upper shadow:** High minus the higher of Open/Close
        - **Lower shadow:** Lower of Open/Close minus Low
        
        ---
        
        **Colors:**
        - 🟢 **Green/Bullish:** Close > Open (price went up)
        - 🔴 **Red/Bearish:** Close < Open (price went down)
        
        ---
        
        **What they show:**
        
        | Part | Information |
        |------|-------------|
        | Body size | Strength of move |
        | Shadow length | Price rejection |
        | Color | Direction |
        
        ---
        
        **Quick patterns:**
        - **Long body:** Strong conviction
        - **Long upper shadow:** Sellers pushed back
        - **Long lower shadow:** Buyers stepped in
        - **Small body:** Indecision
        """)
    
    # INFO BOX: About Volume
    with st.expander(" Understanding Volume", expanded=False):
        st.markdown("""
        **What is Volume?**
        
        Volume represents the number of shares/contracts/units 
        traded during a time period.
        
        ---
        
        **Why Volume Matters:**
        
        Volume confirms price moves:
        
        | Price | Volume | Interpretation |
        |-------|--------|----------------|
        | ↑ Up | High | Strong bullish move |
        | ↑ Up | Low | Weak rally, may reverse |
        | ↓ Down | High | Strong selling pressure |
        | ↓ Down | Low | Weak decline, may bounce |
        
        ---
        
        **Volume anomalies indicate:**
        
        - **Spike in volume:** Something significant happened
          - News event
          - Large institutional trade
          - Options expiration
          - Market manipulation
        
        - **Very low volume:** 
          - Holiday trading
          - Pre-market/after-hours
          - Low interest in asset
        
        ---
        
        **Volume + Price analysis:**
        
        The best signals come when volume confirms price:
        - Breakout + high volume = likely to continue
        - Breakout + low volume = likely false breakout
        """)
    
    # INFO BOX: About Volatility
    with st.expander(" Understanding Volatility", expanded=False):
        st.markdown("""
        **What is Volatility?**
        
        Volatility measures how much prices fluctuate over time.
        In this analysis, we use the **daily range** (High - Low) 
        as a proxy for volatility.
        
        ---
        
        **Types of Volatility:**
        
        | Type | Description |
        |------|-------------|
        | Historical | Past price fluctuations |
        | Implied | Market's expectation (from options) |
        | Realized | Actual observed volatility |
        
        ---
        
        **High volatility indicates:**
        
        - Uncertainty in the market
        - Potential trading opportunities
        - Higher risk
        - News or events driving price
        
        **Low volatility indicates:**
        
        - Market consensus
        - Potential calm before storm
        - Lower trading opportunities
        - Possible breakout coming
        
        ---
        
        **Volatility clustering:**
        
        High volatility tends to follow high volatility, 
        and low follows low. This is why:
        - Risk management is crucial during volatile periods
        - Position sizing should adapt to volatility
        """)
    
    # INFO BOX: Data Granularity
    with st.expander("⏱️ Data Granularity", expanded=False):
        st.markdown("""
        **What is granularity?**
        
        Granularity refers to the time interval between 
        consecutive data points.
        
        ---
        
        **Available granularities:**
        
        | Level | Interval | Use Case |
        |-------|----------|----------|
        | **Minute** | 1 min | Intraday trading, scalping |
        | **Hourly** | 1 hour | Swing trading, day trading |
        | **Daily** | 1 day | Position trading, investing |
        
        ---
        
        **Trade-offs:**
        
        **Higher frequency (minute):**
        - ✅ More detail
        - ✅ Better for short-term
        - ❌ More noise
        - ❌ More data to process
        
        **Lower frequency (daily):**
        - ✅ Clearer trends
        - ✅ Less noise
        - ❌ Miss intraday moves
        - ❌ Delayed signals
        
        ---
        
        **For anomaly detection:**
        
        - **Minute:** Detects flash crashes, HFT anomalies
        - **Hourly:** Detects intraday unusual moves
        - **Daily:** Detects significant daily moves
        """)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data_full(asset: str, granularity: str):
    """Load full dataset."""
    return load_single_asset(asset, granularity)


@st.cache_data
def get_date_bounds(asset: str, granularity: str):
    """Get min/max dates for a dataset without loading all data."""
    return get_date_range_fast(asset, granularity)


@st.cache_data
def process_anomalies(df: pd.DataFrame, threshold: float):
    """Run anomaly detection on data."""
    return detect_anomalies(df, zscore_threshold=threshold, mode="batch")


@st.cache_data
def get_available_weeks(asset: str, granularity: str):
    """Get list of available weeks for minute data."""
    df = load_single_asset(asset, granularity)
    
    # Get unique dates
    dates = pd.Series(df.index.date).unique()
    min_date = min(dates)
    max_date = max(dates)
    
    # Generate week ranges
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


# Get date range for the selected asset and granularity
try:
    min_date_ts, max_date_ts = get_date_bounds(selected_asset, selected_granularity)
    min_date = min_date_ts.date()
    max_date = max_date_ts.date()
except FileNotFoundError as e:
    st.error(f"""
    **Data file not found!**
    
    Please ensure the CSV file exists:
    `data/{selected_granularity}/{config.FILE_NAMES[selected_asset]}`
    
    Error: {e}
    """)
    st.stop()
except Exception as e:
    st.error(f"Error reading data: {e}")
    st.stop()


# =============================================================================
# DATE RANGE SELECTION
# =============================================================================

st.markdown("### 📅 Date Range")

if selected_granularity == "minute":
    # Simple week selectbox for minute data
    st.toast("Minute data is limited to **one week at a time** for performance.", icon="⚠️")
    
    # Get available weeks
    weeks = get_available_weeks(selected_asset, selected_granularity)
    
    # Create selectbox
    selected_week = st.selectbox(
        "Select Week",
        options=range(len(weeks)),
        format_func=lambda i: weeks[i]["label"],
        index=len(weeks) - 1  # Default to most recent week
    )
    
    start_date = weeks[selected_week]["start"]
    end_date = weeks[selected_week]["end"]

else:
    # Standard date selection for hourly/daily
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

# Validate date range
if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()


# =============================================================================
# LOAD AND PROCESS DATA
# =============================================================================

try:
    with st.spinner("Loading data..."):
        df_full = load_data_full(selected_asset, selected_granularity)
        df = filter_by_date_range(df_full, str(start_date), str(end_date))
    
    if len(df) == 0:
        st.warning("No data available for selected date range.")
        st.stop()
    
    # Run anomaly detection
    with st.spinner("Detecting anomalies..."):
        df_processed = process_anomalies(df.copy(), zscore_threshold)

except Exception as e:
    st.error(f"Error processing data: {e}")
    st.stop()

# =============================================================================
# ANOMALY NAVIGATION (Jump to anomaly feature)
# =============================================================================

# Get anomaly table for navigation
anomaly_df = get_anomaly_table(df_processed)

# Store selected date range for zoom
if "selected_zoom_range" not in st.session_state:
    st.session_state.selected_zoom_range = None

# =============================================================================
# MAIN CHARTS
# =============================================================================

st.markdown("---")
st.markdown("###  Price, Volume & Volatility")

# Get column names
open_col = config.COLUMNS["open"]
high_col = config.COLUMNS["high"]
low_col = config.COLUMNS["low"]
close_col = config.COLUMNS["close"]
volume_col = config.COLUMNS["volume"]

# Create subplot figure with candlestick, volume, and volatility
fig_main = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=("Price (OHLC)", "Volume", "Volatility")
)

# Candlestick chart
fig_main.add_trace(
    go.Candlestick(
        x=df_processed.index,
        open=df_processed[open_col],
        high=df_processed[high_col],
        low=df_processed[low_col],
        close=df_processed[close_col],
        name="OHLC",
        increasing_line_color=config.COLOR_BULLISH,
        decreasing_line_color=config.COLOR_BEARISH
    ),
    row=1, col=1
)

# Add anomaly markers on price chart
if show_anomalies:
    anomaly_mask = df_processed["anomaly_price"]
    if anomaly_mask.any():
        anomaly_data = df_processed[anomaly_mask]
        fig_main.add_trace(
            go.Scatter(
                x=anomaly_data.index,
                y=anomaly_data[high_col] * 1.01,
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=config.MARKER_SIZE_ANOMALY,
                    color=config.COLOR_ANOMALY
                ),
                name="Price Anomaly",
                hovertemplate=(
                    "<b>⚠️ ANOMALY</b><br>"
                    "Time: %{x}<br>"
                    "Price: $%{customdata[0]:.2f}<br>"
                    "Z-Score: %{customdata[1]:.2f}σ<br>"
                    "<extra></extra>"
                ),
                customdata=anomaly_data[[close_col, "zscore_close"]].values
            ),
            row=1, col=1
        )

# Volume bar chart
volume_colors = [
    config.COLOR_ANOMALY if a else config.COLOR_NORMAL 
    for a in df_processed["anomaly_volume"]
] if show_anomalies else config.COLOR_NORMAL

fig_main.add_trace(
    go.Bar(
        x=df_processed.index,
        y=df_processed[volume_col],
        name="Volume",
        marker_color=volume_colors,
        hovertemplate=(
            "Time: %{x}<br>"
            "Volume: %{y:,.0f}<br>"
            "<extra></extra>"
        )
    ),
    row=2, col=1
)

# Volatility chart (High-Low range) - LINE CHART
df_processed["volatility_range"] = df_processed[high_col] - df_processed[low_col]

# Separate normal and anomaly data for different styling
anomaly_mask = df_processed["anomaly_volatility"] if show_anomalies else pd.Series([False] * len(df_processed))

# Normal volatility line
fig_main.add_trace(
    go.Scatter(
        x=df_processed.index,
        y=df_processed["volatility_range"],
        mode="lines",
        name="Volatility",
        line=dict(color=config.COLOR_NORMAL, width=2),
        fill='tozeroy',
        fillcolor='rgba(100, 149, 237, 0.2)',
        hovertemplate=(
            "Time: %{x}<br>"
            "Range: $%{y:.2f}<br>"
            "<extra></extra>"
        )
    ),
    row=3, col=1
)

# Anomaly points
if show_anomalies and anomaly_mask.any():
    fig_main.add_trace(
        go.Scatter(
            x=df_processed[anomaly_mask].index,
            y=df_processed[anomaly_mask]["volatility_range"],
            mode="markers",
            name="Volatility Anomaly",
            marker=dict(
                size=config.MARKER_SIZE_ANOMALY,
                color=config.COLOR_ANOMALY,
                symbol="diamond"
            ),
            hovertemplate=(
                "<b>⚠️ VOLATILITY ANOMALY</b><br>"
                "Time: %{x}<br>"
                "Range: $%{y:.2f}<br>"
                "<extra></extra>"
            )
        ),
        row=3, col=1
    )

# Apply zoom if anomaly selected
if st.session_state.selected_zoom_range is not None:
    fig_main.update_xaxes(
        range=[st.session_state.selected_zoom_range["start"], st.session_state.selected_zoom_range["end"]]
    )

# Update layout
fig_main.update_layout(
    height=750,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    xaxis_rangeslider_visible=False,
    hovermode="x unified"
)

fig_main.update_yaxes(title_text="Price", row=1, col=1)
fig_main.update_yaxes(title_text="Volume", row=2, col=1)
fig_main.update_yaxes(title_text="Range ($)", row=3, col=1)
fig_main.update_xaxes(title_text="Date", row=3, col=1)

st.plotly_chart(fig_main, width='stretch')

if len(anomaly_df) > 0:
    col1, col2 = st.columns([4, 1])

    
    with col1:
      st.markdown("#### Jump to Anomaly")  
    with col2:
        if st.button("🔄 Reset Zoom", width='stretch'):
            st.session_state.selected_zoom_range = None
            if "anomaly_selector" in st.session_state:
                st.session_state.anomaly_selector = None
            st.rerun()

    # Create options for selectbox
    anomaly_options = [
        f"#{i}: {row['type']} - {str(row['timestamp'])[:19]} (Z={row['zscore']:.2f})"
        for i, row in anomaly_df.iterrows()
    ]

    def on_anomaly_select():
        """Callback when anomaly is selected"""
        selected = st.session_state.anomaly_selector
        
        if selected != "Select an anomaly...":
            # Extract index from selection (format: "#1: ...")
            anomaly_idx = int(selected.split(":")[0].replace("#", ""))
            anomaly_row = anomaly_df.loc[anomaly_idx]
            anomaly_timestamp = anomaly_row["timestamp"]
            
            # Calculate zoom range (show +/- 5% of data around anomaly)
            total_range = (df_processed.index.max() - df_processed.index.min())
            zoom_delta = total_range * 0.05
            
            st.session_state.selected_zoom_range = {
                "start": anomaly_timestamp - zoom_delta,
                "end": anomaly_timestamp + zoom_delta
            }
    
    selected_anomaly = st.selectbox(
        "Navigate to anomaly",
        options=anomaly_options,
        help="Select an anomaly to zoom the chart to that time period",
        key="anomaly_selector",
        placeholder="Select an anomaly...",
        on_change=on_anomaly_select
    )


# =============================================================================
# Z-SCORE CHARTS
# =============================================================================

st.markdown("---")
st.markdown("### Z-Score Analysis")

# Get threshold lines
thresholds = get_threshold_lines(zscore_threshold)


def create_zscore_chart(
    data: pd.Series, 
    title: str, 
    anomaly_mask: pd.Series
) -> go.Figure:
    """Create a Z-score chart with threshold lines."""
    fig = go.Figure()
    
    # Main Z-score line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.values,
            mode="lines",
            name="Z-Score",
            line=dict(color=config.COLOR_NORMAL, width=1),
            hovertemplate="Time: %{x}<br>Z-Score: %{y:.2f}σ<extra></extra>"
        )
    )
    
    # Anomaly points
    if show_anomalies and anomaly_mask.any():
        anomaly_data = data[anomaly_mask]
        fig.add_trace(
            go.Scatter(
                x=anomaly_data.index,
                y=anomaly_data.values,
                mode="markers",
                name="Anomaly",
                marker=dict(
                    size=config.MARKER_SIZE_ANOMALY,
                    color=config.COLOR_ANOMALY
                ),
                hovertemplate=(
                    "<b>⚠️ ANOMALY</b><br>"
                    "Time: %{x}<br>"
                    "Z-Score: %{y:.2f}σ<br>"
                    "<extra></extra>"
                )
            )
        )
    
    # Threshold lines
    fig.add_hline(
        y=thresholds["anomaly_upper"], 
        line_dash="dash", 
        line_color=config.COLOR_ANOMALY, 
        annotation_text=f"+{zscore_threshold}σ"
    )
    fig.add_hline(
        y=thresholds["anomaly_lower"], 
        line_dash="dash", 
        line_color=config.COLOR_ANOMALY, 
        annotation_text=f"-{zscore_threshold}σ"
    )
    fig.add_hline(
        y=thresholds["warning_upper"], 
        line_dash="dot", 
        line_color=config.COLOR_WARNING, 
        annotation_text=f"+{config.ZSCORE_WARNING_THRESHOLD}σ"
    )
    fig.add_hline(
        y=thresholds["warning_lower"], 
        line_dash="dot", 
        line_color=config.COLOR_WARNING, 
        annotation_text=f"-{config.ZSCORE_WARNING_THRESHOLD}σ"
    )
    fig.add_hline(y=0, line_color="gray", line_width=0.5)
    
    # Add colored regions
    fig.add_hrect(
        y0=thresholds["warning_lower"], 
        y1=thresholds["warning_upper"], 
        fillcolor="green", 
        opacity=0.1, 
        line_width=0
    )
    fig.add_hrect(
        y0=thresholds["warning_upper"], 
        y1=thresholds["anomaly_upper"], 
        fillcolor="yellow", 
        opacity=0.1, 
        line_width=0
    )
    fig.add_hrect(
        y0=thresholds["anomaly_lower"], 
        y1=thresholds["warning_lower"], 
        fillcolor="yellow", 
        opacity=0.1, 
        line_width=0
    )
    
    # Apply zoom if anomaly selected
    if st.session_state.selected_zoom_range is not None:
        fig.update_xaxes(
            range=[st.session_state.selected_zoom_range["start"], st.session_state.selected_zoom_range["end"]]
        )
    
    fig.update_layout(
        title=title,
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Z-Score")
    
    return fig


# Create tabs for different Z-score charts
tab1, tab2, tab3 = st.tabs(["Price Z-Score", "Volume Z-Score", "Volatility Z-Score"])

with tab1:
    fig_zscore_price = create_zscore_chart(
        df_processed["zscore_close"],
        "Price Z-Score",
        df_processed["anomaly_price"]
    )
    st.plotly_chart(fig_zscore_price, width='stretch')

with tab2:
    fig_zscore_volume = create_zscore_chart(
        df_processed["zscore_volume"],
        "Volume Z-Score",
        df_processed["anomaly_volume"]
    )
    st.plotly_chart(fig_zscore_volume, width='stretch')

with tab3:
    fig_zscore_volatility = create_zscore_chart(
        df_processed["zscore_volatility"],
        "Volatility Z-Score",
        df_processed["anomaly_volatility"]
    )
    st.plotly_chart(fig_zscore_volatility, width='stretch')

# =============================================================================
# SUMMARY METRICS
# =============================================================================

st.markdown("---")
st.markdown("###  Summary")

# Count anomalies
anomaly_counts = count_anomalies(df_processed)

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(df_processed):,}")
with col2:
    st.metric("Price Anomalies", anomaly_counts["price"])
with col3:
    st.metric("Volume Anomalies", anomaly_counts["volume"])
with col4:
    st.metric("Volatility Anomalies", anomaly_counts["volatility"])

# =============================================================================
# ANOMALY TABLE
# =============================================================================

st.markdown("---")
st.markdown("### Anomaly Details")

if len(anomaly_df) > 0:
    # Format the table for display
    anomaly_df_display = anomaly_df.copy()
    anomaly_df_display["timestamp"] = anomaly_df_display["timestamp"].astype(str)
    anomaly_df_display["value"] = anomaly_df_display["value"].apply(
        lambda x: f"{x:,.2f}" if pd.notna(x) else "-"
    )
    anomaly_df_display["zscore"] = anomaly_df_display["zscore"].apply(
        lambda x: f"{x:.2f}σ" if pd.notna(x) else "-"
    )
    anomaly_df_display["pct_change"] = anomaly_df_display["pct_change"].apply(
        lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"
    )
    
    # Rename columns for display
    anomaly_df_display.columns = ["Timestamp", "Type", "Value", "Z-Score", "% Change"]
    
    # Add filtering
    anomaly_types = ["All"] + list(anomaly_df["type"].unique())
    selected_type = st.selectbox("Filter by Type", anomaly_types)
    
    if selected_type != "All":
        anomaly_df_display = anomaly_df_display[anomaly_df_display["Type"] == selected_type]
    
    # Display table with 1-based index
    st.dataframe(anomaly_df_display, width='stretch', height=400)
    
    # Download button
    csv = anomaly_df.to_csv(index=True, index_label="ID")
    st.download_button(
        label="📥 Download Anomalies CSV",
        data=csv,
        file_name=f"anomalies_{selected_asset}_{selected_granularity}.csv",
        mime="text/csv"
    )
    
    # Anomaly detail expander
    with st.expander("ℹ️ Understanding the Anomalies"):
        st.markdown(f"""
        **What makes these values anomalies?**
        
        Each anomaly has a Z-Score with absolute value ≥ {zscore_threshold}, meaning 
        the value is at least {zscore_threshold} standard deviations from the mean.
        
        **Probability interpretation:**
        - Z=3σ: ~0.3% chance (1 in 370)
        - Z=4σ: ~0.006% chance (1 in 15,787)
        - Z=5σ: ~0.00006% chance (1 in 1.7 million)
        
        **Anomaly types:**
        - **Price**: Unusual price movement
        - **Volume**: Unusual trading activity
        - **Volatility**: Unusual price range (high-low)
        
        **% Change column:**
        - Shows percentage change from previous period
        - Only displayed for Price anomalies
        - Positive = price increased, Negative = price decreased
        """)

else:
    st.info(f"""
    No anomalies detected with the current threshold (Z-Score ≥ {zscore_threshold}).
    
    Try lowering the threshold in the sidebar to detect more subtle anomalies.
    """)


footer("Single Asset Analysis")