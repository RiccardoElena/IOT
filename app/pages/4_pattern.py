"""
Pattern Recognition Page

This page provides comprehensive pattern analysis including:
- Candlestick patterns (Doji, Hammer, Engulfing) with interactive markers
- Chart patterns (Double Top/Bottom, Head & Shoulders, Cup & Handle)
  displayed as colored regions with clickable legend
- Pattern distribution analysis and timeline visualization
- Calibration sliders for fine-tuning detection sensitivity

Gemini AI assistant provides contextual help on pattern interpretation.

Run with: streamlit run app.py (then navigate to this page)
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Import UI components including Gemini sidebar
from components import (
    footer,
    title,
    render_gemini_sidebar,
)

# Import data and pattern recognition modules
from src.data_loader import (
    filter_by_date_range,
    get_asset_display_name,
    get_date_range_fast,
    load_single_asset,
)
from src.pattern_recognition import (
    detect_all_candlestick_patterns,
    detect_all_chart_patterns,
    get_pattern_summary,
    get_pattern_table,
)

# Import Gemini context builder for this page
from src.gemini_assistant import build_pattern_context


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=f"Pattern Recognition | {config.PAGE_TITLE}",
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT
)

title("🔮 Pattern Recognition",
      "Identify candlestick and chart patterns that may indicate future price movements.")


# =============================================================================
# SIDEBAR - CONTROLS AND INFO
# =============================================================================

with st.sidebar:
    st.header(" Controls")
    
    asset_options = {key: get_asset_display_name(key) for key in config.ASSETS.keys()}
    selected_asset = st.selectbox(
        "Select Asset",
        options=list(asset_options.keys()),
        format_func=lambda x: asset_options[x]
    )
    
    st.info("Using **Daily** data for pattern recognition")
    
    st.markdown("---")
    st.subheader("🎛️ Pattern Calibration")
    
    tolerance = st.slider(
        "Price Tolerance (%)",
        min_value=2.0,
        max_value=15.0,
        value=8.0,
        step=1.0,
        help="How close peak/trough prices must be to match"
    )
    tolerance_decimal = tolerance / 100
    
    prominence = st.slider(
        "Peak Prominence (%)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
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
    
    st.markdown("---")
# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_daily_data(asset: str):
    return load_single_asset(asset, "daily")


@st.cache_data
def get_date_bounds(asset: str):
    return get_date_range_fast(asset, "daily")


try:
    with st.spinner("Loading data..."):
        df = load_daily_data(selected_asset)
        min_date_ts, max_date_ts = get_date_bounds(selected_asset)
        min_date = min_date_ts.date()
        max_date = max_date_ts.date()

except FileNotFoundError as e:
    st.error(f"Data file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# =============================================================================
# DATE RANGE SELECTION
# =============================================================================

st.markdown("---")
st.markdown("### 📅 Date Range")

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

df_filtered = filter_by_date_range(df, str(start_date), str(end_date))

if len(df_filtered) == 0:
    st.warning("No data available for selected date range.")
    st.stop()


# =============================================================================
# PATTERN DETECTION
# =============================================================================

with st.spinner("Detecting patterns..."):
    df_patterns = detect_all_candlestick_patterns(df_filtered)
    
    chart_patterns = detect_all_chart_patterns(
        df_filtered, 
        lookback=chart_lookback,
        tolerance=tolerance_decimal,
        prominence_pct=prominence_decimal
    )


# =============================================================================
# GEMINI CONTEXT AND SIDEBAR
# =============================================================================

# Get pattern summary for context
pattern_summary = get_pattern_summary(df_patterns)

# Format chart patterns for context (simplified list)
chart_pattern_list = []
for p in chart_patterns[:10]:  # Limit to 10 for context
    chart_pattern_list.append({
        "type": p["type"],
        "signal": p["signal"],
        "start": str(p["start_date"])[:10],
        "end": str(p["end_date"])[:10],
        "confidence": f"{p.get('confidence', 0.5) * 100:.0f}%"
    })

# Build Gemini context with pattern data
gemini_context = build_pattern_context(
    asset=selected_asset,
    asset_display=get_asset_display_name(selected_asset),
    start_date=str(start_date),
    end_date=str(end_date),
    candlestick_counts=pattern_summary,
    chart_patterns=chart_pattern_list,
    pattern_distribution={
        "total_candlestick": sum(pattern_summary.values()),
        "total_chart": len(chart_patterns),
        "bullish_signals": pattern_summary.get("hammer", 0) + pattern_summary.get("engulfing_bullish", 0),
        "bearish_signals": pattern_summary.get("engulfing_bearish", 0)
    }
)

# Render Gemini sidebar with pattern context
with st.sidebar:
    render_gemini_sidebar(
        page_context=gemini_context,
        page_type="patterns"
    )


# =============================================================================
# SUMMARY METRICS
# =============================================================================

st.markdown("---")
st.markdown("### Pattern Summary")


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Doji", pattern_summary["doji"])
with col2:
    st.metric("Hammer", pattern_summary["hammer"])
with col3:
    st.metric("Engulfing Bullish", pattern_summary["engulfing_bullish"])
with col4:
    st.metric("Engulfing Bearish", pattern_summary["engulfing_bearish"])
with col5:
    st.metric("Chart Patterns", len(chart_patterns))


# =============================================================================
# CANDLESTICK CHART WITH PATTERNS (markers)
# =============================================================================

st.markdown("---")
st.markdown("### 🕯️ Candlestick Patterns")
st.markdown("*Click on legend items to show/hide patterns*")

open_col = config.COLUMNS["open"]
high_col = config.COLUMNS["high"]
low_col = config.COLUMNS["low"]
close_col = config.COLUMNS["close"]

fig_candle = go.Figure()

fig_candle.add_trace(
    go.Candlestick(
        x=df_patterns.index,
        open=df_patterns[open_col],
        high=df_patterns[high_col],
        low=df_patterns[low_col],
        close=df_patterns[close_col],
        name="Price",
        increasing_line_color=config.COLOR_BULLISH,
        decreasing_line_color=config.COLOR_BEARISH
    )
)

pattern_colors = {
    "pattern_doji": "#9C27B0",
    "pattern_hammer": "#2196F3",
    "pattern_engulfing_bullish": "#4CAF50",
    "pattern_engulfing_bearish": "#F44336"
}

pattern_names = {
    "pattern_doji": "Doji",
    "pattern_hammer": "Hammer",
    "pattern_engulfing_bullish": "Engulfing Bullish",
    "pattern_engulfing_bearish": "Engulfing Bearish"
}

pattern_symbols = {
    "pattern_doji": "diamond",
    "pattern_hammer": "triangle-up",
    "pattern_engulfing_bullish": "star",
    "pattern_engulfing_bearish": "star"
}

for pattern_col, pattern_name in pattern_names.items():
    if pattern_col in df_patterns.columns:
        mask = df_patterns[pattern_col]
        if mask.any():
            pattern_data = df_patterns[mask]
            
            if "Bullish" in pattern_name or pattern_name == "Hammer":
                y_values = pattern_data[low_col] * 0.99
            else:
                y_values = pattern_data[high_col] * 1.01
            
            fig_candle.add_trace(
                go.Scatter(
                    x=pattern_data.index,
                    y=y_values,
                    mode="markers",
                    name=pattern_name,
                    marker=dict(
                        symbol=pattern_symbols[pattern_col],
                        size=12,
                        color=pattern_colors[pattern_col]
                    ),
                    hovertemplate=(
                        f"<b>{pattern_name}</b><br>"
                        "Date: %{x}<br>"
                        "Price: $%{customdata:.2f}<br>"
                        "<extra></extra>"
                    ),
                    customdata=pattern_data[close_col]
                )
            )

fig_candle.update_layout(
    height=500,
    title=f"Candlestick Patterns - {get_asset_display_name(selected_asset)}",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(
        orientation="h", 
        yanchor="bottom", 
        y=1.02,
        itemclick="toggle",
        itemdoubleclick="toggleothers"
    )
)

st.plotly_chart(fig_candle, width='stretch')

# Candlestick pattern table
st.markdown("#### Detected Candlestick Patterns")

pattern_table = get_pattern_table(df_patterns)

if len(pattern_table) > 0:
    pattern_table_display = pattern_table.copy()
    pattern_table_display["timestamp"] = pattern_table_display["timestamp"].astype(str)
    pattern_table_display["price"] = pattern_table_display["price"].apply(lambda x: f"${x:.2f}")
    pattern_table_display.columns = ["Date", "Pattern", "Price", "Signal"]
    
    st.dataframe(pattern_table_display, width='stretch', height=300)
else:
    st.info("No candlestick patterns detected in selected date range.")


# =============================================================================
# CHART PATTERNS - COLORED REGIONS ONLY, CLICKABLE LEGEND
# =============================================================================

st.markdown("---")
st.markdown("### Chart Patterns")
st.markdown("*Click on legend items to show/hide pattern regions*")

if len(chart_patterns) > 0:
    fig_chart = go.Figure()
    
    # Price line
    fig_chart.add_trace(
        go.Scatter(
            x=df_filtered.index,
            y=df_filtered[close_col],
            mode="lines",
            name="Price",
            line=dict(color=config.COLOR_NORMAL, width=1.5),
            hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
        )
    )
    
    # Define colors for each chart pattern type
    chart_pattern_colors = {
        "Double Top": "rgba(244, 67, 54, 0.3)",       # Red
        "Double Bottom": "rgba(76, 175, 80, 0.3)",    # Green
        "Head & Shoulders": "rgba(255, 152, 0, 0.3)", # Orange
        "Cup & Handle": "rgba(33, 150, 243, 0.3)"     # Blue
    }
    
    chart_pattern_border_colors = {
        "Double Top": "rgba(244, 67, 54, 0.8)",
        "Double Bottom": "rgba(76, 175, 80, 0.8)",
        "Head & Shoulders": "rgba(255, 152, 0, 0.8)",
        "Cup & Handle": "rgba(33, 150, 243, 0.8)"
    }
    
    # Group patterns by type
    patterns_by_type = {}
    for p in chart_patterns:
        ptype = p["type"]
        if ptype not in patterns_by_type:
            patterns_by_type[ptype] = []
        patterns_by_type[ptype].append(p)
    
    # For each pattern type, create rectangles using Scatter with fill
    # This makes them appear in legend and be clickable
    for ptype, patterns in patterns_by_type.items():
        fill_color = chart_pattern_colors.get(ptype, "rgba(128, 128, 128, 0.3)")
        border_color = chart_pattern_border_colors.get(ptype, "rgba(128, 128, 128, 0.8)")
        
        # Get y range for rectangles
        y_min = df_filtered[close_col].min()
        y_max = df_filtered[close_col].max()
        y_range = y_max - y_min
        rect_bottom = y_min - y_range * 0.02
        rect_top = y_max + y_range * 0.02
        
        # Create all rectangles for this pattern type as one trace
        x_coords = []
        y_coords = []
        
        for p in patterns:
            start = p["start_date"]
            end = p["end_date"]
            
            # Rectangle coordinates (closed polygon with None to separate)
            x_coords.extend([start, start, end, end, start, None])
            y_coords.extend([rect_bottom, rect_top, rect_top, rect_bottom, rect_bottom, None])
        
        # Add as a single trace (appears in legend)
        fig_chart.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=border_color, width=1),
                name=ptype,
                mode="lines",
                hoverinfo="name",
                hoveron="fills"
            )
        )
        
        # Add necklines for each pattern
        for p in patterns:
            if "neckline" in p:
                fig_chart.add_shape(
                    type="line",
                    x0=p["start_date"],
                    x1=p["end_date"],
                    y0=p["neckline"],
                    y1=p["neckline"],
                    line=dict(color=border_color, width=1, dash="dot"),
                    opacity=0.7
                )
    
    fig_chart.update_layout(
        height=500,
        title=f"Chart Patterns - {get_asset_display_name(selected_asset)}",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02,
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        )
    )
    
    st.plotly_chart(fig_chart, width='stretch')
    
    # Chart pattern details table
    st.markdown("#### Detected Chart Patterns")
    
    chart_pattern_data = []
    for i, p in enumerate(chart_patterns, 1):
        row = {
            "ID": i,
            "Type": p["type"],
            "Start": str(p["start_date"])[:10],
            "End": str(p["end_date"])[:10],
            "Signal": p["signal"],
            "Confidence": f"{p.get('confidence', 0.5) * 100:.0f}%"
        }
        
        if p["type"] == "Double Top":
            row["Details"] = f"Peaks: ${p['peak1_price']:.2f}, ${p['peak2_price']:.2f}"
        elif p["type"] == "Double Bottom":
            row["Details"] = f"Troughs: ${p['trough1_price']:.2f}, ${p['trough2_price']:.2f}"
        elif p["type"] == "Head & Shoulders":
            row["Details"] = f"Head: ${p['head']:.2f}, Shoulders: ${p['left_shoulder']:.2f}"
        elif p["type"] == "Cup & Handle":
            row["Details"] = f"Depth: {p['cup_depth_pct']:.1f}%"
        else:
            row["Details"] = "-"
        
        chart_pattern_data.append(row)
    
    chart_pattern_df = pd.DataFrame(chart_pattern_data)
    st.dataframe(chart_pattern_df, width='stretch')

else:
    st.info(f"""
    No chart patterns detected with current settings.
    
    **Try adjusting:**
    - Increase **Price Tolerance** (currently {tolerance}%)
    - Lower **Peak Prominence** (currently {prominence}%)
    - Increase **Chart Pattern Window** (currently {chart_lookback})
    """)


# =============================================================================
# PATTERN DISTRIBUTION
# =============================================================================

st.markdown("---")
st.markdown("### Pattern Distribution")

all_pattern_counts = {
    "Doji": pattern_summary["doji"],
    "Hammer": pattern_summary["hammer"],
    "Engulfing Bullish": pattern_summary["engulfing_bullish"],
    "Engulfing Bearish": pattern_summary["engulfing_bearish"]
}

chart_type_counts = {}
for p in chart_patterns:
    ptype = p["type"]
    chart_type_counts[ptype] = chart_type_counts.get(ptype, 0) + 1

all_pattern_counts.update(chart_type_counts)
all_pattern_counts = {k: v for k, v in all_pattern_counts.items() if v > 0}

if all_pattern_counts:
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = go.Figure(data=[
            go.Bar(
                x=list(all_pattern_counts.keys()),
                y=list(all_pattern_counts.values()),
                marker_color=[
                    "#9C27B0", "#2196F3", "#4CAF50", "#F44336",
                    "#FF9800", "#00BCD4", "#795548", "#607D8B"
                ][:len(all_pattern_counts)]
            )
        ])
        
        fig_bar.update_layout(
            title="Pattern Frequency",
            xaxis_title="Pattern",
            yaxis_title="Count",
            height=350
        )
        
        st.plotly_chart(fig_bar, width='stretch')
    
    with col2:
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=list(all_pattern_counts.keys()),
                values=list(all_pattern_counts.values()),
                hole=0.4
            )
        ])
        
        fig_pie.update_layout(
            title="Pattern Distribution",
            height=350
        )
        
        st.plotly_chart(fig_pie, width='stretch')

else:
    st.info("No patterns detected to display distribution.")


# =============================================================================
# PATTERN TIMELINE
# =============================================================================

st.markdown("---")
st.markdown("### 📅 Pattern Timeline")
st.markdown("*Click on legend items to show/hide signals*")

timeline_data = []

for pattern_col, pattern_name in pattern_names.items():
    if pattern_col in df_patterns.columns:
        mask = df_patterns[pattern_col]
        for idx in df_patterns[mask].index:
            timeline_data.append({
                "date": idx,
                "pattern": pattern_name,
                "signal": "Bullish" if "Bullish" in pattern_name or pattern_name == "Hammer" else (
                    "Bearish" if "Bearish" in pattern_name else "Neutral"
                )
            })

for p in chart_patterns:
    timeline_data.append({
        "date": p["start_date"],
        "pattern": p["type"],
        "signal": p["signal"]
    })

if timeline_data:
    timeline_df = pd.DataFrame(timeline_data)
    timeline_df = timeline_df.sort_values("date")
    
    fig_timeline = go.Figure()
    
    signal_colors = {
        "Bullish": config.COLOR_BULLISH,
        "Bearish": config.COLOR_BEARISH,
        "Neutral": "#9E9E9E"
    }
    
    for signal in ["Bullish", "Bearish", "Neutral"]:
        mask = timeline_df["signal"] == signal
        if mask.any():
            data = timeline_df[mask]
            fig_timeline.add_trace(
                go.Scatter(
                    x=data["date"],
                    y=[signal] * len(data),
                    mode="markers",
                    name=signal,
                    marker=dict(size=12, color=signal_colors[signal]),
                    text=data["pattern"],
                    hovertemplate="<b>%{text}</b><br>Date: %{x}<extra></extra>"
                )
            )
    
    fig_timeline.update_layout(
        height=300,
        title="Pattern Signals Over Time",
        xaxis_title="Date",
        yaxis_title="Signal Type",
        hovermode="closest",
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02,
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        )
    )
    
    st.plotly_chart(fig_timeline, width='stretch')

else:
    st.info("No patterns to display on timeline.")


# =============================================================================
# FOOTER
# =============================================================================

footer("Pattern Recognition")
