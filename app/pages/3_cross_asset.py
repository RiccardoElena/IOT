"""
Cross-Asset Analysis Page

This page provides multi-asset analysis including:
- Correlation matrix heatmap
- Rolling correlation over time
- Normalized price comparison
- Simultaneous anomaly detection
- Systemic event identification
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.anomaly_detection import detect_anomalies
from src.cross_asset import (
    analyze_asset_pair,
    calculate_correlation_matrix,
    create_price_matrix_from_dict,
    format_pair_name,
    get_asset_pairs,
    get_typical_correlations,
    normalize_prices,
)
from src.data_loader import (
    filter_by_date_range,
    get_asset_display_name,
    load_all_assets,
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=f"Cross-Asset Analysis | {config.PAGE_TITLE}",
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT
)

st.title("🔗 Cross-Asset Analysis")
st.markdown("""
Analyze relationships between multiple assets: correlations, 
simultaneous movements, and systemic events.
""")


# =============================================================================
# HELPER FUNCTIONS FOR CONSISTENT ANOMALY HANDLING
# =============================================================================

def get_anomaly_details_by_date(anomaly_flags: dict) -> pd.DataFrame:
    """
    Create a DataFrame with anomaly details for each date.
    Ensures consistent handling of NaN values.
    
    Returns DataFrame with columns: date, count, assets_list, assets_str
    """
    # Create DataFrame from flags
    anomaly_df = pd.DataFrame(anomaly_flags)
    
    # CRITICAL: Fill NaN with False BEFORE any operations
    anomaly_df = anomaly_df.fillna(False).astype(bool)
    
    results = []
    for timestamp in anomaly_df.index:
        row = anomaly_df.loc[timestamp]
        # Get list of assets with True
        affected_assets = list(row[row].index)
        
        if len(affected_assets) > 0:
            results.append({
                "timestamp": timestamp,
                "count": len(affected_assets),
                "assets_list": affected_assets,
                "assets_str": ", ".join([config.ASSETS.get(a, a) for a in affected_assets])
            })
    
    if not results:
        return pd.DataFrame(columns=["timestamp", "count", "assets_list", "assets_str"])
    
    return pd.DataFrame(results).set_index("timestamp")


def count_simultaneous_anomalies_consistent(anomaly_flags: dict) -> pd.Series:
    """
    Count simultaneous anomalies with consistent NaN handling.
    """
    anomaly_df = pd.DataFrame(anomaly_flags)
    # CRITICAL: Same fillna as get_anomaly_details_by_date
    anomaly_df = anomaly_df.fillna(False).astype(bool)
    return anomaly_df.sum(axis=1)


# =============================================================================
# SIDEBAR - CONTROLS AND INFO
# =============================================================================

with st.sidebar:
    st.header("⚙️ Controls")
    
    # Granularity selection (only daily for cross-asset)
    st.info("📊 Using **Daily** data for cross-asset analysis")
    
    st.markdown("---")
    
    # Correlation window
    correlation_window = st.slider(
        "Correlation Window (days)",
        min_value=10,
        max_value=60,
        value=config.CORRELATION_WINDOW,
        step=5,
        help="Number of days for rolling correlation calculation"
    )
    
    # Systemic event threshold
    systemic_threshold = st.slider(
        "Systemic Event Threshold",
        min_value=2,
        max_value=5,
        value=config.SYSTEMIC_EVENT_THRESHOLD,
        help="Minimum assets with anomalies to flag as systemic event"
    )
    
    st.markdown("---")
    
    # INFO BOX: Pearson Correlation
    with st.expander("📊 Pearson Correlation", expanded=False):
        st.markdown("""
        **What is Pearson Correlation?**
        
        A measure of linear relationship between two variables, 
        ranging from -1 to +1.
        
        **Formula:**
        ```
        r = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² × Σ(y - ȳ)²]
        ```
        
        ---
        
        **Interpretation:**
        
        | Value | Strength | Direction |
        |-------|----------|-----------|
        | +1.0 | Perfect | Positive |
        | +0.7 to +1.0 | Strong | Positive |
        | +0.4 to +0.7 | Moderate | Positive |
        | +0.2 to +0.4 | Weak | Positive |
        | -0.2 to +0.2 | None/Very weak | - |
        | -0.4 to -0.2 | Weak | Negative |
        | -0.7 to -0.4 | Moderate | Negative |
        | -1.0 to -0.7 | Strong | Negative |
        | -1.0 | Perfect | Negative |
        
        ---
        
        **Why use returns, not prices?**
        
        Prices are non-stationary (trending over time), which 
        violates correlation assumptions. Returns are:
        - Mean-reverting
        - Stationary
        - Comparable across assets
        
        ---
        
        **Limitations:**
        
        - Only measures **linear** relationships
        - Sensitive to outliers
        - Doesn't imply causation
        - Can change over time (use rolling)
        """)
    
    # INFO BOX: Rolling Correlation
    with st.expander("🔄 Rolling Correlation", expanded=False):
        st.markdown(f"""
        **Why use rolling correlation?**
        
        Correlations between assets are **not constant**. They 
        change due to:
        
        - Market regimes (bull vs. bear)
        - Economic cycles
        - Policy changes
        - Crisis events
        
        ---
        
        **How it works:**
        
        Current window: **{correlation_window} days**
        
        ```
        Day 1-30:   Calculate correlation
        Day 2-31:   Calculate correlation
        Day 3-32:   Calculate correlation
        ...and so on
        ```
        
        ---
        
        **Use cases:**
        
        1. **Hedge effectiveness:** Is your hedge still working?
        
        2. **Regime detection:** Correlations spike in crises
        
        3. **Diversification check:** Are assets becoming more 
           correlated?
        
        4. **Pair trading:** Entry when correlation deviates, 
           exit when it reverts
        
        ---
        
        **Correlation breakdown:**
        
        When correlations deviate significantly (>2σ) from their 
        historical mean, it signals a potential regime change or 
        market stress.
        """)
    
    # INFO BOX: Systemic Events
    with st.expander("⚠️ Systemic Events", expanded=False):
        st.markdown(f"""
        **What are systemic events?**
        
        When **{systemic_threshold}+ assets** show anomalies 
        simultaneously, it suggests a market-wide event rather 
        than asset-specific news.
        
        ---
        
        **Examples of systemic events:**
        
        | Event | Affected Assets |
        |-------|-----------------|
        | 2008 Financial Crisis | All |
        | COVID-19 Crash (Mar 2020) | All |
        | Fed Rate Decision | S&P, Gold, USD |
        | Oil Price War (2020) | Oil, S&P |
        | Crypto Crash | BTC, Risk assets |
        
        ---
        
        **Why track systemic events?**
        
        1. **Risk management:** Portfolio-wide exposure
        
        2. **Diversification failure:** In crises, correlations 
           go to 1 (everything falls together)
        
        3. **Opportunity detection:** Extreme events can create 
           buying opportunities
        
        4. **Narrative construction:** What caused this?
        
        ---
        
        **Threshold selection:**
        
        - **2 assets:** Common, may be coincidental
        - **3 assets:** Noteworthy, investigate
        - **4+ assets:** Likely systemic, significant event
        
        Current threshold: **{systemic_threshold} assets**
        """)
    
    # INFO BOX: Safe Haven vs Risk Assets
    with st.expander("🛡️ Safe Haven Assets", expanded=False):
        st.markdown("""
        **What are safe haven assets?**
        
        Assets that maintain or increase value during market stress.
        
        ---
        
        **Traditional safe havens:**
        
        | Asset | Behavior in Crisis |
        |-------|-------------------|
        | **Gold** | Rises (flight to safety) |
        | **USD** | Rises (global reserve) |
        | **CHF** | Rises (Swiss stability) |
        | **JPY** | Rises (carry trade unwind) |
        | **US Treasuries** | Rises (risk-off) |
        
        ---
        
        **Risk assets:**
        
        | Asset | Behavior in Crisis |
        |-------|-------------------|
        | **Stocks (S&P 500)** | Falls |
        | **Oil** | Falls (demand drop) |
        | **Bitcoin** | Falls (speculative) |
        | **EM Currencies** | Falls |
        | **High Yield Bonds** | Falls |
        
        ---
        
        **Typical correlations:**
        
        - **Gold ↔ USD:** Negative (gold priced in USD)
        - **Gold ↔ S&P 500:** Low/Negative
        - **Oil ↔ S&P 500:** Positive (economic activity)
        - **BTC ↔ Risk assets:** Variable
        
        ---
        
        **Correlation in crises:**
        
        During severe crises, "all correlations go to 1" as 
        investors sell everything for cash (even safe havens 
        initially, like Gold in March 2020).
        """)
    
    # INFO BOX: Price Normalization
    with st.expander("📈 Price Normalization", expanded=False):
        st.markdown("""
        **Why normalize prices?**
        
        Different assets have vastly different price scales:
        - S&P 500: ~5,000
        - Gold: ~2,000
        - Oil: ~80
        - Bitcoin: ~60,000
        
        Normalizing to base 100 allows visual comparison of 
        **relative performance**.
        
        ---
        
        **How it works:**
        
        ```
        Normalized_Price = (Current_Price / Start_Price) × 100
        ```
        
        Example:
        - Start: Gold = $1,800, S&P = 4,500
        - After: Gold = $1,980 (+10%), S&P = 4,950 (+10%)
        - Normalized: Both = 110
        
        ---
        
        **Use cases:**
        
        1. **Performance comparison:** Which asset performed best?
        
        2. **Trend identification:** Divergence/convergence
        
        3. **Relative strength:** Who's leading/lagging?
        
        4. **Rebalancing signals:** Asset allocation drift
        """)
    
    # INFO BOX: Hedging
    with st.expander("🔀 Hedging with Correlation", expanded=False):
        st.markdown("""
        **What is hedging?**
        
        Reducing portfolio risk by holding negatively correlated 
        assets.
        
        ---
        
        **Effective hedges:**
        
        | Long Position | Hedge | Typical Correlation |
        |---------------|-------|---------------------|
        | S&P 500 | Gold | -0.1 to -0.3 |
        | S&P 500 | USD | -0.2 to -0.4 |
        | Oil | USD | -0.3 to -0.5 |
        | Stocks | Bonds | -0.2 to -0.5 |
        
        ---
        
        **Hedge ratio:**
        
        To fully hedge, you need:
        ```
        Hedge_Ratio = -Correlation × (σ_asset / σ_hedge)
        ```
        
        ---
        
        **Hedge effectiveness over time:**
        
        Use rolling correlation to monitor:
        - Hedge still working?
        - Correlation changed?
        - Need to adjust ratio?
        
        ---
        
        **Warning:**
        
        Hedges can fail during extreme events when correlations 
        break down. Always stress-test your hedges.
        """)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_all_daily_data():
    """Load daily data for all assets."""
    return load_all_assets("daily")


@st.cache_data
def process_cross_asset_data(data_dict, start_date, end_date, zscore_threshold):
    """Process all assets for cross-asset analysis."""
    
    # Filter each asset by date range
    filtered_data = {}
    anomaly_flags = {}
    
    for asset, df in data_dict.items():
        df_filtered = filter_by_date_range(df, start_date, end_date)
        df_processed = detect_anomalies(df_filtered, zscore_threshold=zscore_threshold)
        filtered_data[asset] = df_processed
        anomaly_flags[asset] = df_processed["anomaly_any"]
    
    # Create price matrix
    price_matrix = create_price_matrix_from_dict(filtered_data)
    
    return filtered_data, price_matrix, anomaly_flags


# Load data
try:
    with st.spinner("Loading data for all assets..."):
        all_data = load_all_daily_data()
    
    if len(all_data) == 0:
        st.error("No data loaded. Please check that CSV files exist.")
        st.stop()
    
    if len(all_data) < 2:
        st.error("Need at least 2 assets for cross-asset analysis.")
        st.stop()
    
    # Use toast for temporary notification
    st.toast(f"Loaded {len(all_data)} assets: {', '.join(all_data.keys())}", icon="✅")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# =============================================================================
# DATE RANGE SELECTION
# =============================================================================

st.markdown("---")
st.markdown("### 📅 Date Range")

# Get common date range across all assets
first_asset = list(all_data.values())[0]
min_date = first_asset.index.min().date()
max_date = first_asset.index.max().date()

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


# =============================================================================
# PROCESS DATA
# =============================================================================

try:
    with st.spinner("Processing cross-asset data..."):
        filtered_data, price_matrix, anomaly_flags = process_cross_asset_data(
            all_data,
            str(start_date),
            str(end_date),
            config.ZSCORE_ANOMALY_THRESHOLD
        )

except Exception as e:
    st.error(f"Error processing data: {e}")
    st.stop()


# =============================================================================
# CORRELATION MATRIX
# =============================================================================

st.markdown("---")
st.markdown("### 🔥 Correlation Matrix")

# Calculate correlation matrix on returns
returns_matrix = price_matrix.pct_change().dropna()
corr_matrix = returns_matrix.corr()

# Get display names for axes
display_names = [config.ASSETS.get(a, a) for a in corr_matrix.columns]

# Invert rows for proper matrix orientation (0,0 at top-left)
corr_values_inverted = corr_matrix.values[::-1, :]
display_names_y_inverted = display_names[::-1]

# Create heatmap with inverted Y axis
fig_heatmap = go.Figure(data=go.Heatmap(
    z=corr_values_inverted,
    x=display_names,
    y=display_names_y_inverted,
    colorscale=[
        [0, config.COLOR_BEARISH],      # -1: Red
        [0.5, "white"],                  # 0: White
        [1, config.COLOR_BULLISH]        # +1: Green
    ],
    zmin=-1,
    zmax=1,
    text=np.round(corr_values_inverted, 3),
    texttemplate="%{text:.3f}",
    textfont={"size": 14},
    hovertemplate=(
        "%{x} vs %{y}<br>"
        "Correlation: %{z:.3f}<br>"
        "<extra></extra>"
    )
))

fig_heatmap.update_layout(
    height=400,
    title="Asset Return Correlations"
)

st.plotly_chart(fig_heatmap, width='stretch')

# Typical correlations info
with st.expander("ℹ️ Typical Expected Correlations"):
    typical = get_typical_correlations()
    for pair, description in typical.items():
        if pair[0] in all_data and pair[1] in all_data:
            actual = corr_matrix.loc[pair[0], pair[1]]
            st.markdown(f"**{format_pair_name(pair[0], pair[1])}**: {description}")
            st.markdown(f"   → Current: {actual:.3f}")


# =============================================================================
# NORMALIZED PRICES
# =============================================================================

st.markdown("---")
st.markdown("### 📈 Normalized Price Comparison")

st.markdown("All prices normalized to base 100 for comparison.")

# Normalize prices
norm_prices = normalize_prices(price_matrix)

# Create line chart
fig_normalized = go.Figure()

for asset in norm_prices.columns:
    display_name = config.ASSETS.get(asset, asset)
    fig_normalized.add_trace(
        go.Scatter(
            x=norm_prices.index,
            y=norm_prices[asset],
            mode="lines",
            name=display_name,
            hovertemplate=(
                f"<b>{display_name}</b><br>"
                "Date: %{x}<br>"
                "Value: %{y:.1f}<br>"
                "<extra></extra>"
            )
        )
    )

fig_normalized.update_layout(
    height=450,
    title="Normalized Price Performance (Base = 100)",
    xaxis_title="Date",
    yaxis_title="Normalized Value",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)

st.plotly_chart(fig_normalized, width='stretch')


# =============================================================================
# PAIR ANALYSIS
# =============================================================================

st.markdown("---")
st.markdown("### 🔍 Pair Deep Dive")

# Asset pair selection
pairs = get_asset_pairs()
pair_names = [format_pair_name(a, b) for a, b in pairs]

selected_pair_name = st.selectbox(
    "Select Asset Pair",
    options=pair_names
)

# Get selected pair
selected_pair = pairs[pair_names.index(selected_pair_name)]
asset_a, asset_b = selected_pair

# Analyze pair
pair_analysis = analyze_asset_pair(
    price_matrix, 
    asset_a, 
    asset_b, 
    window=correlation_window
)

# Display pair statistics with info popovers
st.markdown("#### Correlation Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    subcol1, subcol2 = st.columns([4, 1])
    with subcol1:
        st.metric(
            "Static Correlation",
            f"{pair_analysis['static_correlation']:.3f}"
        )
    with subcol2:
        with st.popover("ℹ️"):
            st.markdown("""
            **Static Correlation**
            
            Single correlation value calculated over the **entire selected period**.
            
            Gives an overall view of the relationship, but misses how correlation changes over time.
            """)

with col2:
    subcol1, subcol2 = st.columns([4, 1])
    with subcol1:
        st.metric(
            "Current Correlation",
            f"{pair_analysis['statistics']['current']:.3f}" 
            if pair_analysis['statistics']['current'] else "-"
        )
    with subcol2:
        with st.popover("ℹ️"):
            st.markdown(f"""
            **Current Correlation**
            
            The **most recent** value of the rolling correlation (last {correlation_window} days).
            
            Shows the current state of the relationship between the two assets.
            """)

with col3:
    subcol1, subcol2 = st.columns([4, 1])
    with subcol1:
        st.metric(
            "Correlation Std",
            f"{pair_analysis['statistics']['std']:.3f}"
        )
    with subcol2:
        with st.popover("ℹ️"):
            st.markdown("""
            **Correlation Standard Deviation**
            
            Measures how much the rolling correlation **varies over time**.
            
            - **Low std (< 0.1)**: Stable relationship
            - **High std (> 0.2)**: Volatile relationship, changes frequently
            """)

with col4:
    subcol1, subcol2 = st.columns([4, 1])
    with subcol1:
        st.metric(
            "Correlation Anomalies",
            pair_analysis['anomaly_count']
        )
    with subcol2:
        with st.popover("ℹ️"):
            st.markdown("""
            **Correlation Anomalies**
            
            Number of times the rolling correlation deviated **more than 2 standard deviations** from its mean.
            
            High count indicates unstable or regime-changing relationship.
            """)

# Rolling correlation chart
st.markdown("#### Rolling Correlation Over Time")

fig_rolling = go.Figure()

rolling_corr = pair_analysis["rolling_correlation"]
mean_corr = pair_analysis["statistics"]["mean"]
std_corr = pair_analysis["statistics"]["std"]

# Main line
fig_rolling.add_trace(
    go.Scatter(
        x=rolling_corr.index,
        y=rolling_corr.values,
        mode="lines",
        name="Rolling Correlation",
        line=dict(color=config.COLOR_NORMAL),
        hovertemplate="Date: %{x}<br>Correlation: %{y:.3f}<extra></extra>"
    )
)

# Mean line
fig_rolling.add_hline(
    y=mean_corr,
    line_dash="dash",
    line_color="gray",
    annotation_text=f"Mean: {mean_corr:.3f}"
)

# Anomaly bands
fig_rolling.add_hline(
    y=mean_corr + 2*std_corr,
    line_dash="dot",
    line_color=config.COLOR_WARNING,
    annotation_text="+2σ"
)
fig_rolling.add_hline(
    y=mean_corr - 2*std_corr,
    line_dash="dot",
    line_color=config.COLOR_WARNING,
    annotation_text="-2σ"
)

# Anomaly points
anomaly_mask = pair_analysis["anomaly_mask"]
if anomaly_mask.any():
    anomaly_corr = rolling_corr[anomaly_mask]
    fig_rolling.add_trace(
        go.Scatter(
            x=anomaly_corr.index,
            y=anomaly_corr.values,
            mode="markers",
            name="Correlation Anomaly",
            marker=dict(
                size=10,
                color=config.COLOR_ANOMALY
            ),
            hovertemplate=(
                "<b>⚠️ CORRELATION ANOMALY</b><br>"
                "Date: %{x}<br>"
                "Correlation: %{y:.3f}<br>"
                "<extra></extra>"
            )
        )
    )

fig_rolling.update_layout(
    height=350,
    title=f"Rolling {correlation_window}-Day Correlation: {selected_pair_name}",
    xaxis_title="Date",
    yaxis_title="Correlation",
    yaxis=dict(range=[-1.1, 1.1]),
    hovermode="x unified"
)

st.plotly_chart(fig_rolling, width='stretch')

# Scatter plot of returns
st.markdown("#### Return Scatter Plot")

returns_a = pair_analysis["returns_a"].dropna()
returns_b = pair_analysis["returns_b"].dropna()

# Align indices
common_idx = returns_a.index.intersection(returns_b.index)
returns_a = returns_a.loc[common_idx]
returns_b = returns_b.loc[common_idx]

fig_scatter = go.Figure()

fig_scatter.add_trace(
    go.Scatter(
        x=returns_a,
        y=returns_b,
        mode="markers",
        marker=dict(
            size=5,
            color=config.COLOR_NORMAL,
            opacity=0.6
        ),
        hovertemplate=(
            f"{config.ASSETS.get(asset_a, asset_a)}: " + "%{x:.2f}%<br>"
            f"{config.ASSETS.get(asset_b, asset_b)}: " + "%{y:.2f}%<br>"
            "<extra></extra>"
        )
    )
)

fig_scatter.update_layout(
    height=350,
    title=f"Daily Returns: {selected_pair_name}",
    xaxis_title=f"{config.ASSETS.get(asset_a, asset_a)} Return (%)",
    yaxis_title=f"{config.ASSETS.get(asset_b, asset_b)} Return (%)"
)

st.plotly_chart(fig_scatter, width='stretch')


# =============================================================================
# SIMULTANEOUS ANOMALIES
# =============================================================================

st.markdown("---")
st.markdown("### ⚠️ Simultaneous Anomalies")

# Use consistent functions for both chart and table
anomaly_details = get_anomaly_details_by_date(anomaly_flags)
anomaly_counts = count_simultaneous_anomalies_consistent(anomaly_flags)

# Create systemic mask
systemic_mask = anomaly_counts >= systemic_threshold

# Summary metrics
total_systemic = systemic_mask.sum()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Days Analyzed", len(anomaly_counts))
with col2:
    st.metric("Days with Any Anomaly", int((anomaly_counts > 0).sum()))
with col3:
    subcol1, subcol2 = st.columns([4, 1])
    with subcol1:
        st.metric(f"Systemic Events (≥{systemic_threshold} assets)", int(total_systemic))
    with subcol2:
        with st.popover("ℹ️"):
            st.markdown(f"""
            **Systemic Events**
            
            Days when **{systemic_threshold} or more assets** showed anomalies simultaneously.
            
            This suggests a market-wide event affecting multiple assets at once, rather than asset-specific news.
            """)

# Prepare data for chart with asset names in customdata
chart_dates = anomaly_counts.index.tolist()
chart_counts = anomaly_counts.values.tolist()

# Build assets string for each date
assets_for_hover = []
for date in chart_dates:
    if date in anomaly_details.index:
        assets_for_hover.append(anomaly_details.loc[date, "assets_str"])
    else:
        assets_for_hover.append("None")

# Color bars by severity
colors = [
    config.COLOR_ANOMALY if c >= systemic_threshold else 
    (config.COLOR_WARNING if c >= 2 else config.COLOR_NORMAL)
    for c in chart_counts
]

# Bar chart of simultaneous anomalies
fig_simultaneous = go.Figure()

fig_simultaneous.add_trace(
    go.Bar(
        x=chart_dates,
        y=chart_counts,
        marker_color=colors,
        customdata=assets_for_hover,
        hovertemplate=(
            "<b>Date:</b> %{x}<br>"
            "<b>Anomaly Count:</b> %{y}<br>"
            "<b>Assets:</b> %{customdata}<br>"
            "<extra></extra>"
        )
    )
)

# Threshold line
fig_simultaneous.add_hline(
    y=systemic_threshold,
    line_dash="dash",
    line_color=config.COLOR_ANOMALY,
    annotation_text=f"Systemic threshold ({systemic_threshold})"
)

fig_simultaneous.update_layout(
    height=350,
    title="Simultaneous Anomalies per Day",
    xaxis_title="Date",
    yaxis_title="Number of Assets with Anomalies",
    hovermode="x unified"
)

st.plotly_chart(fig_simultaneous, width='stretch')

# Systemic events table (using same data source as chart)
if total_systemic > 0:
    st.markdown("#### Systemic Event Details")
    
    # Filter anomaly_details for systemic events only
    systemic_dates = anomaly_counts[systemic_mask].index
    systemic_table_data = []
    
    for date in systemic_dates:
        if date in anomaly_details.index:
            row = anomaly_details.loc[date]
            systemic_table_data.append({
                "Date": str(date)[:10],
                "Assets Affected": row["count"],
                "Which Assets": row["assets_str"]
            })
    
    if systemic_table_data:
        systemic_df = pd.DataFrame(systemic_table_data)
        systemic_df.index = range(1, len(systemic_df) + 1)
        st.dataframe(systemic_df, width='stretch')
    else:
        st.info(f"No systemic events detected (threshold: {systemic_threshold}+ assets)")
else:
    st.info(f"No systemic events detected (threshold: {systemic_threshold}+ assets)")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Cross-Asset Analysis | IoT & Data Analytics Project
</div>
""", unsafe_allow_html=True)
