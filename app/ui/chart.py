import plotly.graph_objects as go
import streamlit as st
import config
import pandas as pd
import numpy as np
from typing import Any, Dict, List

from plotly.subplots import make_subplots

def create_zscore_chart(data: pd.Series, anomaly_mask: pd.Series, show_anomalies: bool, thresholds: dict, zscore_threshold: float) -> go.Figure:
    """Create a Z-score chart with threshold lines and anomaly markers."""
    fig = go.Figure()
    
    # Main Z-score line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.values,
            mode="lines",
            name="Z-Score",
            line={"color": config.COLOR_NORMAL, "width": 1},
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
                marker={"size": config.MARKER_SIZE_ANOMALY, "color": config.COLOR_ANOMALY},
                hovertemplate="<b>⚠️ ANOMALY</b><br>Time: %{x}<br>Z-Score: %{y:.2f}σ<extra></extra>"
            )
        )
    
    # Threshold lines
    fig.add_hline(y=thresholds["anomaly_upper"], line_dash="dash", line_color=config.COLOR_ANOMALY, annotation_text=f"+{zscore_threshold}σ")
    fig.add_hline(y=thresholds["anomaly_lower"], line_dash="dash", line_color=config.COLOR_ANOMALY, annotation_text=f"-{zscore_threshold}σ")
    fig.add_hline(y=thresholds["warning_upper"], line_dash="dot", line_color=config.COLOR_WARNING, annotation_text=f"+{config.ZSCORE_WARNING_THRESHOLD}σ")
    fig.add_hline(y=thresholds["warning_lower"], line_dash="dot", line_color=config.COLOR_WARNING, annotation_text=f"-{config.ZSCORE_WARNING_THRESHOLD}σ")
    fig.add_hline(y=0, line_color="gray", line_width=0.5)
    
    # Colored regions
    fig.add_hrect(y0=thresholds["warning_lower"], y1=thresholds["warning_upper"], fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=thresholds["warning_upper"], y1=thresholds["anomaly_upper"], fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=thresholds["anomaly_lower"], y1=thresholds["warning_lower"], fillcolor="yellow", opacity=0.1, line_width=0)
    
    # Apply zoom if set
    if st.session_state.selected_zoom_range is not None:
        fig.update_xaxes(range=[st.session_state.selected_zoom_range["start"], st.session_state.selected_zoom_range["end"]])
    
    # Layout
    fig.update_layout(
        height=350,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        hovermode="x unified",
        margin={"t": 10}  # Reduce top margin to eliminate title space
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Z-Score")
    
    return fig

def create_combined_chart(
    current_idx: int,
    anomalies: List[Dict[str, Any]],
    timestamps: pd.DatetimeIndex,
    prices: np.ndarray,
    volumes: np.ndarray,
    window_size: int,
    zscore_threshold: float,
) -> go.Figure:
    """Create a combined chart with 3 subplots sharing X axis."""
    
    # Create subplots: 3 rows, shared X axis
    fig = make_subplots(
        rows=3, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.30, 0.25],
        subplot_titles=("Price (Streaming)", "Rolling Z-Score", "Volume")
    )
    
    if current_idx > 0:
        display_timestamps = timestamps[:current_idx]
        display_prices = prices[:current_idx]
        display_volumes = volumes[:current_idx]
        
        # =====================================================================
        # ROW 1: PRICE CHART
        # =====================================================================
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=display_timestamps,
                y=display_prices,
                mode="lines",
                name="Price",
                line={"color": config.COLOR_NORMAL, "width": 2},
                hovertemplate="Price: $%{y:.2f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Anomaly markers on price
        visible_anomalies = [a for a in anomalies if a["idx"] < current_idx]
        if visible_anomalies:
            fig.add_trace(
                go.Scatter(
                    x=[a["timestamp"] for a in visible_anomalies],
                    y=[a["price"] for a in visible_anomalies],
                    mode="markers",
                    name="Anomaly",
                    marker={"size": 12, "color": config.COLOR_ANOMALY, "symbol": "x", "line": {"width": 2}},
                    customdata=[a["zscore"] for a in visible_anomalies],
                    hovertemplate="<b>⚠️ ANOMALY</b><br>Price: $%{y:.2f}<br>Z-Score: %{customdata:.2f}σ<extra></extra>"
                ),
                row=1, col=1
            )
        
        # Window rectangle on price chart
        if current_idx > window_size:
            window_start_idx = current_idx - window_size
            fig.add_vrect(
                x0=timestamps[window_start_idx],
                x1=timestamps[current_idx - 1],
                fillcolor="rgba(100, 149, 237, 0.2)",
                line_width=2,
                line_color="rgba(100, 149, 237, 0.8)",
                row=1, col=1 # type: ignore
            )
        else:
            fig.add_vrect(
                x0=timestamps[0],
                x1=timestamps[current_idx - 1],
                fillcolor="rgba(255, 193, 7, 0.2)",
                line_width=2,
                line_color="rgba(255, 193, 7, 0.8)",
                annotation_text=f"Building ({current_idx}/{window_size})",
                annotation_position="top left",
                annotation_font_size=10,
                row=1, col=1 # type: ignore
            )
        
        # =====================================================================
        # ROW 2: Z-SCORE CHART
        # =====================================================================
        
        if current_idx > 1:
            zscores = []
            
            for i in range(current_idx):
                if i < 2:
                    zscores.append(0)
                elif i < window_size:
                    # Building phase: use all available points
                    window_data = np.asarray(prices[:i])
                    current_price = prices[i]
                    mean = window_data.mean()
                    std = window_data.std()
                    if std > 0:
                        zscores.append((current_price - mean) / std)
                    else:
                        zscores.append(0)
                else:
                    # Stable phase: use full window
                    window_data = np.asarray(prices[i - window_size:i])
                    current_price = prices[i]
                    mean = window_data.mean()
                    std = window_data.std()
                    if std > 0:
                        zscores.append((current_price - mean) / std)
                    else:
                        zscores.append(0)
            
            # Split into building (yellow) and stable (blue) segments
            if current_idx <= window_size:
                # All points in building phase
                fig.add_trace(
                    go.Scatter(
                        x=display_timestamps,
                        y=zscores,
                        mode="lines",
                        name="Z-Score (building)",
                        line={"color": "rgba(255, 193, 7, 1)", "width": 2},
                        fill='tozeroy',
                        fillcolor='rgba(255, 193, 7, 0.15)',
                        hovertemplate="Z-Score: %{y:.2f}σ <i>(building)</i><extra></extra>"
                    ),
                    row=2, col=1
                )
            else:
                # Split: yellow for building, blue for stable
                split_idx = window_size
                
                # Building phase (yellow)
                fig.add_trace(
                    go.Scatter(
                        x=display_timestamps[:split_idx + 1],
                        y=zscores[:split_idx + 1],
                        mode="lines",
                        name="Z-Score (building)",
                        line={"color": "rgba(255, 193, 7, 1)", "width": 2},
                        fill='tozeroy',
                        fillcolor='rgba(255, 193, 7, 0.15)',
                        hovertemplate="Z-Score: %{y:.2f}σ <i>(building)</i><extra></extra>"
                    ),
                    row=2, col=1
                )
                
                # Stable phase (blue)
                fig.add_trace(
                    go.Scatter(
                        x=display_timestamps[split_idx:],
                        y=zscores[split_idx:],
                        mode="lines",
                        name="Z-Score (stable)",
                        line={"color": config.COLOR_NORMAL, "width": 2},
                        fill='tozeroy',
                        fillcolor='rgba(100, 149, 237, 0.1)',
                        hovertemplate="Z-Score: %{y:.2f}σ<extra></extra>"
                    ),
                    row=2, col=1
                )
            
            # Threshold lines for Z-Score
            fig.add_hline(
                y=zscore_threshold, 
                line_dash="dash", 
                line_color=config.COLOR_ANOMALY,
                annotation_text=f"+{zscore_threshold}σ",
                annotation_position="right",
                annotation_font_size=10,
                row=2, col=1 # type: ignore
            )
            fig.add_hline(
                y=-zscore_threshold, 
                line_dash="dash", 
                line_color=config.COLOR_ANOMALY,
                annotation_text=f"-{zscore_threshold}σ",
                annotation_position="right",
                annotation_font_size=10,
                row=2, col=1 # type: ignore
            )
            fig.add_hline(y=0, line_color="gray", line_width=1, row=2, col=1) # type: ignore
        
        # =====================================================================
        # ROW 3: VOLUME CHART
        # =====================================================================
        
        fig.add_trace(
            go.Bar(
                x=display_timestamps,
                y=display_volumes,
                name="Volume",
                marker_color=config.COLOR_NORMAL,
                opacity=0.7,
                hovertemplate="Volume: %{y:,.0f}<extra></extra>"
            ),
            row=3, col=1
        )
    
    else:
        # No data yet - show waiting message
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="Press ▶️ Start to begin streaming simulation",
            showarrow=False,
            font={"size": 18, "color": "gray"}
        )
    
    # =========================================================================
    # LAYOUT
    # =========================================================================
    
    fig.update_layout(
        height=650,
        hovermode="x unified",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 10}
        },
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
    )
    
    # Update Y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score (σ)", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    # Only show X-axis label on bottom chart
    fig.update_xaxes(title_text="Time", row=3, col=1)
    
    # Style subplot titles
    annotations = list(fig.layout.annotations) if hasattr(fig.layout, 'annotations') else []  # type: ignore
    for annotation in annotations:
        if hasattr(annotation, 'text') and annotation.text in ["Price (Streaming)", "Rolling Z-Score", "Volume"]:
            annotation.update(font={"size": 12, "color": "gray"})
    
    return fig
