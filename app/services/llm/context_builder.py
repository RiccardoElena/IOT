
from typing import Any, Dict, Optional, List, Callable
from config.ui import PageType

def context_builder_factory(page: PageType) -> Callable:
    """
    Factory function to create a ContextBuilder instance based on the page type.
    
    Args:
        page: An instance of PageType Enum indicating the type of page.
    
    Returns:
        An instance of ContextBuilder tailored for the specified page type.
    """
    if page == PageType.SINGLE_ASSET:
        return _build_single_asset_context
    elif page == PageType.REALTIME:
        return _build_realtime_context
    elif page == PageType.CROSS_ASSET:
        return _build_cross_asset_context
    elif page == PageType.PATTERNS:
        return _build_pattern_context
    else:
        raise ValueError(f"Unsupported page type: {page}")
    
# =============================================================================
# CONTEXT BUILDERS
# =============================================================================

def _build_single_asset_context(
    asset: str,
    asset_display: str,
    granularity: str,
    start_date: str,
    end_date: str,
    total_records: int,
    price_stats: Dict[str, Any],
    anomaly_counts: Dict[str, int],
    anomaly_details: List[Dict[str, Any]],
    zscore_current: Optional[Dict[str, str]] = None,
    volume_stats: Optional[Dict[str, Any]] = None,
    volatility_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build comprehensive context for the Single Asset Analysis page.
    
    Args:
        asset: Internal asset key
        asset_display: Display name of the asset
        granularity: Data granularity (minute/hourly/daily)
        start_date: Start date of the analysis period
        end_date: End date of the analysis period
        total_records: Total number of data points
        price_stats: Price statistics dictionary
        anomaly_counts: Dictionary with anomaly counts by type
        anomaly_details: List of anomaly detail dictionaries
        zscore_current: Current Z-scores (optional)
        volume_stats: Volume statistics (optional)
        volatility_stats: Volatility statistics (optional)
    
    Returns:
        Complete context dictionary
    """
    context = {
        "page": "Single Asset Analysis",
        "asset": asset,
        "asset_display": asset_display,
        "granularity": granularity,
        "period": {
            "start": start_date,
            "end": end_date,
            "total_records": total_records
        },
        "price_statistics": price_stats,
        "anomalies": {
            "counts": anomaly_counts,
            "details": anomaly_details
        }
    }
    
    if zscore_current:
        context["zscore_details"] = zscore_current
    
    if volume_stats:
        context["volume_statistics"] = volume_stats
    
    if volatility_stats:
        context["volatility_statistics"] = volatility_stats
    
    return context


def _build_realtime_context(
    asset: str,
    asset_display: str,
    simulation_day: str,
    window_size: int,
    zscore_threshold: float,
    progress_pct: float,
    points_streamed: int,
    total_points: int,
    anomalies_found: int,
    anomaly_list: List[Dict[str, Any]],
    current_price: Optional[float] = None,
    current_zscore: Optional[float] = None,
    window_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build context for the Real-time IoT Simulation page.
    
    Args:
        asset: Internal asset key
        asset_display: Display name
        simulation_day: The day being simulated
        window_size: Sliding window size
        zscore_threshold: Anomaly detection threshold
        progress_pct: Simulation progress percentage
        points_streamed: Number of points processed
        total_points: Total points in the day
        anomalies_found: Count of anomalies detected
        anomaly_list: List of anomaly details
        current_price: Current price (optional)
        current_zscore: Current Z-score (optional)
        window_stats: Window statistics (optional)
    
    Returns:
        Complete context dictionary
    """
    context = {
        "page": "Real-time IoT Simulation",
        "asset": asset,
        "asset_display": asset_display,
        "granularity": "minute",
        "period": {
            "day": simulation_day,
        },
        "simulation": {
            "window_size": window_size,
            "zscore_threshold": zscore_threshold,
            "progress_pct": f"{progress_pct:.1f}%",
            "points_streamed": points_streamed,
            "total_points": total_points,
            "current_price": f"${current_price:.2f}" if current_price else "N/A",
            "current_zscore": f"{current_zscore:.2f}σ" if current_zscore else "N/A"
        },
        "realtime_anomalies": anomaly_list,
        "anomalies": {
            "counts": {"total": anomalies_found},
            "details": anomaly_list[:10]
        }
    }
    
    if window_stats:
        context["window_statistics"] = window_stats
    
    return context


def _build_cross_asset_context(
    start_date: str,
    end_date: str,
    correlation_matrix: Dict[str, float],
    systemic_events: Dict[str, Any],
    pair_name: Optional[str] = None,
    pair_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build context for the Cross-Asset Analysis page.
    
    Args:
        start_date: Start date
        end_date: End date
        correlation_matrix: Dictionary of asset pair correlations
        systemic_events: Systemic event statistics
        pair_name: Selected pair name (optional)
        pair_analysis: Detailed pair analysis (optional)
    
    Returns:
        Complete context dictionary
    """
    context = {
        "page": "Cross-Asset Analysis",
        "asset": "Multiple",
        "asset_display": "All Assets",
        "granularity": "daily",
        "period": {
            "start": start_date,
            "end": end_date
        },
        "correlations": {
            "matrix": correlation_matrix
        },
        "systemic_events": systemic_events
    }
    
    if pair_name and pair_analysis:
        context["pair_analysis"] = {
            "pair": pair_name,
            **pair_analysis
        }
    
    return context


def _build_pattern_context(
    asset: str,
    asset_display: str,
    start_date: str,
    end_date: str,
    candlestick_counts: Dict[str, int],
    chart_patterns: List[Dict[str, Any]],
    pattern_distribution: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Build context for the Pattern Recognition page.
    
    Args:
        asset: Internal asset key
        asset_display: Display name
        start_date: Start date
        end_date: End date
        candlestick_counts: Counts of each candlestick pattern
        chart_patterns: List of detected chart patterns
        pattern_distribution: Pattern frequency distribution (optional)
    
    Returns:
        Complete context dictionary
    """
    context = {
        "page": "Pattern Recognition",
        "asset": asset,
        "asset_display": asset_display,
        "granularity": "daily",
        "period": {
            "start": start_date,
            "end": end_date
        },
        "candlestick_patterns": candlestick_counts,
        "chart_patterns": chart_patterns
    }
    
    if pattern_distribution:
        context["pattern_distribution"] = pattern_distribution
    
    return context

def filter_context(
    full_context: Dict[str, Any], 
    selected_keys: List[str]
) -> Dict[str, Any]:
    """
    Filter the full context to include only selected data sections.
    
    Always includes: page, asset, period, basic info
    Conditionally includes: detailed statistics based on selection
    
    Args:
        full_context: Complete page context dictionary
        selected_keys: List of selected data option keys
    
    Returns:
        Filtered context dictionary
    """
    # Base context always included
    filtered = {
        "page": full_context.get("page", "Unknown"),
        "asset": full_context.get("asset", "Unknown"),
        "asset_display": full_context.get("asset_display", "Unknown"),
        "granularity": full_context.get("granularity", "daily"),
        "period": full_context.get("period", {}),
    }
    
    # Map of keys to context fields
    key_mapping = {
        "price_stats": "price_statistics",
        "anomalies": "anomalies",
        "zscore_details": "zscore_details",
        "volume_stats": "volume_statistics",
        "volatility_stats": "volatility_statistics",
        "simulation_progress": "simulation",
        "realtime_anomalies": "realtime_anomalies",
        "window_stats": "window_statistics",
        "correlation_matrix": "correlations",
        "systemic_events": "systemic_events",
        "pair_analysis": "pair_analysis",
        "candlestick_patterns": "candlestick_patterns",
        "chart_patterns": "chart_patterns",
        "pattern_distribution": "pattern_distribution",
    }
    
    for key in selected_keys:
        if key in key_mapping:
            field = key_mapping[key]
            if field in full_context:
                filtered[field] = full_context[field]
    
    return filtered