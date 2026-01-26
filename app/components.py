"""
Reusable UI Components for IoT Financial Data Analytics.

This module provides standardized UI components used across all pages:
- Page header and footer
- Gemini AI chat sidebar with data attachment checkboxes
- Auto-scrolling chat messages (scrolls to question, not to end)

Key Features:
- render_gemini_sidebar(): Compact chat in sidebar with data selection
- Data-driven context (no image capture - more reliable for LLM analysis)
- Smart scroll: shows user question + start of response

Usage:
    from components import title, footer, render_gemini_sidebar
    
    with st.sidebar:
        render_gemini_sidebar(
            page_context=context_dict,
            page_type="single_asset"
        )
"""

import streamlit as st
import time
from typing import Any, Dict, List

from utils.logger import logger

# Import constants
try:
    import config
    PAGE_TYPE_SINGLE_ASSET = config.PAGE_TYPE_SINGLE_ASSET
    PAGE_TYPE_REALTIME = config.PAGE_TYPE_REALTIME
    PAGE_TYPE_CROSS_ASSET = config.PAGE_TYPE_CROSS_ASSET
    PAGE_TYPE_PATTERNS = config.PAGE_TYPE_PATTERNS
    CHAT_MESSAGES_HEIGHT = config.CHAT_MESSAGES_HEIGHT
    MAX_CHARTS_IN_CONTEXT = config.MAX_CHARTS_IN_CONTEXT
    INITIAL_MESSAGE_DISPLAY_COUNT = config.INITIAL_MESSAGE_DISPLAY_COUNT
    MAX_ANOMALIES_IN_CONTEXT = config.MAX_ANOMALIES_IN_CONTEXT
    MAX_PATTERNS_IN_CONTEXT = config.MAX_PATTERNS_IN_CONTEXT
    MAX_SYSTEMIC_EVENTS_DISPLAY = config.MAX_SYSTEMIC_EVENTS_DISPLAY
    DEFAULT_CHART_CONFIG = config.DEFAULT_CHART_CONFIG
except ImportError:
    # Fallback if constants not available
    PAGE_TYPE_SINGLE_ASSET = "single_asset"
    PAGE_TYPE_REALTIME = "realtime"
    PAGE_TYPE_CROSS_ASSET = "cross_asset"
    PAGE_TYPE_PATTERNS = "patterns"
    CHAT_MESSAGES_HEIGHT = 300
    MAX_CHARTS_IN_CONTEXT = 5
    INITIAL_MESSAGE_DISPLAY_COUNT = 20
    MAX_ANOMALIES_IN_CONTEXT = 10
    MAX_PATTERNS_IN_CONTEXT = 10
    MAX_SYSTEMIC_EVENTS_DISPLAY = 5
    DEFAULT_CHART_CONFIG = {'displayModeBar': False}
    INITIAL_MESSAGE_DISPLAY_COUNT = 20
    MAX_ANOMALIES_IN_CONTEXT = 10
    MAX_PATTERNS_IN_CONTEXT = 10
    MAX_SYSTEMIC_EVENTS_DISPLAY = 5
    DEFAULT_CHART_CONFIG = {'displayModeBar': False}

# Import Gemini assistant module
try:
    from src.gemini_assistant import (
        get_assistant,
        get_gemini_status,
        is_gemini_available,
    )
    GEMINI_MODULE_AVAILABLE = True
except ImportError:
    GEMINI_MODULE_AVAILABLE = False


# =============================================================================
# BASIC PAGE COMPONENTS
# =============================================================================

def title(page_title: str, description: str) -> None:
    """
    Render a standardized page title with description.
    
    Args:
        page_title: The main title to display
        description: A brief description shown below the title
    """
    st.title(page_title)
    st.markdown(description)


def footer(page_title: str) -> None:
    """
    Render a standardized page footer.
    
    Args:
        page_title: The page name to display in the footer
    """
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray;'>
        {page_title} | IoT & Data Analytics Project
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# DATA ATTACHMENT CONFIGURATION
# =============================================================================

# Available data options for each page type
DATA_OPTIONS = {
    PAGE_TYPE_SINGLE_ASSET: {
        "price_stats": {
            "label": "Statistiche prezzo",
            "description": "Min, max, current, % change",
            "default": True
        },
        "anomalies": {
            "label": "⚠️ Lista anomalie",
            "description": "All detected anomalies with details",
            "default": True
        },
        "zscore_details": {
            "label": "Dettagli Z-Score",
            "description": "Current Z-scores for price, volume, volatility",
            "default": False
        },
        "volume_stats": {
            "label": "Statistiche volume",
            "description": "Volume statistics and trends",
            "default": False
        },
        "volatility_stats": {
            "label": "Statistiche volatilità",
            "description": "Volatility range and patterns",
            "default": False
        }
    },
    PAGE_TYPE_REALTIME: {
        "simulation_progress": {
            "label": "Progresso simulazione",
            "description": "Current progress and points streamed",
            "default": True
        },
        "realtime_anomalies": {
            "label": "⚠️ Anomalie rilevate",
            "description": "Anomalies found during simulation",
            "default": True
        },
        "window_stats": {
            "label": "Statistiche finestra",
            "description": "Rolling window statistics",
            "default": False
        }
    },
    PAGE_TYPE_CROSS_ASSET: {
        "correlation_matrix": {
            "label": "🔗 Matrice correlazioni",
            "description": "Full correlation matrix between assets",
            "default": True
        },
        "systemic_events": {
            "label": "🌐 Eventi sistemici",
            "description": "Days with multiple asset anomalies",
            "default": True
        },
        "pair_analysis": {
            "label": "Analisi coppia",
            "description": "Selected pair detailed statistics",
            "default": False
        }
    },
    PAGE_TYPE_PATTERNS: {
        "candlestick_patterns": {
            "label": "🕯️ Pattern candlestick",
            "description": "Doji, Hammer, Engulfing patterns",
            "default": True
        },
        "chart_patterns": {
            "label": "Pattern grafici",
            "description": "Double Top/Bottom, H&S, Cup & Handle",
            "default": True
        },
        "pattern_distribution": {
            "label": "Distribuzione",
            "description": "Pattern frequency and timeline",
            "default": False
        }
    }
}

# Page-specific suggested questions for welcome message
PAGE_SUGGESTIONS = {
    PAGE_TYPE_SINGLE_ASSET: [
        "Cosa significa Z-score?",
        "Spiega le anomalie rilevate",
        "Analizza il trend del prezzo"
    ],
    PAGE_TYPE_REALTIME: [
        "Come funziona la sliding window?",
        "Spiega le anomalie in tempo reale",
        "Cosa indica la simulazione?"
    ],
    PAGE_TYPE_CROSS_ASSET: [
        "Spiega la matrice di correlazione",
        "Cosa sono gli eventi sistemici?",
        "Analizza le relazioni tra asset"
    ],
    PAGE_TYPE_PATTERNS: [
        "Cosa indica un pattern Doji?",
        "Spiega i pattern rilevati",
        "Qual è il segnale più importante?"
    ]
}


# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def init_gemini_session_state() -> None:
    """
    Initialize Streamlit session state variables for Gemini chat.
    
    Creates:
    - gemini_history: List of chat messages
    - gemini_input_key: Counter for input widget key regeneration
    - gemini_selected_data: Dict of selected data options per page
    - gemini_display_count: Number of messages to display (lazy loading)
    - gemini_selected_charts: Dict of selected chart figures
    - gemini_chart_order: List of chart IDs in selection order
    """
    if "gemini_history" not in st.session_state:
        st.session_state.gemini_history = []
    
    if "gemini_input_key" not in st.session_state:
        st.session_state.gemini_input_key = 0
    
    if "gemini_selected_data" not in st.session_state:
        st.session_state.gemini_selected_data = {}
    
    if "gemini_display_count" not in st.session_state:
        st.session_state.gemini_display_count = INITIAL_MESSAGE_DISPLAY_COUNT
    
    if "gemini_selected_charts" not in st.session_state:
        st.session_state.gemini_selected_charts = {}
    
    if "gemini_chart_order" not in st.session_state:
        st.session_state.gemini_chart_order = []


def get_selected_data_options(page_type: str) -> Dict[str, bool]:
    """
    Get the currently selected data options for a page type.
    
    Args:
        page_type: One of 'single_asset', 'realtime', 'cross_asset', 'patterns'
    
    Returns:
        Dictionary mapping option keys to boolean (selected/not selected)
    """
    init_gemini_session_state()
    
    if page_type not in st.session_state.gemini_selected_data:
        # Initialize with defaults
        if page_type in DATA_OPTIONS:
            st.session_state.gemini_selected_data[page_type] = {
                key: opt["default"] 
                for key, opt in DATA_OPTIONS[page_type].items()
            }
        else:
            st.session_state.gemini_selected_data[page_type] = {}
    
    return st.session_state.gemini_selected_data[page_type]


# =============================================================================
# CHART SELECTION MANAGEMENT
# =============================================================================

def add_chart_to_context(chart_id: str, figure: Any, label: str, page: str) -> None:
    """
    Aggiungi un chart al contesto del chatbot.
    
    Args:
        chart_id: Identificatore unico (es. "single_asset_main_candle")
        figure: Oggetto Plotly Figure
        label: Nome human-readable (es. "Candlestick - BTC Daily")
        page: Nome pagina (es. "single_asset")
    """
    init_gemini_session_state()
    
    # Aggiungi al dict
    st.session_state.gemini_selected_charts[chart_id] = {
        "figure": figure,
        "label": label,
        "page": page,
        "timestamp": time.time()
    }
    
    # Aggiungi all'ordine se non presente
    if chart_id not in st.session_state.gemini_chart_order:
        st.session_state.gemini_chart_order.append(chart_id)


def remove_chart_from_context(chart_id: str) -> None:
    """Rimuovi un chart dal contesto."""
    init_gemini_session_state()
    
    if chart_id in st.session_state.gemini_selected_charts:
        del st.session_state.gemini_selected_charts[chart_id]
    
    if chart_id in st.session_state.gemini_chart_order:
        st.session_state.gemini_chart_order.remove(chart_id)


def is_chart_in_context(chart_id: str) -> bool:
    """Verifica se un chart è già nel contesto."""
    init_gemini_session_state()
    return chart_id in st.session_state.gemini_selected_charts


def get_selected_chart_figures() -> List:
    """
    Ottieni lista di Figure objects nell'ordine di selezione.
    
    Returns:
        Lista di Plotly Figure objects
    """
    init_gemini_session_state()
    
    figures = []
    for chart_id in st.session_state.gemini_chart_order:
        chart_data = st.session_state.gemini_selected_charts.get(chart_id)
        if chart_data and "figure" in chart_data:
            figures.append(chart_data["figure"])
    
    return figures


def clear_all_charts() -> None:
    """Rimuovi tutti i chart dal contesto."""
    init_gemini_session_state()
    st.session_state.gemini_selected_charts = {}
    st.session_state.gemini_chart_order = []


# =============================================================================
# AUTO-SCROLL JAVASCRIPT
# =============================================================================

def inject_auto_scroll_js(anchor_id: str) -> None:
    """
    Execute surgical scroll using components.html (iframe)
    to manipulate parent DOM (window.parent).
    Uses getBoundingClientRect for precise positioning.
    Waits for Streamlit's scroll to stop before applying our scroll.
    """
    import streamlit.components.v1 as components
    
    js_code = f"""
    <script>
        (function() {{
            // Find scrollable parent by walking up the DOM tree
            function getScrollParent(node) {{
                if (!node) return null;
                
                let current = node.parentElement;
                while (current) {{
                    const style = window.parent.getComputedStyle(current);
                    if (style.overflowY === 'auto' || style.overflowY === 'scroll') {{
                        return current;
                    }}
                    current = current.parentElement;
                }}
                return null;
            }}

            function performScroll() {{
                const anchor = window.parent.document.getElementById('{anchor_id}');
                if (!anchor) return;

                const container = getScrollParent(anchor);
                if (!container) return;

                const anchorRect = anchor.getBoundingClientRect();
                const containerRect = container.getBoundingClientRect();
                const relativeTop = anchorRect.top - containerRect.top;
                
                // Scroll to position the anchor 5px from top of container
                container.scrollTop += (relativeTop - 5);
            }}

            // Wait for container to exist, then observe when scrolling stops
            function waitForContainer() {{
                const anchor = window.parent.document.getElementById('{anchor_id}');
                if (!anchor) {{
                    setTimeout(waitForContainer, 50);
                    return;
                }}

                const container = getScrollParent(anchor);
                if (!container) {{
                    setTimeout(waitForContainer, 50);
                    return;
                }}

                // Listen to scroll events and wait for them to stop (debounce)
                let scrollTimeout;
                const scrollHandler = function() {{
                    clearTimeout(scrollTimeout);
                    scrollTimeout = setTimeout(() => {{
                        // Scroll has stopped for 150ms - now apply our scroll
                        container.removeEventListener('scroll', scrollHandler);
                        performScroll();
                    }}, 150);
                }};

                container.addEventListener('scroll', scrollHandler);
                
                // Trigger initial check in case Streamlit hasn't scrolled yet
                setTimeout(() => {{
                    if (scrollTimeout === undefined) {{
                        // No scroll detected after 200ms, scroll immediately
                        container.removeEventListener('scroll', scrollHandler);
                        performScroll();
                    }}
                }}, 200);
            }}

            waitForContainer();
        }})();
    </script>
    """
    
    # components.html creates iframe that ALWAYS executes JS
    components.html(js_code, height=0, width=0)


# =============================================================================
# GEMINI SIDEBAR COMPONENTS
# =============================================================================

def render_gemini_header() -> None:
    """Render the Gemini chat header with status indicator."""
    
    st.markdown("### Gemini Assistant")


def render_status_badge() -> None:
    """Render a compact status badge showing Gemini API availability."""
    if not GEMINI_MODULE_AVAILABLE:
        st.caption("❌ Module not available")
        return
    
    status = get_gemini_status() # type: ignore
    
    if not status["library_installed"]:
        st.caption("❌ Library missing")
    elif not status["api_key_set"]:
        st.caption("Mock mode - set API key")
    else:
        st.caption(f"✅ {status['model']}")


def render_chart_add_button(
    chart_id: str,
    figure: Any,
    label: str,
    page: str,
    position: str = "inline",
    disabled: bool = False
) -> None:
    """
    Renderizza bottone per aggiungere/rimuovere chart dal contesto chat.
    
    Args:
        chart_id: ID unico del chart
        figure: Figura Plotly da salvare
        label: Label descrittivo
        page: Nome pagina corrente
        position: "inline" (accanto al titolo) o "below" (sotto il chart)
        disabled: Se True, disabilita il bottone (es. grafico non pronto)
    
    Usage:
        # Prima di st.plotly_chart()
        render_chart_add_button("main_candle", fig, "Candlestick Chart", "single_asset")
        st.plotly_chart(fig)
    """
    init_gemini_session_state()
    
    is_added = is_chart_in_context(chart_id)
    
    # Limit: max charts
    num_charts = len(st.session_state.gemini_chart_order)
    at_limit = num_charts >= MAX_CHARTS_IN_CONTEXT and not is_added
    
    if position == "inline":
        # Small button con icona
        if is_added:
            if st.button("✅", key=f"chart_btn_{chart_id}", help="Rimuovi dalla chat", type="secondary", disabled=disabled):
                remove_chart_from_context(chart_id)
                st.rerun()
        else:
            if at_limit:
                st.button("📎", key=f"chart_btn_{chart_id}", help=f"Massimo {MAX_CHARTS_IN_CONTEXT} chart. Rimuovine uno per aggiungerne altri.", disabled=True, type="secondary")
            elif disabled:
                st.button("📎", key=f"chart_btn_{chart_id}", help="Grafico non ancora pronto. Completa la simulazione prima di aggiungere.", disabled=True, type="secondary")
            else:
                if st.button("📎", key=f"chart_btn_{chart_id}", help="Aggiungi alla chat Gemini", type="secondary"):
                    add_chart_to_context(chart_id, figure, label, page)
                    st.rerun()
    else:  # below
        # Full-width button sotto il chart
        if is_added:
            if st.button(f"✅ {label} - Rimuovi dalla chat", key=f"chart_btn_{chart_id}", width='stretch', type="secondary", disabled=disabled):
                remove_chart_from_context(chart_id)
                st.rerun()
        else:
            if at_limit:
                st.button(f"📎 {label} - Massimo {MAX_CHARTS_IN_CONTEXT} chart raggiunti", key=f"chart_btn_{chart_id}", width='stretch', disabled=True, type="secondary")
            elif disabled:
                st.button(f"📎 {label} - Non pronto", key=f"chart_btn_{chart_id}", width='stretch', disabled=True, type="secondary")
            else:
                if st.button(f"📎 Aggiungi {label} alla chat", key=f"chart_btn_{chart_id}", width='stretch', type="secondary"):
                    add_chart_to_context(chart_id, figure, label, page)
                    st.rerun()


def render_welcome_message(page_type: str = "single_asset") -> None:
    """
    Render welcome message with page-specific quick suggestions.
    
    Args:
        page_type: Current page type for appropriate suggestions
    """
    st.markdown("""
    <div style='text-align: center; padding: 10px; color: #666;'>
        <div style='font-size: 24px; margin-bottom: 8px;'></div>
        <div style='font-size: 13px;'>
            Chiedimi qualsiasi cosa sui dati!<br>
            Seleziona i dati da includere nel menu sotto.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**💡 Prova:**")
    
    # Get page-specific suggestions
    suggestions = PAGE_SUGGESTIONS.get(page_type, PAGE_SUGGESTIONS[PAGE_TYPE_SINGLE_ASSET])
    
    for suggestion in suggestions:
        if st.button(
            f"› {suggestion}", 
            key=f"sug_{page_type}_{hash(suggestion)}", 
            width='stretch',
            type="secondary"
        ):
            st.session_state.gemini_pending_question = suggestion
            st.rerun()


def render_chat_messages(page_type: str = "single_asset") -> None:
    """
    Render all messages in a scrollable container with smart auto-scroll.
    
    Args:
        page_type: Current page type for welcome message suggestions
    """
    import time
    import streamlit.components.v1 as components
    
    history = st.session_state.gemini_history
    
    if not history:
        render_welcome_message(page_type)
        return
    
    # Find index of last user message
    last_user_index = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role", "user") == "user":
            last_user_index = i
            break

    # Generate unique ID (timestamp) to force fresh anchor
    unique_id = int(time.time() * 1000)
    anchor_id = f"msg_anchor_{unique_id}"

    # Scrollable container
    messages_container = st.container(height=CHAT_MESSAGES_HEIGHT)
    
    with messages_container:
        for i, msg in enumerate(history):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Place anchor BEFORE last user message
            if i == last_user_index:
                st.markdown(f'<div id="{anchor_id}" style="height:1px;"></div>', unsafe_allow_html=True)

            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)
                    if msg.get("data_included"):
                        st.caption(f"📎 {', '.join(msg['data_included'])}")
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(content)
    
    # Inject scroll script via iframe
    inject_auto_scroll_js(anchor_id)


def render_selected_charts_list() -> None:
    """
    Renderizza lista compatta dei chart selezionati.
    Mostra in un expander collassato per non occupare spazio.
    """
    init_gemini_session_state()
    
    num_charts = len(st.session_state.gemini_chart_order)
    
    if num_charts == 0:
        return  # Non mostrare niente se nessun chart
    
    with st.expander(f"📊 Chart allegati ({num_charts})", expanded=False):
        for chart_id in st.session_state.gemini_chart_order:
            chart_data = st.session_state.gemini_selected_charts.get(chart_id)
            if not chart_data:
                continue
            
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.caption(chart_data["label"])
            with col2:
                if st.button("🗑️", key=f"remove_{chart_id}", help="Rimuovi"):
                    remove_chart_from_context(chart_id)
                    st.rerun()
        
        st.markdown("---")
        if st.button("🗑️ Rimuovi tutti", key="clear_all_charts", width='stretch'):
            clear_all_charts()
            st.rerun()


def render_data_selection(page_type: str) -> List[str]:
    """
    Render data selection checkboxes in an expander.
    
    Args:
        page_type: The current page type for appropriate options
    
    Returns:
        List of selected data option keys
    """
    if page_type not in DATA_OPTIONS:
        return []
    
    options = DATA_OPTIONS[page_type]
    selected = get_selected_data_options(page_type)
    
    with st.expander("📎 Dati da allegare", expanded=False):
        for key, opt in options.items():
            default_value = opt["default"] if opt["default"] is not None else False
            new_value = st.checkbox(
                opt["label"],
                value=selected.get(key, default_value),
                key=f"data_opt_{page_type}_{key}",
                help=opt["description"]
            )
            st.session_state.gemini_selected_data[page_type][key] = new_value
        
        st.markdown("---")
        
        if st.button("🗑️ Pulisci chat", key="gem_clear", use_container_width=True):
            st.session_state.gemini_history = []
            if GEMINI_MODULE_AVAILABLE:
                get_assistant().clear_history() # type: ignore
            st.rerun()
    
    return [
        key for key, is_selected 
        in st.session_state.gemini_selected_data[page_type].items() 
        if is_selected
    ]


def render_chat_input(page_context: Dict[str, Any], selected_data: List[str]) -> None:
    """
    Render chat input and handle message submission.
    
    Args:
        page_context: Full page context dictionary
        selected_data: List of selected data option keys to include
    """
    if "gemini_pending_question" in st.session_state:
        pending = st.session_state.gemini_pending_question
        del st.session_state.gemini_pending_question
        _process_user_message(pending, page_context, selected_data)
        return
    
    user_input = st.chat_input(
        placeholder="Scrivi una domanda...",
        key=f"gem_input_{st.session_state.gemini_input_key}"
    )
    
    if user_input:
        _process_user_message(user_input, page_context, selected_data)


def _process_user_message(
    user_input: str, 
    page_context: Dict[str, Any], 
    selected_data: List[str]
) -> None:
    """
    Process a user message and get response from Gemini.
    
    Args:
        user_input: The user's message text
        page_context: Full page context dictionary
        selected_data: List of selected data keys to include
    """
    if not GEMINI_MODULE_AVAILABLE:
        logger.error("Gemini module not available")
        st.error("Gemini module not available")
        return
    
    filtered_context = _filter_context(page_context, selected_data)
    
    page_type = page_context.get("page_type", "single_asset")
    included_names = []
    if page_type in DATA_OPTIONS:
        for key in selected_data:
            if key in DATA_OPTIONS[page_type]:
                included_names.append(DATA_OPTIONS[page_type][key]["label"])
    
    # Aggiungi info sui chart allegati
    selected_figures = get_selected_chart_figures()
    num_charts = len(selected_figures)
    if num_charts > 0:
        included_names.append(f"📊 {num_charts} chart")
    
    user_message = {
        "role": "user",
        "content": user_input,
        "data_included": included_names
    }
    st.session_state.gemini_history.append(user_message)
    
    assistant = get_assistant() # type: ignore
    
    with st.spinner("Gemini sta pensando..."):
        response = assistant.send_message(
            question=user_input,
            page_context=filtered_context,
            history=st.session_state.gemini_history[:-1],
            chart_figures=selected_figures if num_charts > 0 else None
        )
    
    assistant_message = {
        "role": "assistant",
        "content": response,
        "data_included": []
    }
    st.session_state.gemini_history.append(assistant_message)
    
    assistant.add_to_history("user", user_input)
    assistant.add_to_history("assistant", response)
    
    st.session_state.gemini_input_key += 1
    
    st.rerun()


def _filter_context(
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


# =============================================================================
# MAIN SIDEBAR RENDERING FUNCTION
# =============================================================================

def render_gemini_sidebar(
    page_context: Dict[str, Any],
    page_type: str = "single_asset"
) -> None:
    """
    Render the Gemini chat interface inside the sidebar.
    
    Layout order:
    1. Header + status
    2. Chat messages (with page-specific suggestions)
    3. Selected charts list (compact expander)
    4. Text input (above options for natural flow)
    5. Data selection expander (settings, used less frequently)
    
    Args:
        page_context: Dictionary with current page information and data.
        page_type: Type of page for appropriate data options.
            One of: 'single_asset', 'realtime', 'cross_asset', 'patterns'
    """
    init_gemini_session_state()
    
    # 1. Header with status indicator
    render_gemini_header()
    render_status_badge()
    
    # 2. Chat messages (scrollable, with page-specific suggestions)
    render_chat_messages(page_type)
    
    # 2.5 Selected charts list (compact)
    render_selected_charts_list()
    
    # 3. Store page_type in context
    page_context["page_type"] = page_type
    
    # Get current selections from session state
    init_selections = get_selected_data_options(page_type)
    current_selections = [k for k, v in init_selections.items() if v]
    
    # 4. Text input (ABOVE the expander)
    render_chat_input(page_context, current_selections)
    
    # 5. Data selection expander (below input)
    render_data_selection(page_type)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_chat_history() -> list:
    """Get the current chat history from session state."""
    init_gemini_session_state()
    return st.session_state.gemini_history.copy()


def clear_chat_history() -> None:
    """Clear the chat history in session state."""
    init_gemini_session_state()
    st.session_state.gemini_history = []
    
    if GEMINI_MODULE_AVAILABLE:
        get_assistant().clear_history() # type: ignore
