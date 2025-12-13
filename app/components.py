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
from typing import Any, Dict, List

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
    "single_asset": {
        "price_stats": {
            "label": "📈 Statistiche prezzo",
            "description": "Min, max, current, % change",
            "default": True
        },
        "anomalies": {
            "label": "⚠️ Lista anomalie",
            "description": "All detected anomalies with details",
            "default": True
        },
        "zscore_details": {
            "label": "📊 Dettagli Z-Score",
            "description": "Current Z-scores for price, volume, volatility",
            "default": False
        },
        "volume_stats": {
            "label": "📊 Statistiche volume",
            "description": "Volume statistics and trends",
            "default": False
        },
        "volatility_stats": {
            "label": "📉 Statistiche volatilità",
            "description": "Volatility range and patterns",
            "default": False
        }
    },
    "realtime": {
        "simulation_progress": {
            "label": "⏱️ Progresso simulazione",
            "description": "Current progress and points streamed",
            "default": True
        },
        "realtime_anomalies": {
            "label": "⚠️ Anomalie rilevate",
            "description": "Anomalies found during simulation",
            "default": True
        },
        "window_stats": {
            "label": "📊 Statistiche finestra",
            "description": "Rolling window statistics",
            "default": False
        }
    },
    "cross_asset": {
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
            "label": "📊 Analisi coppia",
            "description": "Selected pair detailed statistics",
            "default": False
        }
    },
    "patterns": {
        "candlestick_patterns": {
            "label": "🕯️ Pattern candlestick",
            "description": "Doji, Hammer, Engulfing patterns",
            "default": True
        },
        "chart_patterns": {
            "label": "📈 Pattern grafici",
            "description": "Double Top/Bottom, H&S, Cup & Handle",
            "default": True
        },
        "pattern_distribution": {
            "label": "📊 Distribuzione",
            "description": "Pattern frequency and timeline",
            "default": False
        }
    }
}

# Page-specific suggested questions for welcome message
PAGE_SUGGESTIONS = {
    "single_asset": [
        "Cosa significa Z-score?",
        "Spiega le anomalie rilevate",
        "Analizza il trend del prezzo"
    ],
    "realtime": [
        "Come funziona la sliding window?",
        "Spiega le anomalie in tempo reale",
        "Cosa indica la simulazione?"
    ],
    "cross_asset": [
        "Spiega la matrice di correlazione",
        "Cosa sono gli eventi sistemici?",
        "Analizza le relazioni tra asset"
    ],
    "patterns": [
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
    """
    if "gemini_history" not in st.session_state:
        st.session_state.gemini_history = []
    
    if "gemini_input_key" not in st.session_state:
        st.session_state.gemini_input_key = 0
    
    if "gemini_selected_data" not in st.session_state:
        st.session_state.gemini_selected_data = {}


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
# AUTO-SCROLL JAVASCRIPT
# =============================================================================

def inject_auto_scroll_js(anchor_id: str) -> None:
    """
    Execute surgical scroll using components.html (iframe)
    to manipulate parent DOM (window.parent).
    Uses getBoundingClientRect for precise positioning.
    Keeps scrolling to override Streamlit's auto-scroll.
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

            function attemptScroll() {{
                const anchor = window.parent.document.getElementById('{anchor_id}');
                if (!anchor) return false;

                const container = getScrollParent(anchor);
                if (!container) return false;

                const anchorRect = anchor.getBoundingClientRect();
                const containerRect = container.getBoundingClientRect();
                const relativeTop = anchorRect.top - containerRect.top;
                
                if (Math.abs(relativeTop) > 5) {{
                    container.scrollTop += (relativeTop - 5);
                }}
                
                return true;
            }}

            // Keep scrolling for 1.5 seconds to OVERRIDE Streamlit's auto-scroll
            // Don't stop on success - Streamlit might scroll again after us
            let attempts = 0;
            const interval = setInterval(function() {{
                attemptScroll();  // Always execute
                attempts++;
                if (attempts > 30) {{  // 30 * 50ms = 1.5 seconds
                    clearInterval(interval);
                }}
            }}, 50);
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
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown("### ✨ Gemini Assistant")
    
    with col2:
        if GEMINI_MODULE_AVAILABLE:
            if is_gemini_available():
                st.markdown("🟢")
            else:
                st.markdown("🟡")
        else:
            st.markdown("🔴")


def render_status_badge() -> None:
    """Render a compact status badge showing Gemini API availability."""
    if not GEMINI_MODULE_AVAILABLE:
        st.caption("❌ Module not available")
        return
    
    status = get_gemini_status()
    
    if not status["library_installed"]:
        st.caption("❌ Library missing")
    elif not status["api_key_set"]:
        st.caption("🔧 Mock mode - set API key")
    else:
        st.caption(f"✅ {status['model']}")


def render_welcome_message(page_type: str = "single_asset") -> None:
    """
    Render welcome message with page-specific quick suggestions.
    
    Args:
        page_type: Current page type for appropriate suggestions
    """
    st.markdown("""
    <div style='text-align: center; padding: 10px; color: #666;'>
        <div style='font-size: 24px; margin-bottom: 8px;'>✨</div>
        <div style='font-size: 13px;'>
            Chiedimi qualsiasi cosa sui dati!<br>
            Seleziona i dati da includere nel menu sotto.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**💡 Prova:**")
    
    # Get page-specific suggestions
    suggestions = PAGE_SUGGESTIONS.get(page_type, PAGE_SUGGESTIONS["single_asset"])
    
    for suggestion in suggestions:
        if st.button(
            f"› {suggestion}", 
            key=f"sug_{page_type}_{hash(suggestion)}", 
            use_container_width=True,
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
    messages_container = st.container(height=300)
    
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
                with st.chat_message("assistant", avatar="✨"):
                    st.markdown(content)
    
    # Inject scroll script via iframe
    inject_auto_scroll_js(anchor_id)


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
            new_value = st.checkbox(
                opt["label"],
                value=selected.get(key, opt["default"]),
                key=f"data_opt_{page_type}_{key}",
                help=opt["description"]
            )
            st.session_state.gemini_selected_data[page_type][key] = new_value
        
        st.markdown("---")
        
        if st.button("🗑️ Pulisci chat", key="gem_clear", use_container_width=True):
            st.session_state.gemini_history = []
            if GEMINI_MODULE_AVAILABLE:
                get_assistant().clear_history()
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
        st.error("Gemini module not available")
        return
    
    filtered_context = _filter_context(page_context, selected_data)
    
    page_type = page_context.get("page_type", "single_asset")
    included_names = []
    if page_type in DATA_OPTIONS:
        for key in selected_data:
            if key in DATA_OPTIONS[page_type]:
                included_names.append(DATA_OPTIONS[page_type][key]["label"])
    
    user_message = {
        "role": "user",
        "content": user_input,
        "data_included": included_names
    }
    st.session_state.gemini_history.append(user_message)
    
    assistant = get_assistant()
    
    with st.spinner("✨ Gemini sta pensando..."):
        response = assistant.send_message(
            question=user_input,
            page_context=filtered_context,
            history=st.session_state.gemini_history[:-1]
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
    3. Text input (above options for natural flow)
    4. Data selection expander (settings, used less frequently)
    
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
        get_assistant().clear_history()
