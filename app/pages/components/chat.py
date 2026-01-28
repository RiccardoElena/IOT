import streamlit as st
from .state import init_gemini_session_state, GEMINI_MODULE_AVAILABLE, gemini_status, get_selected_data_options
from config import (PAGE_TYPE_SINGLE_ASSET, PAGE_SUGGESTIONS, CHAT_MESSAGES_HEIGHT, DATA_OPTIONS)
from utils import inject_auto_scroll_js
from .attachment import get_selected_chart_figures, render_selected_charts_list, render_data_selection
from typing import Any, Dict, List

from utils.logger import logger
from src.gemini_assistant import (
    get_assistant,
    filter_context
)


# =============================================================================
# CHAT RENDERING COMPONENTS
# =============================================================================

def _render_gemini_header() -> None:
    """Render the Gemini chat header with status indicator."""
    
    st.markdown("### Gemini Assistant")


def _render_status_badge() -> None:
    """Render a compact status badge showing Gemini API availability."""
    status = gemini_status
    if not status["library_installed"]:
        st.caption("❌ Library missing")
        return
    
    if not status["library_installed"]:
        st.caption("❌ Library missing")
    elif not status["api_key_set"]:
        st.caption("Mock mode - set API key")
    else:
        st.caption(f"✅ {status['model']}")



def _render_welcome_message(page_type: str = "single_asset") -> None:
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


def _render_chat_messages(page_type: str = "single_asset") -> None:
    """
    Render all messages in a scrollable container with smart auto-scroll.
    
    Args:
        page_type: Current page type for welcome message suggestions
    """
    import time
    import streamlit.components.v1 as components
    
    history = st.session_state.gemini_history
    
    if not history:
        _render_welcome_message(page_type)
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

def _render_chat_input(page_context: Dict[str, Any], selected_data: List[str]) -> None:
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
    
    filtered_context = filter_context(page_context, selected_data)
    
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


# =============================================================================
# MAIN SIDEBAR RENDERING FUNCTION
# =============================================================================

def render_chat(
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
    _render_gemini_header()
    _render_status_badge()
    
    # 2. Chat messages (scrollable, with page-specific suggestions)
    _render_chat_messages(page_type)
    
    # 2.5 Selected charts list (compact)
    render_selected_charts_list()
    
    # 3. Store page_type in context
    page_context["page_type"] = page_type
    
    # Get current selections from session state
    init_selections = get_selected_data_options(page_type)
    current_selections = [k for k, v in init_selections.items() if v]
    
    # 4. Text input (ABOVE the expander)
    _render_chat_input(page_context, current_selections)
    
    # 5. Data selection expander (below input)
    render_data_selection(page_type)

