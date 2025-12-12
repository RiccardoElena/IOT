"""
Reusable UI Components for IoT Financial Data Analytics.

This module provides standardized UI components used across all pages:
- Page header and footer
- Gemini AI chat sidebar (compact version integrated in left sidebar)
- Chart rendering with capture-to-Gemini functionality
- Auto-scrolling chat messages
- CSS and JavaScript injection for custom behavior

Key Features:
- render_gemini_sidebar(): Compact chat in sidebar
- render_chart_with_capture(): Chart with "Send to Gemini" button
- Auto-scroll to latest message in chat

Usage:
    from components import (
        title, 
        footer, 
        render_gemini_sidebar,
        render_chart_with_capture
    )
    
    # Render chart with capture button
    render_chart_with_capture(fig, "Price Chart", "main_chart")
    
    # In sidebar
    with st.sidebar:
        render_gemini_sidebar(page_context)
"""

import streamlit as st
from typing import Any, Dict, Optional
import streamlit.components.v1 as components

# Import Gemini assistant module
try:
    from src.gemini_assistant import (
        get_assistant,
        capture_plotly_figure,
        get_gemini_status,
        is_gemini_available,
    )
    GEMINI_MODULE_AVAILABLE = True
except ImportError:
    GEMINI_MODULE_AVAILABLE = False

# Import config for styling constants
try:
    import config
except ImportError:
    config = None


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
# CHART WITH CAPTURE FUNCTIONALITY
# =============================================================================

def render_chart_with_capture(
    fig,
    chart_name: str,
    chart_key: str,
    height: Optional[int] = None
) -> None:
    """
    Render a Plotly chart with a "Send to Gemini" capture button.
    
    Displays the chart with a button that captures it as an image
    and stores it in session state for sending to Gemini.
    
    Args:
        fig: Plotly figure object to render
        chart_name: Human-readable name for the chart (e.g., "Price Chart")
        chart_key: Unique key for the button widget
        height: Optional height override for the chart
    
    Example:
        >>> fig = go.Figure(...)
        >>> render_chart_with_capture(fig, "Main Price Chart", "main_chart")
    """
    # Initialize session state for pending image if needed
    if "gemini_pending_image" not in st.session_state:
        st.session_state.gemini_pending_image = None
    if "gemini_pending_image_name" not in st.session_state:
        st.session_state.gemini_pending_image_name = None
    
    # Create columns: chart takes most space, button on the right
    col_chart, col_btn = st.columns([20, 3])
    
    with col_chart:
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_key}")
    
    with col_btn:
        # Add some vertical spacing to align with chart
        st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
        
        # Check if this chart is already captured
        is_captured = (
            st.session_state.gemini_pending_image is not None and
            st.session_state.gemini_pending_image_name == chart_name
        )
        
        if is_captured:
            # Show "captured" state with option to remove
            if st.button(
                "✅ Allegato",
                key=f"capture_done_{chart_key}",
                help="Clicca per rimuovere",
                use_container_width=True,
                type="primary"
            ):
                st.session_state.gemini_pending_image = None
                st.session_state.gemini_pending_image_name = None
                st.rerun()
        else:
            # Show capture button
            if st.button(
                "📷 Gemini",
                key=f"capture_{chart_key}",
                help=f"Invia '{chart_name}' a Gemini",
                use_container_width=True,
                type="secondary"
            ):
                if GEMINI_MODULE_AVAILABLE:
                    with st.spinner("Cattura..."):
                        image_b64 = capture_plotly_figure(fig)
                        if image_b64:
                            st.session_state.gemini_pending_image = image_b64
                            st.session_state.gemini_pending_image_name = chart_name
                            st.toast(f"📷 {chart_name} catturato!", icon="✅")
                            st.rerun()
                        else:
                            st.toast("Errore nella cattura", icon="❌")
                else:
                    st.toast("Modulo Gemini non disponibile", icon="❌")


def clear_captured_chart() -> None:
    """
    Clear any captured chart from session state.
    
    Call this after the image has been sent to Gemini.
    """
    st.session_state.gemini_pending_image = None
    st.session_state.gemini_pending_image_name = None


# =============================================================================
# GEMINI SIDEBAR CHAT COMPONENTS
# =============================================================================

def init_gemini_session_state() -> None:
    """
    Initialize Streamlit session state variables for Gemini chat.
    
    Creates:
    - gemini_history: List of chat messages
    - gemini_pending_image: Base64 image waiting to be sent
    - gemini_pending_image_name: Name of the captured chart
    - gemini_input_key: Counter for input widget key regeneration
    """
    if "gemini_history" not in st.session_state:
        st.session_state.gemini_history = []
    
    if "gemini_pending_image" not in st.session_state:
        st.session_state.gemini_pending_image = None
    
    if "gemini_pending_image_name" not in st.session_state:
        st.session_state.gemini_pending_image_name = None
    
    if "gemini_input_key" not in st.session_state:
        st.session_state.gemini_input_key = 0


def inject_auto_scroll_js() -> None:
    """
    Inject JavaScript to auto-scroll chat to the latest message.
    
    This script finds the chat container and scrolls it to the bottom
    after each Streamlit rerun.
    """
    js_code = """
    <script>
        // Function to scroll chat container to bottom
        function scrollChatToBottom() {
            // Find all containers with data-testid containing 'stVerticalBlock'
            const containers = document.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
            
            containers.forEach(container => {
                // Check if this looks like a chat container (has scrollable content)
                if (container.scrollHeight > container.clientHeight) {
                    container.scrollTop = container.scrollHeight;
                }
            });
            
            // Also try to find the specific chat message container
            const chatContainers = document.querySelectorAll('.stChatMessageContainer');
            chatContainers.forEach(container => {
                const parent = container.closest('[data-testid="stVerticalBlockBorderWrapper"]');
                if (parent) {
                    parent.scrollTop = parent.scrollHeight;
                }
            });
        }
        
        // Run after a short delay to ensure DOM is updated
        setTimeout(scrollChatToBottom, 100);
        setTimeout(scrollChatToBottom, 300);
        setTimeout(scrollChatToBottom, 500);
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)


def render_gemini_header() -> None:
    """
    Render the Gemini chat header with status indicator.
    """
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
    """
    Render a compact status badge showing Gemini API availability.
    """
    if not GEMINI_MODULE_AVAILABLE:
        st.caption("❌ Modulo non disponibile")
        return
    
    status = get_gemini_status()
    
    if not status["library_installed"]:
        st.caption("❌ Libreria mancante")
    elif not status["api_key_set"]:
        st.caption("🔧 Mock mode - configura API key")
    else:
        st.caption(f"✅ {status['model']}")


def render_pending_image_indicator() -> None:
    """
    Render an indicator showing if a chart is ready to be sent.
    """
    if st.session_state.gemini_pending_image is not None:
        chart_name = st.session_state.gemini_pending_image_name or "Grafico"
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.info(f"📎 **{chart_name}** allegato")
        with col2:
            if st.button("✕", key="remove_pending_image", help="Rimuovi"):
                clear_captured_chart()
                st.rerun()


def render_welcome_message_compact() -> None:
    """
    Render a compact welcome message for the sidebar chat.
    """
    st.markdown("""
    <div style='text-align: center; padding: 10px; color: #666;'>
        <div style='font-size: 24px; margin-bottom: 8px;'>✨</div>
        <div style='font-size: 13px;'>
            Chiedimi qualsiasi cosa sui dati!<br>
            Usa 📷 accanto ai grafici per allegarli.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick suggestions
    st.markdown("**💡 Prova:**")
    
    suggestions = [
        "Cosa significa Z-score?",
        "Spiega i punti rossi",
        "Cos'è un'anomalia?"
    ]
    
    for suggestion in suggestions:
        if st.button(
            f"› {suggestion}", 
            key=f"sug_{hash(suggestion)}", 
            use_container_width=True,
            type="secondary"
        ):
            st.session_state.gemini_pending_question = suggestion
            st.rerun()


def render_chat_messages_compact() -> None:
    """
    Render all messages in a scrollable container with auto-scroll.
    """
    history = st.session_state.gemini_history
    
    if not history:
        render_welcome_message_compact()
        return
    
    # Create scrollable container for messages
    messages_container = st.container(height=280)
    
    with messages_container:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            has_image = msg.get("has_image", False)
            image_name = msg.get("image_name", "")
            
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)
                    if has_image:
                        st.caption(f"📷 {image_name}" if image_name else "📷 Grafico allegato")
            else:
                with st.chat_message("assistant", avatar="✨"):
                    st.markdown(content)
    
    # Inject auto-scroll JavaScript
    inject_auto_scroll_js()


def render_chat_input_compact(page_context: Dict[str, Any]) -> None:
    """
    Render compact chat input area (without capture button - moved to charts).
    
    Args:
        page_context: Dictionary with current page information
    """
    # Check for pending suggestion from button click
    if "gemini_pending_question" in st.session_state:
        pending = st.session_state.gemini_pending_question
        del st.session_state.gemini_pending_question
        _process_user_message(pending, page_context)
        return
    
    # Show pending image indicator
    render_pending_image_indicator()
    
    # Text input
    user_input = st.chat_input(
        placeholder="Scrivi una domanda...",
        key=f"gem_input_{st.session_state.gemini_input_key}"
    )
    
    if user_input:
        _process_user_message(user_input, page_context)


def _process_user_message(user_input: str, page_context: Dict[str, Any]) -> None:
    """
    Process a user message and get response from Gemini.
    
    Args:
        user_input: The user's message text
        page_context: Current page context dictionary
    """
    if not GEMINI_MODULE_AVAILABLE:
        st.error("Modulo Gemini non disponibile")
        return
    
    # Get pending image (if any)
    pending_image = st.session_state.gemini_pending_image
    pending_image_name = st.session_state.gemini_pending_image_name
    
    # Clear pending image BEFORE adding to history
    clear_captured_chart()
    
    # Add user message to history
    user_message = {
        "role": "user",
        "content": user_input,
        "has_image": pending_image is not None,
        "image_name": pending_image_name or ""
    }
    st.session_state.gemini_history.append(user_message)
    
    # Get assistant instance
    assistant = get_assistant()
    
    # Sync history
    assistant.set_history(st.session_state.gemini_history)
    
    # Send message and get response
    with st.spinner("✨ Gemini sta pensando..."):
        response = assistant.send_message(
            question=user_input,
            page_context=page_context,
            image_base64=pending_image,
            history=st.session_state.gemini_history[:-1]
        )
    
    # Add response to history
    assistant_message = {
        "role": "assistant",
        "content": response,
        "has_image": False,
        "image_name": ""
    }
    st.session_state.gemini_history.append(assistant_message)
    
    # Update assistant history
    assistant.add_to_history("user", user_input)
    assistant.add_to_history("assistant", response)
    
    # Reset input key to clear the input box
    st.session_state.gemini_input_key += 1
    
    st.rerun()


def render_chat_actions_compact() -> None:
    """
    Render compact action buttons for the sidebar chat.
    """
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Pulisci", key="gem_clear", use_container_width=True, type="secondary"):
            st.session_state.gemini_history = []
            clear_captured_chart()
            if GEMINI_MODULE_AVAILABLE:
                get_assistant().clear_history()
            st.rerun()
    
    with col2:
        if st.button("ℹ️ Stato", key="gem_status", use_container_width=True, type="secondary"):
            if GEMINI_MODULE_AVAILABLE:
                status = get_gemini_status()
                st.json(status)


# =============================================================================
# MAIN SIDEBAR RENDERING FUNCTION
# =============================================================================

def render_gemini_sidebar(page_context: Dict[str, Any]) -> None:
    """
    Render the Gemini chat interface inside the sidebar.
    
    This is the main entry point for adding Gemini chat to the sidebar.
    Call this function INSIDE the `with st.sidebar:` block.
    
    Note: Chart capture is now handled by render_chart_with_capture(),
    not by this function. Users click 📷 buttons next to charts to
    attach them to their messages.
    
    Args:
        page_context: Dictionary with current page information.
            Should include keys like:
            - "page": Page name
            - "asset": Selected asset
            - "date_range": Selected date range
            - Other page-specific parameters
    
    Example:
        >>> with st.sidebar:
        ...     st.header("Controls")
        ...     # ... controls ...
        ...     st.markdown("---")
        ...     render_gemini_sidebar({"page": "Analysis", "asset": "sp500"})
    """
    # Initialize session state
    init_gemini_session_state()
    
    # Header with status
    render_gemini_header()
    
    # Status badge
    render_status_badge()
    
    # Chat messages (with auto-scroll)
    render_chat_messages_compact()
    
    # Input area
    render_chat_input_compact(page_context)
    
    # Action buttons in expander
    with st.expander("⚙️ Opzioni", expanded=False):
        render_chat_actions_compact()


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
    clear_captured_chart()
    
    if GEMINI_MODULE_AVAILABLE:
        get_assistant().clear_history()


def add_system_message(content: str) -> None:
    """
    Add a system/info message to the chat.
    
    Args:
        content: The message to display
    """
    init_gemini_session_state()
    st.session_state.gemini_history.append({
        "role": "assistant",
        "content": f"ℹ️ {content}",
        "has_image": False,
        "image_name": ""
    })
