"""
Reusable UI Components for IoT Financial Data Analytics.

This module provides standardized UI components used across all pages:
- Page header and footer
- Gemini AI chat sidebar (compact version integrated in left sidebar)
- Chat message rendering
- CSS injection for custom styling

The Gemini sidebar component provides:
- Compact chat interface designed for sidebar width
- Message history with user/assistant styling
- Chart capture functionality (📷 button)
- Persistent conversation across page navigation

Usage:
    from components import title, footer, render_gemini_sidebar
    
    # At the top of the page
    title("Page Title", "Description")
    
    # Inside sidebar, after controls
    with st.sidebar:
        # ... your controls ...
        st.markdown("---")
        render_gemini_sidebar(page_context, current_figure)
    
    # Footer at the end
    footer("Page Title")
"""

import streamlit as st
from typing import Any, Dict, Optional

# Import Gemini assistant module
# Using try/except for graceful degradation if module not yet created
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
    
    Creates a consistent header across all pages with the page title
    and a brief description of the page's functionality.
    
    Args:
        page_title: The main title to display (e.g., "Single Asset Analysis")
        description: A brief description shown below the title
    
    Example:
        >>> title("Pattern Recognition", "Identify candlestick and chart patterns.")
    """
    st.title(page_title)
    st.markdown(description)


def footer(page_title: str) -> None:
    """
    Render a standardized page footer.
    
    Creates a consistent footer across all pages with the page name
    and project attribution.
    
    Args:
        page_title: The page name to display in the footer
    
    Example:
        >>> footer("Single Asset Analysis")
    """
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray;'>
        {page_title} | IoT & Data Analytics Project
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# GEMINI SIDEBAR CHAT COMPONENTS
# =============================================================================

def init_gemini_session_state() -> None:
    """
    Initialize Streamlit session state variables for Gemini chat.
    
    Creates the following session state variables if they don't exist:
    - gemini_chat_open: Whether the chat is expanded (default: True for sidebar)
    - gemini_history: List of chat messages (default: empty list)
    - gemini_pending_image: Base64 image waiting to be sent (default: None)
    - gemini_input_key: Counter for input widget key regeneration
    
    This function is idempotent and safe to call multiple times.
    """
    if "gemini_chat_open" not in st.session_state:
        st.session_state.gemini_chat_open = True
    
    if "gemini_history" not in st.session_state:
        st.session_state.gemini_history = []
    
    if "gemini_pending_image" not in st.session_state:
        st.session_state.gemini_pending_image = None
    
    if "gemini_input_key" not in st.session_state:
        st.session_state.gemini_input_key = 0


def render_gemini_header() -> None:
    """
    Render the Gemini chat header in sidebar.
    
    Displays the Gemini branding and connection status indicator.
    """
    # Header with status
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### ✨ Gemini Assistant")
    
    with col2:
        if GEMINI_MODULE_AVAILABLE:
            if is_gemini_available():
                st.markdown("🟢")  # Connected
            else:
                st.markdown("🟡")  # Mock mode
        else:
            st.markdown("🔴")  # Not available


def render_status_badge() -> None:
    """
    Render a compact status badge showing Gemini API availability.
    
    Shows different indicators based on:
    - Mock mode (API key not configured)
    - Real mode (API key configured and valid)
    - Error state (configuration issues)
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


def render_welcome_message_compact() -> None:
    """
    Render a compact welcome message for the sidebar chat.
    
    Displays a brief greeting and example questions when
    the chat history is empty.
    """
    st.markdown("""
    <div style='text-align: center; padding: 10px; color: #666;'>
        <div style='font-size: 24px; margin-bottom: 8px;'>✨</div>
        <div style='font-size: 13px;'>
            Chiedimi qualsiasi cosa sui dati, anomalie o pattern!
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick suggestions as small buttons
    st.markdown("**💡 Prova a chiedere:**")
    
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


def render_message_compact(role: str, content: str, has_image: bool = False) -> None:
    """
    Render a single chat message with compact styling for sidebar.
    
    Args:
        role: Either "user" or "assistant"
        content: The message text content
        has_image: Whether this message included an attached image
    """
    if role == "user":
        # User message - right aligned, blue background
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #4285F4, #5a95f5);
            color: white;
            padding: 8px 12px;
            border-radius: 12px 12px 4px 12px;
            margin: 4px 0;
            font-size: 13px;
            text-align: right;
        '>
            {content}
            {'<br><small>📷 allegato</small>' if has_image else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Assistant message - left aligned, gray background
        # Process markdown for better display
        st.markdown(f"""
        <div style='
            background: #f0f2f6;
            color: #333;
            padding: 8px 12px;
            border-radius: 12px 12px 12px 4px;
            margin: 4px 0;
            font-size: 13px;
        '>
            {content}
        </div>
        """, unsafe_allow_html=True)


def render_chat_messages_compact() -> None:
    """
    Render all messages in a compact scrollable container.
    
    Uses st.container with fixed height for the sidebar.
    Shows welcome message if history is empty.
    """
    history = st.session_state.gemini_history
    
    if not history:
        render_welcome_message_compact()
        return
    
    # Create scrollable container for messages
    # Using a container with custom height
    messages_container = st.container(height=250)
    
    with messages_container:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            has_image = msg.get("has_image", False)
            
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)
                    if has_image:
                        st.caption("📷 Grafico allegato")
            else:
                with st.chat_message("assistant", avatar="✨"):
                    st.markdown(content)


def render_chat_input_compact(
    page_context: Dict[str, Any],
    current_figure: Optional[Any] = None
) -> None:
    """
    Render compact chat input area for sidebar.
    
    Provides:
    - Camera button to capture current chart
    - Text input for typing messages
    - Image attachment indicator
    
    Args:
        page_context: Dictionary with current page information
        current_figure: Optional Plotly figure for chart capture
    """
    # Check for pending suggestion from button click
    if "gemini_pending_question" in st.session_state:
        pending = st.session_state.gemini_pending_question
        del st.session_state.gemini_pending_question
        _process_user_message(pending, page_context, None)
        return
    
    # Image capture row
    col1, col2 = st.columns([1, 4])
    
    with col1:
        capture_disabled = current_figure is None or not GEMINI_MODULE_AVAILABLE
        
        if st.session_state.gemini_pending_image:
            # Image already captured
            if st.button("📷✓", key="gem_cap_done", help="Rimuovi immagine", use_container_width=True):
                st.session_state.gemini_pending_image = None
                st.rerun()
        else:
            # Capture button
            if st.button(
                "📷", 
                key="gem_capture", 
                help="Cattura grafico" if not capture_disabled else "Nessun grafico",
                disabled=capture_disabled,
                use_container_width=True
            ):
                if current_figure is not None:
                    with st.spinner("📷"):
                        image_b64 = capture_plotly_figure(current_figure)
                        if image_b64:
                            st.session_state.gemini_pending_image = image_b64
                            st.toast("Grafico catturato!", icon="📷")
                            st.rerun()
                        else:
                            st.toast("Errore cattura", icon="❌")
    
    with col2:
        if st.session_state.gemini_pending_image:
            st.caption("📎 Grafico pronto")
    
    # Text input
    user_input = st.chat_input(
        placeholder="Scrivi qui...",
        key=f"gem_input_{st.session_state.gemini_input_key}"
    )
    
    if user_input:
        # Get pending image
        pending_image = st.session_state.gemini_pending_image
        st.session_state.gemini_pending_image = None
        
        # Process message
        _process_user_message(user_input, page_context, pending_image)


def _process_user_message(
    user_input: str,
    page_context: Dict[str, Any],
    image_base64: Optional[str]
) -> None:
    """
    Process a user message and get response from Gemini.
    
    Handles:
    1. Adding user message to history
    2. Sending request to Gemini API
    3. Adding assistant response to history
    4. Triggering UI refresh
    
    Args:
        user_input: The user's message text
        page_context: Current page context dictionary
        image_base64: Optional base64-encoded chart image
    """
    if not GEMINI_MODULE_AVAILABLE:
        st.error("Modulo Gemini non disponibile")
        return
    
    # Add user message to history
    user_message = {
        "role": "user",
        "content": user_input,
        "has_image": image_base64 is not None
    }
    st.session_state.gemini_history.append(user_message)
    
    # Get assistant instance
    assistant = get_assistant()
    
    # Sync history
    assistant.set_history(st.session_state.gemini_history)
    
    # Send message and get response
    with st.spinner("✨ Penso..."):
        response = assistant.send_message(
            question=user_input,
            page_context=page_context,
            image_base64=image_base64,
            history=st.session_state.gemini_history[:-1]
        )
    
    # Add response to history
    assistant_message = {
        "role": "assistant",
        "content": response,
        "has_image": False
    }
    st.session_state.gemini_history.append(assistant_message)
    
    # Update assistant history
    assistant.add_to_history("user", user_input)
    assistant.add_to_history("assistant", response)
    
    # Reset input
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

def render_gemini_sidebar(
    page_context: Dict[str, Any],
    current_figure: Optional[Any] = None
) -> None:
    """
    Render the Gemini chat interface inside the sidebar.
    
    This is the main entry point for adding Gemini chat to the sidebar.
    Call this function INSIDE the `with st.sidebar:` block, typically
    after your page controls.
    
    Args:
        page_context: Dictionary with current page information.
            Should include keys like:
            - "page": Page name
            - "asset": Selected asset (if applicable)
            - "date_range": Selected date range
            - Other page-specific parameters
        
        current_figure: Optional Plotly figure object.
            If provided, enables the "capture chart" functionality
            so users can send chart screenshots to Gemini.
    
    Example:
        >>> with st.sidebar:
        ...     st.header("Controls")
        ...     selected_asset = st.selectbox("Asset", assets)
        ...     # ... other controls ...
        ...     
        ...     st.markdown("---")
        ...     
        ...     page_context = {"page": "Analysis", "asset": selected_asset}
        ...     render_gemini_sidebar(page_context, fig_main)
    
    Note:
        The chat history persists across page navigation using
        st.session_state, so users can continue conversations
        when switching between pages.
    """
    # Initialize session state
    init_gemini_session_state()
    
    # Header
    render_gemini_header()
    
    # Status badge
    render_status_badge()
    
    # Messages
    render_chat_messages_compact()
    
    # Input area
    render_chat_input_compact(page_context, current_figure)
    
    # Action buttons in expander to save space
    with st.expander("⚙️ Opzioni", expanded=False):
        render_chat_actions_compact()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_chat_history() -> list:
    """
    Get the current chat history from session state.
    
    Returns:
        List of message dictionaries with 'role' and 'content' keys.
    """
    init_gemini_session_state()
    return st.session_state.gemini_history.copy()


def clear_chat_history() -> None:
    """
    Clear the chat history in session state.
    
    Also clears the assistant's internal history if available.
    """
    init_gemini_session_state()
    st.session_state.gemini_history = []
    
    if GEMINI_MODULE_AVAILABLE:
        get_assistant().clear_history()


def add_system_message(content: str) -> None:
    """
    Add a system/info message to the chat.
    
    Useful for showing context changes or notifications.
    
    Args:
        content: The message to display
    """
    init_gemini_session_state()
    st.session_state.gemini_history.append({
        "role": "assistant",
        "content": f"ℹ️ {content}",
        "has_image": False
    })
