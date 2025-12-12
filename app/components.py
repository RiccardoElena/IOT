"""
Reusable UI Components for IoT Financial Data Analytics.

This module provides standardized UI components used across all pages:
- Page header and footer
- Gemini AI chat sidebar (floating panel on the right)
- Chat message rendering
- CSS injection for custom styling

The Gemini chat component provides:
- Toggle button (bottom-right corner)
- Expandable chat panel
- Message history with user/assistant styling
- Chart capture functionality
- Responsive design

Usage:
    from components import title, footer, render_gemini_chat
    
    # At the top of the page
    title("Page Title", "Description")
    
    # At the bottom, before footer
    render_gemini_chat(page_context, current_figure)
    
    # Footer
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
# GEMINI CHAT COMPONENTS
# =============================================================================

def init_gemini_session_state() -> None:
    """
    Initialize Streamlit session state variables for Gemini chat.
    
    Creates the following session state variables if they don't exist:
    - gemini_chat_open: Whether the chat panel is open (default: False)
    - gemini_history: List of chat messages (default: empty list)
    - gemini_pending_image: Base64 image waiting to be sent (default: None)
    - gemini_input_key: Counter for input widget key regeneration
    
    This function is idempotent and safe to call multiple times.
    """
    if "gemini_chat_open" not in st.session_state:
        st.session_state.gemini_chat_open = False
    
    if "gemini_history" not in st.session_state:
        st.session_state.gemini_history = []
    
    if "gemini_pending_image" not in st.session_state:
        st.session_state.gemini_pending_image = None
    
    if "gemini_input_key" not in st.session_state:
        st.session_state.gemini_input_key = 0


def inject_gemini_chat_css() -> None:
    """
    Inject custom CSS for the Gemini chat sidebar.
    
    Adds styling for:
    - Fixed position toggle button (bottom-right corner)
    - Chat panel container with shadow and rounded corners
    - Message bubbles (user vs assistant styling)
    - Scrollable message area
    - Input area styling
    - Responsive adjustments
    - Animations for open/close
    
    This CSS is injected once per page load using st.markdown.
    """
    chat_width = getattr(config, "GEMINI_CHAT_WIDTH", 400) if config else 400
    
    css = f"""
    <style>
    /* =========================================
       GEMINI CHAT TOGGLE BUTTON
       ========================================= */
    
    .gemini-toggle-btn {{
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #4285F4 0%, #34A853 50%, #FBBC05 75%, #EA4335 100%);
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        z-index: 9999;
    }}
    
    .gemini-toggle-btn:hover {{
        transform: scale(1.1);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
    }}
    
    /* =========================================
       CHAT PANEL CONTAINER
       ========================================= */
    
    .gemini-chat-panel {{
        position: fixed;
        top: 60px;
        right: 0;
        width: {chat_width}px;
        height: calc(100vh - 60px);
        background: var(--background-color, #ffffff);
        border-left: 1px solid var(--secondary-background-color, #e0e0e0);
        box-shadow: -4px 0 20px rgba(0, 0, 0, 0.1);
        z-index: 9998;
        display: flex;
        flex-direction: column;
        animation: slideIn 0.3s ease-out;
    }}
    
    @keyframes slideIn {{
        from {{
            transform: translateX(100%);
            opacity: 0;
        }}
        to {{
            transform: translateX(0);
            opacity: 1;
        }}
    }}
    
    /* =========================================
       CHAT HEADER
       ========================================= */
    
    .gemini-chat-header {{
        padding: 16px 20px;
        background: linear-gradient(135deg, #4285F4 0%, #34A853 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-weight: 600;
        font-size: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }}
    
    .gemini-chat-header-title {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    
    .gemini-close-btn {{
        background: rgba(255, 255, 255, 0.2);
        border: none;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        cursor: pointer;
        font-size: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.2s ease;
    }}
    
    .gemini-close-btn:hover {{
        background: rgba(255, 255, 255, 0.3);
    }}
    
    /* =========================================
       MESSAGES CONTAINER
       ========================================= */
    
    .gemini-messages {{
        flex: 1;
        overflow-y: auto;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }}
    
    /* Custom scrollbar */
    .gemini-messages::-webkit-scrollbar {{
        width: 6px;
    }}
    
    .gemini-messages::-webkit-scrollbar-track {{
        background: transparent;
    }}
    
    .gemini-messages::-webkit-scrollbar-thumb {{
        background: #c0c0c0;
        border-radius: 3px;
    }}
    
    .gemini-messages::-webkit-scrollbar-thumb:hover {{
        background: #a0a0a0;
    }}
    
    /* =========================================
       MESSAGE BUBBLES
       ========================================= */
    
    .gemini-message {{
        max-width: 85%;
        padding: 12px 16px;
        border-radius: 16px;
        font-size: 14px;
        line-height: 1.5;
        word-wrap: break-word;
    }}
    
    .gemini-message-user {{
        align-self: flex-end;
        background: linear-gradient(135deg, #4285F4, #5a95f5);
        color: white;
        border-bottom-right-radius: 4px;
    }}
    
    .gemini-message-assistant {{
        align-self: flex-start;
        background: var(--secondary-background-color, #f0f0f0);
        color: var(--text-color, #333333);
        border-bottom-left-radius: 4px;
    }}
    
    .gemini-message-assistant strong {{
        color: #4285F4;
    }}
    
    /* Image attachment indicator */
    .gemini-message-image-indicator {{
        font-size: 12px;
        opacity: 0.8;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    
    /* =========================================
       INPUT AREA
       ========================================= */
    
    .gemini-input-area {{
        padding: 16px;
        border-top: 1px solid var(--secondary-background-color, #e0e0e0);
        background: var(--background-color, #ffffff);
    }}
    
    .gemini-input-row {{
        display: flex;
        gap: 8px;
        align-items: flex-end;
    }}
    
    .gemini-capture-btn {{
        background: #f0f0f0;
        border: 1px solid #d0d0d0;
        border-radius: 8px;
        padding: 8px 12px;
        cursor: pointer;
        font-size: 18px;
        transition: background 0.2s ease;
    }}
    
    .gemini-capture-btn:hover {{
        background: #e0e0e0;
    }}
    
    .gemini-capture-btn.active {{
        background: #4285F4;
        color: white;
        border-color: #4285F4;
    }}
    
    /* =========================================
       STATUS INDICATORS
       ========================================= */
    
    .gemini-status {{
        padding: 8px 16px;
        font-size: 12px;
        text-align: center;
        background: #fff3cd;
        color: #856404;
        border-bottom: 1px solid #ffc107;
    }}
    
    .gemini-status.error {{
        background: #f8d7da;
        color: #721c24;
        border-bottom-color: #f5c6cb;
    }}
    
    .gemini-status.success {{
        background: #d4edda;
        color: #155724;
        border-bottom-color: #c3e6cb;
    }}
    
    /* =========================================
       LOADING INDICATOR
       ========================================= */
    
    .gemini-loading {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 12px 16px;
        color: #666;
        font-size: 14px;
    }}
    
    .gemini-loading-dots {{
        display: flex;
        gap: 4px;
    }}
    
    .gemini-loading-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4285F4;
        animation: dotPulse 1.4s infinite ease-in-out;
    }}
    
    .gemini-loading-dot:nth-child(1) {{ animation-delay: -0.32s; }}
    .gemini-loading-dot:nth-child(2) {{ animation-delay: -0.16s; }}
    .gemini-loading-dot:nth-child(3) {{ animation-delay: 0s; }}
    
    @keyframes dotPulse {{
        0%, 80%, 100% {{
            transform: scale(0.6);
            opacity: 0.5;
        }}
        40% {{
            transform: scale(1);
            opacity: 1;
        }}
    }}
    
    /* =========================================
       WELCOME MESSAGE
       ========================================= */
    
    .gemini-welcome {{
        text-align: center;
        padding: 40px 20px;
        color: #666;
    }}
    
    .gemini-welcome-icon {{
        font-size: 48px;
        margin-bottom: 16px;
    }}
    
    .gemini-welcome-title {{
        font-size: 18px;
        font-weight: 600;
        color: #333;
        margin-bottom: 8px;
    }}
    
    .gemini-welcome-text {{
        font-size: 14px;
        line-height: 1.6;
    }}
    
    .gemini-welcome-suggestions {{
        margin-top: 20px;
        text-align: left;
    }}
    
    .gemini-welcome-suggestion {{
        background: var(--secondary-background-color, #f5f5f5);
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 13px;
        cursor: pointer;
        transition: background 0.2s ease;
    }}
    
    .gemini-welcome-suggestion:hover {{
        background: #e8e8e8;
    }}
    
    /* =========================================
       RESPONSIVE ADJUSTMENTS
       ========================================= */
    
    @media (max-width: 768px) {{
        .gemini-chat-panel {{
            width: 100%;
            right: 0;
        }}
        
        .gemini-toggle-btn {{
            bottom: 16px;
            right: 16px;
            width: 50px;
            height: 50px;
            font-size: 24px;
        }}
    }}
    
    /* =========================================
       DARK MODE SUPPORT
       ========================================= */
    
    @media (prefers-color-scheme: dark) {{
        .gemini-chat-panel {{
            background: #1e1e1e;
            border-left-color: #333;
        }}
        
        .gemini-message-assistant {{
            background: #2d2d2d;
            color: #e0e0e0;
        }}
        
        .gemini-input-area {{
            background: #1e1e1e;
            border-top-color: #333;
        }}
        
        .gemini-capture-btn {{
            background: #2d2d2d;
            border-color: #444;
            color: #e0e0e0;
        }}
        
        .gemini-capture-btn:hover {{
            background: #3d3d3d;
        }}
    }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)


def render_toggle_button() -> None:
    """
    Render the floating toggle button to open the Gemini chat.
    
    Displays a circular button with the Gemini sparkle icon in the
    bottom-right corner of the page. Clicking opens the chat panel.
    
    The button uses the Google Gemini gradient colors and includes
    hover effects for better interactivity.
    """
    # Only show toggle button when chat is closed
    if st.session_state.gemini_chat_open:
        return
    
    # Create a placeholder for the toggle button
    toggle_html = """
    <div class="gemini-toggle-btn" onclick="window.parent.postMessage({type: 'gemini_toggle'}, '*')" title="Apri Gemini Assistant">
        ✨
    </div>
    """
    
    st.markdown(toggle_html, unsafe_allow_html=True)


def render_chat_header() -> None:
    """
    Render the chat panel header with title and close button.
    
    The header includes:
    - Gemini icon and title
    - Close button to collapse the panel
    - Gradient background matching Gemini branding
    """
    chat_title = getattr(config, "GEMINI_CHAT_TITLE", "✨ Gemini Assistant") if config else "✨ Gemini Assistant"
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        st.markdown(f"### {chat_title}")
    
    with col2:
        if st.button("✕", key="gemini_close_btn", help="Chiudi chat"):
            st.session_state.gemini_chat_open = False
            st.rerun()


def render_status_indicator() -> None:
    """
    Render a status indicator showing Gemini API availability.
    
    Shows different messages based on:
    - Mock mode (API key not configured)
    - Real mode (API key configured and valid)
    - Error state (configuration issues)
    """
    if not GEMINI_MODULE_AVAILABLE:
        st.warning("⚠️ Modulo Gemini non disponibile. Verifica l'installazione.")
        return
    
    status = get_gemini_status()
    
    if not status["library_installed"]:
        st.error("❌ Libreria `google-generativeai` non installata.")
    elif not status["api_key_set"]:
        st.warning("🔧 **Mock Mode** - Configura `GEMINI_API_KEY` per risposte reali.")
    else:
        st.success(f"✅ Connesso a {status['model']}")


def render_welcome_message() -> None:
    """
    Render a welcome message when the chat history is empty.
    
    Displays:
    - Welcome icon and greeting
    - Brief explanation of capabilities
    - Suggested starter questions
    """
    st.markdown("""
    <div class="gemini-welcome">
        <div class="gemini-welcome-icon">✨</div>
        <div class="gemini-welcome-title">Ciao! Sono il tuo assistente AI</div>
        <div class="gemini-welcome-text">
            Posso aiutarti a comprendere i dati, spiegare le anomalie 
            e analizzare i pattern. Prova a chiedermi qualcosa!
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Suggested questions
    st.markdown("**💡 Suggerimenti:**")
    
    suggestions = [
        "Cosa significa Z-score?",
        "Perché ci sono punti rossi nel grafico?",
        "Come interpreto una correlazione negativa?",
        "Cos'è un pattern Doji?"
    ]
    
    for suggestion in suggestions:
        if st.button(f"📝 {suggestion}", key=f"suggestion_{hash(suggestion)}", use_container_width=True):
            # Add suggestion as user message and trigger response
            st.session_state.gemini_pending_question = suggestion
            st.rerun()


def render_message(role: str, content: str, has_image: bool = False) -> None:
    """
    Render a single chat message with appropriate styling.
    
    Args:
        role: Either "user" or "assistant"
        content: The message text content
        has_image: Whether this message included an attached image
    
    The message is styled differently based on role:
    - User messages: Blue background, aligned right
    - Assistant messages: Gray background, aligned left
    """
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
            if has_image:
                st.caption("📷 Grafico allegato")
    else:
        with st.chat_message("assistant", avatar="✨"):
            st.markdown(content)


def render_chat_messages() -> None:
    """
    Render all messages in the chat history.
    
    Iterates through the session state history and renders each
    message with appropriate styling. Shows welcome message if
    history is empty.
    """
    history = st.session_state.gemini_history
    
    if not history:
        render_welcome_message()
        return
    
    # Render all messages
    for msg in history:
        render_message(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            has_image=msg.get("has_image", False)
        )


def render_chat_input(
    page_context: Dict[str, Any],
    current_figure: Optional[Any] = None
) -> None:
    """
    Render the chat input area with text input and capture button.
    
    Provides:
    - Text input for typing messages
    - Camera button to capture current chart
    - Send functionality
    - Image attachment indicator
    
    Args:
        page_context: Dictionary with current page information
        current_figure: Optional Plotly figure for chart capture
    """
    # Check for pending suggestion
    if "gemini_pending_question" in st.session_state:
        pending = st.session_state.gemini_pending_question
        del st.session_state.gemini_pending_question
        _process_user_message(pending, page_context, None)
        return
    
    # Image capture button and status
    col1, col2 = st.columns([1, 5])
    
    with col1:
        capture_disabled = current_figure is None
        capture_tooltip = "Cattura grafico corrente" if not capture_disabled else "Nessun grafico disponibile"
        
        if st.session_state.gemini_pending_image:
            # Image already captured - show indicator
            if st.button("📷 ✓", key="gemini_capture_attached", help="Immagine allegata (clicca per rimuovere)"):
                st.session_state.gemini_pending_image = None
                st.rerun()
        else:
            # Capture button
            if st.button("📷", key="gemini_capture_btn", help=capture_tooltip, disabled=capture_disabled):
                if current_figure is not None and GEMINI_MODULE_AVAILABLE:
                    with st.spinner("Cattura in corso..."):
                        image_b64 = capture_plotly_figure(current_figure)
                        if image_b64:
                            st.session_state.gemini_pending_image = image_b64
                            st.toast("📷 Grafico catturato!", icon="✅")
                            st.rerun()
                        else:
                            st.toast("Errore nella cattura del grafico", icon="❌")
    
    with col2:
        if st.session_state.gemini_pending_image:
            st.caption("📎 Grafico allegato - verrà inviato con il prossimo messaggio")
    
    # Text input
    placeholder = getattr(config, "GEMINI_CHAT_PLACEHOLDER", "Scrivi una domanda...") if config else "Scrivi una domanda..."
    
    user_input = st.chat_input(
        placeholder=placeholder,
        key=f"gemini_chat_input_{st.session_state.gemini_input_key}"
    )
    
    if user_input:
        # Get pending image (if any)
        pending_image = st.session_state.gemini_pending_image
        st.session_state.gemini_pending_image = None  # Clear after use
        
        # Process the message
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
    
    # Sync history with assistant
    assistant.set_history(st.session_state.gemini_history)
    
    # Send message and get response
    with st.spinner("✨ Gemini sta pensando..."):
        response = assistant.send_message(
            question=user_input,
            page_context=page_context,
            image_base64=image_base64,
            history=st.session_state.gemini_history[:-1]  # Exclude current message
        )
    
    # Add assistant response to history
    assistant_message = {
        "role": "assistant",
        "content": response,
        "has_image": False
    }
    st.session_state.gemini_history.append(assistant_message)
    
    # Update assistant's internal history
    assistant.add_to_history("user", user_input)
    assistant.add_to_history("assistant", response)
    
    # Increment input key to reset input field
    st.session_state.gemini_input_key += 1
    
    # Refresh UI
    st.rerun()


def render_chat_actions() -> None:
    """
    Render additional chat action buttons.
    
    Provides:
    - Clear history button
    - Status information
    """
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🗑️ Pulisci chat", key="gemini_clear_btn", use_container_width=True):
            st.session_state.gemini_history = []
            if GEMINI_MODULE_AVAILABLE:
                get_assistant().clear_history()
            st.rerun()
    
    with col2:
        if st.button("ℹ️ Stato", key="gemini_status_btn", use_container_width=True):
            if GEMINI_MODULE_AVAILABLE:
                status = get_gemini_status()
                st.json(status)


# =============================================================================
# MAIN CHAT RENDERING FUNCTION
# =============================================================================

def render_gemini_chat(
    page_context: Dict[str, Any],
    current_figure: Optional[Any] = None
) -> None:
    """
    Render the complete Gemini chat interface.
    
    This is the main entry point for adding the Gemini chat to any page.
    It handles:
    - Session state initialization
    - CSS injection
    - Toggle button rendering
    - Chat panel rendering (when open)
    
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
        >>> page_context = {
        ...     "page": "Single Asset Analysis",
        ...     "asset": "sp500",
        ...     "granularity": "daily",
        ...     "date_range": "2023-01-01 → 2023-12-31",
        ...     "anomalies_count": 15
        ... }
        >>> render_gemini_chat(page_context, fig_main)
    
    Note:
        Call this function at the end of each page, typically
        just before the footer() call.
    """
    # Initialize session state
    init_gemini_session_state()
    
    # Inject CSS
    inject_gemini_chat_css()
    
    # Render toggle button (when closed)
    if not st.session_state.gemini_chat_open:
        # Use a container with button to toggle chat
        st.markdown("---")
        
        col1, col2, col3 = st.columns([5, 2, 5])
        with col2:
            if st.button("✨ Chiedi a Gemini", key="gemini_open_btn", use_container_width=True):
                st.session_state.gemini_chat_open = True
                st.rerun()
        
        return
    
    # Render chat panel (when open)
    st.markdown("---")
    st.markdown("## ✨ Gemini Assistant")
    
    # Status indicator
    with st.expander("ℹ️ Stato connessione", expanded=False):
        render_status_indicator()
    
    # Messages container
    messages_container = st.container(height=400)
    
    with messages_container:
        render_chat_messages()
    
    # Input area
    render_chat_input(page_context, current_figure)
    
    # Action buttons
    with st.expander("⚙️ Opzioni", expanded=False):
        render_chat_actions()
    
    # Close button at the bottom
    if st.button("✕ Chiudi Gemini", key="gemini_close_bottom", use_container_width=True):
        st.session_state.gemini_chat_open = False
        st.rerun()


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


def is_chat_open() -> bool:
    """
    Check if the chat panel is currently open.
    
    Returns:
        True if chat is open, False otherwise.
    """
    init_gemini_session_state()
    return st.session_state.gemini_chat_open


def toggle_chat() -> None:
    """
    Toggle the chat panel open/closed state.
    """
    init_gemini_session_state()
    st.session_state.gemini_chat_open = not st.session_state.gemini_chat_open
