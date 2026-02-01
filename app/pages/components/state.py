import streamlit as st
from config import INITIAL_MESSAGE_DISPLAY_COUNT, DATA_OPTIONS
from config.ui import PageType
from typing import Dict

try:
    from services import get_gemini_status
    gemini_status = get_gemini_status()
    GEMINI_MODULE_AVAILABLE = True
except ImportError:
    GEMINI_MODULE_AVAILABLE = False

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


def get_selected_data_options(page_type: PageType) -> Dict[str, bool]:
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
            st.session_state.gemini_selected_data[page_type] =dict.fromkeys(
                DATA_OPTIONS[page_type].keys(), False
            )
        else:
            st.session_state.gemini_selected_data[page_type] = {}
    
    return st.session_state.gemini_selected_data[page_type]

