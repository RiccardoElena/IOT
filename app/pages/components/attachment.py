import streamlit as st
import time
from typing import Any, List
from .state import init_gemini_session_state, get_selected_data_options, GEMINI_MODULE_AVAILABLE
from config import MAX_CHARTS_IN_CONTEXT, DATA_OPTIONS
from config.ui import PageType

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

###############################################################################
# RENDERING FUNCTIONS
###############################################################################

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


def render_data_selection(page_type: PageType) -> List[str]:
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
            default_value = bool(opt["default"]) if opt["default"] is not None else False
            new_value = st.checkbox(
                str(opt["label"]),
                value=selected.get(key, default_value),
                key=f"data_opt_{page_type}_{key}",
                help=str(opt["description"])
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