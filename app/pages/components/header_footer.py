import streamlit as st

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