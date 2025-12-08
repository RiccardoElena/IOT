import streamlit as st

# =============================================================================
# FOOTER
# =============================================================================

def footer(page_title: str):
  st.markdown("---")
  st.markdown(f"""
  <div style='text-align: center; color: gray;'>
      {page_title} | IoT & Data Analytics Project
  </div>
  """, unsafe_allow_html=True)

def title(page_title: str, description: str):
    st.title(page_title)
    st.markdown(description)