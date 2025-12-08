"""
IoT Financial Data Analytics - Main Application

Entry point for the Streamlit web application.
This file sets up the page configuration and navigation structure.

Run with: streamlit run app.py
"""

import os
import sys

import streamlit as st

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config
from data_loader import list_available_assets, load_single_asset


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)


# =============================================================================
# CUSTOM CSS (minimal styling)
# =============================================================================

st.markdown("""
    <style>
    /* Reduce padding at the top */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Style for info boxes */
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR - NAVIGATION AND INFO
# =============================================================================

with st.sidebar:
    st.title(f"{config.PAGE_ICON} {config.PAGE_TITLE}")
    st.markdown("---")
    
    # Project description
    st.markdown("""
    **IoT & Data Analytics Project**
    
    Financial data analysis using IoT-inspired 
    techniques for anomaly detection and 
    pattern recognition.
    """)
    
    st.markdown("---")
    
    # Navigation info
    st.markdown("### 📑 Pages")
    st.markdown("""
    1. **Single Asset Analysis**  
       Explore one asset with anomaly detection
       
    2. **Real-time IoT**  
       Streaming simulation with sliding window
       
    3. **Cross-Asset**  
       Correlations and multi-asset analysis
       
    4. **Patterns**  
       Candlestick pattern recognition
    """)
    
    st.markdown("---")
    
    # Technical info expander
    with st.expander("ℹ️ About the techniques"):
        st.markdown("""
        **Z-Score**  
        Measures how many standard deviations 
        a value is from the mean.
        
        - |Z| < 2: Normal
        - |Z| 2-3: Warning
        - |Z| > 3: Anomaly
        
        **Sliding Window**  
        IoT-style processing: statistics calculated 
        on a moving window of recent data points.
        
        **Correlation**  
        Pearson correlation measures linear 
        relationship between assets (-1 to +1).
        """)


# =============================================================================
# MAIN PAGE CONTENT
# =============================================================================

st.title("📊 IoT Financial Data Analytics")

st.markdown("""
Welcome to the Financial Data Analytics Dashboard. This application treats 
high-frequency financial data as IoT sensor streams, applying real-time 
anomaly detection and pattern recognition techniques.
""")

# Overview cards
col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **📈 5 Assets**  
    S&P 500, Gold, Oil, USD, Bitcoin
    """)

with col2:
    st.info("""
    **⏱️ 3 Granularities**  
    Minute, Hourly, Daily
    """)

with col3:
    st.info("""
    **🔍 2 Years**  
    Historical data coverage
    """)

st.markdown("---")

# Quick start guide
st.subheader("🚀 Quick Start")

st.markdown("""
1. **Navigate** using the sidebar menu (pages appear after adding them to `/pages` folder)
2. **Select** an asset and time range
3. **Explore** interactive charts with zoom and pan
4. **Identify** anomalies highlighted in red
5. **Analyze** patterns and correlations

Use the sidebar expanders to learn about each technique.
""")

st.markdown("---")

# Data loading test section
st.subheader("🔧 Data Status")

# Try to load data to verify setup
try:
    # Test loading one asset
    test_asset = list_available_assets()[0]
    test_granularity = "daily"
    
    df = load_single_asset(test_asset, test_granularity)
    record_count = len(df)
    
    st.success(
        f"✅ Data loaded successfully! "
        f"Found {record_count:,} records for {test_asset} ({test_granularity})"
    )
    
    # Show sample
    with st.expander("📋 Preview data"):
        st.dataframe(df.head(10), width='stretch')
        
except FileNotFoundError as e:
    st.warning(f"""
    ⚠️ **Data files not found**
    
    Please ensure your CSV files are placed in the correct folders:
    ```
    data/
    ├── minute/
    │   ├── sp500.csv
    │   ├── gold.csv
    │   └── ...
    ├── hourly/
    │   └── ...
    └── daily/
        └── ...
    ```
    
    Error: {e}
    """)
    
except Exception as e:
    st.error(f"❌ Error loading data: {e}")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    IoT & Data Analytics Project | University Exam
</div>
""", unsafe_allow_html=True)
