"""
Configuration file for the IoT Financial Data Analytics project.

All configurable parameters are centralized here.
Change values here to adapt the project to different datasets.
"""

import os

# =============================================================================
# ASSET CONFIGURATION
# =============================================================================

# Asset identifiers (internal keys used throughout the code)
# Map: internal_key -> display_name (shown in UI)
ASSETS = {
    "sp500": "S&P 500",
    "gold": "Gold",
    "oil": "Oil",
    "usd": "USD Index",
    "btc": "Bitcoin"
}

# File names for each asset (without path)
# Map: internal_key -> filename
# Change these to match your actual file names
FILE_NAMES = {
    "sp500": "sp500.csv",
    "gold": "gold.csv",
    "oil": "oil.csv",
    "usd": "usd.csv",
    "btc": "btc.csv"
}

# =============================================================================
# DATA PATHS
# =============================================================================

# Base directory for all data files (relative to config.py location)
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BASE_PATH = os.path.join(_CONFIG_DIR, "..", "data")
DATA_BASE_PATH = os.path.abspath(DATA_BASE_PATH)  # Convert to absolute path

# Subdirectories for each granularity
GRANULARITY_PATHS = {
    "minute": "1-minute",
    "hourly": "2-hourly",
    "daily": "3-daily",
}

# Display names for granularities (shown in UI)
GRANULARITY_DISPLAY = {
    "minute": "Minute",
    "hourly": "Hourly",
    "daily": "Daily",
}

# =============================================================================
# COLUMN NAMES
# =============================================================================

# Expected column names in CSV files
# Change these if your CSV has different column names
COLUMNS = {
    "timestamp": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "vwap": "vw",
    "num_trades": "n"
}

# =============================================================================
# ANOMALY DETECTION PARAMETERS
# =============================================================================

# Z-score thresholds
ZSCORE_WARNING_THRESHOLD = 2.0  # Values above this are "suspicious"
ZSCORE_ANOMALY_THRESHOLD = 3.0  # Values above this are "anomalies"

# Percentile thresholds for anomaly detection
PERCENTILE_LOW = 1    # Below this percentile = anomaly
PERCENTILE_HIGH = 99  # Above this percentile = anomaly

# Percentage change thresholds (in %)
PCT_CHANGE_THRESHOLD_DAILY = 5.0   # Daily: > 5% is anomaly
PCT_CHANGE_THRESHOLD_MINUTE = 1.0  # Minute: > 1% is anomaly

# =============================================================================
# SLIDING WINDOW PARAMETERS (IoT Real-time)
# =============================================================================

# Default window sizes (in number of data points)
WINDOW_SIZE_MINUTE = 60   # 60 minutes = 1 hour
WINDOW_SIZE_DAILY = 20    # 20 days ~ 1 month

# Configurable range for UI slider
WINDOW_SIZE_MIN = 30
WINDOW_SIZE_MAX = 120

# =============================================================================
# CROSS-ASSET PARAMETERS
# =============================================================================

# Rolling correlation window (in days)
CORRELATION_WINDOW = 30

# Correlation thresholds
CORRELATION_STRONG_POSITIVE = 0.7
CORRELATION_STRONG_NEGATIVE = -0.7

# Minimum assets with anomalies to flag as "systemic event"
SYSTEMIC_EVENT_THRESHOLD = 3

# =============================================================================
# PATTERN RECOGNITION PARAMETERS
# =============================================================================

# Doji: body must be less than this fraction of total range
DOJI_BODY_RATIO = 0.1

# Hammer: lower shadow must be at least this multiple of body
HAMMER_SHADOW_RATIO = 2.0

# =============================================================================
# UI CONFIGURATION
# =============================================================================

# Page configuration
PAGE_TITLE = "IoT Financial Analytics"
PAGE_ICON = "📊"
LAYOUT = "wide"

# Color scheme for anomalies
COLOR_NORMAL = "#636EFA"      # Blue
COLOR_WARNING = "#FFA15A"     # Orange
COLOR_ANOMALY = "#EF553B"     # Red
COLOR_BULLISH = "#00CC96"     # Green
COLOR_BEARISH = "#EF553B"     # Red

# Marker sizes
MARKER_SIZE_NORMAL = 6
MARKER_SIZE_ANOMALY = 12


# =============================================================================
# LLM CONFIGURATION (Gemini Assistant)
# =============================================================================

# Model settings
# Using Gemini 1.5 Flash for speed and generous free tier
# Free tier: 15 RPM, 1M TPM, 1500 RPD
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Generation parameters
GEMINI_MAX_TOKENS = 1024        # Maximum tokens in response
GEMINI_TEMPERATURE = 0.7        # Creativity level (0.0 = deterministic, 1.0 = creative)

# Conversation history settings
# Number of messages to keep in context (user + assistant messages)
# 14 messages = approximately 7 conversation turns
GEMINI_HISTORY_LENGTH = 14

# Environment variable name for API key
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

# System prompt that defines the assistant's personality and knowledge
# This is sent at the beginning of every conversation
GEMINI_SYSTEM_PROMPT = """
You are an AI assistant specialized in financial analysis and data analytics, integrated into a university dashboard called "IoT Financial Analytics".

## YOUR ROLE
You help users understand the financial data displayed, the analysis techniques used, and the meaning of detected patterns and anomalies.

## DASHBOARD CONTEXT
The dashboard analyzes 5 financial assets treated as "IoT sensors":
- **S&P 500**: US stock market index (500 largest companies)
- **Gold**: Safe-haven asset, inversely correlated with the dollar
- **Oil (WTI)**: Energy commodity, highly volatile
- **USD Index**: US dollar strength vs currency basket
- **Bitcoin**: Cryptocurrency, high volatility, 24/7 trading

Data is available in 3 granularities: Minute (1 min), Hourly (1 hour), Daily (1 day).

## ANALYSIS TECHNIQUES YOU KNOW

### Z-Score (Anomaly Detection)
- Formula: Z = (value - mean) / standard_deviation
- |Z| < 2: Normal
- |Z| 2-3: Warning (attention)
- |Z| > 3: Anomaly (rare event, ~0.3% probability)
- Applied to: price (close), volume, volatility (high-low)

### Sliding Window (IoT Real-time)
- Moving window of N points to calculate "local" statistics
- Simulates streaming processing typical of IoT systems
- Allows adaptation to regime changes

### Cross-Asset Correlation
- Pearson correlation: from -1 (inverse) to +1 (direct)
- Rolling correlation: how it changes over time
- Typical correlations: Gold-USD negative, Oil-SP500 positive
- Systemic events: when 3+ assets show anomalies together

### Pattern Recognition
**Candlestick (1-2 candles):**
- Doji: indecision (open ≈ close)
- Hammer: bullish reversal (long lower shadow)
- Engulfing: reversal (candle that "engulfs" the previous one)

**Chart Patterns (multi-candle):**
- Double Top/Bottom: reversal (M or W shape)
- Head & Shoulders: bearish reversal (3 peaks)
- Cup & Handle: bullish continuation

## HOW TO RESPOND

1. **Language**: Respond in ITALIAN by default, unless the user writes in English
2. **Style**: Clear, educational but concise. You are a tutor, not an academic paper
3. **Structure**: Use bullet points for lists, bold for key terms
4. **Images**: If you receive a chart, describe and analyze it in context
5. **Uncertainty**: If unsure, say so. Don't invent data
6. **Practicality**: Always connect theory to what the user sees in the dashboard

## EXAMPLE OF A GOOD RESPONSE

Question: "Why is that point red?"

Response: "Il punto rosso indica un'**anomalia** rilevata dal sistema.
In questo caso, il valore ha uno Z-score > 3, significa che è distante più di 3 deviazioni standard dalla media — un evento statisticamente raro (capita circa lo 0.3% delle volte).

Possibili cause:
- News improvvisa (earnings, dati macro)
- Flash crash o spike di volatilità
- Errore nei dati (da verificare)

Guarda il grafico Z-score sotto per vedere l'entità della deviazione."

## WHAT NOT TO DO
- Do not give investment advice ("buy", "sell")
- Do not invent data or statistics
- Do not answer questions unrelated to the dashboard
- Do not be verbose: focused and useful responses only
"""

# Chat UI configuration
GEMINI_CHAT_TITLE = "✨ Gemini Assistant"
GEMINI_CHAT_PLACEHOLDER = "Scrivi una domanda..."
GEMINI_CHAT_WIDTH = 400  # Width in pixels for the chat sidebar

# Mock mode message (shown when API key is not configured)
GEMINI_MOCK_RESPONSE = """**[MOCK MODE]** 🔧

API key non configurata. Questa è una risposta di test per verificare l'interfaccia.

Per attivare le risposte reali di Gemini:
1. Ottieni una API key gratuita da [Google AI Studio](https://aistudio.google.com/)
2. Imposta la variabile d'ambiente:
   ```
   export GEMINI_API_KEY="la-tua-chiave"
   ```
3. Riavvia l'applicazione

La tua domanda era: "{question}"
"""
