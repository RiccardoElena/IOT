# IoT & Data Analytics - Financial Data Analysis

## Project Overview

A comprehensive Streamlit application for analyzing financial market data using IoT-inspired techniques. The project demonstrates real-time data processing, anomaly detection, cross-asset correlation analysis, and technical pattern recognition.

### Dataset

- **5 Assets**: S&P 500, Gold, Oil (WTI), USD Index, Bitcoin
- **3 Granularities**: Minute, Hourly, Daily
- **Time Range**: 2 years of historical data
- **Data Points**: ~500k minute records, ~17k hourly, ~730 daily per asset

## Features

### Page 1: Single Asset Analysis

- Interactive candlestick charts with OHLC data
- Volume analysis with anomaly highlighting
- Z-score based anomaly detection (price, volume, volatility)
- Jump-to-anomaly navigation feature
- **Minute data**: Date range picker + weekly navigation (7-day chunks for performance)
- Downloadable anomaly reports

### 📡 Page 2: Real-time IoT Simulation

- **True streaming simulation**: Starts from 0 points, advances by 1 (like real IoT)
- Sliding window Z-score calculation
- Visual anomaly detection with severity levels (LOW/MEDIUM/HIGH)
- Real-time metrics dashboard
- **Auto-appearing slider** for post-simulation analysis
- Configurable simulation speed

### 🔗 Page 3: Cross-Asset Analysis

- Correlation matrix heatmap (properly oriented, 3 decimal precision)
- Rolling correlation over time with anomaly detection
- Normalized price comparison (base 100)
- Simultaneous anomaly detection across assets
- Systemic event identification and details
- Asset pair deep-dive with scatter plots

### 🔮 Page 4: Pattern Recognition

- **Candlestick Patterns**: Doji, Hammer, Engulfing Bullish/Bearish
- **Chart Patterns**: Double Top, Double Bottom, Head & Shoulders, Cup & Handle
- **Calibration sliders**: Tolerance (1-10%), Prominence (0.5-5%), Window (30-100)
- Visual pattern highlighting on charts
- Pattern distribution statistics
- Timeline view of bullish/bearish signals

## Technical Stack

- **Python 3.10+**
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical computations
- **SciPy**: Signal processing for pattern detection

## Installation

```bash
# Clone or extract the project
cd progetto

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Project Structure

```bash
progetto/
├── app.py                      # Main entry point
├── config.py                   # Global configuration
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── data/
│   ├── minute/                 # 1-minute granularity CSVs
│   ├── hourly/                 # 1-hour granularity CSVs
│   └── daily/                  # 1-day granularity CSVs
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   ├── anomaly_detection.py    # Z-score anomaly detection
│   ├── cross_asset.py          # Multi-asset analysis
│   └── pattern_recognition.py  # Technical pattern detection
│
└── pages/
    ├── 1_analisi_singolo_asset.py   # Single asset page
    ├── 2_anomaly_realtime.py        # Real-time simulation
    ├── 3_cross_asset.py             # Cross-asset analysis
    └── 4_pattern.py                 # Pattern recognition
```

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ZSCORE_ANOMALY_THRESHOLD` | 3.0 | Z-score threshold for anomalies |
| `ZSCORE_WARNING_THRESHOLD` | 2.0 | Z-score threshold for warnings |
| `CORRELATION_WINDOW` | 20 | Days for rolling correlation |
| `SYSTEMIC_EVENT_THRESHOLD` | 3 | Min assets for systemic event |

## Block 6 Updates (Latest)

### Pattern Recognition Fixes

- **Tolerance increased**: 2% → 5% for Double Top/Bottom matching
- **Cup & Handle stricter**:
  - Minimum depth 12% (was lower)
  - Roundness check (must be U-shaped, not V-shaped)
  - Smoothing window increased to 15
- **Calibration sliders** in sidebar for fine-tuning detection

### Real-time Simulation Fixes

- **Starts from zero**: No more pre-loaded window
- **Advances by 1**: True IoT streaming behavior
- **Z-score delayed**: Only calculated after window_size points
- **Slider auto-appears**: No click required after simulation ends

### Minute Data Navigation

- **Date picker**: Select overall date range
- **Weekly navigation**: ← Previous / Next → buttons
- **Clamped to range**: Can't navigate outside selected dates

### Info Boxes (Complete)

Every page now has comprehensive explanations in sidebar expanders:

**Page 1 (Single Asset)**:
- Asset information (what moves each asset)
- Z-Score explanation with formula and interpretation
- Candlestick anatomy and reading guide
- Volume significance and anomaly interpretation
- Volatility concepts
- Data granularity trade-offs

**Page 2 (Real-time)**:
- IoT streaming concepts
- Sliding window explanation with visual
- Rolling Z-score vs standard Z-score
- Anomaly severity levels and response strategies
- Real-world applications

**Page 3 (Cross-Asset)**:
- Pearson correlation formula and interpretation
- Rolling correlation and regime changes
- Systemic events definition and examples
- Safe haven vs risk assets
- Price normalization methodology
- Hedging with correlation

**Page 4 (Pattern)**:
- Candlestick patterns (Doji, Hammer, Engulfing)
- Chart patterns (Double Top/Bottom, H&S, Cup & Handle)
- Signal interpretation (Bullish/Bearish/Neutral)
- Confidence factors and limitations
- Disclaimer for educational purposes

## Anomaly Detection Methodology

### Z-Score Calculation
```
Z = (x - μ) / σ
```
Where:
- x = observed value
- μ = mean of the series
- σ = standard deviation

### Anomaly Types
1. **Price Anomaly**: Unusual closing price movement
2. **Volume Anomaly**: Unusual trading volume
3. **Volatility Anomaly**: Unusual daily range (High - Low)

### Severity Classification (Real-time)
- **LOW**: threshold ≤ |Z| < threshold + 0.5
- **MEDIUM**: threshold + 0.5 ≤ |Z| < threshold + 1.0
- **HIGH**: |Z| ≥ threshold + 1.0

## Pattern Recognition Details

### Candlestick Patterns (1-2 candles)
| Pattern | Signal | Identification |
|---------|--------|----------------|
| Doji | Neutral | Body < 10% of range |
| Hammer | Bullish | Lower shadow ≥ 2× body, small upper shadow |
| Engulfing Bullish | Bullish | Green engulfs previous red |
| Engulfing Bearish | Bearish | Red engulfs previous green |

### Chart Patterns (multi-candle)
| Pattern | Signal | Key Features |
|---------|--------|--------------|
| Double Top | Bearish | Two peaks at similar levels (M shape) |
| Double Bottom | Bullish | Two troughs at similar levels (W shape) |
| Head & Shoulders | Bearish | Three peaks, middle highest |
| Cup & Handle | Bullish | Rounded bottom + small pullback |

## Usage Tips

1. **Start with Daily data** for clearer patterns and faster loading
2. **Adjust thresholds** based on asset volatility (BTC needs higher thresholds)
3. **Use pattern calibration sliders** to tune detection sensitivity
4. **Check correlation changes** during market stress periods
5. **Run full simulation** then use slider for detailed analysis

## Educational Purpose

This project is designed for educational purposes to demonstrate:
- IoT data streaming concepts
- Statistical anomaly detection
- Technical analysis patterns
- Multi-asset correlation analysis

**Disclaimer**: Pattern recognition and anomaly detection are not financial advice. Always conduct your own research and consult qualified professionals for investment decisions.

## License

MIT License - Educational use
