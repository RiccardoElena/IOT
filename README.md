# IoT Financial Data Analytics

A comprehensive financial data analysis platform that applies IoT-inspired techniques to analyze multiple financial assets. The system treats financial instruments as IoT sensors, implementing real-time anomaly detection, cross-asset correlation analysis, and pattern recognition.

## Project Overview

This project demonstrates the application of IoT data processing methodologies to financial time-series data. It analyzes five major financial assets (S&P 500, Gold, Oil, USD Index, Bitcoin) across three time granularities (minute, hourly, daily), implementing sophisticated analytical techniques for anomaly detection and pattern recognition.

### Key Features

- **Real-time Anomaly Detection**: Statistical analysis using Z-score and percentile-based methods
- **Sliding Window Processing**: IoT-style streaming data simulation with dynamic statistical updates
- **Cross-Asset Analysis**: Correlation matrices, systemic event detection, and multi-asset relationship analysis
- **Pattern Recognition**: Candlestick patterns (Doji, Hammer, Engulfing) and chart patterns (Double Top/Bottom, Head & Shoulders, Cup & Handle)
- **AI-Powered Assistant**: Google Gemini integration with multimodal support (text + chart images) for interactive data analysis
- **Interactive Visualization**: Plotly-based dashboards with zoom, filtering, and export capabilities

## Architecture

### Project Structure

```
IOT/
├── app/                          # Main application
│   ├── app.py                   # Streamlit entry point
│   ├── config.py                # Centralized configuration
│   ├── components.py            # Reusable UI components
│   ├── requirements.txt         # Python dependencies
│   ├── pages/                   # Multi-page Streamlit app
│   │   ├── 1_single_asset_analisys.py
│   │   ├── 2_anomaly_realtime.py
│   │   ├── 3_cross_asset.py
│   │   └── 4_pattern.py
│   └── src/                     # Core analysis modules
│       ├── anomaly_detection.py  # Z-score, percentile detection
│       ├── cross_asset.py        # Correlation analysis
│       ├── data_loader.py        # Data ingestion and preprocessing
│       ├── gemini_assistant.py   # AI assistant integration
│       └── pattern_recognition.py # Candlestick and chart patterns
├── data/                        # Time-series datasets
│   ├── 1-minute/               # High-frequency data
│   ├── 2-hourly/               # Mid-frequency data
│   └── 3-daily/                # Low-frequency data
└── scrapers/                    # Data acquisition
    └── fullscraper.py          # Polygon.io API scraper
```

### Technology Stack

**Frontend & Visualization**
- Streamlit 1.31+: Interactive web framework
- Plotly 5.18+: Interactive charting library
- Kaleido: Chart export for AI analysis

**Data Processing**
- Pandas 2.2+: Time-series data manipulation
- NumPy 1.26+: Numerical computing
- SciPy 1.12+: Statistical analysis and signal processing

**AI Integration**
- Google Generative AI 0.5+: Gemini API for intelligent assistance
- PIL (Pillow): Image processing for multimodal AI

**Data Acquisition**
- polygon-api-client: RESTful API for financial data
- python-dotenv: Environment variable management

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Nix package manager for reproducible environments

### Standard Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd IOT
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
cd app
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Nix Installation (Recommended)

If you use Nix package manager:

```bash
nix develop  # Automatic environment setup
```

## Configuration

### Environment Variables

Create a `.env` file in the `app/` directory:

```env
# Gemini AI Assistant (Optional)
GEMINI_API_KEY=your_gemini_api_key_here

# Data Path (Optional, defaults to ../data)
DATA_BASE_PATH=../data

# Polygon.io API (For data scraping only)
API_KEY=your_polygon_api_key_here
```

### Application Configuration

Edit `app/config.py` to customize:

**Asset Configuration**
```python
ASSETS = {
    "sp500": "S&P 500",
    "gold": "Gold",
    "oil": "Oil",
    "usd": "USD Index",
    "btc": "Bitcoin"
}
```

**Anomaly Detection Parameters**
```python
ZSCORE_WARNING_THRESHOLD = 2.0   # Statistical warning threshold
ZSCORE_ANOMALY_THRESHOLD = 3.0   # Statistical anomaly threshold
WINDOW_SIZE_MINUTE = 60          # Rolling window for streaming
```

**Pattern Recognition Parameters**
```python
DOJI_BODY_RATIO = 0.1           # Doji body/range threshold
HAMMER_SHADOW_RATIO = 2.0       # Hammer shadow/body threshold
```

**AI Assistant Parameters**
```python
GEMINI_MODEL = "gemini-2.5-flash-lite"  # Model selection
GEMINI_MAX_TOKENS = 1024                # Response length limit
GEMINI_TEMPERATURE = 0.7                # Creativity level
```

## Usage

### Running the Application

Start the Streamlit application:

```bash
cd app
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Application Pages

**1. Single Asset Analysis**
- Comprehensive OHLC candlestick visualization
- Multi-panel analysis (price, volume, volatility)
- Z-score anomaly detection with configurable thresholds
- Zoom-to-anomaly navigation feature
- Detailed anomaly table with export functionality

**2. Real-time IoT Simulation**
- Streaming data simulation with sliding window processing
- Real-time anomaly detection visualization
- Dynamic chart updates with progress tracking
- Configurable simulation speed and window size
- IoT-style sensor data processing demonstration

**3. Cross-Asset Analysis**
- Correlation heatmap between all assets
- Rolling correlation time-series
- Normalized price comparison (base 100)
- Systemic event detection (multi-asset anomalies)
- Pair-wise relationship analysis with scatter plots

**4. Pattern Recognition**
- Candlestick pattern detection (Doji, Hammer, Engulfing)
- Chart pattern identification (Double Top/Bottom, H&S, Cup & Handle)
- Interactive pattern visualization with clickable legends
- Pattern distribution analysis and timeline
- Calibration sliders for detection sensitivity

### AI Assistant Features

The integrated Gemini AI assistant provides:

**Multimodal Analysis**
- Text-based data interpretation
- Chart image analysis (up to 5 charts simultaneously)
- Visual pattern recognition and anomaly explanation
- Cross-referencing visual and numerical data

**Context-Aware Responses**
- Page-specific data attachment (statistics, anomalies, patterns)
- Selective data inclusion via checkbox interface
- Conversation history management (last 14 messages)
- Fallback to mock mode without API key

**Smart Chart Integration**
- Add charts to conversation context with inline buttons
- Automatic chart export to PNG format
- Chart limit enforcement (maximum 5 charts)
- Real-time simulation chart disabling (enabled only when complete)

## Data Acquisition

### Using the Scraper

The `scrapers/fullscraper.py` script downloads historical data from Polygon.io:

**Features**
- Multi-asset parallel download
- Three timeframe support (minute, hourly, daily)
- Automatic chunking for API rate limit compliance
- Retry mechanism with exponential backoff
- Dual output format (CSV + JSON)
- Progress tracking and error handling

**Configuration**
```python
# Tickers to download
TICKERS = [
    {"ticker": "SPY", "name": "SP500"},
    {"ticker": "GLD", "name": "Gold"},
    {"ticker": "USO", "name": "Oil"},
    {"ticker": "UUP", "name": "Dollar"},
    {"ticker": "X:BTCUSD", "name": "Bitcoin"}
]

# Time range
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=730)  # 2 years

# Rate limiting
RATE_LIMIT_DELAY = 12  # Seconds between API calls
MAX_RETRIES = 3
```

**Running the Scraper**
```bash
cd scrapers
python fullscraper.py
```

**Output Structure**
```
data/
├── csv/
│   ├── SPY_1min_2022-01-01_to_2024-01-01.csv
│   ├── SPY_1hour_2022-01-01_to_2024-01-01.csv
│   └── SPY_1day_2022-01-01_to_2024-01-01.csv
└── json/
    └── [corresponding JSON files with metadata]
```

### Data Format

**CSV Schema**
```csv
timestamp,open,high,low,close,volume,vw,n
1609459200000,372.75,373.53,372.11,373.25,45234567,373.12,123456
```

**Column Definitions**
- `timestamp`: Unix timestamp (milliseconds)
- `open`: Opening price
- `high`: Highest price in period
- `low`: Lowest price in period
- `close`: Closing price
- `volume`: Trading volume
- `vw`: Volume-weighted average price (VWAP)
- `n`: Number of trades

## Analysis Modules

### Anomaly Detection

**Z-Score Method**
```python
from src.anomaly_detection import detect_anomalies

# Batch mode (entire dataset)
df_with_anomalies = detect_anomalies(df, zscore_threshold=3.0, mode="batch")

# Streaming mode (sliding window)
df_streaming = detect_anomalies(df, zscore_threshold=3.0, mode="rolling", window=60)
```

**Features**
- Batch processing: Global statistics (suitable for daily/hourly)
- Rolling processing: Local statistics (suitable for minute/real-time)
- Multi-metric analysis: Price, volume, volatility
- Configurable thresholds and percentile bounds

### Cross-Asset Analysis

**Correlation Computation**
```python
from src.cross_asset import calculate_correlation_matrix, detect_systemic_events

# Pearson correlation on returns
corr_matrix = calculate_correlation_matrix(price_data)

# Rolling correlation
rolling_corr = calculate_rolling_correlation(asset1, asset2, window=30)

# Systemic event detection
events = detect_systemic_events(anomaly_data, threshold=3)
```

**Typical Correlations**
- Gold vs USD: Negative (-0.3 to -0.7)
- Oil vs S&P 500: Positive (0.3 to 0.6)
- Bitcoin vs Traditional: Low to moderate (0.2 to 0.5)

### Pattern Recognition

**Candlestick Patterns**
```python
from src.pattern_recognition import detect_all_candlestick_patterns

df_patterns = detect_all_candlestick_patterns(df)
# Adds columns: pattern_doji, pattern_hammer, pattern_engulfing_bullish, etc.
```

**Chart Patterns**
```python
from src.pattern_recognition import detect_all_chart_patterns

patterns = detect_all_chart_patterns(
    df,
    lookback=50,           # Window size
    tolerance=0.05,        # Price similarity threshold (5%)
    prominence_pct=0.025   # Peak prominence (2.5%)
)
```

**Pattern Calibration Guidelines**

Conservative Setup (Presentation/High Precision):
```python
tolerance = 0.05        # 5% - Only very similar prices
prominence_pct = 0.025  # 2.5% - Only pronounced peaks
lookback = 60           # 60 days - Well-formed patterns
```

Balanced Setup (General Use):
```python
tolerance = 0.07        # 7%
prominence_pct = 0.015  # 1.5%
lookback = 50           # 50 days
```

Aggressive Setup (High Sensitivity):
```python
tolerance = 0.10        # 10%
prominence_pct = 0.008  # 0.8%
lookback = 40           # 40 days
```

## API Reference

### Anomaly Detection Module

**detect_anomalies(df, zscore_threshold, mode, window)**
- Performs comprehensive anomaly detection on OHLC data
- Returns DataFrame with added columns: `zscore_*`, `anomaly_*`
- Modes: `"batch"` (global statistics) or `"rolling"` (sliding window)

**count_anomalies(df)**
- Counts anomalies by type (price, volume, volatility)
- Returns dictionary with counts

**get_anomaly_table(df)**
- Extracts all anomalies with metadata
- Returns DataFrame with timestamp, type, value, zscore, pct_change

### Cross-Asset Module

**load_all_assets(granularity, start_date, end_date)**
- Loads multiple assets for cross-asset analysis
- Returns dictionary mapping asset keys to DataFrames

**calculate_correlation_matrix(price_dict)**
- Computes Pearson correlation on returns
- Returns correlation matrix DataFrame

**detect_systemic_events(anomaly_dict, threshold)**
- Identifies days with multiple concurrent anomalies
- Returns DataFrame with dates and affected assets

### Pattern Recognition Module

**detect_all_candlestick_patterns(df)**
- Detects Doji, Hammer, Engulfing patterns
- Returns DataFrame with boolean columns

**detect_all_chart_patterns(df, lookback, tolerance, prominence_pct)**
- Detects Double Top/Bottom, H&S, Cup & Handle
- Returns list of dictionaries with pattern details

**get_pattern_summary(df)**
- Counts all detected patterns
- Returns dictionary with counts per pattern type

### Gemini Assistant Module

**GeminiAssistant Class**
- `send_message(question, page_context, history, chart_figures)`: Send query with optional chart images
- `add_to_history(role, content)`: Manage conversation history
- `clear_history()`: Reset conversation
- `get_status()`: Check API availability and configuration

**Context Builders**
- `build_single_asset_context(...)`: Context for single asset page
- `build_realtime_context(...)`: Context for simulation page
- `build_cross_asset_context(...)`: Context for correlation page
- `build_pattern_context(...)`: Context for pattern recognition page

## Performance Considerations

### Data Loading

- CSV caching via `@st.cache_data` decorator
- Lazy loading: Only requested assets and granularities
- Date range filtering at load time
- Memory-efficient pandas operations

### Real-time Simulation

- Batch processing: Multiple points per iteration
- Configurable simulation speed (1-100 points/update)
- Progress tracking with early termination
- Chart update throttling

### Pattern Detection

- Sliding window optimization for large datasets
- Vectorized operations using NumPy
- Signal processing via SciPy for peak detection
- Configurable lookback windows to limit search space

### AI Assistant

- Automatic history truncation (last 14 messages)
- Chart image compression and size limits
- Maximum 5 charts per conversation to manage context
- Fallback to text-only mode if chart export fails

## Troubleshooting

### Common Issues

**Issue: "No module named 'config'"**
- Solution: Ensure you're running from the `app/` directory
- Alternative: Set `PYTHONPATH` environment variable

**Issue: "Data file not found"**
- Solution: Verify data files exist in correct directory structure
- Check `DATA_BASE_PATH` in `.env` file
- Ensure file names match `FILE_NAMES` in `config.py`

**Issue: "Gemini API error"**
- Solution: Verify `GEMINI_API_KEY` in `.env`
- Check API quota at Google AI Studio
- Application continues in mock mode without API key

**Issue: "Chart export failed"**
- Solution: Install kaleido: `pip install kaleido`
- Charts will still display, only AI integration affected

**Issue: "Pattern detection finds nothing"**
- Solution: Adjust calibration parameters (decrease thresholds)
- Try different date ranges with more volatility
- Check console for validation errors

### Debug Mode

Enable detailed logging:

```python
# In config.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Development

### Code Style

- PEP 8 compliance for Python code
- Type hints for function signatures
- Comprehensive docstrings (Google style)
- Modular architecture with clear separation of concerns

### Testing

Run manual tests:
```bash
# Test data loading
python -c "from src.data_loader import load_single_asset; print(load_single_asset('btc', 'daily').head())"

# Test anomaly detection
python -c "from src.anomaly_detection import detect_anomalies; import pandas as pd; ..."

# Test pattern recognition
python -c "from src.pattern_recognition import detect_all_candlestick_patterns; ..."
```

### Extending the Project

**Adding New Assets**
1. Update `ASSETS` and `FILE_NAMES` in `config.py`
2. Add corresponding CSV files to `data/` directories
3. Update scraper `TICKERS` list if acquiring new data

**Adding New Patterns**
1. Implement detection function in `pattern_recognition.py`
2. Add to `detect_all_*_patterns()` function
3. Update UI in `4_pattern.py` page
4. Add to context builder in `gemini_assistant.py`

**Adding New Analysis Techniques**
1. Create new module in `src/`
2. Add page in `app/pages/`
3. Import and integrate in `components.py` if reusable
4. Update Gemini system prompt in `config.py`

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Write clear, documented code
4. Test thoroughly before submitting
5. Submit pull request with detailed description

## License

This project is developed for academic purposes as part of an IoT & Data Analytics university course.

## Acknowledgments

- **Data Source**: Polygon.io for financial market data
- **AI Provider**: Google Gemini for intelligent assistance
- **Framework**: Streamlit for rapid prototyping
- **Visualization**: Plotly for interactive charts

## Contact

For questions, issues, or suggestions, please open an issue in the repository or contact the project maintainers.

## References

### Academic References
- Z-Score Anomaly Detection: Standard statistical method for outlier detection
- Sliding Window Processing: IoT streaming data paradigm
- Pearson Correlation: Statistical relationship measurement
- Technical Analysis: Candlestick and chart pattern recognition

### API Documentation
- [Polygon.io API](https://polygon.io/docs/stocks/getting-started)
- [Google Gemini API](https://ai.google.dev/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)

### Financial Analysis Resources
- [Investopedia - Technical Analysis](https://www.investopedia.com/technical-analysis-4689657)
- [Corporate Finance Institute - Pattern Recognition](https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/technical-analysis/)

---

**Version**: 1.0  
**Last Updated**: January 2026  
**Python Version**: 3.10+  
