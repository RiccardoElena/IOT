# IoT Financial Data Analytics

## Table of Contents

- [General Information](#general-information)
- [Project Overview](#project-overview)
  - [Key Features](#key-features)
- [Architecture](#architecture)
  - [Project Structure](#project-structure)
  - [Technology Stack](#technology-stack)
    - [Application](#application)
    - [Frontend & Visualization](#frontend--visualization)
    - [Data Processing](#data-processing)
    - [AI Integration](#ai-integration)
    - [Data Acquisition](#data-acquisition)
  - [IDEs & Environments](#ides--environments)
  - [Prerequisites](#prerequisites)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Installation and Usage](#installation-and-usage)
- [Input Data](#input-data)
  - [Data Structure](#data-structure)
  - [Data Format](#data-format)
    - [CSV Schema](#csv-schema)
    - [Column Definitions](#column-definitions)
- [Output](#output)
- [License](#license)
- [API Documentation](#api-documentation)

## General Information

- University: University of Naples Federico II
- Department: DIETI
- Program: Master's in Computer Science
- Course: Operating systems for mobile cloud and IoT
- Academic Year: 2025/2026
- Teacher: Prof. Silvio Barra
- Authors: Pasquale Miranda, Riccardo Elena
- Group: 27
- Project Name: IoT Financial Data Analytics

## Project Overview

This project demonstrates the application of IoT data processing methodologies to financial time-series data. It analyses five major financial assets (S&P 500, Gold, Oil, USD Index, Bitcoin) across three time granularities (minute, hourly, daily), implementing sophisticated analytical techniques for anomaly detection and pattern recognition.

### Key Features

- **Real-time Anomaly Detection**: Statistical analysis using Z-score and percentile-based methods
- **Sliding Window Processing**: IoT-style streaming data simulation with dynamic statistical updates
- **Cross-Asset Analysis**: Correlation matrices, systemic event detection, and multi-asset relationship analysis
- **Pattern Recognition**: Candlestick patterns (Doji, Hammer, Engulfing) and chart patterns (Double Top/Bottom, Head & Shoulders, Cup & Handle)
- **AI-Powered Assistant**: Google Gemini integration with multimodal support (text + chart images) for interactive data analysis
- **Interactive Visualization**: Plotly-based dashboards with zoom, filtering, and export capabilities

## Architecture

### Project Structure

```bash
IOT/
├── README.md
├── app
│   ├── Dockerfile
│   ├── app.py
│   ├── config
│   │   ├── __init__.py
│   │   ├── ai.py
│   │   ├── anomaly.py
│   │   ├── assets.py
│   │   ├── attachment.py
│   │   ├── data.py
│   │   ├── logging.py
│   │   ├── patterns.py
│   │   └── ui.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── core.py
│   │   └── data_loader.py
│   ├── docker-compose.yml
│   ├── mypy.ini
│   ├── pages
│   │   ├── 1_single_asset_analisys.py
│   │   ├── 2_anomaly_realtime.py
│   │   ├── 3_cross_asset.py
│   │   ├── 4_pattern.py
│   │   ├── __init__.py
│   │   └── components
│   │       ├── __init__.py
│   │       ├── attachment.py
│   │       ├── chat.py
│   │       ├── controls.py
│   │       ├── header_footer.py
│   │       └── state.py
│   ├── requirements.txt
│   ├── services
│   │   ├── __init__.py
│   │   ├── analisys
│   │   │   ├── __init__.py
│   │   │   ├── anomaly_detection.py
│   │   │   └── cross_asset.py
│   │   ├── llm
│   │   │   ├── __init__.py
│   │   │   ├── context_builder.py
│   │   │   └── gemini_assistant.py
│   │   └── pattern
│   │       ├── __init__.py
│   │       ├── candlestick.py
│   │       ├── chart.py
│   │       └── helpers.py
│   ├── ui
│   │   ├── __init__.py
│   │   ├── chart.py
│   │   └── controls.py
│   └── utils
│       ├── __init__.py
│       ├── autoscroll.py
│       ├── conversions.py
│       ├── dates.py
│       ├── dictionaries.py
│       └── logger.py
└── scrapers
    └── fullscraper.py
```

### Technology Stack

#### Application

#### Frontend & Visualization

- Streamlit 1.31+: Interactive web framework
- Plotly 5.18+: Interactive charting library
- Kaleido: Chart export for AI analysis

#### Data Processing

- Pandas 2.2+: Time-series data manipulation
- NumPy 1.26+: Numerical computing
- SciPy 1.12+: Statistical analysis and signal processing

#### AI Integration

- Google Generative AI 0.5+: Gemini API for intelligent assistance
- PIL (Pillow): Image processing for multimodal AI

#### Data Acquisition

- polygon-api-client: RESTful API for financial data

The application is containerized using Docker for consistent deployment across environments.

### IDEs & Environments

- Visual Studio Code
- NeoVIM

### Prerequisites

- Docker & Docker Compose installed on your machine

## Configuration

### Environment Variables

Two `.env` files are required:

1. **Data Scraper Configuration** (`scrapers/.env`):
    - `POLYGON_API_KEY`: Your Polygon.io API key for data retrieval.
    - `BASE_DATA`: Base directory for storing downloaded data (e.g., `../data/`).
2. **Application Configuration** (`app/.env`):
    - `DATA_PATH_LOCAL`: Path to the local data directory (usually matches `BASE_DATA` in scraper).
    - `GEMINI_API_KEY`: Your Google Gemini API key for AI assistant functionality (optional).
    - `CONTAINER_DATA_DIR`: Directory inside the Docker container where data is mounted.

### Installation and Usage

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd IOT
    ```

2. Follow the [Configuration](#configuration) steps to set up the environment variables.

3. Retrieve data:
    - Use the provided scraper in `scrapers/fullscraper.py` to download historical data from Polygon.io (requires API key).
    - If you already have the data, ensure they follow the strucutre outlined in the [Input Data](#input-data) section.

4. Run the application:

    ```bash
    cd app
    docker-compose up --build
    ```

The application will be accessible at `http://localhost:8501`.

## Input Data

Input data consists of historical OHLC (Open, High, Low, Close) time-series for five financial assets (S&P 500, Gold, Oil, Bitcoin, USD) at three granularities: minute, hourly, and daily.

The data are retrieved from Polygon.io and stored as CSV files.

### Data Structure

```bash
data/
├── 1-minute/
│   ├── btc.csv
│   ├── gold.csv
│   ├── oil.csv
│   ├── sp500.csv
│   └── usd.csv
├── 2-hourly/
│   ├── btc.csv
│   ├── gold.csv
│   ├── oil.csv
│   ├── sp500.csv
│   └── usd.csv
└── 3-daily/
    ├── btc.csv
    ├── gold.csv
    ├── oil.csv
    ├── sp500.csv
    └── usd.csv
```

### Data Format

#### CSV Schema

```csv
timestamp,open,high,low,close,volume,vw,n
1609459200000,372.75,373.53,372.11,373.25,45234567,373.12,123456
```

#### Column Definitions

- `timestamp`: Unix timestamp (milliseconds)
- `open`: Opening price
- `high`: Highest price in period
- `low`: Lowest price in period
- `close`: Closing price
- `volume`: Trading volume
- `vw`: Volume-weighted average price (VWAP)
- `n`: Number of trades

## Output

This application consists in an interactive web dashboard with multiple pages for different analyses:

- **Single Asset Analysis**: Visualize OHLC data with detected anomalies and patterns
- **Real-time Anomaly Detection**: Simulate streaming data with live anomaly detection
- **Cross-Asset Analysis**: Correlation matrices and systemic event detection
- **Pattern Recognition**: Detect and visualize candlestick and chart patterns

## License

This project is developed for academic purposes as part of an IoT & Data Analytics university course.

## API Documentation

- [Polygon.io API](https://polygon.io/docs/stocks/getting-started)
- [Google Gemini API](https://ai.google.dev/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)

---

**Version**: 1.0  
**Last Updated**: January 2026  
**Python Version**: 3.10+
