#!/usr/bin/env python3
"""
Stock Data Downloader - Test Version
Downloads OHLC data for S&P 500 (SPY) with multiple timeframes
Respects 5 API calls/minute rate limit
Saves data in both CSV and JSON formats
"""

import time
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from massive import RESTClient

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = "***REMOVED***"  # ⚠️ REPLACE WITH YOUR ACTUAL API KEY

# Test configuration - Single ticker
TEST_TICKER = "SPY"  # S&P 500 ETF

# Timeframes to download
TIMEFRAMES = [
    {"multiplier": 1, "timespan": "minute", "name": "1min"},
    {"multiplier": 1, "timespan": "hour", "name": "1hour"},
    {"multiplier": 1, "timespan": "day", "name": "1day"}
]

# Date range - 2 years back from today
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=730)  # 2 years

# Rate limiting: 5 calls per minute = 1 call every 13 seconds
RATE_LIMIT_DELAY = 13  # seconds between API calls
MAX_RETRIES = 5
RETRY_DELAY = 7  # seconds

# Output directories
OUTPUT_DIR = Path("stock_data")
CSV_DIR = OUTPUT_DIR / "csv"
JSON_DIR = OUTPUT_DIR / "json"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_directories():
    """Create output directories if they don't exist"""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directories created:")
    print(f"  - CSV: {CSV_DIR}")
    print(f"  - JSON: {JSON_DIR}\n")


def format_date(dt: datetime) -> str:
    """Format datetime to YYYY-MM-DD string"""
    return dt.strftime("%Y-%m-%d")


def wait_for_rate_limit():
    """Pause execution to respect rate limit"""
    print(f"⏳ Waiting {RATE_LIMIT_DELAY}s for rate limit...")
    time.sleep(RATE_LIMIT_DELAY)


def save_to_csv(data: List[Dict], filename: str):
    """Save data to CSV file"""
    filepath = CSV_DIR / filename
    
    if not data:
        print(f"⚠️  No data to save to {filename}")
        return
    
    # CSV headers based on API response
    headers = ["timestamp", "open", "high", "low", "close", "volume", "vw", "n"]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for bar in data:
            # Convert timestamp to readable format
            row = {
                "timestamp": datetime.fromtimestamp(bar["t"] / 1000).isoformat(),
                "open": bar["o"],
                "high": bar["h"],
                "low": bar["l"],
                "close": bar["c"],
                "volume": bar["v"],
                "vw": bar.get("vw", ""),
                "n": bar.get("n", "")
            }
            writer.writerow(row)
    
    print(f"✓ Saved CSV: {filepath} ({len(data)} bars)")


def save_to_json(data: List[Dict], filename: str, metadata: Dict[str, Any]):
    """Save data to JSON file with metadata"""
    filepath = JSON_DIR / filename
    
    output = {
        "metadata": metadata,
        "data_points": len(data),
        "data": data
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Saved JSON: {filepath} ({len(data)} bars)")


def download_data_with_retry(client: RESTClient, ticker: str, timeframe: Dict, 
                             start: str, end: str) -> List[Dict]:
    """
    Download data with retry logic and pagination handling
    """
    all_data = []
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            print(f"  📡 Fetching {timeframe['name']} data (attempt {attempt + 1}/{MAX_RETRIES})...")
            
            # Use the list_aggs method with pagination
            for agg in client.list_aggs(
                ticker,
                timeframe["multiplier"],
                timeframe["timespan"],
                start,
                end,
                adjusted=True,
                sort="asc",
                limit=50000  # Maximum allowed
            ):
                all_data.append(agg)
            
            print(f"  ✓ Downloaded {len(all_data)} data points")
            return all_data
            
        except Exception as e:
            attempt += 1
            print(f"  ❌ Error: {str(e)}")
            
            if attempt < MAX_RETRIES:
                print(f"  ⏳ Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ❌ Failed after {MAX_RETRIES} attempts")
                return []
    
    return all_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("=" * 70)
    print("STOCK DATA DOWNLOADER - TEST MODE")
    print("=" * 70)
    print(f"Ticker: {TEST_TICKER}")
    print(f"Period: {format_date(START_DATE)} to {format_date(END_DATE)}")
    print(f"Timeframes: {', '.join([tf['name'] for tf in TIMEFRAMES])}")
    print(f"Rate Limit: {RATE_LIMIT_DELAY}s between calls\n")
    
    # Setup
    setup_directories()
    
    # Initialize client
    print("🔑 Initializing API client...")
    client = RESTClient(API_KEY)
    print("✓ Client ready\n")
    
    # Download data for each timeframe
    start_str = format_date(START_DATE)
    end_str = format_date(END_DATE)
    
    total_calls = len(TIMEFRAMES)
    current_call = 0
    
    for timeframe in TIMEFRAMES:
        current_call += 1
        
        print("-" * 70)
        print(f"[{current_call}/{total_calls}] Processing {TEST_TICKER} - {timeframe['name']}")
        print("-" * 70)
        
        # Download data
        data = download_data_with_retry(
            client, 
            TEST_TICKER, 
            timeframe, 
            start_str, 
            end_str
        )
        
        if data:
            # Prepare metadata
            metadata = {
                "ticker": TEST_TICKER,
                "timeframe": timeframe["name"],
                "multiplier": timeframe["multiplier"],
                "timespan": timeframe["timespan"],
                "start_date": start_str,
                "end_date": end_str,
                "downloaded_at": datetime.now().isoformat(),
                "adjusted": True
            }
            
            # Generate filenames
            base_filename = f"{TEST_TICKER}_{timeframe['name']}_{start_str}_to_{end_str}"
            csv_filename = f"{base_filename}.csv"
            json_filename = f"{base_filename}.json"
            
            # Save to both formats
            save_to_csv(data, csv_filename)
            save_to_json(data, json_filename, metadata)
            
            print(f"✓ Completed {timeframe['name']}\n")
        else:
            print(f"⚠️  No data retrieved for {timeframe['name']}\n")
        
        # Rate limiting (except for last call)
        if current_call < total_calls:
            wait_for_rate_limit()
    
    # Summary
    print("=" * 70)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"Check your data in:")
    print(f"  📊 CSV files: {CSV_DIR}")
    print(f"  📄 JSON files: {JSON_DIR}")
    print("\nReady for anomaly detection and pattern recognition! 🎯")


if name == "main":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        raise
