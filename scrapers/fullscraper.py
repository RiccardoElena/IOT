#!/usr/bin/env python3.10
import time
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from massive import RESTClient

API_KEY = "***REMOVED***"

TICKERS = [
    {"ticker": "SPY", "name": "SP500"},
    {"ticker": "GLD", "name": "Gold"},
    {"ticker": "USO", "name": "Oil"},
    {"ticker": "UUP", "name": "Dollar"},
    {"ticker": "X:BTCUSD", "name": "Bitcoin"}
]

TIMEFRAMES = [
    {"multiplier": 1, "timespan": "minute", "name": "1min", "chunk_days": 7},
    {"multiplier": 1, "timespan": "hour", "name": "1hour", "chunk_days": 20},
    {"multiplier": 1, "timespan": "day", "name": "1day", "chunk_days": 730}
]

END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=730)
RATE_LIMIT_DELAY = 12
MAX_RETRIES = 3
RETRY_DELAY = 5

OUTPUT_DIR = Path("stock_data_2")
CSV_DIR = OUTPUT_DIR / "csv"
JSON_DIR = OUTPUT_DIR / "json"
api_call_counter = 0

def setup_directories():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directories created:")
    print(f"  - CSV: {CSV_DIR}")
    print(f"  - JSON: {JSON_DIR}\n")

def format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def wait_for_rate_limit():
    print(f"  ⏳ Waiting {RATE_LIMIT_DELAY}s for rate limit...")
    time.sleep(RATE_LIMIT_DELAY)

def generate_date_chunks(start: datetime, end: datetime, chunk_days: int) -> List[tuple]:
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks

def download_chunk(client: RESTClient, ticker: str, multiplier: int, timespan: str, 
                   start: str, end: str, chunk_num: int, total_chunks: int) -> List[Dict]:
    global api_call_counter
    attempt = 0
    
    while attempt < MAX_RETRIES:
        try:
            api_call_counter += 1
            print(f"    📡 Chunk {chunk_num}/{total_chunks} - API Call #{api_call_counter} (attempt {attempt + 1}/{MAX_RETRIES})")
            print(f"       Period: {start} to {end}")
            
            data = []
            for agg in client.list_aggs(
                ticker,
                multiplier,
                timespan,
                start,
                end,
                adjusted="true",
                sort="asc",
                limit=50000
            ):
                data.append(agg)
            
            print(f"    ✓ Downloaded {len(data)} bars")
            return data
            
        except Exception as e:
            attempt += 1
            print(f"    ❌ Error: {str(e)}")
            if attempt < MAX_RETRIES:
                print(f"    ⏳ Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    ❌ Failed after {MAX_RETRIES} attempts")
                return []
    
    return []

def download_with_chunks(client: RESTClient, ticker: str, timeframe: Dict, 
                        start: datetime, end: datetime) -> List[Dict]:
    chunks = generate_date_chunks(start, end, timeframe["chunk_days"])
    total_chunks = len(chunks)
    all_data = []
    
    print(f"  📦 Split into {total_chunks} chunks ({timeframe['chunk_days']} days each)")
    
    for idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        start_str = format_date(chunk_start)
        end_str = format_date(chunk_end)
        
        chunk_data = download_chunk(
            client, 
            ticker, 
            timeframe["multiplier"], 
            timeframe["timespan"],
            start_str,
            end_str,
            idx,
            total_chunks
        )
        
        all_data.extend(chunk_data)
        
        if idx < total_chunks:
            wait_for_rate_limit()
    
    return all_data

def save_to_csv(data: List[Dict], filename: str):
    filepath = CSV_DIR / filename
    if not data:
        print(f"⚠️  No data to save to {filename}")
        return
    
    headers = ["timestamp", "open", "high", "low", "close", "volume", "vw", "n"]
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for bar in data:
            row = {
                "timestamp": datetime.fromtimestamp(bar.timestamp / 1000).isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vw": getattr(bar, 'vwap', ''),
                "n": getattr(bar, 'transactions', '')
            }
            writer.writerow(row)
    print(f"✓ Saved CSV: {filepath} ({len(data)} bars)")

def save_to_json(data: List[Dict], filename: str, metadata: Dict[str, Any]):
    filepath = JSON_DIR / filename
    
    json_data = []
    for bar in data:
        json_data.append({
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "vwap": getattr(bar, 'vwap', None),
            "transactions": getattr(bar, 'transactions', None)
        })
    
    output = {"metadata": metadata, "data_points": len(json_data), "data": json_data}
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"✓ Saved JSON: {filepath} ({len(json_data)} bars)")

def main():
    global api_call_counter
    
    print("=" * 70)
    print("MULTI-ASSET DATA DOWNLOADER - MASSIVE API")
    print("=" * 70)
    print(f"Assets: {', '.join([t['name'] for t in TICKERS])}")
    print(f"Period: {format_date(START_DATE)} to {format_date(END_DATE)}")
    print(f"Timeframes: {', '.join([tf['name'] for tf in TIMEFRAMES])}")
    print(f"Rate Limit: {RATE_LIMIT_DELAY}s between calls")
    print(f"⚠️  This will take a while!\n")
    
    setup_directories()
    
    print("🔑 Initializing Massive API client...")
    client = RESTClient(API_KEY)
    print("✓ Client ready\n")
    
    total_combinations = len(TICKERS) * len(TIMEFRAMES)
    current_combo = 0
    start_time = time.time()
    
    for ticker_info in TICKERS:
        ticker = ticker_info["ticker"]
        ticker_name = ticker_info["name"]
        
        print("=" * 70)
        print(f"PROCESSING ASSET: {ticker_name} ({ticker})")
        print("=" * 70)
        
        for timeframe in TIMEFRAMES:
            current_combo += 1
            
            print("-" * 70)
            print(f"[{current_combo}/{total_combinations}] {ticker_name} - {timeframe['name']}")
            print("-" * 70)
            
            data = download_with_chunks(client, ticker, timeframe, START_DATE, END_DATE)
            
            if data:
                metadata = {
                    "ticker": ticker,
                    "asset_name": ticker_name,
                    "timeframe": timeframe["name"],
                    "multiplier": timeframe["multiplier"],
                    "timespan": timeframe["timespan"],
                    "start_date": format_date(START_DATE),
                    "end_date": format_date(END_DATE),
                    "downloaded_at": datetime.now().isoformat(),
                    "adjusted": True,
                    "total_bars": len(data)
                }
                
                safe_ticker = ticker.replace(":", "_")
                base_filename = f"{safe_ticker}_{timeframe['name']}_{format_date(START_DATE)}_to_{format_date(END_DATE)}"
                save_to_csv(data, f"{base_filename}.csv")
                save_to_json(data, f"{base_filename}.json", metadata)
                print(f"✓ Completed {ticker_name} {timeframe['name']}: {len(data)} total bars\n")
            else:
                print(f"⚠️  No data retrieved for {ticker_name} {timeframe['name']}\n")
            
            if current_combo < total_combinations:
                wait_for_rate_limit()
    
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60
    
    print("=" * 70)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"Total assets downloaded: {len(TICKERS)}")
    print(f"Total API calls made: {api_call_counter}")
    print(f"Total time elapsed: {elapsed_minutes:.1f} minutes ({elapsed_minutes/60:.1f} hours)")
    print(f"\nData saved in:")
    print(f"  📊 CSV files: {CSV_DIR}")
    print(f"  📄 JSON files: {JSON_DIR}")
    print("\n🎯 Ready for anomaly detection and pattern recognition!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted after {api_call_counter} API calls")
    except Exception as e:
        print(f"\n\n❌ Fatal error after {api_call_counter} API calls: {str(e)}")
        raise
