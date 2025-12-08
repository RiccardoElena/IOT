"""
TEST SPY - Massive.com API (ex Polygon.io)
Con logging dettagliato e retry robusto
"""

from massive import RESTClient
import pandas as pd
from datetime import datetime, timedelta
import time
from pathlib import Path
import sys

API_KEY = "TUA_CHIAVE_QUI"
client = RESTClient(API_KEY)

OUTPUT_DIR = Path("test_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

def log(msg, level="INFO"):
    """Log dettagliato"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "ERROR": "❌",
        "RETRY": "🔄",
        "PROGRESS": "⏳"
    }.get(level, "  ")
    
    full_msg = f"[{timestamp}] {prefix} {msg}"
    print(full_msg, flush=True)
    
    with open(OUTPUT_DIR / "download.log", "a") as f:
        f.write(full_msg + "\n")

def download_with_retry(ticker, timespan, start_date, end_date, max_retries=5):
    """Download con retry e logging dettagliato"""
    
    log(f"Starting download: {ticker} | {timespan} | {start_date} to {end_date}")
    
    for attempt in range(1, max_retries + 1):
        try:
            log(f"Attempt {attempt}/{max_retries}", "PROGRESS")
            
            data = []
            row_count = 0
            
            # Chiama list_aggs
            log(f"Calling client.list_aggs()...")
            aggs_iterator = client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                limit=50000
            )
            
            log(f"Iterator created, fetching data...")
            
            # Itera sui risultati
            for agg in aggs_iterator:
                row = {
                    'datetime': datetime.fromtimestamp(agg.timestamp/1000),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                }
                
                if hasattr(agg, 'vwap') and agg.vwap:
                    row['vwap'] = agg.vwap
                if hasattr(agg, 'transactions') and agg.transactions:
                    row['transactions'] = agg.transactions
                
                data.append(row)
                row_count += 1
                
                # Progress ogni 500 righe
                if row_count % 500 == 0:
                    log(f"Fetched {row_count} rows...", "PROGRESS")
            
            # Successo
            df = pd.DataFrame(data)
            log(f"Download completed: {len(df)} rows", "SUCCESS")
            
            if len(df) > 0:
                log(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                log(f"Columns: {', '.join(df.columns)}")
            
            return df
            
        except Exception as e:
            error_type = type(e).name
            error_msg = str(e)
            
            log(f"ERROR: {error_type}", "ERROR")
            log(f"Message: {error_msg}", "ERROR")
            
            # Stampa stack trace completo
            import traceback
            log("Stack trace:", "ERROR")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    log(f"  {line}", "ERROR")
            
            # Determina se è rate limit
            is_rate_limit = any(x in error_msg.lower() for x in ['429', 'rate', 'too many'])
            
            if attempt < max_retries:
                if is_rate_limit:
                    wait = 60 * attempt  # 60s, 120s, 180s...
                    log(f"Rate limit detected. Waiting {wait}s...", "RETRY")
