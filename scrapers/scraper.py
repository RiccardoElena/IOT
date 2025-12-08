"""
TEST SPY - Enhanced Dataset
"""

from massive import RESTClient
import pandas as pd
from datetime import datetime, timedelta
import time
from pathlib import Path

API_KEY = "***REMOVED***"
client = RESTClient(API_KEY)

OUTPUT_DIR = Path("test_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

def download_data(ticker, timespan, start_date, end_date):
    try:
        data = []
        for agg in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan=timespan,
            from_=start_date,
            to=end_date,
            limit=50000
        ):
            row = {
                'datetime': datetime.fromtimestamp(agg.timestamp/1000),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume,
            }
            if agg.vwap:
                row['vwap'] = agg.vwap
            if agg.transactions:
                row['transactions'] = agg.transactions
            
            data.append(row)
        
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

ticker = "SPY"
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

ticker_dir = OUTPUT_DIR / ticker
ticker_dir.mkdir(exist_ok=True)

print(f"\nDownloading {ticker}...")

# DAILY
print("Daily...", end=" ", flush=True)
df = download_data(ticker, "day", start_date, end_date)
if len(df) > 0:
    df.to_csv(ticker_dir / f"{ticker}_daily.csv", index=False)
    print(f"✅ {len(df)} rows")
time.sleep(15)

# HOURLY
print("Hourly...", end=" ", flush=True)
df = download_data(ticker, "hour", start_date, end_date)
if len(df) > 0:
    df.to_csv(ticker_dir / f"{ticker}_hourly.csv", index=False)
    print(f"✅ {len(df)} rows")
time.sleep(15)

# MINUTE
print("Minute...")
all_minutes = []
current = datetime.strptime(start_date, "%Y-%m-%d")
end = datetime.strptime(end_date, "%Y-%m-%d")

while current < end:
    chunk_end = min(current + timedelta(days=90), end)
    print(f"  {current.strftime('%Y-%m-%d')}...", end=" ", flush=True)
    
    df_chunk = download_data(ticker, "minute", current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
    if len(df_chunk) > 0:
        all_minutes.append(df_chunk)
        print(f"✅ {len(df_chunk):,}")
    
    time.sleep(15)
    current = chunk_end

if all_minutes:
    df = pd.concat(all_minutes, ignore_index=True)
    df.to_csv(ticker_dir / f"{ticker}_minute.csv", index=False)
    print(f"\nTotal: {len(df):,} rows")

print(f"\nDone. Files in: {ticker_dir}/")