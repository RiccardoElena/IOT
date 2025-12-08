"""
TEST SPY - Robusto con retry e rate limit
"""

from polygon import RESTClient
import pandas as pd
from datetime import datetime, timedelta
import time
from pathlib import Path

API_KEY = "***REMOVED***"
client = RESTClient(API_KEY)

OUTPUT_DIR = Path("test_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

def download_data_robust(ticker, timespan, start_date, end_date, max_retries=10):
    """
    Download con retry automatico su errori 429
    """
    retries = 0
    wait_time = 13  # Base wait: 13 sec (5 req/min = 12 sec)
    
    while retries < max_retries:
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
            
            # Successo!
            return pd.DataFrame(data) if data else pd.DataFrame()
        
        except Exception as e:
            error_str = str(e)
            
            # Se è 429, retry con exponential backoff
            if "429" in error_str or "too many" in error_str.lower():
                retries += 1
                wait_time = min(wait_time * 2, 300)  # Max 5 min
                print(f"\n  ⚠️  Rate limit! Retry {retries}/{max_retries} in {wait_time}s...", end="", flush=True)
                time.sleep(wait_time)
            else:
                # Altro errore, stampa e ritorna vuoto
                print(f"\n  ❌ Error: {e}")
                return pd.DataFrame()
    
    print(f"\n  ❌ Max retries reached")
    return pd.DataFrame()

ticker = "SPY"
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

# crea una directory con timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ticker_dir = OUTPUT_DIR / f"{ticker}_{timestamp}"
ticker_dir.mkdir(exist_ok=True)

print(f"\nDownloading {ticker} (with auto-retry)...")

# DAILY
print("\n[1/3] Daily...", end=" ", flush=True)
df = download_data_robust(ticker, "day", start_date, end_date)
if len(df) > 0:
    df.to_csv(ticker_dir / f"{ticker}_daily.csv", index=False)
    print(f"✅ {len(df)} rows")
else:
    print("❌")
time.sleep(15)  # Extra safety

# HOURLY
print("[2/3] Hourly...", end=" ", flush=True)
df = download_data_robust(ticker, "hour", start_date, end_date)
if len(df) > 0:
    df.to_csv(ticker_dir / f"{ticker}_hourly.csv", index=False)
    print(f"✅ {len(df)} rows")
else:
    print("❌")
time.sleep(15)

# MINUTE (chunked)
print("[3/3] Minute (chunked)...")
all_minutes = []
current = datetime.strptime(start_date, "%Y-%m-%d")
end = datetime.strptime(end_date, "%Y-%m-%d")

chunk_num = 0
while current < end:
    chunk_end = min(current + timedelta(days=90), end)
    chunk_num += 1
    
    print(f"  Chunk {chunk_num}: {current.strftime('%Y-%m-%d')}...", end=" ", flush=True)
    
    df_chunk = download_data_robust(
        ticker, "minute",
        current.strftime("%Y-%m-%d"),
        chunk_end.strftime("%Y-%m-%d")
    )
    
    if len(df_chunk) > 0:
        all_minutes.append(df_chunk)
        print(f"✅ {len(df_chunk):,}")
    else:
        print("❌ Skip")
    
    time.sleep(15)  # 15 sec per sicurezza
    current = chunk_end

if all_minutes:
    df = pd.concat(all_minutes, ignore_index=True)
    df.to_csv(ticker_dir / f"{ticker}_minute.csv", index=False)
    print(f"\n✅ Total: {len(df):,} minute bars")
else:
    print("\n❌ No minute data")

print(f"\nDone: {ticker_dir.absolute()}/")