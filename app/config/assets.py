"""
Asset configuration for IoT Financial Data Analytics.

Defines the financial assets tracked by the system and their file mappings.
"""

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
