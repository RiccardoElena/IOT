from config import ASSETS, GRANULARITY_DISPLAY, GRANULARITY_PATHS
from typing import List

def get_asset_display_name(asset: str) -> str:
    """
    Get the display name for an asset.
    
    Args:
        asset: Asset key (e.g., 'sp500')
    
    Returns:
        Display name (e.g., 'S&P 500')
    """
    return ASSETS.get(asset, asset)


def get_granularity_display_name(granularity: str) -> str:
    """
    Get the display name for a granularity.
    
    Args:
        granularity: Granularity key (e.g., 'daily')
    
    Returns:
        Display name (e.g., 'Daily')
    """
    return GRANULARITY_DISPLAY.get(granularity, granularity)


def list_available_assets() -> List[str]:
    """
    Get list of available asset keys.
    
    Returns:
        List of asset keys
    """
    return list(ASSETS.keys())


def list_available_granularities() -> List[str]:
    """
    Get list of available granularity keys.
    
    Returns:
        List of granularity keys
    """
    return list(GRANULARITY_PATHS.keys())