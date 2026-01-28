"""
Utils package for common utilities.

Contains reusable utility modules like logging.
"""

from .logger import logger
from .autoscroll import inject_auto_scroll_js
from .dictionaries import get_asset_display_name, get_granularity_display_name, list_available_assets, list_available_granularities
from .dates import get_weeks_in_range, filter_by_date_range, filter_day_data
from .serialization import to_json_string