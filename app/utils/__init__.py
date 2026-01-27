"""
Utils package for common utilities.

Contains reusable utility modules like logging.
"""

from .logger import logger
from .autoscroll import inject_auto_scroll_js

__all__ = ["logger", "inject_auto_scroll_js"]