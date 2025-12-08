"""
Source package for IoT Financial Data Analytics.

This package contains all the core modules:
- data_loader: Data loading and preprocessing
- anomaly_detection: Z-score, percentiles, anomaly detection
- cross_asset: Correlation analysis and multi-asset analysis
- pattern_recognition: Candlestick and chart pattern detection
"""

from . import anomaly_detection
from . import cross_asset
from . import data_loader
from . import pattern_recognition
