"""
Feature Loaders Package.

This package contains modules for loading and processing features for redispatch forecasting:
- prices_loader: Day-ahead electricity prices
- consumption_loader: Load/consumption data with rolling features
- production_loader: Generation forecasts and actual production by type
- bloomberg_loader: Financial data (commodity prices, spreads)
- wind_pv_loader: Wind and PV forecasts and rolling features from actuals
- cross_border_loader: Cross-border physical flows
- imputation: Missing data handling for all feature types
"""

from .prices_loader import load_day_ahead_prices
from .consumption_loader import load_consumption
from .production_loader import load_production
from .bloomberg_loader import load_bloomberg_data
from .wind_pv_loader import load_wind_pv_features
from .cross_border_loader import load_cross_border_flows
from .sce_loader import load_scheduled_exchanges
from .column_selection import select_relevant_sparse_columns
from .imputation import (
    # Core functions
    detect_gap_lengths,
    compute_seasonal_median,
    impute_with_gap_detection,
    # Generic imputation
    impute_forward_fill_seasonal,
    impute_forward_fill_only,
    impute_fill_zero,
    # Feature-specific imputation
    impute_day_ahead_prices,
    impute_production,
    impute_consumption,
    impute_wind_pv,
    impute_cross_border_flows,
    impute_bloomberg,
    impute_redispatch_core,
    # Utilities
    get_imputation_stats,
    validate_no_missing,
    impute_all_features,
)

__all__ = [
    # Loaders
    "load_day_ahead_prices",
    "load_consumption",
    "load_production",
    "load_bloomberg_data",
    "load_wind_pv_features",
    "load_cross_border_flows",
    "load_scheduled_exchanges",
    # Column selection
    "select_relevant_sparse_columns",
    # Imputation - core
    "detect_gap_lengths",
    "compute_seasonal_median",
    "impute_with_gap_detection",
    # Imputation - generic
    "impute_forward_fill_seasonal",
    "impute_forward_fill_only",
    "impute_fill_zero",
    # Imputation - feature-specific
    "impute_day_ahead_prices",
    "impute_production",
    "impute_consumption",
    "impute_wind_pv",
    "impute_cross_border_flows",
    "impute_bloomberg",
    "impute_redispatch_core",
    # Imputation - utilities
    "get_imputation_stats",
    "validate_no_missing",
    "impute_all_features",
]
