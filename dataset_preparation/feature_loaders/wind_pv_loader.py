"""
Wind/PV Loader.

This module provides functions to load wind and PV generation data for a TSO,
including day-ahead forecasts and leakage-free rolling features from actuals.

Feature Reduction Strategy:
- Add wind_total aggregate (onshore + offshore)
- Keep only lag_1h + rollingmean_7d for actuals
- Drop rollingstd, rolling_absdev, lag_48h
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from dataset_preparation.feature_loaders.imputation import impute_wind_pv

from ..tso_config import normalize_tso_name

logger = logging.getLogger(__name__)

# Default path relative to project root
DEFAULT_WIND_PV_PATH = Path("data/wind_pv_generation/entsoe_wind_pv_generation_extended.csv")

# Mapping from canonical TSO names to abbreviations used in wind/PV data
TSO_ABBREV_MAP = {
    "50Hertz": "50HzT",
    "TenneT DE": "TenneT",
    "Amprion": "Amprion",
    "TransnetBW": "TransnetBW",
}

# Types of renewable generation by TSO
# Not all TSOs have offshore wind (only 50Hertz and TenneT)
WIND_PV_TYPES_BY_TSO = {
    "50Hertz": ["Solar", "Wind_Offshore", "Wind_Onshore"],
    "TenneT DE": ["Solar", "Wind_Offshore", "Wind_Onshore"],
    "Amprion": ["Solar", "Wind_Onshore"],
    "TransnetBW": ["Solar", "Wind_Onshore"],
}


def _get_tso_abbrev(tso: str) -> str:
    """Get the abbreviation used in wind/PV data for a TSO."""
    tso_normalized = normalize_tso_name(tso)
    return TSO_ABBREV_MAP.get(tso_normalized, tso_normalized)


def compute_lagged_rolling_features(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "begin_date",
    window_days: int = 7,
    minimal: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling statistics that don't leak future information.
    
    For each timestamp, the rolling window includes the same hour-of-day
    from the past N days (excluding current day).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the value column and date column.
    value_col : str
        Name of the column to compute rolling features for.
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
    window_days : int, optional
        Number of days to include in the rolling window. Default is 7.
    minimal : bool, optional
        If True, compute only rolling mean (not std/absdev). Default is True.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns:
        - {value_col}_rollingmean_{window_days}d
        - {value_col}_rollingstd_{window_days}d (if minimal=False)
        - {value_col}_rolling_absdev_{window_days}d (if minimal=False)
    """
    df = df.copy()
    df = df.sort_values(date_col)
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract hour and date for grouping
    df["_hour"] = df[date_col].dt.hour
    df["_date_only"] = df[date_col].dt.date
    
    # Column names
    rolling_mean_col = f"{value_col}_rollingmean_{window_days}d"
    rolling_std_col = f"{value_col}_rollingstd_{window_days}d"
    rolling_absdev_col = f"{value_col}_rolling_absdev_{window_days}d"
    
    results = []
    
    for hour, hour_df in df.groupby("_hour"):
        hour_df = hour_df.sort_values("_date_only").copy()
        
        # shift(1) excludes the current day's value
        shifted_values = hour_df[value_col].shift(1)
        
        # Rolling mean
        hour_df[rolling_mean_col] = shifted_values.rolling(
            window=window_days, min_periods=max(1, window_days // 2)
        ).mean().bfill()
        
        if not minimal:
            # Rolling std
            hour_df[rolling_std_col] = shifted_values.rolling(
                window=window_days, min_periods=max(1, window_days // 2)
            ).std().bfill()
            
            # Rolling mean of absolute values
            hour_df[rolling_absdev_col] = hour_df[value_col].abs().shift(1).rolling(
                window=window_days, min_periods=max(1, window_days // 2)
            ).mean().bfill()
        
        results.append(hour_df)
    
    result_df = pd.concat(results, ignore_index=True)
    result_df = result_df.sort_values(date_col).reset_index(drop=True)
    result_df = result_df.drop(columns=["_hour", "_date_only"])
    
    return result_df


def compute_lags(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "begin_date",
    lag_hours: list[int] = [1],
) -> pd.DataFrame:
    """
    Compute lagged values for a column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the value column and date column.
    value_col : str
        Name of the column to compute lags for.
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
    lag_hours : list[int], optional
        List of lag hours to compute. Default is [1] (reduced from [24, 48]).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns: {value_col}_lag_{h}h for each h in lag_hours
    """
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    
    for lag_h in lag_hours:
        lag_col = f"{value_col}_lag_{lag_h}h"
        df[lag_col] = df[value_col].shift(lag_h).bfill()
    
    return df


def load_wind_pv_features(
    tso: str,
    rolling_window_days: int = 7,
    wind_pv_path: Path | str | None = None,
    oos_start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load wind/PV forecasts and compute leakage-free rolling features from actuals.
    
    This function loads:
    1. Day-ahead forecasts for wind (onshore/offshore) and PV/solar
    2. Actual generation transformed to leakage-free rolling features and lags
    
    Feature Reduction Strategy:
    - Add wind_total aggregate (onshore + offshore) for forecasts and actuals
    - Keep only lag_1h + rollingmean_7d for actuals
    - Drop rollingstd, rolling_absdev, lag_48h
    - Output: ~4-6 features per TSO (forecasts + lag + rolling for wind_total and solar)
    
    Parameters
    ----------
    tso : str
        The TSO name (e.g., "50Hertz", "TenneT DE", "Amprion", "TransnetBW").
        Will be normalized internally.
    rolling_window_days : int, optional
        Number of days for rolling window. Default is 7.
    wind_pv_path : Path | str | None, optional
        Path to the wind/PV CSV. If None, uses the default path.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - begin_date: datetime column (not index)
        - wind_forecast_total_de_{tso}: aggregated wind forecast (onshore + offshore)
        - pv_forecast_de_{tso}: day-ahead PV/solar forecast
        - wind_actual_total_rollingmean_{window}d_de_{tso}: rolling mean of total wind
        - wind_actual_total_lag_1h_de_{tso}: 1-hour lag of total wind
        - pv_actual_rollingmean_{window}d_de_{tso}: rolling mean
        - pv_actual_lag_1h_de_{tso}: 1-hour lag
        
    Notes
    -----
    - Forecasts are included directly (no leakage issue as they are day-ahead)
    - Actuals are transformed to rolling features and lags only (minimal set)
    - Raw actuals and forecast errors are NOT included to prevent data leakage
    """
    # Determine file path
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    
    if wind_pv_path is None:
        wind_pv_path = project_root / DEFAULT_WIND_PV_PATH
    else:
        wind_pv_path = Path(wind_pv_path)
    
    # Normalize TSO name and get info
    tso_normalized = normalize_tso_name(tso)
    tso_abbrev = _get_tso_abbrev(tso)
    tso_lower = tso_abbrev.lower()
    
    # Get types available for this TSO
    gen_types = WIND_PV_TYPES_BY_TSO.get(tso_normalized, ["Solar", "Wind_Onshore"])
    
    if not wind_pv_path.exists():
        logger.warning(f"Wind/PV file not found: {wind_pv_path}")
        return pd.DataFrame(columns=["begin_date"])
    
    try:
        df = pd.read_csv(wind_pv_path, parse_dates=["begin_date"])
    except Exception as e:
        logger.error(f"Error loading wind/PV data from {wind_pv_path}: {e}")
        return pd.DataFrame(columns=["begin_date"])
    
    result_df = df[["begin_date"]].copy()

    seasonal_fit_mask = None
    if oos_start_date is not None:
        oos_ts = pd.to_datetime(oos_start_date)
        seasonal_fit_mask = df["begin_date"] < oos_ts

    df = impute_wind_pv(
        df, 
        forecast_cols=[col for col in df.columns if col.startswith("forecast_day_ahead_DE_")],
        actual_cols=[col for col in df.columns if col.startswith("generation_DE_")],
        use_sparse_imputation=True,
        pv_night_zero=True,
        pv_level_adjustment_actuals=True,
        seasonal_fit_mask=seasonal_fit_mask,
    )
    
    # -------------------------------------------------------------------------
    # PART 1: Load forecasts and create aggregates
    # -------------------------------------------------------------------------
    # Collect wind forecast columns for aggregation
    wind_forecast_cols = []
    wind_actual_cols = []
    
    for gen_type in gen_types:
        forecast_col = f"forecast_day_ahead_DE_{tso_abbrev}_{gen_type}"
        actual_col = f"generation_DE_{tso_abbrev}_{gen_type}"
        
        if gen_type == "Solar":
            # PV forecast (keep as-is)
            if forecast_col in df.columns:
                result_df[f"pv_forecast_de_{tso_lower}"] = df[forecast_col]
        elif gen_type.startswith("Wind"):
            # Collect wind columns for aggregation
            if forecast_col in df.columns:
                wind_forecast_cols.append(forecast_col)
            if actual_col in df.columns:
                wind_actual_cols.append(actual_col)
    
    # Create wind_total forecast aggregate
    if wind_forecast_cols:
        result_df[f"wind_forecast_total_de_{tso_lower}"] = df[wind_forecast_cols].sum(axis=1)
    
    # Create wind_total actual aggregate for rolling/lag computation
    if wind_actual_cols:
        df["_wind_actual_total"] = df[wind_actual_cols].sum(axis=1)
    
    # -------------------------------------------------------------------------
    # PART 2: Compute rolling features + lags for aggregates only
    # -------------------------------------------------------------------------
    # Wind total actuals
    if "_wind_actual_total" in df.columns:
        prefix = "wind_actual_total"
        suffix = f"de_{tso_lower}"
        
        # Rolling features (only mean)
        temp_df = df[["begin_date", "_wind_actual_total"]].copy()
        temp_df = compute_lagged_rolling_features(
            temp_df, "_wind_actual_total", window_days=rolling_window_days, minimal=True
        )
        rolling_mean_col = f"_wind_actual_total_rollingmean_{rolling_window_days}d"
        if rolling_mean_col in temp_df.columns:
            result_df[f"{prefix}_rollingmean_{rolling_window_days}d_{suffix}"] = temp_df[rolling_mean_col].values
        
        # Lag (only 1h)
        lag_df = compute_lags(df[["begin_date", "_wind_actual_total"]].copy(), "_wind_actual_total", lag_hours=[1])
        lag_1_col = "_wind_actual_total_lag_1h"
        if lag_1_col in lag_df.columns:
            result_df[f"{prefix}_lag_1h_{suffix}"] = lag_df[lag_1_col].values
    
    # PV/Solar actuals
    pv_actual_col = f"generation_DE_{tso_abbrev}_Solar"
    if pv_actual_col in df.columns:
        prefix = "pv_actual"
        suffix = f"de_{tso_lower}"
        
        # Rolling features (only mean)
        temp_df = df[["begin_date", pv_actual_col]].copy()
        temp_df = compute_lagged_rolling_features(
            temp_df, pv_actual_col, window_days=rolling_window_days, minimal=True
        )
        rolling_mean_col = f"{pv_actual_col}_rollingmean_{rolling_window_days}d"
        if rolling_mean_col in temp_df.columns:
            result_df[f"{prefix}_rollingmean_{rolling_window_days}d_{suffix}"] = temp_df[rolling_mean_col].values
        
        # Lag (only 1h)
        lag_df = compute_lags(df[["begin_date", pv_actual_col]].copy(), pv_actual_col, lag_hours=[1])
        lag_1_col = f"{pv_actual_col}_lag_1h"
        if lag_1_col in lag_df.columns:
            result_df[f"{prefix}_lag_1h_{suffix}"] = lag_df[lag_1_col].values
    
    logger.info(
        f"Loaded wind/PV data for {tso_normalized}: "
        f"{len(result_df)} rows, {len(result_df.columns) - 1} feature columns"
    )
    
    return result_df