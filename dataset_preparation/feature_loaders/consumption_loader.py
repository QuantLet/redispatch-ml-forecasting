"""
Consumption Loader.

This module provides functions to load consumption/load data for a TSO
and its neighboring countries, with leakage-free rolling features and lags.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from dataset_preparation.feature_loaders.imputation import impute_consumption

from ..tso_config import get_neighbors, normalize_tso_name

logger = logging.getLogger(__name__)

# Default path to consumption data relative to project root
DEFAULT_CONSUMPTION_PATH = Path("data/consumption_new/entsoe_consumption_full_no_imputation.csv")

# Mapping from canonical TSO names to abbreviations used in consumption data
TSO_ABBREV_MAP = {
    "50Hertz": "50HzT",
    "TenneT DE": "TenneT",
    "Amprion": "Amprion",
    "TransnetBW": "TransnetBW",
}


def _get_tso_abbrev(tso: str) -> str:
    """Get the abbreviation used in consumption data for a TSO."""
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
        If True (default), only compute rollingmean. If False, also compute
        rollingstd and rolling_absdev.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns:
        - {value_col}_rollingmean_{window_days}d
        - {value_col}_rollingstd_{window_days}d (only if minimal=False)
        - {value_col}_rolling_absdev_{window_days}d (only if minimal=False)
        
    Notes
    -----
    For timestamp T on day D at hour H, the rolling window uses:
    - Days D-N to D-1 (N = window_days), same hour H
    - Current day D is EXCLUDED to prevent data leakage
    
    Examples
    --------
    >>> df = compute_lagged_rolling_features(df, 'actual_load_de_tennet', window_days=7)
    """
    df = df.copy()
    df = df.sort_values(date_col)
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract hour and date for grouping
    df["_hour"] = df[date_col].dt.hour
    df["_date_only"] = df[date_col].dt.date
    
    # For each hour, compute rolling stats over past days
    rolling_mean_col = f"{value_col}_rollingmean_{window_days}d"
    
    results = []
    
    for hour, hour_df in df.groupby("_hour"):
        hour_df = hour_df.sort_values("_date_only").copy()
        
        # shift(1) excludes the current day's value
        # rolling window then includes the past window_days values
        shifted_values = hour_df[value_col].shift(1)
        
        # Rolling mean (exclude current day)
        hour_df[rolling_mean_col] = shifted_values.rolling(
            window=window_days, min_periods=max(1, window_days // 2)
        ).mean().bfill()
        
        # Only compute std and absdev if not minimal
        if not minimal:
            rolling_std_col = f"{value_col}_rollingstd_{window_days}d"
            rolling_absdev_col = f"{value_col}_rolling_absdev_{window_days}d"
            
            # Rolling std (exclude current day)
            hour_df[rolling_std_col] = shifted_values.rolling(
                window=window_days, min_periods=max(1, window_days // 2)
            ).std().bfill()
            
            # Rolling mean of absolute values (for deviation tracking)
            hour_df[rolling_absdev_col] = hour_df[value_col].abs().shift(1).rolling(
                window=window_days, min_periods=max(1, window_days // 2)
            ).mean().bfill()
        
        results.append(hour_df)
    
    # Combine and restore original order
    result_df = pd.concat(results, ignore_index=True)
    result_df = result_df.sort_values(date_col).reset_index(drop=True)
    
    # Clean up temporary columns
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
        List of lag hours to compute. Default is [1] (reduced for minimal feature set).
        
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


def load_consumption(
    tso: str,
    data_path: Path | str | None = None,
    window_days: int = 7,
    minimal: bool = True,
    oos_start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load consumption data for TSO and neighboring countries.
    
    This function loads consumption/load data from ENTSO-E, processes it with
    leakage-free rolling features for actuals, and returns forecasts directly.
    
    Parameters
    ----------
    tso : str
        The TSO name (e.g., "50Hertz", "TenneT DE", "Amprion", "TransnetBW").
        Will be normalized internally.
    data_path : Path | str | None, optional
        Path to the consumption CSV file. If None, uses the default path.
    window_days : int, optional
        Number of days for rolling window. Default is 7.
    minimal : bool, optional
        If True (default), only keep rollingmean and lag_1h for actuals.
        If False, include rollingstd, rolling_absdev, and lag_48h.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - begin_date: datetime column (not index)
        - consumption_forecast_load_de_{tso}: TSO's load forecast
        - consumption_forecast_load_{neighbor}: neighbor's load forecast  
        - consumption_load_actual_rollingmean_{window_days}d_de_{tso}: rolling mean of TSO's actual load
        - consumption_load_actual_lag_1h_de_{tso}: 1-hour lag of TSO's actual load
        - Same rolling and lag features for neighbors (if minimal=False, includes more)
        
    Notes
    -----
    - Forecasts are included directly (no leakage issue)
    - Actuals are transformed to rolling features and lags only
    - Raw actuals are NOT included to prevent data leakage
    - Returns empty DataFrame with only 'begin_date' if file not found
    
    Examples
    --------
    >>> df = load_consumption("50Hertz")
    >>> 'consumption_forecast_load_de_50hzt' in df.columns
    True
    """
    # Determine file path
    if data_path is None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        data_path = project_root / DEFAULT_CONSUMPTION_PATH
    else:
        data_path = Path(data_path)
    
    # Check if file exists
    if not data_path.exists():
        logger.warning(f"Consumption file not found: {data_path}")
        return pd.DataFrame(columns=["begin_date"])
    
    # Normalize TSO name and get abbreviation
    tso_normalized = normalize_tso_name(tso)
    tso_abbrev = _get_tso_abbrev(tso)
    neighbors = get_neighbors(tso_normalized)
    
    # Load the data
    try:
        df = pd.read_csv(data_path, parse_dates=["begin_date"])
    except Exception as e:
        logger.error(f"Error loading consumption data from {data_path}: {e}")
        return pd.DataFrame(columns=["begin_date"])
    
    # Start with begin_date
    result_df = df[["begin_date"]].copy()

    seasonal_fit_mask = None
    if oos_start_date is not None:
        oos_ts = pd.to_datetime(oos_start_date)
        seasonal_fit_mask = df["begin_date"] < oos_ts

    df = impute_consumption(df,
        forecast_cols=[col for col in df.columns if col.startswith("forecast_load_")],
        actual_cols=[col for col in df.columns if col.startswith("actual_load_")],
        seasonal_fit_mask=seasonal_fit_mask,
    )
    
    # -------------------------------------------------------------------------
    # FORECASTS - Include directly (no leakage)
    # -------------------------------------------------------------------------
    
    # TSO forecast load
    tso_forecast_col = f"forecast_load_DE_{tso_abbrev}"
    if tso_forecast_col in df.columns:
        new_col_name = f"consumption_forecast_load_de_{tso_abbrev.lower()}"
        result_df[new_col_name] = df[tso_forecast_col]
    else:
        logger.warning(f"TSO forecast column not found: {tso_forecast_col}")
    
    # Neighbor forecast loads
    for neighbor in neighbors:
        # Neighbor codes like DK_1, DK_2, PL, etc.
        neighbor_forecast_col = f"forecast_load_{neighbor}"
        if neighbor_forecast_col in df.columns:
            new_col_name = f"consumption_forecast_load_{neighbor.lower()}"
            result_df[new_col_name] = df[neighbor_forecast_col]
        else:
            logger.debug(f"Neighbor forecast column not found: {neighbor_forecast_col}")
    
    # -------------------------------------------------------------------------
    # ACTUALS - Rolling features and lags only (to prevent leakage)
    # -------------------------------------------------------------------------
    
    # Process TSO actual load
    tso_actual_col = f"actual_load_DE_{tso_abbrev}"
    if tso_actual_col in df.columns:
        # Compute rolling features
        temp_df = df[["begin_date", tso_actual_col]].copy()
        temp_df = compute_lagged_rolling_features(
            temp_df, tso_actual_col, window_days=window_days, minimal=minimal
        )
        
        # Rename rolling columns
        tso_lower = tso_abbrev.lower()
        rolling_mean_col = f"{tso_actual_col}_rollingmean_{window_days}d"
        
        if rolling_mean_col in temp_df.columns:
            result_df[f"consumption_load_actual_rollingmean_{window_days}d_de_{tso_lower}"] = temp_df[rolling_mean_col]
        
        # Include rollingstd and rolling_absdev only if not minimal
        if not minimal:
            rolling_std_col = f"{tso_actual_col}_rollingstd_{window_days}d"
            rolling_absdev_col = f"{tso_actual_col}_rolling_absdev_{window_days}d"
            if rolling_std_col in temp_df.columns:
                result_df[f"consumption_load_actual_rollingstd_{window_days}d_de_{tso_lower}"] = temp_df[rolling_std_col]
            if rolling_absdev_col in temp_df.columns:
                result_df[f"consumption_load_actual_rolling_absdev_{window_days}d_de_{tso_lower}"] = temp_df[rolling_absdev_col]
        
        # Compute lags - only lag_1h by default
        lag_hours = [1] if minimal else [1, 48]
        temp_df = compute_lags(df[["begin_date", tso_actual_col]].copy(), tso_actual_col, lag_hours=lag_hours)
        lag_1_col = f"{tso_actual_col}_lag_1h"
        
        if lag_1_col in temp_df.columns:
            result_df[f"consumption_load_actual_lag_1h_de_{tso_lower}"] = temp_df[lag_1_col]
        
        if not minimal:
            lag_48_col = f"{tso_actual_col}_lag_48h"
            if lag_48_col in temp_df.columns:
                result_df[f"consumption_load_actual_lag_48h_de_{tso_lower}"] = temp_df[lag_48_col]
    else:
        logger.warning(f"TSO actual load column not found: {tso_actual_col}")
    
    # Process neighbor actual loads
    for neighbor in neighbors:
        neighbor_actual_col = f"actual_load_{neighbor}"
        if neighbor_actual_col in df.columns:
            neighbor_lower = neighbor.lower()
            
            # Compute rolling features
            temp_df = df[["begin_date", neighbor_actual_col]].copy()
            temp_df = compute_lagged_rolling_features(
                temp_df, neighbor_actual_col, window_days=window_days, minimal=minimal
            )
            
            # Rename rolling columns
            rolling_mean_col = f"{neighbor_actual_col}_rollingmean_{window_days}d"
            
            if rolling_mean_col in temp_df.columns:
                result_df[f"consumption_load_actual_rollingmean_{window_days}d_{neighbor_lower}"] = temp_df[rolling_mean_col]
            
            # Include rollingstd and rolling_absdev only if not minimal
            if not minimal:
                rolling_std_col = f"{neighbor_actual_col}_rollingstd_{window_days}d"
                rolling_absdev_col = f"{neighbor_actual_col}_rolling_absdev_{window_days}d"
                if rolling_std_col in temp_df.columns:
                    result_df[f"consumption_load_actual_rollingstd_{window_days}d_{neighbor_lower}"] = temp_df[rolling_std_col]
                if rolling_absdev_col in temp_df.columns:
                    result_df[f"consumption_load_actual_rolling_absdev_{window_days}d_{neighbor_lower}"] = temp_df[rolling_absdev_col]
            
            # Compute lags - only lag_1h by default
            lag_hours = [1] if minimal else [1, 48]
            temp_df = compute_lags(df[["begin_date", neighbor_actual_col]].copy(), neighbor_actual_col, lag_hours=lag_hours)
            lag_1_col = f"{neighbor_actual_col}_lag_1h"
            
            if lag_1_col in temp_df.columns:
                result_df[f"consumption_load_actual_lag_1h_{neighbor_lower}"] = temp_df[lag_1_col]
            
            if not minimal:
                lag_48_col = f"{neighbor_actual_col}_lag_48h"
                if lag_48_col in temp_df.columns:
                    result_df[f"consumption_load_actual_lag_48h_{neighbor_lower}"] = temp_df[lag_48_col]
        else:
            logger.debug(f"Neighbor actual load column not found: {neighbor_actual_col}")
    
    logger.info(
        f"Loaded consumption data for {tso_normalized}: "
        f"{len(result_df)} rows, {len(result_df.columns) - 1} feature columns"
    )
    
    return result_df