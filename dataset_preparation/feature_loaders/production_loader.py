"""
Production Loader.

This module provides functions to load production/generation data for a TSO,
including day-ahead forecasts for neighbors and actual generation by fuel type.

Feature Reduction Strategy:
- Aggregate raw fuel types into ~9 groups BEFORE computing rolling/lag features
- Compute only lag_1h + rollingmean_7d for aggregates (not per-fuel)
- Drop rollingstd, rolling_absdev, lag_48h to reduce dimensionality
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from dataset_preparation.feature_loaders.imputation import impute_production

from ..tso_config import get_neighbors, normalize_tso_name

logger = logging.getLogger(__name__)

# Default paths relative to project root
DEFAULT_PRODUCTION_FORECAST_PATH = Path("data/production_new/entsoe_production_full_no_imputation.csv")
DEFAULT_ACTUAL_GENERATION_PATH = Path("data/production_new/entsoe_actual_generation_by_type_extended.csv")

# Mapping from canonical TSO names to abbreviations used in production data
TSO_ABBREV_MAP = {
    "50Hertz": "50HzT",
    "TenneT DE": "TenneT",
    "Amprion": "Amprion",
    "TransnetBW": "TransnetBW",
}

# Raw fuel types in actual generation data (for reference/legacy)
FUEL_TYPES = [
    "Biomass",
    "Fossil_Brown_coal_Lignite",
    "Fossil_Coal-derived_gas",
    "Fossil_Gas",
    "Fossil_Hard_coal",
    "Fossil_Oil",
    "Geothermal",
    "Hydro_Pumped_Storage",
    "Hydro_Run-of-river_and_poundage",
    "Hydro_Water_Reservoir",
    "Nuclear",
    "Other_renewable",
    "Other",
    "Solar",
    "Waste",
    "Wind_Offshore",
    "Wind_Onshore",
]

# Aggregated fuel groups for dimensionality reduction
# Maps group name -> list of raw fuel types to sum
FUEL_GROUPS = {
    "wind_total": ["Wind_Onshore", "Wind_Offshore"],
    "solar": ["Solar"],
    "hydro_ror": ["Hydro_Run-of-river_and_poundage", "Hydro_Water_Reservoir"],
    "pumped_storage": ["Hydro_Pumped_Storage"],
    "thermal_coal": ["Fossil_Hard_coal", "Fossil_Brown_coal_Lignite"],
    "thermal_gas": ["Fossil_Gas", "Fossil_Coal-derived_gas"],
    "thermal_nuclear": ["Nuclear"],
    "thermal_other": ["Fossil_Oil", "Waste", "Other"],
    "bio_other_renewables": ["Biomass", "Geothermal", "Other_renewable"],
}


def _get_tso_abbrev(tso: str) -> str:
    """Get the abbreviation used in production data for a TSO."""
    tso_normalized = normalize_tso_name(tso)
    return TSO_ABBREV_MAP.get(tso_normalized, tso_normalized)


def _normalize_fuel_type_name(fuel_type: str) -> str:
    """
    Normalize fuel type name for column naming.
    
    Converts to lowercase and replaces hyphens/special chars with underscores.
    """
    return fuel_type.lower().replace("-", "_").replace(" ", "_")


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


def _aggregate_fuel_groups(
    df: pd.DataFrame,
    tso_abbrev: str,
) -> pd.DataFrame:
    """
    Aggregate raw fuel types into fuel groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw generation columns per fuel type.
    tso_abbrev : str
        TSO abbreviation for column naming.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with aggregated fuel group columns.
    """
    result = df[["begin_date"]].copy()
    
    for group_name, fuel_types in FUEL_GROUPS.items():
        group_cols = []
        for fuel_type in fuel_types:
            col_name = f"generation_DE_{tso_abbrev}_{fuel_type}"
            if col_name in df.columns:
                group_cols.append(col_name)
        
        if group_cols:
            # Sum the raw fuel types to create aggregate
            result[f"production_{group_name}"] = df[group_cols].sum(axis=1)
        else:
            logger.debug(f"No columns found for fuel group {group_name} for {tso_abbrev}")
    
    return result


def load_production(
    tso: str,
    forecast_path: Path | str | None = None,
    actual_gen_path: Path | str | None = None,
    window_days: int = 7,
    oos_start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load production forecasts (neighbors) and actual generation by fuel groups (TSO).
    
    This function loads:
    1. Day-ahead generation forecasts for neighboring countries
    2. Actual generation aggregated into fuel groups for the specified TSO,
       transformed to leakage-free rolling features and lags
    
    Feature Reduction Strategy:
    - Aggregate raw fuel types into ~9 groups BEFORE computing rolling/lag
    - Compute only lag_1h + rollingmean_7d (not rollingstd, rolling_absdev, lag_48h)
    - Output: ~9 groups × 2 features = ~18 production features per TSO
    
    Parameters
    ----------
    tso : str
        The TSO name (e.g., "50Hertz", "TenneT DE", "Amprion", "TransnetBW").
        Will be normalized internally.
    forecast_path : Path | str | None, optional
        Path to the production forecast CSV. If None, uses the default path.
    actual_gen_path : Path | str | None, optional
        Path to the actual generation by type CSV. If None, uses the default path.
    window_days : int, optional
        Number of days for rolling window. Default is 7.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - begin_date: datetime column (not index)
        - production_forecast_de_{tso}: day-ahead forecast for the TSO
        - production_forecast_{neighbor}: day-ahead forecast for neighbor
        - production_{fuel_group}_de_{tso}_rollingmean_{window_days}d: rolling mean
        - production_{fuel_group}_de_{tso}_lag_1h: 1-hour lag
        
    Notes
    -----
    - Neighbor forecasts are included directly (no leakage issue as they are day-ahead)
    - TSO actual generation is aggregated into fuel groups, then transformed
    - Raw actuals are NOT included to prevent data leakage
    - Fuel groups: wind_total, solar, hydro_ror, pumped_storage, thermal_coal,
      thermal_gas, thermal_nuclear, thermal_other, bio_other_renewables
    """
    # Determine file paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    
    if forecast_path is None:
        forecast_path = project_root / DEFAULT_PRODUCTION_FORECAST_PATH
    else:
        forecast_path = Path(forecast_path)
    
    if actual_gen_path is None:
        actual_gen_path = project_root / DEFAULT_ACTUAL_GENERATION_PATH
    else:
        actual_gen_path = Path(actual_gen_path)
    
    # Normalize TSO name and get info
    tso_normalized = normalize_tso_name(tso)
    tso_abbrev = _get_tso_abbrev(tso)
    tso_lower = tso_abbrev.lower()
    neighbors = get_neighbors(tso_normalized)
    
    result_df = None
    
    # -------------------------------------------------------------------------
    # PART 1: Load neighbor generation forecasts
    # -------------------------------------------------------------------------
    if forecast_path.exists():
        try:
            forecast_df = pd.read_csv(forecast_path, parse_dates=["begin_date"])
            result_df = forecast_df[["begin_date"]].copy()
            forecast_fit_mask = None
            if oos_start_date is not None:
                oos_ts = pd.to_datetime(oos_start_date)
                forecast_fit_mask = forecast_df["begin_date"] < oos_ts

            forecast_df = impute_production(
                forecast_df,
                forecast_cols=[col for col in forecast_df.columns if col.startswith("generation_forecast_")],
                seasonal_fit_mask=forecast_fit_mask,
            )
            
            # Add TSO forecast column (total production forecast)
            tso_forecast_col = f"generation_forecast_DE_{tso_abbrev}"
            if tso_forecast_col in forecast_df.columns:
                result_df[f"production_forecast_de_{tso_lower}"] = forecast_df[tso_forecast_col]
            else:
                logger.warning(f"TSO production forecast column not found: {tso_forecast_col}")

            # Add neighbor forecast columns
            for neighbor in neighbors:
                forecast_col = f"generation_forecast_{neighbor}"
                if forecast_col in forecast_df.columns:
                    new_col_name = f"production_forecast_{neighbor.lower()}"
                    result_df[new_col_name] = forecast_df[forecast_col]
                else:
                    logger.debug(f"Neighbor forecast column not found: {forecast_col}")
                    
        except Exception as e:
            logger.error(f"Error loading production forecasts from {forecast_path}: {e}")
    else:
        logger.warning(f"Production forecast file not found: {forecast_path}")
    
    # -------------------------------------------------------------------------
    # PART 2: Load actual generation by type, aggregate to fuel groups, compute features
    # -------------------------------------------------------------------------
    if actual_gen_path.exists():
        try:
            actual_df = pd.read_csv(actual_gen_path, parse_dates=["begin_date"])
            actual_fit_mask = None
            if oos_start_date is not None:
                oos_ts = pd.to_datetime(oos_start_date)
                actual_fit_mask = actual_df["begin_date"] < oos_ts

            actual_df = impute_production(
                actual_df,
                forecast_cols=[col for col in actual_df.columns if col.startswith("generation_")],
                seasonal_fit_mask=actual_fit_mask,
            )
            
            # Initialize result_df if not already done
            if result_df is None:
                result_df = actual_df[["begin_date"]].copy()
            
            # STEP 1: Aggregate raw fuel types into fuel groups FIRST
            aggregated_df = _aggregate_fuel_groups(actual_df, tso_abbrev)
            
            # STEP 2: Compute rolling/lag features for each fuel group
            for group_name in FUEL_GROUPS.keys():
                raw_col = f"production_{group_name}"
                
                if raw_col not in aggregated_df.columns:
                    continue
                
                base_name = f"production_{group_name}_de_{tso_lower}"
                
                # Compute rolling features (only mean, minimal=True)
                temp_df = aggregated_df[["begin_date", raw_col]].copy()
                temp_df = compute_lagged_rolling_features(
                    temp_df, raw_col, window_days=window_days, minimal=True
                )
                
                # Add rolling mean to result
                rolling_mean_col = f"{raw_col}_rollingmean_{window_days}d"
                if rolling_mean_col in temp_df.columns:
                    result_df[f"{base_name}_rollingmean_{window_days}d"] = temp_df[rolling_mean_col].values
                
                # Compute lags (only 1h)
                temp_df = compute_lags(
                    aggregated_df[["begin_date", raw_col]].copy(), raw_col, lag_hours=[1]
                )
                lag_1_col = f"{raw_col}_lag_1h"
                
                if lag_1_col in temp_df.columns:
                    result_df[f"{base_name}_lag_1h"] = temp_df[lag_1_col].values
                    
        except Exception as e:
            logger.error(f"Error loading actual generation from {actual_gen_path}: {e}")
    else:
        logger.warning(f"Actual generation file not found: {actual_gen_path}")
    
    # If no data was loaded, return empty DataFrame
    if result_df is None:
        logger.warning(f"No production data loaded for {tso_normalized}")
        return pd.DataFrame(columns=["begin_date"])
    
    logger.info(
        f"Loaded production data for {tso_normalized}: "
        f"{len(result_df)} rows, {len(result_df.columns) - 1} feature columns"
    )
    
    return result_df