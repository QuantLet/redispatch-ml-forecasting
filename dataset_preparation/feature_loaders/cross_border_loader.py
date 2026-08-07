"""
Cross-Border Flows Loader.

This module provides functions to load cross-border physical flow data
between DE and neighboring countries/TSOs.

Feature Reduction Strategy:
- Keep only NET flows (drop import/export separately)
- Recompute net flows correctly: net = export - import (positive = export)
- Drop flows with >70% missing data
- Keep only lag_1h + rollingmean_7d (drop rollingstd, rolling_absdev, lag_48h)
- Focus on key borders only
"""

import logging
import re
from pathlib import Path

import pandas as pd
import numpy as np

from dataset_preparation.feature_loaders.imputation import impute_cross_border_flows

from .column_selection import select_relevant_sparse_columns
from ..tso_config import get_neighbors, normalize_tso_name

logger = logging.getLogger(__name__)

# Default path relative to project root
DEFAULT_CROSS_BORDER_PATH = Path("data/cross_border_flows/entsoe_cross_border_flows_extended.csv")

# Mapping from canonical TSO names to abbreviations used in cross-border data
TSO_ABBREV_MAP = {
    "50Hertz": "50HzT",
    "TenneT DE": "TenneT",
    "Amprion": "Amprion",
    "TransnetBW": "TransnetBW",
}

# Mapping for neighbors (country codes to lowercase)
# Cross-border data uses DK_1, DK_2 for Denmark bidding zones
NEIGHBOR_NORMALIZE_MAP = {
    "DK_1": "dk_1",
    "DK_2": "dk_2",
    "DK": "dk",
    "PL": "pl",
    "CZ": "cz",
    "NL": "nl",
    "BE": "be",
    "FR": "fr",
    "LU": "lu",
    "CH": "ch",
    "AT": "at",
}

# Key borders to prioritize (most important for redispatch)
# Limit to ~6-10 borders to reduce dimensionality
KEY_BORDERS = ["PL", "CZ", "DK_1", "DK_2", "NL", "AT", "CH", "FR"]

# Maximum missing data ratio threshold for dropping flows
MAX_MISSING_RATIO = 0.70


def _get_tso_abbrev(tso: str) -> str:
    """Get the abbreviation used in cross-border data for a TSO."""
    tso_normalized = normalize_tso_name(tso)
    return TSO_ABBREV_MAP.get(tso_normalized, tso_normalized)


def _normalize_neighbor(neighbor: str) -> str:
    """Normalize neighbor name to lowercase for column naming."""
    return NEIGHBOR_NORMALIZE_MAP.get(neighbor, neighbor.lower())


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


def _process_flow_column(
    df: pd.DataFrame,
    columns_dict: dict[str, np.ndarray],
    source_col: str,
    base_output_name: str,
    rolling_window_days: int,
) -> None:
    """
    Process a single flow column: compute rolling features and lags.
    
    Feature Reduction: Only compute rollingmean + lag_1h (minimal=True).
    
    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame with begin_date and the flow column.
    columns_dict : dict[str, np.ndarray]
        Dictionary to add computed columns to (modified in place).
    source_col : str
        Name of the source column in df.
    base_output_name : str
        Base name for output columns (e.g., 'cross_border_net_de_to_pl').
    rolling_window_days : int
        Number of days for rolling window.
    """
    if source_col not in df.columns:
        return
    
    # Compute rolling features (minimal=True: only rolling mean)
    temp_df = df[["begin_date", source_col]].copy()
    temp_df = compute_lagged_rolling_features(
        temp_df, source_col, window_days=rolling_window_days, minimal=True
    )
    
    # Rolling column name from compute_lagged_rolling_features
    rolling_mean_col = f"{source_col}_rollingmean_{rolling_window_days}d"
    
    # Add rolling mean to dict
    if rolling_mean_col in temp_df.columns:
        columns_dict[f"{base_output_name}_rollingmean_{rolling_window_days}d"] = np.asarray(
            temp_df[rolling_mean_col].values
        )
    
    # Compute lags (only 1h)
    lag_df = compute_lags(df[["begin_date", source_col]].copy(), source_col, lag_hours=[1])
    lag_1_col = f"{source_col}_lag_1h"
    
    if lag_1_col in lag_df.columns:
        columns_dict[f"{base_output_name}_lag_1h"] = np.asarray(lag_df[lag_1_col].values)


def _compute_net_flow(
    df: pd.DataFrame,
    export_col: str,
    import_col: str,
) -> pd.Series:
    """
    Compute net flow correctly: net = export - import.
    
    Positive net flow = net export from DE to neighbor.
    Negative net flow = net import to DE from neighbor.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing flow columns.
    export_col : str
        Column name for exports (DE -> neighbor).
    import_col : str
        Column name for imports (neighbor -> DE).
        
    Returns
    -------
    pd.Series
        Net flow series (export - import).
    """
    export_vals = df[export_col] if export_col in df.columns else None
    import_vals = df[import_col] if import_col in df.columns else None
    if export_vals is None or import_vals is None:
        logger.warning(f"Missing export or import column for net flow computation: {export_col}, {import_col}")
        return pd.Series(np.nan, index=df.index)
    return export_vals - import_vals


def load_cross_border_flows(
    tso: str,
    rolling_window_days: int = 7,
    cross_border_path: Path | str | None = None,
    force_keep_borders: list[str] | None = None,
    fill_missing_net_with_zero: bool = False,
    activity_threshold: float = 0.3,
    magnitude_threshold: float = 100.0,
    min_nonzero_runs: int = 1,
    oos_start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load cross-border physical flows for TSO (NET FLOWS ONLY).
    
    Feature Reduction Strategy:
    - Keep only NET flows (drop separate import/export)
    - Recompute net flows: net = export - import (positive = net export)
    - Drop flows with >70% missing data
    - Keep only lag_1h + rollingmean_7d (minimal features)
    - Focus on key borders relevant to each TSO
    
    Parameters
    ----------
    tso : str
        The TSO name (e.g., "50Hertz", "TenneT DE", "Amprion", "TransnetBW").
        Will be normalized internally.
    rolling_window_days : int, optional
        Number of days for rolling window. Default is 7.
    cross_border_path : Path | str | None, optional
        Path to the cross-border flows CSV. If None, uses the default path.
    force_keep_borders : list[str] | None, optional
        List of neighbor codes (e.g., ["FR", "NL"]) to keep even if
        missing ratio exceeds MAX_MISSING_RATIO. Default is None.
    fill_missing_net_with_zero : bool, optional
        If True, fill remaining NaNs with 0 for borders in force_keep_borders.
        Default is False.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - begin_date: datetime column (not index)
        - cross_border_net_de_to_{neighbor}_rollingmean_{window}d: rolling mean
        - cross_border_net_de_to_{neighbor}_lag_1h: 1-hour lag
        
    Notes
    -----
    - Only NET flows are included (not separate import/export)
    - Flows with >70% missing are dropped
    - Output: ~6-10 net flows × 2 features = ~12-20 cross-border features
    """
    # Determine file path
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    
    if cross_border_path is None:
        cross_border_path = project_root / DEFAULT_CROSS_BORDER_PATH
    else:
        cross_border_path = Path(cross_border_path)
    
    # Normalize TSO name and get info
    tso_normalized = normalize_tso_name(tso)
    tso_abbrev = _get_tso_abbrev(tso)
    tso_lower = tso_abbrev.lower()
    neighbors = get_neighbors(tso_normalized)

    # Normalize force-keep borders
    if force_keep_borders is None:
        force_keep_borders = []
    force_keep_borders_normalized = {b.upper() for b in force_keep_borders}
    
    if not cross_border_path.exists():
        logger.warning(f"Cross-border flows file not found: {cross_border_path}")
        return pd.DataFrame(columns=["begin_date"])
    
    try:
        df = pd.read_csv(cross_border_path, parse_dates=["begin_date"])
    except Exception as e:
        logger.error(f"Error loading cross-border flows from {cross_border_path}: {e}")
        return pd.DataFrame(columns=["begin_date"])
    
    seasonal_fit_mask = None
    if oos_start_date is not None:
        oos_ts = pd.to_datetime(oos_start_date)
        seasonal_fit_mask = df["begin_date"] < oos_ts

    df = impute_cross_border_flows(
        df,
        flow_cols=[col for col in df.columns if re.match(r"(export|import)_DE_.*_to_.*", col, re.IGNORECASE)],
        seasonal_fit_mask=seasonal_fit_mask,
    )
    
    # Collect all computed columns in a dictionary to avoid DataFrame fragmentation
    columns_dict: dict[str, np.ndarray] = {}
    
    # -------------------------------------------------------------------------
    # Process NET FLOWS ONLY (country-level DE_LU to neighbors)
    # -------------------------------------------------------------------------
    # Get neighbors for filtering (handle DK -> DK_1, DK_2)
    neighbors_for_de_lu = []
    for neighbor in neighbors:
        if neighbor in ["DK_1", "DK_2"]:
            neighbors_for_de_lu.append(neighbor)
        elif neighbor == "DK":
            neighbors_for_de_lu.extend(["DK_1", "DK_2"])
        else:
            neighbors_for_de_lu.append(neighbor)
    
    # Remove duplicates and filter to key borders
    neighbors_for_de_lu = list(set(neighbors_for_de_lu))
    neighbors_for_de_lu = [n for n in neighbors_for_de_lu if n in KEY_BORDERS]

    # --- Phase 1: compute raw net-flow columns ---------------------------
    raw_net_col_map: dict[str, str] = {}   # net_col_name -> base_output_name
    
    for neighbor in neighbors_for_de_lu:
        neighbor_lower = _normalize_neighbor(neighbor)
        
        # Recompute net flow correctly: export - import
        export_col = f"export_DE_LU_to_{neighbor}"
        import_col = f"import_DE_LU_to_{neighbor}"
        
        # Check if we have the necessary columns
        has_export = export_col in df.columns
        has_import = import_col in df.columns
        
        if not has_export and not has_import:
            logger.debug(f"No flow columns found for neighbor {neighbor}")
            continue
        
        # Create a temporary net flow column
        net_col_name = f"_net_de_to_{neighbor_lower}"
        df[net_col_name] = _compute_net_flow(df, export_col, import_col)
        
        # Check missing data ratio
        missing_ratio = df[net_col_name].isna().mean()
        if missing_ratio > MAX_MISSING_RATIO:
            if neighbor in force_keep_borders_normalized:
                logger.info(
                    f"Keeping net flow DE->{neighbor} despite {missing_ratio:.1%} missing "
                    f"(force_keep_borders override)"
                )
                if fill_missing_net_with_zero:
                    df[net_col_name] = df[net_col_name].fillna(0.0)
            else:
                logger.info(
                    f"Dropping net flow DE->{neighbor}: {missing_ratio:.1%} missing "
                    f"(>{MAX_MISSING_RATIO:.0%} threshold)"
                )
                continue
        
        raw_net_col_map[net_col_name] = f"cross_border_net_de_to_{neighbor_lower}"

    # # -------------------------------------------------------------------------
    # # PART 2: TSO-level flows (specific TSO to/from destinations)
    # # -------------------------------------------------------------------------
    # # Find all columns that match this TSO
    # for col in df.columns[df.columns.str.contains(f"DE_{tso_abbrev}", case=False)]:
    #     tso_col_label_re = re.search(f"DE_(.*)_to_(.*)", col, re.IGNORECASE)
    #     if not tso_col_label_re:
    #         continue
    #     tso_col_label = tso_col_label_re.group(1).lower()
    #     neighbor = tso_col_label_re.group(2).lower()
    #     if not tso_col_label or tso_col_label != tso_abbrev.lower() or not neighbor:
    #         continue

    #     # Recompute net flow correctly: export - import
    #     export_col = f"export_DE_{tso_col_label}_to_{neighbor}"
    #     import_col = f"import_DE_{tso_col_label}_to_{neighbor}"
        
    #     # Check if we have the necessary columns
    #     has_export = export_col in df.columns
    #     has_import = import_col in df.columns
        
    #     if not has_export and not has_import:
    #         logger.debug(f"No flow columns found for TSO {tso_abbrev} neighbor {tso_col_label}")
    #         continue

    #     # Create a temporary net flow column
    #     net_col_name = f"_net_de_{tso_col_label}_to_{neighbor}"
    #     df[net_col_name] = _compute_net_flow(df, export_col, import_col)

    #     # Check missing data ratio
    #     missing_ratio = df[net_col_name].isna().mean()
    #     if missing_ratio > MAX_MISSING_RATIO:
    #         if neighbor in force_keep_borders_normalized:
    #             logger.info(
    #                 f"Keeping net flow {tso_col_label}->{neighbor} despite {missing_ratio:.1%} missing "
    #                 f"(force_keep_borders override)"
    #             )
    #             if fill_missing_net_with_zero:
    #                 df[net_col_name] = df[net_col_name].fillna(0.0)
    #         else:
    #             logger.info(
    #                 f"Dropping net flow {tso_col_label}->{neighbor}: {missing_ratio:.1%} missing "
    #                 f"(>{MAX_MISSING_RATIO:.0%} threshold)"
    #             )
    #             continue

    #     raw_net_col_map[net_col_name] = f"cross_border_net_de_{tso_col_label}_to_{neighbor}"

    # --- Phase 2: sparse column selection on raw net flows ----------------
    raw_net_cols = list(raw_net_col_map.keys())
    kept_raw = select_relevant_sparse_columns(
        df,
        raw_net_cols,
        activity_threshold=activity_threshold,
        magnitude_threshold=magnitude_threshold,
        min_nonzero_runs=min_nonzero_runs,
    )

    # --- Phase 3: compute lagged/rolling features only for kept borders ---
    for net_col_name in kept_raw:
        base_name = raw_net_col_map[net_col_name]
        _process_flow_column(
            df, columns_dict, net_col_name, base_name, rolling_window_days
        )

    
    # Build result DataFrame efficiently using pd.concat
    result_df = df[["begin_date"]].copy()
    if columns_dict:
        feature_df = pd.DataFrame(columns_dict, index=result_df.index)
        result_df = pd.concat([result_df, feature_df], axis=1)

    logger.info(
        f"Loaded cross-border flows for {tso_normalized}: "
        f"{len(result_df)} rows, {len(result_df.columns) - 1} feature columns"
    )
    
    return result_df