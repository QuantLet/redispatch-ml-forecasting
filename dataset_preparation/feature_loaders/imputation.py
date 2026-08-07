"""
Imputation Module.

This module provides functions for handling missing data across all feature types
in the redispatch forecasting pipeline. Different imputation strategies are applied
based on the characteristics of each feature type.

Imputation Strategies:
- Forward-fill: For short gaps where autocorrelation is high
- Seasonal median: For longer gaps using hour-of-day, weekday, or month patterns
- Zero-fill: For redispatch features where missing = no activity

Notes
-----
Imputation should be applied BEFORE computing rolling features to prevent
NaN propagation through rolling windows.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# Core Imputation Functions
# =============================================================================


def detect_gap_lengths(series: pd.Series) -> pd.Series:
    """
    Detect the length of consecutive NaN gaps in a series.
    
    Parameters
    ----------
    series : pd.Series
        Series with potential NaN values.
        
    Returns
    -------
    pd.Series
        Series where each NaN position contains its gap length (position within gap).
        Non-NaN positions contain 0.
        
    Examples
    --------
    >>> s = pd.Series([1, np.nan, np.nan, 2, np.nan])
    >>> detect_gap_lengths(s)
    0    0
    1    1
    2    2
    3    0
    4    1
    dtype: int64
    """
    is_missing = series.isna()
    # Group consecutive NaN sequences by counting non-NaN transitions
    gap_groups = (~is_missing).cumsum()
    # Within each gap group, cumsum gives position within gap
    gap_lengths = is_missing.groupby(gap_groups).cumsum()
    return gap_lengths.astype(int)


def _normalize_fit_mask(
    df: pd.DataFrame,
    fit_mask: pd.Series | np.ndarray | list[bool] | None,
) -> pd.Series:
    """Return a boolean mask aligned to df.index for fitting seasonal statistics."""
    if fit_mask is None:
        return pd.Series(True, index=df.index)

    if isinstance(fit_mask, pd.Series):
        return fit_mask.reindex(df.index, fill_value=False).astype(bool)

    return pd.Series(fit_mask, index=df.index).fillna(False).astype(bool)


def compute_seasonal_median(
    df: pd.DataFrame,
    value_col: str,
    grouping_cols: list[str],
    date_col: str = "begin_date",
    fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
) -> pd.Series:
    """
    Compute seasonal median for a value column grouped by time components.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the value column and date column.
    value_col : str
        Name of the column to compute seasonal medians for.
    grouping_cols : list[str]
        Time components to group by. Options: 'hour', 'weekday', 'month', 'dayofyear'.
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
        
    Returns
    -------
    pd.Series
        Series aligned with df index containing seasonal median values.
        Can be used directly to fill NaN values.
        
    Notes
    -----
    This function extracts time components from the date column and computes
    the median for each combination of grouping variables.
    """
    df_work = df.copy()
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df_work[date_col]):
        df_work[date_col] = pd.to_datetime(df_work[date_col])
    
    # Extract time components as needed
    time_component_map = {
        "hour": df_work[date_col].dt.hour,
        "weekday": df_work[date_col].dt.weekday,
        "month": df_work[date_col].dt.month,
        "dayofyear": df_work[date_col].dt.dayofyear,
    }
    
    # Create grouping columns
    group_keys = []
    for col_name in grouping_cols:
        if col_name in time_component_map:
            key_name = f"_grp_{col_name}"
            df_work[key_name] = time_component_map[col_name]
            group_keys.append(key_name)
        elif col_name in df_work.columns:
            # Use existing column (e.g., zone, TSO)
            group_keys.append(col_name)
        else:
            logger.warning(f"Unknown grouping column: {col_name}")
    
    if not group_keys:
        logger.warning("No valid grouping columns, returning NaN")
        return pd.Series(np.nan, index=df.index)
    
    seasonal_median, _ = compute_seasonal_median_with_count(
        df=df_work,
        value_col=value_col,
        grouping_cols=grouping_cols,
        date_col=date_col,
        fit_mask=fit_mask,
    )
    return seasonal_median


def compute_seasonal_median_with_count(
    df: pd.DataFrame,
    value_col: str,
    grouping_cols: list[str],
    date_col: str = "begin_date",
    fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute seasonal median AND per-group count for a value column.
    
    This extended version returns both the median and the number of non-NaN
    samples in each group, enabling reliability-aware imputation.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the value column and date column.
    value_col : str
        Name of the column to compute seasonal medians for.
    grouping_cols : list[str]
        Time components to group by. Options: 'hour', 'weekday', 'month', 'dayofyear'.
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
        
    Returns
    -------
    tuple[pd.Series, pd.Series]
        - seasonal_median: Series aligned with df index containing seasonal median values.
        - seasonal_count: Series aligned with df index containing count of non-NaN values
          in each group.
    """
    df_work = df.copy()
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df_work[date_col]):
        df_work[date_col] = pd.to_datetime(df_work[date_col])
    
    # Extract time components as needed
    time_component_map = {
        "hour": df_work[date_col].dt.hour,
        "weekday": df_work[date_col].dt.weekday,
        "month": df_work[date_col].dt.month,
        "dayofyear": df_work[date_col].dt.dayofyear,
    }
    
    # Create grouping columns
    group_keys = []
    for col_name in grouping_cols:
        if col_name in time_component_map:
            key_name = f"_grp_{col_name}"
            df_work[key_name] = time_component_map[col_name]
            group_keys.append(key_name)
        elif col_name in df_work.columns:
            group_keys.append(col_name)
        else:
            logger.warning(f"Unknown grouping column: {col_name}")
    
    if not group_keys:
        logger.warning("No valid grouping columns, returning NaN")
        return pd.Series(np.nan, index=df.index), pd.Series(0, index=df.index)
    
    fit_mask_aligned = _normalize_fit_mask(df_work, fit_mask)
    if not fit_mask_aligned.any():
        logger.warning("Fit mask has no True values, returning NaN/0 for seasonal stats")
        return pd.Series(np.nan, index=df.index), pd.Series(0, index=df.index)

    fit_df = df_work.loc[fit_mask_aligned, group_keys + [value_col]]
    grouped = fit_df.groupby(group_keys, dropna=False)[value_col]
    medians = grouped.median()
    counts = grouped.count()

    if len(group_keys) == 1:
        key_series = df_work[group_keys[0]]
        seasonal_median = key_series.map(medians)
        seasonal_count = key_series.map(counts).fillna(0)
    else:
        row_keys = pd.MultiIndex.from_frame(df_work[group_keys])
        seasonal_median = pd.Series(medians.reindex(row_keys).to_numpy(), index=df.index)
        seasonal_count = pd.Series(
            counts.reindex(row_keys).fillna(0).to_numpy(),
            index=df.index,
        )

    return seasonal_median, seasonal_count.astype(int)


def compute_tiered_seasonal_fill(
    df: pd.DataFrame,
    value_col: str,
    primary_groups: list[str],
    fallback_groups: list[list[str]] | None = None,
    min_count: int | None = None,
    shrinkage_k: float | None = None,
    shrinkage_target: Literal["next_tier", "global"] = "next_tier",
    missing_fallback: Literal["nan", "global_median"] = "nan",
    date_col: str = "begin_date",
    fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
) -> pd.Series:
    """
    Compute seasonal fill values using tiered fallback and optional shrinkage.
    
    This function implements a cascade of seasonal groupings: if the primary
    grouping has insufficient samples (< min_count) or NaN median, it falls
    back to progressively coarser groupings. Optionally applies shrinkage
    to blend low-count group medians toward coarser estimates.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the value column and date column.
    value_col : str
        Name of the column to compute seasonal fill for.
    primary_groups : list[str]
        Primary grouping columns (e.g., ['hour', 'month']).
    fallback_groups : list[list[str]] | None, optional
        List of fallback groupings in order of preference.
        E.g., [['hour', 'weekday'], ['hour']] means try hour×weekday first,
        then hour-only. Default is None (no fallback).
    min_count : int | None, optional
        Minimum samples required in a group to trust its median.
        Groups with fewer samples trigger fallback or shrinkage.
        Default is None (accept any count ≥ 1).
    shrinkage_k : float | None, optional
        Shrinkage strength parameter. If set, low-count group medians are
        blended toward the next tier's median using weight w = n/(n+k).
        Default is None (no shrinkage, hard fallback).
    missing_fallback : {'nan', 'global_median'}, optional
        What to return if all tiers fail. Default is 'nan'.
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
        
    Returns
    -------
    pd.Series
        Series aligned with df index containing tiered seasonal fill values.
        
    Notes
    -----
    The algorithm proceeds as follows for each row:
    1. Compute median and count for primary_groups
    2. If count >= min_count (or min_count is None) and median is not NaN, use it
    3. Otherwise, try each fallback tier in order
    4. If shrinkage_k is set, blend the best available tier toward the next coarser tier
    5. If all tiers fail, use missing_fallback strategy
    """
    # Build list of all tiers: primary + fallbacks
    all_tiers = [primary_groups]
    if fallback_groups:
        all_tiers.extend(fallback_groups)
    
    fit_mask_aligned = _normalize_fit_mask(df, fit_mask)

    # Compute median and count for each tier
    tier_medians = []
    tier_counts = []
    for groups in all_tiers:
        median, count = compute_seasonal_median_with_count(
            df,
            value_col,
            groups,
            date_col,
            fit_mask=fit_mask_aligned,
        )
        tier_medians.append(median)
        tier_counts.append(count)
    
    # Compute global median as ultimate fallback
    if fit_mask_aligned.any():
        global_med = df.loc[fit_mask_aligned, value_col].median()
    else:
        global_med = df[value_col].median()
    
    # Initialize result with NaN
    result = pd.Series(np.nan, index=df.index)
    
    # Effective min_count (treat None as 1)
    effective_min_count = min_count if min_count is not None else 1
    
    # Process each row: find the best tier and optionally apply shrinkage
    for idx in df.index:
        # Find first tier meeting the count threshold
        best_tier_idx = None
        for i, (median_s, count_s) in enumerate(zip(tier_medians, tier_counts)):
            med_val = median_s.loc[idx]
            cnt_val = count_s.loc[idx]
            if pd.notna(med_val) and cnt_val >= effective_min_count:
                best_tier_idx = i
                break
        
        if best_tier_idx is None:
            # No tier met the threshold; try to find any tier with data
            for i, (median_s, count_s) in enumerate(zip(tier_medians, tier_counts)):
                med_val = median_s.loc[idx]
                cnt_val = count_s.loc[idx]
                if pd.notna(med_val) and cnt_val > 0:
                    best_tier_idx = i
                    break
        
        if best_tier_idx is not None:
            med_val = tier_medians[best_tier_idx].loc[idx]
            cnt_val = tier_counts[best_tier_idx].loc[idx]
            
            # Apply shrinkage if enabled and there's a coarser tier
            if shrinkage_k is not None and shrinkage_k > 0:
                if shrinkage_target == "global":
                    parent_med = global_med
                else:
                    # Find the next coarser tier with valid data
                    parent_med = global_med  # Ultimate parent
                    for j in range(best_tier_idx + 1, len(tier_medians)):
                        parent_val = tier_medians[j].loc[idx]
                        if pd.notna(parent_val):
                            parent_med = parent_val
                            break
                
                if pd.notna(parent_med):
                    # Shrinkage weight: w = n / (n + k)
                    w = cnt_val / (cnt_val + shrinkage_k)
                    result.loc[idx] = w * med_val + (1 - w) * parent_med
                else:
                    result.loc[idx] = med_val
            else:
                result.loc[idx] = med_val
        else:
            # All tiers failed
            if missing_fallback == "global_median" and pd.notna(global_med):
                result.loc[idx] = global_med
            # else: leave as NaN
    
    return result


def compute_level_adjusted_fill(
    df: pd.DataFrame,
    value_col: str,
    seasonal_fill: pd.Series,
    max_residual_gap: int | None = None,
) -> pd.Series:
    """
    Compute level-adjusted fill: seasonal + carried residual.
    
    This preserves local regime shifts by carrying forward the residual
    (observed - seasonal) into gaps, rather than filling with pure seasonal.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the value column.
    value_col : str
        Name of the column to compute fill for.
    seasonal_fill : pd.Series
        Pre-computed seasonal fill values (from tiered or simple seasonal).
    max_residual_gap : int | None, optional
        Maximum gap length to carry residual. Beyond this, decay to pure seasonal.
        Default is None (carry indefinitely).
        
    Returns
    -------
    pd.Series
        Series aligned with df index containing level-adjusted fill values.
        
    Notes
    -----
    For observed values: residual = observed - seasonal
    For missing values: fill = seasonal + ffill(residual)
    
    This is particularly useful for PV actuals where weather causes sustained
    deviations from the typical seasonal pattern.
    """
    observed = df[value_col].copy()
    
    # Compute residuals where we have observations
    residual = observed - seasonal_fill
    
    # Strictly causal residual carry: never use backward fill.
    residual_filled = residual.ffill().fillna(0.0)
    
    # If max_residual_gap is set, decay residual to 0 for very long gaps
    if max_residual_gap is not None:
        is_missing = observed.isna()
        gap_groups = (~is_missing).cumsum()
        gap_position = is_missing.groupby(gap_groups).cumsum()
        
        # Decay factor: 1.0 at gap start, approaching 0 as gap_position approaches max
        # Using linear decay for simplicity
        decay = (1 - gap_position / max_residual_gap).clip(lower=0)
        residual_filled = residual_filled * decay.where(is_missing, 1.0)
    
    # Level-adjusted fill = seasonal + carried residual
    result = seasonal_fill + residual_filled
    
    return result


def impute_with_gap_detection(
    df: pd.DataFrame,
    value_col: str,
    forward_fill_threshold_hours: int,
    seasonal_groups: list[str],
    date_col: str = "begin_date",
    fallback_groups: list[list[str]] | None = None,
    min_count: int | None = None,
    missing_fallback: Literal["nan", "global_median"] = "nan",
    seasonal_fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
) -> pd.Series:
    """
    Impute missing values: forward-fill short gaps, seasonal median for long gaps.
    
    This is the core imputation function used by feature-specific functions.
    It first attempts forward-fill for gaps shorter than the threshold,
    then uses seasonal median for remaining (longer) gaps.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the value column and date column.
    value_col : str
        Name of the column to impute.
    forward_fill_threshold_hours : int
        Maximum gap length (in hours/rows) for forward-fill.
        Gaps longer than this use seasonal median.
    seasonal_groups : list[str]
        Grouping columns for seasonal median. Options: 'hour', 'weekday', 'month'.
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
        
    Returns
    -------
    pd.Series
        Imputed series with same index as input DataFrame.
        
    Notes
    -----
    The function applies strategies in order:
    1. Forward-fill for gaps <= threshold
    2. Seasonal median for remaining NaN values
    """
    result = df[value_col].copy()
    
    # For short gaps: use forward-fill
    # We need to identify which NaN values are in "short" gaps
    # A gap is short if its total length <= threshold
    
    # To get total gap length (not position), we need a different approach
    is_missing = result.isna()
    
    if not is_missing.any():
        return result  # No missing values
    
    # Group consecutive NaNs and find gap lengths
    gap_groups = (~is_missing).cumsum()
    
    # Compute total gap length for each gap
    gap_total_lengths = is_missing.groupby(gap_groups).transform("sum")
    
    # Forward-fill creates values for all NaN positions
    ffilled = result.ffill()
    
    # Apply forward-fill only to short gaps
    short_gap_mask = is_missing & (gap_total_lengths <= forward_fill_threshold_hours)
    result[short_gap_mask] = ffilled[short_gap_mask]
    
    # For remaining NaN values (long gaps), use seasonal median
    still_missing = result.isna()
    if still_missing.any():
        seasonal_fill = compute_tiered_seasonal_fill(
            df=df,
            value_col=value_col,
            primary_groups=seasonal_groups,
            fallback_groups=fallback_groups,
            min_count=min_count,
            shrinkage_k=None,
            missing_fallback=missing_fallback,
            date_col=date_col,
            fit_mask=seasonal_fit_mask,
        )
        result[still_missing] = seasonal_fill[still_missing]
    
    # Log imputation statistics
    n_ffilled = short_gap_mask.sum()
    n_seasonal = still_missing.sum()
    n_total = is_missing.sum()
    if n_total > 0:
        logger.debug(
            f"Imputed {value_col}: {n_total} missing values "
            f"({n_ffilled} forward-fill, {n_seasonal} seasonal median)"
        )
    
    return result


def impute_with_gap_detection_sparse(
    df: pd.DataFrame,
    value_col: str,
    forward_fill_threshold_hours: int,
    seasonal_groups: list[str],
    date_col: str = "begin_date",
    fallback_groups: list[list[str]] | None = None,
    min_count: int | None = None,
    shrinkage_k: float | None = None,
    missing_fallback: Literal["nan", "global_median"] = "nan",
    level_adjustment: bool = False,
    max_residual_gap: int | None = None,
    seasonal_fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
    shrinkage_target: Literal["next_tier", "global"] = "next_tier",
) -> pd.Series:
    """
    Impute missing values with sparsity-aware tiered seasonal fallback.
    
    This is an enhanced version of impute_with_gap_detection that handles
    extremely sparse data (>40% missing) where seasonal medians may be
    unreliable or undefined for certain groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the value column and date column.
    value_col : str
        Name of the column to impute.
    forward_fill_threshold_hours : int
        Maximum gap length (in hours/rows) for forward-fill.
    seasonal_groups : list[str]
        Primary grouping columns for seasonal median (e.g., ['hour', 'month']).
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
    fallback_groups : list[list[str]] | None, optional
        List of fallback groupings in order of preference.
        E.g., [['hour', 'weekday'], ['hour']] tries hour×weekday, then hour-only.
        Default is None (no fallback, same as original behavior).
    min_count : int | None, optional
        Minimum samples required in a seasonal group to trust its median.
        Groups with fewer samples trigger fallback or shrinkage.
        Default is None (accept any count ≥ 1, original behavior).
    shrinkage_k : float | None, optional
        Shrinkage strength parameter. If set, low-count group medians are
        blended toward the next tier using weight w = n/(n+k).
        Default is None (no shrinkage, hard fallback).
    missing_fallback : {'nan', 'global_median'}, optional
        What to return if all seasonal tiers fail. Default is 'nan'.
    level_adjustment : bool, optional
        If True, fill long gaps with seasonal + carried_residual instead
        of pure seasonal. Useful for PV actuals where weather causes
        sustained deviations. Default is False.
    max_residual_gap : int | None, optional
        Maximum gap length to carry residual (only used if level_adjustment=True).
        Beyond this, residual decays to zero. Default is None (no decay).
        
    Returns
    -------
    pd.Series
        Imputed series with same index as input DataFrame.
        
    Notes
    -----
    The function applies strategies in order:
    1. Forward-fill for gaps <= threshold
    2. Tiered seasonal median with optional shrinkage for long gaps
    3. Optional level adjustment to preserve local regime
    """
    result = df[value_col].copy()
    is_missing = result.isna()
    
    if not is_missing.any():
        return result  # No missing values
    
    # Group consecutive NaNs and find gap lengths
    gap_groups = (~is_missing).cumsum()
    gap_total_lengths = is_missing.groupby(gap_groups).transform("sum")
    
    # Forward-fill for short gaps
    ffilled = result.ffill()
    short_gap_mask = is_missing & (gap_total_lengths <= forward_fill_threshold_hours)
    result[short_gap_mask] = ffilled[short_gap_mask]
    
    # For remaining NaN values (long gaps), use tiered seasonal fill
    still_missing = result.isna()
    if still_missing.any():
        # Compute tiered seasonal fill
        seasonal_fill = compute_tiered_seasonal_fill(
            df=df,
            value_col=value_col,
            primary_groups=seasonal_groups,
            fallback_groups=fallback_groups,
            min_count=min_count,
            shrinkage_k=shrinkage_k,
            shrinkage_target=shrinkage_target,
            missing_fallback=missing_fallback,
            date_col=date_col,
            fit_mask=seasonal_fit_mask,
        )
        
        # Optionally apply level adjustment
        if level_adjustment:
            seasonal_fill = compute_level_adjusted_fill(
                df=df,
                value_col=value_col,
                seasonal_fill=seasonal_fill,
                max_residual_gap=max_residual_gap,
            )
        
        result[still_missing] = seasonal_fill[still_missing]
    
    # Log imputation statistics
    n_ffilled = short_gap_mask.sum()
    n_seasonal = still_missing.sum()
    n_total = is_missing.sum()
    if n_total > 0:
        logger.debug(
            f"Imputed (sparse) {value_col}: {n_total} missing values "
            f"({n_ffilled} forward-fill, {n_seasonal} tiered seasonal)"
        )
    
    return result


# =============================================================================
# Generic Imputation Functions
# =============================================================================


def impute_forward_fill_seasonal_sparse(
    df: pd.DataFrame,
    columns: list[str],
    gap_threshold_hours: int,
    seasonal_groups: list[str],
    date_col: str = "begin_date",
    fallback_groups: list[list[str]] | None = None,
    min_count: int | None = None,
    shrinkage_k: float | None = None,
    missing_fallback: Literal["nan", "global_median"] = "nan",
    level_adjustment: bool = False,
    max_residual_gap: int | None = None,
    seasonal_fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
    shrinkage_target: Literal["next_tier", "global"] = "next_tier",
) -> pd.DataFrame:
    """
    Impute columns with sparsity-aware tiered seasonal fallback.
    
    This is an enhanced version of impute_forward_fill_seasonal that handles
    extremely sparse data where certain seasonal groups may have zero or
    very few samples.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to impute.
    columns : list[str]
        List of column names to impute.
    gap_threshold_hours : int
        Maximum gap length for forward-fill.
    seasonal_groups : list[str]
        Primary grouping columns for seasonal median (e.g., ['hour', 'month']).
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
    fallback_groups : list[list[str]] | None, optional
        List of fallback groupings (e.g., [['hour', 'weekday'], ['hour']]).
        Default is None (no fallback).
    min_count : int | None, optional
        Minimum samples required in a group to trust its median.
        Default is None (original behavior).
    shrinkage_k : float | None, optional
        Shrinkage strength (w = n/(n+k)). Default is None (no shrinkage).
    missing_fallback : {'nan', 'global_median'}, optional
        Fallback if all tiers fail. Default is 'nan'.
    level_adjustment : bool, optional
        If True, use seasonal + carried_residual for long gaps. Default is False.
    max_residual_gap : int | None, optional
        Max gap for residual carry (only if level_adjustment=True). Default is None.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with imputed columns.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = impute_with_gap_detection_sparse(
                df=df,
                value_col=col,
                forward_fill_threshold_hours=gap_threshold_hours,
                seasonal_groups=seasonal_groups,
                date_col=date_col,
                fallback_groups=fallback_groups,
                min_count=min_count,
                shrinkage_k=shrinkage_k,
                missing_fallback=missing_fallback,
                level_adjustment=level_adjustment,
                max_residual_gap=max_residual_gap,
                seasonal_fit_mask=seasonal_fit_mask,
                shrinkage_target=shrinkage_target,
            )
        else:
            logger.debug(f"Column not found for imputation: {col}")
    return df


def impute_forward_fill_seasonal(
    df: pd.DataFrame,
    columns: list[str],
    gap_threshold_hours: int,
    seasonal_groups: list[str],
    date_col: str = "begin_date",
    fallback_groups: list[list[str]] | None = None,
    min_count: int | None = None,
    missing_fallback: Literal["nan", "global_median"] = "nan",
    seasonal_fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
) -> pd.DataFrame:
    """
    Impute columns using forward-fill for short gaps, seasonal median for long gaps.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to impute.
    columns : list[str]
        List of column names to impute.
    gap_threshold_hours : int
        Maximum gap length for forward-fill.
    seasonal_groups : list[str]
        Grouping columns for seasonal median (e.g., ['hour', 'weekday']).
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with imputed columns.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = impute_with_gap_detection(
                df=df,
                value_col=col,
                forward_fill_threshold_hours=gap_threshold_hours,
                seasonal_groups=seasonal_groups,
                date_col=date_col,
                fallback_groups=fallback_groups,
                min_count=min_count,
                missing_fallback=missing_fallback,
                seasonal_fit_mask=seasonal_fit_mask,
            )
        else:
            logger.debug(f"Column not found for imputation: {col}")
    return df


def impute_forward_fill_only(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Impute columns using forward-fill only (no seasonal fallback).
    
    Use this for data where further imputation would create artificial patterns
    (e.g., Bloomberg financial data).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to impute.
    columns : list[str]
        List of column names to impute.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with forward-filled columns.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            n_missing_before = df[col].isna().sum()
            df[col] = df[col].ffill()
            n_missing_after = df[col].isna().sum()
            if n_missing_before > 0:
                logger.debug(
                    f"Forward-fill {col}: {n_missing_before} -> {n_missing_after} missing"
                )
    return df


def impute_fill_zero(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Impute columns by filling NaN with 0.
    
    Use for redispatch features where missing = no activity recorded.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to impute.
    columns : list[str]
        List of column names to impute.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with zero-filled columns.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(0)
            if n_missing > 0:
                logger.debug(f"Zero-fill {col}: {n_missing} values imputed")
    return df


# =============================================================================
# Feature-Specific Imputation Functions
# =============================================================================


def impute_day_ahead_prices(
    df: pd.DataFrame,
    price_cols: list[str] | None = None,
    gap_threshold_hours: int = 6,
    date_col: str = "begin_date",
    seasonal_fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
) -> pd.DataFrame:
    """
    Impute missing day-ahead electricity prices.
    
    Strategy:
    1. Forward-fill for short gaps (< threshold)
    2. Seasonal median (hour × month OR hour × weekday) for long gaps
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price columns and date column.
    price_cols : list[str] | None, optional
        List of price column names to impute. If None, auto-detect columns
        starting with 'day_ahead_price_'.
    gap_threshold_hours : int, optional
        Maximum gap for forward-fill. Default is 48 hours.
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with imputed price columns.
        
    Notes
    -----
    Prices are highly autocorrelated short-term, so forward-fill works well
    for short gaps. Long gaps use seasonal patterns (hour + month gives
    good seasonal price patterns).
    """
    df = df.copy()
    
    if price_cols is None:
        price_cols = [
            c
            for c in df.columns
            if c.startswith("day_ahead_price_") or c.startswith("price_")
        ]
    
    # DA prices are better captured by weekly structure than month-only seasonality.
    seasonal_groups = ["hour", "weekday"]
    
    return impute_forward_fill_seasonal(
        df=df,
        columns=price_cols,
        gap_threshold_hours=gap_threshold_hours,
        seasonal_groups=seasonal_groups,
        date_col=date_col,
        fallback_groups=[["hour"]],
        missing_fallback="global_median",
        seasonal_fit_mask=seasonal_fit_mask,
    )


def impute_production_actuals_by_fuel(
    df: pd.DataFrame,
    actual_cols: list[str] | None = None,
    date_col: str = "begin_date",
    seasonal_fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
    uniform_gap_threshold_hours: int | None = None,
) -> pd.DataFrame:
    """
    Impute production actuals with fuel-type-specific sparsity settings.
    
    Uses ACF/PACF patterns and generation profiles to optimize each fuel type.
    """
    df = df.copy()
    
    if actual_cols is None:
        actual_cols = [
            c for c in df.columns 
            if (c.startswith("generation_") or c.startswith("production_"))
            and "forecast" not in c.lower()
        ]
    
    # Category 1: High persistence baseload fuels.
    high_persistence_cols = [
        c for c in actual_cols
        if any(fuel in c.lower() for fuel in ["nuclear", "lignite", "brown_coal"])
    ]

    # Category 2: Medium persistence thermal fuels.
    medium_persistence_cols = [
        c for c in actual_cols
        if c not in high_persistence_cols
        and any(fuel in c.lower() for fuel in ["coal", "hard_coal", "gas"])
    ]

    # Category 3: Lower persistence / flexible fuels.
    low_persistence_cols = [
        c for c in actual_cols
        if c not in high_persistence_cols + medium_persistence_cols
        and any(
            fuel in c.lower()
            for fuel in [
                "hydro",
                "water_reservoir",
                "run-of-river",
                "pumped_storage",
                "oil",
                "geothermal",
            ]
        )
    ]
    
    # Remaining columns (e.g., Other, Other_renewable, Waste)
    other_cols = [
        c for c in actual_cols 
        if c not in high_persistence_cols + medium_persistence_cols + low_persistence_cols
    ]

    seasonal_groups = ["hour", "weekday"]
    fallback_groups = [["hour"]]

    if uniform_gap_threshold_hours is not None:
        high_ff = medium_ff = low_ff = other_ff = uniform_gap_threshold_hours
    else:
        high_ff = 12
        medium_ff = 6
        low_ff = 3
        other_ff = 6
    
    # Impute Category 1: High Persistence
    if high_persistence_cols:
        df = impute_forward_fill_seasonal_sparse(
            df, high_persistence_cols,
            gap_threshold_hours=high_ff,
            seasonal_groups=seasonal_groups,
            date_col=date_col,
            fallback_groups=fallback_groups,
            min_count=12,
            shrinkage_k=None,
            missing_fallback="global_median",
            level_adjustment=False,
            seasonal_fit_mask=seasonal_fit_mask,
        )
    
    # Impute Category 2: Moderate Persistence (thermal)
    if medium_persistence_cols:
        df = impute_forward_fill_seasonal_sparse(
            df, medium_persistence_cols,
            gap_threshold_hours=medium_ff,
            seasonal_groups=seasonal_groups,
            date_col=date_col,
            fallback_groups=fallback_groups,
            min_count=12,
            shrinkage_k=None,
            missing_fallback="global_median",
            level_adjustment=False,
            seasonal_fit_mask=seasonal_fit_mask,
        )
    
    # Impute Category 3: Low Persistence/Flexible
    if low_persistence_cols:
        df = impute_forward_fill_seasonal_sparse(
            df, low_persistence_cols,
            gap_threshold_hours=low_ff,
            seasonal_groups=seasonal_groups,
            date_col=date_col,
            fallback_groups=fallback_groups,
            min_count=8,
            shrinkage_k=None,
            missing_fallback="global_median",
            level_adjustment=False,
            seasonal_fit_mask=seasonal_fit_mask,
        )
    
    # Impute remaining with standard settings
    if other_cols:
        df = impute_forward_fill_seasonal(
            df, other_cols,
            gap_threshold_hours=other_ff,
            seasonal_groups=seasonal_groups,
            date_col=date_col,
            fallback_groups=fallback_groups,
            min_count=8,
            missing_fallback="global_median",
            seasonal_fit_mask=seasonal_fit_mask,
        )
    
    return df


def impute_production(
    df: pd.DataFrame,
    forecast_cols: list[str] | None = None,
    actual_cols: list[str] | None = None,
    gap_threshold_hours: int | None = None,
    date_col: str = "begin_date",
    seasonal_fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
) -> pd.DataFrame:
    """
    Impute missing production/generation data.
    
    Strategy:
    - Forecasts: Forward-fill (same horizon values persist)
    - Actuals: Forward-fill short gaps, seasonal median for long gaps
              Grouping: hour × weekday
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with production columns and date column.
    forecast_cols : list[str] | None, optional
        Forecast column names. If None, auto-detect 'production_forecast_*'.
    actual_cols : list[str] | None, optional
        Actual generation column names. If None, auto-detect 'production_*' 
        columns that contain rolling/lag features.
    gap_threshold_hours : int, optional
        Maximum gap for forward-fill on actuals. Default is 48 hours.
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with imputed production columns.
    """
    df = df.copy()
    
    # Auto-detect forecast columns
    if forecast_cols is None:
        forecast_cols = [
            c
            for c in df.columns
            if c.startswith("production_forecast_")
            or c.startswith("generation_forecast_")
            or c.startswith("forecast_generation_")
        ]
    
    # Auto-detect actual generation columns
    if actual_cols is None:
        actual_cols = [
            c for c in df.columns 
            if (c.startswith("production_") or c.startswith("generation_"))
            and "forecast" not in c.lower()
        ]

    forecast_ff = gap_threshold_hours if gap_threshold_hours is not None else 6

    # Forecasts: short FF + hierarchy fallback.
    df = impute_forward_fill_seasonal(
        df=df,
        columns=forecast_cols,
        gap_threshold_hours=forecast_ff,
        seasonal_groups=["hour", "weekday"],
        date_col=date_col,
        fallback_groups=[["hour"]],
        min_count=8,
        missing_fallback="global_median",
        seasonal_fit_mask=seasonal_fit_mask,
    )
    
    # Actuals: different strategy by fuel type
    df = impute_production_actuals_by_fuel(
        df,
        actual_cols,
        date_col=date_col,
        seasonal_fit_mask=seasonal_fit_mask,
        uniform_gap_threshold_hours=gap_threshold_hours,
    )
    
    return df


def impute_consumption(
    df: pd.DataFrame,
    forecast_cols: list[str] | None = None,
    actual_cols: list[str] | None = None,
    gap_threshold_hours: int | None = None,
    date_col: str = "begin_date",
    seasonal_fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
    forecast_gap_threshold_hours: int = 6,
    actual_gap_threshold_hours: int = 3,
) -> pd.DataFrame:
    """
    Impute missing consumption/load data.
    
    Strategy:
    - Forecasts: Forward-fill only
    - Actuals: Forward-fill short gaps, seasonal median for long gaps
              Grouping: hour × weekday
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with consumption columns and date column.
    forecast_cols : list[str] | None, optional
        Forecast column names. If None, auto-detect 'consumption_forecast_*'.
    actual_cols : list[str] | None, optional
        Actual load column names. If None, auto-detect 'consumption_load_actual_*'.
    gap_threshold_hours : int, optional
        Maximum gap for forward-fill. Default is 48 hours.
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with imputed consumption columns.
    """
    df = df.copy()
    
    # Auto-detect columns
    if forecast_cols is None:
        forecast_cols = [c for c in df.columns if c.startswith("consumption_forecast_")]
    
    if actual_cols is None:
        actual_cols = [c for c in df.columns if c.startswith("consumption_load_actual_")]

    if gap_threshold_hours is not None:
        forecast_gap_threshold_hours = gap_threshold_hours
        actual_gap_threshold_hours = gap_threshold_hours

    # Forecasts: short FF + hierarchical fallback.
    df = impute_forward_fill_seasonal(
        df=df,
        columns=forecast_cols,
        gap_threshold_hours=forecast_gap_threshold_hours,
        seasonal_groups=["hour", "weekday"],
        date_col=date_col,
        fallback_groups=[["hour"]],
        min_count=8,
        missing_fallback="global_median",
        seasonal_fit_mask=seasonal_fit_mask,
    )

    # Actuals: stricter FF horizon + same hierarchy.
    df = impute_forward_fill_seasonal(
        df=df,
        columns=actual_cols,
        gap_threshold_hours=actual_gap_threshold_hours,
        seasonal_groups=["hour", "weekday"],
        date_col=date_col,
        fallback_groups=[["hour"]],
        min_count=8,
        missing_fallback="global_median",
        seasonal_fit_mask=seasonal_fit_mask,
    )
    
    return df


# PV night hours: 21:00 to 01:00 inclusive (no sunlight)
PV_NIGHT_HOURS = {21, 22, 23, 0, 1}


def apply_pv_night_zero(
    df: pd.DataFrame,
    columns: list[str],
    date_col: str = "begin_date",
    night_hours: set[int] | None = None,
) -> pd.DataFrame:
    """
    Force PV columns to zero during night hours.
    
    This is a physical constraint: PV generation is zero when there's no sunlight.
    Applied as a post-processing step after imputation to ensure no spurious
    non-zero values are imputed during night hours.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with PV columns and date column.
    columns : list[str]
        List of PV column names to apply night zeroing to.
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
    night_hours : set[int] | None, optional
        Set of hours (0-23) considered "night". Default is {21, 22, 23, 0, 1}.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with PV columns zeroed during night hours.
    """
    df = df.copy()
    
    if night_hours is None:
        night_hours = PV_NIGHT_HOURS
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Create night mask
    hour = df[date_col].dt.hour
    is_night = hour.isin(night_hours)
    
    # Zero out PV columns during night
    for col in columns:
        if col in df.columns:
            n_zeroed = (is_night & df[col].notna() & (df[col] != 0)).sum()
            df.loc[is_night, col] = 0.0
            if n_zeroed > 0:
                logger.debug(f"PV night zeroing {col}: {n_zeroed} values set to 0")
    
    return df


def impute_wind_pv(
    df: pd.DataFrame,
    forecast_cols: list[str] | None = None,
    actual_cols: list[str] | None = None,
    gap_threshold_hours: int | None = None,
    date_col: str = "begin_date",
    use_sparse_imputation: bool = True,
    pv_night_zero: bool = True,
    pv_level_adjustment_actuals: bool = True,
    seasonal_fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
    forecast_gap_threshold_hours: int = 6,
    actual_gap_threshold_hours: int = 3,
) -> pd.DataFrame:
    """
    Impute missing wind/PV generation data with sparsity-aware handling.
    
    Strategy:
    - Wind forecasts/actuals: Tiered seasonal fallback (hour×month → hour×weekday → hour)
    - PV forecasts: Tiered seasonal fallback, no level adjustment
    - PV actuals: Tiered seasonal fallback WITH level adjustment (preserves weather regime)
    - PV night hours (21:00-01:00): Forced to zero (physical constraint)
    
    This function handles extremely sparse data (>40% missing) where certain
    seasonal groups may have zero samples (e.g., PV at night hours).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with wind/PV columns and date column.
    forecast_cols : list[str] | None, optional
        Forecast column names. If None, auto-detect 'wind_forecast_*' and 'pv_forecast_*'.
    actual_cols : list[str] | None, optional
        Actual generation column names. If None, auto-detect 'wind_actual_*' 
        and 'pv_actual_*' columns.
    gap_threshold_hours : int, optional
        Maximum gap for forward-fill. Default is 48 hours.
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
    use_sparse_imputation : bool, optional
        If True, use sparsity-aware tiered fallback. Default is True.
        Set to False to use original behavior (for backward compatibility).
    pv_night_zero : bool, optional
        If True, force PV columns to zero during night hours (21:00-01:00).
        Default is True.
    pv_level_adjustment_actuals : bool, optional
        If True, apply level adjustment for PV actuals (preserves local weather
        regime by carrying forward residual). Default is True.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with imputed wind/PV columns.
        
    Notes
    -----
    Wind and PV have strong seasonal patterns tied to time of day and month.
    For sparse data, the function uses tiered fallback:
    - Primary: hour × month (captures seasonal patterns)
    - Fallback 1: hour × weekday (if month data is sparse)
    - Fallback 2: hour only (if weekday data is also sparse)
    - Ultimate: global median (prevents NaN when all else fails)
    
    Shrinkage (k=48) is applied to blend low-count group medians toward
    coarser estimates, stabilizing groups with only 1-2 samples.
    """
    df = df.copy()
    
    # Auto-detect forecast columns
    if forecast_cols is None:
        forecast_cols = [
            c for c in df.columns 
            if c.startswith("wind_forecast_") or c.startswith("pv_forecast_")
        ]
    
    # Auto-detect actual columns
    if actual_cols is None:
        actual_cols = [
            c for c in df.columns 
            if c.startswith("wind_actual_") or c.startswith("pv_actual_")
        ]
    
    # Separate wind and PV columns
    wind_forecast_cols = [c for c in forecast_cols if "wind" in c.lower()]
    pv_forecast_cols = [c for c in forecast_cols if "solar" in c.lower() or "pv" in c.lower()]
    wind_actual_cols = [c for c in actual_cols if "wind" in c.lower()]
    pv_actual_cols = [c for c in actual_cols if "solar" in c.lower() or "pv" in c.lower()]

    if gap_threshold_hours is not None:
        forecast_gap_threshold_hours = gap_threshold_hours
        actual_gap_threshold_hours = gap_threshold_hours
    
    # Sparsity-aware settings
    seasonal_groups = ["hour", "month"]
    fallback_groups = [["hour", "weekday"], ["hour"]]
    min_count = 24  # Require ~1 day of samples per group
    shrinkage_k = 12.0  # Count-based shrinkage, then blend toward global median.
    
    if use_sparse_imputation:
        # Wind forecasts: tiered seasonal, no level adjustment
        df = impute_forward_fill_seasonal_sparse(
            df, wind_forecast_cols, forecast_gap_threshold_hours, seasonal_groups, date_col,
            fallback_groups=fallback_groups,
            min_count=min_count,
            shrinkage_k=shrinkage_k,
            missing_fallback="global_median",
            level_adjustment=False,
            seasonal_fit_mask=seasonal_fit_mask,
            shrinkage_target="global",
        )
        
        # Wind actuals: tiered seasonal, no level adjustment
        df = impute_forward_fill_seasonal_sparse(
            df, wind_actual_cols, actual_gap_threshold_hours, seasonal_groups, date_col,
            fallback_groups=fallback_groups,
            min_count=min_count,
            shrinkage_k=shrinkage_k,
            missing_fallback="global_median",
            level_adjustment=False,
            seasonal_fit_mask=seasonal_fit_mask,
            shrinkage_target="global",
        )
        
        # PV forecasts: tiered seasonal, no level adjustment
        df = impute_forward_fill_seasonal_sparse(
            df, pv_forecast_cols, forecast_gap_threshold_hours, seasonal_groups, date_col,
            fallback_groups=fallback_groups,
            min_count=min_count,
            shrinkage_k=shrinkage_k,
            missing_fallback="global_median",
            level_adjustment=False,
            seasonal_fit_mask=seasonal_fit_mask,
            shrinkage_target="global",
        )
        
        # PV actuals: tiered seasonal WITH level adjustment
        df = impute_forward_fill_seasonal_sparse(
            df, pv_actual_cols, actual_gap_threshold_hours, seasonal_groups, date_col,
            fallback_groups=fallback_groups,
            min_count=min_count,
            shrinkage_k=shrinkage_k,
            missing_fallback="global_median",
            level_adjustment=pv_level_adjustment_actuals,
            max_residual_gap=168,  # Decay residual after 1 week
            seasonal_fit_mask=seasonal_fit_mask,
            shrinkage_target="global",
        )
    else:
        # Original behavior for backward compatibility
        df = impute_forward_fill_seasonal(
            df,
            forecast_cols,
            forecast_gap_threshold_hours,
            seasonal_groups,
            date_col,
            fallback_groups=fallback_groups,
            min_count=min_count,
            missing_fallback="global_median",
            seasonal_fit_mask=seasonal_fit_mask,
        )
        df = impute_forward_fill_seasonal(
            df,
            actual_cols,
            actual_gap_threshold_hours,
            seasonal_groups,
            date_col,
            fallback_groups=fallback_groups,
            min_count=min_count,
            missing_fallback="global_median",
            seasonal_fit_mask=seasonal_fit_mask,
        )
    
    # Apply PV night zeroing (physical constraint)
    if pv_night_zero:
        all_pv_cols = pv_forecast_cols + pv_actual_cols
        df = apply_pv_night_zero(df, all_pv_cols, date_col)
    
    return df


def impute_cross_border_flows(
    df: pd.DataFrame,
    flow_cols: list[str] | None = None,
    gap_threshold_hours: int = 10,
    date_col: str = "begin_date",
    seasonal_fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
) -> pd.DataFrame:
    """
    Impute missing cross-border flow data.
    
    Strategy:
    - Forward-fill for short gaps (< 10 hours by default)
    - Seasonal median for long gaps: hour × weekday
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with flow columns and date column.
    flow_cols : list[str] | None, optional
        Flow column names. If None, auto-detect 'cross_border_*' columns.
    gap_threshold_hours : int, optional
        Maximum gap for forward-fill. Default is 10 hours.
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with imputed flow columns.
        
    Notes
    -----
    Cross-border flows are very persistent (high autocorrelation) but have
    weekly patterns (weekday vs weekend traffic). Using a shorter threshold
    (10 hours) exploits this persistence.
    """
    df = df.copy()
    
    # Auto-detect flow columns
    if flow_cols is None:
        flow_cols = [c for c in df.columns if c.startswith("cross_border_")]
    
    # Use hour × weekday for seasonal patterns
    return impute_forward_fill_seasonal(
        df=df,
        columns=flow_cols,
        gap_threshold_hours=gap_threshold_hours,
        seasonal_groups=["hour", "weekday"],
        date_col=date_col,
        fallback_groups=[["hour"]],
        min_count=8,
        missing_fallback="global_median",
        seasonal_fit_mask=seasonal_fit_mask,
    )


def impute_scheduled_exchanges(
    df: pd.DataFrame,
    sce_cols: list[str] | None = None,
    gap_threshold_hours: int = 3,
    date_col: str = "begin_date",
    seasonal_fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
) -> pd.DataFrame:
    """
    Impute scheduled exchange forecasts with minimal short-horizon carry.

    Strategy:
    - Forward-fill for short gaps only (default 3h)
    - Seasonal fallback: hour × weekday -> hour -> global median
    """
    df = df.copy()

    if sce_cols is None:
        sce_cols = [c for c in df.columns if c.startswith("sce_forecast_")]

    return impute_forward_fill_seasonal(
        df=df,
        columns=sce_cols,
        gap_threshold_hours=gap_threshold_hours,
        seasonal_groups=["hour", "weekday"],
        date_col=date_col,
        fallback_groups=[["hour"]],
        min_count=8,
        missing_fallback="global_median",
        seasonal_fit_mask=seasonal_fit_mask,
    )


def impute_bloomberg(
    df: pd.DataFrame,
    bloomberg_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Impute missing Bloomberg financial data.
    
    Strategy: Forward-fill only (no seasonal fallback).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Bloomberg columns.
    bloomberg_cols : list[str] | None, optional
        Bloomberg column names. If None, auto-detect 'bloomberg_*' columns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with forward-filled Bloomberg columns.
        
    Notes
    -----
    Bloomberg prices are already shifted backwards by 1 day in the loader.
    Financial data missing typically means market closed or data unavailable.
    Further imputation (like seasonal patterns) would create artificial patterns
    that don't reflect market reality.
    """
    df = df.copy()
    
    if bloomberg_cols is None:
        bloomberg_cols = [c for c in df.columns if c.startswith("bloomberg_")]
    
    return impute_forward_fill_only(df, bloomberg_cols)


def impute_redispatch_core(
    df: pd.DataFrame,
    data_driven_cols: list[str] | None = None,
    runlength_cols: list[str] | None = None,
    indicator_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Impute redispatch core features.
    
    Strategy:
    - Data-driven features (active_duration_ratio, max_load_q90, etc.): Fill with 0
    - Runlength features (run_len_pos, run_switches, etc.): Fill with 0
    - Indicator columns: Do NOT impute (derived, not imputed)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with redispatch features.
    data_driven_cols : list[str] | None, optional
        Data-driven feature column names. If None, auto-detect common patterns.
    runlength_cols : list[str] | None, optional
        Runlength feature column names. If None, auto-detect 'run_*' columns.
    indicator_cols : list[str] | None, optional
        Indicator column names to skip. If None, auto-detect '*_indicator' columns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with imputed redispatch features.
        
    Notes
    -----
    Missing redispatch data typically means no redispatch activity was recorded
    for that period. Zero is a valid and meaningful fill value.
    """
    df = df.copy()
    
    # Auto-detect data-driven columns
    if data_driven_cols is None:
        data_driven_cols = [
            c for c in df.columns 
            if c.startswith("data_driven_")
            or any(
                token in c.lower()
                for token in [
                    "active_duration_ratio",
                    "max_load_q",
                    "mean_load",
                    "std_load",
                    "energy",
                ]
            )
        ]
    
    # Auto-detect runlength columns
    if runlength_cols is None:
        runlength_cols = [
            c for c in df.columns 
            if c.startswith("runlength_") or c.startswith("run_len_") or c.startswith("run_")
        ]
    
    # Auto-detect indicator columns (to skip)
    if indicator_cols is None:
        indicator_cols = [c for c in df.columns if "indicator" in c.lower()]
    
    # Fill data-driven and runlength with 0
    cols_to_zero_fill = list(set(data_driven_cols + runlength_cols) - set(indicator_cols))
    df = impute_fill_zero(df, cols_to_zero_fill)
    
    # Log indicator columns that are skipped
    if indicator_cols:
        logger.debug(f"Skipping imputation for indicator columns: {indicator_cols}")
    
    return df


# =============================================================================
# Utility Functions
# =============================================================================


def get_imputation_stats(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute imputation statistics: count and percentage of imputed values.
    
    Parameters
    ----------
    df_before : pd.DataFrame
        DataFrame before imputation.
    df_after : pd.DataFrame
        DataFrame after imputation.
    columns : list[str] | None, optional
        Columns to compute stats for. If None, uses all numeric columns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - column: column name
        - missing_before: count of NaN before imputation
        - missing_after: count of NaN after imputation
        - imputed_count: number of values imputed
        - imputed_pct: percentage of values imputed
    """
    if columns is None:
        columns = df_before.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = []
    for col in columns:
        if col in df_before.columns and col in df_after.columns:
            missing_before = df_before[col].isna().sum()
            missing_after = df_after[col].isna().sum()
            imputed = missing_before - missing_after
            total = len(df_before)
            
            stats.append({
                "column": col,
                "missing_before": missing_before,
                "missing_after": missing_after,
                "imputed_count": imputed,
                "imputed_pct": (imputed / total * 100) if total > 0 else 0,
            })
    
    return pd.DataFrame(stats)


def validate_no_missing(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    raise_error: bool = False,
) -> dict[str, int]:
    """
    Validate that columns have no missing values after imputation.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    columns : list[str] | None, optional
        Columns to check. If None, checks all columns except 'begin_date'.
    raise_error : bool, optional
        If True, raise ValueError when missing values found. Default is False.
        
    Returns
    -------
    dict[str, int]
        Dictionary mapping column names to count of remaining NaN values.
        Only includes columns with at least one NaN.
        
    Raises
    ------
    ValueError
        If raise_error=True and any columns have missing values.
    """
    if columns is None:
        columns = [c for c in df.columns if c != "begin_date"]
    
    missing_cols = {}
    for col in columns:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                missing_cols[col] = n_missing
    
    if missing_cols:
        msg = f"Columns with remaining NaN values: {missing_cols}"
        logger.warning(msg)
        if raise_error:
            raise ValueError(msg)
    
    return missing_cols


def impute_all_features(
    df: pd.DataFrame,
    date_col: str = "begin_date",
    seasonal_fit_mask: pd.Series | np.ndarray | list[bool] | None = None,
) -> pd.DataFrame:
    """
    Apply appropriate imputation to all feature types in a combined DataFrame.
    
    This function auto-detects feature types based on column name prefixes
    and applies the appropriate imputation strategy for each.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with mixed feature types.
    date_col : str, optional
        Name of the datetime column. Default is 'begin_date'.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all features imputed.
        
    Notes
    -----
    Feature type detection is based on column prefixes:
    - day_ahead_price_* : Day-ahead prices
    - bloomberg_* : Bloomberg financial data
    - production_* : Production/generation data
    - consumption_* : Consumption/load data
    - wind_*, pv_* : Wind/PV data
    - cross_border_* : Cross-border flows
    - run_*, *_indicator : Redispatch core features
    """
    df = df.copy()
    
    # Track original missing values for logging
    total_missing_before = df.isna().sum().sum()
    
    # Apply feature-specific imputation
    df = impute_day_ahead_prices(df, date_col=date_col, seasonal_fit_mask=seasonal_fit_mask)
    df = impute_bloomberg(df)
    df = impute_production(df, date_col=date_col, seasonal_fit_mask=seasonal_fit_mask)
    df = impute_consumption(df, date_col=date_col, seasonal_fit_mask=seasonal_fit_mask)
    df = impute_wind_pv(df, date_col=date_col, seasonal_fit_mask=seasonal_fit_mask)
    df = impute_scheduled_exchanges(df, date_col=date_col, seasonal_fit_mask=seasonal_fit_mask)
    df = impute_cross_border_flows(df, date_col=date_col, seasonal_fit_mask=seasonal_fit_mask)
    df = impute_redispatch_core(df)
    
    total_missing_after = df.isna().sum().sum()
    
    logger.info(
        f"Imputation complete: {total_missing_before} -> {total_missing_after} missing values"
    )
    
    return df
