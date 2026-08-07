"""
Redispatch Core Module.

This module provides functions for processing and preparing redispatch target data
for forecasting models. It extracts and consolidates the target preparation logic
from the data_processing/redispatch_model_data_processing.ipynb notebook.

The main entry point is `prepare_final_target()`, which orchestrates the full
pipeline from raw redispatch data to hourly aggregated targets with features.
"""

import re
import json
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pytz
from zoneinfo import ZoneInfo

from dataset_preparation.feature_loaders.imputation import impute_redispatch_core

from .tso_config import (
    GERMAN_TSOS,
    BUNDESLAND_TO_TSO,
    TARGET_SPLIT_METHOD_DICT,
    normalize_tso_name,
    get_split_method,
)


# =============================================================================
# Constants
# =============================================================================

# Measurement reason categories for filtering redispatch data
TARGET_RELEVANT_MEASUREMENT_REASONS: dict[str, list[str]] = {
    "electricity_only_redispatch": [
        "Electricty-related redispatch",
        "Electricity- and voltage-related RD"
    ],
    "domestic_redispatch": [
        "Electricty-related redispatch",
        "Voltage-related redispatch",
        "Electricity- and voltage-related RD"
    ],
    "all_redispatch": [
        "Electricty-related redispatch",
        "Voltage-related redispatch",
        "Electricity- and voltage-related RD",
        "Electricity-related countertrade DE-DK1",
        "Electricity-related countertrade DE-DK2",
        "Electricity-related countertrade DE-NO2",
    ]
}

# Default rolling window size for lagged data-driven and runlength rolling features (hours)
ROLLING_WINDOW_DATA_DRIVEN_HOURS: int = 24

# Default rolling window size for other lagged features (days)
ROLLING_WINDOW_LAGGED_DAYS: int = 7

# W&B artifact naming prefix for datasets
WANDB_DATASET_ARTIFACT_PREFIX: str = "redispatch_dataset_"

# Default minimum gap ratio for filtering (allows slight negative gaps)
MIN_GAP_RATIO: float = -0.1

# Feature set type hints
FeatureSet = Literal[
    "data_driven",
    "runlength",
    "day_ahead_price",
    "production_consumption",
    "wind_pv",
    "cross_border",
    "bloomberg",
]

# -------------------------------------------------------------------------
# Feature Reduction: Runlength/Data-Driven Features to Keep
# -------------------------------------------------------------------------
# Reduced feature set for better model efficiency and generalization
# Keep only the most predictive features (8 total instead of ~15+)
RUNLENGTH_FEATURES_TO_KEEP = [
    "runlength_zero",           # How long since activity
    "runlength_switches",       # Volatility measure
    "runlength_indicator",      # Binary activity indicator (kept for internal use)
]

DATA_DRIVEN_FEATURES_TO_KEEP = [
    "data_driven_n_interventions",       # Count of overlapping interventions
    "data_driven_max_load_weighted_proxy",  # Peak load proxy
    "data_driven_total_active_duration",    # Total exposure time
]


# =============================================================================
# DST-Aware Date Parsing
# =============================================================================

def _parse_dst_aware(date_str: str, time_str: str, tz: ZoneInfo) -> datetime:
    """
    Parse a pair (date_str, time_str) where time_str can contain 'A' or 'B' markers
    to indicate the occurrence during DST fallback.

    Parameters
    ----------
    date_str : str
        Date string in format "DD.MM.YYYY".
    time_str : str
        Time string in format "HH:MM" or "HH:MM AM/PM".
        May contain 'A' (first occurrence) or 'B' (second occurrence) for DST.
    tz : ZoneInfo
        Target timezone.

    Returns
    -------
    datetime
        UTC-aware datetime.

    Examples
    --------
    >>> _parse_dst_aware("25.10.2020", "2:00", ZoneInfo("Europe/Berlin"))
    datetime(2020, 10, 25, 0, 0, tzinfo=UTC)  # first occurrence
    >>> _parse_dst_aware("25.10.2020", "2B:00", ZoneInfo("Europe/Berlin"))
    datetime(2020, 10, 25, 1, 0, tzinfo=UTC)  # second occurrence
    """
    s = str(time_str).strip()
    if s == "":
        return pd.NaT

    # Determine fold from suffix marker
    fold = 0
    if "B" in s:
        fold = 1
        s = s.replace("B", "")
    elif "A" in s and not s.endswith("AM") and not s.endswith("PM"):
        fold = 0
        s = s.replace("A", "")

    s = s.strip()

    # Detect 12h clock vs 24h clock
    upper = s.upper()
    is_12h = ("AM" in upper) or ("PM" in upper)

    # Normalize spacing
    s = " ".join(s.split())

    fmt = "%d.%m.%Y %H:%M %p" if is_12h else "%d.%m.%Y %H:%M"
    dt = datetime.strptime(f"{date_str} {s}", fmt)

    # Attach timezone and fold to disambiguate fallback hour
    return dt.replace(tzinfo=tz, fold=fold).astimezone(pytz.UTC)


def parse_redispatch_columns_dst_aware(
    df: pd.DataFrame, 
    tz_name: str = "Europe/Berlin"
) -> pd.DataFrame:
    """
    Parse begin_date/end_date columns with DST awareness.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with begin_date, begin_time, end_date, end_time columns.
    tz_name : str
        Timezone name (default: "Europe/Berlin").

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed datetime columns (UTC, tz-naive).
    """
    tz = ZoneInfo(tz_name)

    for date_col, time_col in [("begin_date", "begin_time"), ("end_date", "end_time")]:
        date_s = df[date_col].astype(str)
        df[date_col] = [
            _parse_dst_aware(d, t, tz) for d, t in zip(date_s, df[time_col])
        ]
        df[date_col] = df[date_col].dt.tz_localize(None)

    return df


def convert_to_utc_hourly(date_series: pd.Series, timezone: str) -> pd.Series:
    """
    Localize naive datetimes to a timezone and convert to UTC.

    Handles ambiguous timestamps (DST fallback) by choosing the 'later' occurrence.
    Non-existent timestamps (spring forward) are shifted forward by 1 hour.

    Parameters
    ----------
    date_series : pd.Series
        Series of naive datetime objects.
    timezone : str
        Source timezone name.

    Returns
    -------
    pd.Series
        UTC datetimes (tz-naive for easier downstream processing).
    """
    tz = pytz.timezone(timezone)
    results = []

    for ts in date_series:
        if pd.isna(ts):
            results.append(pd.NaT)
            continue

        dt = pd.Timestamp(ts).to_pydatetime()

        try:
            aware = tz.localize(dt, is_dst=None)
        except pytz.AmbiguousTimeError:
            aware = tz.localize(dt, is_dst=True)
        except pytz.NonExistentTimeError:
            dt2 = pd.Timestamp(dt) + pd.Timedelta(hours=1)
            aware = tz.localize(dt2.to_pydatetime(), is_dst=True)

        results.append(aware.astimezone(pytz.UTC))

    return pd.Series(results, index=date_series.index).dt.tz_localize(None)


# =============================================================================
# Reading and Filtering Redispatch Data
# =============================================================================

def read_redispatch_data(
    file_path: str,
    translations_path: str,
    timezone: str = "Europe/Berlin",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Read and parse redispatch data from CSV file.

    Parameters
    ----------
    file_path : str
        Path to the redispatch CSV file.
    translations_path : str
        Path to the translations.json file for column renaming.
    timezone : str
        Timezone of the input data ("Europe/Berlin" or "UTC").
    start_date : datetime, optional
        Filter data to start from this date.
    end_date : datetime, optional
        Filter data to end at this date.

    Returns
    -------
    pd.DataFrame
        Parsed redispatch data with datetime columns in UTC.
    """
    with open(translations_path, "r") as f:
        translations = json.load(f)

    redispatch = pd.read_csv(
        file_path,
        sep=";",
        decimal=",",
    ).drop(columns=["ZEITZONE_VON", "ZEITZONE_BIS"], errors="ignore")

    # Translate column values
    for column_name, column_values_dict in translations["column_values"].items():
        for initial_value_name, new_value_name in column_values_dict.items():
            if column_name in redispatch.columns:
                redispatch[column_name] = redispatch[column_name].str.replace(
                    initial_value_name, new_value_name
                )
    redispatch = redispatch.rename(columns=translations["column_names"])

    # Parse dates
    if timezone == "UTC":
        redispatch["begin_date"] = pd.to_datetime(
            redispatch["begin_date"] + " " + redispatch["begin_time"],
            format="%d.%m.%Y %H:%M"
        )
        redispatch["end_date"] = pd.to_datetime(
            redispatch["end_date"] + " " + redispatch["end_time"],
            format="%d.%m.%Y %H:%M"
        )
    else:
        redispatch = parse_redispatch_columns_dst_aware(redispatch, tz_name=timezone)
    redispatch.drop(columns=["begin_time", "end_time"], inplace=True)

    # Filter by date
    last_data_full_month = redispatch["begin_date"][
        redispatch["begin_date"].dt.is_month_end
    ].dt.date.max()

    if end_date is not None:
        end_date_filter = redispatch["end_date"] < end_date
    else:
        end_date_filter = redispatch["end_date"] < last_data_full_month + pd.offsets.Day(1)

    if start_date is not None:
        redispatch = redispatch[
            np.logical_and(
                redispatch["begin_date"] >= start_date,
                end_date_filter
            )
        ]
    else:
        redispatch = redispatch[end_date_filter]

    # Remove zero load and missing affected_unit
    redispatch = redispatch[
        np.logical_and(
            redispatch["total_load"] != 0,
            redispatch["affected_unit"].notna()
        )
    ].copy()

    # Parse requesting_operator
    def _filter_relevant_ops(cell: str):
        if pd.isna(cell):
            return pd.NA
        parts = re.split(r'\s*&\s*', str(cell))
        return parts

    redispatch["requesting_operator"] = redispatch["requesting_operator"].apply(
        _filter_relevant_ops
    )

    return redispatch


def prepare_germany_only_target(
    redispatch_data: pd.DataFrame, 
    include_countertrading: bool = True
) -> pd.DataFrame:
    """
    Filter redispatch data to keep only interventions affecting Germany.

    Parameters
    ----------
    redispatch_data : pd.DataFrame
        Redispatch data with 'Energieträger', 'affected_unit', 'Bundesland der Einheit',
        and 'primary_energy_type' columns.
    include_countertrading : bool
        If True, keep countertrading entries (Börse). Default is True.

    Returns
    -------
    pd.DataFrame
        Filtered redispatch data for Germany-only interventions.
    """
    affected_units_to_filter = ["Vorarlberger Ilwerke"]
    if not include_countertrading:
        affected_units_to_filter.extend(["Börse", "B¿rse"])

    inside_germany_filter = np.logical_and.reduce([
        redispatch_data["Energieträger"] != "outside_germany",
        ~redispatch_data["affected_unit"].isin(affected_units_to_filter),
        np.logical_or(
            redispatch_data["Bundesland der Einheit"].isna(),
            redispatch_data["Bundesland der Einheit"].isin(list(BUNDESLAND_TO_TSO.keys()))
        )
    ])
    germany_redispatch = redispatch_data[inside_germany_filter].copy()

    # Drop values without primary_energy_type
    return germany_redispatch[~germany_redispatch["primary_energy_type"].isna()].copy()


# =============================================================================
# Processing for Directing Operator
# =============================================================================

def process_for_directing_operator(
    data: pd.DataFrame,
    relevant_measurement_reasons: list[str],
    min_gap_ratio: float = -0.1,
    relevant_operators: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Filter and process redispatch data for directing operator analysis.

    This function:
    1. Filters by measurement reasons
    2. Removes entries with anomalous duration gaps
    3. Extracts directing operator as the main operator column

    Parameters
    ----------
    data : pd.DataFrame
        Redispatch data with 'measurement_reason', 'directing_operator',
        'total_load', 'mean_load', 'begin_date', 'end_date', and 'primary_energy_type'.
    relevant_measurement_reasons : list[str]
        List of measurement reasons to include.
    min_gap_ratio : float
        Minimum allowed ratio of (1 - active_duration/total_duration).
        Negative values allow active duration > total duration (data quality issue).
        Default is -0.1.
    relevant_operators : list[str], optional
        List of operator names to include. Defaults to all German TSOs.

    Returns
    -------
    pd.DataFrame
        Processed data with 'operator' column (from directing_operator).
    """
    if relevant_operators is None:
        relevant_operators = GERMAN_TSOS.copy()

    data = data[
        data["measurement_reason"].isin(relevant_measurement_reasons)
    ].copy(deep=False)

    # Calculate duration gap ratio
    active_duration = data["total_load"] / data["mean_load"]
    total_duration = (data["end_date"] - data["begin_date"]).dt.total_seconds() / 3600
    duration_gaps_vs_total_durations = 1 - (active_duration / total_duration)

    # Remove entries with anomalous gaps
    points_to_drop = duration_gaps_vs_total_durations < min_gap_ratio
    print(f"Dropped {points_to_drop.sum() / len(data):.2%} rows due to duration gap ratio < {min_gap_ratio}.")
    data = data.loc[~points_to_drop].copy(deep=False)

    operator_data = data["directing_operator"]
    if operator_data.str.count("&").sum() > 0:
        raise NotImplementedError("Multiple directing operators should not happen.")

    region_redispatch = (
        data.copy(deep=True).loc[operator_data.index]
        .drop(columns=["requesting_operator", "directing_operator"], errors="ignore")
    )
    region_redispatch.drop(columns="region", inplace=True, errors="ignore")
    region_redispatch["operator"] = operator_data

    region_redispatch = region_redispatch[
        np.logical_and(
            region_redispatch["operator"].isin(relevant_operators),
            region_redispatch["primary_energy_type"].notna()
        )
    ]

    return region_redispatch


# =============================================================================
# Hourly Split Methods
# =============================================================================

def bucket_overlap_split(row: pd.Series) -> pd.DataFrame:
    """
    Split intervention energy equally across overlapping hourly buckets.

    This is the "equal" split method - energy is allocated proportionally
    based on the overlap duration with each hour bucket.

    Parameters
    ----------
    row : pd.Series
        Row with 'begin_date', 'end_date', and 'total_load'.

    Returns
    -------
    pd.DataFrame
        Hourly buckets with allocated energy.
    """
    start: pd.Timestamp = row["begin_date"]
    stop: pd.Timestamp = row["end_date"]
    energy = row["total_load"]

    if pd.isna(start) or pd.isna(stop) or stop <= start or energy == 0:
        return pd.DataFrame(columns=["begin_date", "end_date", "total_load"])

    if start.tzinfo != stop.tzinfo:
        raise ValueError("begin_date and end_date must have same timezone")

    total_duration = (stop - start).total_seconds()

    bucket_starts = pd.date_range(
        start=start.floor("h"),
        end=(stop - pd.Timedelta(seconds=1)).floor("h"),
        freq="h",
        tz=start.tzinfo,
    )

    buckets = []
    for bucket_start in bucket_starts:
        bucket_end = bucket_start + pd.Timedelta(hours=1)
        overlap = min(stop, bucket_end) - max(start, bucket_start)
        overlap_seconds = overlap.total_seconds()

        if overlap_seconds > 0:
            buckets.append({
                "begin_date": bucket_start,
                "end_date": bucket_end,
                "total_load": energy * (overlap_seconds / total_duration),
            })

    return pd.DataFrame(buckets)


def split_block_allocation(row: pd.Series) -> pd.DataFrame:
    """
    Allocate energy to both front and back of the intervention period.

    Energy is distributed over [s_i, s_i + L_i/2] ∪ [e_i - L_i/2, e_i],
    where L_i is the active duration computed from total_load/mean_load.

    This is the "split" method - better suited for Amprion data.

    Parameters
    ----------
    row : pd.Series
        Row with 'begin_date', 'end_date', 'total_load', and 'mean_load'.

    Returns
    -------
    pd.DataFrame
        Hourly buckets with allocated energy.
    """
    start: pd.Timestamp = row["begin_date"]
    stop: pd.Timestamp = row["end_date"]
    energy = row["total_load"]
    mean_load = row["mean_load"]

    if pd.isna(start) or pd.isna(stop) or stop <= start or energy == 0:
        return pd.DataFrame(columns=["begin_date", "end_date", "total_load"])

    if start.tzinfo != stop.tzinfo:
        raise ValueError("begin_date and end_date must have same timezone")

    total_duration = (stop - start).total_seconds() / 3600

    # Compute active duration
    if mean_load > 0:
        active_duration = min(energy / mean_load, total_duration)
    else:
        active_duration = total_duration

    # Handle edge cases
    if active_duration < 0:
        return bucket_overlap_split(row)

    if active_duration <= 0.25 or energy < 1.0:
        return bucket_overlap_split(row)

    half_duration = active_duration / 2

    # Define two intervals
    front_end = start + pd.Timedelta(hours=half_duration)
    back_start = stop - pd.Timedelta(hours=half_duration)

    # Check if intervals overlap
    if front_end >= back_start:
        return bucket_overlap_split(row)

    # Front interval buckets
    front_bucket_starts = pd.date_range(
        start=start.floor("h"),
        end=(front_end - pd.Timedelta(seconds=1)).floor("h"),
        freq="h",
        tz=start.tzinfo,
    )

    # Back interval buckets
    back_bucket_starts = pd.date_range(
        start=back_start.floor("h"),
        end=(stop - pd.Timedelta(seconds=1)).floor("h"),
        freq="h",
        tz=start.tzinfo,
    )

    buckets = []

    # Process front interval
    for bucket_start in front_bucket_starts:
        bucket_end = bucket_start + pd.Timedelta(hours=1)
        overlap = min(front_end, bucket_end) - max(start, bucket_start)
        overlap_hours = overlap.total_seconds() / 3600

        if overlap_hours > 0:
            buckets.append({
                "begin_date": bucket_start,
                "end_date": bucket_end,
                "total_load": energy * (overlap_hours / active_duration),
            })

    # Process back interval
    for bucket_start in back_bucket_starts:
        bucket_end = bucket_start + pd.Timedelta(hours=1)
        overlap = min(stop, bucket_end) - max(back_start, bucket_start)
        overlap_hours = overlap.total_seconds() / 3600

        if overlap_hours > 0:
            existing = [b for b in buckets if b["begin_date"] == bucket_start]
            if existing:
                existing[0]["total_load"] += energy * (overlap_hours / active_duration)
            else:
                buckets.append({
                    "begin_date": bucket_start,
                    "end_date": bucket_end,
                    "total_load": energy * (overlap_hours / active_duration),
                })

    return pd.DataFrame(buckets)


def get_isolated_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify intervals that don't overlap with any other intervals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'begin_date' and 'end_date' columns.

    Returns
    -------
    pd.DataFrame
        Subset of df containing only isolated (non-overlapping) intervals.
    """
    starts = df['begin_date'].values
    ends = df['end_date'].values

    overlap_counts = np.array([
        np.sum((starts <= ends[i]) & (ends >= starts[i]))
        for i in range(len(df))
    ])

    df = df.copy()
    df['overlap_count'] = overlap_counts
    return df[df['overlap_count'] == 1]


def apply_hourly_split_method(
    data: pd.DataFrame, 
    method: str = "equal"
) -> pd.DataFrame:
    """
    Apply hourly split method to convert interventions to hourly buckets.

    Parameters
    ----------
    data : pd.DataFrame
        Redispatch interventions with 'begin_date', 'end_date', 'total_load',
        'mean_load', and other columns.
    method : str
        Split method: "equal" (overlap-based) or "split" (front+back loaded).
        Default is "equal".

    Returns
    -------
    pd.DataFrame
        Hourly data with allocated energy and original columns preserved.
    """
    if method == "equal":
        split_func = bucket_overlap_split
    elif method == "split":
        split_func = split_block_allocation
    else:
        raise ValueError(f"Unknown method: {method}. Choose from ['equal', 'split']")

    processed_data = data.apply(split_func, axis=1).reset_index(drop=True)
    original_data_indices = data.index.repeat(processed_data.apply(len))
    concatenated_data = pd.concat(processed_data.to_list(), axis=0).reset_index(drop=True)
    concatenated_data["original_index"] = original_data_indices.values

    return concatenated_data.merge(
        data.drop(columns=["total_load", "begin_date", "end_date"]),
        how="left",
        left_on="original_index",
        right_index=True
    )

# =============================================================================
# Missing Interval Handling
# =============================================================================

def add_missing_intervals(
    data: pd.DataFrame,
    direction: str,
    start_date: datetime,
    end_date: datetime,
    freq: str = "h"
) -> pd.DataFrame:
    """
    Add missing time intervals to the data with zero values.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'begin_date' and 'total_load' columns.
    direction : str
        Direction label ("up" or "down") for the missing intervals.
    start_date : datetime
        Start of the full date range.
    end_date : datetime
        End of the full date range.
    freq : str
        Frequency for the intervals ("h" for hourly, "15min", etc.).

    Returns
    -------
    pd.DataFrame
        DataFrame with all time intervals present.
    """
    full_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    missing_intervals = pd.Series(
        full_range.difference(data["begin_date"]),
        name="begin_date"
    )
    prepared = data.merge(
        missing_intervals, on="begin_date", how="outer"
    ).fillna({"total_load": 0, "direction": direction})
    return prepared


# =============================================================================
# Feature Engineering
# =============================================================================

def compute_runlength_activity_exposure(data: pd.DataFrame, original_data: pd.DataFrame, original_intervention_column: str = "original_index") -> pd.DataFrame:
    data_copy = data.copy(deep=True)
    data_copy = data_copy.merge(
        original_data.loc[:, ["begin_date", "end_date"]],
        left_on=original_intervention_column,
        right_index=True,
        suffixes=("", "_orig"),
        how="left",
    )

    # Compute overlap between [begin_date, end_date) and [begin_date_orig, end_date_orig)
    # Formula: overlap = max(0, min(e_hr, e_ev) - max(b_hr, b_ev))
    b_hr = data_copy["begin_date"]
    e_hr = data_copy["end_date"]
    b_ev = data_copy["begin_date_orig"]
    e_ev = data_copy["end_date_orig"]

    overlap_start = b_hr.where(b_hr > b_ev, b_ev)   # max(b_hr, b_ev)
    overlap_end   = e_hr.where(e_hr < e_ev, e_ev)   # min(e_hr, e_ev)

    overlap_sec = (overlap_end - overlap_start).dt.total_seconds()
    overlap_sec = overlap_sec.clip(lower=0)

    hour_len_sec = (e_hr - b_hr).dt.total_seconds().clip(lower=1)  # avoid /0
    data_copy["runlength_activity_exposure"] = overlap_sec / hour_len_sec

    # Optional numeric hygiene
    data_copy["runlength_activity_exposure"] = data_copy["runlength_activity_exposure"].clip(0.0, 1.0)

    return data_copy

def add_data_driven_features_hourly(
    data: pd.DataFrame,
    original_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    original_intervention_column: str = "original_index",
    eps: float = 1e-1,
) -> pd.DataFrame:
    data_copy = data.copy()

    # --- Event-level: total duration (hours) ---
    total_durations = (
        (original_data["end_date"] - original_data["begin_date"])
        .dt.total_seconds()
        .div(3600.0)
    )

    # --- Event-level: active duration proxy (hours) ---
    mean_load = original_data["mean_load"].replace(0, np.nan)
    active_durations = (original_data["total_load"] / mean_load)

    # Replace near-equality with total duration (avoid tiny rounding errors)
    close_to_total = np.isclose(active_durations, total_durations, atol=eps, rtol=0.0)
    active_durations = active_durations.where(~close_to_total, total_durations)

    # Enforce bounds: 0 <= active_duration <= total_duration
    active_durations = active_durations.clip(lower=0)
    active_durations = np.minimum(active_durations, total_durations)

    # Fill remaining NaNs (e.g., mean_load==0 or missing) to 0
    active_durations = active_durations.fillna(0.0)

    # --- Hour-row exposure (hours overlapped within this hour-row) ---
    # Merge event begin/end for true overlap
    data_copy = compute_runlength_activity_exposure(
        data_copy,
        original_data,
        original_intervention_column,
    )

    # Total exposure per event across all its hour-rows
    data_copy["event_exposure_total"] = data_copy.groupby(original_intervention_column)["runlength_activity_exposure"].transform("sum")

    # Exposure ratio per hour-row; avoid divide-by-zero
    data_copy["event_exposure_ratio"] = np.where(
        data_copy["event_exposure_total"] > 0,
        data_copy["runlength_activity_exposure"] / data_copy["event_exposure_total"],
        0.0,
    )

    # Attach event-level attributes
    data_copy = data_copy.merge(
        active_durations.rename("event_active_duration"),
        left_on=original_intervention_column,
        right_index=True,
        how="left",
    )
    data_copy = data_copy.merge(
        original_data["max_load"].rename("event_max_load"),
        left_on=original_intervention_column,
        right_index=True,
        how="left",
    )

    data_copy["event_active_duration"] = data_copy["event_active_duration"].fillna(0.0)
    data_copy["event_max_load"] = data_copy["event_max_load"].fillna(0.0)

    # Allocate active duration into the hour-row (extensive quantity)
    data_copy["adjusted_active_duration"] = data_copy["event_active_duration"] * data_copy["event_exposure_ratio"]

    # Count of overlapped events
    data_copy["exposure_indicator"] = (data_copy["runlength_activity_exposure"] > 0).astype(int)

    # --- Aggregate to (hour, direction) ---
    g = data_copy.groupby(["begin_date", "direction"], sort=False)
    data_copy["max_load_weighted_component"] = data_copy["event_max_load"] * data_copy["event_exposure_ratio"]
    max_load_weighted_proxy = g["max_load_weighted_component"].max().rename("max_load_weighted_proxy")

    # Feature Reduction: Only compute features we need
    data_driven_features = pd.DataFrame({
        "data_driven_total_active_duration": g["adjusted_active_duration"].sum(),
        "data_driven_n_interventions": g["exposure_indicator"].sum(),
        "data_driven_max_load_weighted_proxy": max_load_weighted_proxy,
    }).reset_index()

    # Compute missing intervals
    data_driven_features_whole_interval_list = []
    for direction, group in data_driven_features.groupby("direction"):
        out = add_missing_intervals(group, direction=direction, start_date=start_date, end_date=end_date, freq="h").fillna({
            feature_name: (0.0 if "n_interventions" not in feature_name else 0)
            for feature_name in data_driven_features.columns if feature_name.startswith("data_driven_")
        })
        data_driven_features_whole_interval_list.append(out)

    data_driven_features_whole_interval = (
        pd.concat(data_driven_features_whole_interval_list, axis=0)
        .sort_values(["begin_date", "direction"]).reset_index(drop=True)
    )

    return data_driven_features_whole_interval


def hourly_union_exposure_15m(
    df_event_hour: pd.DataFrame,
    hour_start_col: str = "begin_date",
    hour_end_col: str = "end_date",
    ev_start_col: str = "begin_date_orig",
    ev_end_col: str = "end_date_orig",
    group_cols=("begin_date", "direction"),
) -> pd.DataFrame:
    """
    Build a 15-min union exposure per (hour, direction) from event-hour rows.

    Returns:
      - indicator: 1 if any 15-min slot overlapped
      - runlength_activity_exposure: (# overlapped 15-min slots) / 4
    """
    d = df_event_hour.copy()
    # 15-min slots within each hour: [h+0, h+15), [h+15, h+30), [h+30, h+45), [h+45, h+60)
    # For each row, mark which slots it overlaps.
    h0 = d[hour_start_col]
    # slot boundaries
    s0 = h0
    s1 = h0 + pd.Timedelta(minutes=15)
    s2 = h0 + pd.Timedelta(minutes=30)
    s3 = h0 + pd.Timedelta(minutes=45)
    s4 = d[hour_end_col]

    ev_b = d[ev_start_col]
    ev_e = d[ev_end_col]

    def overlaps(a_start, a_end, b_start, b_end):
        # interval overlap length > 0
        return (np.minimum(a_end.values.astype("datetime64[ns]"), b_end.values.astype("datetime64[ns]")) >
                np.maximum(a_start.values.astype("datetime64[ns]"), b_start.values.astype("datetime64[ns]")))

    d["_q0"] = overlaps(ev_b, ev_e, s0, s1).astype(np.int8)
    d["_q1"] = overlaps(ev_b, ev_e, s1, s2).astype(np.int8)
    d["_q2"] = overlaps(ev_b, ev_e, s2, s3).astype(np.int8)
    d["_q3"] = overlaps(ev_b, ev_e, s3, s4).astype(np.int8)

    g = d.groupby(list(group_cols), sort=False)
    out = g[["_q0","_q1","_q2","_q3"]].max().reset_index()  # union via OR -> max
    out["runlength_n_quarters_active"] = out[["_q0","_q1","_q2","_q3"]].sum(axis=1)
    out["runlength_activity_exposure"] = out["runlength_n_quarters_active"] / 4.0

    return out.drop(columns=["_q0","_q1","_q2","_q3"])

def process_before_runlength(
    df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    direction_col: str = "direction",
    exposure_col: str = "runlength_activity_exposure",
):
    processed_dfs = []
    for direction, group in df.groupby(direction_col):
        completed_df = hourly_union_exposure_15m(
            group,
            group_cols=("begin_date", direction_col),
        )
        completed_df = add_missing_intervals(
            completed_df,
            direction=direction,
            start_date=start_date,
            end_date=end_date,
        ).fillna({exposure_col: 0.0, "runlength_n_quarters_active": 0.0})
        processed_dfs.append(completed_df)

    return pd.concat(processed_dfs, axis=0).reset_index(drop=True)

def add_runlength_features_hourly(
    df: pd.DataFrame,
    original_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    y_col: str = "y",
    exposure_col: str = "runlength_activity_exposure",
    id_col: str = "unique_id",
    time_col: str = "ds",
    window_size: int = 24,
    state_mode: str = "exposure",   # "y", "exposure", "either"
    y_eps: float = 0.0,
    original_intervention_column: str = "original_index",
    exposure_eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Adds leakage-safe run-length + window features per id on an hourly panel.

    State definition (indicator):
      - "y":        1{y > y_eps}
      - "exposure": 1{runlength_activity_exposure > exposure_eps}
      - "either":   1{(y > y_eps) OR (runlength_activity_exposure > exposure_eps)}

    Features are computed from PAST values only (shift(1)).
    """
    df0 = df.copy()
    df0[time_col] = pd.to_datetime(df0[time_col])

    df0 = df0.sort_values([id_col, time_col], kind="mergesort")

    if exposure_col not in df0.columns:
        df0 = compute_runlength_activity_exposure(
            df0,
            original_data=original_data,
            original_intervention_column=original_intervention_column,
        )
        df0 = process_before_runlength(
            df0,
            direction_col=id_col,
            exposure_col=exposure_col,
            start_date=start_date,
            end_date=end_date,
        )

    # --- indicator ---
    if state_mode == "y":
        indicator = (df0[y_col] > y_eps).astype(np.int8)
    elif state_mode == "exposure":
        indicator = (df0[exposure_col] > exposure_eps).astype(np.int8)
    elif state_mode == "either":
        indicator = ((df0[y_col] > y_eps) | (df0[exposure_col] > exposure_eps)).astype(np.int8)
    else:
        raise ValueError("state_mode must be one of: 'y', 'exposure', 'either'.")

    df0["runlength_indicator"] = indicator

    # Use only PAST info
    df0["_s"] = df0.groupby(id_col, sort=False)["runlength_indicator"].shift(1).fillna(0).astype(np.int8)

    g = df0.groupby(id_col, sort=False)

    # --- run_len_pos / run_len_zero (consecutive past states ending at t-1) ---
    # Create run ID: changes whenever state changes
    df0["_state_change"] = g["_s"].diff().fillna(1).abs().astype(np.int8)
    df0["_run_id"] = g["_state_change"].cumsum()
    
    # Count within each run
    df0["_position_in_run"] = df0.groupby([id_col, "_run_id"], sort=False).cumcount() + 1
    
    # Run length of zeros: position within run, only when _s==0 (KEPT)
    df0["runlength_zero"] = np.where(df0["_s"] == 0, df0["_position_in_run"], 0).astype(np.int32)

    # Optional cap to window_size
    df0["runlength_zero"] = df0["runlength_zero"].clip(upper=window_size)

    # --- switches in rolling window (past window only) (KEPT) ---
    df0["_dswitch"] = g["_s"].diff().abs().fillna(0).astype(np.int8)
    df0["runlength_switches"] = g["_dswitch"].rolling(window_size, min_periods=1).sum().reset_index(level=0, drop=True).astype(np.int32)

    # Clean up - drop all intermediate columns
    df0 = df0.drop(columns=["_s", "_dswitch", "_state_change", "_run_id", "_position_in_run"], errors="ignore")
    
    # Also drop intermediate columns that may have been added earlier
    cols_to_drop = [
        "runlength_n_quarters_active", "runlength_activity_exposure",
        "begin_date_orig", "end_date_orig"
    ]
    df0 = df0.drop(columns=[c for c in cols_to_drop if c in df0.columns], errors="ignore")
    
    return df0

# =============================================================================
# Final Target Preparation
# =============================================================================

def prepare_final_target_data(
    data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime
) -> dict[str, pd.DataFrame]:
    """
    Aggregate and fill missing intervals for each operator.

    Parameters
    ----------
    data : pd.DataFrame
        Hourly redispatch data with 'operator', 'direction', 'begin_date',
        and 'total_load'.
    start_date : datetime
        Start of the date range.
    end_date : datetime
        End of the date range.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping operator names to their processed DataFrames.
    """
    redispatch_operators: dict[str, pd.DataFrame] = {}

    for (operator, direction), group in data.groupby(["operator", "direction"]):
        group_load_aggregated = group.groupby("begin_date")["total_load"].sum().reset_index()
        processed_target_with_intervals = add_missing_intervals(
            group_load_aggregated, direction, start_date, end_date
        )
        processed_target_with_intervals = processed_target_with_intervals.assign(
            direction=direction, operator=operator
        )

        if operator not in redispatch_operators:
            redispatch_operators[operator] = processed_target_with_intervals
        else:
            redispatch_operators[operator] = pd.concat(
                [redispatch_operators[operator], processed_target_with_intervals],
                axis=0
            ).sort_index()

    return redispatch_operators


def prepare_final_target(
    redispatch: pd.DataFrame,
    relevant_measurement_reasons: list[str],
    min_gap_ratio: float,
    feature_set: list[FeatureSet],
    start_date: datetime,
    end_date: datetime,
    split_method_dict: Optional[dict[str, str]] = None,
    relevant_operators: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Orchestrate the full pipeline from raw redispatch data to final target.

    This is the main entry point for preparing redispatch target data. It:
    1. Filters by measurement reasons and directing operator
    2. Applies TSO-specific hourly split methods
    3. Adds data-driven features (optional)
    4. Adds run-length features (optional)
    5. Fills missing intervals

    Parameters
    ----------
    redispatch : pd.DataFrame
        Raw redispatch data with all necessary columns.
    relevant_measurement_reasons : list[str]
        Measurement reasons to include in the target.
    min_gap_ratio : float
        Minimum allowed duration gap ratio for filtering.
    feature_set : list[FeatureSet]
        List of feature sets to compute. Options: "data_driven", "runlength".
    start_date : datetime
        Start of the target date range.
    end_date : datetime
        End of the target date range (inclusive hour).
    split_method_dict : dict[str, str], optional
        Mapping from operator to split method. Defaults to TARGET_SPLIT_METHOD_DICT.
    relevant_operators : list[str], optional
        List of operators to include. Defaults to all German TSOs.

    Returns
    -------
    pd.DataFrame
        Final target DataFrame with columns:
        - begin_date (datetime, UTC)
        - end_date (datetime, UTC)
        - operator (str)
        - direction (str: "up" or "down")
        - total_load (float, MWh)
        - primary_energy_type (str)
        - [engineered features based on feature_set]
    """
    if split_method_dict is None:
        split_method_dict = TARGET_SPLIT_METHOD_DICT.copy()
    if relevant_operators is None:
        relevant_operators = GERMAN_TSOS.copy()

    directing_operator_data = process_for_directing_operator(
        data=redispatch,
        relevant_measurement_reasons=relevant_measurement_reasons,
        min_gap_ratio=min_gap_ratio,
        relevant_operators=relevant_operators
    )

    final_data_list = []

    for operator in relevant_operators:
        split_method = split_method_dict.get(operator, "equal")
        operator_data = directing_operator_data[
            directing_operator_data["operator"] == operator
        ].copy(deep=False)

        if operator_data.empty:
            print(f"Warning: No data for operator {operator}")
            continue

        # Apply hourly split
        hourly_directing_operator_data = apply_hourly_split_method(
            operator_data, method=split_method
        )

        # Compute data-driven features if requested
        # They must be computed before aggregation to avoid 0-duration events
        if "data_driven" in feature_set:
            target_related_features = add_data_driven_features_hourly(
                data=hourly_directing_operator_data,
                original_data=operator_data,
                start_date=start_date,
                end_date=end_date,
                eps=0.3,
            )
        else:
            target_related_features = None

        # Add run-length features if requested
        # These also require pre-aggregation data, and filling missing intervals per direction
        if "runlength" in feature_set:
            runlength_features = add_runlength_features_hourly(
                df=hourly_directing_operator_data,
                original_data=operator_data,
                start_date=start_date,
                end_date=end_date,
                y_col="total_load",
                id_col="direction",
                time_col="begin_date",
                window_size=ROLLING_WINDOW_DATA_DRIVEN_HOURS,
            )
            if target_related_features is not None:
                target_related_features = runlength_features.merge(
                    target_related_features,
                    on=["begin_date", "direction"],
                    how="outer"
                )
            else:
                target_related_features = runlength_features

        # Aggregate and fill missing intervals
        hourly_directing_operator_data = prepare_final_target_data(
            hourly_directing_operator_data,
            start_date=start_date,
            end_date=end_date
        )[operator]

        # Merge data-driven features
        if target_related_features is not None:
            hourly_directing_operator_data = hourly_directing_operator_data.merge(
                target_related_features,
                on=["begin_date", "direction"],
                how="left"
            )
            hourly_directing_operator_data = impute_redispatch_core(
                hourly_directing_operator_data,
            )

        final_data_list.append(hourly_directing_operator_data)

    if not final_data_list:
        raise ValueError("No data produced for any operator")

    return pd.concat(final_data_list, axis=0, ignore_index=True)


def prepare_target_datasets(
    redispatch_data: pd.DataFrame,
    feature_set: list[FeatureSet],
    start_date: datetime,
    end_date: datetime,
    min_gap_ratio: float = -0.1,
    measurement_reasons_dict: Optional[dict[str, list[str]]] = None
) -> dict[str, pd.DataFrame]:
    """
    Prepare target datasets for multiple measurement reason categories.

    Parameters
    ----------
    redispatch_data : pd.DataFrame
        Raw redispatch data.
    feature_set : list[FeatureSet]
        Feature sets to compute.
    start_date : datetime
        Start of the target date range.
    end_date : datetime
        End of the target date range.
    min_gap_ratio : float
        Minimum allowed duration gap ratio.
    measurement_reasons_dict : dict[str, list[str]], optional
        Dictionary mapping reason labels to lists of measurement reasons.
        Defaults to TARGET_RELEVANT_MEASUREMENT_REASONS.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping reason labels to target DataFrames.
    """
    if measurement_reasons_dict is None:
        measurement_reasons_dict = TARGET_RELEVANT_MEASUREMENT_REASONS

    redispatch_datasets = {}

    for reason_label, relevant_reasons in measurement_reasons_dict.items():
        print(f"Preparing dataset for: {reason_label}")
        final_data = prepare_final_target(
            redispatch=redispatch_data,
            relevant_measurement_reasons=relevant_reasons,
            min_gap_ratio=min_gap_ratio,
            feature_set=feature_set,
            start_date=start_date,
            end_date=end_date,
        )
        redispatch_datasets[reason_label] = final_data

    return redispatch_datasets
