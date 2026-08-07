"""
Dataset preparation for shifted-target training.

Handles:
- Loading parquet datasets + JSON metadata
- Classifying columns into future vs. historical covariates
- Adding calendar features (hour, day, month, is_weekend, is_holiday, is_workday)
- Shifting target + time-aligned future covariates by a configurable number of hours
- Recomputing calendar features for the shifted (physical) target time
- Preparing the Nixtla-compatible DataFrame (ds, y, unique_id, covariates)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Column name constants ──────────────────────────────────────────────────────
TARGET_COL = "total_load"
TIME_COL = "begin_date"
DIRECTION_COL = "direction"
OPERATOR_COL = "operator"
META_COLS = {TIME_COL, TARGET_COL, DIRECTION_COL, OPERATOR_COL}

CALENDAR_COLS = ["hour", "day", "month", "is_weekend", "is_holiday", "is_workday"]

# ── Holidays reader ────────────────────────────────────────────────────────────
_HOLIDAYS_PATH = Path(__file__).resolve().parent.parent / "data" / "holidays_new.csv"

# TSO name mapping: the holidays CSV uses "TenneT DE" (with space) as column
# header, but datasets may use "TenneT_DE" (underscore).  We normalise both.
_TSO_HOLIDAY_MAP = {
    "50Hertz": "50Hertz",
    "TenneT DE": "TenneT DE",
    "TenneT_DE": "TenneT DE",
    "Amprion": "Amprion",
    "TransnetBW": "TransnetBW",
}


def _read_holidays(tso: str, holidays_path: str | Path | None = None) -> pd.DataFrame:
    """Return a DataFrame indexed by *date* (normalised) with a single ``is_holiday`` column."""
    path = Path(holidays_path) if holidays_path else _HOLIDAYS_PATH
    holidays = pd.read_csv(path, parse_dates=["day"], date_format="%Y-%m-%d")
    holidays = holidays.drop(columns=["Bundesländer"]).rename(columns={"day": "Date"})
    # Normalise "TenneT DE " -> "TenneT DE" (trailing spaces in CSV headers)
    holidays.columns = holidays.columns.str.strip()
    holidays["Date"] = pd.to_datetime(holidays["Date"]).dt.normalize()

    col = _TSO_HOLIDAY_MAP.get(tso, tso)
    if col not in holidays.columns:
        raise ValueError(
            f"TSO '{tso}' (mapped to '{col}') not found in holidays file. "
            f"Available: {list(holidays.columns)}"
        )
    return holidays[["Date", col]].rename(columns={col: "is_holiday"}).set_index("Date")


# ── Covariate classification ──────────────────────────────────────────────────
def classify_covariates(
    columns: list[str],
) -> tuple[list[str], list[str]]:
    """
    Classify dataset columns into future (time-aligned) and historical covariates.

    Rules
    -----
    * **Future covariates** (will be shifted together with the target):
        - Column name contains ``"forecast"``  (production_forecast_*, consumption_forecast_*,
          sce_forecast_*, pv_forecast_*, wind_forecast_*, residual_load_forecast_*, ...)
        - Column name starts with ``"day_ahead_price"``  (known from D-1 auction)

    * **Historical covariates** (NOT shifted – only observed past values):
        - Everything else that is not a meta column or calendar column.
        - Includes ``*_lag_*``, ``*_rollingmean_*``, ``*_actual_*``,
          ``bloomberg_*``, ``runlength_*``, ``data_driven_*``.

    Returns (future_covariate_cols, historical_covariate_cols).
    """
    future: list[str] = []
    historical: list[str] = []

    skip = META_COLS | set(CALENDAR_COLS)

    for c in columns:
        if c in skip:
            continue
        if "forecast" in c or c.startswith("day_ahead_price"):
            future.append(c)
        else:
            historical.append(c)

    return future, historical


# ── Calendar features ──────────────────────────────────────────────────────────
def add_calendar_features(
    df: pd.DataFrame,
    reference_time: pd.Series,
    tso: str,
    holidays_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Compute calendar features based on *reference_time* (which may differ from ``ds``
    when we recompute after shifting).

    Adds / overwrites: ``hour``, ``day``, ``month``, ``is_weekend``, ``is_holiday``, ``is_workday``.
    """
    df = df.copy()
    df["hour"] = reference_time.dt.hour
    df["day"] = reference_time.dt.day
    df["month"] = reference_time.dt.month
    df["is_weekend"] = (reference_time.dt.weekday >= 5).astype(int)

    # Holidays
    holidays = _read_holidays(tso, holidays_path=holidays_path)
    ref_date = reference_time.dt.normalize()
    # Use a temporary column to avoid index-alignment issues
    df["_ref_date"] = ref_date.values
    hol_joined = df[["_ref_date"]].join(holidays, on="_ref_date")
    df["is_holiday"] = hol_joined["is_holiday"].fillna(0).astype(int).values
    df.drop(columns=["_ref_date"], inplace=True)

    df["is_workday"] = ((reference_time.dt.weekday < 5) & (df["is_holiday"] == 0)).astype(int)
    return df


# ── Dataset loading ────────────────────────────────────────────────────────────
def load_dataset(
    dataset_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a parquet dataset and its companion JSON metadata.

    Returns (df, metadata_dict).
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)

    # Ensure datetime
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    # Load companion metadata
    meta_path = dataset_path.with_suffix(".json")
    metadata: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        logger.warning("No metadata JSON found at %s", meta_path)

    return df, metadata


# ── Core: prepare shifted dataset ──────────────────────────────────────────────
def prepare_shifted_dataset(
    df: pd.DataFrame,
    shift_hours: int,
    tso: str,
    target_col: str = TARGET_COL,
    id_col: str = "unique_id",
    time_col: str = "ds",
    add_calendar: bool = True,
    holidays_path: str | Path | None = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Prepare a shifted dataset for Nixtla training.

    Steps
    -----
    1. Classify columns → future-covariates vs. historical-covariates.
    2. Shift ``y`` and future-covariates by ``-shift_hours`` (per ``unique_id`` group).
    3. Recompute calendar features for ``ds + shift_hours`` (the physical target time).
    4. Drop rows where shifted columns became NaN (tail of each group).

    Parameters
    ----------
    df : pd.DataFrame
        Nixtla-format DataFrame with at least ``ds``, ``y``, ``unique_id`` and feature columns.
    shift_hours : int
        How many hours ahead the target is shifted.  0 = no shift.
    tso : str
        TSO name, used for holiday lookup.
    add_calendar : bool
        Whether to compute calendar features.  If ``False``, existing calendar
        columns (if present) are kept untouched.
    holidays_path : str | Path | None
        Optional override for the holidays CSV path.
    dropna : bool
        Drop rows with NaN in shifted columns (default True).

    Returns
    -------
    (shifted_df, future_covariate_cols, historical_covariate_cols)
        - ``shifted_df`` is ready for ``NeuralForecast.fit()``.
        - ``future_covariate_cols`` includes calendar columns (if added).
        - ``historical_covariate_cols`` are the non-shifted feature columns.
    """
    df = df.sort_values([id_col, time_col]).copy()

    # ── 1. classify covariates ────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in {"y", id_col, time_col}]
    future_cov, hist_cov = classify_covariates(feature_cols)

    logger.info("Future covariates  (%d): %s", len(future_cov), future_cov)
    logger.info("Historical covariates (%d): %s", len(hist_cov), hist_cov)

    # ── 2. shift y + time-aligned future covariates ───────────────────────────
    if shift_hours > 0:
        g = df.groupby(id_col, sort=False)
        df["y"] = g["y"].shift(-shift_hours)
        for c in future_cov:
            df[c] = g[c].shift(-shift_hours)

    # ── 3. calendar features (based on physical target time) ──────────────────
    if add_calendar:
        # Remove any pre-existing calendar columns before recomputing
        existing_cal = [c for c in CALENDAR_COLS if c in df.columns]
        if existing_cal:
            df.drop(columns=existing_cal, inplace=True)

        physical_time = df[time_col] + pd.Timedelta(hours=shift_hours)
        df = add_calendar_features(df, reference_time=physical_time, tso=tso, holidays_path=holidays_path)

        # Calendar columns are future covariates (they describe the target time)
        future_cov = future_cov + CALENDAR_COLS
    else:
        # If calendar cols already exist in df, make sure they're in the right list
        for c in CALENDAR_COLS:
            if c in df.columns and c not in future_cov and c not in hist_cov:
                future_cov.append(c)

    # ── 4. drop NaN rows introduced by shifting ──────────────────────────────
    if dropna and shift_hours > 0:
        need = ["y"] + [c for c in future_cov if c in df.columns]
        df = df.dropna(subset=need)

    return df, future_cov, hist_cov


# ── Convert raw dataset → Nixtla format ──────────────────────────────────────
def to_nixtla_format(
    df: pd.DataFrame,
    direction: str = "both",
) -> pd.DataFrame:
    """
    Convert a raw dataset (with ``begin_date``, ``total_load``, ``direction``, ``operator``)
    into Nixtla convention (``ds``, ``y``, ``unique_id``).

    Parameters
    ----------
    direction : str
        ``"up"``, ``"down"`` or ``"both"``.  When ``"both"``, ``unique_id`` encodes the direction.
    """
    df = df.copy()

    if direction != "both":
        df = df.query(f"{DIRECTION_COL} == @direction").copy()

    # Fill missing hours with forward-fill (consistent with existing code)
    all_frames = []
    for uid in df[DIRECTION_COL].unique():
        sub = df[df[DIRECTION_COL] == uid].copy()
        full_idx = pd.date_range(sub[TIME_COL].min(), sub[TIME_COL].max(), freq="h", name=TIME_COL)
        sub = sub.set_index(TIME_COL).reindex(full_idx).ffill().reset_index()
        sub[DIRECTION_COL] = uid
        all_frames.append(sub)
    df = pd.concat(all_frames, ignore_index=True)

    # Rename to Nixtla convention
    rename_map = {TIME_COL: "ds", TARGET_COL: "y", DIRECTION_COL: "unique_id"}
    df = df.rename(columns=rename_map)

    # Drop the operator column (it's constant per dataset)
    if OPERATOR_COL in df.columns:
        df = df.drop(columns=[OPERATOR_COL])

    return df


def build_static_df() -> pd.DataFrame:
    """Return static exogenous DataFrame encoding direction as one-hot."""
    return pd.DataFrame({
        "unique_id": ["up", "down"],
        "direction_0": [1, 0],
        "direction_1": [0, 1],
    })


STATIC_EXOG_COLS = ["direction_0", "direction_1"]
