"""
Sparse Column Selection Utilities.

This module provides functions for selecting relevant columns from datasets
that contain many zero or near-zero values (e.g., cross-border flows,
scheduled exchanges).

The selection is based on three criteria:
1. Activity: Fraction of non-zero values must exceed a threshold.
2. Magnitude: Standard deviation of non-zero values must exceed a threshold.
3. Temporal: Number of contiguous non-zero runs must exceed a threshold.
"""

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _count_nonzero_runs(series: pd.Series) -> int:
    """
    Count the number of contiguous non-zero runs in a series.

    A "run" is a maximal contiguous block of non-zero (and non-NaN) values.

    Parameters
    ----------
    series : pd.Series
        Numeric series to analyse.

    Returns
    -------
    int
        Number of non-zero runs.
    """
    is_nonzero = (series != 0) & series.notna()
    # A new run starts wherever is_nonzero is True and the previous value was
    # False (or this is the first element).
    run_starts = is_nonzero & (~is_nonzero.shift(1, fill_value=False))
    return int(run_starts.sum())


def select_relevant_sparse_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    activity_threshold: float = 0.3,
    magnitude_threshold: float = 100.0,
    min_nonzero_runs: int = 1,
) -> list[str]:
    """
    Select columns that carry meaningful signal in a sparse dataset.

    A column is **kept** only when it passes all three tests:

    * **Activity** – ``mean(x != 0) >= activity_threshold``
      Ensures the column is "active" often enough.
    * **Magnitude** – ``std(x[x != 0]) >= magnitude_threshold``
      Ensures the non-zero values have enough variability (in MWh).
    * **Temporal** – ``#(non-zero runs) >= min_nonzero_runs``
      Ensures the signal is spread across time rather than concentrated in a
      single burst.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the candidate columns.
    columns : Sequence[str]
        Column names to evaluate.  Missing columns are silently skipped.
    activity_threshold : float, optional
        Minimum fraction of non-zero observations.  Default ``0.3``.
    magnitude_threshold : float, optional
        Minimum standard deviation of the non-zero observations (MWh).
        Default ``100.0``.
    min_nonzero_runs : int, optional
        Minimum number of contiguous non-zero runs.  Default ``1``.

    Returns
    -------
    list[str]
        Sorted list of column names that pass all three criteria.
    """
    kept: list[str] = []

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col]
        valid = series.dropna()

        if valid.empty:
            logger.debug(f"Column '{col}' is entirely NaN – dropped")
            continue

        # 1. Activity
        activity = (valid != 0).mean()
        if activity < activity_threshold:
            logger.debug(
                f"Column '{col}' failed activity check: "
                f"{activity:.2%} < {activity_threshold:.0%}"
            )
            continue

        # 2. Magnitude
        nonzero_vals = valid[valid != 0]
        if nonzero_vals.empty:
            logger.debug(f"Column '{col}' has no non-zero values – dropped")
            continue
        magnitude = nonzero_vals.std()
        if np.isnan(magnitude) or magnitude < magnitude_threshold:
            logger.debug(
                f"Column '{col}' failed magnitude check: "
                f"{magnitude:.1f} < {magnitude_threshold:.0f}"
            )
            continue

        # 3. Temporal spread
        n_runs = _count_nonzero_runs(valid)
        if n_runs < min_nonzero_runs:
            logger.debug(
                f"Column '{col}' failed temporal check: "
                f"{n_runs} runs < {min_nonzero_runs}"
            )
            continue

        kept.append(col)
        logger.debug(
            f"Column '{col}' passed: activity={activity:.2%}, "
            f"magnitude={magnitude:.1f}, runs={n_runs}"
        )

    logger.info(
        f"Sparse column selection: kept {len(kept)}/{len(columns)} columns "
        f"(act>={activity_threshold}, mag>={magnitude_threshold}, "
        f"runs>={min_nonzero_runs})"
    )
    return sorted(kept)
