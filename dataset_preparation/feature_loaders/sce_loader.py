"""
Scheduled Commercial Exchange (SCE) Forecast Loader.

This module loads day-ahead forecasted scheduled commercial exchanges between
DE_LU and its neighbouring bidding zones.  The values are already day-ahead
forecasts, so they are available at prediction time and are included *as-is*
(no lagging or rolling features required).

Only ``DE_LU -> bidding_zone`` entries are kept.  Net flows are computed as:

    net = export(DE_LU → neighbour) − import(neighbour → DE_LU)

Positive net flow  →  net export from DE_LU.
Negative net flow  →  net import into DE_LU.

After computing net flows the sparse-column selector is applied to drop
borders that carry little signal.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .column_selection import select_relevant_sparse_columns

logger = logging.getLogger(__name__)

# Default CSV relative to project root
DEFAULT_SCE_PATH = Path(
    "data/scheduled_exchanges/entsoe_forecasted_scheduled_exchanges_extended.csv"
)

# Neighbours reachable from the DE_LU bidding zone
DE_LU_NEIGHBORS = ["DK_1", "DK_2", "PL", "CZ", "AT", "CH", "FR", "BE", "NL"]


def _normalize_neighbor_label(neighbor: str) -> str:
    """Lower-case a neighbour code for column naming."""
    return neighbor.lower()


def load_scheduled_exchanges(
    tso: str | None = None,
    *,
    sce_path: Path | str | None = None,
    activity_threshold: float = 0.3,
    magnitude_threshold: float = 100.0,
    min_nonzero_runs: int = 1,
    oos_start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load day-ahead scheduled commercial exchange forecasts (net flows).

    Parameters
    ----------
    tso : str or None, optional
        TSO name – currently unused because only DE_LU-level flows are
        returned, but accepted for interface consistency with other loaders.
    sce_path : Path or str or None, optional
        Explicit path to the CSV.  When *None* the default project-relative
        path is used.
    activity_threshold : float, optional
        Minimum fraction of non-zero values for sparse column selection.
        Default ``0.3``.
    magnitude_threshold : float, optional
        Minimum std of non-zero values (MWh) for sparse column selection.
        Default ``100.0``.
    min_nonzero_runs : int, optional
        Minimum number of contiguous non-zero runs for sparse column
        selection.  Default ``1``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ``begin_date`` : datetime
        - ``sce_forecast_net_flow_{neighbour}`` : one column per retained
          border (e.g. ``sce_forecast_net_flow_pl``).
    """
    # ------------------------------------------------------------------
    # Resolve file path
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent.parent
    if sce_path is None:
        sce_path = project_root / DEFAULT_SCE_PATH
    else:
        sce_path = Path(sce_path)

    if not sce_path.exists():
        logger.warning(f"Scheduled exchanges file not found: {sce_path}")
        return pd.DataFrame(columns=["begin_date"])

    # ------------------------------------------------------------------
    # Read CSV
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv(sce_path, parse_dates=["begin_date"])
    except Exception as e:
        logger.error(f"Error loading scheduled exchanges from {sce_path}: {e}")
        return pd.DataFrame(columns=["begin_date"])

    # Drop helper columns that are not needed
    drop_cols = [c for c in ("end_date", "Unnamed: 0") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # ------------------------------------------------------------------
    # Compute net flows for each DE_LU <-> neighbour pair
    # ------------------------------------------------------------------
    net_flow_cols: list[str] = []

    for neighbor in DE_LU_NEIGHBORS:
        neighbor_lower = _normalize_neighbor_label(neighbor)

        export_col = f"schedule_da_DE_LU_to_{neighbor}"   # DE_LU → neighbour
        import_col = f"schedule_da_{neighbor}_to_DE_LU"   # neighbour → DE_LU

        has_export = export_col in df.columns
        has_import = import_col in df.columns

        if not has_export and not has_import:
            logger.debug(f"No scheduled exchange columns for neighbour {neighbor}")
            continue

        net_col = f"sce_forecast_net_flow_{neighbor_lower}"

        if has_export and has_import:
            df[net_col] = df[export_col] - df[import_col]
        elif has_export:
            df[net_col] = df[export_col]
        else:
            df[net_col] = -df[import_col]

        net_flow_cols.append(net_col)

    if not net_flow_cols:
        logger.warning("No DE_LU scheduled exchange columns found")
        return pd.DataFrame(columns=["begin_date"])

    # ------------------------------------------------------------------
    # Sparse-column selection
    # ------------------------------------------------------------------
    kept_cols = select_relevant_sparse_columns(
        df,
        net_flow_cols,
        activity_threshold=activity_threshold,
        magnitude_threshold=magnitude_threshold,
        min_nonzero_runs=min_nonzero_runs,
    )

    result_df = df[["begin_date"] + kept_cols].copy()

    logger.info(
        f"Loaded scheduled exchange forecasts: "
        f"{len(result_df)} rows, {len(kept_cols)} net-flow columns "
        f"(from {len(net_flow_cols)} candidates)"
    )

    return result_df
