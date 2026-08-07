"""
Day-Ahead Prices Loader.

This module provides functions to load day-ahead electricity prices
for a TSO's neighboring countries/bidding zones.

Feature Reduction Strategy:
- Keep DE price (day_ahead_price_de_lu)
- Compute DE vs neighbor SPREADS instead of raw neighbor prices
- Output: DE price + ~4-6 spread features
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from dataset_preparation.feature_loaders.imputation import impute_day_ahead_prices

from ..tso_config import get_neighbors, normalize_tso_name

logger = logging.getLogger(__name__)

# Default path to prices data relative to project root
DEFAULT_PRICES_PATH = Path("data/day_ahead_electricity_prices/entsoe_day_ahead_prices_full_no_imputation.csv")

# Key neighbors for price spreads (most relevant for cross-border impact)
KEY_PRICE_NEIGHBORS = ["PL", "CZ", "DK_1", "DK_2", "NL", "AT", "CH", "FR"]


def load_day_ahead_prices(
    tso: str,
    data_path: Path | str | None = None,
    oos_start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load day-ahead electricity prices with DE-neighbor spreads.
    
    Feature Reduction Strategy:
    - Keep DE price (day_ahead_price_de_lu)
    - Compute spreads: DE - neighbor (positive = DE more expensive)
    - Drop raw neighbor prices to reduce dimensionality
    
    Parameters
    ----------
    tso : str
        The TSO name (e.g., "50Hertz", "TenneT DE", "Amprion", "TransnetBW").
        Will be normalized internally.
    data_path : Path | str | None, optional
        Path to the prices CSV file. If None, uses the default path.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - begin_date: datetime column (not index)
        - day_ahead_price_de_lu: German/Luxembourg price
        - day_ahead_price_spread_de_vs_{neighbor}: DE - neighbor spread
          (positive = DE more expensive, incentive to export)
        
    Notes
    -----
    - Spreads indicate price differentials driving cross-border flows
    - Positive spread = DE more expensive than neighbor = incentive to import
    - Negative spread = DE cheaper = incentive to export
    - Returns an empty DataFrame with only 'begin_date' column if file not found
    """
    # Determine file path
    if data_path is None:
        # Try to find the project root by looking for the data directory
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        data_path = project_root / DEFAULT_PRICES_PATH
    else:
        data_path = Path(data_path)
    
    # Check if file exists
    if not data_path.exists():
        logger.warning(f"Day-ahead prices file not found: {data_path}")
        return pd.DataFrame(columns=["begin_date"])
    
    # Normalize TSO name and get neighbors
    tso_normalized = normalize_tso_name(tso)
    neighbors = get_neighbors(tso_normalized)
    
    if not neighbors:
        logger.warning(f"No neighbors found for TSO: {tso} (normalized: {tso_normalized})")
        return pd.DataFrame(columns=["begin_date"])
    
    # Load the data
    try:
        df = pd.read_csv(data_path, parse_dates=["begin_date"])
    except Exception as e:
        logger.error(f"Error loading day-ahead prices from {data_path}: {e}")
        return pd.DataFrame(columns=["begin_date"])
    
    seasonal_fit_mask = None
    if oos_start_date is not None:
        oos_ts = pd.to_datetime(oos_start_date)
        seasonal_fit_mask = df["begin_date"] < oos_ts

    df = impute_day_ahead_prices(df, seasonal_fit_mask=seasonal_fit_mask)
    
    # Start with begin_date
    result_df = df[["begin_date"]].copy()
    
    # Always include German price (DE_LU)
    de_lu_col = "price_DE_LU"
    if de_lu_col in df.columns:
        result_df["day_ahead_price_de_lu"] = df[de_lu_col]
    else:
        logger.warning("German price column not found: price_DE_LU")
        return result_df
    
    # Compute spreads for neighbors (filter to key neighbors)
    neighbors_to_process = []
    for neighbor in neighbors:
        if neighbor in ["DK_1", "DK_2"]:
            neighbors_to_process.append(neighbor)
        elif neighbor == "DK":
            neighbors_to_process.extend(["DK_1", "DK_2"])
        else:
            neighbors_to_process.append(neighbor)
    
    neighbors_to_process = list(set(neighbors_to_process))
    neighbors_to_process = [n for n in neighbors_to_process if n in KEY_PRICE_NEIGHBORS]
    
    for neighbor in neighbors_to_process:
        price_col = f"price_{neighbor}"
        if price_col not in df.columns:
            logger.debug(f"Price column not found for neighbor {neighbor}: {price_col}")
            continue
        
        # Compute spread: DE - neighbor
        neighbor_lower = neighbor.lower()
        spread_col = f"day_ahead_price_spread_de_vs_{neighbor_lower}"
        result_df[spread_col] = df[de_lu_col] - df[price_col]
    
    # Ensure begin_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df["begin_date"]):
        result_df["begin_date"] = pd.to_datetime(result_df["begin_date"])
    
    logger.info(
        f"Loaded day-ahead prices for {tso_normalized}: "
        f"{len(result_df)} rows, {len(result_df.columns) - 1} price/spread columns"
    )
    
    return result_df