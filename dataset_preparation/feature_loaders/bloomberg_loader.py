"""
Bloomberg Financial Data Loader.

This module provides functions to load Bloomberg financial data from Excel files,
including commodity prices and computed clean spark/dark spreads.

Feature Reduction Strategy:
- Keep daily changes (24h lag) for key commodities
- Keep clean spark spread and clean dark spread
- Drop raw prices (except for computing spreads)
- Output: ~4-6 commodity/spread features
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from dataset_preparation.feature_loaders.imputation import impute_bloomberg

logger = logging.getLogger(__name__)

# Default path to Bloomberg data directory relative to project root
DEFAULT_BLOOMBERG_DIR = Path("data/bloomberg_updated")

# Bloomberg file mappings
# Maps a descriptive name to the expected filename
BLOOMBERG_FILES = {
    "gas": "THE_Gas_Futures_Active_Contract_redispatch.xlsx",
    "coal": "API2_Coal_Futures_redispatch.xlsx",
    "carbon": "EUA_Futures_Active_Contract_redispatch.xlsx",
    "power": "Germany_Monthly_Baseload_Financial_Future_Contracts_redispatch.xlsx",
    "peg": "PEGHANDA_Contracts_redispatch.xlsx",
}

# Efficiency and emission factors for spread calculations
GAS_EFFICIENCY = 0.55  # Combined cycle gas turbine efficiency
COAL_EFFICIENCY = 0.40  # Coal plant efficiency
GAS_EMISSIONS = 0.35   # tCO2/MWh for gas
COAL_EMISSIONS = 0.95  # tCO2/MWh for coal

# Transform from Euro/t to Euro/MWh for API2 coal price
API2_COAL_T_TO_MWH = 3.6 / 25.12  # 1 t of coal = 25.12 GJ, 1 MWh = 3.6 GJ

def _read_bloomberg_file(file_path: Path) -> tuple[pd.DataFrame, str | None]:
    """
    Read a Bloomberg Excel file and return the data with currency info.
    
    Parameters
    ----------
    file_path : Path
        Path to the Excel file.
        
    Returns
    -------
    tuple[pd.DataFrame, str | None]
        - DataFrame with Date and PX_LAST columns
        - Currency code (or None if not found)
    """
    try:
        # Read first few rows to get metadata
        metadata_df = pd.read_excel(file_path, header=None, nrows=6)
        
        # Check for currency (typically row 4, column 1)
        currency = None
        for i in range(len(metadata_df)):
            if metadata_df.iloc[i, 0] == "Currency":
                currency = str(metadata_df.iloc[i, 1])
                break
        
        # Find the header row (row with "Date")
        header_row = None
        for i in range(len(metadata_df)):
            if metadata_df.iloc[i, 0] == "Date":
                header_row = i
                break
        
        if header_row is None:
            # Default to row 6 (0-indexed)
            header_row = 6
        
        # Read the actual data
        df = pd.read_excel(file_path, header=header_row)
        
        # Keep only Date and PX_LAST columns
        if "Date" not in df.columns:
            logger.warning(f"Date column not found in {file_path}")
            return pd.DataFrame(), currency
        
        if "PX_LAST" not in df.columns:
            logger.warning(f"PX_LAST column not found in {file_path}")
            return pd.DataFrame(), currency
        
        result_df = df[["Date", "PX_LAST"]].copy()
        
        # Handle #N/A values
        na_count = result_df["PX_LAST"].isna().sum()
        if na_count > 0:
            # Also check for string "#N/A" values
            str_na_mask = result_df["PX_LAST"].astype(str).str.contains("#N/A", na=False)
            result_df.loc[str_na_mask, "PX_LAST"] = np.nan
            total_na = result_df["PX_LAST"].isna().sum()
            if total_na > 0:
                logger.info(f"Found {total_na} N/A values in {file_path.name}, will forward-fill")
        
        # Ensure Date is datetime
        result_df["Date"] = pd.to_datetime(result_df["Date"])
        
        # Sort by date ascending
        result_df = result_df.sort_values("Date").reset_index(drop=True)
        
        # Forward-fill NaN values
        result_df["PX_LAST"] = result_df["PX_LAST"].ffill()
        
        return result_df, currency
        
    except Exception as e:
        logger.error(f"Error reading Bloomberg file {file_path}: {e}")
        return pd.DataFrame(), None


def _expand_to_hourly(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Expand daily data to hourly by duplicating each day's value for 24 hours.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with daily data.
    date_col : str, optional
        Name of the date column. Default is 'Date'.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with hourly data (24 rows per day).
    """
    if df.empty:
        return df
    
    # Create hourly timestamps for each day
    hourly_rows = []
    for _, row in df.iterrows():
        date = row[date_col]
        for hour in range(24):
            new_row = row.copy()
            new_row[date_col] = date.replace(hour=hour)
            hourly_rows.append(new_row)
    
    return pd.DataFrame(hourly_rows).reset_index(drop=True)


def load_bloomberg_data(
    data_dir: Path | str | None = None,
    prices_path: Path | str | None = None,
    oos_start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load Bloomberg financial data from Excel files.
    
    Feature Reduction Strategy:
    - Keep clean spark spread and clean dark spread (most relevant for merit order)
    - Add 24h changes (daily lags) for key commodities: gas, carbon
    - Drop raw prices to reduce dimensionality
    - Output: ~4-6 bloomberg features
    
    Parameters
    ----------
    data_dir : Path | str | None, optional
        Path to the Bloomberg data directory. If None, uses the default path.
    prices_path : Path | str | None, optional
        Path to the day-ahead prices CSV for power price in spread calculations.
        If None, uses the default prices path.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - begin_date: datetime column (not index), hourly frequency
        - bloomberg_clean_spark_spread: Clean spark spread (gas vs power)
        - bloomberg_clean_dark_spread: Clean dark spread (coal vs power)
        - bloomberg_gas_change_24h: 24h change in gas price
        - bloomberg_carbon_change_24h: 24h change in carbon price
        
    Notes
    -----
    - Prices are shifted backwards by 1 day (T-1) as they are end-of-day prices
    - Data is expanded from daily to hourly frequency
    - Clean spark spread = power_price - (gas_price / gas_efficiency) - carbon_price * gas_emissions
    - Clean dark spread = power_price - (coal_price / coal_efficiency) - carbon_price * coal_emissions
    - 24h changes capture daily price momentum
    """
    # Determine directory path
    if data_dir is None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        data_dir = project_root / DEFAULT_BLOOMBERG_DIR
    else:
        data_dir = Path(data_dir)
    
    # Check if directory exists
    if not data_dir.exists():
        logger.warning(f"Bloomberg data directory not found: {data_dir}")
        return pd.DataFrame(columns=["begin_date"])
    
    # Load each Bloomberg file
    price_data = {}
    currencies = {}
    
    for name, filename in BLOOMBERG_FILES.items():
        file_path = data_dir / filename
        if file_path.exists():
            df, currency = _read_bloomberg_file(file_path)
            if not df.empty:
                price_data[name] = df
                currencies[name] = currency
                if currency and currency != "EUR":
                    logger.warning(f"Bloomberg file {filename} has currency {currency}, expected EUR")
        else:
            logger.warning(f"Bloomberg file not found: {file_path}")
    
    if not price_data:
        logger.warning("No Bloomberg data files loaded")
        return pd.DataFrame(columns=["begin_date"])
    
    # Merge all price data on Date
    # Start with the first available dataset
    merged_df = None
    for name, df in price_data.items():
        df_renamed = df.rename(columns={"PX_LAST": f"bloomberg_{name}_last_price"})
        if merged_df is None:
            merged_df = df_renamed
        else:
            merged_df = pd.merge(merged_df, df_renamed, on="Date", how="outer")
    
    if merged_df is None or merged_df.empty:
        logger.warning("No Bloomberg data after merging")
        return pd.DataFrame(columns=["begin_date"])

    merged_df = impute_bloomberg(merged_df)
    
    # Sort by date
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)
    
    # -------------------------------------------------------------------------
    # Compute 24h changes (daily lags) BEFORE time-shifting
    # -------------------------------------------------------------------------
    gas_col = "bloomberg_gas_last_price"
    carbon_col = "bloomberg_carbon_last_price"
    
    if gas_col in merged_df.columns:
        merged_df["bloomberg_gas_change_24h"] = merged_df[gas_col].diff(1)
    
    if carbon_col in merged_df.columns:
        merged_df["bloomberg_carbon_change_24h"] = merged_df[carbon_col].diff(1)
    
    # Shift backwards by 1 day (price at day D is available at end of day D,
    # so we associate it with day D+1 to avoid look-ahead bias)
    merged_df["Date"] = merged_df["Date"] + pd.Timedelta(days=1)
    
    # Expand to hourly
    merged_df = _expand_to_hourly(merged_df, "Date")
    
    # Rename Date to begin_date
    merged_df = merged_df.rename(columns={"Date": "begin_date"})
    
    # -------------------------------------------------------------------------
    # Compute clean spark and dark spreads
    # -------------------------------------------------------------------------
    coal_col = "bloomberg_coal_last_price"
    power_col = "bloomberg_power_last_price"
    
    result_df = merged_df[["begin_date"]].copy()
    
    # Clean Spark Spread (gas):
    # power_price - (gas_price / gas_efficiency) - carbon_price * gas_emissions
    if all(col in merged_df.columns for col in [power_col, gas_col, carbon_col]):
        result_df["bloomberg_clean_spark_spread"] = (
            merged_df[power_col] 
            - (merged_df[gas_col] / GAS_EFFICIENCY) 
            - (merged_df[carbon_col] * GAS_EMISSIONS / GAS_EFFICIENCY)
        )
    else:
        logger.warning("Missing columns for clean spark spread calculation")
        result_df["bloomberg_clean_spark_spread"] = np.nan
    
    # Clean Dark Spread (coal):
    # power_price - (coal_price_converted / coal_efficiency) - carbon_price * coal_emissions
    if all(col in merged_df.columns for col in [power_col, coal_col, carbon_col]):
        result_df["bloomberg_clean_dark_spread"] = (
            merged_df[power_col] 
            - ((merged_df[coal_col]) * API2_COAL_T_TO_MWH / COAL_EFFICIENCY) 
            - (merged_df[carbon_col] * COAL_EMISSIONS / COAL_EFFICIENCY)
        )
    else:
        logger.warning("Missing columns for clean dark spread calculation")
        result_df["bloomberg_clean_dark_spread"] = np.nan
    
    # Add 24h changes
    if "bloomberg_gas_change_24h" in merged_df.columns:
        result_df["bloomberg_gas_change_24h"] = merged_df["bloomberg_gas_change_24h"]
    
    if "bloomberg_carbon_change_24h" in merged_df.columns:
        result_df["bloomberg_carbon_change_24h"] = merged_df["bloomberg_carbon_change_24h"]
    
    logger.info(
        f"Loaded Bloomberg data: "
        f"{len(result_df)} rows, {len(result_df.columns) - 1} columns"
    )

    return result_df