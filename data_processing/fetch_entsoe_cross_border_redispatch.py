# %% [markdown]
# # ENTSO-E Transparency Platform Data Fetching
# 
# This notebook fetches data from the ENTSO-E Transparency Platform API for:
# - Day-ahead electricity prices
# - Production and consumption data (actual load, forecast load, generation)
# - Cross-border flows
# - Wind and PV generation forecasts and actuals
# 
# All data is converted to UTC and resampled to hourly intervals.

# %%
import os
import io
import time
import random
import requests
import gzip
import zipfile
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Literal, Union
from dotenv import load_dotenv

# %% [markdown]
# ## Configuration

# %%
# Load API key from .env file
load_dotenv()
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

if not ENTSOE_API_KEY:
    raise ValueError("ENTSOE_API_KEY not found in .env file. Please add it.")

# Base URL for ENTSO-E API
ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"

# Date range for fetching data
START_DATE = datetime(2021, 10, 1)
END_DATE = datetime(2026, 1, 5)

# Output directories
DATA_DIR = Path("../data")
DAY_AHEAD_PRICES_DIR = DATA_DIR / "day_ahead_electricity_prices"
PRODUCTION_DIR = DATA_DIR / "production_new"
CONSUMPTION_DIR = DATA_DIR / "consumption_new"
WIND_PV_DIR = DATA_DIR / "wind_pv_generation"

# Create directories if they don't exist
for dir_path in [DAY_AHEAD_PRICES_DIR, PRODUCTION_DIR, CONSUMPTION_DIR, WIND_PV_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## EIC Codes and Constants

# %%
# Bidding zone / control area EIC codes
BIDDING_ZONES = {
    "DE_LU": "10Y1001A1001A82H",       # Germany-Luxembourg
    "DK_1": "10YDK-1--------W",         # Denmark DK1
    "DK_2": "10YDK-2--------M",         # Denmark DK2
    "DK": "10Y1001A1001A796",           # Denmark (entire)
    "PL": "10YPL-AREA-----S",           # Poland
    "CZ": "10YCZ-CEPS-----N",           # Czech Republic
    "AT": "10YAT-APG------L",           # Austria
    "CH": "10YCH-SWISSGRIDZ",           # Switzerland
    "FR": "10YFR-RTE------C",           # France
    "BE": "10YBE----------2",           # Belgium
    "NL": "10YNL----------L",           # Netherlands
}

# German TSO control area EIC codes
GERMAN_TSO_AREAS = {
    "DE_TenneT": "10YDE-EON------1",     # TenneT
    "DE_50HzT": "10YDE-VE-------2",      # 50Hertz
    "DE_Amprion": "10YDE-RWENET---I",    # Amprion
    "DE_TransnetBW": "10YDE-ENBW-----N", # TransnetBW
}

# PSR Type codes for production types
PSR_TYPES = {
    "B01": "Biomass",
    "B02": "Fossil Brown coal/Lignite",
    "B03": "Fossil Coal-derived gas",
    "B04": "Fossil Gas",
    "B05": "Fossil Hard coal",
    "B06": "Fossil Oil",
    "B07": "Fossil Oil shale",
    "B08": "Fossil Peat",
    "B09": "Geothermal",
    "B10": "Hydro Pumped Storage",
    "B11": "Hydro Run-of-river and poundage",
    "B12": "Hydro Water Reservoir",
    "B13": "Marine",
    "B14": "Nuclear",
    "B15": "Other renewable",
    "B16": "Solar",
    "B17": "Waste",
    "B18": "Wind Offshore",
    "B19": "Wind Onshore",
    "B20": "Other",
    "B25": "Energy storage",
}

# Document types
DOCUMENT_TYPES = {
    "day_ahead_prices": "A44",          # Price document
    "actual_load": "A65",               # System total load
    "load_forecast": "A65",             # System total load (with different process type)
    "generation_forecast": "A71",       # Generation forecast
    "wind_solar_forecast": "A69",       # Wind and solar forecast
    "actual_generation": "A75",         # Actual generation per type
    "installed_capacity": "A68",        # Installed generation capacity aggregated
    "cross_border_flows": "A11",        # Aggregated energy data report (scheduled exchanges)
    "outages_prod": "A77",              # Unavailability of generation units
    "outages_gen": "A80",               # Planned unavailability of generation units   
}

DOCUMENT_STATUS = {
    "Active": "A05",
    "Cancelled" : "A09",
    "Withdrawn": "A13",
}

BUSINESS_TYPES = {
    "Planned maintenance": "A53",
    "Forced outage": "A54",
}

# Process types
PROCESS_TYPES = {
    "day_ahead": "A01",
    "intraday": "A40",
    "realised": "A16",
    "current": "A18",
    "year_ahead": "A33",
}

# %% [markdown]
# ## Base ENTSO-E API Client

# %%
class ENTSOEClient:
    """
    Base client for ENTSO-E Transparency Platform API.
    Handles authentication, rate limiting, date chunking, and XML parsing.
    """
    
    def __init__(self, api_key: str, base_url: str = ENTSOE_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        
    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime to ENTSO-E API format (yyyyMMddHHmm)."""
        return dt.strftime("%Y%m%d%H%M")
    
    def _random_sleep(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """Sleep for a random duration to respect rate limits."""
        sleep_time = random.uniform(min_seconds, max_seconds)
        time.sleep(sleep_time)
    
    def _chunk_date_range(self, start: datetime, end: datetime, max_days: int = 365) -> List[tuple]:
        """
        Split date range into chunks (max 1 year for ENTSO-E API).
        Returns list of (start, end) datetime tuples.
        """
        chunks = []
        current_start = start
        
        while current_start < end:
            current_end = min(current_start + timedelta(days=max_days), end)
            chunks.append((current_start, current_end))
            current_start = current_end
            
        return chunks
    
    def _decode_and_parse(self, content: bytes, encoding: str = "utf-8"):
        head = content[:4]

        if head.startswith(b"PK\x03\x04"):  # ZIP
            z = zipfile.ZipFile(io.BytesIO(content))
            zipped_data = []
            for name in z.namelist():
                if name.lower().endswith(".xml"):
                    zipped_data.append(z.read(name).decode(encoding))
            return zipped_data

        if head.startswith(b"PK\x05\x06"):  # Empty ZIP
            return []

        if content[:2] == b"\x1f\x8b":  # GZIP
            xml_bytes = gzip.decompress(content).decode(encoding)
            return xml_bytes

        # Otherwise assume plain XML (possibly with leading whitespace/BOM)
        xml_bytes = content.lstrip()
        if xml_bytes.startswith(b"<"):
            return xml_bytes.decode(encoding)
        with open("inspect.zip", "wb") as f:
            f.write(content)
        raise ValueError(f"Unknown payload. First bytes: {head!r}")
    
    def _make_request(self, params: dict) -> Optional[str | List[str]]:
        """
        Make a request to the ENTSO-E API with error handling and rate limiting.
        Returns XML response as string or None on failure.
        """
        params["securityToken"] = self.api_key
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=60)
            
            # Handle common HTTP errors
            if response.status_code == 200:
                # Check for API error in XML response
                if "<Reason>" in response.text and "No matching data found" in response.text:
                    return None
                return self._decode_and_parse(response.content)
            elif response.status_code == 400:
                print(f"Bad Request (400): Invalid parameters - {params}")
                return None
            elif response.status_code == 401:
                raise ValueError("Unauthorized (401): Invalid API key")
            elif response.status_code == 403:
                print(f"Forbidden (403): Access denied for this data")
                return None
            elif response.status_code == 404:
                print(f"Not Found (404): No data available for parameters - {params}")
                return None
            elif response.status_code == 409:
                print(f"Conflict (409): Request limit exceeded, waiting...")
                time.sleep(60)  # Wait 1 minute before retry
                return self._make_request(params)
            elif response.status_code == 429:
                print(f"Too Many Requests (429): Rate limit exceeded, waiting...")
                time.sleep(60)
                return self._make_request(params)
            else:
                print(f"HTTP Error {response.status_code} for {params}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("Request timed out, retrying...")
            time.sleep(5)
            return self._make_request(params)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

# %%

def _resample_to_hourly(df: pd.DataFrame, agg_method: str = 'sum') -> pd.DataFrame:
    """
    Resample DataFrame from 15-min to hourly resolution, handling mixed resolutions per column.
    
    This function intelligently handles cases where:
    - Some columns are already hourly and should remain unchanged
    - Some columns are 15-min for certain date ranges and need aggregation
    - Different columns may have different resolution patterns
    
    Parameters:
    - df: DataFrame with begin_date, end_date, and value columns
    - agg_method: Aggregation method ('sum', 'mean', 'max', 'min')
    
    Returns:
    DataFrame with hourly resolution
    """
    if df.empty:
        return df
    
    # Calculate time resolution for each row (in minutes)
    df = df.copy()
    df['_resolution_minutes'] = (df['end_date'] - df['begin_date']).dt.total_seconds() / 60
    
    # Check if all data is already hourly
    if (df['_resolution_minutes'] >= 60).all():
        return df.drop(columns=['_resolution_minutes'])
    
    # Check if all data is sub-hourly
    if (df['_resolution_minutes'] < 60).all():
        # Simple case: resample everything
        df_indexed = df.set_index('begin_date')
        numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != '_resolution_minutes']
        
        if agg_method == 'sum':
            resampled = df_indexed[numeric_cols].resample('h').sum()
        elif agg_method == 'mean':
            resampled = df_indexed[numeric_cols].resample('h').mean()
        elif agg_method == 'max':
            resampled = df_indexed[numeric_cols].resample('h').max()
        elif agg_method == 'min':
            resampled = df_indexed[numeric_cols].resample('h').min()
        else:
            resampled = df_indexed[numeric_cols].resample('h').mean()
        
        resampled = resampled.reset_index()
        resampled['end_date'] = resampled['begin_date'] + pd.Timedelta(hours=1)
        
        cols = ['begin_date', 'end_date'] + numeric_cols
        return resampled[cols]
    
    # Mixed resolution case: handle column by column
    print(f"  Detected mixed resolution data (15min and hourly)")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != '_resolution_minutes']
    
    # Create hourly index covering full date range
    hourly_index = pd.date_range(
        df['begin_date'].min().floor('h'),
        df['begin_date'].max().ceil('h'),
        freq='h'
    )
    
    result_df = pd.DataFrame({'begin_date': hourly_index[:-1]})
    result_df['end_date'] = result_df['begin_date'] + pd.Timedelta(hours=1)
    
    # Process each numeric column separately
    for col in numeric_cols:
        # Create a temporary dataframe with just this column
        temp_df = df[['begin_date', 'end_date', '_resolution_minutes', col]].copy()
        temp_df = temp_df.dropna(subset=[col])
        
        if temp_df.empty:
            result_df[col] = np.nan
            continue
        
        # Check resolution for this specific column
        col_resolutions = temp_df.groupby('_resolution_minutes').size()
        
        # If column is purely hourly, just merge it
        if len(col_resolutions) == 1 and col_resolutions.index[0] >= 60:
            hourly_data = temp_df[['begin_date', col]].copy()
            result_df = pd.merge(result_df, hourly_data, on='begin_date', how='left')
        
        # If column has any sub-hourly data, resample it
        elif (temp_df['_resolution_minutes'] < 60).any():
            # Separate hourly and sub-hourly data
            hourly_mask = temp_df['_resolution_minutes'] >= 60
            subhourly_mask = temp_df['_resolution_minutes'] < 60
            
            # Start with hourly data if it exists
            if hourly_mask.any():
                hourly_data = temp_df[hourly_mask][['begin_date', col]].copy()
            else:
                hourly_data = pd.DataFrame(columns=['begin_date', col])
            
            # Resample sub-hourly data if it exists
            if subhourly_mask.any():
                subhourly_data = temp_df[subhourly_mask].copy()
                subhourly_data = subhourly_data.set_index('begin_date')
                
                # Apply aggregation
                if agg_method == 'sum':
                    resampled = subhourly_data[[col]].resample('h').sum()
                elif agg_method == 'mean':
                    resampled = subhourly_data[[col]].resample('h').mean()
                elif agg_method == 'max':
                    resampled = subhourly_data[[col]].resample('h').max()
                elif agg_method == 'min':
                    resampled = subhourly_data[[col]].resample('h').min()
                else:
                    resampled = subhourly_data[[col]].resample('h').mean()
                
                resampled = resampled.reset_index()
                
                # If all hourly entries are missing, they have already been removed by dropna before
                if hourly_data.empty:
                    combined = resampled
                else:
                    # Combine hourly and resampled data, preferring hourly where both exist
                    combined = pd.concat([hourly_data, resampled], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['begin_date'], keep='first')
            else:
                combined = hourly_data
            
            # Merge into result
            result_df = pd.merge(result_df, combined, on='begin_date', how='left')
        
        else:
            # Shouldn't reach here, but handle gracefully
            result_df = pd.merge(result_df, temp_df[['begin_date', col]], on='begin_date', how='left')
    
    return result_df

# %% [markdown]
# ## Data Validation and Export

# %%
def validate_data(df: pd.DataFrame, name: str, min_completeness: float = 0.99) -> bool:
    """
    Validate data quality.
    
    Parameters:
    - df: DataFrame to validate
    - name: Name for logging
    - min_completeness: Minimum required data completeness (0-1)
    
    Returns:
    True if validation passes, False otherwise
    """
    print(f"\n=== Validating {name} ===")
    
    if df.empty:
        print(f"WARNING: {name} is empty!")
        return False
    
    # Check for duplicates
    date_cols = ['begin_date', 'end_date']
    duplicates = df.duplicated(subset=date_cols).sum()
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate rows found")
    
    # Check date range
    print(f"Date range: {df['begin_date'].min()} to {df['begin_date'].max()}")
    
    # Check completeness
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        completeness = 1 - df[col].isna().mean()
        status = "✓" if completeness >= min_completeness else "✗"
        print(f"  {col}: {completeness:.2%} complete {status}")
    
    # Check for gaps in timestamps
    expected_hours = len(pd.date_range(df['begin_date'].min(), df['begin_date'].max(), freq='h'))
    actual_hours = len(df)
    coverage = actual_hours / expected_hours if expected_hours > 0 else 0
    print(f"Temporal coverage: {coverage:.2%} ({actual_hours}/{expected_hours} hours)")
    
    return coverage >= min_completeness


def make_hourly_intervals(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Ensure DataFrame has complete hourly intervals with no gaps.
    Missing values will be NaN.
    
    Parameters:
    - df: Input DataFrame with begin_date, end_date columns
    - start_date: Start of desired range
    - end_date: End of desired range
    
    Returns:
    DataFrame with complete hourly intervals
    """
    if df.empty:
        # Create empty DataFrame with correct structure
        full_index = pd.date_range(start_date, end_date, freq='h')[:-1]
        return pd.DataFrame({
            'begin_date': full_index,
            'end_date': full_index + pd.Timedelta(hours=1)
        })
    
    # Create complete hourly index
    full_index = pd.date_range(start_date, end_date, freq='h')[:-1]
    full_df = pd.DataFrame({
        'begin_date': full_index,
        'end_date': full_index + pd.Timedelta(hours=1)
    })
    
    # Merge with existing data
    result = pd.merge(full_df, df, on=['begin_date', 'end_date'], how='left')
    
    return result


def save_data(df: pd.DataFrame, filepath: Path, name: str):
    """
    Save DataFrame to CSV with validation.
    """
    if df.empty:
        print(f"WARNING: Not saving {name} - DataFrame is empty")
        return
    
    df.to_csv(filepath, index=False)
    print(f"Saved {name} to {filepath} ({len(df)} rows)")

# %% [markdown]
# ## Main Execution

# %%
# Initialize client
client = ENTSOEClient(ENTSOE_API_KEY)
print(f"ENTSO-E client initialized")
print(f"Fetching data from {START_DATE} to {END_DATE}")



# %% [markdown]
# ## Cross-Border Redispatching Data Fetching (13.1.A)
# 
# Fetch Cross-Border Redispatching data from ENTSO-E Transparency Platform.
# This includes redispatch measures between different control areas/bidding zones.

# %%

REDISPATCH_DIR = DATA_DIR / "redispatch_cross_border"
REDISPATCH_DIR.mkdir(parents=True, exist_ok=True)

# Market participant role codes mapping
MARKET_PARTICIPANT_ROLES = {
    "A32": "Market Operator",
    "A33": "Metered Data Responsible",
    "A39": "Data Provider",
    "A44": "Balance Responsible Party",
    "A45": "Metering Point Administrator",
    "A46": "System Operator",
    "A49": "Balance Service Provider",
}

# Additional EIC codes for market participants (ENTSO-E)
MARKET_PARTICIPANTS = {
    "10X1001A1001A450": "ENTSO-E",
}

# Combine all known EIC codes for reverse lookup
ALL_EIC_CODES = {
    **{v: k for k, v in BIDDING_ZONES.items()},
    **{v: k for k, v in GERMAN_TSO_AREAS.items()},
    **MARKET_PARTICIPANTS,
}

FLOW_DIRECTION_MAPPING = {
    "A01": "up",
    "A02": "down",
}

PSR_TYPE_MAPPING = {
    "A04": "generation",
    "A05": "load",
}

REDISPATCH_REASON_CODES = {
    "B24": "Load flow overflow",
    "B25": "Voltage level adjustment",
    "A95": "Complementary information",
}

def _check_max_instances_error(response_text: str) -> bool:
    """
    Check if the response contains a max instances exceeded error (code 999).
    Returns True if error detected, False otherwise.
    """
    if not response_text:
        return False
    
    if "<Reason>" in response_text and "<code>999</code>" in response_text:
        if "exceeds the allowed maximum" in response_text:
            return True
    return False

def _parse_max_instances_error(response_text: str) -> tuple[int, int]:
    """
    Parse the number of instances and max allowed from error message.
    Returns (num_instances, max_allowed) or (None, None) if parsing fails.
    """
    import re
    match = re.search(r'The number of instances \((\d+)\) exceeds the allowed maximum \((\d+)\)', response_text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def parse_cross_border_redispatch_xml_to_dataframe(xml_text: str) -> pd.DataFrame:
    """
    Parse ENTSO-E Cross-Border Redispatching XML response to DataFrame.
    
    Similar to internal redispatch but includes additional fields:
    - sender_mrid / sender_mrid_name: Sender market participant
    - sender_role / sender_role_name: Sender market role
    - receiver_mrid / receiver_mrid_name: Receiver market participant  
    - receiver_role / receiver_role_name: Receiver market role
    
    Returns DataFrame with columns for cross-border redispatch data.
    """
    if not xml_text:
        return pd.DataFrame()
    
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        return pd.DataFrame()
    
    # Detect namespace
    root_ns = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
    ns = {'ns': root_ns} if root_ns else {}
    
    def find_elem(parent, tag):
        """Find element with or without namespace."""
        if ns:
            elem = parent.find(f'.//{{{ns["ns"]}}}{tag}')
            if elem is None:
                elem = parent.find(f'.//{tag}')
        else:
            elem = parent.find(f'.//{tag}')
        return elem
    
    def find_all_elem(parent, tag):
        """Find all elements with or without namespace."""
        if ns:
            elems = parent.findall(f'.//{{{ns["ns"]}}}{tag}')
            if not elems:
                elems = parent.findall(f'.//{tag}')
        else:
            elems = parent.findall(f'.//{tag}')
        return elems
    
    def get_text(parent, tag, default=None):
        """Get text content of element."""
        elem = find_elem(parent, tag)
        return elem.text if elem is not None else default
    
    records = []
    
    # Get document-level metadata
    doc_mrid = get_text(root, 'mRID')
    doc_revision_number = get_text(root, 'revisionNumber', '1')
    
    # Get market participant information (document-level)
    sender_mrid = get_text(root, 'sender_MarketParticipant.mRID')
    sender_role = get_text(root, 'sender_MarketParticipant.marketRole.type')
    receiver_mrid = get_text(root, 'receiver_MarketParticipant.mRID')
    receiver_role = get_text(root, 'receiver_MarketParticipant.marketRole.type')
    
    # Map market participant mRIDs to names
    sender_mrid_name = ALL_EIC_CODES.get(sender_mrid, sender_mrid)
    receiver_mrid_name = ALL_EIC_CODES.get(receiver_mrid, receiver_mrid)
    
    # Map market participant roles to names
    sender_role_name = MARKET_PARTICIPANT_ROLES.get(sender_role, sender_role)
    receiver_role_name = MARKET_PARTICIPANT_ROLES.get(receiver_role, receiver_role)
    
    # Find all TimeSeries elements
    time_series_list = find_all_elem(root, 'TimeSeries')
    
    for ts in time_series_list:
        # Extract metadata
        ts_mrid_raw = get_text(ts, 'mRID')
        ts_mrid = f"{doc_mrid}_{ts_mrid_raw}" if doc_mrid and ts_mrid_raw else ts_mrid_raw
        business_type = get_text(ts, 'businessType')  # Should be A46 for cross-border
        
        # Get domain info
        in_domain = get_text(ts, 'in_Domain.mRID')
        out_domain = get_text(ts, 'out_Domain.mRID')
        
        # Map domains to names
        in_domain_name = ALL_EIC_CODES.get(in_domain, in_domain)
        out_domain_name = ALL_EIC_CODES.get(out_domain, out_domain)
        
        # Get PSR type (A04=generation, A05=load)
        psr_type = get_text(ts, 'mktPSRType.psrType')
        psr_type_name = PSR_TYPE_MAPPING.get(psr_type, psr_type)
        
        # Get flow direction (A01=up, A02=down)
        flow_direction = get_text(ts, 'flowDirection.direction')
        flow_direction_name = FLOW_DIRECTION_MAPPING.get(flow_direction, flow_direction)
        
        # Get unit of measure
        quantity_unit = get_text(ts, 'quantity_Measure_Unit.name')
        
        # Get curve type
        curve_type = get_text(ts, 'curveType')  # A03 = variable sized block
        
        # Get reason codes and map to labels
        reason_elems = find_all_elem(ts, 'Reason')
        reason_labels = []
        for reason_elem in reason_elems:
            code = get_text(reason_elem, 'code')
            if code:
                label = REDISPATCH_REASON_CODES.get(code, code)
                reason_labels.append(label)
        reason_labels_str = ','.join(reason_labels) if reason_labels else None
        
        # Process each Period
        periods = find_all_elem(ts, 'Period')
        
        for period in periods:
            time_interval = find_elem(period, 'timeInterval')
            if time_interval is None:
                continue
            
            period_start = get_text(time_interval, 'start')
            period_end = get_text(time_interval, 'end')
            resolution = get_text(period, 'resolution')
            
            if not period_start or not period_end:
                continue
            
            period_start_dt = pd.to_datetime(period_start)
            period_end_dt = pd.to_datetime(period_end)
            
            # Determine interval duration from resolution
            if resolution == 'PT1M':
                interval_minutes = 1
            elif resolution == 'PT15M':
                interval_minutes = 15
            elif resolution == 'PT30M':
                interval_minutes = 30
            elif resolution == 'PT60M' or resolution == 'PT1H':
                interval_minutes = 60
            elif resolution == 'P1D':
                interval_minutes = 60 * 24
            else:
                interval_minutes = 15  # Default to 15 min for redispatch
            
            # Calculate total number of intervals in the period
            total_intervals = int((period_end_dt - period_start_dt).total_seconds() / (interval_minutes * 60))
            
            # Get all Point elements
            points = find_all_elem(period, 'Point')
            
            if not points:
                continue
            
            # Build a dict of position -> quantity for sparse points
            point_data = {}
            for point in points:
                position = get_text(point, 'position')
                quantity = get_text(point, 'quantity')
                if position is not None:
                    point_data[int(position)] = float(quantity) if quantity else 0.0
            
            # Forward-fill values: for each interval, use the value from the most recent Point
            current_quantity = 0.0  # Default if no point at position 1
            
            for interval_idx in range(1, total_intervals + 1):
                # Check if there's a new value at this position
                if interval_idx in point_data:
                    current_quantity = point_data[interval_idx]
                
                # Calculate timestamps for this interval
                begin_timestamp = period_start_dt + timedelta(minutes=interval_minutes * (interval_idx - 1))
                end_timestamp = begin_timestamp + timedelta(minutes=interval_minutes)
                
                record = {
                    'begin_timestamp_utc': begin_timestamp,
                    'end_timestamp_utc': end_timestamp,
                    'doc_mrid': doc_mrid,
                    'doc_revision_number': int(doc_revision_number) if doc_revision_number else 1,
                    'ts_mrid': ts_mrid,
                    # Market participant info
                    'sender_mrid': sender_mrid,
                    'sender_mrid_name': sender_mrid_name,
                    'sender_role': sender_role,
                    'sender_role_name': sender_role_name,
                    'receiver_mrid': receiver_mrid,
                    'receiver_mrid_name': receiver_mrid_name,
                    'receiver_role': receiver_role,
                    'receiver_role_name': receiver_role_name,
                    # Domain info
                    'in_domain': in_domain,
                    'in_domain_name': in_domain_name,
                    'out_domain': out_domain,
                    'out_domain_name': out_domain_name,
                    # PSR and flow
                    'psr_type': psr_type,
                    'psr_type_name': psr_type_name,
                    'flow_direction': flow_direction,
                    'flow_direction_name': flow_direction_name,
                    'quantity_mwh': current_quantity,
                    'quantity_unit': quantity_unit,
                    'resolution': resolution,
                    'curve_type': curve_type,
                    'reason_labels': reason_labels_str,
                    'business_type': business_type,
                }
                records.append(record)
    
    df = pd.DataFrame(records)
    
    if not df.empty:
        # Ensure UTC timezone-naive
        df['begin_timestamp_utc'] = pd.to_datetime(df['begin_timestamp_utc']).dt.tz_localize(None)
        df['end_timestamp_utc'] = pd.to_datetime(df['end_timestamp_utc']).dt.tz_localize(None)
        df = df.sort_values(['begin_timestamp_utc', 'in_domain_name', 'out_domain_name']).reset_index(drop=True)
    
    return df

# %%
def fetch_cross_border_redispatch_chunk_with_adaptive_window(
    client: ENTSOEClient,
    conn_name: str,
    in_domain: str,
    out_domain: str,
    chunk_start: datetime,
    chunk_end: datetime,
    initial_hours: int,
    min_hours: int = 1,
    min_delay: float = 3.0,
    max_delay: float = 5.0,
) -> tuple[List[pd.DataFrame], int]:
    """
    Fetch cross-border redispatch data for a time range with adaptive window sizing.
    
    If the API returns a 'max instances exceeded' error, progressively
    halve the window size until success or min_hours is reached.
    
    Parameters:
    - client: ENTSOEClient instance
    - conn_name: Name of the connection (for logging)
    - in_domain: EIC code for in_Domain
    - out_domain: EIC code for out_Domain
    - chunk_start: Start datetime
    - chunk_end: End datetime
    - initial_hours: Initial window size in hours
    - min_hours: Minimum window size in hours (default: 1)
    - min_delay: Minimum delay between requests
    - max_delay: Maximum delay between requests
    
    Returns:
    - Tuple of (list of DataFrames, successful_window_hours)
    """
    all_data = []
    current_hours = initial_hours
    current_start = chunk_start
    
    while current_start < chunk_end:
        # Calculate the window end
        window_end = min(current_start + timedelta(hours=current_hours), chunk_end)
        
        params = {
            "documentType": "A63",  # Redispatching document
            "businessType": "A46",  # System Operator re-dispatching (cross-border)
            "in_Domain": in_domain,
            "out_Domain": out_domain,
            "periodStart": client._format_datetime(current_start),
            "periodEnd": client._format_datetime(window_end),
        }
        
        # Make request with explicit handling of max instances error
        params_with_token = {**params, "securityToken": client.api_key}
        
        try:
            response = client.session.get(client.base_url, params=params_with_token, timeout=60)
            
            if response.status_code == 200:
                # Check for max instances error in successful response
                if _check_max_instances_error(response.text):
                    num_inst, max_allowed = _parse_max_instances_error(response.text)
                    print(f"\n⚠️  Max instances exceeded for {conn_name} ({current_start.date()} - {window_end.date()}): "
                          f"{num_inst} instances > {max_allowed} allowed")
                    
                    # Halve the window size
                    new_hours = current_hours // 2
                    if new_hours < min_hours:
                        print(f"❌ Cannot reduce window below {min_hours} hour(s). Skipping this period.")
                        current_start = window_end
                        time.sleep(random.uniform(min_delay, max_delay))
                        continue
                    
                    print(f"↘️  Reducing window from {current_hours}h to {new_hours}h")
                    current_hours = new_hours
                    time.sleep(random.uniform(min_delay, max_delay))
                    continue  # Retry with smaller window, don't advance current_start
                
                # Check for "no data" response
                if "<Reason>" in response.text and "No matching data found" in response.text:
                    # No data for this period, move on
                    current_start = window_end
                    time.sleep(random.uniform(min_delay, max_delay))
                    continue
                
                # Success - parse the data
                xml_response = client._decode_and_parse(response.content)
                
                if xml_response:
                    if isinstance(xml_response, list):
                        for xml_text in xml_response:
                            df = parse_cross_border_redispatch_xml_to_dataframe(xml_text)
                            if not df.empty:
                                df['connection_name'] = conn_name
                                all_data.append(df)
                    else:
                        df = parse_cross_border_redispatch_xml_to_dataframe(xml_response)
                        if not df.empty:
                            df['connection_name'] = conn_name
                            all_data.append(df)
                
                # Success - move to next window
                current_start = window_end
                time.sleep(random.uniform(min_delay, max_delay))
                
            elif response.status_code == 400:
                # Check if it's a max instances error
                if _check_max_instances_error(response.text):
                    num_inst, max_allowed = _parse_max_instances_error(response.text)
                    print(f"\n⚠️  Max instances exceeded (400) for {conn_name} ({current_start.date()} - {window_end.date()}): "
                          f"{num_inst} instances > {max_allowed} allowed")
                    
                    # Halve the window size
                    new_hours = current_hours // 2
                    if new_hours < min_hours:
                        print(f"❌ Cannot reduce window below {min_hours} hour(s). Skipping this period.")
                        current_start = window_end
                        time.sleep(random.uniform(min_delay, max_delay))
                        continue
                    
                    print(f"↘️  Reducing window from {current_hours}h to {new_hours}h")
                    current_hours = new_hours
                    time.sleep(random.uniform(min_delay, max_delay))
                    continue
                else:
                    print(f"Bad Request (400): {response.text[:200]}...")
                    current_start = window_end
                    time.sleep(random.uniform(min_delay, max_delay))
                    
            elif response.status_code == 409 or response.status_code == 429:
                print(f"Rate limit hit ({response.status_code}), waiting 60s...")
                time.sleep(60)
                continue  # Retry same request
                
            else:
                print(f"HTTP Error {response.status_code}: {response.text[:200]}...")
                current_start = window_end
                time.sleep(random.uniform(min_delay, max_delay))
                
        except requests.exceptions.Timeout:
            print("Request timed out, retrying in 5s...")
            time.sleep(5)
            continue
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            current_start = window_end
            time.sleep(random.uniform(min_delay, max_delay))
    
    return all_data, current_hours


def fetch_cross_border_redispatch_data(
    client: ENTSOEClient,
    start_date: datetime,
    end_date: datetime,
    connections: Dict[str, tuple] = None,
    chunk_days: int = 10,
    min_delay: float = 3.0,
    max_delay: float = 5.0,
    adaptive_chunking: bool = True,
    min_chunk_hours: int = 1,
    check_both_directions: bool = True,
    save_steps: int = 100,
) -> pd.DataFrame:
    """
    Fetch Cross-Border Redispatching data (13.1.A) from ENTSO-E.
    
    Checks both directions of any control area combination (A->B and B->A)
    similar to cross-border physical flows extraction strategy.
    
    Parameters:
    - client: ENTSOEClient instance
    - start_date: Start date for data fetching
    - end_date: End date for data fetching
    - connections: Dict mapping connection name to (in_domain, out_domain) EIC codes
                   If None, uses CROSS_BORDER_CONNECTIONS
    - chunk_days: Number of days per request chunk (default: 10)
    - min_delay: Minimum delay between requests in seconds (default: 3.0)
    - max_delay: Maximum delay between requests in seconds (default: 5.0)
    - adaptive_chunking: If True, automatically reduce chunk size on max instances error (default: True)
    - min_chunk_hours: Minimum chunk size in hours when using adaptive chunking (default: 1)
    - check_both_directions: If True, check both A->B and B->A directions (default: True)
    
    Returns:
    DataFrame with all cross-border redispatch records, including in_domain and out_domain columns
    """
    if connections is None:
        connections = REDISPATCHING_CROSS_BORDER_CONNECTIONS
    
    all_data = []
    
    # Create date chunks
    date_chunks = client._chunk_date_range(start_date, end_date, max_days=chunk_days)
    initial_hours = chunk_days * 24
    
    # Calculate total requests (doubled if checking both directions)
    direction_multiplier = 2 if check_both_directions else 1
    total_requests = len(connections) * len(date_chunks) * direction_multiplier
    
    with tqdm(total=total_requests, desc="Fetching cross-border redispatch data") as pbar:
        for conn_name, (in_domain, out_domain) in connections.items():
            # Track successful window size per connection (can be reused across chunks and directions)
            current_window_hours = initial_hours
            
            # Direction 1: in_domain -> out_domain (as defined in connection)
            for chunk_start, chunk_end in date_chunks:
                if adaptive_chunking:
                    chunk_data, successful_hours = fetch_cross_border_redispatch_chunk_with_adaptive_window(
                        client=client,
                        conn_name=f"{conn_name}_fwd",
                        in_domain=in_domain,
                        out_domain=out_domain,
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                        initial_hours=current_window_hours,
                        min_hours=min_chunk_hours,
                        min_delay=min_delay,
                        max_delay=max_delay,
                    )
                    all_data.extend(chunk_data)
                    
                    if successful_hours < current_window_hours:
                        current_window_hours = successful_hours
                        print(f"📝 Adjusted window size to {current_window_hours}h for {conn_name}")
                else:
                    # Original behavior without adaptive chunking
                    params = {
                        "documentType": "A63",
                        "businessType": "A46",
                        "in_Domain": in_domain,
                        "out_Domain": out_domain,
                        "periodStart": client._format_datetime(chunk_start),
                        "periodEnd": client._format_datetime(chunk_end),
                    }
                    
                    xml_response = client._make_request(params)
                    
                    if xml_response:
                        if isinstance(xml_response, list):
                            for xml_text in xml_response:
                                df = parse_cross_border_redispatch_xml_to_dataframe(xml_text)
                                if not df.empty:
                                    df['connection_name'] = f"{conn_name}_fwd"
                                    all_data.append(df)
                        else:
                            df = parse_cross_border_redispatch_xml_to_dataframe(xml_response)
                            if not df.empty:
                                df['connection_name'] = f"{conn_name}_fwd"
                                all_data.append(df)
                    
                    time.sleep(random.uniform(min_delay, max_delay))
                
                pbar.update(1)
                pbar.set_postfix(connection=f"{conn_name}_fwd", chunk=f"{chunk_start.date()} to {chunk_end.date()}")
                if save_steps > 0 and (pbar.n % save_steps == 0):
                    print(f"\n💾 Intermediate save after {pbar.n} requests")
                    intermediate_df = pd.concat(all_data, ignore_index=True)
                    save_data(intermediate_df, REDISPATCH_DIR / "entsoe_redispatch_cross_border_intermediate.csv", "Intermediate Cross-Border Redispatch Data")
            
            # Direction 2: out_domain -> in_domain (reverse direction)
            if check_both_directions:
                for chunk_start, chunk_end in date_chunks:
                    if adaptive_chunking:
                        chunk_data, successful_hours = fetch_cross_border_redispatch_chunk_with_adaptive_window(
                            client=client,
                            conn_name=f"{conn_name}_rev",
                            in_domain=out_domain,  # Swapped
                            out_domain=in_domain,  # Swapped
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            initial_hours=current_window_hours,
                            min_hours=min_chunk_hours,
                            min_delay=min_delay,
                            max_delay=max_delay,
                        )
                        all_data.extend(chunk_data)
                        
                        if successful_hours < current_window_hours:
                            current_window_hours = successful_hours
                            print(f"📝 Adjusted window size to {current_window_hours}h for {conn_name} (reverse)")
                    else:
                        # Original behavior without adaptive chunking
                        params = {
                            "documentType": "A63",
                            "businessType": "A46",
                            "in_Domain": out_domain,  # Swapped
                            "out_Domain": in_domain,  # Swapped
                            "periodStart": client._format_datetime(chunk_start),
                            "periodEnd": client._format_datetime(chunk_end),
                        }
                        
                        xml_response = client._make_request(params)
                        
                        if xml_response:
                            if isinstance(xml_response, list):
                                for xml_text in xml_response:
                                    df = parse_cross_border_redispatch_xml_to_dataframe(xml_text)
                                    if not df.empty:
                                        df['connection_name'] = f"{conn_name}_rev"
                                        all_data.append(df)
                            else:
                                df = parse_cross_border_redispatch_xml_to_dataframe(xml_response)
                                if not df.empty:
                                    df['connection_name'] = f"{conn_name}_rev"
                                    all_data.append(df)
                        
                        time.sleep(random.uniform(min_delay, max_delay))
                    
                    pbar.update(1)
                    pbar.set_postfix(connection=f"{conn_name}_rev", chunk=f"{chunk_start.date()} to {chunk_end.date()}")
                    if save_steps > 0 and (pbar.n % save_steps == 0):
                        print(f"\n💾 Intermediate save after {pbar.n} requests")
                        intermediate_df = pd.concat(all_data, ignore_index=True)
                        save_data(intermediate_df, REDISPATCH_DIR / "entsoe_redispatch_cross_border_intermediate.csv", "Intermediate Cross-Border Redispatch Data")
    
    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        
        result_df = result_df.sort_values(
            ['begin_timestamp_utc', 'in_domain_name', 'out_domain_name', 'psr_type_name', 'flow_direction_name']
        ).reset_index(drop=True)
        
        return result_df
    
    return pd.DataFrame()

# %% [markdown]
# ### Fetch Cross-Border Redispatching Data: German TSO -> DK (entire country)
# 
# For cross-border redispatching, we only consider German TSO to Denmark (entire country) connections.
# This uses the DK EIC code (10Y1001A1001A796) rather than DK_1 or DK_2.

# %%
# Cross-border interconnection EIC codes
REDISPATCHING_CROSS_BORDER_CONNECTIONS = {}

# Get EIC codes for German TSOs and neighboring countries
# For country EIC and Denmark, use only DK_1 and DK_2
# Only include FR and AT, as all other countries do not report cross-border redispatching
german_tsos = GERMAN_TSO_AREAS
neighbors_tso = {k: v for k, v in BIDDING_ZONES.items() if k in ["FR", "AT", "DK"]}
# 1. Connections between each German TSO and neighboring countries
# Here use DK
for tso_name, tso_eic in german_tsos.items():
    for neighbor_name, neighbor_eic in neighbors_tso.items():
        if tso_name == "DE_TenneT" and neighbor_name == "DK":
            continue  # Skip DE_Tennet to DK connection (already included in an early export)
        # TSO to neighbor
        conn_name_fwd = f"{tso_name}_to_{neighbor_name}"
        REDISPATCHING_CROSS_BORDER_CONNECTIONS[conn_name_fwd] = (tso_eic, neighbor_eic)

# 2. Connections between different German TSOs
tso_items = list(german_tsos.items())
for i in range(len(tso_items)):
    for j in range(i + 1, len(tso_items)):
        tso1_name, tso1_eic = tso_items[i]
        tso2_name, tso2_eic = tso_items[j]
        
        # TSO1 to TSO2
        conn_name_fwd = f"{tso1_name}_to_{tso2_name}"
        REDISPATCHING_CROSS_BORDER_CONNECTIONS[conn_name_fwd] = (tso1_eic, tso2_eic)

print(f"Generated {len(REDISPATCHING_CROSS_BORDER_CONNECTIONS)} cross-border connections.")

# %%
# Fetch cross-border redispatch data for German TSO -> country/Germany TSO connections
# Note: Uses adaptive chunking - if API returns "max instances exceeded" error,
# the window will automatically be halved until data can be fetched (min 1 hour)
# Both directions of each connection are checked (A->B and B->A)
redispatch_cross_border_df = fetch_cross_border_redispatch_data(
    client=client,
    start_date=START_DATE,
    end_date=END_DATE,
    connections=REDISPATCHING_CROSS_BORDER_CONNECTIONS,
    chunk_days=2,  # Initial chunk size (will be reduced automatically if needed)
    min_delay=1.0,
    max_delay=3.0,
    adaptive_chunking=True,  # Enable automatic window reduction on error
    min_chunk_hours=1,  # Minimum window size before giving up
    check_both_directions=True,  # Check both A->B and B->A
    save_steps=100,  # Save intermediate results every 100 requests
)

if not redispatch_cross_border_df.empty:
    # Drop columns that are completely empty
    redispatch_cross_border_df = redispatch_cross_border_df.dropna(axis=1, how='all')
    
    print(f"Fetched {len(redispatch_cross_border_df)} cross-border redispatch records")
    print(f"Date range: {redispatch_cross_border_df['begin_timestamp_utc'].min()} to {redispatch_cross_border_df['begin_timestamp_utc'].max()}")
    print(f"\nConnections: {redispatch_cross_border_df['connection_name'].unique()}")
    print(f"in_domain values: {redispatch_cross_border_df['in_domain'].unique()}")
    print(f"out_domain values: {redispatch_cross_border_df['out_domain'].unique()}")
    print(f"PSR types: {redispatch_cross_border_df['psr_type_name'].unique()}")
    print(f"Flow directions: {redispatch_cross_border_df['flow_direction_name'].unique()}")
else:
    print("No cross-border redispatch data found for the specified date range and connections.")

# %%
# Save cross-border redispatch data to CSV
if not redispatch_cross_border_df.empty:
    cross_border_output_file = REDISPATCH_DIR / "entsoe_redispatch_cross_border_germany.csv"
    save_data(redispatch_cross_border_df, cross_border_output_file, "cross-border redispatch data")
    print(f"Saved cross-border redispatch data to {cross_border_output_file}")


