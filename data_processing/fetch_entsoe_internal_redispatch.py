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
    
    def _parse_xml_to_dataframe(self, xml_text: str, value_type: str = "quantity") -> pd.DataFrame:
        """
        Parse ENTSO-E XML response to DataFrame.
        Extracts TimeSeries data with begin_date, end_date, and values.
        
        Parameters:
        - xml_text: Raw XML response
        - value_type: Type of value to extract ('quantity', 'price.amount')
        
        Returns DataFrame with columns: begin_date, end_date, value, [optional metadata]
        """
        if not xml_text:
            return pd.DataFrame()
        
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
            return pd.DataFrame()
        
        # Define namespace
        ns = {
            'gl': 'urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0',
            'pub': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3',
            'tp': 'urn:iec62325.351:tc57wg16:451-3:transmissionnetworkdocument:2:4',
        }
        
        # Try to detect namespace from root element
        root_ns = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
        if root_ns:
            ns['default'] = root_ns
        
        records = []
        
        # Find all TimeSeries elements
        for ts_ns_key in ['default', 'gl', 'pub', 'tp']:
            if ts_ns_key not in ns:
                continue
            time_series_list = root.findall(f'.//{{{ns[ts_ns_key]}}}TimeSeries')
            if time_series_list:
                break
        else:
            # Fallback: try without namespace
            time_series_list = root.findall('.//TimeSeries')
        
        for ts in time_series_list:
            # Extract metadata
            metadata = {}
            
            # Try to get PSR type
            psr_type_elem = ts.find('.//{*}psrType')
            if psr_type_elem is not None:
                metadata['psr_type'] = psr_type_elem.text
                metadata['psr_type_name'] = PSR_TYPES.get(psr_type_elem.text, psr_type_elem.text)
            
            # Try to get domain/area
            for domain_tag in ['inBiddingZone_Domain.mRID', 'in_Domain.mRID', 'outBiddingZone_Domain.mRID', 'out_Domain.mRID']:
                domain_elem = ts.find(f'.//{{*}}{domain_tag}')
                if domain_elem is not None:
                    metadata['domain'] = domain_elem.text
                    break
            
            # Find all Period elements
            periods = ts.findall('.//{*}Period')
            
            for period in periods:
                # Get time interval
                time_interval = period.find('.//{*}timeInterval')
                if time_interval is None:
                    continue
                    
                start_elem = time_interval.find('.//{*}start')
                end_elem = time_interval.find('.//{*}end')
                
                if start_elem is None or end_elem is None:
                    continue
                
                period_start = pd.to_datetime(start_elem.text)
                period_end = pd.to_datetime(end_elem.text)
                
                # Get resolution
                resolution_elem = period.find('.//{*}resolution')
                resolution = resolution_elem.text if resolution_elem is not None else 'PT60M'
                
                # Calculate interval duration
                if resolution == 'PT15M':
                    interval_minutes = 15
                elif resolution == 'PT30M':
                    interval_minutes = 30
                elif resolution == 'PT60M' or resolution == 'P1D':
                    interval_minutes = 60
                elif resolution == 'P1Y':
                    interval_minutes = 60 * 24 * 365  # Approximate
                else:
                    interval_minutes = 60
                
                # Get all Point elements
                points = period.findall('.//{*}Point')
                
                for point in points:
                    position_elem = point.find('.//{*}position')
                    value_elem = point.find(f'.//{{*}}{value_type}')
                    
                    # Also try 'price.amount' for price data
                    if value_elem is None and value_type == 'quantity':
                        value_elem = point.find('.//{*}price.amount')
                    
                    if position_elem is None:
                        continue
                    
                    position = int(position_elem.text)
                    value = float(value_elem.text) if value_elem is not None else np.nan
                    
                    # Calculate exact timestamp
                    point_start = period_start + timedelta(minutes=interval_minutes * (position - 1))
                    point_end = point_start + timedelta(minutes=interval_minutes)
                    
                    record = {
                        'begin_date': point_start,
                        'end_date': point_end,
                        'value': value,
                        'resolution': resolution,
                        **metadata
                    }
                    records.append(record)
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            # Ensure UTC timezone (ENTSO-E data is already in UTC)
            df['begin_date'] = pd.to_datetime(df['begin_date']).dt.tz_localize(None)
            df['end_date'] = pd.to_datetime(df['end_date']).dt.tz_localize(None)
            
            # Sort by timestamp
            df = df.sort_values('begin_date').reset_index(drop=True)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['begin_date', 'end_date'] + 
                                   ([col for col in ['psr_type', 'domain'] if col in df.columns]))
        
        return df

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
# ## Redispatching Internal Data Fetching
# 
# Fetch Redispatching Internal data (13.1.A) from ENTSO-E Transparency Platform.
# This includes internal redispatch measures for Germany (per control area).

# %%
# Mapping codes for redispatch data
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


def parse_redispatch_xml_to_dataframe(xml_text: str) -> pd.DataFrame:
    """
    Parse ENTSO-E Redispatching Internal XML response to DataFrame.
    
    Handles sparse time series data by forward-filling values until the next 
    Point entry or the end of the period.
    
    Returns DataFrame with columns:
    - timestamp_utc: Start time of each interval (UTC)
    - doc_mrid: Document-level mRID
    - doc_revision_number: Document-level revision number
    - ts_mrid: TimeSeries mRID for tracking (doc_mrid + '_' + ts.mRID)
    - tso: TSO name (from GERMAN_TSO_AREAS mapping)
    - in_domain: Control area EIC code
    - out_domain: Control area EIC code
    - psr_type: A04 (generation) or A05 (load)
    - psr_type_name: "generation" or "load"
    - flow_direction: A01 (up) or A02 (down)
    - flow_direction_name: "up" or "down"
    - quantity_mwh: Redispatch volume in MWh
    - quantity_unit: Unit of measure
    - resolution: Time resolution (e.g., PT15M)
    - curve_type: Curve type (A03 = variable sized block)
    - reason_labels: Comma-separated list of reason labels
    - business_type: Business type (A85)
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
    
    # Find all TimeSeries elements
    time_series_list = find_all_elem(root, 'TimeSeries')
    
    for ts in time_series_list:
        # Extract metadata
        ts_mrid_raw = get_text(ts, 'mRID')
        ts_mrid = f"{doc_mrid}_{ts_mrid_raw}" if doc_mrid and ts_mrid_raw else ts_mrid_raw
        business_type = get_text(ts, 'businessType')  # Should be A85
        
        # Get domain info
        in_domain = get_text(ts, 'in_Domain.mRID')
        out_domain = get_text(ts, 'out_Domain.mRID')
        
        # Map domain to TSO name
        tso_name = None
        for tso, eic in GERMAN_TSO_AREAS.items():
            if in_domain == eic:
                tso_name = tso
                break
        
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
                
                # Calculate timestamp for this interval
                timestamp = period_start_dt + timedelta(minutes=interval_minutes * (interval_idx - 1))
                end_timestamp = timestamp + timedelta(minutes=interval_minutes)
                
                record = {
                    'begin_timestamp_utc': timestamp,
                    'end_timestamp_utc': end_timestamp,
                    'doc_mrid': doc_mrid,
                    'doc_revision_number': int(doc_revision_number) if doc_revision_number else 1,
                    'ts_mrid': ts_mrid,
                    'tso': tso_name,
                    'in_domain': in_domain,
                    'out_domain': out_domain,
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
        df = df.sort_values(['begin_timestamp_utc', 'end_timestamp_utc', 'tso', 'psr_type_name', 'flow_direction_name']).reset_index(drop=True)
    
    return df

# %%
class MaxInstancesExceededError(Exception):
    """Raised when ENTSO-E API returns error 999 (max instances exceeded)."""
    def __init__(self, message: str, num_instances: int = None, max_allowed: int = None):
        super().__init__(message)
        self.num_instances = num_instances
        self.max_allowed = max_allowed


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


def fetch_redispatch_chunk_with_adaptive_window(
    client: ENTSOEClient,
    area_name: str,
    eic_code: str,
    chunk_start: datetime,
    chunk_end: datetime,
    initial_hours: int,
    min_hours: int = 1,
    min_delay: float = 3.0,
    max_delay: float = 5.0,
) -> tuple[List[pd.DataFrame], int]:
    """
    Fetch redispatch data for a time range with adaptive window sizing.
    
    If the API returns a 'max instances exceeded' error, progressively
    halve the window size until success or min_hours is reached.
    
    Parameters:
    - client: ENTSOEClient instance
    - area_name: Name of the area (for logging)
    - eic_code: EIC code for the area
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
            "businessType": "A85",  # Internal redispatch
            "in_Domain": eic_code,
            "out_Domain": eic_code,
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
                    print(f"\n⚠️  Max instances exceeded for {area_name} ({current_start.date()} - {window_end.date()}): "
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
                            df = parse_redispatch_xml_to_dataframe(xml_text)
                            if not df.empty:
                                all_data.append(df)
                    else:
                        df = parse_redispatch_xml_to_dataframe(xml_response)
                        if not df.empty:
                            all_data.append(df)
                
                # Success - move to next window
                current_start = window_end
                time.sleep(random.uniform(min_delay, max_delay))
                
            elif response.status_code == 400:
                # Check if it's a max instances error
                if _check_max_instances_error(response.text):
                    num_inst, max_allowed = _parse_max_instances_error(response.text)
                    print(f"\n⚠️  Max instances exceeded (400) for {area_name} ({current_start.date()} - {window_end.date()}): "
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


def fetch_redispatch_internal_data(
    client: ENTSOEClient,
    start_date: datetime,
    end_date: datetime,
    areas: Dict[str, str] = None,
    chunk_days: int = 10,
    min_delay: float = 3.0,
    max_delay: float = 5.0,
    adaptive_chunking: bool = True,
    min_chunk_hours: int = 1,
    save_steps: int = -1,
) -> pd.DataFrame:
    """
    Fetch Redispatching Internal data (13.1.A) from ENTSO-E.
    
    Parameters:
    - client: ENTSOEClient instance
    - start_date: Start date for data fetching
    - end_date: End date for data fetching
    - areas: Dict of area names to EIC codes (default: German TSOs)
    - chunk_days: Number of days per request chunk (default: 10)
    - min_delay: Minimum delay between requests in seconds (default: 3.0)
    - max_delay: Maximum delay between requests in seconds (default: 5.0)
    - adaptive_chunking: If True, automatically reduce chunk size on max instances error (default: True)
    - min_chunk_hours: Minimum chunk size in hours when using adaptive chunking (default: 1)
    
    Returns:
    DataFrame with all redispatch records
    """
    if areas is None:
        areas = GERMAN_TSO_AREAS
    
    all_data = []
    
    # Create date chunks
    date_chunks = client._chunk_date_range(start_date, end_date, max_days=chunk_days)
    initial_hours = chunk_days * 24
    
    total_requests = len(areas) * len(date_chunks)
    
    with tqdm(total=total_requests, desc="Fetching redispatch internal data") as pbar:
        for area_name, eic_code in areas.items():
            # Track successful window size per area (can be reused across chunks)
            current_window_hours = initial_hours
            
            for chunk_start, chunk_end in date_chunks:
                if adaptive_chunking:
                    # Use adaptive chunking with progressive window reduction
                    chunk_data, successful_hours = fetch_redispatch_chunk_with_adaptive_window(
                        client=client,
                        area_name=area_name,
                        eic_code=eic_code,
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                        initial_hours=current_window_hours,
                        min_hours=min_chunk_hours,
                        min_delay=min_delay,
                        max_delay=max_delay,
                    )
                    all_data.extend(chunk_data)
                    
                    # Remember successful window size for next chunk
                    if successful_hours < current_window_hours:
                        current_window_hours = successful_hours
                        print(f"📝 Adjusted window size to {current_window_hours}h for {area_name}")
                else:
                    # Original behavior without adaptive chunking
                    params = {
                        "documentType": "A63",
                        "businessType": "A85",
                        "in_Domain": eic_code,
                        "out_Domain": eic_code,
                        "periodStart": client._format_datetime(chunk_start),
                        "periodEnd": client._format_datetime(chunk_end),
                    }
                    
                    xml_response = client._make_request(params)
                    
                    if xml_response:
                        if isinstance(xml_response, list):
                            for xml_text in xml_response:
                                df = parse_redispatch_xml_to_dataframe(xml_text)
                                if not df.empty:
                                    all_data.append(df)
                        else:
                            df = parse_redispatch_xml_to_dataframe(xml_response)
                            if not df.empty:
                                all_data.append(df)
                    
                    time.sleep(random.uniform(min_delay, max_delay))
                
                pbar.update(1)
                pbar.set_postfix(area=area_name, chunk=f"{chunk_start.date()} to {chunk_end.date()}")
                if save_steps > 0 and (pbar.n % save_steps == 0):
                    print(f"\n💾 Intermediate save after {pbar.n} requests")
                    intermediate_df = pd.concat(all_data, ignore_index=True)
                    save_data(intermediate_df, DATA_DIR / "redispatch_internal" / "entsoe_redispatch_internal_intermediate.csv", "Intermediate Redispatch Internal Data")
    
    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        
        result_df = result_df.sort_values(
            ['begin_timestamp_utc', 'tso']
        ).reset_index(drop=True)
        
        return result_df
    
    return pd.DataFrame()

# %% [markdown]
# ### Fetch Redispatching Internal Data for German TSOs

# %%
# Fetch redispatch internal data for German TSOs
# Note: Uses adaptive chunking - if API returns "max instances exceeded" error,
# the window will automatically be halved until data can be fetched (min 1 hour)
redispatch_internal_df = fetch_redispatch_internal_data(
    client=client,
    start_date=datetime(2021, 10, 1),
    end_date=datetime(2026, 1, 5),
    areas=GERMAN_TSO_AREAS, #{tso: data for tso, data in GERMAN_TSO_AREAS.items() if tso in ["DE_TenneT"]},
    chunk_days=2,  # Initial chunk size (will be reduced automatically if needed)
    min_delay=1.0,
    max_delay=3.0,
    adaptive_chunking=True,  # Enable automatic window reduction on error
    min_chunk_hours=1,  # Minimum window size before giving up
    save_steps=100,
)

print(f"Fetched {len(redispatch_internal_df)} redispatch internal records")
print(f"Date range: {redispatch_internal_df['begin_timestamp_utc'].min()} to {redispatch_internal_df['begin_timestamp_utc'].max()}")
print(f"\nTSOs: {redispatch_internal_df['tso'].unique()}")
print(f"PSR types: {redispatch_internal_df['psr_type_name'].unique()}")
print(f"Flow directions: {redispatch_internal_df['flow_direction_name'].unique()}")

# %%
save_data(redispatch_internal_df, DATA_DIR / "redispatch_internal" / "entsoe_redispatch_internal_germany_raw.csv", "Redispatch Internal Data")