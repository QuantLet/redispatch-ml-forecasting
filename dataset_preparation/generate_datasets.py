"""
Dataset Generation Script.

This script orchestrates the loading of base redispatch data, applies feature 
set combinations, and saves resulting datasets locally and/or to Weights & Biases.

Usage:
    python -m dataset_preparation.generate_datasets \
        --combinations-path dataset_preparation/combinations.json \
        --n-workers 4 \
        --output-dir data/model_data_multi_feature_combinations/ \
        --use-wandb
        
Or from within the dataset_preparation directory:
    cd dataset_preparation && python -c "from generate_datasets import main; main()" ...
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np

# Handle imports whether run as module or script
try:
    # When run as module (python -m dataset_preparation.generate_datasets)
    from .redispatch_core import (
        prepare_final_target,
        read_redispatch_data,
        TARGET_RELEVANT_MEASUREMENT_REASONS,
        WANDB_DATASET_ARTIFACT_PREFIX,
        MIN_GAP_RATIO,
        ROLLING_WINDOW_LAGGED_DAYS,
    )
    from .tso_config import GERMAN_TSOS, get_neighbors, normalize_tso_name, TARGET_SPLIT_METHOD_DICT
    from .feature_loaders.prices_loader import load_day_ahead_prices
    from .feature_loaders.consumption_loader import load_consumption
    from .feature_loaders.production_loader import load_production
    from .feature_loaders.wind_pv_loader import load_wind_pv_features
    from .feature_loaders.cross_border_loader import load_cross_border_flows
    from .feature_loaders.bloomberg_loader import load_bloomberg_data
    from .feature_loaders.sce_loader import load_scheduled_exchanges
except ImportError:
    # When run as script, add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset_preparation.redispatch_core import (
        prepare_final_target,
        read_redispatch_data,
        TARGET_RELEVANT_MEASUREMENT_REASONS,
        WANDB_DATASET_ARTIFACT_PREFIX,
        MIN_GAP_RATIO,
        ROLLING_WINDOW_LAGGED_DAYS,
    )
    from dataset_preparation.tso_config import GERMAN_TSOS, get_neighbors, normalize_tso_name, TARGET_SPLIT_METHOD_DICT
    from dataset_preparation.feature_loaders.prices_loader import load_day_ahead_prices
    from dataset_preparation.feature_loaders.consumption_loader import load_consumption
    from dataset_preparation.feature_loaders.production_loader import load_production
    from dataset_preparation.feature_loaders.wind_pv_loader import load_wind_pv_features
    from dataset_preparation.feature_loaders.cross_border_loader import load_cross_border_flows
    from dataset_preparation.feature_loaders.bloomberg_loader import load_bloomberg_data
    from dataset_preparation.feature_loaders.sce_loader import load_scheduled_exchanges

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class FeatureSetComposer:
    """Orchestrates loading and merging of feature sets onto base redispatch data."""
    
    def __init__(
        self,
        base_df: pd.DataFrame,
        tso: str,
        rolling_window_days: int = ROLLING_WINDOW_LAGGED_DAYS,
        oos_start_date: pd.Timestamp | None = None,
    ):
        """
        Initialize the composer.
        
        Parameters
        ----------
        base_df : pd.DataFrame
            Base redispatch data with 'begin_date' column.
        tso : str
            TSO name for loading TSO-specific features.
        rolling_window_days : int
            Window size for rolling features. Default from ROLLING_WINDOW_LAGGED_DAYS.
        """
        self.base_df = base_df.copy()
        self.tso = normalize_tso_name(tso)
        self.neighbors = get_neighbors(self.tso)
        self.rolling_window_days = rolling_window_days
        self.oos_start_date = oos_start_date
        self.merge_log: list[dict[str, Any]] = []
    
    def add_feature_set(self, feature_set_name: str) -> pd.DataFrame:
        """
        Load and merge a feature set.
        
        Parameters
        ----------
        feature_set_name : str
            Name of the feature set to add. Valid options:
            - "basic": No additional features (already in base_df)
            - "day_ahead_price": Day-ahead electricity prices
            - "production_consumption": Production and consumption data
            - "wind_pv": Wind and PV generation features
            - "cross_border": Cross-border flow data
            - "sce": Scheduled commercial exchange forecasts (net flows)
            - "bloomberg": Bloomberg financial data
            
        Returns
        -------
        pd.DataFrame
            Updated base DataFrame with merged features.
        """
        logger.info(f"  Adding feature set: {feature_set_name}")
        
        if feature_set_name == "basic":
            # Already in base_df
            return self.base_df
        
        elif feature_set_name == "day_ahead_price":
            features = load_day_ahead_prices(self.tso, oos_start_date=self.oos_start_date)
            
        elif feature_set_name == "production_consumption":
            prod = load_production(
                self.tso,
                window_days=self.rolling_window_days,
                oos_start_date=self.oos_start_date,
            )
            cons = load_consumption(
                self.tso,
                window_days=self.rolling_window_days,
                oos_start_date=self.oos_start_date,
            )
            features = prod.merge(cons, on="begin_date", how="outer")
            
        elif feature_set_name == "wind_pv":
            features = load_wind_pv_features(
                self.tso,
                rolling_window_days=self.rolling_window_days,
                oos_start_date=self.oos_start_date,
            )
            
        elif feature_set_name == "cross_border":
            features = load_cross_border_flows(
                self.tso,
                rolling_window_days=self.rolling_window_days,
                oos_start_date=self.oos_start_date,
            )
            
        elif feature_set_name == "sce":
            features = load_scheduled_exchanges(self.tso, oos_start_date=self.oos_start_date)
            
        elif feature_set_name == "bloomberg":
            features = load_bloomberg_data(oos_start_date=self.oos_start_date)
            
        else:
            raise ValueError(f"Unknown feature set: {feature_set_name}")
        
        # Check if features DataFrame is valid
        if features.empty or "begin_date" not in features.columns:
            logger.warning(f"  Feature set '{feature_set_name}' returned empty or invalid DataFrame")
            return self.base_df
        
        # Ensure begin_date is datetime for both DataFrames
        if not pd.api.types.is_datetime64_any_dtype(self.base_df["begin_date"]):
            self.base_df["begin_date"] = pd.to_datetime(self.base_df["begin_date"])
        if not pd.api.types.is_datetime64_any_dtype(features["begin_date"]):
            features["begin_date"] = pd.to_datetime(features["begin_date"])
        
        # Merge with validation
        rows_before = len(self.base_df)
        new_cols = [c for c in features.columns if c != "begin_date" and c not in self.base_df.columns]
        
        if not new_cols:
            logger.warning(f"  No new columns to merge from '{feature_set_name}'")
            return self.base_df
        
        self.base_df = self.base_df.merge(
            features[["begin_date"] + new_cols], 
            on="begin_date", 
            how="left", 
            validate="m:1"  # many-to-one (hours to features)
        )
        rows_after = len(self.base_df)

        # Impute Bloomberg data, which does not include weekends
        if feature_set_name == "bloomberg":
            bloomberg_cols = [c for c in new_cols if c.startswith("bloomberg_")]
            if bloomberg_cols:
                self.base_df[bloomberg_cols] = self.base_df[bloomberg_cols].ffill()
        
        # Log merge stats
        null_pct = self.base_df[new_cols].isnull().mean().mean() * 100
        
        self.merge_log.append({
            "feature_set": feature_set_name,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "new_columns": len(new_cols),
            "null_pct": round(null_pct, 2)
        })
        
        logger.info(f"    Merged {len(new_cols)} columns, {null_pct:.1f}% null values")
        
        return self.base_df

    def _add_production_forecast_other(self) -> None:
        """
        Compute production_forecast_other for each TSO suffix present.

        production_forecast_other = production_forecast
            - wind_forecast_total_de_<tso>
            - pv_forecast_de_<tso>
        
        Note: Updated to use wind_forecast_total (aggregated) instead of separate onshore/offshore.
        """
        prod_cols = [c for c in self.base_df.columns if c.startswith("production_forecast_de_")]
        if not prod_cols:
            return

        for prod_col in prod_cols:
            tso_suffix = prod_col.replace("production_forecast_de_", "", 1)
            # Use aggregated wind_total instead of separate onshore/offshore
            wind_total_col = f"wind_forecast_total_de_{tso_suffix}"
            pv_col = f"pv_forecast_de_{tso_suffix}"

            available_cols = [c for c in [wind_total_col, pv_col] if c in self.base_df.columns]
            if not available_cols:
                continue

            other_col = f"production_forecast_other_de_{tso_suffix}"
            self.base_df[other_col] = self.base_df.pop(prod_col)
            for col in available_cols:
                self.base_df[other_col] = self.base_df[other_col] - self.base_df[col]

    def _add_residual_load(self) -> None:
        """
        Compute residual load feature.
        
        residual_load_forecast = load_forecast - (wind_forecast + solar_forecast)
        
        This represents the load that must be met by dispatchable generation
        (thermal, hydro, imports), which is a key driver of redispatch needs.
        
        Note: Uses consumption_forecast_load_de_<tso> from consumption features
        and wind_forecast_total_de_<tso> + pv_forecast_de_<tso> from wind/pv features.
        """
        # Find load forecast columns
        load_cols = [c for c in self.base_df.columns if c.startswith("consumption_forecast_load_de_")]
        
        for load_col in load_cols:
            tso_suffix = load_col.replace("consumption_forecast_load_de_", "", 1)
            
            # Find corresponding wind and PV forecasts
            wind_total_col = f"wind_forecast_total_de_{tso_suffix}"
            pv_col = f"pv_forecast_de_{tso_suffix}"
            
            # Check if we have the required columns
            has_wind = wind_total_col in self.base_df.columns
            has_pv = pv_col in self.base_df.columns
            
            if not (has_wind or has_pv):
                logger.debug(f"Missing wind/pv forecast for residual load calculation for {tso_suffix}")
                continue
            
            # Compute residual load
            residual_col = f"residual_load_forecast_de_{tso_suffix}"
            self.base_df[residual_col] = self.base_df[load_col].copy()
            
            if has_wind:
                self.base_df[residual_col] = self.base_df[residual_col] - self.base_df[wind_total_col]
            if has_pv:
                self.base_df[residual_col] = self.base_df[residual_col] - self.base_df[pv_col]
            
            logger.info(f"  Added residual_load_forecast_de_{tso_suffix}")

    def validate_merge(self) -> dict[str, int]:
        """
        Check merge quality.
        
        Returns
        -------
        dict[str, int]
            Dictionary with validation metrics:
            - core_nulls: Number of NaN values in core columns
            - duplicates: Number of duplicate rows
            - missing_dates: Number of missing hourly timestamps
        """
        # Core columns should have no NaN
        core_cols = ["begin_date", "operator", "direction", "total_load"]
        available_core_cols = [c for c in core_cols if c in self.base_df.columns]
        core_nulls = self.base_df[available_core_cols].isnull().sum()
        
        if core_nulls.sum() > 0:
            logger.warning(f"  Core columns have NaN values:\n{core_nulls[core_nulls > 0]}")
        
        # Check for duplicate rows
        key_cols = ["begin_date", "operator", "direction"]
        available_key_cols = [c for c in key_cols if c in self.base_df.columns]
        dupes = self.base_df.duplicated(subset=available_key_cols).sum() if available_key_cols else 0
        
        if dupes > 0:
            logger.warning(f"  {dupes} duplicate rows found")
        
        # Check date continuity (simplified - just count unique dates)
        if "begin_date" in self.base_df.columns:
            try:
                date_range = pd.date_range(
                    self.base_df["begin_date"].min(),
                    self.base_df["begin_date"].max(),
                    freq="h"
                )
                # Account for multiple operators/directions
                n_groups = self.base_df.groupby(["operator", "direction"]).ngroups if "operator" in self.base_df.columns and "direction" in self.base_df.columns else 1
                expected_dates = len(date_range) * n_groups
                actual_dates = len(self.base_df)
                missing_dates = max(0, expected_dates - actual_dates)
            except Exception:
                missing_dates = 0
        else:
            missing_dates = 0
        
        if missing_dates > 0:
            logger.info(f"  Note: {missing_dates} potential missing hourly timestamps (may be expected)")
        
        return {
            "core_nulls": int(core_nulls.sum()),
            "duplicates": int(dupes),
            "missing_dates": int(missing_dates)
        }


def process_combination(args_tuple: tuple) -> Optional[dict[str, Any]]:
    """
    Worker function to process a single dataset combination.
    
    Parameters
    ----------
    args_tuple : tuple
        Tuple containing:
        - operator: TSO name
        - combo: List of feature set names
        - base_df: Base DataFrame
        - output_dir: Local output directory (or None)
        - use_wandb: Whether to upload to W&B
        - wandb_project: W&B project name
        - wandb_api_key: W&B API key
        - wandb_entity: W&B entity name
        - rolling_window_days: Rolling window size
        
    Returns
    -------
    dict or None
        Summary of the processed combination, or None if failed.
    """
    (
        operator, 
        combo, 
        base_df, 
        output_dir, 
        use_wandb, 
        wandb_project, 
        wandb_api_key,
        wandb_entity,
        rolling_window_days,
        oos_start_date,
    ) = args_tuple

    combo_name = "_".join(combo)
    process_name = multiprocessing.current_process().name
    logger.info(f"[{process_name}] Processing {operator} [{combo_name}]")
    
    # Compose Dataset
    if "remove_data_driven" in combo:
        # Data-driven features are included by default in base_df; remove if not requested
        combo = tuple(f for f in combo if f != "remove_data_driven")
        # Filter features from df
        base_df = base_df[[col for col in base_df.columns if not col.startswith("data_driven_")]]
    if "remove_runlength" in combo:
        # Run-length features are included by default in base_df; remove if not requested
        combo = tuple(f for f in combo if f != "remove_runlength")
        # Filter features from df
        base_df = base_df[[col for col in base_df.columns if not col.startswith("runlength_")]]

    composer = FeatureSetComposer(
        base_df,
        operator,
        rolling_window_days=rolling_window_days,
        oos_start_date=oos_start_date,
    )
    try:
        for feature_set in combo:
            composer.add_feature_set(feature_set)
    except Exception as e:
        logger.error(f"Error processing {operator} {combo_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    validation = composer.validate_merge()

    composer._add_production_forecast_other()
    composer._add_residual_load()

    dataset = composer.base_df
    
    # Filename handling
    operator_safe = operator.replace(" ", "_").replace("/", "_")
    filename = f"{combo_name}_{operator_safe}.parquet"
    metadata_filename = filename.replace('.parquet', '.json')
    
    meta = {
        "feature_sets": list(combo),
        "operator": operator,
        "directions": ["up", "down"],
        "rows": len(dataset),
        "columns": list(dataset.columns),
        "n_columns": len(dataset.columns),
        "validation": validation,
        "merge_log": composer.merge_log,
        "rolling_window_days": rolling_window_days,
        "creation_timestamp": datetime.now().isoformat(),
        "date_range": {
            "start": str(dataset["begin_date"].min()) if "begin_date" in dataset.columns else None,
            "end": str(dataset["begin_date"].max()) if "begin_date" in dataset.columns else None,
        }
    }

    # Local Save
    out_path = None
    if output_dir:
        out_path = Path(output_dir) / filename
        dataset.to_parquet(out_path, index=False)
        logger.info(f"  Saved to {out_path}")
        
        with open(Path(output_dir) / metadata_filename, "w") as f:
            json.dump(meta, f, indent=2)

    # WandB Save
    if use_wandb:
        try:
            import wandb
            
            # Re-login inside process
            if wandb_api_key:
                os.environ["WANDB_API_KEY"] = wandb_api_key
            
            run = wandb.init(
                project=wandb_project, 
                entity=wandb_entity,
                job_type="dataset-creation",
                reinit="create_new",
                name=f"upload_{filename.replace('.parquet', '')}",
                config=meta,
                notes=f"Dataset for {operator} with features: {', '.join(combo)}"
            )
            
            artifact_name = f"{WANDB_DATASET_ARTIFACT_PREFIX}{combo_name}_{operator_safe}"
            # Sanitize artifact name (wandb allows alphanumeric, dashes, dots, underscores)
            artifact_name = artifact_name.replace(" ", "_").lower()

            artifact = wandb.Artifact(
                name=artifact_name,
                type="dataset",
                metadata=meta
            )
            
            # If we didn't save locally, we need a temp file
            temp_path = None
            if not out_path:
                temp_path = Path(tempfile.gettempdir()) / filename
                dataset.to_parquet(temp_path, index=False)
                file_to_log = temp_path
            else:
                file_to_log = out_path
                
            artifact.add_file(str(file_to_log))
            run.log_artifact(artifact)
            run.finish()
            logger.info(f"  Uploaded to W&B: {artifact_name}")
            
            # Cleanup temp
            if temp_path and temp_path.exists():
                temp_path.unlink()
                
        except Exception as e:
            logger.error(f"W&B upload failed for {operator} {combo_name}: {str(e)}")

    return {
        "filename": filename,
        "operator": operator,
        "feature_sets": combo_name,
        "rows": len(dataset),
        "columns": len(dataset.columns),
        "validation_issues": validation["core_nulls"] + validation["duplicates"] + validation["missing_dates"]
    }


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate Redispatch Datasets with Feature Combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python generate_datasets.py --combinations-path combinations.json --output-dir ../data/model_data/
  python generate_datasets.py --combinations-path combinations.json --n-workers 4 --use-wandb
        """
    )
    parser.add_argument(
        "--combinations-path", 
        type=str, 
        required=True, 
        help="Path to JSON with feature combinations"
    )
    parser.add_argument(
        "--n-workers", 
        type=int, 
        default=1, 
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Local directory to save datasets"
    )
    parser.add_argument(
        "--use-wandb", 
        action="store_true", 
        help="Upload to Weights & Biases"
    )
    parser.add_argument(
        "--redispatch-data-path",
        type=str,
        default="../data/redispatch_data_utc_11_jan_2026.csv",
        help="Path to raw redispatch data CSV"
    )
    parser.add_argument(
        "--translations-path",
        type=str,
        default="../data_processing/translations.json",
        help="Path to translations JSON file"
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="Timezone of the redispatch data (default: UTC)"
    )
    parser.add_argument(
        "--measurement-reasons",
        type=str,
        default="domestic_redispatch",
        choices=["electricity_only_redispatch", "domestic_redispatch", "all_redispatch"],
        help="Measurement reasons category to use (default: domestic_redispatch)"
    )
    parser.add_argument(
        "--operators",
        type=str,
        nargs="+",
        default=None,
        help="TSO operators to process (default: all German TSOs)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for data filtering (YYYY-MM-DD), default: earliest available"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data filtering (YYYY-MM-DD), default: latest full month of data"
    )
    parser.add_argument(
        "--rolling-window-days",
        type=int,
        default=ROLLING_WINDOW_LAGGED_DAYS,
        help=f"Rolling window size in days for lagged features (default: {ROLLING_WINDOW_LAGGED_DAYS})"
    )
    parser.add_argument(
        "--oos-start-date",
        type=str,
        default=None,
        help="Out-of-sample start date (YYYY-MM-DD). Seasonal medians are fit on data before this date.",
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT_NAME", "redispatch-forecasting")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    
    if args.use_wandb and not WANDB_API_KEY:
        try:
            import wandb
            wandb.login()
            WANDB_API_KEY = wandb.api.api_key
        except Exception as e:
            raise ValueError(f"WANDB_API_KEY not found in environment variables and login failed: {e}")

    # Load Combinations
    combinations_path = Path(args.combinations_path)
    if not combinations_path.exists():
        raise FileNotFoundError(f"Combinations file not found: {combinations_path}")
        
    with open(combinations_path, "r") as f:
        combinations = json.load(f)
    
    logger.info(f"Loaded {len(combinations)} feature combinations from {combinations_path}")

    # Resolve redispatch data path
    redispatch_path = Path(args.redispatch_data_path)
    if not redispatch_path.is_absolute():
        redispatch_path = (Path(__file__).parent / redispatch_path).resolve()
    
    if not redispatch_path.exists():
        raise FileNotFoundError(f"Redispatch data file not found: {redispatch_path}")
    
    # Resolve translations path
    translations_path = Path(args.translations_path)
    if not translations_path.is_absolute():
        translations_path = (Path(__file__).parent / translations_path).resolve()
    
    if not translations_path.exists():
        raise FileNotFoundError(f"Translations file not found: {translations_path}")

    # Determine operators
    operators = args.operators if args.operators else GERMAN_TSOS
    operators = [normalize_tso_name(op) for op in operators]
    
    logger.info(f"Processing {len(operators)} operators: {operators}")
    logger.info(f"Using measurement reasons: {args.measurement_reasons}")

    if args.start_date:
        start_date = pd.to_datetime(args.start_date)
        logger.info(f"Filtering data from start date: {start_date}")
    else:
        start_date = None

    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
        logger.info(f"Filtering data until end date: {end_date}")
    else:
        end_date = None

    if args.oos_start_date:
        oos_start_date = pd.to_datetime(args.oos_start_date)
        logger.info(f"Using OOS start date for leakage-safe seasonal fitting: {oos_start_date}")
    else:
        oos_start_date = None

    # Load raw redispatch data using the proper function
    logger.info(f"Loading base redispatch data from {redispatch_path}...")
    logger.info(f"Using translations from {translations_path}")
    redispatch_raw = read_redispatch_data(
        file_path=str(redispatch_path),
        translations_path=str(translations_path),
        timezone=args.timezone,
        start_date=start_date,
        end_date=end_date
    )
    logger.info(f"Loaded {len(redispatch_raw)} rows of raw redispatch data")
    logger.info(f"Start date: {redispatch_raw['begin_date'].min()}, End date: {redispatch_raw['end_date'].max()}")

    # Get the measurement reasons
    relevant_reasons = TARGET_RELEVANT_MEASUREMENT_REASONS.get(
        args.measurement_reasons, 
        TARGET_RELEVANT_MEASUREMENT_REASONS["domestic_redispatch"]
    )

    # Determine date range from data if not specified
    # End date is always the last full month of data
    if start_date is None:
        start_date = redispatch_raw["begin_date"].min()
    end_date = redispatch_raw["end_date"].max()
    logger.info(f"Date range: {start_date} to {end_date}")

    # Prepare base datasets for each operator
    base_datasets = {}
    for operator in operators:
        logger.info(f"Preparing base data for {operator}...")
        try:
            base_datasets[operator] = prepare_final_target(
                redispatch=redispatch_raw,
                relevant_measurement_reasons=relevant_reasons,
                min_gap_ratio=MIN_GAP_RATIO,
                feature_set=["data_driven", "runlength"], # add data-driven and run-length features by default, then filter later
                start_date=start_date,
                end_date=end_date,
                split_method_dict=TARGET_SPLIT_METHOD_DICT,
                relevant_operators=[operator]
            )
            logger.info(f"  Base data for {operator}: {len(base_datasets[operator])} rows")
        except Exception as e:
            logger.error(f"Failed to prepare base data for {operator}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not base_datasets:
        raise ValueError("No base datasets could be prepared")

    # Prepare Tasks
    tasks = []
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for operator in operators:
        if operator not in base_datasets:
            continue
            
        base_slice = base_datasets[operator].copy()
        
        for combo in combinations:
            tasks.append((
                operator, 
                tuple(combo),  # Convert to tuple for hashability
                base_slice, 
                args.output_dir, 
                args.use_wandb,
                WANDB_PROJECT,
                WANDB_API_KEY,
                WANDB_ENTITY,
                args.rolling_window_days,
                oos_start_date,
            ))

    logger.info(f"Prepared {len(tasks)} tasks")

    # Concurrency Control
    max_workers = args.n_workers
    if args.output_dir and max_workers > 2:
        logger.warning("Limiting workers to 2 to prevent I/O bottlenecks with local storage.")
        max_workers = 2

    logger.info(f"Starting execution with {max_workers} workers for {len(tasks)} tasks...")
    
    if max_workers == 1:
        # Single-threaded execution for easier debugging
        results = [process_combination(task) for task in tasks]
    else:
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = pool.map(process_combination, tasks)
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    logger.info(f"Completed {len(results)} / {len(tasks)} tasks successfully")
    
    # Save Summary
    if results:
        summary_df = pd.DataFrame(results)
        
        if args.output_dir:
            summary_path = Path(args.output_dir) / "dataset_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Summary saved to {summary_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATASET GENERATION SUMMARY")
        print("=" * 60)
        print(f"Total datasets created: {len(results)}")
        print(f"Total rows across all datasets: {summary_df['rows'].sum():,}")
        print(f"Datasets with validation issues: {(summary_df['validation_issues'] > 0).sum()}")
        print("\nDatasets by operator:")
        print(summary_df.groupby("operator")["filename"].count())
        print("\nDatasets by feature set:")
        print(summary_df.groupby("feature_sets")["filename"].count())
    else:
        logger.warning("No datasets were successfully created")


if __name__ == "__main__":
    main()
