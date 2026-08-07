# Dataset Preparation Pipeline

This directory contains the complete pipeline for generating redispatch forecasting datasets with various feature combinations.

## Overview

The pipeline orchestrates:
1. Loading and processing base redispatch data
2. Loading multiple feature sets (prices, production, consumption, wind/PV, cross-border flows, Bloomberg data)
3. Merging features with proper validation
4. Generating datasets for all TSO operators
5. Saving datasets locally and/or uploading to Weights & Biases

## Files

- **`generate_datasets.py`**: Main script for dataset generation with parallel processing
- **`run_generation.sh`**: Helper bash script with convenient defaults
- **`combinations.json`**: Configuration file defining feature set combinations
- **`evaluate_datasets.ipynb`**: Jupyter notebook for analyzing and visualizing generated datasets
- **`redispatch_core.py`**: Core functions for redispatch target preparation
- **`tso_config.py`**: TSO configuration and neighbor mappings
- **`feature_loaders/`**: Modules for loading different feature sets

## Quick Start

### 1. Basic Usage

Generate datasets for all TSOs with all feature combinations:

```bash
cd /path/to/redispatch-ml-forecasting
./dataset_preparation/run_generation.sh
```

### 2. Custom Configuration

```bash
./dataset_preparation/run_generation.sh \
    --workers 2 \
    --output-dir data/my_datasets/ \
    --combinations dataset_preparation/combinations.json
```

### 3. Single Operator Test

```bash
python3 -m dataset_preparation.generate_datasets \
    --combinations-path dataset_preparation/combinations_test.json \
    --operators "50Hertz" \
    --output-dir data/test_output/
```

## Feature Combinations

The `combinations.json` file defines which feature sets to combine. Example:

```json
[
    ["basic"],
    ["basic", "day_ahead_price"],
    ["basic", "wind_pv"],
    ["basic", "day_ahead_price", "wind_pv", "cross_border", "production_consumption"]
]
```

Available feature sets:
- **`basic`**: Data-driven features from redispatch history
- **`day_ahead_price`**: Day-ahead electricity prices
- **`production_consumption`**: Generation and load data
- **`wind_pv`**: Wind and PV generation features
- **`cross_border`**: Cross-border flow data
- **`bloomberg`**: Bloomberg financial data (gas, coal, carbon, power prices)

## Command-Line Options

### `generate_datasets.py`

```
--combinations-path PATH          Path to combinations JSON (required)
--n-workers N                     Number of parallel workers (default: 1)
--output-dir DIR                  Local directory for datasets
--use-wandb                       Upload to Weights & Biases
--redispatch-data-path PATH       Path to redispatch CSV
--translations-path PATH          Path to translations JSON
--timezone TIMEZONE               Data timezone (default: Europe/Berlin)
--measurement-reasons TYPE        Filter type (default: domestic_redispatch)
--operators TSO [TSO ...]         Specific TSOs to process
--rolling-window-days N           Rolling window size (default: 7)
```

### `run_generation.sh`

```
-w, --workers N          Number of parallel workers
-o, --output-dir DIR     Output directory
-c, --combinations FILE  Combinations JSON path
--wandb                  Enable W&B upload
-r, --redispatch-data    Redispatch data path
-t, --translations       Translations JSON path
-z, --timezone           Timezone
-m, --reasons TYPE       Measurement reasons type
-h, --help               Show help
```

## Output Structure

Generated datasets are saved as:

```
output_dir/
├── basic_50Hertz.parquet
├── basic_50Hertz.json                          # Metadata
├── basic_day_ahead_price_50Hertz.parquet
├── basic_day_ahead_price_50Hertz.json
├── ...
└── dataset_summary.csv                         # Summary of all datasets
```

### Metadata Files

Each `.json` file contains:
- Feature sets included
- Number of rows and columns
- Date range
- Validation results
- Merge statistics
- Creation timestamp

## Data Validation

The pipeline performs automatic validation:
- **Core columns**: Checks for missing values in essential columns
- **Duplicates**: Detects duplicate rows
- **Date continuity**: Identifies missing timestamps
- **Merge quality**: Tracks null percentages after feature merging

## Evaluation Notebook

Use `evaluate_datasets.ipynb` to:
- Load and inspect dataset summaries
- Visualize data distributions
- Compare datasets across feature combinations
- Analyze validation issues
- Generate analysis reports

## Configuration Constants

Key constants defined in `redispatch_core.py`:

```python
ROLLING_WINDOW_DAYS = 7                           # Rolling window size
WANDB_DATASET_ARTIFACT_PREFIX = "redispatch_dataset_"
MIN_GAP_RATIO = -0.1                              # Duration gap tolerance
```

## Measurement Reasons

Three categories available:
- **`electricity_only_redispatch`**: Electricity-related only
- **`domestic_redispatch`**: Electricity and voltage-related (default)
- **`all_redispatch`**: Including cross-border countertrade

## TSO Operators

Four German transmission system operators:
- **50Hertz**: East Germany
- **TenneT DE**: North and South Germany
- **Amprion**: West Germany
- **TransnetBW**: Southwest Germany

## Performance Tips

1. **Parallelization**: Use 2-4 workers for local storage, more for W&B-only
2. **Memory**: Each worker processes one dataset at a time
3. **I/O**: Local disk writes are throttled to prevent bottlenecks
4. **Testing**: Use `combinations_test.json` with one operator for quick tests

## Examples

### Generate All Datasets Locally

```bash
python3 -m dataset_preparation.generate_datasets \
    --combinations-path dataset_preparation/combinations.json \
    --n-workers 2 \
    --output-dir data/model_data_multi_feature_combinations/
```

### Upload to Weights & Biases

```bash
# Set up W&B credentials first
export WANDB_API_KEY="your_api_key"
export WANDB_PROJECT_NAME="redispatch-forecasting"
export WANDB_ENTITY="your_entity"

python3 -m dataset_preparation.generate_datasets \
    --combinations-path dataset_preparation/combinations.json \
    --use-wandb
```

### Process Specific TSOs

```bash
python3 -m dataset_preparation.generate_datasets \
    --combinations-path dataset_preparation/combinations.json \
    --operators "50Hertz" "TenneT DE" \
    --output-dir data/selected_tsos/
```

## Troubleshooting

### Import Errors

Run from project root:
```bash
cd /path/to/redispatch-ml-forecasting
python3 -m dataset_preparation.generate_datasets ...
```

### Missing Data Files

Check that required data files exist:
- Redispatch data: `data/redispatch_data_3_jan_2026.csv`
- Translations: `data_processing/translations.json`
- Feature data: Files in `data/` subdirectories

### Memory Issues

Reduce the number of workers or process fewer operators at once:
```bash
--n-workers 1 --operators "50Hertz"
```

## Next Steps

After generating datasets:
1. Run `evaluate_datasets.ipynb` to analyze results
2. Implement imputation strategies for missing values
3. Train forecasting models on generated datasets
4. Compare model performance across feature combinations
