#!/bin/bash
# Run the dataset generation script with common configurations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default values
N_WORKERS=1
OUTPUT_DIR="data/model_data_multi_feature_combinations"
COMBINATIONS_PATH="dataset_preparation/combinations.json"
USE_WANDB=""
REDISPATCH_DATA="/home/jovyan/redispatch-ml-forecasting/data/redispatch_data_3_jan_2026.csv"
TRANSLATIONS_PATH="/home/jovyan/redispatch-ml-forecasting/data_processing/translations.json"
TIMEZONE="Europe/Berlin"
MEASUREMENT_REASONS="domestic_redispatch"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers|-w)
            N_WORKERS="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --combinations|-c)
            COMBINATIONS_PATH="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB="--use-wandb"
            shift
            ;;
        --redispatch-data|-r)
            REDISPATCH_DATA="$2"
            shift 2
            ;;
        --translations|-t)
            TRANSLATIONS_PATH="$2"
            shift 2
            ;;
        --timezone|-z)
            TIMEZONE="$2"
            shift 2
            ;;
        --reasons|-m)
            MEASUREMENT_REASONS="$2"
            shift 2
            ;;
        --start-date|-s)
            START_DATE="$2"
            shift 2
            ;;
        --end-date|-e)
            END_DATE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -w, --workers N          Number of parallel workers (default: 1)"
            echo "  -o, --output-dir DIR     Output directory (default: data/model_data_multi_feature_combinations)"
            echo "  -c, --combinations FILE  Path to combinations JSON (default: dataset_preparation/combinations.json)"
            echo "  --wandb                  Enable Weights & Biases upload"
            echo "  -r, --redispatch-data    Path to redispatch CSV data (default: data/redispatch_data_3_jan_2026.csv)"
            echo "  -t, --translations       Path to translations JSON (default: data_processing/translations.json)"
            echo "  -z, --timezone           Timezone of data (default: Europe/Berlin)"
            echo "  -s, --start-date DATE    Start date of data (default: earliest date in redispatch data)"
            echo "  -e, --end-date DATE      End date of data (default: last full month of redispatch data)"
            echo "  -m, --reasons TYPE       Measurement reasons category (default: domestic_redispatch)"

            echo "  -h, --help               Show this help message"
            echo ""
            echo "Measurement reasons options:"
            echo "  electricity_only_redispatch"
            echo "  domestic_redispatch (default)"
            echo "  all_redispatch"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Redispatch Dataset Generation"
echo "========================================"
echo "Project root:      $PROJECT_ROOT"
echo "Workers:           $N_WORKERS"
echo "Output directory:  $OUTPUT_DIR"
echo "Translations:      $TRANSLATIONS_PATH"
echo "Timezone:          $TIMEZONE"
echo "Measurement type:  $MEASUREMENT_REASONS"
echo "W&B upload:        ${USE_WANDB:-disabled}"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the generation script as a module
python3 -m dataset_preparation.generate_datasets \
    --combinations-path "$COMBINATIONS_PATH" \
    --n-workers "$N_WORKERS" \
    --output-dir "$OUTPUT_DIR" \
    --redispatch-data-path "$REDISPATCH_DATA" \
    --translations-path "$TRANSLATIONS_PATH" \
    --timezone "$TIMEZONE" \
    --measurement-reasons "$MEASUREMENT_REASONS" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    $USE_WANDB

echo ""
echo "========================================"
echo "Generation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"