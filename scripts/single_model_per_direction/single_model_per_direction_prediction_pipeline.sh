#!/bin/bash

# TenneT DE
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_model_per_direction_new/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/tennet_de"
python3 -m training.prediction_pipeline_per_direction \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}/evaluation" \
    --dataset-root-dir "data/model_data_extended" \
    --n-threads 40 \
    --start-date "2025-03-01" || echo "Evaluation failed for tenneT_DE, continuing..."

# 50Hertz
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_model_per_direction_new/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/50hertz"
python3 -m training.prediction_pipeline_per_direction \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}/evaluation" \
    --dataset-root-dir "data/model_data_extended" \
    --n-threads 40 \
    --start-date "2025-03-01" || echo "Evaluation failed for 50Hertz, continuing..."

# Amprion
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_model_per_direction_new/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/amprion"
python3 -m training.prediction_pipeline_per_direction \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}/evaluation" \
    --dataset-root-dir "data/model_data_extended" \
    --n-threads 40 \
    --start-date "2025-03-01" || echo "Evaluation failed for Amprion, continuing..."

# TransnetBW
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_model_per_direction_new/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/transnetbw"
python3 -m training.prediction_pipeline_per_direction \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}/evaluation" \
    --dataset-root-dir "data/model_data_extended" \
    --n-threads 40 \
    --start-date "2025-03-01" || echo "Evaluation failed for TransnetBW, continuing..."
