#!/bin/bash

# TenneT DE
# output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window_new/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/tennet_de/2026-02-21_19-04-46/"
python3 -m training.prediction_pipeline_new \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}/evaluation" \
    --dataset-root-dir "data/model_data_extended" \
    --wandb-project "redispatch-forecasting-proper" \
    --n-threads 40 \
    --start-date "2025-03-01" \
    --benchmark-config-path "training/benchmark_config.yaml" || echo "Evaluation failed for tenneT_DE, continuing..."

# 50Hertz
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window_new/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/50hertz/2026-02-21_20-10-16/"
python3 -m training.prediction_pipeline_new \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}/evaluation" \
    --dataset-root-dir "data/model_data_extended" \
    --wandb-project "redispatch-forecasting-proper" \
    --n-threads 40 \
    --start-date "2025-03-01" \
    --benchmark-config-path "training/benchmark_config.yaml" || echo "Evaluation failed for 50Hertz, continuing..."

# Amprion
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window_new/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/amprion/2026-02-21_21-10-51/"
python3 -m training.prediction_pipeline_new \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}/evaluation" \
    --dataset-root-dir "data/model_data_extended" \
    --wandb-project "redispatch-forecasting-proper" \
    --n-threads 40 \
    --start-date "2025-03-01" \
    --benchmark-config-path "training/benchmark_config.yaml" || echo "Evaluation failed for Amprion, continuing..."

# TransnetBW
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window_new/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/transnetbw/2026-02-21_22-34-57/"
python3 -m training.prediction_pipeline_new \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}/evaluation" \
    --dataset-root-dir "data/model_data_extended" \
    --wandb-project "redispatch-forecasting-proper" \
    --n-threads 40 \
    --start-date "2025-03-01" \
    --benchmark-config-path "training/benchmark_config.yaml" || echo "Evaluation failed for TransnetBW, continuing..."
