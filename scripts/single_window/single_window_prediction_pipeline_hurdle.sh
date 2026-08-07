#!/bin/bash

# TenneT_DE
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window_hurdle/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/tennet_de_hurdle/2026-02-25_15-47-05/"
python3 -m training.prediction_pipeline_hurdle_new \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}/evaluation" \
    --dataset-root-dir "data/model_data_extended" \
    --wandb-project "redispatch-forecasting-proper" \
    --n-threads 40 \
    --start-date "2025-03-01" \
    --benchmarks "ridge_regression" \
    --benchmark-config-path "training/benchmark_config.yaml" || echo "Evaluation failed for 50Hertz, continuing..."

# 50Hertz
# output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window_tft_early_stopping/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/50hertz/2026-02-25_10-42-18/"
# python3 -m training.prediction_pipeline_new \
#     --model-path "${output_dir}" \
#     --output-dir "${output_dir}/evaluation" \
#     --dataset-root-dir "data/model_data_extended" \
#     --wandb-project "redispatch-forecasting-proper" \
#     --n-threads 40 \
#     --start-date "2025-03-01" \
#     --benchmark-config-path "training/benchmark_config.yaml" || echo "Evaluation failed for 50Hertz, continuing..."

# # Amprion
# output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window_tft_early_stopping/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/amprion/2026-02-25_11-44-30"
# python3 -m training.prediction_pipeline_new \
#     --model-path "${output_dir}" \
#     --output-dir "${output_dir}/evaluation" \
#     --dataset-root-dir "data/model_data_extended" \
#     --wandb-project "redispatch-forecasting-proper" \
#     --n-threads 40 \
#     --start-date "2025-03-01" \
#     --benchmark-config-path "training/benchmark_config.yaml" || echo "Evaluation failed for Amprion, continuing..."
