#!/bin/bash

# Amprion
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/amprion/2026-02-11_01-45-58"
python3 -m training.prediction_pipeline \
    --model-path "${output_dir}" \
    --dataset-root-dir "data/model_data_new_features_debug" \
    --wandb-project "redispatch-forecasting" \
    --test-best-checkpoint \
    --skip-predictions \
    --explain \
    --n-threads 20 \
    --output-dir "${output_dir}/evaluation" || echo "Evaluation failed for Amprion, continuing..."

# TransnetBW
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/transnetbw/2026-02-11_02-18-59"
python3 -m training.prediction_pipeline \
    --model-path "${output_dir}" \
    --dataset-root-dir "data/model_data_new_features_debug" \
    --wandb-project "redispatch-forecasting" \
    --test-best-checkpoint \
    --skip-predictions \
    --explain \
    --n-threads 20 \
    --output-dir "${output_dir}/evaluation" || echo "Evaluation failed for TransnetBW, continuing..."

# TenneT DE
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/tennet_de/2026-02-11_00-47-09"
python3 -m training.prediction_pipeline \
    --model-path "${output_dir}" \
    --dataset-root-dir "data/model_data_new_features_debug" \
    --wandb-project "redispatch-forecasting" \
    --test-best-checkpoint \
    --skip-predictions \
    --explain \
    --n-threads 20 \
    --output-dir "${output_dir}/evaluation" || echo "Evaluation failed for tenneT_DE, continuing..."

# 50Hertz
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/50hertz/2026-02-11_01-14-22"
python3 -m training.prediction_pipeline \
    --model-path "${output_dir}" \
    --dataset-root-dir "data/model_data_new_features_debug" \
    --wandb-project "redispatch-forecasting" \
    --test-best-checkpoint \
    --skip-predictions \
    --explain \
    --n-threads 20 \
    --output-dir "${output_dir}/evaluation" || echo "Evaluation failed for 50Hertz, continuing..."