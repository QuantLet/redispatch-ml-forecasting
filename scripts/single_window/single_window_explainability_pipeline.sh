#!/bin/bash

# Amprion
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/amprion/2026-02-15_05-12-28"
python3 -m training.explainability_pipeline \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}" \
    --wandb-project "redispatch-forecasting" \
    --model-filter "seed" \
    --mode absolute || echo "Evaluation failed for Amprion, continuing..."

# TransnetBW
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/transnetbw/2026-02-15_06-48-35"
python3 -m training.explainability_pipeline \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}" \
    --wandb-project "redispatch-forecasting" \
    --model-filter "seed" \
    --mode absolute || echo "Evaluation failed for TransnetBW, continuing..."

# TenneT DE
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/tennet_de/2026-02-15_02-05-44"
python3 -m training.explainability_pipeline \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}" \
    --wandb-project "redispatch-forecasting" \
    --model-filter "seed" \
    --mode absolute || echo "Evaluation failed for tenneT_DE, continuing..."

# 50Hertz
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs_single_window/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/50hertz/2026-02-15_03-12-33"
python3 -m training.explainability_pipeline \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}" \
    --wandb-project "redispatch-forecasting" \
    --model-filter "seed" \
    --mode absolute || echo "Evaluation failed for 50Hertz, continuing..."