#!/bin/bash

# TenneT DE
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs/tennet_de/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce"
python3 -m training.explainability_pipeline \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}" \
    --wandb-project "redispatch-forecasting-proper" \
    --model-filter "seed" \
    --start-window 2 \
    --mode absolute || echo "Evaluation failed for tenneT_DE, continuing..."

# 50Hertz
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs/50hertz/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce"
python3 -m training.explainability_pipeline \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}" \
    --wandb-project "redispatch-forecasting-proper" \
    --model-filter "seed" \
    --start-window 2 \
    --mode absolute || echo "Evaluation failed for tenneT_DE, continuing..."

# Amprion
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs/amprion/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce"
python3 -m training.explainability_pipeline \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}" \
    --wandb-project "redispatch-forecasting-proper" \
    --model-filter "seed" \
    --mode absolute || echo "Evaluation failed for Amprion, continuing..."

# TransnetBW
output_dir="/home/jovyan/redispatch-ml-forecasting/outputs/transnetbw/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce"
python3 -m training.explainability_pipeline \
    --model-path "${output_dir}" \
    --output-dir "${output_dir}" \
    --wandb-project "redispatch-forecasting-proper" \
    --model-filter "seed" \
    --mode absolute || echo "Evaluation failed for TransnetBW, continuing..."