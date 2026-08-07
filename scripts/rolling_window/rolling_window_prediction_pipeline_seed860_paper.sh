#!/bin/bash

for tso in "TenneT_DE" "50Hertz" "Amprion" "TransnetBW"; do
    echo "Running training for TSO: $tso"
    python3 -m training.prediction_pipeline_rolling_window \
        --model-path "outputs_nhits_only_seed860/${tso,,}/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/" \
        --dataset-root-dir "data/model_data_extended/" \
        --output-dir "outputs_nhits_only_seed860/${tso,,}/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/evaluation" \
        --wandb-project "redispatch-forecasting-proper" \
        --skip-benchmarks \
        --start-window 2 || echo "Failed for $tso, continuing..."
done