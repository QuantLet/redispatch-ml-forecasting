#!/bin/bash

for tso in "TenneT_DE" "50Hertz" "Amprion" "TransnetBW"; do
    echo "Running training for TSO: $tso"
    python3 -m training.prediction_pipeline_rolling_window \
        --model-path "outputs_paper/${tso,,}/basic_day_ahead_price_wind_pv_production_consumption_sce/" \
        --dataset-root-dir "data/model_data_paper_actuals_1h_lag/" \
        --output-dir "outputs_paper/${tso,,}/basic_day_ahead_price_wind_pv_production_consumption_sce/evaluation" \
        --wandb-project "redispatch-forecasting-proper" \
        --benchmark-config-path "training/benchmark_config.yaml" \
        --checkpoint-selection "both" \
        --start-window 2 || echo "Failed for $tso, continuing..."
done