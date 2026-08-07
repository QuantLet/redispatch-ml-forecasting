#!/bin/bash

for tso in "50Hertz" "Amprion" "TransnetBW"; do
    echo "Running training for TSO: $tso"
    python3 -m training.circular_shift_prediction \
        --dataset-path "data/model_data_extended/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce_${tso}.parquet" \
        --output-dir "outputs_shifted_targets_17/${tso,,}/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/evaluation" \
        --model-dir "outputs_shifted_targets_17/${tso,,}/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce" \
        --wandb-project "redispatch-forecasting-proper" \
        --start-window 2 \
        --rolling-window \
        --n-threads 30  || echo "Failed for $tso, continuing..."
done