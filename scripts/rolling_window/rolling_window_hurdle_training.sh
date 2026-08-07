#!/bin/bash

for tso in "TenneT_DE" "50Hertz" "Amprion" "TransnetBW"; do
    echo "Running training for TSO: $tso"
    
    # Train the model
    python3 -m training.hurdle_runner \
        --dataset-path "data/model_data_extended/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce_${tso}.parquet" \
        --rolling-window \
        --config "training/hurdle_config_rolling_window.yaml" \
        --n-threads 30 || echo "Training failed for $tso, continuing..."
done