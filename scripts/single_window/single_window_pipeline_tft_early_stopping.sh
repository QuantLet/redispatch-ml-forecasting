#!/bin/bash

for tso in "50Hertz" "Amprion"; do
    echo "Running training for TSO: $tso"
    
    # Train the model
    python3 -m training.train_pipeline \
        --dataset-path "data/model_data_extended/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce_${tso}.parquet" \
        --config "training/tft_early_stopping_parameters.yaml" \
        --n-threads 30 || echo "Training failed for $tso, continuing..."
done