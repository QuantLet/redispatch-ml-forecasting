#!/bin/bash

for tso in "Amprion" "50Hertz" "TenneT_DE" "TransnetBW"; do
    echo "Running training for TSO: $tso"
    
    # Train the model
    python3 -m training.train_pipeline_per_tso_direction \
        --dataset-path "data/model_data_new_features_debug/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce_${tso}.parquet" \
        --config "training/neural_model_parameters.yaml" \
        --output-dir /home/jovyan/redispatch-ml-forecasting/outputs_single_model_per_direction_new \
        --n-threads 30 || echo "Training failed for $tso, continuing..."
done