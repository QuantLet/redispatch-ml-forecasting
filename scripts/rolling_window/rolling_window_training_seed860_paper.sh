#!/bin/bash

for tso in "TenneT_DE" "50Hertz" "Amprion" "TransnetBW"; do
    echo "Running training for TSO: $tso"
    
    # Train the model
    python3 -m training.train_pipeline \
        --dataset-path "data/model_data_paper_actuals_1h_lag/basic_day_ahead_price_wind_pv_production_consumption_sce_${tso}.parquet" \
        --config "training/neural_model_parameters_nhits.yaml" \
        --output-dir "outputs_nhits_only_seed860_paper" \
        --rolling-window \
        --start-window 2 \
        --random-seed 860 \
        --n-threads 30 || echo "Training failed for $tso, continuing..."
done