#!/bin/bash

for tso in "TenneT_DE" "50Hertz" "Amprion" "TransnetBW"; do
    echo "Running training for TSO: $tso"
    
    # Train the model
    python3 -m training.train_pipeline \
        --dataset-path "data/model_data_paper_actuals_1h_lag/basic_day_ahead_price_wind_pv_production_consumption_sce_${tso}.parquet" \
        --config "training/neural_model_parameters.yaml" \
        --output-dir outputs_single_window_paper_1h_actuals_lag \
        --eval-dir outputs_single_window_paper_1h_actuals_lag/evaluation \
        --n-threads 30 || echo "Training failed for $tso, continuing..."
done
