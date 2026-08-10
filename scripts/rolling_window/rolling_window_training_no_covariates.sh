#!/bin/bash

for tso in "TenneT_DE" "50Hertz" "Amprion" "TransnetBW"; do
    echo "Running training for TSO: $tso"
    
    # Train the model
    python3 -m training.train_pipeline \
        --dataset-path "data/model_data_paper_actuals_1h_lag/basic_remove_data_driven_remove_runlength_${tso}.parquet" \
        --config "training/neural_model_parameters_nhits.yaml" \
        --output-dir "outputs_no_covariates_best" \
        --rolling-window \
        --start-window 2 \
        --checkpoint-selection best \
        --n-threads 30 || echo "Training failed for $tso, continuing..."
done
