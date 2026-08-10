#!/bin/bash

for tso in "TenneT_DE" "50Hertz" "Amprion" "TransnetBW"; do
    echo "Running training for TSO: $tso"
    python3 -m training.circular_shift_training \
        --dataset-path "data/model_data_paper_actuals_1h_lag/basic_day_ahead_price_wind_pv_production_consumption_sce_${tso}.parquet" \
        --config "training/neural_model_parameters_shift.yaml" \
        --shift-k-days 17 \
        --rolling-window \
        --start-window 2 \
        --n-threads 30 \
        --max-steps 5000 \
        --early-stop-patience 20 \
        --persist-checkpoints \
        --checkpoint-compression 19 \
        --checkpoint-compression-n-threads 30 || echo "Failed for $tso, continuing..."
done
