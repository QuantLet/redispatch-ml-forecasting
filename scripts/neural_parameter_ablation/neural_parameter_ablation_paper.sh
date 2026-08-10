#!/bin/bash

for tso in "Amprion" "TransnetBW"; do  # "TenneT_DE" "50Hertz"; do #
    echo "Running training for TSO: $tso"
    
    # Train the model
    python3 -m training.rolling_window_validation_predictions \
        --dataset-path "data/model_data_paper_actuals_1h_lag/basic_day_ahead_price_wind_pv_production_consumption_sce_${tso}.parquet" \
        --output-dir /home/jovyan/redispatch-ml-forecasting/neural_parameter_ablation_paper \
        --models "nbeatsx, nhits, tft, lstm" \
        --config-path training/ablation_neural_config.yaml \
        --input-sizes 24 \
        --k-windows 2 \
        --max-steps 5000 \
        --early-stop-patience 20 \
        --shift-hours 6 \
        --n-threads 20 \
        --eval-checkpoint-type best \
        --persist-models || echo "Training failed for $tso, continuing..."

done
