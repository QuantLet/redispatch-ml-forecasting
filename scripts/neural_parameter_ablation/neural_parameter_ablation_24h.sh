#!/bin/bash

for tso in "TenneT_DE" "TransnetBW"; do
    echo "Running training for TSO: $tso"
    
    # Train the model
    python3 -m training.rolling_window_validation_predictions \
        --dataset-path "data/model_data_extended/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce_${tso}.parquet" \
        --output-dir /home/jovyan/redispatch-ml-forecasting/neural_parameter_ablation_established_input_size_amprion \
        --models "nbeatsx, nhits, tft, tft_quantile" \
        --config-path training/ablation_neural_config.yaml \
        --input-sizes 24 \
        --early-stop-buffer 30 \
        --k-windows 3 \
        --max-steps 5000 \
        --early-stop-patience 20 \
        --shift-hours 6 \
        --n-threads 20 \
        --persist-models || echo "Training failed for $tso, continuing..."

done