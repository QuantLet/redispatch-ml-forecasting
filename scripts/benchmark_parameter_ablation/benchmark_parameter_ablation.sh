for tso in "TenneT_DE" "50Hertz" "Amprion" "TransnetBW"; do
    echo "Running training for TSO: $tso"
    
    # Train the model
    python3 -m training.benchmark_ablation_validation_predictions \
        --dataset-path "data/model_data_paper_actuals_1h_lag/basic_day_ahead_price_wind_pv_production_consumption_sce_${tso}.parquet" \
        --output-dir /home/jovyan/redispatch-ml-forecasting/benchmark_ablation_paper_v2/${tso} \
        --models "ridge, lightgbm, croston" \
        --input-size 24 \
        --early-stopping-rounds 200 \
        --k-windows 2 \
        --shift-hours 6 \
        --lgb-device gpu \
        --n-jobs 50 || echo "Training failed for $tso, continuing..."

done