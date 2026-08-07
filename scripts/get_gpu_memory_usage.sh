#/bin/bash
for tso in "TenneT_DE" "50Hertz" "Amprion" "TransnetBW"; do
    echo "Running training for TSO: $tso"

    python3 -m training.gpu_memory_usage \
        --model-path outputs/${tso,,}/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce \
        --output-dir outputs/${tso,,}/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/predictions \
        --wandb-project redispatch-forecasting-proper \
        --wandb-entity vlad-bolovaneanu-ase-bues \
        --per-window-aggregation "q95" \
        --start-window 2  || echo "Failed for $tso, continuing..."
done