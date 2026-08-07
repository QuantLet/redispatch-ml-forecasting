# TenneT_DE
python3 -m training.prediction_pipeline_hurdle_rolling_window \
    --dataset-root-dir "data/model_data_extended" \
    --model-path "outputs_single_window_hurdle/tennet_de_hurdle/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce" \
    --output-dir "outputs_single_window_hurdle/tennet_de_hurdle/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/evaluation" \
    --wandb-project "redispatch-forecasting-proper" \
    --benchmarks "ridge_regression" \
    --hurdle-theta-fbeta2-target 0.8 \
    --hurdle-theta-lambda 50.0 \
    --n-threads 30 || echo "Prediction pipeline failed for TenneT_DE, continuing..."

# 50Hertz
python3 -m training.prediction_pipeline_hurdle_rolling_window \
    --dataset-root-dir "data/model_data_extended" \
    --model-path "outputs_single_window_hurdle/50hertz_hurdle/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce" \
    --output-dir "outputs_single_window_hurdle/50hertz_hurdle/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/evaluation" \
    --wandb-project "redispatch-forecasting-proper" \
    --benchmarks "ridge_regression" \
    --hurdle-theta-fbeta2-target 0.8 \
    --hurdle-theta-lambda 50.0 \
    --n-threads 30 || echo "Prediction pipeline failed for 50hertz, continuing..."


# Amprion
python3 -m training.prediction_pipeline_hurdle_rolling_window \
    --dataset-root-dir "data/model_data_extended" \
    --model-path "outputs_single_window_hurdle/amprion_hurdle/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce" \
    --output-dir "outputs_single_window_hurdle/amprion_hurdle/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/evaluation" \
    --wandb-project "redispatch-forecasting-proper" \
    --benchmarks "ridge_regression" \
    --hurdle-theta-fbeta2-target 0.8 \
    --hurdle-theta-lambda 50.0 \
    --n-threads 30 || echo "Prediction pipeline failed for Amprion, continuing..."


# TransnetBW
python3 -m training.prediction_pipeline_hurdle_rolling_window \
    --dataset-root-dir "data/model_data_extended" \
    --model-path "outputs_single_window_hurdle/transnetbw_hurdle/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce" \
    --output-dir "outputs_single_window_hurdle/transnetbw_hurdle/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce/evaluation" \
    --wandb-project "redispatch-forecasting-proper" \
    --benchmarks "ridge_regression" \
    --hurdle-theta-fbeta2-target 0.8 \
    --hurdle-theta-lambda 50.0 \
    --n-threads 30 || echo "Prediction pipeline failed for TransnetBW, continuing..."