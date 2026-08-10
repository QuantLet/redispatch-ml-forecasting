python -m paper_results_cli.plot_best_checkpoint_windows \
  --predictions-dir outputs_paper \
  --dataset-name basic_day_ahead_price_wind_pv_production_consumption_sce \
  --dataset-dir data/model_data_paper_actuals_1h_lag \
  --tso TenneT_DE \
  --windows 2 3 10 11 \
  --models lightgbm_regression_rolling \
  --output-dir paper_results/figures_benchmark
