"""
Benchmark models for redispatch forecasting.

Provides a unified interface for computing baseline / benchmark predictions
that can be compared against the neural models trained via ``training.runner``.

Supported benchmarks
--------------------
* **NaiveSeasonal** – repeats observed values from *K* hours ago (default K = forecast_horizon).
* **LinearRegression** – Darts ``LinearRegressionModel`` with optional lag optimisation.
* **GradientBoostedTrees** – ``LightGBMModel`` (or ``XGBModel`` fallback) with the same
  covariate / lag structure as the linear regression benchmark.
* **AutoARIMA** – ``AutoARIMA`` with optional exogenous (future) covariates.  Supports
  both single-shot and rolling-window modes.
* **SeasonalRegression** – a simple ``LinearRegressionModel`` that uses *only* Fourier
  / calendar features (hour-of-day, day-of-week, month-of-year) as future covariates,
  capturing seasonality without any data-driven covariates.

Shift-awareness (``output_chunk_shift``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All benchmark models operate on **unshifted** (physical-time) data.  When
``shift_hours > 0``, Darts' ``output_chunk_shift`` parameter creates the
required gap between input context and predicted output – no manual dataset
shifting, timestamp relabelling or post-prediction correction is needed.
``NaiveSeasonal`` and ``AutoARIMA`` do not support ``output_chunk_shift`` and
always predict in physical time directly.

Training split and scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~
Benchmark models are trained on the *training* split ``[train_start, valid_start)``.
For sklearn-backed Darts linear benchmarks (Ridge, LASSO, ElasticNet and seasonal
regression), standard scalers are fit on train+validation data up to ``test_start``
(excluding test), matching NeuralForecast ``local_scaler_type="standard"`` usage;
predictions are inverse-transformed before returning.

Rolling window
~~~~~~~~~~~~~~
``run_benchmarks`` accepts an optional list of ``WindowBoundary`` objects
(from ``training.runner``) so that benchmark models are retrained on each window,
exactly mirroring the neural rolling-window scheme.

Lag optimisation (``shift_hours == 0`` only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``optimize_lags`` performs a grid search over historical-covariate and
future-covariate lag depths (12–48h each, 37×37 combinations) evaluated on the
validation set with MAE, returning the best ``(hist_lags, futr_lags)`` pair.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
import lightgbm as lgb

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Optional, Literal, cast

import numpy as np
import pandas as pd

from darts import TimeSeries, concatenate
from darts.metrics import mae as darts_mae
from darts.models import NaiveSeasonal, LinearRegressionModel
from darts.utils.model_selection import train_test_split
from darts.explainability import ShapExplainer
import yaml
from sklearn.preprocessing import StandardScaler

try:
    from darts.models import AutoARIMA
except ImportError:
    from darts.models.forecasting.auto_arima import AutoARIMA  # type: ignore[no-redef]

# Pre-compiled regex used by aggregate_shap_importance
_LAG_SUFFIX_RE = re.compile(r"_lag-?\d+$")

from training.data_prep import (
    CALENDAR_COLS,
    build_static_df,
    classify_covariates,
)

from training.runner import _compute_rolling_windows as compute_rolling_windows, WindowBoundary
from training.benchmark_ablation_validation_predictions import (
    RidgeRegressionModel,
    LassoRegressionModel,
    ElasticNetRegressionModel,
)
from training.train_pipeline import _normalize_tso_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# ── Try to import LightGBM; fall back to XGBoost if unavailable ───────────────
try:
    from darts.models import LightGBMModel
except ImportError:
    raise ImportError("LightGBM is not installed. Please install LightGBM to use the GradientBoostedTrees benchmark.") 


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class BenchmarkConfig:
    """All knobs needed to drive the benchmark computation."""

    forecast_horizon: int = 24
    input_size: int = 24

    shift_hours: int = 0

    tso: str = "TenneT_DE"

    random_seed: int = 42

    test_start: str = "2025-03-01"
    valid_start: str = "2024-11-01"
    train_start: str = "2021-10-01"

    add_calendar: bool = True
    holidays_path: Optional[str] = None

    # Which benchmarks to compute (subset of available keys)
    enabled_benchmarks: list[str] = field(
        default_factory=lambda: [
            "naive_seasonal",
            "ridge_regression",
            "lasso_regression",
            "elasticnet_regression",
            "lightgbm",
            "auto_arima",
            "seasonal_regression",
        ]
    )

    # Rolling window (mirrors runner.py)
    rolling_window: bool = False
    n_train_months: int = 37
    n_valid_months: int = 2
    n_test_months: int = 1
    start_window: int = 0

    # Gradient boosted trees
    gb_device: str = "cpu"
    gb_device_index: int = 0
    gb_n_jobs: int = 50
    gb_min_data_in_leaf: int = 20
    gb_n_estimators: int = 1000
    gb_early_stopping_rounds: int = 50

    # Linear Darts benchmarks
    ridge_alpha: float = 1.0
    lasso_alpha: float = 0.01
    elasticnet_alpha: float = 0.01
    elasticnet_l1_ratio: float = 0.5
    linear_scaler_type: Optional[str] = "standard"

    # AutoARIMA
    arima_use_approximation: bool = True
    arima_max_p: int = 3
    arima_max_q: int = 2
    arima_max_P: int = 3
    arima_max_Q: int = 2

    @property
    def test_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_start)

    @property
    def valid_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.valid_start)

    @property
    def train_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_start)


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    """Container returned by each benchmark runner.

    ``predictions`` is a long-format DataFrame with columns
    ``[ds, unique_id, <model_column>]`` where ``<model_column>`` contains
    the point forecast, clipped to ``[0, ∞)``.
    """

    name: str
    predictions: pd.DataFrame  # columns: ds, unique_id, <name>
    metadata: dict = field(default_factory=dict)


# ── TimeSeries helpers (shared) ───────────────────────────────────────────────
def _make_target_series(
    dataset: pd.DataFrame,
) -> tuple[TimeSeries, list[str]]:
    """Convert Nixtla-format long DataFrame → multivariate Darts ``TimeSeries``."""
    unique_ids = sorted(dataset["unique_id"].dropna().unique().tolist())
    series_list = [
        TimeSeries.from_dataframe(
            dataset.loc[dataset["unique_id"] == uid, ["ds", "y"]],
            time_col="ds",
            value_cols="y",
        )
        for uid in unique_ids
    ]
    target_ts = concatenate(series_list, axis=1)
    target_ts = target_ts.with_columns_renamed(target_ts.components.to_list(), unique_ids)

    static_df = build_static_df().set_index("unique_id")
    target_ts = target_ts.with_static_covariates(static_df.loc[unique_ids])
    return target_ts, unique_ids


def _make_covariate_series(
    dataset: pd.DataFrame,
    covariate_cols: list[str],
    base_unique_id: str,
) -> Optional[TimeSeries]:
    """Build a single ``TimeSeries`` for covariates (shared across unique_ids)."""
    if not covariate_cols:
        return None
    cov_df = dataset.loc[
        dataset["unique_id"] == base_unique_id, ["ds"] + covariate_cols
    ].copy()
    if cov_df.empty:
        return None
    return TimeSeries.from_dataframe(
        cov_df, time_col="ds", value_cols=covariate_cols
    ).with_static_covariates(None)


def _timeseries_to_long(ts: TimeSeries, column_name: str) -> pd.DataFrame:
    """Convert a multivariate Darts TimeSeries back to Nixtla-style long format."""
    df = ts.to_dataframe().copy()
    df.index.name = "ds"
    df = df.reset_index().melt(id_vars=["ds"], var_name="unique_id", value_name=column_name)
    df[column_name] = df[column_name].clip(lower=0)
    return df


def _concat_forecasts(
    forecasts: TimeSeries | list[TimeSeries] | list[list[TimeSeries]],
) -> TimeSeries:
    """Flatten / concatenate the output of ``historical_forecasts``."""
    if isinstance(forecasts, TimeSeries):
        return forecasts
    if not forecasts:
        raise ValueError("No forecasts returned from historical_forecasts.")
    if isinstance(forecasts[0], list):
        flat: list[TimeSeries] = [ts for sub in forecasts for ts in sub]
        return concatenate(flat, axis=0)
    return concatenate(cast(list[TimeSeries], forecasts), axis=0)


def _historical_forecasts_full_tail(
    model,
    target_ts: TimeSeries,
    forecast_horizon: int,
    stride: int,
    start: pd.Timestamp,
    past_covariates: Optional[TimeSeries] = None,
    future_covariates: Optional[TimeSeries] = None,
) -> TimeSeries:
    """Run ``historical_forecasts``."""

    preds = _concat_forecasts(
        model.historical_forecasts(
            cast(TimeSeries, target_ts),
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=forecast_horizon,
            stride=stride,
            last_points_only=False,
            retrain=False,
            verbose=False,
            start=start,
        )
    )
    return preds.slice_intersect(target_ts)


def _scale_timeseries_before_test(
    ts: Optional[TimeSeries],
    test_start: pd.Timestamp,
    scaler_type: Optional[str],
) -> tuple[Optional[TimeSeries], Optional[StandardScaler]]:
    """Fit a scaler on train+validation and transform the full TimeSeries."""
    if ts is None or scaler_type in {None, "none"}:
        return ts, None
    if scaler_type != "standard":
        raise ValueError(
            f"Unsupported linear_scaler_type={scaler_type!r}; use 'standard' or 'none'."
        )

    fit_end = test_start - pd.Timedelta(hours=1)
    ts_start = pd.Timestamp(ts.start_time())
    ts_end = pd.Timestamp(ts.end_time())
    if fit_end < ts_start:
        raise ValueError(
            f"Cannot fit scaler before test_start={test_start}: TimeSeries starts at {ts_start}."
        )
    fit_ts = ts.drop_after(min(fit_end, ts_end))

    scaler = StandardScaler()
    scaler.fit(fit_ts.values(copy=False))
    return ts.with_values(scaler.transform(ts.values(copy=False))), scaler


def _transform_timeseries(
    ts: Optional[TimeSeries],
    scaler: Optional[StandardScaler],
) -> Optional[TimeSeries]:
    if ts is None or scaler is None:
        return ts
    return ts.with_values(scaler.transform(ts.values(copy=False)))


def _inverse_transform_timeseries(
    ts: TimeSeries,
    scaler: Optional[StandardScaler],
) -> TimeSeries:
    if scaler is None:
        return ts
    return ts.with_values(scaler.inverse_transform(ts.values(copy=False)))


# ── Data preparation ──────────────────────────────────────────────────────────
def _prepare_benchmark_data(
    dataset: pd.DataFrame,
    cfg: BenchmarkConfig,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return unshifted dataset prepared for benchmarks with covariate classification.

    All benchmarks operate on *unshifted* (physical-time) data.  When
    ``shift_hours > 0``, models that support it use Darts'
    ``output_chunk_shift`` to create the required gap between input context
    and prediction output, eliminating the need for a separate shifted dataset.

    Returns
    -------
    tuple[pd.DataFrame, list[str], list[str]]
        (df, future_cov_cols, hist_cov_cols)
    """
    feature_cols = [c for c in dataset.columns if c not in {"ds", "y", "unique_id", "horizon"}]
    future_cov_cols, hist_cov_cols = classify_covariates(feature_cols)

    df = dataset.copy()
    if cfg.add_calendar:
        from training.data_prep import add_calendar_features
        df = add_calendar_features(
            df,
            reference_time=df["ds"],
            tso=cfg.tso,
            holidays_path=cfg.holidays_path,
        )
        for c in CALENDAR_COLS:
            if c in df.columns and c not in future_cov_cols:
                future_cov_cols.append(c)

    return df, future_cov_cols, hist_cov_cols


# ── Train/validation split helpers ────────────────────────────────────────────
def _split_for_single_window(
    target_ts: TimeSeries,
    past_cov_ts: Optional[TimeSeries],
    future_cov_ts: Optional[TimeSeries],
    forecast_horizon: int,
    input_length: int,
    valid_start: pd.Timestamp,
    test_start: pd.Timestamp,
) -> tuple:
    """Model-aware split for a single-window benchmark.

    Training data ends at ``valid_start`` (exclusive), matching the
    neural-model convention where ``[train_start, valid_start)`` is the
    training split and ``[valid_start, test_start)`` is held-out validation.
    """
    ts_max = pd.Timestamp(target_ts.time_index.max())
    train_holdout_size = max(1, int((ts_max - valid_start).total_seconds() // 3600) + 2)
    test_holdout_size = max(1, int((ts_max - test_start).total_seconds() // 3600) + 2)

    train_target, temp_target = train_test_split(
        target_ts,
        vertical_split_type="model-aware",
        input_size=input_length,
        horizon=forecast_horizon,
        test_size=train_holdout_size,
    )
    valid_target, test_target = train_test_split(
        temp_target,
        vertical_split_type="model-aware",
        input_size=input_length,
        horizon=forecast_horizon,
        test_size=test_holdout_size,
    )

    train_past = train_future = None
    if past_cov_ts is not None:
        train_past, temp_past = train_test_split(
            past_cov_ts,
            vertical_split_type="model-aware",
            input_size=input_length,
            horizon=forecast_horizon,
            test_size=train_holdout_size,
        )
        valid_past, test_past = train_test_split(
            temp_past,
            vertical_split_type="model-aware",
            input_size=input_length,
            horizon=forecast_horizon,
            test_size=test_holdout_size,
        )
    if future_cov_ts is not None:
        train_future, temp_future = train_test_split(
            future_cov_ts,
            vertical_split_type="model-aware",
            input_size=input_length,
            horizon=forecast_horizon,
            test_size=train_holdout_size,
        )
        valid_future, test_future = train_test_split(
            temp_future,
            vertical_split_type="model-aware",
            input_size=input_length,
            horizon=forecast_horizon,
            test_size=test_holdout_size,
        )
    
    return (
        train_target,
        train_past if past_cov_ts is not None else None,
        train_future if future_cov_ts is not None else None,
        valid_target,
        valid_past if past_cov_ts is not None else None,
        valid_future if future_cov_ts is not None else None,
        test_target,
        test_past if past_cov_ts is not None else None,
        test_future if future_cov_ts is not None else None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Individual benchmark runners
# ═══════════════════════════════════════════════════════════════════════════════


def _run_naive_seasonal(
    target_ts: TimeSeries,
    train_target: TimeSeries,
    cfg: BenchmarkConfig,
) -> BenchmarkResult:
    """NaiveSeasonal(K=forecast_horizon) benchmark.
    
    Note: This benchmark does not use covariates, so it operates on
    unshifted (physical time) data regardless of shift_hours setting.
    """
    model = NaiveSeasonal(K=cfg.forecast_horizon)
    model.fit(cast(TimeSeries, train_target))

    preds = _concat_forecasts(
        model.historical_forecasts(
            target_ts,
            forecast_horizon=cfg.forecast_horizon,
            stride=cfg.forecast_horizon,
            last_points_only=False,
            retrain=True,
            verbose=False,
            start=cfg.test_start_ts,
        )
    )
    df = _timeseries_to_long(preds, "naive_seasonal")
    return BenchmarkResult(name="naive_seasonal", predictions=df)


def _run_regularized_regression(
    result_name: str,
    model_cls: type,
    model_kwargs: dict[str, Any],
    target_ts: TimeSeries,
    train_target: TimeSeries,
    past_cov_ts: Optional[TimeSeries],
    future_cov_ts: Optional[TimeSeries],
    train_past: Optional[TimeSeries],
    train_future: Optional[TimeSeries],
    cfg: BenchmarkConfig,
) -> BenchmarkResult:
    """Shared sklearn-backed Darts regression benchmark with target inversion."""
    target_ts_model, target_scaler = _scale_timeseries_before_test(
        target_ts, cfg.test_start_ts, cfg.linear_scaler_type
    )
    train_target_model = cast(TimeSeries, _transform_timeseries(train_target, target_scaler))

    past_cov_ts_model, past_scaler = _scale_timeseries_before_test(
        past_cov_ts, cfg.test_start_ts, cfg.linear_scaler_type
    )
    train_past_model = cast(Optional[TimeSeries], _transform_timeseries(train_past, past_scaler))

    future_cov_ts_model, future_scaler = _scale_timeseries_before_test(
        future_cov_ts, cfg.test_start_ts, cfg.linear_scaler_type
    )
    train_future_model = cast(Optional[TimeSeries], _transform_timeseries(train_future, future_scaler))

    model = model_cls(
        lags=cfg.input_size,
        output_chunk_length=cfg.forecast_horizon,
        output_chunk_shift=cfg.shift_hours,
        lags_past_covariates=cfg.input_size if past_cov_ts is not None else None,
        lags_future_covariates=(cfg.input_size, 1) if future_cov_ts is not None else None,
        random_state=cfg.random_seed,
        **model_kwargs,
    )

    model.fit(
        train_target_model,
        past_covariates=train_past_model,
        future_covariates=train_future_model,
    )

    start_time = cfg.test_start_ts - pd.Timedelta(hours=cfg.shift_hours)
    preds = _historical_forecasts_full_tail(
        model,
        target_ts=cast(TimeSeries, target_ts_model),
        past_covariates=past_cov_ts_model,
        future_covariates=future_cov_ts_model,
        forecast_horizon=cfg.forecast_horizon,
        stride=cfg.forecast_horizon,
        start=start_time,
    )
    preds = _inverse_transform_timeseries(preds, target_scaler)

    df = _timeseries_to_long(preds, result_name)
    metadata = {
        "hist_lags": cfg.input_size,
        "futr_lags": cfg.input_size,
        "linear_scaler_type": cfg.linear_scaler_type,
        **model_kwargs,
    }
    return BenchmarkResult(name=result_name, predictions=df, metadata=metadata)


def _run_ridge_regression(
    target_ts: TimeSeries,
    train_target: TimeSeries,
    past_cov_ts: Optional[TimeSeries],
    future_cov_ts: Optional[TimeSeries],
    train_past: Optional[TimeSeries],
    train_future: Optional[TimeSeries],
    cfg: BenchmarkConfig,
) -> BenchmarkResult:
    """RidgeRegression benchmark with configurable lags."""
    return _run_regularized_regression(
        result_name="ridge_regression",
        model_cls=RidgeRegressionModel,
        model_kwargs={"alpha": cfg.ridge_alpha},
        target_ts=target_ts,
        train_target=train_target,
        past_cov_ts=past_cov_ts,
        future_cov_ts=future_cov_ts,
        train_past=train_past,
        train_future=train_future,
        cfg=cfg,
    )


def _run_lasso_regression(
    target_ts: TimeSeries,
    train_target: TimeSeries,
    past_cov_ts: Optional[TimeSeries],
    future_cov_ts: Optional[TimeSeries],
    train_past: Optional[TimeSeries],
    train_future: Optional[TimeSeries],
    cfg: BenchmarkConfig,
) -> BenchmarkResult:
    """LASSO regression benchmark with NeuralForecast-compatible scaling."""
    return _run_regularized_regression(
        result_name="lasso_regression",
        model_cls=LassoRegressionModel,
        model_kwargs={"alpha": cfg.lasso_alpha},
        target_ts=target_ts,
        train_target=train_target,
        past_cov_ts=past_cov_ts,
        future_cov_ts=future_cov_ts,
        train_past=train_past,
        train_future=train_future,
        cfg=cfg,
    )


def _run_elasticnet_regression(
    target_ts: TimeSeries,
    train_target: TimeSeries,
    past_cov_ts: Optional[TimeSeries],
    future_cov_ts: Optional[TimeSeries],
    train_past: Optional[TimeSeries],
    train_future: Optional[TimeSeries],
    cfg: BenchmarkConfig,
) -> BenchmarkResult:
    """ElasticNet regression benchmark with NeuralForecast-compatible scaling."""
    return _run_regularized_regression(
        result_name="elasticnet_regression",
        model_cls=ElasticNetRegressionModel,
        model_kwargs={
            "alpha": cfg.elasticnet_alpha,
            "l1_ratio": cfg.elasticnet_l1_ratio,
        },
        target_ts=target_ts,
        train_target=train_target,
        past_cov_ts=past_cov_ts,
        future_cov_ts=future_cov_ts,
        train_past=train_past,
        train_future=train_future,
        cfg=cfg,
    )


def aggregate_shap_importance(
    feat_imp_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate a raw SHAP importance DataFrame from ``log_lightgbm_importance_and_metadata``.

    The raw DataFrame has one row per (input-window timestamp × direction × horizon)
    and one column per Darts lag feature (e.g. ``down_target_lag-24``).  This
    function:

    1. Melts the wide format to long (one row per lag feature value).
    2. Strips ``_lag±N`` suffixes to recover the base feature name.
    3. Takes absolute SHAP values and sums them, returning two aggregated views:

    Returns
    -------
    agg_by_direction : DataFrame
        Columns: feature, direction[, window], abs_shap.
        Sorted by direction (asc) then abs_shap (desc).
    agg_by_direction_horizon : DataFrame
        Columns: feature, direction, horizon[, window], abs_shap.
        Sorted by direction, horizon (asc) then abs_shap (desc).
    """
    known_meta = {"direction", "horizon", "window"}
    meta_cols = [c for c in feat_imp_df.columns if c in known_meta]
    feature_cols = [c for c in feat_imp_df.columns if c not in known_meta]

    # Strip lag suffixes from the ~N column names rather than from N*n_rows values.
    # This is the main speedup: no melt, no per-row string ops.
    base_features = np.array([_LAG_SUFFIX_RE.sub("", c) for c in feature_cols])

    # ── agg_by_direction ─────────────────────────────────────────────────────
    # abs().sum(axis=0) is a single vectorised numpy reduction → Series of length
    # n_feature_cols.  Renaming its index to base names then groupby-sum collapses
    # it to n_base_features.  No intermediate 700 M-row DataFrame needed.
    col_abs_sum = feat_imp_df[feature_cols].abs().sum(axis=0)
    col_abs_sum.index = base_features
    agg_by_direction = (
        col_abs_sum
        .groupby(level=0).sum()
        .rename_axis("feature")
        .reset_index(name="abs_shap")
        .sort_values("abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # ── agg_by_direction_horizon ──────────────────────────────────────────────
    # Compute abs once, group by horizon (usually a single unique value per call
    # since this is invoked inside a horizon loop) → tiny (n_horizons × n_cols)
    # matrix.  Then collapse columns by base feature name and melt the tiny result.
    abs_feat = feat_imp_df[feature_cols].abs()
    abs_by_horizon = abs_feat.groupby(feat_imp_df["horizon"]).sum()
    abs_by_horizon.columns = base_features
    # Sum columns that share the same base feature name
    abs_by_horizon = abs_by_horizon.T.groupby(level=0).sum().T
    # abs_by_horizon is now (n_horizons × n_base_features) - melt is cheap here
    agg_by_direction_horizon = (
        abs_by_horizon.reset_index()
        .melt(id_vars=["horizon"], var_name="feature", value_name="abs_shap")
        .sort_values(["horizon", "abs_shap"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return agg_by_direction, agg_by_direction_horizon


def log_lightgbm_importance_and_metadata(
    model: LightGBMModel, train_target: TimeSeries, 
    test_target: Optional[TimeSeries], test_past: Optional[TimeSeries], 
    test_future: Optional[TimeSeries], 
    cfg: BenchmarkConfig,
    wb: WindowBoundary | None = None
) -> tuple[dict[str, pd.DataFrame], dict[str, str | int | float]]:
    n_estimators = cfg.gb_n_estimators
    min_data_in_leaf = cfg.gb_min_data_in_leaf
    try:
        sub_models: list[lgb.LGBMRegressor] = model.model.estimators_
        best_iters_arr = np.array(
            [
                m.best_iteration_ 
                if getattr(m, "best_iteration_", None) is not None else n_estimators
                for m in sub_models
            ],
            dtype=np.int64,
        )
        n_estimators_arr = np.array(
            [
                max(m.booster_.current_iteration(), m.booster_.num_trees()) + cfg.gb_early_stopping_rounds 
                if getattr(m, "booster_", None) is not None else n_estimators
                for m in sub_models
            ],
            dtype=np.int64,
        )

        n_components = len(train_target.components)
        model_details_dict: dict[str, str | int | float] = {}
        if best_iters_arr.size > 0:
            for i, component in enumerate(train_target.components):
                # step-major layout: sub-model index = step * n_components + component_idx
                component_range = list(range(i, len(sub_models), n_components))
                n_est_min = int(np.min(n_estimators_arr[component_range])) if n_estimators_arr.size > 0 else n_estimators
                n_est_med = float(np.median(n_estimators_arr[component_range])) if n_estimators_arr.size > 0 else float(n_estimators)
                n_est_max = int(np.max(n_estimators_arr[component_range])) if n_estimators_arr.size > 0 else n_estimators

                n_best_min = int(np.min(best_iters_arr[component_range])) if best_iters_arr.size > 0 else n_estimators
                n_best_med = float(np.median(best_iters_arr[component_range])) if best_iters_arr.size > 0 else float(n_estimators)
                n_best_max = int(np.max(best_iters_arr[component_range])) if best_iters_arr.size > 0 else n_estimators
                logger.info(
                    "LightGBM fitted on component %s (%d sub-models): "
                    "best_iteration min=%d median=%.0f max=%d  "
                    "n_estimators_ min=%d median=%.0f max=%d  "
                    "(configured n_estimators=%d, min_data_in_leaf=%d)",
                    component,
                    len(component_range),
                    n_best_min,
                    n_best_med,
                    n_best_max,
                    n_est_min,
                    n_est_med,
                    n_est_max,
                    n_estimators,
                    min_data_in_leaf,
                )
                component_details_dict = {
                    "min_data_in_leaf": min_data_in_leaf,
                    "configured_n_estimators": n_estimators,
                    "n_sub_models": len(component_range),
                    "early_stopping_rounds": cfg.gb_early_stopping_rounds,
                    "best_iteration_min": n_best_min,
                    "best_iteration_median": n_best_med,
                    "best_iteration_max": n_best_max,
                    "n_estimators_min": n_est_min,
                    "n_estimators_median": n_est_med,
                    "n_estimators_max": n_est_max,
                }

                if wb is not None:
                    component_details_dict.update({
                        "validation_window_start": wb.valid_start.strftime("%Y-%m-%d"),
                        "validation_window_end": (wb.test_start - pd.Timedelta(hours=1)).strftime("%Y-%m-%d"),
                    })
                else:
                    component_details_dict.update({
                        "validation_window_start": cfg.valid_start_ts.strftime("%Y-%m-%d"),
                        "validation_window_end": (cfg.test_start_ts - pd.Timedelta(hours=1)).strftime("%Y-%m-%d"),
                    })
                model_details_dict[component] = component_details_dict
        else:
            logger.info(
                "LightGBM fitted (%d sub-models): no early stopping "
                "(n_estimators=%d, min_data_in_leaf=%d)",
                len(sub_models),
                n_estimators,
                min_data_in_leaf,
            )

        # Aggregate feature importances across components and sub-models (mean over all sub-models)
        imp_records: dict[str, list[pd.DataFrame]] = {}
        explainer = ShapExplainer(model=model)
        if cfg.shift_hours > 0 and test_target:
            # If shift_hours > 0, we must use the shifted test data for explanation
            shifted_test_target = test_target.shift(cfg.shift_hours)
        explainer_result = explainer.explain(
            foreground_series=shifted_test_target if cfg.shift_hours > 0 else test_target,
            foreground_past_covariates=test_past,
            foreground_future_covariates=test_future,
        )

        for component in train_target.components:
            for horizon in range(1, cfg.forecast_horizon + 1):
                # Extract importances for this component and horizon
                comp_horiz_imp = explainer_result.get_explanation(
                    component=component,
                    horizon=horizon,
                )
                if comp_horiz_imp is not None:
                    imp_df = comp_horiz_imp.to_dataframe()
                    imp_df["direction"] = str(component)
                    imp_df["horizon"] = horizon
                    per_direction_imp_df, per_direction_horizon_imp_df = aggregate_shap_importance(
                        imp_df
                    )
                    imp_records.setdefault("direction", []).append(per_direction_imp_df)
                    imp_records.setdefault("horizon", []).append(per_direction_horizon_imp_df)

        if imp_records:
            feat_imp_df_final = {
                agg_label: (
                    pd.concat(records, ignore_index=True).groupby(["feature", "horizon"])["abs_shap"].mean().reset_index() 
                    if agg_label == "horizon" else pd.concat(records, ignore_index=True)
                )
                for agg_label, records in imp_records.items()
            }
        else:
            feat_imp_df_final = {
                "direction": pd.DataFrame(columns=["feature", "abs_shap", "direction"]),
                "horizon": pd.DataFrame(columns=["feature", "abs_shap", "horizon"]),
            }
    except Exception as log_exc:
        logger.debug("Could not read LightGBM post-fit stats: %s", log_exc)
        feat_imp_df_final = {
            "direction": pd.DataFrame(columns=["feature", "abs_shap", "direction"]),
            "horizon": pd.DataFrame(columns=["feature", "abs_shap", "horizon"]),
        }
        model_details_dict = {}

    return feat_imp_df_final, model_details_dict


def _run_lightgbm(
    target_ts: TimeSeries,
    train_target: TimeSeries,
    train_past: TimeSeries,
    train_future: TimeSeries,
    valid_target: TimeSeries,
    valid_future: TimeSeries,
    valid_past: TimeSeries,
    test_target: TimeSeries,
    test_future: TimeSeries,
    test_past: TimeSeries,
    past_cov_ts: Optional[TimeSeries],
    future_cov_ts: Optional[TimeSeries],
    cfg: BenchmarkConfig,
) -> Optional[BenchmarkResult]:
    """Gradient-boosted trees (LightGBM) benchmark.

    Uses ``output_chunk_shift`` to handle the decision-time gap when
    ``shift_hours > 0``, operating entirely on unshifted (physical-time) data.
    """

    # If calendar features are used, treat them as categorical future covariates
    future_cov_cols = future_cov_ts.components
    if all(c in future_cov_cols for c in CALENDAR_COLS):
        cat_future_cov_ts = CALENDAR_COLS
    else:
        cat_future_cov_ts = None

    lgbm_kwargs: dict = dict(
        boosting_type="gbdt",
        objective="regression_l1",
        metric="mae",
        n_estimators=cfg.gb_n_estimators,
        min_child_samples=cfg.gb_min_data_in_leaf,
        verbosity=-1,
        n_jobs=cfg.gb_n_jobs,
        random_state=cfg.random_seed,
    )

    if cfg.gb_device == "gpu":
        lgbm_kwargs.update(
            device="gpu",
            gpu_device_id=cfg.gb_device_index,
        )
        os.environ.setdefault("OCL_ICD_VENDORS", "/home/jovyan/opencl_vendors")

    model_params = dict(
        lags=cfg.input_size,
        output_chunk_length=cfg.forecast_horizon,
        output_chunk_shift=cfg.shift_hours,
        lags_past_covariates=cfg.input_size if valid_past is not None else None,
        lags_future_covariates=(cfg.input_size, 1) if valid_future is not None else None,
        categorical_future_covariates=cat_future_cov_ts,
        **lgbm_kwargs,
    )

    use_early_stopping = cfg.gb_early_stopping_rounds > 0
    fit_kwargs: dict = {}
    if use_early_stopping:
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(stopping_rounds=cfg.gb_early_stopping_rounds, verbose=False),
        ]

    model = LightGBMModel(**model_params)
    model.fit(
        cast(TimeSeries, train_target),
        past_covariates=train_past,
        future_covariates=train_future,
        val_series=valid_target,
        val_past_covariates=valid_past,
        val_future_covariates=valid_future,
        **fit_kwargs,
    )

    feature_importance, metadata = log_lightgbm_importance_and_metadata(
        model, train_target, test_target, test_past, test_future, cfg
    )

    metadata.update({
        "hist_lags": cfg.input_size,
        "futr_lags": cfg.input_size,
    })

    # With output_chunk_shift, Darts adds shift_hours to the output position.
    start_time = cfg.test_start_ts - pd.Timedelta(hours=cfg.shift_hours)

    preds = _historical_forecasts_full_tail(
        model,
        target_ts=target_ts,
        forecast_horizon=cfg.forecast_horizon,
        stride=cfg.forecast_horizon,
        past_covariates=past_cov_ts,
        future_covariates=future_cov_ts,
        start=start_time,
    )

    col_name = "lightgbm_regression"
    df = _timeseries_to_long(preds, col_name)
    return BenchmarkResult(
        name=col_name,
        predictions=df,
        metadata={"importance": feature_importance, "model_details": metadata}
    )


def _run_auto_arima_single_window(
    train_target: TimeSeries,
    test_target: TimeSeries,
    cfg: BenchmarkConfig,
) -> BenchmarkResult:
    """AutoARIMA benchmark (single-window, no retrain during backtest).
    
    Handles multivariate series by training separate models per component.
    Note: future_covariates disabled due to rank deficiency issues with many covariates.
    AutoARIMA does not support output_chunk_shift, so it always predicts in
    physical time directly.
    """
    # Determine start time
    start_time = cfg.test_start_ts
    
    # If multivariate, split by component and train separate models
    arima_orders: dict[str, dict] = {}
    all_preds = []
    for component in train_target.components:
        logger.info(f"Training AutoARIMA for component: {component}")
        model = AutoARIMA(
            max_p=cfg.arima_max_p,
            max_q=cfg.arima_max_q,
            max_P=cfg.arima_max_P,
            max_Q=cfg.arima_max_Q,
            season_length=cfg.forecast_horizon,
            approximation=cfg.arima_use_approximation,
            ic="aicc",
        )
        model.fit(train_target[component])
        
        try:
            non_seasonal_order = model.model.model_["arma"][:3]  # type: ignore[union-attr]
            seasonal_order = model.model.model_["arma"][3:]  # type: ignore[union-attr]
            coefficients = model.model.model_["coef"]  # type: ignore[union-attr]
            logger.info(
                "ARIMA fitted (%s): order=%s seasonal_order=%s coefficients=%s",
                component,
                non_seasonal_order,
                seasonal_order,
                coefficients,
            )
            arima_orders[str(component)] = {
                "order_pqd": [int(x) for x in non_seasonal_order],
                "seasonal_order_PQDm": [int(x) for x in seasonal_order],
                "coefficients": {k: float(v) for k, v in coefficients.items()},
            }
        except Exception:
            logger.warning("Could not read ARIMA orders for component %s", component)
        
        component_preds = _historical_forecasts_full_tail(
            model,
            target_ts=test_target[component],
            forecast_horizon=cfg.forecast_horizon,
            stride=cfg.forecast_horizon,
            start=start_time,
        )
        all_preds.append(component_preds)
    
    # Concatenate all component predictions
    preds = concatenate(all_preds, axis=1)
   
    df = _timeseries_to_long(preds, "auto_arima")
    metadata = {"arima_params": arima_orders}
    return BenchmarkResult(name="auto_arima", predictions=df, metadata=metadata)


def _run_auto_arima_rolling(
    target_ts: TimeSeries,
    cfg: BenchmarkConfig,
    windows: list[WindowBoundary],
) -> BenchmarkResult:
    """AutoARIMA benchmark retrained on every rolling window.
    
    Handles multivariate series by training separate models per component.
    AutoARIMA does not support output_chunk_shift, so it always predicts in
    physical time directly.  Training uses only ``[train_start, valid_start)``
    in each window.
    """
    all_preds: list[pd.DataFrame] = []
    arima_orders: dict[str, dict] = {}

    for wi, wb in enumerate(windows):
        logger.info(
            "ARIMA rolling window %d/%d: train→%s, test [%s, %s)",
            wi + 1,
            len(windows),
            wb.valid_start,
            wb.test_start,
            wb.test_end,
        )
        # Slice target to the training period (excluding validation)
        window_train = target_ts.drop_after(wb.valid_start)

        forecast_start = wb.test_start
        
        # If multivariate, train separate models per component
        component_preds = []
        for component in target_ts.components:
            model = AutoARIMA()
            model.fit(window_train[component])  # No future covariates to avoid rank deficiency

            try:
                non_seasonal_order = model.model.model_["arma"][:3]  # type: ignore[union-attr]
                seasonal_order = model.model.model_["arma"][3:]  # type: ignore[union-attr]
                coefficients = model.model.model_["coef"]  # type: ignore[union-attr]
                logger.info(
                    "ARIMA fitted (%s): order=%s seasonal_order=%s coefficients=%s",
                    component,
                    non_seasonal_order,
                    seasonal_order,
                    coefficients,
                )
                arima_orders.setdefault(f"window_{wi}", {})[str(component)] = {
                    "order_pqd": [int(x) for x in non_seasonal_order],
                    "seasonal_order_PQDm": [int(x) for x in seasonal_order],
                    "coefficients": {k: float(v) for k, v in coefficients.items()},
                }
            except Exception:
                logger.warning("Could not read ARIMA orders for component %s", component)

            comp_pred = _historical_forecasts_full_tail(
                model,
                target_ts=target_ts[component],
                forecast_horizon=cfg.forecast_horizon,
                stride=cfg.forecast_horizon,
                start=forecast_start,
            )
            component_preds.append(comp_pred)
        window_preds = concatenate(component_preds, axis=1)
    
        all_preds.append(_timeseries_to_long(window_preds, "auto_arima_rolling"))

    result_df = pd.concat(all_preds, ignore_index=True)
    # De-duplicate in case windows overlap – keep the last (most recent model)
    result_df = result_df.drop_duplicates(subset=["ds", "unique_id"], keep="last")
    
    return BenchmarkResult(name="auto_arima_rolling", predictions=result_df, metadata=arima_orders)


def _run_seasonal_regression(
    target_ts: TimeSeries,
    train_target: TimeSeries,
    train_future: TimeSeries,
    future_cov_ts: TimeSeries,
    cfg: BenchmarkConfig,
) -> BenchmarkResult:
    """Seasonal regression using only Fourier / calendar features.

    This captures hour-of-day, day-of-week and month-of-year patterns without
    any data-driven covariates – a useful "how much does seasonality explain?"
    baseline.  Uses ``output_chunk_shift`` when ``shift_hours > 0``.
    - Trains on shifted data (calendar features are future covariates)
    - Predicts starting at model time
    - Converts predictions back to physical time
    """
    future_cols = future_cov_ts.components

    # Build calendar-only future covariates
    calendar_cols_present = [c for c in CALENDAR_COLS if c in future_cols]
    if not calendar_cols_present:
        logger.warning(
            "No calendar columns found in dataset – skipping seasonal regression."
        )
        return BenchmarkResult(name="seasonal_regression", predictions=pd.DataFrame())

    target_ts_model, target_scaler = _scale_timeseries_before_test(
        target_ts, cfg.test_start_ts, cfg.linear_scaler_type
    )
    train_target_model = cast(TimeSeries, _transform_timeseries(train_target, target_scaler))
    future_cov_ts_model, future_scaler = _scale_timeseries_before_test(
        future_cov_ts, cfg.test_start_ts, cfg.linear_scaler_type
    )
    train_future_model = cast(TimeSeries, _transform_timeseries(train_future, future_scaler))

    # Add day_of_week as a feature (not in CALENDAR_COLS but useful for seasonality)
    model = LinearRegressionModel(
        lags=cfg.input_size,
        output_chunk_length=cfg.forecast_horizon,
        output_chunk_shift=cfg.shift_hours,
        lags_future_covariates=(cfg.input_size, 1) if len(calendar_cols_present) > 0 else None,
        add_encoders={'datetime_attribute': {'future': ['dayofweek']}}
    )
    model.fit(
        train_target_model,
        future_covariates=train_future_model[calendar_cols_present],
    )

    # With output_chunk_shift, Darts adds shift_hours to the output position.
    forecast_start = cfg.test_start_ts - pd.Timedelta(hours=cfg.shift_hours)

    preds = _historical_forecasts_full_tail(
        model,
        target_ts=cast(TimeSeries, target_ts_model),
        future_covariates=cast(TimeSeries, future_cov_ts_model)[calendar_cols_present],
        forecast_horizon=cfg.forecast_horizon,
        stride=cfg.forecast_horizon,
        start=forecast_start,
    )
    preds = _inverse_transform_timeseries(preds, target_scaler)
    
    df = _timeseries_to_long(preds, "seasonal_regression")
    return BenchmarkResult(
        name="seasonal_regression",
        predictions=df,
        metadata={"features": calendar_cols_present, "input_size": cfg.input_size, "linear_scaler_type": cfg.linear_scaler_type},
    )


# ── Rolling window implementations ────────────────────────────────────────────
def _run_regularized_regression_rolling(
    result_name: str,
    model_cls: type,
    model_kwargs: dict[str, Any],
    dataset: pd.DataFrame,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    cfg: BenchmarkConfig,
    windows: list[WindowBoundary],
) -> BenchmarkResult:
    """Shared rolling sklearn-backed regression benchmark with target inversion."""
    all_preds: list[pd.DataFrame] = []

    for wi, wb in enumerate(windows):
        logger.info(
            "%s rolling window %d/%d: train+valid scaler→%s, test [%s, %s)",
            result_name,
            wi + 1,
            len(windows),
            wb.test_start,
            wb.test_start,
            wb.test_end,
        )

        window_df = dataset[
            (dataset["ds"] >= wb.train_start) & (dataset["ds"] < wb.test_end)
        ].copy()

        target_ts, unique_ids = _make_target_series(window_df)
        base_uid = "up" if "up" in unique_ids else unique_ids[0]

        past_cov_ts = _make_covariate_series(window_df, hist_cov_cols, base_uid)
        future_cov_ts = _make_covariate_series(window_df, future_cov_cols, base_uid)

        holdout_size = max(1, int((wb.test_end - wb.valid_start).total_seconds() // 3600) + 2)
        train_target, _ = train_test_split(
            target_ts,
            vertical_split_type="model-aware",
            input_size=cfg.input_size,
            horizon=cfg.forecast_horizon,
            test_size=holdout_size,
        )

        train_past = None
        if past_cov_ts is not None:
            train_past, _ = train_test_split(
                past_cov_ts,
                vertical_split_type="model-aware",
                input_size=cfg.input_size,
                horizon=cfg.forecast_horizon,
                test_size=holdout_size,
            )

        train_future = None
        if future_cov_ts is not None:
            train_future, _ = train_test_split(
                future_cov_ts,
                vertical_split_type="model-aware",
                input_size=cfg.input_size,
                horizon=cfg.forecast_horizon,
                test_size=holdout_size,
            )

        target_ts_model, target_scaler = _scale_timeseries_before_test(
            target_ts, wb.test_start, cfg.linear_scaler_type
        )
        train_target_model = cast(TimeSeries, _transform_timeseries(train_target, target_scaler))
        past_cov_ts_model, past_scaler = _scale_timeseries_before_test(
            past_cov_ts, wb.test_start, cfg.linear_scaler_type
        )
        train_past_model = cast(Optional[TimeSeries], _transform_timeseries(train_past, past_scaler))
        future_cov_ts_model, future_scaler = _scale_timeseries_before_test(
            future_cov_ts, wb.test_start, cfg.linear_scaler_type
        )
        train_future_model = cast(Optional[TimeSeries], _transform_timeseries(train_future, future_scaler))

        model = model_cls(
            lags=cfg.input_size,
            output_chunk_length=cfg.forecast_horizon,
            output_chunk_shift=cfg.shift_hours,
            lags_past_covariates=cfg.input_size if past_cov_ts is not None else None,
            lags_future_covariates=(cfg.input_size, 1) if future_cov_ts is not None else None,
            random_state=cfg.random_seed,
            **model_kwargs,
        )
        model.fit(
            train_target_model,
            past_covariates=train_past_model,
            future_covariates=train_future_model,
        )

        forecast_start = wb.test_start - pd.Timedelta(hours=cfg.shift_hours)
        window_preds = _historical_forecasts_full_tail(
            model,
            target_ts=cast(TimeSeries, target_ts_model),
            past_covariates=past_cov_ts_model,
            future_covariates=future_cov_ts_model,
            forecast_horizon=cfg.forecast_horizon,
            stride=cfg.forecast_horizon,
            start=forecast_start,
        )
        window_preds = _inverse_transform_timeseries(window_preds, target_scaler)

        all_preds.append(_timeseries_to_long(window_preds, result_name))

    result_df = pd.concat(all_preds, ignore_index=True)
    result_df = result_df.drop_duplicates(subset=["ds", "unique_id"], keep="last")
    return BenchmarkResult(
        name=result_name,
        predictions=result_df,
        metadata={
            "hist_lags": cfg.input_size,
            "futr_lags": cfg.input_size,
            "linear_scaler_type": cfg.linear_scaler_type,
            **model_kwargs,
        },
    )


def _run_ridge_regression_rolling(
    dataset: pd.DataFrame,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    cfg: BenchmarkConfig,
    windows: list[WindowBoundary],
) -> BenchmarkResult:
    return _run_regularized_regression_rolling(
        "ridge_regression_rolling",
        RidgeRegressionModel,
        {"alpha": cfg.ridge_alpha},
        dataset,
        future_cov_cols,
        hist_cov_cols,
        cfg,
        windows,
    )


def _run_lasso_regression_rolling(
    dataset: pd.DataFrame,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    cfg: BenchmarkConfig,
    windows: list[WindowBoundary],
) -> BenchmarkResult:
    return _run_regularized_regression_rolling(
        "lasso_regression_rolling",
        LassoRegressionModel,
        {"alpha": cfg.lasso_alpha},
        dataset,
        future_cov_cols,
        hist_cov_cols,
        cfg,
        windows,
    )


def _run_elasticnet_regression_rolling(
    dataset: pd.DataFrame,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    cfg: BenchmarkConfig,
    windows: list[WindowBoundary],
) -> BenchmarkResult:
    return _run_regularized_regression_rolling(
        "elasticnet_regression_rolling",
        ElasticNetRegressionModel,
        {"alpha": cfg.elasticnet_alpha, "l1_ratio": cfg.elasticnet_l1_ratio},
        dataset,
        future_cov_cols,
        hist_cov_cols,
        cfg,
        windows,
    )


def _run_lightgbm_rolling(
    dataset: pd.DataFrame,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    cfg: BenchmarkConfig,
    windows: list[WindowBoundary],
) -> Optional[BenchmarkResult]:
    """Gradient-boosted trees benchmark with rolling window retraining.

    Uses ``output_chunk_shift`` for shift-awareness.  Trains only on
    ``[train_start, valid_start)`` in each window.
    """
    all_preds: list[pd.DataFrame] = []
    all_metadata: dict[str, Any] = {
        "hist_lags": cfg.input_size,
        "futr_lags": cfg.input_size,
        "windows": {}
    }

    # If calendar features are used, treat them as categorical future covariates
    if all(c in future_cov_cols for c in CALENDAR_COLS):
        cat_future_cov_ts = CALENDAR_COLS
    else:
        cat_future_cov_ts = None

    lgbm_kwargs: dict = dict(
        boosting_type="gbdt",
        objective="regression_l1",
        metric="mae",
        n_estimators=cfg.gb_n_estimators,
        min_child_samples=cfg.gb_min_data_in_leaf,
        verbosity=-1,
        n_jobs=cfg.gb_n_jobs,
        random_state=cfg.random_seed,
    )

    if cfg.gb_device == "gpu":
        lgbm_kwargs.update(
            device="gpu",
            gpu_device_id=cfg.gb_device_index,
        )
        os.environ.setdefault("OCL_ICD_VENDORS", "/home/jovyan/opencl_vendors")

    for wi, wb in enumerate(windows):
        logger.info(
            "LightGBM rolling window %d/%d: train→%s, test [%s, %s)",
            wi + 1,
            len(windows),
            wb.valid_start,
            wb.test_start,
            wb.test_end,
        )

        # Slice dataset for this window
        window_df = dataset[
            (dataset["ds"] >= wb.train_start - pd.Timedelta(hours=cfg.input_size)) & (dataset["ds"] < wb.test_end)
        ].copy()

        # Create TimeSeries for this window
        target_ts, unique_ids = _make_target_series(window_df)
        base_uid = "up" if "up" in unique_ids else unique_ids[0]

        past_cov_ts = _make_covariate_series(window_df, hist_cov_cols, base_uid)
        future_cov_ts = _make_covariate_series(window_df, future_cov_cols, base_uid)

        # Split for training (up to valid_start, excluding validation)
        (
            train_target,
            train_past,
            train_future,
            valid_target,
            valid_past,
            valid_future,
            test_target,
            test_past,
            test_future,
        ) = _split_for_single_window(
            target_ts,
            past_cov_ts,
            future_cov_ts,
            forecast_horizon=cfg.forecast_horizon,
            input_length=cfg.input_size,
            valid_start=wb.valid_start,
            test_start=wb.test_start,
        )

        model_params = dict(
            lags=cfg.input_size,
            output_chunk_length=cfg.forecast_horizon,
            output_chunk_shift=cfg.shift_hours,
            lags_past_covariates=cfg.input_size if past_cov_ts is not None else None,
            lags_future_covariates=(cfg.input_size, 1) if future_cov_ts is not None else None,
            categorical_future_covariates=cat_future_cov_ts,
            **lgbm_kwargs,
        )

        use_early_stopping = cfg.gb_early_stopping_rounds > 0
        fit_kwargs: dict = {}
        if use_early_stopping:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=cfg.gb_early_stopping_rounds, verbose=False),
            ]

        # Train model
        model = LightGBMModel(**model_params)
        model.fit(
            cast(TimeSeries, train_target),
            past_covariates=train_past,
            future_covariates=train_future,
            val_series=valid_target,
            val_past_covariates=valid_past,
            val_future_covariates=valid_future,
            **fit_kwargs,
        )

        feature_importance, metadata = log_lightgbm_importance_and_metadata(
            model, train_target, test_target, test_past, test_future, cfg
        )

        metadata.update({
            "hist_lags": cfg.input_size,
            "futr_lags": cfg.input_size,
        })

        # With output_chunk_shift, Darts adds shift_hours to the output position.
        forecast_start = wb.test_start - pd.Timedelta(hours=cfg.shift_hours)

        window_preds = _historical_forecasts_full_tail(
            model,
            target_ts=target_ts,
            forecast_horizon=cfg.forecast_horizon,
            stride=cfg.forecast_horizon,
            past_covariates=past_cov_ts,
            future_covariates=future_cov_ts,
            start=forecast_start,
        )

        col_name = "lightgbm_regression_rolling"
        all_preds.append(_timeseries_to_long(window_preds, col_name))
        all_metadata["windows"][wi + 1] = {"importance": feature_importance, "model_details": metadata}

    col_name = "lightgbm_regression_rolling"
    result_df = pd.concat(all_preds, ignore_index=True)
    result_df = result_df.drop_duplicates(subset=["ds", "unique_id"], keep="last")

    return BenchmarkResult(
        name=col_name,
        predictions=result_df,
        metadata=all_metadata,
    )


def _run_seasonal_regression_rolling(
    dataset: pd.DataFrame,
    cfg: BenchmarkConfig,
    windows: list[WindowBoundary],
) -> BenchmarkResult:
    """Seasonal regression benchmark with rolling window retraining.

    Uses only calendar features and ``output_chunk_shift`` for shift-awareness.
    Trains only on ``[train_start, valid_start)`` in each window.
    """
    all_preds: list[pd.DataFrame] = []

    for wi, wb in enumerate(windows):
        logger.info(
            "SeasonalRegression rolling window %d/%d: train→%s, test [%s, %s)",
            wi + 1,
            len(windows),
            wb.valid_start,
            wb.test_start,
            wb.test_end,
        )

        # Slice dataset for this window
        window_df = dataset[
            (dataset["ds"] >= wb.train_start) & (dataset["ds"] < wb.test_end)
        ].copy()

        # Create TimeSeries for this window
        target_ts, unique_ids = _make_target_series(window_df)
        base_uid = "up" if "up" in unique_ids else unique_ids[0]

        # Build calendar-only future covariates
        calendar_cols_present = [c for c in CALENDAR_COLS if c in window_df.columns]
        if not calendar_cols_present:
            logger.warning(
                "Window %d: No calendar columns found – skipping",
                wi + 1,
            )
            continue

        season_df = window_df.loc[
            window_df["unique_id"] == base_uid, ["ds"] + calendar_cols_present
        ].copy()

        season_ts = TimeSeries.from_dataframe(
            season_df, time_col="ds", value_cols=calendar_cols_present
        ).with_static_covariates(None)

        # Split for training (up to valid_start, excluding validation)
        holdout_size = max(1, int((wb.test_end - wb.valid_start).total_seconds() // 3600) + 2)
        train_target, _ = train_test_split(
            target_ts,
            vertical_split_type="model-aware",
            input_size=cfg.input_size,
            horizon=cfg.forecast_horizon,
            test_size=holdout_size,
        )

        valid_holdout_size = max(1, int((wb.test_start - wb.valid_start).total_seconds() // 3600) + 2)
        test_holdout_size = max(1, int((wb.test_end - wb.test_start).total_seconds() // 3600) + 2)
        train_season, temp_season = train_test_split(
            season_ts,
            vertical_split_type="model-aware",
            input_size=cfg.input_size,
            horizon=cfg.forecast_horizon,
            test_size=valid_holdout_size,
        )
        _, test_season = train_test_split(
            temp_season,
            vertical_split_type="model-aware",
            input_size=cfg.input_size,
            horizon=cfg.forecast_horizon,
            test_size=test_holdout_size,
        )

        target_ts_model, target_scaler = _scale_timeseries_before_test(
            target_ts, wb.test_start, cfg.linear_scaler_type
        )
        train_target_model = cast(TimeSeries, _transform_timeseries(train_target, target_scaler))
        season_ts_model, season_scaler = _scale_timeseries_before_test(
            season_ts, wb.test_start, cfg.linear_scaler_type
        )
        train_season_model = cast(TimeSeries, _transform_timeseries(train_season, season_scaler))

        # Train model
        model = LinearRegressionModel(
            lags=cfg.input_size,
            output_chunk_length=cfg.forecast_horizon,
            output_chunk_shift=cfg.shift_hours,
            lags_future_covariates=(cfg.input_size, 1) if len(calendar_cols_present) > 0 else None,
        )
        model.fit(train_target_model, future_covariates=train_season_model)

        # With output_chunk_shift, Darts adds shift_hours to the output position.
        forecast_start = wb.test_start - pd.Timedelta(hours=cfg.shift_hours)

        window_preds = _historical_forecasts_full_tail(
            model,
            target_ts=cast(TimeSeries, target_ts_model),
            forecast_horizon=cfg.forecast_horizon,
            stride=cfg.forecast_horizon,
            future_covariates=cast(TimeSeries, season_ts_model),
            start=forecast_start,
        )
        window_preds = _inverse_transform_timeseries(window_preds, target_scaler)

        all_preds.append(_timeseries_to_long(window_preds, "seasonal_regression_rolling"))

    if not all_preds:
        logger.warning("No predictions generated for seasonal regression rolling")
        return BenchmarkResult(
            name="seasonal_regression_rolling",
            predictions=pd.DataFrame(),
            metadata={},
        )

    result_df = pd.concat(all_preds, ignore_index=True)
    result_df = result_df.drop_duplicates(subset=["ds", "unique_id"], keep="last")

    return BenchmarkResult(
        name="seasonal_regression_rolling",
        predictions=result_df,
        metadata={"hist_lags": cfg.input_size, "futr_lags": cfg.input_size, "features": calendar_cols_present, "linear_scaler_type": cfg.linear_scaler_type},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def run_benchmarks(
    dataset: pd.DataFrame,
    cfg: BenchmarkConfig,
) -> tuple[pd.DataFrame, dict]:
    """Compute all enabled benchmarks and return predictions with metadata.

    Parameters
    ----------
    dataset : pd.DataFrame
        Nixtla-format DataFrame (``ds``, ``y``, ``unique_id``, feature columns).
    cfg : BenchmarkConfig
        Benchmark configuration.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (predictions_df, metadata_dict)
        - predictions_df: Long-format DataFrame with columns ``[ds, unique_id]`` plus one column
          per benchmark model.
        - metadata_dict: Dictionary mapping benchmark names to their metadata (parameters, etc.)
    """
    # ── 1. Prepare dataset (unshifted / physical time) ─────────────────────────
    df, future_cov_cols, hist_cov_cols = _prepare_benchmark_data(dataset, cfg)
    df = df.sort_values(["unique_id", "ds"]).copy()

    # ── 2. Build Darts TimeSeries objects (all in physical time) ──────────────
    target_ts, unique_ids = _make_target_series(df)
    base_uid = "up" if "up" in unique_ids else unique_ids[0]

    past_cov_ts = _make_covariate_series(df, hist_cov_cols, base_uid)
    future_cov_ts = _make_covariate_series(df, future_cov_cols, base_uid)

    # ── 3. Train split (up to valid_start, excluding validation) ──────────────
    (
        train_target,
        train_past,
        train_future,
        valid_target,
        valid_past,
        valid_future,
        test_target,
        test_past,
        test_future,
    ) = _split_for_single_window(
        target_ts,
        past_cov_ts,
        future_cov_ts,
        cfg.forecast_horizon,
        cfg.input_size,
        cfg.valid_start_ts,
        cfg.test_start_ts,
    )

    # ── 5. Compute rolling windows if needed (from unshifted data) ─────────────
    windows: Optional[list[WindowBoundary]] = None
    if cfg.rolling_window:
        data_start = pd.Timestamp(target_ts.time_index.min())
        data_end = pd.Timestamp(target_ts.time_index.max())
        windows = compute_rolling_windows(
            data_start,
            data_end,
            cfg.n_train_months,
            cfg.n_valid_months,
            cfg.n_test_months,
        )
        logger.info("Computed %d rolling windows for benchmarks", len(windows) if windows else 0)
        if cfg.start_window > 0:
            windows = windows[cfg.start_window:]
            logger.info("Starting from window %d, %d windows remain; start date: %s, end date: %s", cfg.start_window + 1, len(windows), windows[0].train_start, windows[-1].test_end)
    else:
        logger.info("Using single-window split. Train start date: %s, valid start date: %s, test start date: %s",
            train_target.time_index.min() if train_target else "N/A",
            valid_target.time_index.min() if valid_target else "N/A",
            test_target.time_index.min() if test_target else "N/A"
        )

    # ── 6. Run benchmarks ─────────────────────────────────────────────────────
    results: list[BenchmarkResult] = []

    if "naive_seasonal" in cfg.enabled_benchmarks:
        logger.info("Running NaiveSeasonal benchmark...")
        ts_max = pd.Timestamp(target_ts.time_index.max())
        naive_holdout = max(1, int((ts_max - cfg.valid_start_ts).total_seconds() // 3600) + 2)
        train_naive, _ = train_test_split(
            target_ts,
            vertical_split_type="model-aware",
            input_size=cfg.input_size,
            horizon=cfg.forecast_horizon,
            test_size=naive_holdout,
        )
        results.append(
            _run_naive_seasonal(target_ts, cast(TimeSeries, train_naive), cfg)
        )

    if "ridge_regression" in cfg.enabled_benchmarks:
        if windows and cfg.rolling_window:
            logger.info("Running RidgeRegression benchmark (rolling window)...")
            results.append(
                _run_ridge_regression_rolling(
                    df,
                    future_cov_cols,
                    hist_cov_cols,
                    cfg,
                    windows,
                )
            )
        else:
            logger.info("Running RidgeRegression benchmark (single window)...")
            results.append(
                _run_ridge_regression(
                    target_ts=target_ts,
                    train_target=train_target,
                    train_past=train_past,
                    train_future=train_future,
                    past_cov_ts=past_cov_ts,
                    future_cov_ts=future_cov_ts,
                    cfg=cfg,
                )
            )

    if "lasso_regression" in cfg.enabled_benchmarks:
        if windows and cfg.rolling_window:
            logger.info("Running LASSO benchmark (rolling window)...")
            results.append(
                _run_lasso_regression_rolling(
                    df,
                    future_cov_cols,
                    hist_cov_cols,
                    cfg,
                    windows,
                )
            )
        else:
            logger.info("Running LASSO benchmark (single window)...")
            results.append(
                _run_lasso_regression(
                    target_ts=target_ts,
                    train_target=train_target,
                    train_past=train_past,
                    train_future=train_future,
                    past_cov_ts=past_cov_ts,
                    future_cov_ts=future_cov_ts,
                    cfg=cfg,
                )
            )

    if "elasticnet_regression" in cfg.enabled_benchmarks:
        if windows and cfg.rolling_window:
            logger.info("Running ElasticNet benchmark (rolling window)...")
            results.append(
                _run_elasticnet_regression_rolling(
                    df,
                    future_cov_cols,
                    hist_cov_cols,
                    cfg,
                    windows,
                )
            )
        else:
            logger.info("Running ElasticNet benchmark (single window)...")
            results.append(
                _run_elasticnet_regression(
                    target_ts=target_ts,
                    train_target=train_target,
                    train_past=train_past,
                    train_future=train_future,
                    past_cov_ts=past_cov_ts,
                    future_cov_ts=future_cov_ts,
                    cfg=cfg,
                )
            )

    if "lightgbm" in cfg.enabled_benchmarks:
        if windows and cfg.rolling_window:
            logger.info("Running LightGBM benchmark (rolling window)...")
            gbt_result = _run_lightgbm_rolling(
                df,
                future_cov_cols,
                hist_cov_cols,
                cfg,
                windows,
            )
            if gbt_result is not None:
                results.append(gbt_result)
        else:
            logger.info("Running LightGBM benchmark (single window)...")
            gbt_result = _run_lightgbm(
                target_ts=target_ts,
                train_target=train_target,
                train_past=train_past,
                train_future=train_future,
                valid_target=valid_target,
                valid_past=valid_past,
                valid_future=valid_future,
                test_target=test_target,
                test_past=test_past,
                test_future=test_future,
                past_cov_ts=past_cov_ts,
                future_cov_ts=future_cov_ts,
                cfg=cfg,
            )
            if gbt_result is not None:
                results.append(gbt_result)

    if "auto_arima" in cfg.enabled_benchmarks:
        if windows and cfg.rolling_window:
            logger.info("Running AutoARIMA (rolling) benchmark...")
            results.append(
                _run_auto_arima_rolling(target_ts, cfg, windows)
            )
        else:
            logger.info("Running AutoARIMA (single-window) benchmark...")
            results.append(
                _run_auto_arima_single_window(
                    train_target=train_target,
                    test_target=test_target,
                    cfg=cfg,
                )
            )

    if "seasonal_regression" in cfg.enabled_benchmarks:
        if windows and cfg.rolling_window:
            logger.info("Running Seasonal Regression benchmark (rolling window)...")
            results.append(
                _run_seasonal_regression_rolling(
                    df,
                    cfg,
                    windows,
                )
            )
        else:
            logger.info("Running Seasonal Regression benchmark (single window)...")
            results.append(
                _run_seasonal_regression(
                    target_ts=target_ts,
                    train_target=train_target,
                    train_future=train_future,
                    future_cov_ts=future_cov_ts,
                    cfg=cfg,
                )
            )

    # ── 7. Merge all benchmark predictions and collect metadata ──────────────
    if not results:
        logger.warning("No benchmark results were produced.")
        return pd.DataFrame(columns=["ds", "unique_id"]), {}

    merged = results[0].predictions
    all_metadata = {results[0].name: results[0].metadata}
    
    for r in results[1:]:
        if r.predictions.empty:
            continue
        merged = merged.merge(r.predictions, on=["ds", "unique_id"], how="outer")
        all_metadata[r.name] = r.metadata

    merged = merged.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    logger.info(
        "Benchmarks complete: %d rows, columns: %s",
        len(merged),
        list(merged.columns),
    )
    return merged, all_metadata


def merge_changes_from_yaml(config: BenchmarkConfig, tso_name: str, yaml_path: Optional[Path]) -> BenchmarkConfig:
    """Merge benchmark parameters from a YAML file into the metadata dictionary."""
    if yaml_path is None or not yaml_path.exists():
        logger.info("No YAML file provided or file does not exist: %s", yaml_path)
        return config
    try:
        with open(yaml_path, "r") as f:
            yaml_params = yaml.safe_load(f)
            if isinstance(yaml_params, dict):
                for key, value in yaml_params.items():
                    tso_value = value.get(tso_name) if isinstance(value, dict) else value
                    if hasattr(config, key):
                        setattr(config, key, tso_value)
                        logger.info("Updated config parameter from YAML: %s = %s", key, tso_value)
                    else:
                        logger.warning("YAML parameter '%s' does not match any config attribute; skipping.", key)
                logger.info("Merged parameters from YAML file: %s", yaml_path)
            else:
                logger.warning("YAML file does not contain a dictionary at the top level: %s", yaml_path)
    except Exception as e:
        logger.error("Error reading YAML file %s: %s", yaml_path, e)

    return config


def run_benchmarks_from_pipeline_config(
    dataset: pd.DataFrame,
    run_config: dict,
    tso_name: str,
    parameter_yaml_file_path: Optional[Path] = None,
    enabled_benchmarks: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """Convenience wrapper that builds a ``BenchmarkConfig`` from a W&B run config dict.

    This is the main entry point used by ``prediction_pipeline.py``.
    
    Returns
    -------
    tuple[pd.DataFrame, dict]
        (predictions_df, metadata_dict) where metadata_dict contains benchmark parameters
    """
    cfg = BenchmarkConfig(
        forecast_horizon=int(run_config.get("forecast_horizon", 24)),
        input_size=int(run_config.get("input_size", 24)),
        shift_hours=int(run_config.get("shift_hours", 0)),
        tso=tso_name,
        test_start=str(run_config.get("test_start", "2024-04-01")),
        valid_start=str(run_config.get("valid_start", "2024-02-01")),
        train_start=str(run_config.get("train_start", "2020-01-01")),
        add_calendar=bool(run_config.get("add_calendar", True)),
        holidays_path=run_config.get("holidays_path"),
        gb_device=run_config.get("gb_device", "cpu"),
        gb_n_jobs=int(run_config.get("gb_n_jobs", -1)),
        gb_device_index=int(run_config.get("gb_device_index", 0)),
        ridge_alpha=float(run_config.get("ridge_alpha", 1.0)),
        lasso_alpha=float(run_config.get("lasso_alpha", 0.01)),
        elasticnet_alpha=float(run_config.get("elasticnet_alpha", 0.01)),
        elasticnet_l1_ratio=float(run_config.get("elasticnet_l1_ratio", 0.5)),
        linear_scaler_type=run_config.get("linear_scaler_type", run_config.get("local_scaler_type", "standard")),
        rolling_window=str(run_config.get("model_type", "single_window")) == "rolling_window",
        n_train_months=int(run_config.get("n_train_months", 37)),
        n_valid_months=int(run_config.get("n_valid_months", 2)),
        n_test_months=int(run_config.get("n_test_months", 1)),
        start_window=int(run_config.get("start_window", 0)),
        enabled_benchmarks=enabled_benchmarks
        or [
            "naive_seasonal",
            "ridge_regression",
            "lasso_regression",
            "elasticnet_regression",
            "lightgbm",
            "auto_arima",
            "seasonal_regression",
        ],
    )

    cfg = merge_changes_from_yaml(cfg, _normalize_tso_key(tso_name), parameter_yaml_file_path)
    return run_benchmarks(dataset, cfg)
