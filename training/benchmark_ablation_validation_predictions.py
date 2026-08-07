#!/usr/bin/env python
"""
Benchmark knob ablation on rolling windows.

Trains Ridge, LASSO, ElasticNet, LightGBM, and AutoARIMA with different
hyperparameter settings on the first K rolling windows and saves concatenated
test-split predictions.

Output format mirrors ``training/rolling_window_validation_predictions.py`` so
that results can be compared directly against neural-model predictions.

Models / ablated knobs
----------------------
* **Ridge**
    - ``alpha ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100}``
    - Uses all future- and past-covariates with ``input_size`` lags.
    - Handles the decision-time gap via Darts' ``output_chunk_shift``.

* **LASSO**
    - ``alpha ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10}`` by default.
      These values assume the default no-test train+validation standard scaling.
    - Same lag / covariate / shift setup as Ridge, but with L1 regularisation.

* **ElasticNet**
    - ``alpha ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10}`` by default
    - ``l1_ratio = 0.5`` by default
    - Same lag / covariate / shift setup as Ridge, with mixed L1/L2
      regularisation.

* **LightGBM**
        - Fixed ``n_estimators=1_000`` with early stopping evaluated on the
            validation split ``[valid_start, test_start)``.
    - Knob: ``min_data_in_leaf ∈ {20, 50, 100, 200, 400}``.
    - Rationale for choosing ``min_data_in_leaf`` over ``num_leaves``:
      Redispatch features are highly collinear (multiple weather variables,
      production forecasts, price forecasts).  With collinear features the model
      can create near-equivalent splits across many feature combinations, producing
      leaves with very few samples that overfit to noise.  ``min_data_in_leaf``
      (a.k.a. ``min_child_samples`` in the sklearn API) directly enforces a minimum
      leaf support, preventing these noise-capturing collinear splits and reducing
      variance more robustly than ``num_leaves``, which only limits tree width
      without addressing leaf sparsity.

* **AutoARIMA**
    - No covariates (avoids rank-deficiency with many collinear regressors).
    - Orders constrained to ``p ≤ 3, q ≤ 2, P ≤ 3, Q ≤ 2`` with
      seasonal period ``m=24`` (hourly data).
    - Trained once per direction per window; ``retrain=False`` during
      ``historical_forecasts`` (state updated as new observations arrive).
    - **Note:** ``statsforecast.models.AutoARIMA`` has no ``n_jobs`` /
      ``n_cores`` parameter.  The underlying ``auto_arima_f`` search is a
      sequential stepwise algorithm; CPU parallelism cannot be controlled
      at the model level.

* **Croston TSB** (Teunter-Syntetos-Babai)
    - Designed for intermittent / sparse demand series (40–80 % zeros).
    - Fixed ``alpha_d = 0.1`` (demand smoothing).
    - Ablated knob: ``alpha_p ∈ {0.01, 0.03, 0.05, 0.1, 0.2}`` (probability
      smoothing).
    - Applied per window and component (same pattern as AutoARIMA).
    - No covariates (TSB is a univariate exponential-smoothing model).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from darts import TimeSeries, concatenate
from darts.models import AutoARIMA
from darts.models import Croston
from darts.models import LightGBMModel
from darts.explainability import ShapExplainer
from darts.utils.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

from training.data_prep import (
    CALENDAR_COLS,
    add_calendar_features,
    classify_covariates,
    load_dataset,
    to_nixtla_format,
    build_static_df,
)
from training.train_pipeline import set_n_threads
from training.runner import _compute_rolling_windows, WindowBoundary

# Private but stable in Darts 0.40:  SKLearnModel.__init__(model=...) lets us
# plug in any sklearn-compatible estimator while keeping all Darts lag/shift
# machinery.
from darts.models.forecasting.linear_regression_model import SKLearnModel

logger = logging.getLogger(__name__)

# ── Per-TSO input_size helpers ────────────────────────────────────────────────

def _parse_input_sizes(raw: str | list[str]) -> dict[str, int]:
    """Parse ``--input-size`` CLI tokens into a tso→size mapping.

    Accepts:
    * a single bare integer (``"36"``), stored under the key ``"__default__"``
    * one or more ``TSO:INT`` pairs (e.g. ``"TenneT_DE:36"``)
    * a mixture of both (bare integer serves as the fallback default)

    Examples
    --------
    ``["36"]``                          → ``{"__default__": 36}``
    ``["TenneT_DE:36", "Amprion:24"]``  → ``{"TenneT_DE": 36, "Amprion": 24}``
    ``["24", "TenneT_DE:36"]``          → ``{"__default__": 24, "TenneT_DE": 36}``
    """
    result: dict[str, int] = {}
    for token in raw.split() if isinstance(raw, str) else raw:
        if ":" in token:
            tso, size = token.split(":", 1)
            result[tso.strip()] = int(size.strip())
        else:
            result["__default__"] = int(token.strip())
    return result or {"__default__": 24}


def _parse_float_grid(raw: str | list[str] | Sequence[float]) -> list[float]:
    """Parse comma/space-separated float grid CLI values."""
    if isinstance(raw, str):
        tokens = raw.replace(",", " ").split()
    else:
        tokens = []
        for item in raw:
            if isinstance(item, float):
                tokens.append(str(item))
            else:
                tokens.extend(str(item).replace(",", " ").split())
    values = [float(token) for token in tokens if str(token).strip()]
    if not values:
        raise ValueError("Float grid must contain at least one value.")
    return values


def _resolve_input_size(input_size_map: dict[str, int], tso: str) -> int:
    """Return the per-TSO input_size, falling back to ``__default__`` (24)."""
    return input_size_map.get(tso, input_size_map.get("__default__", 24))


# ── Ablation grids ─────────────────────────────────────────────────────────────
LASSO_ALPHAS: list[float] = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1_000.0]
ELASTICNET_ALPHAS: list[float] = LASSO_ALPHAS.copy()
RIDGE_ALPHAS = LASSO_ALPHAS.copy()
ELASTICNET_L1_RATIOS: list[float] = [0.25, 0.5, 0.75]

# min_data_in_leaf (sklearn API: min_child_samples) chosen over num_leaves –
# see module docstring for rationale.
LGBM_MIN_DATA_IN_LEAF: list[int] = [20, 50, 100, 200, 400]

LGBM_N_ESTIMATORS: int = 1_000

# Croston TSB parameters
# alpha_d: demand-level smoothing (fixed per the user requirement)
# alpha_p: probability-of-demand smoothing (ablation grid)
CROSTON_ALPHA_D: list[float] = [0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
CROSTON_ALPHA_P: list[float] = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


# ── Ridge wrapper ──────────────────────────────────────────────────────────────
class RidgeRegressionModel(SKLearnModel):
    """Darts regression model backed by :class:`sklearn.linear_model.Ridge`.

    Identical lag / shift API to :class:`darts.models.LinearRegressionModel`
    but uses L2-regularised regression instead of OLS.
    """

    def __init__(
        self,
        lags: Optional[int] = None,
        lags_past_covariates: Optional[int] = None,
        lags_future_covariates=None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        alpha: float = 1.0,
        multi_models: bool = True,
        use_static_covariates: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            model=Ridge(alpha=alpha, fit_intercept=True),
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            random_state=random_state,
        )
        self.alpha = alpha  # stored for logging / repr


class LassoRegressionModel(SKLearnModel):
    """Darts regression model backed by :class:`sklearn.linear_model.Lasso`."""

    def __init__(
        self,
        lags: Optional[int] = None,
        lags_past_covariates: Optional[int] = None,
        lags_future_covariates=None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        alpha: float = 1.0,
        max_iter: int = 5_000,
        tol: float = 1e-3,
        selection: str = "random",
        multi_models: bool = False,
        use_static_covariates: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            model=Lasso(
                alpha=alpha,
                fit_intercept=True,
                max_iter=max_iter,
                tol=tol,
                selection=selection,
                random_state=random_state,
            ),
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            random_state=random_state,
        )
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection


class ElasticNetRegressionModel(SKLearnModel):
    """Darts regression model backed by :class:`sklearn.linear_model.ElasticNet`."""

    def __init__(
        self,
        lags: Optional[int] = None,
        lags_past_covariates: Optional[int] = None,
        lags_future_covariates=None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 5_000,
        tol: float = 1e-3,
        selection: str = "random",
        multi_models: bool = False,
        use_static_covariates: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            model=ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=max_iter,
                tol=tol,
                selection=selection,
                random_state=random_state,
            ),
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            random_state=random_state,
        )
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection


# ── TimeSeries helpers ─────────────────────────────────────────────────────────
def _make_target_series(
    dataset: pd.DataFrame,
) -> tuple[TimeSeries, list[str]]:
    """Convert Nixtla-format long DataFrame → multivariate Darts TimeSeries."""
    unique_ids = sorted(dataset["unique_id"].dropna().unique().tolist())
    static_df = build_static_df().set_index("unique_id")
    series_list = [
        TimeSeries.from_dataframe(
            dataset.loc[dataset["unique_id"] == uid, ["ds", "y"]],
            time_col="ds",
            value_cols="y",
        )
        for uid in unique_ids
    ]
    target_ts = concatenate(series_list, axis=1)
    target_ts = target_ts.with_columns_renamed(
        target_ts.components.to_list(), unique_ids
    )
    target_ts = target_ts.with_static_covariates(static_df.loc[unique_ids])
    return target_ts, unique_ids


def _make_covariate_series(
    dataset: pd.DataFrame,
    covariate_cols: list[str],
    base_unique_id: str,
) -> Optional[TimeSeries]:
    """Build a shared covariate TimeSeries from one unique_id's rows."""
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


def _concat_forecasts(
    forecasts: Union[TimeSeries, list[TimeSeries], list[list[TimeSeries]]],
) -> TimeSeries:
    if isinstance(forecasts, TimeSeries):
        return forecasts
    if not forecasts:
        raise ValueError("No forecasts returned from historical_forecasts.")
    if isinstance(forecasts[0], list):
        flat: list[TimeSeries] = [ts for sub in forecasts for ts in sub]  # type: ignore[union-attr]
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
    """Run ``historical_forecasts(retrain=False)`` and clip to target range."""
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


def _timeseries_to_long(ts: TimeSeries, col: str) -> pd.DataFrame:
    """Convert multivariate TimeSeries → Nixtla-style long DataFrame."""
    df = ts.to_dataframe().copy()
    df.index.name = "ds"
    df = df.reset_index().melt(id_vars=["ds"], var_name="unique_id", value_name=col)
    df[col] = df[col].clip(lower=0)
    return df


def _scale_timeseries_before_test(
    ts: Optional[TimeSeries],
    test_start: pd.Timestamp,
    scaler_type: Optional[str],
) -> tuple[Optional[TimeSeries], Optional[StandardScaler]]:
    """Fit scaler on train+validation and apply it to the full TimeSeries."""
    if ts is None or scaler_type in {None, "none"}:
        return ts, None
    if scaler_type != "standard":
        raise ValueError(f"Unsupported scaler_type={scaler_type!r}; use 'standard' or 'none'.")

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
    scaled_values = scaler.transform(ts.values(copy=False))
    return ts.with_values(scaled_values), scaler


def _transform_timeseries(
    ts: Optional[TimeSeries],
    scaler: Optional[StandardScaler],
) -> Optional[TimeSeries]:
    """Apply an already-fitted scaler to a TimeSeries."""
    if ts is None or scaler is None:
        return ts
    return ts.with_values(scaler.transform(ts.values(copy=False)))


def _inverse_transform_timeseries(
    ts: TimeSeries,
    scaler: Optional[StandardScaler],
) -> TimeSeries:
    """Map scaled target predictions back to physical units."""
    if scaler is None:
        return ts
    return ts.with_values(scaler.inverse_transform(ts.values(copy=False)))


# ── Data preparation ──────────────────────────────────────────────────────────
def _prepare_benchmark_data(
    nixtla_df: pd.DataFrame,
    tso: str,
    add_calendar: bool,
    holidays_path: Optional[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add calendar features and classify covariates on unshifted (physical) data."""
    df = nixtla_df.copy()
    if add_calendar:
        df = add_calendar_features(
            df,
            reference_time=df["ds"],
            tso=tso,
            holidays_path=holidays_path,
        )
    feature_cols = [c for c in df.columns if c not in {"ds", "y", "unique_id"}]
    future_cov_cols, hist_cov_cols = classify_covariates(feature_cols)
    if add_calendar:
        for c in CALENDAR_COLS:
            if c in df.columns and c not in future_cov_cols:
                future_cov_cols.append(c)
    return df, future_cov_cols, hist_cov_cols


def _split_train_val(
    target_ts: TimeSeries,
    cov_ts: Optional[TimeSeries],
    input_size: int,
    forecast_horizon: int,
    holdout_hours: int,
) -> tuple[TimeSeries, Optional[TimeSeries]]:
    """Split TimeSeries so training ends ``holdout_hours`` before the series end."""
    train, _ = train_test_split(
        target_ts,
        vertical_split_type="model-aware",
        input_size=input_size,
        horizon=forecast_horizon,
        test_size=holdout_hours,
    )
    if cov_ts is None:
        return cast(TimeSeries, train), None
    train_cov, _ = train_test_split(
        cov_ts,
        vertical_split_type="model-aware",
        input_size=input_size,
        horizon=forecast_horizon,
        test_size=holdout_hours,
    )
    return cast(TimeSeries, train), cast(TimeSeries, train_cov)


def _assert_train_ends_before_valid(
    train_ts: TimeSeries,
    valid_start: pd.Timestamp,
    model_name: str,
) -> None:
    """Guard against leakage from validation into training."""
    if pd.Timestamp(train_ts.end_time()) >= valid_start:
        raise RuntimeError(
            f"{model_name} training leaks into validation: "
            f"train_end={train_ts.end_time()} valid_start={valid_start}"
        )


def _expected_n_points(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> int:
    """Return expected count of hourly points in [start, end_exclusive)."""
    return int((end_exclusive - start).total_seconds() // 3600)


def _clip_interval_to_series(
    ts: TimeSeries,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Clip an inclusive interval to a TimeSeries index range.

    Darts ``drop_before`` / ``drop_after`` require the split point to be inside
    the series index. Some prepared datasets can miss one or more edge
    timestamps, so we clamp requested bounds to the available range first.
    """
    ts_start = pd.Timestamp(ts.start_time())
    ts_end = pd.Timestamp(ts.end_time())
    clipped_start = max(start, ts_start)
    clipped_end = min(end, ts_end)
    if clipped_start > clipped_end:
        raise RuntimeError(
            "Requested interval does not overlap TimeSeries bounds: "
            f"requested=[{start}, {end}] available=[{ts_start}, {ts_end}]"
        )
    return clipped_start, clipped_end


def _add_horizon_col(
    df: pd.DataFrame,
    pred_start: pd.Timestamp,
    forecast_horizon: int,
) -> pd.DataFrame:
    """Add a 1-based ``horizon`` column (position within each forecast stride)."""
    df = df.copy()
    h_off = ((df["ds"] - pred_start).dt.total_seconds() / 3600).astype(int)
    df["horizon"] = (h_off % forecast_horizon + 1).astype(int)  # type: ignore[assignment]
    return df


def _assert_test_alignment(
    df: pd.DataFrame,
    wb: WindowBoundary,
    forecast_horizon: int,
    model_name: str,
) -> None:
    """Validate hourly coverage and horizon alignment on [test_start, test_end)."""
    if df.empty:
        raise RuntimeError(f"{model_name} produced no predictions on the test window.")

    expected_end = wb.test_end - pd.Timedelta(hours=1)
    expected_index = pd.date_range(start=wb.test_start, end=expected_end, freq="h")

    for uid, g in df.groupby("unique_id", sort=False):
        g_sorted = g.sort_values("ds")
        got_index = pd.DatetimeIndex(pd.to_datetime(g_sorted["ds"]).unique())
        if len(got_index) != len(expected_index) or not got_index.equals(expected_index):
            raise RuntimeError(
                f"{model_name} prediction timestamps misaligned for unique_id={uid}: "
                f"expected [{wb.test_start}, {wb.test_end}) hourly grid with "
                f"{len(expected_index)} points, got {len(got_index)} points "
                f"from {got_index.min()} to {got_index.max()}."
            )

        if "horizon" in g_sorted.columns:
            got_h = g_sorted["horizon"].to_numpy(dtype=np.int64)
            expected_h = (np.arange(len(expected_index), dtype=np.int64) % forecast_horizon) + 1
            if not np.array_equal(got_h, expected_h):
                raise RuntimeError(
                    f"{model_name} horizon misalignment for unique_id={uid}: "
                    "horizon should cycle 1..forecast_horizon on the hourly test grid."
                )


def _fmt_alpha(alpha: float) -> str:
    """Format a Ridge alpha value for use in column / file names."""
    s = f"{alpha:g}"  # e.g. 0.0001, 0.001, 0.01, 0.1, 1, 10, 100
    return s.replace(".", "p").replace("-", "neg")


# ── Regularized linear-model window ───────────────────────────────────────────
def _run_regularized_linear_window(
    window_df: pd.DataFrame,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    wb: WindowBoundary,
    input_size: int,
    forecast_horizon: int,
    shift_hours: int,
    model_cls: type[SKLearnModel],
    model_label: str,
    col_name: str,
    model_kwargs: Optional[dict[str, Any]] = None,
    scaler_type: Optional[str] = "standard",
) -> tuple[pd.DataFrame, str]:
    """Train one sklearn-backed Darts linear model and predict test period."""
    target_ts, unique_ids = _make_target_series(window_df)
    base_uid = "up" if "up" in unique_ids else unique_ids[0]

    past_cov_ts = _make_covariate_series(window_df, hist_cov_cols, base_uid)
    future_cov_ts = _make_covariate_series(window_df, future_cov_cols, base_uid)

    # Training data ends at valid_start; validation+test are held out.
    holdout_hours = max(
        1, int((wb.test_end - wb.valid_start).total_seconds() // 3600)
    )

    train_target, train_past = _split_train_val(
        target_ts, past_cov_ts, input_size, forecast_horizon, holdout_hours
    )
    _assert_train_ends_before_valid(train_target, wb.valid_start, model_label)
    _, train_future = _split_train_val(
        target_ts, future_cov_ts, input_size, forecast_horizon, holdout_hours
    )

    target_ts_model, target_scaler = _scale_timeseries_before_test(
        target_ts, wb.test_start, scaler_type
    )
    train_target_model = cast(TimeSeries, _transform_timeseries(train_target, target_scaler))
    past_cov_ts_model, past_scaler = _scale_timeseries_before_test(
        past_cov_ts, wb.test_start, scaler_type
    )
    train_past_model = cast(Optional[TimeSeries], _transform_timeseries(train_past, past_scaler))
    future_cov_ts_model, future_scaler = _scale_timeseries_before_test(
        future_cov_ts, wb.test_start, scaler_type
    )
    train_future_model = cast(Optional[TimeSeries], _transform_timeseries(train_future, future_scaler))

    model = model_cls(
        lags=input_size,
        lags_past_covariates=input_size if past_cov_ts is not None else None,
        lags_future_covariates=(input_size, 1) if future_cov_ts is not None else None,
        output_chunk_length=forecast_horizon,
        output_chunk_shift=shift_hours,
        **(model_kwargs or {}),
    )
    model.fit(
        train_target_model,
        past_covariates=train_past_model,
        future_covariates=train_future_model,
    )

    # With output_chunk_shift, start ``shift_hours`` steps early so output
    # lands at physical time test_start.
    start_time = wb.test_start - pd.Timedelta(hours=shift_hours)

    preds_ts = _historical_forecasts_full_tail(
        model,
        target_ts=cast(TimeSeries, target_ts_model),
        forecast_horizon=forecast_horizon,
        stride=forecast_horizon,
        start=start_time,
        past_covariates=past_cov_ts_model,
        future_covariates=future_cov_ts_model,
    )
    preds_ts = _inverse_transform_timeseries(preds_ts, target_scaler)

    pred_end = wb.test_end - pd.Timedelta(hours=1)
    df = _timeseries_to_long(preds_ts, col_name)
    df = df[(df["ds"] >= wb.test_start) & (df["ds"] <= pred_end)].copy()
    df = _add_horizon_col(df, wb.test_start, forecast_horizon)
    _assert_test_alignment(df, wb, forecast_horizon, model_label)
    return df, col_name


# ── LightGBM window ───────────────────────────────────────────────────────────
def _run_lgbm_window(
    window_df: pd.DataFrame,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    wb: WindowBoundary,
    input_size: int,
    forecast_horizon: int,
    shift_hours: int,
    min_data_in_leaf: int,
    n_estimators: int,
    early_stopping_rounds: int,
    n_jobs: int,
    random_seed: int,
    lgb_device: str,
    lgb_device_index: int,
) -> tuple[pd.DataFrame, str, pd.DataFrame, list[dict[str, str | int | float]]]:
    """Train LightGBM with early stopping and predict on the test period.

    Training uses ``[train_start, valid_start)``. Early stopping uses the
    validation split ``[valid_start, test_start)``. The model then generates
    predictions on the test split ``[test_start, test_end)``.

    Returns
    -------
    tuple of (predictions DataFrame, column name, feature importances DataFrame, model details dict).
    The feature importances DataFrame has columns ``feature``, ``importance``,
    ``min_data_in_leaf``;  it is empty when importances could not be extracted.
    The model details dict contains metadata about the trained model.
    """
    target_ts, unique_ids = _make_target_series(window_df)
    base_uid = "up" if "up" in unique_ids else unique_ids[0]

    past_cov_ts = _make_covariate_series(window_df, hist_cov_cols, base_uid)
    future_cov_ts = _make_covariate_series(window_df, future_cov_cols, base_uid)

    # --- Split for LightGBM early stopping -----------------------------------
    # Training: [train_start, valid_start)
    # Val (early stopping): [valid_start, test_start)
    # Historical forecasts: whole target_ts, predictions filtered to test window
    # Holdout for Darts split covers validation + full test period.
    lgbm_holdout_hours = max(
        1,
        int((wb.test_end - wb.valid_start).total_seconds() // 3600),
    )

    train_target, train_past = _split_train_val(
        target_ts, past_cov_ts, input_size, forecast_horizon, lgbm_holdout_hours
    )
    _assert_train_ends_before_valid(train_target, wb.valid_start, "LightGBM")
    _, train_future = _split_train_val(
        target_ts, future_cov_ts, input_size, forecast_horizon, lgbm_holdout_hours
    )

    # If calendar features are used, treat them as categorical future covariates
    if future_cov_cols and future_cov_ts and any(c in future_cov_cols for c in CALENDAR_COLS):
        cat_future_cov_ts = CALENDAR_COLS
        # future_cov_ts = future_cov_ts.astype({col: np.int16 for col in CALENDAR_COLS})
    else:
        cat_future_cov_ts = None

    # Validation slice for early stopping.
    # We need to extend backward by ``input_size`` steps so that
    # ``create_lagged_training_data`` (called internally by Darts'
    # ``_add_val_set_to_kwargs``) can build complete lag feature vectors for
    # the first eval sample at ``valid_start``.
    # output_chunk_shift needs additional history so the first valid timestamp
    # is part of the eval set used by early stopping.
    val_context_hours = input_size + max(0, shift_hours)
    val_context_start = wb.valid_start - pd.Timedelta(hours=val_context_hours)
    requested_val_es_end = wb.test_start - pd.Timedelta(hours=1)
    val_slice_start, val_slice_end = _clip_interval_to_series(
        target_ts,
        val_context_start,
        requested_val_es_end,
    )
    val_target_es = target_ts.drop_after(val_slice_end).drop_before(val_slice_start)
    if past_cov_ts is not None:
        past_slice_start, past_slice_end = _clip_interval_to_series(
            past_cov_ts,
            val_context_start,
            requested_val_es_end,
        )
        val_past_es = past_cov_ts.drop_after(past_slice_end).drop_before(past_slice_start)
    else:
        val_past_es = None

    val_eval_start, val_eval_end = _clip_interval_to_series(
        val_target_es,
        wb.valid_start,
        requested_val_es_end,
    )
    expected_val_points = _expected_n_points(
        val_eval_start,
        val_eval_end + pd.Timedelta(hours=1),
    )
    actual_val_points = len(
        val_target_es.drop_before(val_eval_start).drop_after(val_eval_end)
    )
    if actual_val_points <= 0:
        raise RuntimeError(
            "LightGBM early-stopping validation slice is empty after clipping: "
            f"requested=[{wb.valid_start}, {wb.test_start}) "
            f"effective=[{val_eval_start}, {val_eval_end + pd.Timedelta(hours=1)})"
        )
    if actual_val_points < expected_val_points:
        logger.warning(
            "LightGBM early-stopping validation has missing timestamps: "
            "expected=%d actual=%d for requested=[%s, %s) effective=[%s, %s). "
            "Continuing with clipped/intersected validation samples.",
            expected_val_points,
            actual_val_points,
            wb.valid_start,
            wb.test_start,
            val_eval_start,
            val_eval_end + pd.Timedelta(hours=1),
        )
    if future_cov_ts is not None:
        future_slice_start, future_slice_end = _clip_interval_to_series(
            future_cov_ts,
            val_context_start,
            requested_val_es_end,
        )
        val_future_es = future_cov_ts.drop_after(future_slice_end).drop_before(
            future_slice_start
        )
    else:
        val_future_es = None

    # --- Build LightGBM model ------------------------------------------------
    lgbm_kwargs: dict = dict(
        boosting_type="gbdt",
        objective="regression_l1",
        metric="mae",
        n_estimators=n_estimators,
        min_child_samples=min_data_in_leaf,  # sklearn API ≡ min_data_in_leaf
        verbosity=-1,
        n_jobs=n_jobs,
        random_state=random_seed,
    )
    if lgb_device == "gpu":
        lgbm_kwargs.update(
            device="gpu",
            gpu_device_id=lgb_device_index,
        )
        os.environ.setdefault("OCL_ICD_VENDORS", "/home/jovyan/opencl_vendors")

    model = LightGBMModel(
        lags=input_size,
        lags_past_covariates=input_size if past_cov_ts is not None else None,
        lags_future_covariates=(input_size, 1) if future_cov_ts is not None else None,
        output_chunk_length=forecast_horizon,
        output_chunk_shift=shift_hours,
        categorical_future_covariates=cat_future_cov_ts,
        **lgbm_kwargs,
    )

    # Early stopping is activated by (a) providing val_series so Darts builds
    # the eval_set and (b) passing lgb.early_stopping() as a fit-time callback.
    use_early_stopping = early_stopping_rounds > 0
    fit_kwargs: dict = {}
    if use_early_stopping:
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

    model.fit(
        train_target,
        past_covariates=train_past,
        future_covariates=train_future,
        val_series=val_target_es if use_early_stopping else None,
        val_past_covariates=val_past_es if use_early_stopping else None,
        val_future_covariates=val_future_es if use_early_stopping else None,
        **fit_kwargs,
    )

    # --- Log post-fit LightGBM parameters and extract feature importances ----
    # With multi_models=True, Darts stores one LGBMRegressor per
    # (output_step × component). Summarise best_iteration_ and n_estimators_.
    feat_imp_df: pd.DataFrame = pd.DataFrame(
        columns=["feature", "importance", "min_data_in_leaf"]
    )
    model_details: list[dict[str, str | int | float]] = []

    try:
        sub_models: list[lgb.LGBMRegressor] = model.model.estimators_  # type: ignore[attr-defined]
        best_iters_arr = np.array(
            [
                m.best_iteration_ if getattr(m, "best_iteration_", None) is not None else n_estimators
                for m in sub_models
            ],
            dtype=np.int64,
        )
        n_estimators_arr = np.array(
            [
                max(m.booster_.current_iteration(), m.booster_.num_trees()) if getattr(m, "booster_", None) is not None else n_estimators
                for m in sub_models
            ],
            dtype=np.int64,
        )

        n_components = len(train_target.components)
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
                model_details.append({
                    "validation_window_start": wb.valid_start.strftime("%Y-%m-%d"),
                    "validation_window_end": (wb.test_start - pd.Timedelta(hours=1)).strftime("%Y-%m-%d"),
                    "component": component,
                    "min_data_in_leaf": min_data_in_leaf,
                    "best_iteration_min": n_best_min,
                    "best_iteration_median": n_best_med,
                    "best_iteration_max": n_best_max,
                    "n_estimators_min": n_est_min,
                    "n_estimators_median": n_est_med,
                    "n_estimators_max": n_est_max,
                })
        else:
            logger.info(
                "LightGBM fitted (%d sub-models): no early stopping "
                "(n_estimators=%d, min_data_in_leaf=%d)",
                len(sub_models),
                n_estimators,
                min_data_in_leaf,
            )

        # Aggregate feature importances across components and sub-models (mean over all sub-models)
        imp_records = []
        explainer = ShapExplainer(model=model)
        explainer_result = explainer.explain(
            foreground_series=val_target_es,
            foreground_past_covariates=val_past_es,
            foreground_future_covariates=val_future_es,
        )

        for component in train_target.components:
            for horizon in range(1, forecast_horizon + 1):
                # Extract importances for this component and horizon
                comp_horiz_imp = explainer_result.get_explanation(
                    component=component,
                    horizon=horizon,
                )
                if comp_horiz_imp is not None and hasattr(comp_horiz_imp, "to_dataframe"):
                    imp_df = cast(Any, comp_horiz_imp).to_dataframe()
                    imp_df["direction"] = str(component)
                    imp_df["horizon"] = horizon
                    imp_records.append(imp_df)

        if imp_records:
            feat_imp_df = pd.concat(imp_records, ignore_index=True)
            feat_imp_df["min_data_in_leaf"] = min_data_in_leaf
    except Exception as log_exc:
        logger.debug("Could not read LightGBM post-fit stats: %s", log_exc)

    # --- Generate test predictions -------------------------------------------
    start_time = wb.test_start - pd.Timedelta(hours=shift_hours)
    pred_end = wb.test_end - pd.Timedelta(hours=1)

    preds_ts = _historical_forecasts_full_tail(
        model,
        target_ts=target_ts,
        forecast_horizon=forecast_horizon,
        stride=forecast_horizon,
        start=start_time,
        past_covariates=past_cov_ts,
        future_covariates=future_cov_ts,
    )

    col_name = f"lightgbm_mldl_{min_data_in_leaf}"
    df = _timeseries_to_long(preds_ts, col_name)
    df = df[(df["ds"] >= wb.test_start) & (df["ds"] <= pred_end)].copy()
    df = _add_horizon_col(df, wb.test_start, forecast_horizon)
    _assert_test_alignment(df, wb, forecast_horizon, "LightGBM")
    return df, col_name, feat_imp_df, model_details


# ── AutoARIMA window ───────────────────────────────────────────────────────────

def _run_arima_window(
    window_df: pd.DataFrame,
    pred_col_name: str,
    wb: WindowBoundary,
    input_size: int,
    forecast_horizon: int,
    max_p: int,
    max_q: int,
    max_P: int,
    max_Q: int,
    seasonal_period: int,
    use_approximation: bool,
) -> tuple[pd.DataFrame, str, dict[str, dict]]:
    """Fit AutoARIMA per direction; predict on the test period.

    AutoARIMA does not support ``output_chunk_shift``, so it always predicts in
    physical time directly (consistent with ``benchmarks.py`` convention).
    Separate models are fitted for each direction (``up`` / ``down``).
    ``retrain=False`` is used during ``historical_forecasts``; the fitted model
    updates its state as new observations become available.

    Returns
    -------
    tuple of (predictions DataFrame, column name, selected ARIMA orders per
    component).  The orders dict maps component name to a sub-dict with keys
    ``order_pqd`` (p, q, d) and ``seasonal_order_PQDm`` (P, Q, D, m).
    """
    target_ts, unique_ids = _make_target_series(window_df)
    pred_end = wb.test_end - pd.Timedelta(hours=1)
    holdout_hours = max(
        1, int((wb.test_end - wb.valid_start).total_seconds() // 3600)
    )

    all_component_preds: list[pd.DataFrame] = []
    arima_orders: dict[str, dict] = {}

    for component in target_ts.components:
        comp_ts = target_ts[component]

        # Train on [train_start, valid_start)
        train_comp, _ = train_test_split(
            comp_ts,
            vertical_split_type="model-aware",
            input_size=input_size,
            horizon=forecast_horizon,
            test_size=holdout_hours,
        )
        _assert_train_ends_before_valid(
            cast(TimeSeries, train_comp), wb.valid_start, f"AutoARIMA[{component}]"
        )

        arima = AutoARIMA(
            max_p=max_p,
            max_q=max_q,
            max_P=max_P,
            max_Q=max_Q,
            season_length=seasonal_period,
            ic="aicc",
            approximation=use_approximation,
        )
        arima.fit(cast(TimeSeries, train_comp))

        col_name = pred_col_name
        try:
            non_seasonal_order = arima.model.model_["arma"][:3]  # type: ignore[union-attr]
            seasonal_order = arima.model.model_["arma"][3:]  # type: ignore[union-attr]
            coefficients = arima.model.model_["coef"]  # type: ignore[union-attr]
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

        # No output_chunk_shift → start directly at physical test_start
        comp_preds = _historical_forecasts_full_tail(
            arima,
            target_ts=comp_ts,
            forecast_horizon=forecast_horizon,
            stride=forecast_horizon,
            start=wb.test_start,
        )
        all_component_preds.append(_timeseries_to_long(comp_preds, col_name))  # type: ignore[union-attr]

    df = pd.concat(all_component_preds, ignore_index=True)
    df = df[(df["ds"] >= wb.test_start) & (df["ds"] <= pred_end)].copy()
    df = _add_horizon_col(df, wb.test_start, forecast_horizon)
    return df, col_name, arima_orders


# ── Croston TSB window ────────────────────────────────────────────────────────
def _run_croston_window(
    window_df: pd.DataFrame,
    wb: WindowBoundary,
    forecast_horizon: int,
    alpha_d: float,
    alpha_p: float,
    refit: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Fit Croston TSB per direction and predict on the test period.

    The model is fitted once per component on ``[train_start, valid_start)``.
    ``historical_forecasts(retrain=False)`` is then used to produce rolling
    ``forecast_horizon``-step predictions over the test window.  Because
    Darts' ``StatsForecastModel`` supports transferable-series prediction, the
    model is internally re-applied to each expanding input window at each stride
    step, keeping the demand and probability estimates up to date.

    No covariates are used – TSB is a univariate exponential-smoothing model.
    """
    target_ts, _ = _make_target_series(window_df)
    pred_end = wb.test_end - pd.Timedelta(hours=1)

    alpha_d_str = _fmt_alpha(alpha_d)
    alpha_p_str = _fmt_alpha(alpha_p)
    col_name = f"croston_tsb_ad{alpha_d_str}_ap{alpha_p_str}"

    all_component_preds: list[pd.DataFrame] = []

    for component in target_ts.components:
        comp_ts = target_ts[component]

        # Train on [train_start, valid_start) only.
        train_comp = comp_ts.drop_after(wb.valid_start - pd.Timedelta(hours=1))
        _assert_train_ends_before_valid(
            cast(TimeSeries, train_comp), wb.valid_start, f"Croston[{component}]"
        )

        model = Croston(version="tsb", alpha_d=alpha_d, alpha_p=alpha_p)
        model.fit(cast(TimeSeries, train_comp))

        if refit:
            comp_preds = _concat_forecasts(
                model.historical_forecasts(
                    comp_ts,
                    forecast_horizon=forecast_horizon,
                    stride=forecast_horizon,
                    last_points_only=False,
                    retrain=True,
                    verbose=False,
                    start=wb.test_start,
                )
            )
        else:
            test_n_points = int((wb.test_end - wb.test_start).total_seconds() // 3600) + 1
            comp_preds = model.predict(n=test_n_points).slice_intersect(comp_ts)
        all_component_preds.append(_timeseries_to_long(comp_preds, col_name))

    df = pd.concat(all_component_preds, ignore_index=True)
    df = df[(df["ds"] >= wb.test_start) & (df["ds"] <= pred_end)].copy()
    df = _add_horizon_col(df, wb.test_start, forecast_horizon)
    return df, col_name


# ── Finalize predictions ───────────────────────────────────────────────────────
def _finalize_preds(
    preds_df: pd.DataFrame,
    pred_col: str,
    actuals_df: pd.DataFrame,
    model_name: str,
    model_alias: str,
    input_size: int,
    window_index: int,
    wb: WindowBoundary,
    tso: str,
) -> pd.DataFrame:
    """Merge actuals and attach metadata columns."""
    merged = preds_df.merge(
        actuals_df[["ds", "unique_id", "y"]],
        on=["ds", "unique_id"],
        how="left",
    )
    if pred_col in merged.columns:
        merged = merged.rename(columns={pred_col: "y_hat"})
    elif "y_hat" not in merged.columns:
        raise KeyError(f"Expected column '{pred_col}' not found in predictions.")

    merged["y_hat"] = np.clip(merged["y_hat"], 0, np.inf)
    merged["model_name"] = model_name
    merged["model_alias"] = model_alias
    merged["input_size"] = input_size
    merged["window_index"] = window_index
    merged["eval_split"] = "test"
    merged["test_start"] = wb.test_start
    merged["test_end"] = wb.test_end - pd.Timedelta(hours=1)
    # Legacy naming retained for backward compatibility with existing readers.
    merged["val_start"] = wb.valid_start
    merged["val_end"] = wb.test_end - pd.Timedelta(hours=1)
    merged["tso"] = tso

    cols = [
        "ds",
        "unique_id",
        "y",
        "horizon",
        "y_hat",
        "model_name",
        "model_alias",
        "input_size",
        "window_index",
        "eval_split",
        "test_start",
        "test_end",
        "val_start",
        "val_end",
        "tso",
    ]
    return merged[[c for c in cols if c in merged.columns]]


# ── CLI ────────────────────────────────────────────────────────────────────────
def _normalize_models(raw: list[str]) -> list[str]:
    allowed = {"ridge", "lasso", "elasticnet", "lightgbm", "arima", "croston"}
    models: list[str] = []
    for item in raw:
        models.extend([p.strip().lower() for p in item.split(",") if p.strip()])
    invalid = [m for m in models if m not in allowed]
    if invalid:
        raise SystemExit(
            f"Invalid model(s): {invalid}.  Allowed: {sorted(allowed)}"
        )
    seen: set[str] = set()
    return [m for m in models if not (m in seen or seen.add(m))]  # type: ignore[func-returns-value]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Ablation study over benchmark hyperparameter knobs "
            "(Ridge/LASSO/ElasticNet alpha, LightGBM min_data_in_leaf, AutoARIMA). "
            "Generates rolling test-split predictions in the same format as "
            "rolling_window_validation_predictions.py."
        )
    )

    # --- Data ---
    p.add_argument("--dataset-path", required=True, help="Path to parquet dataset.")
    p.add_argument(
        "--direction", default="both", choices=["up", "down", "both"]
    )
    p.add_argument(
        "--shift-hours",
        type=int,
        default=9,
        help="Decision-time gap in hours (applied via output_chunk_shift for "
        "Ridge / LASSO / ElasticNet / LightGBM; ARIMA always predicts in physical time).",
    )
    p.add_argument("--no-calendar", action="store_true")
    p.add_argument("--holidays-path", default=None)
    p.add_argument(
        "--n-threads", type=int, default=20, help="Thread count for data ops."
    )

    # --- Model architecture ---
    p.add_argument("--forecast-horizon", type=int, default=24)
    p.add_argument(
        "--input-size",
        default="24",
        help=(
            "Number of lag steps for Ridge, LASSO, ElasticNet, and LightGBM.  "
            "Either a single integer applied to all TSOs (e.g. ``36``) or "
            "TSO-keyed pairs (e.g. ``TenneT_DE:36 Amprion:24``).  "
            "A bare integer may be mixed with keyed pairs as a fallback default."
        ),
    )
    p.add_argument(
        "--linear-scaler-type",
        default="standard",
        choices=["standard", "none"],
        help=(
            "Scaler for Ridge/LASSO/ElasticNet targets and covariates. "
            "Fitted on train+validation before each test window; predictions are inverse-transformed."
        ),
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "lasso", "elasticnet", "lightgbm", "arima", "croston"],
        help=(
            "Which benchmark models to ablate (space / comma separated). "
            "Allowed: ridge, lasso, elasticnet, lightgbm, arima, croston."
        ),
    )

    # --- Linear-model grids ---
    p.add_argument(
        "--ridge-alphas",
        nargs="+",
        default=[str(v) for v in RIDGE_ALPHAS],
        help="Ridge alpha grid; accepts space- or comma-separated floats.",
    )
    p.add_argument(
        "--lasso-alphas",
        nargs="+",
        default=[str(v) for v in LASSO_ALPHAS],
        help="LASSO alpha grid; accepts space- or comma-separated floats.",
    )
    p.add_argument(
        "--elasticnet-alphas",
        nargs="+",
        default=[str(v) for v in ELASTICNET_ALPHAS],
        help="ElasticNet alpha grid; accepts space- or comma-separated floats.",
    )
    p.add_argument(
        "--elasticnet-l1-ratios",
        nargs="+",
        default=[str(v) for v in ELASTICNET_L1_RATIOS],
        help="ElasticNet l1_ratio grid; accepts space- or comma-separated floats.",
    )
    p.add_argument(
        "--l1-max-iter",
        type=int,
        default=5_000,
        help="Maximum coordinate-descent iterations for LASSO/ElasticNet.",
    )
    p.add_argument(
        "--l1-tol",
        type=float,
        default=1e-3,
        help="Coordinate-descent tolerance for LASSO/ElasticNet.",
    )
    p.add_argument(
        "--l1-selection",
        default="random",
        choices=["cyclic", "random"],
        help="Coordinate update order for LASSO/ElasticNet.",
    )
    p.add_argument(
        "--l1-multi-models",
        action="store_true",
        help=(
            "Train separate LASSO/ElasticNet submodels per horizon/component. "
            "Default is a single multi-output model per window/knob for speed."
        ),
    )

    # --- Rolling windows ---
    p.add_argument("--n-train-months", type=int, default=37)
    p.add_argument("--n-valid-months", type=int, default=2)
    p.add_argument("--n-test-months", type=int, default=1)
    p.add_argument(
        "--k-windows",
        type=int,
        default=1,
        help="Number of consecutive windows to process.",
    )
    p.add_argument(
        "--start-window",
        type=int,
        default=0,
        help="0-based index of the first window to process.",
    )

    # --- LightGBM early stopping ---
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="LightGBM early-stopping patience (rounds without MAE improvement). "
        "Set to 0 to disable early stopping and always train for n_estimators.",
    )

    # --- LightGBM compute ---
    p.add_argument(
        "--lgb-device",
        default="cpu",
        choices=["cpu", "gpu"],
        help="LightGBM compute device.",
    )
    p.add_argument("--lgb-device-index", type=int, default=0)
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="LightGBM n_jobs (parallelism for CPU training).",
    )

    # --- ARIMA orders ---
    p.add_argument(
        "--arima-max-p", type=int, default=3, help="AutoARIMA max AR order (p)."
    )
    p.add_argument(
        "--arima-max-q", type=int, default=2, help="AutoARIMA max MA order (q)."
    )
    p.add_argument(
        "--arima-max-P",
        type=int,
        default=3,
        help="AutoARIMA max seasonal AR order (P).",
    )
    p.add_argument(
        "--arima-max-Q",
        type=int,
        default=2,
        help="AutoARIMA max seasonal MA order (Q).",
    )
    p.add_argument(
        "--arima-seasonal-period",
        type=int,
        default=24,
        help="Seasonal period for AutoARIMA (24 for hourly data).",
    )
    p.add_argument("--arima-approximation", action="store_true", help="Whether to use approximation in AutoARIMA.  Approximation may be needed to keep runtimes reasonable with large input_size and/or seasonal_period, but it changes the model selection procedure and may impact accuracy.")

    # --- Croston paramaters ---
    p.add_argument("--refit-croston", action="store_true", help="Whether to refit Croston TSB at each prediction step. This makes it incompatbile with the historical forecasts full-tail method, but may improve accuracy.")

    # --- Random seed ---
    p.add_argument("--random-seed", type=int, default=778)

    # --- Output ---
    p.add_argument("--output-dir", default="outputs")

    args = p.parse_args()
    args.models = _normalize_models(args.models)
    args.ridge_alphas = _parse_float_grid(args.ridge_alphas)
    args.lasso_alphas = _parse_float_grid(args.lasso_alphas)
    args.elasticnet_alphas = _parse_float_grid(args.elasticnet_alphas)
    args.elasticnet_l1_ratios = _parse_float_grid(args.elasticnet_l1_ratios)

    if args.k_windows <= 0:
        raise SystemExit("--k-windows must be > 0")

    return args


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    set_n_threads(args.n_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_stem = Path(args.dataset_path).stem
    _log_file = output_dir / f"benchmark_ablation_{dataset_stem}_w{args.start_window}.log"
    _fh = logging.FileHandler(_log_file, mode="a")
    _fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(_fh)
    logger.info("Logging to file: %s", _log_file)

    # --- Load & prepare data -------------------------------------------------
    logger.info("Loading dataset from %s", args.dataset_path)
    raw_df, metadata = load_dataset(args.dataset_path)
    tso = metadata.get("operator") or "unknown_tso"

    nixtla_df = to_nixtla_format(raw_df, direction=args.direction)

    input_size_map = _parse_input_sizes(args.input_size)
    input_size = _resolve_input_size(input_size_map, tso)
    logger.info("Using input_size=%d for TSO=%s", input_size, tso)

    df, future_cov_cols, hist_cov_cols = _prepare_benchmark_data(
        nixtla_df,
        tso=tso,
        add_calendar=not args.no_calendar,
        holidays_path=args.holidays_path,
    )
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    logger.info(
        "Future covariates (%d): %s", len(future_cov_cols), future_cov_cols
    )
    logger.info(
        "Historical covariates (%d): %s", len(hist_cov_cols), hist_cov_cols
    )

    # --- Rolling windows -----------------------------------------------------
    windows = _compute_rolling_windows(
        data_start=df["ds"].min(),
        data_end=df["ds"].max(),
        n_train_months=args.n_train_months,
        n_valid_months=args.n_valid_months,
        n_test_months=args.n_test_months,
    )
    if not windows:
        raise SystemExit(
            "No rolling windows available with the current dataset and month settings."
        )

    start_idx = args.start_window
    end_idx = min(start_idx + args.k_windows, len(windows))
    selected_windows = windows[start_idx:end_idx]
    logger.info(
        "Using windows [%d, %d) out of %d total", start_idx, end_idx, len(windows)
    )

    # --- Main ablation loop --------------------------------------------------
    all_preds: list[pd.DataFrame] = []
    # Per-knob tracking for LightGBM feature importances and best-knob selection
    lgbm_knob_preds: dict[int, list[pd.DataFrame]] = {mldl: [] for mldl in LGBM_MIN_DATA_IN_LEAF}
    lgbm_knob_feat_imps: dict[int, list[pd.DataFrame]] = {mldl: [] for mldl in LGBM_MIN_DATA_IN_LEAF}
    lgbm_model_details = []
    # ARIMA selected orders per window, per component
    arima_window_orders: dict[int, dict] = {}
    # Croston TSB per-(alpha_d, alpha_p) knob tracking
    croston_knob_preds: dict[tuple[float, float], list[pd.DataFrame]] = {}

    for wi, wb in enumerate(selected_windows, start=start_idx):
        logger.info(
            "Window %d: train=[%s, %s)  val=[%s, %s)  test=[%s, %s)",
            wi,
            wb.train_start,
            wb.valid_start,
            wb.valid_start,
            wb.test_start,
            wb.test_start,
            wb.test_end,
        )

        # Slice dataset: train + validation + test (test_end excluded)
        window_df = df[
            (df["ds"] >= wb.train_start) & (df["ds"] < wb.test_end)
        ].copy()

        # ── Ridge ablation ──────────────────────────────────────────────────
        if "ridge" in args.models:
            for alpha in args.ridge_alphas:
                alias = f"ridge_alpha_{_fmt_alpha(alpha)}"
                logger.info("  Ridge alpha=%g  window=%d", alpha, wi)
                try:
                    preds_df, col = _run_regularized_linear_window(
                        window_df=window_df,
                        future_cov_cols=future_cov_cols,
                        hist_cov_cols=hist_cov_cols,
                        wb=wb,
                        input_size=input_size,
                        forecast_horizon=args.forecast_horizon,
                        shift_hours=args.shift_hours,
                        model_cls=RidgeRegressionModel,
                        model_label="Ridge",
                        col_name=alias,
                        model_kwargs={"alpha": alpha},
                        scaler_type=args.linear_scaler_type,
                    )
                    all_preds.append(
                        _finalize_preds(
                            preds_df=preds_df,
                            pred_col=col,
                            actuals_df=nixtla_df,
                            model_name="ridge",
                            model_alias=alias,
                            input_size=input_size,
                            window_index=wi,
                            wb=wb,
                            tso=tso,
                        )
                    )
                except Exception as exc:
                    logger.error(
                        "Ridge alpha=%g window=%d FAILED: %s", alpha, wi, exc,
                        exc_info=True,
                    )

        # ── LASSO ablation ──────────────────────────────────────────────────
        if "lasso" in args.models:
            for alpha in args.lasso_alphas:
                alias = f"lasso_alpha_{_fmt_alpha(alpha)}"
                logger.info("  LASSO alpha=%g  window=%d", alpha, wi)
                try:
                    preds_df, col = _run_regularized_linear_window(
                        window_df=window_df,
                        future_cov_cols=future_cov_cols,
                        hist_cov_cols=hist_cov_cols,
                        wb=wb,
                        input_size=input_size,
                        forecast_horizon=args.forecast_horizon,
                        shift_hours=args.shift_hours,
                        model_cls=LassoRegressionModel,
                        model_label="LASSO",
                        col_name=alias,
                        model_kwargs={
                            "alpha": alpha,
                            "max_iter": args.l1_max_iter,
                            "tol": args.l1_tol,
                            "selection": args.l1_selection,
                            "multi_models": args.l1_multi_models,
                            "random_state": args.random_seed,
                        },
                        scaler_type=args.linear_scaler_type,
                    )
                    all_preds.append(
                        _finalize_preds(
                            preds_df=preds_df,
                            pred_col=col,
                            actuals_df=nixtla_df,
                            model_name="lasso",
                            model_alias=alias,
                            input_size=input_size,
                            window_index=wi,
                            wb=wb,
                            tso=tso,
                        )
                    )
                except Exception as exc:
                    logger.error(
                        "LASSO alpha=%g window=%d FAILED: %s", alpha, wi, exc,
                        exc_info=True,
                    )

        # ── ElasticNet ablation ─────────────────────────────────────────────
        if "elasticnet" in args.models:
            for alpha in args.elasticnet_alphas:
                for l1_ratio in args.elasticnet_l1_ratios:
                    alias = (
                        f"elasticnet_alpha_{_fmt_alpha(alpha)}"
                        f"_l1r{_fmt_alpha(l1_ratio)}"
                    )
                    logger.info(
                        "  ElasticNet alpha=%g  l1_ratio=%g  window=%d",
                        alpha,
                        l1_ratio,
                        wi,
                    )
                    try:
                        preds_df, col = _run_regularized_linear_window(
                            window_df=window_df,
                            future_cov_cols=future_cov_cols,
                            hist_cov_cols=hist_cov_cols,
                            wb=wb,
                            input_size=input_size,
                            forecast_horizon=args.forecast_horizon,
                            shift_hours=args.shift_hours,
                            model_cls=ElasticNetRegressionModel,
                            model_label="ElasticNet",
                            col_name=alias,
                            model_kwargs={
                                "alpha": alpha,
                                "l1_ratio": l1_ratio,
                                "max_iter": args.l1_max_iter,
                                "tol": args.l1_tol,
                                "selection": args.l1_selection,
                                "multi_models": args.l1_multi_models,
                                "random_state": args.random_seed,
                            },
                            scaler_type=args.linear_scaler_type,
                        )
                        all_preds.append(
                            _finalize_preds(
                                preds_df=preds_df,
                                pred_col=col,
                                actuals_df=nixtla_df,
                                model_name="elasticnet",
                                model_alias=alias,
                                input_size=input_size,
                                window_index=wi,
                                wb=wb,
                                tso=tso,
                            )
                        )
                    except Exception as exc:
                        logger.error(
                            "ElasticNet alpha=%g l1_ratio=%g window=%d FAILED: %s",
                            alpha,
                            l1_ratio,
                            wi,
                            exc,
                            exc_info=True,
                        )

        # ── LightGBM ablation ───────────────────────────────────────────────
        if "lightgbm" in args.models:
            for mldl in LGBM_MIN_DATA_IN_LEAF:
                alias = f"lightgbm_mldl_{mldl}"
                logger.info(
                    "  LightGBM min_data_in_leaf=%d  n_estimators=%d  window=%d",
                    mldl,
                    LGBM_N_ESTIMATORS,
                    wi,
                )
                try:
                    preds_df, col, feat_imp_df, model_details = _run_lgbm_window(
                        window_df=window_df,
                        future_cov_cols=future_cov_cols,
                        hist_cov_cols=hist_cov_cols,
                        wb=wb,
                        input_size=input_size,
                        forecast_horizon=args.forecast_horizon,
                        shift_hours=args.shift_hours,
                        min_data_in_leaf=mldl,
                        n_estimators=LGBM_N_ESTIMATORS,
                        early_stopping_rounds=args.early_stopping_rounds,
                        n_jobs=args.n_jobs,
                        random_seed=args.random_seed,
                        lgb_device=args.lgb_device,
                        lgb_device_index=args.lgb_device_index,
                    )
                    finalized = _finalize_preds(
                        preds_df=preds_df,
                        pred_col=col,
                        actuals_df=nixtla_df,
                        model_name="lightgbm",
                        model_alias=alias,
                        input_size=input_size,
                        window_index=wi,
                        wb=wb,
                        tso=tso,
                    )
                    all_preds.append(finalized)
                    lgbm_knob_preds[mldl].append(finalized)
                    if not feat_imp_df.empty:
                        _imp = feat_imp_df.copy()
                        _imp["window_index"] = wi
                        lgbm_knob_feat_imps[mldl].append(_imp)
                    lgbm_model_details.extend(model_details)
                except Exception as exc:
                    logger.error(
                        "LightGBM mldl=%d window=%d FAILED: %s", mldl, wi, exc,
                        exc_info=True,
                    )

        # ── AutoARIMA (single config per window) ────────────────────────────
        if "arima" in args.models:
            alias = (
                f"arima_p{args.arima_max_p}q{args.arima_max_q}"
                f"P{args.arima_max_P}Q{args.arima_max_Q}_m{args.arima_seasonal_period}"
            )
            logger.info("  AutoARIMA  window=%d", wi)
            try:
                preds_df, col, arima_orders_window = _run_arima_window(
                    window_df=window_df,
                    wb=wb,
                    input_size=input_size,
                    forecast_horizon=args.forecast_horizon,
                    max_p=args.arima_max_p,
                    max_q=args.arima_max_q,
                    max_P=args.arima_max_P,
                    max_Q=args.arima_max_Q,
                    seasonal_period=args.arima_seasonal_period,
                    pred_col_name=alias,
                    use_approximation=args.arima_approximation,
                )
                arima_window_orders[wi] = arima_orders_window
                all_preds.append(
                    _finalize_preds(
                        preds_df=preds_df,
                        pred_col=col,
                        actuals_df=nixtla_df,
                        model_name="arima",
                        model_alias=alias,
                        input_size=input_size,
                        window_index=wi,
                        wb=wb,
                        tso=tso,
                    )
                )
            except Exception as exc:
                logger.error(
                    "AutoARIMA window=%d FAILED: %s", wi, exc, exc_info=True
                )

        # ── Croston TSB ablation ────────────────────────────────────────────
        if "croston" in args.models:
            for alpha_p in CROSTON_ALPHA_P:
                for alpha_d in CROSTON_ALPHA_D:
                    alpha_p_str = _fmt_alpha(alpha_p)
                    alpha_d_str = _fmt_alpha(alpha_d)
                    alias = f"croston_tsb_ad{alpha_d_str}_ap{alpha_p_str}"
                    logger.info(
                        "  Croston TSB alpha_d=%g  alpha_p=%g  window=%d",
                        alpha_d,
                        alpha_p,
                        wi,
                    )
                    try:
                        preds_df, col = _run_croston_window(
                            window_df=window_df,
                            wb=wb,
                            forecast_horizon=args.forecast_horizon,
                            alpha_d=alpha_d,
                            alpha_p=alpha_p,
                            refit=args.refit_croston,
                        )
                        finalized_croston = _finalize_preds(
                            preds_df=preds_df,
                            pred_col=col,
                            actuals_df=nixtla_df,
                            model_name="croston",
                            model_alias=alias,
                            input_size=input_size,
                            window_index=wi,
                            wb=wb,
                            tso=tso,
                        )
                        all_preds.append(finalized_croston)
                        _ck = (alpha_d, alpha_p)
                        croston_knob_preds.setdefault(_ck, []).append(finalized_croston)
                    except Exception as exc:
                        logger.error(
                            "Croston TSB alpha_p=%g window=%d FAILED: %s",
                            alpha_p,
                            wi,
                            exc,
                            exc_info=True,
                        )

    # --- Save results ---------------------------------------------------------
    if not all_preds:
        raise SystemExit("No predictions were generated.")

    final_df = pd.concat(all_preds, ignore_index=True)
    final_df = final_df.sort_values(
        ["model_name", "model_alias", "window_index", "unique_id", "ds"]
    ).reset_index(drop=True)

    k_label = len(selected_windows)
    out_parquet = (
        output_dir / f"benchmark_ablation_{dataset_stem}_k{k_label}.parquet"
    )
    out_csv = output_dir / f"benchmark_ablation_{dataset_stem}_k{k_label}.csv"

    final_df.to_parquet(out_parquet, index=False)
    final_df.to_csv(out_csv, index=False)

    logger.info("Saved %d rows to %s", len(final_df), out_parquet)
    logger.info("Saved %d rows to %s", len(final_df), out_csv)

    # Brief per-model summary
    summary = (
        final_df.groupby(["model_name", "model_alias"])
        .agg(
            n_rows=("y_hat", "count"),
            mae=("y_hat", lambda s: (s - final_df.loc[s.index, "y"]).abs().mean()),
        )
        .reset_index()
    )
    logger.info("Test MAE summary:\n%s", summary.to_string(index=False))

    # --- Best LightGBM knob + feature importances ----------------------------
    best_lgbm_knob: Optional[int] = None
    best_lgbm_mae = float("inf")
    lgbm_knob_mae: dict[int, float] = {}
    for mldl, preds_list in lgbm_knob_preds.items():
        if preds_list:
            combined = pd.concat(preds_list, ignore_index=True)
            valid_mask = combined["y"].notna() & combined["y_hat"].notna()
            mae = float(
                (combined.loc[valid_mask, "y_hat"] - combined.loc[valid_mask, "y"])
                .abs()
                .mean()
            )
            lgbm_knob_mae[mldl] = mae
            if mae < best_lgbm_mae:
                best_lgbm_mae = mae
                best_lgbm_knob = mldl

    if best_lgbm_knob is not None:
        logger.info(
            "Best LightGBM knob: min_data_in_leaf=%d (test MAE=%.4f). "
            "All knob MAEs: %s",
            best_lgbm_knob,
            best_lgbm_mae,
            {k: f"{v:.4f}" for k, v in sorted(lgbm_knob_mae.items())},
        )
        imps_list = lgbm_knob_feat_imps.get(best_lgbm_knob, [])
        if imps_list:
            all_imps = pd.concat(imps_list, ignore_index=True)
            feat_imp_path = (
                output_dir
                / f"lgbm_feature_importances_best_mldl{best_lgbm_knob}_{dataset_stem}.csv"
            )
            all_imps.to_csv(feat_imp_path, index=False)
            logger.info(
                "Saved feature importances for best knob "
                "(mldl=%d, %d windows, %d rows) to %s",
                best_lgbm_knob,
                len(imps_list),
                len(all_imps),
                feat_imp_path,
            )

    # --- Best Croston TSB knob -----------------------------------------------
    best_croston_key: Optional[tuple[float, float]] = None
    best_croston_mae = float("inf")
    croston_knob_mae: dict[tuple[float, float], float] = {}
    for _ck, preds_list in croston_knob_preds.items():
        if preds_list:
            combined = pd.concat(preds_list, ignore_index=True)
            valid_mask = combined["y"].notna() & combined["y_hat"].notna()
            mae = float(
                (combined.loc[valid_mask, "y_hat"] - combined.loc[valid_mask, "y"])
                .abs()
                .mean()
            )
            croston_knob_mae[_ck] = mae
            if mae < best_croston_mae:
                best_croston_mae = mae
                best_croston_key = _ck

    if best_croston_key is not None:
        logger.info(
            "Best Croston TSB knob: alpha_d=%g alpha_p=%g (test MAE=%.4f). "
            "All knob MAEs: %s",
            best_croston_key[0],
            best_croston_key[1],
            best_croston_mae,
            {
                f"ad{k[0]}_ap{k[1]}": f"{v:.4f}"
                for k, v in sorted(croston_knob_mae.items())
            },
        )

    # --- Best-parameters JSON ------------------------------------------------
    best_params: dict = {}
    if lgbm_knob_mae:
        best_params["lgbm"] = {
            "best_min_data_in_leaf": best_lgbm_knob,
            "configured_n_estimators": LGBM_N_ESTIMATORS,
            "knob_test_mae": {
                str(k): round(v, 6) for k, v in sorted(lgbm_knob_mae.items())
            },
            "knob_val_mae": {
                str(k): round(v, 6) for k, v in sorted(lgbm_knob_mae.items())
            },
            "model_details": lgbm_model_details,
        }
    if arima_window_orders:
        best_params["arima"] = {
            f"window_{wi}": orders
            for wi, orders in sorted(arima_window_orders.items())
        }
    if croston_knob_mae:
        best_params["croston_tsb"] = {
            "refit": args.refit_croston,
            "best_alpha_d": best_croston_key[0] if best_croston_key else None,
            "best_alpha_p": best_croston_key[1] if best_croston_key else None,
            "knob_test_mae": {
                f"ad{k[0]}_ap{k[1]}": round(v, 6)
                for k, v in sorted(croston_knob_mae.items())
            },
            "knob_val_mae": {
                f"ad{k[0]}_ap{k[1]}": round(v, 6)
                for k, v in sorted(croston_knob_mae.items())
            },
        }
    if best_params:
        params_path = output_dir / f"best_params_{dataset_stem}.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        logger.info("Saved best parameters to %s", params_path)


if __name__ == "__main__":
    main()
