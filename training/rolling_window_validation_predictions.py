#!/usr/bin/env python
"""
Train Nixtla models on the first K rolling windows for multiple input sizes
and save concatenated validation predictions to disk.

- Input sizes: configurable, defaults to [24, 36, 48]
- Models: configurable, defaults to [nhits, nbeatsx, tft, tft_quantile, lstm]
- No Weights & Biases integration
"""

from __future__ import annotations

import argparse
import itertools
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from training.train_pipeline import set_n_threads

try:
    from neuralforecast.core import NeuralForecast, NHITS, TFT, NBEATSx, LSTM
except ImportError:  # pragma: no cover
    from neuralforecast.core import NeuralForecast, NHITS, TFT, NBEATSx
    from neuralforecast.models import LSTM

from neuralforecast.losses.pytorch import MAE, MQLoss
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

from training.data_prep import (
    CALENDAR_COLS,
    STATIC_EXOG_COLS,
    add_calendar_features,
    build_static_df,
    load_dataset,
    prepare_shifted_dataset,
    to_nixtla_format,
)
from training.prediction_pipeline_rolling_window import load_model
from training.runner import (
    CHECKPOINT_BEST_NAME,
    _compute_rolling_windows,
    find_best_valid_checkpoint,
)
from training.losses import MQMedianLoss

logger = logging.getLogger(__name__)


@dataclass
class WindowBoundary:
    train_start: pd.Timestamp
    valid_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _constant_weights(h: int) -> np.ndarray:
    start = max(0, h - 24)
    w = np.zeros(h)
    w[start:] = 1.0
    return w


def _normalize_models(raw_models: list[str]) -> list[str]:
    allowed = {"nhits", "nbeatsx", "tft", "tft_quantile", "lstm"}
    models: list[str] = []
    for item in raw_models:
        models.extend([p.strip().lower() for p in item.split(",") if p.strip()])

    invalid = [m for m in models if m not in allowed]
    if invalid:
        raise SystemExit(
            f"Invalid model(s): {invalid}. Allowed: {sorted(allowed)}"
        )

    seen: set[str] = set()
    deduped: list[str] = []
    for m in models:
        if m not in seen:
            deduped.append(m)
            seen.add(m)
    return deduped


def _parse_input_sizes(raw: list[str]) -> list[int]:
    values: list[int] = []
    for item in raw:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        for p in parts:
            v = int(p)
            if v <= 0:
                raise SystemExit("All input sizes must be > 0")
            values.append(v)

    seen: set[int] = set()
    deduped: list[int] = []
    for v in values:
        if v not in seen:
            deduped.append(v)
            seen.add(v)
    return deduped


def _load_model_config(config_path: str | None) -> dict[str, list[dict[str, Any]]]:
    """Load YAML config and expand parameter combinations per model.
    
    Returns dict mapping model_name -> list of param dicts.
    Each param dict represents one combination to train.
    """
    if not config_path:
        return {}
    
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    
    result: dict[str, list[dict[str, Any]]] = {}
    
    for model_name, params in config.items():
        if not params:
            result[model_name] = [{}]
            continue
        
        # Generate all combinations
        param_names = list(params.keys())
        param_values = [params[k] for k in param_names]
        
        # If all values are lists, create combinations
        # Otherwise, treat single values as one-element lists
        normalized_values = []
        for vals in param_values:
            if isinstance(vals, list):
                normalized_values.append(vals)
            else:
                normalized_values.append([vals])
        
        # Cartesian product of all parameter values
        combinations = list(itertools.product(*normalized_values))
        
        result[model_name] = [
            dict(zip(param_names, combo)) for combo in combinations
        ]
    
    return result


def _make_param_suffix(params: dict[str, Any]) -> str:
    """Create a string suffix from parameter dict for naming."""
    if not params:
        return ""
    
    parts = []
    for key, value in sorted(params.items()):
        # Convert value to string representation
        if isinstance(value, list):
            val_str = "_".join(str(v) for v in value)
        else:
            val_str = str(value)
        # Sanitize for filename
        val_str = val_str.replace(".", "p").replace(",", "").replace(" ", "")
        parts.append(f"{key}_{val_str}")
    
    return "_".join(parts)


def _model_alias(model_name: str, input_size: int, param_suffix: str = "") -> str:
    base = f"{model_name}_i{input_size}"
    if param_suffix:
        base += f"_{param_suffix}"
    return base


def _make_model(
    model_name: str,
    alias: str,
    input_size: int,
    forecast_horizon: int,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    max_steps: int,
    val_check_steps: int,
    early_stop_patience: int,
    batch_size: int,
    windows_batch_size: int,
    learning_rate: float,
    random_seed: int,
    scaler_type: str | None,
    quantile_levels: list[int],
    param_overrides: dict[str, Any] | None = None,
    checkpoint_cb: ModelCheckpoint | None = None,
):
    h = forecast_horizon
    weights = _constant_weights(h)

    callbacks: list[Any] = [
        EarlyStopping(
            monitor="valid_loss",
            mode="min",
            patience=early_stop_patience,
        )
    ]
    if checkpoint_cb is not None:
        callbacks.append(checkpoint_cb)

    common_kwargs: dict[str, object] = dict(
        h=h,
        futr_exog_list=future_cov_cols,
        hist_exog_list=hist_cov_cols,
        stat_exog_list=STATIC_EXOG_COLS,
        batch_size=batch_size,
        windows_batch_size=windows_batch_size,
        loss=MAE(horizon_weight=weights),
        valid_loss=MAE(horizon_weight=weights),
        max_steps=max_steps,
        val_check_steps=val_check_steps,
        early_stop_patience_steps=-1,
        random_seed=random_seed,
        scaler_type=scaler_type,
        alias=alias,
        logger=False,
        enable_checkpointing=checkpoint_cb is not None,
        callbacks=callbacks,
    )

    if model_name == "nhits":
        model_params = {
            "input_size": input_size,
            "n_blocks": [20, 15, 10],
            "learning_rate": learning_rate,
        }
        if param_overrides:
            model_params.update(param_overrides)
        return NHITS(
            **model_params,
            **common_kwargs,  # type: ignore[arg-type]
        )

    if model_name == "nbeatsx":
        model_params = {
            "input_size": input_size,
            "n_blocks": [15, 5, 2],
            "stack_types": ["identity", "seasonality", "trend"],
            "learning_rate": 5e-5,
        }
        if param_overrides:
            model_params.update(param_overrides)
        return NBEATSx(
            **model_params,
            **common_kwargs,  # type: ignore[arg-type]
        )

    if model_name == "tft":
        common_kwargs.update(dict(windows_batch_size=min(windows_batch_size, 32)))
        model_params = {
            "input_size": input_size,
            "hidden_size": 64,
            "dropout": 0.15,
            "n_head": 4,
            "attn_dropout": 0.1,
        }
        if param_overrides:
            model_params.update(param_overrides)
        return TFT(
            **model_params,
            **common_kwargs,  # type: ignore[arg-type]
        )

    if model_name == "tft_quantile":
        q_levels = [float(q) for q in quantile_levels]
        common_kwargs["loss"] = MQLoss(level=q_levels, horizon_weight=weights)
        common_kwargs["valid_loss"] = MQMedianLoss(level=q_levels, horizon_weight=weights)
        common_kwargs["windows_batch_size"] = min(windows_batch_size, 32)
        model_params = {
            "input_size": input_size,
            "hidden_size": 64,
            "dropout": 0.15,
            "n_head": 4,
            "attn_dropout": 0.1,
        }
        if param_overrides:
            model_params.update(param_overrides)
        return TFT(
            **model_params,
            **common_kwargs,  # type: ignore[arg-type]
        )

    if model_name == "lstm":
        common_kwargs.update(dict(windows_batch_size=min(windows_batch_size, 32)))
        model_params = {
            "input_size": input_size,
            "encoder_n_layers": 2,
            "encoder_hidden_size": 128,
            "encoder_dropout": 0.1,
            "decoder_hidden_size": 128,
            "decoder_layers": 2,
            "learning_rate": learning_rate,
        }
        if param_overrides:
            model_params.update(param_overrides)
        return LSTM(
            **model_params,
            **common_kwargs,  # type: ignore[arg-type]
        )

    raise ValueError(f"Unknown model name: {model_name}")


def _predict_with_shift_correction(
    nf: NeuralForecast,
    df_shifted: pd.DataFrame,
    df_unshifted: pd.DataFrame,
    static_df: pd.DataFrame,
    pred_start: pd.Timestamp,
    pred_end: pd.Timestamp,
    future_cov_cols: list[str],
    shift_hours: int,
    forecast_horizon: int,
    tso: str,
    holidays_path: str | None = None,
) -> pd.DataFrame:
    n_series = df_shifted["unique_id"].nunique()

    data_futr_cols = [c for c in future_cov_cols if c not in CALENDAR_COLS]
    need_calendar = any(c in CALENDAR_COLS for c in future_cov_cols)

    missing = [c for c in data_futr_cols if c not in df_unshifted.columns]
    if missing:
        logger.warning(
            "Missing future covariate columns in unshifted data: %s", missing
        )
        data_futr_cols = [c for c in data_futr_cols if c in df_unshifted.columns]

    model_pred_date = pred_start - pd.Timedelta(hours=shift_hours)
    last_stride_start = pred_end - pd.Timedelta(
        hours=shift_hours + forecast_horizon - 1
    )

    preds: list[pd.DataFrame] = []

    while model_pred_date <= last_stride_start:
        hist_df = df_shifted[df_shifted["ds"] < model_pred_date]
        if hist_df.empty:
            model_pred_date += pd.Timedelta(hours=forecast_horizon)
            continue

        phys_start = model_pred_date + pd.Timedelta(hours=shift_hours)
        phys_end = phys_start + pd.Timedelta(hours=forecast_horizon)

        futr_df = df_unshifted.loc[
            (df_unshifted["ds"] >= phys_start) & (df_unshifted["ds"] < phys_end),
            ["unique_id", "ds"] + data_futr_cols,
        ].copy()

        if len(futr_df) < forecast_horizon * n_series:
            break

        if need_calendar:
            futr_df = add_calendar_features(
                futr_df,
                reference_time=futr_df["ds"],
                tso=tso,
                holidays_path=holidays_path,
            )

        if shift_hours > 0:
            futr_df["ds"] = futr_df["ds"] - pd.Timedelta(hours=shift_hours)

        stride_preds = nf.predict(
            df=hist_df,
            futr_df=futr_df,
            static_df=static_df,
            verbose=False,
        )
        stride_preds_df = stride_preds.iloc[: forecast_horizon * n_series].copy()  # type: ignore[attr-defined]

        if shift_hours > 0:
            stride_preds_df["ds"] = stride_preds_df["ds"] + pd.Timedelta(hours=shift_hours)

        stride_preds_df["horizon"] = (
            (stride_preds_df["ds"] - phys_start).dt.total_seconds() / 3600 + 1
        ).astype(int)

        preds.append(stride_preds_df)
        model_pred_date += pd.Timedelta(hours=forecast_horizon)

    if not preds:
        return pd.DataFrame()

    result = pd.concat(preds, ignore_index=True)
    result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    result = result[
        (result["ds"] >= pred_start) & (result["ds"] <= pred_end)
    ].reset_index(drop=True)
    return result


def _make_checkpoint_callback(save_dir: Path) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=str(save_dir),
        monitor="valid_loss",
        filename="{epoch:04d}-{valid_loss:.4f}",
        mode="min",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
    )


def _prepare_predictions_df(preds: pd.DataFrame, actuals_df: pd.DataFrame) -> pd.DataFrame:
    if "y" not in preds.columns:
        merged = preds.merge(
            actuals_df[["ds", "unique_id", "y"]],
            on=["ds", "unique_id"],
            how="left",
        )
    else:
        merged = preds.copy()

    cols_to_keep = [c for c in merged.columns if "-lo-" not in c and "-hi-" not in c]
    merged = merged[cols_to_keep]
    merged.columns = merged.columns.str.replace("-median", "", regex=False)

    pred_cols = merged.columns.difference(["ds", "unique_id", "y", "horizon"])
    for col in pred_cols:
        merged[col] = np.clip(merged[col], 0, np.inf)

    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train each model on the first K rolling windows for input sizes "
            "[24, 36, 48] (default), generate validation predictions, "
            "and save concatenated outputs."
        )
    )

    p.add_argument("--dataset-path", required=True, help="Path to parquet dataset.")
    p.add_argument("--direction", default="both", choices=["up", "down", "both"])
    p.add_argument("--shift-hours", type=int, default=9)
    p.add_argument("--no-calendar", action="store_true")
    p.add_argument("--holidays-path", default=None)

    p.add_argument("--n-threads", type=int, default=20, help="Number of threads for data loading and processing.")

    p.add_argument("--forecast-horizon", type=int, default=24)
    p.add_argument(
        "--input-sizes",
        nargs="+",
        default=["24", "36", "48"],
        help="Input sizes. Supports space or comma separated values.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["nhits", "nbeatsx", "tft", "tft_quantile", "lstm"],
        help="Models to train (space and/or comma separated).",
    )

    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--val-check-steps", type=int, default=50)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--windows-batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--random-seed", type=int, default=778)
    p.add_argument("--scaler-type", default=None)
    p.add_argument("--local-scaler-type", default="standard")
    p.add_argument(
        "--quantile-levels",
        nargs="+",
        type=int,
        default=[54, 60, 64, 70, 74, 80, 84, 90, 94, 98],
    )

    p.add_argument("--n-train-months", type=int, default=37)
    p.add_argument("--n-valid-months", type=int, default=2)
    p.add_argument("--n-test-months", type=int, default=1)
    p.add_argument("--k-windows", type=int, default=1, help="Number of first windows to run.")
    p.add_argument("--start-window", type=int, default=0)

    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--persist-models", action="store_true", help="Save fitted NeuralForecast models.")
    p.add_argument(
        "-eval-checkpoint-type",
        "--eval-checkpoint-type",
        type=str.lower,
        choices=["last", "best"],
        default="last",
        help="Checkpoint type to use for validation prediction: last (default) or best valid checkpoint.",
    )
    p.add_argument(
        "--config-path",
        default=None,
        help="Path to YAML file with model parameter variations.",
    )

    args = p.parse_args()
    args.models = _normalize_models(args.models)
    args.input_sizes = _parse_input_sizes(args.input_sizes)

    if args.k_windows <= 0:
        raise SystemExit("--k-windows must be > 0")

    return args


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

    logger.info("Loading dataset from %s", args.dataset_path)
    raw_df, metadata = load_dataset(args.dataset_path)
    tso = metadata.get("operator") or "unknown_tso"

    nixtla_df = to_nixtla_format(raw_df, direction=args.direction)
    shifted_df, future_cov, hist_cov = prepare_shifted_dataset(
        nixtla_df,
        shift_hours=args.shift_hours,
        tso=tso,
        add_calendar=not args.no_calendar,
        holidays_path=args.holidays_path,
    )

    windows = _compute_rolling_windows(
        data_start=shifted_df["ds"].min(),
        data_end=shifted_df["ds"].max(),
        n_train_months=args.n_train_months,
        n_valid_months=args.n_valid_months,
        n_test_months=args.n_test_months,
    )

    if not windows:
        raise SystemExit("No rolling windows available with current dataset and month settings.")

    start = args.start_window
    end = min(start + args.k_windows, len(windows))
    selected_windows = windows[start:end]
    logger.info("Using windows [%d, %d) out of %d", start, end, len(windows))

    static_df = build_static_df()
    all_preds: list[pd.DataFrame] = []
    
    # Load parameter configurations
    model_configs = _load_model_config(args.config_path)
    logger.info("Loaded parameter configurations for models: %s", list(model_configs.keys()))

    for input_size in args.input_sizes:
        for wi, wb in enumerate(selected_windows, start=start):
            logger.info(
                "Window %d (input_size=%d): train=[%s,%s) val=[%s,%s), test=[%s,%s)",
                wi,
                input_size,
                wb.train_start,
                wb.valid_start,
                wb.valid_start,
                wb.test_start,
                wb.test_start,
                wb.test_end,
            )

            train_window_df = shifted_df[
                (shifted_df["ds"] >= wb.train_start) & (shifted_df["ds"] < wb.valid_start)
            ].copy()
            val_hours = int((wb.test_start - wb.valid_start).total_seconds() / 3600)

            # Generate model specs with parameter combinations
            model_specs = []
            for model_name in args.models:
                param_combos = model_configs.get(model_name, [{}])
                for params in param_combos:
                    param_suffix = _make_param_suffix(params)
                    model_specs.append({
                        "model_name": model_name,
                        "alias": _model_alias(model_name, input_size, param_suffix),
                        "param_overrides": params,
                        "param_suffix": param_suffix,
                    })

            # Best-checkpoint evaluation requires model-isolated checkpoint directories.
            # Bundled multi-model training can overwrite/shared checkpoint files and is
            # incompatible with the single-model best-checkpoint loader.
            if args.eval_checkpoint_type == "best":
                val_pred_start = wb.test_start
                val_pred_end = wb.test_end - pd.Timedelta(hours=1)
                base_cols = ["ds", "unique_id", "y", "horizon"]

                for spec in model_specs:
                    model_root_dir = (
                        output_dir
                        / f"input_size_{input_size}"
                        / f"window_{wi}"
                        / "nf_bundle"
                        / spec["alias"]
                    )
                    checkpoint_dir = model_root_dir / "checkpoints"
                    nf_model_dir = model_root_dir / "nf_model"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_cb = _make_checkpoint_callback(checkpoint_dir)

                    model = _make_model(
                        model_name=spec["model_name"],
                        alias=spec["alias"],
                        input_size=input_size,
                        forecast_horizon=args.forecast_horizon,
                        future_cov_cols=future_cov,
                        hist_cov_cols=hist_cov,
                        max_steps=args.max_steps,
                        val_check_steps=args.val_check_steps,
                        early_stop_patience=args.early_stop_patience,
                        batch_size=args.batch_size,
                        windows_batch_size=args.windows_batch_size,
                        learning_rate=args.learning_rate,
                        random_seed=args.random_seed,
                        scaler_type=args.scaler_type,
                        quantile_levels=args.quantile_levels,
                        param_overrides=spec.get("param_overrides"),
                        checkpoint_cb=checkpoint_cb,
                    )

                    nf_single = NeuralForecast(
                        models=[model],
                        freq="h",
                        local_scaler_type=args.local_scaler_type,
                    )
                    nf_single.fit(df=train_window_df, static_df=static_df, val_size=val_hours)
                    nf_model_dir.mkdir(parents=True, exist_ok=True)
                    nf_single.save(str(nf_model_dir), overwrite=True)

                    best_valid_checkpoint = find_best_valid_checkpoint(checkpoint_dir)
                    if best_valid_checkpoint is None:
                        raise SystemExit(
                            f"Requested eval-checkpoint-type=best but no valid checkpoint was found in {checkpoint_dir}."
                        )
                    best_valid_checkpoint.rename(nf_model_dir / CHECKPOINT_BEST_NAME)

                    nf_eval = load_model(nf_model_dir, checkpoint_best=True)
                    preds = _predict_with_shift_correction(
                        nf=nf_eval,
                        df_shifted=shifted_df,
                        df_unshifted=nixtla_df,
                        static_df=static_df,
                        pred_start=val_pred_start,
                        pred_end=val_pred_end,
                        future_cov_cols=future_cov,
                        shift_hours=args.shift_hours,
                        forecast_horizon=args.forecast_horizon,
                        tso=tso,
                        holidays_path=args.holidays_path,
                    )

                    if preds.empty:
                        logger.warning(
                            "No validation predictions produced for model=%s input_size=%d window=%d",
                            spec["model_name"],
                            input_size,
                            wi,
                        )
                        if not args.persist_models:
                            shutil.rmtree(model_root_dir, ignore_errors=True)
                        continue

                    preds = _prepare_predictions_df(preds, nixtla_df)
                    pred_col = spec["alias"]
                    if pred_col not in preds.columns:
                        logger.warning(
                            "Prediction column %s not found for model=%s input_size=%d window=%d",
                            pred_col,
                            spec["model_name"],
                            input_size,
                            wi,
                        )
                        if not args.persist_models:
                            shutil.rmtree(model_root_dir, ignore_errors=True)
                        continue

                    model_preds = preds[base_cols + [pred_col]].copy()
                    model_preds = model_preds.rename(columns={pred_col: "y_hat"})
                    model_preds["model_name"] = spec["model_name"]
                    model_preds["model_alias"] = pred_col
                    model_preds["input_size"] = input_size
                    model_preds["window_index"] = wi
                    model_preds["val_start"] = wb.valid_start
                    model_preds["val_end"] = val_pred_end
                    model_preds["tso"] = tso
                    all_preds.append(model_preds)

                    if not args.persist_models:
                        shutil.rmtree(model_root_dir, ignore_errors=True)

                continue

            should_persist_bundle = args.persist_models
            bundle_dir = (
                output_dir
                / f"input_size_{input_size}"
                / f"window_{wi}"
                / "nf_bundle"
            )

            bundled_models = [
                _make_model(
                    model_name=spec["model_name"],
                    alias=spec["alias"],
                    input_size=input_size,
                    forecast_horizon=args.forecast_horizon,
                    future_cov_cols=future_cov,
                    hist_cov_cols=hist_cov,
                    max_steps=args.max_steps,
                    val_check_steps=args.val_check_steps,
                    early_stop_patience=args.early_stop_patience,
                    batch_size=args.batch_size,
                    windows_batch_size=args.windows_batch_size,
                    learning_rate=args.learning_rate,
                    random_seed=args.random_seed,
                    scaler_type=args.scaler_type,
                    quantile_levels=args.quantile_levels,
                    param_overrides=spec.get("param_overrides"),
                    checkpoint_cb=None,
                )
                for spec in model_specs
            ]

            logger.info(
                "Training bundled models for window %d, input_size=%d: %s",
                wi,
                input_size,
                ", ".join(spec["alias"] for spec in model_specs),
            )

            nf = NeuralForecast(
                models=bundled_models,
                freq="h",
                local_scaler_type=args.local_scaler_type,
            )
            nf.fit(df=train_window_df, static_df=static_df, val_size=val_hours)

            if should_persist_bundle:
                bundle_dir.mkdir(parents=True, exist_ok=True)
                nf.save(str(bundle_dir), overwrite=True)

            nf_eval = nf

            val_pred_start = wb.test_start
            val_pred_end = wb.test_end - pd.Timedelta(hours=1)
            preds = _predict_with_shift_correction(
                nf=nf_eval,
                df_shifted=shifted_df,
                df_unshifted=nixtla_df,
                static_df=static_df,
                pred_start=val_pred_start,
                pred_end=val_pred_end,
                future_cov_cols=future_cov,
                shift_hours=args.shift_hours,
                forecast_horizon=args.forecast_horizon,
                tso=tso,
                holidays_path=args.holidays_path,
            )

            if preds.empty:
                logger.warning(
                    "No validation predictions produced for input_size=%d window=%d",
                    input_size,
                    wi,
                )
                continue

            preds = _prepare_predictions_df(preds, nixtla_df)
            base_cols = ["ds", "unique_id", "y", "horizon"]

            for spec in model_specs:
                pred_col = spec["alias"]
                if pred_col not in preds.columns:
                    logger.warning(
                        "Prediction column %s not found for model=%s input_size=%d window=%d",
                        pred_col,
                        spec["model_name"],
                        input_size,
                        wi,
                    )
                    continue

                model_preds = preds[base_cols + [pred_col]].copy()
                model_preds = model_preds.rename(columns={pred_col: "y_hat"})
                model_preds["model_name"] = spec["model_name"]
                model_preds["model_alias"] = pred_col
                model_preds["input_size"] = input_size
                model_preds["window_index"] = wi
                model_preds["val_start"] = wb.valid_start
                model_preds["val_end"] = val_pred_end
                model_preds["tso"] = tso
                all_preds.append(model_preds)

    if not all_preds:
        raise SystemExit("No predictions were generated.")

    final_df = pd.concat(all_preds, ignore_index=True)
    final_df = final_df.sort_values(["model_name", "input_size", "window_index", "unique_id", "ds"])

    k_label = len(selected_windows)
    dataset_stem = Path(args.dataset_path).stem
    out_csv = output_dir / f"validation_predictions_{dataset_stem}_k{k_label}_checkpoint_{args.eval_checkpoint_type}.csv"

    final_df.to_csv(out_csv, index=False)

    logger.info("Saved %d rows to %s", len(final_df), out_csv)


if __name__ == "__main__":
    main()
