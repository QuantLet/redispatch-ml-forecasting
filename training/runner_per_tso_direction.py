"""
Per-(TSO, Direction) training runner – no W&B.

Trains one NeuralForecast model per (dataset, direction, neural-model) combination.
Each model sees a *single* time-series (one TSO × one direction), so no static
exogenous direction-encoding is needed.

Unlike ``runner.py`` this module:
- Does **not** require or use Weights & Biases.
- Replaces ``WandbLogger`` with PyTorch-Lightning's ``CSVLogger``.
- Saves predictions and metrics as CSV files under ``output_dir``.
- Sets ``stat_exog_list=[]``  (no direction one-hot – the model is direction-specific).

Public API
----------
- ``PerDirectionConfig``       – configuration dataclass (mirrors ``TrainConfig``)
- ``build_model``              – instantiate a Nixtla model
- ``train_single_window``      – train on a fixed train/val split
- ``train_rolling_windows``    – train over rolling monthly windows
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from neuralforecast.core import NeuralForecast, NHITS, TFT, NBEATSx
try:
    from neuralforecast.core import LSTM
except ImportError:
    from neuralforecast.models import LSTM  # type: ignore[no-redef]
from neuralforecast.losses.pytorch import MAE, MQLoss
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from training.losses import MQMedianLoss
from training.predict import predict_with_shift_correction, prepare_predictions_df
from training.runner import _compute_rolling_windows

CHECKPOINT_BEST_NAME = "best_valid_checkpoint.ckpt"

logger = logging.getLogger(__name__)


# ── Configuration dataclass ───────────────────────────────────────────────────

@dataclass
class PerDirectionConfig:
    """
    All knobs needed for a per-(TSO, direction) training run.

    ``direction`` must be either ``"up"`` or ``"down"`` – a single series per
    model; the combined ``"both"`` mode is handled by the CLI, which iterates
    over both values and creates two separate ``PerDirectionConfig`` objects.
    """

    # Dataset
    dataset_path: str = ""
    tso: str = "TenneT_DE"
    direction: str = "up"          # always "up" or "down" inside this runner

    # Shifting
    shift_hours: int = 9

    # Calendar
    add_calendar: bool = True
    holidays_path: Optional[str] = None

    # Dates
    train_start: str = "2020-01-01"
    valid_start: str = "2024-02-01"
    test_start: str = "2024-04-01"

    # Forecast
    forecast_horizon: int = 24
    input_size: int = 24

    # Training hyper-parameters
    max_steps: int = 5_000
    val_check_steps: int = 50
    early_stop_patience: int = 20
    batch_size: int = 16
    windows_batch_size: int = 64
    learning_rate: float = 1e-4
    random_seed: int = 778
    scaler_type: Optional[str] = None
    local_scaler_type: str = "standard"

    # Models to train
    models: list[str] = field(default_factory=lambda: ["nhits", "tft"])

    # Quantile levels (for tft_quantile)
    quantile_levels: list[int] = field(
        default_factory=lambda: [54, 60, 64, 70, 74, 80, 84, 90, 94, 98]
    )

    # Per-model parameter overrides (nested dict keyed by model name).
    # Populated from the YAML config's model-specific sections.
    # Example: {"nhits": {"n_blocks": [8, 6, 4]}, "lstm": {"encoder_hidden_size": 256}}
    model_params: dict = field(default_factory=dict)

    # Rolling window
    rolling_window: bool = False
    n_train_months: int = 37
    n_valid_months: int = 2
    n_test_months: int = 1

    # Checkpointing / persistence
    persist_checkpoints: bool = True

    # Output
    output_dir: str = "outputs_per_direction"

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def normalized_tso(self) -> str:
        return self.tso.replace(" ", "_").lower()

    @property
    def train_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_start)

    @property
    def valid_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.valid_start)

    @property
    def test_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_start)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _constant_weights(h: int) -> np.ndarray:
    """Return horizon weights: 1.0 for the last 24 steps, 0.0 before."""
    start = max(0, h - 24)
    w = np.zeros(h)
    w[start:] = 1.0
    return w


def _model_alias(model_name: str, config: PerDirectionConfig) -> str:
    if model_name == "tft_quantile":
        return f"tft_quantile_seed{config.random_seed}"
    return f"{model_name}_seed{config.random_seed}"


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


def _find_best_valid_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Return the checkpoint file with the lowest validation loss."""
    if not checkpoint_dir.exists():
        return None
    ckpts = list(checkpoint_dir.glob("*.ckpt"))
    if not ckpts:
        return None

    best, best_loss = None, float("inf")
    for ckpt in ckpts:
        parts = ckpt.stem.split("-")
        val_part = next((p for p in parts if p.startswith("valid_loss=")), None)
        if val_part is None:
            continue
        try:
            val_loss = float(val_part.split("=")[1])
        except (IndexError, ValueError):
            continue
        if val_loss < best_loss:
            best_loss = val_loss
            best = ckpt

    if best:
        logger.info("Best checkpoint: %s (valid_loss=%.4f)", best.name, best_loss)
    return best


def _save_predictions(
    preds: pd.DataFrame,
    out_dir: Path,
    filename: str = "predictions_test.csv",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    preds.to_csv(path, index=False)
    logger.info("Saved predictions → %s", path)


def _save_metrics(
    preds: pd.DataFrame,
    actuals: pd.DataFrame,
    out_dir: Path,
    model_col: str,
    filename: str = "metrics.json",
) -> dict:
    """Compute MAE / RMSE on the test predictions and persist to JSON."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    merged = preds.merge(
        actuals[["unique_id", "ds", "y"]].rename(columns={"y": "y_true"}),
        on=["unique_id", "ds"],
        how="inner",
    )
    merged = merged.dropna(subset=[model_col, "y_true"])

    if merged.empty:
        logger.warning("No overlap between predictions and actuals – skipping metrics.")
        return {}

    y_true = np.asarray(merged["y_true"], dtype=float)
    y_pred = np.asarray(merged[model_col], dtype=float)

    metrics = {
        "model": model_col,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n_samples": int(len(merged)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics → %s  (MAE=%.4f, RMSE=%.4f)", path, metrics["mae"], metrics["rmse"])
    return metrics


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(
    config: PerDirectionConfig,
    model_name: str,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    csv_logger: CSVLogger,
    checkpoint_cb: Optional[ModelCheckpoint],
):
    """
    Instantiate a single Nixtla model object.

    Key difference from ``runner.build_model``:
    - ``stat_exog_list=[]``  – no direction one-hot; the model is direction-specific.
    - ``logger`` is a ``CSVLogger`` instead of a ``WandbLogger``.
    """
    h = config.forecast_horizon
    weights = _constant_weights(h)
    alias = _model_alias(model_name, config)

    # Per-model parameter overrides from the YAML config.
    mp: dict = config.model_params.get(model_name, {})

    callbacks: list = []
    if checkpoint_cb is not None:
        callbacks.append(checkpoint_cb)
    callbacks.append(
        EarlyStopping(
            monitor="valid_loss",
            mode="min",
            patience=config.early_stop_patience,
        )
    )

    common_kwargs = dict(
        h=h,
        futr_exog_list=future_cov_cols,
        hist_exog_list=hist_cov_cols,
        # No static exogenous – this model is already direction-specific.
        stat_exog_list=[],
        batch_size=config.batch_size,
        windows_batch_size=config.windows_batch_size,
        loss=MAE(horizon_weight=weights),
        valid_loss=MAE(horizon_weight=weights),
        max_steps=config.max_steps,
        val_check_steps=config.val_check_steps,
        early_stop_patience_steps=-1,
        random_seed=config.random_seed,
        scaler_type=config.scaler_type,
        alias=alias,
        logger=csv_logger,
        enable_checkpointing=checkpoint_cb is not None,
        callbacks=callbacks,
    )

    if model_name == "nhits":
        defaults = {
            "input_size": config.input_size,
            "n_blocks": [20, 15, 10],
            "learning_rate": config.learning_rate,
        }
        defaults.update(mp)
        return NHITS(**defaults, **common_kwargs)  # type: ignore[arg-type]

    if model_name == "nbeatsx":
        defaults = {
            "input_size": config.input_size,
            "n_blocks": [15, 5, 2],
            "stack_types": ["identity", "seasonality", "trend"],
            "learning_rate": 5e-5,
        }
        defaults.update(mp)
        return NBEATSx(**defaults, **common_kwargs)  # type: ignore[arg-type]

    if model_name == "tft":
        common_kwargs.update(dict(windows_batch_size=min(config.windows_batch_size, 32)))
        defaults = {
            "input_size": config.input_size,
            "hidden_size": 64,
            "dropout": 0.15,
            "n_head": 4,
            "attn_dropout": 0.1,
        }
        defaults.update(mp)
        return TFT(**defaults, **common_kwargs)  # type: ignore[arg-type]

    if model_name == "tft_quantile":
        common_kwargs.update(dict(
            loss=MQLoss(level=config.quantile_levels, horizon_weight=weights),  # type: ignore[arg-type]
            valid_loss=MQMedianLoss(level=config.quantile_levels, horizon_weight=weights),  # type: ignore[arg-type]
            windows_batch_size=min(config.windows_batch_size, 32),
        ))
        defaults = {
            "input_size": config.input_size,
            "hidden_size": 64,
            "dropout": 0.15,
            "n_head": 4,
            "attn_dropout": 0.1,
        }
        defaults.update(mp)
        return TFT(**defaults, **common_kwargs)  # type: ignore[arg-type]

    if model_name == "lstm":
        common_kwargs.update(dict(windows_batch_size=min(config.windows_batch_size, 32)))
        defaults = {
            "input_size": config.input_size,
            "encoder_n_layers": 2,
            "encoder_hidden_size": 128,
            "encoder_dropout": 0.1,
            "decoder_hidden_size": 128,
            "decoder_layers": 2,
            "learning_rate": config.learning_rate,
        }
        defaults.update(mp)
        return LSTM(**defaults, **common_kwargs)  # type: ignore[arg-type]

    raise ValueError(f"Unknown model name: {model_name!r}")


# ── Single-window training ────────────────────────────────────────────────────

def train_single_window(
    df: pd.DataFrame,
    config: PerDirectionConfig,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    metadata: dict | None = None,
    df_unshifted: pd.DataFrame | None = None,
) -> list[NeuralForecast]:
    """
    Train all configured models on a single (TSO, direction) train/val split.

    Split boundaries
    ----------------
    - Training rows: ``[train_start, test_start)`` (shifted time)
    - Validation tail: ``val_size = (test_start - valid_start) * 24`` hours

    Outputs written to ``output_dir/<tso>/<direction>/<model_alias>/``
    -----------------------------------------------------------------------
    - ``nf_model/``        – saved NeuralForecast object (via ``nf.save``)
    - ``metrics.json``     – MAE / RMSE on the test period
    - ``predictions_test.csv`` – raw test-set predictions

    Parameters
    ----------
    df : pd.DataFrame
        Shifted Nixtla-format DataFrame (output of ``prepare_shifted_dataset``).
    config : PerDirectionConfig
    future_cov_cols, hist_cov_cols : list[str]
        Covariate column names (output of ``prepare_shifted_dataset``).
    metadata : dict | None
        Dataset metadata dict (from the companion JSON file).
    df_unshifted : pd.DataFrame | None
        Unshifted Nixtla DataFrame used for future covariates during inference.

    Returns
    -------
    list[NeuralForecast]
        One fitted ``NeuralForecast`` wrapper per model in ``config.models``.
    """
    # ── Date boundaries in model (shifted) time ───────────────────────────────
    if config.shift_hours > 0:
        train_start_ts = config.train_start_ts - pd.Timedelta(hours=config.shift_hours)
        valid_start_ts = config.valid_start_ts - pd.Timedelta(hours=config.shift_hours)
        test_start_ts  = config.test_start_ts  - pd.Timedelta(hours=config.shift_hours)
    else:
        train_start_ts = config.train_start_ts
        valid_start_ts = config.valid_start_ts
        test_start_ts  = config.test_start_ts

    train_df = df[(df["ds"] >= train_start_ts) & (df["ds"] < test_start_ts)].copy()
    val_hours = int((test_start_ts - valid_start_ts).total_seconds() / 3600)

    logger.info(
        "[%s / %s] Single-window training: %d rows, val_size=%d h",
        config.tso, config.direction, len(train_df), val_hours,
    )

    dataset_contents = ", ".join(metadata["feature_sets"]) if metadata else ""
    date_time = (
        datetime.now(tz=pytz.UTC)
        .astimezone(pytz.timezone("Europe/Bucharest"))
        .strftime("%Y-%m-%d_%H-%M-%S")
    )

    nf_list: list[NeuralForecast] = []

    for model_name in config.models:
        alias = _model_alias(model_name, config)
        logger.info("[%s / %s] Training model: %s", config.tso, config.direction, alias)

        run_dir = (
            Path(config.output_dir)
            / (dataset_contents.replace(", ", "_") if dataset_contents else "default")
            / config.normalized_tso
            / config.direction
            / date_time
            / alias
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        # ── CSV Logger ────────────────────────────────────────────────────────
        csv_logger = CSVLogger(
            save_dir=str(run_dir),
            name="training_logs",
            version=0,
        )

        # ── Checkpoint callback ───────────────────────────────────────────────
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_cb = _make_checkpoint_callback(checkpoint_dir)

        # ── Build + fit model ─────────────────────────────────────────────────
        model = build_model(
            config, model_name, future_cov_cols, hist_cov_cols,
            csv_logger, checkpoint_cb,
        )

        nf = NeuralForecast(
            models=[model],
            freq="h",
            local_scaler_type=config.local_scaler_type,
        )

        # No static_df – the model is direction-specific, no direction encoding needed.
        nf.fit(df=train_df, static_df=None, val_size=val_hours)

        # ── Persist model ─────────────────────────────────────────────────────
        if config.persist_checkpoints:
            nf_dir = run_dir / "nf_model"
            nf.save(str(nf_dir), overwrite=True)
            logger.info("Saved NeuralForecast model → %s", nf_dir)

            best_ckpt = _find_best_valid_checkpoint(checkpoint_dir)
            if best_ckpt:
                dest = nf_dir / CHECKPOINT_BEST_NAME
                shutil.copy2(str(best_ckpt), str(dest))
                logger.info("Copied best checkpoint → %s", dest)

        # ── Save run metadata ─────────────────────────────────────────────────
        run_meta = {
            "tso": config.tso,
            "direction": config.direction,
            "model_alias": alias,
            "dataset_path": config.dataset_path,
            "dataset_contents": dataset_contents,
            "date_time": date_time,
            "shift_hours": config.shift_hours,
            "forecast_horizon": config.forecast_horizon,
            "input_size": config.input_size,
            "train_start": config.train_start,
            "valid_start": config.valid_start,
            "test_start": config.test_start,
            "max_steps": config.max_steps,
            "random_seed": config.random_seed,
            "local_scaler_type": config.local_scaler_type,
            "n_future_covariates": len(future_cov_cols),
            "n_hist_covariates": len(hist_cov_cols),
            "future_covariates": future_cov_cols,
            "historical_covariates": hist_cov_cols,
            "dataset_metadata": metadata or {},
        }
        with open(run_dir / "run_meta.json", "w") as f:
            json.dump(run_meta, f, indent=2, default=str)

        # ── Predict on test set ───────────────────────────────────────────────
        _unshifted = df_unshifted if df_unshifted is not None else df
        if df_unshifted is not None or config.shift_hours == 0:
            test_pred_end = _unshifted["ds"].max()
            logger.info(
                "[%s / %s / %s] Generating test predictions: [%s → %s]",
                config.tso, config.direction, alias,
                config.test_start, test_pred_end,
            )
            test_preds = predict_with_shift_correction(
                nf=nf,
                df_shifted=df,
                df_unshifted=_unshifted,
                static_df=None,
                pred_start=config.test_start_ts,
                pred_end=test_pred_end,
                future_cov_cols=future_cov_cols,
                shift_hours=config.shift_hours,
                forecast_horizon=config.forecast_horizon,
                tso=config.tso,
                holidays_path=config.holidays_path,
            )

            if not test_preds.empty:
                test_preds = prepare_predictions_df(test_preds, _unshifted)
                eval_dir = run_dir / "evaluation"
                _save_predictions(test_preds, eval_dir, "predictions_test.csv")

                # Compute metrics for each model column (point forecast)
                model_cols = [c for c in test_preds.columns if c not in {"unique_id", "ds", "y", "horizon"}]
                for mc in model_cols:
                    _save_metrics(test_preds, _unshifted, eval_dir, mc, f"metrics_{mc}.json")
            else:
                logger.warning("[%s / %s / %s] No test predictions generated.", config.tso, config.direction, alias)
        else:
            logger.warning(
                "[%s / %s / %s] Skipping test predictions – df_unshifted not provided and shift_hours > 0.",
                config.tso, config.direction, alias,
            )

        nf_list.append(nf)

    return nf_list


# ── Rolling-window training ──────────────────────────────────────────────────

@dataclass
class WindowBoundary:
    train_start: pd.Timestamp
    valid_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def train_rolling_windows(
    df: pd.DataFrame,
    config: PerDirectionConfig,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    metadata: dict | None = None,
    df_unshifted: pd.DataFrame | None = None,
    start_from_window: int = 0,
) -> list[NeuralForecast]:
    """
    Train models over rolling monthly windows for a single (TSO, direction).

    Each window writes its artefacts to
    ``output_dir/<dataset_contents>/<tso>/<direction>/window_<N>/<alias>/``.

    Parameters
    ----------
    start_from_window : int
        Resume from the N-th window (0-indexed).  Skips earlier windows.

    Returns
    -------
    list[NeuralForecast]
        All fitted ``NeuralForecast`` wrappers, in chronological window order.
    """
    data_start = df["ds"].min()
    data_end   = df["ds"].max()

    windows = _compute_rolling_windows(
        data_start, data_end,
        config.n_train_months, config.n_valid_months, config.n_test_months,
    )
    logger.info(
        "[%s / %s] Rolling-window training: %d windows total",
        config.tso, config.direction, len(windows),
    )

    dataset_contents = ", ".join(metadata["feature_sets"]) if metadata else ""
    date_time = (
        datetime.now(tz=pytz.UTC)
        .astimezone(pytz.timezone("Europe/Bucharest"))
        .strftime("%Y-%m-%d_%H-%M-%S")
    )

    nf_list: list[NeuralForecast] = []

    for wi, wb in enumerate(windows):
        if wi < start_from_window:
            continue

        logger.info(
            "[%s / %s] Window %d/%d: train [%s, %s)  val [%s, %s)  test [%s, %s)",
            config.tso, config.direction,
            wi + 1, len(windows),
            wb.train_start, wb.valid_start,
            wb.valid_start, wb.test_start,
            wb.test_start, wb.test_end,
        )

        window_df = df[
            (df["ds"] >= wb.train_start) & (df["ds"] < wb.test_start)
        ].copy()
        val_hours = int((wb.test_start - wb.valid_start).total_seconds() / 3600)

        for model_name in config.models:
            alias = _model_alias(model_name, config)
            logger.info(
                "[%s / %s] Window %d – training %s", config.tso, config.direction, wi, alias,
            )

            run_dir = (
                Path(config.output_dir)
                / (dataset_contents.replace(", ", "_") if dataset_contents else "default")
                / config.normalized_tso
                / config.direction
                / f"window_{wi:03d}"
                / date_time
                / alias
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            csv_logger = CSVLogger(
                save_dir=str(run_dir),
                name="training_logs",
                version=0,
            )

            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_cb = _make_checkpoint_callback(checkpoint_dir)

            model = build_model(
                config, model_name, future_cov_cols, hist_cov_cols,
                csv_logger, checkpoint_cb,
            )

            nf = NeuralForecast(
                models=[model],
                freq="h",
                local_scaler_type=config.local_scaler_type,
            )

            nf.fit(df=window_df, static_df=None, val_size=val_hours)

            # ── Persist model ─────────────────────────────────────────────────
            if config.persist_checkpoints:
                nf_dir = run_dir / "nf_model"
                nf.save(str(nf_dir), overwrite=True)
                logger.info("Saved model → %s", nf_dir)

                best_ckpt = _find_best_valid_checkpoint(checkpoint_dir)
                if best_ckpt:
                    dest = nf_dir / CHECKPOINT_BEST_NAME
                    shutil.copy2(str(best_ckpt), str(dest))

            # ── Run metadata ──────────────────────────────────────────────────
            run_meta = {
                "tso": config.tso,
                "direction": config.direction,
                "model_alias": alias,
                "window_index": wi,
                "train_start": str(wb.train_start),
                "valid_start": str(wb.valid_start),
                "test_start": str(wb.test_start),
                "test_end": str(wb.test_end),
                "dataset_path": config.dataset_path,
                "dataset_contents": dataset_contents,
                "date_time": date_time,
                "shift_hours": config.shift_hours,
                "forecast_horizon": config.forecast_horizon,
                "input_size": config.input_size,
                "max_steps": config.max_steps,
                "random_seed": config.random_seed,
                "local_scaler_type": config.local_scaler_type,
                "n_future_covariates": len(future_cov_cols),
                "n_hist_covariates": len(hist_cov_cols),
                "future_covariates": future_cov_cols,
                "historical_covariates": hist_cov_cols,
                "dataset_metadata": metadata or {},
            }
            with open(run_dir / "run_meta.json", "w") as f:
                json.dump(run_meta, f, indent=2, default=str)

            # ── Predict on window test period ─────────────────────────────────
            _unshifted = df_unshifted if df_unshifted is not None else df
            if df_unshifted is not None or config.shift_hours == 0:
                test_pred_end = wb.test_end - pd.Timedelta(hours=1)
                logger.info(
                    "[%s / %s / %s] Window %d – generating test predictions [%s → %s]",
                    config.tso, config.direction, alias, wi,
                    wb.test_start, test_pred_end,
                )
                test_preds = predict_with_shift_correction(
                    nf=nf,
                    df_shifted=df,
                    df_unshifted=_unshifted,
                    static_df=None,
                    pred_start=wb.test_start,
                    pred_end=test_pred_end,
                    future_cov_cols=future_cov_cols,
                    shift_hours=config.shift_hours,
                    forecast_horizon=config.forecast_horizon,
                    tso=config.tso,
                    holidays_path=config.holidays_path,
                )
                if not test_preds.empty:
                    test_preds = prepare_predictions_df(test_preds, _unshifted)
                    eval_dir = run_dir / "evaluation"
                    _save_predictions(test_preds, eval_dir, f"predictions_window{wi:03d}.csv")

                    model_cols = [c for c in test_preds.columns if c not in {"unique_id", "ds", "y", "horizon"}]
                    for mc in model_cols:
                        _save_metrics(test_preds, _unshifted, eval_dir, mc, f"metrics_{mc}_window{wi:03d}.json")
            else:
                logger.warning(
                    "[%s / %s / %s] Window %d – skipping predictions (no df_unshifted).",
                    config.tso, config.direction, alias, wi,
                )

            nf_list.append(nf)

    return nf_list
