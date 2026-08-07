#!/usr/bin/env python
"""
Training runner for hurdle (zero-inflated) models.

Trains ``HurdleNHITS``, ``HurdleNBEATSx``, or ``HurdleTFT`` with the
``HurdleCouplingPointLoss`` and a 3-stage learning schedule.

Usage examples
--------------
# Single window, NHITS
python -m training.hurdle_runner \\
    --dataset-path data/.../TenneT_DE_full_combo_dataset.parquet \\
    --tso TenneT_DE \\
    --models nhits nbeatsx \\
    --hurdle-schedule-steps 0.15 0.5 \\
    --hurdle-lambda-bce-final 0.15

# Rolling windows with a YAML config
python -m training.hurdle_runner \\
    --dataset-path data/.../TenneT_DE_full_combo_dataset.parquet \\
    --tso TenneT_DE \\
    --config training/hurdle_config.yaml \\
    --rolling-window
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from scipy.special import expit
from sklearn.metrics import fbeta_score

import numpy as np
import pandas as pd
import pytz
import wandb

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from training.data_prep import (
    STATIC_EXOG_COLS,
    build_static_df,
    load_dataset,
    prepare_shifted_dataset,
    to_nixtla_format,
)
from training.hurdle_models import HurdleNBEATSx, HurdleNHITS, HurdleTFT
from training.hurdle_nf import RedispatchNeuralForecast
from training.losses import HurdleCouplingPointLoss, HurdleEvalLoss
from training.predict import (
    log_predictions_to_wandb,
    predict_with_shift_correction,
    prepare_predictions_df,
)
from training.runner import (
    CHECKPOINT_BEST_NAME,
    TrainConfig,
    WindowBoundary,
    _archive_checkpoint_dir,
    _compute_rolling_windows,
    _log_checkpoint_artifact,
    _make_checkpoint_callback,
    _model_alias,
    _should_persist_checkpoints,
    find_best_valid_checkpoint,
    make_wandb_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────────────
_HURDLE_MODEL_CLASSES = {
    "nhits": HurdleNHITS,
    "nbeatsx": HurdleNBEATSx,
    "tft": HurdleTFT,
}

_ALLOWED_HURDLE_MODELS = set(_HURDLE_MODEL_CLASSES)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HurdleTrainConfig(TrainConfig):
    """
    ``TrainConfig`` extended with hurdle-specific hyperparameters.

    Hurdle schedule
    ---------------
    Training goes through three phases gated by ``hurdle_schedule_steps``:

    * **Phase 1** ``[0, step1)``:  only BCE  (λ_bce=1, λ_mag=0, λ_coup=0)
    * **Phase 2** ``[step1, step2)``:  BCE + mag  (λ_bce ramps down)
    * **Phase 3** ``[step2, ∞)``:  BCE + mag + coupling (λ_bce=``hurdle_lambda_bce_final``)

    ``hurdle_schedule_steps`` contains two floats in ``(0, 1)`` expressing the
    phase boundaries as **fractions of** ``max_steps``.  They are converted to
    absolute steps at model-build time.
    """

    # Models to train - "nhits" | "nbeatsx" | "tft"
    models: list[str] = field(default_factory=lambda: ["nhits"])

    # 3-stage schedule boundaries as fractions of max_steps
    hurdle_schedule_steps: tuple[float, float] = (0.15, 0.50)

    # BCE weight in the final stage (phase 3)
    hurdle_lambda_bce_final: float = 0.15

    # Minimum activation threshold: y < eps → treated as zero
    hurdle_eps: float = 0.1

    # Quantile of non-zero training values used as Huber δ per direction
    hurdle_huber_quantile: float = 0.85

    # Whether to z-score the magnitude head during training
    hurdle_scale_magnitude_head: bool = True

    # Threshold optimisation on validation set
    # Optimises θ to minimise: MAE + λ · max(0, Fβ2_target − Fβ2)
    hurdle_theta_fbeta2_target: float = 0.5
    hurdle_theta_lambda: float = 0.5
    hurdle_theta_n_grid: int = 200


# ─────────────────────────────────────────────────────────────────────────────
# Per-series statistics from training data
# ─────────────────────────────────────────────────────────────────────────────


def _compute_hurdle_stats(
    train_df: pd.DataFrame,
    config: HurdleTrainConfig,
) -> dict[str, dict[str, float]]:
    """
    Compute per-series hurdle statistics from the training DataFrame.

    Returns a dict with keys:
    * ``"base_rates"``  – P(y > eps) per ``unique_id``
    * ``"median_z"``    – median(y[y > eps]) per ``unique_id``
    * ``"pos_weight"``  – n_neg / n_pos per ``unique_id``
    * ``"delta_map"``   – ``hurdle_huber_quantile`` of y[y > eps] per ``unique_id``
    """
    eps = config.hurdle_eps
    base_rates: dict[str, float] = {}
    median_z: dict[str, float] = {}
    pos_weight: dict[str, float] = {}
    delta_map: dict[str, float] = {}

    for uid, grp in train_df.groupby("unique_id"):
        y = grp["y"].values.astype(np.float64)
        total = len(y)
        pos_mask = y >= eps
        n_pos = int(pos_mask.sum())
        n_neg = total - n_pos

        if n_pos == 0:
            logger.warning(
                "Series '%s' has no non-zero values in training data. "
                "Falling back to uniform hurdle stats.",
                uid,
            )
            base_rates[uid] = 0.01
            median_z[uid] = 0.0
            pos_weight[uid] = float(total)
            delta_map[uid] = 1.0
            continue

        base_rates[uid] = float(n_pos / total)
        pos_vals = y[pos_mask]
        median_z[uid] = float(np.median(pos_vals))
        pos_weight[uid] = float(n_neg / max(n_pos, 1))
        delta_map[uid] = float(np.quantile(pos_vals, config.hurdle_huber_quantile))

    logger.info(
        "Hurdle stats - base_rates: %s | median_z: %s | delta_map: %s",
        {k: f"{v:.3f}" for k, v in base_rates.items()},
        {k: f"{v:.1f}" for k, v in median_z.items()},
        {k: f"{v:.1f}" for k, v in delta_map.items()},
    )
    return {
        "base_rates": base_rates,
        "median_z": median_z,
        "pos_weight": pos_weight,
        "delta_map": delta_map,
    }


# ─────────────────────────────────────────────────────────────────────────────
# W&B config
# ─────────────────────────────────────────────────────────────────────────────


def make_hurdle_wandb_config(
    config: HurdleTrainConfig,
    hist_cov_cols: list[str],
    future_cov_cols: list[str],
    hurdle_stats: dict[str, dict] | None = None,
    window_index: int | None = None,
    metadata: dict | None = None,
    window_test_start: str | None = None,
    window_test_end: str | None = None,
) -> tuple[dict, str, str, str]:
    """
    Like ``make_wandb_config`` but:

    * sets ``model_arch = "hurdle"``
    * injects ``"hurdle"`` into the W&B group tag
    * adds hurdle hyperparameters to the base config dict
    """
    base_config, date_time_stamp, dataset_contents, group_tag = make_wandb_config(
        config=config,
        hist_cov_cols=hist_cov_cols,
        future_cov_cols=future_cov_cols,
        window_index=window_index,
        metadata=metadata,
        window_test_start=window_test_start,
        window_test_end=window_test_end,
    )

    # Patch model_arch and group tag
    base_config["model_arch"] = "hurdle"
    # group_tag = group_tag.replace(
    #     config.tso, f"{config.tso}_hurdle", 1
    # )

    # Hurdle-specific config fields
    base_config.update(
        {
            "hurdle_schedule_steps": list(config.hurdle_schedule_steps),
            "hurdle_lambda_bce_final": config.hurdle_lambda_bce_final,
            "hurdle_eps": config.hurdle_eps,
            "hurdle_huber_quantile": config.hurdle_huber_quantile,
            "hurdle_scale_magnitude_head": config.hurdle_scale_magnitude_head,
        }
    )

    if hurdle_stats is not None:
        base_config["hurdle_base_rates"] = hurdle_stats.get("base_rates", {})
        base_config["hurdle_pos_weight"] = hurdle_stats.get("pos_weight", {})
        base_config["hurdle_delta_map"] = hurdle_stats.get("delta_map", {})

    return base_config, date_time_stamp, dataset_contents, group_tag


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────


def _constant_weights_hurdle(h: int) -> np.ndarray:
    """Uniform horizon weights (all steps equally weighted)."""
    start = max(0, h - 24)
    w = np.zeros(h, dtype=np.float32)
    w[start:] = 1.0
    return w


def build_hurdle_model(
    config: HurdleTrainConfig,
    model_name: str,
    hurdle_stats: dict[str, dict],
    static_df: pd.DataFrame,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    pl_logger: WandbLogger,
    checkpoint_cb: ModelCheckpoint | None,
):
    """
    Instantiate a single hurdle model with its coupling loss.

    Parameters
    ----------
    config:
        Training configuration (``HurdleTrainConfig``).
    model_name:
        One of "nhits", "nbeatsx", "tft".
    hurdle_stats:
        Output of ``_compute_hurdle_stats``.
    static_df:
        Static exogenous DataFrame (direction one-hots).
    future_cov_cols, hist_cov_cols:
        Covariate lists.
    pl_logger:
        PyTorch Lightning / W&B logger.
    checkpoint_cb:
        Optional ``ModelCheckpoint`` callback.
    """
    h = config.forecast_horizon
    horizon_weight = _constant_weights_hurdle(h)

    # Convert fractional schedule steps → absolute step counts
    schedule_steps = tuple(
        int(p * config.max_steps) for p in config.hurdle_schedule_steps
    )

    # Per-model parameter overrides from YAML
    mp: dict = config.model_params.get(model_name, {})

    # ── Build losses ──────────────────────────────────────────────────────
    train_loss = HurdleCouplingPointLoss(
        horizon_weight=horizon_weight,
        eps=config.hurdle_eps,
        pos_weight=hurdle_stats["pos_weight"],
        static_df=static_df,
        delta_map=hurdle_stats["delta_map"],
        schedule_steps=schedule_steps,
        lambda_bce_final=config.hurdle_lambda_bce_final,
    )
    valid_loss = HurdleEvalLoss(
        horizon_weight=horizon_weight,
        loss_name="mae",
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks: list = []
    if checkpoint_cb is not None:
        callbacks.append(checkpoint_cb)

    if config.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="valid_loss",
                mode="min",
                patience=config.early_stop_patience,
            )
        )

    # ── Common kwargs ─────────────────────────────────────────────────────
    alias = _model_alias(model_name, config)
    common_kwargs = dict(
        h=h,
        futr_exog_list=future_cov_cols,
        hist_exog_list=hist_cov_cols,
        stat_exog_list=STATIC_EXOG_COLS,
        batch_size=config.batch_size,
        windows_batch_size=config.windows_batch_size,
        loss=train_loss,
        valid_loss=valid_loss,
        max_steps=config.max_steps,
        val_check_steps=config.val_check_steps,
        early_stop_patience_steps=-1,
        random_seed=config.random_seed,
        scaler_type=config.scaler_type,
        alias=alias,
        logger=pl_logger,
        enable_checkpointing=checkpoint_cb is not None,
        callbacks=callbacks,
    )

    # ── Hurdle-specific shared kwargs ─────────────────────────────────────
    hurdle_kwargs = dict(
        per_series_base_rates=hurdle_stats["base_rates"],
        per_series_median_z=hurdle_stats["median_z"],
        scale_magnitude_head=config.hurdle_scale_magnitude_head,
    )

    # ── Architecture-specific defaults ───────────────────────────────────
    ModelClass = _HURDLE_MODEL_CLASSES[model_name]

    if model_name == "nhits":
        defaults = {
            "input_size": config.input_size,
            "n_blocks": [20, 15, 10],
            "learning_rate": config.learning_rate,
        }
        defaults.update(mp)
        return ModelClass(**hurdle_kwargs, **defaults, **common_kwargs)

    if model_name == "nbeatsx":
        defaults = {
            "input_size": config.input_size,
            "n_blocks": [15, 5, 2],
            "stack_types": ["identity", "seasonality", "trend"],
            "learning_rate": 5e-5,
        }
        defaults.update(mp)
        return ModelClass(**hurdle_kwargs, **defaults, **common_kwargs)

    if model_name == "tft":
        common_kwargs["windows_batch_size"] = min(config.windows_batch_size, 32)
        defaults = {
            "input_size": config.input_size,
            "hidden_size": 64,
            "dropout": 0.15,
            "n_head": 4,
            "attn_dropout": 0.1,
        }
        defaults.update(mp)
        return ModelClass(**hurdle_kwargs, **defaults, **common_kwargs)

    raise ValueError(f"Unknown hurdle model name: {model_name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction post-processing
# ─────────────────────────────────────────────────────────────────────────────


def _add_hurdle_point_forecast(preds_df: pd.DataFrame, alias: str) -> pd.DataFrame:
    """
    Combine the two hurdle output channels into a single point forecast.

    Adds a column ``alias`` containing ``sigmoid(logit) * relu(magnitude)``.
    This column can then be used by ``prepare_predictions_df`` / evaluation.
    """
    logit_col = f"{alias}non_zero_logit"
    mag_col = f"{alias}magnitude"

    if logit_col not in preds_df.columns or mag_col not in preds_df.columns:
        logger.warning(
            "Expected hurdle columns '%s' and '%s' not found; "
            "available: %s",
            logit_col, mag_col, list(preds_df.columns),
        )
        return preds_df

    logit_vals = preds_df[logit_col].to_numpy(dtype=np.float64)
    mag_vals = preds_df[mag_col].to_numpy(dtype=np.float64)

    # sigmoid(logit) × relu(magnitude)
    p_hat = expit(logit_vals)
    m_hat = np.maximum(mag_vals, 0.0)
    preds_df = preds_df.copy()
    preds_df[alias] = p_hat * m_hat
    return preds_df


def optimize_hurdle_threshold(
    preds_df: pd.DataFrame,
    alias: str,
    fbeta2_target: float = 0.5,
    lambda_: float = 0.5,
    n_grid: int = 200,
    beta: float = 2.0,
) -> dict:
    """
    Find the optimal probability threshold *θ* for the hurdle binary gate.

    The threshold-gated forecast is defined as::

        ŷ = 1_{p̂ ≥ θ} · max(0, m̂)

    where ``p̂ = sigmoid(logit)`` and ``m̂`` is the (unscaled) magnitude
    prediction.  *θ* is chosen to minimise the combined metric::

        MAE + λ · max(0, Fβ2_target − Fβ2)

    The penalty term λ · max(0, Fβ2_target − Fβ2) is zero when the achieved
    Fβ2 score is at or above *fbeta2_target*, so it only activates to prevent
    the optimiser from collapsing toward a trivially large MAE-optimal
    threshold that sacrifices recall.

    Parameters
    ----------
    preds_df : pd.DataFrame
        DataFrame containing at minimum the columns:

        * ``{alias}-non_zero_logit`` – raw logit for P(non-zero)
        * ``{alias}-magnitude``      – magnitude prediction
        * ``y``                       – ground-truth values
        * ``unique_id``              – series identifier (direction)
    alias : str
        Model alias; determines the hurdle column name prefix.
    fbeta2_target : float
        Target Fβ2 score.  The penalty is proportional to the shortfall
        below this target (zero when Fβ2 ≥ fbeta2_target).
    lambda_ : float
        Weight on the Fβ2 penalty (λ in the formula above).  Default 0.5.
    n_grid : int
        Number of candidate threshold values evaluated in the open interval
        (0, 1).
    beta : float
        β for the F-score (2 ⇒ recall-weighted Fβ2).

    Returns
    -------
    dict
        ``optimal_theta``       – argmin of the combined metric
        ``combined_metric``     – value of the combined metric at optimal_theta
        ``mae``                 – MAE at optimal_theta
        ``fbeta_mean``          – mean Fβ across directions at optimal_theta
        ``fbeta_per_direction`` – per-direction Fβ at optimal_theta
        ``lambda_``             – lambda_ used
        ``fbeta_target``        – fbeta2_target used
        ``beta``                – beta used
        ``grid_thetas``         – list of all evaluated θ values
        ``grid_combined_metrics``, ``grid_mae``, ``grid_fbeta`` – per-θ metrics
    """
    logit_col = f"{alias}non_zero_logit"
    mag_col = f"{alias}magnitude"

    missing = [c for c in [logit_col, mag_col, "y", "unique_id"] if c not in preds_df.columns]
    if missing:
        raise ValueError(
            f"optimize_hurdle_threshold: missing columns {missing}. "
            f"Available: {list(preds_df.columns)}"
        )

    logit_vals = preds_df[logit_col].to_numpy(dtype=np.float64)
    mag_vals   = preds_df[mag_col].to_numpy(dtype=np.float64)
    y_true     = preds_df["y"].to_numpy(dtype=np.float64)
    uids       = preds_df["unique_id"].to_numpy()

    p_hat = expit(logit_vals)          # ∈ (0, 1)
    m_hat = np.maximum(mag_vals, 0.0)  # rectified magnitude
    y_bin = (y_true > 0).astype(int)   # binary ground truth
    dirs  = np.unique(uids)

    # Candidate thresholds in the open interval (0, 1)
    thetas = np.linspace(0.0, 1.0, n_grid + 2)[1:-1]  # shape (n_grid,)

    grid_combined = np.empty(len(thetas))
    grid_mae      = np.empty(len(thetas))
    grid_fbeta    = np.empty(len(thetas))

    for i, theta in enumerate(thetas):
        forecast = (p_hat >= theta).astype(np.float64) * m_hat

        mae_val = float(np.mean(np.abs(forecast - y_true)))

        dir_fb = [
            fbeta_score(
                y_bin[uids == d],
                (p_hat[uids == d] >= theta).astype(int),
                average="binary",
                zero_division=0,
                beta=beta,
            )
            for d in dirs
        ]
        mean_fb = float(np.mean(dir_fb))

        penalty  = float(lambda_) * max(0.0, float(fbeta2_target) - mean_fb)
        grid_mae[i]      = mae_val
        grid_fbeta[i]    = mean_fb
        grid_combined[i] = mae_val + penalty

    best_idx   = int(np.argmin(grid_combined))
    best_theta = float(thetas[best_idx])

    # Per-direction Fβ recomputed at the optimal threshold
    fbeta_per_dir: dict[str, float] = {
        str(d): float(
            fbeta_score(
                y_bin[uids == d],
                (p_hat[uids == d] >= best_theta).astype(int),
                average="binary",
                zero_division=0,
                beta=beta,
            )
        )
        for d in dirs
    }

    return {
        "optimal_theta":       best_theta,
        "combined_metric":     float(grid_combined[best_idx]),
        "mae":                 float(grid_mae[best_idx]),
        "fbeta_mean":          float(grid_fbeta[best_idx]),
        "fbeta_per_direction": fbeta_per_dir,
        "lambda_":             float(lambda_),
        "fbeta_target":        float(fbeta2_target),
        "beta":                float(beta),
        "grid_thetas":              thetas.tolist(),
        "grid_combined_metrics":    grid_combined.tolist(),
        "grid_mae":                 grid_mae.tolist(),
        "grid_fbeta":               grid_fbeta.tolist(),
    }


def _optimize_and_save_valid_threshold(
    nf: RedispatchNeuralForecast,
    alias: str,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    df_shifted: pd.DataFrame,
    df_unshifted: pd.DataFrame,
    static_df: pd.DataFrame,
    future_cov_cols: list[str],
    config: "HurdleTrainConfig",
    output_path: Path,
    window_index: int | None = None,
) -> dict | None:
    """
    Predict on the validation period, optimise the hurdle probability
    threshold θ, and persist the result as a JSON file at *output_path*.

    The threshold is the argmin of::

        MAE + λ · max(0, Fβ2_target − Fβ2)

    evaluated over a uniform grid of 200 candidate values in (0, 1).
    The optimised θ can later be applied at inference time via::

        ŷ = 1_{p̂ ≥ θ} · max(0, m̂)

    Parameters
    ----------
    nf : RedispatchNeuralForecast
        Fitted model wrapper.
    alias : str
        Model alias (determines the ``{alias}-non_zero_logit`` column prefix).
    valid_start, valid_end : pd.Timestamp
        Physical-time boundaries of the validation period (inclusive).
    df_shifted : pd.DataFrame
        Full shifted dataset (Nixtla format, used as history).
    df_unshifted : pd.DataFrame
        Unshifted dataset (for future covariates and ground-truth ``y``).
    static_df : pd.DataFrame
        Direction one-hot static features.
    future_cov_cols : list[str]
        Future covariate column names the model was trained with.
    config : HurdleTrainConfig
        Training config (provides threshold optimisation hyperparameters and
        ``shift_hours``, ``forecast_horizon``, ``tso``, ``holidays_path``).
    output_path : Path
        Where to write the JSON result (parent directory is created if needed).
    window_index : int | None
        Rolling-window index, stored in the JSON for traceability.

    Returns
    -------
    dict | None
        Result dict from :func:`optimize_hurdle_threshold`, or ``None`` if
        predictions could not be generated or optimisation failed.
    """
    label = f" (window {window_index})" if window_index is not None else ""
    logger.info(
        "Optimising hurdle threshold on validation set [%s, %s]%s",
        valid_start, valid_end, label,
    )

    val_preds = predict_with_shift_correction(
        nf=nf,
        df_shifted=df_shifted,
        df_unshifted=df_unshifted,
        static_df=static_df,
        pred_start=valid_start,
        pred_end=valid_end,
        future_cov_cols=future_cov_cols,
        shift_hours=config.shift_hours,
        forecast_horizon=config.forecast_horizon,
        tso=config.tso,
        holidays_path=config.holidays_path,
    )
    if val_preds.empty:
        logger.warning(
            "Empty validation predictions%s – skipping threshold optimisation.", label
        )
        return None

    # Attach ground-truth actuals
    val_preds = val_preds.merge(
        df_unshifted[["unique_id", "ds", "y"]],
        on=["unique_id", "ds"],
        how="left",
    ).dropna(subset=["y"])

    if val_preds.empty:
        logger.warning(
            "No actuals matched validation predictions%s – skipping.", label
        )
        return None

    try:
        result = optimize_hurdle_threshold(
            preds_df=val_preds,
            alias=alias,
            fbeta2_target=config.hurdle_theta_fbeta2_target,
            lambda_=config.hurdle_theta_lambda,
            n_grid=config.hurdle_theta_n_grid,
        )
    except Exception as exc:
        logger.warning(
            "Threshold optimisation failed%s: %s", label, exc, exc_info=True
        )
        return None

    result["alias"] = alias
    result["tso"] = config.tso
    result["valid_start"] = str(valid_start)
    result["valid_end"] = str(valid_end)
    if window_index is not None:
        result["window_index"] = window_index

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(
        "Hurdle θ optimised%s: θ=%.4f  combined=%.4f  MAE=%.4f  Fβ=%.4f  → %s",
        label,
        result["optimal_theta"],
        result["combined_metric"],
        result["mae"],
        result["fbeta_mean"],
        output_path,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Single-window training
# ─────────────────────────────────────────────────────────────────────────────


def train_single_window_hurdle(
    df: pd.DataFrame,
    config: HurdleTrainConfig,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    metadata: dict | None = None,
    df_unshifted: pd.DataFrame | None = None,
) -> list[RedispatchNeuralForecast]:
    """
    Train hurdle models on a single train/val split.

    Parameters
    ----------
    df:
        Shifted Nixtla-format DataFrame.
    config:
        ``HurdleTrainConfig`` with all hyperparameters.
    future_cov_cols, hist_cov_cols:
        Covariate column lists.
    metadata:
        Optional dataset metadata dict (used for W&B grouping).
    df_unshifted:
        Unshifted version of ``df`` for shift-corrected prediction.

    Returns
    -------
    list[RedispatchNeuralForecast]
        One fitted ``RedispatchNeuralForecast`` wrapper per model.
    """
    # ── Slice training window ─────────────────────────────────────────────
    if config.shift_hours > 0:
        train_start_ts = pd.Timestamp(config.train_start) - pd.Timedelta(hours=config.shift_hours)
        valid_start_ts = pd.Timestamp(config.valid_start) - pd.Timedelta(hours=config.shift_hours)
        test_start_ts = pd.Timestamp(config.test_start) - pd.Timedelta(hours=config.shift_hours)
    else:
        train_start_ts = pd.Timestamp(config.train_start)
        valid_start_ts = pd.Timestamp(config.valid_start)
        test_start_ts = pd.Timestamp(config.test_start)

    train_df = df[(df["ds"] >= train_start_ts) & (df["ds"] < test_start_ts)].copy()
    val_hours = int((test_start_ts - valid_start_ts).total_seconds() / 3600)

    logger.info(
        "Single-window hurdle training: %d rows, val_size=%d hours",
        len(train_df), val_hours,
    )

    # ── Compute hurdle statistics ─────────────────────────────────────────
    hurdle_stats = _compute_hurdle_stats(train_df, config)
    static_df = build_static_df()

    # ── W&B group config ──────────────────────────────────────────────────
    base_config, date_time, dataset_contents, group_tag = make_hurdle_wandb_config(
        config=config,
        hist_cov_cols=hist_cov_cols,
        future_cov_cols=future_cov_cols,
        hurdle_stats=hurdle_stats,
        metadata=metadata,
    )

    nf_list: list[RedispatchNeuralForecast] = []

    for model_name in config.models:
        alias = _model_alias(model_name, config)

        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"train_{config.tso}_hurdle_{alias}_{date_time}",
            job_type="training",
            config={
                **base_config,
                "model_name": model_name,
                "model_alias": alias,
            },
            group=group_tag,
            reinit="finish_previous",
        )

        try:
            pl_logger = WandbLogger(experiment=run, log_model=False)

            should_persist = _should_persist_checkpoints(config)
            checkpoint_dir = None
            checkpoint_cb = None
            if should_persist:
                checkpoint_dir = (
                    Path(config.output_dir)
                    / dataset_contents.replace(", ", "_")
                    / f"{config.normalized_tso}_hurdle"
                    / date_time
                    / alias
                )
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_cb = _make_checkpoint_callback(checkpoint_dir)

            model = build_hurdle_model(
                config=config,
                model_name=model_name,
                hurdle_stats=hurdle_stats,
                static_df=static_df,
                future_cov_cols=future_cov_cols,
                hist_cov_cols=hist_cov_cols,
                pl_logger=pl_logger,
                checkpoint_cb=checkpoint_cb,
            )
            run.config.update({"model_params": dict(model.hparams)}, allow_val_change=True)

            nf = RedispatchNeuralForecast(
                models=[model],
                freq="h",
                local_scaler_type=config.local_scaler_type,
            )

            nf.fit(df=train_df, static_df=static_df, val_size=val_hours)

            # ── Persist checkpoint ────────────────────────────────────────
            if should_persist and checkpoint_dir is not None:
                nf_dir = checkpoint_dir / "nf_model"
                nf.save(str(nf_dir), overwrite=True)
                logger.info("Saved RedispatchNeuralForecast model to %s", nf_dir)

                best_ckpt = find_best_valid_checkpoint(checkpoint_dir)
                if best_ckpt:
                    best_ckpt.rename(nf_dir / CHECKPOINT_BEST_NAME)

                archive_path = None
                if config.persist_checkpoints_to_wandb:
                    archive_path = _log_checkpoint_artifact(run, checkpoint_dir, alias, date_time, config)
                elif config.checkpoint_compression:
                    archive_path = _archive_checkpoint_dir(
                        checkpoint_dir, config.checkpoint_compression, config.checkpoint_compression_n_threads
                    )

                if not config.persist_checkpoints and archive_path and archive_path.exists():
                    archive_path.unlink()
                if archive_path and archive_path.exists():
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)

            # ── Predictions ───────────────────────────────────────────────
            _unshifted = df_unshifted if df_unshifted is not None else df
            if df_unshifted is not None or config.shift_hours == 0:
                test_pred_end = _unshifted["ds"].max()
                logger.info(
                    "Generating test predictions for %s (hurdle): [%s, %s]",
                    alias, config.test_start, test_pred_end,
                )
                test_preds = predict_with_shift_correction(
                    nf=nf,
                    df_shifted=df,
                    df_unshifted=_unshifted,
                    static_df=static_df,
                    pred_start=config.test_start_ts,
                    pred_end=test_pred_end,
                    future_cov_cols=future_cov_cols,
                    shift_hours=config.shift_hours,
                    forecast_horizon=config.forecast_horizon,
                    tso=config.tso,
                    holidays_path=config.holidays_path,
                )
                if not test_preds.empty:
                    test_preds = _add_hurdle_point_forecast(test_preds, alias)
                    test_preds = prepare_predictions_df(test_preds, _unshifted)
                    log_predictions_to_wandb(
                        run, test_preds, "test", alias,
                        tso=config.tso, shift_hours=config.shift_hours,
                        timestamp=date_time,
                    )
            else:
                logger.warning("Skipping predictions: df_unshifted not provided and shift_hours > 0")

            # ── Threshold optimisation on validation set ──────────────────────
            if df_unshifted is not None or config.shift_hours == 0:
                valid_pred_start = config.valid_start_ts
                valid_pred_end   = config.test_start_ts - pd.Timedelta(hours=1)
                if should_persist and checkpoint_dir is not None:
                    theta_save_dir = checkpoint_dir.parent
                else:
                    theta_save_dir = Path(config.output_dir)
                theta_save_dir.mkdir(parents=True, exist_ok=True)
                _optimize_and_save_valid_threshold(
                    nf=nf,
                    alias=alias,
                    valid_start=valid_pred_start,
                    valid_end=valid_pred_end,
                    df_shifted=df,
                    df_unshifted=_unshifted,
                    static_df=static_df,
                    future_cov_cols=future_cov_cols,
                    config=config,
                    output_path=theta_save_dir / f"hurdle_theta_opt_{alias}.json",
                )

            nf_list.append(nf)
        except Exception as e:
            logger.error("Error during training %s. Skipping...", alias, exc_info=True)
        finally:
            run.finish()

    return nf_list


# ─────────────────────────────────────────────────────────────────────────────
# Rolling-window training
# ─────────────────────────────────────────────────────────────────────────────


def train_rolling_windows_hurdle(
    df: pd.DataFrame,
    config: HurdleTrainConfig,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    metadata: dict | None = None,
    df_unshifted: pd.DataFrame | None = None,
    start_from_window: int = 0,
) -> list[RedispatchNeuralForecast]:
    """
    Train hurdle models over rolling windows (one-month step).

    The shift-correction logic mirrors ``train_rolling_windows`` in
    ``runner.py``: window boundaries are expressed in *physical* time and
    shifted by ``config.shift_hours`` before slicing ``df``.
    """
    data_start = df["ds"].min()
    data_end = df["ds"].max()

    windows: list[WindowBoundary] = _compute_rolling_windows(
        data_start, data_end,
        config.n_train_months, config.n_valid_months, config.n_test_months,
    )
    logger.info("Rolling-window hurdle training: %d windows", len(windows))

    nf_list: list[RedispatchNeuralForecast] = []

    for wi, wb in enumerate(windows):
        if wi < start_from_window:
            continue

        if config.shift_hours > 0:
            train_start_ts = pd.Timestamp(wb.train_start) - pd.Timedelta(hours=config.shift_hours)
            test_start_ts = pd.Timestamp(wb.test_start) - pd.Timedelta(hours=config.shift_hours)
        else:
            train_start_ts = pd.Timestamp(wb.train_start)
            test_start_ts = pd.Timestamp(wb.test_start)

        logger.info(
            "Window %d/%d  train=[%s, %s)  val=[%s, %s)  test=[%s, %s)",
            wi + 1, len(windows),
            wb.train_start, wb.valid_start,
            wb.valid_start, wb.test_start,
            wb.test_start, wb.test_end,
        )

        window_df = df[(df["ds"] >= train_start_ts) & (df["ds"] < test_start_ts)].copy()
        val_hours = int((wb.test_start - wb.valid_start).total_seconds() / 3600)

        # ── Hurdle statistics on this window's training data ──────────
        hurdle_stats = _compute_hurdle_stats(window_df, config)
        static_df = build_static_df()

        # ── W&B group config ──────────────────────────────────────────
        base_config, date_time, dataset_contents, window_group = make_hurdle_wandb_config(
            config=config,
            hist_cov_cols=hist_cov_cols,
            future_cov_cols=future_cov_cols,
            hurdle_stats=hurdle_stats,
            window_index=wi,
            metadata=metadata,
            window_test_start=str(wb.test_start),
            window_test_end=str(wb.test_end),
        )

        for model_name in config.models:
            alias = _model_alias(model_name, config)

            run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=f"train_{config.normalized_tso}_hurdle_{alias}_{date_time}_w{wi}",
                job_type="training-rolling",
                config={
                    **base_config,
                    "model_name": model_name,
                    "model_alias": alias,
                },
                group=window_group,
                reinit="finish_previous",
            )

            pl_logger = WandbLogger(experiment=run, log_model=False)

            should_persist = _should_persist_checkpoints(config)
            checkpoint_dir = None
            checkpoint_cb = None
            if should_persist:
                checkpoint_dir = (
                    Path(config.output_dir)
                    / f"{config.normalized_tso}_hurdle"
                    / dataset_contents.replace(", ", "_")
                    / f"window_{wi}"
                    / alias
                    / date_time
                )
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_cb = _make_checkpoint_callback(checkpoint_dir)

            model = build_hurdle_model(
                config=config,
                model_name=model_name,
                hurdle_stats=hurdle_stats,
                static_df=static_df,
                future_cov_cols=future_cov_cols,
                hist_cov_cols=hist_cov_cols,
                pl_logger=pl_logger,
                checkpoint_cb=checkpoint_cb,
            )
            run.config.update({"model_params": dict(model.hparams)}, allow_val_change=True)

            nf = RedispatchNeuralForecast(
                models=[model],
                freq="h",
                local_scaler_type=config.local_scaler_type,
            )
            nf.fit(df=window_df, static_df=static_df, val_size=val_hours)

            # ── Persist checkpoint ────────────────────────────────────
            if should_persist and checkpoint_dir is not None:
                nf_dir = checkpoint_dir / "nf_model"
                nf.save(str(nf_dir), overwrite=True)
                logger.info("Saved RedispatchNeuralForecast model (window %d) to %s", wi, nf_dir)

                best_ckpt = find_best_valid_checkpoint(checkpoint_dir)
                if best_ckpt:
                    best_ckpt.rename(nf_dir / CHECKPOINT_BEST_NAME)

                archive_path = None
                if config.persist_checkpoints_to_wandb:
                    archive_path = _log_checkpoint_artifact(
                        run, checkpoint_dir, alias, date_time, config, window_index=wi
                    )
                elif config.checkpoint_compression:
                    archive_path = _archive_checkpoint_dir(
                        checkpoint_dir, config.checkpoint_compression, config.checkpoint_compression_n_threads
                    )

                if not config.persist_checkpoints and archive_path and archive_path.exists():
                    archive_path.unlink()
                if archive_path and archive_path.exists():
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)

            # ── Predictions ───────────────────────────────────────────
            _unshifted = df_unshifted if df_unshifted is not None else df
            if df_unshifted is not None or config.shift_hours == 0:
                test_pred_end = wb.test_end - pd.Timedelta(hours=1)
                logger.info(
                    "Generating test predictions for window %d/%d, %s (hurdle): [%s, %s]",
                    wi + 1, len(windows), alias, wb.test_start, test_pred_end,
                )
                test_preds = predict_with_shift_correction(
                    nf=nf,
                    df_shifted=df,
                    df_unshifted=_unshifted,
                    static_df=static_df,
                    pred_start=wb.test_start,
                    pred_end=test_pred_end,
                    future_cov_cols=future_cov_cols,
                    shift_hours=config.shift_hours,
                    forecast_horizon=config.forecast_horizon,
                    tso=config.tso,
                    holidays_path=config.holidays_path,
                )
                if not test_preds.empty:
                    test_preds = _add_hurdle_point_forecast(test_preds, alias)
                    test_preds = prepare_predictions_df(test_preds, _unshifted)
                    log_predictions_to_wandb(
                        run, test_preds, f"test_window{wi}", alias,
                        tso=config.tso, shift_hours=config.shift_hours,
                        timestamp=date_time, log_table=False,
                    )
            else:
                logger.warning("Skipping predictions: df_unshifted not provided and shift_hours > 0")

            # ── Threshold optimisation on validation set ──────────────────────
            if df_unshifted is not None or config.shift_hours == 0:
                valid_pred_start = wb.valid_start
                valid_pred_end   = wb.test_start - pd.Timedelta(hours=1)
                if should_persist and checkpoint_dir is not None:
                    theta_save_dir = checkpoint_dir.parent.parent
                else:
                    theta_save_dir = Path(config.output_dir) / f"{config.normalized_tso}_hurdle"
                theta_save_dir.mkdir(parents=True, exist_ok=True)
                _optimize_and_save_valid_threshold(
                    nf=nf,
                    alias=alias,
                    valid_start=valid_pred_start,
                    valid_end=valid_pred_end,
                    df_shifted=df,
                    df_unshifted=_unshifted,
                    static_df=static_df,
                    future_cov_cols=future_cov_cols,
                    config=config,
                    output_path=theta_save_dir / f"hurdle_theta_opt_{alias}_w{wi}.json",
                    window_index=wi,
                )

            run.finish()
            nf_list.append(nf)

    return nf_list


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# Additional flat YAML keys beyond those in train_pipeline.py
_YAML_HURDLE_FLAT_KEYS = {
    "hurdle_schedule_steps",
    "hurdle_lambda_bce_final",
    "hurdle_eps",
    "hurdle_huber_quantile",
    "hurdle_scale_magnitude_head",
    "hurdle_theta_fbeta2_target",
    "hurdle_theta_lambda",
    "hurdle_theta_n_grid",
    # Inherit all regular flat keys
    "direction",
    "forecast_horizon",
    "input_size",
    "shift_hours",
    "holidays_path",
    "train_start",
    "valid_start",
    "test_start",
    "max_steps",
    "val_check_steps",
    "early_stop_patience",
    "batch_size",
    "windows_batch_size",
    "learning_rate",
    "random_seed",
    "local_scaler_type",
    "n_train_months",
    "n_valid_months",
    "n_test_months",
    "start_window",
    "models",
    "output_dir",
    "wandb_project",
    "wandb_entity",
    "persist_checkpoints",
    "persist_checkpoints_to_wandb",
    "checkpoint_compression",
    "checkpoint_compression_n_threads",
}


def _load_yaml_config(config_path: str) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    logger.info("Loaded hurdle config from %s", config_path)
    return cfg


def _flat_defaults_from_yaml(yaml_cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in yaml_cfg.items() if k in _YAML_HURDLE_FLAT_KEYS}


def _normalize_tso_key(tso: str) -> str:
    return tso.strip().replace(" ", "_")


def _explicit_cli_keys() -> set[str]:
    explicit: set[str] = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            key_part = arg[2:].split("=")[0]
            explicit.add(key_part.replace("-", "_"))
    return explicit


def _apply_tso_overrides(
    args: argparse.Namespace,
    yaml_cfg: dict[str, Any],
    tso: str,
    explicit_cli_keys: set[str],
) -> None:
    """Apply per-TSO YAML overrides (same logic as ``apply_tso_overrides`` in train_pipeline)."""
    tso_norm = _normalize_tso_key(tso)
    overrides: dict[str, Any] = yaml_cfg.get("tso_overrides", {})
    tso_cfg: dict[str, Any] = overrides.get(tso_norm) or overrides.get(tso) or {}
    if not tso_cfg:
        return

    flat_ov = {k: v for k, v in tso_cfg.items() if not isinstance(v, dict)}
    model_ov = {k: v for k, v in tso_cfg.items() if isinstance(v, dict)}

    for key, value in flat_ov.items():
        if key not in _YAML_HURDLE_FLAT_KEYS:
            continue
        if key in explicit_cli_keys:
            continue
        setattr(args, key, value)
        logger.info("  TSO override %s = %s", key, value)

    if "models" in flat_ov and "models" not in explicit_cli_keys:
        args.models = _normalize_hurdle_models(args.models)

    if model_ov:
        merged: dict[str, Any] = dict(getattr(args, "model_params", {}))
        for model_name, params in model_ov.items():
            if model_name in merged:
                merged[model_name] = {**merged[model_name], **params}
            else:
                merged[model_name] = params
        args.model_params = merged


def _normalize_hurdle_models(raw_models: list[str]) -> list[str]:
    models: list[str] = []
    for item in raw_models:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        models.extend(parts)
    models = [m.lower() for m in models]

    invalid = [m for m in models if m not in _ALLOWED_HURDLE_MODELS]
    if invalid:
        raise SystemExit(
            f"hurdle_runner.py: invalid --models choice(s): {invalid}. "
            f"Choose from {sorted(_ALLOWED_HURDLE_MODELS)}"
        )

    seen: set[str] = set()
    return [m for m in models if not (m in seen or seen.add(m))]  # type: ignore[func-returns-value]


def set_n_threads(n_threads: int | None) -> None:
    if n_threads is not None and n_threads > 0:
        import torch

        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)
        logger.info("Set number of threads to %d", n_threads)


def parse_args() -> argparse.Namespace:
    # ── Two-pass: extract --config before finalising defaults ─────────────
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_args, _ = _pre.parse_known_args()

    yaml_cfg: dict[str, Any] = {}
    if _pre_args.config:
        yaml_cfg = _load_yaml_config(_pre_args.config)

    p = argparse.ArgumentParser(
        description="Train hurdle (zero-inflated) Nixtla models for redispatch forecasting."
    )

    # ── Config file ───────────────────────────────────────────────────────
    p.add_argument("--config", default=None, help="Path to a YAML config file.")

    # ── Dataset ───────────────────────────────────────────────────────────
    p.add_argument("--dataset-path", required=True, help="Path to the .parquet dataset.")
    p.add_argument(
        "--direction", default="both", choices=["up", "down", "both"],
        help="Direction filter (default: both).",
    )

    # ── Shifting ──────────────────────────────────────────────────────────
    p.add_argument("--shift-hours", type=int, default=6, help="Target shift in hours (default: 6).")

    # ── Threads ──────────────────────────────────────────────────────────
    p.add_argument("--n-threads", type=int, default=None, help="Number of threads to use.")

    # ── Date time ─────────────────────────────────────────────────────────
    p.add_argument("--date-time", default=None, type=str, help="Optional W&B date_time override.")

    # ── Calendar ──────────────────────────────────────────────────────────
    p.add_argument("--no-calendar", action="store_true", help="Skip adding calendar features.")
    p.add_argument("--holidays-path", default=None, help="Override for holidays CSV path.")

    # ── Forecast / model ──────────────────────────────────────────────────
    p.add_argument("--forecast-horizon", type=int, default=24, help="Forecast horizon h (default: 24).")
    p.add_argument("--input-size", type=int, default=24, help="Input lookback L (default: 24).")
    p.add_argument(
        "--models", nargs="+", default=["nhits"],
        help="Hurdle models to train: nhits, nbeatsx, tft (default: nhits).",
    )

    # ── Date range ────────────────────────────────────────────────────────
    p.add_argument("--train-start", default="2021-10-01", help="Training start date.")
    p.add_argument("--valid-start", default="2024-10-01", help="Validation start date.")
    p.add_argument("--test-start", default="2025-01-01", help="Test start date.")

    # ── Training hyper-parameters ─────────────────────────────────────────
    p.add_argument("--max-steps", type=int, default=5000, help="Max training steps (default: 5000).")
    p.add_argument("--val-check-steps", type=int, default=50, help="Validate every N steps.")
    p.add_argument("--early-stop-patience", type=int, default=20, help="Early stopping patience.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    p.add_argument("--windows-batch-size", type=int, default=64, help="Windows batch size.")
    p.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--random-seed", type=int, default=778, help="Random seed.")
    p.add_argument("--local-scaler-type", default="standard", help="Local scaler type (default: standard).")

    # ── Hurdle-specific ───────────────────────────────────────────────────
    p.add_argument(
        "--hurdle-schedule-steps", nargs=2, type=float, default=[0.15, 0.50],
        metavar=("STEP1", "STEP2"),
        help=(
            "Two fractions of max_steps marking phase boundaries "
            "(default: 0.15 0.50).  Phase 1 = BCE only, "
            "Phase 2 = BCE+mag, Phase 3 = BCE+mag+coupling."
        ),
    )
    p.add_argument(
        "--hurdle-lambda-bce-final", type=float, default=0.15,
        help="BCE loss weight in final phase (default: 0.15).",
    )
    p.add_argument(
        "--hurdle-eps", type=float, default=0.1,
        help="Threshold below which y is treated as zero (default: 0.1).",
    )
    p.add_argument(
        "--hurdle-huber-quantile", type=float, default=0.85,
        help="Quantile of non-zero training values used as Huber δ (default: 0.85).",
    )
    p.add_argument(
        "--hurdle-no-scale-magnitude", action="store_true",
        help="Disable z-score scaling of the magnitude head.",
    )
    p.add_argument(
        "--hurdle-theta-fbeta2-target", type=float, default=0.5,
        help=(
            "Target Fβ2 score used in the validation-set threshold optimisation "
            "penalty: MAE + λ·max(0, Fβ2_target − Fβ2) (default: 0.5)."
        ),
    )
    p.add_argument(
        "--hurdle-theta-lambda", type=float, default=0.5,
        help="Penalty weight λ for the Fβ2 shortfall in threshold optimisation (default: 0.5).",
    )
    p.add_argument(
        "--hurdle-theta-n-grid", type=int, default=200,
        help="Number of candidate θ values to sweep in (0, 1) (default: 200).",
    )

    # ── Rolling window ────────────────────────────────────────────────────
    p.add_argument("--rolling-window", action="store_true", help="Use rolling-window training.")
    p.add_argument("--n-train-months", type=int, default=37, help="Training period in months.")
    p.add_argument("--n-valid-months", type=int, default=2, help="Validation period in months.")
    p.add_argument("--n-test-months", type=int, default=1, help="Test period in months.")
    p.add_argument("--start-window", type=int, default=0, help="Resume from N-th window (0-indexed).")

    # ── Output / logging ──────────────────────────────────────────────────
    p.add_argument("--output-dir", default="outputs", help="Directory for model artifacts.")
    p.add_argument("--wandb-project", default="redispatch-forecasting", help="W&B project name.")
    p.add_argument("--wandb-entity", default=None, help="W&B entity.")
    p.add_argument("--persist-checkpoints", action="store_true", help="Keep checkpoints locally.")
    p.add_argument("--persist-checkpoints-to-wandb", action="store_true", help="Log checkpoints to W&B.")
    p.add_argument("--checkpoint-compression", default=None, type=int, help="Zstd compression level.")
    p.add_argument("--checkpoint-compression-n-threads", default=20, type=int, help="Compression threads.")

    # ── Apply YAML config as argparse defaults ────────────────────────────
    if yaml_cfg:
        p.set_defaults(**_flat_defaults_from_yaml(yaml_cfg))

    args = p.parse_args()

    # ── Attach model-specific params dict (nested YAML sections) ─────────
    args.model_params = {k: v for k, v in yaml_cfg.items() if isinstance(v, dict) and k != "tso_overrides"}
    args._yaml_cfg = yaml_cfg
    args.models = _normalize_hurdle_models(args.models)
    return args


def main() -> None:
    args = parse_args()
    set_n_threads(args.n_threads)

    # ── 1. Load raw dataset ───────────────────────────────────────────────
    logger.info("Loading dataset from %s", args.dataset_path)
    raw_df, metadata = load_dataset(args.dataset_path)
    logger.info("Raw dataset: %d rows × %d cols", *raw_df.shape)

    # ── 2. Convert to Nixtla format ───────────────────────────────────────
    nixtla_df = to_nixtla_format(raw_df, direction=args.direction)
    logger.info("Nixtla DataFrame: %d rows, unique_ids=%s", len(nixtla_df), nixtla_df["unique_id"].unique().tolist())

    # ── 3. Shift + calendar ───────────────────────────────────────────────
    tso = metadata["operator"]
    _apply_tso_overrides(args, args._yaml_cfg, tso, _explicit_cli_keys())

    logger.info("Applying shift of %d hours", args.shift_hours)
    shifted_df, future_cov, hist_cov = prepare_shifted_dataset(
        nixtla_df,
        shift_hours=args.shift_hours,
        tso=tso,
        add_calendar=not args.no_calendar,
        holidays_path=args.holidays_path,
    )
    logger.info(
        "After shifting: %d rows | %d future covariates | %d historical covariates",
        len(shifted_df), len(future_cov), len(hist_cov),
    )

    # ── 4. Build HurdleTrainConfig ────────────────────────────────────────
    schedule_steps = tuple(float(x) for x in args.hurdle_schedule_steps)
    config = HurdleTrainConfig(
        dataset_path=args.dataset_path,
        tso=tso,
        direction=args.direction,
        shift_hours=args.shift_hours,
        add_calendar=not args.no_calendar,
        holidays_path=args.holidays_path,
        train_start=args.train_start,
        valid_start=args.valid_start,
        test_start=args.test_start,
        date_time=args.date_time,
        forecast_horizon=args.forecast_horizon,
        input_size=args.input_size,
        max_steps=args.max_steps,
        val_check_steps=args.val_check_steps,
        early_stop_patience=args.early_stop_patience,
        batch_size=args.batch_size,
        windows_batch_size=args.windows_batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
        local_scaler_type=args.local_scaler_type,
        models=args.models,
        rolling_window=args.rolling_window,
        n_train_months=args.n_train_months,
        n_valid_months=args.n_valid_months,
        n_test_months=args.n_test_months,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        persist_checkpoints=args.persist_checkpoints,
        persist_checkpoints_to_wandb=args.persist_checkpoints_to_wandb,
        checkpoint_compression=args.checkpoint_compression,
        checkpoint_compression_n_threads=args.checkpoint_compression_n_threads,
        model_params=getattr(args, "model_params", {}),
        # Hurdle-specific
        hurdle_schedule_steps=schedule_steps,
        hurdle_lambda_bce_final=args.hurdle_lambda_bce_final,
        hurdle_eps=args.hurdle_eps,
        hurdle_huber_quantile=args.hurdle_huber_quantile,
        hurdle_scale_magnitude_head=not args.hurdle_no_scale_magnitude,
        # Threshold optimisation
        hurdle_theta_fbeta2_target=args.hurdle_theta_fbeta2_target,
        hurdle_theta_lambda=args.hurdle_theta_lambda,
        hurdle_theta_n_grid=args.hurdle_theta_n_grid,
    )

    # ── 5. Train ──────────────────────────────────────────────────────────
    if config.rolling_window:
        logger.info("Starting rolling-window hurdle training (%d models)", len(config.models))
        train_rolling_windows_hurdle(
            shifted_df, config, future_cov, hist_cov, metadata,
            df_unshifted=nixtla_df, start_from_window=args.start_window,
        )
    else:
        logger.info("Starting single-window hurdle training (%d models)", len(config.models))
        train_single_window_hurdle(
            shifted_df, config, future_cov, hist_cov, metadata,
            df_unshifted=nixtla_df,
        )

    logger.info("Hurdle training complete ✓")


if __name__ == "__main__":
    main()
