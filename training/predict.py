"""
Prediction utilities for shifted-target Nixtla models.

Handles:
- Strided prediction with correct shift / data-leakage prevention
- Merging predictions with actuals
- Logging predictions to Weights & Biases (Table + Artifact)
"""

import logging
import sys
from typing import Optional

import numpy as np
import pandas as pd
import re
import wandb

from neuralforecast.core import NeuralForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse, mae
from sklearn.metrics import r2_score
from itertools import product

from training.data_prep import CALENDAR_COLS, add_calendar_features, build_static_df

logger = logging.getLogger(__name__)


# ── Core strided prediction ───────────────────────────────────────────────────

def predict_with_shift_correction(
    nf: NeuralForecast,
    df_shifted: pd.DataFrame,
    df_unshifted: pd.DataFrame,
    static_df: Optional[pd.DataFrame],
    pred_start: pd.Timestamp,
    pred_end: pd.Timestamp,
    future_cov_cols: list[str],
    shift_hours: int,
    forecast_horizon: int,
    tso: str,
    holidays_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate predictions with correct shift alignment to prevent data leakage.

    **Shifted models** (``shift_hours > 0``):

    * Historical data is drawn from ``df_shifted`` (matching the distribution
      seen during training).
    * Future covariates are taken from ``df_unshifted`` at the *physical*
      target timestamps-these are forecasts / day-ahead prices that would
      genuinely be available at decision time-then relabelled to model time
      so that NeuralForecast's internal scaffold matches.
    * Calendar features (hour, day, month, is_weekend, is_holiday, is_workday)
      are recomputed on the fly for the physical prediction timestamps.
    * Output ``ds`` values are shifted back to physical time.

    **Non-shifted models** (``shift_hours == 0``):

    * ``df_shifted`` and ``df_unshifted`` may be the same DataFrame.
    * Standard strided prediction with no relabelling.

    Parameters
    ----------
    nf : NeuralForecast
        Fitted NeuralForecast wrapper (containing one or more models).
    df_shifted : pd.DataFrame
        Full shifted dataset (Nixtla format: ds, y, unique_id, covariates).
    df_unshifted : pd.DataFrame
        Unshifted dataset with covariate values at their original timestamps.
    static_df : pd.DataFrame
        Static exogenous features (direction one-hot encoding).
    pred_start : pd.Timestamp
        First *physical* timestamp to predict (inclusive).
    pred_end : pd.Timestamp
        Last *physical* timestamp to predict (inclusive).
    future_cov_cols : list[str]
        All future-covariate column names the model was trained with
        (data covariates **and** calendar columns).
    shift_hours : int
        Number of hours by which targets were shifted during training.
    forecast_horizon : int
        Model's forecast horizon (h).
    tso : str
        TSO name (used for holiday lookup when recomputing calendar features).
    holidays_path : str | None
        Optional override for the holidays CSV path.

    Returns
    -------
    pd.DataFrame
        Predictions with ``ds`` in physical time, ``unique_id``, model
        prediction columns, and a ``horizon`` column (1-indexed forecast step).
    """
    n_series = df_shifted["unique_id"].nunique()

    # ── Separate data covariates from calendar features ───────────────────────
    data_futr_cols = [c for c in future_cov_cols if c not in CALENDAR_COLS]
    need_calendar = any(c in CALENDAR_COLS for c in future_cov_cols)

    # Safety: verify data future covariates exist in the unshifted df
    missing = [c for c in data_futr_cols if c not in df_unshifted.columns]
    if missing:
        logger.warning(
            "Columns %s not found in df_unshifted – they will be dropped from futr_df.",
            missing,
        )
        data_futr_cols = [c for c in data_futr_cols if c in df_unshifted.columns]

    # ── Model-time boundaries ─────────────────────────────────────────────────
    # model_pred_date: the *model-time* cutoff for the first stride
    model_pred_date = pred_start - pd.Timedelta(hours=shift_hours)
    # last possible model-time cutoff such that the stride still reaches pred_end
    last_stride_start = pred_end - pd.Timedelta(
        hours=shift_hours + forecast_horizon - 1
    )

    preds: list[pd.DataFrame] = []
    stride_idx = 0

    while model_pred_date <= last_stride_start:
        stride_idx += 1

        # 1. Historical data in model time (shifted, matching training) --------
        hist_df = df_shifted[df_shifted["ds"] < model_pred_date]
        if hist_df.empty:
            logger.warning(
                "Stride %d: empty history at model_pred_date=%s, skipping",
                stride_idx, model_pred_date,
            )
            model_pred_date += pd.Timedelta(hours=forecast_horizon)
            continue

        # 2. Future covariates at physical timestamps --------------------------
        phys_start = model_pred_date + pd.Timedelta(hours=shift_hours)
        phys_end = phys_start + pd.Timedelta(hours=forecast_horizon)

        futr_df = df_unshifted.loc[
            (df_unshifted["ds"] >= phys_start) & (df_unshifted["ds"] < phys_end),
            ["unique_id", "ds"] + data_futr_cols,
        ].copy()

        if len(futr_df) < forecast_horizon * n_series:
            logger.warning(
                "Stride %d: incomplete future covariates (%d/%d rows) "
                "for phys=[%s, %s), stopping prediction loop",
                stride_idx,
                len(futr_df),
                forecast_horizon * n_series,
                phys_start,
                phys_end,
            )
            break

        # 3. Recompute calendar features for the physical timestamps -----------
        if need_calendar:
            futr_df = add_calendar_features(
                futr_df,
                reference_time=futr_df["ds"],
                tso=tso,
                holidays_path=holidays_path,
            )

        # 4. Relabel ds → model time ------------------------------------------
        #    NeuralForecast builds a scaffold from last_date+1h in model time;
        #    our futr_df ds values must match that scaffold exactly.
        if shift_hours > 0:
            futr_df["ds"] = futr_df["ds"] - pd.Timedelta(hours=shift_hours)

        # 5. Predict -----------------------------------------------------------
        stride_preds = nf.predict(
            df=hist_df,
            futr_df=futr_df,
            static_df=static_df,
            verbose=False,
        )
        stride_preds_df = stride_preds.iloc[: forecast_horizon * n_series].copy()

        # 6. Shift prediction timestamps back to physical time -----------------
        if shift_hours > 0:
            stride_preds_df["ds"] = stride_preds_df["ds"] + pd.Timedelta(
                hours=shift_hours
            )

        # 7. Horizon label (1-indexed: hours ahead of the stride boundary) -----
        stride_preds_df["horizon"] = (
            (stride_preds_df["ds"] - phys_start).dt.total_seconds() / 3600 + 1
        ).astype(int)

        preds.append(stride_preds_df)
        model_pred_date += pd.Timedelta(hours=forecast_horizon)

    # ── Assemble results ──────────────────────────────────────────────────────
    if not preds:
        logger.warning(
            "No predictions generated for period [%s, %s]", pred_start, pred_end,
        )
        return pd.DataFrame()

    result = pd.concat(preds, ignore_index=True)
    result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    # Trim to the requested physical window
    result = result[
        (result["ds"] >= pred_start) & (result["ds"] <= pred_end)
    ].reset_index(drop=True)
    return result


# ── Post-processing ──────────────────────────────────────────────────────────

def prepare_predictions_df(
    preds: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge raw model predictions with actual *y* values and tidy up columns.

    * Removes prediction-interval columns (``*-lo-*``, ``*-hi-*``).
    * Strips the ``-median`` suffix from quantile-model columns.
    * Clips predictions to ``[0, ∞)`` (physical load cannot be negative).
    """
    if "y" not in preds.columns:
        merged = preds.merge(
            actuals_df[["ds", "unique_id", "y"]],
            on=["ds", "unique_id"],
            how="left",
        )
    else:
        merged = preds.copy()

    # Drop interval columns
    cols_to_keep = [
        c for c in merged.columns if "-lo-" not in c and "-hi-" not in c
    ]
    merged = merged[cols_to_keep]
    merged.columns = merged.columns.str.replace("-median", "", regex=False)

    # Clip to non-negative
    pred_cols = merged.columns.difference(["ds", "unique_id", "y", "horizon", "tso", "dataset"])
    for col in pred_cols:
        merged[col] = np.clip(merged[col], 0, np.inf)

    return merged


# ── W&B logging ──────────────────────────────────────────────────────────────


def evaluate_models(preds_df: pd.DataFrame, models_to_keep_regex: str | None = None):
    reference_loss_pred_models = [
        model for model in list(preds_df.columns.difference(["y", "ds", "unique_id", "horizon"]))
        if models_to_keep_regex is None or re.search(models_to_keep_regex, model)
    ]
    if len(reference_loss_pred_models) == 0:
        return pd.DataFrame()

    reference_loss_evaluation = evaluate(
        df=preds_df, metrics=[rmse, mae],
        models=reference_loss_pred_models,
    )

    r2_scores = pd.DataFrame(np.array([
        r2_score(
            preds_df.loc[preds_df["unique_id"] == direction, "y"], 
            preds_df.loc[preds_df["unique_id"] == direction, model]
        ) 
        for model, direction in product(reference_loss_pred_models,  ["down", "up"])
    ]).reshape(len(reference_loss_pred_models), 2), index=reference_loss_pred_models, columns=["down", "up"]).T.assign(metric="r2_score", unique_id=["down", "up"])
    evaluation_df = pd.concat([reference_loss_evaluation, r2_scores], axis=0).reset_index(drop=True)
    # Add implicit merge key for wandb
    evaluation_df["merge_key"] = evaluation_df["unique_id"].astype(str) + "_" + evaluation_df["metric"].astype(str)
    return evaluation_df


def log_predictions_to_wandb(
    run: "wandb.Run",
    preds_df: pd.DataFrame,
    split_name: str,
    model_alias: str,
    tso: str,
    shift_hours: int,
    timestamp: str,
    log_table: bool = True,
) -> None:
    """
    Log a prediction DataFrame as both an inline ``wandb.Table`` and a
    parquet ``Artifact`` for easy downstream retrieval.

    Parameters
    ----------
    run : wandb.Run
        Active W&B run.
    preds_df : pd.DataFrame
        Prepared predictions (with ``y`` actuals, cleaned columns).
    split_name : str
        ``"test"``, ``"validation"``, or ``"test_windowN"`` (used for naming).
    model_alias : str
        Model alias (e.g. ``nhits_seed778``).
    tso : str
        TSO identifier.
    shift_hours : int
        Shift hours (stored in artifact metadata).
    timestamp : str
        Timestamp string for artifact naming (e.g. run start time). This ensures each artifact gets a unique name, preventing overwrites across runs.
    """
    if preds_df.empty:
        logger.warning(
            "Empty predictions for %s / %s – skipping wandb log",
            split_name,
            model_alias,
        )
        return

    logger.info(
        "Logging %d prediction rows (%s / %s) to wandb",
        len(preds_df),
        split_name,
        model_alias,
    )

    # ── Inline table (visible in the run's Tables tab) ────────────────────────
    if log_table:
        model_performance_df = evaluate_models(preds_df=preds_df)
        table = wandb.Table(dataframe=model_performance_df)
        run.log({f"performance_metrics/{split_name}": table})

    # ── Artifact (parquet, easy to pull programmatically) ─────────────────────
    tso_norm = tso.replace(" ", "_")
    artifact_name = f"preds_{split_name}_{model_alias}_{tso_norm}_{timestamp}"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="predictions",
        metadata={
            "split": split_name,
            "model_alias": model_alias,
            "tso": tso,
            "shift_hours": shift_hours,
            "timestamp": timestamp,
            "n_rows": len(preds_df),
            "n_unique_ids": int(preds_df["unique_id"].nunique()),
            "ds_min": str(preds_df["ds"].min()),
            "ds_max": str(preds_df["ds"].max()),
        },
    )
    with artifact.new_file(f"{split_name}_predictions.parquet", mode="wb") as f:
        preds_df.to_parquet(f)
    run.log_artifact(artifact)
