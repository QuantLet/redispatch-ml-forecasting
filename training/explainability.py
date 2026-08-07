"""
Integrated Gradients (IG) explainability for shifted-target Nixtla models.

Provides feature-importance explanations via IG, handling:
- Correct shift alignment (same as predict_with_shift_correction) to prevent data leakage
- TFT cudnn compatibility (must disable cudnn for IG)
- Both single-window and rolling-window training modes
- Persisting raw IG tensors, predictions, and summary tables to wandb
- Loading saved IG tensors from disk and aggregating across strides/windows
- Computing overall, per-horizon, and clustered feature importance
- Visualizations (bar charts, heatmaps, line plots, dendrograms) and CSV export

Public API - IG computation
----------------------------
- ``explain_with_shift_correction``  – core IG loop (one model / one window)
- ``explain_all_models``             – wrapper that separates TFT from non-TFT
- ``save_explanation_artifacts``     – persist IG tensors + predictions to wandb
- ``aggregate_ig_stats``             – global + per-horizon importance summaries

Public API - Post-hoc analysis (from saved tensors)
-----------------------------------------------------
- ``load_ig_tensors``                – load saved .pt/.npy IG tensors from disk
- ``combine_attributions``           – aggregate raw tensors into per-feature arrays
- ``build_per_horizon_feature_matrix`` – per-horizon attribution array [B, H, F]
- ``compute_sample_level_stats``     – bootstrap statistics for per-horizon arrays
- ``feature_importance_overall``     – overall (whole test period) importance DataFrame
- ``feature_importance_per_horizon`` – per-horizon importance DataFrame
- ``cluster_features_by_horizon``    – hierarchical clustering of feature profiles
- ``plot_overall_importance_bar``    – horizontal bar chart of overall importance
- ``plot_signed_importance_bar``     – stacked pos/neg bar chart
- ``plot_per_horizon_heatmap``       – signed heatmap (features × horizons)
- ``plot_per_horizon_violin``        – violin plot of attribution distribution across horizons
- ``plot_feature_horizon_lines``     – line plot of selected features across horizons
- ``plot_cluster_heatmap``           – cluster-level attribution heatmap
- ``plot_dendrogram``                – standalone feature dendrogram with cluster-coloured leaves
- ``plot_cluster_top_features``      – per-cluster top-feature bar chart (p% threshold or top-f cap)
- ``build_feature_to_feature_set_map`` – map each feature name to its originating feature set
- ``plot_feature_set_bars``          – per-feature-set bar chart (all features, grouped by data source)

Public API - TFT built-in interpretability
------------------------------------------
- ``save_tft_interpretability``              – extract & save VSN weights + attention from a TFT model
- ``aggregate_tft_interpretability_windows`` – average per-window TFT data and save

Public API - Utilities
----------------------
- ``get_stride_dates_from_wandb``    – generate per-stride timestamps from wandb run config
"""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Callable, Literal, Optional

import re

import numpy as np
import pandas as pd
import torch
import wandb

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

from neuralforecast.core import LSTM, NeuralForecast, TFT

from training.data_prep import CALENDAR_COLS, add_calendar_features, build_static_df

logger = logging.getLogger(__name__)

# Fixed colours per canonical feature set name (used by plot_feature_set_bars)
_FEATURE_SET_PALETTE: dict[str, str] = {
    "basic":                  "#7B7B7B",
    "day_ahead_price":        "#2196F3",
    "wind_pv":                "#4CAF50",
    "production_consumption": "#FF9800",
    "cross_border":           "#9C27B0",
    "bloomberg":              "#F44336",
    "sce":                    "#00BCD4",
    "calendar":               "#8BC34A",
    "static_covariates":      "#795548",   # brown – stand-alone static covariates
    "data_driven":            "#607D8B",   # blue-grey – learned / derived covariates
    "runlength":              "#FF5722",   # deep-orange – run-length features
}

# Canonical display order for feature-set panels and CSV rows.
# Sets not in this list are appended at the end in the order they appear.
_CANONICAL_SET_ORDER: list[str] = [
    "basic",
    "day_ahead_price",
    "wind_pv",
    "production_consumption",
    "cross_border",
    "bloomberg",
    "sce",
    "data_driven",
    "runlength",
    "calendar",
    "static_covariates",
]


# ── Core IG loop with shift correction ────────────────────────────────────────

def explain_with_shift_correction(
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
    holidays_path: Optional[str] = None,
    step_hours: int = 24,
) -> tuple[dict[str, list[dict]], pd.DataFrame]:
    """
    Compute Integrated Gradients explanations with correct shift alignment.

    Mirrors the data-preparation logic of ``predict_with_shift_correction`` to
    ensure explanations are computed on exactly the same inputs the model sees
    at prediction time, preventing data leakage.

    Parameters
    ----------
    nf : NeuralForecast
        Fitted NeuralForecast wrapper (one or more models, but **no** TFT –
        use ``explain_all_models`` to handle TFT separately).
    df_shifted : pd.DataFrame
        Full shifted dataset (Nixtla format).
    df_unshifted : pd.DataFrame
        Unshifted dataset with covariate values at their original timestamps.
    static_df : pd.DataFrame
        Static exogenous features (direction one-hot encoding).
    pred_start : pd.Timestamp
        First *physical* timestamp to explain (inclusive).
    pred_end : pd.Timestamp
        Last *physical* timestamp to explain (inclusive).
    future_cov_cols : list[str]
        All future-covariate column names (data + calendar).
    shift_hours : int
        Number of hours by which targets were shifted during training.
    forecast_horizon : int
        Model's forecast horizon (h).
    tso : str
        TSO name (for holiday lookup).
    holidays_path : str | None
        Override for holidays CSV path.
    step_hours : int
        Stride between explanation windows in hours (default: 24 = daily).

    Returns
    -------
    explanations_dict : dict[str, list[dict]]
        ``{model_alias: [ig_result_per_stride, ...]}``.  Each element is the
        raw output returned by ``nf.explain`` for one stride.
    predictions_df : pd.DataFrame
        Predictions generated alongside explanations, with ``ds`` in physical
        time, ``unique_id``, model columns, and ``horizon``.
    """
    n_series = df_shifted["unique_id"].nunique()

    # ── Relevant horizons (match the original explain pattern) ────────────────
    relevant_horizons = list(range(forecast_horizon))[max(0, forecast_horizon - step_hours):]

    # ── Separate data covariates from calendar features ───────────────────────
    data_futr_cols = [c for c in future_cov_cols if c not in CALENDAR_COLS]
    need_calendar = any(c in CALENDAR_COLS for c in future_cov_cols)

    # Verify columns exist
    missing = [c for c in data_futr_cols if c not in df_unshifted.columns]
    if missing:
        logger.warning(
            "Columns %s not found in df_unshifted – dropped from futr_df.", missing,
        )
        data_futr_cols = [c for c in data_futr_cols if c in df_unshifted.columns]

    # ── Model-time boundaries ─────────────────────────────────────────────────
    # First model-time cutoff: back-calculate from pred_start
    model_pred_date = pred_start - pd.Timedelta(hours=shift_hours)
    # Offset so the *relevant* horizons cover the first physical day
    model_pred_date = model_pred_date - pd.Timedelta(hours=max(0, forecast_horizon - step_hours))
    # Last stride: the relevant horizons must still reach pred_end
    last_stride_phys = pred_end - pd.Timedelta(hours=step_hours - 1)

    explanations_dict: dict[str, list[dict]] = {
        model.hparams["alias"]: [] for model in nf.models
    }
    preds_list: list[pd.DataFrame] = []
    stride_idx = 0

    while True:
        phys_start = model_pred_date + pd.Timedelta(hours=shift_hours)
        # The physical timestamp of the first relevant horizon
        phys_relevant_start = phys_start + pd.Timedelta(hours=max(0, forecast_horizon - step_hours))

        if phys_relevant_start > pred_end:
            break

        stride_idx += 1
        logger.info(
            "IG stride %d: model_time=%s  phys=[%s, +%dh)",
            stride_idx, model_pred_date, phys_start, forecast_horizon,
        )

        # 1. Historical data in model time (shifted) --------------------------
        hist_df = df_shifted[df_shifted["ds"] < model_pred_date]
        if hist_df.empty:
            logger.warning(
                "Stride %d: empty history at model_pred_date=%s, skipping",
                stride_idx, model_pred_date,
            )
            model_pred_date += pd.Timedelta(hours=step_hours)
            continue

        # 2. Future covariates at physical timestamps --------------------------
        phys_end = phys_start + pd.Timedelta(hours=forecast_horizon)
        futr_df = df_unshifted.loc[
            (df_unshifted["ds"] >= phys_start) & (df_unshifted["ds"] < phys_end),
            ["unique_id", "ds"] + data_futr_cols,
        ].copy()

        if len(futr_df) < forecast_horizon * n_series:
            logger.warning(
                "Stride %d: incomplete future covariates (%d/%d rows) "
                "for phys=[%s, %s), stopping",
                stride_idx, len(futr_df), forecast_horizon * n_series,
                phys_start, phys_end,
            )
            break

        # 3. Calendar features for physical timestamps -------------------------
        if need_calendar:
            futr_df = add_calendar_features(
                futr_df,
                reference_time=futr_df["ds"],
                tso=tso,
                holidays_path=holidays_path,
            )

        # 4. Relabel ds → model time ------------------------------------------
        if shift_hours > 0:
            futr_df["ds"] = futr_df["ds"] - pd.Timedelta(hours=shift_hours)

        # 5. Explain -----------------------------------------------------------
        preds, explanations = nf.explain(
            horizons=relevant_horizons,
            df=hist_df,
            futr_df=futr_df,
            static_df=static_df,
            explainer="IntegratedGradients",
        )

        # 6. Collect explanations per model ------------------------------------
        for model_label, explanation in explanations.items():
            explanations_dict[model_label].append(explanation)

        # 7. Shift prediction timestamps back to physical time -----------------
        if isinstance(preds, pd.DataFrame):
            stride_preds = preds.copy()
        else:
            # nf.explain may return a polars DataFrame depending on backend
            stride_preds = pd.DataFrame(preds.to_dict())  # type: ignore[union-attr]
        if shift_hours > 0:
            stride_preds["ds"] = stride_preds["ds"] + pd.Timedelta(hours=shift_hours)

        # Keep only the relevant horizons (physical timestamps ≥ phys_relevant_start)
        stride_preds = stride_preds[stride_preds["ds"] >= phys_relevant_start].copy()

        # Add horizon label
        stride_preds["horizon"] = (
            (stride_preds["ds"] - phys_relevant_start).dt.total_seconds() / 3600 + 1
        ).astype(int)

        preds_list.append(stride_preds)

        model_pred_date += pd.Timedelta(hours=step_hours)

    # ── Assemble ──────────────────────────────────────────────────────────────
    if not preds_list:
        logger.warning(
            "No explanations generated for period [%s, %s]", pred_start, pred_end,
        )
        return explanations_dict, pd.DataFrame()

    predictions_df = pd.concat(preds_list, ignore_index=True)
    predictions_df = predictions_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    # Trim to physical window
    predictions_df = predictions_df[
        (predictions_df["ds"] >= pred_start) & (predictions_df["ds"] <= pred_end)
    ].reset_index(drop=True)

    n_strides = {alias: len(ig_list) for alias, ig_list in explanations_dict.items()}
    logger.info("IG finished: %d strides, models=%s", stride_idx, n_strides)

    return explanations_dict, predictions_df


# ── Handle TFT separately (cudnn disabled) ────────────────────────────────────

def explain_all_models(
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
    holidays_path: Optional[str] = None,
    step_hours: int = 24,
) -> tuple[dict[str, list[dict]], pd.DataFrame]:
    """
    Compute IG explanations for all models, handling TFT/LSTM's cudnn requirement.

    TFT and LSTM models must run with ``torch.backends.cudnn.enabled = False`` to avoid
    errors in Integrated Gradients (double-backward through cuDNN RNNs).
    This function splits TFT/LSTM from non-TFT/LSTM models and handles them separately.
    Parameters
    ----------
    Same as ``explain_with_shift_correction``, but ``nf`` may contain TFT models.

    Returns
    -------
    Same as ``explain_with_shift_correction``.
    """
    has_tft = any(isinstance(m, TFT) or isinstance(m, LSTM) for m in nf.models)
    has_non_tft = any(not isinstance(m, TFT) and not isinstance(m, LSTM) for m in nf.models)

    all_explanations: dict[str, list[dict]] = {}
    all_preds: pd.DataFrame | None = None

    common_kwargs = dict(
        df_shifted=df_shifted,
        df_unshifted=df_unshifted,
        static_df=static_df,
        pred_start=pred_start,
        pred_end=pred_end,
        future_cov_cols=future_cov_cols,
        shift_hours=shift_hours,
        forecast_horizon=forecast_horizon,
        tso=tso,
        holidays_path=holidays_path,
        step_hours=step_hours,
    )

    # ── Non-TFT models ───────────────────────────────────────────────────────
    if has_non_tft:
        nf_non_tft = deepcopy(nf)
        nf_non_tft.models = [m for m in nf_non_tft.models if not isinstance(m, TFT) and not isinstance(m, LSTM)]
        logger.info(
            "Running IG for non-TFT models: %s",
            [m.hparams["alias"] for m in nf_non_tft.models],
        )
        explanations, preds = explain_with_shift_correction(nf=nf_non_tft, **common_kwargs)  # type: ignore[arg-type]
        all_explanations.update(explanations)
        all_preds = preds

    # ── TFT models (cudnn disabled) ──────────────────────────────────────────
    if has_tft:
        nf_tft = deepcopy(nf)
        nf_tft.models = [m for m in nf_tft.models if isinstance(m, TFT) or isinstance(m, LSTM)]
        logger.info(
            "Running IG for TFT/LSTM models (cudnn disabled): %s",
            [m.hparams["alias"] for m in nf_tft.models],
        )
        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            explanations_tft, preds_tft = explain_with_shift_correction(
                nf=nf_tft, **common_kwargs  # type: ignore[arg-type]
            )
        finally:
            torch.backends.cudnn.enabled = prev_cudnn

        all_explanations.update(explanations_tft)

        if all_preds is not None and not preds_tft.empty:
            # Merge TFT predictions with non-TFT predictions on shared keys
            merge_cols = ["unique_id", "ds"]
            extra_cols = [c for c in preds_tft.columns if c not in all_preds.columns]
            if extra_cols:
                all_preds = all_preds.merge(
                    preds_tft[merge_cols + extra_cols],
                    on=merge_cols,
                    how="left",
                )
        elif all_preds is None:
            all_preds = preds_tft

    if all_preds is None:
        all_preds = pd.DataFrame()

    return all_explanations, all_preds


# ── Artifact persistence ─────────────────────────────────────────────────────

def save_explanation_artifacts(
    run: wandb.Run,
    explanations: dict[str, list[dict]],
    predictions: pd.DataFrame,
    model_alias: str,
    tso: str,
    timestamp: str,
    shift_hours: int,
    window_index: int | None = None,
    output_dir: Path | None = None,
    *,
    artifact_type: str = "ig_explanation",
    preds_artifact_type: str = "ig_predictions",
    raw_artifact_name_prefix: str = "ig_raw",
    preds_artifact_name_prefix: str = "ig_preds",
) -> None:
    """
    Persist raw IG tensors and predictions to wandb artifacts.

    Naming convention:
    - IG raw:       ``{raw_artifact_name_prefix}_{model_alias}_{tso}_{timestamp}[_window{i}]``
    - Predictions:  ``{preds_artifact_name_prefix}_{model_alias}_{tso}_{timestamp}[_window{i}]``

    Parameters
    ----------
    run : wandb.Run
        Active W&B run.
    explanations : dict[str, list[dict]]
        Raw IG outputs keyed by model alias → list of per-stride results.
    predictions : pd.DataFrame
        Predictions generated during IG.
    model_alias : str
        Model alias string (e.g. ``nhits_seed778``).
    tso : str
        TSO identifier.
    timestamp : str
        Timestamp string for unique naming.
    shift_hours : int
        Shift hours (stored in metadata).
    window_index : int | None
        Window index for rolling-window training (``None`` for single window).
    output_dir : Path | None
        Optional local directory to also save outputs.
    artifact_type : str
        W&B artifact type for raw IG stride artifacts (default ``"ig_explanation"``).
        Pass ``"ig_dev_explanation"`` for the pre-evaluation dev period.
    preds_artifact_type : str
        W&B artifact type for the predictions artifact (default ``"ig_predictions"``).
        Pass ``"ig_dev_predictions"`` for the pre-evaluation dev period.
    raw_artifact_name_prefix : str
        Prefix for raw IG artifact names (default ``"ig_raw"``).
        Pass ``"ig_dev_raw"`` for the dev period.
    preds_artifact_name_prefix : str
        Prefix for prediction artifact names (default ``"ig_preds"``).
        Pass ``"ig_dev_preds"`` for the dev period.
    """
    tso_norm = tso.replace(" ", "_")
    window_suffix = f"_window{window_index}" if window_index is not None else ""

    # ── Save raw IG tensors ──────────────────────────────────────────────────
    for model_label, ig_list in explanations.items():
        if not ig_list:
            logger.warning("No IG results for model %s, skipping artifact.", model_label)
            continue

        artifact_name = f"{raw_artifact_name_prefix}_{model_label}_{tso_norm}_{timestamp}{window_suffix}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata={
                "model_alias": model_label,
                "tso": tso,
                "shift_hours": shift_hours,
                "timestamp": timestamp,
                "window_index": window_index,
                "n_strides": len(ig_list),
            },
        )

        # Serialize each stride's IG result
        for stride_idx, ig_result in enumerate(ig_list):
            _save_ig_stride_to_artifact(artifact, ig_result, stride_idx)

        run.log_artifact(artifact)
        logger.info("Logged IG artifact: %s (%d strides)", artifact_name, len(ig_list))

        # Also save locally if output_dir is provided
        if output_dir is not None:
            local_dir = output_dir / raw_artifact_name_prefix / f"{model_label}{window_suffix}"
            local_dir.mkdir(parents=True, exist_ok=True)
            for stride_idx, ig_result in enumerate(ig_list):
                _save_ig_stride_locally(local_dir, ig_result, stride_idx)
            logger.info("Saved IG tensors locally to %s", local_dir)

    # ── Save predictions ─────────────────────────────────────────────────────
    if not predictions.empty:
        preds_artifact_name = f"{preds_artifact_name_prefix}_{model_alias}_{tso_norm}_{timestamp}{window_suffix}"
        preds_artifact = wandb.Artifact(
            name=preds_artifact_name,
            type=preds_artifact_type,
            metadata={
                "model_alias": model_alias,
                "tso": tso,
                "shift_hours": shift_hours,
                "timestamp": timestamp,
                "window_index": window_index,
                "n_rows": len(predictions),
                "ds_min": str(predictions["ds"].min()),
                "ds_max": str(predictions["ds"].max()),
            },
        )
        with preds_artifact.new_file("ig_predictions.parquet", mode="wb") as f:
            predictions.to_parquet(f)
        run.log_artifact(preds_artifact)
        logger.info("Logged IG predictions artifact: %s", preds_artifact_name)

        if output_dir is not None:
            local_preds_dir = output_dir / preds_artifact_name_prefix
            local_preds_dir.mkdir(parents=True, exist_ok=True)
            predictions.to_parquet(
                local_preds_dir / f"{preds_artifact_name_prefix}_{model_alias}{window_suffix}.parquet"
            )


def _save_ig_stride_to_artifact(
    artifact: wandb.Artifact,
    ig_result: dict,
    stride_idx: int,
) -> None:
    """Serialize one stride's IG output into a wandb artifact."""
    for key, value in ig_result.items():
        if isinstance(value, torch.Tensor):
            filename = f"stride_{stride_idx:04d}_{key}.pt"
            with artifact.new_file(filename, mode="wb") as f:
                torch.save(value.cpu(), f)
        elif isinstance(value, np.ndarray):
            filename = f"stride_{stride_idx:04d}_{key}.npy"
            with artifact.new_file(filename, mode="wb") as f:
                np.save(f, value)


def _save_ig_stride_locally(
    local_dir: Path,
    ig_result: dict,
    stride_idx: int,
) -> None:
    """Save one stride's IG tensors to local directory."""
    for key, value in ig_result.items():
        if isinstance(value, torch.Tensor):
            torch.save(value.cpu(), local_dir / f"stride_{stride_idx:04d}_{key}.pt")
        elif isinstance(value, np.ndarray):
            np.save(local_dir / f"stride_{stride_idx:04d}_{key}.npy", value)


# ── TFT built-in interpretability ────────────────────────────────────────────

def save_tft_interpretability(
    nf: NeuralForecast,
    output_dir: Path,
    run: "wandb.Run | None",
    model_alias: str,
    window_index: int | None = None,
    tso: str = "",
    timestamp: str = "",
) -> dict | None:
    """
    Extract and persist TFT's built-in interpretability artefacts.

    Saves variable-selection-network (VSN) feature importances and
    batch-averaged attention weights to disk and (optionally) to W&B.

    Should be called **after** ``nf.fit()`` and at least one
    ``nf.predict()`` / ``nf.explain()`` call so that
    ``interpretability_params`` are populated with test-period data.

    Parameters
    ----------
    nf : NeuralForecast
        Fitted NeuralForecast wrapper that must contain at least one TFT model.
    output_dir : Path
        Root output directory; artefacts are written to
        ``<output_dir>/tft_interpretability/<model_alias>[_windowN]/``.
    run : wandb.Run | None
        Active W&B run.  Pass ``None`` to skip W&B logging.
    model_alias : str
        Model alias string (e.g. ``tft_seed778``).
    window_index : int | None
        Window index for rolling-window training (``None`` for single window).
    tso : str
        TSO identifier (used for W&B artifact naming).
    timestamp : str
        Timestamp string for unique W&B artifact naming.

    Returns
    -------
    dict | None
        ``{"feature_importances": dict[str, pd.DataFrame],
           "attention_weights": np.ndarray}``
        for downstream aggregation, or ``None`` if no TFT model is found or
        extraction fails.
    """
    tft_models = [m for m in nf.models if isinstance(m, TFT)]
    if not tft_models:
        return None
    tft_model = tft_models[0]

    try:
        fi_dict: dict[str, pd.DataFrame] = tft_model.feature_importances()
        attn = tft_model.attention_weights()
    except Exception as exc:
        logger.warning(
            "Failed to extract TFT interpretability for %s: %s", model_alias, exc
        )
        return None

    if isinstance(attn, torch.Tensor):
        attn_np: np.ndarray = attn.cpu().numpy()
    else:
        attn_np = np.asarray(attn)

    window_suffix = f"_window{window_index}" if window_index is not None else ""
    out_base = output_dir / "tft_interpretability" / f"{model_alias}{window_suffix}"
    out_base.mkdir(parents=True, exist_ok=True)

    # Save feature importance CSVs
    for key, df_fi in fi_dict.items():
        safe_key = re.sub(r"[^\w]+", "_", key).strip("_").lower()
        df_fi.to_csv(out_base / f"{safe_key}.csv")

    # Save attention weights
    np.save(out_base / "attention_weights.npy", attn_np)
    logger.info("Saved TFT interpretability to %s", out_base)

    # Log to W&B
    if run is not None:
        tso_norm = tso.replace(" ", "_")
        artifact_name = (
            f"tft_interp_{model_alias}_{tso_norm}_{timestamp}{window_suffix}"
        )
        artifact = wandb.Artifact(
            name=artifact_name,
            type="tft_interpretability",
            metadata={
                "model_alias": model_alias,
                "window_index": window_index,
                "tso": tso,
                "timestamp": timestamp,
            },
        )
        artifact.add_dir(str(out_base))
        run.log_artifact(artifact)
        logger.info("Logged TFT interpretability artifact: %s", artifact_name)

        for key, df_fi in fi_dict.items():
            safe_key = re.sub(r"[^\w]+", "_", key).strip("_").lower()
            df_log = df_fi.reset_index() if not isinstance(df_fi.index, pd.RangeIndex) else df_fi.copy()
            try:
                run.log({f"tft_interp/{model_alias}/{safe_key}": wandb.Table(dataframe=df_log)})
            except Exception as exc:
                logger.debug(
                    "Could not log TFT feature importances table '%s': %s", key, exc
                )

    return {"feature_importances": fi_dict, "attention_weights": attn_np}


def aggregate_tft_interpretability_windows(
    per_window_data: list[dict],
    output_dir: Path,
    model_alias: str,
    tso: str = "",
    timestamp: str = "",
) -> None:
    """
    Average TFT interpretability data across rolling-window runs and save.

    Reads the list of dicts returned by :func:`save_tft_interpretability`
    (one per window) and writes:

    - One CSV per feature-importance key (numeric columns averaged).
    - ``attention_weights_avg.npy`` – mean attention array.

    No W&B logging is performed here because all per-window W&B runs have
    already been finished.

    Parameters
    ----------
    per_window_data : list[dict]
        Each dict has keys ``"feature_importances"`` and ``"attention_weights"``
        as returned by :func:`save_tft_interpretability`.
    output_dir : Path
        Root output directory; averaged artefacts are written to
        ``<output_dir>/tft_interpretability/<model_alias>_avg_over_windows/``.
    model_alias : str
        Model alias string.
    tso : str
        TSO identifier (informational only).
    timestamp : str
        Timestamp string (informational only).
    """
    if not per_window_data:
        return

    out_base = (
        output_dir / "tft_interpretability" / f"{model_alias}_avg_over_windows"
    )
    out_base.mkdir(parents=True, exist_ok=True)

    # Average feature importance DataFrames
    first_fi = per_window_data[0]["feature_importances"]
    for key in first_fi.keys():
        dfs = [
            wd["feature_importances"][key]
            for wd in per_window_data
            if key in wd["feature_importances"]
        ]
        if not dfs:
            continue
        try:
            num_cols = dfs[0].select_dtypes(include=[np.number]).columns.tolist()
            non_num = dfs[0].drop(columns=num_cols, errors="ignore")
            avg_num = pd.DataFrame(
                np.mean(
                    np.stack(
                        [df[num_cols].to_numpy() for df in dfs], axis=0
                    ),
                    axis=0,
                ),
                columns=num_cols,
                index=dfs[0].index,
            )
            avg_df = pd.concat([non_num, avg_num], axis=1)
        except Exception as exc:
            logger.warning(
                "Could not average TFT feature importances for key '%s': %s", key, exc
            )
            avg_df = dfs[0]
        safe_key = re.sub(r"[^\w]+", "_", key).strip("_").lower()
        avg_df.to_csv(out_base / f"{safe_key}_avg.csv")

    # Average attention weights
    attn_arrays = [
        wd["attention_weights"]
        for wd in per_window_data
        if wd.get("attention_weights") is not None
    ]
    if attn_arrays:
        try:
            avg_attn = np.mean(np.stack(attn_arrays, axis=0), axis=0)
            np.save(out_base / "attention_weights_avg.npy", avg_attn)
        except Exception as exc:
            logger.warning(
                "Failed to average attention weights across windows: %s", exc
            )

    logger.info(
        "Saved averaged TFT interpretability over %d windows to %s",
        len(per_window_data), out_base,
    )


# ── Summary statistics ────────────────────────────────────────────────────────

def aggregate_ig_stats(
    explanations: dict[str, list[dict]],
    feature_names: list[str],
    feature_type: str = "hist_exog",
    scaler_stats: dict[str, np.ndarray] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute global and per-horizon importance summaries from raw IG tensors.

    Parameters
    ----------
    explanations : dict[str, list[dict]]
        Raw IG outputs keyed by model alias → list of per-stride dicts.
        Each dict is expected to have keys like ``'hist_exog'``, ``'futr_exog'``,
        etc., mapping to tensors.
    feature_names : list[str]
        Ordered feature names matching the last dimension of the IG tensor.
    feature_type : str
        Which IG component to summarise (``'hist_exog'`` or ``'futr_exog'``).
    scaler_stats : dict | None
        Optional ``{feature_name: np.array([mean, std])}`` for converting IG
        from scaled to original units.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{model_alias: summary_df}`` with columns:
        ``feature``, ``mean_abs_ig_scaled``, ``median_abs_ig_scaled``,
        and optionally ``mean_abs_ig_original``.
    """
    summaries: dict[str, pd.DataFrame] = {}

    for model_alias, ig_list in explanations.items():
        if not ig_list:
            continue

        # Collect all IG tensors for this feature type
        all_ig: list[np.ndarray] = []
        for ig_result in ig_list:
            tensor = ig_result.get(feature_type)
            if tensor is None:
                continue
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().numpy()
            all_ig.append(tensor)

        if not all_ig:
            logger.warning(
                "No '%s' IG tensors found for model %s", feature_type, model_alias,
            )
            continue

        # Stack: shape depends on neuralforecast version, but last dim = n_features
        # Sum over input_size / horizon dims to get per-feature importance
        # Typical shape: (n_series, n_horizons, input_size, n_features) or similar
        feature_importance_per_stride: list[np.ndarray] = []
        for ig_np in all_ig:
            # Sum absolute IG over all dims except the last (features)
            axes_to_sum = tuple(range(ig_np.ndim - 1))
            abs_ig = np.abs(ig_np).sum(axis=axes_to_sum)
            feature_importance_per_stride.append(abs_ig)

        # Average across strides
        stacked = np.stack(feature_importance_per_stride)  # (n_strides, n_features)
        mean_abs = stacked.mean(axis=0)
        median_abs = np.median(stacked, axis=0)

        n_features = mean_abs.shape[0]
        if n_features != len(feature_names):
            logger.warning(
                "Feature count mismatch for %s: IG has %d, names has %d. "
                "Truncating to min.",
                model_alias, n_features, len(feature_names),
            )
            n = min(n_features, len(feature_names))
            mean_abs = mean_abs[:n]
            median_abs = median_abs[:n]
            names = feature_names[:n]
        else:
            names = feature_names

        summary = pd.DataFrame({
            "feature": names,
            "mean_abs_ig_scaled": mean_abs,
            "median_abs_ig_scaled": median_abs,
        })

        # Convert to original units if scaler stats available
        if scaler_stats is not None:
            original_scale = []
            for feat in names:
                stats = scaler_stats.get(feat)
                if stats is not None:
                    std = stats[1] if len(stats) > 1 else 1.0
                    original_scale.append(std)
                else:
                    original_scale.append(1.0)
            original_scale_arr = np.array(original_scale)
            summary["mean_abs_ig_original"] = mean_abs * original_scale_arr
            summary["median_abs_ig_original"] = median_abs * original_scale_arr

        # Sort by mean absolute IG (descending)
        summary = summary.sort_values("mean_abs_ig_scaled", ascending=False).reset_index(drop=True)
        summaries[model_alias] = summary

    return summaries


def log_ig_summary_tables(
    run: wandb.Run,
    summaries: dict[str, pd.DataFrame],
    prefix: str = "ig_summary",
) -> None:
    """Log IG summary DataFrames as wandb Tables for visualization."""
    for model_alias, summary_df in summaries.items():
        table = wandb.Table(dataframe=summary_df)
        run.log({f"{prefix}/{model_alias}": table})
        logger.info("Logged IG summary table: %s/%s", prefix, model_alias)


# ══════════════════════════════════════════════════════════════════════════════
#  POST-HOC ANALYSIS - Load saved tensors & aggregate
# ══════════════════════════════════════════════════════════════════════════════


# ── Loading saved IG tensors from disk ────────────────────────────────────────

def load_ig_tensors(ig_dir: Path) -> list[dict[str, torch.Tensor]]:
    """
    Load raw IG tensors produced by ``save_explanation_artifacts`` from a
    single model directory (e.g. ``ig_raw/nhits_seed778_window0``).

    The directory is expected to contain files like::

        stride_0000_insample.pt
        stride_0000_futr_exog.pt
        stride_0000_hist_exog.pt
        stride_0000_stat_exog.pt
        stride_0000_baseline_predictions.pt
        stride_0001_insample.pt
        ...

    Returns
    -------
    list[dict[str, torch.Tensor]]
        One dict per stride, keyed by tensor name (``insample``, ``futr_exog``,
        ``hist_exog``, ``stat_exog``, ``baseline_predictions``).
    """
    ig_dir = Path(ig_dir)
    if not ig_dir.is_dir():
        raise FileNotFoundError(f"IG tensor directory not found: {ig_dir}")

    # Discover strides
    all_files = sorted(ig_dir.glob("stride_*"))
    stride_indices: set[int] = set()
    for f in all_files:
        parts = f.stem.split("_", 2)  # stride, NNNN, key_name
        if len(parts) >= 2:
            try:
                stride_indices.add(int(parts[1]))
            except ValueError:
                pass

    strides_sorted = sorted(stride_indices)
    if not strides_sorted:
        logger.warning("No stride files found in %s", ig_dir)
        return []

    result: list[dict[str, torch.Tensor]] = []
    for si in strides_sorted:
        stride_dict: dict[str, torch.Tensor] = {}
        prefix = f"stride_{si:04d}_"
        for f in ig_dir.glob(f"{prefix}*"):
            key_name = f.stem[len(prefix):]
            if f.suffix == ".pt":
                stride_dict[key_name] = torch.load(f, map_location="cpu", weights_only=True)
            elif f.suffix == ".npy":
                stride_dict[key_name] = torch.from_numpy(np.load(f))
        if stride_dict:
            result.append(stride_dict)

    logger.info("Loaded %d strides from %s", len(result), ig_dir)
    return result


def discover_ig_directories(
    output_dir: Path,
    model_filter: str | None = None,
    start_window: int = 0,
    best_checkpoint: bool = False,
) -> dict[str, list[Path]]:
    """
    Discover all ``window_N/evaluation/ig_raw/<model_alias>`` directories under *output_dir*.

    Returns
    -------
    dict[str, list[Path]]
        ``{model_alias_without_window: [dir_window0, dir_window1, ...]}``.
        For single-window training the list has exactly one element.
    """
    candidate_dirs: list[tuple[str, Path]] = []
    ig_dir_name = "ig_raw_best_checkpoint" if best_checkpoint else "ig_raw"
    ig_root = output_dir / ig_dir_name
    if not ig_root.is_dir():
        # Detect rolling window directories
        for d in output_dir.iterdir():
            if d.is_dir() and d.name.startswith("window_"):
                ig_root_candidate = d / "evaluation" / ig_dir_name
                if ig_root_candidate.is_dir():
                    ig_root = ig_root_candidate
                    candidate_dirs.append((d.name, ig_root_candidate))
        if not candidate_dirs:
            raise FileNotFoundError(f"{ig_dir_name} directory not found under {output_dir}")
    else:
        candidate_dirs.append((ig_root.name, ig_root))
    result: dict[str, list[Path]] = {}
    for dir_name, d in candidate_dirs:
        # If start_window is set, only include windows with index >= start_window
        if "window_" in dir_name:
            window_index = int(dir_name.split("window_")[1])
            if window_index < start_window:
                logger.info("Skipping %s due to window index %d < start_window %d", dir_name, window_index, start_window)
                continue
        for model_dir in d.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                # Strip window suffix from model name if present
                model_name = model_name.rsplit("_window")[0] if "_window" in model_name else model_name
                if model_filter and model_filter not in model_name:
                    continue
                result.setdefault(model_name, []).append(model_dir)

    return result


def get_window_index_from_ig_dir(ig_dir: Path) -> int | None:
    """Extract window index from directory name if any of its parents follow the pattern 'window_N'."""
    current_parrent = ig_dir
    while current_parrent != current_parrent.parent:  # until we reach the root
        if current_parrent.name.startswith("window_"):
            try:
                return int(current_parrent.name.split("window_")[1])
            except ValueError:
                pass
        current_parrent = current_parrent.parent
    return None


# ── Attribution aggregation (window-scaled = no rescale) ──────────────────────

def _squeeze_if_single(x: np.ndarray) -> np.ndarray:
    """Squeeze singleton axes at positions 2 and 3 (series, output dims)."""
    squeeze_axes = [i for i in (2, 3) if i < x.ndim and x.shape[i] == 1]
    return np.squeeze(x, axis=tuple(squeeze_axes)) if squeeze_axes else x


def combine_attributions(
    explanations: dict[str, torch.Tensor],
    nf_model,
    static_df: pd.DataFrame,
    mode: Literal["absolute", "signed"] = "absolute",
    futr_agg: Literal["hist", "futr", "combined"] = "combined",
    output_idx: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Aggregate raw IG tensors into a per-feature importance array.

    This is the **window-scaled** (no rescaling) aggregator.  It works correctly
    for both ``scaler_type`` (window scaling) and ``local_scaler_type`` when you
    only care about relative importance, as advertised by Nixtla.

    Future covariates *before* the forecast horizon are treated as
    historical (they are known at prediction time and concatenated with the
    input window by Nixtla internally). The ``futr_agg`` parameter selects
    which part to keep:

    - ``"combined"`` (default) - sum historical + future tokens
    - ``"futr"`` - only future-horizon tokens (recommended by Nixtla tutorial)
    - ``"hist"`` - only historical tokens of future covariates

    Parameters
    ----------
    explanations : dict
        One stride's IG output (keys: insample, hist_exog, futr_exog, stat_exog,
        baseline_predictions).
    nf_model : neuralforecast model object
        The NeuralForecast model (e.g. ``nf.models[0]``).
    static_df : pd.DataFrame
        Static exogenous DataFrame.
    mode : "absolute" or "signed"
        - ``"absolute"``: take ``|attribution|`` before aggregation (magnitude).
        - ``"signed"``: keep sign (directional contribution).
    futr_agg : "hist", "futr", or "combined"
        How to aggregate historical vs future tokens of future covariates.
    output_idx : int
        Which output head to explain (0 = median/point forecast).

    Returns
    -------
    concatenated_final_values : np.ndarray, shape [n_features, B]
        Feature-only attribution matrix. Baseline predictions are intentionally
        not appended.
    feature_names : list[str]
        Feature names (length ``n_features``).
    """
    past_covariates = nf_model.hparams.get("hist_exog_list", []) or []
    future_covariates = nf_model.hparams.get("futr_exog_list", []) or []

    abs_fn = np.abs if mode == "absolute" else lambda x: x

    attributions_per_batch: list[list[float]] = [[], []]  # assume B=2
    feature_names: list[str] = []

    # 1) insample (y lags)
    # Use abs_fn + sum over H to match hist/futr aggregation exactly.
    # The previous .mean(axis=1) without abs_fn created an H-fold scale
    # deflation: hist/futr features were summed (×H) while y_lags were
    # averaged (÷H), making y_lags appear up to H² times less important.
    y_attr = explanations["insample"][:, :, 0, output_idx, :, 0]
    mask_attr = explanations["insample"][:, :, 0, output_idx, :, 1]
    combined_insample = (y_attr + mask_attr).cpu().numpy()  # [B, H, input_size]
    combined_insample_per_batch = abs_fn(combined_insample).sum(axis=1).T  # [input_size, B]
    for i, attr in enumerate(combined_insample_per_batch):
        for b in range(len(attributions_per_batch)):
            attributions_per_batch[b].append(float(attr[b]))
        feature_names.append(f"y_lag{i + 1}")

    # 2) historical exogenous
    hist_attr = explanations.get("hist_exog")
    if hist_attr is not None:
        hist_np = hist_attr.cpu().numpy()
        pfh_hist = hist_np.sum(axis=-2)  # sum tokens
        pfh_hist = _squeeze_if_single(pfh_hist)  # -> [B, H, n_hist_features]
        agg = abs_fn(pfh_hist).sum(axis=1) if mode == "absolute" else pfh_hist.sum(axis=1)
        agg_T = agg.T  # [n_hist_features, B]
        for t in range(agg_T.shape[0]):
            for b in range(len(attributions_per_batch)):
                attributions_per_batch[b].append(float(agg_T[t, b]))
            feature_names.append(f"hist_exog_{past_covariates[t]}" if t < len(past_covariates) else f"hist_exog_{t}")

    # 3) future exogenous - split tokens
    futr_attr = explanations.get("futr_exog")
    if futr_attr is not None:
        arr = futr_attr.cpu().numpy()
        input_size = int(nf_model.hparams["input_size"])
        hist_tokens = arr[..., :input_size, :]
        futr_tokens = arr[..., input_size:, :]
        pfh_hist_part = hist_tokens.sum(axis=-2)
        pfh_futr_part = futr_tokens.sum(axis=-2)
        pfh_combined = pfh_hist_part + pfh_futr_part

        pfh_hist_part = _squeeze_if_single(pfh_hist_part)
        pfh_futr_part = _squeeze_if_single(pfh_futr_part)
        pfh_combined = _squeeze_if_single(pfh_combined)

        if futr_agg == "hist":
            chosen = pfh_hist_part
        elif futr_agg == "futr":
            chosen = pfh_futr_part
        else:
            chosen = pfh_combined

        agg = abs_fn(chosen).sum(axis=1) if mode == "absolute" else chosen.sum(axis=1)
        agg_T = agg.T
        for t in range(agg_T.shape[0]):
            for b in range(len(attributions_per_batch)):
                attributions_per_batch[b].append(float(agg_T[t, b]))
            feature_names.append(
                f"futr_exog_{future_covariates[t]}" if t < len(future_covariates) else f"futr_exog_{t}"
            )

    # 4) static exogenous
    stat_attr = explanations.get("stat_exog")
    if stat_attr is not None:
        stat_np = stat_attr.cpu().numpy().squeeze(axis=(2, 3))  # [B, H, n_stat]
        stat_per_batch = abs_fn(stat_np).sum(axis=1).T  # [n_stat, B]
        stat_cols = [c for c in static_df.columns if c != "unique_id"]
        for t in range(stat_per_batch.shape[0]):
            for b in range(len(attributions_per_batch)):
                attributions_per_batch[b].append(float(stat_per_batch[t, b]))
            feature_names.append(
                f"stat_exog_{stat_cols[t]}" if t < len(stat_cols) else f"stat_exog_{t}"
            )

    # Stack batches
    B = len(attributions_per_batch)
    batch_arrays = [np.array(attributions_per_batch[b]) for b in range(B)]
    concatenated_batch_values = np.stack(batch_arrays, axis=-1)  # [n_features, B]

    return concatenated_batch_values, feature_names


def combine_attributions_rescaled(
    explanations: dict[str, torch.Tensor],
    nf: NeuralForecast,
    model_idx: int,
    static_df: pd.DataFrame,
    futr_agg: Literal["hist", "futr", "combined"] = "combined",
    output_idx: int = 0,
    min_std: float = 1e-3,
    pooled_std: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Signed aggregator that rescales attributions by per-feature std.

    Rescaling multiplies each attribution by the feature's training-set
    standard deviation, converting from scaled space to original units.
    This is relevant when using ``local_scaler_type`` (standard scaling) and
    you want to express contributions in e.g. MW.

    Parameters are the same as ``combine_attributions`` plus:

    min_std : float
        Floor for feature stds to avoid division-by-near-zero artefacts.
    pooled_std : bool
        If True, use the mean std across series instead of per-series std.

    Returns same shape as ``combine_attributions``.
    """
    nf_model = nf.models[model_idx]
    past_covariates = nf_model.hparams.get("hist_exog_list", []) or []
    future_covariates = nf_model.hparams.get("futr_exog_list", []) or []

    attributions_per_batch: list[list[float]] = [[], []]
    feature_names: list[str] = []

    # 1) insample - rescale by y std
    # Use .sum(axis=1) to match hist/futr (which also sums over H via
    # _rescale_and_aggregate → rescaled.sum(axis=1)). The previous .mean
    # caused an H-fold scale mismatch.
    y_attr = explanations["insample"][:, :, 0, output_idx, :, 0]
    mask_attr = explanations["insample"][:, :, 0, output_idx, :, 1]
    combined_insample = (y_attr + mask_attr).cpu().numpy()
    y_std = nf.scalers_["y"].stats_[:, 1].reshape(-1, 1, 1)
    combined_insample = combined_insample * y_std
    combined_insample_per_batch = combined_insample.sum(axis=1).T
    for i, attr in enumerate(combined_insample_per_batch):
        for b in range(len(attributions_per_batch)):
            attributions_per_batch[b].append(float(attr[b]))
        feature_names.append(f"y_lag{i + 1}")

    # helper for safe scales
    def _get_safe_scales(feature_names_list: list[str]) -> np.ndarray:
        scalers = []
        for col in feature_names_list:
            scaler = nf.scalers_.get(col)
            if scaler is None or getattr(scaler, "stats_", None) is None:
                raise KeyError(f"Scaler.stats_ missing for feature '{col}'")
            stds = scaler.stats_[:, 1].astype(float).copy()
            if pooled_std:
                stds[:] = float(np.nanmean(stds))
            stds = np.where(stds < min_std, min_std, stds)
            scalers.append(stds)
        return np.stack(scalers, axis=0)  # (n_features, n_series)

    def _rescale_and_aggregate(attr_tensor, feature_list, sum_tokens=True):
        arr = attr_tensor.cpu().numpy()
        n_features = arr.shape[-1]
        used_names = feature_list[:n_features]
        scales = _get_safe_scales(used_names)
        n_series = scales.shape[1]
        scales_T = scales.T
        target_shape = [1] * arr.ndim
        target_shape[0] = n_series
        target_shape[-1] = n_features
        scales_expanded = scales_T.reshape(tuple(target_shape))
        rescaled = arr * scales_expanded
        if sum_tokens:
            rescaled = rescaled.sum(axis=-2)
        rescaled = _squeeze_if_single(rescaled)
        return rescaled.sum(axis=1)  # signed sum over horizon -> [B, F]

    # 2) historical exogenous
    hist_attr = explanations.get("hist_exog")
    if hist_attr is not None:
        agg = _rescale_and_aggregate(hist_attr, past_covariates)
        agg_T = agg.T
        for t in range(agg_T.shape[0]):
            for b in range(len(attributions_per_batch)):
                attributions_per_batch[b].append(float(agg_T[t, b]))
            feature_names.append(f"hist_exog_{past_covariates[t]}" if t < len(past_covariates) else f"hist_exog_{t}")

    # 3) future exogenous - split & rescale
    futr_attr = explanations.get("futr_exog")
    if futr_attr is not None:
        arr = futr_attr.cpu().numpy()
        input_size = int(nf_model.hparams["input_size"])
        n_futr_features = arr.shape[-1]
        used_futr = future_covariates[:n_futr_features]
        scales = _get_safe_scales(used_futr)
        n_series = scales.shape[1]
        scales_T = scales.T
        target_shape = [1] * arr.ndim
        target_shape[0] = n_series
        target_shape[-1] = n_futr_features
        scales_expanded = scales_T.reshape(tuple(target_shape))

        hist_tokens = arr[..., :input_size, :] * scales_expanded
        futr_tokens = arr[..., input_size:, :] * scales_expanded
        pfh_hist = _squeeze_if_single(hist_tokens.sum(axis=-2))
        pfh_futr = _squeeze_if_single(futr_tokens.sum(axis=-2))
        pfh_combined = pfh_hist + pfh_futr

        chosen = {"hist": pfh_hist, "futr": pfh_futr, "combined": pfh_combined}[futr_agg]
        agg = chosen.sum(axis=1).T
        for t in range(agg.shape[0]):
            for b in range(len(attributions_per_batch)):
                attributions_per_batch[b].append(float(agg[t, b]))
            feature_names.append(
                f"futr_exog_{future_covariates[t]}" if t < len(future_covariates) else f"futr_exog_{t}"
            )

    # 4) static exogenous - rescale if scaler available
    stat_attr = explanations.get("stat_exog")
    if stat_attr is not None:
        stat_np = stat_attr.cpu().numpy().squeeze(axis=(2, 3))
        stat_cols = [c for c in static_df.columns if c != "unique_id"]
        stat_rescaled = np.empty_like(stat_np)
        for i, col in enumerate(stat_cols[:stat_np.shape[-1]]):
            scaler = nf.scalers_.get(col)
            if scaler is not None and getattr(scaler, "stats_", None) is not None:
                stds = scaler.stats_[:, 1].reshape(-1, 1)
                stat_rescaled[:, :, i] = stat_np[:, :, i] * stds
            else:
                stat_rescaled[:, :, i] = stat_np[:, :, i]
        stat_per_batch = stat_rescaled.sum(axis=1).T
        for t in range(stat_per_batch.shape[0]):
            for b in range(len(attributions_per_batch)):
                attributions_per_batch[b].append(float(stat_per_batch[t, b]))
            feature_names.append(f"stat_exog_{stat_cols[t]}" if t < len(stat_cols) else f"stat_exog_{t}")

    # Stack
    B = len(attributions_per_batch)
    batch_arrays = [np.array(attributions_per_batch[b]) for b in range(B)]
    concatenated_batch_values = np.stack(batch_arrays, axis=-1)
    return concatenated_batch_values, feature_names


# ── Aggregate across strides to produce importance DataFrames ─────────────────

def _aggregate_strides(
    explanations_list: list[dict[str, torch.Tensor]],
    nf_model,
    static_df: pd.DataFrame,
    mode: Literal["absolute", "signed"] = "absolute",
    futr_agg: Literal["hist", "futr", "combined"] = "combined",
) -> tuple[np.ndarray, list[str]]:
    """
    Aggregate attributions across strides for a single model.

    Returns
    -------
    stacked : np.ndarray, shape [n_strides, n_features, B]
        Per-stride feature attributions (baseline row dropped).
    feature_names : list[str]
    """
    all_values: list[np.ndarray] = []
    feature_names: list[str] = []
    for explanations in explanations_list:
        vals, names = combine_attributions(
            explanations=explanations,
            nf_model=nf_model,
            static_df=static_df,
            mode=mode,
            futr_agg=futr_agg,
        )
        all_values.append(vals[:-1])  # drop baseline row
        if not feature_names:
            feature_names = names
    stacked = np.stack(all_values, axis=0)  # [n_strides, n_features, B]
    return stacked, feature_names


def _aggregate_strides_rescaled(
    explanations_list: list[dict[str, torch.Tensor]],
    nf: NeuralForecast,
    model_idx: int,
    static_df: pd.DataFrame,
    futr_agg: Literal["hist", "futr", "combined"] = "combined",
    min_std: float = 1e-3,
    pooled_std: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Same as _aggregate_strides but with std-rescaling (signed only)."""
    all_values: list[np.ndarray] = []
    feature_names: list[str] = []
    for explanations in explanations_list:
        vals, names = combine_attributions_rescaled(
            explanations=explanations,
            nf=nf,
            model_idx=model_idx,
            static_df=static_df,
            futr_agg=futr_agg,
            min_std=min_std,
            pooled_std=pooled_std,
        )
        all_values.append(vals[:-1])
        if not feature_names:
            feature_names = names
    stacked = np.stack(all_values, axis=0)
    return stacked, feature_names


def feature_importance_overall(
    stacked: np.ndarray,
    feature_names: list[str],
    mode: Literal["absolute", "signed"] = "absolute",
) -> pd.DataFrame:
    """
    Compute overall feature importance from stacked per-stride attributions.

    Parameters
    ----------
    stacked : np.ndarray, shape [n_strides, n_features, B]
    feature_names : list[str]
    mode : "absolute" or "signed"

    Returns
    -------
    pd.DataFrame with columns:
        feature, abs_combined, pct_of_total_abs,
        and for signed mode additionally: signed_combined, net_ratio, pos_share, neg_share
    """
    # Average across strides -> [n_features, B]
    avg = np.mean(stacked, axis=0)

    abs_vals = np.abs(avg)
    abs_combined = abs_vals.sum(axis=-1)  # [n_features]
    total_abs = abs_combined.sum() if abs_combined.sum() != 0 else 1.0
    pct_of_total = abs_combined / total_abs * 100.0

    df = pd.DataFrame({
        "feature": feature_names,
        "abs_combined": abs_combined,
        "pct_of_total_abs": pct_of_total,
    })

    if mode == "signed":
        signed_combined = avg.sum(axis=-1)
        net_ratio = np.where(abs_combined == 0, 0.0, signed_combined / abs_combined * 100.0)
        pos_parts = np.maximum(avg, 0).sum(axis=-1)
        neg_parts = np.maximum(-avg, 0).sum(axis=-1)
        pos_share = np.where(abs_combined == 0, 0.0, pos_parts / abs_combined * 100.0)
        neg_share = np.where(abs_combined == 0, 0.0, neg_parts / abs_combined * 100.0)
        df["signed_combined"] = signed_combined
        df["net_ratio"] = net_ratio
        df["pos_share"] = pos_share
        df["neg_share"] = neg_share

    # Per-batch columns
    for b in range(avg.shape[-1]):
        df[f"abs_batch{b}"] = abs_vals[:, b]
        if mode == "signed":
            df[f"signed_batch{b}"] = avg[:, b]

    return df.sort_values("abs_combined", ascending=False).reset_index(drop=True)


def feature_importance_quarterly(
    stacked: np.ndarray,
    feature_names: list[str],
    stride_dates: list[pd.Timestamp],
    mode: Literal["absolute", "signed"] = "absolute",
) -> dict[int, pd.DataFrame]:
    """
    Split stacked attributions by quarter and compute importance per year.

    Parameters
    ----------
    stacked : np.ndarray, shape [n_strides, n_features, B]
    feature_names : list[str]
    stride_dates : list[pd.Timestamp]
        Physical date for each stride (length must match stacked.shape[0]).
    mode : "absolute" or "signed"

    Returns
    -------
    dict[int, pd.DataFrame]
        Keyed by quarter.
    """
    quarters = np.array([d.quarter for d in stride_dates])
    unique_quarters = sorted(set(quarters))
    result: dict[int, pd.DataFrame] = {}
    for quarter in unique_quarters:
        mask = quarters == quarter
        if not mask.any():
            continue
        sub = stacked[mask]
        result[quarter] = feature_importance_overall(sub, feature_names, mode=mode)
    return result


# ── Per-horizon importance ────────────────────────────────────────────────────

def build_per_horizon_feature_matrix(
    explanations: dict[str, torch.Tensor],
    nf_model,
    static_df: pd.DataFrame,
    futr_agg: Literal["hist", "futr", "combined"] = "combined",
    output_idx: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Build unscaled per-horizon attributions array.

    Future covariates before the forecast horizon are treated as historical.

    Returns
    -------
    arr : np.ndarray, shape [B, H, F]
    feature_names : list[str] of length F
    """
    past_covs = nf_model.hparams.get("hist_exog_list", []) or []
    futr_covs = nf_model.hparams.get("futr_exog_list", []) or []

    # insample
    ins = explanations["insample"]
    y_attr = ins[:, :, 0, output_idx, :, 0]
    mask_attr = ins[:, :, 0, output_idx, :, 1]
    insample_tokens = (y_attr + mask_attr).cpu().numpy()  # [B, H, input_size]
    input_size = insample_tokens.shape[-1]
    insample_names = [f"y_lag{i + 1}" for i in range(input_size)]

    # hist_exog
    hist_attr = explanations.get("hist_exog")
    if hist_attr is not None:
        hist_np = hist_attr.cpu().numpy()
        pfh_hist = _squeeze_if_single(hist_np.sum(axis=-2))
        n_hist = pfh_hist.shape[-1]
        hist_names = [f"hist_exog_{past_covs[i]}" if i < len(past_covs) else f"hist_exog_{i}" for i in range(n_hist)]
    else:
        pfh_hist = np.zeros((insample_tokens.shape[0], insample_tokens.shape[1], 0))
        hist_names = []

    # futr_exog
    futr_attr = explanations.get("futr_exog")
    if futr_attr is not None:
        arr = futr_attr.cpu().numpy()
        in_size = int(nf_model.hparams["input_size"])
        hist_part = _squeeze_if_single(arr[..., :in_size, :].sum(axis=-2))
        futr_part = _squeeze_if_single(arr[..., in_size:, :].sum(axis=-2))

        if futr_agg == "hist":
            futr_tokens_by_horizon = hist_part
        elif futr_agg == "futr":
            futr_tokens_by_horizon = futr_part
        else:
            futr_tokens_by_horizon = hist_part + futr_part

        n_futr = futr_tokens_by_horizon.shape[-1]
        futr_names = [f"futr_exog_{futr_covs[i]}" if i < len(futr_covs) else f"futr_exog_{i}" for i in range(n_futr)]
    else:
        futr_tokens_by_horizon = np.zeros((insample_tokens.shape[0], insample_tokens.shape[1], 0))
        futr_names = []

    # stat_exog
    stat_attr = explanations.get("stat_exog")
    if stat_attr is not None:
        stat_np = stat_attr.cpu().numpy().squeeze(axis=(2, 3))
        stat_cols = [c for c in static_df.columns if c != "unique_id"]
        n_stat = stat_np.shape[-1]
        stat_names = [f"stat_exog_{stat_cols[i]}" if i < len(stat_cols) else f"stat_exog_{i}" for i in range(n_stat)]
    else:
        stat_np = np.zeros((insample_tokens.shape[0], insample_tokens.shape[1], 0))
        stat_names = []

    # Concatenate
    parts = [insample_tokens, pfh_hist, futr_tokens_by_horizon, stat_np]
    names = insample_names + hist_names + futr_names + stat_names
    arr_concat = np.concatenate(
        [x if x.size else np.zeros((insample_tokens.shape[0], insample_tokens.shape[1], 0)) for x in parts],
        axis=-1,
    )
    return arr_concat, names


def compute_sample_level_stats(
    arr: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """
    Compute robust per-horizon statistics from stacked per-sample attributions.

    Parameters
    ----------
    arr : np.ndarray, shape (n_samples, B, H, F)

    Returns
    -------
    dict with keys: median, mean, mean_abs, p10, p25, p50, p75, p90,
    frac_pos, frac_neg, ci_mean_lower, ci_mean_upper.
    Each value has shape (B, H, F).
    """
    if arr.ndim != 4:
        raise ValueError(f"Expected shape (n_samples, B, H, F), got {arr.shape}")

    stats: dict[str, np.ndarray] = {
        "median": np.median(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "mean_abs": np.mean(np.abs(arr), axis=0),
        "frac_pos": (arr > 0).mean(axis=0),
        "frac_neg": (arr < 0).mean(axis=0),
    }
    for q, label in [(10, "p10"), (25, "p25"), (50, "p50"), (75, "p75"), (90, "p90")]:
        stats[label] = np.percentile(arr, q, axis=0)

    # Bootstrap CI for mean
    n = arr.shape[0]
    rng = np.random.default_rng(42)
    boots = np.stack([np.mean(arr[rng.integers(0, n, size=n)], axis=0) for _ in range(n_boot)])
    stats["ci_mean_lower"] = np.percentile(boots, 100 * alpha / 2, axis=0)
    stats["ci_mean_upper"] = np.percentile(boots, 100 * (1 - alpha / 2), axis=0)

    return stats


def feature_importance_per_horizon(
    stats: dict[str, np.ndarray],
    feature_names: list[str],
    top_k: int | None = None,
    rank_metric: str = "mean_abs",
    include_metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a per-horizon importance DataFrame from pre-computed statistics.

    Parameters
    ----------
    stats : dict from ``compute_sample_level_stats``
    feature_names : list[str]
    top_k : int | None
        Keep only top K features per horizon. None = keep all.
    rank_metric : str
        Metric to use for ranking features (default: ``mean_abs``).
    include_metrics : list[str] | None
        Which metrics to include in the output. None = all available.

    Returns
    -------
    pd.DataFrame with columns: batch, horizon, feature, + selected metrics.
    """
    rank_data = stats[rank_metric]  # (B, H, F)
    B, H, F = rank_data.shape
    if top_k is None or top_k <= 0:
        top_k = F
    top_k = min(top_k, F)

    if include_metrics is None:
        include_metrics = list(stats.keys())

    # Per-horizon ranking: mean_abs averaged over batches
    horizon_ranking = np.mean(np.abs(rank_data), axis=0)  # (H, F)
    top_idx_per_h = np.argsort(horizon_ranking, axis=1)[:, ::-1][:, :top_k]

    rows: list[dict] = []
    for b in range(B):
        for h in range(H):
            for rank, fi in enumerate(top_idx_per_h[h]):
                row = {
                    "batch": b,
                    "horizon": h,
                    "rank": rank,
                    "feature": feature_names[fi],
                }
                for metric_name in include_metrics:
                    if metric_name in stats:
                        row[metric_name] = float(stats[metric_name][b, h, fi])
                rows.append(row)

    return pd.DataFrame(rows)


# ── Feature clustering ────────────────────────────────────────────────────────

def cluster_features_by_horizon(
    arr_bhf: np.ndarray,
    feature_names: list[str],
    n_clusters: int = 8,
    method: str = "average",
    metric: str = "correlation",
    collapse_agg: Callable = np.mean,
) -> dict:
    """
    Hierarchical clustering of features based on their per-horizon attribution profiles.

    Parameters
    ----------
    arr_bhf : np.ndarray, shape [B, H, F]
        Mean attributions (averaged across strides).
    feature_names : list[str]
    n_clusters : int
    method : str
        Linkage method for scipy.cluster.hierarchy.
    metric : str
        Distance metric for pdist.
    collapse_agg : callable
        Function to aggregate features within each cluster.

    Returns
    -------
    dict with keys:
        linkage, labels (len F), cluster_agg (n_clusters, H) per batch,
        order (dendrogram leaf order), cluster_names, feature_cluster_map (DataFrame).
    """
    B, H, F = arr_bhf.shape
    k = min(n_clusters, max(1, F))

    # Build profile: [F, H*B]
    profile = arr_bhf.transpose(2, 1, 0).reshape(F, H * B).astype(float)
    profile = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)

    # Drop constant features
    feature_var = profile.var(axis=1)
    nonconst_mask = feature_var > 0.0
    kept_idx = np.where(nonconst_mask)[0]

    if kept_idx.size == 0:
        warnings.warn("No features with non-zero variance for clustering.")
        return {
            "linkage": None, "labels": np.full(F, -1, dtype=int),
            "cluster_agg": {}, "order": list(range(F)),
            "cluster_names": [], "feature_cluster_map": pd.DataFrame(),
        }

    profile_kept = profile[kept_idx]

    # Distance & linkage
    try:
        condensed = pdist(profile_kept, metric=metric)  # type: ignore[arg-type]
        if not np.isfinite(condensed).all():
            raise ValueError("Non-finite distances")
    except Exception:
        condensed = pdist(profile_kept, metric="euclidean")  # type: ignore[arg-type]

    linkage = sch.linkage(condensed, method=method)
    dendro = sch.dendrogram(linkage, no_plot=True)
    plt.close()
    order_kept = dendro["leaves"]
    order = list(kept_idx[order_kept])

    labels_kept = sch.fcluster(linkage, t=min(k, profile_kept.shape[0]), criterion="maxclust") - 1
    labels = np.full(F, -1, dtype=int)
    labels[kept_idx] = labels_kept
    actual_k = int(labels_kept.max() + 1)
    cluster_names = [f"cluster_{i}" for i in range(actual_k)]

    # Cluster-level aggregation per batch.
    # Build dynamically (list → stack) to avoid static shape mismatches.
    # Use arr_bhf[b][:, idx] instead of arr_bhf[b, :, idx]: the latter triggers
    # numpy's non-adjacent advanced-indexing rule when b is a scalar and idx is
    # an array, returning shape (len(idx), H) instead of (H, len(idx)), which
    # causes collapse_agg(axis=1) to return (len(idx),) rather than (H,).
    cluster_agg: dict[int, np.ndarray] = {}
    for b in range(B):
        rows: list[np.ndarray] = []
        for c in range(actual_k):
            idx = np.where(labels == c)[0]
            if idx.size > 0:
                # arr_bhf[b][:, idx] → shape (H, n_in_cluster)
                rows.append(collapse_agg(arr_bhf[b][:, idx], axis=1))  # (H,)
            else:
                rows.append(np.zeros(H))
        cluster_agg[b] = np.stack(rows, axis=0)  # (actual_k, H)

    # Feature → cluster mapping
    mapping_rows = []
    for fi in range(F):
        mapping_rows.append({"feature": feature_names[fi], "cluster": int(labels[fi])})
    feature_cluster_map = pd.DataFrame(mapping_rows)

    return {
        "linkage": linkage,
        "labels": labels,
        "cluster_agg": cluster_agg,
        "order": order,
        "cluster_names": cluster_names,
        "feature_cluster_map": feature_cluster_map,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════


def plot_overall_importance_bar(
    df: pd.DataFrame,
    top_n: int = 25,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 7),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Horizontal bar chart of overall feature importance (by ``pct_of_total_abs``).
    """
    df_sorted = df.sort_values("pct_of_total_abs", ascending=False).head(top_n).iloc[::-1]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    sns.barplot(x="pct_of_total_abs", y="feature", data=df_sorted, color="steelblue", ax=ax)
    ax.set_xlabel("Importance (% of total absolute attribution)")
    ax.set_ylabel("")
    if title:
        ax.set_title(title)
    for i, (v, _) in enumerate(zip(df_sorted["pct_of_total_abs"], df_sorted["feature"])):
        ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_signed_importance_bar(
    df: pd.DataFrame,
    top_n: int = 25,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 7),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Stacked horizontal bar chart showing positive vs negative share
    (requires ``pos_share`` and ``neg_share`` columns from signed mode).
    """
    if "pos_share" not in df.columns:
        raise ValueError("DataFrame must have 'pos_share' column (use mode='signed')")

    df_sorted = df.sort_values("pct_of_total_abs", ascending=False).head(top_n).iloc[::-1]
    y = np.arange(len(df_sorted))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    ax.barh(y, np.asarray(df_sorted["pos_share"].values, dtype=float), color="tab:green", label="positive (%)")
    ax.barh(y, -np.asarray(df_sorted["neg_share"].values, dtype=float), color="tab:red", label="negative (%)")

    if "net_ratio" in df_sorted.columns:
        for i, net in enumerate(df_sorted["net_ratio"].values):
            ax.plot([net], [y[i]], marker="D", color="k", markersize=5)

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["feature"].values)
    ax.set_xlabel("% of feature magnitude (pos right, neg left); ◆ = net %")
    ax.axvline(0, color="k", linewidth=0.5)
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_per_horizon_heatmap(
    arr: np.ndarray,
    feature_names: list[str],
    batch_idx: int = 0,
    top_k: int = 40,
    signed: bool = True,
    figsize: tuple = (14, 10),
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Heatmap of features × horizons for a chosen batch.

    Parameters
    ----------
    arr : np.ndarray, shape [B, H, F]
    """
    B, H, F = arr.shape
    mat = arr[batch_idx]
    if not signed:
        mat = np.abs(mat)

    ranking = np.abs(mat).sum(axis=0)  # [F]
    order = np.argsort(ranking)[::-1]
    top_idx = order[:min(top_k, F)]
    mat_top = mat[:, top_idx].T  # [top_k, H]
    names_top = [feature_names[i] for i in top_idx]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    cmap = "RdBu_r" if signed else "viridis"
    sns.heatmap(
        mat_top,
        cmap=cmap,
        center=0 if signed else None,
        xticklabels=[f"h{h}" for h in range(H)],
        yticklabels=names_top,
        ax=ax,
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel(f"Feature (top {len(names_top)})")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Per-horizon attributions (batch {batch_idx}) - {'signed' if signed else 'abs'}")
    fig.tight_layout()
    return fig, ax


def plot_per_horizon_violin(
    arr_bhf: np.ndarray,
    feature_names: list[str],
    top_k: int = 20,
    batch_idxs: tuple[int, ...] | None = None,
    signed: bool = True,
    figsize: tuple = (10, 8),
    title: str | None = None,
    inner: str = "box",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Violin plot showing per-feature attribution distributions across forecast horizons.

    For each of the top-K features (ranked by mean absolute attribution),
    draws a horizontal violin showing the spread of signed or absolute
    attributions across all forecast horizons and the selected batch(es).
    This complements ``plot_per_horizon_heatmap`` by revealing the *shape*
    of each feature's horizon-attribution profile (e.g. whether a feature
    is concentrated on early horizons or uniformly important).

    Parameters
    ----------
    arr_bhf : np.ndarray, shape [B, H, F]
        Per-horizon attributions (e.g. mean over strides from
        ``build_per_horizon_feature_matrix``).
    feature_names : list[str]
    top_k : int
        Number of top features to show.
    batch_idxs : tuple[int, ...] | None
        Which batch indices to pool when building the violin.
        ``None`` pools all batches.
    signed : bool
        If ``True`` keep the sign of the attribution; if ``False`` use
        absolute values.
    figsize : tuple
    title : str | None
    inner : str
        Seaborn ``violinplot`` inner style: ``"box"``, ``"point"``,
        ``"stick"``, or ``None``.
    ax : Axes | None

    Returns
    -------
    fig, ax
    """
    B, H, F = arr_bhf.shape
    if batch_idxs is None:
        batch_idxs = tuple(range(B))

    # Pool selected batches → shape (n_batches * H, F)
    slices = np.concatenate([arr_bhf[b] for b in batch_idxs], axis=0)
    if not signed:
        slices = np.abs(slices)

    # Rank features by mean absolute attribution over all batches and horizons
    ranking = np.abs(arr_bhf).mean(axis=(0, 1))  # (F,)
    order = np.argsort(ranking)[::-1]
    top_idx = order[:min(top_k, F)]

    # Build long-form DataFrame: one row per (feature, horizon_×_batch) sample
    rows: list[dict] = []
    for fi in top_idx:
        for val in slices[:, fi]:
            rows.append({"feature": feature_names[fi], "attribution": float(val)})
    df_long = pd.DataFrame(rows)

    # Feature order: most important at top
    feat_order = [feature_names[i] for i in top_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.violinplot(
        data=df_long,
        y="feature",
        x="attribution",
        order=feat_order,
        orient="h",
        inner=inner,
        cut=0,
        ax=ax,
    )
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Attribution" + (" (signed)" if signed else " (absolute)"))
    ax.set_ylabel("")
    n_batches = len(batch_idxs)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"Per-horizon attribution distribution - top {len(top_idx)} features"
            f" ({n_batches} batch{'es' if n_batches != 1 else ''} × {H} horizons)"
        )
    fig.tight_layout()
    return fig, ax


def plot_feature_horizon_lines(
    arr: np.ndarray,
    feature_names: list[str],
    top_k: int = 5,
    batch_idxs: tuple[int, ...] = (0, 1),
    figsize: tuple = (10, 4),
    title: str | None = None,
) -> tuple[Figure, list[Axes]]:
    """
    Line plot of top-K features across horizons.
    """
    B, H, F = arr.shape
    ranking = np.abs(arr).sum(axis=(0, 1))  # [F]
    order = np.argsort(ranking)[::-1][:top_k]
    features = [feature_names[i] for i in order]

    fig, axes = plt.subplots(len(features), 1, figsize=(figsize[0], figsize[1] * len(features)), sharex=True)
    if len(features) == 1:
        axes = [axes]
    for i, (feat, fi) in enumerate(zip(features, order)):
        for b in batch_idxs:
            axes[i].plot(np.arange(H), arr[b, :, fi], label=f"batch{b}", marker="o", markersize=3)
        axes[i].axhline(0, color="k", linewidth=0.5)
        axes[i].set_ylabel(feat, fontsize=8)
        axes[i].legend(fontsize=7)
    axes[-1].set_xlabel("Horizon")
    fig.suptitle(title or "Feature contributions across horizon")
    fig.tight_layout()
    return fig, axes


def plot_cluster_heatmap(
    cluster_info: dict,
    batch_idx: int = 0,
    figsize: tuple = (14, 6),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Heatmap of cluster-level attributions across horizons.
    """
    cluster_agg = cluster_info["cluster_agg"][batch_idx]
    cluster_names = cluster_info["cluster_names"]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    H = cluster_agg.shape[1]
    sns.heatmap(
        cluster_agg,
        cmap="viridis",
        xticklabels=[f"h{h}" for h in range(H)],
        yticklabels=cluster_names,
        ax=ax,
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Cluster")
    ax.set_title(title or f"Cluster-level attributions (batch {batch_idx})")
    fig.tight_layout()
    return fig, ax


def plot_dendrogram(
    cluster_info: dict,
    feature_names: list[str],
    figsize: tuple | None = None,
    title: str | None = None,
) -> Figure:
    """
    Standalone dendrogram of the hierarchical feature clustering.

    Leaf labels are coloured by their cluster membership so the relationship
    between the tree topology and the cluster assignment is immediately visible.
    A small cluster-colour legend is added in the lower-right corner.

    Parameters
    ----------
    cluster_info : dict
        Output of :func:`cluster_features_by_horizon`.
    feature_names : list[str]
        Full feature name list (same order used during clustering).
    figsize : tuple | None
        ``(width, height)`` in inches.  Height is auto-scaled to
        ``max(6, F * 0.22)`` when ``None``.
    title : str | None
        Optional axes title.  Defaults to ``"Feature dendrogram"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    linkage   = cluster_info["linkage"]
    labels    = cluster_info["labels"]
    cluster_names = cluster_info["cluster_names"]
    actual_k  = len(cluster_names)
    F         = len(feature_names)

    palette = sns.color_palette("tab10", n_colors=max(actual_k, 1))
    cluster_colors: dict[int, tuple] = {
        c: tuple(palette[c % len(palette)]) for c in range(actual_k)
    }
    name_to_cluster: dict[str, int] = {
        feature_names[fi]: int(labels[fi])
        for fi in range(F)
        if fi < len(labels)
    }

    if figsize is None:
        figsize = (10, max(6, F * 0.22))

    fig, ax = plt.subplots(figsize=figsize)

    if linkage is not None:
        sch.dendrogram(
            linkage,
            orientation="left",
            labels=feature_names,
            ax=ax,
            leaf_font_size=max(4, min(8, int(300 / max(F, 1)))),
            color_threshold=0,
            above_threshold_color="#555555",
        )
        ax.set_xlabel("Distance", fontsize=9)
        # Colour leaf labels by cluster membership
        for tick_label in ax.get_yticklabels():
            c = name_to_cluster.get(tick_label.get_text(), -1)
            tick_label.set_color(cluster_colors.get(c, (0.4, 0.4, 0.4)))
        # Cluster legend
        legend_handles = [
            mpatches.Patch(color=cluster_colors[c], label=cluster_names[c])
            for c in range(actual_k)
        ]
        ax.legend(
            handles=legend_handles, title="Cluster",
            loc="lower right", fontsize=7, title_fontsize=8,
        )
    else:
        ax.text(
            0.5, 0.5, "Dendrogram\nnot available",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.axis("off")

    ax.set_title(title or "Feature dendrogram", fontsize=11)
    fig.tight_layout()
    return fig


def plot_cluster_top_features(
    arr_bhf: np.ndarray,
    feature_names: list[str],
    cluster_info: dict,
    top_f: int = 5,
    top_p: float | None = 80.0,
    figsize_per_cluster: tuple = (9, 2.2),
    title: str | None = None,
) -> Figure:
    """
    Per-cluster top-feature importance bar chart.

    For each cluster the function selects the *smallest* set of features
    whose combined mean absolute attribution accounts for at least *top_p* %
    of the cluster’s total attribution (when ``top_p`` is not ``None``).  The
    number of selected features is capped at *top_f* regardless of whether the
    threshold has been reached.  When ``top_p`` is ``None``, exactly *top_f*
    features are shown.  At least one feature is always displayed.

    Each cluster gets its own subplot panel arranged in a two-column grid.
    Bars show mean absolute attribution; error bars show the standard deviation
    of ``|attribution|`` across forecast horizons (averaged across batches),
    capturing how much a feature’s importance varies over the horizon.
    The bar colour matches the cluster colour used in
    :func:`plot_dendrogram`.

    Parameters
    ----------
    arr_bhf : np.ndarray, shape [B, H, F]
        Per-horizon attribution array (e.g. mean over strides from
        ``build_per_horizon_feature_matrix``).
    feature_names : list[str]
    cluster_info : dict
        Output of :func:`cluster_features_by_horizon`.
    top_f : int
        Hard upper bound on the number of features shown per cluster
        (default 5).  Also used as the exact count when ``top_p`` is ``None``.
    top_p : float | None
        Coverage threshold in % of cluster attribution (default 80.0).
        Features are added in descending importance order until the cumulative
        attribution reaches this fraction of the cluster total.  Set to
        ``None`` to always show exactly ``top_f`` features.
    figsize_per_cluster : tuple
        ``(width, height_per_panel)`` in inches.  Total figure height is
        scaled by the number of cluster rows.
    title : str | None
        Optional figure-level suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels        = cluster_info["labels"]
    cluster_names = cluster_info["cluster_names"]
    actual_k      = len(cluster_names)
    B, H, F       = arr_bhf.shape

    palette = sns.color_palette("tab10", n_colors=max(actual_k, 1))
    cluster_colors: dict[int, tuple] = {
        c: tuple(palette[c % len(palette)]) for c in range(actual_k)
    }

    # Per-feature summary stats
    mean_abs: np.ndarray = np.abs(arr_bhf).mean(axis=(0, 1))   # [F]
    # Std of |attribution| over horizons, averaged over batches → [F]
    std_over_h: np.ndarray = np.abs(arr_bhf).std(axis=1).mean(axis=0)

    # ── Per-cluster feature selection ────────────────────────────────────
    cluster_selected: dict[int, list[int]] = {}
    for c in range(actual_k):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            cluster_selected[c] = []
            continue
        # Descending order of importance within the cluster
        sorted_idx = idx_c[np.argsort(mean_abs[idx_c])[::-1]]
        if top_p is not None:
            cluster_total = float(mean_abs[idx_c].sum())
            if cluster_total <= 0.0:
                n_sel = max(1, top_f)
            else:
                threshold = (top_p / 100.0) * cluster_total
                cumsum = np.cumsum(mean_abs[sorted_idx])
                # First index where cumsum >= threshold
                i_over = int(np.searchsorted(cumsum, threshold, side="left"))
                # +1 because searchsorted gives insertion point, not element count
                n_sel = max(1, min(i_over + 1, top_f, len(sorted_idx)))
        else:
            n_sel = max(1, min(top_f, len(sorted_idx)))
        cluster_selected[c] = list(sorted_idx[:n_sel])

    non_empty = [c for c in range(actual_k) if cluster_selected[c]]
    n_panels  = len(non_empty)

    if n_panels == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No features to display", ha="center", va="center")
        ax.axis("off")
        if title:
            ax.set_title(title)
        return fig

    n_cols = min(2, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    # Dynamic row height: base height + extra for additional features
    max_feats = max(len(cluster_selected[c]) for c in non_empty)
    row_h = figsize_per_cluster[1] + max(0.0, (max_feats - 2) * 0.35)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cluster[0] * n_cols, row_h * n_rows),
        squeeze=False,
    )

    for panel_idx, c in enumerate(non_empty):
        row, col = divmod(panel_idx, n_cols)
        ax = axes[row][col]
        color = cluster_colors[c]

        feat_idx = cluster_selected[c]     # already sorted best→worst
        names    = [feature_names[i] for i in feat_idx]
        vals     = mean_abs[feat_idx]
        errs     = std_over_h[feat_idx]

        # Reverse so most-important bar appears at the top of the y-axis
        y = np.arange(len(names))
        ax.barh(
            y, vals[::-1], xerr=errs[::-1],
            color=color, alpha=0.82,
            error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "#444"},
        )
        ax.set_yticks(y)
        ax.set_yticklabels(names[::-1], fontsize=9)
        ax.set_xlabel("Mean |attribution|  (± horizon std)", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.axvline(0, color="k", linewidth=0.4)

        # Subtitle: cluster name + coverage stats (matching feature-set format)
        total_abs = float(mean_abs.sum()) or 1.0
        cluster_idx = np.where(labels == c)[0]
        cluster_total = float(mean_abs[cluster_idx].sum())
        cluster_pct = cluster_total / total_abs * 100
        shown_pct   = float(vals.sum()) / total_abs * 100
        n_total     = len(cluster_idx)
        ax.set_title(
            f"{cluster_names[c]}  -  set: {cluster_pct:.1f}\u202f%  ·  shown: {shown_pct:.1f}\u202f% of total"
            f"  ({len(feat_idx)}/{n_total} features)",
            fontsize=9, color=color, fontweight="bold",
        )

    # Hide unused subplot panels
    for panel_idx in range(len(non_empty), n_rows * n_cols):
        row, col = divmod(panel_idx, n_cols)
        axes[row][col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Feature-set importance analysis
# ══════════════════════════════════════════════════════════════════════════════


def build_feature_to_feature_set_map(
    feature_names: list[str],
    feature_sets: list[str],
) -> dict[str, str]:
    """
    Map each feature name to its originating feature set.

    Feature names may carry a NeuralForecast covariate-type prefix
    (``hist_exog_``, ``futr_exog_``, or ``static_exog_``).  The prefix is
    handled as follows before content classification:

    * ``static_exog_*`` - assigned to the standalone ``"static_exog"``
      category regardless of the rest of the name.
    * ``hist_exog_*`` / ``futr_exog_*`` - prefix is stripped and the bare
      name is classified using the patterns below.

    Recognised feature sets / categories
    -------------------------------------
    * ``"basic"`` - target lags (``y_lag*``), calendar columns
      (``hour``, ``day``, ``month``, ``is_weekend``, ``is_holiday``,
      ``is_workday``), residual-load columns.
    * ``"day_ahead_price"`` - ``day_ahead_price_*`` columns.
    * ``"wind_pv"`` - ``wind_forecast_*``, ``pv_forecast_*``,
      ``wind_actual_*``, ``pv_actual_*``.
    * ``"production_consumption"`` - ``production_*`` / ``consumption_*``.
    * ``"cross_border"`` - ``cross_border_*`` columns.
    * ``"bloomberg"`` - ``bloomberg_*`` columns.
    * ``"sce"`` - ``sce_forecast_*`` columns.
    * ``"static_exog"`` - any ``static_exog_*`` column (standalone).
    * ``"data_driven"`` - ``data_driven_*`` columns.
    * ``"runlength"`` - ``runlength_*`` columns.

    The three structural categories (``static_exog``, ``data_driven``,
    ``runlength``) bypass the ``feature_sets`` validation because they are
    covariate-type groupings rather than content-based feature sets recorded
    in the dataset metadata.  All other inferred sets fall back to
    ``"basic"`` when absent from *feature_sets*.

    Parameters
    ----------
    feature_names : list[str]
        All feature names used by the model (including ``y_lag*``, calendar
        columns, etc.).
    feature_sets : list[str]
        Feature sets actually present in the dataset, e.g. from
        ``run_config["dataset_metadata"]["feature_sets"]`` or a local
        dataset metadata JSON.  Used to validate content-set assignments.

    Returns
    -------
    dict[str, str]
        ``{feature_name: feature_set_name}``.
    """
    known_sets = set(feature_sets)
    fallback = "basic" if "basic" in known_sets else (feature_sets[0] if feature_sets else "basic")
    # Structural categories that are always valid regardless of known_sets
    _STRUCTURAL = frozenset({"static_covariates", "data_driven", "runlength", "calendar"})
    mapping: dict[str, str] = {}

    for feat in feature_names:
        # ── static_covariates prefix → standalone category ──────────────────────
        if feat.startswith("stat_exog_"):
            mapping[feat] = "static_covariates"
            continue

        # ── Strip hist_exog_ / futr_exog_ to get the bare feature name ────
        bare = feat
        for pfx in ("hist_exog_", "futr_exog_"):
            if feat.startswith(pfx):
                bare = feat[len(pfx):]
                break

        # ── Classify by bare name ──────────────────────────────────────────
        # Target lags (IG-internal names for the lagged y series)
        if bare.startswith("y_lag"):
            fs = "basic"

        # Calendar features (training.data_prep.CALENDAR_COLS)
        elif bare in ("hour", "day", "month", "is_weekend", "is_holiday", "is_workday"):
            fs = "calendar"

        # Bloomberg
        elif bare.startswith("bloomberg_"):
            fs = "bloomberg"

        # Scheduled commercial exchanges
        elif bare.startswith("sce_forecast_"):
            fs = "sce"

        # Cross-border physical flows
        elif bare.startswith("cross_border_"):
            fs = "cross_border"

        # Day-ahead electricity prices
        elif bare.startswith("day_ahead_price"):
            fs = "day_ahead_price"

        # Wind / PV
        elif (
            bare.startswith("wind_forecast_")
            or bare.startswith("wind_actual_")
            or bare.startswith("pv_forecast_")
            or bare.startswith("pv_actual_")
        ):
            fs = "wind_pv"

        # Production / Consumption (combined feature set)
        elif bare.startswith("production_") or bare.startswith("consumption_") or "residual_load" in bare:
            fs = "production_consumption"

        # Data-driven / learned covariates (structural - always valid)
        elif bare.startswith("data_driven_") or bare == "data_driven":
            fs = "data_driven"

        # Run-length features (structural - always valid)
        elif bare.startswith("runlength_") or bare == "runlength":
            fs = "runlength"

        # Residual load / total load → basic
        elif bare.startswith("residual_load_") or bare == "total_load":
            fs = "basic"

        # Unknown → safe fallback
        else:
            fs = "basic"

        # Validate: structural sets bypass the check; content sets must be
        # present in known_sets; otherwise fall back.
        if fs not in _STRUCTURAL and fs not in known_sets:
            fs = fallback

        mapping[feat] = fs

    return mapping


def plot_feature_set_bars(
    arr_bhf: np.ndarray,
    feature_names: list[str],
    feature_set_map: dict[str, str],
    feature_sets: list[str],
    top_f: int = 10,
    top_p: float | None = 80.0,
    figsize_per_set: tuple[float, float] = (9.0, 2.5),
    title: str | None = None,
) -> Figure:
    """
    Per-feature-set importance bar chart.

    Analogous to :func:`plot_cluster_top_features` but groups features by
    their originating feature set instead of by hierarchical clustering.
    The top contributing features of each set are shown (sorted descending
    by mean |attribution|), selected by the ``top_p`` coverage threshold and
    ``top_f`` hard cap - the same logic used in
    :func:`plot_cluster_top_features`.

    Parameters
    ----------
    arr_bhf : np.ndarray, shape [B, H, F]
        Per-horizon attribution array, usually the mean over strides from
        :func:`build_per_horizon_feature_matrix`.
    feature_names : list[str]
        Feature names in the same order as the F dimension of *arr_bhf*.
    feature_set_map : dict[str, str]
        ``{feature_name: feature_set_name}`` from
        :func:`build_feature_to_feature_set_map`.
    feature_sets : list[str]
        Ordered list of feature set names to display (determines panel order).
        Sets not present in the data are silently skipped.
    top_f : int
        Hard cap on the number of features displayed per panel.  Defaults to
        ``10``.
    top_p : float | None
        Coverage threshold in percent.  For each panel only the top features
        that together account for at least ``top_p``\u202f% of the set's total
        mean absolute attribution are shown, subject to the ``top_f`` cap.
        Pass ``None`` to always show exactly ``top_f`` features (no coverage
        trimming).  Defaults to ``80.0``.
    figsize_per_set : tuple[float, float]
        ``(width_per_column, height_per_panel)`` in inches.  Total height is
        scaled by the number of displayed rows.
    title : str | None
        Optional figure-level suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    B, H, F = arr_bhf.shape

    # ── Per-feature summary statistics ─────────────────────────────────────
    mean_abs: np.ndarray = np.abs(arr_bhf).mean(axis=(0, 1))        # [F]
    # Std of |attribution| over horizons, then mean over batches → [F]
    std_over_h: np.ndarray = np.abs(arr_bhf).std(axis=1).mean(axis=0)

    # ── Augment feature_sets with structural categories found in the map ────
    # (static_exog, data_driven, runlength may not appear in the metadata list)
    _map_vals = set(feature_set_map.values())
    extra_sets = [
        s for s in _CANONICAL_SET_ORDER
        if s not in set(feature_sets) and s in _map_vals
    ]
    effective_sets = list(feature_sets) + extra_sets

    # ── Group feature indices by feature set ────────────────────────────────
    feat_idx_by_set: dict[str, list[int]] = {fs: [] for fs in effective_sets}
    fallback_set = "basic" if "basic" in feat_idx_by_set else (effective_sets[0] if effective_sets else None)
    for i, feat in enumerate(feature_names):
        fs = feature_set_map.get(feat, fallback_set or "basic")
        if fs in feat_idx_by_set:
            feat_idx_by_set[fs].append(i)
        elif fallback_set and fallback_set in feat_idx_by_set:
            feat_idx_by_set[fallback_set].append(i)

    non_empty_sets = [fs for fs in effective_sets if feat_idx_by_set.get(fs)]

    if not non_empty_sets:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No features to display", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        if title:
            ax.set_title(title)
        return fig

    total_abs = float(mean_abs.sum()) or 1.0

    # ── Per-set feature selection (apply top_p / top_f trimming) ────────────
    selected_by_set: dict[str, list[int]] = {}
    for fs in non_empty_sets:
        idx_c = feat_idx_by_set[fs]
        # Sort descending by mean |attribution|
        s_idx = sorted(idx_c, key=lambda i: mean_abs[i], reverse=True)
        if top_p is not None:
            set_total_val = float(sum(mean_abs[i] for i in s_idx))
            if set_total_val > 0.0:
                threshold = (top_p / 100.0) * set_total_val
                cumsum = np.cumsum([mean_abs[i] for i in s_idx])
                i_over = int(np.searchsorted(cumsum, threshold, side="left"))
                n_sel = max(1, min(i_over + 1, top_f, len(s_idx)))
            else:
                n_sel = max(1, min(top_f, len(s_idx)))
        else:
            n_sel = max(1, min(top_f, len(s_idx)))
        selected_by_set[fs] = s_idx[:n_sel]

    # ── Layout ──────────────────────────────────────────────────────────────
    n_cols = min(2, len(non_empty_sets))
    n_rows = (len(non_empty_sets) + n_cols - 1) // n_cols

    # Dynamic row height: scale with the tallest *selected* feature list
    max_feats = max(len(selected_by_set[fs]) for fs in non_empty_sets)
    row_h = figsize_per_set[1] + max(0.0, (max_feats - 3) * 0.35)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_set[0] * n_cols, row_h * n_rows),
        squeeze=False,
    )

    for panel_idx, fs in enumerate(non_empty_sets):
        row, col = divmod(panel_idx, n_cols)
        ax = axes[row][col]

        color = _FEATURE_SET_PALETTE.get(fs, "#888888")
        n_total = len(feat_idx_by_set[fs])

        sorted_sel = np.array(selected_by_set[fs])
        names = [feature_names[i] for i in sorted_sel]
        vals  = mean_abs[sorted_sel]
        errs  = std_over_h[sorted_sel]

        y = np.arange(len(names))
        ax.barh(
            y, vals[::-1], xerr=errs[::-1],
            color=color, alpha=0.82,
            error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "#444"},
        )
        ax.set_yticks(y)
        ax.set_yticklabels(names[::-1], fontsize=9)
        ax.set_xlabel("Mean |attribution|  (± horizon std)", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.axvline(0, color="k", linewidth=0.4)

        set_total_abs = float(mean_abs[list(feat_idx_by_set[fs])].sum())
        set_pct   = set_total_abs / total_abs * 100
        shown_pct = float(vals.sum()) / total_abs * 100
        n_sel     = len(sorted_sel)
        ax.set_title(
            f"{fs}  -  set: {set_pct:.1f}\u202f%  ·  shown: {shown_pct:.1f}\u202f% of total"
            f"  ({n_sel}/{n_total} features)",
            fontsize=9, color=color, fontweight="bold",
        )

    # Hide unused subplot panels
    for panel_idx in range(len(non_empty_sets), n_rows * n_cols):
        row, col = divmod(panel_idx, n_cols)
        axes[row][col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def get_stride_dates_from_wandb(
    run_config: dict,
    n_strides: int,
    test_start_key: str = "test_start",
    horizon_key: str = "forecast_horizon",
    default_test_start: str = "2025-01-01 00:00:00",
    default_forecast_horizon: int = 24,
) -> list[pd.Timestamp]:
    """
    Generate one start-timestamp per stride from a wandb run config.

    Data is assumed to be **hourly**, so consecutive strides are separated by
    ``forecast_horizon`` hours.  This produces timestamps that align perfectly
    with the rolling-window evaluation schedule used during training.

    Parameters
    ----------
    run_config : dict
        wandb run config dict (e.g. ``run.config`` or a loaded JSON config).
    n_strides : int
        Number of strides, i.e. the length of the loaded IG tensor list.
    test_start_key : str
        Key in ``run_config`` holding the test-period start timestamp.
        Defaults to ``"test_start"``.
    horizon_key : str
        Key in ``run_config`` holding the forecast horizon (integer, in hours).
        Defaults to ``"forecast_horizon"``.
    default_test_start : str
        Fallback value used when ``test_start_key`` is absent from the config.
    default_forecast_horizon : int
        Fallback value used when ``horizon_key`` is absent from the config.
        
    Returns
    -------
    list[pd.Timestamp] of length ``n_strides``
        ``stride_dates[i]`` is the physical start timestamp of the *i*-th
        stride's forecast window.

    Examples
    --------
    >>> dates = get_stride_dates_from_wandb(run.config, n_strides=len(ig_list))
    >>> df_yearly = feature_importance_yearly(stacked, feature_names, dates)
    """
    raw_start = run_config.get(test_start_key, default_test_start)
    test_start = pd.Timestamp(raw_start)
    forecast_horizon = int(run_config.get(horizon_key, default_forecast_horizon))
    return list(
        pd.date_range(
            start=test_start,
            periods=n_strides,
            freq=f"{forecast_horizon}h",
        )
    )
