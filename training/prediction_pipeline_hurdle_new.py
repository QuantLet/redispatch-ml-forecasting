"""Single-window prediction & evaluation pipeline for hurdle (zero-inflated) models.

Analogous to ``prediction_pipeline_new.py`` but handles hurdle-specific logic:

* Loads checkpoints via :class:`~training.hurdle_nf.RedispatchNeuralForecast`.
* Produces **two point-forecast columns** per model alias:

    ``{alias}``       – mean:  ``sigmoid(logit) × relu(magnitude)``
    ``{alias}_theta`` – threshold-gated: ``1_{p̂ ≥ θ} × relu(magnitude)``

* Looks for the optimised θ as a JSON file written by
  :func:`~training.hurdle_runner._optimize_and_save_valid_threshold` at
  training time (``hurdle_theta_opt_{alias}.json`` next to the checkpoint).
  **If the file is missing the threshold is re-optimised at prediction time**
  on the validation period derived from ``run_config``, and the result is
  saved locally (and logged to W&B) so subsequent runs can reuse it.
* Saves raw predictions (logit + magnitude preserved) as Parquet locally.
* Logs processed predictions (point forecasts only) to W&B.
* Evaluates each forecast type and saves a combined evaluation CSV.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import wandb
import zstandard as zstd
from scipy.special import expit
from sklearn.metrics import fbeta_score

from neuralforecast import NeuralForecast

from training.benchmarks import run_benchmarks_from_pipeline_config
from training.data_prep import (
    build_static_df,
    load_dataset,
    prepare_shifted_dataset,
    to_nixtla_format,
)
from training.hurdle_nf import RedispatchNeuralForecast
from training.hurdle_runner import (
    _optimize_and_save_valid_threshold,
    optimize_hurdle_threshold,
)
from training.predict import (
    evaluate_models,
    log_predictions_to_wandb,
    predict_with_shift_correction,
    prepare_predictions_df,
)
from training.prediction_pipeline_new import (
    check_dataset_compatibility,
    get_dataset,
    identify_wandb_run,
    move_files_from_last_nonempty_dir,
    persist_to_wandb,
    save_lightgbm_feature_importance,
)
from training.prediction_pipeline_rolling_window import (
    _save_predictions_locally,
    classification_metrics_df,
    evaluation_pipeline,
    get_last_checkpoint_path,
    get_model_name,
    save_evaluation_locally,
)
from training.runner import CHECKPOINT_BEST_NAME
from training.train_pipeline import _normalize_tso_key, set_n_threads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Archive / model discovery
# ---------------------------------------------------------------------------

def read_models(model_path: Path):
    """Yield extracted model directories, mirroring prediction_pipeline_new.read_models."""
    archive_files = list(model_path.glob("*.tar.zst"))
    if archive_files:
        logger.info("Found %d .tar.zst archived models in %s", len(archive_files), model_path)
        for archive_path in archive_files:
            logger.info("Inflating model from %s", archive_path)
            model_dir = archive_path.with_suffix("").with_suffix("")
            model_dir.mkdir(exist_ok=True)
            model_dir = model_dir.resolve()
            with open(archive_path, "rb") as fh:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as reader:
                    with tarfile.open(fileobj=reader, mode="r|*") as tar:
                        tar.extractall(path=model_dir)
            move_files_from_last_nonempty_dir(model_dir, model_dir)
            yield model_dir
    else:
        logger.info(
            "No archived models found in %s – looking for directories.", model_path
        )
        for model_dir in model_path.iterdir():
            if model_dir.is_dir() and any(
                "nf_model" in d.name for d in model_dir.iterdir() if d.is_dir()
            ):
                yield (model_dir / "nf_model").resolve()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _dedupe_callbacks(callbacks: list[object]) -> list[object]:
    seen: set[type] = set()
    return [cb for cb in callbacks if not (type(cb) in seen or seen.add(type(cb)))]  # type: ignore[func-returns-value]


def _dedupe_model_callbacks(nf: NeuralForecast) -> None:
    for model in getattr(nf, "models", []):
        if isinstance(getattr(model, "hparams", None), dict):
            cbs = model.hparams.get("callbacks")
            if isinstance(cbs, list):
                model.hparams["callbacks"] = _dedupe_callbacks(cbs)
        trainer_kwargs = getattr(model, "trainer_kwargs", None)
        if isinstance(trainer_kwargs, dict) and isinstance(trainer_kwargs.get("callbacks"), list):
            trainer_kwargs["callbacks"] = _dedupe_callbacks(trainer_kwargs["callbacks"])


def load_hurdle_model(model_dir: Path, checkpoint_best: bool) -> RedispatchNeuralForecast:
    """Load a hurdle model checkpoint via :class:`RedispatchNeuralForecast`."""
    best_valid_ckpt = model_dir / CHECKPOINT_BEST_NAME
    best_valid_ckpt_tmp = best_valid_ckpt.with_suffix(".bkp")
    last_ckpt = get_last_checkpoint_path(model_dir)
    last_ckpt_tmp = last_ckpt.with_suffix(".bkp")
    try:
        if not checkpoint_best:
            if best_valid_ckpt.exists():
                best_valid_ckpt.rename(best_valid_ckpt_tmp)
            nf = RedispatchNeuralForecast.load(str(model_dir.resolve()))
        else:
            if best_valid_ckpt_tmp.exists():
                best_valid_ckpt_tmp.rename(best_valid_ckpt)
            if not best_valid_ckpt.exists():
                raise FileNotFoundError(f"Best-valid checkpoint not found in {model_dir}")
            last_ckpt.rename(last_ckpt_tmp)
            best_valid_ckpt.rename(last_ckpt)
            nf = RedispatchNeuralForecast.load(str(model_dir.resolve()))
        _dedupe_model_callbacks(nf)
        return nf
    except Exception as exc:
        raise RuntimeError(
            f"Error loading hurdle model from {model_dir} "
            f"(checkpoint_best={checkpoint_best})"
        ) from exc
    finally:
        if best_valid_ckpt_tmp.exists():
            best_valid_ckpt_tmp.rename(best_valid_ckpt)
        elif last_ckpt_tmp.exists():
            last_ckpt.rename(best_valid_ckpt)
            last_ckpt_tmp.rename(last_ckpt)


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def _require_config(run_config: dict, key: str):
    value = run_config.get(key)
    if value is None:
        raise KeyError(f"Missing required wandb config key: {key}")
    return value


def retrieve_raw_predictions_from_wandb(
    wandb_entity: str | None,
    wandb_project: str,
    model_name: str,
    tso_name: str,
    timestamp: str,
    best_checkpoint: bool,
) -> pd.DataFrame:
    """Retrieve the raw (logit + magnitude) hurdle predictions artifact from W&B.

    The artifact is named ``hurdle_raw_preds_{split}_{model}_{tso}_{timestamp}``.
    Returns an empty DataFrame when the artifact is not found.
    """
    split_label = "test_best_valid" if best_checkpoint else "test"
    artifact_name = f"hurdle_raw_preds_{split_label}_{model_name}_{tso_name}_{timestamp}"
    project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
    try:
        art = wandb.Api().artifact(f"{project_path}/{artifact_name}:latest")
        dl_path = Path(art.download(root=Path("wandb")))
        parquet_path = dl_path / f"{split_label}_hurdle_raw.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found at {parquet_path}")
        return pd.read_parquet(parquet_path)
    except Exception as exc:
        logger.debug(
            "Raw-predictions artifact '%s' not found in W&B: %s",
            artifact_name, exc,
        )
        return pd.DataFrame()


def log_raw_predictions_to_wandb(
    wandb_run: "wandb.Run",
    raw_preds: pd.DataFrame,
    model_name: str,
    tso_name: str,
    timestamp: str,
    best_checkpoint: bool,
) -> None:
    """Log raw (logit + magnitude) predictions as a W&B artifact."""
    split_label = "test_best_valid" if best_checkpoint else "test"
    artifact_name = f"hurdle_raw_preds_{split_label}_{model_name}_{tso_name}_{timestamp}"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="hurdle_raw_predictions",
        metadata={
            "tso": tso_name,
            "timestamp": timestamp,
            "best_checkpoint": best_checkpoint,
            "n_rows": len(raw_preds),
            "ds_min": str(raw_preds["ds"].min()),
            "ds_max": str(raw_preds["ds"].max()),
        },
    )
    with artifact.new_file(f"{split_label}_hurdle_raw.parquet", mode="wb") as fh:
        raw_preds.to_parquet(fh, index=False)
    wandb_run.log_artifact(artifact)
    logger.info("Logged raw hurdle predictions to W&B artifact '%s'", artifact_name)


def retrieve_theta_from_wandb(
    wandb_entity: str | None,
    wandb_project: str,
    model_name: str,
    tso_name: str,
    timestamp: str,
) -> dict | None:
    """Retrieve the theta optimisation JSON artifact from W&B.

    Returns ``None`` when not found.
    """
    artifact_name = f"hurdle_theta_{model_name}_{tso_name}_{timestamp}"
    project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
    try:
        art = wandb.Api().artifact(f"{project_path}/{artifact_name}:latest")
        dl_path = Path(art.download(root=Path("wandb")))
        json_path = dl_path / "hurdle_theta_opt.json"
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found at {json_path}")
        with open(json_path) as fh:
            return json.load(fh)
    except Exception as exc:
        logger.debug(
            "Theta artifact '%s' not found in W&B: %s", artifact_name, exc
        )
        return None


def log_theta_to_wandb(
    wandb_run: "wandb.Run",
    theta_result: dict,
    model_name: str,
    tso_name: str,
    timestamp: str,
) -> None:
    """Log the theta optimisation result as a W&B artifact."""
    artifact_name = f"hurdle_theta_{model_name}_{tso_name}_{timestamp}"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="hurdle_theta",
        metadata={
            "optimal_theta": theta_result.get("optimal_theta"),
            "mae": theta_result.get("mae"),
            "fbeta_mean": theta_result.get("fbeta_mean"),
            "combined_metric": theta_result.get("combined_metric"),
        },
    )
    with artifact.new_file("hurdle_theta_opt.json", mode="w") as fh:
        json.dump(theta_result, fh, indent=2)
    wandb_run.log_artifact(artifact)
    logger.info(
        "Logged theta result (θ=%.4f) to W&B artifact '%s'",
        theta_result.get("optimal_theta", float("nan")),
        artifact_name,
    )


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_hurdle_checkpoint(
    nf_model: RedispatchNeuralForecast,
    dataset: pd.DataFrame,
    run_config: dict,
    tso_name: str,
    add_calendar_features: bool,
    holidays_path: str | None,
) -> pd.DataFrame:
    """Run inference for one hurdle model checkpoint.

    Returns raw predictions with logit and magnitude columns intact.
    """
    shift_hours = int(_require_config(run_config, "shift_hours"))
    test_start = pd.Timestamp(_require_config(run_config, "test_start"))
    forecast_horizon = int(_require_config(run_config, "forecast_horizon"))

    shifted_dataset, future_cov_cols, _ = prepare_shifted_dataset(
        df=dataset,
        shift_hours=shift_hours,
        tso=tso_name,
        add_calendar=add_calendar_features,
        holidays_path=holidays_path,
    )
    return predict_with_shift_correction(
        nf=nf_model,
        df_unshifted=dataset,
        df_shifted=shifted_dataset,
        forecast_horizon=forecast_horizon,
        future_cov_cols=future_cov_cols,
        pred_start=test_start,
        pred_end=dataset["ds"].max(),
        holidays_path=holidays_path,
        shift_hours=shift_hours,
        static_df=build_static_df(),
        tso=tso_name,
    )


def _apply_hurdle_forecasts(
    raw_preds: pd.DataFrame,
    alias: str,
    theta: float | None,
) -> pd.DataFrame:
    """Add mean and (optionally) θ-gated point-forecast columns to *raw_preds*.

    * ``{alias}``       – ``sigmoid(logit) × relu(magnitude)``
    * ``{alias}_theta`` – ``1_{p̂ ≥ theta} × relu(magnitude)`` (when theta is not None)
    """
    logit_col = f"{alias}non_zero_logit"
    mag_col = f"{alias}magnitude"

    if logit_col not in raw_preds.columns or mag_col not in raw_preds.columns:
        logger.warning(
            "Hurdle columns '%s' / '%s' not found; available: %s",
            logit_col, mag_col, list(raw_preds.columns),
        )
        return raw_preds

    df = raw_preds.copy()
    p_hat = expit(df[logit_col].to_numpy(dtype=np.float64))
    m_hat = np.maximum(df[mag_col].to_numpy(dtype=np.float64), 0.0)

    df[alias] = p_hat * m_hat
    if theta is not None:
        df[f"{alias}_theta"] = np.where(p_hat >= theta, m_hat, 0.0)
        logger.info(
            "Applied θ=%.4f: %d/%d steps predict non-zero.",
            theta, int((p_hat >= theta).sum()), len(p_hat),
        )
    return df


def _optimize_theta_at_inference(
    nf_model: RedispatchNeuralForecast,
    dataset: pd.DataFrame,
    run_config: dict,
    tso_name: str,
    add_calendar_features: bool,
    holidays_path: str | None,
    alias: str,
    output_path: Path,
    fbeta2_target: float,
    lambda_: float,
    n_grid: int,
) -> float | None:
    """Optimise θ on the validation split when no pre-saved JSON is available.

    Runs inference on [valid_start, test_start) from *run_config*, minimises
    the combined MAE + λ·max(0, Fβ2_target − Fβ2) metric, and writes the
    result to *output_path*.

    Returns the optimal theta, or ``None`` on failure.
    """
    valid_start_raw = run_config.get("valid_start")
    test_start_raw = run_config.get("test_start")
    shift_hours = int(run_config.get("shift_hours", 0))
    forecast_horizon = int(run_config.get("forecast_horizon", 24))

    if valid_start_raw is None or test_start_raw is None:
        logger.warning(
            "Cannot optimise θ at inference: 'valid_start' / 'test_start' "
            "missing from run_config."
        )
        return None

    valid_start = pd.Timestamp(valid_start_raw)
    test_start = pd.Timestamp(test_start_raw)
    valid_end = test_start - pd.Timedelta(hours=1)

    if shift_hours > 0:
        valid_start = valid_start - pd.Timedelta(hours=shift_hours)

    logger.info(
        "Optimising θ on validation period [%s, %s] for alias=%s",
        valid_start, valid_end, alias,
    )

    shifted_dataset, future_cov_cols, _ = prepare_shifted_dataset(
        df=dataset,
        shift_hours=shift_hours,
        tso=tso_name,
        add_calendar=add_calendar_features,
        holidays_path=holidays_path,
    )

    val_preds = predict_with_shift_correction(
        nf=nf_model,
        df_unshifted=dataset,
        df_shifted=shifted_dataset,
        forecast_horizon=forecast_horizon,
        future_cov_cols=future_cov_cols,
        pred_start=valid_start,
        pred_end=valid_end,
        holidays_path=holidays_path,
        shift_hours=shift_hours,
        static_df=build_static_df(),
        tso=tso_name,
    )

    if val_preds.empty:
        logger.warning("Empty validation predictions for θ optimisation – skipping.")
        return None

    val_preds = val_preds.merge(
        dataset[["unique_id", "ds", "y"]],
        on=["unique_id", "ds"],
        how="left",
    ).dropna(subset=["y"])

    if val_preds.empty:
        logger.warning("No ground-truth matched validation predictions – skipping θ optimisation.")
        return None

    try:
        result = optimize_hurdle_threshold(
            preds_df=val_preds,
            alias=alias,
            fbeta2_target=fbeta2_target,
            lambda_=lambda_,
            n_grid=n_grid,
        )
    except Exception as exc:
        logger.warning("θ optimisation failed: %s", exc, exc_info=True)
        return None

    result["alias"] = alias
    result["tso"] = tso_name
    result["valid_start"] = str(valid_start)
    result["valid_end"] = str(valid_end)
    result["computed_at_inference"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(
        "θ optimised at inference: θ=%.4f  combined=%.4f  MAE=%.4f  Fβ=%.4f → %s",
        result["optimal_theta"],
        result["combined_metric"],
        result["mae"],
        result["fbeta_mean"],
        output_path,
    )
    return float(result["optimal_theta"])


def _load_theta_json(json_path: Path) -> float | None:
    """Load θ from a JSON file.  Returns ``None`` if missing or unreadable."""
    if not json_path.exists():
        return None
    try:
        with open(json_path) as fh:
            return float(json.load(fh)["optimal_theta"])
    except Exception as exc:
        logger.warning("Could not read theta JSON at %s: %s", json_path, exc)
        return None


# ---------------------------------------------------------------------------
# Per-model prediction flow
# ---------------------------------------------------------------------------

def run_predictions_for_hurdle_model(
    *,
    model_dir: Path,
    model_dataset: pd.DataFrame,
    run_config: dict,
    run_id: str,
    tso_name: str,
    dataset_compat_status: str,
    benchmarks_df: pd.DataFrame,
    wandb_project: str,
    wandb_entity: str | None,
    output_dir: Path,
    add_calendar_features: bool,
    holidays_path: str | None,
    best_checkpoint: bool,
    start_date: pd.Timestamp | None,
    fbeta2_target: float = 0.5,
    theta_lambda: float = 0.5,
    theta_n_grid: int = 200,
) -> pd.DataFrame:
    """Fetch or compute hurdle predictions for one checkpoint type, evaluate, and persist.

    Workflow
    --------
    1. Try to retrieve **raw** predictions (logit + magnitude) from W&B.
    2. If missing (or extended dataset), run inference from the checkpoint and
       log the raw artifact to W&B.
    3. Resolve θ:
       a. Look for ``hurdle_theta_opt_{alias}.json`` next to the model checkpoint.
       b. If not found locally, try the W&B theta artifact.
       c. If still not found, run θ optimisation on the validation split (requires
          loading the model if not already loaded) and save + log the result.
    4. Apply mean and θ-gated forecasts.
    5. Save predictions locally (raw Parquet) and log processed predictions.
    6. Evaluate and persist evaluation CSV + W&B table.

    Returns
    -------
    pd.DataFrame
        Processed predictions (mean + theta columns, logit/magnitude dropped).
    """
    timestamp = _require_config(run_config, "date_time")
    shift_hours = int(_require_config(run_config, "shift_hours"))
    model_alias = _require_config(run_config, "model_alias")
    model_name = get_model_name(model_dir)
    tso_norm = _normalize_tso_key(tso_name)
    split_label = "test_best_valid" if best_checkpoint else "test"

    # ── θ JSON: canonical location written by training runner ─────────────────
    # model_dir is the extracted checkpoint directory; the JSON is written next
    # to the parent (one level up) by hurdle_runner when using archive layout,
    # or inside model_dir itself when run directly.  We probe both.
    theta_json_candidates = [
        model_dir / f"hurdle_theta_opt_{model_alias}.json",
        model_dir.parent / f"hurdle_theta_opt_{model_alias}.json",
    ]

    with wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=run_id,
        resume="allow",
        mode="online",
    ) as wandb_run:
        # ── Step 1: raw predictions ───────────────────────────────────────────
        nf: RedispatchNeuralForecast | None = None

        raw_preds = retrieve_raw_predictions_from_wandb(
            wandb_entity, wandb_project,
            model_name=model_name,
            tso_name=tso_norm,
            timestamp=timestamp,
            best_checkpoint=best_checkpoint,
        )

        needs_inference = raw_preds.empty or dataset_compat_status == "extended"

        if needs_inference:
            if raw_preds.empty:
                logger.info(
                    "Raw predictions not found in W&B – running inference for %s (%s)",
                    model_alias, split_label,
                )
            else:
                # Extended dataset: keep existing predictions and append new tail
                last_pred_ts = raw_preds["ds"].max()
                if model_dataset["ds"].max() <= last_pred_ts:
                    needs_inference = False  # already fully covered
                else:
                    logger.info(
                        "Extending hurdle predictions for %s from %s",
                        model_alias,
                        last_pred_ts + pd.Timedelta(hours=1),
                    )

            if needs_inference:
                nf = load_hurdle_model(model_dir, checkpoint_best=best_checkpoint)
                if raw_preds.empty:
                    raw_preds = predict_hurdle_checkpoint(
                        nf_model=nf,
                        dataset=model_dataset,
                        run_config=run_config,
                        tso_name=tso_norm,
                        add_calendar_features=add_calendar_features,
                        holidays_path=holidays_path,
                    )
                else:
                    # Append tail
                    tail_start = raw_preds["ds"].max() + pd.Timedelta(hours=1)
                    tail_raw = predict_hurdle_checkpoint(
                        nf_model=nf,
                        dataset=model_dataset,
                        run_config={**run_config, "test_start": str(tail_start)},
                        tso_name=tso_norm,
                        add_calendar_features=add_calendar_features,
                        holidays_path=holidays_path,
                    )
                    raw_preds = (
                        pd.concat([raw_preds, tail_raw], ignore_index=True)
                        .drop_duplicates(subset=["unique_id", "ds"], keep="last")
                        .sort_values(["unique_id", "ds"])
                        .reset_index(drop=True)
                    )

                # Log raw artifact to W&B
                raw_with_y = raw_preds.merge(
                    model_dataset[["unique_id", "ds", "y"]],
                    on=["unique_id", "ds"],
                    how="left",
                )
                log_raw_predictions_to_wandb(
                    wandb_run, raw_with_y, model_name, tso_norm, timestamp, best_checkpoint,
                )

        # ── Step 2: resolve θ ─────────────────────────────────────────────────
        theta: float | None = None

        # 2a. Local JSON next to checkpoint
        for candidate in theta_json_candidates:
            theta = _load_theta_json(candidate)
            if theta is not None:
                logger.info("Loaded θ=%.4f from %s", theta, candidate)
                break

        # 2b. W&B artifact
        if theta is None:
            theta_data = retrieve_theta_from_wandb(
                wandb_entity, wandb_project, model_name, tso_norm, timestamp,
            )
            if theta_data is not None:
                theta = float(theta_data["optimal_theta"])
                logger.info("Loaded θ=%.4f from W&B artifact", theta)

        # 2c. Compute at inference time on the validation split
        if theta is None:
            logger.info(
                "θ JSON not found locally or in W&B – optimising on validation set for %s",
                model_alias,
            )
            # Use the output_dir as the fallback save location
            theta_save_path = output_dir / f"hurdle_theta_opt_{model_alias}_{tso_norm}_{timestamp}.json"

            if nf is None:
                nf = load_hurdle_model(model_dir, checkpoint_best=best_checkpoint)

            theta = _optimize_theta_at_inference(
                nf_model=nf,
                dataset=model_dataset,
                run_config=run_config,
                tso_name=tso_norm,
                add_calendar_features=add_calendar_features,
                holidays_path=holidays_path,
                alias=model_alias,
                output_path=theta_save_path,
                fbeta2_target=fbeta2_target,
                lambda_=theta_lambda,
                n_grid=theta_n_grid,
            )
            # Log theta to W&B so it is available for future runs
            if theta is not None:
                try:
                    with open(theta_save_path) as fh:
                        theta_result = json.load(fh)
                    log_theta_to_wandb(wandb_run, theta_result, model_name, tso_norm, timestamp)
                except Exception as exc:
                    logger.warning("Could not log theta to W&B: %s", exc)

        # ── Step 3: apply hurdle post-processing ──────────────────────────────
        processed = _apply_hurdle_forecasts(raw_preds, model_alias, theta)

        # Drop logit/magnitude columns before evaluation
        logit_col = f"{model_alias}non_zero_logit"
        mag_col = f"{model_alias}magnitude"
        keep_cols = [c for c in processed.columns if c not in (logit_col, mag_col)]
        processed_clean = processed[keep_cols].copy()

        # Merge ground truth
        processed_with_y = processed_clean.merge(
            model_dataset[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="left",
        )

        # ── Step 4: log processed predictions to W&B ─────────────────────────
        log_predictions_to_wandb(
            wandb_run,
            processed_with_y,
            split_name=split_label,
            shift_hours=shift_hours,
            model_alias=model_alias,
            tso=tso_norm,
            timestamp=timestamp,
            log_table=False,
        )

        # ── Step 5: save processed predictions locally ────────────────────────
        _save_predictions_locally(
            output_dir,
            processed_with_y,
            model_name,
            tso_norm,
            timestamp=timestamp,
            best_checkpoint=best_checkpoint,
        )

        # ── Step 6: evaluate ──────────────────────────────────────────────────
        evaluation_df = evaluation_pipeline(
            model_dataset=model_dataset,
            predictions=processed_clean,
            benchmarks_df=benchmarks_df,
            start_date=start_date,
            beta=2.0,
        )
        save_evaluation_locally(
            output_dir,
            evaluation_df,
            model_name,
            tso_norm,
            timestamp=timestamp,
            best_checkpoint=best_checkpoint,
        )
        persist_to_wandb(
            wandb_run,
            evaluation_df,
            model_name,
            tso_norm,
            best_checkpoint=best_checkpoint,
            timestamp=timestamp,
        )

    return processed_clean


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    root_model_path: Path,
    dataset_root_dir: Path,
    wandb_project: str,
    wandb_entity: str | None,
    output_dir: Path,
    add_calendar_features: bool,
    holidays_path: str | None,
    benchmark_config_path: Path | None = None,
    enabled_benchmarks: list[str] | None = None,
    test_best_checkpoint: bool = False,
    run_predictions: bool = True,
    start_date: pd.Timestamp | None = None,
    fbeta2_target: float = 0.5,
    theta_lambda: float = 0.5,
    theta_n_grid: int = 200,
) -> None:
    """Run the hurdle single-window prediction and evaluation pipeline.

    Parameters
    ----------
    root_model_path : Path
        Directory containing model archives (``.tar.zst``) or model sub-dirs.
    dataset_root_dir : Path
        Root directory where pre-processed Parquet datasets are stored.
    wandb_project, wandb_entity : str
        W&B project / entity for run resumption and logging.
    output_dir : Path
        Directory to write predictions, evaluations, and theta JSONs.
    add_calendar_features : bool
        Whether to add calendar features when preparing the input dataset.
    holidays_path : str or None
        Path override for the holidays CSV.
    benchmark_config_path : Path or None
        Optional YAML with benchmark hyper-parameters.
    enabled_benchmarks : list[str] or None
        Benchmark models to compute.  ``None`` runs all available.
    test_best_checkpoint : bool
        Also evaluate the best-validation checkpoint alongside the last one.
    run_predictions : bool
        Set to ``False`` to skip predictions and evaluation (benchmarks still run).
    start_date : pd.Timestamp or None
        If set, evaluation metrics are computed only from this date onwards.
    fbeta2_target : float
        Target Fβ2 recall score used in θ optimisation penalty.
    theta_lambda : float
        Penalty weight λ for the Fβ2 shortfall in θ optimisation.
    theta_n_grid : int
        Number of candidate threshold values swept during θ optimisation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # File logging
    log_file = output_dir / "prediction_pipeline_hurdle.log"
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(fh)
    logger.info("=" * 80)
    logger.info("Starting hurdle single-window prediction pipeline")
    logger.info("root_model_path = %s", root_model_path)
    logger.info("=" * 80)

    model_dirs = list(read_models(root_model_path))
    if not model_dirs:
        logger.warning("No models found in %s – exiting.", root_model_path)
        return

    # ── Collect per-model metadata ────────────────────────────────────────────
    model_metas: list[dict] = []
    for model_dir in model_dirs:
        run_info = identify_wandb_run(model_dir, wandb_entity, wandb_project)
        run_config = run_info["config"]
        tso_name = str(_require_config(run_config, "tso")).replace(" ", "_")
        dataset_name = str(_require_config(run_config, "dataset_contents")).replace(", ", "_")
        model_metas.append(
            {
                "model_dir": model_dir,
                "run_info": run_info,
                "tso_name": tso_name,
                "dataset_name": dataset_name,
            }
        )

    # ── Group by (tso, dataset) for shared benchmark computation ─────────────
    dataset_keys: dict[tuple[str, str], list[dict]] = {}
    for meta in model_metas:
        key = (meta["tso_name"], meta["dataset_name"])
        dataset_keys.setdefault(key, []).append(meta)

    for (tso_name, dataset_name), group_metas in dataset_keys.items():
        model_dataset = get_dataset(dataset_root_dir, tso_name, dataset_name)
        supplied_dataset_path = dataset_root_dir / f"{dataset_name}_{tso_name}.parquet"
        representative_config = group_metas[0]["run_info"]["config"]
        tso_norm = _normalize_tso_key(tso_name)

        # ── Benchmarks (once per dataset) ─────────────────────────────────────
        if run_predictions:
            logger.info("Computing benchmarks for tso=%s dataset=%s", tso_norm, dataset_name)
            benchmarks_df, benchmark_metadata = run_benchmarks_from_pipeline_config(
                dataset=model_dataset,
                run_config=representative_config,
                tso_name=tso_norm,
                enabled_benchmarks=enabled_benchmarks,
                parameter_yaml_file_path=benchmark_config_path,
            )

            if not enabled_benchmarks or "lightgbm" in enabled_benchmarks:
                for key in [k for k in benchmark_metadata if "lightgbm" in k]:
                    benchmark_metadata = save_lightgbm_feature_importance(
                        benchmark_metadata, output_dir, key
                    )

            benchmarks_path = output_dir / f"benchmarks_{tso_norm}_{dataset_name}.csv"
            benchmarks_df.assign(tso=tso_norm, dataset=dataset_name).to_csv(
                benchmarks_path, index=False
            )
            logger.info("Saved benchmarks to %s", benchmarks_path)

            bm_meta_path = output_dir / f"benchmark_metadata_{tso_norm}_{dataset_name}.json"
            with open(bm_meta_path, "w") as fh_meta:
                json.dump(benchmark_metadata, fh_meta, indent=2, default=str)
        else:
            benchmarks_df = pd.DataFrame()
            logger.info("Skipping benchmarks (run_predictions=False).")

        # ── Per-model loop ────────────────────────────────────────────────────
        last_preds_list: list[pd.DataFrame] = []
        best_preds_list: list[pd.DataFrame] = []

        for meta in group_metas:
            model_dir = meta["model_dir"]
            run_config = meta["run_info"]["config"]
            run_id = meta["run_info"]["run_id"]
            model_alias = _require_config(run_config, "model_alias")

            logger.info("Processing hurdle model '%s' in %s", model_alias, model_dir)

            # Dataset compatibility check
            dataset_compat_status, _ = check_dataset_compatibility(
                run_config=run_config,
                supplied_dataset=model_dataset,
                supplied_dataset_path=supplied_dataset_path,
            )
            if dataset_compat_status == "incompatible":
                logger.warning(
                    "Skipping model %s: supplied dataset is incompatible with the "
                    "training dataset. A new training run is recommended.",
                    model_alias,
                )
                continue

            if not run_predictions:
                logger.info("Skipping predictions for %s (run_predictions=False).", model_alias)
                continue

            # Last checkpoint
            preds = run_predictions_for_hurdle_model(
                model_dir=model_dir,
                model_dataset=model_dataset,
                run_config=run_config,
                run_id=run_id,
                tso_name=tso_norm,
                dataset_compat_status=dataset_compat_status,
                benchmarks_df=benchmarks_df,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                output_dir=output_dir,
                add_calendar_features=add_calendar_features,
                holidays_path=holidays_path,
                best_checkpoint=False,
                start_date=start_date,
                fbeta2_target=fbeta2_target,
                theta_lambda=theta_lambda,
                theta_n_grid=theta_n_grid,
            )
            last_preds_list.append(preds)

            # Best-validation checkpoint (optional)
            if test_best_checkpoint:
                best_preds = run_predictions_for_hurdle_model(
                    model_dir=model_dir,
                    model_dataset=model_dataset,
                    run_config=run_config,
                    run_id=run_id,
                    tso_name=tso_norm,
                    dataset_compat_status=dataset_compat_status,
                    benchmarks_df=benchmarks_df,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    output_dir=output_dir,
                    add_calendar_features=add_calendar_features,
                    holidays_path=holidays_path,
                    best_checkpoint=True,
                    start_date=start_date,
                    fbeta2_target=fbeta2_target,
                    theta_lambda=theta_lambda,
                    theta_n_grid=theta_n_grid,
                )
                best_preds_list.append(best_preds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def prepare_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-window prediction pipeline for hurdle (zero-inflated) models."
    )
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model directory or path to a .tar.zst archive.",
    )
    p.add_argument(
        "--dataset-root-dir",
        type=str,
        required=True,
        help="Directory where pre-processed Parquet datasets are stored.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for predictions, evaluations, and theta JSONs.",
    )
    p.add_argument("--wandb-project", type=str, required=True, help="W&B project name.")
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B entity name.")
    p.add_argument(
        "--test-best-checkpoint",
        action="store_true",
        help="Also predict and evaluate the best-validation checkpoint.",
    )
    p.add_argument("--no-calendar", action="store_true", help="Skip calendar features.")
    p.add_argument("--holidays-path", default=None, help="Override for the holidays CSV path.")
    p.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help=(
            "Benchmark models to compute.  Accepted: naive_seasonal, "
            "linear_regression, gradient_boosted_trees, auto_arima, "
            "seasonal_regression.  Defaults to all."
        ),
    )
    p.add_argument(
        "--benchmark-config-path",
        type=str,
        default=None,
        help="YAML file with benchmark hyper-parameters.",
    )
    p.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Skip predictions/evaluation (benchmarks still run).",
    )
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Evaluation start date (ISO format, e.g. '2024-06-01').",
    )
    p.add_argument(
        "--hurdle-theta-fbeta2-target",
        type=float,
        default=0.5,
        help="Target Fβ2 score used in the θ optimisation penalty (default: 0.5).",
    )
    p.add_argument(
        "--hurdle-theta-lambda",
        type=float,
        default=0.5,
        help="Penalty weight λ for the Fβ2 shortfall in θ optimisation (default: 0.5).",
    )
    p.add_argument(
        "--hurdle-theta-n-grid",
        type=int,
        default=200,
        help="Number of candidate threshold values in (0, 1) (default: 200).",
    )
    p.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="Number of PyTorch threads.  Defaults to all available.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = prepare_args()
    if args.n_threads is not None:
        set_n_threads(args.n_threads)
    main(
        root_model_path=Path(args.model_path),
        dataset_root_dir=Path(args.dataset_root_dir),
        output_dir=Path(args.output_dir),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        add_calendar_features=not args.no_calendar,
        holidays_path=args.holidays_path,
        enabled_benchmarks=args.benchmarks,
        benchmark_config_path=Path(args.benchmark_config_path) if args.benchmark_config_path else None,
        test_best_checkpoint=args.test_best_checkpoint,
        run_predictions=not args.skip_predictions,
        start_date=pd.Timestamp(args.start_date) if args.start_date else None,
        fbeta2_target=args.hurdle_theta_fbeta2_target,
        theta_lambda=args.hurdle_theta_lambda,
        theta_n_grid=args.hurdle_theta_n_grid,
    )
