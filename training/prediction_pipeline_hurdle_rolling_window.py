"""Rolling-window prediction & evaluation pipeline for hurdle (zero-inflated) models.

Separate from ``prediction_pipeline_rolling_window.py`` to keep hurdle-specific
logic self-contained.  Key differences:

* Loads checkpoints via :class:`~training.hurdle_nf.RedispatchNeuralForecast`
  which correctly handles hurdle model state dicts.
* Always re-runs inference from the saved checkpoint (the W&B artifacts logged
  by the trainer clip the logit column, making them unsuitable for computing the
  θ-gated forecast).  A local Parquet cache (raw logit + magnitude preserved)
  is used to skip inference on subsequent runs when ``--force`` is not set.
* Produces **two point-forecast columns** per model alias:

    ``{alias}``       – mean: ``sigmoid(logit) × relu(magnitude)``
    ``{alias}_theta`` – threshold-gated: ``1_{p̂ ≥ θ} × relu(magnitude)``

  where ``θ`` is loaded from the JSON written by
  :func:`~training.hurdle_runner._optimize_and_save_valid_threshold` during
  training.  If no JSON is found the theta column is omitted silently.

* Saves raw predictions (logit + magnitude preserved) as Parquet locally.
* Logs processed predictions (point forecasts only) to W&B by resuming the
  training run for each window.
* Evaluates each forecast type and saves a combined evaluation CSV.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tarfile
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
    classify_covariates,
    load_dataset,
    prepare_shifted_dataset,
    to_nixtla_format,
)
from training.hurdle_nf import RedispatchNeuralForecast, _HURDLE_MODEL_REGISTRY
from training.hurdle_runner import _add_hurdle_point_forecast, optimize_hurdle_threshold
from training.predict import (
    evaluate_models,
    log_predictions_to_wandb,
    predict_with_shift_correction,
    prepare_predictions_df,
)
from training.prediction_pipeline_new import save_lightgbm_feature_importance
from training.prediction_pipeline_rolling_window import (
    _require_config,
    _save_predictions_locally,
    classification_metrics_df,
    evaluation_pipeline,
    get_dataset,
    get_last_checkpoint_path,
    get_model_name,
    identify_wandb_run,
    move_files_from_last_nonempty_dir,
    read_models,
    save_evaluation_locally,
)
# Shared hurdle W&B / theta helpers
from training.prediction_pipeline_hurdle_new import (
    _load_theta_json,
    _optimize_theta_at_inference,
    log_raw_predictions_to_wandb,
    log_theta_to_wandb,
    retrieve_raw_predictions_from_wandb as _retrieve_raw_preds_wandb,
    retrieve_theta_from_wandb,
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
    """Load a hurdle model checkpoint using :class:`RedispatchNeuralForecast`.

    Uses the same checkpoint-swapping logic as the standard pipeline:
    when *checkpoint_best* is ``False`` the best-valid checkpoint is hidden so
    that ``NeuralForecast.load`` picks up the last checkpoint, and vice-versa.

    Parameters
    ----------
    model_dir : Path
        Extracted model directory containing ``*.ckpt`` files.
    checkpoint_best : bool
        ``True`` to load the best-validation checkpoint.

    Returns
    -------
    RedispatchNeuralForecast
        Loaded (fitted) hurdle model wrapper.
    """
    best_valid_checkpoint = model_dir / CHECKPOINT_BEST_NAME
    best_valid_checkpoint_tmp = best_valid_checkpoint.with_suffix(".bkp")
    last_checkpoint_path = get_last_checkpoint_path(model_dir)
    last_checkpoint_tmp = last_checkpoint_path.with_suffix(".bkp")

    try:
        if not checkpoint_best:
            if best_valid_checkpoint.exists():
                best_valid_checkpoint.rename(best_valid_checkpoint_tmp)
            nf_model = RedispatchNeuralForecast.load(str(model_dir.resolve()))
        else:
            if best_valid_checkpoint_tmp.exists():
                best_valid_checkpoint_tmp.rename(best_valid_checkpoint)
            if not best_valid_checkpoint.exists():
                raise FileNotFoundError(
                    f"Best-valid checkpoint not found in {model_dir}"
                )
            last_checkpoint_path.rename(last_checkpoint_tmp)
            best_valid_checkpoint.rename(last_checkpoint_path)
            nf_model = RedispatchNeuralForecast.load(str(model_dir.resolve()))

        _dedupe_model_callbacks(nf_model)
        return nf_model

    except Exception as exc:
        raise RuntimeError(
            f"Error loading hurdle model from {model_dir} "
            f"(checkpoint_best={checkpoint_best})"
        ) from exc

    finally:
        # Restore original checkpoint layout regardless of success/failure.
        if best_valid_checkpoint_tmp.exists():
            best_valid_checkpoint_tmp.rename(best_valid_checkpoint)
        elif last_checkpoint_tmp.exists():
            last_checkpoint_path.rename(best_valid_checkpoint)
            last_checkpoint_tmp.rename(last_checkpoint_path)


# ---------------------------------------------------------------------------
# Hurdle post-processing helpers
# ---------------------------------------------------------------------------


def _apply_hurdle_forecasts(
    raw_preds: pd.DataFrame,
    alias: str,
    theta: float | None,
) -> pd.DataFrame:
    """Add hurdle point-forecast columns to *raw_preds* (in-place copy).

    Adds:

    * ``{alias}``       – mean forecast: ``sigmoid(logit) × relu(magnitude)``
    * ``{alias}_theta`` – threshold-gated forecast (only if *theta* is given):
      ``1_{p̂ ≥ theta} × relu(magnitude)``

    Parameters
    ----------
    raw_preds : pd.DataFrame
        Raw predictions with ``{alias}-non_zero_logit`` and
        ``{alias}-magnitude`` columns.
    alias : str
        Model alias (column-name prefix).
    theta : float or None
        Optimal probability threshold.  When ``None`` only the mean column is
        added.

    Returns
    -------
    pd.DataFrame
        Copy of *raw_preds* with point-forecast column(s) added.
    """
    logit_col = f"{alias}non_zero_logit"
    mag_col = f"{alias}magnitude"

    if logit_col not in raw_preds.columns or mag_col not in raw_preds.columns:
        logger.warning(
            "Hurdle columns '%s' / '%s' not found in predictions; "
            "available: %s",
            logit_col, mag_col, list(raw_preds.columns),
        )
        return raw_preds

    df = raw_preds.copy()

    logit_vals = df[logit_col].to_numpy(dtype=np.float64)
    mag_vals = df[mag_col].to_numpy(dtype=np.float64)

    p_hat = expit(logit_vals)           # sigmoid
    m_hat = np.maximum(mag_vals, 0.0)   # relu

    # Mean forecast
    df[alias] = p_hat * m_hat

    # Threshold-gated forecast
    if theta is not None:
        df[f"{alias}_theta"] = np.where(p_hat >= theta, m_hat, 0.0)
        logger.info(
            "Applied θ=%.4f gate: %d/%d steps predict non-zero.",
            theta,
            int((p_hat >= theta).sum()),
            len(p_hat),
        )

    return df


def _raw_cache_path(
    output_dir: Path,
    model_name: str,
    tso_name: str,
    timestamp: str,
    window_index: int,
    best_checkpoint: bool,
) -> Path:
    """Return the path for the per-window **raw** (logit + magnitude) cache."""
    stem = f"hurdle_raw_preds_{model_name}_{tso_name}_{timestamp}_window{window_index}"
    if best_checkpoint:
        stem += "_best_checkpoint"
    return output_dir / f"{stem}.parquet"


# ---------------------------------------------------------------------------
# Per-window prediction
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

    Returns raw predictions (logit + magnitude columns, **not** clipped).
    """
    shift_hours = int(_require_config(run_config, "shift_hours"))
    test_start = pd.Timestamp(_require_config(run_config, "test_start"))
    forecast_horizon = int(_require_config(run_config, "forecast_horizon"))

    if run_config.get("test_end") is not None:
        test_end = pd.Timestamp(run_config["test_end"]) - pd.Timedelta(hours=1)
    elif run_config.get("model_type", "single_window") == "rolling_window":
        n_test_months = run_config.get("n_test_months", 1)
        test_end = test_start + pd.DateOffset(months=n_test_months) - pd.Timedelta(hours=1)
    else:
        test_end = dataset["ds"].max()

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
        pred_end=test_end,
        holidays_path=holidays_path,
        shift_hours=shift_hours,
        static_df=build_static_df(),
        tso=tso_name,
    )


def run_hurdle_window(
    model_dataset: pd.DataFrame,
    add_calendar_features: bool,
    holidays_path: str | None,
    meta: dict,
    output_dir: Path,
    wandb_project: str,
    wandb_entity: str | None,
    best_checkpoint: bool = False,
    force: bool = False,
    fbeta2_target: float = 0.5,
    theta_lambda: float = 0.5,
    theta_n_grid: int = 200,
) -> pd.DataFrame:
    """Predict, post-process, cache, log, and return predictions for one window.

    Workflow
    --------
    1. Look for a local raw-predictions cache (``hurdle_raw_preds_*.parquet``).
       Skip inference if found and *force* is ``False``.
    2. Otherwise load the model checkpoint and run inference, then save raw
       cache and log the raw artifact to W&B.
    3. Resolve θ (three-tier lookup):
       a. Local JSON written by training runner
          (``{window_dir}/hurdle_theta_opt_{alias}_w{wi}.json``).
       b. W&B theta artifact.
       c. Compute at inference on the validation split – saves JSON locally and
          logs to W&B.
    4. Apply mean and θ-gated forecasts via ``_apply_hurdle_forecasts``.
    5. Log *processed* predictions (point forecasts, no raw logit/magnitude)
       to W&B by resuming the training run.
    6. Save *processed* predictions locally as ``predictions_*.parquet``.

    Returns
    -------
    pd.DataFrame
        Processed predictions DataFrame including ``{alias}`` and optionally
        ``{alias}_theta`` columns (but **not** logit/magnitude).
    """
    window_index = meta["window_index"]
    model_dir = meta["model_dir"]
    run_config = meta["run_info"]["config"]
    timestamp = _require_config(run_config, "date_time")
    model_name = meta["model_name"]
    tso_name = _normalize_tso_key(meta["tso_name"])
    shift_hours = int(_require_config(run_config, "shift_hours"))
    model_alias = _require_config(run_config, "model_alias")

    # ── Raw predictions cache ─────────────────────────────────────────────────
    cache_path = _raw_cache_path(
        output_dir, model_name, tso_name, timestamp, window_index, best_checkpoint
    )

    nf_model: RedispatchNeuralForecast | None = None

    with wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=meta["run_info"]["run_id"],
        resume="allow",
        mode="online",
    ) as wandb_run:
        if not force and cache_path.exists():
            logger.info(
                "Window %d: loading raw predictions from cache %s",
                window_index, cache_path,
            )
            raw_preds = pd.read_parquet(cache_path)
        else:
            logger.info("Window %d: running inference from checkpoint %s", window_index, model_dir)
            nf_model = load_hurdle_model(model_dir, checkpoint_best=best_checkpoint)
            raw_preds = predict_hurdle_checkpoint(
                nf_model=nf_model,
                dataset=model_dataset,
                run_config=run_config,
                tso_name=tso_name,
                add_calendar_features=add_calendar_features,
                holidays_path=holidays_path,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            raw_preds.to_parquet(cache_path, index=False)
            logger.info("Window %d: raw predictions cached to %s", window_index, cache_path)

        # ── Three-tier θ lookup ───────────────────────────────────────────────────
        # Tier 1: local JSON written by training runner
        # model_dir = {root}/window_{wi}/model_extraction_{alias}/
        window_dir = model_dir.parent
        theta_json_candidates = [
            window_dir / f"hurdle_theta_opt_{model_alias}_w{window_index}.json",
            model_dir / f"hurdle_theta_opt_{model_alias}_w{window_index}.json",
            model_dir / f"hurdle_theta_opt_{model_alias}.json",
        ]
        theta: float | None = None
        for candidate in theta_json_candidates:
            theta = _load_theta_json(candidate)
            if theta is not None:
                logger.info("Window %d: loaded θ=%.4f from %s", window_index, theta, candidate)
                break

        # Log raw predictions artifact if we just computed them
        if nf_model is not None:
            raw_with_y = raw_preds.merge(
                model_dataset[["unique_id", "ds", "y"]],
                on=["unique_id", "ds"],
                how="left",
            )
            log_raw_predictions_to_wandb(
                wandb_run, raw_with_y, model_name, tso_name, timestamp, best_checkpoint,
            )

        # Tier 2: W&B theta artifact
        if theta is None:
            theta_data = retrieve_theta_from_wandb(
                wandb_entity, wandb_project, model_name, tso_name, timestamp,
            )
            if theta_data is not None:
                theta = float(theta_data["optimal_theta"])
                logger.info("Window %d: loaded θ=%.4f from W&B artifact", window_index, theta)

        # Tier 3: compute at inference on the validation split
        if theta is None:
            logger.info(
                "Window %d: θ not found – optimising on validation set for %s",
                window_index, model_alias,
            )
            theta_save_path = (
                output_dir
                / f"hurdle_theta_opt_{model_alias}_{tso_name}_{timestamp}_w{window_index}.json"
            )
            if nf_model is None:
                nf_model = load_hurdle_model(model_dir, checkpoint_best=best_checkpoint)
            theta = _optimize_theta_at_inference(
                nf_model=nf_model,
                dataset=model_dataset,
                run_config=run_config,
                tso_name=tso_name,
                add_calendar_features=add_calendar_features,
                holidays_path=holidays_path,
                alias=model_alias,
                output_path=theta_save_path,
                fbeta2_target=fbeta2_target,
                lambda_=theta_lambda,
                n_grid=theta_n_grid,
            )
            if theta is not None:
                try:
                    with open(theta_save_path) as fh:
                        theta_result = json.load(fh)
                    log_theta_to_wandb(wandb_run, theta_result, model_name, tso_name, timestamp)
                except Exception as exc:
                    logger.warning("Window %d: could not log theta to W&B: %s", window_index, exc)

        # ── Apply hurdle post-processing ──────────────────────────────────────
        processed = _apply_hurdle_forecasts(raw_preds, model_alias, theta)

        # Drop raw logit & magnitude before evaluation/logging
        logit_col = f"{model_alias}non_zero_logit"
        mag_col = f"{model_alias}magnitude"
        forecast_cols = [c for c in processed.columns if c not in (logit_col, mag_col)]
        processed_clean = processed[forecast_cols].copy()

        # Merge ground truth for logging
        processed_with_y = processed_clean.merge(
            model_dataset[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="left",
        )

        # ── Log processed predictions to W&B ─────────────────────────────────
        split_label = (
            f"test_best_valid_window{window_index}"
            if best_checkpoint
            else f"test_window{window_index}"
        )
        log_predictions_to_wandb(
            wandb_run,
            processed_with_y,
            split_name=split_label,
            shift_hours=shift_hours,
            model_alias=model_alias,
            tso=tso_name,
            timestamp=timestamp,
            log_table=False,
        )

    # ── Save processed predictions locally ───────────────────────────────────
    _save_predictions_locally(
        output_dir,
        processed_with_y,
        model_name,
        tso_name,
        timestamp,
        best_checkpoint=best_checkpoint,
        window_index=window_index,
    )

    return processed_clean


# ---------------------------------------------------------------------------
# Final evaluation (across all windows)
# ---------------------------------------------------------------------------

def evaluate_hurdle_final(
    preds_list: list[pd.DataFrame],
    model_dataset: pd.DataFrame,
    benchmarks_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    tso_name: str,
    timestamp: str,
    best_checkpoint: bool = False,
) -> None:
    """Concatenate per-window predictions, evaluate, and save locally.

    Evaluation is performed for **every** prediction column present (both
    ``{alias}`` and ``{alias}_theta`` if available), producing a single
    combined evaluation CSV.
    """
    all_preds = pd.concat(preds_list, ignore_index=True)
    all_preds = all_preds.drop_duplicates(subset=["unique_id", "ds"], keep="first")

    eval_df = evaluation_pipeline(
        model_dataset=model_dataset,
        predictions=all_preds,
        benchmarks_df=cast(pd.DataFrame, benchmarks_df),
        beta=2.0,
    )

    save_evaluation_locally(
        output_dir,
        eval_df,
        model_name,
        tso_name,
        timestamp=timestamp,
        best_checkpoint=best_checkpoint,
    )

    # Save combined predictions with y merged
    preds_with_y = all_preds.merge(
        model_dataset[["unique_id", "ds", "y"]],
        on=["unique_id", "ds"],
        how="left",
    )
    _save_predictions_locally(
        output_dir,
        preds_with_y,
        model_name,
        tso_name,
        timestamp=timestamp,
        best_checkpoint=best_checkpoint,
    )


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
    enabled_benchmarks: list[str] | None = None,
    benchmark_config_path: Path | None = None,
    test_best_checkpoint: bool = False,
    persist_archive_dir: bool = False,
    start_window: int = 0,
    force: bool = False,
    fbeta2_target: float = 0.5,
    theta_lambda: float = 0.5,
    theta_n_grid: int = 200,
) -> None:
    """Run the hurdle rolling-window prediction and evaluation pipeline.

    Parameters
    ----------
    root_model_path : Path
        Root directory containing ``window_N/`` sub-directories with
        ``.tar.zst`` model archives.
    dataset_root_dir : Path
        Root directory where pre-processed Parquet datasets are stored.
    wandb_project, wandb_entity : str
        Weights & Biases project / entity for run resumption and logging.
    output_dir : Path
        Directory to write raw caches, processed predictions, and evaluations.
    add_calendar_features : bool
        Whether to add calendar features when preparing the input dataset.
    holidays_path : str or None
        Path override for the holidays CSV.
    enabled_benchmarks : list[str] or None
        Benchmark models to compute.  ``None`` runs all available.
    benchmark_config_path : Path or None
        Optional YAML with benchmark hyper-parameters.
    test_best_checkpoint : bool
        Also evaluate the best-validation checkpoint alongside the last one.
    persist_archive_dir : bool
        Keep the temporary extraction directories instead of cleaning up.
    start_window : int
        Index of the first window to process (inclusive).
    force : bool
        Force re-inference even when a local raw-predictions cache exists.
    fbeta2_target : float
        Target Fβ2 score used in the θ optimisation penalty (default 0.5).
    theta_lambda : float
        Penalty weight λ for the Fβ2 shortfall in θ optimisation (default 0.5).
    theta_n_grid : int
        Number of candidate θ values swept during optimisation (default 200).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # File logging
    log_file = output_dir / "prediction_pipeline_hurdle_rolling_window.log"
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
    logger.info("Starting hurdle rolling-window prediction pipeline")
    logger.info("root_model_path = %s", root_model_path)
    logger.info("=" * 80)

    temp_dirs_to_cleanup: list[Path] = []

    try:
        model_data = list(read_models(root_model_path, temp_dirs_to_cleanup, window_start=start_window))
        if not model_data:
            logger.warning("No models found in %s – exiting.", root_model_path)
            return

        # ── Collect metadata for every extracted model ────────────────────────
        model_metas: list[dict] = []
        for window_index, model_dir in model_data:
            run_info = identify_wandb_run(model_dir, window_index, wandb_entity, wandb_project)
            run_config = run_info["config"]
            tso_name = str(_require_config(run_config, "tso")).replace(" ", "_")
            dataset_name = str(_require_config(run_config, "dataset_contents")).replace(", ", "_")
            model_name = get_model_name(model_dir)
            model_metas.append(
                {
                    "window_index": window_index,
                    "model_dir": model_dir,
                    "model_name": model_name,
                    "run_info": run_info,
                    "tso_name": tso_name,
                    "dataset_name": dataset_name,
                }
            )

        # ── Group by (tso, dataset) for shared benchmark computation ──────────
        dataset_keys: dict[tuple[str, str], list[dict]] = {}
        for meta in model_metas:
            key = (meta["tso_name"], meta["dataset_name"])
            dataset_keys.setdefault(key, []).append(meta)

        benchmark_cache: dict[tuple[str, str], pd.DataFrame] = {}

        for (tso_name, dataset_name), ds_metas in dataset_keys.items():
            logger.info(
                "Computing benchmarks for tso=%s dataset=%s", tso_name, dataset_name
            )
            model_dataset = get_dataset(dataset_root_dir, tso_name, dataset_name)
            representative_config = ds_metas[0]["run_info"]["config"]
            representative_config["start_window"] = start_window
            timestamp = _require_config(representative_config, "date_time")

            benchmarks_df, benchmark_metadata = run_benchmarks_from_pipeline_config(
                dataset=model_dataset,
                run_config=representative_config,
                tso_name=tso_name,
                enabled_benchmarks=enabled_benchmarks,
                parameter_yaml_file_path=benchmark_config_path,
            )
            benchmark_cache[(tso_name, dataset_name)] = benchmarks_df

            # Save LightGBM feature importance if present
            if enabled_benchmarks and "lightgbm" in enabled_benchmarks:
                for key in [k for k in benchmark_metadata if "lightgbm" in k]:
                    benchmark_metadata = save_lightgbm_feature_importance(
                        benchmark_metadata, output_dir, key
                    )

            benchmarks_path = output_dir / f"benchmarks_{tso_name}_{dataset_name}_{timestamp}.csv"
            benchmarks_df.assign(tso=tso_name, dataset=dataset_name).to_csv(
                benchmarks_path, index=False
            )
            logger.info("Saved benchmark predictions to %s", benchmarks_path)

            meta_path = output_dir / f"benchmark_metadata_{tso_name}_{dataset_name}.json"
            with open(meta_path, "w") as fh:
                json.dump(benchmark_metadata, fh, indent=2, default=str)

        # ── Group by (tso, dataset, model_name) for model processing ─────────
        model_keys: dict[tuple[str, str, str], list[dict]] = {}
        for meta in model_metas:
            key = (meta["tso_name"], meta["dataset_name"], meta["model_name"])
            model_keys.setdefault(key, []).append(meta)

        for (tso_name, dataset_name, model_name), group_metas in model_keys.items():
            logger.info(
                "Processing hurdle model=%s tso=%s dataset=%s (%d windows)",
                model_name, tso_name, dataset_name, len(group_metas),
            )

            model_dataset = get_dataset(dataset_root_dir, tso_name, dataset_name)
            benchmarks_df = benchmark_cache[(tso_name, dataset_name)]

            representative_config = group_metas[0]["run_info"]["config"]
            timestamp = _require_config(representative_config, "date_time")

            group_metas.sort(key=lambda m: m["window_index"])

            # ── Last checkpoint ───────────────────────────────────────────────
            last_preds_list: list[pd.DataFrame] = []
            for meta in group_metas:
                # Refresh timestamp per window (may differ in multi-alias setups)
                timestamp = _require_config(meta["run_info"]["config"], "date_time")
                processed = run_hurdle_window(
                    model_dataset=model_dataset,
                    add_calendar_features=add_calendar_features,
                    holidays_path=holidays_path,
                    meta=meta,
                    output_dir=output_dir,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    best_checkpoint=False,
                    force=force,
                    fbeta2_target=fbeta2_target,
                    theta_lambda=theta_lambda,
                    theta_n_grid=theta_n_grid,
                )
                last_preds_list.append(processed)

            logger.info("Evaluating last-checkpoint predictions across %d windows", len(group_metas))
            evaluate_hurdle_final(
                preds_list=last_preds_list,
                model_dataset=model_dataset,
                benchmarks_df=benchmarks_df,
                output_dir=output_dir,
                model_name=model_name,
                tso_name=tso_name,
                timestamp=timestamp,
                best_checkpoint=False,
            )

            # ── Best-validation checkpoint (optional) ─────────────────────────
            if test_best_checkpoint:
                best_preds_list: list[pd.DataFrame] = []
                for meta in group_metas:
                    timestamp = _require_config(meta["run_info"]["config"], "date_time")
                    processed = run_hurdle_window(
                        model_dataset=model_dataset,
                        add_calendar_features=add_calendar_features,
                        holidays_path=holidays_path,
                        meta=meta,
                        output_dir=output_dir,
                        wandb_project=wandb_project,
                        wandb_entity=wandb_entity,
                        best_checkpoint=True,
                        force=force,
                        fbeta2_target=fbeta2_target,
                        theta_lambda=theta_lambda,
                        theta_n_grid=theta_n_grid,
                    )
                    best_preds_list.append(processed)

                logger.info(
                    "Evaluating best-checkpoint predictions across %d windows", len(group_metas)
                )
                evaluate_hurdle_final(
                    preds_list=best_preds_list,
                    model_dataset=model_dataset,
                    benchmarks_df=benchmarks_df,
                    output_dir=output_dir,
                    model_name=model_name,
                    tso_name=tso_name,
                    timestamp=timestamp,
                    best_checkpoint=True,
                )

    finally:
        if temp_dirs_to_cleanup and not persist_archive_dir:
            logger.info("Cleaning up %d temporary extraction directories", len(temp_dirs_to_cleanup))
            for tmp in temp_dirs_to_cleanup:
                if tmp.exists() and tmp.is_dir() and "model_extraction_" in tmp.name:
                    try:
                        shutil.rmtree(tmp)
                        logger.info("Deleted %s", tmp)
                    except Exception as exc:
                        logger.warning("Could not delete %s: %s", tmp, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def prepare_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rolling-window prediction pipeline for hurdle (zero-inflated) models."
    )
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Root directory containing window_N/ sub-directories with .tar.zst archives.",
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
        help="Output directory for predictions, evaluation, and raw caches.",
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
        "--persist-archive-dir",
        action="store_true",
        help="Keep temporary extraction directories (useful for debugging).",
    )
    p.add_argument(
        "--start-window",
        type=int,
        default=0,
        help="Index of the first window to process (inclusive, default: 0).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run inference even when a local raw-predictions cache exists.",
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
        persist_archive_dir=args.persist_archive_dir,
        start_window=args.start_window,
        force=args.force,
        fbeta2_target=args.hurdle_theta_fbeta2_target,
        theta_lambda=args.hurdle_theta_lambda,
        theta_n_grid=args.hurdle_theta_n_grid,
    )
