import logging
import argparse
import json
import os
import pickle
import shutil
import tempfile
from contextlib import contextmanager
import pandas as pd
from sklearn.metrics import fbeta_score
import zstandard as zstd
import tarfile
import wandb

from pathlib import Path
from neuralforecast import NeuralForecast
from training.prediction_pipeline_new import save_lightgbm_feature_importance
from training.runner import CHECKPOINT_BEST_NAME
from typing import cast

from training.predict import predict_with_shift_correction, evaluate_models, log_predictions_to_wandb, prepare_predictions_df
from training.hurdle_runner import optimize_hurdle_threshold
from training.data_prep import (
    load_dataset,
    to_nixtla_format,
    build_static_df,
    prepare_shifted_dataset,
    classify_covariates,
)
from training.benchmarks import run_benchmarks_from_pipeline_config
from training.explainability import (
    explain_all_models,
    save_explanation_artifacts,
    aggregate_ig_stats,
    log_ig_summary_tables,
)
from training.train_pipeline import _normalize_tso_key, set_n_threads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _require_config(run_config: dict, key: str):
    value = run_config.get(key)
    if value is None:
        raise KeyError(f"Missing required wandb config key: {key}")
    return value


def _dedupe_callbacks(callbacks: list[object]) -> list[object]:
    seen: set[type] = set()
    deduped: list[object] = []
    for cb in callbacks:
        cb_type = type(cb)
        if cb_type in seen:
            continue
        seen.add(cb_type)
        deduped.append(cb)
    return deduped

def _dedupe_model_callbacks(nf: NeuralForecast) -> None:
    for model in getattr(nf, "models", []):
        callbacks = model.hparams.get("callbacks", None)
        if isinstance(callbacks, list) and callbacks:
            model.hparams["callbacks"] = _dedupe_callbacks(callbacks)

        trainer_kwargs = getattr(model, "trainer_kwargs", None)
        if isinstance(trainer_kwargs, dict) and isinstance(trainer_kwargs.get("callbacks"), list):
            trainer_kwargs["callbacks"] = _dedupe_callbacks(trainer_kwargs["callbacks"])

def move_files_from_last_nonempty_dir(source_dir: Path, root_dir: Path):
    """Recursively find the first directory with non-empty data starting from source_dir and move its files to root_dir."""
    if not source_dir.is_dir():
        raise NotADirectoryError(f"{source_dir} is not a directory.")
    
    for item in source_dir.iterdir():
        if item.is_file() and item.stat().st_size > 0:
            item.rename(root_dir / item.name)  # Move file to root_dir
        elif item.is_dir():
            move_files_from_last_nonempty_dir(item, root_dir)  # Recurse into subdirectory
            # After returning from recursion, check if the current directory is empty and remove it
            if not any(source_dir.iterdir()):
                source_dir.rmdir()


def read_models(model_path: Path, temp_dirs_to_cleanup: list[Path], window_start: int = 0):
    """Read models from window directories and extract archives to temporary directories.
    
    Args:
        model_path: Root path containing window directories
        temp_dirs_to_cleanup: List to track temporary directories for later cleanup
        window_start: Index of the first window to process (inclusive)
    
    Yields:
        Tuple of (window_index, extracted_model_dir)
    """
    window_directories = [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("window_")]

    if window_directories:
        logger.info(f"Found {len(window_directories)} window directories in {model_path}, extracting models from each.")
        for window_dir in window_directories:
            window_index = int(window_dir.name.replace("window_", ""))
            if window_index < window_start:
                logger.info(f"Skipping window {window_index} because it is before the specified start_window {window_start}.")
                continue
            model_directories = [dir for dir in window_dir.iterdir() if dir.is_dir()]
            for model_path in model_directories:
                archive_paths = [path for path in model_path.glob("*.tar.zst") if path.is_file()]
                if len(archive_paths) == 0:
                    logger.warning(f"No .tar.zst archive found in {model_path}, skipping this model.")
                    continue
                elif len(archive_paths) > 1:
                    logger.warning(f"Multiple .tar.zst archives found in {model_path}, expected only one. Skipping this model.")
                    continue
                archive_path = archive_paths[0]
                logger.info(f"Inflating model from {archive_path}")
                
                # Create a safe temporary directory for extraction (in the same parent directory to avoid later reference issues)
                model_dir = model_path.with_name("model_extraction_" + model_path.name)
                temp_dirs_to_cleanup.append(model_dir)
                
                # Extract the archive to the temporary directory
                if model_dir.exists():
                    logger.info(f"Temporary extraction directory {model_dir} already exists, skipping extraction.")
                else:
                    model_dir.mkdir(parents=True, exist_ok=False)
                    with open(archive_path, "rb") as f:
                        dctx = zstd.ZstdDecompressor()
                        with dctx.stream_reader(f) as reader:
                            with tarfile.open(fileobj=reader, mode="r|*") as tar:
                                tar.extractall(path=model_dir)
                move_files_from_last_nonempty_dir(model_dir, model_dir)  # Move files from last non-empty subdirectory to model_dir
                yield window_index, model_dir
    else:
        logger.info(f"No window directories found in {model_path}, script will exit.")


def get_last_checkpoint_path(nf_dir: Path, require: bool = True) -> Path | None:
    checkpoints = list(nf_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {nf_dir / 'checkpoints'}")
    
    best_checkpoint_path = nf_dir / CHECKPOINT_BEST_NAME
    last_checkpoint = set(checkpoints) - {best_checkpoint_path}
    if not len(last_checkpoint) == 1:
        if not require and len(last_checkpoint) == 0:
            return None
        raise ValueError(f"Expected exactly one last checkpoint in {nf_dir / 'checkpoints'}, found {len(last_checkpoint)}")
    
    return last_checkpoint.pop()


def _get_single_model_alias(model_dir: Path) -> str | None:
    alias_map_path = model_dir / "alias_to_model.pkl"
    if not alias_map_path.exists():
        return None

    try:
        with open(alias_map_path, "rb") as f:
            alias_to_model = pickle.load(f)
    except Exception:
        return None

    if not isinstance(alias_to_model, dict) or not alias_to_model:
        return None

    aliases = sorted(alias_to_model.keys())
    if len(aliases) > 1:
        logger.warning(
            "Multiple model aliases found in %s; using %s to load best checkpoint.",
            alias_map_path,
            aliases[0],
        )
    return aliases[0]


def load_model(model_dir: Path, checkpoint_best: bool) -> NeuralForecast:
    best_valid_checkpoint = model_dir / CHECKPOINT_BEST_NAME
    best_valid_checkpoint_tmp_path = best_valid_checkpoint.with_suffix(".bkp")
    last_checkpoint_path = get_last_checkpoint_path(model_dir, require=not checkpoint_best)
    last_checkpoint_tmp_path = last_checkpoint_path.with_suffix(".bkp") if last_checkpoint_path is not None else None
    best_swapped_to_last = False
    last_moved_to_tmp = False
    try:
        if not checkpoint_best:
            if best_valid_checkpoint.exists():
                best_valid_checkpoint.rename(best_valid_checkpoint_tmp_path)
            nf_model = NeuralForecast.load(str(model_dir.resolve()))
        else:
            # If no separate "last" checkpoint exists, synthesize a loadable
            # checkpoint filename from the saved model alias.
            if best_valid_checkpoint_tmp_path.exists():
                best_valid_checkpoint_tmp_path.rename(best_valid_checkpoint)
            if not best_valid_checkpoint.exists():
                logger.warning(
                    "Best valid checkpoint not found in %s; falling back to default checkpoint loading.",
                    model_dir,
                )
                nf_model = NeuralForecast.load(str(model_dir.resolve()))
                _dedupe_model_callbacks(nf_model)
                return nf_model

            if last_checkpoint_path is not None and last_checkpoint_tmp_path is not None:
                # Rename the best valid checkpoint to the expected name for loading
                # and rename the last checkpoint to avoid confusion.
                last_checkpoint_path.rename(last_checkpoint_tmp_path)
                last_moved_to_tmp = True
                best_valid_checkpoint.rename(last_checkpoint_path)
                best_swapped_to_last = True
            elif last_checkpoint_path is None:
                model_alias = _get_single_model_alias(model_dir)
                if model_alias is None:
                    raise FileNotFoundError(
                        f"Could not infer model alias from {model_dir / 'alias_to_model.pkl'} to load best checkpoint"
                    )
                synthetic_checkpoint_path = model_dir / f"{model_alias}_0.ckpt"
                if synthetic_checkpoint_path.exists():
                    synthetic_checkpoint_tmp = synthetic_checkpoint_path.with_suffix(".bkp")
                    synthetic_checkpoint_path.rename(synthetic_checkpoint_tmp)
                    last_checkpoint_tmp_path = synthetic_checkpoint_tmp
                    last_moved_to_tmp = True
                last_checkpoint_path = synthetic_checkpoint_path
                best_valid_checkpoint.rename(last_checkpoint_path)
                best_swapped_to_last = True

            nf_model = NeuralForecast.load(str(model_dir.resolve()))
        _dedupe_model_callbacks(nf_model)
        return nf_model
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_dir} with checkpoint_best={checkpoint_best}") from e
    finally:
        if best_valid_checkpoint_tmp_path.exists():
            best_valid_checkpoint_tmp_path.rename(best_valid_checkpoint)

        if best_swapped_to_last and last_checkpoint_path is not None and last_checkpoint_path.exists():
            last_checkpoint_path.rename(best_valid_checkpoint)

        if last_moved_to_tmp and last_checkpoint_path is not None and last_checkpoint_tmp_path is not None and last_checkpoint_tmp_path.exists():
            last_checkpoint_tmp_path.rename(last_checkpoint_path)


def get_model_name(model_dir: Path) -> str:
    # Extract the model name from the model directory structure
    model_name = model_dir.name.replace("model_extraction_", "")  # Remove temp prefix if present
    return model_name


def identify_wandb_run(model_dir: Path, window_index: int, wandb_entity: str | None, wandb_project: str) -> dict:
    """Locate the existing W&B run for this model checkpoint and return metadata.

    Returns a dict with keys: run_id, config.
    """
    model_label = get_model_name(model_dir)
    timestamp = [dir for dir in model_dir.iterdir() if dir.is_dir()][0].name

    api = wandb.Api()
    project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
    filters = {
        "config.date_time": timestamp,
        "config.model_alias": model_label,
        "config.window_index": window_index,
    }
    runs = api.runs(project_path, filters=filters)
    if not runs:
        raise FileNotFoundError(
            f"No W&B run found for model_alias={model_label} and date_time={timestamp} in {project_path}."
        )

    if len(runs) > 1:
        logger.warning(
            "Multiple W&B runs found for model_alias=%s and date_time=%s; using the most recent.",
            model_label,
            timestamp,
        )

    run = runs[0]
    logger.info("Identified W&B run ID: %s", run.id)
    return {
        "run_id": run.id,
        "config": dict(run.config),
    }


def get_dataset(dataset_root_dir: Path, tso_name: str, dataset_name: str) -> pd.DataFrame:
    logger.info("Looking for dataset with tso_name=%s and dataset_name=%s in %s", tso_name, dataset_name, dataset_root_dir)
    if dataset_name == "basic":
        dataset_variants = ["basic", "basic_remove_data_driven", "basic_remove_runlength", "basic_remove_data_driven_remove_runlength"]
        for variant in dataset_variants:
            dataset_path = dataset_root_dir / f"{variant}_{tso_name}.parquet"
            if dataset_path.exists():
                logger.info(f"Found dataset at {dataset_path}, using this variant.")
                break
        else:
            raise FileNotFoundError(
                f"No dataset found for any of the variants {dataset_variants} with tso_name={tso_name} in {dataset_root_dir}."
            )
    else:
        dataset_path = dataset_root_dir / f"{dataset_name}_{tso_name}.parquet"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    dataset, _ = load_dataset(dataset_path)
    nixtla_dataset = to_nixtla_format(dataset)
    return nixtla_dataset


def retrieve_model_predictions_from_wandb(
    wandb_entity: str | None,
    wandb_project: str,
    model_name: str,
    tso_name: str,
    timestamp: str,
    window_index: int,
    best_checkpoint: bool,
) -> pd.DataFrame:
    """Retrieve predictions artifact logged by training.predict.log_predictions_to_wandb."""
    if best_checkpoint:
        checkpoint_label = f"test_best_valid_window{window_index}"
    else:
        checkpoint_label = f"test_window{window_index}"
    artifact_name = f"preds_{checkpoint_label}_{model_name}_{tso_name}_{timestamp}"
    try:
        project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
        predictions_artifact = wandb.Api().artifact(f"{project_path}/{artifact_name}:latest")
        predictions_dir = Path(predictions_artifact.download(root=Path("wandb")))
        predictions_path = predictions_dir / f"{checkpoint_label}_predictions.parquet"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found at {predictions_path}")

        return pd.read_parquet(predictions_path)
    except Exception as e:
        logger.warning("Error retrieving predictions from W&B artifact %s: %s", artifact_name, e)
        return pd.DataFrame()


def run_model_predictions_window(
    model_dataset: pd.DataFrame,
    add_calendar_features: bool,
    holidays_path: str | None,
    shift_hours: int,
    model_alias: str,
    wandb_project: str,
    wandb_entity: str | None,
    meta: dict,
    checkpoint_preds_list: list[pd.DataFrame],
    best_checkpoint: bool = False,
    force: bool = False,
    output_dir: Path | None = None,
):
    with wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=meta["run_info"]["run_id"],
        resume="allow",
        mode="online",
    ) as wandb_run:
        window_index = meta["window_index"]
        model_dir = meta["model_dir"]
        run_config = meta["run_info"]["config"]
        timestamp = _require_config(run_config, "date_time")
        model_name = meta["model_name"]
        tso_name = _normalize_tso_key(meta["tso_name"])

        logger.info("Processing window %d: %s", window_index, model_dir)

        # ── Last checkpoint predictions for this window ───────────────
        checkpoint_predictions = retrieve_model_predictions_from_wandb(
            wandb_entity,
            wandb_project,
            model_name=model_name,
            tso_name=tso_name,
            timestamp=timestamp,
            window_index=window_index,
            best_checkpoint=best_checkpoint,
        )
        if best_checkpoint:
            checkpoint_label = f"test_best_valid_window{window_index}"
        else:
            checkpoint_label = f"test_window{window_index}"
        if force or checkpoint_predictions.empty:
            last_checkpoint = load_model(model_dir, checkpoint_best=best_checkpoint)
            checkpoint_predictions = predict_checkpoint(
                nf_model=last_checkpoint,
                dataset=model_dataset,
                tso_name=tso_name,
                run_config=run_config,
                add_calendar_features=add_calendar_features,
                holidays_path=holidays_path,
            )
            last_checkpoint_predictions_df = checkpoint_predictions.merge(
                model_dataset[["unique_id", "ds", "y"]],
                on=["unique_id", "ds"],
                how="left",
            )
            log_predictions_to_wandb(
                wandb_run,
                last_checkpoint_predictions_df,
                split_name=checkpoint_label,
                shift_hours=shift_hours,
                model_alias=model_alias,
                tso=tso_name,
                timestamp=timestamp,
                log_table=False,
            )

        # ── Save per-window predictions locally ────────────────────────────
        if output_dir is not None:
            preds_to_save = checkpoint_predictions.copy()
            if "y" not in preds_to_save.columns:
                preds_to_save = preds_to_save.merge(
                    model_dataset[["unique_id", "ds", "y"]],
                    on=["unique_id", "ds"],
                    how="left",
                )
            _save_predictions_locally(
                output_dir,
                preds_to_save,
                model_name,
                tso_name,
                timestamp,
                best_checkpoint=best_checkpoint,
                window_index=window_index,
            )

        checkpoint_preds_list.append(checkpoint_predictions)

    return checkpoint_preds_list


def _wandb_artifact_exists(
    wandb_entity: str | None,
    wandb_project: str,
    artifact_name: str,
) -> bool:
    """Return True if *artifact_name* already exists as a W&B artifact."""
    project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
    try:
        wandb.Api().artifact(f"{project_path}/{artifact_name}:latest")
        return True
    except Exception:
        return False


def run_explainability_window(
    model_dataset: pd.DataFrame,
    add_calendar_features: bool,
    holidays_path: str | None,
    shift_hours: int,
    meta: dict,
    model_name: str,
    model_alias: str,
    model_dir: Path,
    output_dir: Path,
    wandb_project: str,
    wandb_entity: str | None,
    checkpoint_best: bool = False,
    explain_step_hours: int = 24,
    force: bool = False,
):
    window_index = meta["window_index"]

    with wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=meta["run_info"]["run_id"],
        resume="allow",
        mode="online",
    ) as wandb_run:
        run_config = meta["run_info"]["config"]
        tso_name = _normalize_tso_key(meta["tso_name"])
        timestamp = _require_config(run_config, "date_time")

        # Skip if explainability output already exists for this window
        if checkpoint_best:
            ig_artifact_name = f"ig_raw_best_checkpoint_{model_alias}_{tso_name}_{timestamp}"
        else:
            ig_artifact_name = f"ig_raw_{model_alias}_{tso_name}_{timestamp}"
        if not force and _wandb_artifact_exists(wandb_entity, wandb_project, ig_artifact_name):
            logger.info(
                "Explainability artifact already exists in W&B: %s",
                ig_artifact_name,
            )
            return

        logger.info(
            "Running IG explainability for window %d, model %s",
            window_index, model_name,
        )
        loaded_checkpoint = load_model(model_dir, checkpoint_best=checkpoint_best)
        ig_explanations, ig_preds = explain_checkpoint(
            nf_model=loaded_checkpoint,
            dataset=model_dataset,
            run_config=run_config,
            tso_name=tso_name,
            add_calendar_features=add_calendar_features,
            holidays_path=holidays_path,
            step_hours=explain_step_hours,
        )
        save_explanation_artifacts(
            run=wandb_run,
            explanations=ig_explanations,
            predictions=ig_preds,
            model_alias=model_alias,
            tso=tso_name,
            timestamp=timestamp,
            shift_hours=shift_hours,
            window_index=window_index,
            output_dir=output_dir,
        )

        # Log per-window summary tables
        hist_cov = run_config.get("historical_covariates", [])
        futr_cov = run_config.get("future_covariates", [])
        for feat_type, feat_names in [
            ("hist_exog", hist_cov),
            ("futr_exog", futr_cov),
        ]:
            if feat_names:
                summaries = aggregate_ig_stats(
                    ig_explanations,
                    feature_names=feat_names,
                    feature_type=feat_type,
                )
                log_ig_summary_tables(
                    wandb_run, summaries,
                    prefix=f"ig_{feat_type}_window{window_index}",
                )


def evaluate_final_predictions(
    checkpoint_preds_list: list[pd.DataFrame],
    model_dataset: pd.DataFrame,
    benchmarks_df: pd.DataFrame | None,
    output_dir: Path,
    model_name: str,
    tso_name: str,
    timestamp: str,
    best_checkpoint: bool = False,
):
    # ── Concatenate predictions across all windows ────────────────────
    checkpoint_predictions_all = pd.concat(checkpoint_preds_list, ignore_index=True)

    # Remove duplicates keeping the first occurrence (in case of overlapping windows)
    checkpoint_predictions_all = checkpoint_predictions_all.drop_duplicates(
        subset=["unique_id", "ds"],
        keep="first"
    )

    # ── Evaluate last checkpoint (benchmarks_df may be None when --skip-benchmarks is set) ─────
    last_checkpoint_evaluation_df = evaluation_pipeline(
        model_dataset=model_dataset,
        predictions=checkpoint_predictions_all,
        benchmarks_df=benchmarks_df,
        beta=2.0,
    )

    save_evaluation_locally(
        output_dir,
        last_checkpoint_evaluation_df,
        model_name,
        tso_name,
        timestamp=timestamp,
        best_checkpoint=best_checkpoint,
    )

    # ── Save combined (all-windows) predictions as Parquet ────────────────
    preds_to_save = checkpoint_predictions_all.copy()
    if "y" not in preds_to_save.columns:
        preds_to_save = preds_to_save.merge(
            model_dataset[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="left",
        )
    _save_predictions_locally(
        output_dir,
        preds_to_save,
        model_name,
        tso_name,
        timestamp=timestamp,
        best_checkpoint=best_checkpoint,
    )


def optimize_and_save_hurdle_thresholds(
    checkpoint_preds_list: list[pd.DataFrame],
    window_indices: list[int],
    model_dataset: pd.DataFrame,
    output_dir: Path,
    model_alias: str,
    model_name: str,
    tso_name: str,
    timestamp: str,
    fbeta2_target: float = 0.5,
    lambda_: float = 0.5,
    n_grid: int = 200,
    beta: float = 2.0,
) -> None:
    """
    Optimise the hurdle probability threshold θ for each rolling window and
    for all windows combined.

    For each window the threshold is chosen to minimise::

        MAE + λ · max(0, Fβ2_target − Fβ2)

    where Fβ2 is the F-β score averaged across redispatch directions.
    Results are written to JSON files in *output_dir*:

    * Per-window: ``hurdle_theta_opt_{model_name}_{tso_name}_{timestamp}_window{N}.json``
    * Combined:   ``hurdle_theta_opt_{model_name}_{tso_name}_{timestamp}_all_windows.json``

    If the prediction DataFrames do not contain the expected hurdle columns
    (``{model_alias}-non_zero_logit`` / ``{model_alias}-magnitude``), the
    function logs a debug message and returns without writing any files.

    Parameters
    ----------
    checkpoint_preds_list : list[pd.DataFrame]
        Raw model predictions per window (in the same order as *window_indices*).
        Need not contain a ``y`` column – actuals are pulled from *model_dataset*.
    window_indices : list[int]
        Window index for each DataFrame in *checkpoint_preds_list*.
    model_dataset : pd.DataFrame
        Full dataset (Nixtla format) used to look up ground-truth ``y``.
    output_dir : Path
        Directory where JSON files are written (created if it does not exist).
    model_alias : str
        Model alias used as the column-name prefix for hurdle outputs.
    model_name, tso_name, timestamp : str
        Used for output file naming.
    fbeta2_target : float
        Target Fβ2 score that anchors the recall-preserving penalty.
    lambda_ : float
        Penalty weight λ (default 0.5).
    n_grid : int
        Number of candidate thresholds swept in (0, 1).
    beta : float
        β for the F-score (2 ⇒ recall-weighted Fβ2).
    """
    logit_col = f"{model_alias}-non_zero_logit"

    # Fast-path: none of the predictions come from a hurdle model
    if not any(logit_col in df.columns for df in checkpoint_preds_list):
        logger.debug(
            "optimize_and_save_hurdle_thresholds: column '%s' not found in any "
            "prediction DataFrame – skipping (model_alias=%s).",
            logit_col, model_alias,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    window_results: dict[int, dict] = {}
    preds_for_combined: list[pd.DataFrame] = []

    for wi, preds_df in zip(window_indices, checkpoint_preds_list):
        if logit_col not in preds_df.columns:
            logger.warning(
                "Window %d: hurdle column '%s' missing – skipping threshold optimisation.",
                wi, logit_col,
            )
            continue

        # Merge ground-truth actuals
        preds_with_y = preds_df.merge(
            model_dataset[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="left",
        ).dropna(subset=["y"])

        if preds_with_y.empty:
            logger.warning("Window %d: no matching actuals found – skipping.", wi)
            continue

        try:
            result = optimize_hurdle_threshold(
                preds_df=preds_with_y,
                alias=model_alias,
                fbeta2_target=fbeta2_target,
                lambda_=lambda_,
                n_grid=n_grid,
                beta=beta,
            )
        except Exception as exc:
            logger.warning(
                "Window %d: threshold optimisation failed: %s", wi, exc, exc_info=True
            )
            continue

        result["window_index"] = wi
        result["model_alias"]  = model_alias
        result["tso_name"]     = tso_name
        window_results[wi] = result
        preds_for_combined.append(preds_with_y)

        per_window_path = output_dir / (
            f"hurdle_theta_opt_{model_name}_{tso_name}_{timestamp}_window{wi}.json"
        )
        with open(per_window_path, "w") as fh:
            json.dump(result, fh, indent=2)
        logger.info(
            "Window %d: optimal θ=%.4f  combined=%.4f  MAE=%.4f  Fβ=%.4f  → %s",
            wi, result["optimal_theta"], result["combined_metric"],
            result["mae"], result["fbeta_mean"], per_window_path,
        )

    if not window_results:
        logger.info(
            "optimize_and_save_hurdle_thresholds: no per-window results produced – "
            "skipping combined JSON."
        )
        return

    # ── Combined optimisation across all windows ──────────────────────────────
    combined_meta: dict = {
        "model_alias":    model_alias,
        "model_name":     model_name,
        "tso_name":       tso_name,
        "timestamp":      timestamp,
        "lambda_":        float(lambda_),
        "fbeta_target":   float(fbeta2_target),
        "beta":           float(beta),
        "n_windows":      len(window_results),
        "window_results": window_results,
    }

    if preds_for_combined:
        all_preds = pd.concat(preds_for_combined, ignore_index=True)
        all_preds = all_preds.drop_duplicates(subset=["unique_id", "ds"], keep="first")

        try:
            combined_opt = optimize_hurdle_threshold(
                preds_df=all_preds,
                alias=model_alias,
                fbeta2_target=fbeta2_target,
                lambda_=lambda_,
                n_grid=n_grid,
                beta=beta,
            )
            combined_meta.update({
                "overall_optimal_theta":         combined_opt["optimal_theta"],
                "overall_combined_metric":       combined_opt["combined_metric"],
                "overall_mae":                   combined_opt["mae"],
                "overall_fbeta_mean":            combined_opt["fbeta_mean"],
                "overall_fbeta_per_direction":   combined_opt["fbeta_per_direction"],
                "overall_grid_thetas":           combined_opt["grid_thetas"],
                "overall_grid_combined_metrics": combined_opt["grid_combined_metrics"],
                "overall_grid_mae":              combined_opt["grid_mae"],
                "overall_grid_fbeta":            combined_opt["grid_fbeta"],
            })
            logger.info(
                "All windows combined: optimal θ=%.4f  combined=%.4f  MAE=%.4f  Fβ=%.4f",
                combined_opt["optimal_theta"], combined_opt["combined_metric"],
                combined_opt["mae"], combined_opt["fbeta_mean"],
            )
        except Exception as exc:
            logger.warning("Combined threshold optimisation failed: %s", exc, exc_info=True)

    combined_path = output_dir / (
        f"hurdle_theta_opt_{model_name}_{tso_name}_{timestamp}_all_windows.json"
    )
    with open(combined_path, "w") as fh:
        json.dump(combined_meta, fh, indent=2)
    logger.info("Saved combined threshold optimisation results to %s", combined_path)


def predict_checkpoint(
    nf_model: NeuralForecast,
    dataset: pd.DataFrame,
    run_config: dict,
    tso_name: str,
    add_calendar_features: bool,
    holidays_path: str | None,
) -> pd.DataFrame:
    shift_hours = int(_require_config(run_config, "shift_hours"))
    test_start_date = pd.Timestamp(_require_config(run_config, "test_start"))
    forecast_horizon = int(_require_config(run_config, "forecast_horizon"))

    # Prefer the explicitly stored test_end (set correctly per-window since the
    # runner.py fix).  Fall back to the n_test_months computation for configs
    # logged before that fix, and finally to the dataset maximum.
    if run_config.get("test_end") is not None:
        test_end_date = pd.Timestamp(run_config["test_end"]) - pd.Timedelta(hours=1)
    elif run_config.get("model_type", "single_window") == "rolling_window":
        n_test_months = run_config.get("n_test_months", 1)
        test_end_date = test_start_date + pd.DateOffset(months=n_test_months) - pd.Timedelta(hours=1)
    else:
        test_end_date = dataset["ds"].max()
    
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
        pred_start=test_start_date,
        pred_end=test_end_date,
        holidays_path=holidays_path,
        shift_hours=shift_hours,
        static_df=build_static_df(),
        tso=tso_name,
    )


def explain_checkpoint(
    nf_model: NeuralForecast,
    dataset: pd.DataFrame,
    run_config: dict,
    tso_name: str,
    add_calendar_features: bool,
    holidays_path: str | None,
    step_hours: int = 24,
) -> tuple[dict, pd.DataFrame]:
    """Run IG explainability for a rolling-window model checkpoint."""
    shift_hours = int(_require_config(run_config, "shift_hours"))
    test_start_date = pd.Timestamp(_require_config(run_config, "test_start"))
    forecast_horizon = int(_require_config(run_config, "forecast_horizon"))

    # Prefer the explicitly stored test_end (set correctly per-window since the
    # runner.py fix).  Fall back to the n_test_months computation for configs
    # logged before that fix, and finally to the dataset maximum.
    if run_config.get("test_end") is not None:
        test_end_date = pd.Timestamp(run_config["test_end"]) - pd.Timedelta(hours=1)
    elif run_config.get("model_type", "single_window") == "rolling_window":
        n_test_months = run_config.get("n_test_months", 1)
        test_end_date = test_start_date + pd.DateOffset(months=n_test_months) - pd.Timedelta(hours=1)
    else:
        test_end_date = dataset["ds"].max()

    shifted_dataset, future_cov_cols, _ = prepare_shifted_dataset(
        df=dataset,
        shift_hours=shift_hours,
        tso=tso_name,
        add_calendar=add_calendar_features,
        holidays_path=holidays_path,
    )
    return explain_all_models(
        nf=nf_model,
        df_shifted=shifted_dataset,
        df_unshifted=dataset,
        static_df=build_static_df(),
        pred_start=test_start_date,
        pred_end=test_end_date,
        future_cov_cols=future_cov_cols,
        shift_hours=shift_hours,
        forecast_horizon=forecast_horizon,
        tso=tso_name,
        holidays_path=holidays_path,
        step_hours=step_hours,
    )


def _save_predictions_locally(
    output_dir: Path,
    preds_df: pd.DataFrame,
    model_name: str,
    tso_name: str,
    timestamp: str,
    best_checkpoint: bool = False,
    window_index: int | None = None,
) -> None:
    """Save raw predictions as a Parquet file.

    Parameters
    ----------
    output_dir : Path
        Directory to write the file to (created if needed).
    preds_df : pd.DataFrame
        Predictions DataFrame (should include ``y`` when available).
    model_name, tso_name, timestamp : str
        Used to construct the output filename.
    best_checkpoint : bool
        Appends ``_best_checkpoint`` to the filename stem when True.
    window_index : int or None
        When provided, ``_window{N}`` is inserted before the checkpoint suffix
        to distinguish per-window from all-windows files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"predictions_{model_name}_{tso_name}_{timestamp}"
    if window_index is not None:
        stem += f"_window{window_index}"
    if best_checkpoint:
        stem += "_best_checkpoint"
    output_path = output_dir / f"{stem}.parquet"
    preds_df.to_parquet(output_path, index=False)
    logger.info("Saved predictions to %s", output_path)


def save_evaluation_locally(output_dir: Path, evaluation_df: pd.DataFrame, model_name: str, tso_name: str, timestamp: str, best_checkpoint: bool):
    """Save the evaluation dataframe locally as csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"evaluation_{model_name}_{tso_name}_{timestamp}"
    if best_checkpoint:
        filename += "_best_checkpoint"
    filename += ".csv"
    output_path = output_dir / filename
    evaluation_df.to_csv(output_path)
    logger.info(f"Saved evaluation to {output_path}")


def classification_metrics_df(prepared_preds_df, id_col='unique_id', time_col='ds', target_col='y', beta: float = 1.0):
    # basic validation
    if id_col not in prepared_preds_df.columns or time_col not in prepared_preds_df.columns or target_col not in prepared_preds_df.columns:
        raise ValueError(f"prepared_preds_df must contain columns {id_col}, {time_col}, {target_col}")

    # choose model columns automatically if not provided
    exclude = {id_col, time_col, target_col, "horizon"}
    models = [c for c in prepared_preds_df.columns if c not in exclude]
    if len(models) == 0:
        raise ValueError("No model columns found in prepared_preds_df")

    # directions to evaluate
    directions = prepared_preds_df[id_col].unique()

    results = []
    for direction in directions:
        sub = prepared_preds_df[prepared_preds_df[id_col] == direction]
        # ground truth binary
        y_true = (sub[target_col] > 0).astype(int)
        result_row = {"unique_id": direction, "metric": f"fbeta_{beta}"}
        for model in models:
            y_pred = (sub[model] > 0).astype(int)
            fbeta = fbeta_score(y_true, y_pred, average='binary', zero_division=0, beta=beta)
            result_row[model] = fbeta
        results.append(result_row)
    return pd.DataFrame(results)


def evaluation_pipeline(predictions: pd.DataFrame, benchmarks_df: pd.DataFrame | None, model_dataset: pd.DataFrame, start_date: pd.Timestamp | None = None, beta: float = 1.0):
    if start_date is not None:
        effective_start = max(start_date, predictions["ds"].min())
        predictions = predictions[predictions["ds"] >= effective_start]
        if benchmarks_df is not None and not benchmarks_df.empty:
            benchmarks_df = benchmarks_df[benchmarks_df["ds"] >= effective_start]

    # Regression split may remove last days from target_ts, so forward-fill benchmarks to align.
    # When benchmarks were skipped (benchmarks_df is None or empty) we evaluate model columns only.
    if benchmarks_df is not None and not benchmarks_df.empty:
        final_predictions_df = predictions.merge(benchmarks_df, on=["unique_id", "ds"], how="left").ffill()
    else:
        final_predictions_df = predictions.copy()

    final_predictions_df = prepare_predictions_df(final_predictions_df, model_dataset)
    evaluted_group_overall = evaluate_models(final_predictions_df)
    evaluted_group_overall_to_keep = evaluted_group_overall[evaluted_group_overall["metric"].isin(["mae", "rmse"])].copy()
    evaluated_conditional_group = evaluate_models(final_predictions_df[final_predictions_df["y"] > 0])
    evaluated_conditional_group_to_keep = evaluated_conditional_group[evaluated_conditional_group["metric"].isin(["mae", "rmse"])].copy()
    evaluated_conditional_group_to_keep["metric"] = evaluated_conditional_group_to_keep["metric"].map({"mae": "mae_conditional", "rmse": "rmse_conditional"})
    evaluated_fbeta_group = classification_metrics_df(final_predictions_df, beta=beta)
    evaluated_fbeta_group_to_keep = evaluated_fbeta_group[evaluated_fbeta_group["metric"] == f"fbeta_{beta}"].copy()
    evaluted_group = pd.concat([evaluted_group_overall_to_keep, evaluated_conditional_group_to_keep, evaluated_fbeta_group_to_keep], ignore_index=True)
    return evaluted_group


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
    checkpoint_selection: str = "last",
    persist_archive_dir: bool = False,
    run_explainability: bool = False,
    run_predictions: bool = True,
    skip_benchmarks: bool = False,
    explain_step_hours: int = 24,
    start_window: int = 0,
    force: bool = False,
    optimize_hurdle_thresholds: bool = False,
    hurdle_theta_fbeta2_target: float = 0.5,
    hurdle_theta_lambda: float = 0.5,
    hurdle_theta_n_grid: int = 200,
):
    # Setup file logging
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "prediction_pipeline_rolling_window.log"
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info("=" * 80)
    logger.info("Starting rolling window prediction pipeline")
    logger.info("Logging to file: %s", log_file)
    logger.info("=" * 80)
    
    evaluate_last_checkpoint = checkpoint_selection in {"last", "both"}
    evaluate_best_checkpoint = checkpoint_selection in {"best", "both"}

    # Track temporary directories for cleanup
    temp_dirs_to_cleanup: list[Path] = []
    
    try:
        # ── Discover models and group by (tso, dataset_name, model_name) across windows
        #    so we can concatenate predictions from all windows before evaluation.
        model_data = list(read_models(root_model_path, temp_dirs_to_cleanup, window_start=start_window))
        if not model_data:
            logger.warning("No models found in %s", root_model_path)
            return

        # Collect per-model metadata first
        model_metas: list[dict] = []
        for window_index, model_dir in model_data:
            run_info = identify_wandb_run(model_dir, window_index, wandb_entity, wandb_project)
            run_config = run_info["config"]
            tso_name = str(_require_config(run_config, "tso")).replace(" ", "_")
            dataset_name = str(_require_config(run_config, "dataset_contents")).replace(", ", "_")
            model_name = get_model_name(model_dir)
            model_metas.append({
                "window_index": window_index,
                "model_dir": model_dir,
                "model_name": model_name,
                "run_info": run_info,
                "tso_name": tso_name,
                "dataset_name": dataset_name,
            })

        # Group models first by (tso_name, dataset_name) for benchmark computation
        dataset_keys: dict[tuple[str, str], list[dict]] = {}
        for meta in model_metas:
            key = (meta["tso_name"], meta["dataset_name"])
            dataset_keys.setdefault(key, []).append(meta)

        # ── Process each dataset group (compute benchmarks once per dataset) ─────
        benchmark_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

        if run_predictions:
            for (tso_name, dataset_name), dataset_metas in dataset_keys.items():
                if skip_benchmarks:
                    logger.info(
                        "Skipping benchmarks for tso=%s dataset=%s (--skip-benchmarks)",
                        tso_name,
                        dataset_name,
                    )
                    benchmark_cache[(tso_name, dataset_name)] = None
                    continue

                logger.info(
                    "Computing benchmarks for tso=%s dataset=%s (shared across all models)",
                    tso_name,
                    dataset_name,
                )

                model_dataset = get_dataset(dataset_root_dir, tso_name, dataset_name)

                # Use the first model's first window config as representative
                representative_config = dataset_metas[0]["run_info"]["config"]
                representative_config["start_window"] = start_window
                timestamp = _require_config(representative_config, "date_time")

                benchmarks_df_path = output_dir / f"benchmarks_{tso_name}_{dataset_name}_{timestamp}.csv"

                # Compute benchmarks ONCE for this dataset and cache it
                benchmarks_df, benchmark_metadata = run_benchmarks_from_pipeline_config(
                    dataset=model_dataset,
                    run_config=representative_config,
                    tso_name=tso_name,
                    enabled_benchmarks=enabled_benchmarks,
                    parameter_yaml_file_path=benchmark_config_path,
                )
                benchmark_cache[(tso_name, dataset_name)] = benchmarks_df
    
                # Save LightGBM feature importance to CSV if present in metadata
                if not enabled_benchmarks or "lightgbm" in enabled_benchmarks:
                    lightgbm_keys = [key for key in benchmark_metadata.keys() if "lightgbm" in key]
                    for key in lightgbm_keys:
                        benchmark_metadata = save_lightgbm_feature_importance(
                            benchmark_metadata, output_dir, key
                        )

                # Save benchmark predictions
                benchmarks_df.assign(tso=tso_name, dataset=dataset_name).to_csv(benchmarks_df_path, index=False)
                logger.info("Saved benchmark predictions to %s", benchmarks_df_path)

                # Save benchmark metadata to JSON file
                benchmark_metadata_path = output_dir / f"benchmark_metadata_{tso_name}_{dataset_name}.json"
                import json
                with open(benchmark_metadata_path, "w") as f:
                    json.dump(benchmark_metadata, f, indent=2, default=str)
                logger.info("Saved benchmark metadata to %s", benchmark_metadata_path)
        else:
            logger.info("Skipping benchmark computation because predictions/evaluation are disabled.")

        # Now group by (tso_name, dataset_name, model_name) for model processing
        model_keys: dict[tuple[str, str, str], list[dict]] = {}
        for meta in model_metas:
            key = (meta["tso_name"], meta["dataset_name"], meta["model_name"])
            model_keys.setdefault(key, []).append(meta)

        # ── Process each model group (across all windows) ────────────────────────
        for (tso_name, dataset_name, model_name), group_metas in model_keys.items():
            logger.info(
                "Processing model=%s for tso=%s dataset=%s across %d windows",
                model_name,
                tso_name,
                dataset_name,
                len(group_metas),
            )
            
            model_dataset = get_dataset(dataset_root_dir, tso_name, dataset_name)
            
            # Retrieve cached benchmarks for this dataset when evaluating predictions
            benchmarks_df = benchmark_cache[(tso_name, dataset_name)] if run_predictions else None

            # Use the first window's config for other metadata
            representative_config = group_metas[0]["run_info"]["config"]
            timestamp = _require_config(representative_config, "date_time")
            shift_hours = int(_require_config(representative_config, "shift_hours"))
            model_alias = _require_config(representative_config, "model_alias")

            # Sort windows by index to ensure proper ordering
            group_metas.sort(key=lambda x: x["window_index"])

            # ── Collect predictions from all windows ──────────────────────────────
            last_checkpoint_preds_list = [] if run_predictions and evaluate_last_checkpoint else None
            best_checkpoint_preds_list = [] if run_predictions and evaluate_best_checkpoint else None

            for meta in group_metas:
                window_index = meta["window_index"]
                model_dir = meta["model_dir"]
                run_config = meta["run_info"]["config"]
                timestamp = _require_config(run_config, "date_time")
                model_name = meta["model_name"]


                if run_predictions:
                    if evaluate_last_checkpoint and last_checkpoint_preds_list is not None:
                        last_checkpoint_preds_list = run_model_predictions_window(
                            model_dataset=model_dataset,
                            add_calendar_features=add_calendar_features,
                            holidays_path=holidays_path,
                            shift_hours=shift_hours,
                            model_alias=model_alias,
                            wandb_project=wandb_project,
                            wandb_entity=wandb_entity,
                            meta=meta,
                            checkpoint_preds_list=last_checkpoint_preds_list,
                            best_checkpoint=False,
                            output_dir=output_dir,
                        )

                    if evaluate_best_checkpoint and best_checkpoint_preds_list is not None:
                        best_checkpoint_preds_list = run_model_predictions_window(
                            model_dataset=model_dataset,
                            add_calendar_features=add_calendar_features,
                            holidays_path=holidays_path,
                            shift_hours=shift_hours,
                            model_alias=model_alias,
                            wandb_project=wandb_project,
                            wandb_entity=wandb_entity,
                            meta=meta,
                            checkpoint_preds_list=best_checkpoint_preds_list,
                            best_checkpoint=True,
                            output_dir=output_dir,
                        )

                    # ── IG Explainability for this window
                    if run_explainability:
                        if evaluate_last_checkpoint and last_checkpoint_preds_list is not None:
                            run_explainability_window(
                                model_dataset=model_dataset,
                                add_calendar_features=add_calendar_features,
                                holidays_path=holidays_path,
                                shift_hours=shift_hours,
                                meta=meta,
                                model_name=model_name,
                                model_alias=model_alias,
                                model_dir=model_dir,
                                output_dir=output_dir,
                                wandb_project=wandb_project,
                                wandb_entity=wandb_entity,
                                checkpoint_best=False,
                                explain_step_hours=explain_step_hours,
                            )
                        if evaluate_best_checkpoint and best_checkpoint_preds_list is not None:
                            run_explainability_window(
                                model_dataset=model_dataset,
                                add_calendar_features=add_calendar_features,
                                holidays_path=holidays_path,
                                shift_hours=shift_hours,
                                meta=meta,
                                model_name=model_name,
                                model_alias=model_alias,
                                model_dir=model_dir,
                                output_dir=output_dir,
                                wandb_project=wandb_project,
                                wandb_entity=wandb_entity,
                                checkpoint_best=True,
                                explain_step_hours=explain_step_hours,
                            )

            if run_predictions:
                # ── Concatenate predictions across all windows ────────────────────
                logger.info("Concatenating predictions from %d windows", len(group_metas))

                if evaluate_last_checkpoint and last_checkpoint_preds_list is not None:
                    evaluate_final_predictions(
                        checkpoint_preds_list=last_checkpoint_preds_list,
                        model_dataset=model_dataset,
                        benchmarks_df=benchmarks_df,
                        output_dir=output_dir,
                        model_name=model_name,
                        tso_name=tso_name,
                        timestamp=timestamp,
                        best_checkpoint=False,
                    )

                if evaluate_best_checkpoint and best_checkpoint_preds_list is not None:
                    evaluate_final_predictions(
                        checkpoint_preds_list=best_checkpoint_preds_list,
                        model_dataset=model_dataset,
                        benchmarks_df=benchmarks_df,
                        output_dir=output_dir,
                        model_name=model_name,
                        tso_name=tso_name,
                        timestamp=timestamp,
                        best_checkpoint=True,
                    )

                # ── Hurdle threshold optimisation (optional) ──────────────────────
                if optimize_hurdle_thresholds:
                    if last_checkpoint_preds_list is None:
                        logger.warning(
                            "Skipping hurdle threshold optimisation because it requires --checkpoint-selection last|both."
                        )
                        continue
                    window_indices = [m["window_index"] for m in group_metas]
                    optimize_and_save_hurdle_thresholds(
                        checkpoint_preds_list=last_checkpoint_preds_list,
                        window_indices=window_indices,
                        model_dataset=model_dataset,
                        output_dir=output_dir,
                        model_alias=model_alias,
                        model_name=model_name,
                        tso_name=tso_name,
                        timestamp=timestamp,
                        fbeta2_target=hurdle_theta_fbeta2_target,
                        lambda_=hurdle_theta_lambda,
                        n_grid=hurdle_theta_n_grid,
                    )
            else:
                logger.info(
                    "Skipping prediction aggregation/evaluation for model=%s because predictions are disabled.",
                    model_name,
                )
    finally:
        # Clean up all temporary extraction directories
        if temp_dirs_to_cleanup and not persist_archive_dir:
            logger.info(f"Cleaning up {len(temp_dirs_to_cleanup)} temporary extraction directories")
            for temp_dir in temp_dirs_to_cleanup:
                if temp_dir.exists() and temp_dir.is_dir():
                    # Safety check: only delete directories in our temp pattern
                    if "model_extraction_" in temp_dir.name: # and temp_dir.parent == root_model_path.parent:
                        try:
                            shutil.rmtree(temp_dir)
                            logger.info(f"Deleted temporary directory: {temp_dir}")
                        except Exception as e:
                            logger.warning(f"Failed to delete temporary directory {temp_dir}: {e}")
                    else:
                        logger.warning(f"Skipping deletion of directory {temp_dir} - does not match safety criteria")


def prepare_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model-path", type=str, help="Model directory or path to a zstd archive containing the whole neuralforecast saved data.")
    p.add_argument("--dataset-root-dir", type=str, help="Path to the main directory where datasets are kept.")
    p.add_argument(
        "--checkpoint-selection",
        choices=["last", "best", "both"],
        default="last",
        help="Which checkpoint(s) to evaluate (default: last).",
    )
    p.add_argument("--wandb-project", type=str, help="Weights and Biases project name for logging.")
    p.add_argument("--wandb-entity", type=str, default=None, help="Weights and Biases entity name for logging.")
    p.add_argument("--no-calendar", action="store_true", help="Skip adding calendar features.")
    p.add_argument("--n-threads", type=int, default=None, help="Number of threads to use torch. Default to all available.")
    p.add_argument("--persist-archive-dir", action="store_true", help="Whether to persist the extracted archive directory (with a 'model_extraction_' prefix) instead of cleaning up after evaluation. Useful for debugging.")
    p.add_argument("--holidays-path", default=None, help="Override for holidays CSV path.")
    p.add_argument("--output-dir", type=str, help="Output directory path.")
    p.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help=(
            "Which benchmarks to compute.  Accepted values: "
            "naive_seasonal, linear_regression, gradient_boosted_trees, "
            "auto_arima, seasonal_regression.  Defaults to all."
        ),
    )
    p.add_argument("--benchmark-config-path", type=str, default=None, help="Path to yaml file with benchmark configuration (lags, hyperparameters, etc.).")
    p.add_argument(
        "--explain",
        action="store_true",
        help="Run Integrated Gradients explainability per window after predictions.",
    )
    p.add_argument(
        "--explain-step-hours",
        type=int,
        default=24,
        help="Stride (in hours) between IG explanation windows (default: 24 = daily).",
    )
    p.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Skip prediction/benchmark/evaluation steps and run only explainability.",
    )
    p.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help=(
            "Skip benchmark computation. Predictions are still generated and evaluated, "
            "but no comparison against naive/statistical baselines is performed. "
            "Useful for faster iteration when benchmark scores are not needed."
        ),
    )
    p.add_argument("--start-window", type=int, default=0, help="Index of the first window to process (inclusive).")
    p.add_argument("--force", action="store_true", help="Force re-computation of predictions, benchmarks, and explainability even if they already exist in W&B.")
    p.add_argument(
        "--optimize-hurdle-thresholds",
        action="store_true",
        help=(
            "Optimise the hurdle probability threshold θ for each rolling window and "
            "overall.  Results are saved to JSON files in --output-dir.  Only applies "
            "to hurdle models (predictions must contain a non_zero_logit column)."
        ),
    )
    p.add_argument(
        "--hurdle-theta-fbeta2-target",
        type=float,
        default=0.5,
        help="Target Fβ2 score used in the threshold optimisation penalty (default: 0.5).",
    )
    p.add_argument(
        "--hurdle-theta-lambda",
        type=float,
        default=0.5,
        help="Penalty weight λ for the Fβ2 shortfall in threshold optimisation (default: 0.5).",
    )
    p.add_argument(
        "--hurdle-theta-n-grid",
        type=int,
        default=200,
        help="Number of candidate threshold values to evaluate in (0, 1) (default: 200).",
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
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        add_calendar_features=not args.no_calendar,
        holidays_path=args.holidays_path,
        enabled_benchmarks=args.benchmarks,
        benchmark_config_path=Path(args.benchmark_config_path) if args.benchmark_config_path else None,
        checkpoint_selection=args.checkpoint_selection,
        persist_archive_dir=args.persist_archive_dir,
        run_explainability=args.explain,
        run_predictions=not args.skip_predictions,
        skip_benchmarks=args.skip_benchmarks,
        explain_step_hours=args.explain_step_hours,
        start_window=args.start_window,
        force=args.force,
        optimize_hurdle_thresholds=args.optimize_hurdle_thresholds,
        hurdle_theta_fbeta2_target=args.hurdle_theta_fbeta2_target,
        hurdle_theta_lambda=args.hurdle_theta_lambda,
        hurdle_theta_n_grid=args.hurdle_theta_n_grid,
    )

