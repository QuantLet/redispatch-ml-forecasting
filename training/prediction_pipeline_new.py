import json
import logging
import argparse
import os
import tempfile
from contextlib import contextmanager
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
import zstandard as zstd
import tarfile
import wandb

from pathlib import Path
from neuralforecast import NeuralForecast
from training.runner import CHECKPOINT_BEST_NAME, _load_tft_interp_local
from typing import cast, Literal

from training.predict import predict_with_shift_correction, evaluate_models, log_predictions_to_wandb, prepare_predictions_df
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
    save_tft_interpretability,
    aggregate_ig_stats,
    log_ig_summary_tables,
    _save_ig_stride_to_artifact,
)
from training.train_pipeline import set_n_threads

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


def read_models(model_path: Path):
    archive_files = [file for file in model_path.glob("*.tar.zst")]

    if len(archive_files) > 0:
        logger.info(f"Found {len(archive_files)} zst archived models")
        for archive_path in archive_files:
            logger.info(f"Inflating model from {archive_path}")
            model_dir = archive_path.with_suffix("").with_suffix("")
            model_dir.mkdir(exist_ok=True)
            model_dir = model_dir.resolve()
            # Extract the archive to the model directory
            with open(archive_path, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    with tarfile.open(fileobj=reader, mode="r|*") as tar:
                        tar.extractall(path=model_dir)
            move_files_from_last_nonempty_dir(model_dir, model_dir)  # Move files from last non-empty subdirectory to model_dir
            yield model_dir
    else:
        logger.info(f"No archived models found in {model_path}, looking for directories.")
        for model_dir in model_path.iterdir():
            if model_dir.is_dir() and any("nf_model" in dir.name for dir in model_dir.iterdir() if dir.is_dir()):
                yield (model_dir / "nf_model").resolve()


def get_last_checkpoint_path(nf_dir: Path):
    checkpoints = list(nf_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {nf_dir / 'checkpoints'}")
    
    best_checkpoint_path = nf_dir / CHECKPOINT_BEST_NAME
    last_checkpoint = set(checkpoints) - {best_checkpoint_path}
    if not len(last_checkpoint) == 1:
        raise ValueError(f"Expected exactly one last checkpoint in {nf_dir / 'checkpoints'}, found {len(last_checkpoint)}")
    
    return last_checkpoint.pop()


def load_model(model_dir: Path, checkpoint_best: bool) -> NeuralForecast:
    best_valid_checkpoint = model_dir / CHECKPOINT_BEST_NAME
    best_valid_checkpoint_tmp_path = best_valid_checkpoint.with_suffix(".bkp")
    last_checkpoint_path = get_last_checkpoint_path(model_dir)
    last_checkpoint_tmp_path = last_checkpoint_path.with_suffix(".bkp")
    try:
        if not checkpoint_best:
            if best_valid_checkpoint.exists():
                best_valid_checkpoint.rename(best_valid_checkpoint_tmp_path)
            nf_model = NeuralForecast.load(str(model_dir.resolve()))
        else:
            # Rename the best valid checkpoint to the expected name for loading
            # and rename the last checkpoint to avoid confusion
            if best_valid_checkpoint_tmp_path.exists():
                best_valid_checkpoint_tmp_path.rename(best_valid_checkpoint)
            if not best_valid_checkpoint.exists():
                raise FileNotFoundError(f"Best valid checkpoint not found in {model_dir}")
            last_checkpoint_path.rename(last_checkpoint_tmp_path)
            best_valid_checkpoint.rename(last_checkpoint_path)
            nf_model = NeuralForecast.load(str(model_dir.resolve()))
        _dedupe_model_callbacks(nf_model)
        return nf_model
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_dir} with checkpoint_best={checkpoint_best}") from e
    finally:
        if best_valid_checkpoint_tmp_path.exists():
            best_valid_checkpoint_tmp_path.rename(best_valid_checkpoint)
        elif last_checkpoint_tmp_path.exists():
            last_checkpoint_path.rename(best_valid_checkpoint)
            last_checkpoint_tmp_path.rename(last_checkpoint_path)


def get_model_name(model_dir: Path) -> str:
    # Extract the model name from the model directory structure
    model_name = model_dir.with_suffix("").with_suffix("").name  # Remove .tar.zst and potential suffixes
    return model_name


def identify_wandb_run(model_dir: Path, wandb_entity: str | None, wandb_project: str) -> dict:
    """Locate the existing W&B run for this model checkpoint and return metadata.

    Returns a dict with keys: run_id, config.
    """
    model_label = get_model_name(model_dir)
    timestamp = model_dir.parent.name

    api = wandb.Api()
    project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
    filters = {
        "config.date_time": timestamp,
        "config.model_alias": model_label,
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
    best_checkpoint: bool,
) -> pd.DataFrame:
    """Retrieve predictions artifact logged by training.predict.log_predictions_to_wandb."""
    if best_checkpoint:
        checkpoint_label = "test_best_valid"
    else:
        checkpoint_label = "test"
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


def get_naive_benchmark(dataset: pd.DataFrame, forecast_horizon: int):
    naive_benchmark = (
        dataset.groupby("unique_id")[["y", "ds"]].apply(lambda g: g.set_index("ds")["y"].shift(forecast_horizon)).T
        .reset_index().melt(id_vars=["ds"], value_name="naive_benchmark")
    )
    merged_dataset = dataset[["ds", "unique_id", "y"]].merge(naive_benchmark, how="left", on=["unique_id", "ds"])
    return merged_dataset


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
        pred_end=dataset["ds"].max(),
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
    pred_end_override: pd.Timestamp | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Run IG explainability for a model checkpoint with shift correction.

    Parameters
    ----------
    pred_end_override : pd.Timestamp | None
        When set, overrides the default ``dataset["ds"].max()`` as the
        last physical timestamp to explain (inclusive).  Used to restrict
        the run to a dev / pre-evaluation period.
    """
    shift_hours = int(_require_config(run_config, "shift_hours"))
    test_start_date = pd.Timestamp(_require_config(run_config, "test_start"))
    forecast_horizon = int(_require_config(run_config, "forecast_horizon"))
    pred_end = pred_end_override if pred_end_override is not None else dataset["ds"].max()
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
        pred_end=pred_end,
        future_cov_cols=future_cov_cols,
        shift_hours=shift_hours,
        forecast_horizon=forecast_horizon,
        tso=tso_name,
        holidays_path=holidays_path,
        step_hours=step_hours,
    )


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


def _retrieve_tft_interp_from_wandb(
    wandb_entity: str | None,
    wandb_project: str,
    artifact_name: str,
    download_dir: Path,
) -> dict | None:
    """Download a TFT interpretability artifact from W&B and load it.

    Downloads into *download_dir* so the files are available locally as a
    side-effect.  Returns the loaded dict or ``None`` if the artifact is
    not found or cannot be parsed.
    """
    project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
    try:
        art = wandb.Api().artifact(f"{project_path}/{artifact_name}:latest")
        dl_path = Path(art.download(root=str(download_dir)))
        result = _load_tft_interp_local(dl_path)
        if result is not None:
            logger.info("Loaded TFT interpretability from W&B artifact: %s", artifact_name)
        return result
    except Exception as exc:
        logger.debug(
            "TFT interp artifact '%s' not found in W&B: %s", artifact_name, exc
        )
        return None


def _load_metadata_json(dataset_path: str | Path) -> dict | None:
    """Load the JSON sidecar metadata file for a dataset parquet file."""
    path = Path(dataset_path).with_suffix(".json")
    if not path.exists():
        logger.debug("No metadata JSON found at %s", path)
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load dataset metadata from %s: %s", path, exc)
        return None


def check_dataset_compatibility(
    run_config: dict,
    supplied_dataset: pd.DataFrame,
    supplied_dataset_path: Path,
) -> tuple[Literal["compatible", "extended", "incompatible"], pd.Timestamp | None]:
    """Compare the supplied dataset against the dataset used during training.

    Returns
    -------
    ("compatible", None)
        Supplied dataset matches training data or ends earlier; no new period to cover.
    ("extended", first_new_ts)
        Supplied dataset extends further in time; predictions/explainability needed
        from *first_new_ts* onwards.
    ("incompatible", None)
        Structural (columns, feature-sets) or value mismatch detected; a new
        training run is recommended.
    """
    train_dataset_path = run_config.get("dataset_local_path", "")
    if not train_dataset_path:
        logger.warning("No 'dataset_local_path' in run config – skipping compatibility check.")
        return "compatible", None

    train_meta = _load_metadata_json(train_dataset_path)
    supplied_meta = _load_metadata_json(supplied_dataset_path)

    if train_meta is None or supplied_meta is None:
        logger.warning(
            "Could not load one or both dataset metadata JSONs – skipping compatibility check."
        )
        return "compatible", None

    # ── Column check ─────────────────────────────────────────────────────────
    train_cols = set(train_meta.get("columns", []))
    supplied_cols = set(supplied_meta.get("columns", []))
    if train_cols != supplied_cols:
        missing = train_cols - supplied_cols
        extra = supplied_cols - train_cols
        logger.warning(
            "Dataset column mismatch (missing=%s, extra=%s). "
            "A new training run is recommended.",
            missing,
            extra,
        )
        return "incompatible", None

    # ── Feature-set check ────────────────────────────────────────────────────
    train_fs = set(train_meta.get("feature_sets", []))
    supplied_fs = set(supplied_meta.get("feature_sets", []))
    if train_fs != supplied_fs:
        logger.warning(
            "Dataset feature-set mismatch (%s vs %s). "
            "A new training run is recommended.",
            train_fs,
            supplied_fs,
        )
        return "incompatible", None

    # ── Timeline check ───────────────────────────────────────────────────────
    train_start = pd.Timestamp(train_meta["date_range"]["start"])
    train_end = pd.Timestamp(train_meta["date_range"]["end"])
    supplied_start = pd.Timestamp(supplied_meta["date_range"]["start"])
    supplied_end = pd.Timestamp(supplied_meta["date_range"]["end"])

    if supplied_start > train_start:
        logger.warning(
            "Supplied dataset starts at %s, but training data started at %s. "
            "The supplied dataset must cover the full training history. "
            "A new training run is recommended.",
            supplied_start,
            train_start,
        )
        return "incompatible", None

    # ── Value check on the overlapping portion (sample-based) ────────────────
    train_path_resolved = Path(train_dataset_path).resolve()
    supplied_path_resolved = supplied_dataset_path.resolve()
    overlap_end = min(train_end, supplied_end)

    # if train_path_resolved != supplied_path_resolved:
    #     try:
    #         train_df_raw, _ = load_dataset(str(train_path_resolved))
    #         train_nixtla = to_nixtla_format(train_df_raw)
    #         sample_ts = sorted(t for t in supplied_dataset["ds"].unique() if t <= overlap_end)[-100:]
    #         if sample_ts:
    #             sup_sample = (
    #                 supplied_dataset[supplied_dataset["ds"].isin(sample_ts)]
    #                 .sort_values(["unique_id", "ds"])
    #                 .reset_index(drop=True)
    #             )
    #             trn_sample = (
    #                 train_nixtla[train_nixtla["ds"].isin(sample_ts)]
    #                 .sort_values(["unique_id", "ds"])
    #                 .reset_index(drop=True)
    #             )
    #             numeric_cols = [
    #                 c for c in sup_sample.columns
    #                 if c in trn_sample.columns
    #                 and c not in ("ds", "unique_id")
    #                 and pd.api.types.is_numeric_dtype(sup_sample[c])
    #             ]
    #             for col in numeric_cols:
    #                 a = sup_sample[col].fillna(0).to_numpy()
    #                 b = trn_sample[col].fillna(0).to_numpy()
    #                 if len(a) == len(b) and not np.allclose(a, b, rtol=1e-4, atol=1e-6):
    #                     logger.warning(
    #                         "Dataset value mismatch detected in column '%s' over "
    #                         "the overlapping period. A new training run is recommended.",
    #                         col,
    #                     )
    #                     return "incompatible", None
    #     except Exception as exc:
    #         logger.warning(
    #             "Could not perform value-level dataset comparison (%s); "
    #             "proceeding with caution.",
    #             exc,
    #         )

    # ── Extension check ───────────────────────────────────────────────────────
    if supplied_end > train_end:
        first_new_ts = train_end + pd.Timedelta(hours=1)
        logger.info(
            "Dataset is extended: new data from %s to %s.",
            first_new_ts,
            supplied_end,
        )
        return "extended", first_new_ts

    logger.info(
        "Dataset is compatible and not extended "
        "(supplied end: %s, training end: %s).",
        supplied_end,
        train_end,
    )
    return "compatible", None


def _get_artifact_ds_max(
    wandb_entity: str | None,
    wandb_project: str,
    artifact_name: str,
) -> pd.Timestamp | None:
    """Return the ``ds_max`` field from an artifact's metadata, or ``None``."""
    project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
    try:
        art = wandb.Api().artifact(f"{project_path}/{artifact_name}:latest")
        ds_max_str = art.metadata.get("ds_max")
        if ds_max_str:
            return pd.Timestamp(ds_max_str)
    except Exception:
        pass
    return None


def _extend_ig_artifacts(
    run: "wandb.Run",
    wandb_entity: str | None,
    wandb_project: str,
    ig_artifact_name: str,
    ig_preds_artifact_name: str,
    new_ig_explanations: dict,
    new_ig_preds: pd.DataFrame,
    model_alias: str,
    tso: str,
    timestamp: str,
    shift_hours: int,
    output_dir: Path | None = None,
) -> None:
    """Extend existing IG W&B artifacts (raw strides + predictions) with new strides.

    Downloads the current ``:latest`` version of each artifact, appends the new
    strides (renumbered to continue from the last existing index), and logs a new
    artifact version so that ``:latest`` always covers the full period.
    """
    project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
    tso_norm = tso.replace(" ", "_")

    # ── Extend per-model raw IG stride artifacts ──────────────────────────────
    for model_label, new_strides in new_ig_explanations.items():
        if not new_strides:
            continue

        art_name = f"ig_raw_{model_label}_{tso_norm}_{timestamp}"
        n_old_strides = 0
        old_art = None
        try:
            old_art = wandb.Api().artifact(f"{project_path}/{art_name}:latest")
            n_old_strides = old_art.metadata.get("n_strides", 0)
        except Exception as exc:
            logger.debug("Could not retrieve existing IG artifact %s: %s", art_name, exc)

        new_art = wandb.Artifact(
            name=art_name,
            type="ig_explanation",
            metadata={
                "model_alias": model_label,
                "tso": tso,
                "shift_hours": shift_hours,
                "timestamp": timestamp,
                "window_index": None,
                "n_strides": n_old_strides + len(new_strides),
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            if old_art is not None and n_old_strides > 0:
                old_art.download(root=tmpdir)
                for fpath in sorted(Path(tmpdir).glob("stride_*.pt")):
                    new_art.add_file(str(fpath), name=fpath.name)
                for fpath in sorted(Path(tmpdir).glob("stride_*.npy")):
                    new_art.add_file(str(fpath), name=fpath.name)
            for i, ig_result in enumerate(new_strides):
                _save_ig_stride_to_artifact(new_art, ig_result, n_old_strides + i)
            run.log_artifact(new_art)
        logger.info(
            "Extended IG artifact '%s': %d old + %d new strides",
            art_name,
            n_old_strides,
            len(new_strides),
        )

        if output_dir is not None:
            local_dir = output_dir / "ig_raw" / model_label
            local_dir.mkdir(parents=True, exist_ok=True)
            for i, ig_result in enumerate(new_strides):
                for key, value in ig_result.items():
                    global_idx = n_old_strides + i
                    if hasattr(value, "cpu"):
                        import torch
                        torch.save(
                            value.cpu(),
                            local_dir / f"stride_{global_idx:04d}_{key}.pt",
                        )
                    elif isinstance(value, np.ndarray):
                        np.save(
                            local_dir / f"stride_{global_idx:04d}_{key}.npy",
                            value,
                        )

    # ── Extend IG predictions artifact ───────────────────────────────────────
    if not new_ig_preds.empty:
        preds_art_name = f"ig_preds_{model_alias}_{tso_norm}_{timestamp}"
        old_preds_df = pd.DataFrame()
        try:
            old_preds_art = wandb.Api().artifact(f"{project_path}/{preds_art_name}:latest")
            with tempfile.TemporaryDirectory() as tmpdir:
                dl_path = Path(old_preds_art.download(root=tmpdir))
                pq_path = dl_path / "ig_predictions.parquet"
                if pq_path.exists():
                    old_preds_df = pd.read_parquet(pq_path)
        except Exception as exc:
            logger.debug("Could not retrieve existing IG predictions artifact: %s", exc)

        combined_preds = (
            pd.concat([old_preds_df, new_ig_preds], ignore_index=True)
            .drop_duplicates(subset=["unique_id", "ds"], keep="last")
            .sort_values(["unique_id", "ds"])
            .reset_index(drop=True)
        ) if not old_preds_df.empty else new_ig_preds

        new_preds_art = wandb.Artifact(
            name=preds_art_name,
            type="ig_predictions",
            metadata={
                "model_alias": model_alias,
                "tso": tso,
                "shift_hours": shift_hours,
                "timestamp": timestamp,
                "window_index": None,
                "n_rows": len(combined_preds),
                "ds_min": str(combined_preds["ds"].min()),
                "ds_max": str(combined_preds["ds"].max()),
            },
        )
        with new_preds_art.new_file("ig_predictions.parquet", mode="wb") as f:
            combined_preds.to_parquet(f)
        run.log_artifact(new_preds_art)

        if output_dir is not None:
            local_preds_dir = output_dir / "ig_preds"
            local_preds_dir.mkdir(parents=True, exist_ok=True)
            combined_preds.to_parquet(
                local_preds_dir / f"ig_preds_{model_alias}.parquet"
            )


def persist_to_wandb(run: wandb.Run, evaluation_df: pd.DataFrame, model_name: str, tso_name: str, timestamp: str, best_checkpoint: bool,):
    """Persist the evaluation dataframe to wandb as a Table object."""
    try:
        table = wandb.Table(dataframe=evaluation_df)
        artifact_table_name = f"{model_name}_{tso_name}_{timestamp}_{run.id}"
        if best_checkpoint:
            artifact_table_name += "_best_checkpoint"
        artifact = wandb.Artifact(artifact_table_name, type="evaluation")
        artifact.add(table, f"evaluation_test/{artifact_table_name}")
        run.log_artifact(artifact)
    except Exception:
        logger.exception("Error logging evaluation to W&B for model %s", model_name)


def save_predictions_locally(output_dir: Path, preds_df: pd.DataFrame, tso_name: str, best_checkpoint: bool):
    """Save the prediction dataframe locally as csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"predictions_{tso_name}"
    if best_checkpoint:
        filename += "_best_checkpoint"
    filename += ".csv"
    output_path = output_dir / filename
    preds_df.to_csv(output_path)
    logger.info(f"Saved predictions to {output_path}")


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

def evaluation_pipeline(predictions: pd.DataFrame, benchmarks_df: pd.DataFrame, model_dataset: pd.DataFrame, start_date: pd.Timestamp | None = None, beta: float = 1.0):
    if start_date is not None:
        effective_start = max(start_date, predictions["ds"].min())
        predictions = predictions[predictions["ds"] >= effective_start]
        benchmarks_df = benchmarks_df[benchmarks_df["ds"] >= effective_start]

    # Regression split may remove last days from target_ts, so forward-fill benchmarks to align
    final_predictions_df = predictions.merge(benchmarks_df, on=["unique_id", "ds"], how="left").ffill()

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


def run_predictions_for_model(
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
) -> pd.DataFrame:
    """Fetch or compute predictions for one checkpoint type, evaluate, and persist."""
    timestamp = _require_config(run_config, "date_time")
    shift_hours = int(_require_config(run_config, "shift_hours"))
    model_alias = _require_config(run_config, "model_alias")
    split_name = "test_best_valid" if best_checkpoint else "test"

    with wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=run_id,
        resume="allow",
        mode="online",
    ) as wandb_run:
        nf = load_model(model_dir, checkpoint_best=best_checkpoint)

        predictions = retrieve_model_predictions_from_wandb(
            wandb_entity,
            wandb_project,
            model_name=get_model_name(model_dir),
            tso_name=tso_name,
            timestamp=timestamp,
            best_checkpoint=best_checkpoint,
        )
        if predictions.empty:
            predictions = predict_checkpoint(
                nf_model=nf,
                dataset=model_dataset,
                tso_name=tso_name,
                run_config=run_config,
                add_calendar_features=add_calendar_features,
                holidays_path=holidays_path,
            )
            log_predictions_to_wandb(
                wandb_run,
                predictions.merge(
                    model_dataset[["unique_id", "ds", "y"]],
                    on=["unique_id", "ds"],
                    how="left",
                ),
                split_name=split_name,
                model_alias=model_alias,
                shift_hours=shift_hours,
                tso=tso_name,
                timestamp=timestamp,
                log_table=False,
            )
        elif dataset_compat_status == "extended":
            last_pred_ts = predictions["ds"].max()
            if model_dataset["ds"].max() > last_pred_ts:
                tail_pred_start = last_pred_ts + pd.Timedelta(hours=1)
                logger.info(
                    "Extending %s predictions for %s from %s",
                    split_name, model_alias, tail_pred_start,
                )
                tail_preds = predict_checkpoint(
                    nf_model=nf,
                    dataset=model_dataset,
                    run_config={**run_config, "test_start": str(tail_pred_start)},
                    tso_name=tso_name,
                    add_calendar_features=add_calendar_features,
                    holidays_path=holidays_path,
                )
                # Normalise both parts to the same (processed) column format
                # before concatenation.  The W&B artifact may have been stored
                # in either raw (contains "-median") or already-processed format
                # depending on which pipeline version originally ran, while fresh
                # tail predictions are always raw.  Applying prepare_predictions_df
                # to both eliminates duplicate columns that would otherwise arise
                # when the "-median" rename step encounters both "model-median" and
                # "model" as separate columns.
                predictions = prepare_predictions_df(predictions, model_dataset)
                tail_preds_with_y = tail_preds.merge(
                    model_dataset[["unique_id", "ds", "y"]],
                    on=["unique_id", "ds"],
                    how="left",
                )
                tail_preds_processed = prepare_predictions_df(tail_preds_with_y, model_dataset)
                predictions = (
                    pd.concat(
                        [predictions, tail_preds_processed],
                        ignore_index=True,
                    )
                    .drop_duplicates(subset=["unique_id", "ds"], keep="last")
                    .sort_values(["unique_id", "ds"])
                    .reset_index(drop=True)
                )
                log_predictions_to_wandb(
                    wandb_run,
                    predictions,
                    split_name=split_name,
                    model_alias=model_alias,
                    shift_hours=shift_hours,
                    tso=tso_name,
                    timestamp=timestamp,
                    log_table=False,
                )

        evaluation_df = evaluation_pipeline(
            model_dataset=model_dataset,
            predictions=predictions,
            benchmarks_df=benchmarks_df,
            start_date=start_date,
            beta=2.0,
        )
        save_evaluation_locally(
            output_dir,
            evaluation_df=evaluation_df,
            model_name=get_model_name(model_dir),
            tso_name=tso_name,
            timestamp=timestamp,
            best_checkpoint=best_checkpoint,
        )
        persist_to_wandb(
            wandb_run,
            evaluation_df,
            get_model_name(model_dir),
            tso_name,
            best_checkpoint=best_checkpoint,
            timestamp=timestamp,
        )
    return predictions


def run_explainability_for_model(
    *,
    model_dir: Path,
    model_dataset: pd.DataFrame,
    run_config: dict,
    run_id: str,
    tso_name: str,
    dataset_compat_status: str,
    wandb_project: str,
    wandb_entity: str | None,
    output_dir: Path,
    add_calendar_features: bool,
    holidays_path: str | None,
    explain_step_hours: int,
    start_date: pd.Timestamp | None,
    best_checkpoint: bool = False,
    force: bool = False,
) -> None:
    """Compute or retrieve IG explainability and TFT interpretability for one model."""
    timestamp = _require_config(run_config, "date_time")
    shift_hours = int(_require_config(run_config, "shift_hours"))
    model_alias = _require_config(run_config, "model_alias")
    test_start = pd.Timestamp(_require_config(run_config, "test_start"))
    tso_norm = tso_name.replace(" ", "_")

    # Resolve the effective evaluation start: clip to test_start if necessary.
    eval_start = max(start_date, test_start) if start_date is not None else test_start
    has_dev_period = eval_start > test_start

    ig_artifact_name = f"ig_raw_{model_alias}_{tso_norm}_{timestamp}"
    ig_dev_artifact_name = f"ig_dev_raw_{model_alias}_{tso_norm}_{timestamp}"
    tft_artifact_name = f"tft_interp_{model_alias}_{tso_norm}_{timestamp}"
    tft_local_dir = output_dir / "tft_interpretability" / model_alias

    ig_art_exists = _wandb_artifact_exists(
        wandb_entity, wandb_project, ig_artifact_name
    )
    ig_art_dev_exists = _wandb_artifact_exists(
        wandb_entity, wandb_project, ig_dev_artifact_name
    )

    with wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=run_id,
        resume="allow",
        mode="online",
    ) as wandb_run:
        nf_loaded: NeuralForecast | None = None

        # ── Integrated-Gradients explainability ──────────────────────────────
        if force or not ig_art_exists:
            logger.info("Running IG explainability for %s", model_alias)
            nf_loaded = load_model(model_dir, checkpoint_best=best_checkpoint)

            # Main period: [eval_start, pred_end]
            ig_explanations, ig_preds = explain_checkpoint(
                nf_model=nf_loaded,
                dataset=model_dataset,
                run_config={
                    **run_config,
                    "test_start": str(eval_start),
                },
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
                output_dir=output_dir,
            )
            _log_ig_summary_tables(wandb_run, ig_explanations, run_config)

        else:
            logger.info(
                "IG artifact '%s' already in W&B – skipping recomputation.",
                ig_artifact_name,
            )

        # Dev period: [test_start, eval_start) – saved to separate artifact
        if force or (has_dev_period and not ig_art_dev_exists):
            logger.info("Running IG dev  explainability for %s", model_alias)
            logger.info("Dev period: %s to %s", test_start, eval_start - pd.Timedelta(hours=1))
            nf_loaded = load_model(model_dir, checkpoint_best=best_checkpoint)
            dev_explanations, dev_preds = explain_checkpoint(
                nf_model=nf_loaded,
                dataset=model_dataset,
                run_config={
                    **run_config,
                    "test_start": str(test_start),
                },
                tso_name=tso_name,
                add_calendar_features=add_calendar_features,
                holidays_path=holidays_path,
                step_hours=explain_step_hours,
                pred_end_override=eval_start - pd.Timedelta(hours=1),
            )
            save_explanation_artifacts(
                run=wandb_run,
                explanations=dev_explanations,
                predictions=dev_preds,
                model_alias=model_alias,
                tso=tso_name,
                timestamp=timestamp,
                shift_hours=shift_hours,
                output_dir=output_dir,
                artifact_type="ig_dev_explanation",
                preds_artifact_type="ig_dev_predictions",
                raw_artifact_name_prefix="ig_dev_raw",
                preds_artifact_name_prefix="ig_dev_preds",
            )
        else:
            logger.info(
                "IG dev artifact '%s' already in W&B – skipping recomputation.",
                ig_dev_artifact_name,
            )

        # ── TFT interpretability ─────────────────────────────────────────────
        if dataset_compat_status == "extended":
            logger.info(
                "Recomputing TFT interpretability for extended dataset (%s)",
                model_alias,
            )
            if nf_loaded is None:
                nf_loaded = load_model(model_dir, checkpoint_best=best_checkpoint)
            predict_checkpoint(
                nf_model=nf_loaded,
                dataset=model_dataset,
                run_config=run_config,
                tso_name=tso_name,
                add_calendar_features=add_calendar_features,
                holidays_path=holidays_path,
            )
            save_tft_interpretability(
                nf=nf_loaded,
                output_dir=output_dir,
                run=wandb_run,
                model_alias=model_alias,
                window_index=None,
                tso=tso_name,
                timestamp=timestamp,
            )
        else:
            tft_interp = _retrieve_tft_interp_from_wandb(
                wandb_entity, wandb_project, tft_artifact_name, tft_local_dir,
            )
            if tft_interp is None:
                tft_interp = _load_tft_interp_local(tft_local_dir)
                if tft_interp is not None:
                    logger.info(
                        "Loaded TFT interpretability from local files: %s",
                        tft_local_dir,
                    )
            if tft_interp is None:
                logger.info(
                    "TFT interpretability not found; computing from model checkpoint for %s",
                    model_alias,
                )
                if nf_loaded is None:
                    nf_loaded = load_model(model_dir, checkpoint_best=best_checkpoint)
                    predict_checkpoint(
                        nf_model=nf_loaded,
                        dataset=model_dataset,
                        run_config=run_config,
                        tso_name=tso_name,
                        add_calendar_features=add_calendar_features,
                        holidays_path=holidays_path,
                    )
                tft_interp = save_tft_interpretability(
                    nf=nf_loaded,
                    output_dir=output_dir,
                    run=wandb_run,
                    model_alias=model_alias,
                    window_index=None,
                    tso=tso_name,
                    timestamp=timestamp,
                )
                if tft_interp is None:
                    logger.info(
                        "No TFT model found in checkpoint – "
                        "skipping TFT interpretability for %s.",
                        model_alias,
                    )


def _log_ig_summary_tables(
    wandb_run: "wandb.Run",
    ig_explanations: dict,
    run_config: dict,
) -> None:
    """Log IG summary tables for hist/futr covariates to W&B."""
    hist_cov = run_config.get("historical_covariates", [])
    futr_cov = run_config.get("future_covariates", [])
    for feat_type, feat_names in [("hist_exog", hist_cov), ("futr_exog", futr_cov)]:
        if feat_names:
            summaries = aggregate_ig_stats(
                ig_explanations,
                feature_names=feat_names,
                feature_type=feat_type,
            )
            log_ig_summary_tables(
                wandb_run, summaries, prefix=f"ig_{feat_type}",
            )


def save_lightgbm_feature_importance(metadata: dict, output_dir: Path, key: str) -> dict:
    """Aggregate SHAP feature importances and save two CSV files.

    Handles both single-window (``metadata[key]["importance"]``) and
    rolling-window (``metadata[key]["windows"][w]["importance"]``) structures.

    Writes:
    * ``{key}_feature_importance_by_direction.csv``
    * ``{key}_feature_importance_by_direction_horizon.csv``
    """
    key_meta = metadata.get(key)
    if not isinstance(key_meta, dict):
        raise KeyError(f"Missing or invalid metadata for benchmark key '{key}'.")

    if "windows" in key_meta:
        # Rolling-window structure: each window holds
        # {"importance": {"direction": df, "horizon": df}, ...}
        windows_meta = key_meta.get("windows", {})
        if not isinstance(windows_meta, dict) or not windows_meta:
            raise ValueError(f"Rolling metadata for '{key}' has no windows.")

        logger.info("Found %d LightGBM rolling windows for %s", len(windows_meta), key)

        direction_dfs: list[pd.DataFrame] = []
        horizon_dfs: list[pd.DataFrame] = []

        for window, window_dict in windows_meta.items():
            if not isinstance(window_dict, dict):
                continue
            importance = window_dict.get("importance")
            if not isinstance(importance, dict):
                continue

            direction_df = importance.get("direction")
            if isinstance(direction_df, pd.DataFrame) and not direction_df.empty:
                direction_dfs.append(direction_df.assign(window=window))

            horizon_df = importance.get("horizon")
            if isinstance(horizon_df, pd.DataFrame) and not horizon_df.empty:
                horizon_dfs.append(horizon_df.assign(window=window))

        if not direction_dfs or not horizon_dfs:
            raise ValueError(
                f"No LightGBM feature-importance DataFrames found in rolling metadata for '{key}'."
            )

        agg_df = pd.concat(direction_dfs, ignore_index=True)
        agg_df_horiz = pd.concat(horizon_dfs, ignore_index=True)
    elif "importance" in key_meta:
        # Single-window structure: importance is directly in the benchmark metadata
        importance = key_meta.get("importance", {})
        if not isinstance(importance, dict):
            raise ValueError(f"Invalid single-window importance structure for '{key}'.")

        agg_df = importance.get("direction")
        agg_df_horiz = importance.get("horizon")
        if not isinstance(agg_df, pd.DataFrame) or not isinstance(agg_df_horiz, pd.DataFrame):
            raise ValueError(f"Missing direction/horizon importance DataFrames for '{key}'.")
    else:
        raise ValueError(f"No supported LightGBM importance structure found for '{key}'.")

    dir_path = output_dir / f"{key}_feature_importance_by_direction.csv"
    dir_horiz_path = output_dir / f"{key}_feature_importance_by_direction_horizon.csv"

    agg_df.to_csv(dir_path, index=False)
    agg_df_horiz.to_csv(dir_horiz_path, index=False)

    logger.info(
        "Saved feature importance (by direction, %d features) to %s",
        agg_df["feature"].nunique(), dir_path,
    )
    logger.info(
        "Saved feature importance (by direction+horizon, %d features) to %s",
        agg_df_horiz["feature"].nunique(), dir_horiz_path,
    )
    return metadata


def merge_tso_data(predictions_list: list[pd.DataFrame], original_df: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["unique_id", "ds", "horizon", "y"]
    predictions_list_processed = [
        prepare_predictions_df(preds_df, original_df) for preds_df in predictions_list
    ]
    if predictions_list:
        result = predictions_list_processed[0]
        for to_merge in predictions_list_processed[1:]:
            result = pd.merge(result, to_merge, on=merge_cols, how="outer")
    else:
        result = pd.DataFrame()
    return result


def main(
    root_model_path: Path,
    dataset_root_dir: Path,
    wandb_project: str,
    wandb_entity: str | None,
    output_dir: Path,
    add_calendar_features: bool,
    holidays_path: str | None,
    benchmark_config_path: Path | None,
    enabled_benchmarks: list[str] | None = None,
    test_best_checkpoint: bool = False,
    run_explainability: bool = False,
    run_predictions: bool = False,
    explain_step_hours: int = 24,
    start_date: pd.Timestamp | None = None,
):
    # Setup file logging
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "prediction_pipeline.log"
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info("=" * 80)
    logger.info("Starting single window prediction pipeline")
    logger.info("Logging to file: %s", log_file)
    logger.info("=" * 80)
    
    # ── Discover models and group by (tso, dataset_name) so benchmarks are
    #    computed *once* per dataset rather than once per neural checkpoint.
    model_dirs = list(read_models(root_model_path))
    if not model_dirs:
        logger.warning("No models found in %s", root_model_path)
        return

    # Collect per-model metadata first
    model_metas: list[dict] = []
    for model_dir in model_dirs:
        run_info = identify_wandb_run(model_dir, wandb_entity, wandb_project)
        run_config = run_info["config"]
        tso_name = str(_require_config(run_config, "tso")).replace(" ", "_")
        dataset_name = str(_require_config(run_config, "dataset_contents")).replace(", ", "_")
        model_metas.append({
            "model_dir": model_dir,
            "run_info": run_info,
            "tso_name": tso_name,
            "dataset_name": dataset_name,
        })

    # Group models by (tso_name, dataset_name)
    dataset_keys: dict[tuple[str, str], list[dict]] = {}
    for meta in model_metas:
        key = (meta["tso_name"], meta["dataset_name"])
        dataset_keys.setdefault(key, []).append(meta)

    # ── Process each dataset group ────────────────────────────────────────────
    for (tso_name, dataset_name), group_metas in dataset_keys.items():
        model_dataset = get_dataset(dataset_root_dir, tso_name, dataset_name)
        supplied_dataset_path = dataset_root_dir / f"{dataset_name}_{tso_name}.parquet"

        # Use the first model's config as the representative for benchmark parameters
        representative_config = group_metas[0]["run_info"]["config"]

        # Compute benchmarks ONCE for this dataset
        logger.info(
            "Computing benchmarks for tso=%s dataset=%s",
            tso_name,
            dataset_name,
        )
        if run_predictions:            
            benchmarks_df, benchmark_metadata = run_benchmarks_from_pipeline_config(
                dataset=model_dataset,
                run_config=representative_config,
                tso_name=tso_name,
                enabled_benchmarks=enabled_benchmarks,
                parameter_yaml_file_path=benchmark_config_path,
            )

            # Save LightGBM feature importance to CSV if present in metadata
            if not enabled_benchmarks or "lightgbm" in enabled_benchmarks:
                logger.info("Exporting LightGBM feature importance to CSV")
                lightgbm_keys = [key for key in benchmark_metadata.keys() if "lightgbm" in key]
                for key in lightgbm_keys:
                    benchmark_metadata = save_lightgbm_feature_importance(
                        benchmark_metadata, output_dir, key
                    )
            else:
                logger.info("Skipping LightGBM feature importance export (not in enabled benchmarks)")

            # Save benchmarks
            benchmarks_path = output_dir / f"benchmarks_{tso_name}_{dataset_name}.csv"
            benchmarks_df.assign(tso=tso_name, dataset=dataset_name).to_csv(benchmarks_path)
            logger.info("Saved benchmarks to %s", benchmarks_path)
            
            # Save benchmark metadata to JSON file
            benchmark_metadata_path = output_dir / f"benchmark_metadata_{tso_name}_{dataset_name}.json"
            with open(benchmark_metadata_path, "w") as f:
                json.dump(benchmark_metadata, f, indent=2, default=str)
            logger.info("Saved benchmark metadata to %s", benchmark_metadata_path)

        # ── Now iterate over each neural model sharing this dataset ───────────
        last_predictions_list, best_predictions_list = [], []
        for meta in group_metas:
            model_dir = meta["model_dir"]
            run_config = meta["run_info"]["config"]
            run_id = meta["run_info"]["run_id"]

            logger.info("Processing model in %s", model_dir)

            model_alias = _require_config(run_config, "model_alias")

            # ── Dataset compatibility check ───────────────────────────────────
            dataset_compat_status, _new_data_start = check_dataset_compatibility(
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

            if run_predictions:
                predictions = run_predictions_for_model(
                    model_dir=model_dir,
                    model_dataset=model_dataset,
                    run_config=run_config,
                    run_id=run_id,
                    tso_name=tso_name,
                    dataset_compat_status=dataset_compat_status,
                    benchmarks_df=benchmarks_df,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    output_dir=output_dir,
                    add_calendar_features=add_calendar_features,
                    holidays_path=holidays_path,
                    best_checkpoint=False,
                    start_date=start_date,
                )
                last_predictions_list.append(predictions)
                if test_best_checkpoint:
                    best_predictions = run_predictions_for_model(
                        model_dir=model_dir,
                        model_dataset=model_dataset,
                        run_config=run_config,
                        run_id=run_id,
                        tso_name=tso_name,
                        dataset_compat_status=dataset_compat_status,
                        benchmarks_df=benchmarks_df,
                        wandb_project=wandb_project,
                        wandb_entity=wandb_entity,
                        output_dir=output_dir,
                        add_calendar_features=add_calendar_features,
                        holidays_path=holidays_path,
                        best_checkpoint=True,
                        start_date=start_date,
                    )
                    best_predictions_list.append(best_predictions)

        if last_predictions_list:
            merged_last_predictions = merge_tso_data(last_predictions_list, model_dataset)
            save_predictions_locally(
                output_dir,
                merged_last_predictions,
                tso_name,
                best_checkpoint=False,
            )
        
        if best_predictions_list:
            merged_best_predictions = merge_tso_data(best_predictions_list, model_dataset)
            save_predictions_locally(
                output_dir,
                merged_best_predictions,
                tso_name,
                best_checkpoint=True,
            )

        # ── Explainability (Integrated Gradients) + TFT interpretability ──
        if run_explainability:
            run_explainability_for_model(
                model_dir=model_dir,
                model_dataset=model_dataset,
                run_config=run_config,
                run_id=run_id,
                tso_name=tso_name,
                dataset_compat_status=dataset_compat_status,
                wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    output_dir=output_dir,
                    add_calendar_features=add_calendar_features,
                    holidays_path=holidays_path,
                    explain_step_hours=explain_step_hours,
                    start_date=start_date,
                )


def prepare_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model-path", type=str, help="Model directory or path to a zstd archive containing the whole neuralforecast saved data.")
    p.add_argument("--dataset-root-dir", type=str, help="Path to the main directory where datasets are kept.")
    p.add_argument("--test-best-checkpoint", action="store_true", help="Whether to predict and evaluate the best validation checkpoint (in addition to the last checkpoint).")
    p.add_argument("--wandb-project", type=str, help="Weights and Biases project name for logging.")
    p.add_argument("--wandb-entity", type=str, default=None, help="Weights and Biases entity name for logging.")
    p.add_argument("--no-calendar", action="store_true", help="Skip adding calendar features.")
    p.add_argument("--holidays-path", default=None, help="Override for holidays CSV path.")
    p.add_argument("--n-threads", type=int, default=-1, help="Number of torch dataloader threads to use.")
    p.add_argument("--output-dir", type=str, help="Output directory path.")

    p.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help=(
            "Which benchmarks to compute.  Accepted values: "
            "naive_seasonal, ridge_regression, lightgbm, "
            "auto_arima, seasonal_regression.  Defaults to all."
        ),
    )
    p.add_argument("--benchmark-config-path", type=str, default=None, help="Path to yaml file with benchmark configuration (lags, hyperparameters, etc.).")
    p.add_argument(
        "--explain",
        action="store_true",
        help="Run Integrated Gradients explainability after predictions.",
    )
    p.add_argument(
        "--explain-step-hours",
        type=int,
        default=24,
        help="Stride (in hours) between IG explanation windows (default: 24 = daily).",
    )
    p.add_argument("--skip-predictions", action="store_true", help="Skip prediction and evaluation steps, only explain. Useful when predictions are already logged to W&B.")
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help=(
            "Evaluation start date (ISO format, e.g. '2024-06-01').  "
            "Evaluation metrics are computed only from this date onwards.  "
            "For explainability, strides before this date are saved to a separate "
            "'ig_dev_explanation' artifact; stride numbering in the main artifact "
            "starts from 0 at this date.  "
            "Clipped to the earliest timestamp in the test set if earlier."
        ),
    )

    args = p.parse_args()

    return args


if __name__ == "__main__":
    args = prepare_args()
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
        test_best_checkpoint=args.test_best_checkpoint,
        run_explainability=args.explain,
        explain_step_hours=args.explain_step_hours,
        run_predictions=not args.skip_predictions,
        start_date=pd.Timestamp(args.start_date) if args.start_date else None,
    )



