"""
Prediction pipeline for per-(TSO, direction) single-window models.

Loads models trained with ``train_pipeline_per_tso_direction.py``,
re-runs / extends predictions, and evaluates them locally.

No Weights & Biases integration – all outputs are written to disk.

Directory structure expected under ``--model-path``
-----------------------------------------------------
<feature_set>/
  <tso>/
    <direction>/
      <timestamp>/
        <model_alias>/
          run_meta.json
          nf_model/
            alias_to_model.pkl
            best_valid_checkpoint.ckpt   (optional)
            <model_alias>_0.ckpt         (last checkpoint)
            ...

Outputs
-------
For each model the pipeline writes into
``<output_dir>/<feature_set>/<tso>/<direction>/<timestamp>/<model_alias>/``:
  predictions_last.parquet
  predictions_best.parquet          (only with --test-best-checkpoint)
  evaluation_last.csv
  evaluation_best.csv               (only with --test-best-checkpoint)

Usage examples
--------------
# Predict & evaluate all models under a feature-set directory
python -m training.prediction_pipeline_per_direction \\
    --model-path outputs_single_model_per_direction_new/basic_.../ \\
    --output-dir outputs_single_model_per_direction_new/eval \\
    --start-date 2025-01-01

# Also evaluate the best-validation checkpoint
python -m training.prediction_pipeline_per_direction \\
    --model-path outputs_single_model_per_direction_new/ \\
    --output-dir results/per_direction \\
    --test-best-checkpoint \\
    --start-date 2025-01-01
"""
from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from neuralforecast import NeuralForecast

from training.runner import CHECKPOINT_BEST_NAME
from training.predict import (
    predict_with_shift_correction,
    evaluate_models,
    prepare_predictions_df,
)
from training.data_prep import (
    load_dataset,
    to_nixtla_format,
    prepare_shifted_dataset,
)
from training.train_pipeline import set_n_threads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Model loading helpers ─────────────────────────────────────────────────────

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


def get_last_checkpoint_path(nf_dir: Path) -> Path:
    """Return the single 'last' checkpoint (everything except best_valid_checkpoint.ckpt)."""
    checkpoints = list(nf_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {nf_dir}")

    best_checkpoint_path = nf_dir / CHECKPOINT_BEST_NAME
    last_checkpoints = set(checkpoints) - {best_checkpoint_path}
    if len(last_checkpoints) != 1:
        raise ValueError(
            f"Expected exactly one last checkpoint in {nf_dir}, "
            f"found {len(last_checkpoints)}: {last_checkpoints}"
        )
    return last_checkpoints.pop()


def load_model(model_dir: Path, checkpoint_best: bool) -> NeuralForecast:
    """Load a NeuralForecast model, optionally using the best-validation checkpoint."""
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
            if best_valid_checkpoint_tmp_path.exists():
                best_valid_checkpoint_tmp_path.rename(best_valid_checkpoint)
            if not best_valid_checkpoint.exists():
                raise FileNotFoundError(
                    f"Best valid checkpoint not found in {model_dir}"
                )
            last_checkpoint_path.rename(last_checkpoint_tmp_path)
            best_valid_checkpoint.rename(last_checkpoint_path)
            nf_model = NeuralForecast.load(str(model_dir.resolve()))
        _dedupe_model_callbacks(nf_model)
        return nf_model
    except Exception as e:
        raise RuntimeError(
            f"Error loading model from {model_dir} with checkpoint_best={checkpoint_best}"
        ) from e
    finally:
        if best_valid_checkpoint_tmp_path.exists():
            best_valid_checkpoint_tmp_path.rename(best_valid_checkpoint)
        elif last_checkpoint_tmp_path.exists():
            last_checkpoint_path.rename(best_valid_checkpoint)
            last_checkpoint_tmp_path.rename(last_checkpoint_path)


# ── Model discovery ───────────────────────────────────────────────────────────

def discover_models(root_path: Path) -> list[Path]:
    """
    Recursively find all ``run_meta.json`` files under *root_path*.

    Returns the parent directory of each ``run_meta.json`` (i.e. the
    ``<model_alias>/`` directory that also contains ``nf_model/``).
    """
    meta_files = sorted(root_path.rglob("run_meta.json"))
    if not meta_files:
        logger.warning("No run_meta.json files found under %s", root_path)
    return [p.parent for p in meta_files]


def load_run_meta(model_dir: Path) -> dict:
    """Load and return the ``run_meta.json`` for a model directory."""
    meta_path = model_dir / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_meta.json not found in {model_dir}")
    with open(meta_path) as f:
        return json.load(f)


# ── Dataset loading ───────────────────────────────────────────────────────────

def _resolve_dataset_path(
    run_meta: dict,
    dataset_root_dir: Path | None,
    workspace_root: Path,
) -> Path:
    """
    Resolve the dataset parquet path.

    Priority:
    1. ``dataset_root_dir`` supplied on CLI, combined with TSO & dataset-contents name.
    2. ``run_meta["dataset_path"]`` resolved relative to ``workspace_root``.
    """
    tso_name = run_meta["tso"].replace(" ", "_")
    dataset_contents = run_meta.get("dataset_contents", "")
    # Build the canonical dataset name (same convention as train_pipeline_per_tso_direction)
    dataset_name = dataset_contents.replace(", ", "_")

    if dataset_root_dir is not None:
        candidate = dataset_root_dir / f"{dataset_name}_{tso_name}.parquet"
        if candidate.exists():
            return candidate
        logger.warning(
            "Dataset not found at %s, falling back to run_meta path.", candidate
        )

    stored_path = Path(run_meta["dataset_path"])
    if stored_path.is_absolute() and stored_path.exists():
        return stored_path

    # Try relative to workspace root
    resolved = workspace_root / stored_path
    if resolved.exists():
        return resolved

    raise FileNotFoundError(
        f"Dataset not found. Tried:\n"
        f"  {dataset_root_dir / f'{dataset_name}_{tso_name}.parquet' if dataset_root_dir else 'N/A'}\n"
        f"  {stored_path}\n"
        f"  {resolved}"
    )


def load_direction_dataset(
    dataset_path: Path,
    direction: str,
) -> pd.DataFrame:
    """Load a dataset parquet and return the Nixtla-format slice for *direction*."""
    df_raw, _ = load_dataset(dataset_path)
    nixtla_df = to_nixtla_format(df_raw, direction=direction)
    return nixtla_df


# ── Dataset compatibility check ───────────────────────────────────────────────

def check_dataset_compatibility(
    run_meta: dict,
    supplied_dataset_path: Path,
) -> tuple[Literal["compatible", "extended", "incompatible"], pd.Timestamp | None]:
    """
    Compare the supplied dataset against the one used during training.

    Uses the ``dataset_metadata`` block embedded in ``run_meta.json`` as the
    training-time reference (no need to re-load the original parquet).

    Returns
    -------
    ("compatible", None)
        Dataset matches training data (same columns/features, not newer).
    ("extended", first_new_ts)
        Dataset extends further in time; predictions are needed from
        *first_new_ts* onwards.
    ("incompatible", None)
        Column or feature-set mismatch – a new training run is recommended.
    """
    train_meta = run_meta.get("dataset_metadata", {})
    if not train_meta:
        logger.warning("No 'dataset_metadata' in run_meta.json – skipping compatibility check.")
        return "compatible", None

    # Load companion JSON for the supplied dataset
    supplied_meta_path = supplied_dataset_path.with_suffix(".json")
    if not supplied_meta_path.exists():
        logger.warning(
            "No sidecar metadata JSON found at %s – skipping compatibility check.",
            supplied_meta_path,
        )
        return "compatible", None

    with open(supplied_meta_path) as f:
        supplied_meta = json.load(f)

    # ── Column check ─────────────────────────────────────────────────────────
    train_cols = set(train_meta.get("columns", []))
    supplied_cols = set(supplied_meta.get("columns", []))
    if train_cols and supplied_cols and train_cols != supplied_cols:
        missing = train_cols - supplied_cols
        extra = supplied_cols - train_cols
        logger.warning(
            "Dataset column mismatch (missing=%s, extra=%s). "
            "A new training run is recommended.",
            missing,
            extra,
        )
        return "incompatible", None

    # ── Feature-set check ─────────────────────────────────────────────────────
    train_fs = set(train_meta.get("feature_sets", []))
    supplied_fs = set(supplied_meta.get("feature_sets", []))
    if train_fs and supplied_fs and train_fs != supplied_fs:
        logger.warning(
            "Dataset feature-set mismatch (%s vs %s). "
            "A new training run is recommended.",
            train_fs,
            supplied_fs,
        )
        return "incompatible", None

    # ── Timeline check ────────────────────────────────────────────────────────
    train_range = train_meta.get("date_range", {})
    supplied_range = supplied_meta.get("date_range", {})

    if not train_range or not supplied_range:
        logger.warning("Missing date_range in one of the metadata files – skipping timeline check.")
        return "compatible", None

    train_start = pd.Timestamp(train_range["start"])
    train_end = pd.Timestamp(train_range["end"])
    supplied_start = pd.Timestamp(supplied_range["start"])
    supplied_end = pd.Timestamp(supplied_range["end"])

    if supplied_start > train_start:
        logger.warning(
            "Supplied dataset starts at %s, but training data started at %s. "
            "The supplied dataset must cover the full training history. "
            "A new training run is recommended.",
            supplied_start,
            train_start,
        )
        return "incompatible", None

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


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_checkpoint(
    nf_model: NeuralForecast,
    dataset: pd.DataFrame,
    run_meta: dict,
    add_calendar_features: bool,
    holidays_path: str | None,
    pred_start_override: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate predictions for a single checkpoint.

    Parameters
    ----------
    pred_start_override
        When set, overrides ``run_meta["test_start"]`` as the first timestamp to
        predict.  Used when extending predictions for an updated dataset.
    """
    shift_hours = int(run_meta["shift_hours"])
    test_start = pd.Timestamp(pred_start_override or run_meta["test_start"])
    forecast_horizon = int(run_meta["forecast_horizon"])
    tso = run_meta["tso"]

    shifted_dataset, future_cov_cols, _ = prepare_shifted_dataset(
        df=dataset,
        shift_hours=shift_hours,
        tso=tso,
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
        static_df=None,  # per-direction models have no static exogenous
        tso=tso,
    )


# ── Evaluation helpers ────────────────────────────────────────────────────────

def classification_metrics_df(
    prepared_preds_df: pd.DataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    beta: float = 1.0,
) -> pd.DataFrame:
    exclude = {id_col, time_col, target_col, "horizon"}
    models = [c for c in prepared_preds_df.columns if c not in exclude]
    if not models:
        raise ValueError("No model columns found in prepared_preds_df")

    results = []
    for direction in prepared_preds_df[id_col].unique():
        sub = prepared_preds_df[prepared_preds_df[id_col] == direction]
        y_true = (sub[target_col] > 0).astype(int)
        row = {"unique_id": direction, "metric": f"fbeta_{beta}"}
        for model in models:
            y_pred = (sub[model] > 0).astype(int)
            row[model] = fbeta_score(
                y_true, y_pred, average="binary", zero_division=0, beta=beta
            )
        results.append(row)
    return pd.DataFrame(results)


def evaluation_pipeline(
    predictions: pd.DataFrame,
    model_dataset: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    beta: float = 2.0,
) -> pd.DataFrame:
    """Run all evaluation metrics on *predictions*.

    Parameters
    ----------
    start_date
        Only evaluate from this timestamp onwards (useful for a dedicated test
        window that excludes the warm-up / dev period).
    """
    if start_date is not None:
        effective_start = max(start_date, predictions["ds"].min())
        predictions = predictions[predictions["ds"] >= effective_start]

    final_df = prepare_predictions_df(predictions, model_dataset)

    overall = evaluate_models(final_df)
    overall = overall[overall["metric"].isin(["mae", "rmse"])].copy()

    conditional = evaluate_models(final_df[final_df["y"] > 0])
    conditional = conditional[conditional["metric"].isin(["mae", "rmse"])].copy()
    conditional["metric"] = conditional["metric"].map(
        {"mae": "mae_conditional", "rmse": "rmse_conditional"}
    )

    fbeta_df = classification_metrics_df(final_df, beta=beta)
    fbeta_df = fbeta_df[fbeta_df["metric"] == f"fbeta_{beta}"].copy()

    return pd.concat([overall, conditional, fbeta_df], ignore_index=True)


def save_predictions_locally(
    output_dir: Path,
    predictions: pd.DataFrame,
    checkpoint_label: str,
) -> None:
    """Persist predictions to ``<output_dir>/predictions_<checkpoint_label>.parquet``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"predictions_{checkpoint_label}.parquet"
    predictions.to_parquet(path, index=False)
    logger.info("Saved predictions → %s", path)


def load_predictions_locally(
    output_dir: Path,
    checkpoint_label: str,
) -> pd.DataFrame:
    """Load previously saved predictions if they exist."""
    path = output_dir / f"predictions_{checkpoint_label}.parquet"
    if path.exists():
        logger.info("Loading cached predictions from %s", path)
        return pd.read_parquet(path)
    return pd.DataFrame()


def save_evaluation_locally(
    output_dir: Path,
    evaluation_df: pd.DataFrame,
    checkpoint_label: str,
) -> None:
    """Save evaluation CSV to ``<output_dir>/evaluation_<checkpoint_label>.csv``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"evaluation_{checkpoint_label}.csv"
    evaluation_df.to_csv(path)
    logger.info("Saved evaluation → %s", path)


# ── Per-model prediction + evaluation ────────────────────────────────────────

def run_predictions_for_model(
    *,
    model_dir: Path,
    nf_dir: Path,
    run_meta: dict,
    model_dataset: pd.DataFrame,
    dataset_compat_status: Literal["compatible", "extended", "incompatible"],
    output_dir: Path,
    add_calendar_features: bool,
    holidays_path: str | None,
    best_checkpoint: bool,
    start_date: pd.Timestamp | None,
) -> None:
    """Fetch or compute predictions for one checkpoint type, evaluate, and persist."""
    checkpoint_label = "best" if best_checkpoint else "last"
    model_alias = run_meta["model_alias"]
    tso = run_meta["tso"]
    direction = run_meta["direction"]

    logger.info(
        "[%s / %s / %s] Running predictions (checkpoint=%s)",
        tso, direction, model_alias, checkpoint_label,
    )

    # Try to load cached predictions
    predictions = load_predictions_locally(output_dir, checkpoint_label)

    if predictions.empty:
        # No cached predictions – run from scratch
        nf = load_model(nf_dir, checkpoint_best=best_checkpoint)
        predictions = predict_checkpoint(
            nf_model=nf,
            dataset=model_dataset,
            run_meta=run_meta,
            add_calendar_features=add_calendar_features,
            holidays_path=holidays_path,
        )
        if predictions.empty:
            logger.warning(
                "[%s / %s / %s] No predictions generated – skipping.", tso, direction, model_alias
            )
            return
        # Attach actuals
        predictions = predictions.merge(
            model_dataset[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="left",
        )
        save_predictions_locally(output_dir, predictions, checkpoint_label)

    elif dataset_compat_status == "extended":
        # Cached predictions exist but dataset has grown – extend the tail
        last_pred_ts = predictions["ds"].max()
        if model_dataset["ds"].max() > last_pred_ts:
            tail_pred_start = last_pred_ts + pd.Timedelta(hours=1)
            logger.info(
                "[%s / %s / %s] Extending predictions from %s",
                tso, direction, model_alias, tail_pred_start,
            )
            nf = load_model(nf_dir, checkpoint_best=best_checkpoint)
            tail_preds = predict_checkpoint(
                nf_model=nf,
                dataset=model_dataset,
                run_meta=run_meta,
                add_calendar_features=add_calendar_features,
                holidays_path=holidays_path,
                pred_start_override=tail_pred_start,
            )
            if not tail_preds.empty:
                tail_preds = tail_preds.merge(
                    model_dataset[["unique_id", "ds", "y"]],
                    on=["unique_id", "ds"],
                    how="left",
                )
                # Normalise both parts to the same processed-column format
                predictions = prepare_predictions_df(predictions, model_dataset)
                tail_preds = prepare_predictions_df(tail_preds, model_dataset)
                predictions = (
                    pd.concat([predictions, tail_preds], ignore_index=True)
                    .drop_duplicates(subset=["unique_id", "ds"], keep="last")
                    .sort_values(["unique_id", "ds"])
                    .reset_index(drop=True)
                )
                save_predictions_locally(output_dir, predictions, checkpoint_label)
        else:
            logger.info(
                "[%s / %s / %s] Cached predictions are already up-to-date.",
                tso, direction, model_alias,
            )

    # Evaluate
    evaluation_df = evaluation_pipeline(
        predictions=predictions,
        model_dataset=model_dataset,
        start_date=start_date,
        beta=2.0,
    )
    save_evaluation_locally(output_dir, evaluation_df, checkpoint_label)


# ── Main orchestration ────────────────────────────────────────────────────────

def main(
    root_model_path: Path,
    dataset_root_dir: Path | None,
    output_dir: Path,
    add_calendar_features: bool,
    holidays_path: str | None,
    test_best_checkpoint: bool,
    start_date: pd.Timestamp | None,
) -> None:
    # Set up file logging
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "prediction_pipeline_per_direction.log"
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(file_handler)
    logger.info("=" * 80)
    logger.info("Starting per-direction prediction pipeline")
    logger.info("model path  : %s", root_model_path)
    logger.info("dataset dir : %s", dataset_root_dir)
    logger.info("output dir  : %s", output_dir)
    logger.info("start date  : %s", start_date)
    logger.info("best ckpt   : %s", test_best_checkpoint)
    logger.info("=" * 80)

    workspace_root = Path(__file__).resolve().parent.parent

    model_dirs = discover_models(root_model_path)
    if not model_dirs:
        logger.warning("No models found under %s – nothing to do.", root_model_path)
        return

    logger.info("Discovered %d model(s).", len(model_dirs))

    for model_dir in model_dirs:
        try:
            run_meta = load_run_meta(model_dir)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", model_dir, exc)
            continue

        tso = run_meta["tso"]
        direction = run_meta["direction"]
        model_alias = run_meta["model_alias"]
        nf_dir = model_dir / "nf_model"

        if not nf_dir.exists():
            logger.warning(
                "Skipping %s: nf_model/ directory not found.", model_dir
            )
            continue

        logger.info(
            "Processing model: tso=%s  direction=%s  alias=%s",
            tso, direction, model_alias,
        )

        # ── Resolve dataset ───────────────────────────────────────────────────
        try:
            dataset_path = _resolve_dataset_path(run_meta, dataset_root_dir, workspace_root)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", model_dir, exc)
            continue

        logger.info("Using dataset: %s", dataset_path)

        # ── Compatibility check ───────────────────────────────────────────────
        compat_status, _first_new_ts = check_dataset_compatibility(run_meta, dataset_path)
        if compat_status == "incompatible":
            logger.warning(
                "Skipping %s / %s / %s: supplied dataset is incompatible with "
                "the training dataset.  A new training run is recommended.",
                tso, direction, model_alias,
            )
            continue

        # ── Load direction-specific dataset ───────────────────────────────────
        try:
            model_dataset = load_direction_dataset(dataset_path, direction)
        except Exception as exc:
            logger.warning("Skipping %s: failed to load dataset: %s", model_dir, exc)
            continue

        # ── Determine per-model output directory ─────────────────────────────
        # Mirror the training output structure so results live alongside the model.
        # Path components: feature_set / tso / direction / timestamp / model_alias
        # We derive these from the path relative to root_model_path when possible,
        # otherwise fall back to building from run_meta fields.
        try:
            rel = model_dir.relative_to(root_model_path)
            model_output_dir = output_dir / rel
        except ValueError:
            # model_dir is not under root_model_path (e.g. absolute path mismatch)
            dataset_contents_slug = run_meta.get("dataset_contents", "default").replace(", ", "_")
            tso_slug = tso.replace(" ", "_").lower()
            date_time = run_meta.get("date_time", "unknown")
            model_output_dir = (
                output_dir / dataset_contents_slug / tso_slug / direction / date_time / model_alias
            )

        # ── Predict (last checkpoint) ─────────────────────────────────────────
        try:
            run_predictions_for_model(
                model_dir=model_dir,
                nf_dir=nf_dir,
                run_meta=run_meta,
                model_dataset=model_dataset,
                dataset_compat_status=compat_status,
                output_dir=model_output_dir,
                add_calendar_features=add_calendar_features,
                holidays_path=holidays_path,
                best_checkpoint=False,
                start_date=start_date,
            )
        except Exception:
            logger.exception(
                "Error running last-checkpoint predictions for %s / %s / %s",
                tso, direction, model_alias,
            )

        # ── Predict (best-validation checkpoint, optional) ────────────────────
        if test_best_checkpoint:
            best_ckpt_path = nf_dir / CHECKPOINT_BEST_NAME
            if not best_ckpt_path.exists():
                logger.warning(
                    "[%s / %s / %s] Best-valid checkpoint not found – skipping.",
                    tso, direction, model_alias,
                )
            else:
                try:
                    run_predictions_for_model(
                        model_dir=model_dir,
                        nf_dir=nf_dir,
                        run_meta=run_meta,
                        model_dataset=model_dataset,
                        dataset_compat_status=compat_status,
                        output_dir=model_output_dir,
                        add_calendar_features=add_calendar_features,
                        holidays_path=holidays_path,
                        best_checkpoint=True,
                        start_date=start_date,
                    )
                except Exception:
                    logger.exception(
                        "Error running best-checkpoint predictions for %s / %s / %s",
                        tso, direction, model_alias,
                    )

    logger.info("Per-direction prediction pipeline complete.")


def prepare_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prediction pipeline for per-(TSO, direction) single-window models."
    )
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help=(
            "Root directory to scan recursively for run_meta.json files.  "
            "Can be the output root (e.g. outputs_single_model_per_direction_new/) "
            "or a feature-set sub-directory."
        ),
    )
    p.add_argument(
        "--dataset-root-dir",
        type=str,
        default=None,
        help=(
            "Directory that contains dataset parquet files named "
            "<dataset_contents>_<tso>.parquet.  When supplied, takes priority "
            "over the dataset_path stored in run_meta.json."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory for evaluation outputs.",
    )
    p.add_argument(
        "--test-best-checkpoint",
        action="store_true",
        help=(
            "Also evaluate the best-validation checkpoint "
            "(in addition to the last checkpoint)."
        ),
    )
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help=(
            "Evaluation start date (ISO format, e.g. '2025-01-01').  "
            "Metrics are computed only from this date onwards.  "
            "Clipped to the earliest timestamp in the test set if earlier."
        ),
    )
    p.add_argument(
        "--no-calendar",
        action="store_true",
        help="Skip adding calendar features (hour, day, month, is_weekend, ...).",
    )
    p.add_argument(
        "--holidays-path",
        default=None,
        help="Override for the holidays CSV path used for calendar features.",
    )
    p.add_argument(
        "--n-threads",
        type=int,
        default=-1,
        help="Number of PyTorch dataloader threads (default: -1 = auto).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = prepare_args()
    set_n_threads(args.n_threads)
    main(
        root_model_path=Path(args.model_path),
        dataset_root_dir=Path(args.dataset_root_dir) if args.dataset_root_dir else None,
        output_dir=Path(args.output_dir),
        add_calendar_features=not args.no_calendar,
        holidays_path=args.holidays_path,
        test_best_checkpoint=args.test_best_checkpoint,
        start_date=pd.Timestamp(args.start_date) if args.start_date else None,
    )
