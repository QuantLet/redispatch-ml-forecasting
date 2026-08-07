#!/usr/bin/env python
"""
Prediction pipeline for circular-shift augmented models.

Loads saved NeuralForecast checkpoints produced by ``circular_shift_training.py``
and generates test-set predictions **without** re-applying the circular
covariate shift.  Predictions use real (unshifted) covariate values so results
are directly comparable with non-augmented baselines.

Provide the same dataset / date / forecast-config arguments as you used during
training; the rolling-window boundaries are reconstructed from them
identically.

Model discovery
---------------
``--model-dir`` must point to the directory that contains ``window_*``
subdirectories, i.e. the directory that was ``<output-dir>/<tso>/<dataset>/``
during training.  Both checkpoint layouts produced by the training script are
supported:

1. **Uncompressed** (``--persist-checkpoints``)::

       <model-dir>/window_N/<alias>/<date-time>/nf_model/

2. **Compressed** (``--checkpoint-compression <level>``)::

       <model-dir>/window_N/<alias>/<date-time>.tar.zst

For single-window models (no ``--rolling-window``), point ``--model-dir``
directly at the directory that contains the alias subdirectory and supply
``--test-start`` (and optionally ``--test-end``).

Outputs
-------
* Per-window parquet files in ``--output-dir``
* A combined parquet (all windows concatenated, duplicates removed) per alias
* A CSV evaluation file per alias (MAE, RMSE, R²)

W&B
---
When ``--wandb-project`` is given, the pipeline looks up and **resumes** the
original training run for each checkpoint (required so that the
``WandbLogger`` embedded in the saved checkpoint can reconnect).  No new
prediction artifacts are uploaded to W&B.

Usage examples
--------------
# Rolling window – local output only
python -m training.circular_shift_prediction \\
    --dataset-path data/.../TenneT_DE.parquet \\
    --model-dir outputs/TenneT_DE/basic_day_ahead_price_wind_pv_.../ \\
    --shift-hours 6 \\
    --rolling-window \\
    --n-train-months 37 --n-valid-months 2 --n-test-months 1 \\
    --output-dir preds/cs_run_2026

# Same run but also log to W&B
python -m training.circular_shift_prediction \\
    --dataset-path ... \\
    --model-dir ... \\
    --shift-hours 6 --rolling-window \\
    --n-train-months 37 --n-valid-months 2 --n-test-months 1 \\
    --wandb-project redispatch-forecasting \\
    --output-dir preds/cs_run_2026

# Use best-validation checkpoint instead of last checkpoint
python -m training.circular_shift_prediction \\
    ... \\
    --best-checkpoint

# Resume from window 5 (skip already-done windows)
python -m training.circular_shift_prediction \\
    ... \\
    --start-window 5
"""

import argparse
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Iterator

import pandas as pd
import zstandard as zstd

from training.predict import (
    predict_with_shift_correction,
    prepare_predictions_df,
    evaluate_models,
)
from training.data_prep import (
    load_dataset,
    to_nixtla_format,
    prepare_shifted_dataset,
    build_static_df,
)
from training.prediction_pipeline_rolling_window import (
    load_model,
    move_files_from_last_nonempty_dir,
    _save_predictions_locally,
)
from training.runner import _compute_rolling_windows
from training.train_pipeline import (
    _flat_defaults_from_yaml,
    load_yaml_config,
    set_n_threads,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Model discovery ───────────────────────────────────────────────────────────

def _extract_archive(archive_path: Path, temp_dir: Path) -> None:
    """Extract a .tar.zst archive into *temp_dir* and flatten the structure."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    with open(archive_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|*") as tar:
                tar.extractall(path=temp_dir)
    # Flatten: move files from deepest non-empty subdir up to temp_dir root
    move_files_from_last_nonempty_dir(temp_dir, temp_dir)


def _discover_uncompressed(
    model_dir: Path,
    start_window: int,
) -> Iterator[tuple[int, Path, str, str]]:
    """
    Yield ``(window_index, nf_model_dir, alias, date_time)`` for uncompressed
    checkpoints matching the layout::

        model_dir/window_N/alias/date_time/nf_model/
    """
    for window_dir in sorted(
        model_dir.glob("window_*"),
        key=lambda d: int(d.name.split("_")[1]),
    ):
        wi = int(window_dir.name.split("_")[1])
        if wi < start_window:
            continue
        for alias_dir in sorted(window_dir.iterdir()):
            if not alias_dir.is_dir():
                continue
            alias = alias_dir.name
            for dt_dir in sorted(alias_dir.iterdir()):
                if not dt_dir.is_dir():
                    continue
                nf_model_dir = dt_dir / "nf_model"
                if nf_model_dir.is_dir():
                    yield wi, nf_model_dir, alias, dt_dir.name
                    break  # take first (most recent) date_time dir


def _discover_compressed(
    model_dir: Path,
    start_window: int,
    temp_dirs: list[Path],
) -> Iterator[tuple[int, Path, str, str]]:
    """
    Yield ``(window_index, extracted_dir, alias, date_time)`` for compressed
    checkpoints matching the layout::

        model_dir/window_N/alias/date_time.tar.zst

    Archives are extracted to a sibling temporary directory registered in
    *temp_dirs* for cleanup after the pipeline completes.
    """
    for window_dir in sorted(
        model_dir.glob("window_*"),
        key=lambda d: int(d.name.split("_")[1]),
    ):
        wi = int(window_dir.name.split("_")[1])
        if wi < start_window:
            continue
        for alias_dir in sorted(window_dir.iterdir()):
            if not alias_dir.is_dir():
                continue
            alias = alias_dir.name
            archives = sorted(alias_dir.glob("*.tar.zst"))
            if not archives:
                continue
            archive = archives[-1]  # most recent
            date_time = archive.stem.replace(".tar", "")  # e.g. "2026-03-05_10-00-00"
            temp_dir = alias_dir.with_name(
                f"model_extraction_{alias_dir.name}_w{wi}"
            )
            temp_dirs.append(temp_dir)
            if not temp_dir.exists():
                logger.info(
                    "Extracting %s → %s", archive.name, temp_dir
                )
                _extract_archive(archive, temp_dir)
            else:
                logger.info(
                    "Using existing extraction dir %s", temp_dir
                )
            yield wi, temp_dir, alias, date_time


def _discover_models(
    model_dir: Path,
    start_window: int,
    temp_dirs: list[Path],
) -> Iterator[tuple[int, Path, str, str]]:
    """
    Yield ``(window_index, nf_model_dir, alias, date_time)`` from both
    uncompressed and compressed checkpoint layouts.  When both exist for the
    same (window, alias), the uncompressed version takes precedence.
    """
    seen: set[tuple[int, str]] = set()

    for wi, nf_dir, alias, dt in _discover_uncompressed(model_dir, start_window):
        seen.add((wi, alias))
        yield wi, nf_dir, alias, dt

    for wi, nf_dir, alias, dt in _discover_compressed(
        model_dir, start_window, temp_dirs
    ):
        if (wi, alias) not in seen:
            seen.add((wi, alias))
            yield wi, nf_dir, alias, dt


# ── Single-model prediction ───────────────────────────────────────────────────

def _predict_window(
    wi: int,
    nf_model_dir: Path,
    alias: str,
    date_time: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    pred_df: pd.DataFrame,
    nixtla_df: pd.DataFrame,
    future_cov_cols: list[str],
    shift_hours: int,
    forecast_horizon: int,
    tso: str,
    holidays_path: str | None,
    best_checkpoint: bool,
    output_dir: Path | None,
) -> pd.DataFrame:
    """
    Load one checkpoint, run strided prediction, post-process and save locally.
    Returns the prepared predictions DataFrame (may be empty).

    The caller is responsible for establishing the correct W&B context
    (resuming the original training run) **before** calling this function so
    that the WandbLogger embedded in the loaded checkpoint can reconnect.
    """
    logger.info(
        "Window %d / %s  [%s – %s]: loading model from %s",
        wi, alias, test_start.date(), test_end.date(), nf_model_dir,
    )
    nf = load_model(nf_model_dir, checkpoint_best=best_checkpoint)
    static_df = build_static_df()

    pred_end = test_end - pd.Timedelta(hours=1)
    raw_preds = predict_with_shift_correction(
        nf=nf,
        df_shifted=pred_df,
        df_unshifted=nixtla_df,
        static_df=static_df,
        pred_start=test_start,
        pred_end=pred_end,
        future_cov_cols=future_cov_cols,
        shift_hours=shift_hours,
        forecast_horizon=forecast_horizon,
        tso=tso,
        holidays_path=holidays_path,
    )

    if raw_preds.empty:
        logger.warning("Window %d / %s: no predictions generated.", wi, alias)
        return pd.DataFrame()

    preds = prepare_predictions_df(raw_preds, nixtla_df)

    # ── Local save ────────────────────────────────────────────────────────────
    if output_dir is not None:
        preds_with_y = preds.copy()
        if "y" not in preds_with_y.columns:
            preds_with_y = preds_with_y.merge(
                nixtla_df[["ds", "unique_id", "y"]],
                on=["ds", "unique_id"],
                how="left",
            )
        _save_predictions_locally(
            output_dir,
            preds_with_y,
            model_name=alias,
            tso_name=tso,
            timestamp=date_time,
            window_index=wi,
            best_checkpoint=best_checkpoint,
        )

    return preds


# ── W&B run identification ────────────────────────────────────────────────────

def _find_wandb_run_id(
    wandb_entity: str | None,
    wandb_project: str,
    alias: str,
    date_time: str,
    window_index: int | None,
) -> str | None:
    """
    Query the W&B API for the original training run that produced this
    checkpoint and return its run ID, or None if not found.

    The lookup uses the fields stored by ``runner.make_wandb_config``:
    ``config.date_time``, ``config.model_alias``, and (for rolling windows)
    ``config.window_index``.
    """
    import wandb as _wandb

    project_path = (
        f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
    )
    filters: dict = {
        "config.date_time": date_time,
        "config.model_alias": alias,
    }
    if window_index is not None:
        filters["config.window_index"] = window_index
    try:
        runs = _wandb.Api().runs(project_path, filters=filters)
        if not runs:
            logger.warning(
                "No W&B run found for alias=%s date_time=%s window=%s in %s",
                alias, date_time, window_index, project_path,
            )
            return None
        if len(runs) > 1:
            logger.warning(
                "Multiple W&B runs matched for alias=%s date_time=%s; "
                "using the most recent.",
                alias, date_time,
            )
        return runs[0].id
    except Exception:
        logger.exception(
            "Failed to look up W&B run for alias=%s date_time=%s", alias, date_time
        )
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_args, _ = _pre.parse_known_args()

    yaml_cfg: dict[str, Any] = {}
    if _pre_args.config:
        yaml_cfg = load_yaml_config(_pre_args.config)

    p = argparse.ArgumentParser(
        description=(
            "Prediction pipeline for circular-shift augmented models.\n"
            "Loads saved checkpoints and generates predictions with real\n"
            "(unshifted) covariate values for fair baseline comparison."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Config file ───────────────────────────────────────────────────────────
    p.add_argument("--config", default=None,
                   help="YAML config file (same format as train_pipeline.py).")

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument("--dataset-path", required=True,
                   help="Path to the .parquet dataset file.")
    p.add_argument("--direction", default="both", choices=["up", "down", "both"],
                   help="Direction filter (default: both).")

    # ── Model checkpoints ─────────────────────────────────────────────────────
    p.add_argument(
        "--model-dir", required=True,
        help=(
            "Root directory containing window_* checkpoint subdirs "
            "(the output_dir/tso/dataset from training)."
        ),
    )
    p.add_argument(
        "--best-checkpoint", action="store_true",
        help="Use the best-validation checkpoint instead of the last checkpoint.",
    )
    p.add_argument(
        "--start-window", type=int, default=0,
        help="Skip windows with index < this value (0-indexed, default: 0).",
    )
    p.add_argument(
        "--date-time-filter", default=None,
        metavar="DATETIME",
        help=(
            "Only process checkpoints whose date_time directory name matches "
            "this string (e.g. 2026-03-05_10-00-00).  Useful when multiple "
            "training runs are stored under the same model-dir."
        ),
    )

    # ── Shift / calendar ──────────────────────────────────────────────────────
    p.add_argument("--shift-hours", type=int, default=6,
                   help="Target shift in hours used during training (default: 6).")
    p.add_argument("--no-calendar", action="store_true",
                   help="Skip adding calendar features (must match training).")
    p.add_argument("--holidays-path", default=None,
                   help="Override for the holidays CSV path.")
    p.add_argument("--n-threads", type=int, default=None,
                   help="Number of CPU threads (default: all available).")

    # ── Forecast params ───────────────────────────────────────────────────────
    p.add_argument("--forecast-horizon", type=int, default=24,
                   help="Forecast horizon h (default: 24).")

    # ── Rolling window (must match training) ──────────────────────────────────
    p.add_argument("--rolling-window", action="store_true",
                   help="Reconstruct rolling window boundaries (recommended).")
    p.add_argument("--n-train-months", type=int, default=37,
                   help="Training months used during training (default: 37).")
    p.add_argument("--n-valid-months", type=int, default=2,
                   help="Validation months used during training (default: 2).")
    p.add_argument("--n-test-months", type=int, default=1,
                   help="Test months used during training (default: 1).")

    # ── Single-window dates ───────────────────────────────────────────────────
    p.add_argument("--test-start", default=None,
                   help="Test start date (single-window mode only).")
    p.add_argument("--test-end", default=None,
                   help="Test end date (single-window, exclusive; default: dataset end).")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="predictions_circ_shift",
                   help="Directory for output parquet / CSV files.")
    p.add_argument(
        "--wandb-project", default=None,
        help=(
            "W&B project name.  When set, the pipeline resumes the original "
            "training run for each checkpoint (required for WandbLogger to load "
            "correctly).  No new prediction artifacts are uploaded."
        ),
    )
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity / team name (optional).")
    p.add_argument(
        "--force", action="store_true",
        help=(
            "Re-run predictions even if per-window output files already exist "
            "locally.  Existing files are overwritten.  Without this flag, "
            "already-computed windows are skipped (their existing parquet is "
            "still included in the combined output)."
        ),
    )
    p.add_argument(
        "--keep-extracted", action="store_true",
        help="Keep temporary archive-extraction directories (useful for debugging).",
    )

    if yaml_cfg:
        p.set_defaults(**_flat_defaults_from_yaml(yaml_cfg))

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_n_threads(args.n_threads)

    use_wandb = args.wandb_project is not None

    # ── 1. Load dataset & convert ─────────────────────────────────────────────
    logger.info("Loading dataset from %s", args.dataset_path)
    raw_df, metadata = load_dataset(args.dataset_path)
    tso = metadata.get("operator", "")

    logger.info("Converting to Nixtla format (direction=%s)", args.direction)
    nixtla_df = to_nixtla_format(raw_df, direction=args.direction)
    logger.info(
        "Dataset: %d rows, unique_ids=%s",
        len(nixtla_df), nixtla_df["unique_id"].unique().tolist(),
    )

    # ── 2. Prepare prediction df (target shift only, no circular covariate shift)
    logger.info("Preparing prediction df (shift_hours=%d)", args.shift_hours)
    pred_df, future_cov_cols, hist_cov_cols = prepare_shifted_dataset(
        nixtla_df,
        shift_hours=args.shift_hours,
        tso=tso,
        add_calendar=not args.no_calendar,
        holidays_path=args.holidays_path,
    )
    logger.info(
        "Prediction df: %d rows, %d future covariates, %d historical covariates",
        len(pred_df), len(future_cov_cols), len(hist_cov_cols),
    )

    # ── 3. Reconstruct rolling window boundaries (if applicable) ─────────────
    if args.rolling_window:
        windows = _compute_rolling_windows(
            data_start=nixtla_df["ds"].min(),
            data_end=nixtla_df["ds"].max(),
            n_train_months=args.n_train_months,
            n_valid_months=args.n_valid_months,
            n_test_months=args.n_test_months,
        )
        window_map = {i: wb for i, wb in enumerate(windows)}
        logger.info("Reconstructed %d rolling windows from dataset boundaries.", len(windows))
    else:
        window_map = None
        if args.test_start is None:
            raise SystemExit(
                "--test-start is required in single-window mode (--rolling-window not set)."
            )

    # ── 4. Discover models and run predictions ────────────────────────────────
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging
    log_file = output_dir / "circular_shift_prediction.log"
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to %s", log_file)

    temp_dirs: list[Path] = []
    all_preds_by_alias: dict[str, list[pd.DataFrame]] = {}

    try:
        model_iter = list(
            _discover_models(model_dir, args.start_window, temp_dirs)
        )

        if not model_iter:
            logger.warning(
                "No models discovered under %s (start_window=%d). "
                "Check --model-dir and checkpoint layout.",
                model_dir, args.start_window,
            )
            return

        logger.info("Discovered %d model checkpoint(s).", len(model_iter))

        for wi, nf_model_dir, alias, date_time in model_iter:
            # Optional filter by date_time
            if (
                args.date_time_filter is not None
                and date_time != args.date_time_filter
            ):
                logger.info(
                    "Skipping window %d / %s (date_time=%s ≠ filter=%s)",
                    wi, alias, date_time, args.date_time_filter,
                )
                continue

            # Determine the test period for this window
            if args.rolling_window:
                wb = window_map.get(wi)
                if wb is None:
                    logger.warning(
                        "Window %d not in reconstructed boundaries – "
                        "dataset or config may differ from training. Skipping.",
                        wi,
                    )
                    continue
                test_start = wb.test_start
                test_end = wb.test_end
            else:
                test_start = pd.Timestamp(args.test_start)
                test_end = (
                    nixtla_df["ds"].max() + pd.Timedelta(hours=1)
                    if args.test_end is None
                    else pd.Timestamp(args.test_end)
                )

            # ── Check if this window was already computed locally ──────────────
            suffix = "_best_checkpoint" if args.best_checkpoint else ""
            local_pred_path = (
                output_dir
                / f"predictions_{alias}_{tso}_{date_time}_window{wi}{suffix}.parquet"
            )
            if local_pred_path.exists() and not args.force:
                logger.info(
                    "Window %d / %s: predictions already exist at %s – "
                    "skipping (use --force to recompute).",
                    wi, alias, local_pred_path,
                )
                existing = pd.read_parquet(local_pred_path)
                all_preds_by_alias.setdefault(alias, []).append(existing)
                continue

            # ── Resume the original training W&B run (required for WandbLogger) ─
            # The WandbLogger baked into the checkpoint needs an active run
            # with the original run ID to reconnect without errors.
            wandb_run = None
            if use_wandb:
                import wandb as _wandb
                run_id = _find_wandb_run_id(
                    args.wandb_entity, args.wandb_project, alias, date_time,
                    wi if args.rolling_window else None,
                )
                if run_id is not None:
                    wandb_run = _wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        id=run_id,
                        resume="allow",
                        reinit="finish_previous",
                    )
                else:
                    logger.warning(
                        "Window %d / %s: could not find original W&B run; "
                        "model will be loaded without a W&B context.",
                        wi, alias,
                    )

            preds = _predict_window(
                wi=wi,
                nf_model_dir=nf_model_dir,
                alias=alias,
                date_time=date_time,
                test_start=test_start,
                test_end=test_end,
                pred_df=pred_df,
                nixtla_df=nixtla_df,
                future_cov_cols=future_cov_cols,
                shift_hours=args.shift_hours,
                forecast_horizon=args.forecast_horizon,
                tso=tso,
                holidays_path=args.holidays_path,
                best_checkpoint=args.best_checkpoint,
                output_dir=output_dir,
            )

            if wandb_run is not None:
                wandb_run.finish()

            if not preds.empty:
                all_preds_by_alias.setdefault(alias, []).append(preds)

        # ── 5. Combine windows & save per-alias summary ───────────────────────
        if not all_preds_by_alias:
            logger.warning("No predictions were generated for any window.")
            return

        for alias, preds_list in all_preds_by_alias.items():
            combined = (
                pd.concat(preds_list, ignore_index=True)
                .drop_duplicates(subset=["unique_id", "ds"], keep="first")
                .sort_values(["unique_id", "ds"])
                .reset_index(drop=True)
            )

            combined_path = (
                output_dir / f"predictions_{alias}_{tso}_all_windows.parquet"
            )
            combined.to_parquet(combined_path, index=False)
            logger.info(
                "Saved %d combined predictions (%d windows) to %s",
                len(combined), len(preds_list), combined_path,
            )

            # Evaluation summary
            eval_df = evaluate_models(combined)
            if not eval_df.empty:
                logger.info("Evaluation for %s:\n%s", alias, eval_df.to_string(index=False))
                eval_path = (
                    output_dir / f"evaluation_{alias}_{tso}_all_windows.csv"
                )
                eval_df.to_csv(eval_path, index=False)
                logger.info("Saved evaluation to %s", eval_path)

    finally:
        if not args.keep_extracted:
            for tmp in temp_dirs:
                if tmp.exists() and "model_extraction_" in tmp.name:
                    shutil.rmtree(tmp, ignore_errors=True)
                    logger.debug("Removed temp dir %s", tmp)

    logger.info("Circular-shift prediction pipeline complete ✓")


if __name__ == "__main__":
    main()
