import argparse
import logging
import re
from pathlib import Path
from statistics import mean, median
from typing import Literal

import pandas as pd
import wandb
from wandb.apis.public import Run

from training.prediction_pipeline_rolling_window import (
    _require_config,
    identify_wandb_run,
    read_models,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CUDA_VISIBLE_DEVICES_PATTERN = re.compile(r"CUDA_VISIBLE_DEVICES:\s*\[(\d+)\]")
LOCAL_WANDB_RUN_DIR_PATTERN = re.compile(r"^run-(\d{8}_\d{6})-([a-z0-9]+)$")

PerWindowAggregation = Literal["max", "median", "q95", "q90"]


def _list_prediction_files(prediction_dir: Path) -> list[Path]:
    return sorted(prediction_dir.rglob("predictions_*.parquet"))


def _prediction_file_matches_run(
    filename: str,
    model_alias: str,
    timestamp: str,
    window_index: int,
) -> bool:
    if not filename.startswith(f"predictions_{model_alias}_"):
        return False
    if f"_{timestamp}_window{window_index}" not in filename:
        return False
    return filename.endswith(".parquet")


def _extract_gpu_index_from_log_text(log_text: str) -> int | None:
    match = CUDA_VISIBLE_DEVICES_PATTERN.search(log_text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _detect_gpu_index_from_local_wandb(run_id: str, wandb_dir: Path) -> int | None:
    if not wandb_dir.exists():
        logger.warning("Local wandb directory does not exist: %s", wandb_dir)
        return None

    candidates: list[tuple[str, Path]] = []
    for entry in wandb_dir.iterdir():
        if not entry.is_dir():
            continue
        match = LOCAL_WANDB_RUN_DIR_PATTERN.match(entry.name)
        if not match:
            continue
        datetime_token, entry_run_id = match.group(1), match.group(2)
        if entry_run_id != run_id:
            continue
        candidates.append((datetime_token, entry))

    if not candidates:
        logger.warning("No local wandb run directories found for run_id=%s in %s", run_id, wandb_dir)
        return None

    candidates.sort(key=lambda x: x[0])

    for datetime_token, run_dir in candidates:
        output_log = run_dir / "files" / "output.log"
        if not output_log.exists():
            logger.debug("Missing output.log in %s", run_dir)
            continue
        try:
            log_text = output_log.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.debug("Could not read %s: %s", output_log, exc)
            continue

        gpu_index = _extract_gpu_index_from_log_text(log_text)
        if gpu_index is not None:
            logger.debug(
                "Detected GPU index %d for run %s from local dir %s (datetime=%s)",
                gpu_index,
                run_id,
                run_dir,
                datetime_token,
            )
            return gpu_index

    logger.warning("Could not determine GPU index from local output.log for run %s", run_id)
    return None


def _aggregate_memory_bytes(values: pd.Series, aggregation: PerWindowAggregation) -> float | None:
    if values.empty:
        return None
    if aggregation == "max":
        return float(values.max())
    if aggregation == "median":
        return float(values.median())
    if aggregation == "q95":
        return float(values.quantile(0.95))
    if aggregation == "q90":
        return float(values.quantile(0.90))
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def _aggregate_memory_gb_from_history(
    run: Run,
    gpu_index: int,
    per_window_aggregation: PerWindowAggregation,
) -> float | None:
    # First, try the key requested in the original requirement.
    preferred_key = f"gpu.{gpu_index}.memoryAllocatedBytes"

    def _scan_values(key: str) -> pd.Series:
        values: list[float] = []
        history_iter = run.scan_history(keys=[key])
        for row in history_iter:
            value = row.get(key)
            if value is None:
                continue
            values.append(float(value))
        return pd.Series(values, dtype="float64")

    try:
        preferred_values = _scan_values(preferred_key)
    except Exception as exc:
        logger.warning("Failed to scan history for run %s (%s): %s", run.id, preferred_key, exc)
        preferred_values = pd.Series(dtype="float64")

    aggregated_bytes = _aggregate_memory_bytes(preferred_values, per_window_aggregation)
    if aggregated_bytes is not None:
        return aggregated_bytes / (1024 ** 3)

    # Fallback for runs where metrics are stored under system stream keys.
    fallback_key = f"system.gpu.{gpu_index}.memoryAllocatedBytes"
    try:
        system_history_raw = run.history(stream="system", pandas=True)
    except Exception as exc:
        logger.warning(
            "Failed to load system history for run %s (%s): %s",
            run.id,
            fallback_key,
            exc,
        )
        return None

    if isinstance(system_history_raw, pd.DataFrame):
        system_history = system_history_raw
    else:
        system_history = pd.DataFrame(system_history_raw)

    if fallback_key not in system_history.columns:
        logger.warning(
            "No values found for %s in run %s (preferred key %s also missing).",
            fallback_key,
            run.id,
            preferred_key,
        )
        return None

    fallback_values = pd.to_numeric(system_history[fallback_key], errors="coerce").dropna()
    if fallback_values.empty:
        logger.warning(
            "Only NaN values found for %s in run %s (preferred key %s also missing).",
            fallback_key,
            run.id,
            preferred_key,
        )
        return None

    aggregated_bytes = _aggregate_memory_bytes(fallback_values, per_window_aggregation)
    if aggregated_bytes is None:
        return None
    return aggregated_bytes / (1024 ** 3)


def main(
    root_model_path: Path,
    wandb_project: str,
    wandb_entity: str | None,
    output_dir: Path,
    per_window_aggregation: PerWindowAggregation = "max",
    start_window: int = 0,
    persist_archive_dir: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    local_wandb_dir = repo_root / "wandb"

    log_file = output_dir / "gpu_memory_usage.log"
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(file_handler)

    prediction_files = _list_prediction_files(output_dir)
    prediction_names = {p.name for p in prediction_files}
    if not prediction_names:
        raise FileNotFoundError(f"No prediction parquet files found in {output_dir}")

    logger.info("Found %d prediction parquet files in %s", len(prediction_names), output_dir)

    temp_dirs_to_cleanup: list[Path] = []
    per_model_records: dict[str, list[dict]] = {}
    try:
        model_data = list(read_models(root_model_path, temp_dirs_to_cleanup, window_start=start_window))
        if not model_data:
            raise FileNotFoundError(f"No rolling window models found in {root_model_path}")

        api = wandb.Api()
        project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project

        for window_index, model_dir in model_data:
            run_info = identify_wandb_run(model_dir, window_index, wandb_entity, wandb_project)
            run_id = run_info["run_id"]
            run_config = run_info["config"]

            model_alias = str(_require_config(run_config, "model_alias"))
            timestamp = str(_require_config(run_config, "date_time"))

            has_prediction_for_run = any(
                _prediction_file_matches_run(name, model_alias, timestamp, window_index)
                for name in prediction_names
            )
            if not has_prediction_for_run:
                logger.info(
                    "Skipping run %s (%s window %d): no matching prediction file in %s",
                    run_id,
                    model_alias,
                    window_index,
                    output_dir,
                )
                continue

            run = api.run(f"{project_path}/{run_id}")
            gpu_index = _detect_gpu_index_from_local_wandb(run_id=run_id, wandb_dir=local_wandb_dir)
            if gpu_index is None:
                continue

            window_memory_gb = _aggregate_memory_gb_from_history(
                run,
                gpu_index,
                per_window_aggregation,
            )
            if window_memory_gb is None:
                continue

            per_model_records.setdefault(model_alias, []).append(
                {
                    "model_alias": model_alias,
                    "window_index": window_index,
                    "run_id": run_id,
                    "gpu_index": gpu_index,
                    "per_window_aggregation": per_window_aggregation,
                    "window_memory_gb": window_memory_gb,
                }
            )
            logger.info(
                "Run %s | model %s | window %d | gpu %d | %s memory %.3f GB",
                run_id,
                model_alias,
                window_index,
                gpu_index,
                per_window_aggregation,
                window_memory_gb,
            )

        all_records = [record for records in per_model_records.values() for record in records]
        if not all_records:
            raise RuntimeError("No GPU memory records collected. Check run logs/history availability.")

        per_window_df = pd.DataFrame(all_records).sort_values(["model_alias", "window_index"])
        per_window_path = output_dir / "gpu_memory_per_window.csv"
        per_window_df.to_csv(per_window_path, index=False)

        summary_rows = []
        for model_alias, records in sorted(per_model_records.items()):
            values = [r["window_memory_gb"] for r in records]
            summary_rows.append(
                {
                    "model_alias": model_alias,
                    "n_windows": len(values),
                    "per_window_aggregation": per_window_aggregation,
                    "window_memory_gb_max": max(values),
                    "window_memory_gb_mean": mean(values),
                    "window_memory_gb_median": median(values),
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values("model_alias")
        summary_path = output_dir / "gpu_memory_summary_per_model.csv"
        summary_df.to_csv(summary_path, index=False)

        logger.info("Saved per-window GPU memory to %s", per_window_path)
        logger.info("Saved per-model GPU memory summary to %s", summary_path)
    finally:
        if temp_dirs_to_cleanup and not persist_archive_dir:
            for temp_dir in temp_dirs_to_cleanup:
                if temp_dir.exists() and temp_dir.is_dir() and "model_extraction_" in temp_dir.name:
                    try:
                        import shutil

                        shutil.rmtree(temp_dir)
                    except Exception as exc:
                        logger.warning("Failed to cleanup temp directory %s: %s", temp_dir, exc)


def prepare_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model directory or path to rolling-window model archives.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory path.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        required=True,
        help="Weights and Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights and Biases entity name.",
    )
    parser.add_argument(
        "--per-window-aggregation",
        type=str,
        default="max",
        choices=["max", "median", "q95", "q90"],
        help="Aggregation used to reduce per-window GPU memory history.",
    )
    parser.add_argument(
        "--start-window",
        type=int,
        default=0,
        help="Index of the first window to process (inclusive).",
    )
    parser.add_argument(
        "--persist-archive-dir",
        action="store_true",
        help=(
            "Whether to persist extracted archive directories (with a 'model_extraction_' prefix) "
            "instead of cleaning up."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = prepare_args()
    main(
        root_model_path=Path(args.model_path),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        output_dir=Path(args.output_dir),
        per_window_aggregation=args.per_window_aggregation,
        start_window=args.start_window,
        persist_archive_dir=args.persist_archive_dir,
    )