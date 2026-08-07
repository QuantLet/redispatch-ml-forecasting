#!/usr/bin/env python
"""
CLI entry-point for the shifted-target Nixtla training pipeline.

Usage examples
--------------
# Single window, NHITS + TFT, shift=9 hours
python train.py \\
    --dataset-path data/model_data_new_features_debug/TransnetBW_full_combo_dataset.parquet \\
    --tso TransnetBW \\
    --shift-hours 9 \\
    --models nhits tft

# Rolling windows, all models
python train.py \\
    --dataset-path data/model_data_new_features_debug/TenneT_DE_full_combo_dataset.parquet \\
    --tso TenneT_DE \\
    --rolling-window \\
    --models nhits nbeatsx tft tft_quantile
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from training.data_prep import load_dataset, to_nixtla_format, prepare_shifted_dataset
from training.runner import TrainConfig, train_single_window, train_rolling_windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def set_n_threads(n_threads: int | None) -> None:
    if n_threads is not None and n_threads > 0:
        import torch

        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)
        logger.info("Set number of threads to %d", n_threads)
    else:
        logger.info("Using default number of threads")


# ── YAML config helpers ───────────────────────────────────────────────────────

# Flat keys in the YAML config that map 1-to-1 to argparse dest names.
_YAML_FLAT_KEYS = {
    # Dataset / direction
    "direction",
    # Forecast
    "forecast_horizon",
    "input_size",
    # Shifting / calendar
    "shift_hours",
    "holidays_path",
    # Date range
    "train_start",
    "valid_start",
    "test_start",
    # Training hyper-parameters
    "max_steps",
    "val_check_steps",
    "early_stop_patience",
    "batch_size",
    "windows_batch_size",
    "learning_rate",
    "random_seed",
    "local_scaler_type",
    # Rolling window
    "n_train_months",
    "n_valid_months",
    "n_test_months",
    # Models
    "models",
    # Output / logging
    "output_dir",
    "eval_dir",
    "wandb_project",
    "wandb_entity",
    # Checkpointing
    "persist_checkpoints",
    "persist_checkpoints_to_wandb",
    "checkpoint_compression",
    "checkpoint_compression_n_threads",
    "checkpoint_selection",
    "skip_explainability",
}


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load a YAML training config and return it as a dict."""
    if yaml is None:
        raise ImportError(
            "PyYAML is required to use --config.  Install it with: pip install pyyaml"
        )
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    logger.info("Loaded config from %s", config_path)
    return cfg


def _flat_defaults_from_yaml(yaml_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract flat (non-nested) values from the YAML config for argparse set_defaults()."""
    return {k: v for k, v in yaml_cfg.items() if k in _YAML_FLAT_KEYS}


def _normalize_tso_key(tso: str) -> str:
    """Normalise a TSO name for YAML key lookup (spaces → underscores)."""
    return tso.strip().replace(" ", "_")


def _explicit_cli_keys() -> set[str]:
    """Return argparse dest names that were explicitly supplied on the command line."""
    explicit: set[str] = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            key_part = arg[2:].split("=")[0]
            explicit.add(key_part.replace("-", "_"))
    return explicit


def apply_tso_overrides(
    args: argparse.Namespace,
    yaml_cfg: dict[str, Any],
    tso: str,
    explicit_cli_keys: set[str],
) -> None:
    """
    Mutate *args* in-place with per-TSO overrides from the YAML config.

    Precedence: CLI (explicit) > TSO override > base YAML > hardcoded default.

    Flat keys (``input_size``, ``shift_hours``, ``models``, ...) are applied only
    when the user did NOT supply that flag on the command line.

    Model-param dicts are *always* deep-merged: global → TSO-specific,
    so TSO-level keys win while unmentioned keys keep their global value.
    """
    tso_norm = _normalize_tso_key(tso)
    overrides: dict[str, Any] = yaml_cfg.get("tso_overrides", {})

    # Accept both "TenneT DE" and "TenneT_DE" as keys
    tso_cfg: dict[str, Any] = overrides.get(tso_norm) or overrides.get(tso) or {}
    if not tso_cfg:
        logger.debug("No TSO overrides found for '%s'", tso_norm)
        return

    flat_ov = {k: v for k, v in tso_cfg.items() if not isinstance(v, dict)}
    model_ov = {k: v for k, v in tso_cfg.items() if isinstance(v, dict)}

    if flat_ov:
        logger.info("Applying TSO overrides for '%s': %s", tso_norm, flat_ov)

    for key, value in flat_ov.items():
        if key not in _YAML_FLAT_KEYS:
            continue
        if key in explicit_cli_keys:
            logger.debug("  Skipping TSO override for '%s' (set via CLI)", key)
            continue
        setattr(args, key, value)
        logger.info("  %s = %s", key, value)

    # Re-normalise model list if it was overridden
    if "models" in flat_ov and "models" not in explicit_cli_keys:
        args.models = _normalize_models(args.models)

    # Deep-merge model params: global → TSO-specific (TSO wins per key)
    if model_ov:
        merged: dict[str, Any] = dict(getattr(args, "model_params", {}))
        for model_name, params in model_ov.items():
            if model_name in merged:
                merged[model_name] = {**merged[model_name], **params}
            else:
                merged[model_name] = params
        args.model_params = merged
        logger.info("  Applied TSO model param overrides for: %s", list(model_ov))


def parse_args() -> argparse.Namespace:
    # ── Two-pass: extract --config before finalising defaults ─────────────────
    # (argparse cannot read set_defaults after add_argument, so we pre-parse
    #  --config with a minimal parser to get the YAML path early.)
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_args, _ = _pre.parse_known_args()

    yaml_cfg: dict[str, Any] = {}
    if _pre_args.config:
        yaml_cfg = load_yaml_config(_pre_args.config)

    p = argparse.ArgumentParser(
        description="Train shifted-target Nixtla models for redispatch forecasting."
    )

    # ── Config file ───────────────────────────────────────────────────────────
    p.add_argument(
        "--config", default=None,
        help=(
            "Path to a YAML config file (e.g. training/train_config.yaml).  "
            "Values override hardcoded defaults; CLI flags override the YAML."
        ),
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--dataset-path", required=True,
        help="Path to the .parquet dataset (companion .json metadata is loaded automatically).",
    )
    p.add_argument(
        "--direction", default="both", choices=["up", "down", "both"],
        help="Direction filter – 'up', 'down', or 'both' (default: both).",
    )

    # ── Shifting ──────────────────────────────────────────────────────────────
    p.add_argument("--shift-hours", type=int, default=6, help="Target shift in hours (default: 6).")

    # ── Date time ──────────────────────────────────────────────────────────────
    p.add_argument("--date-time", default=None, type=str, help="Optional wandb date_time experiment override. Useful if some experiments are run later than others, but they should be grouped.")

    # ── Threads ──────────────────────────────────────────────────────────────
    p.add_argument("--n-threads", type=int, default=None, help="Number of threads to use (default: all available).")

    # ── Calendar ──────────────────────────────────────────────────────────────
    p.add_argument("--no-calendar", action="store_true", help="Skip adding calendar features.")
    p.add_argument("--holidays-path", default=None, help="Override for holidays CSV path.")

    # ── Forecast / model ──────────────────────────────────────────────────────
    p.add_argument("--forecast-horizon", type=int, default=24, help="Forecast horizon h (default: 24).")
    p.add_argument("--input-size", type=int, default=24, help="Input lookback L (default: 24).")
    p.add_argument(
        "--models", nargs="+", default=["nhits", "tft"],
        help=(
            "Which models to train. Accepts space-separated and/or comma-separated values. "
            "Choices: nhits, nbeatsx, tft, tft_quantile, lstm. "
            "Examples: --models nhits nbeatsx | --models nhits,nbeatsx,lstm"
        ),
    )

    # ── Date range ────────────────────────────────────────────────────────────
    p.add_argument("--train-start", default="2021-10-01", help="Training start date.")
    p.add_argument("--valid-start", default="2024-10-01", help="Validation start date.")
    p.add_argument("--test-start", default="2025-01-01", help="Test start date.")

    # ── Training hyper-parameters ─────────────────────────────────────────────
    p.add_argument("--max-steps", type=int, default=5000, help="Max training steps (default: 5000).")
    p.add_argument("--val-check-steps", type=int, default=50, help="Validate every N steps (default: 50).")
    p.add_argument("--early-stop-patience", type=int, default=20, help="Early stopping patience steps.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    p.add_argument("--windows-batch-size", type=int, default=64, help="Windows batch size (default: 64).")
    p.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4).")
    p.add_argument("--random-seed", type=int, default=778, help="Random seed (default: 778).")
    p.add_argument("--local-scaler-type", default="standard", help="Nixtla local scaler (default: standard).")

    # ── Rolling window ────────────────────────────────────────────────────────
    p.add_argument("--rolling-window", action="store_true", help="Use rolling-window training.")
    p.add_argument("--n-train-months", type=int, default=37, help="Training period in months (default: 36).")
    p.add_argument("--n-valid-months", type=int, default=2, help="Validation period in months (default: 2).")
    p.add_argument("--n-test-months", type=int, default=1, help="Test period in months (default: 1).")
    p.add_argument("--start-window", type=int, default=0, help="Start from N-th window (0-indexed, default: 0). Useful for resuming rolling-window training.")

    # ── Output / logging ──────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="outputs", help="Directory for model artifacts.")
    p.add_argument(
        "--eval-dir",
        default=None,
        help="Directory for saving evaluation predictions (per-model/per-window + model-level rolling files).",
    )
    p.add_argument("--wandb-project", default="redispatch-forecasting", help="W&B project name.")
    p.add_argument("--wandb-entity", default=None, help="W&B entity (team or user).")
    p.add_argument(
        "--persist-checkpoints",
        action="store_true",
        help="Persist checkpoints locally in the output directory.",
    )
    p.add_argument(
        "--persist-checkpoints-to-wandb",
        action="store_true",
        help="Log checkpoints to W&B artifacts (saved locally first).",
    )
    p.add_argument(
        "--checkpoint-compression",
        default=None,
        type=int,
        help="Optional zstd compression level for checkpoint artifacts.",
    )
    p.add_argument(
        "--checkpoint-compression-n-threads",
        default=20,
        type=int,
        help="Number of threads for checkpoint compression (default: 20).",
    )
    p.add_argument(
        "--checkpoint-selection",
        choices=["last", "best", "both"],
        default="last",
        help="Which checkpoint(s) to evaluate and explain (default: last).",
    )
    p.add_argument(
        "--skip-explanability",
        "--skip-explainability",
        dest="skip_explainability",
        action="store_true",
        help="Skip explainability for selected checkpoints.",
    )

    # ── Apply YAML config as argparse defaults (CLI overrides YAML) ───────────
    if yaml_cfg:
        p.set_defaults(**_flat_defaults_from_yaml(yaml_cfg))

    args = p.parse_args()

    # ── Attach model-specific params dict (nested YAML sections) ─────────────
    args.model_params = {k: v for k, v in yaml_cfg.items() if isinstance(v, dict)}
    args._yaml_cfg = yaml_cfg  # carried through to main() for TSO override lookup

    args.models = _normalize_models(args.models)
    return args


def _normalize_models(raw_models: list[str]) -> list[str]:
    allowed = {"nhits", "nbeatsx", "tft", "tft_quantile", "lstm"}
    models: list[str] = []
    for item in raw_models:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        models.extend(parts)

    # Normalize to lowercase
    models = [m.lower() for m in models]

    invalid = [m for m in models if m not in allowed]
    if invalid:
        raise SystemExit(
            f"train_pipeline.py: error: argument --models: invalid choice(s): {invalid}. "
            f"Choose from {sorted(allowed)}"
        )

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for m in models:
        if m not in seen:
            deduped.append(m)
            seen.add(m)
    return deduped


def main() -> None:
    args = parse_args()
    set_n_threads(args.n_threads)

    # ── 1. Load raw dataset ───────────────────────────────────────────────────
    logger.info("Loading dataset from %s", args.dataset_path)
    raw_df, metadata = load_dataset(args.dataset_path)
    logger.info("Raw dataset: %d rows × %d cols", *raw_df.shape)

    # ── 2. Convert to Nixtla format (ds, y, unique_id + covariates) ──────────
    logger.info("Converting to Nixtla format (direction=%s)", args.direction)
    nixtla_df = to_nixtla_format(raw_df, direction=args.direction)
    logger.info("Nixtla DataFrame: %d rows, unique_ids=%s", len(nixtla_df), nixtla_df["unique_id"].unique().tolist())

    # ── 3. Shift + calendar + covariate classification ───────────────────────
    tso = metadata["operator"]

    # Apply per-TSO YAML overrides now that the TSO is known.
    # CLI flags (explicit_cli_keys) always win over these overrides.
    apply_tso_overrides(args, args._yaml_cfg, tso, _explicit_cli_keys())

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

    # ── 4. Build config ──────────────────────────────────────────────────────
    config = TrainConfig(
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
        eval_dir=args.eval_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        persist_checkpoints=args.persist_checkpoints,
        persist_checkpoints_to_wandb=args.persist_checkpoints_to_wandb,
        checkpoint_compression=args.checkpoint_compression,
        checkpoint_compression_n_threads=args.checkpoint_compression_n_threads,
        checkpoint_selection=args.checkpoint_selection,
        skip_explainability=args.skip_explainability,
        model_params=getattr(args, "model_params", {}),
    )

    # ── 5. Train + predict ─────────────────────────────────────────────────
    if config.rolling_window:
        logger.info("Starting rolling-window training")
        train_rolling_windows(
            shifted_df, config, future_cov, hist_cov, metadata,
            df_unshifted=nixtla_df, start_from_window=args.start_window,
        )
    else:
        logger.info("Starting single-window training")
        train_single_window(
            shifted_df, config, future_cov, hist_cov, metadata,
            df_unshifted=nixtla_df,
        )

    logger.info("Training complete ✓")


if __name__ == "__main__":
    main()

