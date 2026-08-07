#!/usr/bin/env python
"""
CLI entry-point for the per-(TSO, Direction) training pipeline.

Trains *separate* models for each TSO and direction combination.
No Weights & Biases login required – all artefacts are saved locally.

Usage examples
--------------
# Single window, two TSO datasets, both directions, NHITS + TFT
python train_pipeline_per_tso_direction.py \\
    --dataset-paths \\
        data/model_data_new_features_debug/basic_TenneT_DE.parquet \\
        data/model_data_new_features_debug/basic_TransnetBW.parquet \\
    --models nhits tft \\
    --shift-hours 9 \\
    --output-dir outputs_per_direction

# Discover all datasets in a directory, train only "up" direction
python train_pipeline_per_tso_direction.py \\
    --datasets-dir data/model_data_new_features_debug \\
    --directions up \\
    --models nhits \\
    --output-dir outputs_per_direction

# Rolling-window training for all TSOs, both directions
python train_pipeline_per_tso_direction.py \\
    --datasets-dir data/model_data_new_features_debug \\
    --rolling-window \\
    --n-train-months 37 \\
    --n-valid-months 2 \\
    --n-test-months 1 \\
    --output-dir outputs_per_direction_rolling

Training produces, for each (TSO, direction, model) tuple:
    output_dir/<feature_set>/<tso>/<direction>/<timestamp>/<model_alias>/
        nf_model/               – saved NeuralForecast object
        checkpoints/            – best checkpoint (.ckpt)
        training_logs/          – CSVLogger output (metrics per step)
        evaluation/
            predictions_test.csv
            metrics_<model>.json
        run_meta.json           – full run configuration
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
from training.runner_per_tso_direction import PerDirectionConfig, train_single_window, train_rolling_windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ALLOWED_MODELS = {"nhits", "nbeatsx", "tft", "tft_quantile", "lstm"}
ALLOWED_DIRECTIONS = {"up", "down"}


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
        description=(
            "Train separate NeuralForecast models for each TSO × direction combination. "
            "No Weights & Biases login required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Config file ───────────────────────────────────────────────────────────
    p.add_argument(
        "--config", default=None,
        help=(
            "Path to a YAML config file (e.g. training/train_config_per_direction.yaml).  "
            "Values override hardcoded defaults; CLI flags override the YAML."
        ),
    )

    # ── Dataset input ──────────────────────────────────────────────────────────
    dataset_group = p.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset-paths",
        nargs="+",
        metavar="PATH",
        help="One or more explicit .parquet dataset paths (one per TSO).",
    )
    dataset_group.add_argument(
        "--datasets-dir",
        metavar="DIR",
        help=(
            "Directory to scan for .parquet datasets.  "
            "Use --dataset-glob to filter (default: '*.parquet')."
        ),
    )
    p.add_argument(
        "--dataset-glob",
        default="*.parquet",
        help="Glob pattern applied inside --datasets-dir (default: '*.parquet').",
    )

    # ── Directions ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--directions",
        nargs="+",
        default=["up", "down"],
        choices=["up", "down"],
        help="Directions to train.  Default: both (up and down).",
    )

    # ── Shifting ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--shift-hours",
        type=int,
        default=6,
        help="Target shift in hours (default: 6).",
    )

    # ── Threads ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="PyTorch thread count (default: system default).",
    )

    # ── Calendar ───────────────────────────────────────────────────────────────
    p.add_argument("--no-calendar", action="store_true", help="Skip adding calendar features.")
    p.add_argument("--holidays-path", default=None, help="Override holidays CSV path.")

    # ── Forecast / model ───────────────────────────────────────────────────────
    p.add_argument(
        "--forecast-horizon",
        type=int,
        default=24,
        help="Forecast horizon h (default: 24).",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=24,
        help="Input lookback L (default: 24).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["nhits", "tft"],
        help=(
            "Models to train.  Space- or comma-separated. "
            "Choices: nhits, nbeatsx, tft, tft_quantile. "
            "Default: nhits tft."
        ),
    )

    # ── Date range ─────────────────────────────────────────────────────────────
    p.add_argument("--train-start", default="2021-10-01", help="Training start date.")
    p.add_argument("--valid-start", default="2024-10-01", help="Validation start date.")
    p.add_argument("--test-start",  default="2025-01-01", help="Test start date.")

    # ── Training hyper-parameters ──────────────────────────────────────────────
    p.add_argument("--max-steps",           type=int,   default=5000,  help="Max training steps (default: 5000).")
    p.add_argument("--val-check-steps",     type=int,   default=50,    help="Validate every N steps (default: 50).")
    p.add_argument("--early-stop-patience", type=int,   default=20,    help="Early-stopping patience (default: 20).")
    p.add_argument("--batch-size",          type=int,   default=16,    help="Batch size (default: 16).")
    p.add_argument("--windows-batch-size",  type=int,   default=64,    help="Windows batch size (default: 64).")
    p.add_argument("--learning-rate",       type=float, default=1e-4,  help="Learning rate (default: 1e-4).")
    p.add_argument("--random-seed",         type=int,   default=778,   help="Random seed (default: 778).")
    p.add_argument("--local-scaler-type",   default="standard",        help="Nixtla local scaler (default: standard).")

    # ── Rolling window ─────────────────────────────────────────────────────────
    p.add_argument("--rolling-window",     action="store_true", help="Use rolling-window training.")
    p.add_argument("--n-train-months",     type=int, default=37, help="Training period in months (default: 37).")
    p.add_argument("--n-valid-months",     type=int, default=2,  help="Validation period in months (default: 2).")
    p.add_argument("--n-test-months",      type=int, default=1,  help="Test period in months (default: 1).")
    p.add_argument("--start-window",       type=int, default=0,
                   help="Resume rolling-window from the N-th window (0-indexed, default: 0).")

    # ── Checkpointing / output ─────────────────────────────────────────────────
    p.add_argument(
        "--output-dir",
        default="outputs_per_direction",
        help="Root output directory for all artefacts (default: outputs_per_direction).",
    )
    p.add_argument(
        "--no-persist-checkpoints",
        action="store_true",
        help="Skip persisting NeuralForecast model and checkpoints to disk.",
    )

    # ── Apply YAML config as argparse defaults (CLI overrides YAML) ───────────
    if yaml_cfg:
        p.set_defaults(**_flat_defaults_from_yaml(yaml_cfg))

    args = p.parse_args()

    # ── Attach model-specific params dict (nested YAML sections) ─────────────
    # Exclude known non-model top-level dicts (e.g. tso_overrides) so they do
    # not show up as phantom model entries in build_model().
    _NON_MODEL_DICT_KEYS = {"tso_overrides"}
    args.model_params = {
        k: v for k, v in yaml_cfg.items()
        if isinstance(v, dict) and k not in _NON_MODEL_DICT_KEYS
    }
    args._yaml_cfg = yaml_cfg  # carried through to main() for TSO override lookup

    args.models = _normalize_models(args.models)
    args.directions = list(dict.fromkeys(args.directions))  # deduplicate, preserve order
    return args


def _normalize_models(raw: list[str]) -> list[str]:
    models: list[str] = []
    for item in raw:
        models.extend(p.strip() for p in item.split(",") if p.strip())

    models = [m.lower() for m in models]
    invalid = [m for m in models if m not in ALLOWED_MODELS]
    if invalid:
        raise SystemExit(
            f"train_pipeline_per_tso_direction.py: invalid model(s): {invalid}. "
            f"Choose from {sorted(ALLOWED_MODELS)}"
        )

    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for m in models:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def _set_n_threads(n: int | None) -> None:
    if n and n > 0:
        import torch
        torch.set_num_threads(n)
        torch.set_num_interop_threads(n)
        logger.info("PyTorch threads set to %d", n)


def _collect_dataset_paths(args: argparse.Namespace) -> list[Path]:
    if args.dataset_paths:
        paths = [Path(p) for p in args.dataset_paths]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"Dataset file(s) not found: {missing}")
        return paths

    # Discover from directory
    datasets_dir = Path(args.datasets_dir)
    if not datasets_dir.is_dir():
        raise SystemExit(f"--datasets-dir is not a directory: {datasets_dir}")

    paths = sorted(datasets_dir.glob(args.dataset_glob))
    if not paths:
        raise SystemExit(
            f"No files matching '{args.dataset_glob}' found in {datasets_dir}"
        )
    logger.info("Discovered %d dataset(s) in %s", len(paths), datasets_dir)
    return paths


# ── Main entry-point ──────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    _set_n_threads(args.n_threads)

    # ── Snapshot the post-YAML / pre-TSO-override state ───────────────────────
    # apply_tso_overrides() mutates args in-place, so we restore to this base
    # snapshot before each TSO to prevent settings from one TSO bleeding into
    # the next when a later TSO omits an override key.
    _explicit = _explicit_cli_keys()  # sys.argv never changes – compute once
    _base_flat = {k: getattr(args, k) for k in _YAML_FLAT_KEYS if hasattr(args, k)}
    _base_model_params = {k: dict(v) for k, v in args.model_params.items()}

    dataset_paths = _collect_dataset_paths(args)

    logger.info(
        "Per-(TSO × direction) training  |  datasets=%d  directions=%s  models=%s",
        len(dataset_paths),
        args.directions,
        args.models,
    )

    total_runs = len(dataset_paths) * len(args.directions)
    run_idx = 0

    for dataset_path in dataset_paths:
        # ── 1. Load raw dataset ───────────────────────────────────────────────
        logger.info("═" * 70)
        logger.info("Loading dataset: %s", dataset_path)
        raw_df, metadata = load_dataset(dataset_path)
        tso = metadata.get("operator", dataset_path.stem)
        logger.info("TSO=%s  |  shape=%s", tso, raw_df.shape)

        # ── Restore global YAML base, then apply per-TSO overrides ────────────
        # Reset to base values first so each TSO starts from the same clean
        # slate regardless of what previous iterations may have changed.
        for key, value in _base_flat.items():
            if key not in _explicit:
                setattr(args, key, value)
        args.model_params = {k: dict(v) for k, v in _base_model_params.items()}
        apply_tso_overrides(args, args._yaml_cfg, tso, _explicit)

        for direction in args.directions:
            run_idx += 1
            logger.info(
                "─" * 70 + "\n[Run %d/%d]  TSO=%s  direction=%s",
                run_idx, total_runs, tso, direction,
            )

            # ── 2. Convert to Nixtla format (single-direction series) ─────────
            nixtla_df = to_nixtla_format(raw_df, direction=direction)
            logger.info(
                "Nixtla format: %d rows  unique_ids=%s",
                len(nixtla_df),
                nixtla_df["unique_id"].unique().tolist(),
            )

            # ── 3. Shift + calendar + covariate classification ────────────────
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

            # ── 4. Build config ───────────────────────────────────────────────
            config = PerDirectionConfig(
                dataset_path=str(dataset_path),
                tso=tso,
                direction=direction,
                shift_hours=args.shift_hours,
                add_calendar=not args.no_calendar,
                holidays_path=args.holidays_path,
                train_start=args.train_start,
                valid_start=args.valid_start,
                test_start=args.test_start,
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
                persist_checkpoints=not args.no_persist_checkpoints,
                output_dir=args.output_dir,
                model_params=getattr(args, "model_params", {}),
            )

            # ── 5. Train ──────────────────────────────────────────────────────
            if config.rolling_window:
                logger.info("Starting rolling-window training")
                train_rolling_windows(
                    shifted_df, config, future_cov, hist_cov, metadata,
                    df_unshifted=nixtla_df,
                    start_from_window=args.start_window,
                )
            else:
                logger.info("Starting single-window training")
                train_single_window(
                    shifted_df, config, future_cov, hist_cov, metadata,
                    df_unshifted=nixtla_df,
                )

    logger.info("═" * 70)
    logger.info("All %d run(s) complete ✓", total_runs)


if __name__ == "__main__":
    main()
