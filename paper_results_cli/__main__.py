"""Command-line entrypoint for the paper-results pipeline.

Usage
-----
Run a specific result module::

    python -m paper_results_cli --mcs
    python -m paper_results_cli --fbeta --mae
    python -m paper_results_cli --ablation --ablation-seeds 778 860

Run everything::

    python -m paper_results_cli --all

Override defaults::

    python -m paper_results_cli --all \\
        --predictions-dir path/to/outputs \\
        --output-dir paper_results/v2 \\
        --models nhits nbeatsx \\
        --regimes-file custom_regimes.json

Custom regimes JSON format (``--regimes-file``)::

    {
        "high_pv":         ["2025-03-01", "2025-03-31"],
        "pv_distribution": ["2025-04-01", "2025-06-30"]
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import (
    BENCHMARK_MODELS,
    DEFAULT_MODELS,
    DEFAULT_REGIMES,
    DEFAULT_START_DATE,
    DATASET_NAME,
    PaperConfig,
)
from .style import apply_paper_style, save_color_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m paper_results_cli",
        description="Generate publication-quality tables and figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Module selection ────────────────────────────────────────────────────
    sel = p.add_argument_group("module selection")
    sel.add_argument(
        "--all", action="store_true",
        help="Run every result module (overrides individual flags).",
    )
    sel.add_argument(
        "--mcs", action="store_true",
        help="MCS inclusion-rate tables (overall + per-regime).",
    )
    sel.add_argument(
        "--fbeta", action="store_true",
        help="F-beta bootstrap dominance tables.",
    )
    sel.add_argument(
        "--mae", action="store_true",
        help="MAE evaluation table and per-horizon difference heatmap.",
    )
    sel.add_argument(
        "--ablation", action="store_true",
        help="Circular-shift ablation DM dominance table.",
    )
    sel.add_argument(
        "--descriptive", action="store_true",
        help="Descriptive statistics and plots.",
    )
    sel.add_argument(
        "--interpretability", action="store_true",
        help="IG feature-group table and horizon-block heatmap.",
    )

    # ── I/O paths ───────────────────────────────────────────────────────────
    io = p.add_argument_group("I/O paths")
    io.add_argument(
        "--predictions-dir", type=Path, default=Path("outputs/"),
        metavar="DIR",
        help="Root of full-covariate rolling-window predictions "
             "(default: %(default)s).",
    )
    io.add_argument(
        "--benchmarks-dir", type=Path, default=None,
        metavar="DIR",
        help="Root of benchmark predictions (default: same as --predictions-dir).",
    )
    io.add_argument(
        "--no-cov-dir", type=Path,
        default=Path("outputs_no_covariates/"),
        metavar="DIR",
        help="No-covariate predictions root for ablation "
             "(default: %(default)s).",
    )
    io.add_argument(
        "--shift-dir", type=Path,
        default=Path("outputs_shifted_targets_17/"),
        metavar="DIR",
        help="Shifted-target predictions root for ablation "
             "(default: %(default)s).",
    )
    io.add_argument(
        "--output-dir", type=Path, default=Path("paper_results"),
        metavar="DIR",
        help="Root directory for generated tables, figures, and style files "
             "(default: %(default)s).",
    )
    io.add_argument(
        "--ablation-cache-dir", type=Path, default=None,
        metavar="DIR",
        help="Optional parquet cache for ablation predictions "
             "(disabled by default; pass a directory to enable).",
    )
    io.add_argument(
        "--interpretability-ig-root", type=Path,
        default=Path("outputs_paper"),
        metavar="DIR",
        help="Root containing per-TSO IG outputs for interpretability "
             "(default: %(default)s).",
    )
    io.add_argument(
        "--interpretability-dataset-root", type=Path,
        default=Path("data/model_data_paper_actuals_1h_lag"),
        metavar="DIR",
        help="Dataset metadata root for interpretability "
             "(default: %(default)s).",
    )

    # ── Dataset ─────────────────────────────────────────────────────────────
    ds = p.add_argument_group("dataset")
    ds.add_argument(
        "--dataset-name", default=DATASET_NAME,
        metavar="NAME",
        help="Feature-set sub-directory name (default: %(default)s).",
    )
    ds.add_argument(
        "--start-date", default=DEFAULT_START_DATE,
        metavar="YYYY-MM-DD",
        help="Evaluation start date (default: %(default)s).",
    )
    ds.add_argument(
        "--regimes-file", type=Path, default=None,
        metavar="FILE",
        help="Optional JSON file that overrides the default stress regimes.  "
             "Format: {\"label\": [\"YYYY-MM-DD\", \"YYYY-MM-DD\"], ...}",
    )
    ds.add_argument(
        "--dataset-root-dir-path", type=Path,
        default=Path("data/model_data_paper_actuals_1h_lag"), metavar="FILE",
        help="Path to the dataset root directory (default: %(default)s). This is required for the descriptive outputs.",
    )
    ds.add_argument(
        "--train-config-yaml-path", type=Path, default=Path("training/neural_model_parameters.yaml"), metavar="FILE",
        help="Path to the training config YAML file (default: %(default)s). This is required for the descriptive outputs.",
    )
    ds.add_argument(
        "--interpretability-dataset-name",
        default="basic_day_ahead_price_wind_pv_production_consumption_sce",
        metavar="NAME",
        help="Feature-set directory/name used for interpretability IG outputs "
             "(default: %(default)s).",
    )

    # ── Model selection ──────────────────────────────────────────────────────
    mod = p.add_argument_group("model selection")
    mod.add_argument(
        "--models", nargs="+",
        default=DEFAULT_MODELS + BENCHMARK_MODELS,
        metavar="MODEL",
        help="Seed-agnostic model names to include "
             "(default: %(default)s).",
    )
    mod.add_argument(
        "--interpretability-model-filter",
        default="nhits",
        metavar="MODEL",
        help="Substring used to select IG model directories for interpretability "
             "(default: %(default)s).",
    )

    # ── Statistical parameters ───────────────────────────────────────────────
    stat = p.add_argument_group("statistical parameters")
    stat.add_argument(
        "--mcs-alpha", type=float, default=0.25,
        metavar="ALPHA",
        help="MCS significance level (default: %(default)s).",
    )
    stat.add_argument(
        "--fbeta-beta", type=float, default=2.0,
        metavar="BETA",
        help="Beta for the F-beta score (default: %(default)s).",
    )
    stat.add_argument(
        "--fbeta-eps", type=float, default=0.05,
        metavar="EPS",
        help="Epsilon for the bootstrap acceptance rule (default: %(default)s).",
    )
    stat.add_argument(
        "--ablation-seeds", nargs="+", type=int, default=[778, 860],
        metavar="SEED",
        help="Training seeds for the ablation module (default: %(default)s).",
    )
    stat.add_argument(
        "--dm-sparsity-threshold", type=float, default=0.7,
        metavar="THRESH",
        help="Sparsity threshold for DM dominance classification (default: %(default)s).",
    )
    stat.add_argument(
        "--dm-alpha", type=float, default=0.05,
        metavar="ALPHA",
        help="Significance level for DM tests (default: %(default)s).",
    )
    stat.add_argument(
        "--interpretability-start-window", type=int, default=2,
        metavar="N",
        help="First rolling window to include in interpretability outputs "
             "(default: %(default)s).",
    )

    # ── Checkpoint selection ────────────────────────────────────────────────
    ckpt = p.add_argument_group("checkpoint selection")
    ckpt.add_argument(
        "--best-checkpoint", action="store_true",
        help="Use best-validation-checkpoint neural predictions/IG artifacts. "
             "Benchmarks are merged unchanged.",
    )

    # ── Output settings ──────────────────────────────────────────────────────
    out = p.add_argument_group("output settings")
    out.add_argument(
        "--figure-format", choices=["png", "pdf"], default="png",
        help="Plot file format (default: %(default)s).",
    )

    # ── Logging ─────────────────────────────────────────────────────────────
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return p


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> PaperConfig:
    """Convert parsed CLI arguments to a ``PaperConfig`` instance."""
    regimes = dict(DEFAULT_REGIMES)
    if args.regimes_file is not None:
        try:
            raw = json.loads(args.regimes_file.read_text())
            regimes = {k: tuple(v) for k, v in raw.items()}
        except Exception as exc:
            raise SystemExit(
                f"Error reading --regimes-file '{args.regimes_file}': {exc}"
            ) from exc

    return PaperConfig(
        predictions_dir=args.predictions_dir,
        benchmarks_dir=args.benchmarks_dir,
        no_cov_dir=args.no_cov_dir,
        shift_dir=args.shift_dir,
        ablation_cache_dir=(None if str(args.ablation_cache_dir) == "" else args.ablation_cache_dir),
        best_checkpoint=args.best_checkpoint,
        dataset_name=args.dataset_name,
        train_config_yaml_path=args.train_config_yaml_path,
        dataset_dir=args.dataset_root_dir_path,
        start_date=args.start_date,
        output_dir=args.output_dir,
        figure_format=args.figure_format,
        models=list(args.models),
        stress_regimes=regimes,
        mcs_alpha=args.mcs_alpha,
        fbeta_beta=args.fbeta_beta,
        fbeta_eps_acceptance=args.fbeta_eps,
        ablation_seeds=list(args.ablation_seeds),
        dm_sparsity_threshold=args.dm_sparsity_threshold,
        dm_alpha=args.dm_alpha,
    )


# ---------------------------------------------------------------------------
# Module dispatch stubs (Phase-2 bodies will fill these in)
# ---------------------------------------------------------------------------


def _run_mcs(config: PaperConfig) -> None:
    logger.info("[mcs] Starting MCS evaluation ...")
    from .modules import mcs as mcs_mod  # noqa: PLC0415

    mcs_mod.run(config)
    logger.info("[mcs] Done.")


def _run_fbeta(config: PaperConfig) -> None:
    logger.info("[fbeta] Starting F-beta bootstrap evaluation ...")
    from .modules import fbeta as fbeta_mod  # noqa: PLC0415

    fbeta_mod.run(config)
    logger.info("[fbeta] Done.")


def _run_mae(config: PaperConfig) -> None:
    logger.info("[mae] Starting MAE evaluation ...")
    from .modules import mae as mae_mod  # noqa: PLC0415

    mae_mod.run(config)
    logger.info("[mae] Done.")


def _run_ablation(config: PaperConfig) -> None:
    logger.info("[ablation] Starting circular-shift ablation DM tests ...")
    from .modules import ablation as ablation_mod  # noqa: PLC0415

    ablation_mod.run(config)
    logger.info("[ablation] Done.")


def run_descriptive(config: PaperConfig) -> None:
    logger.info("[descriptive] Starting descriptive outputs ...")
    from .modules import descriptive as descriptive_mod  # noqa: PLC0415

    descriptive_mod.run(config)
    logger.info("[descriptive] Done.")


def _run_interpretability(config: PaperConfig, args: argparse.Namespace) -> None:
    logger.info("[interpretability] Starting IG feature-group outputs ...")
    from .modules import interpretability as interpretability_mod  # noqa: PLC0415

    interpretability_mod.run(
        config,
        ig_root=args.interpretability_ig_root,
        dataset_root=args.interpretability_dataset_root,
        dataset_name=args.interpretability_dataset_name,
        model_filter=args.interpretability_model_filter,
        start_window=args.interpretability_start_window,
    )
    logger.info("[interpretability] Done.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    config = _build_config(args)
    config.make_output_dirs()

    # Always persist the colour registry so figures can reference it.
    apply_paper_style()
    save_color_registry(config.style_dir)

    # Determine which modules to run.
    run_all = args.all
    run_mcs     = run_all or args.mcs
    run_fbeta   = run_all or args.fbeta
    run_mae     = run_all or args.mae
    run_ablation_flag = run_all or args.ablation
    run_descriptive_flag = run_all or args.descriptive
    run_interpretability_flag = run_all or args.interpretability

    if not any([run_mcs, run_fbeta, run_mae, run_ablation_flag, run_descriptive_flag, run_interpretability_flag]):
        parser.print_help()
        raise SystemExit(
            "\nNo module selected.  Pass --mcs, --fbeta, --mae, --ablation, --descriptive, --interpretability, or --all."
        )

    errors: list[str] = []

    if run_mcs:
        try:
            _run_mcs(config)
        except Exception as exc:
            logger.error("MCS module failed: %s", exc, exc_info=True)
            errors.append(f"mcs: {exc}")

    if run_fbeta:
        try:
            _run_fbeta(config)
        except Exception as exc:
            logger.error("F-beta module failed: %s", exc, exc_info=True)
            errors.append(f"fbeta: {exc}")

    if run_mae:
        try:
            _run_mae(config)
        except Exception as exc:
            logger.error("MAE module failed: %s", exc, exc_info=True)
            errors.append(f"mae: {exc}")

    if run_ablation_flag:
        try:
            _run_ablation(config)
        except Exception as exc:
            logger.error("Ablation module failed: %s", exc, exc_info=True)
            errors.append(f"ablation: {exc}")

    if run_descriptive_flag:
        try:
            run_descriptive(config)
        except Exception as exc:
            logger.error("Descriptive module failed: %s", exc, exc_info=True)
            errors.append(f"descriptive: {exc}")

    if run_interpretability_flag:
        try:
            _run_interpretability(config, args)
        except Exception as exc:
            logger.error("Interpretability module failed: %s", exc, exc_info=True)
            errors.append(f"interpretability: {exc}")

    if errors:
        logger.error(
            "The following modules raised errors:\n  %s",
            "\n  ".join(errors),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
