#!/usr/bin/env python
"""
CLI entry-point for circular-shift augmented rolling-window training.

Applies a circular shift of K days to the **training portion** of each
rolling window (excluding calendar, meta and target columns), then trains
via runner.py.  The validation slice is left with real covariate values so
that early stopping and model selection work correctly.

Shift procedure (applied independently per unique_id group)
-----------------------------------------------------------
The formula is:

    ŷ_t = y_{(t + K) mod T}

where ŷ_t refers to **covariate** columns only – the target ``y`` is never
circularly shifted.  The shift is a pure circular (modular) index mapping:

    shifted[t] = original[(t + K_h) % T_h]   for t in the shifted block

where K_h = K × 24 (shift in hours) and T_h is the length of the block
being shifted.  Training and validation portions of each window are shifted
**separately** - each uses its own T_h (``train_hours`` and ``val_hours``
respectively) so that no training-period covariate values leak into
validation or vice-versa.  This gives the model one consistent
"shifted-covariate" mapping to both fit and early-stop on.

Prediction
----------
At test time, ``predict_with_shift_correction`` draws lookback context and
future covariates from the **clean** (unshifted) base dataset.  The first
``input_size`` rows of context come from the end of the validation period
with real covariates - a deliberate distribution mismatch that measures
how much the trained model relied on temporal covariate patterns.

Ordering
--------
The target shift (``prepare_shifted_dataset``) is applied first to the whole
dataset.  Then the per-window callback circularly shifts the *training* rows
of each window - the already-target-shifted future covariates are permuted
together with the historical covariates, so the target-shift alignment is
preserved.

Feature-group shifting
----------------------
All columns except the target ``y``, calendar features (hour, day, month,
is_weekend, is_holiday, is_workday) and Nixtla meta columns (ds, unique_id)
are shifted *jointly* – the same K is applied to every column so relative
alignment within a feature group is preserved.  Calendar features always
represent wall-clock time and are never shifted.

Exclusion of degenerate shifts
-------------------------------
The user may supply ``--exclude-multiples`` to reject shift values K whose
number of days divides evenly into any of the listed base periods.  For
example ``--exclude-multiples 7`` rejects K = 7, 14, 21, ... (weekly repeats)
and ``--exclude-multiples 7 365`` additionally rejects K = 365, 730, ... .

Usage examples
--------------
# Sample K randomly (seed 42), exclude weekly multiples, rolling window
# T_h is automatically set to each window's training-set length.
python -m training.circular_shift_training \\
    --dataset-path data/model_data_new_features_debug/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce_TenneT_DE.parquet \\
    --max-shift-days 90 \\
    --exclude-multiples 7 \\
    --shift-seed 42 \\
    --rolling-window \\
    --models nhits tft

# Fixed K = 15 days, weekly + yearly exclusion
python -m training.circular_shift_training \\
    --dataset-path data/model_data_new_features_debug/basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce_TenneT_DE.parquet \\
    --shift-k-days 15 \\
    --exclude-multiples 7 365 \\
    --rolling-window \\
    --models nhits tft \\
    --start-window 0
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from training.data_prep import (
    CALENDAR_COLS,
    load_dataset,
    to_nixtla_format,
    prepare_shifted_dataset,
)
from training.runner import TrainConfig, train_rolling_windows, train_single_window
from training.train_pipeline import (
    _flat_defaults_from_yaml,
    _normalize_models,
    _normalize_tso_key,
    _explicit_cli_keys,
    apply_tso_overrides,
    load_yaml_config,
    set_n_threads,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _normalize_eval_dir_keys(yaml_cfg: dict[str, Any]) -> dict[str, Any]:
    """Support both YAML keys: eval_dir and evaluation_dir."""
    if not yaml_cfg:
        return yaml_cfg

    if "evaluation_dir" in yaml_cfg and "eval_dir" not in yaml_cfg:
        yaml_cfg["eval_dir"] = yaml_cfg["evaluation_dir"]

    tso_overrides = yaml_cfg.get("tso_overrides")
    if isinstance(tso_overrides, dict):
        for tso_cfg in tso_overrides.values():
            if (
                isinstance(tso_cfg, dict)
                and "evaluation_dir" in tso_cfg
                and "eval_dir" not in tso_cfg
            ):
                tso_cfg["eval_dir"] = tso_cfg["evaluation_dir"]

    return yaml_cfg

# ── Columns that must never be circularly shifted ────────────────────────────
# The shift is a covariate-only augmentation: the target y is left in its
# original temporal position so the model still learns to predict real
# future redispatch.  Calendar columns represent wall-clock time and must
# also remain untouched.  ds and unique_id are Nixtla meta columns.
_NEVER_SHIFT_COLS: frozenset[str] = frozenset(CALENDAR_COLS) | frozenset(
    {"ds", "unique_id", "y"}
)


# ── Shift-K sampling ─────────────────────────────────────────────────────────

def _is_excluded(k: int, exclude_multiples: list[int]) -> bool:
    """Return True if *k* is an integer multiple of any value in *exclude_multiples*."""
    return any(m > 0 and k % m == 0 for m in exclude_multiples)


def sample_shift_k(
    max_shift_days: int,
    exclude_multiples: list[int],
    rng: random.Random,
    min_shift_days: int = 1,
    max_attempts: int = 10_000,
) -> int:
    """
    Sample a random shift K (in days) in [min_shift_days, max_shift_days]
    that is not a multiple of any value in *exclude_multiples*.

    Raises
    ------
    RuntimeError
        If no valid K can be found within *max_attempts* draws.
    """
    if max_shift_days < min_shift_days:
        raise ValueError(
            f"max_shift_days ({max_shift_days}) must be >= min_shift_days ({min_shift_days})"
        )

    # Build candidate list first for small ranges (fast + reproducible)
    candidates = [
        k for k in range(min_shift_days, max_shift_days + 1)
        if not _is_excluded(k, exclude_multiples)
    ]
    if not candidates:
        raise RuntimeError(
            f"No valid shift K in [{min_shift_days}, {max_shift_days}] "
            f"after excluding multiples of {exclude_multiples}."
        )

    return rng.choice(candidates)


# ── Per-column circular shift ─────────────────────────────────────────────────

def _shift_array(vals: np.ndarray, K_hours: int, T_hours: int) -> np.ndarray:
    """
    Pure circular shift:  ``shifted[t] = vals[(t + K_hours) % T_hours]``.

    Index 0 receives ``vals[K_hours % T_hours]`` - a left-rotation by K steps,
    matching the spec  ŷ_t = y_{(t+K) mod T}.  No values are dropped and no
    back-fill is needed – the modular arithmetic wraps around cleanly.

    Parameters
    ----------
    vals : 1-D numpy array (float or int)
    K_hours : shift magnitude in hours (>= 0)
    T_hours : period in hours (> 0).  Indices always land in ``[0, T_hours)``;
              if ``T_hours <= len(vals)`` they index valid positions directly.
    """
    N = len(vals)
    if K_hours <= 0 or N == 0:
        return vals.copy()

    T_eff = min(T_hours, N)          # period cannot exceed series length
    K_eff = K_hours % T_eff          # normalise K into [0, T_eff)

    if K_eff == 0:
        return vals.copy()

    indices = (np.arange(N, dtype=np.intp) + K_eff) % T_eff
    return vals[indices]


# ── Main circular-shift transform ────────────────────────────────────────────

def apply_circular_shift(
    df: pd.DataFrame,
    K_days: int,
    T_hours: int = 0,
) -> pd.DataFrame:
    """
    Circularly shift all covariate columns in a Nixtla-format DataFrame by
    ``K_days × 24`` hours.  The target ``y``, calendar features, ``ds`` and
    ``unique_id`` are never touched.

    This must be called **before** ``prepare_shifted_dataset`` so that the
    subsequent target shift keeps ``y`` and future covariates aligned.

    Feature groups are shifted **jointly** (same K for every column) so
    relative alignment within a group is preserved.  The shift is purely
    circular (no missing values, no back-fill).

    Parameters
    ----------
    df : pd.DataFrame
        Nixtla-format frame with at least ``ds``, ``y``, ``unique_id``.
    K_days : int
        Shift magnitude in days (>= 1).
    T_hours : int
        Period in hours for the modular wrap.  Should equal the training-set
        length so the shift cycles over exactly the data the model trains on.
        0 (default) = use each group's full length.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with covariate columns circularly shifted.
    """
    if K_days <= 0:
        logger.info("apply_circular_shift: K_days=%d <= 0 – returning unchanged.", K_days)
        return df.copy()

    K_hours = K_days * 24

    # Columns that will be shifted (everything except y, meta, and calendar)
    shift_cols = [c for c in df.columns if c not in _NEVER_SHIFT_COLS]
    if not shift_cols:
        logger.warning("apply_circular_shift: no shiftable columns found.")
        return df.copy()

    logger.info(
        "Circular shift: K=%d days (%d hours), T_hours=%s, %d column(s) – "
        "y/calendar/meta untouched",
        K_days, K_hours,
        T_hours if T_hours > 0 else "full-group-length",
        len(shift_cols),
    )

    out = df.copy()
    for uid, grp_idx in df.groupby("unique_id", sort=False).groups.items():
        n_rows = len(grp_idx)
        T_eff = T_hours if T_hours > 0 else n_rows
        for col in shift_cols:
            vals = df.loc[grp_idx, col].to_numpy().copy()
            shifted = _shift_array(vals, K_hours, T_eff)
            out.loc[grp_idx, col] = shifted

    logger.info("Circular shift applied (K=%d days, T_h=%s hours).", K_days, T_hours or "full")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    # ── Two-pass: extract --config before finalising defaults ─────────────────
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_args, _ = _pre.parse_known_args()

    yaml_cfg: dict[str, Any] = {}
    if _pre_args.config:
        yaml_cfg = load_yaml_config(_pre_args.config)
        yaml_cfg = _normalize_eval_dir_keys(yaml_cfg)

    p = argparse.ArgumentParser(
        description=(
            "Circular-shift data-augmentation training pipeline.\n"
            "Shifts all non-calendar covariate columns by K days per rolling window.\n"
            "The shift period T_h is automatically set to each window's training-set\n"
            "length; the target y is never shifted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Config file ───────────────────────────────────────────────────────────
    p.add_argument(
        "--config", default=None,
        help="Path to a YAML config file (same format as train_pipeline.py).",
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--dataset-path", required=True,
        help="Path to the .parquet dataset.",
    )
    p.add_argument(
        "--direction", default="both", choices=["up", "down", "both"],
        help="Direction filter (default: both).",
    )

    # ── Circular shift ────────────────────────────────────────────────────────
    shift_grp = p.add_argument_group(
        "circular shift",
        "Parameters controlling the circular day-shift applied to the raw features.",
    )
    shift_grp.add_argument(
        "--shift-k-days", type=int, default=None,
        metavar="K",
        help=(
            "Number of days to shift (K ≥ 1).  "
            "If omitted the script samples K randomly from "
            "[1, --max-shift-days] (excluding multiples)."
        ),
    )
    shift_grp.add_argument(
        "--max-shift-days", type=int, default=90,
        metavar="MAX_K",
        help=(
            "Upper bound for random K sampling (only used when --shift-k-days is "
            "omitted, default: 90)."
        ),
    )
    shift_grp.add_argument(
        "--min-shift-days", type=int, default=1,
        metavar="MIN_K",
        help="Lower bound for random K sampling (default: 1).",
    )
    shift_grp.add_argument(
        "--exclude-multiples", type=int, nargs="+", default=[],
        metavar="M",
        help=(
            "Exclude shift values K that are an integer multiple of any of these "
            "base periods (in days).  Example: --exclude-multiples 7 365 rejects "
            "K = 7, 14, 21 ... and K = 365, 730, ..."
        ),
    )
    shift_grp.add_argument(
        "--shift-seed", type=int, default=None,
        metavar="SEED",
        help="Random seed for K sampling (default: None = non-deterministic).",
    )

    # ── Shifting / calendar ───────────────────────────────────────────────────
    p.add_argument("--shift-hours", type=int, default=6,
                   help="Target shift in hours for the NeuralForecast pipeline (default: 6).")
    p.add_argument("--date-time", default=None, type=str,
                   help="Optional wandb date_time experiment override.")
    p.add_argument("--n-threads", type=int, default=None,
                   help="Number of CPU threads to use (default: all available).")
    p.add_argument("--no-calendar", action="store_true",
                   help="Skip adding calendar features.")
    p.add_argument("--holidays-path", default=None,
                   help="Override for holidays CSV path.")

    # ── Forecast / model ──────────────────────────────────────────────────────
    p.add_argument("--forecast-horizon", type=int, default=24,
                   help="Forecast horizon h (default: 24).")
    p.add_argument("--input-size", type=int, default=24,
                   help="Input lookback L (default: 24).")
    p.add_argument(
        "--models", nargs="+", default=["nhits", "tft"],
        help=(
            "Models to train.  Accepts space- or comma-separated values.  "
            "Choices: nhits, nbeatsx, tft, tft_quantile, lstm."
        ),
    )

    # ── Date range ────────────────────────────────────────────────────────────
    p.add_argument("--train-start", default="2021-10-01",
                   help="Training start date.")
    p.add_argument("--valid-start", default="2024-10-01",
                   help="Validation start date.")
    p.add_argument("--test-start", default="2025-01-01",
                   help="Test start date.")

    # ── Training hyper-parameters ─────────────────────────────────────────────
    p.add_argument("--max-steps", type=int, default=5000,
                   help="Max training steps (default: 5000).")
    p.add_argument("--val-check-steps", type=int, default=50,
                   help="Validate every N steps (default: 50).")
    p.add_argument("--early-stop-patience", type=int, default=20,
                   help="Early stopping patience steps.")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size (default: 16).")
    p.add_argument("--windows-batch-size", type=int, default=64,
                   help="Windows batch size (default: 64).")
    p.add_argument("--learning-rate", type=float, default=1e-4,
                   help="Learning rate (default: 1e-4).")
    p.add_argument("--random-seed", type=int, default=778,
                   help="Random seed for model training (default: 778).")
    p.add_argument("--local-scaler-type", default="standard",
                   help="Nixtla local scaler (default: standard).")

    # ── Rolling window ────────────────────────────────────────────────────────
    p.add_argument("--rolling-window", action="store_true",
                   help="Use rolling-window training (strongly recommended).")
    p.add_argument("--n-train-months", type=int, default=37,
                   help="Training period in months (default: 37).")
    p.add_argument("--n-valid-months", type=int, default=2,
                   help="Validation period in months (default: 2).")
    p.add_argument("--n-test-months", type=int, default=1,
                   help="Test period in months (default: 1).")
    p.add_argument(
        "--start-window", type=int, default=0,
        help=(
            "Resume rolling-window training from this window index (0-indexed).  "
            "Useful for continuing interrupted runs."
        ),
    )

    # ── Output / logging ──────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="outputs",
                   help="Directory for model artefacts.")
    p.add_argument(
        "--evaluation-dir",
        "--eval-dir",
        dest="eval_dir",
        default=None,
        help=(
            "Directory for saving evaluation predictions "
            "(per-model/per-window + model-level rolling files)."
        ),
    )
    p.add_argument("--wandb-project", default="redispatch-forecasting",
                   help="W&B project name.")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity (team or user).")
    p.add_argument("--persist-checkpoints", action="store_true",
                   help="Persist checkpoints locally in the output directory.")
    p.add_argument("--persist-checkpoints-to-wandb", action="store_true",
                   help="Log checkpoints to W&B artefacts.")
    p.add_argument("--checkpoint-compression", default=None, type=int,
                   help="Optional zstd compression level for checkpoint artefacts.")
    p.add_argument("--checkpoint-compression-n-threads", default=20, type=int,
                   help="Threads for checkpoint compression (default: 20).")
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

    # Attach model-specific params dict (nested YAML sections)
    args.model_params = {k: v for k, v in yaml_cfg.items() if isinstance(v, dict)}
    args._yaml_cfg = yaml_cfg

    args.models = _normalize_models(args.models)
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_n_threads(args.n_threads)

    # ── 1. Resolve K ─────────────────────────────────────────────────────────
    if args.shift_k_days is not None:
        K_days = args.shift_k_days
        if K_days <= 0:
            raise SystemExit("--shift-k-days must be a positive integer.")
        if _is_excluded(K_days, args.exclude_multiples):
            raise SystemExit(
                f"--shift-k-days {K_days} is excluded because it is a multiple "
                f"of one of {args.exclude_multiples}.  Choose a different K."
            )
        logger.info("Using fixed shift K = %d day(s).", K_days)
    else:
        rng = random.Random(args.shift_seed)
        K_days = sample_shift_k(
            max_shift_days=args.max_shift_days,
            exclude_multiples=args.exclude_multiples,
            rng=rng,
            min_shift_days=args.min_shift_days,
        )
        logger.info(
            "Sampled random shift K = %d day(s) "
            "(seed=%s, range=[%d, %d], exclude_multiples=%s).",
            K_days, args.shift_seed, args.min_shift_days,
            args.max_shift_days, args.exclude_multiples,
        )

    # ── 2. Load dataset ───────────────────────────────────────────────────────
    logger.info("Loading dataset from %s", args.dataset_path)
    raw_df, metadata = load_dataset(args.dataset_path)
    logger.info("Raw dataset: %d rows × %d cols", *raw_df.shape)

    # ── 3. Convert to Nixtla format ───────────────────────────────────────────
    logger.info("Converting to Nixtla format (direction=%s)", args.direction)
    nixtla_df = to_nixtla_format(raw_df, direction=args.direction)
    logger.info(
        "Nixtla DataFrame: %d rows, unique_ids=%s",
        len(nixtla_df), nixtla_df["unique_id"].unique().tolist(),
    )

    # ── 4. Apply per-TSO YAML overrides ───────────────────────────────────────
    tso = metadata.get("operator", "")
    apply_tso_overrides(args, args._yaml_cfg, tso, _explicit_cli_keys())

    # ── 5. Prepare shifted dataset (target shift + calendar) ────────────────
    # This is the CLEAN dataset - no circular shift.  It is used:
    #   a) as the base df for the runner (window slicing)
    #   b) as df_for_prediction (test-time inference)
    # The circular covariate shift is applied ONLY to the training rows of
    # each window via the per_window_transform callback (see step 7).
    logger.info("Applying target shift of %d hours", args.shift_hours)
    shifted_df, future_cov, hist_cov = prepare_shifted_dataset(
        nixtla_df,
        shift_hours=args.shift_hours,
        tso=tso,
        add_calendar=not args.no_calendar,
        holidays_path=args.holidays_path,
    )
    logger.info(
        "After target-shifting: %d rows | %d future covariates | %d historical covariates",
        len(shifted_df), len(future_cov), len(hist_cov),
    )

    # Identify which columns in the shifted_df should be circularly permuted.
    # Calendar columns were recomputed on physical target time, so they belong
    # to _NEVER_SHIFT_COLS.  The columns eligible for circular shifting are
    # everything else minus ds, unique_id, and y.
    circ_shift_cols = [
        c for c in shifted_df.columns if c not in _NEVER_SHIFT_COLS
    ]
    logger.info("Columns that will be circularly shifted per window: %s", circ_shift_cols)

    # ── 6. Build per_window_transform ─────────────────────────────────────────
    # The callback receives window_df (train + val) and train_hours.
    # Both the training and validation portions are circularly shifted, but
    # *separately* - each portion permutes only within itself so that no
    # training-period covariate values leak into validation or vice-versa.
    #
    # Why shift validation too?
    #   The model experiences one consistent "shifted-covariate world" during
    #   fitting.  Early stopping evaluates on shifted validation covariates,
    #   which is the same distribution the model trains on.  This gives a
    #   clean placebo test: if the model does well despite covariates being
    #   scrambled, the covariates didn't help.
    #
    # At prediction time the model receives real (unshifted) covariates from
    # the clean base df.  The first L lookback rows come from the unshifted
    # validation period - a deliberate distribution mismatch that measures
    # how much the model relied on temporal covariate patterns.
    K_h = K_days * 24

    def _circ_shift_block(
        out: pd.DataFrame,
        idx: pd.Index,
        K_hours: int,
    ) -> None:
        """Circularly shift `circ_shift_cols` in-place for the given row index."""
        T = len(idx)
        if T == 0:
            return
        K_eff = K_hours % T
        if K_eff == 0:
            return
        # shifted[t] = original[(t + K_eff) % T]
        # Index 0 receives original[K_eff] - a left-rotation by K steps,
        # matching the original specification  ŷ_t = y_{(t+K) mod T}.
        indices = (np.arange(T, dtype=np.intp) + K_eff) % T
        for col in circ_shift_cols:
            vals = out.loc[idx, col].to_numpy()
            out.loc[idx, col] = vals[indices]

    def _per_window_circ_shift(
        window_df: pd.DataFrame,
        train_hours: int,
    ) -> pd.DataFrame:
        """Circularly shift covariates on training and validation *separately*."""
        out = window_df.copy()
        for _uid, grp_idx in out.groupby("unique_id", sort=False).groups.items():
            train_idx = grp_idx[:train_hours]
            val_idx = grp_idx[train_hours:]
            # Shift training rows within training period
            _circ_shift_block(out, train_idx, K_h)
            # Shift validation rows within validation period
            _circ_shift_block(out, val_idx, K_h)
        return out

    # ── 7. Build TrainConfig ──────────────────────────────────────────────────
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
        circ_shift_k_days=K_days,
        circ_shift_T_hours=0,  # T_h is per-window (= train_hours), not a global
    )

    logger.info(
        "TrainConfig: tso=%s, models=%s, K=%d day(s), rolling=%s",
        config.tso, config.models, K_days, config.rolling_window,
    )

    # ── 8. Train ──────────────────────────────────────────────────────────────
    # The main df is the CLEAN target-shifted dataset (no circular shift).
    # The per_window_transform applies the circular shift to each window's
    # training rows only - validation keeps real covariates.
    # df_for_prediction is NOT needed because the base df is already clean.
    if config.rolling_window:
        logger.info("Starting rolling-window training (start_window=%d)", args.start_window)
        train_rolling_windows(
            shifted_df,
            config,
            future_cov,
            hist_cov,
            metadata=metadata,
            df_unshifted=nixtla_df,
            start_from_window=args.start_window,
            per_window_transform=_per_window_circ_shift,
        )
    else:
        logger.info("Starting single-window training")
        train_single_window(
            shifted_df,
            config,
            future_cov,
            hist_cov,
            metadata=metadata,
            df_unshifted=nixtla_df,
            per_window_transform=_per_window_circ_shift,
        )

    logger.info("Circular-shift training complete ✓  (K=%d days)", K_days)


if __name__ == "__main__":
    main()
