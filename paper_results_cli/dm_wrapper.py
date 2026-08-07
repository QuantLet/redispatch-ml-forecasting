"""Safer wrapper around :mod:`covariate_effect.dm` for the ablation analysis.

The original ``compare_models`` / ``pairwise_dm_comparison`` functions in
``covariate_effect/dm.py`` make unrestricted assumptions about column
alignment and swallow errors silently.  This module adds:

* Column-intersection validation (raises when the intersection is empty).
* Explicit logging of dropped or mismatched model columns.
* A ``seed`` column on the returned DataFrame so that multi-seed results can
  be stacked and compared.
* An orchestration function that reproduces the three-variant DM test loop
  from ``circular_shift_ablation.ipynb`` with proper error handling.
"""
from __future__ import annotations

import logging
import warnings
from typing import Literal

import pandas as pd

from covariate_effect.dm import compare_models, show_dm_dominance

from .config import METADATA_COLS, PaperConfig
from .data import filter_to_common_models, load_ablation_predictions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABLATION_LABELS: dict[str, str] = {
    "full_cov":       "full_cov",
    "no_cov":         "no_cov",
    "full_cov_shift": "full_cov_shift",
}

# Triplets of (reference, alternative, label) for the three pairwise tests.
_DM_TRIPLETS: list[tuple[str, str, str]] = [
    ("full_cov",       "no_cov",         "full_vs_no_cov"),
    ("full_cov",       "full_cov_shift",  "full_vs_shift"),
    ("no_cov",         "full_cov_shift",  "no_cov_vs_shift"),
]

# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------


def _model_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in METADATA_COLS]


def _validate_pair_columns(
    ref: pd.DataFrame,
    alt: pd.DataFrame,
    ref_label: str,
    alt_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return *ref* and *alt* restricted to their common model columns.

    Raises
    ------
    ValueError
        When there are no common model columns.
    """
    ref_models = set(_model_cols(ref))
    alt_models = set(_model_cols(alt))
    common = ref_models & alt_models

    if not common:
        raise ValueError(
            f"No common model columns between '{ref_label}' "
            f"({sorted(ref_models)}) and '{alt_label}' ({sorted(alt_models)})."
        )

    dropped_ref = ref_models - common
    dropped_alt = alt_models - common
    if dropped_ref:
        logger.warning(
            "DM validation: dropping from '%s' (not in '%s'): %s",
            ref_label, alt_label, sorted(dropped_ref),
        )
    if dropped_alt:
        logger.warning(
            "DM validation: dropping from '%s' (not in '%s'): %s",
            alt_label, ref_label, sorted(dropped_alt),
        )

    meta_ref = [c for c in ref.columns if c in METADATA_COLS]
    meta_alt = [c for c in alt.columns if c in METADATA_COLS]
    return (
        ref[meta_ref + sorted(common)],
        alt[meta_alt + sorted(common)],
    )


# ---------------------------------------------------------------------------
# Single pairwise DM wrapper
# ---------------------------------------------------------------------------


def run_pairwise_dm(
    ref: pd.DataFrame,
    alt: pd.DataFrame,
    ref_label: str,
    alt_label: str,
    tso_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Run ``compare_models`` on a validated (ref, alt) pair.

    Parameters
    ----------
    ref, alt:
        Wide prediction DataFrames with metadata + model columns.
    ref_label, alt_label:
        Human-readable names used in log messages.
    tso_filter:
        Optional list of TSO names to restrict the comparison to.

    Returns
    -------
    pd.DataFrame
        The DataFrame returned by ``covariate_effect.dm.compare_models``,
        with added columns ``comparison`` (``"ref_vs_alt"`` label) and
        ``ref_label`` / ``alt_label``.
    """
    ref_v, alt_v = _validate_pair_columns(ref, alt, ref_label, alt_label)

    if tso_filter:
        ref_v = ref_v[ref_v["tso"].isin(tso_filter)]
        alt_v = alt_v[alt_v["tso"].isin(tso_filter)]
        if ref_v.empty or alt_v.empty:
            warnings.warn(
                f"After TSO filter {tso_filter}, one of the prediction sets is "
                "empty - skipping this DM comparison.",
                stacklevel=2,
            )
            return pd.DataFrame()

    try:
        result = compare_models(ref_v, alt_v)
    except Exception as exc:
        raise RuntimeError(
            f"compare_models raised an error for '{ref_label}' vs '{alt_label}': {exc}"
        ) from exc

    result["comparison"] = f"{ref_label}_vs_{alt_label}"
    result["ref_label"] = ref_label
    result["alt_label"] = alt_label
    return result


# ---------------------------------------------------------------------------
# Multi-seed ablation orchestration
# ---------------------------------------------------------------------------


def run_ablation_dm_tests(
    config: PaperConfig,
    seeds: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run all three pairwise DM comparisons for each seed.

    Reproduces the ``compare_models_dm_tests`` notebook function but with
    proper validation, logging, and multi-seed support.

    Parameters
    ----------
    config:
        ``PaperConfig`` instance (used for paths and TSO lists).
    seeds:
        Training random seeds.  Defaults to ``config.ablation_seeds``.

    Returns
    -------
    dict
        Keys are the comparison labels (``"full_vs_no_cov"``,
        ``"full_vs_shift"``, ``"no_cov_vs_shift"``).  Values are DataFrames
        with all seeds stacked (each seed adds a ``"seed"`` column).
    """
    if seeds is None:
        seeds = list(config.ablation_seeds)

    all_results: dict[str, list[pd.DataFrame]] = {
        label: [] for _, _, label in _DM_TRIPLETS
    }

    for seed in seeds:
        logger.info("Loading ablation predictions for seed=%d ...", seed)
        try:
            dfs = load_ablation_predictions(config, seed)
        except RuntimeError as exc:
            logger.error("Skipping seed=%d: %s", seed, exc)
            continue

        # Restrict all three sets to their common model columns before testing.
        dfs = filter_to_common_models(dfs)

        for ref_key, alt_key, label in _DM_TRIPLETS:
            logger.info(
                "  seed=%d - DM test: %s vs %s", seed, ref_key, alt_key
            )
            try:
                result = run_pairwise_dm(
                    ref=dfs[ref_key],
                    alt=dfs[alt_key],
                    ref_label=ref_key,
                    alt_label=alt_key,
                )
            except (ValueError, RuntimeError) as exc:
                logger.error(
                    "  seed=%d - DM test %s FAILED: %s", seed, label, exc
                )
                continue

            if not result.empty:
                result["seed"] = seed
                all_results[label].append(result)

    return {
        label: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for label, frames in all_results.items()
    }


# ---------------------------------------------------------------------------
# Dominance aggregation
# ---------------------------------------------------------------------------


def aggregate_dm_dominance(
    dm_results: dict[str, pd.DataFrame],
    target_df: pd.DataFrame,
    config: PaperConfig,
    dominance_direction: Literal["lower", "higher"] = "lower",
) -> pd.DataFrame:
    """Aggregate DM comparison results into a dominance table.

    Calls ``show_dm_dominance`` once per comparison label (across all TSOs)
    using *target_df* to compute sparsity, then stacks all labels.

    Parameters
    ----------
    dm_results:
        Mapping of comparison-label → DM result DataFrame (output of
        :func:`run_pairwise_dm` or a subset of :func:`run_ablation_dm_tests`).
    target_df:
        Full-covariate predictions DataFrame used solely to compute per-unit
        sparsity (must have ``tso``, ``unique_id``, ``y`` columns).
    config:
        ``PaperConfig`` instance (provides ``dm_alpha`` and
        ``dm_sparsity_threshold``).
    dominance_direction:
        ``"lower"`` for MAE (lower is better); ``"higher"`` for F-beta.

    Returns
    -------
    pd.DataFrame
        Stacked dominance table with columns ``comparison``, ``is_sparse``,
        ``success``, ``failure``.
    """
    lower_better = dominance_direction == "lower"
    panels: list[pd.DataFrame] = []

    for label, df in dm_results.items():
        if df.empty:
            logger.warning("No DM results for comparison '%s'; skipping.", label)
            continue

        try:
            dom = show_dm_dominance(
                comparison_df=df,
                target=target_df,
                sparsity_threshold=config.dm_sparsity_threshold,
                alpha=config.dm_alpha,
                lower_better=lower_better,
            )
        except Exception as exc:
            logger.warning(
                "show_dm_dominance failed for comparison '%s': %s", label, exc
            )
            continue

        dom["comparison"] = label
        panels.append(dom)

    if not panels:
        return pd.DataFrame()

    return pd.concat(panels, ignore_index=True)
