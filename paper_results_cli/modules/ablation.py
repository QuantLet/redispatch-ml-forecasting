"""Circular-shift ablation: DM dominance table across seeds and comparisons.

For each seed the three prediction variants are loaded:

* **full_cov** - full-covariate model (``outputs/``)
* **no_cov** - no-covariate model (``outputs_no_covariates/``)
* **full_cov_shift** - full-covariate model trained on circularly-shifted
  targets (``outputs_shifted_targets_17/``)

Three pairwise DM comparisons are run (matching the notebook triplets):

1. ``shift_vs_full_cov`` - full_cov_shift (ref) vs full_cov (alt)
2. ``no_cov_vs_full_cov`` - no_cov (ref) vs full_cov (alt)
3. ``shift_vs_no_cov`` - full_cov_shift (ref) vs no_cov (alt)

``show_dm_dominance`` is called with the full-covariate predictions as
the sparsity reference.

Outputs
-------
tables/ablation_dm_dominance.tex
    Dominance counts (success / failure) broken down by comparison ×
    sparsity class × seed.
"""
from __future__ import annotations

import logging

import pandas as pd

from covariate_effect.dm import show_dm_dominance

from ..config import PaperConfig
from ..data import filter_to_common_models, load_ablation_predictions
from ..dm_wrapper import run_pairwise_dm
from ..tables import add_index_names, make_percentage_formatter, save_latex_table

logger = logging.getLogger(__name__)

# Triplets: (ref_key, alt_key, output_label)
_TRIPLETS: list[tuple[str, str, str]] = [
    ("full_cov_shift", "full_cov",       "shift_vs_full_cov"),
    ("no_cov",         "full_cov",       "no_cov_vs_full_cov"),
    ("no_cov",         "full_cov_shift", "no_cov_vs_shift"),
]

# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def _rename_index_levels(latex: str, df: pd.DataFrame) -> str:
    index_names_mapping = {
        "comparison": "Comparison",
        "is_sparse": "Type",
    }
    latex = add_index_names(latex, df)
    for current, new in index_names_mapping.items():
        latex = latex.replace(current, new)
    return latex


def _inject_multiindex_group_header(latex: str, df: pd.DataFrame) -> str:
    """Insert a grouped header row for tables with a MultiIndex row index."""
    latex = _rename_index_levels(latex, df)
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels <= 1:
        left_span = 0
    else:
        left_span = df.index.nlevels

    right_span_half = len(df.columns) // 2
    if right_span_half <= 0:
        return latex


    seed_values = df.columns.get_level_values("seed").unique()
    total_span = left_span + right_span_half * 2
    if left_span == 0:
        total_span += 1
    group_header = (
        (
            f"\\multicolumn{{{left_span}}}{{c}}{{Comparison details}} & "
            if left_span > 0 else "& "
        )
        + (
            f"\\multicolumn{{{right_span_half}}}{{c}}{{{seed_values[0]}}} & "
            f"\\multicolumn{{{right_span_half}}}{{c}}{{{seed_values[1]}}} \\\\"
        )
    )
    group_rules = (
        f"\\cmidrule(lr){{1-{left_span}}}"
        f"\\cmidrule(lr){{{left_span + 1}-{left_span + right_span_half}}}"
        f"\\cmidrule(lr){{{left_span + right_span_half + 1}-{total_span}}}"
        if left_span > 0 else ""
        f"\\cmidrule(lr){{2-{1 + right_span_half}}}"
        f"\\cmidrule(lr){{{2 + right_span_half}-{total_span}}}"
    )

    lines = latex.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == r"\toprule":
            # First remove the seed header line, which will be replaced by the group header
            lines.pop(i + 1)
            lines.insert(i + 1, group_header)
            lines.insert(i + 2, group_rules)
            out = "\n".join(lines)
            if latex.endswith("\n"):
                out += "\n"
            return out
    return latex


# ---------------------------------------------------------------------------
# Per-seed computation
# ---------------------------------------------------------------------------


def _run_for_seed(config: PaperConfig, seed: int) -> pd.DataFrame | None:
    """Run all DM comparisons for *seed* and return a dominance DataFrame.

    Returns ``None`` if predictions cannot be loaded or no comparisons succeed.

    The returned DataFrame has columns ``[is_sparse, success, failure,
    comparison]`` and is ready for multi-seed stacking.
    """
    logger.info("  Ablation - seed %d: loading predictions ...", seed)
    try:
        dfs = load_ablation_predictions(config, seed)
    except RuntimeError as exc:
        logger.error("  Ablation - seed %d: SKIPPED (%s)", seed, exc)
        return None

    dfs = filter_to_common_models(dfs)
    target_df = dfs["full_cov"]  # used only for sparsity computation

    frames: list[pd.DataFrame] = []
    for ref_key, alt_key, label in _TRIPLETS:
        logger.info(
            "  Ablation - seed %d: DM %s vs %s ...", seed, ref_key, alt_key
        )
        try:
            comp_df = run_pairwise_dm(
                ref=dfs[ref_key],
                alt=dfs[alt_key],
                ref_label=ref_key,
                alt_label=alt_key,
            )
        except (ValueError, RuntimeError) as exc:
            logger.error(
                "  Ablation - seed %d: DM failed for '%s': %s", seed, label, exc
            )
            continue

        if comp_df.empty:
            logger.warning(
                "  Ablation - seed %d: empty DM result for '%s'.", seed, label
            )
            continue

        try:
            dom = show_dm_dominance(
                comparison_df=comp_df,
                target=target_df,
                sparsity_threshold=config.dm_sparsity_threshold,
                alpha=config.dm_alpha,
                lower_better=False, # the way triplets are constructed, first model is always the "reference" and should be worse (higher values) if DM success is to be counted
            )
        except Exception as exc:
            logger.error(
                "  Ablation - seed %d: show_dm_dominance failed for '%s': %s",
                seed,
                label,
                exc,
            )
            continue

        frames.append(dom.assign(comparison=label))

    if not frames:
        logger.warning("  Ablation - seed %d: no comparisons succeeded.", seed)
        return None

    return (
        pd.concat(frames, ignore_index=True)
        .drop(columns="model")
        .set_index(["comparison", "is_sparse"])
        .unstack(level=1)
        .assign(seed=seed)
    )


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------


def _aggregate_seeds(seed_results: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Stack per-seed dominance DataFrames into the final layout."""
    combined = pd.concat(seed_results, ignore_index=False).reset_index()
    combined["comparison"] = combined["comparison"].map({
        "shift_vs_full_cov": "Shift vs Full Cov",
        "no_cov_vs_full_cov": "No Cov vs Full Cov",
        "no_cov_vs_shift": "No Cov vs Shift",
    })
    combined["seed"] = combined["seed"].map(lambda s: f"Seed {s}")

    combined_processed = (
        combined.set_index(["comparison", "seed"])
        .unstack(level=1)
        .stack(level=1)
        .swaplevel(0, 1, axis=1)
        .sort_index(ascending=False, axis=1)
    )

    combined_favorable = combined_processed.loc[
        ["Shift vs Full Cov", "No Cov vs Full Cov"], :
    ].copy()
    combined_neutral = combined_processed.loc[
        ["No Cov vs Shift"], :
    ].copy()

    return {"All": combined_processed, "Favorable": combined_favorable, "Neutral": combined_neutral}



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(config: PaperConfig) -> None:
    """Run the ablation DM dominance analysis and write the LaTeX table."""
    seed_results: list[pd.DataFrame] = []
    for seed in config.ablation_seeds:
        result = _run_for_seed(config, seed)
        if result is not None:
            seed_results.append(result)

    if not seed_results:
        logger.warning(
            "  Ablation: no seeds produced results; skipping table save."
        )
        return

    all_seeds_comparison_dict = _aggregate_seeds(seed_results)

    for comparison, df in all_seeds_comparison_dict.items():
        logger.info(
            "  Ablation - comparison '%s': %d seeds included in table.",
            comparison,
            df.shape[1] // 2,
        )
        if comparison == "All":
            caption = "DM dominance counts (success / failure) for all comparisons and seeds."
        if comparison == "Favorable":
            caption = (
                "DM dominance counts (success / failure) for comparisons "
                "where the first model is expected to outperform the second."
            )
        else:
            caption = (
                "DM dominance counts (success / failure) for comparisons "
                "where no clear dominance is expected."
            )
        if comparison == "Neutral":
            # Only one comparison in this category, so we can drop the "comparison" index level for a cleaner table
            df = df.droplevel(0, axis=0)
        save_latex_table(
            df,
            config.tables_dir / f"ablation_dm_dominance_{comparison.lower()}.tex",
            caption=caption,
            formatters={col: make_percentage_formatter(2) for col in df.columns},
            final_form_callback=_inject_multiindex_group_header,
        )
    logger.info("  Ablation - done (%d seeds).", len(seed_results))
