"""F-beta bootstrap dominance tables and per-horizon line plot.

The per-horizon run is parallelised over ``n_workers`` processes, matching the
notebook's ``ProcessPoolExecutor`` approach.

Outputs
-------
tables/fbeta_overall.tex
    Overall model dominance and bootstrap acceptance rate.
tables/fbeta_per_regime.tex
    Per-regime model dominance (model × regime matrix).
figures/fbeta_horizon_line.{png,pdf}
    Mean dominance per horizon, one line per model.
figures/fbeta_per_regime_boxplot.{png,pdf}
    Per-regime dominance distribution per model.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from covariate_effect.block_bootstrap_fbeta import run_bootstrap_fbeta_tests
from .descriptive import save_model_colors

from ..config import DEFAULT_MODEL_ORDER, PaperConfig
from ..data import apply_model_ordering, load_predictions, select_models
from ..style import get_model_color, get_model_display_name, no_legend, save_figure
from ..tables import add_index_names, make_percentage_formatter, save_latex_table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core dominance computation
# ---------------------------------------------------------------------------


def _aggregate_model_dominance(test_probs: dict[tuple[str, str], pd.DataFrame]) -> pd.Series:
    """Aggregate per-group model dominance as the mean across all groups."""
    per_group_dominance: list[pd.Series] = []
    for key, superiority in test_probs.items():
        group_dominance = (superiority > 0.5).sum(axis=1) / max(1, superiority.shape[1] - 1)
        group_dominance.name = f"{key[0]}::{key[1]}"
        per_group_dominance.append(group_dominance)

    if not per_group_dominance:
        return pd.Series(dtype=float, name="model_dominance")

    return pd.concat(per_group_dominance, axis=1).mean(axis=1).rename("model_dominance")


def _compute_dominance(
    predictions_df: pd.DataFrame,
    config: PaperConfig,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Call ``run_bootstrap_fbeta_tests`` and return a dominance summary.

    Returns a DataFrame with columns:

    * ``model_dominance`` - fraction of pairwise contests won (bootstrap superiority prob > 0.5)
    * ``bootstrap_max_1sd_acceptance`` - mean bootstrap acceptance rate
    """
    test_probs, acceptance = run_bootstrap_fbeta_tests(
        data=predictions_df,
        eps_acceptance=config.fbeta_eps_acceptance,
        beta=config.fbeta_beta,
        show_progress=show_progress,
    )
    dominance = _aggregate_model_dominance(test_probs)
    acceptance_mean = (
        acceptance.mean(axis=0, numeric_only=True)
        .rename("bootstrap_max_1sd_acceptance")
    )
    acceptance_mean.index = acceptance_mean.index.str.replace("_delta", "")
    concatenated = pd.concat([dominance, acceptance_mean], axis=1)
    concatenated.index.name = "models"
    return concatenated

def _rename_columns(latex: str, df: pd.DataFrame) -> str:
    column_mapping = {
        "models": "Model",
        "model_dominance": "Model dominance",
        "high_pv": "High PV",
        "pv_distribution": "PV-DSO",
        "mixed_pv_wind": r"Mixed PV/Wind",
        "winter_reversion": "Winter wind",
    }
    latex = add_index_names(latex, df)
    for initial, new in column_mapping.items():
        latex = latex.replace(initial, new)
    return latex



# ---------------------------------------------------------------------------
# Overall table
# ---------------------------------------------------------------------------


def run_fbeta_overall(predictions_df: pd.DataFrame, config: PaperConfig) -> pd.DataFrame:
    """Compute and save the overall-sample F-beta dominance table."""
    logger.info("  F-beta - overall sample ...")
    df = _compute_dominance(predictions_df, config)
    df_filtered = select_models(df, config.models, by_index=True)
    df_sorted = apply_model_ordering(df_filtered, by_index=True)
    df_renamed: pd.DataFrame = df_sorted.rename(index=get_model_display_name)
    save_latex_table(
        df_renamed[["model_dominance"]],
        config.tables_dir / "fbeta_dominance_overall.tex",
        caption=(
            rf"F$_{{\beta={config.fbeta_beta:.0f}}}$ bootstrap model dominance "
            rf"($\varepsilon={config.fbeta_eps_acceptance}$)"
        ),
        formatters={"model_dominance": make_percentage_formatter(2)},
        final_form_callback=_rename_columns,
    )
    return df


def run_fbeta_overall_unfiltered(predictions_df: pd.DataFrame, config: PaperConfig) -> pd.DataFrame:
    """Compute and save the overall-sample F-beta dominance table."""
    logger.info("  F-beta - overall sample ...")
    df = _compute_dominance(predictions_df, config)
    df_sorted = apply_model_ordering(df, by_index=True)
    df_renamed: pd.DataFrame = df_sorted.rename(index=get_model_display_name)
    save_latex_table(
        df_renamed[["model_dominance"]],
        config.tables_dir / "fbeta_dominance_overall_unfiltered.tex",
        caption=(
            rf"F$_{{\beta={config.fbeta_beta:.0f}}}$ bootstrap model dominance "
            rf"($\varepsilon={config.fbeta_eps_acceptance}$)"
        ),
        formatters={"model_dominance": make_percentage_formatter(2)},
        final_form_callback=_rename_columns,
    )
    return df


# ---------------------------------------------------------------------------
# Per-regime table
# ---------------------------------------------------------------------------


def run_fbeta_per_regime(predictions_df: pd.DataFrame, config: PaperConfig) -> pd.DataFrame:
    """Compute and save the per-regime F-beta dominance table."""
    logger.info("  F-beta - per regime ...")
    frames: list[pd.DataFrame] = []
    with tqdm(total=len(config.stress_regimes), desc="F-beta per regime", leave=False) as pbar:
        for label, (regime_start, regime_end) in config.stress_regimes.items():
            regime_df = predictions_df[
                (predictions_df["ds"] >= regime_start)
                & (predictions_df["ds"] < regime_end)
            ].copy()
            if regime_df.empty:
                logger.warning("  F-beta: no data for regime '%s'; skipping.", label)
                continue
            regime_dominance = _compute_dominance(regime_df, config, show_progress=False)
            regime_dominance_filtered = select_models(regime_dominance, config.models, by_index=True)
            regime_dominance_sorted = apply_model_ordering(regime_dominance_filtered, by_index=True)
            frames.append(regime_dominance_sorted[["model_dominance"]].rename(columns={"model_dominance": label}))
            pbar.update(1)

    if not frames:
        logger.warning("  F-beta per-regime: no regimes produced data.")
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined = combined.rename(index=get_model_display_name)
    save_latex_table(
        combined,
        config.tables_dir / "fbeta_dominance_per_regime.tex",
        caption=rf"Per-regime F$_{{\beta={config.fbeta_beta:.0f}}}$ model dominance",
        formatters={col: make_percentage_formatter(2) for col in combined.columns},
        final_form_callback=_rename_columns,
    )
    return combined


def run_fbeta_per_regime_boxplot(
    per_regime_dominance_df: pd.DataFrame,
    config: PaperConfig,
) -> None:
    """Save a per-regime F-beta dominance boxplot using precomputed results."""
    logger.info("  F-beta - per regime boxplot ...")
    if per_regime_dominance_df.empty:
        logger.warning("  F-beta per-regime boxplot: empty input; skipping.")
        return

    regime_order = [
        regime for regime in config.stress_regimes
        if regime in per_regime_dominance_df.columns
    ]
    if not regime_order:
        logger.warning("  F-beta per-regime boxplot: no regime columns; skipping.")
        return

    plot_df = (
        per_regime_dominance_df
        .reset_index(names="model")
        .melt(
            id_vars=["model"],
            value_vars=regime_order,
            var_name="regime",
            value_name="model_dominance",
        )
        .dropna(subset=["model_dominance"])
    )
    if plot_df.empty:
        logger.warning("  F-beta per-regime boxplot: no rows after melt; skipping.")
        return

    model_order = [
        get_model_display_name(str(model))
        for model in DEFAULT_MODEL_ORDER
        if get_model_display_name(str(model)) in set(plot_df["model"])
    ]
    if model_order:
        plot_df["model"] = pd.Categorical(
            plot_df["model"],
            categories=model_order,
            ordered=True,
        )
    plot_df["regime"] = pd.Categorical(
        plot_df["regime"],
        categories=regime_order,
        ordered=True,
    )

    cmap = plt.get_cmap("Spectral")
    color_positions = np.linspace(0.15, 0.85, num=len(regime_order))
    regime_colors = {
        regime: cmap(float(pos))
        for regime, pos in zip(regime_order, color_positions)
    }

    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="model",
        y="model_dominance",
        hue="regime",
        data=plot_df,
        palette=regime_colors,
    )
    plt.xlabel("")
    plt.ylabel(rf"F$_{{\beta={config.fbeta_beta:.0f}}}$ dominance")
    no_legend(fig.axes[0])
    save_figure(
        fig,
        config.figures_dir / "fbeta_per_regime_boxplot",
        fmt=config.figure_format,
    )
    save_model_colors(config, "fbeta_per_regime_boxplot_colors.tex", regime_colors)


# ---------------------------------------------------------------------------
# Per-horizon (parallelised)
# ---------------------------------------------------------------------------


def _fbeta_single_horizon(
    horizon: int,
    horizon_df: pd.DataFrame,
    eps_acceptance: float,
    fbeta_beta: float,
) -> pd.DataFrame:
    """Worker function - runs the bootstrap for a single horizon."""
    test_probs, _ = run_bootstrap_fbeta_tests(
        data=horizon_df,
        eps_acceptance=eps_acceptance,
        beta=fbeta_beta,
        show_progress=False,
    )
    dominance = _aggregate_model_dominance(test_probs)
    return dominance.to_frame().assign(horizon=horizon)


def run_fbeta_horizon_plots(
    predictions_df: pd.DataFrame,
    config: PaperConfig,
    n_workers: int = 4,
) -> None:
    """Run per-horizon F-beta bootstrap (parallel) and save a line-plot figure."""
    logger.info(
        "  F-beta - per horizon line plot (%d parallel workers) ...", n_workers
    )
    horizon_groups = [
        (int(h), grp.copy())
        for h, grp in predictions_df.groupby("horizon", sort=True)
    ]

    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _fbeta_single_horizon,
                h,
                df,
                config.fbeta_eps_acceptance,
                config.fbeta_beta,
            ): h
            for h, df in horizon_groups
        }
        with tqdm(total=len(futures), desc="F-beta per horizon", leave=False) as pbar:
            for fut in as_completed(futures):
                h = futures[fut]
                try:
                    horizon_result = fut.result()
                    horizon_result_filtered = select_models(horizon_result, config.models, by_index=True)
                    horizon_result_sorted: pd.DataFrame = apply_model_ordering(horizon_result_filtered, by_index=True)
                    frames.append(horizon_result_sorted)
                except Exception as exc:
                    logger.error(
                        "  F-beta per-horizon failed for horizon=%d: %s", h, exc
                    )
                finally:
                    pbar.update(1)

    if not frames:
        logger.warning("  F-beta per-horizon: no results; skipping figure.")
        return

    horizon_df = (
        pd.concat(frames, axis=0, ignore_index=False)
        .reset_index()
        .rename(columns={"index": "model"})
    )

    # Pivot the data to create a heatmap: rows=model, columns=horizon, values=model_dominance
    heatmap_data = horizon_df.pivot(index="model", columns="horizon", values="model_dominance")
    heatmap_data = heatmap_data.rename(index=get_model_display_name)

    if heatmap_data.empty:
        logger.warning("  F-beta per-horizon: empty after pivot; skipping heatmap.")
        return

    fig, ax = plt.subplots(figsize=(12, max(4, heatmap_data.shape[0] * 0.7)))
    im = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        cmap="coolwarm",
    )
    ax.set_xticks(range(heatmap_data.shape[1]))
    ax.set_xticklabels([str(h) for h in heatmap_data.columns])
    ax.set_yticks(range(heatmap_data.shape[0]))
    ax.set_yticklabels(list(heatmap_data.index))
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Model")
    plt.colorbar(im, ax=ax, label=rf"F$_{{\beta={config.fbeta_beta:.0f}}}$ dominance")
    no_legend(ax)
    save_figure(
        fig,
        config.figures_dir / "fbeta_horizon_heatmap",
        fmt=config.figure_format,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for model, grp in horizon_df.groupby("model"):
        grp = grp.sort_values("horizon")
        ax.plot(
            grp["horizon"],
            grp["model_dominance"],
            color=get_model_color(str(model)),
            label=str(model),
        )
    tick_positions = np.arange(1, 25)
    tick_labels = [str(h) if (h == 1 or h % 5 == 0 or h % 24 == 0) else "" for h in tick_positions]
    ax.set_xlabel("Horizon")
    ax.set_xlim(1, 24)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(rf"F$_{{\beta={config.fbeta_beta:.0f}}}$ dominance")
    no_legend(ax)
    save_figure(
        fig,
        config.figures_dir / "fbeta_horizon_line",
        fmt=config.figure_format,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(config: PaperConfig) -> None:
    """Run all F-beta outputs and write them to *config.output_dir*."""
    predictions_df = load_predictions(config, apply_model_selection=False)
    run_fbeta_overall(predictions_df, config)
    run_fbeta_overall_unfiltered(predictions_df, config)
    per_regime_dominance = run_fbeta_per_regime(predictions_df, config)
    run_fbeta_per_regime_boxplot(per_regime_dominance, config)
    run_fbeta_horizon_plots(predictions_df, config, n_workers=7)