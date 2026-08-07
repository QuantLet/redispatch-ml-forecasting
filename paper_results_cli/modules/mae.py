"""MAE and F-beta evaluation tables, per-horizon heatmap, and line plot.

Outputs
-------
tables/mae_overall_mae.tex
    Overall MAE per model and TSO direction (best value bolded per row).
tables/mae_overall_fbeta_2_0.tex
    Same layout for F-beta (β = 2).
tables/mae_per_regime_mae.tex
    Per-regime MAE (model × regime columns, index = TSO × direction).
tables/mae_per_regime_fbeta_2_0.tex
    Per-regime F-beta.
figures/mae_diff_heatmap.{png,pdf}
    Per-horizon MAE difference between two configurable models
    (using the diverging ``mae_diff`` colormap).
figures/mae_horizon_line.{png,pdf}
    Mean MAE across TSO/direction pairs, one line per model.
"""
from __future__ import annotations

import logging
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from covariate_effect.evaluate import perform_tso_evaluations_rolling_window
from .descriptive import save_model_colors

from ..config import BENCHMARK_MODELS, DEFAULT_MODEL_ORDER, DEFAULT_MODELS, PaperConfig
from ..data import apply_model_ordering, load_predictions, select_models
from ..style import get_model_color, get_model_display_name, no_legend, save_figure
from ..tables import (
    METRIC_HIGHER_BETTER,
    METRIC_PRECISION,
    add_index_names,
    save_latex_table,
)

logger = logging.getLogger(__name__)

_EVAL_META = frozenset(
    {"tso", "unique_id", "metric", "sparsity", "merge_key", "horizon", "regime", "volume"}
)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _run_evaluations(
    df: pd.DataFrame,
    config: PaperConfig,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Call ``perform_tso_evaluations_rolling_window``."""
    return perform_tso_evaluations_rolling_window(
        df,
        beta=config.fbeta_beta,
        start_date=start_date,
        end_date=end_date,
    )


def _run_evaluations_per_horizon(
    df_with_baselines: pd.DataFrame,
    config: PaperConfig,
) -> pd.DataFrame:
    """Loop over every unique horizon and collect evaluations."""
    frames: list[pd.DataFrame] = []
    for h in df_with_baselines["horizon"].unique():
        h_df = df_with_baselines[df_with_baselines["horizon"] == h]
        ev = perform_tso_evaluations_rolling_window(
            h_df,
            beta=config.fbeta_beta,
        ).assign(horizon=int(h))
        frames.append(ev)
    return pd.concat(frames, axis=0)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def _bold_best_row(row: pd.Series, model_cols: list[str], higher: bool, prec: int) -> pd.Series:
    """Return *row* with the best-model cell wrapped in ``\\textbf{}``."""
    vals = row[model_cols].astype(float)
    best_col = vals.idxmax() if higher else vals.idxmin()
    row = row.copy().astype(object)
    for c in model_cols:
        v = f"{float(row[c]):.{prec}f}"
        row[c] = rf"\textbf{{{v}}}" if c == best_col else v
    return row


def _format_metric_table(eval_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot to a wide model × direction table with best cell bolded."""
    sub = eval_df[eval_df["metric"] == metric].copy()
    model_cols = [c for c in sub.columns if c not in _EVAL_META]
    if not model_cols:
        return pd.DataFrame()
    higher = METRIC_HIGHER_BETTER.get(metric, False)
    prec = METRIC_PRECISION.get(metric, 3)
    for c in model_cols:
        sub[c] = sub[c].astype(float)
    formatted = sub.apply(
        _bold_best_row, axis=1, model_cols=model_cols, higher=higher, prec=prec
    )
    formatted["tso"] = formatted["tso"].str.replace(r"_", " ", regex=False)
    out = formatted[["tso", "unique_id"] + model_cols].set_index(["tso", "unique_id"])
    out: pd.DataFrame = apply_model_ordering(out, by_index=False)
    out = out.rename(columns=lambda c: get_model_display_name(str(c)))
    return out


def get_metric_name_for_tabel(metric_name: str, config: PaperConfig) -> str:
    if metric_name == "mae":
        return "MAE"
    elif "fbeta" in metric_name:
        return rf"$F_{{\beta={config.fbeta_beta:.0f}}}$"
    elif metric_name == "mae_conditional":
        return "MAE | y > 0"
    elif metric_name == "r2_score":
        return r"$R^2$"
    else:
        return metric_name.upper()


def _metric_label_order(config: PaperConfig) -> list[str]:
    """Return metric labels in the same order as ``config.mae_metrics``."""
    labels: list[str] = []
    for metric in config.mae_metrics:
        metric_label = get_metric_name_for_tabel(metric.replace(".", "_"), config)
        if metric_label not in labels:
            labels.append(metric_label)
    return labels


def _sort_eval_index(
    df: pd.DataFrame,
    metric_order: list[str],
) -> pd.DataFrame:
    """Sort ``(tso, metric, unique_id)`` index with custom metric ordering."""
    if df.empty or not isinstance(df.index, pd.MultiIndex):
        return df

    level_names = list(df.index.names)
    if "metric" not in level_names:
        return df.sort_index()

    index_frame = df.index.to_frame(index=False)
    index_frame["metric"] = pd.Categorical(
        index_frame["metric"],
        categories=metric_order,
        ordered=True,
    )
    sorted_frame = index_frame.sort_values(level_names, kind="stable")
    sorted_index = pd.MultiIndex.from_frame(sorted_frame[level_names])
    return df.reindex(sorted_index)


def _rename_index_levels(latex: str, df: pd.DataFrame) -> str:
    index_names_mapping = {
        "tso": "TSO",
        "metric": "Metric",
        "unique_id": "Direction",
        "models": "Model",
    }
    latex = add_index_names(latex, df)
    for current, new in index_names_mapping.items():
        latex = latex.replace(current, new)
    return latex


def _inject_multiindex_group_header(latex: str, df: pd.DataFrame) -> str:
    """Insert a grouped header row for tables with a MultiIndex row index."""
    latex = _rename_index_levels(latex, df)
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels <= 1:
        return latex
    if "Dataset details" in latex:
        return latex

    left_span = df.index.nlevels
    right_span = len(df.columns)
    if right_span <= 0:
        return latex

    total_span = left_span + right_span
    group_header = (
        f"\\multicolumn{{{left_span}}}{{c}}{{Dataset details}} & "
        f"\\multicolumn{{{right_span}}}{{c}}{{Models}} \\\\"
    )
    group_rules = (
        f"\\cmidrule(lr){{1-{left_span}}}"
        f"\\cmidrule(lr){{{left_span + 1}-{total_span}}}"
    )

    lines = latex.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == r"\toprule":
            lines.insert(i + 1, group_header)
            lines.insert(i + 2, group_rules)
            out = "\n".join(lines)
            if latex.endswith("\n"):
                out += "\n"
            return out
    return latex


# ---------------------------------------------------------------------------
# Overall tables
# ---------------------------------------------------------------------------


def run_mae_overall(predictions_df: pd.DataFrame, config: PaperConfig, file_name: str, model_subset: list[str] = []) -> pd.DataFrame:
    """Run full-sample evaluation and save MAE + F-beta tables."""
    logger.info("  MAE - overall sample ...")
    if model_subset:
        predictions_df = select_models(predictions_df, model_subset, by_index=False)
    eval_df = _run_evaluations(predictions_df, config, start_date=config.start_date)
    combined_metrics = []
    for metric in config.mae_metrics:
        formatted = _format_metric_table(eval_df, metric)
        if formatted.empty:
            logger.warning("  MAE overall: no rows for metric '%s'.", metric)
            continue
        processed_metric = metric.replace(".", "_")
        metric_label = get_metric_name_for_tabel(processed_metric, config)
        combined_metrics.append(formatted.assign(metric=metric_label))
    
    metric_order = _metric_label_order(config)
    combined_metrics_df = pd.concat(combined_metrics, axis=0).reset_index().set_index([
        "tso", "unique_id", "metric"
    ]).swaplevel(1, 2, axis=0)
    combined_metrics_df = _sort_eval_index(combined_metrics_df, metric_order)
    save_latex_table(
        combined_metrics_df,
        config.tables_dir / file_name,
        caption="Out-of-sample evaluations per model, whole period (MAE and F-beta). Best value(s) per row are bolded.",
        final_form_callback=_inject_multiindex_group_header,
    )
    return eval_df


def run_mae_sparsity_table(predictions_df: pd.DataFrame, config: PaperConfig, file_name: str, model_subset: list[str] = []) -> pd.DataFrame:
    """Run full-sample evaluation and compute dense/sparse MAE and F-beta tables."""

    def _aggregate_by_sparsity(
        sub: pd.DataFrame,
        *,
        weighted: bool,
    ) -> pd.DataFrame:
        model_cols = [c for c in sub.columns if c not in {"volume", "sparsity"}]
        if not model_cols:
            return pd.DataFrame()

        grouped = sub.assign(
            sparsity_class=np.where(sub["sparsity"] >= config.dm_sparsity_threshold, "Sparse", "Dense")
        )
        if weighted:
            aggregated = grouped.groupby("sparsity_class")[["volume"] + model_cols].apply(
                lambda group: group[model_cols].multiply(group["volume"], axis=0).sum() / group["volume"].sum()
            )
        else:
            aggregated = grouped.groupby("sparsity_class")[model_cols].mean()

        aggregated = aggregated.reindex(["Dense", "Sparse"])
        aggregated = aggregated.T
        aggregated.index.name = "Model"
        aggregated.columns.name = None
        aggregated = apply_model_ordering(aggregated, by_index=True)
        aggregated = aggregated.rename(index=get_model_display_name)
        return aggregated

    if model_subset:
        predictions_df = select_models(predictions_df, model_subset, by_index=False)
    eval_df = _run_evaluations(predictions_df, config, start_date=config.start_date)
    model_cols = [c for c in eval_df.columns if c not in _EVAL_META]
    mapped_model_cols = [get_model_display_name(str(c)) for c in model_cols]

    mae_results = eval_df.loc[
        eval_df["metric"] == "mae", ["volume", "sparsity"] + model_cols
    ].copy()
    mae_for_table = _aggregate_by_sparsity(mae_results, weighted=True)
    mae_for_table_formatted = mae_for_table.apply(
        _bold_best_row, axis=0, model_cols=mapped_model_cols, higher=False, prec=3
    )

    fbeta_metric = f"fbeta_{config.fbeta_beta}"
    fbeta_results = eval_df.loc[
        eval_df["metric"] == fbeta_metric, ["sparsity"] + model_cols
    ].copy()
    fbeta_for_table = _aggregate_by_sparsity(fbeta_results, weighted=False)
    fbeta_for_table_formatted = fbeta_for_table.apply(
        _bold_best_row, axis=0, model_cols=mapped_model_cols, higher=True, prec=3
    )

    r2_metric = "r2_score"
    r2_results = eval_df.loc[
        eval_df["metric"] == r2_metric, ["sparsity"] + model_cols
    ].copy()
    r2_for_table = _aggregate_by_sparsity(r2_results, weighted=False)
    r2_for_table_formatted = r2_for_table.apply(
        _bold_best_row, axis=0, model_cols=mapped_model_cols, higher=True, prec=2
    )

    table_column_index = pd.MultiIndex.from_product(
        [["MAE", rf"F$_{{\beta={config.fbeta_beta:.0f}}}$", r"$R^2$"], ["Dense", "Sparse"]],
        names=["Metric", "Sparsity"],
    )

    combined_metrics_df_formatted = pd.concat(
        [mae_for_table_formatted, fbeta_for_table_formatted, r2_for_table_formatted],
        axis=1,
    ).set_axis(table_column_index, axis=1)
    combined_metrics_df_formatted.index.name = "Model"

    save_latex_table(
        combined_metrics_df_formatted,
        config.tables_dir / file_name,
        caption=(
            "Volume-weighted MAE, average F-beta, and $R^2$ per model, grouped by "
            f"dense/sparse class using a sparsity threshold of {config.dm_sparsity_threshold:.2f}. "
            "Best value(s) per row are bolded."
        ),
    )
    return eval_df


# ---------------------------------------------------------------------------
# Per-regime tables
# ---------------------------------------------------------------------------


def run_mae_per_regime(predictions_df: pd.DataFrame, config: PaperConfig) -> pd.DataFrame:
    """Run per-regime evaluations and save MAE + F-beta regime tables."""
    logger.info("  MAE - per regime ...")
    frames: list[pd.DataFrame] = []
    for label, (regime_start, regime_end) in config.stress_regimes.items():
        ev = _run_evaluations(
            predictions_df,
            config,
            start_date=regime_start,
            end_date=regime_end,
        ).assign(regime=label)
        frames.append(ev)

    combined = pd.concat(frames, axis=0)

    metric_order = _metric_label_order(config)
    for regime in combined["regime"].unique():
        combined_metrics = []
        for metric in config.mae_metrics:
            regime_eval = combined[np.logical_and(
                combined["metric"] == metric,
                combined["regime"] == regime
            )].copy()

            formatted = _format_metric_table(regime_eval, metric)
            if formatted.empty:
                continue

            processed_metric = metric.replace(".", "_")
            metric_label = get_metric_name_for_tabel(processed_metric, config)

            combined_metrics.append(formatted.assign(metric=metric_label))

        if not combined_metrics:
            logger.warning("  MAE per regime: no model columns for regime '%s'; skipping.", regime)
            continue
        regime_metric = pd.concat(combined_metrics, axis=0).reset_index().set_index([
            "tso", "unique_id", "metric"
        ]).swaplevel(1, 2, axis=0)
        regime_metric = _sort_eval_index(regime_metric, metric_order)
        save_latex_table(
            regime_metric,
            config.tables_dir / f"per_regime_{regime}_eval.tex",
            caption=f"Out-of-sample evaluations per model, regime '{regime}'. Best value(s) per row are bolded.",
            final_form_callback=_inject_multiindex_group_header,
        )

    return combined


# ---------------------------------------------------------------------------
# Per-horizon MAE difference heatmap
# ---------------------------------------------------------------------------


def run_mae_diff_heatmap(
    df_with_baselines: pd.DataFrame,
    config: PaperConfig,
) -> None:
    """Compute per-horizon MAE difference and save a diverging heatmap.

    The two models being compared are taken from ``config.mae_diff_models``.
    The colormap ``mae_diff`` (red=negative, white=zero, green=positive) is
    registered by :func:`paper_results_cli.style.apply_paper_style`.
    """
    model_a, model_b = config.mae_diff_models
    available = set(df_with_baselines.columns)
    if model_a not in available or model_b not in available:
        logger.warning(
            "  MAE diff heatmap: model(s) not in predictions ('%s', '%s'); "
            "skipping. Available: %s",
            model_a,
            model_b,
            sorted(available - _EVAL_META),
        )
        return

    logger.info("  MAE - diff heatmap (%s − %s) ...", model_a, model_b)
    per_horizon = _run_evaluations_per_horizon(df_with_baselines, config)
    per_horizon_mae = per_horizon[per_horizon["metric"] == "mae"].copy()
    per_horizon_mae["diff"] = (
        per_horizon_mae[model_a].astype(float)
        - per_horizon_mae[model_b].astype(float)
    )

    heatmap_data = per_horizon_mae.set_index(["tso", "unique_id", "horizon"])["diff"].unstack(
        level=2
    )
    if heatmap_data.empty:
        logger.warning("  MAE diff heatmap: empty after pivot; skipping.")
        return

    vabs = float(np.nanmax(np.abs(heatmap_data.values)))
    if vabs == 0:
        vabs = 1.0
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    n_rows = len(heatmap_data)
    fig, ax = plt.subplots(figsize=(12, max(4, n_rows * 0.7)))
    im = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        cmap="mae_diff",
        norm=norm,
    )
    ax.set_xticks(range(heatmap_data.shape[1]))
    ax.set_xticklabels([str(h) for h in heatmap_data.columns])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [f"{tso}, {uid}" for tso, uid in heatmap_data.index]
    )
    ax.set_xlabel("Horizon")
    plt.colorbar(im, ax=ax, label="MAE diff (MWh)")
    no_legend(ax)
    save_figure(
        fig,
        config.figures_dir / "mae_diff_heatmap",
        fmt=config.figure_format,
    )


# ---------------------------------------------------------------------------
# Per-horizon line plot
# ---------------------------------------------------------------------------


def run_mae_horizon_lineplot(
    df_with_baselines: pd.DataFrame,
    config: PaperConfig,
    metric_name: str = "mae",
) -> None:
    """Save a per-horizon mean-MAE line plot (one line per model)."""
    logger.info("  MAE - horizon line plot ...")
    per_horizon = _run_evaluations_per_horizon(df_with_baselines, config)
    model_cols = [c for c in per_horizon.columns if c not in _EVAL_META]
    per_horizon_mae = per_horizon.loc[per_horizon["metric"] == metric_name, ["tso", "unique_id", "horizon"] + model_cols].copy()
    if not model_cols:
        logger.warning("  MAE horizon line: no model columns; skipping.")
        return

    melted = per_horizon_mae.melt(
        id_vars=["tso", "unique_id", "horizon"],
        value_vars=model_cols,
        var_name="model",
        value_name="prediction",
    )
    per_model_horizon = (
        melted.groupby(["model", "horizon"])["prediction"].mean().reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for model, grp in per_model_horizon.groupby("model"):
        grp = grp.sort_values("horizon")
        ax.plot(
            grp["horizon"].astype(int),
            grp["prediction"],
            color=get_model_color(str(model)),
            label=str(model),
        )
    tick_positions = np.arange(1, 25)
    tick_labels = [str(h) if (h == 1 or h % 5 == 0 or h % 24 == 0) else "" for h in tick_positions]
    ax.set_xlabel("Horizon")
    ax.set_xlim(1, 24)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="both", labelsize=10)
    if metric_name == "mae":
        ax.set_ylabel("MAE (MWh)")
    elif metric_name == "fbeta_2.0":
        ax.set_ylabel(r"$F_{\beta=2}$")
    elif metric_name == "mae_conditional":
        ax.set_ylabel("MAE | y > 0 (MWh)")
    elif metric_name == "r2_score":
        ax.set_ylabel(r"$R^2$")
    else:
        ax.set_ylabel(metric_name.upper())
    
    no_legend(ax)
    processed_metric = metric_name.replace(".", "_")
    save_figure(
        fig,
        config.figures_dir / f"{processed_metric}_horizon_line",
        fmt=config.figure_format,
    )


# ---------------------------------------------------------------------------
# Per-horizon box plot
# ---------------------------------------------------------------------------


def run_mae_horizon_boxplot(
    df_with_baselines: pd.DataFrame,
    config: PaperConfig,
):
    """Save a per-horizon MAE boxplot grouped by horizon aggregation."""
    horizon_agg_colors = {
        "1-9h": "#7A6529",
        "10-16h": "#5999E2",
        "17-24h": "#E574AC16",
    }
    logger.info("  MAE - horizon boxplot ...")
    per_horizon = _run_evaluations_per_horizon(df_with_baselines, config)
    per_horizon_mae = per_horizon[per_horizon["metric"] == "mae"].copy()
    model_cols = [c for c in per_horizon_mae.columns if c not in _EVAL_META]
    if not model_cols:
        logger.warning("  MAE horizon line: no model columns; skipping.")
        return

    per_horizon_eval_melted_df = per_horizon_mae.melt(
        id_vars=["tso", "unique_id", "horizon"],
        value_vars=model_cols,
        var_name="model", 
        value_name="prediction"
    )
    # per_horizon_eval_melted_df = per_horizon_eval_melted_df.rename(columns=lambda c: get_model_display_name(str(c)))
    per_horizon_eval_melted_df["horizon_agg"] = np.where(
        per_horizon_eval_melted_df["horizon"] <= 9, "1-9h", 
        np.where(
            per_horizon_eval_melted_df["horizon"] <= 16, "10-16h", 
            "17-24h"
        )
    )
    palette = [horizon_agg_colors.get(agg, "#808080") for agg in sorted(per_horizon_eval_melted_df["horizon_agg"].unique())]

    per_horizon_eval_melted_df["model"] = pd.Categorical(
        per_horizon_eval_melted_df["model"].map(lambda m: get_model_display_name(str(m))),
        categories=[get_model_display_name(str(m)) for m in DEFAULT_MODEL_ORDER if str(m) in per_horizon_eval_melted_df["model"].str.replace(r"_seed\d+$", "", regex=True).unique()],
        ordered=True,
    )
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="model", y="prediction", hue="horizon_agg", 
        data=per_horizon_eval_melted_df, palette=palette
    )
    plt.ylabel("MAE (MWh)")
    plt.xlabel("")
    no_legend(fig.axes[0])
    save_figure(
        fig,
        config.figures_dir / "mae_horizon_boxplot",
        fmt=config.figure_format,
    )
    save_model_colors(config, "boxplot_colors.tex", horizon_agg_colors)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(config: PaperConfig) -> None:
    """Run all MAE/F-beta outputs and write them to *config.output_dir*."""
    predictions_df = load_predictions(config, apply_model_selection=True)
    predictions_unfiltered_df = load_predictions(config, apply_model_selection=False)

    run_mae_overall(predictions_df, config, "overall_eval.tex")
    run_mae_overall(predictions_unfiltered_df, config, "eval_benchmarks_only.tex", model_subset=BENCHMARK_MODELS)
    run_mae_overall(predictions_unfiltered_df, config, "eval_default_models.tex", model_subset=DEFAULT_MODELS)
    run_mae_sparsity_table(predictions_unfiltered_df, config, "sparsity_eval.tex")
    run_mae_per_regime(predictions_df, config)
    run_mae_horizon_boxplot(predictions_df, config)
    run_mae_diff_heatmap(predictions_df, config)
    for metric in config.mae_metrics:
        run_mae_horizon_lineplot(predictions_df, config, metric_name=metric)
