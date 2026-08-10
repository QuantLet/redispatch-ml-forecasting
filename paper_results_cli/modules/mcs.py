"""MCS (Model Confidence Set) inclusion-rate tables and heatmap.

Outputs
-------
tables/mcs_overall.tex
    Overall MCS inclusion rate (one row per model, sorted descending).
tables/mcs_per_regime.tex
    Regime × model inclusion-rate table.
figures/mcs_horizon_heatmap.{png,pdf}
    Model (y-axis) × horizon (x-axis) heat-map of inclusion rates.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from covariate_effect.mcs import compare_models_mcs

from ..config import BENCHMARK_MODELS, PaperConfig
from ..data import apply_model_ordering, load_predictions, select_models
from ..style import get_model_display_name, no_legend, save_figure
from ..tables import make_percentage_formatter, save_latex_table, add_index_names

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _inclusion_rate(
    predictions_df: pd.DataFrame,
    alpha: float,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series:
    """Run MCS and return the per-model inclusion-rate Series.

    The *denominator* is the total number of (TSO, direction) pairs present in
    *predictions_df*.
    """
    mcs_df = compare_models_mcs(
        predictions_df,
        alpha=alpha,
        start_date=start_date,
        end_date=end_date,
    )
    n_pairs = predictions_df.drop_duplicates(subset=["tso", "unique_id"]).shape[0]
    if n_pairs == 0:
        return pd.Series(dtype=float)
    return (
        mcs_df.pivot_table(
            index=["tso", "direction"],
            columns="models",
            values="status",
            aggfunc=lambda g: g.iloc[0] == "included",
        )
        .sum(axis=0)
        / n_pairs
    )


def _inclusion_rate_by_sparsity(
    predictions_df: pd.DataFrame,
    alpha: float,
    start_date: str | None = None,
    end_date: str | None = None,
    sparsity_threshold: float = 0.7,
) -> pd.DataFrame:
    """Run MCS and return per-model inclusion rates for dense/sparse groups."""
    mcs_df = compare_models_mcs(
        predictions_df,
        alpha=alpha,
        start_date=start_date,
        end_date=end_date,
    )
    sparsity = (
        predictions_df.groupby(["tso", "unique_id"])["y"]
        .apply(lambda values: (values == 0).mean() >= sparsity_threshold)
        .reset_index(name="sparsity")
    )
    sparsity_count = sparsity["sparsity"].value_counts()
    if sparsity_count.empty:
        return pd.DataFrame(columns=["Dense", "Sparse"])

    mcs_with_sparsity = (
        mcs_df.merge(
            sparsity,
            left_on=["tso", "direction"],
            right_on=["tso", "unique_id"],
            how="left",
        )
        .drop(columns="unique_id")
        .dropna(subset=["sparsity"])
    )
    aggregated = (
        mcs_with_sparsity.pivot_table(
            index=["tso", "direction", "sparsity"],
            columns="models",
            values="status",
            aggfunc=lambda g: g.iloc[0] == "included",
        )
        .unstack(level=2)
        .sum(axis=0)
    )
    rates = aggregated.div(sparsity_count, level="sparsity").unstack(level=1)
    rates = rates.rename(columns={False: "Dense", True: "Sparse"})
    for col in ["Dense", "Sparse"]:
        if col not in rates.columns:
            rates[col] = pd.NA
    rates = rates[["Dense", "Sparse"]]
    rates.index.name = "models"
    rates.columns.name = None
    return rates


def _table_models(config: PaperConfig) -> list[str]:
    return list(dict.fromkeys(list(config.models) + list(BENCHMARK_MODELS)))


def _select_order_rename(df: pd.DataFrame | pd.Series, models: list[str]) -> pd.DataFrame | pd.Series:
    selected = select_models(df, models, by_index=True)
    ordered = apply_model_ordering(selected, by_index=True)
    return ordered.rename(index=get_model_display_name)


def _fbeta_dominance(
    predictions_df: pd.DataFrame,
    config: PaperConfig,
    table_models: list[str],
) -> pd.Series:
    from .fbeta import _compute_dominance  # noqa: PLC0415

    dominance = _compute_dominance(
        predictions_df,
        config,
        show_progress=False,
    )["model_dominance"]
    return _select_order_rename(dominance, table_models)  # type: ignore[return-value]


def _fbeta_dominance_by_sparsity(
    predictions_df: pd.DataFrame,
    config: PaperConfig,
    table_models: list[str],
) -> pd.DataFrame:
    """Return F-beta dominance computed independently on dense/sparse series."""
    series_sparsity = (
        predictions_df.groupby(["tso", "unique_id"])["y"]
        .apply(lambda values: (values == 0).mean() >= config.dm_sparsity_threshold)
        .rename("sparse")
    )
    columns: dict[str, pd.Series] = {}
    for label, is_sparse in (("Dense", False), ("Sparse", True)):
        keys = series_sparsity[series_sparsity == is_sparse].index
        mask = pd.MultiIndex.from_frame(
            predictions_df[["tso", "unique_id"]]
        ).isin(keys)
        subset = predictions_df.loc[mask]
        if subset.empty:
            columns[label] = pd.Series(dtype=float)
        else:
            columns[label] = _fbeta_dominance(subset, config, table_models)
    return pd.DataFrame(columns).reindex(columns=["Dense", "Sparse"])


def _mcs_inclusion_marks(
    predictions_df: pd.DataFrame,
    config: PaperConfig,
) -> pd.DataFrame:
    """Return exact MCS membership for every selected model and target series."""
    filtered = select_models(predictions_df, config.models, by_index=False)
    mcs_df = compare_models_mcs(
        filtered,
        alpha=config.mcs_alpha,
        start_date=config.start_date,
    )
    marks = mcs_df.pivot_table(
        index="models",
        columns=["tso", "direction"],
        values="status",
        aggfunc=lambda values: values.iloc[0] == "included",
    )
    marks = apply_model_ordering(marks, by_index=True)
    marks = marks.rename(index=get_model_display_name)
    marks.index.name = "Model"
    marks.columns.names = ["TSO", "Series"]
    return marks


def _rename_columns(latex: str, df: pd.DataFrame) -> str:
    column_mapping = {
        "models": "Model",
        "inclusion_rate": "MCS inclusion rate",
        "overall_mcs": "Overall MCS",
        "benchmark_mcs": "Benchmark-only MCS",
        "fbeta_dominance": r"F$_{\beta}$ dominance",
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
# Overall
# ---------------------------------------------------------------------------


def run_mcs_overall(predictions_df: pd.DataFrame, config: PaperConfig, file_name: str, model_subset: list[str] = [], apply_filtering: bool = True) -> pd.DataFrame:
    """Compute and save the overall-sample MCS inclusion rate table."""
    logger.info("  MCS - overall sample ...")
    if apply_filtering:
        predictions_df = select_models(predictions_df, config.models, by_index=False)
    elif model_subset:
        predictions_df = select_models(predictions_df, model_subset, by_index=False)
    rates = _inclusion_rate(
        predictions_df, config.mcs_alpha, start_date=config.start_date
    )
    rates_sorted = apply_model_ordering(rates, by_index=True)

    df = rates_sorted.to_frame(name="inclusion_rate").rename(index=get_model_display_name)
    save_latex_table(
        df,
        config.tables_dir / file_name,
        caption=f"Overall MCS inclusion rate ($\\alpha={config.mcs_alpha}$)",
        formatters={"inclusion_rate": make_percentage_formatter(2)},
        final_form_callback=_rename_columns,
    )
    return df


def run_mcs_by_sparsity(
    predictions_df: pd.DataFrame,
    config: PaperConfig,
    file_name: str,
    model_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Compute and save dense/sparse MCS inclusion rates."""
    logger.info("  MCS - by sparsity ...")
    table_models = _table_models(config) if model_subset is None else model_subset
    if model_subset is not None:
        predictions_df = select_models(predictions_df, model_subset, by_index=False)
    rates = _inclusion_rate_by_sparsity(
        predictions_df,
        config.mcs_alpha,
        start_date=config.start_date,
        sparsity_threshold=config.dm_sparsity_threshold,
    )
    df: pd.DataFrame = _select_order_rename(rates, table_models)  # type: ignore[assignment]
    save_latex_table(
        df,
        config.tables_dir / file_name,
        caption=(
            f"MCS inclusion rate by target sparsity "
            f"($\\alpha={config.mcs_alpha}$)"
        ),
        formatters={col: make_percentage_formatter(2) for col in df.columns},
        final_form_callback=_rename_columns,
    )
    return df


def run_mcs_overall_benchmark_sparsity(
    predictions_df: pd.DataFrame,
    config: PaperConfig,
) -> pd.DataFrame:
    """Save one table combining overall and benchmark-only MCS by sparsity."""
    logger.info("  MCS - overall/benchmark by sparsity ...")
    table_models = _table_models(config)
    overall = _select_order_rename(
        _inclusion_rate_by_sparsity(
            predictions_df,
            config.mcs_alpha,
            start_date=config.start_date,
            sparsity_threshold=config.dm_sparsity_threshold,
        ),
        table_models,
    )
    benchmark_predictions = select_models(predictions_df, BENCHMARK_MODELS, by_index=False)
    benchmarks = _select_order_rename(
        _inclusion_rate_by_sparsity(
            benchmark_predictions,
            config.mcs_alpha,
            start_date=config.start_date,
            sparsity_threshold=config.dm_sparsity_threshold,
        ),
        BENCHMARK_MODELS,
    )
    benchmarks = benchmarks.reindex(overall.index)
    df = pd.concat(
        {"Overall MCS": overall, "Benchmark-only MCS": benchmarks},
        axis=1,
    )
    df.columns.names = [None, None]
    df.index.name = "models"
    save_latex_table(
        df,
        config.tables_dir / "mcs_overall_benchmark_sparsity.tex",
        caption=(
            f"Overall and benchmark-only MCS inclusion rates by target sparsity "
            f"($\\alpha={config.mcs_alpha}$)"
        ),
        formatters={col: make_percentage_formatter(2) for col in df.columns},
        final_form_callback=_rename_columns,
    )
    return df


def run_mcs_overall_benchmark_fbeta(
    predictions_df: pd.DataFrame,
    config: PaperConfig,
    fbeta_dominance: pd.Series | None = None,
) -> pd.DataFrame:
    """Save overall MCS, benchmark-only MCS, and overall F-beta dominance."""
    logger.info("  MCS - overall/benchmark/F-beta summary ...")
    table_models = _table_models(config)
    overall = _select_order_rename(
        _inclusion_rate(predictions_df, config.mcs_alpha, start_date=config.start_date),
        table_models,
    )
    benchmark_predictions = select_models(predictions_df, BENCHMARK_MODELS, by_index=False)
    benchmarks = _select_order_rename(
        _inclusion_rate(
            benchmark_predictions,
            config.mcs_alpha,
            start_date=config.start_date,
        ),
        BENCHMARK_MODELS,
    ).reindex(overall.index)
    fbeta = (
        _fbeta_dominance(predictions_df, config, table_models)
        if fbeta_dominance is None
        else fbeta_dominance
    )
    df = pd.concat(
        [
            overall.rename("overall_mcs"),
            benchmarks.rename("benchmark_mcs"),
            fbeta.rename("fbeta_dominance"),
        ],
        axis=1,
    )
    df.index.name = "models"
    save_latex_table(
        df,
        config.tables_dir / "mcs_overall_benchmark_fbeta.tex",
        caption=(
            f"Overall MCS, benchmark-only MCS, and "
            rf"F$_{{\beta={config.fbeta_beta:.0f}}}$ dominance"
        ),
        formatters={col: make_percentage_formatter(2) for col in df.columns},
        final_form_callback=_rename_columns,
    )
    return df


def run_mcs_glued_summary(
    predictions_df: pd.DataFrame,
    config: PaperConfig,
) -> pd.DataFrame:
    """Save dense/sparse MCS inclusion and F-beta dominance percentages."""
    logger.info("  MCS - glued summary ...")
    table_models = _table_models(config)
    overall_sparsity = _select_order_rename(
        _inclusion_rate_by_sparsity(
            predictions_df,
            config.mcs_alpha,
            start_date=config.start_date,
            sparsity_threshold=config.dm_sparsity_threshold,
        ),
        table_models,
    )
    fbeta_sparsity = _fbeta_dominance_by_sparsity(
        predictions_df, config, table_models
    ).reindex(overall_sparsity.index)

    rows: list[dict[str, object]] = []
    for model in overall_sparsity.index:
        rows.append(
            {
                "Model": model,
                "Measure": "MCS inclusion",
                "Dense": overall_sparsity.loc[model, "Dense"],
                "Sparse": overall_sparsity.loc[model, "Sparse"],
            }
        )
        rows.append(
            {
                "Model": model,
                "Measure": rf"F$_{{\beta={config.fbeta_beta:.0f}}}$ dominance",
                "Dense": fbeta_sparsity.loc[model, "Dense"],
                "Sparse": fbeta_sparsity.loc[model, "Sparse"],
            }
        )
    df = pd.DataFrame(rows).set_index(["Model", "Measure"])
    save_latex_table(
        df,
        config.tables_dir / "mcs_glued_summary.tex",
        caption=(
            f"MCS inclusion and "
            rf"F$_{{\beta={config.fbeta_beta:.0f}}}$ dominance by target sparsity"
        ),
        formatters={col: make_percentage_formatter(2) for col in df.columns},
        final_form_callback=_rename_columns,
    )
    return df


def run_mcs_inclusion_by_series(
    predictions_df: pd.DataFrame,
    config: PaperConfig,
) -> pd.DataFrame:
    """Save binary MCS inclusion for each selected model and target series."""
    logger.info("  MCS - exact inclusion by series ...")
    df = _mcs_inclusion_marks(predictions_df, config)

    def mark(value: object) -> str:
        return r"\textbf{X}" if pd.notna(value) and bool(value) else ""

    def tidy_header(latex: str, table: pd.DataFrame) -> str:
        return _rename_columns(latex, table).replace("SeriesModel &", "Model &")

    save_latex_table(
        df,
        config.tables_dir / "mcs_inclusion_by_series.tex",
        caption=f"MCS inclusion by target series ($\\alpha={config.mcs_alpha}$); "
        r"an \textbf{X} denotes inclusion.",
        formatters={column: mark for column in df.columns},
        final_form_callback=tidy_header,
    )
    return df


# ---------------------------------------------------------------------------
# Per-regime
# ---------------------------------------------------------------------------


def run_mcs_per_regime(predictions_df: pd.DataFrame, config: PaperConfig) -> pd.DataFrame:
    """Compute and save the per-regime MCS inclusion rate table."""
    logger.info("  MCS - per regime ...")
    regime_series: dict[str, pd.Series] = {}
    for regime_name, (regime_start, regime_end) in config.stress_regimes.items():
        regime_rates = _inclusion_rate(
            predictions_df,
            config.mcs_alpha,
            start_date=regime_start,
            end_date=regime_end,
        )
        regime_rates_filtered: pd.Series = select_models(regime_rates, config.models, by_index=True)
        regime_rates_sorted: pd.Series = apply_model_ordering(regime_rates_filtered, by_index=True)
        regime_series[regime_name] = regime_rates_sorted
    df = pd.DataFrame(regime_series)
    df = df.rename(index=get_model_display_name)
    save_latex_table(
        df,
        config.tables_dir / "mcs_per_regime.tex",
        caption=f"Per-regime MCS inclusion rate ($\\alpha={config.mcs_alpha}$)",
        formatters={col: make_percentage_formatter(2) for col in df.columns},
        final_form_callback=_rename_columns,
    )
    return df


# ---------------------------------------------------------------------------
# Per-horizon heat-map
# ---------------------------------------------------------------------------


def run_mcs_horizon_heatmap(
    predictions_df: pd.DataFrame,
    config: PaperConfig,
) -> None:
    """Compute per-horizon MCS inclusion rates and save a heat-map figure."""
    logger.info("  MCS - per horizon heatmap ...")
    horizons = sorted(predictions_df["horizon"].unique())
    n_pairs = predictions_df.drop_duplicates(subset=["tso", "unique_id"]).shape[0]

    horizon_series: list[pd.DataFrame] = []
    for horizon in tqdm(horizons, desc="MCS per horizon", leave=False):
        h_df = predictions_df[predictions_df["horizon"] == horizon].copy()
        mcs_h = compare_models_mcs(
            h_df,
            alpha=config.mcs_alpha,
            start_date=config.start_date,
        )
        rates_h = (
            mcs_h.pivot_table(
                index=["tso", "direction"],
                columns="models",
                values="status",
                aggfunc=lambda g: g.iloc[0] == "included",
            )
            .sum(axis=0)
            / (n_pairs if n_pairs > 0 else 1)
        )
        rates_h_filtered: pd.Series = select_models(rates_h, config.models, by_index=True)
        horizon_series.append(
            rates_h_filtered.to_frame(name="mcs_prob").assign(horizon=int(horizon))
        )

    if not horizon_series:
        logger.warning("  MCS per-horizon: no data - skipping heatmap.")
        return

    long_df = pd.concat(horizon_series, axis=0)
    heatmap_data = (
        long_df.reset_index()
        .set_index(["models", "horizon"])["mcs_prob"]
        .unstack(level=1)
    )
    heatmap_data: pd.DataFrame = apply_model_ordering(heatmap_data, by_index=True)

    fig, ax = plt.subplots(figsize=(12, max(4, len(heatmap_data) * 0.5)))
    im = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
    )
    ax.set_xticks(range(heatmap_data.shape[1]))
    ax.set_xticklabels([str(h) for h in heatmap_data.columns])
    ax.set_yticks(range(heatmap_data.shape[0]))
    ax.set_yticklabels([get_model_display_name(str(m)) for m in heatmap_data.index])
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Model")
    plt.colorbar(im, ax=ax, label="MCS inclusion rate")
    no_legend(ax)
    save_figure(
        fig,
        config.figures_dir / "mcs_horizon_heatmap",
        fmt=config.figure_format,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(config: PaperConfig) -> None:
    """Run all MCS outputs and write them to *config.output_dir*."""
    # We run all models in the same MCS, so we don't apply model selection here, but after obtaining results
    predictions_df = load_predictions(config, apply_model_selection=False)
    run_mcs_overall(predictions_df, config, file_name="mcs_overall.tex")
    run_mcs_overall(predictions_df, config, file_name="mcs_overall_unfiltered.tex", apply_filtering=False)
    run_mcs_overall(predictions_df, config, file_name="mcs_overall_benchmarks.tex", apply_filtering=False, model_subset=BENCHMARK_MODELS)
    run_mcs_by_sparsity(predictions_df, config, file_name="mcs_by_sparsity.tex")
    run_mcs_glued_summary(predictions_df, config)
    run_mcs_inclusion_by_series(predictions_df, config)
    run_mcs_per_regime(predictions_df, config)
    run_mcs_horizon_heatmap(predictions_df, config)
