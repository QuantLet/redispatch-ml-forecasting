"""Plot best-checkpoint rolling-window predictions.

This module is intentionally not wired into ``paper_results_cli.__main__``.
Run it directly when you want a quick visual check of one or more model
predictions for selected rolling windows.

Example
-------
python -m paper_results_cli.plot_best_checkpoint_windows \\
    --predictions-dir outputs_paper \\
    --benchmark-dir outputs_paper \\
    --dataset-name basic_day_ahead_price_wind_pv_production_consumption_sce \\
    --dataset-dir data/model_data_paper_actuals_1h_lag \\
    --tso 50Hertz \\
    --windows 2 3 \\
    --models nhits tft ridge_regression_rolling
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_results_cli.config import (
    BENCHMARK_MODELS,
    DATASET_NAME,
    DEFAULT_MODELS,
    DEFAULT_START_DATE,
    PaperConfig,
)
from paper_results_cli.data import (
    load_benchmark_predictions_with_target,
    read_per_window_predictions,
    _resolve_dataset_name,
    resolve_model_columns,
    select_models,
    validate_predictions,
)
from paper_results_cli.style import (
    apply_paper_style,
    get_model_color,
    no_legend,
    save_figure,
)

logger = logging.getLogger(__name__)

TSO_DISPLAY_NAMES = {
    "50hertz": "50Hertz",
    "amprion": "Amprion",
    "transnetbw": "TransnetBW",
    "tennet_de": "TenneT_DE",
    "tennet": "TenneT_DE",
}

ACTUAL_COLOR = "#424242"


def _normalise_tso(value: str) -> str:
    """Return the display TSO name used by prediction and dataset files."""
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    return TSO_DISPLAY_NAMES.get(key, value.strip())


def _safe_name(value: str) -> str:
    """Return a filesystem-friendly stem fragment."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def _save_plot_colors(
    style_dir: Path,
    file_name: str,
    color_dict: dict[str, str],
) -> None:
    """Save the colors used by this plot as LaTeX color definitions."""
    lines: list[str] = [
        "% Colour definitions for best-checkpoint prediction plots",
        "% Paste into LaTeX preamble after: \\usepackage[dvipsnames]{xcolor}",
        "",
    ]
    for label, hex_color in color_dict.items():
        r = int(hex_color[1:3], 16) / 255
        g = int(hex_color[3:5], 16) / 255
        b = int(hex_color[5:7], 16) / 255
        tex_name = (
            label.replace("_", " ")
            .replace("-", " ")
            .title()
            .replace(" ", "")
        )
        lines.append(
            f"\\definecolor{{{tex_name}}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}"
            f"  % {label}: {hex_color}"
        )

    style_dir.mkdir(parents=True, exist_ok=True)
    (style_dir / file_name).write_text("\n".join(lines) + "\n")


def _load_actuals(
    dataset_dir: Path,
    dataset_name: str,
    tso: str,
) -> pd.DataFrame | None:
    """Load actual target values in Nixtla format for optional plot context."""
    actuals_path = dataset_dir / f"{dataset_name}_{tso}.parquet"
    if not actuals_path.exists():
        logger.warning("Actuals file not found: %s", actuals_path)
        return None

    raw = pd.read_parquet(actuals_path)
    rename_map = {
        "begin_date": "ds",
        "direction": "unique_id",
        "total_load": "y",
    }
    raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})
    required = {"unique_id", "ds", "y"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Actuals file '{actuals_path}' is missing columns {sorted(missing)}.")

    actuals = raw[["unique_id", "ds", "y"]].copy()
    actuals["ds"] = pd.to_datetime(actuals["ds"])
    return actuals.drop_duplicates(["unique_id", "ds"]).sort_values(["unique_id", "ds"])


def _filter_predictions(
    predictions: pd.DataFrame,
    *,
    tsos: list[str] | None,
    windows: list[int],
    unique_ids: list[str] | None,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """Apply window, TSO, direction, and date filters."""
    filtered = predictions[predictions["window_index"].isin(windows)].copy()
    if tsos:
        filtered = filtered[filtered["tso"].isin(tsos)]
    if unique_ids:
        filtered = filtered[filtered["unique_id"].isin(unique_ids)]
    if start_date:
        filtered = filtered[filtered["ds"] >= pd.Timestamp(start_date)]
    if end_date:
        filtered = filtered[filtered["ds"] <= pd.Timestamp(end_date)]
    if filtered.empty:
        raise ValueError("No predictions remain after applying the requested filters.")
    return filtered


def _merge_benchmark_predictions(
    predictions: pd.DataFrame,
    *,
    benchmark_dir: Path,
    dataset_name: str,
    start_date: str,
    requested_models: list[str],
) -> pd.DataFrame:
    """Attach requested benchmark columns to per-window neural predictions."""
    csv_benchmark_models = set(BENCHMARK_MODELS) - {"lstm"}
    requested_benchmarks = [
        model for model in requested_models if model in csv_benchmark_models
    ]
    if not requested_benchmarks:
        return predictions

    key_cols = ["tso", "unique_id", "ds", "horizon"]
    window_key_cols = [*key_cols, "y", "window_index"]
    anchor_cols = resolve_model_columns(predictions, DEFAULT_MODELS)
    if not anchor_cols:
        raise ValueError(
            "Benchmark loading needs at least one neural prediction column "
            "to attach targets, but none of the default neural models were found."
        )

    target_source = (
        predictions[[*key_cols, "y", anchor_cols[0]]]
        .drop_duplicates(key_cols)
        .copy()
    )
    config = PaperConfig(
        predictions_dir=benchmark_dir,
        benchmarks_dir=benchmark_dir,
        dataset_name=dataset_name,
        start_date=start_date,
        models=requested_benchmarks,
    )
    benchmarks = load_benchmark_predictions_with_target(config, target_source)
    benchmarks = select_models(benchmarks, requested_benchmarks)

    benchmark_cols = resolve_model_columns(benchmarks, requested_benchmarks)
    window_keys = predictions[window_key_cols].drop_duplicates().copy()
    benchmarks_by_window = window_keys.merge(
        benchmarks[[*key_cols, *benchmark_cols]],
        on=key_cols,
        how="left",
    )

    missing_cols = [
        col for col in benchmark_cols if benchmarks_by_window[col].isna().all()
    ]
    if missing_cols:
        raise ValueError(
            "Benchmark columns could not be aligned to selected windows: "
            f"{missing_cols}"
        )

    return predictions.merge(
        benchmarks_by_window,
        on=window_key_cols,
        how="left",
    )


def _make_window_plot(
    window_df: pd.DataFrame,
    actuals_df: pd.DataFrame | None,
    model_cols: list[str],
    *,
    tso: str,
    window: int,
    output_dir: Path,
    figure_format: str,
    history_hours: int,
    max_insample_length: int | None,
    width: float,
    height: float,
) -> Path:
    """Create and save one plot for a TSO/window slice."""
    forecast_df = (
        window_df[["unique_id", "ds", *model_cols]]
        .drop_duplicates(["unique_id", "ds"])
        .sort_values(["unique_id", "ds"])
    )

    min_ds = forecast_df["ds"].min()
    max_ds = forecast_df["ds"].max()
    if actuals_df is not None:
        history_start = min_ds - pd.Timedelta(hours=history_hours)
        actual_plot_df = actuals_df[
            (actuals_df["unique_id"].isin(forecast_df["unique_id"].unique()))
            & (actuals_df["ds"] >= history_start)
            & (actuals_df["ds"] <= max_ds)
        ].copy()
    else:
        actual_plot_df = (
            window_df[["unique_id", "ds", "y"]]
            .drop_duplicates(["unique_id", "ds"])
            .sort_values(["unique_id", "ds"])
        )

    if actual_plot_df.empty:
        raise ValueError(f"No actual values available for TSO={tso}, window={window}.")

    unique_ids = sorted(forecast_df["unique_id"].unique())
    if max_insample_length is not None:
        actual_plot_df = (
            actual_plot_df.groupby("unique_id", group_keys=False)
            .tail(max_insample_length)
            .copy()
        )

    fig, axes = plt.subplots(
        nrows=len(unique_ids),
        ncols=1,
        figsize=(width, height * len(unique_ids)),
        sharex=True,
        squeeze=False,
    )
    for ax, unique_id in zip(axes[:, 0], unique_ids, strict=True):
        actual_direction = actual_plot_df[actual_plot_df["unique_id"] == unique_id]
        forecast_direction = forecast_df[forecast_df["unique_id"] == unique_id]
        ax.plot(
            actual_direction["ds"],
            actual_direction["y"],
            color=ACTUAL_COLOR,
            alpha=0.45,
            linewidth=1.2,
            label="Actual",
        )
        for model in model_cols:
            ax.plot(
                forecast_direction["ds"],
                forecast_direction[model],
                color=get_model_color(model),
                linewidth=1.5,
                label=model,
            )
        ax.set_title("Direction: {}".format(unique_id))
        ax.set_ylabel("Redispatch load (MWh)")
        ax.set_xlabel("Time")
        no_legend(ax)

    model_part = "_".join(_safe_name(model) for model in model_cols)
    output_path = output_dir / f"best_checkpoint_predictions_{_safe_name(tso)}_window{window}_{model_part}"
    save_figure(fig, output_path, fmt=figure_format)
    return output_path.with_suffix(f".{figure_format}")


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone parser for this plotting helper."""
    parser = argparse.ArgumentParser(
        prog="python -m paper_results_cli.plot_best_checkpoint_windows",
        description="Plot selected best-checkpoint rolling-window predictions.",
    )
    parser.add_argument("--predictions-dir", type=Path, default=Path("outputs_paper/"))
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Root of benchmark predictions (default: same as --predictions-dir).",
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/model_data_extended"),
        help="Root containing '<dataset-name>_<TSO>.parquet' actuals files.",
    )
    parser.add_argument(
        "--actuals-dataset-name",
        default=None,
        help="Dataset parquet stem for actuals. Defaults to --dataset-name.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results/figures"))
    parser.add_argument("--style-dir", type=Path, default=Path("paper_results/style"))
    parser.add_argument("--windows", nargs="+", type=int, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS + BENCHMARK_MODELS)
    parser.add_argument(
        "--tso",
        nargs="+",
        default=None,
        help="Optional TSO filter, e.g. 50Hertz Amprion TenneT_DE.",
    )
    parser.add_argument(
        "--unique-id",
        nargs="+",
        default=None,
        help="Optional direction filter, usually 'up' and/or 'down'.",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--history-hours", type=int, default=72)
    parser.add_argument(
        "--max-insample-length",
        type=int,
        default=None,
        help="Limit the number of actual/history points shown per direction.",
    )
    parser.add_argument("--figure-format", choices=["png", "pdf"], default="png")
    parser.add_argument("--width", type=float, default=16.0)
    parser.add_argument("--height", type=float, default=3.5)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the plotting helper."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    apply_paper_style()
    dataset_name = _resolve_dataset_name(args.predictions_dir, args.dataset_name)
    actuals_dataset_name = args.actuals_dataset_name or dataset_name
    benchmark_dir = args.benchmark_dir or args.predictions_dir
    tsos = [_normalise_tso(tso) for tso in args.tso] if args.tso else None

    predictions = read_per_window_predictions(
        root_dir=args.predictions_dir,
        dataset_name=dataset_name,
        checkpoint_best=True,
    )
    validate_predictions(predictions, context="best-checkpoint predictions")
    predictions = _merge_benchmark_predictions(
        predictions,
        benchmark_dir=benchmark_dir,
        dataset_name=dataset_name,
        start_date=args.start_date or DEFAULT_START_DATE,
        requested_models=args.models,
    )
    predictions = select_models(predictions, args.models)
    predictions = _filter_predictions(
        predictions,
        tsos=tsos,
        windows=args.windows,
        unique_ids=args.unique_id,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    model_cols = resolve_model_columns(predictions, args.models)
    _save_plot_colors(
        args.style_dir,
        "best_checkpoint_prediction_colors.tex",
        {"actual": ACTUAL_COLOR, **{model: get_model_color(model) for model in model_cols}},
    )

    saved_paths: list[Path] = []
    for (tso, window), window_df in predictions.groupby(["tso", "window_index"], sort=True):
        actuals = _load_actuals(args.dataset_dir, actuals_dataset_name, str(tso))
        saved = _make_window_plot(
            window_df,
            actuals,
            model_cols,
            tso=str(tso),
            window=int(window),
            output_dir=args.output_dir,
            figure_format=args.figure_format,
            history_hours=args.history_hours,
            max_insample_length=args.max_insample_length,
            width=args.width,
            height=args.height,
        )
        logger.info("Saved %s", saved)
        saved_paths.append(saved)

    if not saved_paths:
        raise ValueError("No plots were created.")


if __name__ == "__main__":
    main()
