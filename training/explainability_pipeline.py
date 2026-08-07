"""
Post-hoc explainability pipeline for saved Integrated Gradients (IG) tensors.

Works with IG tensors produced by **both** ``prediction_pipeline.py``
(single-window) and ``prediction_pipeline_rolling_window.py``, which store
raw IG results under ``<output_dir>/ig_raw/<model_alias>[_windowN]/``.

Analyses produced
-----------------
1. **Overall feature importance** (whole test period + per-year)
   - absolute & signed bar charts (PNG)
   - CSV with pct_of_total, net_ratio, pos/neg share
2. **Per-horizon importance**
   - heatmap (features × horizons, signed & absolute) PNG
   - line plots for top-K features across horizon PNG
   - CSV with bootstrap CIs, percentiles, frac_pos/neg per horizon
3. **Clustered feature importance**
   - hierarchical clustering of feature-horizon profiles
   - dendrogram + cluster heatmaps PNG
   - cluster membership CSV

Usage
-----
::

    python -m training.explainability_pipeline \\
        --output-dir outputs/my_run \\
        --wandb-project redispatch-ml \\
        --mode absolute signed \\
        --top-k 25 \\
        --n-clusters 8

Or from a script::

    from training.explainability_pipeline import run_explainability_analysis
    run_explainability_analysis(output_dir=Path("outputs/my_run"), ...)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Literal

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb

from neuralforecast.core import NeuralForecast

from training.data_prep import build_static_df
from training.explainability import (
    get_stride_dates_from_wandb,
    load_ig_tensors,
    discover_ig_directories,
    combine_attributions,
    build_per_horizon_feature_matrix,
    compute_sample_level_stats,
    feature_importance_overall,
    feature_importance_quarterly,
    feature_importance_per_horizon,
    cluster_features_by_horizon,
    plot_overall_importance_bar,
    plot_signed_importance_bar,
    plot_per_horizon_heatmap,
    plot_per_horizon_violin,
    plot_feature_horizon_lines,
    plot_cluster_heatmap,
    plot_dendrogram,
    plot_cluster_top_features,
    build_feature_to_feature_set_map,
    plot_feature_set_bars,
    _CANONICAL_SET_ORDER,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Model discovery & loading helpers
# ══════════════════════════════════════════════════════════════════════════════


def _load_nf_model(model_dir: Path) -> NeuralForecast:
    """Load a NeuralForecast checkpoint from disk (last checkpoint only)."""
    from training.prediction_pipeline import load_model
    return load_model(model_dir, checkpoint_best=False)


def _resolve_model_for_ig_dir(
    ig_subdir: Path,
    model_root: Path | None,
    wandb_entity: str | None,
    wandb_project: str | None,
    output_dir: Path,
) -> NeuralForecast | None:
    """
    Attempt to load the NeuralForecast model matching an ig_raw sub-directory.

    Strategy:
    1. If ``model_root`` is given, scan it for a matching model directory.
    2. Fall back to loading the NF model from a matching name under model_root.

    Returns None if the model cannot be found.
    """
    if model_root is None:
        return None

    ig_name = ig_subdir.name  # e.g. nhits_seed778 or nhits_seed778_window0
    # Strip _windowN suffix
    if "_window" in ig_name:
        base_alias = ig_name.rsplit("_window", 1)[0]
    else:
        base_alias = ig_name

    # Search for nf_model dirs under model_root
    candidates: list[Path] = []

    # Check for top-level model dirs
    for p in model_root.iterdir():
        if p.is_dir() and base_alias in p.name:
            nf_dir = p / "nf_model"
            if nf_dir.is_dir():
                candidates.append(nf_dir)
            elif any(f.suffix == ".ckpt" for f in p.rglob("*.ckpt")):
                candidates.append(p)

    if not candidates:
        logger.warning("Could not find NF model for %s under %s", ig_name, model_root)
        return None

    model_dir = candidates[0]
    logger.info("Loading NF model for %s from %s", ig_name, model_dir)
    try:
        return _load_nf_model(model_dir)
    except Exception:
        logger.exception("Failed to load model from %s", model_dir)
        return None


def extract_model_timestamp_from_root_dir(model_path: Path, model_alias: str, window_index: int | None = None) -> str | None:
    """Extract timestamp from the model directory structure. For single window, the timestamp is the name of 'model_dir', and for rolling window, each window's .tar.zst file is named with the timestamp."""
    if model_path is None:
        return None

    if model_path.name == "evaluation":
        model_path = model_path.parent

    if window_index is not None:
        model_path = model_path / f"window_{window_index}" / model_alias
    
    timestmap_pattern_regex = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
    # Check for single-window pattern: .../model_dir/nf_model/
    match = re.search(timestmap_pattern_regex, model_path.name)
    if match:
        return match.group(0)
    files_in_dir = list(model_path.glob("*.tar.zst"))
    if files_in_dir:
        match = [file for file in files_in_dir if re.search(timestmap_pattern_regex, file.name)]
        if match:
            # Choose the latest timestamp if multiple matches are found
            match.sort(key=lambda x: re.search(timestmap_pattern_regex, x.name).group(0), reverse=True)
            if len(match) > 1:
                logger.info(f"Multiple .tar.zst files with timestamps found in {model_path}. Using the latest one: {match[0].name}")
                logger.info(f"Usng the latest available: {match[0].name}")
            return match[0].name.split(".tar.zst")[0]  # Return just the timestamp part without extension
    else:
        raise ValueError(f"No .tar.zst files found in {model_path} to extract timestamp from.")
    return None


def _extract_model_hparams_from_wandb(
    model_alias: str,
    model_path: Path | None,
    wandb_entity: str | None,
    wandb_project: str,
    window_index: int | None = None,
) -> dict | None:
    """Fetch run config from wandb to recover feature lists when the model
    checkpoint is not available."""
    try:
        api = wandb.Api()
        project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
        if model_path:
            timestamp = extract_model_timestamp_from_root_dir(model_path, model_alias=model_alias, window_index=window_index)
        else:
            timestamp = None
        if timestamp:
            filters = {
                "config.model_alias": model_alias,
                "config.date_time": timestamp,
            }
            runs = api.runs(project_path, filters=filters)
        else:
            runs = None
        if not runs:
            return None
        return dict(runs[0].config)
    except Exception:
        logger.warning("Could not fetch wandb config for %s", model_alias)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Rolling-window helpers
# ══════════════════════════════════════════════════════════════════════════════


def _extract_window_index(dirname: Path, model_path: Path) -> int | None:
    """Return the first parent directory of the form ``window_N`` from *dirname*, or ``None``.
    """
    current_dir = dirname
    while current_dir.exists() and str(current_dir.resolve()) != str(model_path.resolve()):
        if current_dir.name.startswith("window_"):
            try:
                return int(current_dir.name.split("_")[1])
            except (ValueError, IndexError):
                return None
        current_dir = current_dir.parent
    return None


def _sort_ig_dirs_by_window(
    ig_dirs: list[Path],
    model_path: Path,
) -> list[tuple[int | None, Path]]:
    """Sort *ig_dirs* by their window index (numerically, not lexicographically).

    Directories without a ``_windowN`` suffix (single-window runs) sort first.
    Directories **with** a suffix are ordered by the integer *N* so that
    ``window2`` correctly follows ``window1`` (and precedes ``window10``).

    Returns
    -------
    list of ``(window_index_or_None, path)`` pairs
    """
    pairs: list[tuple[int | None, Path]] = [
        (_extract_window_index(d, model_path=model_path), d) for d in ig_dirs
    ]
    pairs.sort(key=lambda x: (x[0] is not None, x[0] if x[0] is not None else 0))
    return pairs


def _build_rolling_window_stride_dates(
    ig_dirs_with_idx: list[tuple[int | None, Path]],
    n_strides_per_window: list[int],
    run_config: dict,
) -> list[pd.Timestamp]:
    """Generate one timestamp per IG stride for a rolling-window model.

    In the rolling-window training scheme every model checkpoint is evaluated
    on the **same** 1-month test period (global ``test_start`` →
    ``test_start + 1 month``).  For visualisation and quarterly breakdowns it
    is therefore useful to treat each checkpoint window as if it occupied a
    *different* calendar month: window 0 is labelled starting at
    ``test_start``, window 1 at ``test_start + 1 month``, etc.

    Strides within a window are spaced ``forecast_horizon`` hours apart
    (matching the prediction stride used during IG computation).

    Parameters
    ----------
    ig_dirs_with_idx :
        Sorted ``(window_index, path)`` pairs from
        :func:`_sort_ig_dirs_by_window`.
    n_strides_per_window :
        Number of IG strides found in each corresponding window directory.
    run_config :
        W&B run config dict.  Keys ``"test_start"`` (ISO string) and
        ``"forecast_horizon"`` (int, hours) are used.  Falls back to
        ``"2025-01-01"`` / ``24`` if absent.

    Returns
    -------
    list[pd.Timestamp]
        Length equals ``sum(n_strides_per_window)``.
    """
    raw_start = run_config.get("test_start", "2025-01-01 00:00:00")
    test_start = pd.Timestamp(raw_start)
    forecast_horizon = int(run_config.get("forecast_horizon", 24))

    dates: list[pd.Timestamp] = []
    for (window_idx, _), n_strides in zip(ig_dirs_with_idx, n_strides_per_window):
        window_num = window_idx if window_idx is not None else 0
        window_start = test_start + pd.DateOffset(months=window_num)
        for j in range(n_strides):
            dates.append(window_start + pd.Timedelta(hours=j * forecast_horizon))
    return dates


# ══════════════════════════════════════════════════════════════════════════════
#  Feature-set metadata resolution helpers
# ══════════════════════════════════════════════════════════════════════════════

# Maps the lowercase TSO output-directory name to the TSO suffix used in
# dataset metadata JSON filenames (e.g. "tennet_de" → "TenneT_DE").
_DIR_NAME_TO_TSO_FILE_SUFFIX: dict[str, str] = {
    "50hertz":    "50Hertz",
    "tennet_de":  "TenneT_DE",
    "amprion":    "Amprion",
    "transnetbw": "TransnetBW",
}


def _extract_feature_sets_from_run_config(run_config: dict) -> list[str] | None:
    """
    Extract the ``feature_sets`` list from a wandb run config dict.

    Handles both the normal case where ``dataset_metadata`` is a nested dict
    and the (rare) legacy case where ``feature_sets`` was logged at the top
    level.

    Returns ``None`` when the information cannot be found.
    """
    dm = run_config.get("dataset_metadata")
    if isinstance(dm, dict):
        fs = dm.get("feature_sets")
        if isinstance(fs, list) and fs:
            return fs
    # Legacy / flat config
    fs = run_config.get("feature_sets")
    if isinstance(fs, list) and fs:
        return fs
    return None


def _resolve_feature_sets_from_dir(
    ig_dir: Path,
    dataset_root_dir: Path,
) -> list[str] | None:
    """
    Infer the dataset feature-set list from the directory tree and a local
    dataset metadata JSON.

    The ig_dir path always has the form::

        .../<tso_dir>/<dataset_name>/[<timestamp>/]         # single-window
        .../<tso_dir>/<dataset_name>/window_N/              # rolling-window
            evaluation/ig_raw/<model_alias>               # (both)

    In both cases:

    * ``ig_dir.parents[3]`` → dataset dir  (e.g. ``basic_day_ahead_price_...``)
    * ``ig_dir.parents[4]`` → TSO dir      (e.g. ``tennet_de``)

    The function looks for a JSON file named
    ``{dataset_name}_{tso_suffix}.json`` under *dataset_root_dir* and reads
    the ``feature_sets`` key.

    Returns ``None`` when no matching file is found or the file cannot be
    parsed.
    """
    try:
        dataset_name = ig_dir.parents[3].name   # e.g. "basic_day_ahead_price_..."
        tso_dir_name = ig_dir.parents[4].name   # e.g. "tennet_de"
    except IndexError:
        logger.debug(
            "Cannot infer dataset dir from ig_dir %s (too few parent levels)", ig_dir
        )
        return None

    tso_suffix = _DIR_NAME_TO_TSO_FILE_SUFFIX.get(tso_dir_name.lower())
    if tso_suffix is None:
        logger.debug(
            "Unknown TSO directory name '%s'; cannot resolve feature-set JSON.", tso_dir_name
        )
        return None

    json_name = f"{dataset_name}_{tso_suffix}.json"
    json_path = dataset_root_dir / json_name

    if not json_path.exists():
        logger.warning(
            "Dataset metadata JSON not found at %s - skipping feature-set chart.", json_path
        )
        return None

    try:
        with open(json_path) as fh:
            meta = json.load(fh)
        fs = meta.get("feature_sets")
        if isinstance(fs, list) and fs:
            logger.info("Loaded feature_sets from %s: %s", json_path, fs)
            return fs
        logger.warning("No 'feature_sets' list found in %s", json_path)
        return None
    except Exception:
        logger.warning("Failed to read %s", json_path, exc_info=True)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Core analysis runners
# ══════════════════════════════════════════════════════════════════════════════


def _save_fig(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved CSV: %s", path)


def run_overall_importance(
    stacked: np.ndarray,
    feature_names: list[str],
    model_label: str,
    out_dir: Path,
    modes: list[Literal["absolute", "signed"]],
    top_n: int = 25,
    stride_dates: list[pd.Timestamp] | None = None,
) -> None:
    """Run overall importance analysis and save outputs."""
    for mode in modes:
        df = feature_importance_overall(stacked, feature_names, mode=mode)
        _save_csv(df, out_dir / f"overall_importance_{mode}_{model_label}.csv")

        fig, _ = plot_overall_importance_bar(
            df, top_n=top_n,
            title=f"Overall Feature Importance ({mode}) - {model_label}",
        )
        _save_fig(fig, out_dir / f"overall_importance_{mode}_{model_label}.png")

        if mode == "signed" and "pos_share" in df.columns:
            fig_s, _ = plot_signed_importance_bar(
                df, top_n=top_n,
                title=f"Signed Feature Importance - {model_label}",
            )
            _save_fig(fig_s, out_dir / f"signed_importance_{model_label}.png")

    # Yearly breakdown
    if stride_dates is not None and len(stride_dates) == stacked.shape[0]:
        for mode in modes:
            quarterly = feature_importance_quarterly(stacked, feature_names, stride_dates, mode=mode)
            for quarter, df_quarter in quarterly.items():
                _save_csv(
                    df_quarter,
                    out_dir / f"overall_importance_{mode}_{model_label}_{quarter}.csv",
                )
                fig, _ = plot_overall_importance_bar(
                    df_quarter, top_n=top_n,
                    title=f"Feature Importance ({mode}) - {model_label} - {quarter}",
                )
                _save_fig(fig, out_dir / f"overall_importance_{mode}_{model_label}_{quarter}.png")


def run_per_horizon_importance(
    per_horizon_stacked: np.ndarray,
    feature_names: list[str],
    model_label: str,
    out_dir: Path,
    top_k: int = 25,
) -> None:
    """Run per-horizon analysis and save outputs."""
    n_strides, B, H, F = per_horizon_stacked.shape

    # Compute stats
    stats = compute_sample_level_stats(per_horizon_stacked)

    # Per-horizon CSV with top-K features
    df_per_h = feature_importance_per_horizon(stats, feature_names, top_k=top_k)
    _save_csv(df_per_h, out_dir / f"per_horizon_importance_{model_label}.csv")

    # Mean across strides for heatmaps/line plots
    mean_arr = np.mean(per_horizon_stacked, axis=0)  # [B, H, F]

    # Signed heatmap
    fig, _ = plot_per_horizon_heatmap(
        mean_arr, feature_names, batch_idx=0, signed=True,
        title=f"Per-horizon attributions (signed, batch=up) - {model_label}",
    )
    _save_fig(fig, out_dir / f"per_horizon_heatmap_signed_up_{model_label}.png")

    fig, _ = plot_per_horizon_heatmap(
        mean_arr, feature_names, batch_idx=1 if B > 1 else 0, signed=True,
        title=f"Per-horizon attributions (signed, batch=down) - {model_label}",
    )
    _save_fig(fig, out_dir / f"per_horizon_heatmap_signed_down_{model_label}.png")

    # Absolute heatmap
    fig, _ = plot_per_horizon_heatmap(
        mean_arr, feature_names, batch_idx=0, signed=False,
        title=f"Per-horizon attributions (absolute, batch=up) - {model_label}",
    )
    _save_fig(fig, out_dir / f"per_horizon_heatmap_abs_up_{model_label}.png")


    fig, _ = plot_per_horizon_heatmap(
        mean_arr, feature_names, batch_idx=1, signed=False,
        title=f"Per-horizon attributions (absolute, batch=down) - {model_label}",
    )
    _save_fig(fig, out_dir / f"per_horizon_heatmap_abs_down_{model_label}.png")

    # Violin: per-feature attribution distribution across all horizons (pools batches)
    fig, _ = plot_per_horizon_violin(
        mean_arr, feature_names,
        batch_idxs=tuple(range(B)),
        signed=True,
        title=f"Per-horizon attribution distribution (signed) - {model_label}",
    )
    _save_fig(fig, out_dir / f"per_horizon_violin_signed_{model_label}.png")

    fig, _ = plot_per_horizon_violin(
        mean_arr, feature_names,
        batch_idxs=tuple(range(B)),
        signed=False,
        title=f"Per-horizon attribution distribution (absolute) - {model_label}",
    )
    _save_fig(fig, out_dir / f"per_horizon_violin_abs_{model_label}.png")

    # Line plots
    fig, _ = plot_feature_horizon_lines(
        mean_arr, feature_names, top_k=min(5, F),
        title=f"Top features across horizon - {model_label}",
    )
    _save_fig(fig, out_dir / f"per_horizon_lines_{model_label}.png")


def run_cluster_analysis(
    per_horizon_stacked: np.ndarray,
    feature_names: list[str],
    model_label: str,
    out_dir: Path,
    n_clusters: int = 8,
    top_k: int = 25,
    top_f: int = 5,
    top_p: float | None = 80.0,
) -> None:
    """Run hierarchical clustering analysis and save outputs."""
    # Mean across strides
    mean_arr = np.mean(per_horizon_stacked, axis=0)  # [B, H, F]
    B, H, F = mean_arr.shape

    if F < 3:
        logger.warning("Too few features (%d) for clustering - skipping.", F)
        return

    cluster_info = cluster_features_by_horizon(
        mean_arr, feature_names, n_clusters=n_clusters,
    )

    if cluster_info["linkage"] is None:
        logger.warning("Clustering failed (no non-constant features) - skipping.")
        return

    # Save cluster membership
    _save_csv(
        cluster_info["feature_cluster_map"],
        out_dir / f"cluster_membership_{model_label}.csv",
    )

    # Save per-cluster detailed importance CSV
    mean_abs_all: np.ndarray = np.abs(mean_arr).mean(axis=(0, 1))  # [F]
    total_abs = float(mean_abs_all.sum()) or 1.0
    labels = cluster_info["labels"]
    cluster_names = cluster_info["cluster_names"]
    actual_k = len(cluster_names)

    cluster_rows = []
    for c in range(actual_k):
        cluster_idx = np.where(labels == c)[0]
        if len(cluster_idx) == 0:
            continue
        # Sort features in this cluster by descending attribution
        sorted_idx = cluster_idx[np.argsort(mean_abs_all[cluster_idx])[::-1]]
        cluster_total = float(mean_abs_all[cluster_idx].sum())
        cluster_pct = cluster_total / total_abs * 100

        for rank, feat_idx in enumerate(sorted_idx, start=1):
            cluster_rows.append({
                "cluster_id":        c,
                "cluster_name":      cluster_names[c],
                "cluster_pct_total": round(cluster_pct, 2),
                "feature":           feature_names[feat_idx],
                "rank_in_cluster":   rank,
                "mean_abs":          round(float(mean_abs_all[feat_idx]), 6),
                "pct_of_total":      round(float(mean_abs_all[feat_idx]) / total_abs * 100, 2),
                "pct_of_cluster":    round(float(mean_abs_all[feat_idx]) / cluster_total * 100, 2) if cluster_total > 0 else 0.0,
            })

    if cluster_rows:
        _save_csv(
            pd.DataFrame(cluster_rows),
            out_dir / f"cluster_importance_{model_label}.csv",
        )

    # Cluster-level attribution heatmap (unchanged)
    fig, _ = plot_cluster_heatmap(
        cluster_info, batch_idx=0,
        title=f"Cluster attributions (batch=up) - {model_label}",
    )
    _save_fig(fig, out_dir / f"cluster_heatmap_up_{model_label}.png")

    if B > 1:
        fig, _ = plot_cluster_heatmap(
            cluster_info, batch_idx=1,
            title=f"Cluster attributions (batch=down) - {model_label}",
        )
        _save_fig(fig, out_dir / f"cluster_heatmap_down_{model_label}.png")

    # ── Plot 1: standalone dendrogram (cluster-coloured leaves) ──────────
    fig_dendro = plot_dendrogram(
        cluster_info, feature_names,
        title=f"Feature dendrogram - {model_label}",
    )
    _save_fig(fig_dendro, out_dir / f"cluster_dendrogram_{model_label}.png")

    # ── Plot 2: per-cluster top-feature bars (separate canvas) ─────────
    fig_top = plot_cluster_top_features(
        mean_arr, feature_names, cluster_info,
        top_f=top_f,
        top_p=top_p,
        title=f"Top features per cluster (batch=up) - {model_label}",
    )
    _save_fig(fig_top, out_dir / f"cluster_top_features_up_{model_label}.png")

    if B > 1:
        # For the down-batch, flip axis 0 to recompute with batch_idx=1
        fig_top_dn = plot_cluster_top_features(
            mean_arr, feature_names, cluster_info,
            top_f=top_f,
            top_p=top_p,
            title=f"Top features per cluster (batch=down) - {model_label}",
        )
        _save_fig(fig_top_dn, out_dir / f"cluster_top_features_down_{model_label}.png")


def run_feature_set_analysis(
    per_horizon_stacked: np.ndarray,
    feature_names: list[str],
    feature_sets: list[str],
    model_label: str,
    out_dir: Path,
    top_f: int = 10,
    top_p: float | None = 80.0,
) -> None:
    """
    Per-feature-set importance bar chart - one subplot per data source.

    For each feature set (e.g. ``"wind_pv"``, ``"bloomberg"``, ...) a subplot
    is drawn showing **all** features of that set sorted by mean absolute
    attribution, with horizon-std error bars.  Results are saved as:

    * ``feature_set_bars_up_{model_label}.png``   (batch direction = up)
    * ``feature_set_bars_down_{model_label}.png`` (batch direction = down,
      only when B > 1)
    * ``feature_set_importance_{model_label}.csv``  (per-set aggregate stats)

    Parameters
    ----------
    per_horizon_stacked : np.ndarray, shape [n_strides, B, H, F]
        Per-horizon attribution tensor (stacked over all IG strides).
    feature_names : list[str]
        Feature names in F-order.
    feature_sets : list[str]
        Ordered feature set names (from wandb config or metadata JSON).
    model_label : str
        Model alias used in file names and plot titles.
    out_dir : Path
        Directory where outputs are saved.
    """
    mean_arr = np.mean(per_horizon_stacked, axis=0)   # [B, H, F]
    B = mean_arr.shape[0]

    feature_set_map = build_feature_to_feature_set_map(feature_names, feature_sets)

    # ── Per-batch charts ────────────────────────────────────────────────────
    batch_specs = [("up", 0)]
    if B > 1:
        batch_specs.append(("down", 1))

    for b_label, b_idx in batch_specs:
        arr_1bhf = mean_arr[b_idx : b_idx + 1]   # [1, H, F] → proper per-batch stats
        fig = plot_feature_set_bars(
            arr_1bhf,
            feature_names,
            feature_set_map,
            feature_sets,
            top_f=top_f,
            top_p=top_p,
            title=f"Feature-Set Importance (batch={b_label}) - {model_label}",
        )
        _save_fig(fig, out_dir / f"feature_set_bars_{b_label}_{model_label}.png")

    # ── Aggregate CSV ───────────────────────────────────────────────────────
    mean_abs_all: np.ndarray = np.abs(mean_arr).mean(axis=(0, 1))   # [F]
    total_abs = float(mean_abs_all.sum()) or 1.0

    # Build per-feature-set summary - include structural sets (static_exog,
    # data_driven, runlength) that may not appear in the metadata feature_sets.
    _all_sets_in_map = set(feature_set_map.values())
    csv_sets: list[str] = [s for s in _CANONICAL_SET_ORDER if s in _all_sets_in_map]
    for fs in feature_sets:
        if fs not in csv_sets:
            csv_sets.append(fs)
    rows = []
    for fs in csv_sets:
        idx_c = [i for i, f in enumerate(feature_names) if feature_set_map.get(f) == fs]
        if not idx_c:
            continue
        set_abs = float(mean_abs_all[idx_c].sum())
        rows.append({
            "feature_set":       fs,
            "n_features":        len(idx_c),
            "mean_abs_total":    round(set_abs, 6),
            "pct_of_total":      round(set_abs / total_abs * 100, 2),
        })
    if rows:
        _save_csv(
            pd.DataFrame(rows),
            out_dir / f"feature_set_importance_{model_label}.csv",
        )

    # Save per-feature detailed importance CSV
    feature_rows = []
    for feat_idx, feat_name in enumerate(feature_names):
        fs = feature_set_map.get(feat_name, "unknown")
        feat_abs = float(mean_abs_all[feat_idx])
        feature_rows.append({
            "feature":        feat_name,
            "feature_set":    fs,
            "mean_abs":       round(feat_abs, 6),
            "pct_of_total":   round(feat_abs / total_abs * 100, 2),
        })

    # Sort by feature set (canonical order) then by descending attribution
    def set_sort_key(row):
        fs = row["feature_set"]
        if fs in _CANONICAL_SET_ORDER:
            set_rank = _CANONICAL_SET_ORDER.index(fs)
        else:
            set_rank = len(_CANONICAL_SET_ORDER)
        return (set_rank, -row["mean_abs"])

    feature_rows.sort(key=set_sort_key)

    if feature_rows:
        _save_csv(
            pd.DataFrame(feature_rows),
            out_dir / f"feature_set_features_{model_label}.csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════


def run_explainability_analysis(
    output_dir: Path,
    model_root: Path | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    model_filter: str | None = None,
    modes: list[Literal["absolute", "signed"]] | None = None,
    futr_agg: Literal["hist", "futr", "combined"] = "combined",
    top_k: int = 25,
    n_clusters: int = 8,
    top_f: int = 5,
    top_p: float | None = 80.0,
    skip_overall: bool = False,
    skip_per_horizon: bool = False,
    skip_clustering: bool = False,
    dataset_root_dir: Path | None = None,
    start_window: int = 0,
    dpi: int = 150,
) -> None:
    """
    End-to-end post-hoc explainability from saved IG tensors.

    Supports both **single-window** and **rolling-window** training runs:

    * **Single-window** – pass the run directory (e.g.
      ``outputs/.../2026-02-15_02-05-44/``).  IG tensors are read from
      ``<output_dir>/evaluation/ig_raw/`` and results are saved under
      ``<output_dir>/explainability_results/``.

    * **Rolling-window** – pass the *parent* directory that contains
      ``window_0/``, ``window_1/``, ..., and ``evaluation/`` (e.g.
      ``outputs/tennet_de/basic_.../``).  Windows are sorted numerically,
      their IG strides are concatenated into a **single per-model** analysis
      (no per-window split), and results are saved under
      ``<output_dir>/explainability_results/``.

    Parameters
    ----------
    output_dir : Path
        * Rolling-window: parent dir containing ``window_N/`` subdirs
          **and** ``evaluation/ig_raw/``.
        * Single-window: run dir containing ``evaluation/ig_raw/``.
    model_root : Path | None
        Root directory where NeuralForecast checkpoints live. Needed to
        resolve feature names from model hparams. If ``None``, the pipeline
        will attempt to infer from wandb config.
    wandb_project : str | None
        W&B project (used to look up feature names when model_root is absent).
    wandb_entity : str | None
        W&B entity/org.
    model_filter : str | None
        Only process IG dirs whose name contains this substring.
    modes : list of "absolute" and/or "signed"
        Attribution aggregation modes to run (default: both).
    futr_agg : "hist", "futr", or "combined"
        How to treat future covariates before the forecast horizon.
    top_k : int
        Number of top features to include in per-horizon analysis.
    n_clusters : int
        Number of clusters for hierarchical feature clustering.
    top_f : int
        Maximum number of features shown per cluster in the top-feature
        chart (hard cap; default 5).
    top_p : float | None
        Attribution coverage threshold in % per cluster (default 80.0).
        Features are added until their cumulative attribution reaches this
        fraction of the cluster total.  ``None`` shows exactly ``top_f``
        features per cluster.
    skip_overall : bool
        Skip overall importance analysis.
    skip_per_horizon : bool
        Skip per-horizon importance analysis.
    skip_clustering : bool
        Skip feature clustering analysis.
    dataset_root_dir : Path | None
        Root directory that contains dataset metadata JSON files named
        ``{dataset_feature_set}_{tso_normalized}.json`` (e.g.
        ``data/model_data_new_features_debug/``).  Used as a fallback to
        resolve the ``feature_sets`` list when the wandb run config does not
        contain ``dataset_metadata.feature_sets``.  If ``None`` and the
        wandb config is also unavailable the per-feature-set bar chart is
        silently skipped.
    dpi : int
        DPI for saved figures.
    """
    if modes is None:
        modes = ["absolute", "signed"]

    output_dir = Path(output_dir)
    results_dir = output_dir / "explainability_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = results_dir / "explainability_pipeline.log"
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("Starting explainability pipeline")
    logger.info("Output dir: %s", output_dir)
    logger.info("Results dir: %s", results_dir)
    logger.info("Modes: %s | futr_agg: %s | top_k: %d | n_clusters: %d",
                modes, futr_agg, top_k, n_clusters)
    logger.info("=" * 80)

    # ── Discover IG directories ───────────────────────────────────────────────
    # For rolling-window runs the caller should pass the *parent* directory
    # (containing ``window_0/``, ``window_1/``, ..., ``evaluation/``); the
    # pipeline will look for ``<output_dir>/evaluation/ig_raw/``.
    # For single-window runs pass the run directory; the pipeline looks for
    # ``<output_dir>/evaluation/ig_raw/`` and falls back to
    # ``<output_dir>/ig_raw/``.
    ig_map: dict[str, list[Path]] = {}
    try:
        ig_map = discover_ig_directories(output_dir / "evaluation", model_filter=model_filter, start_window=start_window)
    except FileNotFoundError:
        pass
    if not ig_map:
        try:
            # Fallback: ig_raw directly under output_dir (legacy layout)
            ig_map = discover_ig_directories(output_dir, model_filter=model_filter, start_window=start_window)
        except FileNotFoundError:
            pass
    if not ig_map:
        logger.error("No ig_raw directories found under %s", output_dir)
        return

    logger.info("Discovered %d model(s): %s", len(ig_map), list(ig_map.keys()))

    static_df = build_static_df()

    run_configs_cache: dict[str, dict] = {}
    nf_cache: dict[str, NeuralForecast] = {}

    # ── Process each model ────────────────────────────────────────────────────
    for base_alias, ig_dirs in ig_map.items():
        logger.info("─" * 60)
        logger.info("Processing model: %s (%d window(s))", base_alias, len(ig_dirs))

        model_results_dir = results_dir / base_alias
        model_results_dir.mkdir(parents=True, exist_ok=True)

        window_index = _extract_window_index(ig_dirs[0], model_path=model_root if model_root is not None else output_dir)
        if window_index is not None:
            logger.info("Extracted window index: %s", window_index)

        for ig_dir in ig_dirs:
            # Try to log parameters for at least one window
            # Assumption: all windows for the same model have the same feature names, feature sets, and parameters, so it's sufficient to resolve from one IG directory.
            if base_alias in run_configs_cache or base_alias in nf_cache:
                break  # already resolved for this model
            logger.info("IG directory: %s", ig_dir)

            # Load model to get feature names from hparams
            # Try wandb first, if not then locally
            nf_model = None
            run_config: dict | None = None

            if wandb_project:
                run_config = _extract_model_hparams_from_wandb(
                    base_alias, model_root, wandb_entity, wandb_project, window_index=window_index
                )

            if run_config is None and model_root is not None:
                nf = _resolve_model_for_ig_dir(
                    ig_dir, model_root, wandb_entity, wandb_project, output_dir,
                )
                if nf is not None:
                    nf_model = nf.models[0]
                    if base_alias not in nf_cache:
                        nf_cache[base_alias] = nf_model
                    logger.info("Model loaded for %s (window index: %s)", base_alias, window_index)
            elif run_config is not None:
                if base_alias not in run_configs_cache:
                    run_configs_cache[base_alias] = run_config
                logger.info("Wandb config loaded for %s (window index: %s)", base_alias, window_index)

            if run_config is None and nf_model is None:
                logger.warning(
                    "Could not resolve model or wandb config for %s. "
                    "Feature names cannot be extracted and this file's analysis will be skipped.",
                    base_alias,
                )

        # ── Resolve feature_sets (for per-feature-set bar chart) ─────────────
        # Primary: extract from wandb run config dataset_metadata.
        # Fallback: navigate the directory tree and read the dataset metadata JSON.
        feature_sets: list[str] | None = None
        _cached_run_config = run_configs_cache.get(base_alias)
        if _cached_run_config:
            feature_sets = _extract_feature_sets_from_run_config(_cached_run_config)
        if feature_sets is None and dataset_root_dir is not None:
            # Use the first ig_dir as anchor for directory navigation
            feature_sets = _resolve_feature_sets_from_dir(ig_dirs[0], dataset_root_dir)
        if feature_sets is None:
            logger.warning(
                "feature_sets not resolved for %s - per-feature-set chart will be skipped. "
                "Pass --dataset-root-dir or ensure wandb run config has dataset_metadata.",
                base_alias,
            )
        else:
            logger.info("Resolved feature_sets for %s: %s", base_alias, feature_sets)

        # ── Sort window directories numerically, then load IG tensors ──────────
        ig_dirs_with_idx = _sort_ig_dirs_by_window(ig_dirs, model_path=model_root if model_root is not None else output_dir)
        logger.info(
            "Window order for %s: %s",
            base_alias,
            [(idx, d.name) for idx, d in ig_dirs_with_idx],
        )

        all_explanations: list[dict[str, torch.Tensor]] = []
        window_stride_counts: list[int] = []
        for _, ig_dir in ig_dirs_with_idx:
            window_strides = load_ig_tensors(ig_dir)
            window_stride_counts.append(len(window_strides))
            all_explanations.extend(window_strides)

        if not all_explanations:
            logger.warning("No strides loaded for %s - skipping.", base_alias)
            continue

        logger.info("Loaded %d total strides for %s", len(all_explanations), base_alias)

        # ── Resolve feature names ────────────────────────────────────────────
        # We need a model-like object to extract feature names.
        # If the actual NF model isn't available, build a lightweight proxy.
        nf_model = nf_cache.get(base_alias)
        run_config = run_configs_cache.get(base_alias)
        if nf_model is None:
            nf_model = _build_hparams_proxy(all_explanations[0], run_config, static_df)
            if nf_model is None:
                logger.error(
                    "Cannot determine feature names for %s. "
                    "Provide --model-path or --wandb-project.",
                    base_alias,
                )
                continue

        # ── OVERALL IMPORTANCE ────────────────────────────────────────────────
        if not skip_overall:
            logger.info("Computing overall importance for %s ...", base_alias)
            try:
                all_values: list[np.ndarray] = []
                feature_names: list[str] = []
                for explanations in all_explanations:
                    vals, names = combine_attributions(
                        explanations=explanations,
                        nf_model=nf_model,
                        static_df=static_df,
                        mode="signed",  # signed preserves info for both modes
                        futr_agg=futr_agg,
                    )
                    all_values.append(vals[:-1])  # drop baseline
                    if not feature_names:
                        feature_names = names

                stacked = np.stack(all_values, axis=0)  # [n_strides, n_features, B]

                if run_config is None:
                    run_config = {}
                # Use rolling-window date generation when there are multiple
                # windowed directories (each window covers 1 calendar month
                # starting at test_start + N months, so that quarterly
                # breakdowns reflect model evolution over time rather than
                # spilling far beyond the actual test period).
                is_rolling = (
                    len(ig_dirs_with_idx) > 1
                    and any(idx is not None for idx, _ in ig_dirs_with_idx)
                )
                if is_rolling:
                    dates = _build_rolling_window_stride_dates(
                        ig_dirs_with_idx=ig_dirs_with_idx,
                        n_strides_per_window=window_stride_counts,
                        run_config=run_config,
                    )
                else:
                    dates = get_stride_dates_from_wandb(
                        run_config=run_config,
                        n_strides=stacked.shape[0],
                    )
                run_overall_importance(
                    stacked=stacked,
                    feature_names=feature_names,
                    model_label=base_alias,
                    out_dir=model_results_dir,
                    modes=modes,
                    top_n=top_k,
                    stride_dates=dates,
                )
                logger.info("✓ Overall importance complete for %s", base_alias)
            except Exception:
                logger.exception("Error computing overall importance for %s", base_alias)

        # ── PER-HORIZON IMPORTANCE ────────────────────────────────────────────
        per_horizon_stacked: np.ndarray | None = None
        # Build the per-horizon matrix when needed by any downstream analysis
        # (per-horizon plots, clustering, or feature-set bars).
        need_per_horizon = (
            not skip_per_horizon
            or not skip_clustering
            or feature_sets is not None
        )
        if need_per_horizon:
            logger.info("Building per-horizon feature matrix for %s ...", base_alias)
            try:
                per_h_list: list[np.ndarray] = []
                ph_names: list[str] = []
                for explanations in all_explanations:
                    arr, names = build_per_horizon_feature_matrix(
                        explanations=explanations,
                        nf_model=nf_model,
                        static_df=static_df,
                        futr_agg=futr_agg,
                    )
                    per_h_list.append(arr)
                    if not ph_names:
                        ph_names = names

                per_horizon_stacked = np.stack(per_h_list, axis=0)  # [n_strides, B, H, F]
                logger.info(
                    "Per-horizon matrix shape: %s (strides=%d, B=%d, H=%d, F=%d)",
                    per_horizon_stacked.shape, *per_horizon_stacked.shape,
                )
            except Exception:
                logger.exception("Error building per-horizon matrix for %s", base_alias)

        if per_horizon_stacked is not None and not skip_per_horizon:
            try:
                run_per_horizon_importance(
                    per_horizon_stacked=per_horizon_stacked,
                    feature_names=ph_names,
                    model_label=base_alias,
                    out_dir=model_results_dir,
                    top_k=top_k,
                )
                logger.info("✓ Per-horizon importance complete for %s", base_alias)
            except Exception:
                logger.exception("Error in per-horizon analysis for %s", base_alias)

        # ── CLUSTERING ────────────────────────────────────────────────────────
        if per_horizon_stacked is not None and not skip_clustering:
            try:
                run_cluster_analysis(
                    per_horizon_stacked=per_horizon_stacked,
                    feature_names=ph_names,
                    model_label=base_alias,
                    out_dir=model_results_dir,
                    n_clusters=n_clusters,
                    top_k=top_k,
                    top_f=top_f,
                    top_p=top_p,
                )
                logger.info("✓ Clustering complete for %s", base_alias)
            except Exception:
                logger.exception("Error in cluster analysis for %s", base_alias)

        # ── FEATURE-SET BARS ──────────────────────────────────────────────────
        if per_horizon_stacked is not None and feature_sets is not None:
            try:
                run_feature_set_analysis(
                    per_horizon_stacked=per_horizon_stacked,
                    feature_names=ph_names,
                    feature_sets=feature_sets,
                    model_label=base_alias,
                    out_dir=model_results_dir,
                    top_f=top_f,
                    top_p=top_p,
                )
                logger.info("✓ Feature-set analysis complete for %s", base_alias)
            except Exception:
                logger.exception("Error in feature-set analysis for %s", base_alias)

        logger.info("Done with model %s", base_alias)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("Explainability pipeline complete. Results in: %s", results_dir)
    logger.info("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  Lightweight model proxy (when checkpoint is unavailable)
# ══════════════════════════════════════════════════════════════════════════════


class _HParamsProxy:
    """Minimal duck-type substitute for a NeuralForecast model object,
    providing just enough ``hparams`` for the aggregation utilities."""

    def __init__(self, hparams: dict):
        self.hparams = hparams


def _build_hparams_proxy(
    sample_explanation: dict[str, torch.Tensor],
    run_config: dict | None,
    static_df: pd.DataFrame,
) -> _HParamsProxy | None:
    """
    Build a minimal model proxy from a sample IG stride and wandb config.

    Uses the tensor shapes + run config to infer feature lists and input_size.
    """
    # Infer input_size from insample tensor
    insample = sample_explanation.get("insample")
    if insample is None:
        return None
    input_size = insample.shape[-2]  # [..., input_size, 2]

    # Infer futr/hist feature counts from tensor shapes
    hist_exog = sample_explanation.get("hist_exog")
    futr_exog = sample_explanation.get("futr_exog")
    stat_exog = sample_explanation.get("stat_exog")

    n_hist = hist_exog.shape[-1] if hist_exog is not None else 0
    n_futr = futr_exog.shape[-1] if futr_exog is not None else 0
    n_stat = stat_exog.shape[-1] if stat_exog is not None else 0

    # Get feature names from run_config if available
    hist_exog_list: list[str] = []
    futr_exog_list: list[str] = []

    if run_config:
        hist_exog_list = run_config.get("historical_covariates", []) or []
        futr_exog_list = run_config.get("future_covariates", []) or []

    # Pad with generic names if config doesn't have enough
    while len(hist_exog_list) < n_hist:
        hist_exog_list.append(f"hist_feat_{len(hist_exog_list)}")
    while len(futr_exog_list) < n_futr:
        futr_exog_list.append(f"futr_feat_{len(futr_exog_list)}")

    hparams = {
        "input_size": input_size,
        "hist_exog_list": hist_exog_list[:n_hist],
        "futr_exog_list": futr_exog_list[:n_futr],
    }

    logger.info(
        "Built hparams proxy: input_size=%d, n_hist=%d, n_futr=%d, n_stat=%d",
        input_size, n_hist, n_futr, n_stat,
    )
    return _HParamsProxy(hparams)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-hoc explainability analysis from saved IG tensors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--output-dir", type=str, required=True,
        help=(
            "For rolling-window runs: parent directory containing window_N/ "
            "subdirs and evaluation/ig_raw/. "
            "For single-window runs: run directory containing evaluation/ig_raw/. "
            "Results are always saved under <output_dir>/explainability_results/."
        ),
    )
    p.add_argument(
        "--start-window", type=int, default=0,
        help=(
            "For rolling-window runs, only process windows with index >= this value. "
            "Ignored for single-window runs. Default: 0 (process all windows)."
        )
    )
    p.add_argument(
        "--model-path", type=str, default=None,
        help="Root directory of NF model checkpoints (for hparams resolution).",
    )
    p.add_argument(
        "--wandb-project", type=str, default=None,
        help="W&B project name (alternative to --model-path for feature names).",
    )
    p.add_argument(
        "--wandb-entity", type=str, default=None,
        help="W&B entity/organization.",
    )
    p.add_argument(
        "--model-filter", type=str, default=None,
        help="Only process models whose alias contains this substring.",
    )
    p.add_argument(
        "--mode", nargs="+",
        choices=["absolute", "signed"], default=["absolute", "signed"],
        help="Attribution aggregation modes (default: both).",
    )
    p.add_argument(
        "--futr-agg",
        choices=["hist", "futr", "combined"], default="combined",
        help=(
            "How to aggregate future covariate tokens: "
            "'combined' (default) treats pre-horizon tokens as historical."
        ),
    )
    p.add_argument(
        "--top-k", type=int, default=25,
        help="Number of top features for per-horizon tables & plots (default: 25).",
    )
    p.add_argument(
        "--top-f", type=int, default=5,
        help="Max features per cluster in the top-feature chart (default: 5).",
    )
    p.add_argument(
        "--top-p", type=float, default=80.0,
        help=(
            "Attribution coverage threshold %% per cluster (default: 80.0). "
            "Features are added until they explain this fraction of the cluster ’s "
            "total attribution. Set to 0 to show exactly --top-f features per cluster."
        ),
    )
    p.add_argument(
        "--n-clusters", type=int, default=8,
        help="Number of clusters for hierarchical feature clustering (default: 8).",
    )
    p.add_argument("--skip-overall", action="store_true", help="Skip overall importance analysis.")
    p.add_argument("--skip-per-horizon", action="store_true", help="Skip per-horizon analysis.")
    p.add_argument("--skip-clustering", action="store_true", help="Skip clustering analysis.")
    p.add_argument(
        "--dataset-root-dir", type=str, default=None,
        help=(
            "Directory containing dataset metadata JSON files named "
            "'{dataset_feature_set}_{tso_normalized}.json' "
            "(e.g. 'data/model_data_new_features_debug/'). "
            "Used as a fallback to obtain the feature_sets list when the "
            "wandb run config does not contain dataset_metadata. "
            "If omitted and wandb config is also unavailable, the "
            "per-feature-set bar chart is silently skipped."
        ),
    )
    p.add_argument("--dpi", type=int, default=150, help="DPI for saved figures (default: 150).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_explainability_analysis(
        output_dir=Path(args.output_dir),
        model_root=Path(args.model_path) if args.model_path else None,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        model_filter=args.model_filter,
        modes=args.mode,
        futr_agg=args.futr_agg,
        top_k=args.top_k,
        n_clusters=args.n_clusters,
        top_f=args.top_f,
        top_p=args.top_p if args.top_p > 0 else None,
        skip_overall=args.skip_overall,
        skip_per_horizon=args.skip_per_horizon,
        skip_clustering=args.skip_clustering,
        dataset_root_dir=Path(args.dataset_root_dir) if args.dataset_root_dir else None,
        start_window=args.start_window,
        dpi=args.dpi,
    )
