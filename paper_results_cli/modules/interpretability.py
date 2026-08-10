"""Paper-ready Integrated Gradients feature-group summaries.

Outputs
-------
tables/global_feature_group_importance.csv
tables/global_feature_group_importance.tex
tables/horizon_group_attribution_shares.csv
tables/top_feature_attribution_shares.csv
tables/global_top_feature_importance.tex
tables/horizon_top_feature_importance.tex
figures/horizon_group_attribution_heatmap.{png,pdf}
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from training.data_prep import CALENDAR_COLS, classify_covariates
from training.explainability import (
    build_per_horizon_feature_matrix,
    discover_ig_directories,
    load_ig_tensors,
)

from ..config import PaperConfig
from ..style import save_figure
from ..tables import add_index_names, escape_latex, save_latex_table

logger = logging.getLogger(__name__)

TSO_SUFFIXES: dict[str, str] = {
    "50hertz": "50Hertz",
    "amprion": "Amprion",
    "transnetbw": "TransnetBW",
    "tennet_de": "TenneT_DE",
}

TSO_DISPLAY: dict[str, str] = {
    "50hertz": "50Hertz",
    "amprion": "Amprion",
    "transnetbw": "TransnetBW",
    "tennet_de": "TenneT_DE",
}

GROUP_ORDER: list[str] = [
    "target_lags/basic",
    "runlength",
    "data_driven",
    "calendar",
    "wind_pv",
    "production_consumption",
    "sce",
    "cross_border",
    "day_ahead_price",
]

PLOT_GROUP_LABELS: dict[str, str] = {
    "target_lags/basic": "target lags\n/basic",
    "runlength": "runlength",
    "data_driven": "data driven",
    "calendar": "calendar",
    "wind_pv": "wind/PV",
    "production_consumption": "production\nconsumption",
    "sce": "SCE",
    "cross_border": "cross-border",
    "day_ahead_price": "day-ahead\nprice",
}

HORIZON_BLOCKS: dict[str, tuple[int, int]] = {
    "early": (1, 9),
    "middle": (10, 17),
    "late": (18, 24),
}


@dataclass(frozen=True)
class InterpretabilityInputs:
    ig_root: Path
    dataset_root: Path
    dataset_name: str
    model_filter: str
    start_window: int
    best_checkpoint: bool
    exclude_groups: tuple[str, ...]


def _metadata_path(dataset_root: Path, dataset_name: str, tso_key: str) -> Path:
    return dataset_root / f"{dataset_name}_{TSO_SUFFIXES[tso_key]}.json"


def _load_metadata(dataset_root: Path, dataset_name: str, tso_key: str) -> dict:
    path = _metadata_path(dataset_root, dataset_name, tso_key)
    if not path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {path}")
    return json.loads(path.read_text())


def _infer_hparams(metadata: dict, input_size: int) -> SimpleNamespace:
    columns = list(metadata["columns"])
    future_covariates, historical_covariates = classify_covariates(columns)
    future_covariates = future_covariates + CALENDAR_COLS
    return SimpleNamespace(
        hparams={
            "hist_exog_list": historical_covariates,
            "futr_exog_list": future_covariates,
            "input_size": input_size,
        }
    )


def _group_feature(feature: str) -> str:
    if feature.startswith("stat_exog_"):
        return "static_covariates"

    bare = feature
    for prefix in ("hist_exog_", "futr_exog_"):
        if feature.startswith(prefix):
            bare = feature[len(prefix):]
            break

    if bare.startswith("y_lag"):
        return "target_lags/basic"
    if bare in CALENDAR_COLS:
        return "calendar"
    if bare.startswith("bloomberg_"):
        return "bloomberg"
    if bare.startswith("runlength_") or bare == "runlength":
        return "runlength"
    if bare.startswith("data_driven_") or bare == "data_driven":
        return "data_driven"
    if bare.startswith("sce_forecast_"):
        return "sce"
    if bare.startswith("cross_border_"):
        return "cross_border"
    if bare.startswith("day_ahead_price"):
        return "day_ahead_price"
    if bare.startswith(("wind_forecast_", "pv_forecast_", "wind_actual_", "pv_actual_")):
        return "wind_pv"
    if bare.startswith(("production_", "consumption_")) or "residual_load" in bare:
        return "production_consumption"
    return "target_lags/basic"


def _available_groups(feature_names: list[str], exclude_groups: tuple[str, ...]) -> list[str]:
    present = {_group_feature(f) for f in feature_names} - set(exclude_groups)
    ordered = [g for g in GROUP_ORDER if g in present]
    return ordered + sorted(present - set(ordered))


def _group_share_tensor(
    arr: np.ndarray,
    feature_names: list[str],
    groups: list[str],
    exclude_groups: tuple[str, ...],
) -> np.ndarray:
    """Return per-sample relative group reliance with shape [sample, B, H, G]."""
    abs_arr = np.abs(arr)
    group_arr = np.zeros((*abs_arr.shape[:3], len(groups)), dtype=float)
    group_index = {g: i for i, g in enumerate(groups)}

    for feature_idx, feature_name in enumerate(feature_names):
        group = _group_feature(feature_name)
        if group in exclude_groups or group not in group_index:
            continue
        group_arr[..., group_index[group]] += abs_arr[..., feature_idx]

    denominator = group_arr.sum(axis=-1, keepdims=True)
    return np.divide(
        group_arr,
        denominator,
        out=np.zeros_like(group_arr),
        where=denominator > 0,
    )




def _feature_share_tensor(
    arr: np.ndarray,
    feature_names: list[str],
    exclude_groups: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    """Return per-sample relative feature reliance with shape [sample, B, H, F]."""
    keep_indices = [
        i for i, feature_name in enumerate(feature_names)
        if _group_feature(feature_name) not in exclude_groups
    ]
    kept_features = [feature_names[i] for i in keep_indices]
    abs_arr = np.abs(arr[..., keep_indices])
    denominator = abs_arr.sum(axis=-1, keepdims=True)
    shares = np.divide(
        abs_arr,
        denominator,
        out=np.zeros_like(abs_arr),
        where=denominator > 0,
    )
    return shares, kept_features


def _feature_display_name(feature: str) -> str:
    for prefix in ("hist_exog_", "futr_exog_", "stat_exog_"):
        if feature.startswith(prefix):
            return feature[len(prefix):]
    return feature


def _load_tso_direction_shares(
    inputs: InterpretabilityInputs,
    tso_key: str,
    input_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    metadata = _load_metadata(inputs.dataset_root, inputs.dataset_name, tso_key)
    model_stub = _infer_hparams(metadata, input_size=input_size)
    static_df = pd.DataFrame({
        "unique_id": ["up", "down"],
        "direction_0": [1, 0],
        "direction_1": [0, 1],
    })

    output_dir = inputs.ig_root / tso_key / inputs.dataset_name
    ig_dirs = discover_ig_directories(
        output_dir,
        model_filter=inputs.model_filter,
        start_window=inputs.start_window,
        best_checkpoint=inputs.best_checkpoint,
    )
    if not ig_dirs:
        raise FileNotFoundError(f"No IG directories found under {output_dir}")

    per_stride: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for model_name, model_dirs in ig_dirs.items():
        logger.info("Loading %s/%s from %d window directories", tso_key, model_name, len(model_dirs))
        for ig_dir in sorted(model_dirs, key=lambda p: str(p)):
            for stride in load_ig_tensors(ig_dir):
                arr, names = build_per_horizon_feature_matrix(stride, model_stub, static_df)
                if feature_names is None:
                    feature_names = names
                elif names != feature_names:
                    raise ValueError(f"Feature names changed while reading {ig_dir}")
                per_stride.append(arr)

    if not per_stride or feature_names is None:
        raise ValueError(f"No IG strides loaded for {tso_key}")

    stacked = np.stack(per_stride, axis=0)
    groups = _available_groups(feature_names, inputs.exclude_groups)
    shares = _group_share_tensor(stacked, feature_names, groups, inputs.exclude_groups)
    feature_shares, kept_features = _feature_share_tensor(
        stacked,
        feature_names,
        inputs.exclude_groups,
    )

    group_rows: list[dict] = []
    for b_idx, direction in enumerate(["up", "down"][: shares.shape[1]]):
        for h_idx in range(shares.shape[2]):
            horizon = h_idx + 1
            for g_idx, group in enumerate(groups):
                group_rows.append({
                    "tso": TSO_DISPLAY[tso_key],
                    "direction": direction,
                    "horizon": horizon,
                    "group": group,
                    "share": float(shares[:, b_idx, h_idx, g_idx].mean()),
                    "n_predictions": int(shares.shape[0]),
                })

    feature_rows: list[dict] = []
    for b_idx, direction in enumerate(["up", "down"][: feature_shares.shape[1]]):
        for h_idx in range(feature_shares.shape[2]):
            horizon = h_idx + 1
            for f_idx, feature in enumerate(kept_features):
                feature_rows.append({
                    "tso": TSO_DISPLAY[tso_key],
                    "direction": direction,
                    "horizon": horizon,
                    "feature": _feature_display_name(feature),
                    "raw_feature": feature,
                    "group": _group_feature(feature),
                    "share": float(feature_shares[:, b_idx, h_idx, f_idx].mean()),
                    "n_predictions": int(feature_shares.shape[0]),
                })

    return pd.DataFrame(group_rows), pd.DataFrame(feature_rows), groups


def build_horizon_group_shares(
    inputs: InterpretabilityInputs,
    tso_keys: list[str] | None = None,
    input_size: int = 24,
) -> pd.DataFrame:
    tso_keys = tso_keys or list(TSO_SUFFIXES)
    frames = [
        _load_tso_direction_shares(inputs, tso_key=tso_key, input_size=input_size)[0]
        for tso_key in tso_keys
    ]
    shares = pd.concat(frames, ignore_index=True)
    shares["horizon_block"] = pd.cut(
        shares["horizon"],
        bins=[0, 9, 17, 24],
        labels=["early", "middle", "late"],
    ).astype(str)
    return (
        shares.groupby(["tso", "direction", "horizon_block", "group"], as_index=False)
        .agg(share=("share", "mean"), n_predictions=("n_predictions", "sum"))
    )


def build_top_feature_shares(
    inputs: InterpretabilityInputs,
    *,
    tso_keys: list[str] | None = None,
    input_size: int = 24,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the top-N feature attribution shares overall and by horizon block."""
    tso_keys = tso_keys or list(TSO_SUFFIXES)
    frames = [
        _load_tso_direction_shares(inputs, tso_key=tso_key, input_size=input_size)[1]
        for tso_key in tso_keys
    ]
    feature_shares = pd.concat(frames, ignore_index=True)
    feature_shares["horizon_block"] = pd.cut(
        feature_shares["horizon"],
        bins=[0, 9, 17, 24],
        labels=["early", "middle", "late"],
    ).astype(str)

    block = (
        feature_shares
        .groupby(["tso", "direction", "horizon_block", "feature", "raw_feature", "group"], as_index=False)
        .agg(share=("share", "mean"), n_predictions=("n_predictions", "sum"))
    )
    overall = (
        feature_shares
        .groupby(["tso", "direction", "feature", "raw_feature", "group"], as_index=False)
        .agg(share=("share", "mean"), n_predictions=("n_predictions", "sum"))
        .assign(horizon_block="overall")
    )
    ranked = pd.concat([overall, block], ignore_index=True)
    block_order = ["overall", *HORIZON_BLOCKS]
    ranked["horizon_block"] = pd.Categorical(
        ranked["horizon_block"],
        categories=block_order,
        ordered=True,
    )
    ranked = ranked.sort_values(
        ["tso", "direction", "horizon_block", "share", "feature"],
        ascending=[True, True, True, False, True],
    )
    ranked["horizon_block"] = ranked["horizon_block"].astype(str)
    ranked["rank"] = ranked.groupby(["tso", "direction", "horizon_block"]).cumcount() + 1
    columns = [
        "tso",
        "direction",
        "horizon_block",
        "rank",
        "feature",
        "raw_feature",
        "group",
        "share",
        "n_predictions",
    ]
    return ranked.loc[ranked["rank"] <= top_n, columns].reset_index(drop=True)


def _format_group_share(row: pd.Series) -> str:
    return f"{row['group']} ({row['share'] * 100:.0f}%)"


def _format_feature_share(row: pd.Series) -> str:
    return f"{escape_latex(str(row['feature']))} ({row['share'] * 100:.1f}\\%)"


def _interpretation(top_groups: list[str]) -> str:
    top = set(top_groups[:2])
    if top & {"production_consumption", "target_lags/basic", "runlength", "data_driven"}:
        if top & {"wind_pv", "sce", "day_ahead_price"}:
            return "mixed state/forecast driven"
        return "system-state driven"
    if top & {"wind_pv", "sce", "day_ahead_price"}:
        return "forecast driven"
    return "diffuse feature reliance"


def build_global_importance_table(block_shares: pd.DataFrame) -> pd.DataFrame:
    overall = (
        block_shares.groupby(["tso", "direction", "group"], as_index=False)["share"]
        .mean()
        .sort_values(["tso", "direction", "share"], ascending=[True, True, False])
    )
    rows: list[dict] = []
    for (tso, direction), sub in overall.groupby(["tso", "direction"], sort=False):
        top = sub.head(3).reset_index(drop=True)
        group_names = top["group"].tolist()
        rows.append({
            "TSO": tso,
            "direction": direction,
            "dominant group": _format_group_share(top.iloc[0]) if len(top) > 0 else "",
            "second group": _format_group_share(top.iloc[1]) if len(top) > 1 else "",
            "third group": _format_group_share(top.iloc[2]) if len(top) > 2 else "",
            "interpretation": _interpretation(group_names),
        })
    return pd.DataFrame(rows)


def build_global_top_feature_table(top_feature_shares: pd.DataFrame) -> pd.DataFrame:
    overall = top_feature_shares[top_feature_shares["horizon_block"] == "overall"].copy()
    rows: list[dict] = []
    for (tso, direction), sub in overall.groupby(["tso", "direction"], sort=False):
        top = sub.sort_values("rank").head(2).reset_index(drop=True)
        rows.append({
            "TSO": escape_latex(str(tso)),
            "Direction": direction,
            "First feature": _format_feature_share(top.iloc[0]) if len(top) > 0 else "",
            "Second feature": _format_feature_share(top.iloc[1]) if len(top) > 1 else "",
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["TSO", "Direction"])


def build_horizon_top_feature_table(top_feature_shares: pd.DataFrame) -> pd.DataFrame:
    top1 = top_feature_shares[
        (top_feature_shares["horizon_block"].isin(HORIZON_BLOCKS))
        & (top_feature_shares["rank"] == 1)
    ].copy()
    if top1.empty:
        return pd.DataFrame()
    top1["cell"] = top1.apply(_format_feature_share, axis=1)
    top1["tso"] = top1["tso"].map(lambda value: escape_latex(str(value)))
    table = (
        top1
        .pivot(index=["tso", "direction"], columns="horizon_block", values="cell")
        .reindex(columns=list(HORIZON_BLOCKS))
        .rename_axis(index=["TSO", "Direction"], columns=None)
        .rename(columns={
            "early": "Early horizons",
            "middle": "Middle horizons",
            "late": "Late horizons",
        })
    )
    return table


def plot_horizon_group_heatmap(block_shares: pd.DataFrame) -> plt.Figure:
    block_shares = block_shares.copy()
    block_shares["row"] = block_shares["tso"] + " | " + block_shares["direction"]
    rows = sorted(block_shares["row"].unique())
    groups = [g for g in GROUP_ORDER if g in set(block_shares["group"])]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(max(15.5, len(groups) * 1.75), 6.2),
        sharey=True,
        constrained_layout=True,
    )
    vmax = max(float(block_shares["share"].max()), 0.01)
    for ax, block in zip(axes, HORIZON_BLOCKS, strict=True):
        matrix = (
            block_shares[block_shares["horizon_block"] == block]
            .pivot(index="row", columns="group", values="share")
            .reindex(index=rows, columns=groups)
            .fillna(0.0)
        )
        matrix = matrix.rename(columns=PLOT_GROUP_LABELS)
        sns.heatmap(
            matrix * 100,
            ax=ax,
            cmap="YlGnBu",
            vmin=0,
            vmax=vmax * 100,
            cbar=block == "late",
            cbar_kws={"label": "Attribution share (%)", "shrink": 0.86},
            linewidths=0.35,
            linecolor="white",
        )
        ax.set_title(f"{block.capitalize()} horizons", fontsize=16, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35, labelsize=9, pad=2)
        ax.tick_params(axis="y", labelsize=10)
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation_mode("anchor")
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
    return fig


def _save_global_table(table: pd.DataFrame, config: PaperConfig) -> None:
    csv_path = config.tables_dir / "global_feature_group_importance.csv"
    table.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    tex_path = config.tables_dir / "global_feature_group_importance.tex"
    tex_path.write_text(table.to_latex(index=False, escape=True, column_format="llllll"))
    logger.info("Saved %s", tex_path)


def _save_top_feature_outputs(top_feature_shares: pd.DataFrame, config: PaperConfig) -> None:
    csv_path = config.tables_dir / "top_feature_attribution_shares.csv"
    top_feature_shares.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    global_table = build_global_top_feature_table(top_feature_shares)
    if not global_table.empty:
        save_latex_table(
            global_table,
            config.tables_dir / "global_top_feature_importance.tex",
            caption="Top Integrated-Gradients feature attribution shares by TSO and direction.",
            column_format="llll",
            final_form_callback=add_index_names,
        )
        logger.info("Saved %s", config.tables_dir / "global_top_feature_importance.tex")

    horizon_table = build_horizon_top_feature_table(top_feature_shares)
    if not horizon_table.empty:
        save_latex_table(
            horizon_table,
            config.tables_dir / "horizon_top_feature_importance.tex",
            caption="Top Integrated-Gradients feature attribution share by horizon block.",
            column_format="lllll",
            final_form_callback=add_index_names,
        )
        logger.info("Saved %s", config.tables_dir / "horizon_top_feature_importance.tex")



def run(
    config: PaperConfig,
    *,
    ig_root: Path | None = None,
    dataset_root: Path | None = None,
    dataset_name: str | None = None,
    model_filter: str = "nhits",
    start_window: int = 2,
    top_n_features: int = 5,
    exclude_groups: tuple[str, ...] = ("static_covariates", "bloomberg"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate Table 1 and the horizon-block attribution heatmap."""
    inputs = InterpretabilityInputs(
        ig_root=ig_root or Path("outputs_paper"),
        dataset_root=dataset_root or Path("data/model_data_paper_actuals_1h_lag"),
        dataset_name=dataset_name or "basic_day_ahead_price_wind_pv_production_consumption_sce",
        model_filter=model_filter,
        start_window=start_window,
        best_checkpoint=config.best_checkpoint,
        exclude_groups=exclude_groups,
    )

    config.make_output_dirs()
    block_shares = build_horizon_group_shares(
        inputs,
        tso_keys=list(TSO_SUFFIXES),
        input_size=24,
    )
    shares_path = config.tables_dir / "horizon_group_attribution_shares.csv"
    block_shares.to_csv(shares_path, index=False)
    logger.info("Saved %s", shares_path)

    global_table = build_global_importance_table(block_shares)
    _save_global_table(global_table, config)

    top_feature_shares = build_top_feature_shares(
        inputs,
        tso_keys=list(TSO_SUFFIXES),
        input_size=24,
        top_n=top_n_features,
    )
    _save_top_feature_outputs(top_feature_shares, config)

    fig = plot_horizon_group_heatmap(block_shares)
    save_figure(fig, config.figures_dir / "horizon_group_attribution_heatmap", config.figure_format)
    plt.close(fig)
    return global_table, block_shares
