"""Prediction loading, schema validation, and model-column resolution.

All result modules should obtain their prediction DataFrames exclusively
through the helpers in this module so that validation, benchmark merging,
and model-name normalisation happen in exactly one place.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from covariate_effect.read_predictions import (
    read_benchmark_predictions,
    read_local_rolling_window_predictions,
)

from .config import BENCHMARK_MODELS, METADATA_COLS, PaperConfig, DEFAULT_MODEL_ORDER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLS: frozenset[str] = frozenset({"tso", "unique_id", "ds", "horizon", "y"})
TSO_DIR_NAMES: frozenset[str] = frozenset({"50hertz", "amprion", "transnetbw", "tennet_de"})


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _model_columns(df: pd.DataFrame) -> list[str]:
    """Return the model (non-metadata) column names in *df*."""
    return [c for c in df.columns if c not in METADATA_COLS]


def validate_predictions(df: pd.DataFrame, context: str = "") -> None:
    """Raise *ValueError* when the DataFrame is empty, missing required columns,
    or contains null timestamps."""
    tag = f" ({context})" if context else ""
    if df.empty:
        raise ValueError(f"Predictions DataFrame is empty{tag}.")
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Predictions missing required columns {sorted(missing)}{tag}."
        )
    if df["ds"].isnull().any():
        raise ValueError(f"Predictions contain null timestamps{tag}.")


def _read_local_predictions_with_layout_fallback(
    root_dir: Path,
    dataset_name: str,
    *,
    checkpoint_best: bool,
    model_filter: str = "seed",
) -> pd.DataFrame:
    """Read local rolling predictions across known training output layouts.

    Newer training runs commonly save neural predictions under ``evaluation/``;
    some paper-result runs use ``predictions/``; older runs used
    ``tree_importance/``.  The checkpoint suffix is handled by
    ``read_local_rolling_window_predictions`` via *checkpoint_best*.
    """
    candidate_dirs = ["predictions", "evaluation", "tree_importance"]
    errors: list[str] = []
    for evaluation_dir_name in candidate_dirs:
        try:
            df = read_local_rolling_window_predictions(
                rolling_window_shift_dir=root_dir,
                dataset_name=dataset_name,
                evaluation_dir_name=evaluation_dir_name,
                model_filter=model_filter,
                checkpoint_best=checkpoint_best,
            )
            if not df.empty:
                return df
            errors.append(f"{evaluation_dir_name}: empty")
        except Exception as exc:
            errors.append(f"{evaluation_dir_name}: {exc}")

    checkpoint_label = "best checkpoint" if checkpoint_best else "last checkpoint"
    raise FileNotFoundError(
        f"No local rolling-window {checkpoint_label} predictions found for "
        f"dataset '{dataset_name}' under '{root_dir}'. Tried: "
        + "; ".join(errors)
    )


def _seed_specific_dir_candidates(base_dir: Path, seed: int, *, shifted: bool = False) -> list[Path]:
    """Return plausible seed-specific roots, preferring paper-suffixed layouts.

    Paper runs use names such as ``outputs_nhits_only_seed860_paper`` and
    ``outputs_shifted_targets_17_seed860_paper``. Older notebook paths used
    ``..._nhits_only_seed860``. Keep both so ablation loading follows the files
    that are actually present instead of assuming one string pattern.
    """
    stem = str(base_dir).rstrip("/")
    suffixes = [f"_seed{seed}"] if shifted else [f"_nhits_only_seed{seed}", f"_seed{seed}"]
    candidates: list[Path] = []

    for suffix in suffixes:
        if stem.endswith("_paper"):
            candidates.append(Path(f"{stem[:-len('_paper')]}{suffix}_paper"))
        candidates.append(Path(f"{stem}{suffix}"))

    return list(dict.fromkeys(candidates))


def _resolve_seed_specific_dir(base_dir: Path, seed: int, *, shifted: bool = False) -> Path:
    """Resolve the seed-specific directory for ablation runs."""
    candidates = _seed_specific_dir_candidates(base_dir, seed, shifted=shifted)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _neural_requested_models(config: PaperConfig) -> list[str]:
    """Return requested models that are expected in neural prediction sets."""
    benchmark_models = set(BENCHMARK_MODELS) - {"lstm"}
    return [model for model in config.models if model not in benchmark_models]


def _available_dataset_names(root_dir: Path) -> list[str]:
    """Return dataset subdirectory names found below TSO directories."""
    names: list[str] = []
    for tso_dir in root_dir.glob("*"):
        if not tso_dir.is_dir() or tso_dir.name.lower() not in TSO_DIR_NAMES:
            continue
        for dataset_dir in tso_dir.glob("*"):
            if dataset_dir.is_dir():
                names.append(dataset_dir.name)
    return list(dict.fromkeys(names))


def _resolve_dataset_name(root_dir: Path, requested_name: str) -> str:
    """Use *requested_name* when present, otherwise fall back to a single dataset dir."""
    if any(
        (tso_dir / requested_name).is_dir()
        for tso_dir in root_dir.glob("*")
        if tso_dir.is_dir() and tso_dir.name.lower() in TSO_DIR_NAMES
    ):
        return requested_name

    available = _available_dataset_names(root_dir)
    if len(available) == 1:
        return available[0]
    return requested_name


def _read_ablation_ig_predictions(
    root_dir: Path,
    dataset_name: str,
    seed: int,
    checkpoint_best: bool,
    dataset_root: Path,
    actuals_dataset_name: str,
) -> pd.DataFrame:
    """Read forecast predictions saved beside IG tensors for ablation."""
    from training.data_prep import load_dataset, to_nixtla_format

    frames: list[pd.DataFrame] = []
    pred_layouts = (
        [("ig_preds_best_checkpoint", "ig_preds_best_checkpoint"), ("ig_preds", "ig_preds")]
        if checkpoint_best
        else [("ig_preds", "ig_preds")]
    )
    tso_display = {
        "50hertz": "50Hertz",
        "amprion": "Amprion",
        "transnetbw": "TransnetBW",
        "tennet_de": "TenneT_DE",
    }

    for tso_dir in root_dir.glob("*"):
        if not tso_dir.is_dir() or tso_dir.name.lower() not in TSO_DIR_NAMES:
            continue
        ig_root = tso_dir / dataset_name
        if not ig_root.is_dir():
            continue
        tso_name = tso_display.get(tso_dir.name.lower(), tso_dir.name)
        model_dataset = None
        for actuals_name in dict.fromkeys([actuals_dataset_name, dataset_name]):
            actuals_path = dataset_root / f"{actuals_name}_{tso_name}.parquet"
            if actuals_path.exists():
                model_dataset, _ = load_dataset(actuals_path)
                model_dataset = to_nixtla_format(model_dataset)
                break

        for pred_dir_name, file_prefix in pred_layouts:
            pattern = re.compile(rf"{file_prefix}_(?P<model>.+_seed{seed})_window(?P<window>\d+)\.parquet$")
            for pred_file in ig_root.glob(f"window_*/evaluation/{pred_dir_name}/{file_prefix}_*.parquet"):
                match = pattern.match(pred_file.name)
                if match is None:
                    continue
                df = pd.read_parquet(pred_file)
                if "y" not in df.columns:
                    if model_dataset is None:
                        raise FileNotFoundError(
                            f"Cannot attach y for '{pred_file}'; no dataset parquet found under '{dataset_root}'."
                        )
                    df = df.merge(
                        model_dataset[["unique_id", "ds", "y"]],
                        on=["unique_id", "ds"],
                        how="left",
                    )
                model_name = match.group("model")
                window_index = int(match.group("window"))
                id_vars = ["unique_id", "ds", "horizon", "y"]
                missing = set(id_vars) - set(df.columns)
                if missing:
                    raise ValueError(f"IG prediction file '{pred_file}' is missing columns {sorted(missing)}.")
                model_cols = [
                    c for c in df.columns
                    if c not in set(id_vars) | {"dataset", "tso"}
                ]
                if model_name not in model_cols and len(model_cols) == 1:
                    df = df.rename(columns={model_cols[0]: model_name})
                    model_cols = [model_name]
                melted = df.melt(
                    id_vars=id_vars,
                    value_vars=model_cols,
                    var_name="model",
                    value_name="y_pred",
                )
                melted["tso"] = tso_name
                melted["window_index"] = window_index
                frames.append(melted)

    if not frames:
        raise FileNotFoundError(
            f"No IG prediction files found for seed={seed}, dataset='{dataset_name}' under '{root_dir}'."
        )

    concatenated = pd.concat(frames, ignore_index=True)
    return concatenated.pivot_table(
        index=["tso", "unique_id", "ds", "horizon", "y", "window_index"],
        columns="model",
        values="y_pred",
    ).reset_index()


# ---------------------------------------------------------------------------
# Model-column resolution
# ---------------------------------------------------------------------------


def resolve_model_columns(
    df: pd.DataFrame | pd.Series,
    requested_models: list[str],
    by_index: bool = False,
) -> list[str]:
    """Map seed-agnostic model names to the actual column names present in *df*.

    Matching strategy (in order):

    1. Exact match - ``"nhits"`` matches the column ``"nhits"``.
    2. Prefix match - ``"nhits"`` matches ``"nhits_seed778"`` or any column
       that starts with ``"nhits_"``.

    Parameters
    ----------
    df:
        Wide predictions DataFrame with one column per model.
    requested_models:
        Seed-agnostic canonical names supplied on the CLI (``--models``).

    Returns
    -------
    list[str]
        Deduplicated list of resolved column names, in encounter order.

    Warns
    -----
    UserWarning
        When a requested model has no matching column.
    """
    if by_index:
        available = df.index
    else:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("resolve_model_columns: expected DataFrame when by_index=False.")
        available = _model_columns(df)
    resolved: list[str] = []

    for model in requested_models:
        if model == "tft":
            # Include plain TFT and seed-suffixed TFT only, but exclude quantile variants.
            matches = [
                c for c in available
                if (
                    c == "tft"
                    or bool(re.fullmatch(r"tft_seed\d+", c))
                )
            ]
        elif model == "tft_quantile":
            # Include only quantile TFT variants.
            matches = [
                c for c in available
                if (
                    c == "tft_quantile"
                    or bool(re.fullmatch(r"tft_quantile_seed\d+", c))
                )
            ]
        else:
            if model in available:
                resolved.append(model)
                continue
            matches = [
                c for c in available if c == model or c.startswith(f"{model}_")
            ]
        if matches:
            resolved.extend(matches)
        else:
            warnings.warn(
                f"Requested model '{model}' has no matching column in predictions. "
                f"Available model columns: {available}",
                stacklevel=2,
            )

    # Deduplicate while preserving order.
    return list(dict.fromkeys(resolved))


def select_models(
    df: pd.DataFrame | pd.Series,
    requested_models: list[str],
    by_index: bool = False,
) -> pd.DataFrame | pd.Series:
    """Keep only metadata + requested model columns.

    Selection is seed-agnostic: e.g. requesting ``nhits`` keeps
    ``nhits_seed778``/``nhits_seed860`` if present.  TFT selection is stricter:

    * ``tft`` keeps ``tft`` and ``tft_seed<digits>`` only.
    * ``tft_quantile`` keeps ``tft_quantile`` and ``tft_quantile_seed<digits>``.
    """
    if not requested_models:
        return df

    selected_models = resolve_model_columns(df, requested_models, by_index=by_index)
    if by_index:
        available_models = df.index
    else:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("select_models: expected DataFrame when by_index=False.")
        available_models = _model_columns(df)
    if not selected_models:
        raise ValueError(
            "None of the requested models were found. "
            f"Requested={requested_models}; available={available_models}"
        )

    if by_index:
        return df.loc[selected_models]
    else:
        keep_meta = [c for c in df.columns if c in METADATA_COLS]
        return df[keep_meta + selected_models]


def intersect_models_across_sets(dfs: list[pd.DataFrame]) -> list[str]:
    """Return the sorted list of model columns common to *all* DataFrames.

    Raises
    ------
    ValueError
        If the intersection is empty or *dfs* is empty.
    """
    if not dfs:
        raise ValueError("intersect_models_across_sets: no DataFrames provided.")
    sets = [set(_model_columns(df)) for df in dfs]
    common = sets[0].intersection(*sets[1:])
    if not common:
        raise ValueError(
            "No common model columns found across prediction sets.\n"
            f"  Per-set columns: {[sorted(s) for s in sets]}"
        )
    return sorted(common)


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------


def load_predictions(config: PaperConfig, apply_model_selection: bool) -> pd.DataFrame:
    """Load full-covariate rolling-window predictions merged with benchmarks.

    Returns a wide DataFrame (one column per model) with metadata columns
    ``tso``, ``unique_id``, ``ds``, ``horizon``, ``y``.  The ``window_index``
    column is dropped after merging because downstream modules work with the
    pooled evaluation set.

    Parameters
    ----------
    config:
        Active ``PaperConfig`` instance.
    apply_model_selection:
        Whether to apply model selection.
    """
    preds = _read_local_predictions_with_layout_fallback(
        root_dir=config.predictions_dir,
        dataset_name=config.dataset_name,
        checkpoint_best=config.best_checkpoint,
    )
    validate_predictions(preds, context="full-cov predictions")

    benchmarks = read_benchmark_predictions(
        root_dir=config.benchmarks_dir,
        dataset_name=config.dataset_name,
        start_date=pd.Timestamp(config.start_date),
    )

    if benchmarks.empty:
        warnings.warn(
            "Benchmark predictions are empty; proceeding without benchmarks.",
            stacklevel=2,
        )
        merged = preds.copy()
    else:
        merged = preds.merge(
            benchmarks,
            on=["tso", "ds", "unique_id", "horizon"],
            how="inner",
        )

    merged = merged.drop(columns=["window_index"], errors="ignore")
    validate_predictions(merged, context="merged predictions")
    if apply_model_selection:
        merged = select_models(merged, config.models)
    return merged


def load_benchmark_predictions_with_target(
    config: PaperConfig,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return benchmark-only predictions with the ``y`` actuals attached.

    The benchmark-only MCS and DM comparisons need the target column ``y``.
    This function attaches it by joining with the column from *predictions_df*.

    Parameters
    ----------
    config:
        Active ``PaperConfig`` instance.
    predictions_df:
        Full merged predictions DataFrame (returned by :func:`load_predictions`).
    """
    benchmarks = read_benchmark_predictions(
        root_dir=config.benchmarks_dir,
        dataset_name=config.dataset_name,
        start_date=pd.Timestamp(config.start_date),
    )
    if benchmarks.empty:
        raise ValueError(
            "Benchmark predictions are empty; cannot build benchmark-only set."
        )

    # Attach 'y' (and one model column as a cross-check key) from full preds.
    model_cols = _model_columns(predictions_df)
    if not model_cols:
        raise ValueError(
            "No model columns found in full predictions; cannot attach target."
        )
    anchor = model_cols[0]
    result = benchmarks.merge(
        predictions_df[["tso", "ds", "unique_id", "horizon", "y", anchor]],
        on=["tso", "ds", "unique_id", "horizon"],
        how="inner",
    )
    validate_predictions(result, context="benchmark predictions with target")
    return result


def load_ablation_predictions(
    config: PaperConfig,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Load the three prediction sets used in the circular-shift ablation.

    The directory layout mirrors ``get_model_output_directories`` from
    ``circular_shift_ablation.ipynb``:

    * seed 778 (canonical) → ``outputs/``, ``outputs_no_covariates/``,
      ``outputs_shifted_targets_17/``
    * other seeds → ``outputs_nhits_only_seed{seed}/``, etc.

    Parameters
    ----------
    config:
        Active ``PaperConfig`` instance.  For seed 778, ``config.predictions_dir``,
        ``config.no_cov_dir``, and ``config.shift_dir`` are used directly.
    seed:
        Training random seed.

    Returns
    -------
    dict with keys ``"full_cov"``, ``"no_cov"``, ``"full_cov_shift"``.
    """
    if seed == 778:
        dirs: dict[str, tuple[Path, str]] = {
            "full_cov":       (config.predictions_dir, config.dataset_name),
            "no_cov":         (config.no_cov_dir,      "basic"),
            "full_cov_shift": (config.shift_dir,       config.dataset_name),
        }
    else:
        dirs = {
            "full_cov":       (_resolve_seed_specific_dir(config.predictions_dir, seed),         config.dataset_name),
            "no_cov":         (_resolve_seed_specific_dir(config.no_cov_dir, seed),              "basic"),
            "full_cov_shift": (_resolve_seed_specific_dir(config.shift_dir, seed, shifted=True), config.dataset_name),
        }

    requested_models = _neural_requested_models(config)
    if not requested_models:
        requested_models = config.models

    result: dict[str, pd.DataFrame] = {}
    for label, (d, ds_name) in dirs.items():
        ds_name = _resolve_dataset_name(d, ds_name)
        cache_file: Path | None = None
        if config.ablation_cache_dir is not None:
            checkpoint_label = "best_checkpoint" if config.best_checkpoint else "last_checkpoint"
            cache_file = config.ablation_cache_dir / f"{label}_seed{seed}_{checkpoint_label}_ig_predictions.parquet"
            if cache_file.exists():
                try:
                    cached = pd.read_parquet(cache_file)
                    validate_predictions(cached, context=f"seed={seed} {label} (cache)")
                    result[label] = select_models(
                        cached.drop(columns=["window_index"], errors="ignore"),
                        requested_models,
                    )
                    continue
                except Exception as exc:
                    warnings.warn(
                        f"Failed to read cache '{cache_file}' ({exc}); falling back to local files.",
                        stacklevel=2,
                    )

        try:
            try:
                df = _read_ablation_ig_predictions(
                    root_dir=d,
                    dataset_name=ds_name,
                    seed=seed,
                    checkpoint_best=config.best_checkpoint,
                    dataset_root=config.dataset_dir,
                    actuals_dataset_name=config.dataset_name,
                )
            except Exception:
                df = _read_local_predictions_with_layout_fallback(
                    root_dir=d,
                    dataset_name=ds_name,
                    checkpoint_best=config.best_checkpoint,
                    model_filter=f"seed{seed}",
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ablation predictions for seed={seed}, "
                f"label='{label}' from '{d}': {exc}"
            ) from exc

        validate_predictions(df, context=f"seed={seed} {label}")
        no_window = df.drop(columns=["window_index"], errors="ignore")
        result[label] = select_models(no_window, requested_models)

        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                no_window.to_parquet(cache_file)
            except Exception as exc:
                warnings.warn(
                    f"Could not write ablation cache '{cache_file}': {exc}",
                    stacklevel=2,
                )

    return result


def filter_to_common_models(
    dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Restrict all DataFrames to their common model columns.

    This must be called before running pairwise DM tests across prediction
    sets, because each set may have been trained with a different random seed
    and therefore carry differently-named model columns.

    Parameters
    ----------
    dfs:
        Mapping of label → DataFrame.  All DataFrames must contain the standard
        metadata columns (``tso``, ``unique_id``, ``ds``, ``horizon``, ``y``).

    Returns
    -------
    dict
        Same keys as *dfs*; each value has only the intersection of model
        columns (plus metadata).

    Warns
    -----
    UserWarning
        When model columns are dropped in any particular DataFrame.
    """
    common = intersect_models_across_sets(list(dfs.values()))
    row_keys = ["tso", "unique_id", "ds", "horizon"]
    common_rows: pd.MultiIndex | None = None
    for df in dfs.values():
        row_index = pd.MultiIndex.from_frame(df[row_keys].drop_duplicates())
        common_rows = row_index if common_rows is None else common_rows.intersection(row_index)
    if common_rows is None or common_rows.empty:
        raise ValueError("No common prediction rows found across ablation sets.")

    filtered: dict[str, pd.DataFrame] = {}
    for key, df in dfs.items():
        dropped = set(_model_columns(df)) - set(common)
        if dropped:
            warnings.warn(
                f"[{key}] Dropping model columns not in all sets: {sorted(dropped)}",
                stacklevel=2,
            )
        meta_cols = [c for c in df.columns if c in METADATA_COLS]
        aligned = df.set_index(row_keys).loc[common_rows].reset_index()
        filtered[key] = aligned[meta_cols + list(common)]
    return filtered


def apply_model_ordering(
    df: pd.DataFrame | pd.Series,
    by_index: bool,
) -> pd.DataFrame | pd.Series:
    """Reorder model columns in *df* according to the default ordering."""
    available_models = df.index if by_index else pd.Index(_model_columns(df))
    available_models = available_models.str.replace(r"_seed\d+", "", regex=True)
    filtered_order = [m for m in DEFAULT_MODEL_ORDER if m in available_models]
    model_col_idx = [available_models.get_loc(c)for c in filtered_order]
    if by_index:
        return df.loc[df.index[model_col_idx]]
    else:
        return df.iloc[:, model_col_idx]
