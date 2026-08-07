"""
Training runner – model construction, single-window and rolling-window training.

Uses NeuralForecast (Nixtla) with WandbLogger for loss tracking.
"""

import logging
import shutil
import subprocess
import tarfile
import pytz
import wandb
import tempfile

from os import cpu_count
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Literal

import numpy as np
import pandas as pd

from neuralforecast.core import NeuralForecast, NHITS, TFT, NBEATSx
try:
    from neuralforecast.core import LSTM
except ImportError:
    from neuralforecast.models import LSTM  # type: ignore[no-redef]
from neuralforecast.losses.pytorch import MAE, MQLoss
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from training.data_prep import STATIC_EXOG_COLS, build_static_df
from training.explainability import (
    explain_all_models,
    explain_with_shift_correction,
    save_explanation_artifacts,
    save_tft_interpretability,
    aggregate_tft_interpretability_windows,
)
from training.losses import MQMedianLoss
from training.predict import (
    predict_with_shift_correction,
    prepare_predictions_df,
    log_predictions_to_wandb,
)

CHECKPOINT_BEST_NAME = "best_valid_checkpoint.ckpt"

logger = logging.getLogger(__name__)


# ── Configuration dataclass ───────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """All knobs needed to drive a training run."""

    # Dataset
    dataset_path: str = ""
    tso: str = "TenneT_DE"
    direction: str = "both"

    # Shifting
    shift_hours: int = 9

    # Calendar
    add_calendar: bool = True
    holidays_path: Optional[str] = None

    # Dates
    train_start: str = "2020-01-01"
    valid_start: str = "2024-02-01"
    test_start: str = "2024-04-01"

    # Date time
    date_time: Optional[str] = None  # If None, will be set to current time in make_wandb_config()

    # Forecast
    forecast_horizon: int = 24
    input_size: int = 24

    # Training
    max_steps: int = 5_000
    val_check_steps: int = 50
    early_stop_patience: int = 20
    batch_size: int = 16
    windows_batch_size: int = 64
    learning_rate: float = 1e-4
    random_seed: int = 778
    scaler_type: Optional[str] = None  # None → NeuralForecast uses local_scaler_type
    local_scaler_type: str = "standard"

    # Models to train (subset of: nhits, nbeatsx, tft, tft_quantile, lstm)
    models: list[str] = field(default_factory=lambda: ["nhits", "tft"])

    # Quantile levels for MQLoss models
    quantile_levels: list[int] = field(
        default_factory=lambda: [54, 60, 64, 70, 74, 80, 84, 90, 94, 98]
    )

    # Per-model parameter overrides (nested dict keyed by model name).
    # Populated from the YAML config's model-specific sections.
    # Example: {"nhits": {"n_blocks": [8, 6, 4]}, "lstm": {"encoder_hidden_size": 256}}
    model_params: dict = field(default_factory=dict)

    # Rolling window
    rolling_window: bool = False
    n_train_months: int = 37
    n_valid_months: int = 2
    n_test_months: int = 1

    # Checkpointing
    persist_checkpoints: bool = False
    persist_checkpoints_to_wandb: bool = False
    checkpoint_compression: Optional[int] = None
    checkpoint_compression_n_threads: int = 20
    checkpoint_selection: Literal["last", "best", "both"] = "last"
    skip_explainability: bool = False

    # Circular-shift augmentation (set by circular_shift_training.py; None = not used)
    circ_shift_k_days: Optional[int] = None
    circ_shift_T_hours: Optional[int] = None

    # Output / logging
    output_dir: str = "outputs"
    eval_dir: Optional[str] = None
    wandb_project: str = "redispatch-forecasting"
    wandb_entity: Optional[str] = None

    @property
    def train_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_start)

    @property
    def valid_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.valid_start)

    @property
    def test_start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_start)

    @property
    def normalized_tso(self) -> str:
        return self.tso.replace(" ", "_").lower()


# ── Horizon weighting ─────────────────────────────────────────────────────────
def _constant_weights(h: int) -> np.ndarray:
    """Constant weight = 1 for all horizon steps (used by the existing shifted models)."""
    start = max(0, h - 24)
    w = np.zeros(h)
    w[start:] = 1.0
    return w


# ── Model factory ─────────────────────────────────────────────────────────────
def _model_alias(model_name: str, config: TrainConfig) -> str:
    if model_name == "tft_quantile":
        return f"tft_quantile_seed{config.random_seed}"
    return f"{model_name}_seed{config.random_seed}"


def _make_checkpoint_callback(save_dir: Path) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=str(save_dir),
        monitor="valid_loss",
        filename="{epoch:04d}-{valid_loss:.4f}",
        mode="min",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
    )


def _should_persist_checkpoints(config: TrainConfig) -> bool:
    return config.persist_checkpoints or config.persist_checkpoints_to_wandb


def test_if_tar_is_available():
    # Create a simple file in a temporary directory, try to archive it with tar and zstd, and check if the archive was created successfully.

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("This is a test.")

            archive_path = Path(tmpdir) / "test.tar.zst"
            try:
                subprocess.run(
                    ["tar", "-I", "zstd", "-cf", str(archive_path), "-C", tmpdir, "test.txt"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if not archive_path.exists():
                    logger.warning("Tar command ran but archive was not created. Will not archive checkpoints.")
                    return False
            except Exception as e:
                logger.exception("tar with zstd compression errorred out. Will not archive checkpoints.")
                return False
            
            return True
    except Exception as e:
        logger.exception("Unexpected error during tar availability test. Will not archive checkpoints.")
        return False


def _archive_checkpoint_dir(save_dir: Path, compression_level: int | None, compression_n_threads: int | None) -> Optional[Path]:
    if not compression_level:
        return None
    
    archive_path = save_dir.parent / f"{save_dir.name}.tar.zst"

    if archive_path.exists():
        logger.warning("Archive path %s already exists. Will not archive again.", archive_path)
        return archive_path

    if compression_level < 1 or compression_level > 22:
        logger.warning("Invalid compression level %d; setting to 15.", compression_level)
        compression_level = 15

    if not test_if_tar_is_available():
        return None
    
    cpus = cpu_count()
    if cpus is not None and compression_n_threads is not None:
        n_compression_threads = max(min(compression_n_threads, (cpus - 1) // 2), 1)
    elif cpus is not None:
        n_compression_threads = max((cpus - 1) // 2, 1)
    else:
        n_compression_threads = 1

    logger.info("Archiving checkpoints in %s with zstd compression level %d using %d threads", save_dir, compression_level, n_compression_threads)
    arvhive_command_result = subprocess.run(
        ["tar", "-I", f"zstd -{compression_level} --threads={n_compression_threads}", "-cf", f"{save_dir}.tar.zst", "-C", str(save_dir.parent), save_dir.name],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if arvhive_command_result.returncode != 0:
        logger.warning("Tar command failed with return code %d. Will not archive checkpoints.", arvhive_command_result.returncode)
        return None
    

    if not archive_path.exists():
        logger.warning("Archive was not created at expected path %s. Will not archive checkpoints.", archive_path)
        return None

    return archive_path


def _log_checkpoint_artifact(
    run: wandb.Run,
    save_dir: Path,
    alias: str,
    date_time: str,
    config: TrainConfig,
    window_index: int | None = None,
) -> Optional[Path]:

    archive_path = _archive_checkpoint_dir(save_dir, config.checkpoint_compression, config.checkpoint_compression_n_threads)
    artifact_name = f"{alias}_{date_time}"
    if window_index is not None:
        artifact_name = f"{artifact_name}_window{window_index}"

    artifact = wandb.Artifact(
        name=artifact_name,
        type="model_checkpoint",
        metadata={
            "compression": config.checkpoint_compression or "none",
            "save_dir": str(save_dir),
        },
    )
    if archive_path:
        artifact.add_file(str(archive_path))
    else:
        artifact.add_dir(str(save_dir))

    run.log_artifact(artifact)
    return archive_path

def make_wandb_config(
    config: TrainConfig,
    hist_cov_cols: list[str],
    future_cov_cols: list[str],
    window_index: int | None = None,
    metadata: dict | None = None,
    window_test_start: str | None = None,
    window_test_end: str | None = None,
) -> tuple[dict, str, str, str]:
    if config.date_time is not None:
        date_time_stamp = config.date_time
    else:
        date_time_stamp = datetime.now(tz=pytz.UTC).astimezone(pytz.timezone("Europe/Bucharest")).strftime("%Y-%m-%d_%H-%M-%S")
    dataset_contents = ", ".join(metadata["feature_sets"]) if metadata else ""
    group_tag_interface = f"{config.tso}{'_' + dataset_contents if dataset_contents else ''}_{date_time_stamp}"
    base_config = {
        "tso": config.tso,
        "date_time": date_time_stamp,
        "dataset_contents": dataset_contents,
        "dataset_local_path": config.dataset_path,
        "model_arch": "paper",
        "model_type": "single_window",
        "shift_hours": config.shift_hours,
        "direction": config.direction,
        "forecast_horizon": config.forecast_horizon,
        "input_size": config.input_size,
        "train_start": config.train_start,
        "valid_start": config.valid_start,
        "test_start": config.test_start,
        "max_steps": config.max_steps,
        "random_seed": config.random_seed,
        "early_stop_patience_steps_correct": config.early_stop_patience,
        "local_scaler_type": config.local_scaler_type,
        "n_future_covariates": len(future_cov_cols),
        "n_hist_covariates": len(hist_cov_cols),
        "future_covariates": future_cov_cols,
        "historical_covariates": hist_cov_cols,
        "add_calendar": config.add_calendar,
        "holidays_path": config.holidays_path,
        "dataset_metadata": metadata or {},
    }

    if config.circ_shift_k_days is not None:
        base_config["circ_shift_k_days"] = config.circ_shift_k_days
        base_config["circ_shift_T_hours"] = config.circ_shift_T_hours
        base_config["augmentation"] = "circular_shift"
        group_tag_interface += f"_cs{config.circ_shift_k_days}d"

    if window_index is not None:
        base_config.update({
            "model_type": "rolling_window",
            "window_index": window_index,
            "n_test_months": config.n_test_months,
        })
        # Override with the per-window dates so downstream consumers
        # (prediction_pipeline_rolling_window.py, explainability_pipeline.py)
        # always use the correct test window, not the global TrainConfig value.
        if window_test_start is not None:
            base_config["test_start"] = window_test_start
        if window_test_end is not None:
            base_config["test_end"] = window_test_end
        window_suffix = f"_window{window_index}"
        # Keep the _window{N} suffix readable; truncate the prefix if needed.
        max_prefix = 128 - len(window_suffix)
        group_tag_interface = group_tag_interface[:max_prefix] + window_suffix
    else:
        group_tag_interface = group_tag_interface[:128]

    return base_config, date_time_stamp, dataset_contents, group_tag_interface


def _load_tft_interp_local(interp_dir: Path) -> dict | None:
    """
    Load TFT interpretability data saved by ``save_tft_interpretability``
    from a local directory.

    Parameters
    ----------
    interp_dir : Path
        The per-window directory, e.g.
        ``<output_dir>/<tso>/<dataset>/window_N/evaluation/
        tft_interpretability/<alias>_windowN/``.

    Returns
    -------
    dict | None
        ``{"feature_importances": dict[str, pd.DataFrame],
           "attention_weights": np.ndarray}``
        or ``None`` if the directory does not exist or contains no artefacts.
    """
    if not interp_dir.exists():
        return None

    fi_dict: dict[str, pd.DataFrame] = {}
    for csv_file in sorted(interp_dir.glob("*.csv")):
        try:
            fi_dict[csv_file.stem] = pd.read_csv(csv_file, index_col=0)
        except Exception as exc:
            logger.warning(
                "Failed to load TFT feature importance CSV %s: %s", csv_file, exc
            )

    attn_path = interp_dir / "attention_weights.npy"
    attn_np: np.ndarray | None = None
    if attn_path.exists():
        try:
            attn_np = np.load(attn_path)
        except Exception as exc:
            logger.warning(
                "Failed to load TFT attention weights from %s: %s", attn_path, exc
            )

    if not fi_dict and attn_np is None:
        return None

    return {"feature_importances": fi_dict, "attention_weights": attn_np}


def find_best_valid_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the checkpoint file with the best validation loss in the given directory."""
    if not checkpoint_dir.exists():
        logger.warning("Checkpoint directory does not exist: %s", checkpoint_dir)
        return None

    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        logger.warning("No checkpoint files found in: %s", checkpoint_dir)
        return None

    best_checkpoint = None
    best_loss = float("inf")
    for ckpt_file in checkpoint_files:
        try:
            # Extract validation loss from filename.
            # Supported formats include:
            # - "epoch=XX-valid_loss=YY.ckpt"
            # - "0001-0.1234.ckpt" (ModelCheckpoint filename="{epoch:04d}-{valid_loss:.4f}")
            parts = ckpt_file.stem.split("-")
            val_loss_part = next((p for p in parts if p.startswith("valid_loss=")), None)
            if val_loss_part is not None:
                val_loss = float(val_loss_part.split("=")[1])
            else:
                if len(parts) < 2:
                    continue
                # Fallback to the last token when only the metric value is present.
                val_loss = float(parts[-1])
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = ckpt_file
        except Exception as e:
            logger.warning("Failed to parse checkpoint file %s: %s", ckpt_file, e)

    if best_checkpoint is None:
        logger.warning("No valid checkpoint files found in: %s", checkpoint_dir)
    else:
        logger.info("Best checkpoint found: %s with valid_loss=%.4f", best_checkpoint, best_loss)

    return best_checkpoint


def _selected_checkpoints(selection: Literal["last", "best", "both"]) -> list[str]:
    if selection == "last":
        return ["last"]
    if selection == "best":
        return ["best"]
    if selection == "both":
        return ["last", "best"]
    raise ValueError(f"Unsupported checkpoint selection: {selection}")


def _ensure_checkpoint_dir_unzipped(checkpoint_dir: Path) -> None:
    """
    Ensure the checkpoint directory exists on disk uncompressed.

    If the directory was archived as ``<checkpoint_dir>.tar.zst`` and removed,
    restore it before model loading for prediction.
    """
    nf_model_dir = checkpoint_dir / "nf_model"
    if nf_model_dir.exists():
        return

    archive_path = checkpoint_dir.parent / f"{checkpoint_dir.name}.tar.zst"
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory is missing and no archive was found: {checkpoint_dir}"
        )

    logger.info(
        "Restoring checkpoint directory from archive before prediction: %s",
        archive_path,
    )
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "tar",
            "-I",
            "zstd",
            "-xf",
            str(archive_path),
            "-C",
            str(checkpoint_dir.parent),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not nf_model_dir.exists():
        raise FileNotFoundError(
            f"Archive extraction completed but nf_model is still missing in {checkpoint_dir}"
        )


def _cleanup_unzipped_checkpoint_dir(checkpoint_dir: Path | None, config: TrainConfig) -> None:
    """
    Remove an uncompressed checkpoint directory when a compressed archive exists.

    This is primarily used after best-checkpoint evaluation, where the directory
    may have been restored from ``.tar.zst`` just for prediction.
    """
    if checkpoint_dir is None:
        return

    if not checkpoint_dir.exists():
        return

    archive_path = checkpoint_dir.parent / f"{checkpoint_dir.name}.tar.zst"
    if not archive_path.exists():
        # Avoid deleting the only remaining copy if no archive exists.
        return

    shutil.rmtree(checkpoint_dir, ignore_errors=True)


def _load_selected_nf(
    checkpoint_key: str,
    nf_last: NeuralForecast,
    checkpoint_dir: Path | None,
) -> NeuralForecast:
    if checkpoint_key == "last":
        return nf_last

    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required to load best checkpoint")

    from training.prediction_pipeline_rolling_window import load_model

    _ensure_checkpoint_dir_unzipped(checkpoint_dir)
    nf_model_dir = checkpoint_dir / "nf_model"
    return load_model(nf_model_dir, checkpoint_best=True)


def _save_predictions_locally(
    eval_dir: str | None,
    normalized_tso: str,
    model_alias: str,
    timestamp: str,
    split_name: str,
    preds_df: pd.DataFrame,
    window_index: int | None = None,
) -> None:
    if eval_dir is None or preds_df.empty:
        return

    out_root = Path(eval_dir) / normalized_tso / model_alias / timestamp
    if window_index is not None:
        out_root = out_root / f"window_{window_index}"
    out_root.mkdir(parents=True, exist_ok=True)

    output_path = out_root / f"{split_name}_predictions.parquet"
    preds_df.to_parquet(output_path, index=False)
    logger.info("Saved predictions locally to %s", output_path)


def _save_rolling_predictions_global(
    eval_dir: str | None,
    normalized_tso: str,
    model_alias: str,
    timestamp: str,
    checkpoint_key: str,
    preds_df: pd.DataFrame,
) -> None:
    if eval_dir is None or preds_df.empty:
        return

    out_root = Path(eval_dir) / normalized_tso / model_alias / timestamp
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_suffix = "_best_checkpoint" if checkpoint_key == "best" else ""
    output_path = out_root / f"predictions_{model_alias}_{normalized_tso}_{timestamp}{ckpt_suffix}.parquet"
    preds_df.to_parquet(output_path, index=False)
    logger.info("Saved rolling global predictions locally to %s", output_path)


def build_model(
    config: TrainConfig,
    model_name: str,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    pl_logger: WandbLogger,
    checkpoint_cb: ModelCheckpoint | None,
):
    """Instantiate a single Nixtla model object with logging and checkpointing."""
    h = config.forecast_horizon
    weights = _constant_weights(h)
    alias = _model_alias(model_name, config)

    # Per-model parameter overrides from the YAML config.
    mp: dict = config.model_params.get(model_name, {})

    callbacks: list = []
    if checkpoint_cb is not None:
        callbacks.append(checkpoint_cb)
    
    if config.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="valid_loss",
                mode="min",
                patience=config.early_stop_patience,
            )
        )

    common_kwargs = dict(
        h=h,
        futr_exog_list=future_cov_cols,
        hist_exog_list=hist_cov_cols,
        stat_exog_list=STATIC_EXOG_COLS,
        batch_size=config.batch_size,
        windows_batch_size=config.windows_batch_size,
        loss=MAE(horizon_weight=weights),
        valid_loss=MAE(horizon_weight=weights),
        max_steps=config.max_steps,
        val_check_steps=config.val_check_steps,
        early_stop_patience_steps=-1,
        random_seed=config.random_seed,
        scaler_type=config.scaler_type,
        alias=alias,
        logger=pl_logger,
        enable_checkpointing=checkpoint_cb is not None,
        callbacks=callbacks,
    )

    if model_name == "nhits":
        defaults = {
            "input_size": config.input_size,
            "n_blocks": [20, 15, 10],
            "learning_rate": config.learning_rate,
        }
        defaults.update(mp)
        return NHITS(**defaults, **common_kwargs)  # type: ignore[arg-type]

    if model_name == "nbeatsx":
        defaults = {
            "input_size": config.input_size,
            "n_blocks": [15, 5, 2],
            "stack_types": ["identity", "seasonality", "trend"],
            "learning_rate": 5e-5,
        }
        defaults.update(mp)
        return NBEATSx(**defaults, **common_kwargs)  # type: ignore[arg-type]

    if model_name == "tft":
        common_kwargs.update(dict(windows_batch_size=min(config.windows_batch_size, 32)))
        defaults = {
            "input_size": config.input_size,
            "hidden_size": 64,
            "dropout": 0.15,
            "n_head": 4,
            "attn_dropout": 0.1,
        }
        defaults.update(mp)
        return TFT(**defaults, **common_kwargs)  # type: ignore[arg-type]

    if model_name == "tft_quantile":
        common_kwargs.update(dict(
            loss=MQLoss(level=config.quantile_levels, horizon_weight=weights),  # type: ignore[arg-type]
            valid_loss=MQMedianLoss(level=config.quantile_levels, horizon_weight=weights),  # type: ignore[arg-type]
            windows_batch_size=min(config.windows_batch_size, 32)
        ))
        defaults = {
            "input_size": config.input_size,
            "hidden_size": 64,
            "dropout": 0.15,
            "n_head": 4,
            "attn_dropout": 0.1,
        }
        defaults.update(mp)
        return TFT(**defaults, **common_kwargs)  # type: ignore[arg-type]

    if model_name == "lstm":
        common_kwargs.update(dict(windows_batch_size=min(config.windows_batch_size, 32)))
        defaults = {
            "input_size": config.input_size,
            "encoder_n_layers": 2,
            "encoder_hidden_size": 128,
            "encoder_dropout": 0.1,
            "decoder_hidden_size": 128,
            "decoder_layers": 2,
            "learning_rate": config.learning_rate,
        }
        defaults.update(mp)
        return LSTM(**defaults, **common_kwargs)  # type: ignore[arg-type]

    raise ValueError(f"Unknown model name: {model_name}")


# ── Single-window training ────────────────────────────────────────────────────
def train_single_window(
    df: pd.DataFrame,
    config: TrainConfig,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    metadata: dict | None = None,
    df_unshifted: pd.DataFrame | None = None,
    df_for_prediction: Optional[pd.DataFrame] = None,
    per_window_transform: Optional[Callable[[pd.DataFrame, int], pd.DataFrame]] = None,
) -> list[NeuralForecast]:
    """
    Train all configured models on a single train/val split.

    The split is defined by ``config.valid_start`` and ``config.test_start``:
    * Training data:   ``[train_start, test_start)``
    * Validation tail: ``val_size = (test_start - valid_start) * 24`` hours

    df_for_prediction : optional DataFrame
        If provided, used as ``df_shifted`` inside ``predict_with_shift_correction``
        and ``explain_all_models`` instead of ``df``.  Pass a version of the dataset
        that has only the *target shift* applied (no circular-covariate shift) so that
        test-time predictions use real covariate history and are comparable with
        non-augmented baselines.  Defaults to ``df`` when not supplied.

    per_window_transform : optional callable ``(window_df, train_hours) -> window_df``
        Applied to the training slice after it is cut but before fitting.  The
        second argument is the number of *training* hours (excluding validation),
        which the transform may use as the circular-shift period T_h.

    Returns the fitted ``NeuralForecast`` wrapper.
    """
    import wandb

    # Convert start dates into model time
    if config.shift_hours > 0:
        train_start_ts = pd.Timestamp(config.train_start) - pd.Timedelta(hours=config.shift_hours)
        valid_start_ts = pd.Timestamp(config.valid_start) - pd.Timedelta(hours=config.shift_hours)
        test_start_ts = pd.Timestamp(config.test_start) - pd.Timedelta(hours=config.shift_hours)
    else:
        train_start_ts = pd.Timestamp(config.train_start)
        valid_start_ts = pd.Timestamp(config.valid_start)
        test_start_ts = pd.Timestamp(config.test_start)

    train_df = df[
        (df["ds"] >= train_start_ts) & (df["ds"] < test_start_ts)
    ].copy()

    val_hours = int((test_start_ts - valid_start_ts).total_seconds() / 3600)
    train_hours = int((valid_start_ts - train_start_ts).total_seconds() / 3600)
    if per_window_transform is not None:
        train_df = per_window_transform(train_df, train_hours)
        logger.info(
            "Single-window: applied per-window transform (train_hours=%d)",
            train_hours,
        )
    logger.info(
        "Single-window training: %d rows, val_size=%d hours",
        len(train_df),
        val_hours,
    )

    static_df = build_static_df()
    nf_list: list[NeuralForecast] = []

    base_config, date_time, dataset_contents, group_tag_interface = make_wandb_config(config, hist_cov_cols, future_cov_cols, metadata=metadata)

    for model_name in config.models:
        alias = _model_alias(model_name, config)
        _cs_suffix = f"_cs{config.circ_shift_k_days}d" if config.circ_shift_k_days is not None else ""
        _run_name = f"train_{config.tso}_{alias}_{date_time}{_cs_suffix}"
        _run_name = _run_name[:128]  # wandb run names must not exceed 128 chars
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=_run_name,
            job_type="training",
            config={
                **base_config,
                "model_name": model_name,
                "model_alias": alias,
            },
            group=group_tag_interface,
            reinit="finish_previous",
        )

        pl_logger = WandbLogger(
            experiment=run,
            log_model=False,
        )

        should_persist = _should_persist_checkpoints(config)
        should_prepare_checkpoint = should_persist or config.checkpoint_selection in ("best", "both")
        checkpoint_dir = None
        checkpoint_cb = None
        if should_prepare_checkpoint:
            checkpoint_dir = (
                Path(config.output_dir)
                / dataset_contents.replace(", ", "_")
                / f"{config.normalized_tso}"
                / date_time
                / alias
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_cb = _make_checkpoint_callback(checkpoint_dir)

        model = build_model(
            config,
            model_name,
            future_cov_cols,
            hist_cov_cols,
            pl_logger,
            checkpoint_cb,
        )
        run.config.update({"model_params": dict(model.hparams)}, allow_val_change=True)

        nf = NeuralForecast(
            models=[model],
            freq="h",
            local_scaler_type=config.local_scaler_type,
        )

        nf.fit(
            df=train_df,
            static_df=static_df,
            val_size=val_hours,
        )

        if should_prepare_checkpoint and checkpoint_dir is not None:
            nf_dir = checkpoint_dir / "nf_model"
            nf.save(str(nf_dir), overwrite=True)
            logger.info("Saved NeuralForecast model to %s", nf_dir)
            best_valid_checkpoint = find_best_valid_checkpoint(checkpoint_dir)
            if best_valid_checkpoint:
                best_valid_checkpoint.rename(nf_dir / CHECKPOINT_BEST_NAME)

            if should_persist:
                archive_path = None
                if config.persist_checkpoints_to_wandb:
                    archive_path = _log_checkpoint_artifact(
                        run,
                        checkpoint_dir,
                        alias,
                        date_time,
                        config,
                    )
                elif config.checkpoint_compression:
                    archive_path = _archive_checkpoint_dir(
                        checkpoint_dir,
                        config.checkpoint_compression,
                        config.checkpoint_compression_n_threads,
                    )
                # Keep the archive but drop the uncompressed directory if local
                # checkpoint persistence is disabled.
                if archive_path and archive_path.exists():
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)

        # ── Generate & log predictions on test set ─────────────────────
        _unshifted = df_unshifted if df_unshifted is not None else df
        # Use df_for_prediction as the shifted context for inference so that
        # circular-covariate augmentation does not bleed into test predictions.
        _pred_df = df_for_prediction if df_for_prediction is not None else df
        if df_unshifted is not None or config.shift_hours == 0:
            test_pred_end = _unshifted["ds"].max()
            logger.info(
                "Generating test predictions for %s: [%s, %s]",
                alias, config.test_start, test_pred_end,
            )
            for checkpoint_key in _selected_checkpoints(config.checkpoint_selection):
                nf_eval = _load_selected_nf(checkpoint_key, nf, checkpoint_dir)
                split_name = "test_best_valid" if checkpoint_key == "best" else "test"

                test_preds = predict_with_shift_correction(
                    nf=nf_eval,
                    df_shifted=_pred_df,
                    df_unshifted=_unshifted,
                    static_df=static_df,
                    pred_start=config.test_start_ts,
                    pred_end=test_pred_end,
                    future_cov_cols=future_cov_cols,
                    shift_hours=config.shift_hours,
                    forecast_horizon=config.forecast_horizon,
                    tso=config.tso,
                    holidays_path=config.holidays_path,
                )
                if not test_preds.empty:
                    test_preds = prepare_predictions_df(test_preds, _unshifted)
                    log_predictions_to_wandb(
                        run, test_preds, split_name, alias,
                        tso=config.tso, shift_hours=config.shift_hours,
                        timestamp=date_time,
                    )
                    _save_predictions_locally(
                        eval_dir=config.eval_dir,
                        normalized_tso=config.normalized_tso,
                        model_alias=alias,
                        timestamp=date_time,
                        split_name=split_name,
                        preds_df=test_preds,
                    )

                if config.skip_explainability:
                    continue

                test_explanations_df, test_predictions_df = explain_all_models(
                    nf=nf_eval,
                    df_shifted=_pred_df,
                    df_unshifted=_unshifted,
                    static_df=static_df,
                    pred_start=config.test_start_ts,
                    pred_end=test_pred_end,
                    future_cov_cols=future_cov_cols,
                    shift_hours=config.shift_hours,
                    forecast_horizon=config.forecast_horizon,
                    tso=config.tso,
                    holidays_path=config.holidays_path,
                )
                persist_dir = (
                    Path(config.output_dir)
                    / dataset_contents.replace(", ", "_")
                    / f"{config.normalized_tso}"
                    / date_time
                    / "evaluation"
                )
                raw_prefix = "ig_raw_best_checkpoint" if checkpoint_key == "best" else "ig_raw"
                preds_prefix = "ig_preds_best_checkpoint" if checkpoint_key == "best" else "ig_preds"
                save_explanation_artifacts(
                    run=run,
                    explanations=test_explanations_df,
                    predictions=test_predictions_df,
                    model_alias=alias,
                    timestamp=date_time,
                    shift_hours=config.shift_hours,
                    tso=config.tso,
                    output_dir=persist_dir,
                    raw_artifact_name_prefix=raw_prefix,
                    preds_artifact_name_prefix=preds_prefix,
                )
                # TFT built-in interpretability (VSN weights + attention)
                # interpretability_params are populated by the predict calls above.
                save_tft_interpretability(
                    nf=nf_eval,
                    output_dir=persist_dir,
                    run=run,
                    model_alias=alias,
                    window_index=None,
                    tso=config.tso,
                    timestamp=date_time,
                )

                if checkpoint_key == "best":
                    _cleanup_unzipped_checkpoint_dir(checkpoint_dir, config)
        else:
            logger.warning(
                "Skipping predictions: df_unshifted not provided and shift_hours > 0"
            )

        if (
            checkpoint_dir is not None
            and not config.persist_checkpoints
            and not config.persist_checkpoints_to_wandb
        ):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

        run.finish()
        nf_list.append(nf)

    return nf_list


# ── Rolling-window training ──────────────────────────────────────────────────
@dataclass
class WindowBoundary:
    train_start: pd.Timestamp
    valid_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _compute_rolling_windows(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    n_train_months: int,
    n_valid_months: int,
    n_test_months: int,
    min_month_coverage: float = 0.5,
) -> list[WindowBoundary]:
    """Compute rolling-window boundaries, advancing one month at a time."""
    windows = []
    current_train_start = data_start

    while True:
        valid_start = current_train_start + pd.DateOffset(months=n_train_months)
        test_start = valid_start + pd.DateOffset(months=n_valid_months)
        test_end = test_start + pd.DateOffset(months=n_test_months)

        # Keep the last window, even if test_end slightly exceeds data_end
        current_month_covarage = (data_end - test_start).total_seconds() / (test_end - test_start).total_seconds()
        if test_end > data_end + pd.Timedelta(hours=1) and current_month_covarage < min_month_coverage:
            break

        windows.append(WindowBoundary(
            train_start=current_train_start,
            valid_start=valid_start,
            test_start=test_start,
            test_end=test_end,
        ))
        current_train_start += pd.DateOffset(months=1)

    return windows


def train_rolling_windows(
    df: pd.DataFrame,
    config: TrainConfig,
    future_cov_cols: list[str],
    hist_cov_cols: list[str],
    metadata: dict | None = None,
    df_unshifted: pd.DataFrame | None = None,
    df_for_prediction: Optional[pd.DataFrame] = None,
    start_from_window: int = 0,
    per_window_transform: Optional[Callable[[pd.DataFrame, int], pd.DataFrame]] = None,
) -> list[NeuralForecast]:
    """
    Train models over rolling windows (one month step).

    per_window_transform : optional callable ``(window_df, train_hours) -> window_df``
        If supplied, called once per window on ``window_df`` (train+val slice)
        before fitting.  ``train_hours`` is the length of the *training* portion
        of the window (excluding validation), e.g. for use as the circular-shift
        period T_h.

    Each window produces its own ``NeuralForecast`` object.

    Shift correction
    ----------------
    ``df`` is the *shifted* dataset: ``ds`` holds the original physical timestamp
    while ``y`` and time-aligned future covariates were shifted *forward* by
    ``shift_hours``.  A row at model-time ``t`` therefore represents a physical
    target at ``t + shift_hours``.

    Window boundaries (``wb.train_start``, ``wb.test_start``, ...) are expressed
    in **physical time**.  To slice ``df`` correctly we subtract ``shift_hours``
    from each boundary before filtering on ``ds``:

    * Without the correction the first ``shift_hours`` physical training rows
      (physical targets ``[train_start, train_start + shift_hours)``) would be
      excluded, and the last ``shift_hours`` rows fed to NeuralForecast would
      have physical targets that fall *inside* the test period (leakage).
    * With the correction, model-time ``[train_start − shift_hours, test_start − shift_hours)``
      maps exactly to physical targets ``[train_start, test_start)``.  The
      ``val_size`` is still computed from the physical gap
      ``(test_start − valid_start)`` in hours, which is the same length in both
      time frames.
    """
    import wandb

    data_start = df["ds"].min()
    data_end = df["ds"].max()

    windows = _compute_rolling_windows(
        data_start, data_end,
        config.n_train_months, config.n_valid_months, config.n_test_months,
    )
    logger.info("Rolling-window training: %d windows", len(windows))

    nf_list: list[NeuralForecast] = []
    # Accumulate per-window TFT interpretability for post-loop averaging
    tft_interp_accumulator: dict[str, list[dict]] = {}
    # Accumulate per-window predictions for model-level global files
    rolling_preds_accumulator: dict[tuple[str, str], list[pd.DataFrame]] = {}
    rolling_preds_timestamp: dict[tuple[str, str], str] = {}

    # ── Pre-populate accumulator with already-completed windows ──────────────
    if start_from_window > 0:
        _, _, dataset_contents_prev, _ = make_wandb_config(
            config, hist_cov_cols, future_cov_cols, metadata=metadata
        )
        for wi_prev in range(start_from_window):
            for model_name in config.models:
                if "tft" not in model_name:
                    continue
                alias_prev = _model_alias(model_name, config)
                # Try local disk (no date_time required in the path)
                local_interp_dir = (
                    Path(config.output_dir)
                    / f"{config.normalized_tso}"
                    / dataset_contents_prev.replace(", ", "_")
                    / f"window_{wi_prev}"
                    / "evaluation"
                    / "tft_interpretability"
                    / f"{alias_prev}_window{wi_prev}"
                )
                tft_prev = _load_tft_interp_local(local_interp_dir)
                if tft_prev is not None:
                    tft_interp_accumulator.setdefault(alias_prev, []).append(tft_prev)
                    logger.info(
                        "Pre-loaded TFT interpretability for window %d / %s",
                        wi_prev, alias_prev,
                    )
                else:
                    logger.warning(
                        "Could not recover TFT interpretability for window %d / %s "
                        "from local disk – aggregate will be incomplete.",
                        wi_prev, alias_prev,
                    )

    for wi, wb in enumerate(windows):
        if wi < start_from_window:
            continue
        # Convert physical boundaries → model-time boundaries for shifted df slicing
        if config.shift_hours > 0:
            train_start_ts = pd.Timestamp(wb.train_start) - pd.Timedelta(hours=config.shift_hours)
            test_start_ts = pd.Timestamp(wb.test_start) - pd.Timedelta(hours=config.shift_hours)
        else:
            train_start_ts = pd.Timestamp(wb.train_start)
            test_start_ts = pd.Timestamp(wb.test_start)

        logger.info(
            "Window %d/%d  (physical)  train [%s, %s), val [%s, %s), test [%s, %s)",
            wi + 1, len(windows),
            wb.train_start, wb.valid_start,
            wb.valid_start, wb.test_start,
            wb.test_start, wb.test_end,
        )
        if config.shift_hours > 0:
            logger.info(
                "Window %d/%d  (model-time, shift=%dh)  df slice [%s, %s)",
                wi + 1, len(windows), config.shift_hours,
                train_start_ts, test_start_ts,
            )

        window_df = df[
            (df["ds"] >= train_start_ts) & (df["ds"] < test_start_ts)
        ].copy()

        val_hours = int((wb.test_start - wb.valid_start).total_seconds() / 3600)
        train_hours = int((wb.valid_start - wb.train_start).total_seconds() / 3600)
        if per_window_transform is not None:
            window_df = per_window_transform(window_df, train_hours)
            logger.info(
                "Window %d: applied per-window transform (train_hours=%d)",
                wi + 1, train_hours,
            )

        static_df = build_static_df()
        base_config, date_time, dataset_contents, window_group = make_wandb_config(
            config, hist_cov_cols, future_cov_cols,
            window_index=wi, metadata=metadata,
            window_test_start=str(wb.test_start),
            window_test_end=str(wb.test_end),
        )

        for model_name in config.models:
            alias = _model_alias(model_name, config)
            _cs_suffix = f"_cs{config.circ_shift_k_days}d" if config.circ_shift_k_days is not None else ""
            _run_name = f"train_{config.normalized_tso}_{alias}_{date_time}_w{wi}{_cs_suffix}"
            _run_name = _run_name[:128]  # wandb run names must not exceed 128 chars
            run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=_run_name,
                job_type="training-rolling",
                config={
                    **base_config,
                    "model_name": model_name,
                    "model_alias": alias,
                },
                group=window_group,
                reinit="finish_previous",
            )

            pl_logger = WandbLogger(
                experiment=run,
                log_model=False,
            )

            should_persist = _should_persist_checkpoints(config)
            should_prepare_checkpoint = should_persist or config.checkpoint_selection in ("best", "both")
            checkpoint_dir = None
            checkpoint_cb = None
            if should_prepare_checkpoint:
                checkpoint_dir = (
                    Path(config.output_dir)
                    / f"{config.normalized_tso}"
                    / dataset_contents.replace(", ", "_")
                    / f"window_{wi}"
                    / alias
                    / date_time
                )
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_cb = _make_checkpoint_callback(checkpoint_dir)

            model = build_model(
                config,
                model_name,
                future_cov_cols,
                hist_cov_cols,
                pl_logger,
                checkpoint_cb,
            )
            run.config.update({"model_params": dict(model.hparams)}, allow_val_change=True)

            nf = NeuralForecast(
                models=[model],
                freq="h",
                local_scaler_type=config.local_scaler_type,
            )

            nf.fit(df=window_df, static_df=static_df, val_size=val_hours)

            if should_prepare_checkpoint and checkpoint_dir is not None:
                nf_dir = checkpoint_dir / "nf_model"
                nf.save(str(nf_dir), overwrite=True)
                logger.info("Saved NeuralForecast model to %s", nf_dir)
                best_valid_checkpoint = find_best_valid_checkpoint(checkpoint_dir)
                if best_valid_checkpoint:
                    best_valid_checkpoint.rename(nf_dir / CHECKPOINT_BEST_NAME)

                if should_persist:
                    archive_path = None
                    if config.persist_checkpoints_to_wandb:
                        archive_path = _log_checkpoint_artifact(
                            run,
                            checkpoint_dir,
                            alias,
                            date_time,
                            config,
                            window_index=wi,
                        )
                    elif config.checkpoint_compression:
                        archive_path = _archive_checkpoint_dir(
                            checkpoint_dir,
                            config.checkpoint_compression,
                            config.checkpoint_compression_n_threads,
                        )
                    # Keep the archive but drop the uncompressed directory if local
                    # checkpoint persistence is disabled.
                    if archive_path and archive_path.exists():
                        shutil.rmtree(checkpoint_dir, ignore_errors=True)

            # ── Generate & log predictions on window test set ─────────
            _unshifted = df_unshifted if df_unshifted is not None else df
            # Use df_for_prediction as the shifted context for inference so that
            # circular-covariate augmentation does not bleed into test predictions.
            _pred_df = df_for_prediction if df_for_prediction is not None else df
            if df_unshifted is not None or config.shift_hours == 0:
                test_pred_end = wb.test_end - pd.Timedelta(hours=1)
                logger.info(
                    "Generating test predictions for window %d/%d, %s: [%s, %s]",
                    wi + 1, len(windows), alias, wb.test_start, test_pred_end,
                )
                for checkpoint_key in _selected_checkpoints(config.checkpoint_selection):
                    nf_eval = _load_selected_nf(checkpoint_key, nf, checkpoint_dir)
                    checkpoint_split = (
                        f"test_best_valid_window{wi}"
                        if checkpoint_key == "best"
                        else f"test_window{wi}"
                    )
                    test_preds = predict_with_shift_correction(
                        nf=nf_eval,
                        df_shifted=_pred_df,
                        df_unshifted=_unshifted,
                        static_df=static_df,
                        pred_start=wb.test_start,
                        pred_end=test_pred_end,
                        future_cov_cols=future_cov_cols,
                        shift_hours=config.shift_hours,
                        forecast_horizon=config.forecast_horizon,
                        tso=config.tso,
                        holidays_path=config.holidays_path,
                    )
                    if not test_preds.empty:
                        test_preds = prepare_predictions_df(test_preds, _unshifted)
                        log_predictions_to_wandb(
                            run, test_preds, checkpoint_split, alias,
                            tso=config.tso, shift_hours=config.shift_hours,
                            timestamp=date_time, log_table=False,
                        )
                        _save_predictions_locally(
                            eval_dir=config.eval_dir,
                            normalized_tso=config.normalized_tso,
                            model_alias=alias,
                            timestamp=date_time,
                            split_name=checkpoint_split,
                            preds_df=test_preds,
                            window_index=wi,
                        )

                        key = (alias, checkpoint_key)
                        rolling_preds_timestamp.setdefault(key, date_time)
                        if rolling_preds_timestamp[key] != date_time:
                            logger.warning(
                                "Mixed timestamps for %s/%s (%s vs %s). Global file uses first timestamp.",
                                alias,
                                checkpoint_key,
                                rolling_preds_timestamp[key],
                                date_time,
                            )
                        test_preds_with_window = test_preds.copy()
                        test_preds_with_window["window_index"] = wi
                        rolling_preds_accumulator.setdefault(key, []).append(test_preds_with_window)

                    if config.skip_explainability:
                        continue

                    # ── IG explanations for this window ───────────────────────
                    ig_persist_dir = (
                        Path(config.output_dir)
                        / f"{config.normalized_tso}"
                        / dataset_contents.replace(", ", "_")
                        / f"window_{wi}"
                        / "evaluation"
                    )
                    try:
                        win_explanations, win_ig_preds = explain_all_models(
                            nf=nf_eval,
                            df_shifted=_pred_df,
                            df_unshifted=_unshifted,
                            static_df=static_df,
                            pred_start=wb.test_start,
                            pred_end=test_pred_end,
                            future_cov_cols=future_cov_cols,
                            shift_hours=config.shift_hours,
                            forecast_horizon=config.forecast_horizon,
                            tso=config.tso,
                            holidays_path=config.holidays_path,
                        )
                        raw_prefix = "ig_raw_best_checkpoint" if checkpoint_key == "best" else "ig_raw"
                        preds_prefix = "ig_preds_best_checkpoint" if checkpoint_key == "best" else "ig_preds"
                        save_explanation_artifacts(
                            run=run,
                            explanations=win_explanations,
                            predictions=win_ig_preds,
                            model_alias=alias,
                            timestamp=date_time,
                            shift_hours=config.shift_hours,
                            tso=config.tso,
                            output_dir=ig_persist_dir,
                            window_index=wi,
                            raw_artifact_name_prefix=raw_prefix,
                            preds_artifact_name_prefix=preds_prefix,
                        )
                        # TFT built-in interpretability for this window
                        tft_data = save_tft_interpretability(
                            nf=nf_eval,
                            output_dir=ig_persist_dir,
                            run=run,
                            model_alias=alias,
                            window_index=wi,
                            tso=config.tso,
                            timestamp=date_time,
                        )
                        if tft_data is not None:
                            tft_interp_accumulator.setdefault(alias, []).append(tft_data)
                    except Exception:
                        logger.exception(
                            "Error computing IG explanations for window %d, model %s",
                            wi, alias,
                        )

                    if checkpoint_key == "best":
                        _cleanup_unzipped_checkpoint_dir(checkpoint_dir, config)

            else:
                logger.warning(
                    "Skipping predictions: df_unshifted not provided and shift_hours > 0"
                )

            if (
                checkpoint_dir is not None
                and not config.persist_checkpoints
                and not config.persist_checkpoints_to_wandb
            ):
                shutil.rmtree(checkpoint_dir, ignore_errors=True)

            run.finish()
            nf_list.append(nf)

    # ── Save global per-model rolling predictions ─────────────────────────
    for (alias_key, checkpoint_key), preds_parts in rolling_preds_accumulator.items():
        if not preds_parts:
            continue
        preds_all = pd.concat(preds_parts, ignore_index=True)
        dedupe_cols = [c for c in ["unique_id", "ds"] if c in preds_all.columns]
        if dedupe_cols:
            preds_all = preds_all.drop_duplicates(subset=dedupe_cols, keep="last")
        _save_rolling_predictions_global(
            eval_dir=config.eval_dir,
            normalized_tso=config.normalized_tso,
            model_alias=alias_key,
            timestamp=rolling_preds_timestamp[(alias_key, checkpoint_key)],
            checkpoint_key=checkpoint_key,
            preds_df=preds_all,
        )

    # ── Aggregate TFT interpretability across all windows ─────────────────
    if tft_interp_accumulator:
        _, _, dataset_contents_agg, _ = make_wandb_config(
            config, [], [], metadata=metadata
        )
        agg_persist_dir = (
            Path(config.output_dir)
            / f"{config.normalized_tso}"
            / dataset_contents_agg.replace(", ", "_")
            / "evaluation_rolling_avg"
        )
        for alias_key, window_data in tft_interp_accumulator.items():
            logger.info(
                "Aggregating TFT interpretability for %s across %d windows",
                alias_key, len(window_data),
            )
            aggregate_tft_interpretability_windows(
                per_window_data=window_data,
                output_dir=agg_persist_dir,
                model_alias=alias_key,
                tso=config.tso,
            )

    return nf_list
