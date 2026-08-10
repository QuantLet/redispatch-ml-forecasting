"""Central configuration for the paper-results CLI.

All result modules share one ``PaperConfig`` instance, so there is a single
place to adjust paths, regime windows, statistical hyper-parameters, and
output conventions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset / evaluation defaults
# ---------------------------------------------------------------------------

DATASET_NAME = (
    "basic_day_ahead_price_wind_pv_cross_border_production_consumption_bloomberg_sce"
)
DEFAULT_START_DATE = "2025-03-01"

#: Four stress-regime windows used in the per-regime MCS / evaluation tables.
#: Keys are human-readable labels used as column headers in LaTeX tables.
DEFAULT_REGIMES: dict[str, tuple[str, str]] = {
    "high_pv":         ("2025-03-01", "2025-03-31"),
    "pv_distribution": ("2025-04-01", "2025-06-30"),
    "mixed_pv_wind":   ("2025-07-01", "2025-09-30"),
    "winter_reversion":("2025-10-01", "2026-02-21"),
}

#: Canonical seed-agnostic model names accepted on the CLI (``--models``).
#: ``data.resolve_model_columns`` maps these to the actual seed-suffixed
#: column names present in each prediction DataFrame.
DEFAULT_MODELS: list[str] = ["nbeatsx", "nhits", "tft"]

#: Benchmark models are always included and carry no seed suffix.
BENCHMARK_MODELS: list[str] = [
    "lightgbm_regression_rolling",
    "auto_arima_rolling",
    "ridge_regression_rolling",
    "naive_seasonal",
    "seasonal_regression_rolling",
    "lstm",
]

#: Canonical publication display names (seed-agnostic keys).
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "nbeatsx": "NBEATSx",
    "nhits": "NHiTS",
    "tft": "TFT",
    "lstm": "LSTM",
    "auto_arima_rolling": "ARIMA",
    "lightgbm_regression_rolling": "LightGBM",
    "naive_seasonal": "Naive",
    "ridge_regression_rolling": "Ridge",
    "seasonal_regression_rolling": "Seasonal Ridge",
}

#: Columns that are *not* model prediction columns in any predictions DataFrame.
METADATA_COLS: frozenset[str] = frozenset(
    {
        "tso",
        "unique_id",
        "ds",
        "horizon",
        "y",
        "window_index",
        "null",
        "moving_avg_hourly",
        "merge_key",
        "sparsity",
        "metric",
        "regime",
    }
)

# Model default ordering for all tables and figures (benchmarks are always first, in the order defined above).
DEFAULT_MODEL_ORDER: list[str] = BENCHMARK_MODELS + DEFAULT_MODELS


@dataclass
class PaperConfig:
    """Central configuration for the paper-results CLI.

    Parameters
    ----------
    predictions_dir:
        Full-covariate rolling-window predictions root (``outputs/``).
    benchmarks_dir:
        Benchmark predictions root; defaults to *predictions_dir* when ``None``.
    best_checkpoint:
        When True, paper modules read neural-model predictions and IG tensors
        produced from the best validation checkpoint. Benchmark predictions are
        unchanged because they are not checkpointed.
    no_cov_dir:
        No-covariate predictions root (``outputs_no_covariates/``).
    shift_dir:
        Shifted-target predictions root (``outputs_shifted_targets_17/``).
    dataset_name:
        Sub-directory name that identifies the feature set used during training.
    start_date:
        ISO date string; predictions before this date are excluded from all
        evaluations.
    output_dir:
        Root for all paper artifacts.  Sub-directories ``tables/``,
        ``figures/``, and ``style/`` are created automatically.
    figure_format:
        ``"png"`` (default) or ``"pdf"``.
    models:
        Seed-agnostic model names forwarded to ``data.resolve_model_columns``.
    stress_regimes:
        Mapping of regime label → (start_date, end_date) used for per-regime
        outputs.  Defaults to the four notebook windows.
    mcs_alpha:
        Significance level for the MCS procedure (default 0.25).
    fbeta_beta:
        Beta for the F-beta score (default 2.0).
    fbeta_eps_acceptance:
        Epsilon for the bootstrap acceptance rule (default 0.05).
    ablation_seeds:
        Random seeds for the circular-shift ablation module.
    mae_diff_models:
        Two model column names (seed-specific) used in the MAE-difference
        heatmap.  Defaults to ``("nhits_seed778", "tft_seed778")``.
    """

    # --- Input paths ---
    predictions_dir: Path = field(default_factory=lambda: Path("outputs/"))
    benchmarks_dir: Path | None = None  # resolved to predictions_dir in __post_init__
    no_cov_dir: Path = field(default_factory=lambda: Path("outputs_no_covariates/"))
    shift_dir: Path = field(
        default_factory=lambda: Path("outputs_shifted_targets_17/")
    )
    ablation_cache_dir: Path | None = None
    best_checkpoint: bool = False

    # --- Dataset ---
    dataset_name: str = DATASET_NAME
    start_date: str = DEFAULT_START_DATE
    dataset_dir: Path = field(
        default_factory=lambda: Path("data/model_data_paper_actuals_1h_lag")
    )
    train_config_yaml_path: Path = field(default_factory=lambda: Path("training/neural_model_parameters.yaml"))

    # --- Output ---
    output_dir: Path = field(default_factory=lambda: Path("paper_results"))
    figure_format: str = "png"

    # --- Model selection ---
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))

    # --- Regimes ---
    stress_regimes: dict[str, tuple[str, str]] = field(
        default_factory=lambda: dict(DEFAULT_REGIMES)
    )

    # --- Statistical hyper-parameters ---
    mcs_alpha: float = 0.25
    fbeta_beta: float = 2.0
    fbeta_eps_acceptance: float = 0.05

    # --- Ablation ---
    ablation_seeds: list[int] = field(default_factory=lambda: [778, 860])

    # --- DM test ---
    dm_sparsity_threshold: float = 0.7
    dm_alpha: float = 0.05

    # --- MAE diff heatmap ---
    mae_diff_models: tuple[str, str] = ("nhits_seed778", "tft_seed778")

    mae_metrics: list[str] = field(default_factory=lambda: ["mae", "r2_score", "fbeta_2.0"])

    def __post_init__(self) -> None:
        if self.benchmarks_dir is None:
            self.benchmarks_dir = self.predictions_dir

    # --- Output sub-directories (read-only properties) ---

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def style_dir(self) -> Path:
        return self.output_dir / "style"

    def make_output_dirs(self) -> None:
        """Create all output sub-directories."""
        for d in (self.tables_dir, self.figures_dir, self.style_dir):
            d.mkdir(parents=True, exist_ok=True)
