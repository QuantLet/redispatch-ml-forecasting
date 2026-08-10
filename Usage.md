# Day-ahead Forecasting for Redispatch Measures using Machine Learning Models

Reproducibility package for the paper **"Day-ahead Forecasting for Redispatch Measures using Machine Learning Models."**
This repository contains the full data-processing, training, and results-generation code. Every table and figure in the paper is regenerated from **saved model predictions** by the `paper_results_cli` module — no model retraining is required to reproduce the reported results.

- **Paper:**
- **Code:**
- **Prediction artifacts:** hosted on Zenodo - `https://zenodo.org/records/21867081` (see [Data & prediction artifacts](#data--prediction-artifacts))

---

## Table of contents

1. [Overview](#overview)
2. [Repository structure](#repository-structure)
3. [Installation](#installation)
4. [Data & prediction artifacts](#data--prediction-artifacts)
5. [Reproducing the paper](#reproducing-the-paper)
7. [CLI reference](#cli-reference)
9. [Citation](#citation-license-contact)

---

## Overview

The pipeline has three stages:

1. **Data preparation** (`dataset_preparation/`, `data/`) — assembles the redispatch target series and exogenous covariates from public ENTSO-E Transparency and Netztransparenz data.
2. **Model training** (`training/`, `scripts/`) — trains the benchmark and neural forecasters (Naive, ARIMA, Ridge, LightGBM, LSTM, NBEATSx, NHiTS, TFT) under the rolling-window protocol and writes per-window predictions to disk.
3. **Results generation** (`paper_results_cli/`) — consumes the saved predictions (csv/parquet) and produces every table and figure in the paper.

Reproducing the **paper results** only requires stages 3 + the saved predictions; stages 1–2 are provided for full transparency but are compute-intensive (GPU) and not needed to regenerate the reported numbers.

---

## Repository structure

```
redispatch-ml-forecasting/  (branch: new_dataset)
├── dataset_preparation/         # Build target + covariate datasets from raw sources
├── data/                        # Processed datasets and metadata
│   └── model_data_paper_actuals_1h_lag/ # final paper datasets (tracked in Git)
├── training/                    # Training machinery
├── scripts/                     # Rolling-window training entrypoints
├── covariate_effect/            # Covariate ablation experiment support
├── check_input_size_ablation.ipynb   # Lag-size (input window) selection results
├── paper_results/               # Default output dir for generated tables & figures
└── paper_results_cli/           # Results-generation package (main entrypoint)
    ├── __main__.py              #   CLI entrypoint (python -m paper_results_cli)
    ├── config.py                #   Configuration
    ├── data.py                  #   Prediction loading / alignment
    ├── dm_wrapper.py            #   Diebold–Mariano test wrapper
    ├── tables.py                #   Shared table formatting
    ├── style.py                 #   Figure style + colour registry
    ├── plot_best_checkpoint_windows.py   # Prediction-window figure
    ├── monthly_congestion_cost_table.py # ENTSO-E redispatching costs
    ├── download_paper_artifacts.py # Bring training artifacts locally
    ├── package_paper_artifacts.py # Prepare artifacts for storage
    └── modules/
        ├── descriptive.py       #   Descriptive stats + ACF/PACF
        ├── mae.py               #   MAE evaluation + per-horizon MAE
        ├── fbeta.py             #   F-beta bootstrap dominance
        ├── mcs.py               #   Model Confidence Set
        ├── ablation.py          #   Circular-shift DM dominance
        └── interpretability.py  #   Integrated-Gradients feature importance
```

---

## Installation

Tested on Python 3.11. A CUDA GPU is only needed for stage 1–2 (training); results generation runs on CPU. Conda-like enviroment is given as reference. 

Recommended path:

```bash
git clone -b new_dataset https://github.com/vlad-bolovaneanu-ase/redispatch-ml-forecasting.git
cd redispatch-ml-forecasting

conda create -n --file redispatch_env.yml
```

---

## Data & prediction artifacts

The final datasets are tracked in `data/model_data_paper_actuals_1h_lag/`.

Data can also be fetched via the ENTSO-E Transparency API:

1. Create a root-level `.env` file with the `ENTSOE_API_KEY` key. If you don't have an API key, follow [this](https://www.amsleser.no/blog/post/21-obtaining-api-token-from-entso-e) tutorial.
1. Run the `data_processing/entsoe_data_fetching.ipynb` notebook until the `Summary` section (the other datasets were not used). This operation may take some time.
1. Execute the `dataset_preparation/run_generation.sh` helper. Explore the README file in the directory beforehand.

Predictions, ablation inputs, and raw Integrated Gradients tensors are archived
externally because they are too large for the Git repository:

**Download:** `https://doi.org/10.5281/zenodo.21867081` or `https://zenodo.org/records/21867081`

Download, verify, and restore all external artifacts with:

```bash
cd scripts
python download_paper_artifacts.py --all
```

For a selective restore, use any combination of `--predictions`,
`--interpretability`, and `--ablations`. Existing archives and destination
files are protected unless `--overwrite` is explicitly supplied.

After downloading, the artifacts are unpacked into the repository root on their expected paths:

```
redispatch-ml-forecasting/
├── outputs_paper/                  # Full-covariate predictions + NHiTS IG
├── outputs_no_covariates/          # No-covariate predictions (ablation H1)
├── outputs_shifted_targets_17_paper/ # Circular-shift predictions (ablation H2)
├── outputs_nhits_only_seed860_paper/ # Seed-860 full-covariate ablation
├── outputs_shifted_targets_17_seed860_paper/ # Ablation H2 for the second seed
├── outputs_no_covariates_nhits_only_seed860/ # No-covariate predictions for the second seed (ablation H1)
└── data/
    └── model_data_paper_actuals_1h_lag/ # supplied by Git
```

---

## Reproducing the paper

### Outputs only

Regenerate into `paper_results/`:

```bash
python3 -m paper_results_cli \
    --predictions-dir outputs_paper \
    --shift-dir outputs_shifted_targets_17_paper \
    --interpretability-ig-root outputs_paper \
    --interpretability-dataset-root data/model_data_paper_actuals_1h_lag \
    --dataset-root-dir-path data/model_data_paper_actuals_1h_lag \
    --dataset-name basic_day_ahead_price_wind_pv_production_consumption_sce \
    --interpretability-model-filter nhits \
    --best-checkpoint \
    --all \
    --models "lstm" "lightgbm_regression_rolling" "nhits" "nbeatsx" "tft"

```

or 

```bash
source scripts/get_paper_results.sh
```

Or run individual result modules:

```bash
python -m paper_results_cli --descriptive        # descriptive stats + ACF/PACF
python -m paper_results_cli --mae                # MAE tables + per-horizon MAE
python -m paper_results_cli --fbeta              # F-beta bootstrap dominance
python -m paper_results_cli --mcs                # Model Confidence Set tables/heatmap
python -m paper_results_cli --ablation           # circular-shift DM dominance
python -m paper_results_cli --interpretability   # IG feature-group importance
```

The prediction-window figure is produced by:

```bash
python -m paper_results_cli.plot_best_checkpoint_windows \
  --predictions-dir outputs_paper \
  --dataset-name basic_day_ahead_price_wind_pv_production_consumption_sce \
  --dataset-dir data/model_data_paper_actuals_1h_lag \
  --windows 11 \
  --models nbeatsx \
  --tso TenneT_DE
```

The congestion management cost table is produced by:

```bash
python3 -m paper_results_cli.monthly_congestion_cost_table \
  --input_file data/redispatching_costs/entsoe_redispatching_congestion_costs.csv \
  --output_file paper_results/tables/monthly_congestion_cost_table.tex
```

### Training

Two important prerequisites:
- A GPU with at least 16 GB of VRAM
- Weights&Biases (wandb) account - a free one should suffice, but there is the possibility of a free Academia account, see [this](https://community.wandb.ai/t/how-to-upgrade-my-team-to-an-academic-account/7135) post.

1. Authenticate using [this](https://docs.wandb.ai/models/ref/cli/wandb-login) guide. Add the following keys to the project-root `.env` file: `WANDB_API_KEY`, `WANDB_PROJECT_NAME`, `WANDB_ENTITY`.
1. Familiarize with the configuration options available in `training/runner.py`. We recommend specifying common arguments in a yaml configuration file, then overriding specifics for each training.
1. Start the session of your choice. There are prepared recipes in the `scripts` directory. For example, the main training loop is exemplified in `scripts/rolling_window/rolling_window_prediction_pipeline_paper.sh`.

---

## CLI reference

```
python -m paper_results_cli [MODULE FLAGS] [OPTIONS]
```

**Module flags:** `--all`, `--descriptive`, `--mae`, `--fbeta`, `--mcs`, `--ablation`, `--interpretability`

**I/O paths** (defaults in parentheses):

| Option | Default | Purpose |
|---|---|---|
| `--predictions-dir` | `outputs/` | full-covariate rolling-window predictions |
| `--benchmarks-dir` | = predictions-dir | benchmark predictions root |
| `--no-cov-dir` | `outputs_no_covariates/` | no-covariate predictions (ablation) |
| `--shift-dir` | `outputs_shifted_targets_17/` | circular-shift predictions (ablation) |
| `--output-dir` | `paper_results` | destination for tables/figures |
| `--ablation-cache-dir` | `pred_cache/` | parquet cache for ablation (empty string disables) |
| `--interpretability-ig-root` | `outputs_nhits_only_seed860_paper` | per-TSO IG outputs |
| `--interpretability-dataset-root` | `data/model_data_paper_actuals_1h_lag` | dataset metadata for IG |
| `--dataset-root-dir-path` | `data/model_data_paper_actuals_1h_lag` | dataset root (descriptive) |
| `--train-config-yaml-path` | `training/neural_model_parameters.yaml` | training config (descriptive) |

**Statistical parameters** (paper defaults):

| Option | Default | Meaning |
|---|---|---|
| `--mcs-alpha` | `0.25` | MCS significance level |
| `--fbeta-beta` | `2.0` | β for the F-beta score |
| `--fbeta-eps` | `0.05` | bootstrap acceptance epsilon |
| `--ablation-seeds` | `778 860` | training seeds for ablation |
| `--dm-sparsity-threshold` | `0.7` | sparsity split for DM dominance |
| `--dm-alpha` | `0.05` | DM test significance level |
| `--interpretability-start-window` | `2` | first rolling window for IG |
| `--interpretability-top-n-features` | `5` | ranked feature-level IG attributions to keep per TSO/direction/block |
| `--best-checkpoint` | off | use best-validation-checkpoint predictions |
| `--figure-format` | `png` | `png` or `pdf` |
| `--start-date` | dataset default | evaluation start date |
| `--models` | defaults + benchmarks | models to include |
| `--regimes-file` | built-in regimes | JSON override for stress regimes |
| `-v, --verbose` | off | DEBUG logging |

Custom stress-regime JSON (`--regimes-file`):

```json
{
  "high_pv": ["2025-03-01", "2025-03-31"],
  "pv_distribution": ["2025-04-01", "2025-06-30"]
}
```

---

## Citation, license, contact

**Citation:**

```bibtex
@article{bolovaneanu_redispatch_2026,
  title  = {Day-ahead Forecasting for Redispatch Measures using Machine Learning Models},
  author = {Basangova, M. and Bolovaneanu, V. and Conda, A. and Pele, D. T. and Erlwein-Sayer, C. and Melzer, A. and Petukhina, A. and Phan, M.},
  year   = {2026},
  note   = {Preprint / under review}
}
```
