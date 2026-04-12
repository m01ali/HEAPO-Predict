# HEAPO-Predict

A full supervised machine-learning pipeline for predicting daily heat-pump energy consumption using the [HEAPO dataset](https://zenodo.org/records/15056919) (Brudermueller et al. 2025). The pipeline covers every step from raw data ingestion through academic report generation, across thirteen sequential phases.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Analysis Tracks](#3-analysis-tracks)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Models](#5-models)
6. [Key Results](#6-key-results)
7. [Project Structure](#7-project-structure)
8. [Setup & Installation](#8-setup--installation)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [Configuration](#10-configuration)
11. [Outputs Reference](#11-outputs-reference)
12. [Reproducibility](#12-reproducibility)

---

## 1. Project Overview

HEAPO-Predict answers a single core question:

> **Given a household's smart-meter history, weather conditions, and building characteristics, how accurately can we predict its daily heat-pump electricity consumption (kWh)?**

The project trains and evaluates ten regression models — from simple baselines to gradient-boosted trees, a linear family, and an artificial neural network — on 1,298 Swiss households monitored over roughly two years. A second, protocol-enriched track exploits detailed technician inspection data for a treatment subgroup of 109 households.

### What the pipeline does end-to-end

```
Raw HEAPO files
      │
      ▼
Phase 1  Load & validate           ─┐
Phase 2  Clean & flag              ─┤  Data engineering
Phase 3  Merge (Track A / B)       ─┤
Phase 4  Feature engineering       ─┘
      │
      ▼
Phase 5  Exploratory Data Analysis
Phase 6  Train/Val/Test splits + scaling
      │
      ▼
Phase 7  Model training (10 models × 2 tracks)
Phase 8  Hyperparameter tuning (Optuna, Bayesian)
Phase 8b Model refit on final data
      │
      ▼
Phase 9  Evaluation (metrics, CV, significance tests)
Phase 10 Interpretability (SHAP, permutation importance)
Phase 11 Subgroup & bias analysis
      │
      ▼
Phase 12 Academic report generation
Phase 13 Final reproducibility checks
```

---

## 2. Dataset

**Source:** HEAPO — Heat Pump Operation dataset  
**Zenodo record:** `15056919` (pinned; all results are tied to this specific release)  
**Reference:** Brudermueller et al. (2025), arXiv:2503.16993v1

### Dataset components

| File group | Description | Size |
|---|---|---|
| Smart Meter Data (SMD) | One CSV per household, daily kWh totals + reactive energy | ~1,298 files, ~900k rows |
| Metadata | Building survey: type, living area, construction year, HP specs | 1,408 households |
| Protocols | Technician inspection reports for treatment households | 410 rows |
| Weather | 8 regional stations: temperature, sunshine, precipitation | ~8 stations × 2 years |
| Master mapping | Links Household_ID → Weather_ID and group assignment | 1,408 rows |

### Obtaining the data

Download and extract the HEAPO dataset from Zenodo record `15056919` into a directory called `heapo_data/` at the project root (or set `data.dataset_path` in `config/params.yaml`):

```bash
# Using the Zenodo record ID
wget https://zenodo.org/records/15056919/files/<archive>.zip
unzip <archive>.zip -d heapo_data/
```

Column names across all data files are validated against **Tables 1, 4, 5, and 6** of the HEAPO paper before any processing step.

---

## 3. Analysis Tracks

The pipeline runs two parallel tracks that share the same preprocessing but diverge at merging:

### Track A — Full sample

- **Households:** ~1,272 (after minimum-days filter of 180 days)
- **Feature set:** 45 features for tree models; 30 scaled features for linear/ANN models
- **Target:** raw `kWh_received_Total` per day (trees), `log1p(kWh)` for linear/ANN
- **Models:** all 10 models (baselines, linear family, DT, RF, XGBoost, LightGBM, ANN)
- **Purpose:** benchmark performance on the general heat-pump population

### Track B — Protocol-enriched

- **Households:** ~109 treatment households (those with technician visit records)
- **Feature set:** 75 features (Track A features + protocol variables: heating curve temperature, night setback, HP age, technician-flagged issues)
- **Target:** same as Track A
- **Models:** XGBoost B, Ridge B
- **Purpose:** quantify the uplift from including protocol/inspection features

---

## 4. Pipeline Architecture

### Phase 1 — Data Loading (`scripts/01_data_loading.py`)

Loads all five HEAPO data sources, validates schema and row counts against the paper, profiles each dataset, and saves five Parquet files to `data/processed/`.

**Outputs:**
- `data/processed/smd_daily.parquet` — ~900k rows × 15 cols
- `data/processed/metadata.parquet` — survey variables
- `data/processed/protocols.parquet` — 410 rows with `is_orphan`, `visit_number`
- `data/processed/weather_daily.parquet` — 8 stations stacked
- `data/processed/households.parquet` — 1,408-row master mapping
- `outputs/tables/phase1_profiling_report.txt`

---

### Phase 2 — Data Cleaning (`scripts/02_data_cleaning.py`)

Applies rule-based cleaning to each dataset independently:

- **SMD:** hard cap at 500 kWh/day (meter errors), removes households with ≥30% null target rows, per-household IQR flagging (×3.0 multiplier, flags rows but retains them), derives `Date`, `is_iqr_outlier`, `post_intervention`, `has_pv`, `has_reactive_energy`
- **Metadata:** plausibility bounds on construction year (1900–2025), HP install year (1980–2025), HP capacity (2–60 kW), normpoint COP (2.0–6.0), heating curve temperature (15–70 °C), living area flag >1,000 m²
- **Weather:** cross-station temperature consistency check (flags station-days >5 °C from network mean), marks stations with no sunshine sensor (`HbsbG`, `ceOxS`, `sV3mR`), fills short gaps with linear interpolation, adds `sunshine_available`, `interpolated_flag`, `temp_cross_station_flag`
- **Protocols:** removes duplicate visits, assigns `visit_number`, flags orphan records

**Outputs:**
- `data/processed/smd_daily_clean.parquet`
- `data/processed/metadata_clean.parquet`
- `data/processed/protocols_clean.parquet`
- `data/processed/weather_daily_clean.parquet`
- `outputs/tables/phase2_cleaning_report.txt`

---

### Phase 3 — Data Merging (`scripts/03_data_merging.py`)

Joins the four cleaned datasets into two analysis-track Parquets:

- **Track A** (`merged_full.parquet`): SMD × weather × metadata for all ~1,272 households. ~913,620 rows × 47 cols.
- **Track B** (`merged_protocol.parquet`): Track A join × protocol table, restricted to treatment households. ~84,367 rows × 110 cols.

Runs 8 integrity checks (row counts, no duplicate household-days, join completeness).

**Outputs:**
- `data/processed/merged_full.parquet`
- `data/processed/merged_protocol.parquet`
- `outputs/tables/phase3_merge_report.txt`

---

### Phase 4 — Feature Engineering (`scripts/04_feature_engineering.py`)

Builds the full feature matrices on top of the merged Parquets. Feature groups:

| Group | Features | Notes |
|---|---|---|
| Temporal | Month, day-of-week, day-of-year, is_weekend, season | Calendar features |
| Weather | `temp_avg`, `temp_min`, `temp_max`, HDD (base 15 °C), sunshine hours, precipitation | Station-matched per household |
| Weather rolling/lag | 3-day and 7-day rolling temp averages; 1-day temp lag | Configurable via `params.yaml` |
| Household metadata | Living area bucket, building age bucket, HP type, heat distribution type, building type | Ordinal-encoded |
| Reactive energy proxy | `power_factor_proxy` from kVArh inductive/capacitive | Optional (default: on) |
| Protocol features (Track B only) | Heating curve temperature, night setback temperature, HP age, technician issue flags | Adds 30 features |

Runs 6 integrity checks and writes a feature catalog report.

**Outputs:**
- `data/processed/features_full.parquet`
- `data/processed/features_protocol.parquet`
- `outputs/tables/phase4_feature_report.txt`

---

### Phase 5 — Exploratory Data Analysis (`scripts/05_eda.py`)

Eleven EDA tasks producing 35+ figures and two summary tables:

- Target distribution (histograms, log-transformed, per-household means, monthly box plots)
- Target vs. temperature scatter (50,000-row sample with LOWESS smoother)
- Temporal patterns: year overlay, day-of-week, seasonal coverage heatmap
- Univariate distributions for all numeric and categorical features
- Bivariate feature–target relationships
- Correlation matrix + VIF multicollinearity analysis (threshold: |r| > 0.85, VIF > 10)
- Missing data maps (Track A and Track B)
- Subgroup comparisons (HP type, building type, PV presence, EV ownership, treatment/control)
- Protocol EDA (Track B): HP age, heating curve, building age, pre/post-intervention splits

**Outputs:**
- `outputs/figures/05_*.png` (35+ figures)
- `outputs/tables/phase5_eda_summary.txt`
- `outputs/tables/phase5_vif_table.txt`

---

### Phase 6 — Data Preparation (`scripts/06_data_preparation.py`)

Prepares train/validation/test splits and fitting artifacts:

**Temporal split boundaries** (defined in `config/params.yaml`):

| Split | Date range | Track A rows | Track A HH |
|---|---|---|---|
| Train | up to 2023-05-31 | 646,258 | 1,119 |
| Validation | 2023-06-01 – 2023-11-30 | 153,594 | 856 |
| Test | 2023-12-01 – 2024-03-21 | 74,368 | 826 |

- Households with fewer than 180 days of data are dropped
- Median imputation per column with an imputation registry (for inference-time consistency)
- `StandardScaler` fitted on training data only, saved as `scaler_linear_A.pkl` and `scaler_linear_B.pkl`
- 5-fold `GroupKFold` CV folds (grouped by `Household_ID`) for cross-validation
- `log1p` target transformation for linear models / ANN
- Feature lists exported to `phase6_feature_lists.json`

**Outputs:**
- `data/processed/train_full.parquet`, `val_full.parquet`, `test_full.parquet`
- `data/processed/train_protocol.parquet`, `val_protocol.parquet`, `test_protocol.parquet`
- `outputs/models/scaler_linear_A.pkl`, `scaler_linear_B.pkl`
- `outputs/models/imputation_registry.json`
- `outputs/tables/phase6_feature_lists.json`

---

### Phase 7 — Model Training (`scripts/07_model_training.py`)

Trains all models with default hyperparameters on the train split and evaluates on validation:

**Track A models (45 tree features / 30 linear features):**

| Model | Type | Target space |
|---|---|---|
| Baseline: Global Mean | Baseline | raw kWh |
| Baseline: Per-HH Mean | Baseline | raw kWh |
| Baseline: HDD-Linear | OLS on HDD | raw kWh |
| OLS | Linear Regression | log1p(kWh) |
| Ridge | L2-regularised linear | log1p(kWh) |
| Lasso | L1-regularised linear | log1p(kWh) |
| ElasticNet | L1+L2 linear | log1p(kWh) |
| Decision Tree | Tree | raw kWh |
| Random Forest | Ensemble (500 trees) | raw kWh |
| XGBoost | Gradient boosting | raw kWh |
| LightGBM | Gradient boosting | raw kWh |
| ANN (MLP 128→64→1) | Neural network | log1p(kWh) |

**Track B models (75 protocol features):**
- `XGBoost_B` — XGBoost on protocol-enriched features
- `Ridge_B` — Ridge regression on protocol-enriched features (scaled)

All models and predictions are serialised to `outputs/models/`.

---

### Phase 8 — Hyperparameter Tuning (`scripts/08_hyperparameter_tuning.py`)

Bayesian optimisation via **Optuna** using the validation set RMSE (raw kWh) as the objective. The test set is never seen inside an Optuna trial.

| Model | Trials | Search space highlights |
|---|---|---|
| ElasticNet | 30 | alpha, l1_ratio |
| Decision Tree | 40 | max_depth, min_samples_split, min_samples_leaf |
| Random Forest | 60 | n_estimators (≤150 during search, 500 at refit), max_depth, min_samples |
| XGBoost | 80 | learning_rate, max_depth, subsample, colsample, reg terms |
| LightGBM | 80 | learning_rate, num_leaves, max_depth, min_child_samples |
| ANN | 60 | hidden_layer_sizes, activation, alpha, learning_rate_init |
| XGBoost B | 40 | Same space as XGBoost |

Best parameters are saved to `outputs/models/best_params.json`. Optuna study databases are stored in `outputs/models/optuna_studies/`.

### Phase 8b — Model Refit (`scripts/08b_refit_models.py`)

Re-fits tree models (RF, XGBoost, LightGBM, DT, XGBoost B) on the current train split using the best hyperparameters from `best_params.json`. This step exists because Phase 6 was re-run after the initial Phase 7/8 training, making stored tree models incompatible with the current data splits. ANN and ElasticNet are unaffected (they use the scaler, which was already re-fitted).

---

### Phase 9 — Evaluation (`scripts/09_evaluation.py`)

Comprehensive evaluation across four dimensions:

**Metrics computed (all in raw kWh space):**
- RMSE, MAE, R², Median Absolute Error, sMAPE (with 0.5 kWh floor to exclude near-zero summer days)

**Evaluation dimensions:**
1. **Validation and test set metrics** — full model comparison table
2. **5-fold GroupKFold cross-validation** — mean ± std across folds
3. **Seasonal performance** — Winter, Spring, Summer, Autumn breakdown
4. **Statistical significance** — pairwise Wilcoxon signed-rank tests between all model pairs (α = 0.05)

**Diagnostic figures produced:**
- Predicted vs. actual scatter plots
- Residual histograms
- Residuals vs. predicted (heteroscedasticity check)
- Time-series overlays (6 sampled households)
- Seasonal bar charts
- CV error bars
- Data volume vs. accuracy scatter
- Ablation bar plot
- Significance heatmap

---

### Phase 10 — Interpretability (`scripts/10_interpretability.py`)

Ten interpretability tasks:

| Task | Method | Models |
|---|---|---|
| 10.1 | Permutation importance | All 6 Track A models |
| 10.2 | SHAP global (bar + beeswarm) | TreeExplainer (RF/XGB/LGBM/DT), LinearExplainer (EN), KernelExplainer (ANN) |
| 10.3 | SHAP dependence plots | RF, XGBoost — top 5 features |
| 10.4 | SHAP local (waterfall + force) | XGBoost — 4 representative cases |
| 10.5 | Decision Tree structure visualisation | DT |
| 10.6 | ElasticNet standardised coefficients | ElasticNet |
| 10.7 | XGBoost B SHAP | Protocol features (75-dim) |
| 10.8 | Cross-model feature ranking table + heatmap | All models |
| 10.9 | Accuracy–interpretability tradeoff plot | All models |
| 10.10 | Consolidated interpretability report | — |

> **Memory note:** The script defaults to memory-safe settings for 8 GB machines (SHAP sampled to 5,000 rows, 1 permutation job). Comments in the script show full-fidelity settings for ≥32 GB RAM.

---

### Phase 11 — Subgroup & Bias Analysis (`scripts/11_subgroup_analysis.py`)

Examines whether model errors are systematically biased across household characteristics:

**Subgroups analysed:**
- HP type (Air-Source vs. Ground-Source)
- Building type (House vs. Apartment)
- Heat distribution (Floor / Radiator / Both)
- PV presence
- EV ownership
- Living area bucket (<100 / 100–150 / 150–200 / 200–300 / >300 m²)
- Group (Control vs. Treatment)
- Intervention status (pre-visit / post-visit / control)
- Month

**Statistical tests:** Mann-Whitney U (pairwise) + Kruskal-Wallis (multi-group) with Bonferroni correction.

**Track B subgroup analysis:** XGBoost B residuals broken down by building age, heating curve temperature, and night setback temperature.

---

### Phase 12 — Report Generation (`scripts/12_generate_report.py`)

Generates `outputs/report/HEAPO_Predict_Report.md` — a complete academic-style Markdown report. Every number is pulled directly from phase output CSVs (no hardcoding), guaranteeing consistency between reported results and computed artefacts.

---

### Phase 13 — Final Checks (`scripts/13_final_checks.py`)

Seven automated reproducibility checks:

| Check | What it verifies |
|---|---|
| CHECK 1 | All 30 expected output files are present |
| CHECK 2 | Key metric values match Phase 9 CSV values to 3 decimal places |
| CHECK 3 | All 21 report figures are present |
| CHECK 4 | Temporal boundaries, row counts, residual sign convention, feature list completeness |
| CHECK 5 | RF model re-prediction on 100 test rows matches stored predictions (< 1×10⁻³ kWh) |
| CHECK 6 | All 9 required config keys present in `params.yaml` |
| CHECK 7 | Writes `outputs/tables/phase13_final_checks_report.txt` |

Exit code `0` if all pass, `1` if any fail. **All 6/6 checks currently pass.**

---

## 5. Models

### Track A — 10 models

| Model | Val RMSE (kWh) | Val R² | Test RMSE (kWh) | Test R² |
|---|---|---|---|---|
| Random Forest | 7.81 | 0.736 | 11.54 | 0.728 |
| XGBoost | 7.84 | 0.734 | 11.59 | 0.726 |
| LightGBM | 7.92 | 0.729 | 11.65 | 0.723 |
| Decision Tree | 9.23 | 0.631 | 14.44 | 0.575 |
| ANN (MLP 128→64→1) | 9.87 | 0.579 | 15.56 | 0.506 |
| ElasticNet | 12.18 | 0.359 | 20.40 | 0.151 |
| Baseline: HDD-Linear | 14.52 | 0.088 | 21.08 | 0.094 |
| Baseline: Per-HH Mean | 17.46 | −0.317 | 20.32 | 0.158 |
| Baseline: Global Mean | 18.68 | −0.509 | 24.61 | −0.235 |

### Track B — Protocol-enriched

| Model | Val RMSE (kWh) | Val R² | Test RMSE (kWh) | Test R² |
|---|---|---|---|---|
| XGBoost B | 5.81 | 0.841 | 8.42 | 0.847 |
| Ridge B | 10.18 | 0.513 | — | — |

**XGBoost B** is the best-performing model overall, achieving R² = 0.847 on the test set by leveraging protocol inspection features available only for treatment households.

**Random Forest** is the best Track A model (R² = 0.728, RMSE = 11.54 kWh/day on the test set), outperforming XGBoost and LightGBM by a small margin.

---

## 6. Key Results

### Model performance summary (test set, raw kWh space)

- **Best Track A model:** Random Forest — RMSE 11.54 kWh, MAE 7.47 kWh, R² 0.728
- **Best overall model:** XGBoost B (Track B) — RMSE 8.42 kWh, MAE 6.06 kWh, R² 0.847
- **Linear models** perform near the HDD baseline, confirming the non-linear relationship between weather and heat-pump consumption
- **Protocol features** provide a substantial uplift: XGBoost B improves over Track A XGBoost by ~27% in RMSE

### Top predictive features (SHAP, Track A RF)

1. `temp_avg_rolling_3d` — 3-day rolling average temperature (dominant driver)
2. `hdd_15` — heating degree days (base 15 °C)
3. `temp_avg_lag_1d` — yesterday's temperature
4. `Survey_Building_LivingArea` — household floor area
5. `kvarh_received_inductive_total` — inductive reactive energy (HP proxy)

### Subgroup bias findings

- Ground-source HP households are systematically underestimated (higher consumption, harder to predict)
- Households with radiator heat distribution show higher residuals than floor heating
- Treatment households pre-visit have the largest prediction errors (sparse training data)
- Model accuracy is consistent across PV and EV subgroups

### Reproducibility

All 6/6 final checks pass. RF re-prediction error < 4.3×10⁻¹⁴ kWh (floating-point precision only).

---

## 7. Project Structure

```
HEAPO-Predict/
│
├── config/
│   └── params.yaml                  # Central config — all hyperparameters and thresholds
│
├── data/
│   └── processed/                   # Parquet files written by each phase (git-ignored)
│       ├── smd_daily.parquet
│       ├── smd_daily_clean.parquet
│       ├── metadata.parquet
│       ├── metadata_clean.parquet
│       ├── protocols.parquet
│       ├── protocols_clean.parquet
│       ├── weather_daily.parquet
│       ├── weather_daily_clean.parquet
│       ├── households.parquet
│       ├── merged_full.parquet
│       ├── merged_protocol.parquet
│       ├── features_full.parquet
│       ├── features_protocol.parquet
│       ├── train_full.parquet        train_protocol.parquet
│       ├── val_full.parquet          val_protocol.parquet
│       └── test_full.parquet         test_protocol.parquet
│
├── heapo_data/                      # Raw HEAPO dataset (not tracked, must be downloaded)
│
├── outputs/
│   ├── figures/                     # All PNG plots (05_*.png, phase10_*.png, phase11_*.png)
│   ├── logs/                        # Per-phase run logs
│   ├── models/                      # Serialised models (.pkl, .json, .txt) + Optuna studies
│   ├── report/                      # HEAPO_Predict_Report.md (academic report)
│   └── tables/                      # CSV metrics, TXT reports, JSON feature lists
│
├── scripts/                         # Orchestration scripts — one per phase
│   ├── 00_smoke_test.py
│   ├── 01_data_loading.py
│   ├── 02_data_cleaning.py
│   ├── 03_data_merging.py
│   ├── 04_feature_engineering.py
│   ├── 05_eda.py
│   ├── 06_data_preparation.py
│   ├── 07_model_training.py
│   ├── 08_hyperparameter_tuning.py
│   ├── 08b_refit_models.py
│   ├── 09_evaluation.py
│   ├── 10_interpretability.py
│   ├── 11_subgroup_analysis.py
│   ├── 12_generate_report.py
│   └── 13_final_checks.py
│
├── src/                             # Library modules imported by scripts
│   ├── __init__.py
│   ├── ann.py                       # sklearn MLPRegressor wrapper (ANN)
│   ├── data_cleaner.py
│   ├── data_loader.py
│   ├── data_merger.py
│   ├── data_preparation.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── feature_engineer.py
│   ├── interpretability.py
│   ├── models.py                    # All model definitions + metrics
│   ├── preprocessing.py
│   └── subgroup_analysis.py
│
├── CLAUDE.md                        # AI assistant instructions
├── README.md                        # This file
└── .venv/                           # Python virtual environment (not tracked)
```

---

## 8. Setup & Installation

### Prerequisites

- Python 3.11 or 3.12 (tested on 3.14 on Apple M1; PyTorch is not used — sklearn MLP only)
- ~4 GB disk space for processed Parquets and model artifacts
- ~8 GB RAM minimum (16 GB recommended for full-fidelity SHAP in Phase 10)

### Install

```bash
# Clone the repository
git clone https://github.com/m01ali/HEAPO-Predict.git
cd HEAPO-Predict

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Core dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `pyarrow` | Parquet I/O |
| `scikit-learn` | Linear models, DT, RF, ANN, scalers, CV |
| `xgboost` | XGBoost gradient boosting |
| `lightgbm` | LightGBM gradient boosting |
| `optuna` | Bayesian hyperparameter optimisation |
| `shap` | SHAP explanations |
| `matplotlib`, `seaborn` | Visualisation |
| `scipy` | Statistical tests (Wilcoxon, Mann-Whitney, Kruskal-Wallis) |
| `joblib` | Model serialisation |
| `pyyaml` | Config loading |

---

## 9. Running the Pipeline

All scripts must be run from the **project root** directory. Activate the virtual environment first.

```bash
source .venv/bin/activate
```

### Option A — Run all phases in sequence

```bash
python scripts/01_data_loading.py
python scripts/02_data_cleaning.py
python scripts/03_data_merging.py
python scripts/04_feature_engineering.py
python scripts/05_eda.py
python scripts/06_data_preparation.py
python scripts/07_model_training.py
python scripts/08_hyperparameter_tuning.py
python scripts/08b_refit_models.py
python scripts/09_evaluation.py
python scripts/10_interpretability.py
python scripts/11_subgroup_analysis.py
python scripts/12_generate_report.py
python scripts/13_final_checks.py
```

### Option B — Smoke test first

Before running the full pipeline, verify that imports and the config load correctly:

```bash
python scripts/00_smoke_test.py
```

### Option C — Start from a checkpoint

If you already have processed Parquets from a previous run, you can resume from any phase. Each script only reads its declared input files from `data/processed/` and `outputs/`. For example, to re-run evaluation only:

```bash
python scripts/09_evaluation.py
```

### Approximate runtimes (Apple M1 Air, 8 GB RAM)

| Phase | Runtime |
|---|---|
| 01 Data loading | ~2 min |
| 02 Data cleaning | ~1 min |
| 03 Data merging | ~1 min |
| 04 Feature engineering | ~2 min |
| 05 EDA | ~3 min |
| 06 Data preparation | ~2 min |
| 07 Model training | ~15 min |
| 08 Hyperparameter tuning | ~3–6 hours (configurable via `n_trials_*`) |
| 08b Model refit | ~20 min |
| 09 Evaluation | ~5 min |
| 10 Interpretability | ~15 min (memory-safe) / ~60 min (full fidelity) |
| 11 Subgroup analysis | ~3 min |
| 12 Report generation | ~1 min |
| 13 Final checks | <1 min |

> **Tuning time:** Phase 8 is the most expensive step. To reduce it, lower `n_trials_rf`, `n_trials_xgb`, and `n_trials_lgbm` in `config/params.yaml`. The stored `best_params.json` can be reused via Phase 8b without re-running Phase 8.

---

## 10. Configuration

All pipeline parameters live in `config/params.yaml`. Nothing is hardcoded across scripts.

### Key sections

```yaml
data:
  dataset_path: "./heapo_data/"      # Path to extracted HEAPO dataset
  zenodo_record_id: 15056919         # Pinned Zenodo release
  min_days_threshold: 180            # Minimum days per household

splits:
  train_end: "2023-05-31"            # Training window end
  val_end:   "2023-11-30"            # Validation window end
  test_end:  "2024-03-21"            # Test window end (dataset end)

modeling:
  random_seed: 42
  xgboost_early_stopping_rounds: 50
  ann_early_stopping_patience: 15
  shap_background_samples: 200       # KernelExplainer background size (ANN)
  optuna_n_trials: 80                # Default trials (overridden per model below)

evaluation:
  mape_floor_kwh: 0.5                # Exclude days below this from sMAPE
  cv_n_splits: 5                     # GroupKFold folds
  stat_test_alpha: 0.05              # Wilcoxon / Mann-Whitney alpha
  n_bootstrap: 1000                  # Bootstrap CI iterations

tuning:
  n_trials_rf:   60                  # Optuna trials per model
  n_trials_xgb:  80
  n_trials_lgbm: 80
  n_trials_ann:  60
  rf_n_estimators_search: 150        # Cap during search; final refit uses 500
  rf_n_estimators_final:  500

cleaning:
  smd_hard_cap_kwh: 500              # Above this = meter error
  null_fraction_threshold: 0.30      # Drop household if ≥30% target rows null
  iqr_multiplier: 3.0                # Per-household IQR flag multiplier

feature_engineering:
  include_reactive_energy: true      # Power factor proxy feature
  include_autoregressive: false      # Lag/rolling kWh features (off by default)
  rolling_windows_days: [3, 7]       # Rolling temp window sizes
  lag_days: [1]                      # Temp lag days
```

> **Note on autoregressive features:** Setting `include_autoregressive: true` enables lag/rolling kWh consumption features. This requires that Phase 6 enforces a strictly temporal (non-shuffled) split, which it already does. However, these features were disabled by default in this study to keep the feature set clean and comparable with the HEAPO paper's own analysis.

---

## 11. Outputs Reference

### Tables (`outputs/tables/`)

| File | Contents |
|---|---|
| `phase1_profiling_report.txt` | Row counts, null rates, dtype summary per dataset |
| `phase2_cleaning_report.txt` | Records removed/flagged per cleaning rule |
| `phase3_merge_report.txt` | Join statistics, integrity check results |
| `phase4_feature_report.txt` | Feature catalog with dtype and null rate |
| `phase5_eda_summary.txt` | Key EDA findings (distribution stats, VIF flags) |
| `phase5_vif_table.txt` | Variance inflation factors for all numeric features |
| `phase6_feature_lists.json` | Exact feature column lists for tree / linear / protocol models |
| `phase6_preparation_report.txt` | Split sizes, imputation decisions, scaler stats |
| `phase7_training_report.txt` | Validation metrics for all models at default hyperparameters |
| `phase8_tuning_report.txt` | Best hyperparameters and val RMSE per model |
| `phase9_metrics_val.csv` | Validation set: RMSE, MAE, R², MedAE, sMAPE for all models |
| `phase9_metrics_test.csv` | Test set metrics |
| `phase9_metrics_cv.csv` | 5-fold CV mean ± std |
| `phase9_metrics_seasonal.csv` | Per-season metrics |
| `phase9_wilcoxon_matrix.csv` | Pairwise Wilcoxon p-values |
| `phase9_ablation_metrics.csv` | Feature ablation results |
| `phase9_test_predictions.parquet` | Row-level predictions for all Track A models |
| `phase9_test_predictions_b.parquet` | Row-level predictions for XGBoost B |
| `phase10_permutation_importance.csv` | Permutation importance scores |
| `phase10_shap_mean_abs.csv` | Mean absolute SHAP per feature per model |
| `phase10_feature_ranking_table.csv` | Cross-model feature ranking |
| `phase10_interpretability_report.txt` | Consolidated interpretability findings |
| `phase11_subgroup_metrics.csv` | Per-subgroup RMSE/MAE/bias for all Track A models |
| `phase11_mannwhitney_results.csv` | Mann-Whitney U test results with Bonferroni correction |
| `phase11_subgroup_composition.csv` | Household and row counts per subgroup |
| `phase11_subgroup_report.txt` | Full subgroup analysis narrative |
| `phase11_track_b_subgroup_metrics.csv` | XGBoost B subgroup metrics (Track B) |
| `phase13_final_checks_report.txt` | Pass/fail status for all 6 reproducibility checks |

### Models (`outputs/models/`)

| File | Contents |
|---|---|
| `model_rf_tuned.pkl` | Tuned Random Forest (500 trees) |
| `model_xgboost_tuned.pkl` / `.json` | Tuned XGBoost |
| `model_lgbm_tuned.pkl` / `.txt` | Tuned LightGBM |
| `model_dt_tuned.pkl` | Tuned Decision Tree |
| `model_elasticnet_tuned.pkl` | Tuned ElasticNet |
| `model_ann_tuned.pkl` | Tuned ANN (sklearn MLPRegressor) |
| `model_xgboost_b_tuned.pkl` / `.json` | Tuned XGBoost B (Track B) |
| `scaler_linear_A.pkl` | StandardScaler for Track A linear/ANN models |
| `scaler_linear_B.pkl` | StandardScaler for Track B linear models |
| `imputation_registry.json` | Median imputation values per column |
| `best_params.json` | Best hyperparameters from Optuna |
| `optuna_studies/` | Optuna SQLite study databases |
| `baseline_hh_means.parquet` | Per-household mean kWh (for Per-HH baseline) |
| `baseline_hdd_linear.pkl` | Fitted HDD-linear baseline model |

### Logs (`outputs/logs/`)

Every phase writes a structured log to `outputs/logs/phaseN_run.log`. Logs include timestamps, INFO/WARNING records, and a copy of all `print()` output for full traceability.

---

## 12. Reproducibility

This pipeline is designed to be fully reproducible:

- **Pinned dataset version:** Zenodo record `15056919`
- **Fixed random seeds:** `random_seed: 42` used in all models, bootstrap, and sampling
- **Temporal data split:** no data leakage — scaler and imputer fitted on train only; test set is never seen during tuning
- **Config-driven:** no magic numbers in scripts
- **Column validation:** all column names are checked against the HEAPO paper (Tables 1, 4, 5, 6) before use
- **Final check script:** Phase 13 verifies file existence, metric consistency, model reproducibility, and config completeness on every run

To reproduce results from scratch:

```bash
# 1. Download the dataset (Zenodo record 15056919) into heapo_data/
# 2. Install dependencies
pip install -r requirements.txt
# 3. Run the full pipeline
for i in 01 02 03 04 05 06 07 08 08b 09 10 11 12 13; do
    python scripts/${i}_*.py
done
# 4. Verify
python scripts/13_final_checks.py   # expect: 6/6 checks passed
```

---

## Citation

If you use this pipeline or build on it, please cite the underlying dataset:

> Brudermueller et al. (2025). *HEAPO: A Dataset for Heat Pump Operation Prediction.* arXiv:2503.16993v1.
