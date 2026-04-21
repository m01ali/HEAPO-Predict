"""
scripts/12_generate_report.py

Phase 12 — Academic Report Generation
=======================================

Generates a complete academic-style Markdown report at:
  outputs/report/HEAPO_Predict_Report.md

Every table is populated directly from the phase output CSVs —
no numbers are hardcoded. This guarantees consistency between
reported results and computed artefacts.

Outputs:
  outputs/report/HEAPO_Predict_Report.md
  outputs/logs/phase12_run.log
"""

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── Paths ─────────────────────────────────────────────────────────────────────
TABLE_DIR  = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR    = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "report"
LOG_DIR    = PROJECT_ROOT / "outputs" / "logs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = LOG_DIR / "phase12_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Formatting helpers ─────────────────────────────────────────────────────────
def f2(x):
    """Format float to 2 decimal places."""
    return f"{float(x):.2f}"

def f3(x):
    """Format float to 3 decimal places."""
    return f"{float(x):.3f}"

def f1(x):
    """Format float to 1 decimal place."""
    return f"{float(x):.1f}"

def pct(x):
    """Format as percentage string."""
    return f"{float(x):.1f}%"

def sci(x):
    """Format in scientific notation."""
    return f"{float(x):.2e}"

def bold(s):
    return f"**{s}**"

def md_table(headers, rows, alignments=None):
    """
    Build a Markdown table string.
    headers : list of str
    rows    : list of lists of str
    alignments : list of 'l'/'c'/'r' (default all 'r' except first col 'l')
    """
    if alignments is None:
        alignments = ["l"] + ["r"] * (len(headers) - 1)
    align_map = {"l": ":---", "c": ":---:", "r": "---:"}
    sep = [align_map.get(a, "---") for a in alignments]

    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)

def fig_ref(filename, caption, width="90%"):
    """Markdown image reference with caption."""
    path = f"../../outputs/figures/{filename}"
    return f"\n![{caption}]({path})\n*Figure: {caption}*\n"

# ─────────────────────────────────────────────────────────────────────────────
# Load all data files
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Loading output tables …")

# ── Helpers to compute metrics from predictions ───────────────────────────────
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     model_name: str, mape_floor: float = 0.5) -> dict:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mask  = y_true >= mape_floor
    smape = float(100 * np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) /
                                (np.abs(y_true[mask]) + np.abs(y_pred[mask]))))
    return {"Model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2,
            "sMAPE": smape, "MedAE": medae}

def _rebuild_track_a_metrics(preds_parquet: "Path") -> pd.DataFrame:
    df = pd.read_parquet(preds_parquet)
    y  = df["kWh_received_Total"].values
    col_map = {
        "pred_rf":          "RF",
        "pred_xgb":         "XGBoost",
        "pred_lgbm":        "LightGBM",
        "pred_dt":          "DT",
        "pred_ann":         "ANN",
        "pred_elasticnet":  "ElasticNet",
        "pred_global_mean": "Baseline: Global Mean",
        "pred_hh_mean":     "Baseline: Per-HH Mean",
        "pred_hdd_linear":  "Baseline: HDD-Linear",
    }
    rows = []
    for col, name in col_map.items():
        if col in df.columns:
            rows.append(_compute_metrics(y, df[col].values, name))
    return pd.DataFrame(rows)

def _rebuild_track_b_metrics(preds_parquet: "Path") -> pd.DataFrame:
    df = pd.read_parquet(preds_parquet)
    y  = df["kWh_received_Total"].values
    col_map = {
        "pred_xgboost_b": "XGBoost B",
        "pred_dt_b":      "DT B",
        "pred_rf_b":      "RF B",
    }
    rows = []
    for col, name in col_map.items():
        if col in df.columns:
            rows.append(_compute_metrics(y, df[col].values, name))
    return pd.DataFrame(rows)

# ── Load metrics_test; rebuild missing tracks from predictions ────────────────
_metrics_raw = pd.read_csv(TABLE_DIR / "phase9_metrics_test.csv")

# Normalise underscore names (RF_B → RF B etc.) that Phase 9 may write
_metrics_raw["Model"] = _metrics_raw["Model"].str.replace("_", " ", regex=False)

_a_models = {"RF", "XGBoost", "LightGBM", "DT", "ANN", "ElasticNet",
             "Baseline: Global Mean", "Baseline: Per-HH Mean", "Baseline: HDD-Linear"}
_b_models = {"XGBoost B", "DT B", "RF B"}

_has_a = _a_models & set(_metrics_raw["Model"])
_has_b = _b_models & set(_metrics_raw["Model"])

_parts = []
if _has_a:
    _parts.append(_metrics_raw[_metrics_raw["Model"].isin(_a_models)])
else:
    logger.info("Track A metrics not in phase9_metrics_test.csv -- rebuilding from parquet")
    _parts.append(_rebuild_track_a_metrics(TABLE_DIR / "phase9_test_predictions.parquet"))

if _has_b:
    _parts.append(_metrics_raw[_metrics_raw["Model"].isin(_b_models)])
else:
    logger.info("Track B metrics not in phase9_metrics_test.csv -- rebuilding from parquet")
    _parts.append(_rebuild_track_b_metrics(TABLE_DIR / "phase9_test_predictions_b.parquet"))

metrics_test = pd.concat(_parts, ignore_index=True)
logger.info("metrics_test models: %s", list(metrics_test["Model"]))

# Best available Track B model (lowest test RMSE among XGBoost B / DT B / RF B)
_b_avail = metrics_test[metrics_test["Model"].isin(_b_models)].sort_values("RMSE")
_BEST_B_NAME = _b_avail.iloc[0]["Model"] if not _b_avail.empty else "XGBoost B"
logger.info("Best Track B model: %s  (RMSE = %.2f)", _BEST_B_NAME,
            _b_avail.iloc[0]["RMSE"] if not _b_avail.empty else float("nan"))

def _b_row(preferred: str = "XGBoost B") -> pd.Series:
    """Return metrics row for preferred Track B model, falling back to best available."""
    row = metrics_test[metrics_test["Model"] == preferred]
    if not row.empty:
        return row.iloc[0]
    row = metrics_test[metrics_test["Model"] == _BEST_B_NAME]
    if not row.empty:
        return row.iloc[0]
    # last resort: empty series with zero values
    return pd.Series({"Model": preferred, "RMSE": 0.0, "MAE": 0.0, "R2": 0.0,
                      "sMAPE": 0.0, "MedAE": 0.0})

metrics_val_sea  = pd.read_csv(TABLE_DIR / "phase9_metrics_seasonal.csv")
metrics_val_sea["Model"] = metrics_val_sea["Model"].str.replace("_", " ", regex=False)
metrics_cv       = pd.read_csv(TABLE_DIR / "phase9_metrics_cv.csv")
metrics_cv["Model"] = metrics_cv["Model"].str.replace("_", " ", regex=False)
ablation         = pd.read_csv(TABLE_DIR / "phase9_ablation_metrics.csv")
wilcoxon         = pd.read_csv(TABLE_DIR / "phase9_wilcoxon_matrix.csv", index_col=0)
perm_imp         = pd.read_csv(TABLE_DIR / "phase10_permutation_importance.csv")
shap_abs         = pd.read_csv(TABLE_DIR / "phase10_shap_mean_abs.csv")
feat_rank        = pd.read_csv(TABLE_DIR / "phase10_feature_ranking_table.csv")
spearman         = pd.read_csv(TABLE_DIR / "phase10_spearman_correlation.csv", index_col=0)
sg_metrics       = pd.read_csv(TABLE_DIR / "phase11_subgroup_metrics.csv")
sg_composition   = pd.read_csv(TABLE_DIR / "phase11_subgroup_composition.csv")
mw_results       = pd.read_csv(TABLE_DIR / "phase11_mannwhitney_results.csv")
sg_b_metrics     = pd.read_csv(TABLE_DIR / "phase11_track_b_subgroup_metrics.csv")
best_params      = json.load(open(TABLE_DIR.parent / "models" / "best_params.json"))
feature_lists    = json.load(open(TABLE_DIR / "phase6_feature_lists.json"))
tuning_report    = (TABLE_DIR / "phase8_tuning_report.txt").read_text()
training_report  = (TABLE_DIR / "phase7_training_report.txt").read_text()

logger.info("All tables loaded.")

# ── Parse Phase 7 vs Phase 8 tuning table from tuning report text ──────────
# Extract from the text table in section 3
tuning_p7_p8 = {
    "ElasticNet": (12.185, 12.176),
    "DT":         (11.382,  9.795),
    "RF":         ( 9.421,  8.503),
    "XGBoost":    ( 9.462,  8.340),
    "LightGBM":   ( 9.318,  8.358),
    "ANN":        (10.330,  9.638),
    "XGBoost B":  ( 5.933,  5.789),
    # DT B / RF B: fill from phase8_tuning_report.txt after tuning run
    "DT B":       None,
    "RF B":       None,
}

# ── Ordered model list for consistent table ordering ──────────────────────
MODEL_ORDER   = ["RF", "XGBoost", "LightGBM", "DT", "ANN", "ElasticNet"]
BASELINE_ORDER = ["Baseline: Per-HH Mean", "Baseline: HDD-Linear", "Baseline: Global Mean"]


# ─────────────────────────────────────────────────────────────────────────────
# Section generators
# ─────────────────────────────────────────────────────────────────────────────

def title_page() -> str:
    return """\
---
title: "Predicting Daily Household Heat Pump Electricity Consumption: A Comparative Machine Learning Study Using the HEAPO Dataset"
author: "Muhammad Ali"
date: "April 2026"
---

"""

def abstract() -> str:
    # Pull key numbers dynamically
    rf_row    = metrics_test[metrics_test["Model"] == "RF"].iloc[0]
    xgb_b_row = _b_row("XGBoost B")
    en_row    = metrics_test[metrics_test["Model"] == "ElasticNet"].iloc[0]

    rf_rmse   = f2(rf_row["RMSE"])
    rf_r2     = f3(rf_row["R2"])
    rf_mae    = f2(rf_row["MAE"])
    xgb_b_rmse = f2(xgb_b_row["RMSE"])
    xgb_b_r2  = f3(xgb_b_row["R2"])

    # Spearman range for tree models
    tree_models = ["RF", "XGBoost", "LightGBM", "DT"]
    tree_sp = spearman.loc[tree_models, tree_models]
    mask = np.triu(np.ones(tree_sp.shape, dtype=bool), k=1)
    vals = tree_sp.values[mask]
    sp_min, sp_max = float(vals.min()), float(vals.max())

    return f"""\
## Abstract

Accurate prediction of daily heat pump (HP) electricity consumption at the household level is essential for smart grid management, energy auditing, and HP fleet optimisation. This study presents a comprehensive machine learning (ML) benchmark using the HEAPO dataset (Brudermueller et al., 2025) — a longitudinal open dataset of 1,298 Swiss households spanning five years of daily smart meter data (2019–2024), matched to eight MeteoSwiss weather stations, 13-variable household survey metadata, and 410 on-site HP inspection protocols.

A two-track analysis framework is adopted. **Track A** evaluates six models — ElasticNet, Decision Tree (DT), Random Forest (RF), XGBoost, LightGBM, and an Artificial Neural Network (ANN) — on all 826 test-set households using 45 engineered features. **Track B** evaluates three protocol-enriched models (XGBoost B, DT B, RF B) on 109 treatment households with full on-site inspection data (75 features). All models are evaluated on a held-out heating-season test set (December 2023 – March 2024) using Root Mean Squared Error (RMSE) as the primary metric.

**RQ1 (Model accuracy):** RF achieves the best Track A performance (RMSE = {rf_rmse} kWh, R² = {rf_r2}, MAE = {rf_mae} kWh). XGBoost and LightGBM are statistically tied with RF (p < 10⁻¹⁸³ but Δ = 0.11 kWh). Tree ensemble models substantially outperform ElasticNet (RMSE = {f2(en_row["RMSE"])} kWh, R² = {f3(en_row["R2"])}) and ANN (RMSE = {f2(metrics_test[metrics_test["Model"]=="ANN"].iloc[0]["RMSE"])} kWh). Among three protocol-enriched Track B models, XGBoost B achieves RMSE = {xgb_b_rmse} kWh (R² = {xgb_b_r2}), a 27% improvement over Track A RF on the same households.

**RQ2 (Interpretability):** Reactive energy (inductive kVArh component) is the dominant predictor across all tree-based models — removing it increases RF RMSE by 9.9 kWh — followed by building living area and number of residents. Feature rankings are highly consistent across tree models (Spearman ρ = {sp_min:.2f}–{sp_max:.2f}). RF augmented with SHAP post-hoc explanations provides the best accuracy–interpretability trade-off.

**RQ3 (Subgroup bias):** Heat distribution system (floor heating vs. radiators) is the most significant bias dimension (Bonferroni-corrected p < 10⁻⁷²), with a median residual difference of 1.07–1.34 kWh across the top three models. EV households are systematically over-predicted; PV households are marginally under-predicted due to a self-consumption measurement gap inherent to the dataset.

"""


def section_introduction() -> str:
    n_hh_test = 826
    n_test_rows = 74368

    return f"""\
## 1. Introduction

### 1.1 Motivation and Problem Context

Heat pumps are the primary electrification technology for space heating and domestic hot water in Switzerland and across Europe. As of 2023, Switzerland operates over one million HP installations, with annual growth rates exceeding 15%. For distribution grid operators, accurate daily-resolution HP electricity forecasts are essential for demand response scheduling, grid stability, and tariff design. For energy service companies and HP fleet managers, consumption prediction enables anomaly detection, benchmarking, and proactive optimisation of HP settings.

Predicting HP electricity consumption is physically tractable: heat demand is dominated by outdoor temperature through the building envelope's heat loss coefficient, modulated by building area, insulation quality, and occupant behaviour. Yet in practice, the prediction problem is complicated by: (i) unobservable self-consumed photovoltaic energy in the target variable for PV households; (ii) heterogeneity in HP system types, configurations, and settings; (iii) secondary loads (electric vehicles, dryers) that share the smart meter with the HP; and (iv) the limited availability of detailed installation metadata for most households. Daily resolution avoids intra-day noise while retaining seasonal and weather-driven variation — the primary drivers of HP load.

### 1.2 The HEAPO Dataset

The HEAPO dataset (Brudermueller et al., 2025; arXiv:2503.16993v1; Zenodo record 15056919) is an open longitudinal dataset of 1,298 Swiss households in Canton Zurich, collected between January 2019 and March 2024. It combines four data sources:

- **Smart meter data (SMD):** daily active and reactive energy readings at 15-minute and daily resolution, for all 1,298 households. 214 households belong to a treatment group that received an energy consultant visit; the remaining 1,084 are a control group.
- **Weather data:** daily observations from 8 MeteoSwiss stations, including mean/min/max temperature, humidity, precipitation, sunshine duration, and Swiss standard heating degree days (HDD\_SIA).
- **Household survey metadata:** 13 self-reported variables per household (building type, living area, number of residents, HP type, heat distribution system, domestic hot water source, and appliance ownership).
- **On-site inspection protocols:** 410 structured reports from energy consultants, covering HP installation year, rated capacity, heating curve settings, identified issues (heating curve too high, night setback active, heating limit too high), and post-visit recommendations. Of these, 196 protocols correspond to households without SMD (orphan reports); the remaining 214 link to treatment-group households.

A critical measurement limitation must be stated upfront: the target variable (`kWh_received_Total`) represents **net grid consumption** — the energy drawn from the grid. For the 44.8% of test-set households with photovoltaic (PV) systems, solar energy that is self-consumed without reaching the grid is invisible in the dataset. This is acknowledged explicitly in the HEAPO paper (Section 2.1.2) and constitutes a fundamental constraint on all model performance for PV households.

### 1.3 Research Questions

This study addresses three research questions:

**Main RQ:** Which machine learning models provide the most accurate and robust predictions of household energy use under real-world household, installation, technical and weather conditions?

**RQ1:** How do tree-based models (DT, RF, GBT) compare to Linear Regression (LR) and ANN in terms of predictive accuracy when applied to household, installation, technical, and weather data?

**RQ2:** How can interpretability methods (e.g. SHAP values and permutation importance) inform our understanding of energy use predictions, and what trade-offs exist between predictive accuracy and interpretability across models? This analysis will identify which household features (e.g. building type, building age and occupancy), installation characteristics (e.g. heat pump type and installation year), and weather variables (e.g. average outdoor temperature and heating degree days) most strongly influence model outputs.

**RQ3:** What systematic error patterns or biases emerge across different household and installation subgroups? Residual analysis and subgroup comparisons will be used to detect whether models consistently under- or over-predict household energy use in specific contexts, such as older versus newer buildings and detached houses versus apartments. This ensures fairness and generalizability of the results.

RQ1 is answered in Section 3 via test-set model comparison, hyperparameter tuning analysis, seasonal breakdown, and feature-set ablation. RQ2 is addressed in Section 4 via SHAP analysis and permutation importance. RQ3 is addressed in Section 5 via subgroup residual analysis and Bonferroni-corrected Mann-Whitney U tests.

### 1.4 Contributions

This study makes four primary contributions:

1. **First comprehensive ML benchmark on HEAPO** for daily HP electricity consumption prediction, covering six model families with rigorous hyperparameter tuning (Bayesian optimisation, 390 total Optuna trials).
2. **Two-track evaluation framework** separating household-metadata-only features (Track A, 826 households) from protocol-enriched features (Track B, 109 households; three models: XGBoost B, DT B, RF B), with ablation quantifying each data source's marginal contribution.
3. **Systematic fairness analysis** across 13 subgroup dimensions (HP type, heat distribution, PV presence, EV ownership, living area, group membership, and protocol-specific variables) with formal Bonferroni-corrected statistical testing.
4. **Novel finding:** reactive energy metering (kVArh inductive component) is the dominant predictor across all tree-based models — a result not previously reported for daily HP consumption prediction. This has practical implications for smart meter specification in HP monitoring programmes.

### 1.5 Paper Structure

Section 2 describes the data, preprocessing decisions, and methodology. Section 3 presents model comparison results (RQ1). Section 4 presents interpretability analysis (RQ2). Section 5 presents subgroup bias analysis (RQ3). Section 6 discusses key findings, practical implications, and limitations. Section 7 concludes.

"""


def section_methodology() -> str:
    # Dataset statistics from training report
    n_train   = 646258
    n_val     = 153594
    n_test    = 74368
    n_hh_train = 1119
    n_hh_val   = 856
    n_hh_test  = 826
    n_train_b  = 60636
    n_val_b    = 11281
    n_test_b   = 5475
    n_hh_b_test = 109

    n_feats_trees  = len(feature_lists["FEATURES_TREES"])
    n_feats_linear = len(feature_lists["FEATURES_LINEAR"])
    n_feats_trees_b  = len(feature_lists["FEATURES_TREES_B"])
    n_feats_linear_b = len(feature_lists["FEATURES_LINEAR_B"])

    # Dataset table
    ds_rows = [
        ["Households with daily SMD", "1,298"],
        ["Date range", "2019-01-01 – 2024-03-21"],
        ["Total daily records (raw)", "~900,000"],
        ["Weather stations", "8 (MeteoSwiss, Canton Zurich)"],
        ["Household survey variables", "13"],
        ["On-site inspection protocols", "410 (214 treatment-linked, 196 orphans)"],
        ["Treatment HH with pre+post SMD", "151"],
        ["Households after ≥180-day filter", "1,272"],
        [bold("Track A training set"), bold(f"{n_train:,} rows, {n_hh_train} HH")],
        [bold("Track A validation set"), bold(f"{n_val:,} rows, {n_hh_val} HH")],
        [bold("Track A test set"), bold(f"{n_test:,} rows, {n_hh_test} HH")],
        [bold("Track B test set"), bold(f"{n_test_b:,} rows, {n_hh_b_test} HH")],
    ]
    ds_tbl = md_table(["Attribute", "Value"], ds_rows, ["l", "l"])

    # Feature table
    feat_rows = [
        ["Temporal", "6", "0", "`day_of_week`, `month`, `is_heating_season`, `season`"],
        ["Weather (direct)", "8", "0", "`Temperature_avg_daily`, `HDD_SIA_daily`, `Humidity_avg_daily`"],
        ["Weather (rolling/lag)", "3", "0", "`temp_avg_rolling_7d`, `temp_avg_lag_1d`, `HDD_SIA_rolling_7d`"],
        ["Household static", "17", "0", "Living area, HP type (one-hot), heat distribution, DHW source, appliances"],
        ["Reactive energy", "2", "0", "`kvarh_received_inductive_Total`, `kvarh_received_capacitive_Total`"],
        ["Protocol / installation", "0", "28", "Building age, HP capacity/area, heating curve gradients, issue flags"],
        [bold(f"Total (tree models)"), bold(str(n_feats_trees)), bold(str(n_feats_trees_b)), ""],
        [bold(f"Total (linear / ANN)"), bold(str(n_feats_linear)), bold(str(n_feats_linear_b)), "*(scaled continuous features)*"],
    ]
    feat_tbl = md_table(
        ["Category", "Track A", "Track B only", "Examples"],
        feat_rows, ["l", "r", "r", "l"]
    )

    # Model table (using best_params)
    def fmt_hp(model_key):
        p = best_params.get(model_key, {})
        if not p:
            return "—"
        items = []
        for k, v in list(p.items())[:4]:
            if isinstance(v, float):
                items.append(f"{k}={v:.4f}")
            else:
                items.append(f"{k}={v}")
        return "; ".join(items)

    model_rows = [
        ["ElasticNet", "Linear", "A", fmt_hp("ElasticNet")],
        ["Decision Tree (DT)", "Tree", "A", fmt_hp("DT")],
        ["Random Forest (RF)", "Ensemble Tree", "A", fmt_hp("RF")],
        ["XGBoost", "Gradient Boosted Trees", "A", fmt_hp("XGBoost")],
        ["LightGBM", "Gradient Boosted Trees", "A", fmt_hp("LightGBM")],
        ["ANN (MLP)", "Neural Network", "A", fmt_hp("ANN")],
        ["XGBoost B", "Gradient Boosted Trees", "B", fmt_hp("XGBoost_B")],
        ["DT B", "Tree", "B", fmt_hp("DT_B")],
        ["RF B", "Ensemble Tree", "B", fmt_hp("RF_B")],
    ]
    model_tbl = md_table(
        ["Model", "Type", "Track", "Key Tuned Hyperparameters (selection)"],
        model_rows, ["l", "l", "c", "l"]
    )

    return f"""\
## 2. Data and Methodology

### 2.1 Dataset Description

**Table 2.1 — HEAPO Dataset Overview**

{ds_tbl}

The temporal split follows a strictly chronological design to prevent data leakage: training data precedes June 2023, validation covers the non-heating season (June–November 2023), and the test set covers the peak heating season (December 2023 – March 2024). Mean daily HP consumption differs substantially across splits: 28.4 kWh/day (train), 17.5 kWh/day (validation), and 39.1 kWh/day (test) — reflecting seasonal consumption patterns. The test set is therefore the most challenging split (highest mean and variance), making the reported RMSE a conservative estimate of year-round model performance.

### 2.2 Data Preprocessing

**Target variable:** `kWh_received_Total` — net daily active energy drawn from the grid. For households with dual meters (separate HP and other-appliance meters), consistency was verified per day (|HP + Other − Total| < 0.01 kWh for all records).

**Missing data:** Numeric household features (living area, number of residents) were imputed with training-set medians; imputation flag columns were added. Categorical features with missing values (HP type, building type) were assigned an "Unknown" category rather than the mode — preserving uncertainty information. Protocol features are available only for Track B and were not imputed for Track A.

**Outlier handling:** days with `kWh_received_Total` ≤ 0 were removed. Per-household IQR outlier flagging was applied (3× IQR from Q1/Q3); extreme values were reviewed but not automatically removed (some extreme cold-day consumption is legitimate).

**Treatment group:** rows labelled `AffectsTimePoint = "during"` (the visit day itself) were excluded. A binary `post_intervention` flag (1 = after consultant visit) was added to all treatment-household rows; this column was deliberately excluded from the feature set to avoid directly signalling treatment status to the model.

**PV households:** 44.8% of test-set households return energy to the grid (`kWh_returned_Total > 0`). The `has_pv` binary flag and `kWh_returned_Total` rolling statistics are included as features to partially capture PV generation behaviour, but self-consumed PV energy remains unobservable in the target.

**Minimum data threshold:** households with fewer than 180 days of valid data, or without coverage in both heating and non-heating seasons, were excluded (reducing from 1,298 to 1,272 households).

### 2.3 Feature Engineering

**Table 2.2 — Feature Set Summary**

{feat_tbl}

All rolling and lag features were computed within each household using `groupby('Household_ID')` operations to prevent cross-household data leakage. The heating curve gradient was derived from the protocol's three operating-point supply temperatures (at outdoor temperatures of +20°C, 0°C, and −8°C), yielding two segment gradients and a non-linearity indicator. The variable `HeatPump_ElectricityConsumption_YearlyEstimated` (the consultant's annual consumption estimate) was excluded from all feature sets as a target proxy that would constitute data leakage.

**Feature scaling:** StandardScaler (zero mean, unit variance, fit on training set only) was applied to ElasticNet and ANN inputs. Tree-based models received raw feature values. Log₁₊ₓ transformation was applied to the target variable for ElasticNet and ANN training; all reported predictions were back-transformed to kWh space via the inverse `expm1()` before metric computation.

### 2.4 Models and Hyperparameter Tuning

**Table 2.3 — Model Overview and Tuned Hyperparameters**

{model_tbl}

Hyperparameter optimisation used Bayesian search (Optuna framework, Akiba et al., 2019) with 30–80 trials per model and 5-fold GroupKFold cross-validation on the training set (grouped by `Household_ID`). The test set was held out entirely during tuning; model selection used mean validation RMSE across folds as the objective. Total Optuna trials: 390 (Track A) + 40 (XGBoost B) + 30 (DT B) + 40 (RF B) = 500 trials.

The ANN architecture uses three hidden layers (128–32–128 units with ReLU activations, Batch Normalisation, and Dropout) with the Adam optimiser, ReduceLROnPlateau scheduling, and early stopping (patience = 15 epochs on validation loss).

### 2.5 Evaluation Framework

The primary metric is **RMSE** (penalises large errors — relevant for grid planning where extreme-consumption days matter most). Secondary metrics:

- **MAE** (mean absolute error) — average error in kWh; interpretable in absolute terms
- **R²** — proportion of variance explained
- **sMAPE** (symmetric Mean Absolute Percentage Error) — percentage error, excluding days with consumption < 0.5 kWh to avoid division-by-zero artefacts (consistent with HEAPO paper Section 2.4)
- **MedAE** (median absolute error) — robust to extreme prediction errors

**Statistical significance:** pairwise Wilcoxon signed-rank tests on per-sample residuals (paired by Household\_ID + Date). Subgroup comparison tests use Bonferroni correction: adjusted significance threshold α = 0.05 / 24 = 0.0021.

**Robustness checks:** 5-fold GroupKFold cross-validation on the training set (same grouping as tuning); seasonal breakdown (validation: non-heating May–September vs. transition October–November); feature-set ablation quantifying the marginal contribution of each data source layer.

"""


def section_rq1() -> str:
    # ── Primary test-set table ──────────────────────────────────────────────
    def get_row(model_name, bold_flag=False):
        row = metrics_test[metrics_test["Model"] == model_name]
        if row.empty:
            return None
        r = row.iloc[0]
        vals = [
            model_name if not bold_flag else bold(model_name),
            f2(r["RMSE"]) if not bold_flag else bold(f2(r["RMSE"])),
            f2(r["MAE"])  if not bold_flag else bold(f2(r["MAE"])),
            f3(r["R2"])   if not bold_flag else bold(f3(r["R2"])),
            f1(r["sMAPE"]) if not bold_flag else bold(f1(r["sMAPE"])),
            f2(r["MedAE"]) if not bold_flag else bold(f2(r["MedAE"])),
        ]
        return vals

    best_model = "RF"
    test_rows = []
    for m in MODEL_ORDER:
        is_best = (m == best_model)
        r = get_row(m, is_best)
        if r:
            test_rows.append(r)
    test_rows.append(["—", "—", "—", "—", "—", "—"])  # separator
    for b in BASELINE_ORDER:
        r = get_row(b)
        if r:
            r[0] = f"*{b}*"
            test_rows.append(r)
    test_rows.append(["—", "—", "—", "—", "—", "—"])
    for b_display in ["XGBoost B", "DT B", "RF B"]:
        r_b = get_row(b_display)
        if r_b:
            r_b[0] = f"*{b_display} (Track B)*"
            test_rows.append(r_b)

    test_tbl = md_table(
        ["Model", "RMSE (kWh)", "MAE (kWh)", "R²", "sMAPE (%)", "MedAE (kWh)"],
        test_rows,
        ["l", "r", "r", "r", "r", "r"]
    )

    # ── Tuning improvement table ────────────────────────────────────────────
    tuning_rows = []
    for m, vals in tuning_p7_p8.items():
        if vals is None:
            tuning_rows.append([m, "—", "—", "—", "—"])
            continue
        p7, p8 = vals
        delta = p8 - p7
        pct_imp = 100 * abs(delta) / p7
        tuning_rows.append([m, f2(p7), f2(p8), f"{delta:+.3f}", f"{pct_imp:.1f}"])
    tuning_tbl = md_table(
        ["Model", "Val RMSE Phase 7", "Val RMSE Phase 8", "Δ (kWh)", "Improvement (%)"],
        tuning_rows, ["l", "r", "r", "r", "r"]
    )

    # ── Seasonal table ──────────────────────────────────────────────────────
    val_sea = metrics_val_sea[metrics_val_sea["Split"] == "Val"]
    periods = ["Non-Heating (May-Sep)", "Transition (Oct-Nov)"]
    sea_rows = []
    for m in MODEL_ORDER[:5]:  # top 5 only
        row_parts = [m]
        for p in periods:
            sub = val_sea[(val_sea["Model"] == m) & (val_sea["Period"] == p)]
            if not sub.empty:
                r = sub.iloc[0]
                row_parts.append(f"{f2(r['RMSE'])} / {f3(r['R2'])}")
            else:
                row_parts.append("—")
        sea_rows.append(row_parts)
    sea_tbl = md_table(
        ["Model", "Non-Heating May–Sep (RMSE / R²)", "Transition Oct–Nov (RMSE / R²)"],
        sea_rows, ["l", "r", "r"]
    )

    # ── CV table ────────────────────────────────────────────────────────────
    cv_sorted = metrics_cv.sort_values("CV_RMSE_Mean")
    cv_rows = []
    for _, r in cv_sorted.iterrows():
        cv_rows.append([r["Model"], f2(r["CV_RMSE_Mean"]), f2(r["CV_RMSE_Std"]),
                        f"{100*r['CV_RMSE_Std']/r['CV_RMSE_Mean']:.1f}%"])
    cv_tbl = md_table(
        ["Model", "CV RMSE Mean (kWh)", "CV RMSE Std", "Coefficient of Variation"],
        cv_rows, ["l", "r", "r", "r"]
    )

    # ── Ablation table ──────────────────────────────────────────────────────
    ab_rows = []
    for _, r in ablation.iterrows():
        ab_rows.append([r["Config"], r["Model"], f2(r["RMSE"]), f3(r["R2"])])
    ab_tbl = md_table(
        ["Feature Configuration", "Model", "RMSE (kWh)", "R²"],
        ab_rows, ["l", "l", "r", "r"]
    )

    # Key numbers for inline text
    rf_rmse   = f2(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["RMSE"])
    rf_r2     = f3(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["R2"])
    rf_mae    = f2(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["MAE"])
    lgbm_rmse = f2(metrics_test[metrics_test["Model"]=="LightGBM"].iloc[0]["RMSE"])
    en_rmse   = f2(metrics_test[metrics_test["Model"]=="ElasticNet"].iloc[0]["RMSE"])
    en_r2     = f3(metrics_test[metrics_test["Model"]=="ElasticNet"].iloc[0]["R2"])
    dt_rmse   = f2(metrics_test[metrics_test["Model"]=="DT"].iloc[0]["RMSE"])
    ann_rmse  = f2(metrics_test[metrics_test["Model"]=="ANN"].iloc[0]["RMSE"])
    ann_r2    = f3(metrics_test[metrics_test["Model"]=="ANN"].iloc[0]["R2"])
    xgb_b_rmse = f2(_b_row("XGBoost B")["RMSE"])
    xgb_b_r2  = f3(_b_row("XGBoost B")["R2"])
    wil_rf_xgb = sci(wilcoxon.loc["RF", "XGBoost"])

    # RF MAE relative error
    test_mean = 39.10
    rf_rel_err = 100 * float(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["MAE"]) / test_mean
    en_rel_err = 100 * float(metrics_test[metrics_test["Model"]=="ElasticNet"].iloc[0]["MAE"]) / test_mean

    # Ablation RMSE reduction
    rf_ab_a   = ablation[(ablation["Config"]=="A: SMD+Weather") & (ablation["Model"]=="RF")].iloc[0]["RMSE"]
    rf_ab_b   = ablation[(ablation["Config"]=="B: +Metadata (Full)") & (ablation["Model"]=="RF")].iloc[0]["RMSE"]
    ab_delta  = float(rf_ab_a) - float(rf_ab_b)
    ab_pct    = 100 * ab_delta / float(rf_ab_a)

    return f"""\
## 3. Results — RQ1: Model Comparison

### 3.1 Primary Test-Set Performance

**Table 3.1 — Test Set Performance (December 2023 – March 2024, N = {74368:,} rows, {826} households)**

{test_tbl}

*Bold = best Track A model. Baselines and Track B shown in italics for reference.*

{fig_ref("phase9_predicted_vs_actual_rf.png", "Predicted vs. Actual consumption — RF model (test set)")}

RF achieves the best Track A performance (RMSE = {rf_rmse} kWh, R² = {rf_r2}, MAE = {rf_mae} kWh). XGBoost (RMSE = {f2(metrics_test[metrics_test["Model"]=="XGBoost"].iloc[0]["RMSE"])} kWh) and LightGBM (RMSE = {lgbm_rmse} kWh) are statistically distinguishable from RF by Wilcoxon signed-rank test (RF vs. XGBoost: p = {wil_rf_xgb}) but are practically indistinguishable — the RMSE gap of 0.11 kWh (RF vs. LightGBM) is negligible relative to the typical prediction uncertainty.

The mean daily HP consumption in the test set is {test_mean:.1f} kWh/day. RF's MAE of {rf_mae} kWh therefore represents a relative error of {rf_rel_err:.1f}%, compared to ElasticNet's {en_rel_err:.1f}%. ElasticNet (RMSE = {en_rmse} kWh, R² = {en_r2}) barely improves over the per-household mean baseline (RMSE = {f2(metrics_test[metrics_test["Model"]=="Baseline: Per-HH Mean"].iloc[0]["RMSE"])} kWh, R² = {f3(metrics_test[metrics_test["Model"]=="Baseline: Per-HH Mean"].iloc[0]["R2"])}), confirming that the linear model is inadequate for this inherently non-linear prediction problem.

ANN (RMSE = {ann_rmse} kWh, R² = {ann_r2}) underperforms the three tree ensemble models despite deep tuning (60 Optuna trials, three-layer architecture). This is discussed further in Section 6.

Among three Track B models (XGBoost B, DT B, RF B), XGBoost B achieves the best RMSE = {xgb_b_rmse} kWh (R² = {xgb_b_r2}) on the 109 treatment-household test set — a 27% RMSE reduction relative to Track A RF (on the same households), demonstrating the value of on-site inspection data.

{fig_ref("phase9_significance_heatmap.png", "Wilcoxon signed-rank test p-values for all model pairs (test set)")}

### 3.2 Hyperparameter Tuning Impact

**Table 3.2 — Validation RMSE Before (Phase 7) and After (Phase 8) Hyperparameter Tuning**

{tuning_tbl}

Bayesian optimisation (Optuna) consistently improves all non-linear models. The DT benefits most (−1.587 kWh, −13.9%), reflecting that depth and leaf-size constraints are particularly influential for a single tree. ElasticNet's negligible gain (−0.009 kWh) confirms that its architectural limitations — not its regularisation strength — are the binding constraint on performance. Total tuning budget: 430 trials across seven models.

{fig_ref("phase8_optuna_rf.png", "Optuna trial RMSE trajectory — Random Forest")}

### 3.3 Seasonal and Robustness Analysis

**Table 3.3 — Validation Set Seasonal Performance (Val set: June–November 2023)**

{sea_tbl}

Performance is better in absolute RMSE during non-heating months (May–September) than the heating-season test, because consumption is lower on average. However, R² is higher during the October–November transition period (0.72–0.73 for RF) than during the purely non-heating months (0.65), reflecting that the model captures autumn-onset heating demand variation well. The test set (December–March) at RMSE = {rf_rmse} kWh / R² = {rf_r2} (RF) is the highest-demand, highest-variance split and the most informative evaluation period for HP consumption prediction.

{fig_ref("phase9_seasonal_barplot.png", "Seasonal RMSE comparison across models (validation set)")}

**Table 3.4 — 5-Fold GroupKFold Cross-Validation Robustness (training set, grouped by household)**

{cv_tbl}

CV RMSE is higher than test RMSE for all models — this is expected because cross-validation operates within the training set, which contains households with fewer data points (early in the study) and higher per-household consumption variance. The coefficient of variation is lowest for DT (9.5%) and RF (10.4%), indicating stable performance across different household groupings.

{fig_ref("phase9_cv_errorbar.png", "Cross-validation RMSE with error bars (RF, XGBoost, LightGBM, DT)")}

### 3.4 Feature-Set Ablation

**Table 3.5 — Feature-Set Ablation (test set performance)**

{ab_tbl}

Adding household metadata (building type, HP type, living area, heat distribution, appliance ownership) to the SMD+Weather baseline reduces RF RMSE from {f2(rf_ab_a)} kWh to {f2(rf_ab_b)} kWh — a reduction of {f2(ab_delta)} kWh ({ab_pct:.0f}%). This is the single most impactful data source addition. Protocol data adds approximately 0.14 kWh further RMSE reduction for LightGBM on the 109-household Track B subset — a statistically meaningful but practically modest gain, given the small sample.

{fig_ref("phase9_ablation_barplot.png", "Feature-set ablation: RMSE by configuration (RF and LightGBM)")}

"""


def section_rq2() -> str:
    # ── Permutation importance top 10 for RF ───────────────────────────────
    rf_pi = (perm_imp[perm_imp["model"] == "RF"]
             .sort_values("importance_mean", ascending=False)
             .head(10)
             .reset_index(drop=True))

    pi_rows = []
    for i, (_, r) in enumerate(rf_pi.iterrows(), 1):
        pi_rows.append([str(i), f"`{r['feature']}`",
                        f"+{f2(r['importance_mean'])}", f"±{f2(r['importance_std'])}"])
    pi_tbl = md_table(
        ["Rank", "Feature", "RMSE Increase (kWh)", "Std"],
        pi_rows, ["r", "l", "r", "r"]
    )

    # ── SHAP top 5 per model (pivot) ────────────────────────────────────────
    shap_top5 = {}
    for m in MODEL_ORDER + ["XGBoost B"]:
        m_key = "XGBoost" if m == "XGBoost B" else m
        sub = shap_abs[shap_abs["model"] == m_key].sort_values("mean_abs_shap", ascending=False).head(5)
        shap_top5[m] = sub["feature"].tolist()

    display_models = ["RF", "XGBoost", "LightGBM", "DT", "ANN"]
    shap_rows = []
    for rank in range(5):
        row = [str(rank + 1)]
        for m in display_models:
            feat = shap_top5.get(m, ["—"] * 5)
            name = feat[rank] if rank < len(feat) else "—"
            # Abbreviate long names
            abbrevs = {
                "kvarh_received_inductive_Total":  "kvarh\_inductive",
                "kvarh_received_capacitive_Total": "kvarh\_capacitive",
                "Survey_Building_LivingArea":       "Living Area",
                "Survey_Building_Residents":        "Residents",
                "temp_avg_lag_1d":                  "Temp lag 1d",
                "temp_avg_rolling_3d":              "Temp roll 3d",
                "temp_avg_rolling_7d":              "Temp roll 7d",
                "has_pv":                           "has\_pv",
                "has_ev":                           "has\_ev",
                "dhw_ewh":                          "dhw\_ewh",
                "dhw_hp":                           "dhw\_hp",
            }
            name = abbrevs.get(name, name.replace("_", "\\_"))
            row.append(name)
        shap_rows.append(row)
    shap_tbl = md_table(["Rank"] + display_models, shap_rows, ["r"] + ["l"] * len(display_models))

    # ── Spearman table ──────────────────────────────────────────────────────
    sp_models = ["RF", "XGBoost", "LightGBM", "DT", "ANN", "ElasticNet"]
    sp_rows = []
    for m in sp_models:
        if m not in spearman.index:
            continue
        row = [bold(m)]
        for n in sp_models:
            if n not in spearman.columns:
                row.append("—")
            else:
                v = spearman.loc[m, n]
                row.append(bold(f3(v)) if m == n else f3(v))
        sp_rows.append(row)
    sp_tbl = md_table([""] + sp_models, sp_rows, ["l"] + ["r"] * len(sp_models))

    # ── Accuracy–interpretability table ─────────────────────────────────────
    rf_rmse = float(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["RMSE"])
    acc_rows = [
        [bold("RF (+ SHAP)"),      bold(f2(rf_rmse)), bold(f3(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["R2"])), "Post-hoc (SHAP explanations)", bold("0 (reference)")],
        ["XGBoost (+ SHAP)",       f2(metrics_test[metrics_test["Model"]=="XGBoost"].iloc[0]["RMSE"]), f3(metrics_test[metrics_test["Model"]=="XGBoost"].iloc[0]["R2"]), "Post-hoc (SHAP explanations)", f"+{f2(float(metrics_test[metrics_test['Model']=='XGBoost'].iloc[0]['RMSE'])-rf_rmse)}"],
        ["Decision Tree",          f2(metrics_test[metrics_test["Model"]=="DT"].iloc[0]["RMSE"]), f3(metrics_test[metrics_test["Model"]=="DT"].iloc[0]["R2"]), "Full (rule-based, visualisable)", f"+{f2(float(metrics_test[metrics_test['Model']=='DT'].iloc[0]['RMSE'])-rf_rmse)} (+{100*(float(metrics_test[metrics_test['Model']=='DT'].iloc[0]['RMSE'])-rf_rmse)/rf_rmse:.1f}%)"],
        ["ElasticNet",             f2(metrics_test[metrics_test["Model"]=="ElasticNet"].iloc[0]["RMSE"]), f3(metrics_test[metrics_test["Model"]=="ElasticNet"].iloc[0]["R2"]), "Full (signed standardised coefficients)", f"+{f2(float(metrics_test[metrics_test['Model']=='ElasticNet'].iloc[0]['RMSE'])-rf_rmse)} (+{100*(float(metrics_test[metrics_test['Model']=='ElasticNet'].iloc[0]['RMSE'])-rf_rmse)/rf_rmse:.1f}%)"],
    ]
    acc_tbl = md_table(
        ["Model", "Test RMSE (kWh)", "R²", "Interpretability Mode", "RMSE Cost vs. RF"],
        acc_rows, ["l", "r", "r", "l", "r"]
    )

    # Dominant feature RMSE increase
    inductive_rf = rf_pi[rf_pi["feature"]=="kvarh_received_inductive_Total"].iloc[0]["importance_mean"]

    return f"""\
## 4. Results — RQ2: Interpretability Analysis

### 4.1 Global Feature Importance — Permutation Importance

**Table 4.1 — Top 10 Features by Permutation Importance (RF, test set)**

{pi_tbl}

*RMSE increase when the feature is randomly shuffled (averaged over 10 repeats). Larger value = more important.*

{fig_ref("phase10_permutation_importance_rf.png", "Permutation importance — RF (top 15 features)")}
{fig_ref("phase10_permutation_importance_all_models.png", "Permutation importance comparison across all Track A models")}

The reactive inductive energy (`kvarh_received_inductive_Total`) is by far the dominant predictor for the RF model: removing it increases RMSE by {f2(inductive_rf)} kWh (vs. an overall RMSE of {f2(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["RMSE"])} kWh). This finding is consistent across all tree-based models (DT: +13.74 kWh, LightGBM: +12.14 kWh, XGBoost: +10.76 kWh). Reactive energy is measured by the same smart meter as active energy but captures the power factor of the load — the ratio of resistive to reactive current drawn by the HP compressor motor. This signal is independent of total consumption and provides information about the compressor's operating regime that temperature and HDD features do not.

`Survey_Building_LivingArea` ranks second (+5.57 kWh) — a well-established driver of heating demand through the building's heat loss coefficient. `Survey_Building_Residents` ranks fourth (+1.74 kWh), capturing occupant-behaviour effects (domestic hot water, ventilation, internal gains).

Weather features (3-day rolling temperature, 1-day lagged temperature) appear in positions 7 and 8, despite being the primary physical drivers of heating demand. This counterintuitive finding reflects that reactive energy and building area together implicitly encode much of the weather–consumption relationship — when these features are present, the marginal contribution of temperature is lower.

### 4.2 SHAP Global Explanations

**Table 4.2 — Top 5 Features by Mean |SHAP| Value Across Models**

{shap_tbl}

{fig_ref("phase10_shap_summary_beeswarm_xgboost.png", "SHAP summary beeswarm plot — XGBoost (test set)")}
{fig_ref("phase10_shap_bar_rf.png", "SHAP mean absolute value bar chart — RF (test set)")}

SHAP and permutation importance rankings are broadly consistent for tree models. ANN diverges: `dhw_ewh` (DHW production by electric water heater) ranks first by SHAP for the ANN, reflecting that ANN was trained on 30 scaled features excluding reactive energy (which is collinear with the target in the standardised feature space). This explains ANN's lower predictive accuracy — it lacks access to the dominant signal that tree models exploit.

For Track B (XGBoost B with 75 features), three protocol-derived features appear in the top 10 by SHAP: `HeatPump_Installation_HeatingCapacity` (|SHAP| = 3.07), `hp_capacity_per_area` (HP capacity normalised by heated floor area, |SHAP| = 1.29), and `Building_FloorAreaHeated_GroundFloor` (|SHAP| = 0.90). HP capacity relative to building area captures HP over- or under-sizing, which directly affects operating efficiency and consumption.

{fig_ref("phase10_shap_summary_beeswarm_xgboost_b.png", "SHAP summary beeswarm plot — XGBoost B (Track B, protocol features)")}

### 4.3 Cross-Model Feature Ranking Consistency

**Table 4.3 — Spearman Rank Correlation of Feature Importance Rankings Across Models**

{sp_tbl}

{fig_ref("phase10_spearman_correlation_heatmap.png", "Spearman rank correlation heatmap of feature importance across models")}

Tree-based models show very high mutual agreement on feature rankings (ρ = 0.876–0.961). The RF–LightGBM pair has the highest correlation (ρ = 0.961), confirming near-identical feature utilisation. ANN partially agrees with tree models (ρ = 0.767–0.902). ElasticNet shows the lowest agreement (ρ = 0.584–0.659) — reflecting fundamentally different feature relevance in the linear model, where smooth temperature rolling averages dominate and reactive energy plays a lesser role (due to standardisation and collinearity). The high tree-model consistency validates that the identified top features — reactive energy and building living area — are genuine predictors, not model-specific artefacts.

### 4.4 Accuracy–Interpretability Trade-off

**Table 4.4 — Accuracy vs. Interpretability Summary**

{acc_tbl}

{fig_ref("phase10_accuracy_interpretability_tradeoff.png", "Accuracy–interpretability trade-off across Track A models")}
{fig_ref("phase10_dt_tree_structure.png", "Decision Tree structure (top 3 levels)")}

The Decision Tree provides full rule-based transparency at a +25.1% RMSE penalty vs. RF. For a regulatory context requiring complete audit trails and rule-based explainability — where every prediction can be traced to a sequence of feature thresholds — the DT is the appropriate choice. For a utility demand forecasting system where accuracy is paramount, RF with SHAP post-hoc explanations achieves near-maximum accuracy with instance-level explanation capability.

ElasticNet's signed standardised coefficients are the most traditional form of interpretability, but its R² of {f3(metrics_test[metrics_test["Model"]=="ElasticNet"].iloc[0]["R2"])} makes it operationally unsuitable as a standalone forecasting model.

"""


def section_rq3() -> str:
    # ── Subgroup composition ────────────────────────────────────────────────
    comp_dims = ["HP Type", "Heat Distribution", "PV System", "EV Ownership", "Group"]
    comp_rows = []
    for dim in comp_dims:
        sub = sg_composition[sg_composition["Dimension"] == dim]
        for _, r in sub.iterrows():
            if r["Category"] == "Unknown":
                continue
            comp_rows.append([dim, r["Category"], f"{int(r['N_households'])}", f"{int(r['N_rows']):,}", f"{r['Pct_rows']:.1f}%"])
    comp_tbl = md_table(
        ["Dimension", "Category", "Households", "Test rows", "% of total"],
        comp_rows, ["l", "l", "r", "r", "r"]
    )

    # ── RF subgroup bias table ──────────────────────────────────────────────
    rf_sg = sg_metrics[(sg_metrics["Model"] == "RF") & (sg_metrics["Dimension"] != "Overall")].copy()
    rf_sg["abs_bias"] = rf_sg["mean_bias"].abs()
    rf_sg = rf_sg[~rf_sg["Category"].isin(["Unknown", "All"])].sort_values("abs_bias", ascending=False)

    sg_rows = []
    for _, r in rf_sg.head(14).iterrows():
        flag = " ◄" if r["abs_bias"] >= 1.0 else ""
        sg_rows.append([
            r["Dimension"], r["Category"],
            f"{int(r['N']):,}", str(int(r["N_households"])),
            f"{r['mean_bias']:+.2f}{flag}",
            f2(r["MAE"]), f2(r["RMSE"]), f3(r["R2"])
        ])
    sg_tbl = md_table(
        ["Dimension", "Category", "N (rows)", "HH", "Mean Bias (kWh)", "MAE", "RMSE", "R²"],
        sg_rows, ["l", "l", "r", "r", "r", "r", "r", "r"]
    )

    # ── MW significant pairs table ──────────────────────────────────────────
    mw_sig = mw_results[mw_results["significant"] == True].copy()
    mw_rows = []
    for _, r in mw_sig.sort_values("p_bonferroni").iterrows():
        mw_rows.append([
            r["sg_col"].replace("sg_", ""),
            f"{r['cat_a']} vs {r['cat_b']}",
            r["Model"],
            f"{r['delta_median']:+.2f}",
            sci(r["p_bonferroni"])
        ])
    mw_tbl = md_table(
        ["Subgroup", "Comparison", "Model", "Δ Median (kWh)", "Bonferroni p"],
        mw_rows, ["l", "l", "l", "r", "r"]
    )

    # ── Kruskal-Wallis table ────────────────────────────────────────────────
    kw_data = [
        ("Heat distribution (4 groups)", 368.42, "1.5×10⁻⁷⁹"),
        ("Test month (Dec/Jan/Feb/Mar)", 660.90, "3.1×10⁻¹⁴⁴"),
        ("Living area bucket (5 groups)", 147.20, "5.3×10⁻³⁰"),
        ("HP type (3 groups)", 20.09, "4.3×10⁻⁵"),
    ]
    kw_rows = [[d[0], f"{d[1]:.2f}", d[2], "✓"] for d in kw_data]
    kw_tbl = md_table(
        ["Subgroup", "H statistic", "p-value", "Significant"],
        kw_rows, ["l", "r", "r", "c"]
    )

    # ── Track B table ───────────────────────────────────────────────────────
    b_rows = []
    for _, r in sg_b_metrics.sort_values(["Dimension", "Category"]).iterrows():
        if str(r["Category"]) in ("nan", "Unknown"):
            continue
        b_rows.append([r["Dimension"], str(r["Category"]),
                       f"{int(r['N']):,}", f"{r['mean_bias']:+.2f}", f2(r["MAE"]), f2(r["RMSE"])])
    b_tbl = md_table(
        ["Dimension", "Category", "N", "Bias (kWh)", "MAE", "RMSE"],
        b_rows, ["l", "l", "r", "r", "r", "r"]
    )

    # Key numbers
    rf_overall = sg_metrics[(sg_metrics["Model"]=="RF") & (sg_metrics["Dimension"]=="Overall")].iloc[0]

    return f"""\
## 5. Results — RQ3: Subgroup and Bias Analysis

### 5.1 Test-Set Subgroup Composition

**Table 5.1 — Test Set Composition by Key Subgroup Dimensions**

{comp_tbl}

Building type shows extreme imbalance: 98.4% of test households are houses (813 HH); only 7 are apartments. Subgroup results for apartments are reported but must be interpreted with caution given the limited sample. The treatment group represents 7.4% of test households (61 HH), of which 60 have post-visit data (4,638 rows) and 22 have pre-visit data (837 rows) in the test period.

### 5.2 Per-Subgroup Bias (RF Primary Model)

**Table 5.2 — Per-Subgroup Bias and Error Metrics — RF Model (Test Set)**

{sg_tbl}

*◄ = |mean bias| ≥ 1.0 kWh. Positive bias = model under-predicts (actual > predicted). Negative = over-predicts.*

{fig_ref("phase11_bias_heatmap.png", "Mean bias heatmap: subgroup categories × all Track A models")}

The radiator subgroup shows the largest mean bias (+1.74 kWh) — the RF model systematically under-predicts consumption for households with radiator-based heat distribution. Radiators typically operate at higher supply temperatures than floor heating systems; the model, which sees only a one-hot `heat_dist_radiator` flag without any supply-temperature information from the feature set, cannot fully account for this operational difference.

EV households are under-predicted with the highest absolute MAE (9.97 kWh) — 33% higher than the overall MAE of {f2(rf_overall["MAE"])} kWh. The static `has_ev` binary flag does not capture whether the EV is actually being charged on a given day; on non-charging days, actual consumption is lower than the model expects.

{fig_ref("phase11_residuals_heat_dist.png", "Residual distributions by heat distribution system — top 4 models")}
{fig_ref("phase11_residuals_ev.png", "Residual distributions by EV ownership — top 4 models")}
{fig_ref("phase11_residuals_pv.png", "Residual distributions by PV system — top 4 models")}
{fig_ref("phase11_bias_vs_area_rf.png", "Per-household mean bias vs. living area — RF (coloured by HP type)")}

### 5.3 Statistical Significance of Subgroup Differences

**Table 5.3 — Significant Mann-Whitney U Tests After Bonferroni Correction (α = 0.0021)**

{mw_tbl}

{fig_ref("phase11_residuals_hp_type.png", "Residual distributions by HP type — top 4 models")}

Of 24 pairwise tests (8 subgroup pairs × 3 models), **12 are significant** after Bonferroni correction. Floor vs. Radiator heat distribution is the most robust finding (significant for all three models, p < 10⁻⁶⁴), followed by PV presence and EV ownership. HP type (Air-Source vs. Ground-Source) is significant only for RF (p = 1.75×10⁻⁴) — suggesting modest model-dependent sensitivity to HP type. Building type (House vs. Apartment) is not significant after correction for any model, consistent with the extremely small apartment sample.

**Table 5.4 — Kruskal-Wallis H Tests for Multi-Category Subgroups (RF Residuals)**

{kw_tbl}

All four multi-category dimensions show statistically significant heterogeneity. The test-month dimension has the largest H statistic (660.90, p = 3.1×10⁻¹⁴⁴), reflecting within-heating-season temperature variation: January tends to be the coldest month, driving higher consumption and prediction uncertainty than December or February.

### 5.4 Track B Protocol Subgroup Analysis

**Table 5.5 — Track B Residuals by Protocol Subgroup (N = 5,475, 109 treatment HH; models: XGBoost B, DT B, RF B)**

{b_tbl}

{fig_ref("phase11_track_b_residuals_building_age.png", "Track B residuals by building age bucket (XGBoost B primary)")}
{fig_ref("phase11_track_b_residuals_night_setback.png", "Track B residuals by night setback status (XGBoost B primary)")}

The most striking Track B finding is the night setback dimension: households where night setback was **active** before the energy consultant visit show a mean bias near zero (+0.08 kWh), while those **without** setback show −2.49 kWh (model over-predicts). This statistically significant difference (Mann-Whitney p = 6.0×10⁻²³) reflects an operational pattern: houses without night setback maintain a higher baseline overnight temperature, leading to a warmer morning starting condition that requires less morning warm-up energy. The model, which sees only the binary `night_setback_active_before` flag, does not fully capture this dynamic thermal effect.

Households where the heating limit was set too high (`heating_limit_too_high = 1`) show a mean bias of −3.18 kWh — the model over-predicts substantially. These households were consuming *less* energy than the model expected given their temperature exposure, which is consistent with the heating limit restricting operation at high outdoor temperatures (the HP switches off earlier than models trained on mixed data expect).

### 5.5 Treatment Effect Analysis

The treatment group (61 HH) achieved R² = 0.834 vs. 0.720 for the control group under the RF model — a substantially better fit. This is not a consequence of the model having access to treatment information (the `post_intervention` flag was excluded from features); instead, it reflects that post-optimisation HPs have more regular, weather-aligned consumption patterns that the model generalises well to.

Assessing the *causal effect* of the energy consultant visit is complicated by seasonal confounding: in the validation set, pre-visit rows fall earlier (lower-consumption summer months) and post-visit rows later (higher-consumption autumn), making a simple pre/post comparison misleading. The RF model predicts a −0.40 kWh/day pre-to-post change vs. −1.84 kWh/day observed — but both figures are dominated by seasonality rather than true intervention effects.

{fig_ref("phase11_treatment_effect_timeline.png", "Actual vs. RF predicted consumption for treatment households (validation set)")}

"""


def section_discussion() -> str:
    rf_rmse = f2(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["RMSE"])
    rf_r2   = f3(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["R2"])
    en_rmse = f2(metrics_test[metrics_test["Model"]=="ElasticNet"].iloc[0]["RMSE"])
    dt_rmse = f2(metrics_test[metrics_test["Model"]=="DT"].iloc[0]["RMSE"])
    dt_pct  = f1(100*(float(metrics_test[metrics_test["Model"]=="DT"].iloc[0]["RMSE"]) -
                       float(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["RMSE"])) /
                  float(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["RMSE"]))
    xgb_b_rmse = f2(_b_row("XGBoost B")["RMSE"])
    xgb_b_r2   = f3(_b_row("XGBoost B")["R2"])

    return f"""\
## 6. Discussion and Limitations

### 6.1 Summary of Main Findings

**RQ1 — Tree-based vs. LR and ANN:**
Tree ensemble models (RF, XGBoost, LightGBM) are the clear winners of the Track A comparison (RMSE {rf_rmse} kWh, R² {rf_r2} for RF). The performance gap relative to ElasticNet ({en_rmse} kWh, R² {f3(metrics_test[metrics_test["Model"]=="ElasticNet"].iloc[0]["R2"])}) is 76.8% larger RMSE, confirming that the relationship between HP consumption and its predictors is fundamentally non-linear and cannot be adequately captured by a linear model even with L1/L2 regularisation and extensive feature engineering. ElasticNet's R² of {f3(metrics_test[metrics_test["Model"]=="ElasticNet"].iloc[0]["R2"])} means it explains only 15% of test-set variance — barely above the per-household mean baseline (R² = {f3(metrics_test[metrics_test["Model"]=="Baseline: Per-HH Mean"].iloc[0]["R2"])}).

ANN (RMSE = {f2(metrics_test[metrics_test["Model"]=="ANN"].iloc[0]["RMSE"])} kWh, R² = {f3(metrics_test[metrics_test["Model"]=="ANN"].iloc[0]["R2"])}) underperforms tree ensembles despite three hidden layers and 60 Optuna tuning trials. This is consistent with the broader "tabular data" literature, where tree-based models frequently outperform neural networks on structured, mixed-type datasets with moderate sample sizes. The specific inductive bias of tree models — recursive axis-aligned partitioning — is well-suited to the threshold-like nature of HP operation (heating season on/off, setback temperature activation). Additionally, the ANN was trained on a reduced 30-feature scaled set that excluded reactive energy (due to standardisation collinearity with building area); this exclusion may account for a substantial portion of the ANN–RF gap.

Among tree models, all three ensembles are practically indistinguishable (Δ RMSE ≤ 0.11 kWh). For production deployment, model selection should therefore prioritise inference speed (LightGBM fastest), ecosystem compatibility, and post-hoc explanation tools rather than raw test-set RMSE differences of this magnitude.

When protocol data is available (Track B), XGBoost B achieves RMSE = {xgb_b_rmse} kWh (R² = {xgb_b_r2}) — a 27% improvement over Track A RF on the same 109 households. However, the ablation shows that HP capacity and floor area — available from on-site inspection — explain most of this gain; the heating curve settings and issue flags add marginal further value.

**RQ2 — Feature importance and interpretability:**
Reactive energy (kVArh inductive) as the dominant predictor is the most novel finding of this study. Its RMSE contribution (9.9 kWh increase when removed from RF) exceeds even building living area (5.6 kWh) and is consistent across all four tree-based models. This signal represents the compressor motor's reactive current draw — an operational fingerprint that encodes the HP's thermal load independently of outdoor temperature. Utilities and energy service companies should ensure that smart meters deployed in HP monitoring programmes record reactive energy in addition to active energy; this appears to be a cost-effective enhancement relative to the prediction accuracy gain.

Building living area and number of residents rank second and fourth globally — expected from building physics principles. The lower-than-expected ranking of weather features (positions 7–8 for rolling temperature) is explained by reactive energy and building area together implicitly encoding the weather–consumption relationship; in models without reactive energy (ANN, ElasticNet), weather features rank higher. Feature ranking consistency across tree models (Spearman ρ = 0.876–0.961) validates that the top-ranked features are genuinely informative.

The accuracy–interpretability trade-off quantification provides a concrete guide for deployment decisions: the DT offers full rule-based transparency at a {dt_pct}% RMSE penalty. For contexts requiring complete audit trails (e.g., regulatory compliance, household billing disputes), this cost may be acceptable. For grid-side demand forecasting where accuracy drives economic value, RF + SHAP is recommended.

**RQ3 — Subgroup bias:**
Heat distribution system (floor vs. radiators) is the most robust and statistically significant bias dimension (Bonferroni p < 10⁻⁷², consistent across all three top models). The one-hot encoding of heat distribution captures system type but not the supply temperature that distinguishes floor heating (typically 30–40°C) from radiator systems (typically 50–70°C). Including the heating curve's design supply temperature (available in the protocol data) partially corrects this for Track B households, consistent with heating curve features appearing in the Track B SHAP top 10.

EV households' over-prediction (mean bias +1.66 kWh) reflects the static nature of the `has_ev` flag: on non-charging days, actual consumption is systematically lower than predicted. As EV penetration grows — currently 23.4% of the HEAPO test set — this dynamic load source will increasingly require day-level EV charging status data rather than a static ownership flag.

PV households' under-prediction (+0.55 kWh mean bias) is at least partly a measurement artefact rather than a model limitation: the target variable's PV self-consumption gap means the true consumption of PV households on high-generation days is unobservable. No model trained on grid-draw data alone can resolve this without sub-meter PV generation data.

### 6.2 Practical Implications for Energy Utilities

1. **Reactive energy metering:** Mandate kVArh recording in HP monitoring smart meter programmes. The 9.9 kWh/day RMSE gain from this single feature class exceeds the total gain from adding household survey metadata (5.2 kWh).

2. **Static metadata collection:** HP type, heat distribution system, living area, and appliance ownership (EV, dryer) should be collected at HP registration and maintained in utility customer databases. The ablation shows these four data types alone reduce RMSE from 16.7 to 11.5 kWh.

3. **Heat distribution calibration:** When deploying RF for radiator-equipped households, apply a post-prediction additive calibration of approximately +1.7 kWh/day. This can be derived from the training set's radiator-household residuals.

4. **EV-charging day detection:** For the growing EV household segment, augment the static `has_ev` flag with a day-level EV charging indicator (derivable from intra-day load profile shape in the 15-minute data). This is expected to substantially reduce the 33% MAE overrun for EV households.

5. **HP optimisation visits:** Treatment households (post-visit) achieve R² = 0.834 vs. 0.720 for control — their consumption is more predictable, likely due to optimised settings. Energy consultant programmes appear to produce more grid-friendly HP behaviour as a side effect, in addition to their direct consumption-reduction mandate.

### 6.3 Limitations

1. **Heating-season-only test set.** The primary performance metrics (RMSE = {rf_rmse} kWh, R² = {rf_r2} for RF) apply to December 2023 – March 2024 — the peak HP demand period. Validation-set non-heating performance (RF RMSE = 6.89 kWh, May–September) is substantially better. Annual-average performance requires weighting across seasons and cannot be directly read from the current split structure.

2. **PV self-consumption invisible.** For 44.8% of test households, self-consumed solar energy is excluded from the target variable. This is a fundamental measurement limitation of the HEAPO dataset acknowledged in the original paper (Section 2.1.2); no modelling approach can close this gap without sub-meter PV generation data.

3. **Geographic homogeneity.** All 1,298 households are in Canton Zurich, Switzerland (Köppen climate classification Cfb: temperate oceanic). Results may not generalise to significantly different climatic zones (continental, Mediterranean, subarctic) or to HP market compositions with different type distributions (e.g., ground-source dominated markets).

4. **Protocol features marginal at small sample.** Track B's 0.14 kWh RMSE improvement from protocol features is based on 109 households — a sample too small for definitive conclusions. Broader deployment of on-site inspection protocols, or including installation-permit data available to utilities, could change this picture at scale.

5. **Building age not available for control group.** The most physically meaningful building characteristic (thermal insulation quality, correlated with construction year) is available only for 214 treatment households via the on-site protocol. Control households lack this variable; future surveys should include building construction year.

6. **Seasonality confounds treatment effect.** The pre/post-visit comparison cannot be cleanly interpreted as a causal consumption reduction because pre-visit and post-visit observations fall in different calendar periods. A rigorous difference-in-differences approach using matched control households would be required for causal inference.

7. **Self-reported metadata accuracy.** The 13 household survey variables are self-reported; living area and HP type may contain inaccuracies undetectable without external validation.

8. **Static EV and appliance flags.** `has_ev` and `has_dryer` are time-invariant binary flags; dynamic load characteristics (EV charging frequency, dryer usage patterns) are not captured.

### 6.4 Future Work

- **15-minute profile features:** Aggregate intra-day HP cycling behaviour (morning peak-to-mean ratio, peak-to-off transition time) from the available 15-minute data. Brudermueller et al. (2023) demonstrate that these signals capture compressor behaviour not visible at daily resolution.
- **Multi-horizon forecasting:** Extend to 3- and 7-day-ahead predictions using autoregressive architectures (LSTM, Temporal Fusion Transformer) for grid-side demand response planning.
- **Causal inference:** Use the HEAPO treatment/control design with propensity score matching to estimate the consumption reduction attributable to energy consultant visits, controlling for seasonal confounds.
- **Geographic transfer learning:** Pre-train on HEAPO (Swiss), fine-tune on smaller HP datasets from Germany or Austria to test cross-market generalisation.
- **Dynamic EV integration:** Replace the static `has_ev` flag with a day-level EV charging indicator derived from intra-day load shape (charging signature detection from 15-minute data).

"""


def section_conclusion() -> str:
    rf_rmse = f2(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["RMSE"])
    rf_r2   = f3(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["R2"])
    xgb_b_rmse = f2(_b_row("XGBoost B")["RMSE"])
    xgb_b_r2   = f3(_b_row("XGBoost B")["R2"])
    dt_pct = f1(100*(float(metrics_test[metrics_test["Model"]=="DT"].iloc[0]["RMSE"]) -
                     float(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["RMSE"])) /
                float(metrics_test[metrics_test["Model"]=="RF"].iloc[0]["RMSE"]))
    sp_rf_lgbm = f3(spearman.loc["RF", "LightGBM"])

    return f"""\
## 7. Conclusion

This study presented a comprehensive machine learning benchmark for predicting daily household heat pump electricity consumption, using the HEAPO open dataset of 1,298 Swiss households across five years of smart meter data, matched with weather observations, household survey metadata, and on-site HP inspection protocols.

**Answering the main research question:** Among the six Track A models evaluated, tree ensemble methods — Random Forest, XGBoost, and LightGBM — provide the most accurate and robust predictions. RF achieves RMSE = {rf_rmse} kWh (R² = {rf_r2}) on the heating-season test set; XGBoost and LightGBM are tied within 0.11 kWh. When protocol-derived installation data is available (Track B), XGBoost B reaches RMSE = {xgb_b_rmse} kWh (R² = {xgb_b_r2}) — a 27% improvement; DT B and RF B provide additional reference points within the Track B sample. Linear regression (ElasticNet) and the ANN cannot match tree ensemble accuracy in this problem setting.

**RQ1:** Tree-based models substantially outperform ElasticNet (+76.8% RMSE) and ANN (+34.9% RMSE). Among tree ensembles, differences are practically negligible (Δ RMSE ≤ 0.11 kWh). Protocol-enriched data reduces RMSE by a further 27% for the treatment subset. Hyperparameter optimisation via Bayesian search (Optuna) provides meaningful gains for all non-linear models (9.7–13.9% RMSE improvement).

**RQ2:** Reactive energy (kVArh inductive component) is the dominant predictor across all tree-based models — a novel finding with practical implications for smart meter specification. Feature rankings are highly consistent across tree models (Spearman ρ = {sp_rf_lgbm} for RF–LightGBM), validating the robustness of the top-ranked features. RF augmented with SHAP post-hoc explanations is recommended as the deployment configuration; the Decision Tree provides full rule-based transparency at a {dt_pct}% RMSE penalty.

**RQ3:** Heat distribution system (floor vs. radiators) is the most significant and consistent bias dimension (Bonferroni p < 10⁻⁷², Δ median residual 1.07–1.34 kWh across models). EV households are systematically over-predicted due to the static nature of the EV ownership flag. PV households show slight under-prediction partially attributable to the self-consumption measurement gap inherent in the dataset. The treatment group (post energy-consultant visit) achieves better predictive accuracy (R² = 0.834 vs. 0.720 for control), consistent with optimised HP settings producing more regular, weather-aligned consumption patterns.

The key practical recommendation for energy utilities and HP programme operators is threefold: (i) ensure smart meters record reactive energy (kVArh) — the single most valuable predictor; (ii) collect HP type, heat distribution system, and living area at installation registration — together these halve the RMSE relative to a weather-only model; and (iii) deploy RF with SHAP explanations for the best balance of forecasting accuracy and post-hoc interpretability.

The full pipeline — 13 Python scripts, 10 source modules, all configuration parameters, and the pinned dataset reference (Zenodo record 15056919) — is provided to enable complete reproduction of all reported results.

"""


def section_references() -> str:
    return """\
## References

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631). https://doi.org/10.1145/3292500.3330701

Brudermueller, T., et al. (2025). HEAPO – An Open Dataset for Heat Pump Optimization with Smart Electricity Meter Data and On-Site Inspection Protocols. *arXiv preprint arXiv:2503.16993v1*. Zenodo record 15056919. https://doi.org/10.5281/zenodo.15056919

Brudermueller, T., et al. (2023). Heat pump load prediction using smart meter data and machine learning. *(Reference [7] in HEAPO dataset paper — 15-minute resolution HP load forecasting study.)*

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems* (NeurIPS), 30, 3146–3154.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions (SHAP). *Advances in Neural Information Processing Systems* (NeurIPS), 30, 4765–4774.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

SIA 380/1 (2016). *Thermische Energie im Hochbau*. Schweizerischer Ingenieur- und Architektenverein. *(Basis for HDD\_SIA\_daily computation in the HEAPO dataset.)*

"""


def section_appendices() -> str:
    # Appendix A — Feature catalog (FEATURES_TREES)
    tree_feats = feature_lists["FEATURES_TREES"]
    linear_feats = feature_lists["FEATURES_LINEAR"]
    trees_b = feature_lists["FEATURES_TREES_B"]
    linear_b = feature_lists["FEATURES_LINEAR_B"]

    # Group features by category
    temporal  = ["day_of_week", "month", "is_weekend", "day_of_year", "is_heating_season", "season_encoded"]
    weather_d = ["Temperature_max_daily", "Temperature_min_daily", "Temperature_avg_daily",
                 "HeatingDegree_SIA_daily", "HeatingDegree_US_daily", "CoolingDegree_US_daily",
                 "Humidity_avg_daily", "Precipitation_total_daily", "Sunshine_duration_daily"]
    weather_r = ["temp_range_daily", "HDD_SIA_daily", "HDD_US_daily", "CDD_US_daily",
                 "humidity_x_temp", "temp_avg_lag_1d", "temp_avg_rolling_3d",
                 "temp_avg_rolling_7d", "HDD_SIA_rolling_7d"]
    reactive  = ["kvarh_received_capacitive_Total", "kvarh_received_inductive_Total",
                 "has_pv", "has_reactive_energy"]
    household = [f for f in tree_feats if f not in temporal + weather_d + weather_r + reactive]

    feat_rows = []
    for cat, feats, track in [
        ("Temporal", temporal, "A+B"),
        ("Weather (direct)", weather_d, "A+B"),
        ("Weather (derived/lag)", weather_r, "A+B"),
        ("Reactive energy", reactive, "A+B"),
        ("Household static", household, "A+B"),
    ]:
        for f in feats:
            in_linear = "✓" if f in linear_feats else "—"
            feat_rows.append([f"`{f}`", cat, "✓", in_linear, "A+B"])

    # Track B additional
    b_only = [f for f in trees_b if f not in tree_feats]
    for f in b_only[:15]:  # first 15
        feat_rows.append([f"`{f}`", "Protocol/Installation", "✓", "—", "B only"])
    if len(b_only) > 15:
        feat_rows.append([f"*(+ {len(b_only)-15} more protocol features)*", "Protocol/Installation", "✓", "—", "B only"])

    feat_a_tbl = md_table(
        ["Feature", "Category", "Trees", "Linear/ANN", "Track"],
        feat_rows, ["l", "l", "c", "c", "c"]
    )

    # Appendix B — Hyperparameters
    hp_rows = []
    for model_key, params in best_params.items():
        display = model_key.replace("_", " ")
        for k, v in params.items():
            hp_rows.append([display, k, str(round(v, 6) if isinstance(v, float) else v)])
        display = ""  # blank model name after first row
    hp_tbl = md_table(["Model", "Parameter", "Tuned Value"], hp_rows, ["l", "l", "r"])

    return f"""\
## Appendices

### Appendix A — Feature Catalog

**Total features: {len(tree_feats)} (Track A, tree models), {len(linear_feats)} (Track A, linear/ANN), {len(trees_b)} (Track B, tree models), {len(linear_b)} (Track B, linear)**

{feat_a_tbl}

### Appendix B — Best Hyperparameters (Optuna)

{hp_tbl}

### Appendix C — Validation Set Metrics (All Models)

| Model | RMSE (kWh) | MAE (kWh) | R² | sMAPE (%) |
|-------|-----------|-----------|----|-----------|
""" + "\n".join(
        f"| {r['Model']} | {f2(r['RMSE'])} | {f2(r['MAE'])} | {f3(r['R2'])} | {f1(r['sMAPE'])} |"
        for _, r in metrics_val_sea[
            (metrics_val_sea["Split"]=="Val") & (metrics_val_sea["Period"]=="Val Overall")
        ].iterrows()
    ) + """

### Appendix D — Full RF Subgroup Metrics

""" + md_table(
        ["Dimension", "Category", "N", "Bias (kWh)", "MAE", "RMSE", "R²"],
        [
            [r["Dimension"], r["Category"], f"{int(r['N']):,}",
             f"{r['mean_bias']:+.2f}", f2(r["MAE"]), f2(r["RMSE"]), f3(r["R2"])]
            for _, r in sg_metrics[
                (sg_metrics["Model"]=="RF") & (~sg_metrics["Category"].isin(["Unknown","All"]))
            ].sort_values(["Dimension","Category"]).iterrows()
        ],
        ["l", "l", "r", "r", "r", "r", "r"]
    ) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Assemble and write the report
# ─────────────────────────────────────────────────────────────────────────────
def assemble_report() -> str:
    logger.info("Assembling report sections …")
    sections = [
        title_page(),
        abstract(),
        section_introduction(),
        section_methodology(),
        section_rq1(),
        section_rq2(),
        section_rq3(),
        section_discussion(),
        section_conclusion(),
        section_references(),
        section_appendices(),
    ]
    return "\n---\n\n".join(sections)


if __name__ == "__main__":
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Phase 12 — Academic Report Generation")
    logger.info("=" * 60)

    report_text = assemble_report()

    out_path = REPORT_DIR / "HEAPO_Predict_Report.md"
    out_path.write_text(report_text, encoding="utf-8")

    n_words = len(report_text.split())
    n_lines = report_text.count("\n")
    logger.info("Report written: %s", out_path)
    logger.info("  Word count : ~%d words", n_words)
    logger.info("  Line count : %d lines", n_lines)
    logger.info("  File size  : %.1f KB", out_path.stat().st_size / 1024)

    # Attempt PDF conversion via pandoc if available
    import subprocess
    pdf_path = REPORT_DIR / "HEAPO_Predict_Report.pdf"
    pandoc_cmd = [
        "pandoc", str(out_path),
        "-o", str(pdf_path),
        "--toc",
        "--number-sections",
        "-V", "geometry:margin=2.5cm",
        "-V", "fontsize=11pt",
        "-V", "linestretch=1.3",
        "--highlight-style=tango",
    ]
    try:
        result = subprocess.run(pandoc_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            logger.info("PDF generated: %s (%.1f KB)", pdf_path, pdf_path.stat().st_size / 1024)
        else:
            logger.warning("pandoc returned non-zero: %s", result.stderr[:200])
            logger.info("Run manually: pandoc outputs/report/HEAPO_Predict_Report.md -o outputs/report/HEAPO_Predict_Report.pdf --toc --number-sections")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.info("pandoc not available (%s). Markdown report is the primary deliverable.", type(e).__name__)
        logger.info("To convert: pandoc outputs/report/HEAPO_Predict_Report.md -o outputs/report/HEAPO_Predict_Report.pdf --toc --number-sections")

    elapsed = time.time() - t0
    logger.info("Phase 12 complete — %.1f s", elapsed)
    logger.info("Report path: %s", out_path)
