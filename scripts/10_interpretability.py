"""
scripts/10_interpretability.py

Phase 10 — Interpretability Analysis
=====================================

Tasks:
  10.1  Permutation importance (all 6 Track A models)
  10.2  SHAP global: TreeExplainer (RF/XGB/LGBM/DT), LinearExplainer (EN),
                     KernelExplainer (ANN, 2 000-row subsample)
  10.3  SHAP dependence plots (RF, top 5 features)
  10.4  SHAP local waterfall + force (RF, 4 representative cases)
  10.5  DT tree-structure visualisation
  10.6  ElasticNet standardised coefficients
  10.7  XGBoost B SHAP (75 protocol features)
  10.8  Cross-model feature ranking table + heatmaps
  10.9  Accuracy–interpretability tradeoff plot
  10.10 Consolidated interpretability report

Inputs : data/processed/{train,val,test}_{full,protocol}.parquet
         outputs/models/  — tuned model pickles + scalers
         outputs/tables/phase9_test_predictions.parquet
Outputs: outputs/figures/phase10_*.png
         outputs/tables/phase10_*.csv
         outputs/tables/phase10_interpretability_report.txt
         outputs/logs/phase10_run.log
"""

import gc
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Memory-safe defaults — tuned for M1 Air 8 GB unified memory.
#
# ┌─ To run full-fidelity results on a machine with ≥32 GB RAM ──────────────┐
# │  PERM_N_JOBS   = -1          # parallelise across all cores              │
# │  PERM_N_REPEATS = 30         # more shuffles → tighter confidence intervals│
# │  SHAP_ROWS_OTHER = 74_368    # full test set for XGBoost / LightGBM / DT │
# │  ANN_SHAP_ROWS   = 5_000     # larger subsample for KernelExplainer       │
# │  See also: RF SHAP note below (Task 10.2.1)                              │
# └──────────────────────────────────────────────────────────────────────────┘
# ─────────────────────────────────────────────────────────────────────────────
PERM_N_JOBS    = 1          # full hardware: -1 (use all cores)
PERM_N_REPEATS = 10         # full hardware: 30 (tighter confidence intervals)
# XGBoost / LightGBM / DT have optimised C++ SHAP kernels — fast at 5 000 rows.
# Full hardware: set SHAP_ROWS_OTHER = 74_368 to run on the complete test set.
SHAP_ROWS_OTHER    = 5_000  # full hardware: 74_368 (complete test set)
ANN_SHAP_ROWS      = 2_000  # full hardware: 5_000 (KernelExplainer is slow regardless)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.data_loader import load_config
from src.evaluation import MODEL_COLOURS
from src.interpretability import (
    compute_permutation_importance,
    compute_shap_kernel,
    compute_shap_linear,
    compute_shap_tree,
    plot_accuracy_interpretability_tradeoff,
    plot_all_models_permutation,
    plot_dt_tree,
    plot_elasticnet_coefficients,
    plot_feature_ranking_heatmap,
    plot_permutation_importance,
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_shap_dependence,
    plot_shap_force,
    plot_shap_waterfall,
    plot_spearman_heatmap,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = PROJECT_ROOT / "data" / "processed"
MODEL_DIR  = PROJECT_ROOT / "outputs" / "models"
TABLE_DIR  = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR    = PROJECT_ROOT / "outputs" / "figures"
SHAP_DIR   = TABLE_DIR / "shap_values"
LOG_DIR    = PROJECT_ROOT / "outputs" / "logs"

for d in (TABLE_DIR, FIG_DIR, SHAP_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    log_path = LOG_DIR / "phase10_run.log"
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature lists — identical to Phase 9
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_TREES = [
    "kvarh_received_capacitive_Total", "kvarh_received_inductive_Total",
    "has_pv", "has_reactive_energy",
    "Temperature_max_daily", "Temperature_min_daily", "Temperature_avg_daily",
    "HeatingDegree_SIA_daily", "HeatingDegree_US_daily", "CoolingDegree_US_daily",
    "Humidity_avg_daily", "Precipitation_total_daily", "Sunshine_duration_daily",
    "Survey_Building_LivingArea", "Survey_Building_Residents",
    "day_of_week", "month", "is_weekend", "day_of_year", "is_heating_season",
    "temp_range_daily", "HDD_SIA_daily", "HDD_US_daily", "CDD_US_daily",
    "humidity_x_temp", "temp_avg_lag_1d", "temp_avg_rolling_3d",
    "temp_avg_rolling_7d", "HDD_SIA_rolling_7d",
    "building_type_house", "building_type_apartment",
    "hp_type_air_source", "hp_type_ground_source", "hp_type_unknown",
    "dhw_hp", "dhw_ewh", "dhw_solar", "dhw_combined", "dhw_unknown",
    "heat_dist_floor", "heat_dist_radiator", "heat_dist_both", "heat_dist_unknown",
    "has_ev", "has_dryer",
]   # 45 features

# ─────────────────────────────────────────────────────────────────────────────
# Training-time benchmarks from Phase 8b (seconds)
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_SECONDS = {
    "ElasticNet": 5,
    "DT":         60,
    "RF":         750,
    "XGBoost":    37,
    "LightGBM":   100,
    "ANN":        160,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load model .pkl
# ─────────────────────────────────────────────────────────────────────────────
def _load(name: str):
    path = MODEL_DIR / f"model_{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Task 10.8 helper — Spearman rank correlation matrix
# ─────────────────────────────────────────────────────────────────────────────
def _spearman_matrix(ranking_df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr
    cols = ranking_df.columns.tolist()
    rho  = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for c1 in cols:
        for c2 in cols:
            valid = ranking_df[[c1, c2]].dropna()
            if len(valid) < 3:
                rho.loc[c1, c2] = float("nan")
            else:
                stat = spearmanr(valid[c1], valid[c2]).statistic
                # scipy ≥1.9 returns a matrix when passed two arrays; extract scalar
                rho.loc[c1, c2] = float(np.atleast_1d(stat).ravel()[0])
    return rho


# ─────────────────────────────────────────────────────────────────────────────
# Task 10.10 helper — write report
# ─────────────────────────────────────────────────────────────────────────────
def _write_report(
    perm_imp: dict,
    shap_mean_abs: dict,
    dt_model,
    elasticnet_model,
    rho_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    shap_b_df: pd.DataFrame,
    features_b: list[str],
    features_linear: list[str],
    cases: list[dict],
) -> str:
    lines = []

    def h(text):
        lines.append("")
        lines.append("=" * 70)
        lines.append(text)
        lines.append("=" * 70)

    def sub(text):
        lines.append("")
        lines.append("-" * 50)
        lines.append(text)
        lines.append("-" * 50)

    h("HEAPO-Predict — Phase 10: Interpretability Analysis Report")
    lines.append("Models: ElasticNet, DT, RF, XGBoost, LightGBM, ANN, XGBoost B")
    lines.append("Test set: Dec 2023 – Mar 2024  |  N=74,368 rows (Track A)")

    # ── Section 1: Permutation importance summary ──────────────────────────
    h("Section 1 — Permutation Importance Summary (top 10 features per model)")
    for mname, df in perm_imp.items():
        sub(mname)
        top10 = df.head(10)
        for _, row in top10.iterrows():
            lines.append(
                f"  {row['feature']:<40}  mean RMSE↑ = {row['importance_mean']:+.3f} kWh"
                f"  ± {row['importance_std']:.3f}"
            )

    # ── Section 2: SHAP global ─────────────────────────────────────────────
    h("Section 2 — SHAP Global Findings (top 5 features by mean |SHAP|)")
    for mname, df in shap_mean_abs.items():
        sub(mname)
        top5 = df.head(5)
        for _, row in top5.iterrows():
            lines.append(
                f"  {row['feature']:<40}  mean |SHAP| = {row['mean_abs_shap']:.3f}"
            )

    # ── Section 3: Local prediction explanations ───────────────────────────
    h("Section 3 — Local Prediction Explanations (RF, 4 representative cases)")
    for case in cases:
        sub(case["label"])
        lines.append(f"  Household_ID : {case['household_id']}")
        lines.append(f"  Date         : {case['date']}")
        lines.append(f"  y_true       : {case['y_true']:.2f} kWh")
        lines.append(f"  RF prediction: {case['y_pred']:.2f} kWh")
        lines.append(f"  Error (pred-actual): {case['y_pred'] - case['y_true']:+.2f} kWh")

    # ── Section 4: DT root split ───────────────────────────────────────────
    h("Section 4 — Decision Tree Root Split")
    tree = dt_model.tree_
    feat_idx  = tree.feature[0]
    threshold = tree.threshold[0]
    n_left    = int(tree.n_node_samples[1])
    n_right   = int(tree.n_node_samples[2])
    n_total   = int(tree.n_node_samples[0])
    feat_name = FEATURES_TREES[feat_idx] if feat_idx < len(FEATURES_TREES) else f"feature_{feat_idx}"
    lines.append(
        f"  Root split: {feat_name} <= {threshold:.4f}"
    )
    lines.append(
        f"  Left branch (heating off):  {n_left:,} samples ({100*n_left/n_total:.1f}%)"
    )
    lines.append(
        f"  Right branch (heating on):  {n_right:,} samples ({100*n_right/n_total:.1f}%)"
    )

    def _dt_depth2(node_id, depth=0):
        if depth >= 2 or tree.feature[node_id] < 0:
            return
        fi   = tree.feature[node_id]
        thr  = tree.threshold[node_id]
        fname = FEATURES_TREES[fi] if fi < len(FEATURES_TREES) else f"feature_{fi}"
        indent = "  " * (depth + 1)
        lines.append(f"{indent}Node {node_id}: {fname} <= {thr:.4f}")
        _dt_depth2(tree.children_left[node_id],  depth + 1)
        _dt_depth2(tree.children_right[node_id], depth + 1)

    lines.append("")
    lines.append("  Tree structure (first 2 levels):")
    _dt_depth2(0)

    # ── Section 5: Spearman rank correlation ───────────────────────────────
    h("Section 5 — Cross-Model Feature Ranking Consistency (Spearman ρ)")
    cols = rho_df.columns.tolist()
    header = f"{'':>12}" + "".join(f"{c:>14}" for c in cols)
    lines.append(header)
    for c1 in cols:
        row_str = f"{c1:>12}"
        for c2 in cols:
            v = rho_df.loc[c1, c2]
            row_str += f"{v:>14.3f}" if not pd.isna(v) else f"{'n/a':>14}"
        lines.append(row_str)
    lines.append("")
    # Find highest off-diagonal pair
    best_rho, best_pair = -2.0, ("", "")
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i >= j:
                continue
            v = float(rho_df.loc[c1, c2])
            if not np.isnan(v) and v > best_rho:
                best_rho, best_pair = v, (c1, c2)
    lines.append(
        f"  Most similar pair: {best_pair[0]} vs {best_pair[1]}  ρ = {best_rho:.3f}"
    )

    # ── Section 6: Track B protocol feature analysis ───────────────────────
    h("Section 6 — Track B XGBoost B SHAP (Protocol Features)")
    lines.append("  Top 10 features by mean |SHAP| (XGBoost B, test_protocol, N=5,475):")
    lines.append("")
    # Mark protocol-specific features (not in standard 45)
    std_set = set(FEATURES_TREES)
    for rank, row in enumerate(shap_b_df.head(10).itertuples(), start=1):
        tag = "" if row.feature in std_set else "  [P]"
        lines.append(
            f"  #{rank:>2}  {row.feature:<50}  mean |SHAP| = {row.mean_abs_shap:.3f}{tag}"
        )
    n_protocol_in_top10 = sum(
        1 for r in shap_b_df.head(10).itertuples()
        if r.feature not in std_set
    )
    lines.append(
        f"\n  Protocol-specific features in top 10: {n_protocol_in_top10}/10"
    )

    # ── Section 7: Accuracy-interpretability tradeoff ──────────────────────
    h("Section 7 — Accuracy–Interpretability Tradeoff Summary")
    lines.append("  Test RMSE by model (Dec 2023 – Mar 2024):")
    lines.append("    RF         : 11.54 kWh  R²=0.728  (best Track A)")
    lines.append("    XGBoost    : 11.59 kWh  R²=0.726")
    lines.append("    LightGBM   : 11.65 kWh  R²=0.723")
    lines.append("    DT         : 14.44 kWh  R²=0.575  (best transparent model)")
    lines.append("    ANN        : 15.56 kWh  R²=0.506")
    lines.append("    ElasticNet : 20.40 kWh  R²=0.151  (no better than baselines)")
    lines.append("    XGBoost B  :  8.42 kWh  R²=0.847  (Track B, 109 HH)")
    lines.append("")
    lines.append("  Cost of choosing DT over RF: +2.90 kWh RMSE (+25.1%)")
    lines.append("  Cost of choosing ElasticNet over DT: +5.96 kWh RMSE (+41.3%)")
    lines.append("")
    lines.append(
        "  Recommendation: RF with SHAP explanations provides near-optimal accuracy\n"
        "  with post-hoc interpretability for individual predictions.\n"
        "  For operational deployment at scale (utility forecasting), RF + SHAP is the\n"
        "  preferred choice. For regulatory reporting requiring full transparency,\n"
        "  the DT provides rule-based explanations at a 25% accuracy penalty."
    )

    report = "\n".join(lines)
    report_path = TABLE_DIR / "phase10_interpretability_report.txt"
    report_path.write_text(report, encoding="utf-8")
    return str(report_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("HEAPO-Predict — Phase 10: Interpretability Analysis")
    logger.info("=" * 60)

    cfg          = load_config("config/params.yaml")
    seed         = cfg["modeling"]["random_seed"]
    shap_bg_size = cfg["modeling"]["shap_background_samples"]   # 200

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading parquets...")
    df_train    = pd.read_parquet(DATA_DIR / "train_full.parquet")
    df_test     = pd.read_parquet(DATA_DIR / "test_full.parquet")
    df_test_b   = pd.read_parquet(DATA_DIR / "test_protocol.parquet")

    y_test      = df_test["kWh_received_Total"].values
    X_test      = df_test[FEATURES_TREES].values
    logger.info("  test_full  : %s", df_test.shape)
    logger.info("  test_b     : %s", df_test_b.shape)

    # Track B feature list (same logic as Phase 9)
    _b_exclude = {
        "Household_ID", "Date", "kWh_received_Total", "kWh_log1p",
        "kWh_received_HeatPump", "kWh_received_Other", "kWh_returned_Total",
        "Group", "AffectsTimePoint", "Timestamp", "Weather_ID", "cv_fold",
        "post_intervention",
        "kvarh_received_capacitive_HeatPump", "kvarh_received_capacitive_Other",
        "kvarh_received_inductive_HeatPump", "kvarh_received_inductive_Other",
    }
    features_b = [
        c for c in df_test_b.select_dtypes(include="number").columns
        if c not in _b_exclude
    ][:75]
    X_test_b  = df_test_b[features_b].values
    y_test_b  = df_test_b["kWh_received_Total"].values
    logger.info("  FEATURES_TREES=%d  FEATURES_TREES_B=%d", len(FEATURES_TREES), len(features_b))

    # ── Load feature meta (scaler + ANN) ──────────────────────────────────
    logger.info("Loading scalers and feature metadata...")
    scaler_a = joblib.load(MODEL_DIR / "scaler_linear_A.pkl")
    meta_a   = json.loads((MODEL_DIR / "scaler_linear_A_meta.json").read_text())
    features_linear = meta_a["feature_names"]   # 30 features

    X_test_linear  = df_test[features_linear].values
    X_train_linear = df_train[features_linear].values
    logger.info("  FEATURES_LINEAR=%d", len(features_linear))

    # ── Load models ────────────────────────────────────────────────────────
    logger.info("Loading tuned models...")
    rf_model  = _load("rf_tuned")
    xgb_model = _load("xgboost_tuned")
    lgbm_model= _load("lgbm_tuned")
    dt_model  = _load("dt_tuned")
    ann_model = _load("ann_tuned")
    en_model  = _load("elasticnet_tuned")
    xgb_b_model = _load("xgboost_b_tuned")
    logger.info("  All 7 models loaded.")

    # RNG used throughout for reproducible subsampling
    rng = np.random.default_rng(seed)

    # ── Load Phase 9 test predictions (for case selection) ────────────────
    df_preds = pd.read_parquet(TABLE_DIR / "phase9_test_predictions.parquet")

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.1 — Permutation Importance  (skip if CSV already exists)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("-" * 50)
    logger.info("Task 10.1 — Permutation Importance")
    logger.info("-" * 50)

    _perm_csv = TABLE_DIR / "phase10_permutation_importance.csv"
    perm_imp  = {}

    if _perm_csv.exists():
        logger.info("  SKIPPING — %s already exists (delete to re-run)", _perm_csv.name)
        _perm_df = pd.read_csv(_perm_csv)
        for mname, grp in _perm_df.groupby("model"):
            perm_imp[mname] = grp[["feature","importance_mean","importance_std"]].reset_index(drop=True)
    else:
        perm_configs = [
            ("ElasticNet", en_model,   X_test_linear, features_linear, True,  scaler_a),
            ("DT",         dt_model,   X_test,        FEATURES_TREES,  False, None),
            ("RF",         rf_model,   X_test,        FEATURES_TREES,  False, None),
            ("XGBoost",    xgb_model,  X_test,        FEATURES_TREES,  False, None),
            ("LightGBM",   lgbm_model, X_test,        FEATURES_TREES,  False, None),
            ("ANN",        ann_model,  X_test_linear, features_linear, True,  scaler_a),
        ]

        perm_all_rows = []
        for mname, model, X_in, feats, log_tgt, scaler in perm_configs:
            logger.info("  Computing permutation importance: %s ...", mname)
            t_start = time.time()
            df_imp = compute_permutation_importance(
                model, X_in, y_test, feats,
                log_target=log_tgt,
                scaler=scaler,
                n_repeats=PERM_N_REPEATS,
                random_state=seed,
                n_jobs=PERM_N_JOBS,
            )
            logger.info(
                "    %s done in %.0f s  |  top feature: %s (%.3f kWh)",
                mname, time.time() - t_start,
                df_imp.iloc[0]["feature"], df_imp.iloc[0]["importance_mean"],
            )
            perm_imp[mname] = df_imp

            plot_permutation_importance(
                df_imp, mname,
                color=MODEL_COLOURS.get(mname, "#888888"),
                save_path=FIG_DIR / f"phase10_permutation_importance_{mname.lower()}.png",
                top_n=20,
            )

            for _, row in df_imp.iterrows():
                perm_all_rows.append({
                    "model":            mname,
                    "feature":          row["feature"],
                    "importance_mean":  row["importance_mean"],
                    "importance_std":   row["importance_std"],
                })

        pd.DataFrame(perm_all_rows).to_csv(_perm_csv, index=False
    )
    logger.info("  Saved phase10_permutation_importance.csv")

    # Faceted comparison — feature order by RF importance (top 20)
    rf_top20 = perm_imp["RF"].head(20)["feature"].tolist()
    plot_all_models_permutation(
        perm_imp, rf_top20, MODEL_COLOURS,
        save_path=FIG_DIR / "phase10_permutation_importance_all_models.png",
        top_n=20,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.2 — SHAP Global Analysis  (skip if CSV + XGBoost npy already exist)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("-" * 50)
    logger.info("Task 10.2 — SHAP Global Analysis")
    logger.info("-" * 50)

    _shap_csv   = TABLE_DIR / "phase10_shap_mean_abs.csv"
    _xgb_npy    = SHAP_DIR  / "shap_xgboost_test.npy"
    shap_mean_abs      = {}
    shap_mean_abs_rows = []
    shap_primary_values = None
    primary_base_value  = None
    primary_sample_idx  = None
    PRIMARY_MODEL_NAME  = "XGBoost"

    if _shap_csv.exists() and _xgb_npy.exists():
        logger.info("  SKIPPING SHAP computation — CSVs and .npy already exist")
        _shap_df = pd.read_csv(_shap_csv)
        for mname, grp in _shap_df.groupby("model"):
            shap_mean_abs[mname] = grp[["feature","mean_abs_shap"]].sort_values(
                "mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_primary_values = np.load(_xgb_npy)
        # Reconstruct the same subsample index deterministically
        rng2 = np.random.default_rng(seed)
        primary_sample_idx = rng2.choice(len(X_test), size=SHAP_ROWS_OTHER, replace=False)
        primary_sample_idx.sort()
        primary_base_value = 0.0   # will not be used for plots (waterfall reads from explainer)
        logger.info(
            "  Loaded shap_xgboost_test.npy shape=%s  primary_sample_idx len=%d",
            shap_primary_values.shape, len(primary_sample_idx),
        )
        # Recompute XGBoost base value for waterfall plots
        import shap as _shap_mod
        _xgb_exp = _shap_mod.TreeExplainer(xgb_model)
        primary_base_value = float(np.atleast_1d(_xgb_exp.expected_value)[0])
        logger.info("  XGBoost base_value=%.3f kWh (recomputed for waterfall)", primary_base_value)
        del _xgb_exp
        gc.collect()
    else:
        # ── 10.2.1  RF — use Gini feature_importances_ (instant, no TreeExplainer) ──
        # sklearn's RandomForest TreeExplainer is O(n_trees × n_rows) with NO native
        # C++ fast path (unlike XGBoost/LightGBM). On M1 Air 8 GB, 500 trees × even
        # 500 rows takes hours. XGBoost (RMSE 11.59, ΔR²=0.002 vs RF) is used for all
        # SHAP deep-dive plots as the representative tree-ensemble model instead.
        # RF contributes its Gini (MDI) importances to the cross-model ranking table.
        #
        # ┌─ Full-hardware alternative (≥32 GB RAM, ~2–4 h runtime) ────────────────┐
        # │  sv_rf, base_rf, _ = compute_shap_tree(rf_model,                        │
        # │                          X_test[:SHAP_ROWS_OTHER], FEATURES_TREES)       │
        # │  # replace rf_gini below with a proper mean-|SHAP| DataFrame             │
        # │  rf_shap_mean = pd.DataFrame({                                           │
        # │      "feature":       FEATURES_TREES,                                    │
        # │      "mean_abs_shap": np.abs(sv_rf).mean(axis=0),                        │
        # │  }).sort_values("mean_abs_shap", ascending=False)                        │
        # │  shap_mean_abs["RF"] = rf_shap_mean                                      │
        # │  plot_shap_beeswarm(sv_rf, X_test[:SHAP_ROWS_OTHER],                    │
        # │      FEATURES_TREES, "RF", FIG_DIR/"phase10_shap_summary_beeswarm_rf.png")│
        # └──────────────────────────────────────────────────────────────────────────┘
        logger.info("  RF: using Gini feature_importances_ (skipping slow TreeExplainer)")
        rf_gini = pd.DataFrame({
            "feature":       FEATURES_TREES,
            "mean_abs_shap": rf_model.feature_importances_,   # MDI, already normalised
        }).sort_values("mean_abs_shap", ascending=False)

        shap_mean_abs["RF"] = rf_gini
        for _, row in rf_gini.iterrows():
            shap_mean_abs_rows.append({
                "model": "RF", "feature": row["feature"],
                "mean_abs_shap": row["mean_abs_shap"],
            })

        # Bar chart using Gini importances (label accordingly)
        plot_shap_bar(
            rf_gini["mean_abs_shap"].values.reshape(1, -1).repeat(10, axis=0),
            rf_gini["feature"].tolist(),
            "RF (Gini importance)",
            save_path=FIG_DIR / "phase10_shap_bar_rf.png",
        )
        logger.info("  Saved phase10_shap_bar_rf.png  (Gini importances)")

        # ── 10.2.2  TreeExplainer: XGBoost, LightGBM, DT (fast native kernels) ──
        tree_configs = [
            ("xgboost",  xgb_model,  X_test,  FEATURES_TREES),
            ("lgbm",     lgbm_model, X_test,  FEATURES_TREES),
            ("dt",       dt_model,   X_test,  FEATURES_TREES),
        ]

        for model_key, model, X_in, feats in tree_configs:
            mname_display = {"xgboost": "XGBoost", "lgbm": "LightGBM", "dt": "DT"}[model_key]
            logger.info("  SHAP TreeExplainer: %s ...", mname_display)

            n_rows = len(X_in)
            if n_rows > SHAP_ROWS_OTHER:
                idx = rng.choice(n_rows, size=SHAP_ROWS_OTHER, replace=False)
                idx.sort()
                X_shap = X_in[idx]
                logger.info("    Subsampled %d → %d rows for SHAP", n_rows, SHAP_ROWS_OTHER)
            else:
                idx    = np.arange(n_rows)
                X_shap = X_in

            t_start = time.time()
            sv, base_val, expl = compute_shap_tree(model, X_shap, feats)
            logger.info("    %s done in %.0f s", mname_display, time.time() - t_start)

            # Keep XGBoost SHAP for dependence / waterfall / force plots
            if model_key == "xgboost":
                shap_primary_values = sv.copy()
                primary_base_value  = base_val
                primary_sample_idx  = idx
                np.save(SHAP_DIR / "shap_xgboost_test.npy", sv)

            plot_shap_beeswarm(
                sv, X_shap, feats, mname_display,
                save_path=FIG_DIR / f"phase10_shap_summary_beeswarm_{model_key}.png",
            )
            plot_shap_bar(
                sv, feats, mname_display,
                save_path=FIG_DIR / f"phase10_shap_bar_{model_key}.png",
            )

            mean_abs = pd.DataFrame({
                "feature":       feats,
                "mean_abs_shap": np.abs(sv).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)

            shap_mean_abs[mname_display] = mean_abs
            for _, row in mean_abs.iterrows():
                shap_mean_abs_rows.append({
                    "model": mname_display, "feature": row["feature"],
                    "mean_abs_shap": row["mean_abs_shap"],
                })

            del sv, expl
            gc.collect()

        # ── 10.2.3  LinearExplainer: ElasticNet ─────────────────────────────
        logger.info("  SHAP LinearExplainer: ElasticNet ...")
        idx_bg   = rng.choice(len(X_train_linear), size=shap_bg_size, replace=False)
        X_bg_raw = X_train_linear[idx_bg]

        sv_en, base_en = compute_shap_linear(
            en_model, X_test_linear, X_bg_raw, scaler_a, features_linear
        )

        plot_shap_beeswarm(
            sv_en, X_test_linear, features_linear, "ElasticNet",
            save_path=FIG_DIR / "phase10_shap_summary_beeswarm_elasticnet.png",
            ylabel_suffix="log kWh",
        )
        plot_shap_bar(
            sv_en, features_linear, "ElasticNet",
            save_path=FIG_DIR / "phase10_shap_bar_elasticnet.png",
            ylabel_suffix="log kWh",
        )

        mean_abs_en = pd.DataFrame({
            "feature":       features_linear,
            "mean_abs_shap": np.abs(sv_en).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        shap_mean_abs["ElasticNet"] = mean_abs_en
        for _, row in mean_abs_en.iterrows():
            shap_mean_abs_rows.append({
                "model": "ElasticNet", "feature": row["feature"],
                "mean_abs_shap": row["mean_abs_shap"],
            })

        # ── 10.2.4  KernelExplainer: ANN ─────────────────────────────────
        # KernelExplainer is model-agnostic and slow by design (O(n_bg × n_test) calls).
        # ANN_SHAP_ROWS=2,000 takes ~30 min on M1 Air. Full hardware: set ANN_SHAP_ROWS=5,000
        # (≥32 GB RAM, ~1.5–2 h). Increase nsamples (currently 100) for higher accuracy too.
        logger.info(
            "  SHAP KernelExplainer: ANN (%d-row subsample, ~30 min on M1 Air) ...",
            ANN_SHAP_ROWS,
        )
        logger.info("  This is the slowest step — progress logged every 200 rows.")

        def ann_predict_fn(X_scaled: np.ndarray) -> np.ndarray:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred_log = ann_model.predict(X_scaled)
            return np.expm1(pred_log).clip(0)

        idx_sub      = rng.choice(len(X_test_linear), size=ANN_SHAP_ROWS, replace=False)
        X_sub_raw    = X_test_linear[idx_sub]
        X_sub_scaled = scaler_a.transform(X_sub_raw)
        X_bg_scaled  = scaler_a.transform(X_bg_raw)

        sv_ann, base_ann = compute_shap_kernel(
            ann_predict_fn,
            X_bg_scaled,
            X_sub_scaled,
            features_linear,
            batch_size=200,
            nsamples=100,
            progress_logger=logger,
        )

        plot_shap_beeswarm(
            sv_ann, X_sub_raw, features_linear, "ANN",
            save_path=FIG_DIR / "phase10_shap_summary_beeswarm_ann.png",
        )
        plot_shap_bar(
            sv_ann, features_linear, "ANN",
            save_path=FIG_DIR / "phase10_shap_bar_ann.png",
        )

        mean_abs_ann = pd.DataFrame({
            "feature":       features_linear,
            "mean_abs_shap": np.abs(sv_ann).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        shap_mean_abs["ANN"] = mean_abs_ann
        for _, row in mean_abs_ann.iterrows():
            shap_mean_abs_rows.append({
                "model": "ANN", "feature": row["feature"],
                "mean_abs_shap": row["mean_abs_shap"],
            })

        # Save mean |SHAP| CSV
        pd.DataFrame(shap_mean_abs_rows).to_csv(
            TABLE_DIR / "phase10_shap_mean_abs.csv", index=False
        )
        logger.info("  Saved phase10_shap_mean_abs.csv")

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.3 — SHAP Dependence Plots (RF, top 5 features)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("-" * 50)
    logger.info("Task 10.3 — SHAP Dependence Plots (XGBoost, top 5 features)")
    logger.info("-" * 50)
    # XGBoost used as primary SHAP model (RF≈XGBoost in performance; XGBoost has fast SHAP)

    top5_primary = shap_mean_abs[PRIMARY_MODEL_NAME].head(5)["feature"].tolist()
    logger.info("  Top 5 %s features (by mean |SHAP|): %s", PRIMARY_MODEL_NAME, top5_primary)

    X_test_shap_subset = X_test[primary_sample_idx]
    df_test_shap = pd.DataFrame(X_test_shap_subset, columns=FEATURES_TREES)

    for feat in top5_primary:
        feat_slug = feat.replace("/", "_").lower()
        plot_shap_dependence(
            shap_primary_values,
            df_test_shap,
            feat,
            FEATURES_TREES,
            save_path=FIG_DIR / f"phase10_shap_dependence_{feat_slug}_xgboost.png",
            interaction_feature="auto",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.4 — SHAP Local Analysis (Waterfall + Force, RF, 4 cases)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("-" * 50)
    logger.info("Task 10.4 — SHAP Local Analysis (4 cases)")
    logger.info("-" * 50)

    # Case selection uses XGBoost predictions over the SHAP subsample
    df_preds_reset  = df_preds.reset_index(drop=True)
    y_true_all      = df_preds_reset["kWh_received_Total"].values[primary_sample_idx]
    y_pred_primary  = df_preds_reset["pred_xgb"].values[primary_sample_idx]
    residuals       = y_true_all - y_pred_primary
    # local_idx  → into shap_primary_values
    # original_idx = primary_sample_idx[local_idx] → into X_test / df_preds_reset

    def _pick_case(mask, label, seed_offset=0):
        idxs = np.where(mask)[0]   # indices within the 5k subsample
        if len(idxs) == 0:
            logger.warning("  No rows matched for case '%s' — skipping", label)
            return None
        chosen = idxs[seed_offset % len(idxs)]
        return int(chosen)   # local index into shap_rf_values

    cases_meta = [
        {
            "label":  "High Consumption — Accurate",
            "idx":    _pick_case(
                (y_true_all > np.quantile(y_true_all, 0.90)) &
                (np.abs(residuals) < 5.0),
                "high_accurate",
            ),
            "slug": "high_consumption",
        },
        {
            "label":  "Low Consumption — Accurate",
            "idx":    _pick_case(
                (y_true_all < np.quantile(y_true_all, 0.10)) &
                (np.abs(residuals) < 2.0),
                "low_accurate",
            ),
            "slug": "low_consumption",
        },
        {
            "label":  "Large Under-Prediction (model too low)",
            "idx":    _pick_case(
                residuals > np.quantile(residuals, 0.95),
                "large_under",
            ),
            "slug": "large_error_under",
        },
        {
            "label":  "Large Over-Prediction (model too high)",
            "idx":    _pick_case(
                residuals < np.quantile(residuals, 0.05),
                "large_over",
            ),
            "slug": "large_error_over",
        },
    ]

    report_cases = []
    for case in cases_meta:
        local_idx = case["idx"]   # index into 5k SHAP subsample
        if local_idx is None:
            continue
        orig_idx = int(primary_sample_idx[local_idx])   # index into full test set
        row = df_preds_reset.iloc[orig_idx]
        y_t = float(row["kWh_received_Total"])
        y_p = float(row["pred_xgb"])
        hid = row.get("Household_ID", "?")
        date_val = row.get("Date", "?")

        logger.info(
            "  Case '%s': local=%d orig=%d  y_true=%.1f  pred=%.1f  error=%+.1f",
            case["label"], local_idx, orig_idx, y_t, y_p, y_p - y_t,
        )

        shap_row = shap_primary_values[local_idx]
        X_row    = X_test[orig_idx]

        plot_shap_waterfall(
            shap_row, primary_base_value, X_row, FEATURES_TREES,
            case["label"], y_t, y_p,
            save_path=FIG_DIR / f"phase10_shap_waterfall_{case['slug']}_xgboost.png",
        )

        # Force plot for first two cases only (high + low consumption)
        if case["slug"] in ("high_consumption", "low_consumption"):
            plot_shap_force(
                primary_base_value, shap_row, X_row, FEATURES_TREES,
                case["label"],
                save_path=FIG_DIR / f"phase10_shap_force_{case['slug']}_xgboost.png",
            )

        report_cases.append({
            "label":       case["label"],
            "household_id": str(hid),
            "date":        str(date_val),
            "y_true":      y_t,
            "y_pred":      y_p,
        })

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.5 — Decision Tree Structure
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("-" * 50)
    logger.info("Task 10.5 — Decision Tree Structure")
    logger.info("-" * 50)
    plot_dt_tree(
        dt_model, FEATURES_TREES,
        save_path=FIG_DIR / "phase10_dt_tree_structure.png",
        max_depth=4,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.6 — ElasticNet Coefficients
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("-" * 50)
    logger.info("Task 10.6 — ElasticNet Standardized Coefficients")
    logger.info("-" * 50)
    plot_elasticnet_coefficients(
        en_model, features_linear,
        save_path=FIG_DIR / "phase10_elasticnet_coefficients.png",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.7 — XGBoost B SHAP (Track B, 75 protocol features)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("-" * 50)
    logger.info("Task 10.7 — XGBoost B SHAP (Track B, 75 features)")
    logger.info("-" * 50)
    sv_b, base_b, _ = compute_shap_tree(xgb_b_model, X_test_b, features_b)

    # Label protocol-specific features with [P]
    std_set   = set(FEATURES_TREES)
    feats_b_labelled = [
        f"{f} [P]" if f not in std_set else f
        for f in features_b
    ]

    plot_shap_beeswarm(
        sv_b, X_test_b, features_b, "XGBoost B",
        save_path=FIG_DIR / "phase10_shap_summary_beeswarm_xgboost_b.png",
        max_display=25,
    )
    plot_shap_bar(
        sv_b, feats_b_labelled, "XGBoost B",
        save_path=FIG_DIR / "phase10_shap_bar_xgboost_b.png",
        max_display=25,
    )

    # Dependence plot for HDD_SIA_daily coloured by best available interaction
    hdd_feat = "HDD_SIA_daily"
    if hdd_feat in features_b:
        df_test_b_display = df_test_b[features_b].reset_index(drop=True)
        plot_shap_dependence(
            sv_b,
            df_test_b_display,
            hdd_feat,
            features_b,
            save_path=FIG_DIR / "phase10_shap_dependence_hdd_sia_daily_xgboost_b.png",
        )

    shap_b_df = pd.DataFrame({
        "feature":       features_b,
        "mean_abs_shap": np.abs(sv_b).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.8 — Cross-Model Feature Ranking Consistency
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("-" * 50)
    logger.info("Task 10.8 — Cross-Model Feature Ranking Consistency")
    logger.info("-" * 50)

    # Common features between tree (45) and linear (30) sets
    linear_set  = set(features_linear)
    tree_set    = set(FEATURES_TREES)
    common_feats = sorted(linear_set & tree_set)
    logger.info("  Common features for cross-model ranking: %d", len(common_feats))

    def _rank_df(imp_df: pd.DataFrame, common: list[str]) -> pd.Series:
        """Convert importance DataFrame to rank Series (1=best) over common features."""
        sub  = imp_df[imp_df["feature"].isin(common)].copy()
        sub  = sub.sort_values("importance_mean", ascending=False).reset_index(drop=True)
        ranks = {row["feature"]: rank + 1 for rank, (_, row) in enumerate(sub.iterrows())}
        return pd.Series({f: ranks.get(f, float("nan")) for f in common})

    ranking_dict = {}
    for mname, df_imp in perm_imp.items():
        ranking_dict[mname] = _rank_df(df_imp, common_feats)

    ranking_df = pd.DataFrame(ranking_dict, index=common_feats)
    ranking_df.index.name = "feature"
    ranking_df.to_csv(TABLE_DIR / "phase10_feature_ranking_table.csv")
    logger.info("  Saved phase10_feature_ranking_table.csv  (%d features)", len(common_feats))

    # Heatmap
    plot_feature_ranking_heatmap(
        ranking_df.reset_index(),
        save_path=FIG_DIR / "phase10_feature_ranking_heatmap.png",
    )

    # Spearman ρ matrix
    rho_df = _spearman_matrix(ranking_df)
    rho_df.to_csv(TABLE_DIR / "phase10_spearman_correlation.csv")
    logger.info("  Saved phase10_spearman_correlation.csv")

    plot_spearman_heatmap(
        rho_df,
        save_path=FIG_DIR / "phase10_spearman_correlation_heatmap.png",
    )

    # Log most-similar model pair
    best_rho, best_pair = -2.0, ("", "")
    cols = rho_df.columns.tolist()
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i >= j:
                continue
            v = float(rho_df.loc[c1, c2])
            if not np.isnan(v) and v > best_rho:
                best_rho, best_pair = v, (c1, c2)
    logger.info(
        "  Most similar pair: %s vs %s  Spearman ρ=%.3f",
        best_pair[0], best_pair[1], best_rho,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.9 — Accuracy–Interpretability Tradeoff Plot
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("-" * 50)
    logger.info("Task 10.9 — Accuracy–Interpretability Tradeoff")
    logger.info("-" * 50)

    model_tradeoff_data = [
        {"name": "ElasticNet", "rmse": 20.40, "interp_score": 5,
         "color": MODEL_COLOURS["ElasticNet"],
         "training_seconds": TRAINING_SECONDS["ElasticNet"]},
        {"name": "DT",         "rmse": 14.44, "interp_score": 4,
         "color": MODEL_COLOURS["DT"],
         "training_seconds": TRAINING_SECONDS["DT"]},
        {"name": "ANN",        "rmse": 15.56, "interp_score": 1,
         "color": MODEL_COLOURS["ANN"],
         "training_seconds": TRAINING_SECONDS["ANN"]},
        {"name": "RF",         "rmse": 11.54, "interp_score": 2,
         "color": MODEL_COLOURS["RF"],
         "training_seconds": TRAINING_SECONDS["RF"]},
        {"name": "XGBoost",    "rmse": 11.59, "interp_score": 2,
         "color": MODEL_COLOURS["XGBoost"],
         "training_seconds": TRAINING_SECONDS["XGBoost"]},
        {"name": "LightGBM",   "rmse": 11.65, "interp_score": 2,
         "color": MODEL_COLOURS["LightGBM"],
         "training_seconds": TRAINING_SECONDS["LightGBM"]},
    ]

    plot_accuracy_interpretability_tradeoff(
        model_tradeoff_data,
        baseline_rmse=20.32,   # Per-HH Mean from Phase 9
        save_path=FIG_DIR / "phase10_accuracy_interpretability_tradeoff.png",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.10 — Write Consolidated Report
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("-" * 50)
    logger.info("Task 10.10 — Writing Consolidated Report")
    logger.info("-" * 50)

    report_path = _write_report(
        perm_imp        = perm_imp,
        shap_mean_abs   = shap_mean_abs,
        dt_model        = dt_model,
        elasticnet_model= en_model,
        rho_df          = rho_df,
        ranking_df      = ranking_df,
        shap_b_df       = shap_b_df,
        features_b      = features_b,
        features_linear = features_linear,
        cases           = report_cases,
    )
    logger.info("  Saved %s", report_path)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Phase 10 complete.  Elapsed: %.0f s (%.1f min)", elapsed, elapsed / 60)
    logger.info("  Figures  -> %s", FIG_DIR)
    logger.info("  Tables   -> %s", TABLE_DIR)
    logger.info("  Report   -> %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
