"""
scripts/10_interpretability.py

Phase 10 -- Interpretability Analysis
======================================

Tasks:
  10.1.A  Permutation importance (selected Track A models)
  10.1.B  Permutation importance (selected Track B models)
  10.2.A  SHAP global: TreeExplainer (XGB/LGBM/DT), Gini (RF),
                       LinearExplainer (EN), KernelExplainer (ANN)
  10.2.B  SHAP global: TreeExplainer for all selected Track B models
  10.3.A  SHAP dependence plots (XGBoost, top 5 features)
  10.3.B  SHAP dependence plots (top 5 per selected Track B model)
  10.4.A  SHAP local waterfall + force (XGBoost, 4 representative cases)
  10.4.B  SHAP local waterfall (2 cases per selected Track B model)
  10.5.A  DT tree-structure visualisation
  10.5.B  DT_B tree-structure visualisation
  10.6    ElasticNet standardised coefficients
  10.8.A  Cross-model feature ranking table + heatmaps (Track A)
  10.8.B  Cross-model feature ranking table + heatmaps (Track B)
  10.9.A  Accuracy-interpretability tradeoff (Track A)
  10.9.B  Accuracy-interpretability tradeoff (Track B)
  10.10   Consolidated interpretability report

Inputs : data/processed/{train,val,test}_{full,protocol}.parquet
         outputs/models/  -- tuned model pickles + scalers
         outputs/tables/phase9_test_predictions{,_b}.parquet
Outputs: outputs/figures/phase10_*.png
         outputs/tables/phase10_*.csv
         outputs/tables/phase10_interpretability_report.txt
         outputs/logs/phase10_run.log
"""

from __future__ import annotations

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
# Memory-safe defaults -- tuned for M1 Air 8 GB unified memory.
#
# ┌─ To run full-fidelity results on a machine with >=32 GB RAM ─────────────┐
# │  PERM_N_JOBS    = -1          # parallelise across all cores             │
# │  PERM_N_REPEATS = 30          # more shuffles -> tighter CIs             │
# │  SHAP_ROWS_OTHER = 74_368     # full test set for XGBoost / LightGBM / DT│
# │  ANN_SHAP_ROWS   = 5_000      # larger subsample for KernelExplainer     │
# └──────────────────────────────────────────────────────────────────────────┘
PERM_N_JOBS     = 1
PERM_N_REPEATS  = 10
SHAP_ROWS_OTHER = 5_000
ANN_SHAP_ROWS   = 2_000

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
DATA_DIR  = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR   = PROJECT_ROOT / "outputs" / "figures"
SHAP_DIR  = TABLE_DIR / "shap_values"
LOG_DIR   = PROJECT_ROOT / "outputs" / "logs"

for _d in (TABLE_DIR, FIG_DIR, SHAP_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


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
# Feature lists -- identical to Phases 9
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
# Phase 8 training-time benchmarks (seconds) -- for tradeoff bubble sizes
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
# Fallback colours for Track B models if not in src/evaluation.py
# ─────────────────────────────────────────────────────────────────────────────
_B_COLOUR_DEFAULTS = {
    "XGBoost_B": "#e377c2",
    "DT_B":      "#bcbd22",
    "RF_B":      "#17becf",
}


# ─────────────────────────────────────────────────────────────────────────────
# Runtime selection -- prompts (print body + short input, avoids readline garble)
# ─────────────────────────────────────────────────────────────────────────────
_TRACK_PROMPT = (
    "\n=== Phase 10 -- Interpretability Analysis ===\n\n"
    "Select track(s) to analyse:\n\n"
    "  [1]  Track A  (full sample, 1,119 HH)\n"
    "       Models: ElasticNet, DT, RF, XGBoost, LightGBM, ANN\n"
    "  [2]  Track B  (protocol-enriched, 109 HH)\n"
    "       Models: XGBoost_B, DT_B, RF_B\n"
    "  [3]  Both\n"
    "  [0]  Exit"
)

_TRACK_A_PROMPT = (
    "\nSelect Track A model(s) to analyse:\n\n"
    "  [1]  ElasticNet\n"
    "  [2]  DT\n"
    "  [3]  RF\n"
    "  [4]  XGBoost\n"
    "  [5]  LightGBM\n"
    "  [6]  ANN\n"
    "  [7]  All\n\n"
    "  Tip: comma-separate for combinations  e.g. 3,4 = RF + XGBoost"
)

_TRACK_B_PROMPT = (
    "\nSelect Track B model(s) to analyse:\n\n"
    "  [1]  XGBoost_B\n"
    "  [2]  DT_B\n"
    "  [3]  RF_B\n"
    "  [4]  All\n\n"
    "  Tip: comma-separate for combinations  e.g. 2,3 = DT_B + RF_B"
)

_ALL_A: set[str] = {"elasticnet", "dt", "rf", "xgboost", "lgbm", "ann"}
_ALL_B: set[str] = {"xgboost_b", "dt_b", "rf_b"}

_A_MAP: dict[str, set[str]] = {
    "1": {"elasticnet"}, "2": {"dt"},   "3": {"rf"},
    "4": {"xgboost"},    "5": {"lgbm"}, "6": {"ann"},
}
_B_MAP: dict[str, set[str]] = {
    "1": {"xgboost_b"}, "2": {"dt_b"}, "3": {"rf_b"},
}


def _parse_multi(
    raw: str,
    mapping: dict,
    all_key: str,
    all_result: set,
    range_hint: str,
) -> set | None:
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        print(f"  No input. Enter numbers {range_hint}, comma-separated.")
        return None
    bad = [t for t in tokens if t not in set(mapping) | {all_key}]
    if bad:
        print(f"  Unrecognised: {bad}. Enter numbers {range_hint}, comma-separated.")
        return None
    if all_key in tokens:
        return all_result.copy()
    result: set[str] = set()
    for t in tokens:
        result |= mapping[t]
    return result


def _get_track_choice() -> set[str]:
    while True:
        print(_TRACK_PROMPT)
        sys.stdout.flush()
        raw = input("Enter choice: ").strip()
        if raw == "0":
            sys.exit(0)
        elif raw == "1":
            return {"A"}
        elif raw == "2":
            return {"B"}
        elif raw == "3":
            return {"A", "B"}
        else:
            print("  Enter 1, 2, 3, or 0.")


def _get_track_a_models() -> set[str]:
    while True:
        print(_TRACK_A_PROMPT)
        sys.stdout.flush()
        raw = input("Enter choice(s): ").strip()
        result = _parse_multi(raw, _A_MAP, "7", _ALL_A, "1-7")
        if result is not None:
            return result


def _get_track_b_models() -> set[str]:
    while True:
        print(_TRACK_B_PROMPT)
        sys.stdout.flush()
        raw = input("Enter choice(s): ").strip()
        result = _parse_multi(raw, _B_MAP, "4", _ALL_B, "1-4")
        if result is not None:
            return result


# ─────────────────────────────────────────────────────────────────────────────
# Model loader -- warns and returns None if file missing, tries fallback
# ─────────────────────────────────────────────────────────────────────────────
def _load_optional(name: str) -> object | None:
    path = MODEL_DIR / f"model_{name}.pkl"
    if path.exists():
        return joblib.load(path)
    _fallbacks = {
        "dt_B_tuned":      "model_dt_B.pkl",
        "rf_B_tuned":      "model_rf_B.pkl",
        "xgboost_b_tuned": "model_xgboost_b.pkl",
    }
    alt = _fallbacks.get(name)
    if alt:
        alt_path = MODEL_DIR / alt
        if alt_path.exists():
            logging.getLogger(__name__).warning(
                "  %s not found -- loading fallback: %s", name, alt
            )
            return joblib.load(alt_path)
    logging.getLogger(__name__).warning("  model_%s.pkl not found and no fallback -- skipping", name)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Task 10.8 helper -- Spearman rank correlation matrix
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
                rho.loc[c1, c2] = float(np.atleast_1d(stat).ravel()[0])
    return rho


# ─────────────────────────────────────────────────────────────────────────────
# Task 10.10 -- consolidated report writer
# ─────────────────────────────────────────────────────────────────────────────
def _write_report(
    perm_imp:         dict,
    shap_mean_abs:    dict,
    dt_model,
    elasticnet_model,
    rho_df:           pd.DataFrame | None,
    ranking_df:       pd.DataFrame | None,
    shap_b_df_all:    dict[str, pd.DataFrame],
    features_b:       list[str],
    features_linear:  list[str],
    cases:            list[dict],
    perm_imp_b:       dict | None = None,
    rho_b_df:         pd.DataFrame | None = None,
    dt_b_model                             = None,
    b_rmse:           dict | None          = None,
) -> str:
    lines: list[str] = []

    def h(text: str) -> None:
        lines.append("")
        lines.append("=" * 70)
        lines.append(text)
        lines.append("=" * 70)

    def sub(text: str) -> None:
        lines.append("")
        lines.append("-" * 50)
        lines.append(text)
        lines.append("-" * 50)

    h("HEAPO-Predict -- Phase 10: Interpretability Analysis Report")
    lines.append("Models: ElasticNet, DT, RF, XGBoost, LightGBM, ANN, XGBoost_B, DT_B, RF_B")
    lines.append("Test set: Dec 2023 - Mar 2024  |  Track A N=74,368  |  Track B N=5,475")

    # ── Section 1: Track A permutation importance ──────────────────────────
    if perm_imp:
        h("Section 1 -- Permutation Importance Summary (top 10 per Track A model)")
        for mname, df in perm_imp.items():
            sub(mname)
            for _, row in df.head(10).iterrows():
                lines.append(
                    f"  {row['feature']:<40}  mean RMSE up = {row['importance_mean']:+.3f} kWh"
                    f"  +/- {row['importance_std']:.3f}"
                )

    # ── Section 1b: Track B permutation importance ─────────────────────────
    if perm_imp_b:
        h("Section 1b -- Permutation Importance Summary (top 10 per Track B model)")
        std_set = set(FEATURES_TREES)
        for mname, df in perm_imp_b.items():
            sub(mname)
            for rank, (_, row) in enumerate(df.head(10).iterrows(), start=1):
                tag = "  [P]" if row["feature"] not in std_set else ""
                lines.append(
                    f"  #{rank:>2}  {row['feature']:<50}  mean RMSE up = {row['importance_mean']:+.3f} kWh"
                    f"  +/- {row['importance_std']:.3f}{tag}"
                )

    # ── Section 2: Track A SHAP global ────────────────────────────────────
    if shap_mean_abs:
        h("Section 2 -- SHAP Global Findings (top 5 by mean |SHAP| per Track A model)")
        for mname, df in shap_mean_abs.items():
            sub(mname)
            for _, row in df.head(5).iterrows():
                lines.append(
                    f"  {row['feature']:<40}  mean |SHAP| = {row['mean_abs_shap']:.3f}"
                )

    # ── Section 2b: Track B SHAP global ───────────────────────────────────
    if shap_b_df_all:
        h("Section 2b -- Track B SHAP Global Findings (top 5 by mean |SHAP|)")
        std_set = set(FEATURES_TREES)
        for mname, df in shap_b_df_all.items():
            sub(mname)
            n_prot_top5 = 0
            for _, row in df.head(5).iterrows():
                tag = "  [P]" if row["feature"] not in std_set else ""
                if tag:
                    n_prot_top5 += 1
                lines.append(
                    f"  {row['feature']:<50}  mean |SHAP| = {row['mean_abs_shap']:.3f}{tag}"
                )
            lines.append(f"\n  Protocol-specific in top-5: {n_prot_top5}/5")

    # ── Section 3: Local prediction explanations ───────────────────────────
    if cases:
        h("Section 3 -- Local Prediction Explanations (XGBoost, 4 representative cases)")
        for case in cases:
            sub(case["label"])
            lines.append(f"  Household_ID       : {case['household_id']}")
            lines.append(f"  Date               : {case['date']}")
            lines.append(f"  y_true             : {case['y_true']:.2f} kWh")
            lines.append(f"  XGBoost prediction : {case['y_pred']:.2f} kWh")
            lines.append(f"  Error (pred-actual): {case['y_pred'] - case['y_true']:+.2f} kWh")

    # ── Section 4: DT root split ───────────────────────────────────────────
    if dt_model is not None:
        h("Section 4 -- Decision Tree Root Split (Track A)")
        tree = dt_model.tree_
        feat_idx  = tree.feature[0]
        threshold = tree.threshold[0]
        n_left    = int(tree.n_node_samples[1])
        n_right   = int(tree.n_node_samples[2])
        n_total   = int(tree.n_node_samples[0])
        fname = FEATURES_TREES[feat_idx] if feat_idx < len(FEATURES_TREES) else f"feature_{feat_idx}"
        lines.append(f"  Root split: {fname} <= {threshold:.4f}")
        lines.append(f"  Left  branch: {n_left:,} samples ({100*n_left/n_total:.1f}%)")
        lines.append(f"  Right branch: {n_right:,} samples ({100*n_right/n_total:.1f}%)")

        def _dt_depth2(node_id: int, depth: int = 0) -> None:
            if depth >= 2 or tree.feature[node_id] < 0:
                return
            fi  = tree.feature[node_id]
            thr = tree.threshold[node_id]
            fn  = FEATURES_TREES[fi] if fi < len(FEATURES_TREES) else f"feature_{fi}"
            indent = "  " * (depth + 1)
            lines.append(f"{indent}Node {node_id}: {fn} <= {thr:.4f}")
            _dt_depth2(tree.children_left[node_id],  depth + 1)
            _dt_depth2(tree.children_right[node_id], depth + 1)

        lines.append("\n  Tree structure (first 2 levels):")
        _dt_depth2(0)

    # ── Section 4b: DT_B root split ───────────────────────────────────────
    if dt_b_model is not None and features_b:
        h("Section 4b -- Decision Tree B Root Split (Track B, 75 features)")
        tree_b    = dt_b_model.tree_
        fi_b      = tree_b.feature[0]
        thr_b     = tree_b.threshold[0]
        n_left_b  = int(tree_b.n_node_samples[1])
        n_right_b = int(tree_b.n_node_samples[2])
        n_tot_b   = int(tree_b.n_node_samples[0])
        fname_b   = features_b[fi_b] if fi_b < len(features_b) else f"feature_{fi_b}"
        is_prot   = fname_b not in set(FEATURES_TREES)
        lines.append(
            f"  Root split: {fname_b} <= {thr_b:.4f}  "
            f"[{'PROTOCOL' if is_prot else 'standard'}]"
        )
        lines.append(f"  Left  branch: {n_left_b:,} samples ({100*n_left_b/n_tot_b:.1f}%)")
        lines.append(f"  Right branch: {n_right_b:,} samples ({100*n_right_b/n_tot_b:.1f}%)")

    # ── Section 5: Track A Spearman rank correlation ───────────────────────
    if rho_df is not None:
        h("Section 5 -- Track A Cross-Model Ranking Consistency (Spearman rho)")
        cols = rho_df.columns.tolist()
        lines.append(f"{'':>12}" + "".join(f"{c:>14}" for c in cols))
        for c1 in cols:
            row_str = f"{c1:>12}"
            for c2 in cols:
                v = rho_df.loc[c1, c2]
                row_str += f"{v:>14.3f}" if not pd.isna(v) else f"{'n/a':>14}"
            lines.append(row_str)
        lines.append("")
        best_rho, best_pair = -2.0, ("", "")
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i >= j:
                    continue
                v = float(rho_df.loc[c1, c2])
                if not np.isnan(v) and v > best_rho:
                    best_rho, best_pair = v, (c1, c2)
        lines.append(f"  Most similar pair: {best_pair[0]} vs {best_pair[1]}  rho = {best_rho:.3f}")

    # ── Section 5b: Track B Spearman rank correlation ─────────────────────
    if rho_b_df is not None:
        h("Section 5b -- Track B Cross-Model Ranking Consistency (Spearman rho)")
        cols_b = rho_b_df.columns.tolist()
        lines.append(f"{'':>12}" + "".join(f"{c:>14}" for c in cols_b))
        for c1 in cols_b:
            row_str = f"{c1:>12}"
            for c2 in cols_b:
                v = rho_b_df.loc[c1, c2]
                row_str += f"{v:>14.3f}" if not pd.isna(v) else f"{'n/a':>14}"
            lines.append(row_str)
        lines.append("")
        best_rho_b, best_pair_b = -2.0, ("", "")
        for i, c1 in enumerate(cols_b):
            for j, c2 in enumerate(cols_b):
                if i >= j:
                    continue
                v = float(rho_b_df.loc[c1, c2])
                if not np.isnan(v) and v > best_rho_b:
                    best_rho_b, best_pair_b = v, (c1, c2)
        lines.append(
            f"  Most similar pair: {best_pair_b[0]} vs {best_pair_b[1]}  rho = {best_rho_b:.3f}"
        )

    # ── Section 6: Track B protocol feature analysis (all models) ─────────
    if shap_b_df_all:
        h("Section 6 -- Track B SHAP: Protocol Feature Analysis")
        std_set = set(FEATURES_TREES)
        for mname, df in shap_b_df_all.items():
            sub(f"{mname} -- top 10 features")
            n_prot = 0
            for rank, row in enumerate(df.head(10).itertuples(), start=1):
                tag = "  [P]" if row.feature not in std_set else ""
                if tag:
                    n_prot += 1
                lines.append(
                    f"  #{rank:>2}  {row.feature:<50}  mean |SHAP| = {row.mean_abs_shap:.3f}{tag}"
                )
            lines.append(f"\n  Protocol-specific in top-10: {n_prot}/10")

    # ── Section 7: Accuracy-interpretability tradeoff ─────────────────────
    h("Section 7 -- Accuracy-Interpretability Tradeoff Summary")
    lines.append("  Track A test RMSE (Dec 2023 - Mar 2024):")
    lines.append("    RF         : 11.54 kWh  R2=0.728  (best Track A)")
    lines.append("    XGBoost    : 11.59 kWh  R2=0.726")
    lines.append("    LightGBM   : 11.65 kWh  R2=0.723")
    lines.append("    DT         : 14.44 kWh  R2=0.575  (best transparent model)")
    lines.append("    ANN        : 15.56 kWh  R2=0.506")
    lines.append("    ElasticNet : 20.40 kWh  R2=0.151")
    lines.append("    XGBoost_B  :  8.42 kWh  R2=0.847  (Track B, 109 HH)")
    lines.append("")
    lines.append("  Cost of choosing DT over RF: +2.90 kWh RMSE (+25.1%)")
    lines.append("  Cost of choosing ElasticNet over DT: +5.96 kWh RMSE (+41.3%)")
    lines.append("")
    lines.append(
        "  Recommendation: RF with SHAP explanations provides near-optimal accuracy\n"
        "  with post-hoc interpretability. For regulatory reporting requiring full\n"
        "  transparency, DT provides rule-based explanations at a 25% accuracy penalty."
    )

    if b_rmse:
        h("Section 7b -- Track B Accuracy-Interpretability Tradeoff")
        lines.append("  Track B test RMSE:")
        for mname, rmse in sorted(b_rmse.items(), key=lambda x: x[1]):
            lines.append(f"    {mname:<12}: {rmse:.2f} kWh")
        if "DT_B" in b_rmse and "RF_B" in b_rmse:
            cost = b_rmse["DT_B"] - b_rmse["RF_B"]
            lines.append(
                f"\n  Cost of choosing DT_B (transparent) over RF_B (ensemble): "
                f"+{cost:.2f} kWh RMSE ({100*cost/b_rmse['RF_B']:.1f}%)"
            )

    report = "\n".join(lines)
    report_path = TABLE_DIR / "phase10_interpretability_report.txt"
    report_path.write_text(report, encoding="utf-8")
    return str(report_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("HEAPO-Predict -- Phase 10: Interpretability Analysis")
    logger.info("=" * 60)

    cfg          = load_config("config/params.yaml")
    seed         = cfg["modeling"]["random_seed"]
    shap_bg_size = cfg["modeling"]["shap_background_samples"]   # 200

    # ── Runtime selection (all prompts before any work) ───────────────────
    tracks = _get_track_choice()
    track_a_models: set[str] = _get_track_a_models() if "A" in tracks else set()
    track_b_models: set[str] = _get_track_b_models() if "B" in tracks else set()
    logger.info("Track A selection: %s", sorted(track_a_models) or "none")
    logger.info("Track B selection: %s", sorted(track_b_models) or "none")

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading parquets...")
    df_train = pd.read_parquet(DATA_DIR / "train_full.parquet")
    df_test  = pd.read_parquet(DATA_DIR / "test_full.parquet")
    y_test   = df_test["kWh_received_Total"].values
    X_test   = df_test[FEATURES_TREES].values
    logger.info("  train_full : %s", df_train.shape)
    logger.info("  test_full  : %s", df_test.shape)

    # Track B parquets -- only if Track B is selected
    features_b: list[str] = []
    X_test_b:  np.ndarray | None = None
    y_test_b:  np.ndarray | None = None
    df_test_b: pd.DataFrame | None = None

    if track_b_models:
        df_train_b = pd.read_parquet(DATA_DIR / "train_protocol.parquet")
        df_test_b  = pd.read_parquet(DATA_DIR / "test_protocol.parquet")
        logger.info("  train_B    : %s", df_train_b.shape)
        logger.info("  test_B     : %s", df_test_b.shape)

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

        # Null-fill using training-set medians only (DT/RF crash on NaN)
        _train_b_df = df_train_b[features_b].copy()
        _test_b_df  = df_test_b[features_b].copy()
        _null_cols_b = _train_b_df.columns[_train_b_df.isnull().any()].tolist()
        if _null_cols_b:
            _train_b_medians = _train_b_df[_null_cols_b].median()
            _train_b_df[_null_cols_b] = _train_b_df[_null_cols_b].fillna(_train_b_medians)
            _test_b_df[_null_cols_b]  = _test_b_df[_null_cols_b].fillna(_train_b_medians)
            logger.info("  Track B: filled NaN in %d columns with training-set medians",
                        len(_null_cols_b))

        X_test_b = _test_b_df.values
        y_test_b = df_test_b["kWh_received_Total"].values
        logger.info("  FEATURES_TREES_B=%d  X_test_b shape=%s", len(features_b), X_test_b.shape)

    # ── Load scalers and feature meta (Track A only) ──────────────────────
    scaler_a:       object | None = None
    features_linear: list[str]   = []
    X_test_linear:  np.ndarray | None = None
    X_train_linear: np.ndarray | None = None

    if track_a_models:
        logger.info("Loading scalers and feature metadata...")
        scaler_a = joblib.load(MODEL_DIR / "scaler_linear_A.pkl")
        meta_a   = json.loads((MODEL_DIR / "scaler_linear_A_meta.json").read_text())
        features_linear = meta_a["feature_names"]   # 30 features
        X_test_linear   = df_test[features_linear].values
        X_train_linear  = df_train[features_linear].values
        logger.info("  FEATURES_LINEAR=%d", len(features_linear))

    # ── Load models (file-existence based) ────────────────────────────────
    logger.info("Loading models...")
    rf_model    = _load_optional("rf_tuned")        if "rf"         in track_a_models else None
    xgb_model   = _load_optional("xgboost_tuned")   if "xgboost"    in track_a_models else None
    lgbm_model  = _load_optional("lgbm_tuned")      if "lgbm"       in track_a_models else None
    dt_model    = _load_optional("dt_tuned")        if "dt"         in track_a_models else None
    ann_model   = _load_optional("ann_tuned")       if "ann"        in track_a_models else None
    en_model    = _load_optional("elasticnet_tuned") if "elasticnet" in track_a_models else None
    xgb_b_model = _load_optional("xgboost_b_tuned") if "xgboost_b"  in track_b_models else None
    dt_b_model  = _load_optional("dt_B_tuned")      if "dt_b"       in track_b_models else None
    rf_b_model  = _load_optional("rf_B_tuned")      if "rf_b"       in track_b_models else None

    # Track B model dict (non-None only)
    models_b: dict[str, object] = {}
    if xgb_b_model is not None:
        models_b["XGBoost_B"] = xgb_b_model
    if dt_b_model is not None:
        models_b["DT_B"]      = dt_b_model
    if rf_b_model is not None:
        models_b["RF_B"]      = rf_b_model

    # RNG -- used throughout for reproducible subsampling
    rng = np.random.default_rng(seed)

    # ── Load Phase 9 predictions (Track A) ───────────────────────────────
    df_preds: pd.DataFrame | None = None
    if track_a_models:
        _pred_path = TABLE_DIR / "phase9_test_predictions.parquet"
        if _pred_path.exists():
            df_preds = pd.read_parquet(_pred_path)
        else:
            logger.warning("  phase9_test_predictions.parquet not found -- waterfall cases skipped")

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.1.A -- Permutation Importance (Track A)
    # ═══════════════════════════════════════════════════════════════════════
    perm_imp: dict[str, pd.DataFrame] = {}

    if track_a_models:
        logger.info("-" * 50)
        logger.info("Task 10.1.A -- Permutation Importance (Track A)")
        logger.info("-" * 50)

        _perm_csv = TABLE_DIR / "phase10_permutation_importance.csv"

        if _perm_csv.exists():
            logger.info("  SKIPPING -- %s already exists (delete to re-run)", _perm_csv.name)
            _perm_df = pd.read_csv(_perm_csv)
            for mname, grp in _perm_df.groupby("model"):
                perm_imp[mname] = grp[["feature", "importance_mean", "importance_std"]].reset_index(drop=True)
        else:
            perm_configs: list[tuple] = []
            if "elasticnet" in track_a_models and en_model is not None:
                perm_configs.append(("ElasticNet", en_model, X_test_linear, features_linear, True, scaler_a))
            if "dt" in track_a_models and dt_model is not None:
                perm_configs.append(("DT",        dt_model,   X_test,        FEATURES_TREES,  False, None))
            if "rf" in track_a_models and rf_model is not None:
                perm_configs.append(("RF",        rf_model,   X_test,        FEATURES_TREES,  False, None))
            if "xgboost" in track_a_models and xgb_model is not None:
                perm_configs.append(("XGBoost",   xgb_model,  X_test,        FEATURES_TREES,  False, None))
            if "lgbm" in track_a_models and lgbm_model is not None:
                perm_configs.append(("LightGBM",  lgbm_model, X_test,        FEATURES_TREES,  False, None))
            if "ann" in track_a_models and ann_model is not None:
                perm_configs.append(("ANN",       ann_model,  X_test_linear, features_linear, True, scaler_a))

            perm_all_rows: list[dict] = []
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
                        "model": mname, "feature": row["feature"],
                        "importance_mean": row["importance_mean"],
                        "importance_std":  row["importance_std"],
                    })

            pd.DataFrame(perm_all_rows).to_csv(_perm_csv, index=False)
            logger.info("  Saved phase10_permutation_importance.csv")

        if len(perm_imp) >= 2:
            ref_a  = "RF" if "RF" in perm_imp else next(iter(perm_imp))
            top20  = perm_imp[ref_a].head(20)["feature"].tolist()
            plot_all_models_permutation(
                perm_imp, top20, MODEL_COLOURS,
                save_path=FIG_DIR / "phase10_permutation_importance_all_models.png",
                top_n=20,
            )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.1.B -- Permutation Importance (Track B)
    # ═══════════════════════════════════════════════════════════════════════
    perm_imp_b: dict[str, pd.DataFrame] = {}

    if models_b:
        logger.info("-" * 50)
        logger.info("Task 10.1.B -- Permutation Importance (Track B)")
        logger.info("-" * 50)

        _perm_b_csv = TABLE_DIR / "phase10_permutation_importance_b.csv"

        if _perm_b_csv.exists():
            logger.info("  SKIPPING -- %s already exists (delete to re-run)", _perm_b_csv.name)
            _df = pd.read_csv(_perm_b_csv)
            for mname, grp in _df.groupby("model"):
                perm_imp_b[mname] = grp[["feature", "importance_mean", "importance_std"]].reset_index(drop=True)
        else:
            perm_b_rows: list[dict] = []
            for b_name, b_model in models_b.items():
                logger.info("  Computing Track B permutation importance: %s ...", b_name)
                t_start = time.time()
                df_imp_b = compute_permutation_importance(
                    b_model, X_test_b, y_test_b, features_b,
                    log_target=False,
                    scaler=None,
                    n_repeats=PERM_N_REPEATS,
                    random_state=seed,
                    n_jobs=PERM_N_JOBS,
                )
                logger.info(
                    "    %s done in %.0f s  |  top feature: %s (%.3f kWh)",
                    b_name, time.time() - t_start,
                    df_imp_b.iloc[0]["feature"], df_imp_b.iloc[0]["importance_mean"],
                )
                perm_imp_b[b_name] = df_imp_b
                _b_color = MODEL_COLOURS.get(b_name, _B_COLOUR_DEFAULTS.get(b_name, "#888888"))
                plot_permutation_importance(
                    df_imp_b, b_name,
                    color=_b_color,
                    save_path=FIG_DIR / f"phase10_permutation_importance_{b_name.lower()}.png",
                    top_n=25,
                )
                for _, row in df_imp_b.iterrows():
                    perm_b_rows.append({
                        "model": b_name, "feature": row["feature"],
                        "importance_mean": row["importance_mean"],
                        "importance_std":  row["importance_std"],
                    })

            pd.DataFrame(perm_b_rows).to_csv(_perm_b_csv, index=False)
            logger.info("  Saved phase10_permutation_importance_b.csv")

        if len(perm_imp_b) >= 2:
            ref_b  = "XGBoost_B" if "XGBoost_B" in perm_imp_b else next(iter(perm_imp_b))
            top25b = perm_imp_b[ref_b].head(25)["feature"].tolist()
            plot_all_models_permutation(
                perm_imp_b, top25b,
                {k: MODEL_COLOURS.get(k, _B_COLOUR_DEFAULTS.get(k, "#888888")) for k in perm_imp_b},
                save_path=FIG_DIR / "phase10_permutation_importance_all_b_models.png",
                top_n=25,
            )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.2.A -- SHAP Global Analysis (Track A)
    # ═══════════════════════════════════════════════════════════════════════
    shap_mean_abs: dict[str, pd.DataFrame] = {}
    shap_primary_values: np.ndarray | None = None
    primary_base_value:  float             = 0.0
    primary_sample_idx:  np.ndarray | None = None
    PRIMARY_MODEL_NAME   = "XGBoost"

    if track_a_models:
        logger.info("-" * 50)
        logger.info("Task 10.2.A -- SHAP Global Analysis (Track A)")
        logger.info("-" * 50)

        _shap_csv = TABLE_DIR / "phase10_shap_mean_abs.csv"
        _xgb_npy  = SHAP_DIR  / "shap_xgboost_test.npy"

        if _shap_csv.exists() and _xgb_npy.exists():
            logger.info("  SKIPPING SHAP computation -- CSVs and .npy already exist")
            _shap_df = pd.read_csv(_shap_csv)
            for mname, grp in _shap_df.groupby("model"):
                shap_mean_abs[mname] = grp[["feature", "mean_abs_shap"]].sort_values(
                    "mean_abs_shap", ascending=False).reset_index(drop=True)
            shap_primary_values = np.load(_xgb_npy)
            _rng2 = np.random.default_rng(seed)
            primary_sample_idx = _rng2.choice(len(X_test), size=SHAP_ROWS_OTHER, replace=False)
            primary_sample_idx.sort()
            logger.info(
                "  Loaded shap_xgboost_test.npy shape=%s  idx len=%d",
                shap_primary_values.shape, len(primary_sample_idx),
            )
            if xgb_model is not None:
                import shap as _shap_mod
                _xgb_exp = _shap_mod.TreeExplainer(xgb_model)
                primary_base_value = float(np.atleast_1d(_xgb_exp.expected_value)[0])
                logger.info("  XGBoost base_value=%.3f kWh (recomputed)", primary_base_value)
                del _xgb_exp
                gc.collect()
        else:
            shap_mean_abs_rows: list[dict] = []

            # RF -- Gini MDI (sklearn TreeExplainer is prohibitively slow for 500 trees)
            if "rf" in track_a_models and rf_model is not None:
                logger.info("  RF: using Gini feature_importances_ (skipping slow TreeExplainer)")
                rf_gini = pd.DataFrame({
                    "feature":       FEATURES_TREES,
                    "mean_abs_shap": rf_model.feature_importances_,
                }).sort_values("mean_abs_shap", ascending=False)
                shap_mean_abs["RF"] = rf_gini
                plot_shap_bar(
                    rf_gini["mean_abs_shap"].values.reshape(1, -1).repeat(10, axis=0),
                    rf_gini["feature"].tolist(),
                    "RF (Gini importance)",
                    save_path=FIG_DIR / "phase10_shap_bar_rf.png",
                )
                logger.info("  Saved phase10_shap_bar_rf.png  (Gini importances)")
                for _, row in rf_gini.iterrows():
                    shap_mean_abs_rows.append({"model": "RF", "feature": row["feature"],
                                               "mean_abs_shap": row["mean_abs_shap"]})

            # XGBoost, LightGBM, DT -- TreeExplainer
            tree_configs_a: list[tuple] = []
            if "xgboost" in track_a_models and xgb_model is not None:
                tree_configs_a.append(("xgboost",  xgb_model,  FEATURES_TREES))
            if "lgbm" in track_a_models and lgbm_model is not None:
                tree_configs_a.append(("lgbm",     lgbm_model, FEATURES_TREES))
            if "dt" in track_a_models and dt_model is not None:
                tree_configs_a.append(("dt",       dt_model,   FEATURES_TREES))

            for model_key, model, feats in tree_configs_a:
                mname_display = {"xgboost": "XGBoost", "lgbm": "LightGBM", "dt": "DT"}[model_key]
                logger.info("  SHAP TreeExplainer: %s ...", mname_display)
                n_rows = len(X_test)
                if n_rows > SHAP_ROWS_OTHER:
                    idx = rng.choice(n_rows, size=SHAP_ROWS_OTHER, replace=False)
                    idx.sort()
                    X_shap = X_test[idx]
                    logger.info("    Subsampled %d -> %d rows", n_rows, SHAP_ROWS_OTHER)
                else:
                    idx    = np.arange(n_rows)
                    X_shap = X_test
                t_start = time.time()
                sv, base_val, expl = compute_shap_tree(model, X_shap, feats)
                logger.info("    %s done in %.0f s", mname_display, time.time() - t_start)

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
                    shap_mean_abs_rows.append({"model": mname_display, "feature": row["feature"],
                                               "mean_abs_shap": row["mean_abs_shap"]})
                del sv, expl
                gc.collect()

            # ElasticNet -- LinearExplainer
            if "elasticnet" in track_a_models and en_model is not None:
                logger.info("  SHAP LinearExplainer: ElasticNet ...")
                idx_bg   = rng.choice(len(X_train_linear), size=shap_bg_size, replace=False)
                X_bg_raw = X_train_linear[idx_bg]
                sv_en, _ = compute_shap_linear(en_model, X_test_linear, X_bg_raw, scaler_a, features_linear)
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
                    shap_mean_abs_rows.append({"model": "ElasticNet", "feature": row["feature"],
                                               "mean_abs_shap": row["mean_abs_shap"]})

            # ANN -- KernelExplainer (slowest step, ~30 min on M1 Air)
            if "ann" in track_a_models and ann_model is not None:
                logger.info(
                    "  SHAP KernelExplainer: ANN (%d-row subsample, ~30 min on M1 Air) ...",
                    ANN_SHAP_ROWS,
                )
                logger.info("  This is the slowest step -- progress logged every 200 rows.")

                def ann_predict_fn(X_scaled: np.ndarray) -> np.ndarray:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pred_log = ann_model.predict(X_scaled)
                    return np.expm1(pred_log).clip(0)

                if "X_bg_raw" not in dir():  # may be unset if elasticnet was skipped
                    idx_bg   = rng.choice(len(X_train_linear), size=shap_bg_size, replace=False)
                    X_bg_raw = X_train_linear[idx_bg]

                idx_sub      = rng.choice(len(X_test_linear), size=ANN_SHAP_ROWS, replace=False)
                X_sub_raw    = X_test_linear[idx_sub]
                X_sub_scaled = scaler_a.transform(X_sub_raw)
                X_bg_scaled  = scaler_a.transform(X_bg_raw)

                sv_ann, _ = compute_shap_kernel(
                    ann_predict_fn, X_bg_scaled, X_sub_scaled,
                    features_linear, batch_size=200, nsamples=100, progress_logger=logger,
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
                    shap_mean_abs_rows.append({"model": "ANN", "feature": row["feature"],
                                               "mean_abs_shap": row["mean_abs_shap"]})

            pd.DataFrame(shap_mean_abs_rows).to_csv(
                TABLE_DIR / "phase10_shap_mean_abs.csv", index=False
            )
            logger.info("  Saved phase10_shap_mean_abs.csv")

        # Fall back primary model if XGBoost wasn't selected
        if shap_primary_values is None and shap_mean_abs:
            PRIMARY_MODEL_NAME = next(iter(shap_mean_abs))
            logger.info("  Primary SHAP model: %s (XGBoost not selected)", PRIMARY_MODEL_NAME)

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.2.B -- SHAP Global Analysis (Track B)
    # ═══════════════════════════════════════════════════════════════════════
    shap_mean_abs_b: dict[str, pd.DataFrame] = {}
    shap_primary_b:  dict[str, np.ndarray]   = {}
    base_vals_b:     dict[str, float]        = {}
    b_shap_idx: np.ndarray | None = None

    if models_b:
        logger.info("-" * 50)
        logger.info("Task 10.2.B -- SHAP Global Analysis (Track B)")
        logger.info("-" * 50)

        # Deterministic subsample index for all Track B models (same idx = same rows)
        _n_rows_b = len(X_test_b)
        if _n_rows_b > SHAP_ROWS_OTHER:
            _rng_b   = np.random.default_rng(seed + 1)   # offset to decouple from Track A rng
            b_shap_idx = _rng_b.choice(_n_rows_b, size=SHAP_ROWS_OTHER, replace=False)
            b_shap_idx.sort()
        else:
            b_shap_idx = np.arange(_n_rows_b)
        X_shap_b_base = X_test_b[b_shap_idx]

        _shap_b_csv = TABLE_DIR / "phase10_shap_mean_abs_b.csv"
        # RF_B uses Gini MDI (no TreeExplainer, no .npy) -- exclude from npy check
        _b_needs_npy = [b for b in models_b if b != "RF_B"]
        _all_b_npy_exist = all(
            (SHAP_DIR / f"shap_{b.lower()}_test.npy").exists()
            for b in _b_needs_npy
        )

        if _shap_b_csv.exists() and _all_b_npy_exist:
            logger.info("  SKIPPING Track B SHAP -- CSVs and .npy already exist")
            _df_b = pd.read_csv(_shap_b_csv)
            for mname, grp in _df_b.groupby("model"):
                shap_mean_abs_b[mname] = grp[["feature", "mean_abs_shap"]].sort_values(
                    "mean_abs_shap", ascending=False).reset_index(drop=True)
            import shap as _shap_mod
            for b_name, b_model in models_b.items():
                if b_name == "RF_B":
                    continue   # no npy for RF_B; Gini values already in CSV
                b_key = b_name.lower()
                npy   = SHAP_DIR / f"shap_{b_key}_test.npy"
                if npy.exists():
                    shap_primary_b[b_name] = np.load(npy)
                _exp = _shap_mod.TreeExplainer(b_model)
                base_vals_b[b_name] = float(np.atleast_1d(_exp.expected_value)[0])
                del _exp
            gc.collect()
            logger.info("  Loaded Track B SHAP .npy files: %s", list(shap_primary_b.keys()))
        else:
            shap_b_rows: list[dict] = []
            std_set = set(FEATURES_TREES)

            for b_name, b_model in models_b.items():
                b_key    = b_name.lower()
                npy_path = SHAP_DIR / f"shap_{b_key}_test.npy"

                # RF_B: sklearn TreeExplainer has no fast C++ path for RandomForest.
                # Use Gini MDI (feature_importances_) -- instant, comparable signal.
                # Same approach as RF in Track A (Task 10.2.A).
                if b_name == "RF_B":
                    logger.info(
                        "  RF_B: using Gini feature_importances_ (skipping slow TreeExplainer)"
                    )
                    rf_b_gini = pd.DataFrame({
                        "feature":       features_b,
                        "mean_abs_shap": b_model.feature_importances_,
                    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

                    feats_b_labelled_gini = [
                        f"{f} [P]" if f not in std_set else f for f in features_b
                    ]
                    # Gini bar chart -- replicate array so plot_shap_bar gets a 2-D input
                    plot_shap_bar(
                        rf_b_gini["mean_abs_shap"].values.reshape(1, -1).repeat(10, axis=0),
                        feats_b_labelled_gini,
                        "RF_B (Gini importance)",
                        save_path=FIG_DIR / "phase10_shap_bar_rf_b.png",
                        max_display=25,
                    )
                    logger.info("  Saved phase10_shap_bar_rf_b.png  (Gini importances)")

                    shap_mean_abs_b["RF_B"] = rf_b_gini
                    for _, row in rf_b_gini.iterrows():
                        shap_b_rows.append({"model": "RF_B", "feature": row["feature"],
                                            "mean_abs_shap": row["mean_abs_shap"]})
                    # No npy saved; no entry in shap_primary_b.
                    # Downstream tasks (dependence, waterfall) skip RF_B gracefully.
                    continue

                logger.info("  SHAP TreeExplainer: %s (%d rows) ...", b_name, len(X_shap_b_base))
                t_start = time.time()
                sv_b, base_b, _ = compute_shap_tree(b_model, X_shap_b_base, features_b)
                logger.info("    %s done in %.0f s", b_name, time.time() - t_start)

                np.save(npy_path, sv_b)
                shap_primary_b[b_name] = sv_b.copy()
                base_vals_b[b_name]    = base_b

                feats_b_labelled = [
                    f"{f} [P]" if f not in std_set else f
                    for f in features_b
                ]
                plot_shap_beeswarm(
                    sv_b, X_shap_b_base, features_b, b_name,
                    save_path=FIG_DIR / f"phase10_shap_summary_beeswarm_{b_key}.png",
                    max_display=25,
                )
                plot_shap_bar(
                    sv_b, feats_b_labelled, b_name,
                    save_path=FIG_DIR / f"phase10_shap_bar_{b_key}.png",
                    max_display=25,
                )

                mean_abs_b = pd.DataFrame({
                    "feature":       features_b,
                    "mean_abs_shap": np.abs(sv_b).mean(axis=0),
                }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
                shap_mean_abs_b[b_name] = mean_abs_b
                for _, row in mean_abs_b.iterrows():
                    shap_b_rows.append({"model": b_name, "feature": row["feature"],
                                        "mean_abs_shap": row["mean_abs_shap"]})

                del sv_b
                gc.collect()

            pd.DataFrame(shap_b_rows).to_csv(_shap_b_csv, index=False)
            logger.info("  Saved phase10_shap_mean_abs_b.csv")

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.3.A -- SHAP Dependence Plots (XGBoost, top 5 features)
    # ═══════════════════════════════════════════════════════════════════════
    if shap_primary_values is not None and PRIMARY_MODEL_NAME in shap_mean_abs:
        logger.info("-" * 50)
        logger.info("Task 10.3.A -- SHAP Dependence Plots (%s, top 5 features)", PRIMARY_MODEL_NAME)
        logger.info("-" * 50)

        top5_primary = shap_mean_abs[PRIMARY_MODEL_NAME].head(5)["feature"].tolist()
        logger.info("  Top 5 %s features: %s", PRIMARY_MODEL_NAME, top5_primary)

        X_shap_subset = X_test[primary_sample_idx]
        df_shap_a = pd.DataFrame(X_shap_subset, columns=FEATURES_TREES)

        for feat in top5_primary:
            feat_slug = feat.replace("/", "_").lower()
            plot_shap_dependence(
                shap_primary_values, df_shap_a, feat, FEATURES_TREES,
                save_path=FIG_DIR / f"phase10_shap_dependence_{feat_slug}_xgboost.png",
                interaction_feature="auto",
            )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.3.B -- SHAP Dependence Plots (Track B, top 5 per model)
    # ═══════════════════════════════════════════════════════════════════════
    if shap_primary_b and features_b:
        logger.info("-" * 50)
        logger.info("Task 10.3.B -- SHAP Dependence Plots (Track B)")
        logger.info("-" * 50)

        df_shap_b_base = pd.DataFrame(X_shap_b_base, columns=features_b)

        for b_name, b_shap in shap_primary_b.items():
            b_key = b_name.lower()
            if b_name not in shap_mean_abs_b:
                continue
            top5_b = shap_mean_abs_b[b_name].head(5)["feature"].tolist()
            logger.info("  %s top-5 features: %s", b_name, top5_b)
            for feat in top5_b:
                feat_slug = feat.replace("/", "_").lower()
                plot_shap_dependence(
                    b_shap, df_shap_b_base, feat, features_b,
                    save_path=FIG_DIR / f"phase10_shap_dependence_{feat_slug}_{b_key}.png",
                    interaction_feature="auto",
                )
            logger.info("  Saved %d dependence plots for %s", len(top5_b), b_name)

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.4.A -- SHAP Local Analysis (XGBoost, 4 representative cases)
    # ═══════════════════════════════════════════════════════════════════════
    report_cases: list[dict] = []

    if shap_primary_values is not None and df_preds is not None:
        logger.info("-" * 50)
        logger.info("Task 10.4.A -- SHAP Local Analysis (4 cases, XGBoost)")
        logger.info("-" * 50)

        df_preds_reset = df_preds.reset_index(drop=True)
        y_true_all     = df_preds_reset["kWh_received_Total"].values[primary_sample_idx]
        y_pred_primary = df_preds_reset["pred_xgb"].values[primary_sample_idx]
        residuals      = y_true_all - y_pred_primary

        def _pick_case(mask: np.ndarray, label: str, seed_offset: int = 0) -> int | None:
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                logger.warning("  No rows matched for case '%s' -- skipping", label)
                return None
            return int(idxs[seed_offset % len(idxs)])

        cases_meta = [
            {"label": "High Consumption -- Accurate",           "slug": "high_consumption",
             "idx": _pick_case((y_true_all > np.quantile(y_true_all, 0.90)) & (np.abs(residuals) < 5.0), "high_accurate")},
            {"label": "Low Consumption -- Accurate",            "slug": "low_consumption",
             "idx": _pick_case((y_true_all < np.quantile(y_true_all, 0.10)) & (np.abs(residuals) < 2.0), "low_accurate")},
            {"label": "Large Under-Prediction (model too low)", "slug": "large_error_under",
             "idx": _pick_case(residuals > np.quantile(residuals, 0.95), "large_under")},
            {"label": "Large Over-Prediction (model too high)", "slug": "large_error_over",
             "idx": _pick_case(residuals < np.quantile(residuals, 0.05), "large_over")},
        ]

        for case in cases_meta:
            local_idx = case["idx"]
            if local_idx is None:
                continue
            orig_idx = int(primary_sample_idx[local_idx])
            row      = df_preds_reset.iloc[orig_idx]
            y_t      = float(row["kWh_received_Total"])
            y_p      = float(row["pred_xgb"])
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
            if case["slug"] in ("high_consumption", "low_consumption"):
                plot_shap_force(
                    primary_base_value, shap_row, X_row, FEATURES_TREES,
                    case["label"],
                    save_path=FIG_DIR / f"phase10_shap_force_{case['slug']}_xgboost.png",
                )
            report_cases.append({
                "label": case["label"],
                "household_id": str(row.get("Household_ID", "?")),
                "date":         str(row.get("Date", "?")),
                "y_true": y_t, "y_pred": y_p,
            })

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.4.B -- SHAP Local Waterfall (Track B, 2 cases per model)
    # ═══════════════════════════════════════════════════════════════════════
    if shap_primary_b and features_b:
        logger.info("-" * 50)
        logger.info("Task 10.4.B -- SHAP Local Waterfall (Track B, 2 cases per model)")
        logger.info("-" * 50)

        _b_pred_parquet = TABLE_DIR / "phase9_test_predictions_b.parquet"
        if not _b_pred_parquet.exists():
            logger.warning(
                "  phase9_test_predictions_b.parquet not found -- Task 10.4.B skipped"
            )
        else:
            df_preds_b = pd.read_parquet(_b_pred_parquet).reset_index(drop=True)

            for b_name, b_shap in shap_primary_b.items():
                b_key    = b_name.lower()
                pred_col = f"pred_{b_key}"

                if pred_col not in df_preds_b.columns:
                    logger.warning("  Column %s not in Track B predictions -- skipping %s",
                                   pred_col, b_name)
                    continue

                y_true_b_all = df_preds_b["kWh_received_Total"].values
                y_pred_b_all = df_preds_b[pred_col].values
                res_b_all    = y_true_b_all - y_pred_b_all

                # Align with the SHAP subsample index
                y_true_sub_b = y_true_b_all[b_shap_idx]
                y_pred_sub_b = y_pred_b_all[b_shap_idx]
                res_sub_b    = res_b_all[b_shap_idx]

                def _pick_b_case(mask: np.ndarray, label: str) -> int | None:
                    idxs = np.where(mask)[0]
                    if len(idxs) == 0:
                        logger.warning("  No rows for Track B case '%s' (%s)", label, b_name)
                        return None
                    return int(idxs[0])

                b_cases_meta = [
                    {"label": "High Consumption", "slug": "high_consumption",
                     "idx": _pick_b_case(
                         (y_true_sub_b > np.quantile(y_true_sub_b, 0.90)) & (np.abs(res_sub_b) < 5.0),
                         "high_consumption")},
                    {"label": "Large Error", "slug": "large_error",
                     "idx": _pick_b_case(
                         res_sub_b > np.quantile(res_sub_b, 0.95),
                         "large_error")},
                ]

                import shap as _shap_mod_b
                _b_exp = _shap_mod_b.TreeExplainer(models_b[b_name])
                base_b_val = float(np.atleast_1d(_b_exp.expected_value)[0])
                del _b_exp
                gc.collect()

                for case_b in b_cases_meta:
                    local_idx_b = case_b["idx"]
                    if local_idx_b is None:
                        continue
                    orig_idx_b = int(b_shap_idx[local_idx_b])
                    y_t_b = float(y_true_b_all[orig_idx_b])
                    y_p_b = float(y_pred_b_all[orig_idx_b])
                    logger.info(
                        "  %s / %s: local=%d orig=%d  y_true=%.1f  pred=%.1f  error=%+.1f",
                        b_name, case_b["label"], local_idx_b, orig_idx_b, y_t_b, y_p_b, y_p_b - y_t_b,
                    )
                    plot_shap_waterfall(
                        b_shap[local_idx_b], base_b_val,
                        X_test_b[orig_idx_b], features_b,
                        case_b["label"], y_t_b, y_p_b,
                        save_path=FIG_DIR / f"phase10_shap_waterfall_{case_b['slug']}_{b_key}.png",
                    )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.5.A -- Decision Tree Structure (Track A)
    # ═══════════════════════════════════════════════════════════════════════
    if "dt" in track_a_models and dt_model is not None:
        logger.info("-" * 50)
        logger.info("Task 10.5.A -- Decision Tree Structure (Track A)")
        logger.info("-" * 50)
        plot_dt_tree(
            dt_model, FEATURES_TREES,
            save_path=FIG_DIR / "phase10_dt_tree_structure.png",
            max_depth=4,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.5.B -- Decision Tree B Structure (Track B)
    # ═══════════════════════════════════════════════════════════════════════
    if "dt_b" in track_b_models and dt_b_model is not None:
        logger.info("-" * 50)
        logger.info("Task 10.5.B -- Decision Tree B Structure (Track B)")
        logger.info("-" * 50)
        plot_dt_tree(
            dt_b_model, features_b,
            save_path=FIG_DIR / "phase10_dt_b_tree_structure.png",
            max_depth=4,
        )
        _tree_b   = dt_b_model.tree_
        _fi_b     = _tree_b.feature[0]
        _thr_b    = _tree_b.threshold[0]
        _fn_b     = features_b[_fi_b] if _fi_b < len(features_b) else f"feat_{_fi_b}"
        _is_prot  = _fn_b not in set(FEATURES_TREES)
        logger.info(
            "  DT_B root split: %s <= %.4f  [%s]",
            _fn_b, _thr_b, "PROTOCOL" if _is_prot else "standard",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.6 -- ElasticNet Standardised Coefficients
    # ═══════════════════════════════════════════════════════════════════════
    if "elasticnet" in track_a_models and en_model is not None:
        logger.info("-" * 50)
        logger.info("Task 10.6 -- ElasticNet Standardised Coefficients")
        logger.info("-" * 50)
        plot_elasticnet_coefficients(
            en_model, features_linear,
            save_path=FIG_DIR / "phase10_elasticnet_coefficients.png",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.8.A -- Cross-Model Feature Ranking (Track A)
    # ═══════════════════════════════════════════════════════════════════════
    rho_df:     pd.DataFrame | None = None
    ranking_df: pd.DataFrame | None = None

    if len(perm_imp) >= 2:
        logger.info("-" * 50)
        logger.info("Task 10.8.A -- Cross-Model Feature Ranking (Track A)")
        logger.info("-" * 50)

        linear_set   = set(features_linear)
        tree_set     = set(FEATURES_TREES)
        common_feats = sorted(linear_set & tree_set)
        logger.info("  Common features for ranking: %d", len(common_feats))

        def _rank_df(imp_df: pd.DataFrame, common: list[str]) -> pd.Series:
            sub   = imp_df[imp_df["feature"].isin(common)].sort_values("importance_mean", ascending=False).reset_index(drop=True)
            ranks = {row["feature"]: rank + 1 for rank, (_, row) in enumerate(sub.iterrows())}
            return pd.Series({f: ranks.get(f, float("nan")) for f in common})

        ranking_dict = {mname: _rank_df(df_imp, common_feats) for mname, df_imp in perm_imp.items()}
        ranking_df   = pd.DataFrame(ranking_dict, index=common_feats)
        ranking_df.index.name = "feature"
        ranking_df.to_csv(TABLE_DIR / "phase10_feature_ranking_table.csv")
        logger.info("  Saved phase10_feature_ranking_table.csv")

        plot_feature_ranking_heatmap(
            ranking_df.reset_index(),
            save_path=FIG_DIR / "phase10_feature_ranking_heatmap.png",
        )

        rho_df = _spearman_matrix(ranking_df)
        rho_df.to_csv(TABLE_DIR / "phase10_spearman_correlation.csv")
        logger.info("  Saved phase10_spearman_correlation.csv")
        plot_spearman_heatmap(rho_df, save_path=FIG_DIR / "phase10_spearman_correlation_heatmap.png")

        _cols = rho_df.columns.tolist()
        _best_rho, _best_pair = -2.0, ("", "")
        for _i, _c1 in enumerate(_cols):
            for _j, _c2 in enumerate(_cols):
                if _i >= _j:
                    continue
                _v = float(rho_df.loc[_c1, _c2])
                if not np.isnan(_v) and _v > _best_rho:
                    _best_rho, _best_pair = _v, (_c1, _c2)
        logger.info("  Most similar pair: %s vs %s  rho=%.3f", _best_pair[0], _best_pair[1], _best_rho)

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.8.B -- Cross-Model Feature Ranking (Track B)
    # ═══════════════════════════════════════════════════════════════════════
    rho_b_df:     pd.DataFrame | None = None
    ranking_b_df: pd.DataFrame | None = None

    if len(perm_imp_b) >= 2:
        logger.info("-" * 50)
        logger.info("Task 10.8.B -- Cross-Model Feature Ranking (Track B)")
        logger.info("-" * 50)

        def _rank_b(imp_df: pd.DataFrame) -> pd.Series:
            sub   = imp_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
            ranks = {row["feature"]: rank + 1 for rank, (_, row) in enumerate(sub.iterrows())}
            return pd.Series({f: ranks.get(f, float("nan")) for f in features_b})

        ranking_b_df = pd.DataFrame(
            {b_name: _rank_b(df_imp) for b_name, df_imp in perm_imp_b.items()},
            index=features_b,
        )
        ranking_b_df.index.name = "feature"
        ranking_b_df.to_csv(TABLE_DIR / "phase10_feature_ranking_table_b.csv")
        logger.info("  Saved phase10_feature_ranking_table_b.csv  (%d features)", len(features_b))

        # Heatmap -- top 30 by mean rank to keep plot readable
        mean_rank_b = ranking_b_df.mean(axis=1).sort_values()
        top30_b     = mean_rank_b.head(30).index.tolist()
        plot_feature_ranking_heatmap(
            ranking_b_df.loc[top30_b].reset_index(),
            save_path=FIG_DIR / "phase10_feature_ranking_heatmap_b.png",
        )

        rho_b_df = _spearman_matrix(ranking_b_df)
        rho_b_df.to_csv(TABLE_DIR / "phase10_spearman_correlation_b.csv")
        logger.info("  Saved phase10_spearman_correlation_b.csv")
        plot_spearman_heatmap(rho_b_df, save_path=FIG_DIR / "phase10_spearman_correlation_heatmap_b.png")

        _cols_b = rho_b_df.columns.tolist()
        _best_rho_b, _best_pair_b = -2.0, ("", "")
        for _i, _c1 in enumerate(_cols_b):
            for _j, _c2 in enumerate(_cols_b):
                if _i >= _j:
                    continue
                _v = float(rho_b_df.loc[_c1, _c2])
                if not np.isnan(_v) and _v > _best_rho_b:
                    _best_rho_b, _best_pair_b = _v, (_c1, _c2)
        logger.info(
            "  Most similar Track B pair: %s vs %s  rho=%.3f",
            _best_pair_b[0], _best_pair_b[1], _best_rho_b,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.9.A -- Accuracy-Interpretability Tradeoff (Track A)
    # ═══════════════════════════════════════════════════════════════════════
    if track_a_models:
        logger.info("-" * 50)
        logger.info("Task 10.9.A -- Accuracy-Interpretability Tradeoff (Track A)")
        logger.info("-" * 50)

        model_tradeoff_data = [
            {"name": "ElasticNet", "rmse": 20.40, "interp_score": 5,
             "color": MODEL_COLOURS.get("ElasticNet", "#aec7e8"),
             "training_seconds": TRAINING_SECONDS["ElasticNet"]},
            {"name": "DT",         "rmse": 14.44, "interp_score": 4,
             "color": MODEL_COLOURS.get("DT",        "#ffbb78"),
             "training_seconds": TRAINING_SECONDS["DT"]},
            {"name": "ANN",        "rmse": 15.56, "interp_score": 1,
             "color": MODEL_COLOURS.get("ANN",       "#f7b6d2"),
             "training_seconds": TRAINING_SECONDS["ANN"]},
            {"name": "RF",         "rmse": 11.54, "interp_score": 2,
             "color": MODEL_COLOURS.get("RF",        "#98df8a"),
             "training_seconds": TRAINING_SECONDS["RF"]},
            {"name": "XGBoost",    "rmse": 11.59, "interp_score": 2,
             "color": MODEL_COLOURS.get("XGBoost",   "#ff9896"),
             "training_seconds": TRAINING_SECONDS["XGBoost"]},
            {"name": "LightGBM",   "rmse": 11.65, "interp_score": 2,
             "color": MODEL_COLOURS.get("LightGBM",  "#c5b0d5"),
             "training_seconds": TRAINING_SECONDS["LightGBM"]},
        ]
        plot_accuracy_interpretability_tradeoff(
            model_tradeoff_data,
            baseline_rmse=20.32,
            save_path=FIG_DIR / "phase10_accuracy_interpretability_tradeoff.png",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.9.B -- Accuracy-Interpretability Tradeoff (Track B)
    # ═══════════════════════════════════════════════════════════════════════
    b_rmse: dict[str, float] | None = None

    if models_b:
        logger.info("-" * 50)
        logger.info("Task 10.9.B -- Accuracy-Interpretability Tradeoff (Track B)")
        logger.info("-" * 50)

        b_rmse = {"XGBoost_B": 8.42, "DT_B": 8.753, "RF_B": 6.731}  # Phase 7.1 fallbacks
        _metrics_csv = TABLE_DIR / "phase9_test_metrics.csv"
        if _metrics_csv.exists():
            _met = pd.read_csv(_metrics_csv)
            for b_name in models_b:
                _row = _met[_met["Model"] == b_name]
                if not _row.empty:
                    b_rmse[b_name] = float(_row["RMSE"].iloc[0])
                    logger.info("  %s RMSE from Phase 9: %.3f", b_name, b_rmse[b_name])

        b_interp = {"XGBoost_B": 2, "DT_B": 4, "RF_B": 2}
        model_b_tradeoff = [
            {
                "name":             b_name,
                "rmse":             b_rmse.get(b_name, 10.0),
                "interp_score":     b_interp.get(b_name, 2),
                "color":            MODEL_COLOURS.get(b_name, _B_COLOUR_DEFAULTS.get(b_name, "#888888")),
                "training_seconds": 60,
            }
            for b_name in models_b
        ]
        if model_b_tradeoff:
            plot_accuracy_interpretability_tradeoff(
                model_b_tradeoff,
                baseline_rmse=b_rmse.get("DT_B", 8.753),
                save_path=FIG_DIR / "phase10_accuracy_interpretability_tradeoff_b.png",
            )

    # ═══════════════════════════════════════════════════════════════════════
    # Task 10.10 -- Consolidated Report
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("-" * 50)
    logger.info("Task 10.10 -- Writing Consolidated Report")
    logger.info("-" * 50)

    report_path = _write_report(
        perm_imp         = perm_imp,
        shap_mean_abs    = shap_mean_abs,
        dt_model         = dt_model,
        elasticnet_model = en_model,
        rho_df           = rho_df,
        ranking_df       = ranking_df,
        shap_b_df_all    = shap_mean_abs_b,
        features_b       = features_b,
        features_linear  = features_linear,
        cases            = report_cases,
        perm_imp_b       = perm_imp_b  or None,
        rho_b_df         = rho_b_df,
        dt_b_model       = dt_b_model,
        b_rmse           = b_rmse,
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
