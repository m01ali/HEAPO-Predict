"""
scripts/11_subgroup_analysis.py

Phase 11 — Subgroup and Bias Analysis
======================================

Tasks:
  11.0  Setup, load data, build master analysis table
  11.1  Subgroup label engineering
  11.2  Subgroup composition analysis
  11.3  Per-subgroup residual metrics (all Track A models)
  11.4  Treatment effect analysis (val set pre/post)
  11.5  Visualizations (bias heatmap, MAE bars, box plots, scatter, timeline, table)
  11.6  Statistical testing (Mann-Whitney U + Kruskal-Wallis + Bonferroni)
  11.7  Track B protocol subgroup analysis (XGBoost B)
  11.8  Consolidated report

Inputs : data/processed/test_full.parquet
         data/processed/val_full.parquet
         data/processed/test_protocol.parquet
         outputs/tables/phase9_test_predictions.parquet
         outputs/tables/phase9_val_predictions.parquet
         outputs/tables/phase9_test_predictions_b.parquet

Outputs: outputs/figures/phase11_*.png
         outputs/tables/phase11_*.csv
         outputs/tables/phase11_subgroup_report.txt
         outputs/logs/phase11_run.log
"""

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

from src.subgroup_analysis import (
    AGE_ORDER,
    AREA_ORDER,
    MODEL_COLOURS,
    MODEL_PRED_COLS,
    MODEL_RESID_COLS,
    build_subgroup_labels,
    compute_subgroup_metrics,
    kruskal_wallis,
    mannwhitney_pairwise,
    plot_bias_heatmap,
    plot_bias_vs_area,
    plot_composition_bar,
    plot_mae_grouped_bar,
    plot_residual_boxplots,
    plot_subgroup_rmse_table,
    plot_track_b_bias_heatmap,
    plot_track_b_residual_boxplot,
    plot_treatment_timeline,
    run_subgroup_metrics,
    write_report,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR  = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR   = PROJECT_ROOT / "outputs" / "figures"
LOG_DIR   = PROJECT_ROOT / "outputs" / "logs"

for d in (TABLE_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
log_path = LOG_DIR / "phase11_run.log"
fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=fmt,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Model display names (ordered by test RMSE, best first)
# ─────────────────────────────────────────────────────────────────────────────
MODELS_A = ["RF", "XGBoost", "LightGBM", "DT", "ANN", "ElasticNet"]

# Subgroup dimension map: display_name → column_name  (populated after label engineering)
SG_COLS_MAP = {
    "HP Type":             "sg_hp_type",
    "Building Type":       "sg_building_type",
    "Heat Distribution":   "sg_heat_dist",
    "PV System":           "sg_pv",
    "Living Area":         "sg_area",
    "Group":               "sg_group",
    "Intervention Status": "sg_intervention",
    "EV Ownership":        "sg_ev",
    "Test Month":          "sg_month",
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 11.0 — Load data and build master analysis table
# ─────────────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Task 11.0 — Loading data")
logger.info("=" * 60)
t0 = time.time()

# --- Track A test predictions (already includes Group, has_pv) ---
preds = pd.read_parquet(TABLE_DIR / "phase9_test_predictions.parquet")
logger.info("Test predictions loaded: %s", preds.shape)

# --- Test features needed for subgroup labels not in preds file ---
FEAT_COLS = [
    "Household_ID", "Date",
    "hp_type_air_source", "hp_type_ground_source", "hp_type_unknown",
    "building_type_house", "building_type_apartment",
    "heat_dist_floor", "heat_dist_radiator", "heat_dist_both", "heat_dist_unknown",
    "has_ev", "has_dryer",
    "Survey_Building_LivingArea",
    "living_area_bucket",
    "post_intervention",
    "AffectsTimePoint",
]
feats = pd.read_parquet(DATA_DIR / "test_full.parquet", columns=FEAT_COLS)
logger.info("Test features loaded: %s", feats.shape)

# --- Merge predictions with feature columns ---
# preds already has: Household_ID, Date, kWh_received_Total, Group, has_pv,
#                    pred_rf, pred_xgb, pred_lgbm, pred_dt, pred_ann, pred_elasticnet
#                    residual_pred_* (actual - predicted)
df = preds.merge(feats, on=["Household_ID", "Date"], how="left")
logger.info("Master table shape after merge: %s", df.shape)

# Verify all prediction and residual columns are present
for m, col in {**MODEL_PRED_COLS, **MODEL_RESID_COLS}.items():
    if col not in df.columns:
        logger.error("Missing column: %s (%s)", col, m)
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Task 11.1 — Subgroup label engineering
# ─────────────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Task 11.1 — Building subgroup labels")
logger.info("=" * 60)

df = build_subgroup_labels(df)

# Spot-check label distributions
for col in ["sg_hp_type", "sg_pv", "sg_group", "sg_intervention", "sg_area"]:
    logger.info("%s: %s", col, dict(df[col].value_counts()))

# ─────────────────────────────────────────────────────────────────────────────
# Task 11.2 — Composition analysis
# ─────────────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Task 11.2 — Subgroup composition analysis")
logger.info("=" * 60)

composition_rows = []
for dim_name, sg_col in SG_COLS_MAP.items():
    for cat, sub in df.groupby(sg_col, dropna=False):
        cat_str = str(cat) if not pd.isna(cat) else "Unknown"
        composition_rows.append({
            "Dimension":    dim_name,
            "Category":     cat_str,
            "N_rows":       len(sub),
            "N_households": sub["Household_ID"].nunique(),
            "Pct_rows":     round(100 * len(sub) / len(df), 2),
            "Mean_kWh":     round(sub["kWh_received_Total"].mean(), 2),
        })
composition_df = pd.DataFrame(composition_rows)
composition_df.to_csv(TABLE_DIR / "phase11_subgroup_composition.csv", index=False)
logger.info("Composition table saved: %d rows", len(composition_df))

# Composition bar chart
plot_composition_bar(composition_df, FIG_DIR / "phase11_composition_bar.png")

# Composition bias: treatment vs. control cross-tabs
hh_df = df.groupby("Household_ID").agg(
    group     =("sg_group", "first"),
    hp_type   =("sg_hp_type", "first"),
    pv        =("sg_pv", "first"),
    ev        =("sg_ev", "first"),
    building  =("sg_building_type", "first"),
).reset_index()

ct_hp = pd.crosstab(hh_df["group"], hh_df["hp_type"], normalize="index").round(3) * 100
ct_pv = pd.crosstab(hh_df["group"], hh_df["pv"],      normalize="index").round(3) * 100
ct_ev = pd.crosstab(hh_df["group"], hh_df["ev"],      normalize="index").round(3) * 100
logger.info("Treatment/Control × HP type (%%):  \n%s", ct_hp.to_string())
logger.info("Treatment/Control × PV (%%):        \n%s", ct_pv.to_string())
logger.info("Treatment/Control × EV (%%):        \n%s", ct_ev.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Task 11.3 — Per-subgroup residual metrics
# ─────────────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Task 11.3 — Computing per-subgroup metrics")
logger.info("=" * 60)

metrics_df = run_subgroup_metrics(
    df, SG_COLS_MAP, MODEL_PRED_COLS, y_col="kWh_received_Total", min_n=30
)
logger.info("Subgroup metrics computed: %d (subgroup × model) combinations", len(metrics_df))

# Add overall test-set metrics as a reference row
overall_rows = []
for model_name, pred_col in MODEL_PRED_COLS.items():
    m = compute_subgroup_metrics(df, pred_col, "kWh_received_Total", min_n=1)
    if m:
        overall_rows.append({"Dimension": "Overall", "Category": "All", "Model": model_name, **m})
overall_df = pd.DataFrame(overall_rows)
metrics_df = pd.concat([metrics_df, overall_df], ignore_index=True)

metrics_df.to_csv(TABLE_DIR / "phase11_subgroup_metrics.csv", index=False)
logger.info("Subgroup metrics saved.")

# ─────────────────────────────────────────────────────────────────────────────
# Task 11.4 — Treatment effect analysis (validation set)
# ─────────────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Task 11.4 — Treatment effect analysis")
logger.info("=" * 60)

val_preds = pd.read_parquet(TABLE_DIR / "phase9_val_predictions.parquet")
VAL_FEAT_COLS = [
    "Household_ID", "Date", "post_intervention", "living_area_bucket",
    "Survey_Building_LivingArea", "hp_type_air_source", "hp_type_ground_source",
    "hp_type_unknown", "building_type_house", "building_type_apartment",
    "heat_dist_floor", "heat_dist_radiator", "heat_dist_both", "heat_dist_unknown",
    "has_ev", "has_dryer",
]
val_feats = pd.read_parquet(DATA_DIR / "val_full.parquet", columns=VAL_FEAT_COLS)
df_val = val_preds.merge(val_feats, on=["Household_ID", "Date"], how="left")
df_val = build_subgroup_labels(df_val)

# Filter to treatment households
treat_val = df_val[df_val["Group"] == "treatment"].copy()
logger.info("Treatment HHs in val set: %d unique HHs, %d rows",
            treat_val["Household_ID"].nunique(), len(treat_val))

treatment_stats: dict = {}

# Pre vs. post consumption
pre  = treat_val[treat_val["post_intervention"] == 0]
post = treat_val[treat_val["post_intervention"] == 1]
treatment_stats["Val set — Treatment HHs pre-visit (rows)"]  = f"{len(pre):,}  ({pre['Household_ID'].nunique()} HH)"
treatment_stats["Val set — Treatment HHs post-visit (rows)"] = f"{len(post):,}  ({post['Household_ID'].nunique()} HH)"
treatment_stats["Val set — Mean consumption pre-visit (kWh)"]  = f"{pre['kWh_received_Total'].mean():.2f}"
treatment_stats["Val set — Mean consumption post-visit (kWh)"] = f"{post['kWh_received_Total'].mean():.2f}"
actual_reduction = pre["kWh_received_Total"].mean() - post["kWh_received_Total"].mean()
treatment_stats["Val set — Actual consumption change pre→post (kWh)"] = f"{actual_reduction:+.2f}  (NOTE: confounded by seasonality)"

# Model-predicted reduction for RF and XGBoost
for m_name, pred_col in [("RF", "pred_rf"), ("XGBoost", "pred_xgb")]:
    if pred_col in treat_val.columns:
        pre_pred  = treat_val[treat_val["post_intervention"] == 0][pred_col].mean()
        post_pred = treat_val[treat_val["post_intervention"] == 1][pred_col].mean()
        model_reduction = pre_pred - post_pred
        treatment_stats[f"Val set — {m_name} predicted change pre→post (kWh)"] = f"{model_reduction:+.2f}"

# Test set treatment analysis (mostly post-visit)
treat_test = df[df["Group"] == "treatment"].copy()
treatment_stats["Test set — Treatment HHs pre-visit (rows)"]  = f"{(treat_test['post_intervention']==0).sum():,}  ({treat_test[treat_test['post_intervention']==0]['Household_ID'].nunique()} HH)"
treatment_stats["Test set — Treatment HHs post-visit (rows)"] = f"{(treat_test['post_intervention']==1).sum():,}  ({treat_test[treat_test['post_intervention']==1]['Household_ID'].nunique()} HH)"

# Residual bias: pre vs post in test set (RF)
if len(treat_test[treat_test["post_intervention"] == 0]) >= 30:
    pre_bias  = treat_test[treat_test["post_intervention"] == 0]["residual_pred_rf"].mean()
    post_bias = treat_test[treat_test["post_intervention"] == 1]["residual_pred_rf"].mean()
    treatment_stats["Test set — RF mean bias pre-visit (kWh)"]  = f"{pre_bias:+.2f}"
    treatment_stats["Test set — RF mean bias post-visit (kWh)"] = f"{post_bias:+.2f}"
else:
    treatment_stats["Test set — RF mean bias pre-visit (kWh)"] = "n<30 — not computed"
    post_bias = treat_test[treat_test["post_intervention"] == 1]["residual_pred_rf"].mean()
    treatment_stats["Test set — RF mean bias post-visit (kWh)"] = f"{post_bias:+.2f}"

for key, val in treatment_stats.items():
    logger.info("  %s: %s", key, val)

# Treatment timeline figure (val set)
plot_treatment_timeline(
    df_val, "pred_rf", "RF",
    FIG_DIR / "phase11_treatment_effect_timeline.png",
    n_hh=5,
)

# ─────────────────────────────────────────────────────────────────────────────
# Task 11.5 — Visualizations
# ─────────────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Task 11.5 — Generating visualizations")
logger.info("=" * 60)

# 11.5.1 — Bias heatmap (all models)
plot_bias_heatmap(
    metrics_df, MODELS_A,
    FIG_DIR / "phase11_bias_heatmap.png",
)

# 11.5.2 — MAE grouped bar charts (top 3 models)
overall_mae = {
    m: metrics_df[(metrics_df["Model"] == m) & (metrics_df["Dimension"] == "Overall")]["MAE"].values[0]
    for m in MODELS_A
    if not metrics_df[(metrics_df["Model"] == m) & (metrics_df["Dimension"] == "Overall")].empty
}

MAE_BAR_DIMS = ["HP Type", "PV System", "Building Type", "Living Area", "Group"]
for model in ["RF", "XGBoost", "LightGBM"]:
    mae = overall_mae.get(model, 7.47)
    plot_mae_grouped_bar(
        metrics_df, model, MAE_BAR_DIMS,
        FIG_DIR / f"phase11_mae_by_subgroup_{model.lower().replace(' ','_')}.png",
        overall_mae=mae,
    )

# 11.5.3 — Residual box plots by subgroup
BOXPLOT_MODELS = {m: MODEL_RESID_COLS[m] for m in ["RF", "XGBoost", "LightGBM", "DT"]}

# HP type
plot_residual_boxplots(
    df, "sg_hp_type", "HP Type", BOXPLOT_MODELS,
    FIG_DIR / "phase11_residuals_hp_type.png",
    cat_order=["Air-Source", "Ground-Source", "Unknown"],
)
# Building type
plot_residual_boxplots(
    df, "sg_building_type", "Building Type", BOXPLOT_MODELS,
    FIG_DIR / "phase11_residuals_building_type.png",
    cat_order=["House", "Apartment", "Unknown"],
)
# PV system
plot_residual_boxplots(
    df, "sg_pv", "PV System", BOXPLOT_MODELS,
    FIG_DIR / "phase11_residuals_pv.png",
    cat_order=["With PV", "Without PV"],
)
# Living area buckets
plot_residual_boxplots(
    df, "sg_area", "Living Area", BOXPLOT_MODELS,
    FIG_DIR / "phase11_residuals_area.png",
    cat_order=AREA_ORDER,
    exclude_cats=["Unknown"],
)
# Intervention status
plot_residual_boxplots(
    df, "sg_intervention", "Intervention Status", BOXPLOT_MODELS,
    FIG_DIR / "phase11_residuals_intervention.png",
    cat_order=["Control (no visit)", "Treatment (pre-visit)", "Treatment (post-visit)"],
    exclude_cats=[],
)
# EV ownership
plot_residual_boxplots(
    df, "sg_ev", "EV Ownership", BOXPLOT_MODELS,
    FIG_DIR / "phase11_residuals_ev.png",
    cat_order=["With EV", "Without EV"],
)
# Heat distribution
plot_residual_boxplots(
    df, "sg_heat_dist", "Heat Distribution", BOXPLOT_MODELS,
    FIG_DIR / "phase11_residuals_heat_dist.png",
    cat_order=["Floor", "Radiator", "Both", "Unknown"],
)
# Test month
plot_residual_boxplots(
    df, "sg_month", "Test Month", BOXPLOT_MODELS,
    FIG_DIR / "phase11_residuals_month.png",
    cat_order=["Dec", "Jan", "Feb", "Mar"],
    exclude_cats=[],
)

# 11.5.4 — Bias vs. living area scatter (RF)
plot_bias_vs_area(
    df, "residual_pred_rf", "RF",
    FIG_DIR / "phase11_bias_vs_area_rf.png",
)

# 11.5.5 — Subgroup RMSE styled table
plot_subgroup_rmse_table(
    metrics_df, MODELS_A,
    FIG_DIR / "phase11_subgroup_rmse_table.png",
)

logger.info("Task 11.5 complete — all figures saved.")

# ─────────────────────────────────────────────────────────────────────────────
# Task 11.6 — Statistical testing
# ─────────────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Task 11.6 — Statistical testing")
logger.info("=" * 60)

# Mann-Whitney U — binary subgroup pairs, for the top 3 models
BINARY_PAIRS = [
    ("sg_hp_type",      "Air-Source",             "Ground-Source"),
    ("sg_building_type","House",                   "Apartment"),
    ("sg_pv",           "With PV",                 "Without PV"),
    ("sg_group",        "Treatment",               "Control"),
    ("sg_ev",           "With EV",                 "Without EV"),
    ("sg_intervention", "Treatment (pre-visit)",   "Treatment (post-visit)"),
    ("sg_intervention", "Control (no visit)",      "Treatment (post-visit)"),
    ("sg_heat_dist",    "Floor",                   "Radiator"),
]

mw_rows = []
for sg_col, cat_a, cat_b in BINARY_PAIRS:
    for model_name in ["RF", "XGBoost", "LightGBM"]:
        resid_col = MODEL_RESID_COLS[model_name]
        result = mannwhitney_pairwise(df, sg_col, cat_a, cat_b, resid_col, min_n=30)
        if result is not None:
            mw_rows.append({"Model": model_name, **result})

mw_df = pd.DataFrame(mw_rows)

# Bonferroni correction
if not mw_df.empty:
    n_tests = len(mw_df)
    mw_df["p_bonferroni"] = (mw_df["p_value"] * n_tests).clip(upper=1.0)
    mw_df["significant"]  = mw_df["p_bonferroni"] < 0.05
    logger.info("MW tests: %d total, %d significant (Bonferroni p<0.05)",
                n_tests, mw_df["significant"].sum())
    for _, row in mw_df[mw_df["significant"]].iterrows():
        logger.info("  SIGNIFICANT: %s  %s vs %s  [%s]  Δmedian=%.2f  p_bonf=%.3e",
                    row["sg_col"], row["cat_a"], row["cat_b"], row["Model"],
                    row["delta_median"], row["p_bonferroni"])
else:
    mw_df = pd.DataFrame(columns=["sg_col", "cat_a", "cat_b", "Model",
                                   "n_a", "n_b", "median_a", "median_b",
                                   "delta_median", "stat", "p_value",
                                   "p_bonferroni", "significant"])

mw_df.to_csv(TABLE_DIR / "phase11_mannwhitney_results.csv", index=False)
logger.info("Mann-Whitney results saved.")

# Kruskal-Wallis — multi-category subgroups, RF residuals
kw_results = []
for sg_col in ["sg_hp_type", "sg_heat_dist", "sg_area", "sg_month"]:
    res = kruskal_wallis(df, sg_col, "residual_pred_rf", min_n=30)
    if res is not None:
        kw_results.append(res)
        logger.info("Kruskal-Wallis %s: H=%.2f  p=%.3e", sg_col, res["stat"], res["p_value"])

# ─────────────────────────────────────────────────────────────────────────────
# Task 11.7 — Track B protocol subgroup analysis
# ─────────────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Task 11.7 — Track B protocol subgroup analysis")
logger.info("=" * 60)

preds_b = pd.read_parquet(TABLE_DIR / "phase9_test_predictions_b.parquet")
PROTO_COLS = [
    "Household_ID", "Date",
    "building_age_bucket",
    "HeatPump_Installation_CorrectlyPlanned",
    "heating_curve_too_high",
    "heating_limit_too_high",
    "night_setback_active_before",
    "post_intervention",
    "Survey_Building_LivingArea",
    "living_area_bucket",
    "hp_type_air_source", "hp_type_ground_source", "hp_type_unknown",
]
available_proto = [c for c in PROTO_COLS if c in pd.read_parquet(
    DATA_DIR / "test_protocol.parquet", columns=["Household_ID"]).columns
    or True]
# Load only existing columns
proto_all = pd.read_parquet(DATA_DIR / "test_protocol.parquet")
available_proto = [c for c in PROTO_COLS if c in proto_all.columns]
proto_sub = proto_all[available_proto]

df_b = preds_b.merge(proto_sub, on=["Household_ID", "Date"], how="left")
df_b["residual_xgb_b"] = df_b["kWh_received_Total"] - df_b["pred_xgb_b"]
logger.info("Track B merged table: %s", df_b.shape)

# Protocol subgroup definitions
PROTOCOL_SG_MAP = {}
if "building_age_bucket" in df_b.columns:
    PROTOCOL_SG_MAP["Building Age Bucket"] = "building_age_bucket"
if "HeatPump_Installation_CorrectlyPlanned" in df_b.columns:
    PROTOCOL_SG_MAP["HP Correctly Planned"] = "HeatPump_Installation_CorrectlyPlanned"
if "heating_curve_too_high" in df_b.columns:
    PROTOCOL_SG_MAP["Heating Curve Too High"] = "heating_curve_too_high"
if "heating_limit_too_high" in df_b.columns:
    PROTOCOL_SG_MAP["Heating Limit Too High"] = "heating_limit_too_high"
if "night_setback_active_before" in df_b.columns:
    PROTOCOL_SG_MAP["Night Setback Active (before)"] = "night_setback_active_before"

from src.subgroup_analysis import run_subgroup_metrics as _run_sg
track_b_metrics = _run_sg(
    df_b, PROTOCOL_SG_MAP,
    pred_cols={"XGBoost B": "pred_xgb_b"},
    y_col="kWh_received_Total",
    min_n=10,
)
track_b_metrics.to_csv(TABLE_DIR / "phase11_track_b_subgroup_metrics.csv", index=False)
logger.info("Track B metrics saved: %d rows", len(track_b_metrics))

# Figures
plot_track_b_bias_heatmap(
    track_b_metrics,
    FIG_DIR / "phase11_track_b_bias_heatmap.png",
)

if "building_age_bucket" in df_b.columns:
    # Map numeric age buckets to readable labels if needed
    df_b["building_age_bucket"] = df_b["building_age_bucket"].astype(str)
    plot_track_b_residual_boxplot(
        df_b, "building_age_bucket", "Building Age Bucket",
        "residual_xgb_b",
        FIG_DIR / "phase11_track_b_residuals_building_age.png",
        cat_order=AGE_ORDER,
    )

if "HeatPump_Installation_CorrectlyPlanned" in df_b.columns:
    df_b["hp_sizing_label"] = df_b["HeatPump_Installation_CorrectlyPlanned"].map(
        {1.0: "Correctly Planned", 0.0: "Incorrectly Planned", True: "Correctly Planned", False: "Incorrectly Planned"}
    ).fillna("Unknown")
    plot_track_b_residual_boxplot(
        df_b, "hp_sizing_label", "HP Sizing",
        "residual_xgb_b",
        FIG_DIR / "phase11_track_b_residuals_hp_sizing.png",
        cat_order=["Correctly Planned", "Incorrectly Planned"],
    )

if "heating_curve_too_high" in df_b.columns:
    df_b["heating_curve_label"] = df_b["heating_curve_too_high"].map(
        {1.0: "Too High", 0.0: "Appropriate", 1: "Too High", 0: "Appropriate"}
    ).fillna("Unknown")
    plot_track_b_residual_boxplot(
        df_b, "heating_curve_label", "Heating Curve Setting",
        "residual_xgb_b",
        FIG_DIR / "phase11_track_b_residuals_heating_curve.png",
        cat_order=["Too High", "Appropriate"],
    )

if "night_setback_active_before" in df_b.columns:
    df_b["night_setback_label"] = df_b["night_setback_active_before"].map(
        {1.0: "Active", 0.0: "Not Active", 1: "Active", 0: "Not Active"}
    ).fillna("Unknown")
    plot_track_b_residual_boxplot(
        df_b, "night_setback_label", "Night Setback (before visit)",
        "residual_xgb_b",
        FIG_DIR / "phase11_track_b_residuals_night_setback.png",
        cat_order=["Active", "Not Active"],
    )

# HP issue flag analysis (Mann-Whitney for Track B)
issue_flag_stats = {}
for flag_col, flag_label in [
    ("heating_curve_too_high", "Heating curve too high"),
    ("heating_limit_too_high", "Heating limit too high"),
    ("night_setback_active_before", "Night setback active"),
]:
    if flag_col not in df_b.columns:
        continue
    yes = df_b[df_b[flag_col] == 1]["residual_xgb_b"].dropna()
    no  = df_b[df_b[flag_col] == 0]["residual_xgb_b"].dropna()
    if len(yes) >= 10 and len(no) >= 10:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(yes, no, alternative="two-sided")
        issue_flag_stats[flag_label] = {
            "n_flagged": len(yes), "n_not_flagged": len(no),
            "mean_bias_flagged": float(yes.mean()),
            "mean_bias_not_flagged": float(no.mean()),
            "p_value": float(p),
        }
        logger.info("HP issue [%s]: mean bias (flagged)=%.2f vs (not)=%.2f  p=%.3e",
                    flag_label, yes.mean(), no.mean(), p)

# ─────────────────────────────────────────────────────────────────────────────
# Task 11.8 — Consolidated report
# ─────────────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Task 11.8 — Writing consolidated report")
logger.info("=" * 60)

write_report(
    metrics_df=metrics_df,
    composition_df=composition_df,
    mw_df=mw_df,
    kw_results=kw_results,
    treatment_stats=treatment_stats,
    track_b_metrics=track_b_metrics,
    save_path=TABLE_DIR / "phase11_subgroup_report.txt",
    n_test_rows=len(df),
    n_test_hh=df["Household_ID"].nunique(),
)

elapsed = time.time() - t0
logger.info("=" * 60)
logger.info("Phase 11 complete — %.1f s", elapsed)
logger.info("Outputs written to:")
logger.info("  Figures : %s", FIG_DIR)
logger.info("  Tables  : %s", TABLE_DIR)
logger.info("  Log     : %s", log_path)
logger.info("=" * 60)
