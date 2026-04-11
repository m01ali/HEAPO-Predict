"""
scripts/07_model_training.py

Phase 7 — Model Training orchestration.

Trains all models defined in src/models.py across Track A (full sample)
and Track B (protocol-enriched) and saves artifacts, predictions, and a
training report.

Dataset assignment:
  Track A (*_full.parquet):
    - Baselines, OLS, Ridge, Lasso, ElasticNet, DT, RF, XGBoost, LightGBM, ANN
    - Trees use FEATURES_TREES (45 cols), raw kWh target
    - Linear/ANN use FEATURES_LINEAR (30 cols, scaled), log1p kWh target
  Track B (*_protocol.parquet):
    - XGBoost_B (75 features), Ridge_B (46 features, scaled)
    - Same architectures, richer feature set, treatment-only households

Usage:
    source .venv/bin/activate
    python scripts/07_model_training.py
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor as _DT  # used in CV lambdas
from src.data_loader import load_config
from src.models import (
    compute_metrics,
    cv_evaluate,
    fit_decision_tree,
    fit_hdd_baseline,
    fit_lightgbm,
    fit_linear_variants,
    fit_random_forest,
    fit_xgboost,
    predict_hh_mean,
    predict_overall_mean,
)
from src.ann import ANN, fit_ann   # sklearn-based MLP, no PyTorch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "outputs" / "models"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
LOG_DIR = ROOT / "outputs" / "logs"

for d in (MODEL_DIR, TABLE_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "phase7_run.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config + seeds
# ---------------------------------------------------------------------------
cfg = load_config(str(ROOT / "config" / "params.yaml"))
RANDOM_SEED = cfg["modeling"]["random_seed"]
EARLY_STOP = cfg["modeling"]["xgboost_early_stopping_rounds"]
ANN_PATIENCE = cfg["modeling"]["ann_early_stopping_patience"]
MAPE_FLOOR = cfg["evaluation"]["mape_floor_kwh"]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logger.info("=" * 70)
logger.info("HEAPO-Predict  Phase 7 — Model Training")
logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Random seed: {RANDOM_SEED}  |  XGB early stop: {EARLY_STOP}  |  ANN patience: {ANN_PATIENCE}")
logger.info("=" * 70)

# ---------------------------------------------------------------------------
# Task 7.0 — Load data + pre-training assertions
# ---------------------------------------------------------------------------
logger.info("\n--- Task 7.0  Load data & assertions ---")

EXPECTED_SHAPES = {
    "train_full":     (646_258, 89),
    "val_full":       (153_594, 88),
    "test_full":      (74_368,  88),
    "train_protocol": (60_636,  180),
    "val_protocol":   (11_281,  179),
    "test_protocol":  (5_475,   179),
}

splits: dict[str, pd.DataFrame] = {}
for name, expected in EXPECTED_SHAPES.items():
    path = DATA_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    assert df.shape == expected, (
        f"Shape mismatch for {name}: got {df.shape}, expected {expected}"
    )
    splits[name] = df
    logger.info(f"  {name:20s}: {df.shape}  ✓")

train_full = splits["train_full"]
val_full   = splits["val_full"]
test_full  = splits["test_full"]
train_prot = splits["train_protocol"]
val_prot   = splits["val_protocol"]
test_prot  = splits["test_protocol"]

# Feature lists
with open(TABLE_DIR / "phase6_feature_lists.json") as f:
    feat = json.load(f)

FEATURES_TREES    = feat["FEATURES_TREES"]     # 45
FEATURES_LINEAR   = feat["FEATURES_LINEAR"]    # 30
FEATURES_TREES_B  = feat["FEATURES_TREES_B"]   # 75
FEATURES_LINEAR_B = feat["FEATURES_LINEAR_B"]  # 46
TARGET_RAW        = feat["TARGET_RAW"]          # "kWh_received_Total"
TARGET_LOG        = feat["TARGET_LOG"]          # "kWh_log1p"

# Scalers — loaded, never refitted
scaler_A = joblib.load(MODEL_DIR / "scaler_linear_A.pkl")
scaler_B = joblib.load(MODEL_DIR / "scaler_linear_B.pkl")

# Assertions
assert train_full[FEATURES_TREES].isnull().sum().sum() == 0,  "Nulls in FEATURES_TREES (train)"
assert train_full[FEATURES_LINEAR].isnull().sum().sum() == 0, "Nulls in FEATURES_LINEAR (train)"
assert val_full[FEATURES_TREES].isnull().sum().sum() == 0,    "Nulls in FEATURES_TREES (val)"
assert val_full[FEATURES_LINEAR].isnull().sum().sum() == 0,   "Nulls in FEATURES_LINEAR (val)"
assert train_full[TARGET_RAW].min() > 0, "Non-positive target in training set"
assert set(train_full["cv_fold"].unique()) == {0, 1, 2, 3, 4}, "cv_fold values not {0,1,2,3,4}"
assert scaler_A.n_features_in_ == len(FEATURES_LINEAR), (
    f"scaler_A expects {scaler_A.n_features_in_} features, FEATURES_LINEAR has {len(FEATURES_LINEAR)}"
)
assert scaler_B.n_features_in_ == len(FEATURES_LINEAR_B), (
    f"scaler_B expects {scaler_B.n_features_in_} features, FEATURES_LINEAR_B has {len(FEATURES_LINEAR_B)}"
)
actual_hh_B = train_prot["Household_ID"].nunique()
assert actual_hh_B == 109, f"Track B train HH count: expected 109, got {actual_hh_B}"
logger.info("All pre-training assertions passed ✓")

# ---------------------------------------------------------------------------
# Feature matrices
# ---------------------------------------------------------------------------
logger.info("\n--- Building feature matrices ---")

# Track A — trees (raw target)
X_train_trees  = train_full[FEATURES_TREES].values
X_val_trees    = val_full[FEATURES_TREES].values
X_test_trees   = test_full[FEATURES_TREES].values
y_train_raw    = train_full[TARGET_RAW].values
y_val_raw      = val_full[TARGET_RAW].values
y_test_raw     = test_full[TARGET_RAW].values

# Track A — linear (scaled, log target)
X_train_lin = scaler_A.transform(train_full[FEATURES_LINEAR].values)
X_val_lin   = scaler_A.transform(val_full[FEATURES_LINEAR].values)
X_test_lin  = scaler_A.transform(test_full[FEATURES_LINEAR].values)
y_train_log = train_full[TARGET_LOG].values
y_val_log   = val_full[TARGET_LOG].values

# CV folds (aligned to train_full)
cv_folds = train_full["cv_fold"].values

# Track B — trees
X_train_trees_B  = train_prot[FEATURES_TREES_B].values
X_val_trees_B    = val_prot[FEATURES_TREES_B].values
y_train_raw_B    = train_prot[TARGET_RAW].values
y_val_raw_B      = val_prot[TARGET_RAW].values

# Track B — linear (scaled, log target)
X_train_lin_B = scaler_B.transform(train_prot[FEATURES_LINEAR_B].values)
X_val_lin_B   = scaler_B.transform(val_prot[FEATURES_LINEAR_B].values)
y_train_log_B = train_prot[TARGET_LOG].values
y_val_log_B   = val_prot[TARGET_LOG].values

logger.info(
    f"Track A  train: {X_train_trees.shape[0]:,} rows  "
    f"trees={X_train_trees.shape[1]}  linear={X_train_lin.shape[1]}"
)
logger.info(
    f"Track B  train: {X_train_trees_B.shape[0]:,} rows  "
    f"trees={X_train_trees_B.shape[1]}  linear={X_train_lin_B.shape[1]}"
)

# ---------------------------------------------------------------------------
# Collector for all validation predictions
# ---------------------------------------------------------------------------
val_preds = val_full[
    ["Household_ID", "Timestamp", TARGET_RAW, "is_heating_season", "AffectsTimePoint"]
].copy()

# Metrics list (dicts, one per model)
all_metrics: list[dict] = []
training_times: dict[str, float] = {}

# ===========================================================================
# Task 7.1 — Baselines
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 7.1  Baseline predictors")
logger.info("=" * 60)

# 7.1.1 Overall mean
t0 = time.perf_counter()
pred_val_overall = predict_overall_mean(y_train_raw, len(y_val_raw))
pred_test_overall = np.full(len(y_test_raw), float(y_train_raw.mean()))
training_times["baseline_overall_mean"] = time.perf_counter() - t0

m = compute_metrics(y_val_raw, pred_val_overall, name="Baseline: overall mean", floor=MAPE_FLOOR)
all_metrics.append(m)
val_preds["pred_overall_mean"] = pred_val_overall

# 7.1.2 Per-household mean
t0 = time.perf_counter()
pred_val_hh = predict_hh_mean(train_full, val_full, TARGET_RAW)
pred_test_hh = predict_hh_mean(train_full, test_full, TARGET_RAW)
training_times["baseline_hh_mean"] = time.perf_counter() - t0

m = compute_metrics(y_val_raw, pred_val_hh, name="Baseline: per-HH mean", floor=MAPE_FLOOR)
all_metrics.append(m)
val_preds["pred_hh_mean"] = pred_val_hh

# Save HH mean lookup table
hh_means_df = (
    train_full.groupby("Household_ID")[TARGET_RAW]
    .mean()
    .reset_index()
    .rename(columns={TARGET_RAW: "train_mean_kwh"})
)
hh_means_df.to_parquet(MODEL_DIR / "baseline_hh_means.parquet", index=False)
logger.info(f"Saved HH mean lookup: {MODEL_DIR / 'baseline_hh_means.parquet'}")

# 7.1.3 HDD-proportional baseline
t0 = time.perf_counter()
hdd_model = fit_hdd_baseline(train_full, TARGET_RAW)
pred_val_hdd = np.clip(
    hdd_model.predict(val_full[["HDD_SIA_daily"]].values), 0, None
)
pred_test_hdd = np.clip(
    hdd_model.predict(test_full[["HDD_SIA_daily"]].values), 0, None
)
training_times["baseline_hdd_linear"] = time.perf_counter() - t0

m = compute_metrics(y_val_raw, pred_val_hdd, name="Baseline: HDD-linear", floor=MAPE_FLOOR)
all_metrics.append(m)
val_preds["pred_hdd_linear"] = pred_val_hdd

joblib.dump(hdd_model, MODEL_DIR / "baseline_hdd_linear.pkl")
logger.info(f"Saved {MODEL_DIR / 'baseline_hdd_linear.pkl'}")

# Test metrics for baselines (baselines have no hyperparameters)
logger.info("\nBaseline TEST-set metrics (for anchoring Phase 9 table):")
compute_metrics(y_test_raw, pred_test_overall, name="[TEST] Baseline: overall mean", floor=MAPE_FLOOR)
compute_metrics(y_test_raw, pred_test_hh,      name="[TEST] Baseline: per-HH mean",  floor=MAPE_FLOOR)
compute_metrics(y_test_raw, pred_test_hdd,     name="[TEST] Baseline: HDD-linear",   floor=MAPE_FLOOR)

# ===========================================================================
# Task 7.2 — Linear Regression Variants
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 7.2  Linear Regression Variants (OLS / Ridge / Lasso / ElasticNet)")
logger.info("=" * 60)

t0 = time.perf_counter()
linear_models = fit_linear_variants(
    X_train_lin, y_train_log, FEATURES_LINEAR, random_state=RANDOM_SEED
)
training_times["linear_variants"] = time.perf_counter() - t0

for model_name, estimator in linear_models.items():
    pred_log = estimator.predict(X_val_lin)
    pred_kwh = np.expm1(pred_log).clip(0)
    m = compute_metrics(y_val_raw, pred_kwh, name=model_name, floor=MAPE_FLOOR)
    all_metrics.append(m)
    val_preds[f"pred_{model_name.lower()}"] = pred_kwh

    pkl_path = MODEL_DIR / f"model_{model_name.lower()}.pkl"
    joblib.dump(estimator, pkl_path)
    logger.info(f"Saved {pkl_path}")

# 5-fold CV sanity check for Ridge
logger.info("\nLinear CV check (Ridge):")
cv_evaluate(
    model_fn=lambda Xtr, ytr: Ridge(alpha=1.0, random_state=RANDOM_SEED).fit(Xtr, ytr),
    X=X_train_lin,
    y_log=y_train_log,
    y_raw=y_train_raw,
    folds=cv_folds,
    model_name="Ridge",
    is_log_target=True,
)

# ===========================================================================
# Task 7.3 — Decision Tree
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 7.3  Decision Tree")
logger.info("=" * 60)

t0 = time.perf_counter()
dt_model = fit_decision_tree(X_train_trees, y_train_raw, random_state=RANDOM_SEED)
training_times["decision_tree"] = time.perf_counter() - t0

pred_dt = dt_model.predict(X_val_trees)
m = compute_metrics(y_val_raw, pred_dt, name="Decision Tree", floor=MAPE_FLOOR)
all_metrics.append(m)
val_preds["pred_dt"] = pred_dt

joblib.dump(dt_model, MODEL_DIR / "model_dt.pkl")
logger.info(f"Saved {MODEL_DIR / 'model_dt.pkl'}")

# ===========================================================================
# Task 7.4 — Random Forest
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 7.4  Random Forest  (300 trees — this will take a few minutes)")
logger.info("=" * 60)

t0 = time.perf_counter()
rf_model = fit_random_forest(X_train_trees, y_train_raw, random_state=RANDOM_SEED)
training_times["random_forest"] = time.perf_counter() - t0

pred_rf = rf_model.predict(X_val_trees)
m = compute_metrics(y_val_raw, pred_rf, name="Random Forest", floor=MAPE_FLOOR)
all_metrics.append(m)
val_preds["pred_rf"] = pred_rf

joblib.dump(rf_model, MODEL_DIR / "model_rf.pkl")
logger.info(f"Saved {MODEL_DIR / 'model_rf.pkl'}")

# ===========================================================================
# Task 7.5 — Gradient Boosted Trees
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 7.5  Gradient Boosted Trees (XGBoost + LightGBM)")
logger.info("=" * 60)

# -- XGBoost Track A --
t0 = time.perf_counter()
xgb_model = fit_xgboost(
    X_train_trees, y_train_raw,
    X_val_trees, y_val_raw,
    early_stopping_rounds=EARLY_STOP,
    random_state=RANDOM_SEED,
)
training_times["xgboost"] = time.perf_counter() - t0

pred_xgb = xgb_model.predict(X_val_trees)
m = compute_metrics(y_val_raw, pred_xgb, name="XGBoost", floor=MAPE_FLOOR)
all_metrics.append(m)
val_preds["pred_xgboost"] = pred_xgb

joblib.dump(xgb_model, MODEL_DIR / "model_xgboost.pkl")
xgb_model.save_model(str(MODEL_DIR / "model_xgboost.json"))
logger.info(f"Saved {MODEL_DIR / 'model_xgboost.pkl'} + .json")

# -- LightGBM Track A --
t0 = time.perf_counter()
lgb_model = fit_lightgbm(
    X_train_trees, y_train_raw,
    X_val_trees, y_val_raw,
    early_stopping_rounds=EARLY_STOP,
    random_state=RANDOM_SEED,
)
training_times["lightgbm"] = time.perf_counter() - t0

pred_lgb = lgb_model.predict(X_val_trees)
m = compute_metrics(y_val_raw, pred_lgb, name="LightGBM", floor=MAPE_FLOOR)
all_metrics.append(m)
val_preds["pred_lgbm"] = pred_lgb

joblib.dump(lgb_model, MODEL_DIR / "model_lgbm.pkl")
lgb_model.booster_.save_model(str(MODEL_DIR / "model_lgbm.txt"))
logger.info(f"Saved {MODEL_DIR / 'model_lgbm.pkl'} + .txt")

# ===========================================================================
# Task 7.6 — ANN
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 7.6  Artificial Neural Network (2-layer MLP, PyTorch)")
logger.info("=" * 60)

# Free large tree models from RAM before allocating ANN tensors.
# RF alone is ~4 GB; keeping it live while PyTorch allocates 600k×30
# tensors causes heavy memory pressure on 8 GB M1 systems.
import gc
del rf_model, lgb_model   # xgb_model kept — reused in Task 7.7 Track B comparison
gc.collect()
logger.info("Freed RF/XGB/LGBM from RAM before ANN tensor allocation")

t0 = time.perf_counter()
ann_model, train_losses, val_losses, ann_meta = fit_ann(
    X_train=X_train_lin,
    y_train=y_train_log,
    X_val=X_val_lin,
    y_val=y_val_log,
    feature_names=FEATURES_LINEAR,
    patience=200,        # run all 200 epochs — no early stopping
    random_state=RANDOM_SEED,
)
training_times["ann"] = time.perf_counter() - t0

ann_model.eval()
ann_log_pred = ann_model(X_val_lin)   # numpy in, numpy out
pred_ann = np.expm1(ann_log_pred).clip(0)

m = compute_metrics(y_val_raw, pred_ann, name="ANN (MLP 128-64)", floor=MAPE_FLOOR)
all_metrics.append(m)
val_preds["pred_ann"] = pred_ann

# Save ANN artifacts — sklearn model via joblib, meta via json
joblib.dump(ann_model._model, MODEL_DIR / "model_ann.pkl")
with open(MODEL_DIR / "model_ann_meta.json", "w") as f:
    json.dump(ann_meta, f, indent=2)
logger.info(f"Saved {MODEL_DIR / 'model_ann.pkl'} + model_ann_meta.json")

# Loss curve plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label="Train MSE (log-space)", linewidth=1.5)
ax.plot(val_losses,   label="Val MSE (log-space)",   linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE (log kWh)")
ax.set_title("ANN Training and Validation Loss — Phase 7")
ax.legend()
ax.set_yscale("log")
fig.tight_layout()
fig.savefig(FIG_DIR / "phase7_ann_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info(f"Saved {FIG_DIR / 'phase7_ann_loss_curves.png'}")

# ===========================================================================
# Task 7.7 — Track B Models
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 7.7  Track B Models (Protocol-Enriched, 109 HH, treatment only)")
logger.info("=" * 60)

track_b_metrics: list[dict] = []

# -- Ridge Track B --
t0 = time.perf_counter()
ridge_B = Ridge(alpha=1.0, random_state=RANDOM_SEED)
ridge_B.fit(X_train_lin_B, y_train_log_B)
training_times["ridge_B"] = time.perf_counter() - t0

pred_log_B = ridge_B.predict(X_val_lin_B)
pred_kwh_B = np.expm1(pred_log_B).clip(0)
m_B = compute_metrics(y_val_raw_B, pred_kwh_B, name="Ridge_B (Track B, 109 HH)", floor=MAPE_FLOOR)
track_b_metrics.append(m_B)

joblib.dump(ridge_B, MODEL_DIR / "model_ridge_B.pkl")
logger.info(f"Saved {MODEL_DIR / 'model_ridge_B.pkl'}")

# -- XGBoost Track B --
t0 = time.perf_counter()
xgb_B = fit_xgboost(
    X_train_trees_B, y_train_raw_B,
    X_val_trees_B, y_val_raw_B,
    early_stopping_rounds=EARLY_STOP,
    random_state=RANDOM_SEED,
    suffix="Track B",
)
training_times["xgboost_B"] = time.perf_counter() - t0

pred_xgb_B = xgb_B.predict(X_val_trees_B)
m_B = compute_metrics(y_val_raw_B, pred_xgb_B, name="XGBoost_B (Track B, 109 HH)", floor=MAPE_FLOOR)
track_b_metrics.append(m_B)

joblib.dump(xgb_B, MODEL_DIR / "model_xgboost_B.pkl")
xgb_B.save_model(str(MODEL_DIR / "model_xgboost_B.json"))
logger.info(f"Saved {MODEL_DIR / 'model_xgboost_B.pkl'} + .json")

# Compare Track B vs Track A XGBoost on the Track B val households
# (run Track A XGBoost on the 64 Track B val households for apples-to-apples)
# xgb_model was trained on FEATURES_TREES (45 cols) — slice to those columns only
X_val_trees_B_trackA = val_prot[FEATURES_TREES].values
xgb_A_pred_on_B_val = xgb_model.predict(X_val_trees_B_trackA)
m_A_on_B = compute_metrics(
    y_val_raw_B, xgb_A_pred_on_B_val,
    name="XGBoost_A applied to Track B val (for comparison)",
    floor=MAPE_FLOOR,
)
logger.info(
    f"\nProtocol feature gain: "
    f"XGBoost_B RMSE={m_B['rmse']:.3f}  vs  XGBoost_A (same HH)={m_A_on_B['rmse']:.3f} kWh"
)

# ===========================================================================
# Task 7.9 — Save unified validation predictions
# ===========================================================================
logger.info("\n--- Task 7.9  Save val predictions parquet ---")

val_preds.to_parquet(TABLE_DIR / "phase7_val_predictions.parquet", index=False)
logger.info(f"Saved {TABLE_DIR / 'phase7_val_predictions.parquet'}  shape={val_preds.shape}")

# ===========================================================================
# Task 7.11 — 5-fold CV RMSE for all Track A models
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 7.11  5-Fold CV RMSE — all Track A models")
logger.info("=" * 60)

cv_results: dict[str, list[float]] = {}

# Linear models (log target)
for model_name in ("OLS", "Ridge", "Lasso", "ElasticNet"):
    est = linear_models[model_name]
    cv_results[model_name] = cv_evaluate(
        model_fn=lambda Xtr, ytr, _est=est.__class__(**est.get_params()): _est.fit(Xtr, ytr),
        X=X_train_lin,
        y_log=y_train_log,
        y_raw=y_train_raw,
        folds=cv_folds,
        model_name=model_name,
        is_log_target=True,
    )

# DT (raw target)
dt_params = dict(
    max_depth=8, min_samples_split=20, min_samples_leaf=10,
    max_features=None, random_state=RANDOM_SEED
)
cv_results["Decision Tree"] = cv_evaluate(
    model_fn=lambda Xtr, ytr: _DT(**dt_params).fit(Xtr, ytr),
    X=X_train_trees,
    y_log=y_train_log,
    y_raw=y_train_raw,
    folds=cv_folds,
    model_name="Decision Tree",
    is_log_target=False,
)

# RF (raw target) — use fewer trees for speed in CV
from sklearn.ensemble import RandomForestRegressor as RFR
cv_results["Random Forest"] = cv_evaluate(
    model_fn=lambda Xtr, ytr: RFR(
        n_estimators=50, max_depth=None, min_samples_split=5,
        min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=RANDOM_SEED
    ).fit(Xtr, ytr),
    X=X_train_trees,
    y_log=y_train_log,
    y_raw=y_train_raw,
    folds=cv_folds,
    model_name="Random Forest",
    is_log_target=False,
)

# Note: XGBoost and LightGBM CV omitted here — they need eval_set (val) inside
# each fold, which the current cv_evaluate interface doesn't support.
# Their CV will be covered properly in Phase 8 Optuna tuning.
logger.info("Note: XGBoost/LightGBM/ANN CV skipped (require in-fold eval_set — done in Phase 8)")

# ===========================================================================
# Task 7.10 — Training Report
# ===========================================================================
logger.info("\n--- Task 7.10  Write training report ---")

report_lines: list[str] = []

def rline(s=""):
    report_lines.append(s)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
rline("=" * 72)
rline("HEAPO-Predict Phase 7 — Model Training Report")
rline(f"Generated: {timestamp}")
rline("=" * 72)

rline("")
rline("SECTION 1 — INPUT DATA SUMMARY")
rline(f"  Track A  train : {train_full.shape[0]:>8,} rows, {train_full['Household_ID'].nunique():>4} HH  "
      f"(control={int((train_full['Group']=='control').sum()):,}  "
      f"treatment={int((train_full['Group']=='treatment').sum()):,})")
rline(f"  Track A  val   : {val_full.shape[0]:>8,} rows, {val_full['Household_ID'].nunique():>4} HH")
rline(f"  Track A  test  : {test_full.shape[0]:>8,} rows, {test_full['Household_ID'].nunique():>4} HH")
rline(f"  Track B  train : {train_prot.shape[0]:>8,} rows, {train_prot['Household_ID'].nunique():>4} HH  (treatment only)")
rline(f"  Track B  val   : {val_prot.shape[0]:>8,} rows, {val_prot['Household_ID'].nunique():>4} HH")
rline(f"  Track B  test  : {test_prot.shape[0]:>8,} rows, {test_prot['Household_ID'].nunique():>4} HH")
rline(f"  Target mean  train/val/test (Track A): {y_train_raw.mean():.2f} / {y_val_raw.mean():.2f} / {y_test_raw.mean():.2f} kWh/day")
rline(f"  Target mean  train/val/test (Track B): {y_train_raw_B.mean():.2f} / {y_val_raw_B.mean():.2f} / {y_test_raw.mean():.2f} kWh/day (test approx)")
rline(f"  Log-target mean (train A): {y_train_log.mean():.4f}  std: {y_train_log.std():.4f}")

rline("")
rline("SECTION 2 — VALIDATION SET METRICS (Track A models, kWh space)")
rline(
    f"  {'Model':<38}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}  {'MedAE':>7}  {'sMAPE%':>7}  {'Time(s)':>8}"
)
rline("  " + "-" * 78)

time_map = {
    "Baseline: overall mean": training_times.get("baseline_overall_mean", 0),
    "Baseline: per-HH mean":  training_times.get("baseline_hh_mean", 0),
    "Baseline: HDD-linear":   training_times.get("baseline_hdd_linear", 0),
    "OLS":                    training_times.get("linear_variants", 0) / 4,
    "Ridge":                  training_times.get("linear_variants", 0) / 4,
    "Lasso":                  training_times.get("linear_variants", 0) / 4,
    "ElasticNet":             training_times.get("linear_variants", 0) / 4,
    "Decision Tree":          training_times.get("decision_tree", 0),
    "Random Forest":          training_times.get("random_forest", 0),
    "XGBoost":                training_times.get("xgboost", 0),
    "LightGBM":               training_times.get("lightgbm", 0),
    "ANN (MLP 128-64)":       training_times.get("ann", 0),
}

for m in all_metrics:
    name = m["model"]
    t = time_map.get(name, 0)
    rline(
        f"  {name:<38}  {m['rmse']:>7.3f}  {m['mae']:>7.3f}  {m['r2']:>7.4f}  "
        f"{m['medae']:>7.3f}  {m['smape']:>7.2f}  {t:>8.1f}"
    )

rline("")
rline("SECTION 3 — TRACK B VALIDATION METRICS (protocol-enriched, 109 HH training)")
rline(f"  {'Model':<42}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}  {'sMAPE%':>7}")
rline("  " + "-" * 72)
for m in track_b_metrics:
    rline(
        f"  {m['model']:<42}  {m['rmse']:>7.3f}  {m['mae']:>7.3f}  "
        f"{m['r2']:>7.4f}  {m['smape']:>7.2f}"
    )
rline(f"  {'XGBoost_A applied to Track B val (comparison)':<42}  "
      f"{m_A_on_B['rmse']:>7.3f}  {m_A_on_B['mae']:>7.3f}  "
      f"{m_A_on_B['r2']:>7.4f}  {m_A_on_B['smape']:>7.2f}")
rline(f"  NOTE: Track B val has only 64 HH — interpret metrics with caution (high variance).")

rline("")
rline("SECTION 4 — 5-FOLD CV RMSE (Track A, training data only)")
rline(f"  {'Model':<22}  {'Mean RMSE':>10}  {'Std RMSE':>10}  {'Per-fold'}")
rline("  " + "-" * 70)
for model_name, fold_rmses in cv_results.items():
    rline(
        f"  {model_name:<22}  {np.mean(fold_rmses):>10.3f}  {np.std(fold_rmses):>10.3f}  "
        f"{[round(r, 3) for r in fold_rmses]}"
    )
rline("  XGBoost / LightGBM / ANN CV: deferred to Phase 8 (require in-fold eval_set).")

rline("")
rline("SECTION 5 — TRAINING TIMES")
for name, t in training_times.items():
    rline(f"  {name:<30}: {t:>8.1f}s")

rline("")
rline("SECTION 6 — MODEL ARTIFACTS SAVED")
artifacts = [
    "outputs/models/baseline_hh_means.parquet",
    "outputs/models/baseline_hdd_linear.pkl",
    "outputs/models/model_ols.pkl",
    "outputs/models/model_ridge.pkl",
    "outputs/models/model_lasso.pkl",
    "outputs/models/model_elasticnet.pkl",
    "outputs/models/model_dt.pkl",
    "outputs/models/model_rf.pkl",
    "outputs/models/model_xgboost.pkl",
    "outputs/models/model_xgboost.json",
    "outputs/models/model_lgbm.pkl",
    "outputs/models/model_lgbm.txt",
    "outputs/models/model_ann_state_dict.pt",
    "outputs/models/model_ann_meta.json",
    "outputs/models/model_ridge_B.pkl",
    "outputs/models/model_xgboost_B.pkl",
    "outputs/models/model_xgboost_B.json",
    "outputs/tables/phase7_val_predictions.parquet",
    "outputs/figures/phase7_ann_loss_curves.png",
]
for a in artifacts:
    rline(f"  {a}")

rline("")
rline("SECTION 7 — PHASE 8 READINESS CHECKLIST")
phase8_items = [
    ("model_ridge.pkl",        "Ridge — starting alpha=1.0, needs tuning"),
    ("model_lasso.pkl",        "Lasso — starting alpha=0.01, needs tuning"),
    ("model_elasticnet.pkl",   "ElasticNet — starting alpha=0.01 l1=0.5, needs tuning"),
    ("model_dt.pkl",           "DT — starting max_depth=8, needs tuning"),
    ("model_rf.pkl",           "RF — starting 300 trees, needs tuning"),
    ("model_xgboost.pkl",      "XGBoost — starting params, needs tuning"),
    ("model_lgbm.pkl",         "LightGBM — starting params, needs tuning"),
    ("model_ann_state_dict.pt","ANN — starting 128-64 arch, needs tuning"),
]
for fname, note in phase8_items:
    exists = (MODEL_DIR / fname).exists()
    rline(f"  [{'x' if exists else ' '}] {fname:<35} — {note}")

rline("")
rline("=" * 72)

report_text = "\n".join(report_lines)
report_path = TABLE_DIR / "phase7_training_report.txt"
report_path.write_text(report_text)
logger.info(f"Training report saved → {report_path}")

# Print summary table to console
logger.info("\n" + "=" * 60)
logger.info("VALIDATION METRICS SUMMARY")
logger.info("=" * 60)
logger.info(f"  {'Model':<38}  {'RMSE':>7}  {'MAE':>6}  {'R²':>7}  {'sMAPE%':>7}")
logger.info("  " + "-" * 70)
for m in all_metrics:
    logger.info(
        f"  {m['model']:<38}  {m['rmse']:>7.3f}  {m['mae']:>6.3f}  "
        f"{m['r2']:>7.4f}  {m['smape']:>7.2f}%"
    )

logger.info("\n" + "=" * 70)
logger.info(f"Phase 7 complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 70)
