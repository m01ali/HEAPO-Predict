"""
scripts/07_model_training.py

Phase 7 — Model Training orchestration.

Trains models defined in src/models.py across Track A (full sample) and/or
Track B (protocol-enriched). An interactive prompt at startup lets you select
which track(s) and which specific model(s) to train, so selective re-runs are
fast without re-running the entire pipeline.

Prompt flow:
  Step 1 — Track selection:   1 (Track A) / 2 (Track B) / 3 (All) / 0 (Exit)
  Step 2a — Track A models:   number(s) 1-8, comma-separated for combinations
  Step 2b — Track B models:   number(s) 1-5, comma-separated for combinations

For a full first-run: enter 3 → 8 → 5.

Dataset assignment:
  Track A (*_full.parquet):
    - Baselines, OLS, Ridge, Lasso, ElasticNet, DT, RF, XGBoost, LightGBM, ANN
    - Trees use FEATURES_TREES (45 cols), raw kWh target
    - Linear/ANN use FEATURES_LINEAR (30 cols, scaled), log1p kWh target
  Track B (*_protocol.parquet):
    - XGBoost_B, Ridge_B, DT_B, RF_B
    - Trees use FEATURES_TREES_B (75 cols), raw kWh target
    - Ridge_B uses FEATURES_LINEAR_B (46 cols, scaled), log1p kWh target

Usage:
    source .venv/bin/activate
    python scripts/07_model_training.py
"""

from __future__ import annotations

import json
import logging
import random
import sys
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
    fit_decision_tree_b,
    fit_hdd_baseline,
    fit_lightgbm,
    fit_linear_variants,
    fit_random_forest,
    fit_random_forest_b,
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
FIG_DIR   = ROOT / "outputs" / "figures"
LOG_DIR   = ROOT / "outputs" / "logs"

for d in (MODEL_DIR, TABLE_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging — append mode so selective re-runs accumulate in the same log
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "phase7_run.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Interactive selection prompts
# ---------------------------------------------------------------------------

TRACK_PROMPT = """\

=== Phase 7 — Model Training ===

Select a track:

  [1]  Track A      — full sample (1,119 HH, 646k rows, 45 features)
  [2]  Track B      — protocol-enriched (109 HH, 60k rows, 75 features)
  [3]  All          — both tracks (full first-run)
  [0]  Exit

Enter choice [0-3]: """

TRACK_A_PROMPT = """\

--- Track A: Select model(s) ---

  [1]  Baselines        (Overall Mean, Per-HH Mean, HDD-Linear)
  [2]  Linear models    (OLS, Ridge, Lasso, ElasticNet)
  [3]  Decision Tree    (DT)
  [4]  Random Forest    (RF)       ~5-15 min on CPU
  [5]  XGBoost          (XGB)
  [6]  LightGBM         (LGBM)
  [7]  ANN              (MLP, sklearn MLPRegressor)
  [8]  All models

  Tip: comma-separate for combinations  e.g. 3,4 → DT + RF   |   3,4,5 → DT + RF + XGB

Enter choice(s): """

TRACK_B_PROMPT = """\

--- Track B: Select model(s) ---

  [1]  XGBoost_B        (XGB on 75 protocol features)
  [2]  Ridge_B          (Ridge on 46 linear protocol features)
  [3]  Decision Tree B  (DT_B on 75 protocol features)
  [4]  Random Forest B  (RF_B on 75 protocol features)  ~1-3 min on CPU
  [5]  All models

  Tip: comma-separate for combinations  e.g. 3,4 → DT_B + RF_B

Enter choice(s): """

_TRACK_A_MAP: dict[str, set[str]] = {
    "1": {"baselines"},
    "2": {"linear"},
    "3": {"dt"},
    "4": {"rf"},
    "5": {"xgb"},
    "6": {"lgbm"},
    "7": {"ann"},
}

_TRACK_B_MAP: dict[str, set[str]] = {
    "1": {"xgb_b"},
    "2": {"ridge_b"},
    "3": {"dt_b"},
    "4": {"rf_b"},
}

_ALL_A = {"baselines", "linear", "dt", "rf", "xgb", "lgbm", "ann"}
_ALL_B = {"xgb_b", "ridge_b", "dt_b", "rf_b"}


def get_track_choice() -> set[str]:
    valid = {"0", "1", "2", "3"}
    while True:
        raw = input(TRACK_PROMPT).strip()
        if raw not in valid:
            print(f"  Invalid input '{raw}'. Enter 0, 1, 2, or 3.")
            continue
        if raw == "0":
            print("Exiting.")
            sys.exit(0)
        if raw == "3":
            return {"A", "B"}
        return {"A"} if raw == "1" else {"B"}


def _parse_multi(raw: str, mapping: dict[str, set[str]], all_key: str,
                 all_result: set[str], range_hint: str) -> set[str] | None:
    """Parse a comma-separated token string against a mapping. Returns None on error."""
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return None
    valid = set(mapping) | {all_key}
    bad = [t for t in tokens if t not in valid]
    if bad:
        print(f"  Unrecognised input(s): {bad}. Enter numbers {range_hint}, comma-separated.")
        return None
    if all_key in tokens:
        return all_result
    result: set[str] = set()
    for t in tokens:
        result |= mapping[t]
    return result


def get_track_a_models() -> set[str]:
    while True:
        raw = input(TRACK_A_PROMPT).strip()
        result = _parse_multi(raw, _TRACK_A_MAP, "8", _ALL_A, "1-8")
        if result is not None:
            return result


def get_track_b_models() -> set[str]:
    while True:
        raw = input(TRACK_B_PROMPT).strip()
        result = _parse_multi(raw, _TRACK_B_MAP, "5", _ALL_B, "1-5")
        if result is not None:
            return result


# ---------------------------------------------------------------------------
# Interactive selection — runs before any data loading
# ---------------------------------------------------------------------------
tracks         = get_track_choice()
track_a_models = get_track_a_models() if "A" in tracks else set()
track_b_models = get_track_b_models() if "B" in tracks else set()

logger.info("=" * 70)
logger.info("HEAPO-Predict  Phase 7 — Model Training")
logger.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Tracks  : {sorted(tracks)}")
logger.info(f"Track A : {sorted(track_a_models) if track_a_models else 'none'}")
logger.info(f"Track B : {sorted(track_b_models) if track_b_models else 'none'}")
logger.info("=" * 70)

# ---------------------------------------------------------------------------
# Config + seeds
# ---------------------------------------------------------------------------
cfg         = load_config(str(ROOT / "config" / "params.yaml"))
RANDOM_SEED = cfg["modeling"]["random_seed"]
EARLY_STOP  = cfg["modeling"]["xgboost_early_stopping_rounds"]
ANN_PATIENCE = cfg["modeling"]["ann_early_stopping_patience"]
MAPE_FLOOR  = cfg["evaluation"]["mape_floor_kwh"]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logger.info(f"Random seed: {RANDOM_SEED}  |  XGB early stop: {EARLY_STOP}  |  ANN patience: {ANN_PATIENCE}")

# ---------------------------------------------------------------------------
# Task 7.0 — Conditional data loading + assertions
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

_to_load = []
if "A" in tracks:
    _to_load += ["train_full", "val_full", "test_full"]
if "B" in tracks:
    _to_load += ["train_protocol", "val_protocol", "test_protocol"]

for name in _to_load:
    path = DATA_DIR / f"{name}.parquet"
    df   = pd.read_parquet(path)
    assert df.shape == EXPECTED_SHAPES[name], (
        f"Shape mismatch for {name}: got {df.shape}, expected {EXPECTED_SHAPES[name]}"
    )
    splits[name] = df
    logger.info(f"  {name:20s}: {df.shape}  ✓")

# Feature lists (always needed)
with open(TABLE_DIR / "phase6_feature_lists.json") as f:
    feat = json.load(f)

FEATURES_TREES    = feat["FEATURES_TREES"]     # 45
FEATURES_LINEAR   = feat["FEATURES_LINEAR"]    # 30
FEATURES_TREES_B  = feat["FEATURES_TREES_B"]   # 75
FEATURES_LINEAR_B = feat["FEATURES_LINEAR_B"]  # 46
TARGET_RAW        = feat["TARGET_RAW"]          # "kWh_received_Total"
TARGET_LOG        = feat["TARGET_LOG"]          # "kWh_log1p"

# Scalers — loaded only for the selected tracks, never refitted
if "A" in tracks:
    scaler_A = joblib.load(MODEL_DIR / "scaler_linear_A.pkl")
if "B" in tracks:
    scaler_B = joblib.load(MODEL_DIR / "scaler_linear_B.pkl")

# Assertions scoped to selected tracks
if "A" in tracks:
    train_full = splits["train_full"]
    val_full   = splits["val_full"]
    test_full  = splits["test_full"]

    assert train_full[FEATURES_TREES].isnull().sum().sum() == 0,  "Nulls in FEATURES_TREES (train)"
    assert train_full[FEATURES_LINEAR].isnull().sum().sum() == 0, "Nulls in FEATURES_LINEAR (train)"
    assert val_full[FEATURES_TREES].isnull().sum().sum() == 0,    "Nulls in FEATURES_TREES (val)"
    assert val_full[FEATURES_LINEAR].isnull().sum().sum() == 0,   "Nulls in FEATURES_LINEAR (val)"
    assert train_full[TARGET_RAW].min() > 0, "Non-positive target in training set"
    assert set(train_full["cv_fold"].unique()) == {0, 1, 2, 3, 4}, "cv_fold values not {0,1,2,3,4}"
    assert scaler_A.n_features_in_ == len(FEATURES_LINEAR), (
        f"scaler_A expects {scaler_A.n_features_in_} features, FEATURES_LINEAR has {len(FEATURES_LINEAR)}"
    )

if "B" in tracks:
    train_prot = splits["train_protocol"]
    val_prot   = splits["val_protocol"]
    test_prot  = splits["test_protocol"]

    null_b = train_prot[FEATURES_TREES_B].isnull().sum().sum()
    if null_b > 0:
        _null_per_col = train_prot[FEATURES_TREES_B].isnull().sum()
        _null_per_col = _null_per_col[_null_per_col > 0].sort_values(ascending=False)
        logger.warning(
            f"Track B: {null_b:,} NaN values in FEATURES_TREES_B across "
            f"{len(_null_per_col)} columns. "
            f"XGBoost handles NaN natively; sklearn DT/RF will receive "
            f"median-filled matrices (filling applied in feature matrix step).\n"
            f"  Top null columns: {_null_per_col.head(8).to_dict()}"
        )
    assert train_prot[TARGET_RAW].min() > 0, "Non-positive target in Track B training set"
    assert scaler_B.n_features_in_ == len(FEATURES_LINEAR_B), (
        f"scaler_B expects {scaler_B.n_features_in_} features, FEATURES_LINEAR_B has {len(FEATURES_LINEAR_B)}"
    )
    actual_hh_B = train_prot["Household_ID"].nunique()
    assert actual_hh_B == 109, f"Track B train HH count: expected 109, got {actual_hh_B}"
    assert len(FEATURES_TREES_B) == 75, (
        f"Expected 75 FEATURES_TREES_B, got {len(FEATURES_TREES_B)}"
    )

logger.info("All pre-training assertions passed ✓")

# ---------------------------------------------------------------------------
# Feature matrices (built only for selected tracks)
# ---------------------------------------------------------------------------
logger.info("\n--- Building feature matrices ---")

if "A" in tracks:
    X_train_trees = train_full[FEATURES_TREES].values
    X_val_trees   = val_full[FEATURES_TREES].values
    X_test_trees  = test_full[FEATURES_TREES].values
    y_train_raw   = train_full[TARGET_RAW].values
    y_val_raw     = val_full[TARGET_RAW].values
    y_test_raw    = test_full[TARGET_RAW].values

    X_train_lin   = scaler_A.transform(train_full[FEATURES_LINEAR].values)
    X_val_lin     = scaler_A.transform(val_full[FEATURES_LINEAR].values)
    X_test_lin    = scaler_A.transform(test_full[FEATURES_LINEAR].values)
    y_train_log   = train_full[TARGET_LOG].values
    y_val_log     = val_full[TARGET_LOG].values

    cv_folds      = train_full["cv_fold"].values

    logger.info(
        f"Track A  train: {X_train_trees.shape[0]:,} rows  "
        f"trees={X_train_trees.shape[1]}  linear={X_train_lin.shape[1]}"
    )

if "B" in tracks:
    _train_B_df = train_prot[FEATURES_TREES_B].copy()
    _val_B_df   = val_prot[FEATURES_TREES_B].copy()

    # Fill any remaining protocol nulls with training-set column medians.
    # sklearn DT/RF cannot handle NaN; XGBoost can but filling is harmless.
    # Medians are computed on training data only to avoid val/test leakage.
    _null_cols_B = _train_B_df.columns[_train_B_df.isnull().any()].tolist()
    if _null_cols_B:
        _train_B_medians = _train_B_df[_null_cols_B].median()
        _train_B_df[_null_cols_B] = _train_B_df[_null_cols_B].fillna(_train_B_medians)
        _val_B_df[_null_cols_B]   = _val_B_df[_null_cols_B].fillna(_train_B_medians)
        logger.info(
            f"Track B: filled NaN in {len(_null_cols_B)} columns "
            f"with training-set medians (no val leakage)"
        )

    X_train_trees_B = _train_B_df.values
    X_val_trees_B   = _val_B_df.values
    y_train_raw_B   = train_prot[TARGET_RAW].values
    y_val_raw_B     = val_prot[TARGET_RAW].values

    X_train_lin_B   = scaler_B.transform(train_prot[FEATURES_LINEAR_B].values)
    X_val_lin_B     = scaler_B.transform(val_prot[FEATURES_LINEAR_B].values)
    y_train_log_B   = train_prot[TARGET_LOG].values
    y_val_log_B     = val_prot[TARGET_LOG].values

    logger.info(
        f"Track B  train: {X_train_trees_B.shape[0]:,} rows  "
        f"trees={X_train_trees_B.shape[1]}  linear={X_train_lin_B.shape[1]}"
    )

# ---------------------------------------------------------------------------
# Prediction collectors
# ---------------------------------------------------------------------------
if "A" in tracks:
    val_preds = val_full[
        ["Household_ID", "Timestamp", TARGET_RAW, "is_heating_season", "AffectsTimePoint"]
    ].copy()
else:
    val_preds = None

val_preds_B: pd.DataFrame | None = None
if "B" in tracks:
    val_preds_B = val_prot[
        ["Household_ID", "Timestamp", TARGET_RAW, "is_heating_season", "AffectsTimePoint"]
    ].copy()

all_metrics:    list[dict] = []
track_b_metrics: list[dict] = []
training_times: dict[str, float] = {}

# ===========================================================================
# Task 7.1 — Baselines
# ===========================================================================
if "baselines" in track_a_models:
    logger.info("\n" + "=" * 60)
    logger.info("Task 7.1  Baseline predictors")
    logger.info("=" * 60)

    # 7.1.1 Overall mean
    t0 = time.perf_counter()
    pred_val_overall  = predict_overall_mean(y_train_raw, len(y_val_raw))
    pred_test_overall = np.full(len(y_test_raw), float(y_train_raw.mean()))
    training_times["baseline_overall_mean"] = time.perf_counter() - t0

    m = compute_metrics(y_val_raw, pred_val_overall, name="Baseline: overall mean", floor=MAPE_FLOOR)
    all_metrics.append(m)
    val_preds["pred_overall_mean"] = pred_val_overall

    # 7.1.2 Per-household mean
    t0 = time.perf_counter()
    pred_val_hh  = predict_hh_mean(train_full, val_full,  TARGET_RAW)
    pred_test_hh = predict_hh_mean(train_full, test_full, TARGET_RAW)
    training_times["baseline_hh_mean"] = time.perf_counter() - t0

    m = compute_metrics(y_val_raw, pred_val_hh, name="Baseline: per-HH mean", floor=MAPE_FLOOR)
    all_metrics.append(m)
    val_preds["pred_hh_mean"] = pred_val_hh

    hh_means_df = (
        train_full.groupby("Household_ID")[TARGET_RAW]
        .mean().reset_index()
        .rename(columns={TARGET_RAW: "train_mean_kwh"})
    )
    hh_means_df.to_parquet(MODEL_DIR / "baseline_hh_means.parquet", index=False)
    logger.info(f"Saved {MODEL_DIR / 'baseline_hh_means.parquet'}")

    # 7.1.3 HDD-proportional baseline
    t0 = time.perf_counter()
    hdd_model     = fit_hdd_baseline(train_full, TARGET_RAW)
    pred_val_hdd  = np.clip(hdd_model.predict(val_full[["HDD_SIA_daily"]].values),  0, None)
    pred_test_hdd = np.clip(hdd_model.predict(test_full[["HDD_SIA_daily"]].values), 0, None)
    training_times["baseline_hdd_linear"] = time.perf_counter() - t0

    m = compute_metrics(y_val_raw, pred_val_hdd, name="Baseline: HDD-linear", floor=MAPE_FLOOR)
    all_metrics.append(m)
    val_preds["pred_hdd_linear"] = pred_val_hdd

    joblib.dump(hdd_model, MODEL_DIR / "baseline_hdd_linear.pkl")
    logger.info(f"Saved {MODEL_DIR / 'baseline_hdd_linear.pkl'}")

    logger.info("\nBaseline TEST-set metrics (for anchoring Phase 9 table):")
    compute_metrics(y_test_raw, pred_test_overall, name="[TEST] Baseline: overall mean", floor=MAPE_FLOOR)
    compute_metrics(y_test_raw, pred_test_hh,      name="[TEST] Baseline: per-HH mean",  floor=MAPE_FLOOR)
    compute_metrics(y_test_raw, pred_test_hdd,     name="[TEST] Baseline: HDD-linear",   floor=MAPE_FLOOR)

# ===========================================================================
# Task 7.2 — Linear Regression Variants
# ===========================================================================
if "linear" in track_a_models:
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
else:
    linear_models = {}

# ===========================================================================
# Task 7.3 — Decision Tree
# ===========================================================================
if "dt" in track_a_models:
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
if "rf" in track_a_models:
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
if "xgb" in track_a_models or "lgbm" in track_a_models:
    logger.info("\n" + "=" * 60)
    logger.info("Task 7.5  Gradient Boosted Trees (XGBoost + LightGBM)")
    logger.info("=" * 60)

if "xgb" in track_a_models:
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

if "lgbm" in track_a_models:
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
if "ann" in track_a_models:
    logger.info("\n" + "=" * 60)
    logger.info("Task 7.6  Artificial Neural Network (2-layer MLP, sklearn backend)")
    logger.info("=" * 60)

    # Free large tree models from RAM before allocating ANN tensors.
    import gc
    _freed: list[str] = []
    if "rf" in track_a_models and "rf_model" in dir():
        del rf_model
        _freed.append("rf_model")
    if "lgbm" in track_a_models and "lgb_model" in dir():
        del lgb_model
        _freed.append("lgb_model")
    gc.collect()
    if _freed:
        logger.info(f"Freed from RAM before ANN: {_freed}")

    t0 = time.perf_counter()
    ann_model, train_losses, val_losses, ann_meta = fit_ann(
        X_train=X_train_lin,
        y_train=y_train_log,
        X_val=X_val_lin,
        y_val=y_val_log,
        feature_names=FEATURES_LINEAR,
        patience=200,
        random_state=RANDOM_SEED,
    )
    training_times["ann"] = time.perf_counter() - t0

    ann_model.eval()
    ann_log_pred = ann_model(X_val_lin)
    pred_ann = np.expm1(ann_log_pred).clip(0)

    m = compute_metrics(y_val_raw, pred_ann, name="ANN (MLP 128-64)", floor=MAPE_FLOOR)
    all_metrics.append(m)
    val_preds["pred_ann"] = pred_ann

    joblib.dump(ann_model._model, MODEL_DIR / "model_ann.pkl")
    with open(MODEL_DIR / "model_ann_meta.json", "w") as f:
        json.dump(ann_meta, f, indent=2)
    logger.info(f"Saved {MODEL_DIR / 'model_ann.pkl'} + model_ann_meta.json")

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
if "B" in tracks:
    logger.info("\n" + "=" * 60)
    logger.info("Task 7.7  Track B Models (Protocol-Enriched, 109 HH, treatment only)")
    logger.info("=" * 60)
    logger.info(
        f"  NOTE: Track B val has only 64 HH — "
        "metric variance is ±3-6 kWh/fold. Interpret with caution."
    )

    # ── Ridge_B ──────────────────────────────────────────────────────────────
    if "ridge_b" in track_b_models:
        t0 = time.perf_counter()
        ridge_B = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        ridge_B.fit(X_train_lin_B, y_train_log_B)
        training_times["ridge_B"] = time.perf_counter() - t0

        pred_log_B = ridge_B.predict(X_val_lin_B)
        pred_kwh_B = np.expm1(pred_log_B).clip(0)
        m_B = compute_metrics(
            y_val_raw_B, pred_kwh_B, name="Ridge_B (Track B, 109 HH)", floor=MAPE_FLOOR
        )
        track_b_metrics.append(m_B)
        val_preds_B["pred_ridge_B"] = pred_kwh_B

        joblib.dump(ridge_B, MODEL_DIR / "model_ridge_B.pkl")
        logger.info(f"Saved {MODEL_DIR / 'model_ridge_B.pkl'}")

    # ── XGBoost_B ─────────────────────────────────────────────────────────────
    if "xgb_b" in track_b_models:
        t0 = time.perf_counter()
        xgb_B = fit_xgboost(
            X_train_trees_B, y_train_raw_B,
            X_val_trees_B,   y_val_raw_B,
            early_stopping_rounds=EARLY_STOP,
            random_state=RANDOM_SEED,
            suffix="Track B",
        )
        training_times["xgboost_B"] = time.perf_counter() - t0

        pred_xgb_B = xgb_B.predict(X_val_trees_B)
        m_B = compute_metrics(
            y_val_raw_B, pred_xgb_B, name="XGBoost_B (Track B, 109 HH)", floor=MAPE_FLOOR
        )
        track_b_metrics.append(m_B)
        val_preds_B["pred_xgboost_B"] = pred_xgb_B

        joblib.dump(xgb_B, MODEL_DIR / "model_xgboost_B.pkl")
        xgb_B.save_model(str(MODEL_DIR / "model_xgboost_B.json"))
        logger.info(f"Saved {MODEL_DIR / 'model_xgboost_B.pkl'} + .json")

    # ── DT_B ─────────────────────────────────────────────────────────────────
    if "dt_b" in track_b_models:
        logger.info("\nDT_B: Decision Tree on 75 protocol features")
        t0 = time.perf_counter()
        dt_B = fit_decision_tree_b(X_train_trees_B, y_train_raw_B, random_state=RANDOM_SEED)
        training_times["dt_B"] = time.perf_counter() - t0

        pred_dt_B = dt_B.predict(X_val_trees_B)
        m_B = compute_metrics(
            y_val_raw_B, pred_dt_B, name="DT_B (Track B, 109 HH)", floor=MAPE_FLOOR
        )
        if m_B["r2"] < 0.575:
            logger.warning(
                f"DT_B val R²={m_B['r2']:.4f} is below Track A DT (0.575) — "
                "protocol features may not be helping DT on this small sample. "
                "Inspect overfitting gap (train vs val R²) after training."
            )
        track_b_metrics.append(m_B)
        val_preds_B["pred_dt_B"] = pred_dt_B

        joblib.dump(dt_B, MODEL_DIR / "model_dt_B.pkl")
        logger.info(f"Saved {MODEL_DIR / 'model_dt_B.pkl'}")

    # ── RF_B ─────────────────────────────────────────────────────────────────
    if "rf_b" in track_b_models:
        logger.info("\nRF_B: Random Forest on 75 protocol features  (~1-3 min)")
        t0 = time.perf_counter()
        rf_B = fit_random_forest_b(X_train_trees_B, y_train_raw_B, random_state=RANDOM_SEED)
        training_times["rf_B"] = time.perf_counter() - t0

        pred_rf_B = rf_B.predict(X_val_trees_B)
        m_B = compute_metrics(
            y_val_raw_B, pred_rf_B, name="RF_B (Track B, 109 HH)", floor=MAPE_FLOOR
        )
        track_b_metrics.append(m_B)
        val_preds_B["pred_rf_B"] = pred_rf_B

        joblib.dump(rf_B, MODEL_DIR / "model_rf_B.pkl")
        logger.info(f"Saved {MODEL_DIR / 'model_rf_B.pkl'}")

    # ── XGBoost_A applied to Track B val (apples-to-apples comparison) ───────
    # Only possible if XGBoost_A was trained this run or already saved to disk.
    _xgb_for_comparison = None
    if "xgb" in track_a_models and "xgb_model" in dir():
        _xgb_for_comparison = xgb_model
    elif (MODEL_DIR / "model_xgboost.pkl").exists():
        _xgb_for_comparison = joblib.load(MODEL_DIR / "model_xgboost.pkl")
        logger.info("Loaded saved model_xgboost.pkl for Track B comparison")

    if _xgb_for_comparison is not None and track_b_metrics:
        X_val_trees_B_trackA = val_prot[FEATURES_TREES].values
        xgb_A_pred_on_B_val  = _xgb_for_comparison.predict(X_val_trees_B_trackA)
        m_A_on_B = compute_metrics(
            y_val_raw_B, xgb_A_pred_on_B_val,
            name="XGBoost_A applied to Track B val (for comparison)",
            floor=MAPE_FLOOR,
        )
        best_b_rmse = min(m["rmse"] for m in track_b_metrics)
        logger.info(
            f"\nProtocol feature gain: "
            f"best Track B RMSE={best_b_rmse:.3f}  vs  "
            f"XGBoost_A (same HH)={m_A_on_B['rmse']:.3f} kWh"
        )
    else:
        m_A_on_B = None
        logger.info(
            "XGBoost_A comparison skipped "
            "(model_xgboost.pkl not found and not trained this run)"
        )

# ===========================================================================
# Task 7.9 — Save unified validation predictions
# ===========================================================================
logger.info("\n--- Task 7.9  Save val predictions parquets ---")

if val_preds is not None and len(val_preds.columns) > 5:
    val_preds.to_parquet(TABLE_DIR / "phase7_val_predictions.parquet", index=False)
    logger.info(
        f"Saved {TABLE_DIR / 'phase7_val_predictions.parquet'}  shape={val_preds.shape}"
    )

if val_preds_B is not None and len(val_preds_B.columns) > 5:
    out_B = TABLE_DIR / "phase7_val_predictions_B.parquet"
    if out_B.exists():
        existing_B = pd.read_parquet(out_B)
        for col in val_preds_B.columns:
            if col not in ["Household_ID", "Timestamp", TARGET_RAW,
                           "is_heating_season", "AffectsTimePoint"]:
                if col in existing_B.columns:
                    logger.warning(f"Overwriting existing column {col} in {out_B.name}")
                existing_B[col] = val_preds_B[col].values
        existing_B.to_parquet(out_B, index=False)
    else:
        val_preds_B.to_parquet(out_B, index=False)
    logger.info(f"Saved {out_B}  shape={val_preds_B.shape}")

# ===========================================================================
# Task 7.11 — 5-fold CV RMSE for Track A models
# ===========================================================================
cv_results: dict[str, list[float]] = {}

if "A" in tracks and any(m in track_a_models for m in ("linear", "dt", "rf")):
    logger.info("\n" + "=" * 60)
    logger.info("Task 7.11  5-Fold CV RMSE — Track A models")
    logger.info("=" * 60)

    if "linear" in track_a_models:
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

    if "dt" in track_a_models:
        dt_params = dict(
            max_depth=8, min_samples_split=20, min_samples_leaf=10,
            max_features=None, random_state=RANDOM_SEED,
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

    if "rf" in track_a_models:
        from sklearn.ensemble import RandomForestRegressor as RFR
        cv_results["Random Forest"] = cv_evaluate(
            model_fn=lambda Xtr, ytr: RFR(
                n_estimators=50, max_depth=None, min_samples_split=5,
                min_samples_leaf=3, max_features="sqrt", n_jobs=-1,
                random_state=RANDOM_SEED,
            ).fit(Xtr, ytr),
            X=X_train_trees,
            y_log=y_train_log,
            y_raw=y_train_raw,
            folds=cv_folds,
            model_name="Random Forest",
            is_log_target=False,
        )

    logger.info(
        "Note: XGBoost/LightGBM/ANN CV skipped "
        "(require in-fold eval_set — done in Phase 8)"
    )

# ===========================================================================
# Task 7.10 — Training Report
# ===========================================================================
logger.info("\n--- Task 7.10  Write training report ---")

report_lines: list[str] = []

def rline(s: str = "") -> None:
    report_lines.append(s)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
rline("=" * 72)
rline("HEAPO-Predict Phase 7 — Model Training Report")
rline(f"Generated : {timestamp}")
rline(f"Tracks    : {sorted(tracks)}")
rline(f"Track A   : {sorted(track_a_models) if track_a_models else 'none'}")
rline(f"Track B   : {sorted(track_b_models) if track_b_models else 'none'}")
rline("=" * 72)

if "A" in tracks:
    rline("")
    rline("SECTION 1 — INPUT DATA SUMMARY")
    rline(
        f"  Track A  train : {train_full.shape[0]:>8,} rows, "
        f"{train_full['Household_ID'].nunique():>4} HH  "
        f"(control={int((train_full['Group']=='control').sum()):,}  "
        f"treatment={int((train_full['Group']=='treatment').sum()):,})"
    )
    rline(f"  Track A  val   : {val_full.shape[0]:>8,} rows, {val_full['Household_ID'].nunique():>4} HH")
    rline(f"  Track A  test  : {test_full.shape[0]:>8,} rows, {test_full['Household_ID'].nunique():>4} HH")
    rline(f"  Target mean  train/val/test (Track A): {y_train_raw.mean():.2f} / {y_val_raw.mean():.2f} / {y_test_raw.mean():.2f} kWh/day")
    rline(f"  Log-target mean (train A): {y_train_log.mean():.4f}  std: {y_train_log.std():.4f}")

if "B" in tracks:
    rline("")
    rline("SECTION 1B — TRACK B INPUT DATA SUMMARY")
    rline(f"  Track B  train : {train_prot.shape[0]:>8,} rows, {train_prot['Household_ID'].nunique():>4} HH  (treatment only)")
    rline(f"  Track B  val   : {val_prot.shape[0]:>8,} rows,  {val_prot['Household_ID'].nunique():>4} HH")
    rline(f"  Track B  test  : {test_prot.shape[0]:>8,} rows,   {test_prot['Household_ID'].nunique():>4} HH")
    rline(f"  Target mean  train/val (Track B): {y_train_raw_B.mean():.2f} / {y_val_raw_B.mean():.2f} kWh/day")

if all_metrics:
    rline("")
    rline("SECTION 2 — VALIDATION SET METRICS (Track A models, kWh space)")
    rline(
        f"  {'Model':<38}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}  "
        f"{'MedAE':>7}  {'sMAPE%':>7}  {'Time(s)':>8}"
    )
    rline("  " + "-" * 78)
    time_map = {
        "Baseline: overall mean": training_times.get("baseline_overall_mean", 0),
        "Baseline: per-HH mean":  training_times.get("baseline_hh_mean", 0),
        "Baseline: HDD-linear":   training_times.get("baseline_hdd_linear", 0),
        "OLS":         training_times.get("linear_variants", 0) / max(len(linear_models), 1),
        "Ridge":       training_times.get("linear_variants", 0) / max(len(linear_models), 1),
        "Lasso":       training_times.get("linear_variants", 0) / max(len(linear_models), 1),
        "ElasticNet":  training_times.get("linear_variants", 0) / max(len(linear_models), 1),
        "Decision Tree":    training_times.get("decision_tree", 0),
        "Random Forest":    training_times.get("random_forest", 0),
        "XGBoost":          training_times.get("xgboost", 0),
        "LightGBM":         training_times.get("lightgbm", 0),
        "ANN (MLP 128-64)": training_times.get("ann", 0),
    }
    for m in all_metrics:
        name = m["model"]
        t = time_map.get(name, 0)
        rline(
            f"  {name:<38}  {m['rmse']:>7.3f}  {m['mae']:>7.3f}  {m['r2']:>7.4f}  "
            f"{m['medae']:>7.3f}  {m['smape']:>7.2f}  {t:>8.1f}"
        )

if track_b_metrics:
    rline("")
    rline(
        "SECTION 3 — TRACK B VALIDATION METRICS "
        "(protocol-enriched, 109 HH training, 64 HH val)"
    )
    rline(
        "  NOTE: Small val set (64 HH) — metric variance is ±3-6 kWh/fold. "
        "Interpret with caution."
    )
    rline(f"  {'Model':<44}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}  {'sMAPE%':>7}  {'Time(s)':>8}")
    rline("  " + "-" * 84)
    for m in track_b_metrics:
        t = training_times.get(
            {"Ridge_B (Track B, 109 HH)": "ridge_B",
             "XGBoost_B (Track B, 109 HH)": "xgboost_B",
             "DT_B (Track B, 109 HH)": "dt_B",
             "RF_B (Track B, 109 HH)": "rf_B"}.get(m["model"], ""), 0
        )
        rline(
            f"  {m['model']:<44}  {m['rmse']:>7.3f}  {m['mae']:>7.3f}  "
            f"{m['r2']:>7.4f}  {m['smape']:>7.2f}  {t:>8.1f}"
        )
    if m_A_on_B is not None:
        rline(
            f"  {'XGBoost_A on Track B val (comparison)':<44}  "
            f"{m_A_on_B['rmse']:>7.3f}  {m_A_on_B['mae']:>7.3f}  "
            f"{m_A_on_B['r2']:>7.4f}  {m_A_on_B['smape']:>7.2f}  {'—':>8}"
        )

if cv_results:
    rline("")
    rline("SECTION 4 — 5-FOLD CV RMSE (Track A, training data only)")
    rline(f"  {'Model':<22}  {'Mean RMSE':>10}  {'Std RMSE':>10}  Per-fold")
    rline("  " + "-" * 70)
    for model_name, fold_rmses in cv_results.items():
        rline(
            f"  {model_name:<22}  {np.mean(fold_rmses):>10.3f}  "
            f"{np.std(fold_rmses):>10.3f}  "
            f"{[round(r, 3) for r in fold_rmses]}"
        )
    rline("  XGBoost / LightGBM / ANN CV: deferred to Phase 8.")

if training_times:
    rline("")
    rline("SECTION 5 — TRAINING TIMES")
    for name, t in training_times.items():
        rline(f"  {name:<30}: {t:>8.1f}s")

rline("")
rline("SECTION 6 — MODEL ARTIFACTS SAVED")
_saved = []
if "baselines" in track_a_models:
    _saved += ["outputs/models/baseline_hh_means.parquet",
               "outputs/models/baseline_hdd_linear.pkl"]
if "linear" in track_a_models:
    _saved += ["outputs/models/model_ols.pkl", "outputs/models/model_ridge.pkl",
               "outputs/models/model_lasso.pkl", "outputs/models/model_elasticnet.pkl"]
if "dt"   in track_a_models: _saved.append("outputs/models/model_dt.pkl")
if "rf"   in track_a_models: _saved.append("outputs/models/model_rf.pkl")
if "xgb"  in track_a_models: _saved += ["outputs/models/model_xgboost.pkl",
                                          "outputs/models/model_xgboost.json"]
if "lgbm" in track_a_models: _saved += ["outputs/models/model_lgbm.pkl",
                                          "outputs/models/model_lgbm.txt"]
if "ann"  in track_a_models: _saved += ["outputs/models/model_ann.pkl",
                                          "outputs/models/model_ann_meta.json"]
if "ridge_b" in track_b_models: _saved.append("outputs/models/model_ridge_B.pkl")
if "xgb_b"   in track_b_models: _saved += ["outputs/models/model_xgboost_B.pkl",
                                              "outputs/models/model_xgboost_B.json"]
if "dt_b"    in track_b_models: _saved.append("outputs/models/model_dt_B.pkl")
if "rf_b"    in track_b_models: _saved.append("outputs/models/model_rf_B.pkl")
if val_preds is not None and len(val_preds.columns) > 5:
    _saved.append("outputs/tables/phase7_val_predictions.parquet")
if val_preds_B is not None and len(val_preds_B.columns) > 5:
    _saved.append("outputs/tables/phase7_val_predictions_B.parquet")
if "ann" in track_a_models:
    _saved.append("outputs/figures/phase7_ann_loss_curves.png")
for a in _saved:
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
    ("model_ann.pkl",          "ANN — starting 128-64 arch, needs tuning"),
    ("model_xgboost_B.pkl",    "XGBoost_B — starting params, needs tuning"),
    ("model_ridge_B.pkl",      "Ridge_B — starting alpha=1.0, needs tuning"),
    ("model_dt_B.pkl",         "DT_B (Phase 7.1) — starting max_depth=8, needs tuning"),
    ("model_rf_B.pkl",         "RF_B (Phase 7.1) — starting 300 trees, needs tuning"),
]
for fname, note in phase8_items:
    exists = (MODEL_DIR / fname).exists()
    rline(f"  [{'x' if exists else ' '}] {fname:<35} — {note}")

rline("")
rline("=" * 72)

report_text = "\n".join(report_lines)
report_path = TABLE_DIR / "phase7_training_report.txt"

# Append mode on selective re-runs; overwrite on a full run.
_is_full_run = (track_a_models == _ALL_A and track_b_models == _ALL_B)
report_mode = "w" if _is_full_run else "a"
with open(report_path, report_mode) as f:
    if report_mode == "a" and report_path.exists():
        f.write("\n\n")  # blank separator before appended run
    f.write(report_text)
logger.info(f"Training report {'written' if _is_full_run else 'appended'} → {report_path}")

# Print summary to console
logger.info("\n" + "=" * 60)
logger.info("VALIDATION METRICS SUMMARY")
logger.info("=" * 60)
if all_metrics:
    logger.info(f"  {'Model':<38}  {'RMSE':>7}  {'MAE':>6}  {'R²':>7}  {'sMAPE%':>7}")
    logger.info("  " + "-" * 70)
    for m in all_metrics:
        logger.info(
            f"  {m['model']:<38}  {m['rmse']:>7.3f}  {m['mae']:>6.3f}  "
            f"{m['r2']:>7.4f}  {m['smape']:>7.2f}%"
        )
if track_b_metrics:
    logger.info("\n  Track B:")
    for m in track_b_metrics:
        logger.info(
            f"  {m['model']:<38}  {m['rmse']:>7.3f}  {m['mae']:>6.3f}  "
            f"{m['r2']:>7.4f}  {m['smape']:>7.2f}%"
        )

logger.info("\n" + "=" * 70)
logger.info(f"Phase 7 complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 70)
