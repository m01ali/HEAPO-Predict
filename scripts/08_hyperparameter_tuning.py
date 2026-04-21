"""
scripts/08_hyperparameter_tuning.py

Phase 8 — Hyperparameter Tuning via Optuna Bayesian Optimisation.

Tunes:
  Track A  — ElasticNet, DT, RF, XGBoost, LightGBM, ANN
  Track B  — XGBoost_B, DT_B, RF_B (protocol-enriched, 109 HH)

Objective for every model: minimise val-set RMSE in raw kWh space.
Test set is NEVER used inside an Optuna objective.

Run from repo root:
    python scripts/08_hyperparameter_tuning.py
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb
import lightgbm as lgb

# ── silence noisy libraries during search ──────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_config
from src.models import compute_metrics
from src.ann import ANN, fit_ann

# ── logging ────────────────────────────────────────────────────────────────
_LOG_DIR = ROOT / "outputs" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "phase8.log"

_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

_file_handler = logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8")
_file_handler.setFormatter(_fmt)

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
logger = logging.getLogger(__name__)
logger.info(f"Logging to {_LOG_FILE}")

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "outputs" / "models"
FIG_DIR   = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"

cfg     = load_config()
SEED    = cfg["modeling"]["random_seed"]
MAPE_FL = cfg["evaluation"]["mape_floor_kwh"]
T       = cfg["tuning"]

STUDY_DIR = ROOT / T["study_dir"]
STUDY_DIR.mkdir(parents=True, exist_ok=True)
STORAGE   = f"sqlite:///{STUDY_DIR}/optuna.db"

ES_ROUNDS = T["early_stopping_rounds"]

# ── report helper ──────────────────────────────────────────────────────────
_report_lines: list[str] = []

def _w(s: str = "") -> None:
    _report_lines.append(s)


# ===========================================================================
# Load data
# ===========================================================================
logger.info("=" * 70)
logger.info("Phase 8 — Hyperparameter Tuning")
logger.info(f"Started: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 70)

logger.info("\n--- Loading data ---")
train_full = pd.read_parquet(DATA_DIR / "train_full.parquet")
val_full   = pd.read_parquet(DATA_DIR / "val_full.parquet")
test_full  = pd.read_parquet(DATA_DIR / "test_full.parquet")
train_prot = pd.read_parquet(DATA_DIR / "train_protocol.parquet")
val_prot   = pd.read_parquet(DATA_DIR / "val_protocol.parquet")
test_prot  = pd.read_parquet(DATA_DIR / "test_protocol.parquet")

with open(TABLE_DIR / "phase6_feature_lists.json") as f:
    feat = json.load(f)

FEATURES_TREES    = feat["FEATURES_TREES"]
FEATURES_LINEAR   = feat["FEATURES_LINEAR"]
FEATURES_TREES_B  = feat["FEATURES_TREES_B"]
FEATURES_LINEAR_B = feat["FEATURES_LINEAR_B"]
TARGET_RAW = feat["TARGET_RAW"]
TARGET_LOG = feat["TARGET_LOG"]

scaler_A = joblib.load(MODEL_DIR / "scaler_linear_A.pkl")
scaler_B = joblib.load(MODEL_DIR / "scaler_linear_B.pkl")

# ── Track A feature matrices ───────────────────────────────────────────────
X_train_trees = train_full[FEATURES_TREES].values
X_val_trees   = val_full[FEATURES_TREES].values
X_test_trees  = test_full[FEATURES_TREES].values
y_train_raw   = train_full[TARGET_RAW].values
y_val_raw     = val_full[TARGET_RAW].values
y_test_raw    = test_full[TARGET_RAW].values

X_train_lin = scaler_A.transform(train_full[FEATURES_LINEAR].values)
X_val_lin   = scaler_A.transform(val_full[FEATURES_LINEAR].values)
X_test_lin  = scaler_A.transform(test_full[FEATURES_LINEAR].values)
y_train_log = train_full[TARGET_LOG].values
y_val_log   = val_full[TARGET_LOG].values

# ── Track B feature matrices (null-fill with training medians) ─────────────
_train_B_df = train_prot[FEATURES_TREES_B].copy()
_val_B_df   = val_prot[FEATURES_TREES_B].copy()
_test_B_df  = test_prot[FEATURES_TREES_B].copy()

_null_cols_B = _train_B_df.columns[_train_B_df.isnull().any()].tolist()
if _null_cols_B:
    _train_B_medians = _train_B_df[_null_cols_B].median()
    _train_B_df[_null_cols_B] = _train_B_df[_null_cols_B].fillna(_train_B_medians)
    _val_B_df[_null_cols_B]   = _val_B_df[_null_cols_B].fillna(_train_B_medians)
    _test_B_df[_null_cols_B]  = _test_B_df[_null_cols_B].fillna(_train_B_medians)
    logger.info(
        f"Track B: filled NaN in {len(_null_cols_B)} columns "
        f"with training-set medians (no val/test leakage)"
    )

X_train_trees_B = _train_B_df.values
X_val_trees_B   = _val_B_df.values
X_test_trees_B  = _test_B_df.values
y_train_raw_B   = train_prot[TARGET_RAW].values
y_val_raw_B     = val_prot[TARGET_RAW].values
y_test_raw_B    = test_prot[TARGET_RAW].values

logger.info(f"Track A  train: {X_train_trees.shape[0]:,} rows  |  val: {X_val_trees.shape[0]:,}  |  test: {X_test_trees.shape[0]:,}")
logger.info(f"Track B  train: {X_train_trees_B.shape[0]:,} rows  |  val: {X_val_trees_B.shape[0]:,}  |  test: {X_test_trees_B.shape[0]:,}")

# Phase 7 baseline RMSEs (for delta table in report)
P7 = {
    "ElasticNet":  12.185,
    "DT":          11.382,
    "RF":           9.421,
    "XGBoost":      9.462,
    "LightGBM":     9.318,
    "ANN":         10.330,
    "XGBoost_B":    5.933,
    "DT_B":         8.753,
    "RF_B":         6.731,
}

best_params_all: dict[str, dict] = {}


# ===========================================================================
# Runtime selection — Track and model prompts
# ===========================================================================
_TRACK_PROMPT = (
    "\n=== Phase 8 -- Hyperparameter Tuning ===\n\n"
    "Select track(s) to tune:\n\n"
    "  [1]  Track A  (full sample, 1,119 HH)\n"
    "       Models: ElasticNet, DT, RF, XGBoost, LightGBM, ANN\n"
    "  [2]  Track B  (protocol-enriched, 109 HH)\n"
    "       Models: XGBoost_B, DT_B, RF_B\n"
    "  [3]  Both\n"
    "  [0]  Exit"
)

_TRACK_A_PROMPT = (
    "\nSelect Track A model(s) to tune:\n\n"
    "  [1]  ElasticNet\n"
    "  [2]  DT\n"
    "  [3]  RF\n"
    "  [4]  XGBoost\n"
    "  [5]  LightGBM\n"
    "  [6]  ANN\n"
    "  [7]  All\n\n"
    "  Tip: comma-separate for combinations  e.g. 1,4 = ElasticNet + XGBoost"
)

_TRACK_B_PROMPT = (
    "\nSelect Track B model(s) to tune:\n\n"
    "  [1]  XGBoost_B\n"
    "  [2]  DT_B\n"
    "  [3]  RF_B\n"
    "  [4]  All\n\n"
    "  Tip: comma-separate for combinations  e.g. 2,3 = DT_B + RF_B"
)

_ALL_A = {"elasticnet", "dt", "rf", "xgboost", "lgbm", "ann"}
_ALL_B = {"xgboost_b", "dt_b", "rf_b"}

_A_MAP: dict[str, set[str]] = {
    "1": {"elasticnet"}, "2": {"dt"}, "3": {"rf"},
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


tracks = _get_track_choice()
track_a_models: set[str] = set()
track_b_models: set[str] = set()

if "A" in tracks:
    track_a_models = _get_track_a_models()
    logger.info(f"Track A tuning selection: {sorted(track_a_models)}")

if "B" in tracks:
    track_b_models = _get_track_b_models()
    logger.info(f"Track B tuning selection: {sorted(track_b_models)}")

# prerequisite check for DT_B / RF_B
if "dt_b" in track_b_models and not (MODEL_DIR / "model_dt_B.pkl").exists():
    logger.warning(
        "model_dt_B.pkl not found — DT_B was not trained in Phase 7.1. "
        "Run scripts/07_model_training.py and select Track B → [3] DT_B first."
    )
if "rf_b" in track_b_models and not (MODEL_DIR / "model_rf_B.pkl").exists():
    logger.warning(
        "model_rf_B.pkl not found — RF_B was not trained in Phase 7.1. "
        "Run scripts/07_model_training.py and select Track B → [4] RF_B first."
    )

# ── initialise elapsed times and study handles ─────────────────────────────
elapsed_en = elapsed_dt = elapsed_rf = elapsed_xgb = elapsed_lgbm = elapsed_ann = 0.0
elapsed_xgb_b = elapsed_dt_b = elapsed_rf_b = 0.0
en_study = dt_study = rf_study = xgb_study = lgbm_study = ann_study = None
xgb_b_study = dt_b_study = rf_b_study = None


# ── shared helpers ─────────────────────────────────────────────────────────
def _trial_cb(study: optuna.Study, trial: optuna.Trial) -> None:
    n = len(study.trials)
    best = study.best_value
    cur  = trial.value if trial.value is not None else float("nan")
    logger.info(
        f"  Trial {n:3d}  val_RMSE={cur:.4f}  best={best:.4f}  "
        f"params={json.dumps(trial.params, default=str)}"
    )


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred.clip(0))))


# ===========================================================================
# Task 8.1 — ElasticNet (30 trials)
# ===========================================================================
if "elasticnet" in track_a_models:
    logger.info("\n" + "=" * 60)
    logger.info("Task 8.1  ElasticNet Hyperparameter Tuning")
    logger.info("=" * 60)

    def _en_objective(trial: optuna.Trial) -> float:
        alpha    = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=SEED)
        model.fit(X_train_lin, y_train_log)
        pred = np.expm1(model.predict(X_val_lin)).clip(0)
        return _rmse(y_val_raw, pred)

    en_study = optuna.create_study(
        study_name="phase8_elasticnet", storage=STORAGE,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    t0 = time.perf_counter()
    _n = max(0, T["n_trials_linear"] - len(en_study.trials))
    if _n: en_study.optimize(_en_objective, n_trials=_n, callbacks=[_trial_cb], show_progress_bar=False)
    elapsed_en = time.perf_counter() - t0

    best_params_all["ElasticNet"] = en_study.best_params
    logger.info(f"ElasticNet best: {en_study.best_params}  val_RMSE={en_study.best_value:.4f}  ({elapsed_en:.0f}s)")


# ===========================================================================
# Task 8.2 — Decision Tree Track A (40 trials)
# ===========================================================================
if "dt" in track_a_models:
    logger.info("\n" + "=" * 60)
    logger.info("Task 8.2  Decision Tree Hyperparameter Tuning")
    logger.info("=" * 60)

    def _dt_objective(trial: optuna.Trial) -> float:
        params = dict(
            max_depth         = trial.suggest_int("max_depth", 4, 20),
            min_samples_split = trial.suggest_int("min_samples_split", 5, 100, log=True),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 3, 50, log=True),
            max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state      = SEED,
        )
        model = DecisionTreeRegressor(**params).fit(X_train_trees, y_train_raw)
        return _rmse(y_val_raw, model.predict(X_val_trees))

    dt_study = optuna.create_study(
        study_name="phase8_dt", storage=STORAGE,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    t0 = time.perf_counter()
    _n = max(0, T["n_trials_dt"] - len(dt_study.trials))
    if _n: dt_study.optimize(_dt_objective, n_trials=_n, callbacks=[_trial_cb], show_progress_bar=False)
    elapsed_dt = time.perf_counter() - t0

    best_params_all["DT"] = dt_study.best_params
    logger.info(f"DT best: {dt_study.best_params}  val_RMSE={dt_study.best_value:.4f}  ({elapsed_dt:.0f}s)")


# ===========================================================================
# Task 8.3 — XGBoost Track B (40 trials)
# ===========================================================================
if "xgboost_b" in track_b_models:
    logger.info("\n" + "=" * 60)
    logger.info("Task 8.3  XGBoost Track B Hyperparameter Tuning")
    logger.info("=" * 60)

    def _xgb_b_objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators          = 2000,
            max_depth             = trial.suggest_int("max_depth", 3, 7),
            learning_rate         = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample             = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree      = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight      = trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha             = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda            = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            gamma                 = trial.suggest_float("gamma", 0.0, 3.0),
            early_stopping_rounds = ES_ROUNDS,
            tree_method           = "hist",
            random_state          = SEED,
            n_jobs                = -1,
        )
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_trees_B, y_train_raw_B,
                  eval_set=[(X_val_trees_B, y_val_raw_B)], verbose=False)
        return _rmse(y_val_raw_B, model.predict(X_val_trees_B))

    xgb_b_study = optuna.create_study(
        study_name="phase8_xgboost_b", storage=STORAGE,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    t0 = time.perf_counter()
    _n = max(0, T["n_trials_xgb_b"] - len(xgb_b_study.trials))
    if _n: xgb_b_study.optimize(_xgb_b_objective, n_trials=_n, callbacks=[_trial_cb], show_progress_bar=False)
    elapsed_xgb_b = time.perf_counter() - t0

    best_params_all["XGBoost_B"] = xgb_b_study.best_params
    logger.info(f"XGBoost_B best: {xgb_b_study.best_params}  val_RMSE={xgb_b_study.best_value:.4f}  ({elapsed_xgb_b:.0f}s)")


# ===========================================================================
# Task 8.3b — DT_B Hyperparameter Tuning (30 trials)
# ===========================================================================
if "dt_b" in track_b_models:
    logger.info("\n" + "=" * 60)
    logger.info("Task 8.3b  DT_B Hyperparameter Tuning")
    logger.info("=" * 60)

    def _dt_b_objective(trial: optuna.Trial) -> float:
        if not (MODEL_DIR / "model_dt_B.pkl").exists():
            raise optuna.exceptions.TrialPruned()
        params = dict(
            max_depth         = trial.suggest_int("max_depth", 4, 15),
            min_samples_split = trial.suggest_int("min_samples_split", 10, 200, log=True),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 5, 100, log=True),
            max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state      = SEED,
        )
        model = DecisionTreeRegressor(**params)
        model.fit(X_train_trees_B, y_train_raw_B)
        return _rmse(y_val_raw_B, model.predict(X_val_trees_B))

    dt_b_study = optuna.create_study(
        study_name="phase8_dt_b", storage=STORAGE,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    t0 = time.perf_counter()
    _n = max(0, T["n_trials_dt_b"] - len(dt_b_study.trials))
    if _n:
        dt_b_study.optimize(_dt_b_objective, n_trials=_n,
                            callbacks=[_trial_cb], show_progress_bar=False)
    elapsed_dt_b = time.perf_counter() - t0

    best_params_all["DT_B"] = dt_b_study.best_params
    logger.info(
        f"DT_B best: {dt_b_study.best_params}  "
        f"val_RMSE={dt_b_study.best_value:.4f}  ({elapsed_dt_b:.0f}s)"
    )


# ===========================================================================
# Task 8.3c — RF_B Hyperparameter Tuning (40 trials)
# ===========================================================================
if "rf_b" in track_b_models:
    logger.info("\n" + "=" * 60)
    logger.info("Task 8.3c  RF_B Hyperparameter Tuning  (n_estimators=50 during search)")
    logger.info("=" * 60)

    def _rf_b_objective(trial: optuna.Trial) -> float:
        if not (MODEL_DIR / "model_rf_B.pkl").exists():
            raise optuna.exceptions.TrialPruned()
        params = dict(
            n_estimators      = 50,
            max_depth         = trial.suggest_categorical("max_depth", [10, 20, 30, None]),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 30),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 20),
            max_features      = trial.suggest_categorical(
                                    "max_features", ["sqrt", "log2", 0.3, 0.5]
                                ),
            bootstrap         = trial.suggest_categorical("bootstrap", [True, False]),
            n_jobs            = -1,
            random_state      = SEED,
        )
        model = RandomForestRegressor(**params)
        model.fit(X_train_trees_B, y_train_raw_B)
        rmse = _rmse(y_val_raw_B, model.predict(X_val_trees_B))
        del model; gc.collect()
        return rmse

    rf_b_study = optuna.create_study(
        study_name="phase8_rf_b", storage=STORAGE,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    t0 = time.perf_counter()
    _n = max(0, T["n_trials_rf_b"] - len(rf_b_study.trials))
    if _n:
        rf_b_study.optimize(_rf_b_objective, n_trials=_n,
                            callbacks=[_trial_cb], show_progress_bar=False)
    elapsed_rf_b = time.perf_counter() - t0

    best_params_all["RF_B"] = rf_b_study.best_params
    logger.info(
        f"RF_B best: {rf_b_study.best_params}  "
        f"val_RMSE={rf_b_study.best_value:.4f}  ({elapsed_rf_b:.0f}s)"
    )


# ===========================================================================
# Task 8.4 — XGBoost Track A (80 trials)
# ===========================================================================
if "xgboost" in track_a_models:
    logger.info("\n" + "=" * 60)
    logger.info("Task 8.4  XGBoost Track A Hyperparameter Tuning")
    logger.info("=" * 60)

    def _xgb_objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators          = 2000,
            max_depth             = trial.suggest_int("max_depth", 3, 10),
            learning_rate         = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample             = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree      = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_weight      = trial.suggest_int("min_child_weight", 1, 15),
            reg_alpha             = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda            = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            gamma                 = trial.suggest_float("gamma", 0.0, 5.0),
            early_stopping_rounds = ES_ROUNDS,
            tree_method           = "hist",
            random_state          = SEED,
            n_jobs                = -1,
        )
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_trees, y_train_raw,
                  eval_set=[(X_val_trees, y_val_raw)], verbose=False)
        return _rmse(y_val_raw, model.predict(X_val_trees))

    xgb_study = optuna.create_study(
        study_name="phase8_xgboost", storage=STORAGE,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    t0 = time.perf_counter()
    _n = max(0, T["n_trials_xgb"] - len(xgb_study.trials))
    if _n: xgb_study.optimize(_xgb_objective, n_trials=_n, callbacks=[_trial_cb], show_progress_bar=False)
    elapsed_xgb = time.perf_counter() - t0

    best_params_all["XGBoost"] = xgb_study.best_params
    logger.info(f"XGBoost best: {xgb_study.best_params}  val_RMSE={xgb_study.best_value:.4f}  ({elapsed_xgb:.0f}s)")


# ===========================================================================
# Task 8.5 — LightGBM Track A (80 trials)
# ===========================================================================
if "lgbm" in track_a_models:
    logger.info("\n" + "=" * 60)
    logger.info("Task 8.5  LightGBM Track A Hyperparameter Tuning")
    logger.info("=" * 60)

    def _lgbm_objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators      = 3000,
            num_leaves        = trial.suggest_int("num_leaves", 20, 300),
            max_depth         = trial.suggest_int("max_depth", 3, 15),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples = trial.suggest_int("min_child_samples", 10, 200, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_split_gain    = trial.suggest_float("min_split_gain", 0.0, 1.0),
            random_state      = SEED,
            n_jobs            = -1,
            verbose           = -1,
        )
        model = lgb.LGBMRegressor(**params)
        callbacks = [
            lgb.early_stopping(stopping_rounds=ES_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        model.fit(
            X_train_trees, y_train_raw,
            eval_set=[(X_val_trees, y_val_raw)],
            callbacks=callbacks,
        )
        pred = model.predict(X_val_trees)
        if isinstance(pred, pd.DataFrame):
            pred = pred.values
        return _rmse(y_val_raw, np.asarray(pred))

    lgbm_study = optuna.create_study(
        study_name="phase8_lgbm", storage=STORAGE,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    t0 = time.perf_counter()
    _n = max(0, T["n_trials_lgbm"] - len(lgbm_study.trials))
    if _n: lgbm_study.optimize(_lgbm_objective, n_trials=_n, callbacks=[_trial_cb], show_progress_bar=False)
    elapsed_lgbm = time.perf_counter() - t0

    best_params_all["LightGBM"] = lgbm_study.best_params
    logger.info(f"LightGBM best: {lgbm_study.best_params}  val_RMSE={lgbm_study.best_value:.4f}  ({elapsed_lgbm:.0f}s)")


# ===========================================================================
# Task 8.6 — ANN (60 trials)
# ===========================================================================
if "ann" in track_a_models:
    logger.info("\n" + "=" * 60)
    logger.info("Task 8.6  ANN Hyperparameter Tuning")
    logger.info("=" * 60)

    from sklearn.neural_network import MLPRegressor

    def _ann_objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layer_sizes = tuple(
            trial.suggest_categorical(f"n_units_l{i}", [32, 64, 128, 256])
            for i in range(n_layers)
        )
        lr    = trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True)
        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
        batch = trial.suggest_categorical("batch_size", [128, 256, 512])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MLPRegressor(
                hidden_layer_sizes  = layer_sizes,
                activation          = "relu",
                solver              = "adam",
                alpha               = alpha,
                batch_size          = batch,
                learning_rate_init  = lr,
                learning_rate       = "constant",
                max_iter            = 1,
                warm_start          = True,
                tol                 = 1e-10,
                n_iter_no_change    = 300,
                early_stopping      = False,
                verbose             = False,
                random_state        = SEED,
            )
            best_val = float("inf")
            patience_counter = 0
            pat = T["ann_patience_search"]

            for _epoch in range(100):
                model.fit(X_train_lin, y_train_log)
                val_pred = np.expm1(model.predict(X_val_lin)).clip(0)
                val_rmse = _rmse(y_val_raw, val_pred)
                if val_rmse < best_val - 1e-5:
                    best_val = val_rmse
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= pat:
                    break

        return best_val

    ann_study = optuna.create_study(
        study_name="phase8_ann", storage=STORAGE,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    t0 = time.perf_counter()
    _n = max(0, T["n_trials_ann"] - len(ann_study.trials))
    if _n: ann_study.optimize(_ann_objective, n_trials=_n, callbacks=[_trial_cb], show_progress_bar=False)
    elapsed_ann = time.perf_counter() - t0

    best_params_all["ANN"] = ann_study.best_params
    logger.info(f"ANN best: {ann_study.best_params}  val_RMSE={ann_study.best_value:.4f}  ({elapsed_ann:.0f}s)")


# ===========================================================================
# Task 8.7 — Random Forest Track A (60 trials) — heaviest, runs last
# ===========================================================================
if "rf" in track_a_models:
    logger.info("\n" + "=" * 60)
    logger.info("Task 8.7  Random Forest Hyperparameter Tuning  (n_estimators=150 during search)")
    logger.info("=" * 60)

    def _rf_objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators      = T["rf_n_estimators_search"],
            max_depth         = trial.suggest_categorical("max_depth", [10, 15, 20, 30, None]),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
            max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            bootstrap         = trial.suggest_categorical("bootstrap", [True, False]),
            n_jobs            = -1,
            random_state      = SEED,
        )
        model = RandomForestRegressor(**params).fit(X_train_trees, y_train_raw)
        rmse  = _rmse(y_val_raw, model.predict(X_val_trees))
        del model; gc.collect()
        return rmse

    rf_study = optuna.create_study(
        study_name="phase8_rf", storage=STORAGE,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    t0 = time.perf_counter()
    _n = max(0, T["n_trials_rf"] - len(rf_study.trials))
    if _n: rf_study.optimize(_rf_objective, n_trials=_n, callbacks=[_trial_cb], show_progress_bar=False)
    elapsed_rf = time.perf_counter() - t0

    best_params_all["RF"] = rf_study.best_params
    logger.info(f"RF best: {rf_study.best_params}  val_RMSE={rf_study.best_value:.4f}  ({elapsed_rf:.0f}s)")


# ===========================================================================
# Task 8.8 — Save best params JSON (load-then-update preserves params from
#            prior runs when only a subset of models are tuned in this run)
# ===========================================================================
_bp_path = MODEL_DIR / "best_params.json"
if _bp_path.exists():
    with open(_bp_path) as f:
        _existing_params = json.load(f)
    _existing_params.update(best_params_all)
    best_params_all = _existing_params

with open(_bp_path, "w") as f:
    json.dump(best_params_all, f, indent=2, default=str)
logger.info(f"\nSaved best_params.json → {_bp_path}  (keys: {sorted(best_params_all)})")


# ===========================================================================
# Task 8.9 — Retrain with best hyperparameters on full training set
#             Each block runs only if params exist in best_params.json
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 8.9  Final Retraining with Best Hyperparameters")
logger.info("=" * 60)

if "ElasticNet" in best_params_all:
    en_tuned = ElasticNet(**best_params_all["ElasticNet"], max_iter=5000, random_state=SEED)
    en_tuned.fit(X_train_lin, y_train_log)
    joblib.dump(en_tuned, MODEL_DIR / "model_elasticnet_tuned.pkl")
    logger.info("Retrained ElasticNet_tuned ✓")

if "DT" in best_params_all:
    dt_tuned = DecisionTreeRegressor(**best_params_all["DT"], random_state=SEED)
    dt_tuned.fit(X_train_trees, y_train_raw)
    joblib.dump(dt_tuned, MODEL_DIR / "model_dt_tuned.pkl")
    logger.info("Retrained DT_tuned ✓")

if "RF" in best_params_all:
    use_oob = best_params_all["RF"].get("bootstrap", True)
    rf_params = {
        **best_params_all["RF"],
        "n_estimators": T["rf_n_estimators_final"],
        "n_jobs": -1, "random_state": SEED, "oob_score": use_oob, "verbose": 1,
    }
    logger.info(f"Fitting RF_tuned with n_estimators={T['rf_n_estimators_final']}...")
    rf_tuned = RandomForestRegressor(**rf_params)
    rf_tuned.fit(X_train_trees, y_train_raw)
    if use_oob:
        logger.info(f"RF_tuned OOB R²={rf_tuned.oob_score_:.4f}")
    else:
        logger.info("RF_tuned fitted (bootstrap=False, no OOB score)")
    joblib.dump(rf_tuned, MODEL_DIR / "model_rf_tuned.pkl")
    logger.info("Retrained RF_tuned ✓")
    del rf_tuned; gc.collect()

if "XGBoost" in best_params_all:
    _xgb_params = {
        **best_params_all["XGBoost"],
        "n_estimators": 2000, "early_stopping_rounds": ES_ROUNDS,
        "tree_method": "hist", "random_state": SEED, "n_jobs": -1,
    }
    xgb_tuned = xgb.XGBRegressor(**_xgb_params)
    xgb_tuned.fit(X_train_trees, y_train_raw,
                  eval_set=[(X_val_trees, y_val_raw)], verbose=100)
    joblib.dump(xgb_tuned, MODEL_DIR / "model_xgboost_tuned.pkl")
    xgb_tuned.save_model(str(MODEL_DIR / "model_xgboost_tuned.json"))
    logger.info("Retrained XGBoost_tuned ✓")

if "LightGBM" in best_params_all:
    _lgbm_params = {
        **best_params_all["LightGBM"],
        "n_estimators": 3000, "random_state": SEED, "n_jobs": -1, "verbose": -1,
    }
    lgbm_tuned = lgb.LGBMRegressor(**_lgbm_params)
    lgbm_tuned.fit(
        X_train_trees, y_train_raw,
        eval_set=[(X_val_trees, y_val_raw)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=ES_ROUNDS, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    joblib.dump(lgbm_tuned, MODEL_DIR / "model_lgbm_tuned.pkl")
    lgbm_tuned.booster_.save_model(str(MODEL_DIR / "model_lgbm_tuned.txt"))
    logger.info("Retrained LightGBM_tuned ✓")

if "ANN" in best_params_all:
    ann_bp = best_params_all["ANN"]
    n_layers_best = ann_bp["n_layers"]
    best_layers = tuple(ann_bp[f"n_units_l{i}"] for i in range(n_layers_best))
    logger.info(
        f"Retraining ANN_tuned: arch={best_layers}  "
        f"lr={ann_bp['learning_rate_init']:.5f}  alpha={ann_bp['alpha']:.2e}  batch={ann_bp['batch_size']}"
    )
    ann_tuned, ann_train_losses, ann_val_losses, ann_meta = fit_ann(
        X_train=X_train_lin,
        y_train=y_train_log,
        X_val=X_val_lin,
        y_val=y_val_log,
        feature_names=FEATURES_LINEAR,
        patience=T["ann_patience_final"],
        random_state=SEED,
        hidden_layers=list(best_layers),
        learning_rate=ann_bp["learning_rate_init"],
        batch_size=ann_bp["batch_size"],
        max_epochs=200,
    )
    joblib.dump(ann_tuned._model, MODEL_DIR / "model_ann_tuned.pkl")
    ann_meta["best_optuna_params"] = ann_bp
    with open(MODEL_DIR / "model_ann_tuned_meta.json", "w") as f:
        json.dump(ann_meta, f, indent=2)
    logger.info("Retrained ANN_tuned ✓")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ann_train_losses, label="Train MSE (log)", linewidth=1.5)
    ax.plot(ann_val_losses,   label="Val MSE (log)",   linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (log kWh)")
    ax.set_title("ANN Tuned — Training and Validation Loss")
    ax.legend(); ax.set_yscale("log"); fig.tight_layout()
    fig.savefig(FIG_DIR / "phase8_ann_tuned_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

if "XGBoost_B" in best_params_all:
    _xgb_b_params = {
        **best_params_all["XGBoost_B"],
        "n_estimators": 2000, "early_stopping_rounds": ES_ROUNDS,
        "tree_method": "hist", "random_state": SEED, "n_jobs": -1,
    }
    xgb_b_tuned = xgb.XGBRegressor(**_xgb_b_params)
    xgb_b_tuned.fit(X_train_trees_B, y_train_raw_B,
                    eval_set=[(X_val_trees_B, y_val_raw_B)], verbose=100)
    joblib.dump(xgb_b_tuned, MODEL_DIR / "model_xgboost_b_tuned.pkl")
    xgb_b_tuned.save_model(str(MODEL_DIR / "model_xgboost_b_tuned.json"))
    logger.info("Retrained XGBoost_B_tuned ✓")

if "DT_B" in best_params_all:
    dt_B_tuned = DecisionTreeRegressor(**best_params_all["DT_B"], random_state=SEED)
    dt_B_tuned.fit(X_train_trees_B, y_train_raw_B)
    joblib.dump(dt_B_tuned, MODEL_DIR / "model_dt_B_tuned.pkl")
    logger.info("Retrained DT_B_tuned ✓")

if "RF_B" in best_params_all:
    use_oob_b = best_params_all["RF_B"].get("bootstrap", True)
    rf_B_tuned = RandomForestRegressor(
        **{k: v for k, v in best_params_all["RF_B"].items() if k != "n_estimators"},
        n_estimators = 300,
        oob_score    = use_oob_b,
        n_jobs       = -1,
        random_state = SEED,
        verbose      = 1,
    )
    rf_B_tuned.fit(X_train_trees_B, y_train_raw_B)
    if use_oob_b:
        logger.info(f"RF_B_tuned OOB R²={rf_B_tuned.oob_score_:.4f}")
    joblib.dump(rf_B_tuned, MODEL_DIR / "model_rf_B_tuned.pkl")
    logger.info("Retrained RF_B_tuned ✓")
    del rf_B_tuned; gc.collect()


# ===========================================================================
# Task 8.10 — Evaluate all tuned models on val and test sets
#             Loads from disk so evaluation includes results from prior runs
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 8.10  Evaluation — Val and Test Sets")
logger.info("=" * 60)


def _predict_and_eval(model, X, y_raw, X_lin=None, name="", is_log=False, floor=MAPE_FL):
    if is_log:
        pred_log = model.predict(X_lin if X_lin is not None else X)
        pred = np.expm1(np.asarray(pred_log)).clip(0)
    else:
        pred = np.asarray(model.predict(X)).clip(0)
    return compute_metrics(y_raw, pred, name=name, floor=floor), pred


val_preds_tuned  = {
    "Household_ID": val_full["Household_ID"].values,
    "Date": (val_full.index if val_full.index.name == "Date"
             else val_full.get("Date", pd.Series(range(len(val_full)))).values),
    "y_true": y_val_raw,
}
test_preds_tuned = {"Household_ID": test_full["Household_ID"].values, "y_true": y_test_raw}

all_val_metrics  = []
all_test_metrics = []


def _eval_both_A(model, name: str, is_log: bool = False) -> None:
    vm, vp = _predict_and_eval(model, X_val_trees,  y_val_raw,  X_val_lin,  f"{name} [VAL]",  is_log)
    tm, tp = _predict_and_eval(model, X_test_trees, y_test_raw, X_test_lin, f"{name} [TEST]", is_log)
    all_val_metrics.append(vm)
    all_test_metrics.append(tm)
    val_preds_tuned[f"pred_{name.lower().replace(' ', '_')}"]  = vp
    test_preds_tuned[f"pred_{name.lower().replace(' ', '_')}"] = tp


def _eval_both_B(model, name: str) -> None:
    vm, _ = _predict_and_eval(model, X_val_trees_B,  y_val_raw_B,  name=f"{name} [VAL]")
    tm, _ = _predict_and_eval(model, X_test_trees_B, y_test_raw_B, name=f"{name} [TEST]")
    all_val_metrics.append(vm)
    all_test_metrics.append(tm)


# Track A — file-existence-based so prior-run models are included
_A_eval = [
    ("ElasticNet_tuned", MODEL_DIR / "model_elasticnet_tuned.pkl", True),
    ("DT_tuned",         MODEL_DIR / "model_dt_tuned.pkl",         False),
    ("RF_tuned",         MODEL_DIR / "model_rf_tuned.pkl",         False),
    ("XGBoost_tuned",    MODEL_DIR / "model_xgboost_tuned.pkl",    False),
    ("LightGBM_tuned",   MODEL_DIR / "model_lgbm_tuned.pkl",       False),
    ("ANN_tuned",        MODEL_DIR / "model_ann_tuned.pkl",        True),
]
for _name, _path, _is_log in _A_eval:
    if _path.exists():
        _m = joblib.load(_path)
        _eval_both_A(_m, _name, is_log=_is_log)
        del _m; gc.collect()

# Track B
_B_eval = [
    ("XGBoost_B_tuned", MODEL_DIR / "model_xgboost_b_tuned.pkl"),
    ("DT_B_tuned",      MODEL_DIR / "model_dt_B_tuned.pkl"),
    ("RF_B_tuned",      MODEL_DIR / "model_rf_B_tuned.pkl"),
]
for _name, _path in _B_eval:
    if _path.exists():
        _m = joblib.load(_path)
        _eval_both_B(_m, _name)
        del _m; gc.collect()

pd.DataFrame(val_preds_tuned).to_parquet(TABLE_DIR / "phase8_val_predictions.parquet", index=False)
pd.DataFrame(test_preds_tuned).to_parquet(TABLE_DIR / "phase8_test_predictions.parquet", index=False)
logger.info(f"Saved phase8_val_predictions.parquet ({len(val_preds_tuned['y_true']):,} rows)")
logger.info(f"Saved phase8_test_predictions.parquet ({len(test_preds_tuned['y_true']):,} rows)")


# ===========================================================================
# Task 8.11 — Optuna convergence plots (only for studies run this session)
# ===========================================================================
logger.info("\n--- Saving convergence plots ---")

studies_map: dict[str, optuna.Study] = {}
for _sname, _study in [
    ("elasticnet", en_study),
    ("dt",         dt_study),
    ("rf",         rf_study),
    ("xgboost",    xgb_study),
    ("lgbm",       lgbm_study),
    ("ann",        ann_study),
    ("xgboost_b",  xgb_b_study),
    ("dt_b",       dt_b_study),
    ("rf_b",       rf_b_study),
]:
    if _study is not None:
        studies_map[_sname] = _study

for name, study in studies_map.items():
    vals = [t.value for t in study.trials if t.value is not None]
    if not vals:
        continue
    best_so_far = [min(vals[:i+1]) for i in range(len(vals))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(vals)+1), vals, alpha=0.4, linewidth=1, label="Trial RMSE")
    ax.plot(range(1, len(best_so_far)+1), best_so_far, linewidth=2, color="red", label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("Val RMSE (kWh)")
    ax.set_title(f"Optuna Convergence — {name}")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / f"phase8_optuna_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

logger.info("Convergence plots saved.")


# ===========================================================================
# Task 8.12 — Training Report
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 8.12  Writing Phase 8 Training Report")
logger.info("=" * 60)

_w("=" * 70)
_w("HEAPO-Predict  Phase 8 — Hyperparameter Tuning Report")
_w(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
_w("=" * 70)

_w("\nSECTION 1 — TUNING BUDGET AND WALL-CLOCK TIME  (this session only)")
_budget = [
    ("ElasticNet",  T["n_trials_linear"], elapsed_en),
    ("DT",          T["n_trials_dt"],     elapsed_dt),
    ("RF",          T["n_trials_rf"],     elapsed_rf),
    ("XGBoost",     T["n_trials_xgb"],    elapsed_xgb),
    ("LightGBM",    T["n_trials_lgbm"],   elapsed_lgbm),
    ("ANN",         T["n_trials_ann"],    elapsed_ann),
    ("XGBoost_B",   T["n_trials_xgb_b"], elapsed_xgb_b),
    ("DT_B",        T["n_trials_dt_b"],   elapsed_dt_b),
    ("RF_B",        T["n_trials_rf_b"],   elapsed_rf_b),
]
for _mname, _n_trials, _elapsed in _budget:
    if _elapsed > 0:
        _w(f"  {_mname:<12}: {_n_trials:3d} trials  {_elapsed:6.0f}s")

_w("\nSECTION 2 — BEST HYPERPARAMETERS")
for model_name, params in best_params_all.items():
    _w(f"\n  [{model_name}]")
    for k, v in params.items():
        _w(f"    {k:25s} = {v}")

_w("\nSECTION 3 — VAL RMSE: PHASE 7 vs PHASE 8")
_w(f"  {'Model':<30}  {'P7 val RMSE':>12}  {'P8 val RMSE':>12}  {'Delta':>8}")
_w("  " + "-" * 68)
_study_rmse: dict[str, float] = {}
for _sname, _study in studies_map.items():
    _key_map = {
        "elasticnet": "ElasticNet", "dt": "DT", "rf": "RF",
        "xgboost": "XGBoost", "lgbm": "LightGBM", "ann": "ANN",
        "xgboost_b": "XGBoost_B", "dt_b": "DT_B", "rf_b": "RF_B",
    }
    _study_rmse[_key_map[_sname]] = _study.best_value

for mname, p7 in P7.items():
    if mname in _study_rmse:
        p8 = _study_rmse[mname]
        delta = p8 - p7
        arrow = "↓" if delta < 0 else "↑"
        _w(f"  {mname:<30}  {p7:>12.3f}  {p8:>12.3f}  {delta:>+7.3f} {arrow}")

_w("\nSECTION 4 — VALIDATION SET METRICS (Tuned Models)")
_w(f"  {'Model':<40}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}  {'sMAPE%':>8}")
_w("  " + "-" * 75)
for m in all_val_metrics:
    _w(f"  {m['model']:<40}  {m['rmse']:>7.3f}  {m['mae']:>7.3f}  {m['r2']:>7.4f}  {m['smape']:>7.2f}%")

_w("\nSECTION 5 — TEST SET METRICS (Final — run once)")
_w(f"  {'Model':<40}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}  {'sMAPE%':>8}")
_w("  " + "-" * 75)
for m in all_test_metrics:
    _w(f"  {m['model']:<40}  {m['rmse']:>7.3f}  {m['mae']:>7.3f}  {m['r2']:>7.4f}  {m['smape']:>7.2f}%")

_w("\nSECTION 6 — PHASE 9 READINESS CHECKLIST")
_checks = [
    ("best_params.json",               MODEL_DIR / "best_params.json"),
    ("model_elasticnet_tuned.pkl",     MODEL_DIR / "model_elasticnet_tuned.pkl"),
    ("model_dt_tuned.pkl",             MODEL_DIR / "model_dt_tuned.pkl"),
    ("model_rf_tuned.pkl",             MODEL_DIR / "model_rf_tuned.pkl"),
    ("model_xgboost_tuned.pkl",        MODEL_DIR / "model_xgboost_tuned.pkl"),
    ("model_lgbm_tuned.pkl",           MODEL_DIR / "model_lgbm_tuned.pkl"),
    ("model_ann_tuned.pkl",            MODEL_DIR / "model_ann_tuned.pkl"),
    ("model_xgboost_b_tuned.pkl",      MODEL_DIR / "model_xgboost_b_tuned.pkl"),
    ("model_dt_B_tuned.pkl",           MODEL_DIR / "model_dt_B_tuned.pkl"),
    ("model_rf_B_tuned.pkl",           MODEL_DIR / "model_rf_B_tuned.pkl"),
    ("phase8_val_predictions.parquet", TABLE_DIR  / "phase8_val_predictions.parquet"),
    ("phase8_test_predictions.parquet",TABLE_DIR  / "phase8_test_predictions.parquet"),
]
for label, path in _checks:
    tick = "✓" if path.exists() else "✗"
    _w(f"  [{tick}] {label}")

_w("\n" + "=" * 70)

report_path = TABLE_DIR / "phase8_tuning_report.txt"
report_path.write_text("\n".join(_report_lines), encoding="utf-8")
logger.info(f"Report saved → {report_path}")

# ── Final summary table to terminal ────────────────────────────────────────
if _study_rmse:
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 8 COMPLETE — VAL RMSE IMPROVEMENT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  {'Model':<28}  {'Phase 7':>8}  {'Phase 8':>8}  {'Δ RMSE':>8}")
    logger.info("  " + "-" * 58)
    for mname, p8 in _study_rmse.items():
        p7 = P7.get(mname, float("nan"))
        logger.info(f"  {mname:<28}  {p7:>8.3f}  {p8:>8.3f}  {p8-p7:>+7.3f}")

logger.info("=" * 70)
logger.info(f"Phase 8 complete: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
