"""
scripts/08_hyperparameter_tuning.py

Phase 8 — Hyperparameter Tuning via Optuna Bayesian Optimisation.

Tunes:
  Track A  — ElasticNet, Decision Tree, Random Forest, XGBoost, LightGBM, ANN
  Track B  — XGBoost_B (protocol-enriched, 109 HH)

Objective for every model: minimise val-set RMSE in raw kWh space.
Test set is NEVER used inside an Optuna objective.

Run from repo root:
    cd scripts && python 08_hyperparameter_tuning.py
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
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

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

def rline(s: str = "") -> None:
    logger.info(s)
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

FEATURES_TREES   = feat["FEATURES_TREES"]
FEATURES_LINEAR  = feat["FEATURES_LINEAR"]
FEATURES_TREES_B = feat["FEATURES_TREES_B"]
FEATURES_LINEAR_B = feat["FEATURES_LINEAR_B"]
TARGET_RAW = feat["TARGET_RAW"]
TARGET_LOG = feat["TARGET_LOG"]

scaler_A = joblib.load(MODEL_DIR / "scaler_linear_A.pkl")
scaler_B = joblib.load(MODEL_DIR / "scaler_linear_B.pkl")

# ── feature matrices ───────────────────────────────────────────────────────
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

# Track B
X_train_trees_B = train_prot[FEATURES_TREES_B].values
X_val_trees_B   = val_prot[FEATURES_TREES_B].values
X_test_trees_B  = test_prot[FEATURES_TREES_B].values
y_train_raw_B   = train_prot[TARGET_RAW].values
y_val_raw_B     = val_prot[TARGET_RAW].values
y_test_raw_B    = test_prot[TARGET_RAW].values

logger.info(f"Track A  train: {X_train_trees.shape[0]:,} rows  |  val: {X_val_trees.shape[0]:,}  |  test: {X_test_trees.shape[0]:,}")
logger.info(f"Track B  train: {X_train_trees_B.shape[0]:,} rows  |  val: {X_val_trees_B.shape[0]:,}  |  test: {X_test_trees_B.shape[0]:,}")

# Phase 7 baseline RMSEs (for delta logging)
P7 = {
    "ElasticNet":   12.185,
    "DT":           11.382,
    "RF":            9.421,
    "XGBoost":       9.462,
    "LightGBM":      9.318,
    "ANN":          10.330,
    "XGBoost_B":     5.933,
}

best_params_all: dict[str, dict] = {}
tuned_metrics:   dict[str, dict] = {}

# ── helper: log trial result ───────────────────────────────────────────────
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
logger.info("\n" + "=" * 60)
logger.info("Task 8.1  ElasticNet Hyperparameter Tuning")
logger.info("=" * 60)

from sklearn.linear_model import ElasticNet as _EN

def _en_objective(trial: optuna.Trial) -> float:
    alpha    = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)
    model = _EN(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=SEED)
    model.fit(X_train_lin, y_train_log)
    pred = np.expm1(model.predict(X_val_lin)).clip(0)
    return _rmse(y_val_raw, pred)

en_study = optuna.create_study(
    study_name="phase8_elasticnet",
    storage=STORAGE,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    load_if_exists=True,
)
t0 = time.perf_counter()
_n = max(0, T["n_trials_linear"] - len(en_study.trials))
if _n: en_study.optimize(_en_objective, n_trials=_n, callbacks=[_trial_cb], show_progress_bar=False)
elapsed_en = time.perf_counter() - t0

best_params_all["ElasticNet"] = en_study.best_params
logger.info(f"ElasticNet best: {en_study.best_params}  val_RMSE={en_study.best_value:.4f}  ({elapsed_en:.0f}s)")


# ===========================================================================
# Task 8.2 — Decision Tree (40 trials)
# ===========================================================================
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
    study_name="phase8_dt",
    storage=STORAGE,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    load_if_exists=True,
)
t0 = time.perf_counter()
_n = max(0, T["n_trials_dt"] - len(dt_study.trials))
if _n: dt_study.optimize(_dt_objective, n_trials=_n, callbacks=[_trial_cb], show_progress_bar=False)
elapsed_dt = time.perf_counter() - t0

best_params_all["DT"] = dt_study.best_params
logger.info(f"DT best: {dt_study.best_params}  val_RMSE={dt_study.best_value:.4f}  ({elapsed_dt:.0f}s)")


# ===========================================================================
# Task 8.3 — XGBoost Track B (40 trials) — small dataset, fast
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 8.3  XGBoost Track B Hyperparameter Tuning")
logger.info("=" * 60)

import xgboost as xgb

def _xgb_b_objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators       = 2000,
        max_depth          = trial.suggest_int("max_depth", 3, 7),
        learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample          = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree   = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight   = trial.suggest_int("min_child_weight", 1, 10),
        reg_alpha          = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda         = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        gamma              = trial.suggest_float("gamma", 0.0, 3.0),
        early_stopping_rounds = ES_ROUNDS,
        tree_method        = "hist",
        random_state       = SEED,
        n_jobs             = -1,
    )
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_trees_B, y_train_raw_B,
              eval_set=[(X_val_trees_B, y_val_raw_B)], verbose=False)
    return _rmse(y_val_raw_B, model.predict(X_val_trees_B))

xgb_b_study = optuna.create_study(
    study_name="phase8_xgboost_b",
    storage=STORAGE,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    load_if_exists=True,
)
t0 = time.perf_counter()
_n = max(0, T["n_trials_xgb_b"] - len(xgb_b_study.trials))
if _n: xgb_b_study.optimize(_xgb_b_objective, n_trials=_n, callbacks=[_trial_cb], show_progress_bar=False)
elapsed_xgb_b = time.perf_counter() - t0

best_params_all["XGBoost_B"] = xgb_b_study.best_params
logger.info(f"XGBoost_B best: {xgb_b_study.best_params}  val_RMSE={xgb_b_study.best_value:.4f}  ({elapsed_xgb_b:.0f}s)")


# ===========================================================================
# Task 8.4 — XGBoost Track A (80 trials)
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 8.4  XGBoost Track A Hyperparameter Tuning")
logger.info("=" * 60)

def _xgb_objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators       = 2000,
        max_depth          = trial.suggest_int("max_depth", 3, 10),
        learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample          = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree   = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        min_child_weight   = trial.suggest_int("min_child_weight", 1, 15),
        reg_alpha          = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda         = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        gamma              = trial.suggest_float("gamma", 0.0, 5.0),
        early_stopping_rounds = ES_ROUNDS,
        tree_method        = "hist",
        random_state       = SEED,
        n_jobs             = -1,
    )
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_trees, y_train_raw,
              eval_set=[(X_val_trees, y_val_raw)], verbose=False)
    return _rmse(y_val_raw, model.predict(X_val_trees))

xgb_study = optuna.create_study(
    study_name="phase8_xgboost",
    storage=STORAGE,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
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
logger.info("\n" + "=" * 60)
logger.info("Task 8.5  LightGBM Track A Hyperparameter Tuning")
logger.info("=" * 60)

import lightgbm as lgb

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
    study_name="phase8_lgbm",
    storage=STORAGE,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
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
logger.info("\n" + "=" * 60)
logger.info("Task 8.6  ANN Hyperparameter Tuning")
logger.info("=" * 60)

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as _mse

def _ann_objective(trial: optuna.Trial) -> float:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_sizes = tuple(
        trial.suggest_categorical(f"n_units_l{i}", [32, 64, 128, 256])
        for i in range(n_layers)
    )
    lr     = trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True)
    alpha  = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
    batch  = trial.suggest_categorical("batch_size", [128, 256, 512])

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

        SAMPLE = 10_000
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_train_lin), size=min(SAMPLE, len(X_train_lin)), replace=False)
        X_s, y_s = X_train_lin[idx], y_train_log[idx]

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
    study_name="phase8_ann",
    storage=STORAGE,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    load_if_exists=True,
)
t0 = time.perf_counter()
_n = max(0, T["n_trials_ann"] - len(ann_study.trials))
if _n: ann_study.optimize(_ann_objective, n_trials=_n, callbacks=[_trial_cb], show_progress_bar=False)
elapsed_ann = time.perf_counter() - t0

best_params_all["ANN"] = ann_study.best_params
logger.info(f"ANN best: {ann_study.best_params}  val_RMSE={ann_study.best_value:.4f}  ({elapsed_ann:.0f}s)")


# ===========================================================================
# Task 8.7 — Random Forest (60 trials) — last because it's the heaviest
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 8.7  Random Forest Hyperparameter Tuning  (n_estimators=150 during search)")
logger.info("=" * 60)

from sklearn.ensemble import RandomForestRegressor

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
    del model
    gc.collect()
    return rmse

rf_study = optuna.create_study(
    study_name="phase8_rf",
    storage=STORAGE,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    load_if_exists=True,
)
t0 = time.perf_counter()
_n = max(0, T["n_trials_rf"] - len(rf_study.trials))
if _n: rf_study.optimize(_rf_objective, n_trials=_n, callbacks=[_trial_cb], show_progress_bar=False)
elapsed_rf = time.perf_counter() - t0

best_params_all["RF"] = rf_study.best_params
logger.info(f"RF best: {rf_study.best_params}  val_RMSE={rf_study.best_value:.4f}  ({elapsed_rf:.0f}s)")


# ===========================================================================
# Task 8.8 — Save best params JSON
# ===========================================================================
with open(MODEL_DIR / "best_params.json", "w") as f:
    json.dump(best_params_all, f, indent=2, default=str)
logger.info(f"\nSaved best_params.json → {MODEL_DIR / 'best_params.json'}")


# ===========================================================================
# Task 8.9 — Retrain with best hyperparameters on full training set
# ===========================================================================
logger.info("\n" + "=" * 60)
logger.info("Task 8.9  Final Retraining with Best Hyperparameters")
logger.info("=" * 60)

# ── ElasticNet ────────────────────────────────────────────────────────────
from sklearn.linear_model import ElasticNet
en_tuned = ElasticNet(**best_params_all["ElasticNet"], max_iter=5000, random_state=SEED)
en_tuned.fit(X_train_lin, y_train_log)
joblib.dump(en_tuned, MODEL_DIR / "model_elasticnet_tuned.pkl")
logger.info("Retrained ElasticNet_tuned ✓")

# ── Decision Tree ─────────────────────────────────────────────────────────
dt_tuned = DecisionTreeRegressor(**best_params_all["DT"], random_state=SEED)
dt_tuned.fit(X_train_trees, y_train_raw)
joblib.dump(dt_tuned, MODEL_DIR / "model_dt_tuned.pkl")
logger.info("Retrained DT_tuned ✓")

# ── Random Forest (n_estimators=500 for final) ────────────────────────────
use_oob = best_params_all["RF"].get("bootstrap", True)
rf_params = {**best_params_all["RF"], "n_estimators": T["rf_n_estimators_final"],
             "n_jobs": -1, "random_state": SEED, "oob_score": use_oob, "verbose": 1}
logger.info(f"Fitting RF_tuned with n_estimators={T['rf_n_estimators_final']} (best arch + full tree budget)...")
rf_tuned = RandomForestRegressor(**rf_params)
rf_tuned.fit(X_train_trees, y_train_raw)
if use_oob:
    logger.info(f"RF_tuned OOB R²={rf_tuned.oob_score_:.4f}")
else:
    logger.info("RF_tuned fitted (bootstrap=False, no OOB score)")
joblib.dump(rf_tuned, MODEL_DIR / "model_rf_tuned.pkl")
logger.info("Retrained RF_tuned ✓")
del rf_tuned; gc.collect()   # free RAM before XGBoost

# ── XGBoost Track A ───────────────────────────────────────────────────────
# Use best_iteration × 1.2 from best trial as fixed n_estimators (no early stop on final)
xgb_best_trial = xgb_study.best_trial
_xgb_params = {**best_params_all["XGBoost"],
               "n_estimators": 2000, "early_stopping_rounds": ES_ROUNDS,
               "tree_method": "hist", "random_state": SEED, "n_jobs": -1}
xgb_tuned = xgb.XGBRegressor(**_xgb_params)
xgb_tuned.fit(X_train_trees, y_train_raw,
              eval_set=[(X_val_trees, y_val_raw)], verbose=100)
joblib.dump(xgb_tuned, MODEL_DIR / "model_xgboost_tuned.pkl")
xgb_tuned.save_model(str(MODEL_DIR / "model_xgboost_tuned.json"))
logger.info("Retrained XGBoost_tuned ✓")

# ── LightGBM Track A ──────────────────────────────────────────────────────
_lgbm_params = {**best_params_all["LightGBM"],
                "n_estimators": 3000, "random_state": SEED, "n_jobs": -1, "verbose": -1}
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

# ── ANN ───────────────────────────────────────────────────────────────────
ann_bp = best_params_all["ANN"]
n_layers_best = ann_bp["n_layers"]
best_layers = tuple(ann_bp[f"n_units_l{i}"] for i in range(n_layers_best))
logger.info(f"Retraining ANN_tuned: architecture={best_layers}  lr={ann_bp['learning_rate_init']:.5f}  alpha={ann_bp['alpha']:.2e}  batch={ann_bp['batch_size']}")

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

# ANN loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ann_train_losses, label="Train MSE (log)", linewidth=1.5)
ax.plot(ann_val_losses,   label="Val MSE (log)",   linewidth=1.5)
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (log kWh)")
ax.set_title("ANN Tuned — Training and Validation Loss")
ax.legend(); ax.set_yscale("log"); fig.tight_layout()
fig.savefig(FIG_DIR / "phase8_ann_tuned_loss.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── XGBoost Track B ───────────────────────────────────────────────────────
_xgb_b_params = {**best_params_all["XGBoost_B"],
                 "n_estimators": 2000, "early_stopping_rounds": ES_ROUNDS,
                 "tree_method": "hist", "random_state": SEED, "n_jobs": -1}
xgb_b_tuned = xgb.XGBRegressor(**_xgb_b_params)
xgb_b_tuned.fit(X_train_trees_B, y_train_raw_B,
                eval_set=[(X_val_trees_B, y_val_raw_B)], verbose=100)
joblib.dump(xgb_b_tuned, MODEL_DIR / "model_xgboost_b_tuned.pkl")
xgb_b_tuned.save_model(str(MODEL_DIR / "model_xgboost_b_tuned.json"))
logger.info("Retrained XGBoost_B_tuned ✓")

# reload RF (was deleted to free RAM)
rf_tuned = joblib.load(MODEL_DIR / "model_rf_tuned.pkl")


# ===========================================================================
# Task 8.10 — Evaluate all tuned models on val and test sets
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

val_preds_tuned  = {"Household_ID": val_full["Household_ID"].values,
                    "Date":         val_full.index if val_full.index.name == "Date" else val_full.get("Date", pd.Series(range(len(val_full)))).values,
                    "y_true":       y_val_raw}
test_preds_tuned = {"Household_ID": test_full["Household_ID"].values,
                    "y_true":       y_test_raw}

all_val_metrics  = []
all_test_metrics = []

# Helper to collect both val and test
def _eval_both(model, name, is_log=False):
    vm, vp = _predict_and_eval(model, X_val_trees,  y_val_raw,  X_val_lin,  f"{name} [VAL]",  is_log)
    tm, tp = _predict_and_eval(model, X_test_trees, y_test_raw, X_test_lin, f"{name} [TEST]", is_log)
    all_val_metrics.append(vm);  val_preds_tuned[f"pred_{name.lower().replace(' ', '_')}"]  = vp
    all_test_metrics.append(tm); test_preds_tuned[f"pred_{name.lower().replace(' ', '_')}"] = tp

_eval_both(en_tuned,   "ElasticNet_tuned", is_log=True)
_eval_both(dt_tuned,   "DT_tuned",         is_log=False)
_eval_both(rf_tuned,   "RF_tuned",         is_log=False)
_eval_both(xgb_tuned,  "XGBoost_tuned",    is_log=False)
_eval_both(lgbm_tuned, "LightGBM_tuned",   is_log=False)
_eval_both(ann_tuned,  "ANN_tuned",        is_log=True)

# Track B
vm_b, vp_b = _predict_and_eval(xgb_b_tuned, X_val_trees_B,  y_val_raw_B,  name="XGBoost_B_tuned [VAL]")
tm_b, tp_b = _predict_and_eval(xgb_b_tuned, X_test_trees_B, y_test_raw_B, name="XGBoost_B_tuned [TEST]")
all_val_metrics.append(vm_b)
all_test_metrics.append(tm_b)

# Save predictions parquet
pd.DataFrame(val_preds_tuned).to_parquet(TABLE_DIR / "phase8_val_predictions.parquet", index=False)
pd.DataFrame(test_preds_tuned).to_parquet(TABLE_DIR / "phase8_test_predictions.parquet", index=False)
logger.info(f"Saved phase8_val_predictions.parquet ({len(val_preds_tuned['y_true']):,} rows)")
logger.info(f"Saved phase8_test_predictions.parquet ({len(test_preds_tuned['y_true']):,} rows)")


# ===========================================================================
# Task 8.11 — Optuna convergence plots
# ===========================================================================
logger.info("\n--- Saving convergence plots ---")

studies_map = {
    "elasticnet": en_study,
    "dt":         dt_study,
    "rf":         rf_study,
    "xgboost":    xgb_study,
    "lgbm":       lgbm_study,
    "ann":        ann_study,
    "xgboost_b":  xgb_b_study,
}

for name, study in studies_map.items():
    vals = [t.value for t in study.trials if t.value is not None]
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

def _w(s=""):
    _report_lines.append(s)

_w("=" * 70)
_w("HEAPO-Predict  Phase 8 — Hyperparameter Tuning Report")
_w(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
_w("=" * 70)

_w("\nSECTION 1 — TUNING BUDGET AND WALL-CLOCK TIME")
_w(f"  ElasticNet  : {T['n_trials_linear']:3d} trials  {elapsed_en:6.0f}s")
_w(f"  DT          : {T['n_trials_dt']:3d} trials  {elapsed_dt:6.0f}s")
_w(f"  XGBoost_B   : {T['n_trials_xgb_b']:3d} trials  {elapsed_xgb_b:6.0f}s")
_w(f"  XGBoost     : {T['n_trials_xgb']:3d} trials  {elapsed_xgb:6.0f}s")
_w(f"  LightGBM    : {T['n_trials_lgbm']:3d} trials  {elapsed_lgbm:6.0f}s")
_w(f"  ANN         : {T['n_trials_ann']:3d} trials  {elapsed_ann:6.0f}s")
_w(f"  RF          : {T['n_trials_rf']:3d} trials  {elapsed_rf:6.0f}s")

_w("\nSECTION 2 — BEST HYPERPARAMETERS")
for model_name, params in best_params_all.items():
    _w(f"\n  [{model_name}]")
    for k, v in params.items():
        _w(f"    {k:25s} = {v}")

_w("\nSECTION 3 — VAL RMSE: PHASE 7 vs PHASE 8")
_w(f"  {'Model':<30}  {'P7 val RMSE':>12}  {'P8 val RMSE':>12}  {'Delta':>8}")
_w("  " + "-" * 68)
study_rmse = {
    "ElasticNet":  en_study.best_value,
    "DT":          dt_study.best_value,
    "RF":          rf_study.best_value,
    "XGBoost":     xgb_study.best_value,
    "LightGBM":    lgbm_study.best_value,
    "ANN":         ann_study.best_value,
    "XGBoost_B":   xgb_b_study.best_value,
}
for mname, p7 in P7.items():
    p8 = study_rmse.get(mname, float("nan"))
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
checks = [
    ("best_params.json", MODEL_DIR / "best_params.json"),
    ("model_xgboost_tuned.pkl", MODEL_DIR / "model_xgboost_tuned.pkl"),
    ("model_lgbm_tuned.pkl",    MODEL_DIR / "model_lgbm_tuned.pkl"),
    ("model_rf_tuned.pkl",      MODEL_DIR / "model_rf_tuned.pkl"),
    ("model_ann_tuned.pkl",     MODEL_DIR / "model_ann_tuned.pkl"),
    ("model_elasticnet_tuned.pkl", MODEL_DIR / "model_elasticnet_tuned.pkl"),
    ("model_dt_tuned.pkl",      MODEL_DIR / "model_dt_tuned.pkl"),
    ("model_xgboost_b_tuned.pkl", MODEL_DIR / "model_xgboost_b_tuned.pkl"),
    ("phase8_val_predictions.parquet",  TABLE_DIR / "phase8_val_predictions.parquet"),
    ("phase8_test_predictions.parquet", TABLE_DIR / "phase8_test_predictions.parquet"),
]
for label, path in checks:
    tick = "✓" if path.exists() else "✗"
    _w(f"  [{tick}] {label}")

_w("\n" + "=" * 70)

report_path = TABLE_DIR / "phase8_tuning_report.txt"
report_path.write_text("\n".join(_report_lines), encoding="utf-8")
logger.info(f"Report saved → {report_path}")

# ── Final summary table to terminal ────────────────────────────────────────
logger.info("\n" + "=" * 70)
logger.info("PHASE 8 COMPLETE — VAL RMSE IMPROVEMENT SUMMARY")
logger.info("=" * 70)
logger.info(f"  {'Model':<28}  {'Phase 7':>8}  {'Phase 8':>8}  {'Δ RMSE':>8}")
logger.info("  " + "-" * 58)
for mname, p7 in P7.items():
    p8 = study_rmse.get(mname, float("nan"))
    logger.info(f"  {mname:<28}  {p7:>8.3f}  {p8:>8.3f}  {p8-p7:>+7.3f}")
logger.info("=" * 70)
logger.info(f"Phase 8 complete: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
