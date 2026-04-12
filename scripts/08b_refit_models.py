"""
scripts/08b_refit_models.py

Re-fit all tree models with the best hyperparameters from best_params.json
on the CURRENT data/processed/ parquets.

Background: Phase 6 was re-run after Phase 7/8 training, producing new
parquets with different train/val/test distributions. The tree models in
outputs/models/ were trained on the old parquets and are incompatible with
the current data. This script performs the final-refit step from Phase 8
(Task 8.8) using the already-identified hyperparameters.

The ANN and ElasticNet are NOT re-trained here because they use the scaler
which was fitted on the current data and are already working correctly.

Outputs (overwrites broken models):
  outputs/models/model_rf_tuned.pkl
  outputs/models/model_xgboost_tuned.pkl + .json
  outputs/models/model_lgbm_tuned.pkl + .txt
  outputs/models/model_dt_tuned.pkl
  outputs/models/model_xgboost_b_tuned.pkl + .json
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.data_loader import load_config

DATA_DIR  = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
LOG_DIR   = PROJECT_ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
def setup_logging():
    log_path = LOG_DIR / "phase8b_refit.log"
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


# ── Feature lists ──────────────────────────────────────────────────────────
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
]   # 45 features — verified against RF/XGB/LGBM n_features_in_


def main():
    t0 = time.time()
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Phase 8b — Re-fit tree models on current parquets")
    logger.info("=" * 60)

    best_params = json.loads((MODEL_DIR / "best_params.json").read_text())
    cfg = load_config("config/params.yaml")
    seed = cfg["modeling"]["random_seed"]

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading parquets...")
    df_train   = pd.read_parquet(DATA_DIR / "train_full.parquet")
    df_val     = pd.read_parquet(DATA_DIR / "val_full.parquet")
    df_b_train = pd.read_parquet(DATA_DIR / "train_protocol.parquet")
    df_b_val   = pd.read_parquet(DATA_DIR / "val_protocol.parquet")

    # Derive Track B features (first 75 numeric cols, same as original training)
    _b_exclude = {
        "Household_ID", "Date", "kWh_received_Total", "kWh_log1p",
        "kWh_received_HeatPump", "kWh_received_Other", "kWh_returned_Total",
        "Group", "AffectsTimePoint", "Timestamp", "Weather_ID", "cv_fold",
        "post_intervention",
        "kvarh_received_capacitive_HeatPump", "kvarh_received_capacitive_Other",
        "kvarh_received_inductive_HeatPump", "kvarh_received_inductive_Other",
    }
    features_b = [
        c for c in df_b_train.select_dtypes(include="number").columns
        if c not in _b_exclude
    ][:75]

    X_train = df_train[FEATURES_TREES].values
    y_train = df_train["kWh_received_Total"].values
    X_val   = df_val[FEATURES_TREES].values
    y_val   = df_val["kWh_received_Total"].values

    X_b_train = df_b_train[features_b].values
    y_b_train = df_b_train["kWh_received_Total"].values
    X_b_val   = df_b_val[features_b].values
    y_b_val   = df_b_val["kWh_received_Total"].values

    from sklearn.metrics import mean_squared_error, r2_score

    def _eval(name, model, X_v, y_v):
        pred = np.clip(model.predict(X_v), 0, None)
        rmse = float(np.sqrt(mean_squared_error(y_v, pred)))
        r2   = float(r2_score(y_v, pred))
        logger.info("  %s  val RMSE=%.3f  R2=%.4f", name, rmse, r2)
        return rmse, r2

    # ── Decision Tree (already re-fitted at 03:24, skip) ─────────────────
    logger.info("Skipping DT — already re-fitted (model_dt_tuned.pkl Apr 12 03:24)")

    # ── Random Forest ─────────────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Re-fitting Random Forest (n_estimators=500)...")
    from sklearn.ensemble import RandomForestRegressor
    rf_params = {k: v for k, v in best_params["RF"].items()}
    logger.info("  Training RF on %d rows × %d features...", X_train.shape[0], X_train.shape[1])
    rf = RandomForestRegressor(
        **rf_params,
        n_estimators=500,
        n_jobs=-1,
        random_state=seed,
        verbose=1,
    )
    rf.fit(X_train, y_train)
    _eval("RF", rf, X_val, y_val)
    joblib.dump(rf, MODEL_DIR / "model_rf_tuned.pkl")
    logger.info("  Saved model_rf_tuned.pkl")

    # ── XGBoost Track A ───────────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Re-fitting XGBoost (Track A)...")
    from xgboost import XGBRegressor
    early_stop = cfg["modeling"]["xgboost_early_stopping_rounds"]
    xgb_params = {k: v for k, v in best_params["XGBoost"].items()}
    xgb = XGBRegressor(
        **xgb_params,
        n_estimators=2000,
        early_stopping_rounds=early_stop,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("  Best iteration: %d", xgb.best_iteration)
    _eval("XGBoost", xgb, X_val, y_val)
    joblib.dump(xgb, MODEL_DIR / "model_xgboost_tuned.pkl")
    xgb.save_model(str(MODEL_DIR / "model_xgboost_tuned.json"))
    logger.info("  Saved model_xgboost_tuned.pkl + .json")

    # ── LightGBM ──────────────────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Re-fitting LightGBM...")
    import lightgbm as lgb
    lgbm_params = {k: v for k, v in best_params["LightGBM"].items()}
    lgbm = lgb.LGBMRegressor(
        **lgbm_params,
        n_estimators=3000,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    logger.info("  Best iteration: %d", lgbm.best_iteration_)
    _eval("LightGBM", lgbm, X_val, y_val)
    joblib.dump(lgbm, MODEL_DIR / "model_lgbm_tuned.pkl")
    lgbm.booster_.save_model(str(MODEL_DIR / "model_lgbm_tuned.txt"))
    logger.info("  Saved model_lgbm_tuned.pkl + .txt")

    # ── XGBoost Track B ───────────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Re-fitting XGBoost Track B...")
    xgb_b_params = {k: v for k, v in best_params["XGBoost_B"].items()}
    xgb_b = XGBRegressor(
        **xgb_b_params,
        n_estimators=2000,
        early_stopping_rounds=early_stop,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )
    xgb_b.fit(
        X_b_train, y_b_train,
        eval_set=[(X_b_val, y_b_val)],
        verbose=False,
    )
    logger.info("  Best iteration: %d", xgb_b.best_iteration)
    pred_b = np.clip(xgb_b.predict(X_b_val), 0, None)
    rmse_b = float(np.sqrt(mean_squared_error(y_b_val, pred_b)))
    r2_b   = float(r2_score(y_b_val, pred_b))
    logger.info("  XGBoost_B  val RMSE=%.3f  R2=%.4f", rmse_b, r2_b)
    joblib.dump(xgb_b, MODEL_DIR / "model_xgboost_b_tuned.pkl")
    xgb_b.save_model(str(MODEL_DIR / "model_xgboost_b_tuned.json"))
    logger.info("  Saved model_xgboost_b_tuned.pkl + .json")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Phase 8b complete.  Elapsed: %.0f s", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
