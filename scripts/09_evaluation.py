"""
scripts/09_evaluation.py

Phase 9 -- Model Evaluation
============================

Inputs  : data/processed/{train,val,test}_{full,protocol}.parquet
          outputs/models/  -- tuned model pickles + scalers + best_params.json
Outputs : outputs/tables/  -- metrics CSVs + evaluation report
          outputs/figures/ -- diagnostic PNGs
          outputs/logs/phase9_run.log
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
from scipy.stats import wilcoxon
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

# ── Project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.data_loader import load_config
from src.evaluation import (
    assert_predictions_valid,
    compute_all_metrics,
    plot_ablation_barplot,
    plot_cv_errorbar,
    plot_data_volume_scatter,
    plot_predicted_vs_actual,
    plot_residual_histogram,
    plot_residuals_vs_predicted,
    plot_seasonal_barplot,
    plot_significance_heatmap,
    plot_timeseries,
    plot_timeseries_comparison,
    predict_raw,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = PROJECT_ROOT / "data" / "processed"
MODEL_DIR  = PROJECT_ROOT / "outputs" / "models"
TABLE_DIR  = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR    = PROJECT_ROOT / "outputs" / "figures"
LOG_DIR    = PROJECT_ROOT / "outputs" / "logs"

for d in (TABLE_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    log_path = LOG_DIR / "phase9_run.log"
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
# Feature lists (derived from model artefacts)
# ─────────────────────────────────────────────────────────────────────────────
_ALL_51 = [
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
    # last 6 were excluded from the fitted tree models (RF/XGB/LGBM use 45)
    "has_freezer", "power_factor_proxy", "season_encoded",
    "living_area_bucket_encoded",
    "Survey_Building_LivingArea_imputed", "Survey_Building_Residents_imputed",
]

FEATURES_TREES   = _ALL_51[:45]   # 45 features — matches RF / XGB / LGBM n_features_in_
FEATURES_LINEAR  = None           # loaded from scaler meta below

# Weather + temporal only (ablation A — no Survey_* / household metadata)
_WEATHER_TEMPORAL_EXCLUDE = {
    "Survey_Building_LivingArea", "Survey_Building_Residents",
    "building_type_house", "building_type_apartment",
    "hp_type_air_source", "hp_type_ground_source", "hp_type_unknown",
    "dhw_hp", "dhw_ewh", "dhw_solar", "dhw_combined", "dhw_unknown",
    "heat_dist_floor", "heat_dist_radiator", "heat_dist_both", "heat_dist_unknown",
    "has_ev", "has_dryer",
    "has_pv",  # derived from SMD but considered household-level
}
FEATURES_ABLATION_A = [f for f in FEATURES_TREES if f not in _WEATHER_TEMPORAL_EXCLUDE]

# Track B: first 75 numeric columns in train_protocol (confirmed matching XGB_B)
FEATURES_TREES_B = None   # populated during data loading


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load a model and scaler once, cache for reuse
# ─────────────────────────────────────────────────────────────────────────────
def _load_model(name: str):
    path = MODEL_DIR / f"model_{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Task 9.2: Baseline predictions
# ─────────────────────────────────────────────────────────────────────────────
def make_baseline_predictions(
    df_train: pd.DataFrame,
    df_target: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Return dict of {baseline_name: predictions} for df_target."""
    y_train = df_train["kWh_received_Total"].values

    # Global mean
    global_mean = float(y_train.mean())
    pred_global = np.full(len(df_target), global_mean)

    # Per-household mean (unknown households fall back to global mean)
    hh_means = (
        df_train.groupby("Household_ID")["kWh_received_Total"]
        .mean()
        .rename("hh_mean")
    )
    pred_hh = (
        df_target[["Household_ID"]]
        .merge(hh_means, on="Household_ID", how="left")["hh_mean"]
        .fillna(global_mean)
        .values
    )

    # HDD-linear baseline (pre-fitted in Phase 7)
    hdd_baseline_path = MODEL_DIR / "baseline_hdd_linear.pkl"
    if hdd_baseline_path.exists():
        hdd_model = joblib.load(hdd_baseline_path)
        X_hdd = df_target[["HDD_SIA_daily"]].values
        pred_hdd = np.clip(hdd_model.predict(X_hdd), 0, None)
    else:
        pred_hdd = pred_global.copy()
        logging.getLogger(__name__).warning(
            "baseline_hdd_linear.pkl not found — using global mean as HDD-Linear fallback"
        )

    return {
        "Baseline: Global Mean": pred_global,
        "Baseline: Per-HH Mean": pred_hh,
        "Baseline: HDD-Linear":  pred_hdd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 9.3: Generate all tuned model predictions
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_predictions(
    df: pd.DataFrame,
    models: dict,
    scaler_a,
    linear_feats: list,
) -> dict[str, np.ndarray]:
    """Return {model_name: raw_kWh_predictions} for a given DataFrame."""
    logger = logging.getLogger(__name__)
    preds = {}

    X_trees  = df[FEATURES_TREES].values
    X_linear = df[linear_feats].values

    for name, mdl in models.items():
        log_target = name in ("ElasticNet", "ANN")
        scaler     = scaler_a if log_target else None
        X          = X_linear if log_target else X_trees

        logger.info("  Predicting: %s", name)
        y_pred = predict_raw(mdl, X, log_target=log_target, scaler=scaler)
        assert_predictions_valid(y_pred, name)
        preds[name] = y_pred

    return preds


def generate_b_predictions(
    df_b: pd.DataFrame,
    xgb_b,
    features_b: list,
) -> np.ndarray:
    X = df_b[features_b].values
    y_pred = predict_raw(xgb_b, X, log_target=False, scaler=None)
    assert_predictions_valid(y_pred, "XGBoost B")
    return y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Task 9.3: Compute metrics table
# ─────────────────────────────────────────────────────────────────────────────
def build_metrics_table(
    preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    floor_kwh: float,
    track: str = "A",
) -> pd.DataFrame:
    rows = []
    for name, yp in preds.items():
        m = compute_all_metrics(y_true, yp, floor_kwh)
        m["Model"] = name
        m["Track"] = track
        rows.append(m)
    df = pd.DataFrame(rows)[["Model", "RMSE", "MAE", "R2", "sMAPE", "MedAE", "N", "Track"]]
    return df.sort_values("RMSE").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Task 9.5: Seasonal breakdown
# ─────────────────────────────────────────────────────────────────────────────
def compute_seasonal_metrics(
    df: pd.DataFrame,
    preds: dict[str, np.ndarray],
    floor_kwh: float,
    split_label: str,
) -> pd.DataFrame:
    """Compute metrics per period within a given split (val or test)."""
    rows = []

    def _period_label(m: int) -> str:
        if m in (12, 1, 2):
            return "Peak Winter (Dec-Feb)"
        elif m in (3, 4):
            return "Shoulder (Mar-Apr)"
        elif m in (5, 6, 7, 8, 9):
            return "Non-Heating (May-Sep)"
        else:
            return "Transition (Oct-Nov)"

    # Use Date column (timezone-aware) to extract month safely
    dates = pd.to_datetime(df["Date"]).dt.month.values

    for name, yp in preds.items():
        for period_fn, period_name in [
            (lambda m: m >= 1,                      f"{split_label} Overall"),
            (lambda m: m in [12, 1, 2],             "Peak Winter (Dec-Feb)"),
            (lambda m: m in [3, 4],                 "Shoulder (Mar-Apr)"),
            (lambda m: m in [5, 6, 7, 8, 9],        "Non-Heating (May-Sep)"),
            (lambda m: m in [10, 11],               "Transition (Oct-Nov)"),
        ]:
            mask = np.array([period_fn(mo) for mo in dates])
            if mask.sum() < 10:
                continue
            m = compute_all_metrics(
                df["kWh_received_Total"].values[mask],
                yp[mask],
                floor_kwh,
            )
            m["Model"]  = name
            m["Split"]  = split_label
            m["Period"] = period_name
            rows.append(m)

    return pd.DataFrame(rows)[["Model", "Split", "Period", "RMSE", "MAE", "R2", "sMAPE", "MedAE", "N"]]


# ─────────────────────────────────────────────────────────────────────────────
# Task 9.6: Cross-validation
# ─────────────────────────────────────────────────────────────────────────────
def run_cross_validation(
    df_train: pd.DataFrame,
    models_cfg: dict,
    scaler_a,
    linear_feats: list,
    n_splits: int,
    floor_kwh: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """5-fold GroupKFold CV on train_full using tuned hyperparameters."""
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    best_params = json.loads((MODEL_DIR / "best_params.json").read_text())
    groups  = df_train["Household_ID"].values
    y_raw   = df_train["kWh_received_Total"].values
    y_log   = df_train["kWh_log1p"].values
    X_trees = df_train[FEATURES_TREES].values
    X_lin   = df_train[linear_feats].values

    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    model_builders = {
        "ElasticNet": lambda: ElasticNet(
            **best_params["ElasticNet"], max_iter=5000, random_state=42
        ),
        "DT": lambda: DecisionTreeRegressor(
            **best_params["DT"], random_state=42
        ),
        "RF": lambda: RandomForestRegressor(
            **best_params["RF"], n_estimators=150, n_jobs=-1, random_state=42
        ),
        "XGBoost": lambda: XGBRegressor(
            **best_params["XGBoost"], n_estimators=500,
            tree_method="hist", n_jobs=-1, random_state=42, verbosity=0,
        ),
        "LightGBM": lambda: LGBMRegressor(
            **best_params["LightGBM"], n_estimators=500,
            n_jobs=-1, random_state=42, verbose=-1,
        ),
        "ANN": lambda: MLPRegressor(
            hidden_layer_sizes=tuple(
                best_params["ANN"][f"n_units_l{i}"]
                for i in range(best_params["ANN"]["n_layers"])
            ),
            learning_rate_init=best_params["ANN"]["learning_rate_init"],
            alpha=best_params["ANN"]["alpha"],
            batch_size=best_params["ANN"]["batch_size"],
            max_iter=200,
            n_iter_no_change=20,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            tol=1e-5,
        ),
    }

    for model_name, builder in model_builders.items():
        log_target = model_name in ("ElasticNet", "ANN")
        rmse_folds = []

        logger.info("  CV: %s (%d folds)", model_name, n_splits)
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_trees, y_raw, groups)):
            mdl = builder()

            if log_target:
                X_tr  = scaler_a.transform(X_lin[tr_idx])
                X_vl  = scaler_a.transform(X_lin[val_idx])
                y_tr  = y_log[tr_idx]
                y_vl  = y_raw[val_idx]
                mdl.fit(X_tr, y_tr)
                yp = np.expm1(mdl.predict(X_vl)).clip(0)
            else:
                X_tr  = X_trees[tr_idx]
                X_vl  = X_trees[val_idx]
                y_tr  = y_raw[tr_idx]
                y_vl  = y_raw[val_idx]
                mdl.fit(X_tr, y_tr)
                yp = np.clip(mdl.predict(X_vl), 0, None)

            fold_rmse = float(np.sqrt(mean_squared_error(y_vl, yp)))
            rmse_folds.append(fold_rmse)
            logger.info("    Fold %d: RMSE=%.3f", fold + 1, fold_rmse)

        rows.append({
            "Model":          model_name,
            "CV_RMSE_Mean":   float(np.mean(rmse_folds)),
            "CV_RMSE_Std":    float(np.std(rmse_folds)),
            "CV_RMSE_Folds":  rmse_folds,
        })

    return pd.DataFrame(rows)[["Model", "CV_RMSE_Mean", "CV_RMSE_Std", "CV_RMSE_Folds"]]


# ─────────────────────────────────────────────────────────────────────────────
# Task 9.7: Per-household MAE vs training volume
# ─────────────────────────────────────────────────────────────────────────────
def compute_data_volume_df(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    test_preds: dict[str, np.ndarray],
    top_models: list[str],
) -> pd.DataFrame:
    training_days = (
        df_train.groupby("Household_ID").size().rename("training_days")
    )
    rows = []
    for hid in df_test["Household_ID"].unique():
        mask = df_test["Household_ID"].values == hid
        n_tr = int(training_days.get(hid, 0))
        row = {"Household_ID": hid, "training_days": n_tr}
        for m in top_models:
            if m in test_preds:
                y_t = df_test["kWh_received_Total"].values[mask]
                y_p = test_preds[m][mask]
                if len(y_t) > 0:
                    row[f"mae_{m.lower().replace(' ', '_')}"] = float(
                        np.mean(np.abs(y_t - y_p))
                    )
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Task 9.8: Feature-set ablation
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    df_b_train: pd.DataFrame,
    df_b_test: pd.DataFrame,
    best_params: dict,
    features_b: list,
    floor_kwh: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Ablation A/B/C as per spec Task 9.8."""
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor

    y_test    = df_test["kWh_received_Total"].values
    y_train   = df_train["kWh_received_Total"].values

    # --- Config B (baseline = full Track A, already computed in main) ---
    # We need to retrain from scratch on train_full for fair ablation comparison.
    configs = []

    # Config A: SMD + Weather only (no Survey_* / household metadata)
    logger.info("  Ablation A: SMD+Weather only")
    for mname, Builder in [
        ("LightGBM", lambda: LGBMRegressor(
            **best_params["LightGBM"], n_estimators=500,
            n_jobs=-1, random_state=42, verbose=-1)),
        ("RF", lambda: RandomForestRegressor(
            **best_params["RF"], n_estimators=300, n_jobs=-1, random_state=42)),
    ]:
        mdl = Builder()
        mdl.fit(df_train[FEATURES_ABLATION_A].values, y_train)
        yp = np.clip(mdl.predict(df_test[FEATURES_ABLATION_A].values), 0, None)
        m = compute_all_metrics(y_test, yp, floor_kwh)
        configs.append({"Config": "A: SMD+Weather", "Model": mname, **m})
        logger.info("    %s Ablation-A RMSE=%.3f", mname, m["RMSE"])

    # Config B: Full Track A (45 features)
    logger.info("  Ablation B: SMD+Weather+Metadata (full Track A)")
    for mname, Builder in [
        ("LightGBM", lambda: LGBMRegressor(
            **best_params["LightGBM"], n_estimators=500,
            n_jobs=-1, random_state=42, verbose=-1)),
        ("RF", lambda: RandomForestRegressor(
            **best_params["RF"], n_estimators=300, n_jobs=-1, random_state=42)),
    ]:
        mdl = Builder()
        mdl.fit(df_train[FEATURES_TREES].values, y_train)
        yp = np.clip(mdl.predict(df_test[FEATURES_TREES].values), 0, None)
        m = compute_all_metrics(y_test, yp, floor_kwh)
        configs.append({"Config": "B: +Metadata (Full)", "Model": mname, **m})
        logger.info("    %s Ablation-B RMSE=%.3f", mname, m["RMSE"])

    # Config B-109: Same as B but on the 109 treatment households only
    logger.info("  Ablation B-109: +Metadata on 109 treatment HH subset")
    # Filter full train/test to the 109 protocol households
    protocol_hhs = df_b_train["Household_ID"].unique()
    df_train_109 = df_train[df_train["Household_ID"].isin(protocol_hhs)]
    df_test_109  = df_test[df_test["Household_ID"].isin(protocol_hhs)]
    y_test_109   = df_test_109["kWh_received_Total"].values

    if len(df_test_109) > 0:
        for mname, Builder in [
            ("LightGBM", lambda: LGBMRegressor(
                **best_params["LightGBM"], n_estimators=500,
                n_jobs=-1, random_state=42, verbose=-1)),
        ]:
            mdl = Builder()
            mdl.fit(df_train_109[FEATURES_TREES].values,
                    df_train_109["kWh_received_Total"].values)
            yp = np.clip(mdl.predict(df_test_109[FEATURES_TREES].values), 0, None)
            m = compute_all_metrics(y_test_109, yp, floor_kwh)
            configs.append({"Config": "B-109: +Metadata (109 HH)", "Model": mname, **m})
            logger.info("    %s Ablation-B-109 RMSE=%.3f", mname, m["RMSE"])

    # Config C: Full protocol features (Track B)
    logger.info("  Ablation C: +Protocol (Track B, 109 HH)")
    y_b_test = df_b_test["kWh_received_Total"].values
    for mname, Builder in [
        ("LightGBM", lambda: LGBMRegressor(
            **best_params["LightGBM"], n_estimators=500,
            n_jobs=-1, random_state=42, verbose=-1)),
    ]:
        mdl = Builder()
        mdl.fit(df_b_train[features_b].values,
                df_b_train["kWh_received_Total"].values)
        yp = np.clip(mdl.predict(df_b_test[features_b].values), 0, None)
        m = compute_all_metrics(y_b_test, yp, floor_kwh)
        configs.append({"Config": "C: +Protocol (Track B)", "Model": mname, **m})
        logger.info("    %s Ablation-C RMSE=%.3f", mname, m["RMSE"])

    return pd.DataFrame(configs)


# ─────────────────────────────────────────────────────────────────────────────
# Task 9.9: Wilcoxon pairwise tests
# ─────────────────────────────────────────────────────────────────────────────
def run_wilcoxon_tests(
    preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    model_names: list[str],
) -> pd.DataFrame:
    """Return symmetric DataFrame of Wilcoxon p-values."""
    ae = {m: np.abs(y_true - preds[m]) for m in model_names if m in preds}
    names = list(ae.keys())
    p_matrix = pd.DataFrame(
        np.ones((len(names), len(names))),
        index=names, columns=names,
    )
    for i, m1 in enumerate(names):
        for j, m2 in enumerate(names):
            if i == j:
                p_matrix.loc[m1, m2] = 1.0
            elif i < j:
                try:
                    _, p = wilcoxon(ae[m1], ae[m2])
                except Exception:
                    p = float("nan")
                p_matrix.loc[m1, m2] = p
                p_matrix.loc[m2, m1] = p
    return p_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Task 9.10: Consolidated report
# ─────────────────────────────────────────────────────────────────────────────
def write_report(
    cfg: dict,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seasonal_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    best_model: str,
    phase7_baselines: dict,
    start_time: float,
) -> None:
    lines = []
    W = 86

    def hr(char="="):
        lines.append(char * W)

    def sec(title):
        lines.append("")
        hr()
        lines.append(f"  {title}")
        hr()

    hr()
    lines.append("  PHASE 9 -- MODEL EVALUATION REPORT")
    lines.append("  HEAPO-Predict  |  Daily Household Heat Pump Electricity Consumption")
    lines.append(f"  Script : scripts/09_evaluation.py")
    import datetime
    lines.append(f"  Date   : {datetime.date.today().isoformat()}")
    hr()

    # 1. Dataset summary
    sec("1. DATASET SUMMARY")
    lines.append(f"  Test  set : 2023-12-01 -> 2024-03-21  N=74,368  HH=826")
    lines.append(f"  Val   set : 2023-06-01 -> 2023-11-30  N=153,594")
    lines.append(f"  Train set : up to 2023-05-31  N=646,258  HH=1,119")
    lines.append("")
    lines.append("  NOTE: Test set is heating season only (Dec-Mar). Target mean = 39.10 kWh/day")
    lines.append("        vs val mean 17.51 kWh/day. Compare R2 across seasons, not absolute RMSE.")

    # 2. Test set performance
    sec("2. TEST SET PERFORMANCE")
    lines.append(_fmt_metrics_table(test_df))

    # 3. Validation set + Phase 7 delta
    sec("3. VALIDATION SET PERFORMANCE (vs Phase 7 baseline)")
    lines.append(_fmt_metrics_table(val_df))
    lines.append("")
    lines.append("  Improvement vs Phase 7 val RMSE:")
    for _, row in val_df.iterrows():
        model = row["Model"]
        if model in phase7_baselines:
            delta = phase7_baselines[model] - row["RMSE"]
            sign  = "+" if delta > 0 else ""
            lines.append(
                f"    {model:<28s}  Phase7={phase7_baselines[model]:.3f}"
                f"  Phase9={row['RMSE']:.3f}  delta={sign}{delta:.3f}"
            )

    # 4. Seasonal breakdown
    sec("4. SEASONAL BREAKDOWN (Val Set)")
    val_seasonal = seasonal_df[seasonal_df["Split"] == "Val"]
    if not val_seasonal.empty:
        lines.append(_fmt_seasonal_table(val_seasonal, ["Non-Heating (May-Sep)", "Transition (Oct-Nov)"]))
    lines.append("")
    lines.append("  Test Set Month-Level Breakdown:")
    test_seasonal = seasonal_df[seasonal_df["Split"] == "Test"]
    if not test_seasonal.empty:
        lines.append(_fmt_seasonal_table(test_seasonal, ["Peak Winter (Dec-Feb)", "Shoulder (Mar-Apr)"]))

    # 5. Cross-validation
    sec("5. CROSS-VALIDATION ROBUSTNESS (Train Set, 5-Fold GroupKFold)")
    if cv_df is not None and len(cv_df) > 0:
        lines.append(
            f"  {'Model':<28s}  {'CV_RMSE_Mean':>12s}  {'CV_RMSE_Std':>12s}"
        )
        hr("-")
        for _, row in cv_df.sort_values("CV_RMSE_Mean").iterrows():
            lines.append(
                f"  {row['Model']:<28s}  {row['CV_RMSE_Mean']:>12.3f}  {row['CV_RMSE_Std']:>12.3f}"
            )
    else:
        lines.append("  (CV skipped)")

    # 6. Feature set ablation
    sec("6. FEATURE-SET ABLATION")
    if ablation_df is not None and len(ablation_df) > 0:
        lines.append(_fmt_ablation_table(ablation_df))

    # 7. Statistical significance
    sec("7. STATISTICAL SIGNIFICANCE (Wilcoxon Signed-Rank, Test Set)")
    bonferroni = 0.0033
    if pval_df is not None and not pval_df.empty:
        sig_pairs = []
        names = list(pval_df.index)
        for i, m1 in enumerate(names):
            for j, m2 in enumerate(names):
                if i < j:
                    p = pval_df.loc[m1, m2]
                    if not pd.isna(p) and float(p) < bonferroni:
                        sig_pairs.append(f"  {m1} vs {m2}: p={float(p):.4f} (sig)")
        lines.append(f"  Bonferroni threshold: alpha=0.05/15={bonferroni}")
        lines.append(f"  Statistically significant pairs ({len(sig_pairs)}):")
        lines.extend(sig_pairs if sig_pairs else ["  (none)"])

    # 8. Key findings
    sec("8. KEY FINDINGS")
    best_test = test_df[~test_df["Model"].str.startswith("Baseline")].iloc[0]
    lines.append(f"  Best overall model : {best_test['Model']}")
    lines.append(f"    Test  RMSE={best_test['RMSE']:.3f} kWh  MAE={best_test['MAE']:.3f}  R2={best_test['R2']:.4f}")
    val_best = val_df[val_df["Model"] == best_test["Model"]]
    if not val_best.empty:
        lines.append(f"    Val   RMSE={val_best['RMSE'].values[0]:.3f} kWh  R2={val_best['R2'].values[0]:.4f}")
    lines.append("")
    lines.append("  Linear models plateau ~12 kWh RMSE -- confirmed by ElasticNet tuning.")
    lines.append("  Protocol features (Track B) provide additional signal for treatment HH.")

    # 9. Limitations
    sec("9. LIMITATIONS AND CAVEATS")
    lines += [
        "  - Test set covers heating season only (Dec-Mar). Val set used for non-heating assessment.",
        "  - PV self-consumed energy is invisible in kWh_received_Total (36.65% of HH).",
        "  - Track B trains on 109 HH. CV is more reliable than a single test-set read-out.",
        "  - All households in canton Zurich (8 weather stations). Geographic scope limited.",
        "  - Building age available only for 214 treatment HH (protocol data).",
    ]

    elapsed = time.time() - start_time
    lines.append("")
    hr()
    lines.append(f"  Total elapsed: {elapsed:.0f} s")
    hr()

    report_path = TABLE_DIR / "phase9_evaluation_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logging.getLogger(__name__).info("Saved %s", report_path)


def _fmt_metrics_table(df: pd.DataFrame) -> str:
    hdr = f"  {'Model':<28s}  {'RMSE':>7s}  {'MAE':>7s}  {'R2':>7s}  {'sMAPE%':>7s}  {'MedAE':>7s}  {'N':>7s}"
    sep = "  " + "-" * 70
    rows = [hdr, sep]
    for _, r in df.iterrows():
        rows.append(
            f"  {r['Model']:<28s}  {r['RMSE']:>7.3f}  {r['MAE']:>7.3f}"
            f"  {r['R2']:>7.4f}  {r['sMAPE']:>7.2f}  {r['MedAE']:>7.3f}  {int(r['N']):>7d}"
        )
    return "\n".join(rows)


def _fmt_seasonal_table(df: pd.DataFrame, periods: list) -> str:
    df_f = df[df["Period"].isin(periods)]
    hdr = f"  {'Model':<28s}  {'Period':<28s}  {'RMSE':>7s}  {'R2':>7s}  {'N':>6s}"
    sep = "  " + "-" * 80
    rows = [hdr, sep]
    for _, r in df_f.iterrows():
        rows.append(
            f"  {r['Model']:<28s}  {r['Period']:<28s}  {r['RMSE']:>7.3f}  {r['R2']:>7.4f}  {int(r['N']):>6d}"
        )
    return "\n".join(rows)


def _fmt_ablation_table(df: pd.DataFrame) -> str:
    hdr = f"  {'Config':<30s}  {'Model':<12s}  {'RMSE':>7s}  {'MAE':>7s}  {'R2':>7s}"
    sep = "  " + "-" * 65
    rows = [hdr, sep]
    for _, r in df.iterrows():
        rows.append(
            f"  {r['Config']:<30s}  {r['Model']:<12s}  {r['RMSE']:>7.3f}  {r['MAE']:>7.3f}  {r['R2']:>7.4f}"
        )
    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Task 9.11: Save prediction parquets
# ─────────────────────────────────────────────────────────────────────────────
def save_prediction_parquet(
    df: pd.DataFrame,
    preds: dict[str, np.ndarray],
    path: Path,
    col_map: dict[str, str],
    extra_cols: list[str],
) -> None:
    """Build a slim prediction parquet with residuals and metadata columns."""
    out = df[["Household_ID", "Date", "kWh_received_Total"] + extra_cols].copy()
    for display_name, col_name in col_map.items():
        if display_name in preds:
            out[col_name] = preds[display_name]
            out[f"residual_{col_name}"] = (
                out["kWh_received_Total"] - out[col_name]
            )
    out.to_parquet(path, index=False)
    logging.getLogger(__name__).info("Saved %s  shape=%s", path, out.shape)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    global FEATURES_LINEAR, FEATURES_TREES_B

    t0 = time.time()
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("HEAPO-Predict -- Phase 9: Model Evaluation")
    logger.info("=" * 60)

    cfg       = load_config("config/params.yaml")
    floor_kwh = cfg["evaluation"]["mape_floor_kwh"]

    # ── Task 9.0: Load feature metadata ──────────────────────────────────────
    logger.info("Loading feature metadata...")
    scaler_a_meta   = json.loads((MODEL_DIR / "scaler_linear_A_meta.json").read_text())
    FEATURES_LINEAR = scaler_a_meta["feature_names"]
    best_params     = json.loads((MODEL_DIR / "best_params.json").read_text())

    scaler_a = joblib.load(MODEL_DIR / "scaler_linear_A.pkl")
    scaler_b = joblib.load(MODEL_DIR / "scaler_linear_B.pkl")

    # ── Task 9.0: Load data ───────────────────────────────────────────────────
    logger.info("Loading parquets...")
    df_train    = pd.read_parquet(DATA_DIR / "train_full.parquet")
    df_val      = pd.read_parquet(DATA_DIR / "val_full.parquet")
    df_test     = pd.read_parquet(DATA_DIR / "test_full.parquet")
    df_b_train  = pd.read_parquet(DATA_DIR / "train_protocol.parquet")
    df_b_val    = pd.read_parquet(DATA_DIR / "val_protocol.parquet")
    df_b_test   = pd.read_parquet(DATA_DIR / "test_protocol.parquet")

    logger.info("  train_full  : %s", df_train.shape)
    logger.info("  val_full    : %s", df_val.shape)
    logger.info("  test_full   : %s", df_test.shape)
    logger.info("  train_B     : %s", df_b_train.shape)
    logger.info("  val_B       : %s", df_b_val.shape)
    logger.info("  test_B      : %s", df_b_test.shape)

    # Derive FEATURES_TREES_B from protocol train columns
    _b_exclude = {
        "Household_ID", "Date", "kWh_received_Total", "kWh_log1p",
        "kWh_received_HeatPump", "kWh_received_Other", "kWh_returned_Total",
        "Group", "AffectsTimePoint", "Timestamp", "Weather_ID", "cv_fold",
        "post_intervention",
        "kvarh_received_capacitive_HeatPump", "kvarh_received_capacitive_Other",
        "kvarh_received_inductive_HeatPump", "kvarh_received_inductive_Other",
    }
    _b_num_cols = [
        c for c in df_b_train.select_dtypes(include="number").columns
        if c not in _b_exclude
    ]
    FEATURES_TREES_B = _b_num_cols[:75]   # XGB_B was trained on first 75
    logger.info("  FEATURES_TREES=%d  FEATURES_LINEAR=%d  FEATURES_TREES_B=%d",
                len(FEATURES_TREES), len(FEATURES_LINEAR), len(FEATURES_TREES_B))

    # ── Load tuned models ─────────────────────────────────────────────────────
    logger.info("Loading tuned models...")
    models_a = {
        "ElasticNet": joblib.load(MODEL_DIR / "model_elasticnet_tuned.pkl"),
        "DT":         joblib.load(MODEL_DIR / "model_dt_tuned.pkl"),
        "RF":         joblib.load(MODEL_DIR / "model_rf_tuned.pkl"),
        "XGBoost":    joblib.load(MODEL_DIR / "model_xgboost_tuned.pkl"),
        "LightGBM":   joblib.load(MODEL_DIR / "model_lgbm_tuned.pkl"),
        "ANN":        joblib.load(MODEL_DIR / "model_ann_tuned.pkl"),
    }
    xgb_b = joblib.load(MODEL_DIR / "model_xgboost_b_tuned.pkl")

    # ── Task 9.2: Baselines ───────────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Task 9.2 -- Baselines")
    baselines_test = make_baseline_predictions(df_train, df_test)
    baselines_val  = make_baseline_predictions(df_train, df_val)

    # ── Task 9.3: Generate predictions ───────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Task 9.3 -- Generating predictions (test)")
    preds_test = generate_all_predictions(df_test, models_a, scaler_a, FEATURES_LINEAR)
    preds_test.update(baselines_test)
    pred_b_test = generate_b_predictions(df_b_test, xgb_b, FEATURES_TREES_B)

    logger.info("Task 9.3 -- Generating predictions (val)")
    preds_val = generate_all_predictions(df_val, models_a, scaler_a, FEATURES_LINEAR)
    preds_val.update(baselines_val)
    pred_b_val = generate_b_predictions(df_b_val, xgb_b, FEATURES_TREES_B)

    # ── Task 9.3: Metrics tables ──────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Task 9.3 -- Computing metrics")
    y_test = df_test["kWh_received_Total"].values
    y_val  = df_val["kWh_received_Total"].values
    y_b_test = df_b_test["kWh_received_Total"].values
    y_b_val  = df_b_val["kWh_received_Total"].values

    metrics_test = build_metrics_table(preds_test, y_test, floor_kwh, track="A")
    metrics_val  = build_metrics_table(preds_val,  y_val,  floor_kwh, track="A")

    # Add XGBoost B row
    m_b_test = compute_all_metrics(y_b_test, pred_b_test, floor_kwh)
    m_b_test.update({"Model": "XGBoost B", "Track": "B"})
    m_b_val  = compute_all_metrics(y_b_val, pred_b_val, floor_kwh)
    m_b_val.update({"Model": "XGBoost B", "Track": "B"})

    metrics_test = pd.concat(
        [metrics_test, pd.DataFrame([m_b_test])], ignore_index=True
    )
    metrics_val  = pd.concat(
        [metrics_val,  pd.DataFrame([m_b_val])],  ignore_index=True
    )

    metrics_test.to_csv(TABLE_DIR / "phase9_metrics_test.csv", index=False)
    metrics_val.to_csv(TABLE_DIR  / "phase9_metrics_val.csv",  index=False)
    logger.info("Saved phase9_metrics_test.csv and phase9_metrics_val.csv")

    # Log test metrics
    logger.info("\n%s", metrics_test[["Model", "RMSE", "MAE", "R2", "sMAPE"]].to_string(index=False))

    # ── Task 9.4: Diagnostic plots ────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Task 9.4 -- Diagnostic plots")
    model_pred_names = [m for m in preds_test if not m.startswith("Baseline")]
    for mname in model_pred_names:
        slug = mname.lower().replace(" ", "_")
        plot_predicted_vs_actual(
            y_test, preds_test[mname], mname,
            FIG_DIR / f"phase9_predicted_vs_actual_{slug}.png", floor_kwh,
        )
        plot_residuals_vs_predicted(
            y_test, preds_test[mname], mname,
            FIG_DIR / f"phase9_residuals_vs_predicted_{slug}.png",
        )
        plot_residual_histogram(
            y_test, preds_test[mname], mname,
            FIG_DIR / f"phase9_residual_histogram_{slug}.png",
        )

    # Time-series: best model (LightGBM expected)
    best_model = metrics_test[
        ~metrics_test["Model"].str.startswith("Baseline")
        & (metrics_test["Track"] == "A")
    ].iloc[0]["Model"]
    logger.info("Best Track A model: %s", best_model)

    # Choose 6 sample households: 2 treatment (high), 2 treatment (mid), 2 control
    treatment_hhs = df_test[df_test["Group"] == "treatment"]["Household_ID"].unique()
    control_hhs   = df_test[df_test["Group"] == "control"]["Household_ID"].unique()
    # Pick by mean consumption on test set
    hh_means_test = (
        df_test.groupby("Household_ID")["kWh_received_Total"].mean()
    )
    rng = np.random.default_rng(cfg["evaluation"]["ts_sample_seed"])

    def _pick_hhs(hhs, n):
        valid = [h for h in hhs if h in hh_means_test.index]
        if len(valid) < n:
            return list(valid)
        sorted_hhs = sorted(valid, key=lambda h: hh_means_test[h])
        tercile = len(sorted_hhs) // 3
        high = sorted_hhs[-tercile:]
        mid  = sorted_hhs[tercile: 2 * tercile]
        chosen = (
            list(rng.choice(high, min(n // 2, len(high)), replace=False))
            + list(rng.choice(mid, min(n // 2, len(mid)), replace=False))
        )
        return chosen[:n]

    ts_hhs = _pick_hhs(treatment_hhs, 4) + _pick_hhs(control_hhs, 2)
    ts_hhs = ts_hhs[:cfg["evaluation"]["ts_sample_households"]]

    # Add best model pred column to val and test for time-series plot
    best_slug = best_model.lower().replace(" ", "_")
    pred_col  = f"pred_{best_slug}"
    df_val_ts  = df_val.copy()
    df_test_ts = df_test.copy()
    df_val_ts[pred_col]  = preds_val[best_model]
    df_test_ts[pred_col] = preds_test[best_model]

    if ts_hhs:
        plot_timeseries(
            df_val_ts, df_test_ts, pred_col, best_model, ts_hhs,
            FIG_DIR / f"phase9_timeseries_{best_slug}.png",
        )

    # Multi-model comparison: 1 treatment + 1 control
    comp_hhs = []
    if len(treatment_hhs) > 0:
        comp_hhs.append(int(rng.choice(treatment_hhs[:50])))
    if len(control_hhs) > 0:
        comp_hhs.append(int(rng.choice(control_hhs[:50])))

    if comp_hhs:
        # Add all model pred columns to val/test frames
        df_val_cmp  = df_val.copy()
        df_test_cmp = df_test.copy()
        pred_col_map = {}
        for mn in model_pred_names:
            slug_mn = mn.lower().replace(" ", "_")
            col = f"pred_{slug_mn}"
            df_val_cmp[col]  = preds_val[mn]
            df_test_cmp[col] = preds_test[mn]
            pred_col_map[mn] = col

        plot_timeseries_comparison(
            df_val_cmp, df_test_cmp, pred_col_map, comp_hhs,
            FIG_DIR / "phase9_timeseries_comparison.png",
        )

    # ── Task 9.5: Seasonal breakdown ──────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Task 9.5 -- Seasonal breakdown")
    seasonal_test = compute_seasonal_metrics(df_test, preds_test, floor_kwh, "Test")
    seasonal_val  = compute_seasonal_metrics(df_val,  preds_val,  floor_kwh, "Val")
    seasonal_df   = pd.concat([seasonal_test, seasonal_val], ignore_index=True)
    seasonal_df.to_csv(TABLE_DIR / "phase9_metrics_seasonal.csv", index=False)
    logger.info("Saved phase9_metrics_seasonal.csv")

    # Seasonal bar plot (test set)
    seasonal_models_only = seasonal_test[~seasonal_test["Model"].str.startswith("Baseline")]
    if not seasonal_models_only.empty:
        plot_seasonal_barplot(
            seasonal_models_only, "RMSE",
            FIG_DIR / "phase9_seasonal_barplot.png",
        )

    # ── Task 9.6: Cross-validation ────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Task 9.6 -- Cross-validation (5-fold GroupKFold on train_full)")
    cv_df = run_cross_validation(
        df_train, models_a, scaler_a, FEATURES_LINEAR,
        n_splits=cfg["evaluation"]["cv_n_splits"],
        floor_kwh=floor_kwh,
        logger=logger,
    )
    cv_df.to_csv(TABLE_DIR / "phase9_metrics_cv.csv", index=False)
    logger.info("Saved phase9_metrics_cv.csv")

    plot_cv_errorbar(
        cv_df[["Model", "CV_RMSE_Mean", "CV_RMSE_Std"]],
        metrics_test[metrics_test["Track"] == "A"],
        FIG_DIR / "phase9_cv_errorbar.png",
    )

    # ── Task 9.7: Data volume scatter ─────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Task 9.7 -- Per-household MAE vs training days")
    top3 = list(
        metrics_test[
            ~metrics_test["Model"].str.startswith("Baseline")
            & (metrics_test["Track"] == "A")
        ]["Model"].head(3)
    )
    volume_df = compute_data_volume_df(df_train, df_test, preds_test, top3)
    volume_df.to_csv(TABLE_DIR / "phase9_data_volume.csv", index=False)
    plot_data_volume_scatter(
        volume_df,
        FIG_DIR / "phase9_data_volume_scatter.png",
        min_days_threshold=cfg["data"]["min_days_threshold"],
    )

    # ── Task 9.8: Feature-set ablation ────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Task 9.8 -- Feature-set ablation")
    ablation_df = run_ablation(
        df_train, df_val, df_test,
        df_b_train, df_b_test,
        best_params, FEATURES_TREES_B,
        floor_kwh, logger,
    )
    ablation_df.to_csv(TABLE_DIR / "phase9_ablation_metrics.csv", index=False)
    logger.info("Saved phase9_ablation_metrics.csv")
    plot_ablation_barplot(ablation_df, FIG_DIR / "phase9_ablation_barplot.png")

    # ── Task 9.9: Wilcoxon tests ──────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Task 9.9 -- Pairwise Wilcoxon tests (test set)")
    test_model_names = [m for m in model_pred_names if m in preds_test]
    pval_df = run_wilcoxon_tests(preds_test, y_test, test_model_names)
    pval_df.to_csv(TABLE_DIR / "phase9_wilcoxon_matrix.csv")
    logger.info("Saved phase9_wilcoxon_matrix.csv")
    plot_significance_heatmap(
        pval_df, FIG_DIR / "phase9_significance_heatmap.png",
        alpha_bonferroni=0.05 / 15,
        alpha_nominal=cfg["evaluation"]["stat_test_alpha"],
    )

    # ── Task 9.11: Save prediction parquets ───────────────────────────────────
    logger.info("-" * 40)
    logger.info("Task 9.11 -- Saving prediction parquets")
    pred_col_map_parquet = {
        "Baseline: Global Mean": "pred_global_mean",
        "Baseline: Per-HH Mean": "pred_hh_mean",
        "Baseline: HDD-Linear":  "pred_hdd_linear",
        "ElasticNet":            "pred_elasticnet",
        "DT":                    "pred_dt",
        "RF":                    "pred_rf",
        "XGBoost":               "pred_xgb",
        "LightGBM":              "pred_lgbm",
        "ANN":                   "pred_ann",
    }
    extra_meta = ["is_heating_season", "has_pv", "Group", "Survey_HeatPump_Installation_Type"]

    save_prediction_parquet(
        df_test, preds_test,
        TABLE_DIR / "phase9_test_predictions.parquet",
        pred_col_map_parquet, extra_meta,
    )
    save_prediction_parquet(
        df_val, preds_val,
        TABLE_DIR / "phase9_val_predictions.parquet",
        pred_col_map_parquet, extra_meta,
    )

    # Track B parquet
    df_b_out = df_b_test[["Household_ID", "Date", "kWh_received_Total", "is_heating_season"]].copy()
    df_b_out["pred_xgb_b"]      = pred_b_test
    df_b_out["residual_xgb_b"]  = df_b_out["kWh_received_Total"] - pred_b_test
    df_b_out.to_parquet(TABLE_DIR / "phase9_test_predictions_b.parquet", index=False)
    logger.info("Saved phase9_test_predictions_b.parquet  shape=%s", df_b_out.shape)

    # ── Task 9.10: Consolidated report ────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Task 9.10 -- Writing consolidated report")
    phase7_baselines = {
        "LightGBM":  9.318,
        "XGBoost":   9.462,
        "RF":        9.421,
        "ANN":       10.330,
        "DT":        11.382,
        "ElasticNet": 12.185,
        "XGBoost B": 5.933,
    }
    write_report(
        cfg=cfg,
        test_df=metrics_test,
        val_df=metrics_val,
        seasonal_df=seasonal_df,
        cv_df=cv_df,
        ablation_df=ablation_df,
        pval_df=pval_df,
        best_model=best_model,
        phase7_baselines=phase7_baselines,
        start_time=t0,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Phase 9 complete.  Elapsed: %.0f s", elapsed)
    logger.info("  Tables  -> %s", TABLE_DIR)
    logger.info("  Figures -> %s", FIG_DIR)
    logger.info("  Report  -> %s", TABLE_DIR / "phase9_evaluation_report.txt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
