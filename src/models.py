"""
src/models.py

Model definitions, training routines, and evaluation helpers for Phase 7.
Covers: baselines, Linear Regression variants, Decision Tree, Random Forest,
XGBoost, LightGBM, and 5-fold CV evaluation.

ANN (MLP) lives in src/ann.py — sklearn MLPRegressor backend, no PyTorch.

All column names verified against Tables 1, 4, 5, 6 of the HEAPO paper
(Brudermueller et al. 2025, arXiv:2503.16993v1).

Target conventions:
  - Tree models (DT, RF, XGBoost, LightGBM): train on raw kWh, predict raw kWh.
  - Linear models: train on log1p(kWh), back-transform via np.expm1().clip(0).
  - All metrics always computed in raw kWh space.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = 0.5,
) -> float:
    """Symmetric MAPE, excluding days where actual consumption < floor kWh.

    This is consistent with the HEAPO paper's own validation methodology
    (Section 2.4). The floor prevents near-zero summer days from inflating
    the percentage error.

    Returns NaN if no rows remain after filtering.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true >= floor
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if len(y_t) == 0:
        return float("nan")
    return float(100.0 * np.mean(
        2.0 * np.abs(y_t - y_p) / (np.abs(y_t) + np.abs(y_p) + 1e-8)
    ))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "",
    floor: float = 0.5,
) -> dict[str, Any]:
    """Compute RMSE, MAE, R², MedAE, sMAPE for a model in raw kWh space.

    Args:
        y_true: Ground-truth kWh values.
        y_pred: Predicted kWh values (will be clipped to ≥0).
        name:   Model name for logging.
        floor:  sMAPE floor threshold (kWh); rows below are excluded from %.

    Returns:
        Dict with keys: model, rmse, mae, r2, medae, smape, smape_excluded_pct.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).clip(0)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    smape_val = smape(y_true, y_pred, floor=floor)
    excluded_pct = float(100.0 * (y_true < floor).mean())

    logger.info(
        f"{name:35s} | RMSE={rmse:7.3f}  MAE={mae:6.3f}  R²={r2:7.4f}  "
        f"MedAE={medae:6.3f}  sMAPE={smape_val:5.2f}%  "
        f"(excl {excluded_pct:.1f}% rows <{floor} kWh)"
    )
    return {
        "model": name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "medae": medae,
        "smape": smape_val,
        "smape_excluded_pct": excluded_pct,
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def predict_overall_mean(
    y_train: np.ndarray,
    n_predictions: int,
) -> np.ndarray:
    """Predict the training-set mean for every sample (global constant baseline)."""
    overall_mean = float(np.asarray(y_train).mean())
    logger.info(f"Overall-mean baseline: mean={overall_mean:.4f} kWh/day")
    return np.full(n_predictions, overall_mean)


def predict_hh_mean(
    df_train: pd.DataFrame,
    df_target: pd.DataFrame,
    target_col: str = "kWh_received_Total",
    hh_col: str = "Household_ID",
) -> np.ndarray:
    """Per-household mean baseline.

    For each row in df_target, look up that household's training-period mean.
    Households absent from training (cold-start) fall back to the global mean.

    Returns predictions aligned to df_target's index.
    """
    hh_means = df_train.groupby(hh_col)[target_col].mean()
    global_mean = float(df_train[target_col].mean())

    preds = df_target[hh_col].map(hh_means).fillna(global_mean).to_numpy()

    n_cold = int((~df_target[hh_col].isin(hh_means.index)).sum())
    logger.info(
        f"Per-HH mean baseline: {hh_means.shape[0]} HH trained, "
        f"{n_cold} cold-start rows → fallback to global mean ({global_mean:.3f} kWh)"
    )
    return preds


def fit_hdd_baseline(
    df_train: pd.DataFrame,
    target_col: str = "kWh_received_Total",
    hdd_col: str = "HDD_SIA_daily",
) -> LinearRegression:
    """Fit a global HDD-proportional linear model: kWh = α×HDD + β.

    Only uses heating-season rows (HDD > 0) to avoid a degenerate summer fit.
    """
    mask = df_train[hdd_col] > 0
    X = df_train.loc[mask, [hdd_col]].values
    y = df_train.loc[mask, target_col].values
    model = LinearRegression().fit(X, y)
    alpha, beta = float(model.coef_[0]), float(model.intercept_)
    n_rows = int(mask.sum())
    logger.info(
        f"HDD baseline fitted on {n_rows} heating-season rows: "
        f"α={alpha:.4f} kWh/HDD,  β={beta:.4f} kWh"
    )
    return model


# ---------------------------------------------------------------------------
# Linear Regression Variants
# ---------------------------------------------------------------------------

def fit_linear_variants(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit OLS, Ridge, Lasso, and ElasticNet.

    All models train on log1p(kWh). Predictions must be back-transformed
    via np.expm1(...).clip(0) before metric computation.

    Args:
        X_train:      Scaled feature matrix (output of StandardScaler.transform).
        y_train:      Log1p-transformed target.
        feature_names: List matching X_train columns (for coefficient logging).
        random_state: Random seed.

    Returns:
        Dict mapping model name → fitted sklearn estimator.
    """
    models: dict[str, Any] = {}

    specs = [
        ("OLS",        LinearRegression(fit_intercept=True)),
        ("Ridge",      Ridge(alpha=1.0, random_state=random_state)),
        ("Lasso",      Lasso(alpha=0.01, max_iter=5000, random_state=random_state)),
        ("ElasticNet", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000,
                                  random_state=random_state)),
    ]

    for name, estimator in specs:
        t0 = time.perf_counter()
        estimator.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        coef = np.asarray(estimator.coef_).ravel()
        top_idx = np.argsort(np.abs(coef))[::-1][:10]
        top_feats = [(feature_names[i], round(float(coef[i]), 5)) for i in top_idx]
        logger.info(f"{name} fitted in {elapsed:.2f}s — top-10 |coef|: {top_feats}")

        if name == "Lasso":
            zero_feats = [feature_names[i] for i, c in enumerate(coef) if c == 0.0]
            logger.info(f"Lasso: {len(zero_feats)} zero-coef features: {zero_feats}")

        models[name] = estimator

    return models


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

def fit_decision_tree(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
) -> DecisionTreeRegressor:
    """Fit a Decision Tree regressor on raw kWh target."""
    t0 = time.perf_counter()
    dt = DecisionTreeRegressor(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=None,
        random_state=random_state,
    )
    dt.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    train_r2 = float(dt.score(X_train, y_train))
    logger.info(
        f"DecisionTree fitted in {elapsed:.1f}s — "
        f"depth={dt.get_depth()}, leaves={dt.get_n_leaves()}, "
        f"train_R²={train_r2:.4f}"
    )
    return dt


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def fit_random_forest(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Fit a Random Forest regressor on raw kWh target.

    OOB score is enabled as a free internal validation signal.
    Training on 646k rows × 300 trees is slow (5–15 min on CPU).
    """
    t0 = time.perf_counter()
    # verbose=1 prints joblib parallel progress (e.g. "[Parallel]: Done 50/300...")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        oob_score=True,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    logger.info(
        f"RandomForest fitted in {elapsed:.1f}s — "
        f"OOB R²={rf.oob_score_:.4f}"
    )
    return rf


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def fit_xgboost(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    suffix: str = "",
) -> Any:
    """Fit XGBoost regressor with early stopping on raw kWh target.

    CRITICAL: eval_set is always passed so early_stopping_rounds is active.
    Without eval_set, early stopping silently has no effect.

    Args:
        suffix: Appended to log messages to distinguish Track A vs Track B.
    """
    import xgboost as xgb  # local import — not always installed in test envs

    t0 = time.perf_counter()
    model = xgb.XGBRegressor(
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.0,
        reg_lambda=1.0,
        early_stopping_rounds=early_stopping_rounds,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,   # print val-RMSE every 100 rounds
    )
    elapsed = time.perf_counter() - t0

    tag = f" [{suffix}]" if suffix else ""
    logger.info(
        f"XGBoost{tag} fitted in {elapsed:.1f}s — "
        f"best_iteration={model.best_iteration}, "
        f"best_val_RMSE={model.best_score:.4f} kWh"
    )
    return model


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def fit_lightgbm(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
) -> Any:
    """Fit LightGBM regressor with early stopping on raw kWh target.

    Inputs are always converted to numpy to avoid the sklearn feature-name
    warning when predicting later with numpy arrays.
    """
    import lightgbm as lgb  # local import

    # Convert to numpy so fit and predict use the same representation
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.values

    t0 = time.perf_counter()
    model = lgb.LGBMRegressor(
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=100),  # print val-RMSE every 100 rounds
    ]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    elapsed = time.perf_counter() - t0

    best_iter = model.best_iteration_
    # LightGBM default regression metric key is 'l2' (MSE), not 'rmse'.
    valid_scores = model.best_score_.get("valid_0", {})
    best_score_l2 = valid_scores.get("l2", float("nan"))
    best_score_rmse = float(np.sqrt(best_score_l2)) if not np.isnan(best_score_l2) else float("nan")
    logger.info(
        f"LightGBM fitted in {elapsed:.1f}s — "
        f"best_iteration={best_iter}, "
        f"best_val_RMSE={best_score_rmse:.4f} kWh"
    )
    return model


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cv_evaluate(
    model_fn,
    X: np.ndarray | pd.DataFrame,
    y_log: np.ndarray,
    y_raw: np.ndarray,
    folds: np.ndarray,
    model_name: str,
    is_log_target: bool = False,
) -> list[float]:
    """5-fold cross-validated RMSE using pre-assigned fold column.

    Args:
        model_fn:      Callable(X_tr, y_tr) → fitted model.
        X:             Feature matrix aligned to training set.
        y_log:         Log1p target (used if is_log_target=True).
        y_raw:         Raw kWh target (always used for RMSE computation).
        folds:         Integer fold assignments per row (values 0–4).
        model_name:    For logging.
        is_log_target: If True, model trains on y_log; preds back-transformed.

    Returns:
        List of per-fold RMSE values (length 5).
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    y_train_target = np.asarray(y_log) if is_log_target else np.asarray(y_raw)
    y_raw_arr = np.asarray(y_raw)
    folds_arr = np.asarray(folds)

    fold_rmses: list[float] = []
    for fold_id in range(5):
        mask_val = folds_arr == fold_id
        mask_tr = ~mask_val
        fitted = model_fn(X[mask_tr], y_train_target[mask_tr])
        preds = np.asarray(fitted.predict(X[mask_val]))
        if is_log_target:
            preds = np.expm1(preds).clip(0)
        preds = preds.clip(0)
        rmse = float(np.sqrt(mean_squared_error(y_raw_arr[mask_val], preds)))
        fold_rmses.append(rmse)

    mean_rmse = float(np.mean(fold_rmses))
    std_rmse = float(np.std(fold_rmses))
    logger.info(
        f"CV RMSE [{model_name:20s}]: {mean_rmse:.3f} ± {std_rmse:.3f} kWh  "
        f"per fold: {[round(r, 3) for r in fold_rmses]}"
    )
    return fold_rmses
