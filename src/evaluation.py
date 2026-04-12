"""
src/evaluation.py

Evaluation metrics, diagnostic plots, and back-transform helpers for Phase 9.

Public API
----------
compute_all_metrics(y_true, y_pred, floor_kwh) -> dict
predict_raw(model, X, log_target, scaler) -> np.ndarray

plot_predicted_vs_actual(y_true, y_pred, model_name, save_path)
plot_residuals_vs_predicted(y_true, y_pred, model_name, save_path)
plot_residual_histogram(y_true, y_pred, model_name, save_path)
plot_timeseries(df_val, df_test, pred_col, model_name, household_ids, save_path)
plot_timeseries_comparison(df_val, df_test, pred_cols, household_ids, save_path)
plot_seasonal_barplot(seasonal_df, metric, save_path)
plot_cv_errorbar(cv_df, test_metrics_df, save_path)
plot_data_volume_scatter(volume_df, save_path, min_days_threshold)
plot_ablation_barplot(ablation_df, save_path)
plot_significance_heatmap(pval_df, save_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# ── Colour palette (consistent across all Phase 9 figures) ───────────────────
MODEL_COLOURS = {
    "Baseline: Global Mean":    "#aaaaaa",
    "Baseline: Per-HH Mean":    "#888888",
    "Baseline: HDD-Linear":     "#555555",
    "ElasticNet":               "#4e79a7",
    "DT":                       "#f28e2b",
    "RF":                       "#59a14f",
    "XGBoost":                  "#e15759",
    "LightGBM":                 "#76b7b2",
    "ANN":                      "#edc948",
    "XGBoost B":                "#b07aa1",
}

FIG_DPI = 150


# ─────────────────────────────────────────────────────────────────────────────
# 1. Metric functions
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def smape(y_true: np.ndarray, y_pred: np.ndarray, floor_kwh: float = 0.5) -> float:
    """Symmetric MAPE consistent with paper Section 2.4.
    Rows with y_true < floor_kwh are excluded (near-zero HP days blow up % metrics).
    Returns percentage (0-100).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mask = y_t >= floor_kwh
    if mask.sum() == 0:
        return float("nan")
    denom = (np.abs(y_t[mask]) + np.abs(y_p[mask])) / 2.0
    return float(100.0 * np.mean(np.abs(y_t[mask] - y_p[mask]) / denom))


def medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor_kwh: float = 0.5,
) -> dict:
    """Return dict with RMSE, MAE, R2, sMAPE, MedAE, N."""
    return {
        "RMSE":  rmse(y_true, y_pred),
        "MAE":   mae(y_true, y_pred),
        "R2":    r2(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred, floor_kwh),
        "MedAE": medae(y_true, y_pred),
        "N":     int(len(y_true)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def predict_raw(
    model,
    X: np.ndarray,
    log_target: bool,
    scaler=None,
) -> np.ndarray:
    """Generate clipped, raw-kWh predictions regardless of training target.

    Parameters
    ----------
    model      : fitted sklearn / XGBoost / LightGBM model
    X          : feature matrix (2D numpy array, columns already in correct order)
    log_target : if True, model was trained on log1p(y) — apply expm1 back-transform
    scaler     : if not None, apply scaler.transform(X) before predicting
                 (MUST be the already-fitted scaler — never refit here)
    """
    X_input = scaler.transform(X) if scaler is not None else X
    pred = model.predict(X_input)
    if log_target:
        pred = np.expm1(pred)
    return np.clip(pred, 0.0, None)


def assert_predictions_valid(y_pred: np.ndarray, model_name: str) -> None:
    """Sanity checks on a prediction array before metrics are computed."""
    if np.isnan(y_pred).any():
        raise ValueError(f"{model_name}: NaN in predictions — check input features / transform")
    if not (y_pred >= 0).all():
        raise ValueError(f"{model_name}: negative predictions — forgot clip(0)?")
    if y_pred.max() >= 1000:
        raise ValueError(
            f"{model_name}: prediction >= 1000 kWh — likely a transform error "
            f"(max={y_pred.max():.1f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", path)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path,
    floor_kwh: float = 0.5,
) -> None:
    """Hexbin predicted vs. actual scatter with diagonal reference line."""
    fig, ax = plt.subplots(figsize=(6, 5))

    hb = ax.hexbin(y_true, y_pred, gridsize=80, mincnt=1,
                   cmap="Blues", linewidths=0.2)
    fig.colorbar(hb, ax=ax, label="Count")

    lim = max(float(y_true.max()), float(y_pred.max())) * 1.02
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, label="y = x (perfect)")

    metrics = compute_all_metrics(y_true, y_pred, floor_kwh)
    ax.set_title(
        f"{model_name}  —  Predicted vs. Actual (Test)\n"
        f"RMSE={metrics['RMSE']:.2f} kWh  |  R\u00b2={metrics['R2']:.3f}",
        fontsize=11,
    )
    ax.set_xlabel("Actual (kWh/day)")
    ax.set_ylabel("Predicted (kWh/day)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.tight_layout()
    _save(fig, save_path)


def plot_residuals_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path,
    sample_n: int = 10_000,
    seed: int = 42,
) -> None:
    """Residuals (y_true - y_pred) vs. predicted, with LOWESS trend."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=min(sample_n, len(y_true)), replace=False)
    yt_s = y_true[idx]
    yp_s = y_pred[idx]
    residuals = yt_s - yp_s

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(yp_s, residuals, alpha=0.15, s=8, color="#4e79a7", rasterized=True)
    ax.axhline(0, color="red", linewidth=1.2, linestyle="--")

    order = np.argsort(yp_s)
    smooth = lowess(residuals[order], yp_s[order], frac=0.2, return_sorted=False)
    ax.plot(yp_s[order], smooth, color="orange", linewidth=2, label="LOWESS trend")

    rmse_val = rmse(y_true, y_pred)
    ax.set_title(f"{model_name}  —  Residuals vs. Predicted\nRMSE={rmse_val:.2f} kWh", fontsize=11)
    ax.set_xlabel("Predicted (kWh/day)")
    ax.set_ylabel("Residual = Actual \u2212 Predicted (kWh)")
    ax.legend(fontsize=9)
    subtitle = f"(Showing {len(idx):,} of {len(y_true):,} test points)"
    ax.text(0.98, 0.02, subtitle, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="grey")
    fig.tight_layout()
    _save(fig, save_path)


def plot_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path,
) -> None:
    """Residual histogram with KDE and RMSE markers."""
    from scipy.stats import gaussian_kde

    residuals = y_true - y_pred
    rmse_val = rmse(y_true, y_pred)
    mean_res = float(np.mean(residuals))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=80, density=True, alpha=0.55,
            color="#4e79a7", label="Residuals")

    kde = gaussian_kde(residuals, bw_method="silverman")
    xs = np.linspace(float(residuals.min()), float(residuals.max()), 400)
    ax.plot(xs, kde(xs), color="navy", linewidth=2, label="KDE")

    ax.axvline(mean_res, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_res:.2f}")
    ax.axvline(-rmse_val, color="grey", linestyle=":", linewidth=1.2)
    ax.axvline(+rmse_val, color="grey", linestyle=":", linewidth=1.2,
               label=f"\u00b1RMSE ({rmse_val:.2f})")

    ax.set_title(f"{model_name}  —  Residual Distribution (Test)", fontsize=11)
    ax.set_xlabel("Residual (kWh/day)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)


def plot_timeseries(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    pred_col: str,
    model_name: str,
    household_ids: list,
    save_path: Path,
) -> None:
    """Actual vs. predicted time series for sampled households (best model)."""
    nrows = len(household_ids)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 3 * nrows), sharex=False)
    if nrows == 1:
        axes = [axes]

    for ax, hid in zip(axes, household_ids):
        hval  = df_val[df_val["Household_ID"]  == hid].sort_values("Date")
        htest = df_test[df_test["Household_ID"] == hid].sort_values("Date")
        hdf   = pd.concat([hval, htest])

        ax.plot(hdf["Date"], hdf["kWh_received_Total"],
                color="steelblue", linewidth=1.0, label="Actual", alpha=0.9)
        if pred_col in hdf.columns:
            ax.plot(hdf["Date"], hdf[pred_col],
                    color="darkorange", linewidth=1.0, linestyle="--",
                    label="Predicted", alpha=0.9)

        if not htest.empty:
            ax.axvspan(htest["Date"].min(), htest["Date"].max(),
                       alpha=0.08, color="red", label="Test (heating season)")

        ax.set_ylabel("kWh/day", fontsize=8)
        ax.set_title(f"HH {hid}", fontsize=9)
        ax.legend(fontsize=7, loc="upper left", ncol=3)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
        ax.tick_params(axis="x", labelsize=7)

    fig.suptitle(f"{model_name}  —  Daily Consumption: Actual vs. Predicted", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, save_path)


def plot_timeseries_comparison(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    pred_cols: dict,
    household_ids: list,
    save_path: Path,
) -> None:
    """All models overlaid for 2 households (one treatment, one control).

    pred_cols : dict mapping display_name -> column_name present in df_val / df_test
    """
    nrows = len(household_ids)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 4 * nrows), sharex=False)
    if nrows == 1:
        axes = [axes]

    colours = list(MODEL_COLOURS.values())

    for ax, hid in zip(axes, household_ids):
        hval  = df_val[df_val["Household_ID"]  == hid].sort_values("Date")
        htest = df_test[df_test["Household_ID"] == hid].sort_values("Date")
        hdf   = pd.concat([hval, htest])

        ax.plot(hdf["Date"], hdf["kWh_received_Total"],
                color="black", linewidth=1.4, label="Actual", zorder=10)

        for i, (name, col) in enumerate(pred_cols.items()):
            if col in hdf.columns:
                ax.plot(hdf["Date"], hdf[col],
                        color=colours[i % len(colours)],
                        linewidth=0.9, linestyle="--", alpha=0.8, label=name)

        if not htest.empty:
            ax.axvspan(htest["Date"].min(), htest["Date"].max(),
                       alpha=0.07, color="red")

        group = str(hdf["Group"].iloc[0]) if "Group" in hdf.columns else "?"
        ax.set_title(f"HH {hid}  ({group})", fontsize=10)
        ax.set_ylabel("kWh/day")
        ax.legend(fontsize=7, loc="upper left", ncol=4)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(8))

    fig.suptitle("All Models  —  Daily Predictions Comparison", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, save_path)


def plot_seasonal_barplot(
    seasonal_df: pd.DataFrame,
    metric: str,
    save_path: Path,
) -> None:
    """Grouped bar chart: RMSE or MAE by model x season.

    seasonal_df columns: Model, Period, <metric>
    """
    models  = [m for m in seasonal_df["Model"].unique() if "Baseline" not in m]
    periods = list(seasonal_df["Period"].unique())
    x = np.arange(len(models))
    width = 0.8 / len(periods)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, period in enumerate(periods):
        vals = []
        for m in models:
            row = seasonal_df[(seasonal_df["Model"] == m) & (seasonal_df["Period"] == period)]
            vals.append(float(row[metric].values[0]) if len(row) > 0 else 0.0)
        offset = (i - len(periods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=period, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"{metric} (kWh/day)")
    ax.set_title(f"Seasonal Performance \u2014 {metric} by Model and Period")
    ax.legend(title="Period", fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)


def plot_cv_errorbar(
    cv_df: pd.DataFrame,
    test_metrics_df: Optional[pd.DataFrame],
    save_path: Path,
) -> None:
    """Horizontal bar chart: CV RMSE mean +/- std; test RMSE as diamond overlay.

    cv_df columns: Model, CV_RMSE_Mean, CV_RMSE_Std
    """
    df = cv_df.sort_values("CV_RMSE_Mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.6)))

    colours = [MODEL_COLOURS.get(m, "#4e79a7") for m in df["Model"]]
    ax.barh(df["Model"], df["CV_RMSE_Mean"], xerr=df["CV_RMSE_Std"],
            color=colours, alpha=0.75, capsize=4, error_kw={"linewidth": 1.5})

    if test_metrics_df is not None:
        plotted_label = False
        for _, row in df.iterrows():
            match = test_metrics_df[test_metrics_df["Model"] == row["Model"]]
            if not match.empty:
                label = "Test RMSE" if not plotted_label else ""
                ax.scatter(match["RMSE"].values[0], row["Model"],
                           marker="D", color="red", s=55, zorder=5, label=label)
                plotted_label = True

    ax.set_xlabel("RMSE (kWh/day)")
    ax.set_title("Cross-Validation RMSE (mean \u00b1 std)  \u2014  Train Set\nRed diamonds = Test RMSE")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)


def plot_data_volume_scatter(
    volume_df: pd.DataFrame,
    save_path: Path,
    min_days_threshold: int = 180,
) -> None:
    """Per-household MAE vs. training days, with LOWESS trend.

    volume_df columns: Household_ID, training_days, mae_<model> ...
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    mae_cols = [c for c in volume_df.columns if c.startswith("mae_")]
    if not mae_cols:
        logger.warning("plot_data_volume_scatter: no mae_* columns found — skipping")
        return

    n = len(mae_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, col in zip(axes[0], mae_cols):
        model_name = col.replace("mae_", "").replace("_", " ").title()
        valid = volume_df[["training_days", col]].dropna()
        ax.scatter(valid["training_days"], valid[col],
                   alpha=0.3, s=15, color="#4e79a7", rasterized=True)
        ax.axvline(min_days_threshold, color="grey", linestyle="--",
                   linewidth=1.2, label=f"min_days={min_days_threshold}")

        order = np.argsort(valid["training_days"].values)
        smooth = lowess(valid[col].values[order],
                        valid["training_days"].values[order],
                        frac=0.3, return_sorted=False)
        ax.plot(valid["training_days"].values[order], smooth,
                color="orange", linewidth=2, label="LOWESS")

        ax.set_xlabel("Training days available")
        ax.set_ylabel("Per-HH MAE (kWh/day)")
        ax.set_title(model_name)
        ax.legend(fontsize=8)

    fig.suptitle("Per-Household MAE vs. Training Data Volume", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, save_path)


def plot_ablation_barplot(
    ablation_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Grouped bar chart for feature-set ablation.

    ablation_df columns: Config, Model, RMSE
    """
    models  = list(ablation_df["Model"].unique())
    configs = list(ablation_df["Config"].unique())
    x = np.arange(len(configs))
    width = 0.7 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = []
        for cfg in configs:
            row = ablation_df[(ablation_df["Model"] == model) & (ablation_df["Config"] == cfg)]
            vals.append(float(row["RMSE"].values[0]) if len(row) > 0 else float("nan"))
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=model, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("RMSE (kWh/day)")
    ax.set_title("Feature-Set Ablation  \u2014  RMSE by Configuration")
    ax.legend(title="Model", fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)


def plot_significance_heatmap(
    pval_df: pd.DataFrame,
    save_path: Path,
    alpha_bonferroni: float = 0.0033,
    alpha_nominal: float = 0.05,
) -> None:
    """Heatmap of Wilcoxon p-values with three-tier significance colouring."""
    # Build annotation matrix
    annot = pval_df.copy().astype(object)
    for r in pval_df.index:
        for c in pval_df.columns:
            v = pval_df.loc[r, c]
            if r == c:
                annot.loc[r, c] = "\u2014"
            elif pd.isna(v):
                annot.loc[r, c] = "N/A"
            else:
                annot.loc[r, c] = f"{float(v):.3f}"

    # Numeric colour matrix: 0.9=not sig, 0.5=nominally sig, 0.1=Bonferroni sig
    colour_vals = pd.DataFrame(
        np.full(pval_df.shape, 0.9),
        index=pval_df.index, columns=pval_df.columns,
    )
    for r in pval_df.index:
        for c in pval_df.columns:
            v = pval_df.loc[r, c]
            if r == c:
                colour_vals.loc[r, c] = 1.0
            elif not pd.isna(v):
                if float(v) < alpha_bonferroni:
                    colour_vals.loc[r, c] = 0.1
                elif float(v) < alpha_nominal:
                    colour_vals.loc[r, c] = 0.5

    n = len(pval_df)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
    cmap = sns.color_palette(["#d73027", "#fee08b", "#ffffff", "#1a9850"], as_cmap=True)
    sns.heatmap(
        colour_vals.astype(float), annot=annot, fmt="",
        cmap=cmap, vmin=0, vmax=1,
        linewidths=0.5, linecolor="lightgrey",
        cbar=False,
        ax=ax,
    )
    ax.set_title(
        "Pairwise Wilcoxon Signed-Rank p-values\n"
        f"Green p<{alpha_bonferroni} (Bonferroni)  |  "
        f"Yellow p<{alpha_nominal} (nominal)  |  "
        "Red = not significant",
        fontsize=10,
    )
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, save_path)
