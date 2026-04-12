"""
src/interpretability.py

SHAP analysis (TreeExplainer, LinearExplainer, KernelExplainer)
and permutation importance computation for Phase 10.

Public API
----------
compute_permutation_importance(model, X, y, features, log_target, scaler,
                               n_repeats, random_state) -> pd.DataFrame
compute_shap_tree(model, X, feature_names) -> tuple[np.ndarray, float]
compute_shap_linear(model, X_raw, X_bg_raw, scaler, feature_names) -> tuple[np.ndarray, float]
compute_shap_kernel(predict_fn, X_bg_scaled, X_test_scaled, feature_names,
                    batch_size, nsamples, logger) -> np.ndarray

plot_shap_beeswarm(shap_values, X_display, feature_names, model_name,
                   max_display, save_path)
plot_shap_bar(shap_values, feature_names, model_name, max_display, save_path)
plot_shap_dependence(shap_values, X_display_df, feature_name, feature_names,
                     save_path, interaction_feature)
plot_shap_waterfall(shap_values_row, base_value, X_row, feature_names,
                    case_label, y_true_val, y_pred_val, save_path)
plot_shap_force(base_value, shap_values_row, X_row, feature_names,
                case_label, save_path)
plot_permutation_importance(imp_df, model_name, color, save_path, top_n)
plot_all_models_permutation(imp_dict, feature_order, model_colours, save_path)
plot_dt_tree(dt_model, feature_names, save_path, max_depth)
plot_elasticnet_coefficients(model, feature_names, save_path)
plot_feature_ranking_heatmap(ranking_df, save_path)
plot_spearman_heatmap(rho_df, save_path)
plot_accuracy_interpretability_tradeoff(model_data, baseline_rmse, save_path)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

FIG_DPI = 150


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", path)


def _save_current(path: Path) -> None:
    """Save the current matplotlib figure (used after shap plot calls)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close("all")
    logger.info("  Saved %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Permutation Importance
# ─────────────────────────────────────────────────────────────────────────────

def compute_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    log_target: bool = False,
    scaler=None,
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute permutation importance in raw-kWh space for any model.

    The scorer back-transforms log1p predictions before computing RMSE, so
    importance values are always in kWh space regardless of training target.
    Larger values = more important (mean RMSE increase when feature is shuffled).

    Parameters
    ----------
    model       : fitted sklearn / XGBoost / LightGBM model
    X           : raw (unscaled) feature matrix, shape (N, len(features))
    y           : raw kWh target
    features    : feature name list (same order as X columns)
    log_target  : if True, model was trained on log1p(y) — apply expm1
    scaler      : if not None, applied inside the scorer wrapper
    n_repeats   : number of shuffles per feature
    random_state: reproducibility seed

    Returns
    -------
    DataFrame with columns: feature, importance_mean, importance_std
    sorted descending by importance_mean.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_squared_error

    def _scorer(estimator, X_in, y_in):
        X_input = scaler.transform(X_in) if scaler is not None else X_in
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = estimator.predict(X_input)
        if log_target:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0.0, None)
        return -float(np.sqrt(mean_squared_error(y_in, pred)))  # negative RMSE

    result = permutation_importance(
        model,
        X,
        y,
        scoring=_scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    df = pd.DataFrame({
        "feature":          features,
        "importance_mean":  result.importances_mean,
        "importance_std":   result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHAP computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_tree(
    model,
    X: np.ndarray,
    feature_names: list[str],
):
    """Compute SHAP values for a tree-based model (RF, XGB, LGBM, DT).

    Uses shap.TreeExplainer which is exact and fast.

    Returns
    -------
    shap_values  : np.ndarray, shape (N, len(feature_names))
    base_value   : float, the mean prediction (intercept)
    explanation  : shap.Explanation object for waterfall/force plots
    """
    import shap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X, check_additivity=False)

    shap_values = explanation.values
    if shap_values.ndim == 3:
        # Some versions return (N, features, 1) for single-output regressors
        shap_values = shap_values[:, :, 0]

    base_val = float(np.atleast_1d(explainer.expected_value)[0])
    logger.info(
        "  TreeExplainer: shape=%s  base_value=%.3f kWh",
        shap_values.shape,
        base_val,
    )
    return shap_values, base_val, explanation


def compute_shap_linear(
    model,
    X_raw: np.ndarray,
    X_bg_raw: np.ndarray,
    scaler,
    feature_names: list[str],
):
    """Compute SHAP values for the ElasticNet (log-kWh space).

    SHAP values are in log-kWh space because the model was trained on log1p(y).
    Axes on plots must be labelled 'SHAP value (log kWh)'.

    Parameters
    ----------
    model    : fitted ElasticNet
    X_raw    : raw (unscaled) test features, shape (N, 30)
    X_bg_raw : raw background dataset for masker, shape (M, 30)
    scaler   : fitted StandardScaler

    Returns
    -------
    shap_values : np.ndarray, shape (N, 30)
    base_value  : float
    """
    import shap

    X_scaled    = scaler.transform(X_raw)
    X_bg_scaled = scaler.transform(X_bg_raw)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer   = shap.LinearExplainer(model, X_bg_scaled, feature_names=feature_names)
        shap_values = explainer.shap_values(X_scaled)

    base_val = float(np.atleast_1d(explainer.expected_value)[0])
    logger.info(
        "  LinearExplainer: shape=%s  base_value=%.4f log-kWh (≈ %.2f kWh)",
        shap_values.shape,
        base_val,
        float(np.expm1(base_val)),
    )
    return shap_values, base_val


def compute_shap_kernel(
    predict_fn,
    X_bg_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    feature_names: list[str],
    batch_size: int = 200,
    nsamples: int = 100,
    progress_logger=None,
) -> np.ndarray:
    """Compute SHAP values for the ANN using KernelExplainer.

    KernelExplainer is model-agnostic but slow. Values are in raw kWh because
    predict_fn applies expm1 back-transformation internally.

    Parameters
    ----------
    predict_fn     : callable(X_scaled) -> kWh predictions (already back-transformed)
    X_bg_scaled    : scaled background dataset, shape (bg_n, 30)
    X_test_scaled  : scaled test rows to explain, shape (N, 30)
    batch_size     : rows to process per batch (for progress logging)
    nsamples       : KernelExplainer sample budget per row

    Returns
    -------
    shap_values : np.ndarray, shape (N, 30) — in raw kWh space
    """
    import shap
    log = progress_logger or logger

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(predict_fn, X_bg_scaled)

    n_rows = len(X_test_scaled)
    shap_batches = []
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        batch = X_test_scaled[start:end]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv_batch = explainer.shap_values(batch, nsamples=nsamples, silent=True)
        shap_batches.append(sv_batch)
        log.info("  ANN KernelExplainer: %d / %d rows processed", end, n_rows)

    shap_values = np.vstack(shap_batches)
    base_val = float(np.atleast_1d(explainer.expected_value)[0])
    log.info(
        "  KernelExplainer done: shape=%s  base_value=%.3f kWh",
        shap_values.shape,
        base_val,
    )
    return shap_values, base_val


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_display: np.ndarray,
    feature_names: list[str],
    model_name: str,
    save_path: Path,
    max_display: int = 20,
    ylabel_suffix: str = "",
) -> None:
    """SHAP summary beeswarm plot.

    X_display should be the UNSCALED feature matrix for interpretable colour axis.
    For log-target models, ylabel_suffix should indicate the space (e.g., 'log kWh').
    """
    import shap

    X_df = pd.DataFrame(X_display, columns=feature_names)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.summary_plot(
            shap_values,
            X_df,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
            plot_size=(9, 7),
        )

    fig = plt.gcf()
    ax  = fig.axes[0]
    units = ylabel_suffix if ylabel_suffix else "kWh"
    ax.set_xlabel(f"SHAP value ({units})")
    ax.set_title(
        f"SHAP Beeswarm — {model_name}\n"
        f"(top {max_display} features, N={len(shap_values):,})",
        fontsize=11,
    )
    _save_current(save_path)


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    model_name: str,
    save_path: Path,
    max_display: int = 20,
    ylabel_suffix: str = "",
) -> None:
    """SHAP mean |value| bar plot."""
    import shap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
            plot_size=(8, 6),
        )

    fig = plt.gcf()
    ax  = fig.axes[0]
    units = ylabel_suffix if ylabel_suffix else "kWh"
    ax.set_xlabel(f"Mean |SHAP value| ({units})")
    ax.set_title(
        f"SHAP Feature Importance — {model_name}\n"
        f"(top {max_display} features by mean |SHAP|, N={len(shap_values):,})",
        fontsize=11,
    )
    _save_current(save_path)


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_display_df: pd.DataFrame,
    feature_name: str,
    feature_names: list[str],
    save_path: Path,
    interaction_feature: str = "auto",
) -> None:
    """SHAP dependence plot for a single feature with interaction colouring."""
    import shap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.dependence_plot(
            feature_name,
            shap_values,
            X_display_df[feature_names],
            interaction_index=interaction_feature,
            show=False,
            alpha=0.4,
        )

    fig = plt.gcf()
    ax  = fig.axes[0]
    ax.set_ylabel(f"SHAP value for {feature_name} (kWh)")
    ax.set_title(f"SHAP Dependence — RF — {feature_name}", fontsize=11)
    fig.tight_layout()
    _save(fig, save_path)


def plot_shap_waterfall(
    shap_values_row: np.ndarray,
    base_value: float,
    X_row: np.ndarray,
    feature_names: list[str],
    case_label: str,
    y_true_val: float,
    y_pred_val: float,
    save_path: Path,
    max_display: int = 15,
) -> None:
    """SHAP waterfall plot for one prediction."""
    import shap

    expl = shap.Explanation(
        values=shap_values_row,
        base_values=base_value,
        data=X_row,
        feature_names=feature_names,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.plots.waterfall(expl, max_display=max_display, show=False)

    fig = plt.gcf()
    fig.suptitle(
        f"RF — {case_label}\n"
        f"y_true = {y_true_val:.1f} kWh  |  pred = {y_pred_val:.1f} kWh  |  "
        f"error = {y_pred_val - y_true_val:+.1f} kWh",
        fontsize=10,
        y=1.01,
    )
    _save(fig, save_path)


def plot_shap_force(
    base_value: float,
    shap_values_row: np.ndarray,
    X_row: np.ndarray,
    feature_names: list[str],
    case_label: str,
    save_path: Path,
) -> None:
    """SHAP force plot (horizontal) for one prediction."""
    import shap

    feat_series = pd.Series(X_row, index=feature_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.force_plot(
            base_value,
            shap_values_row,
            feat_series,
            matplotlib=True,
            show=False,
            figsize=(16, 3),
            text_rotation=15,
        )

    fig = plt.gcf()
    fig.suptitle(f"RF Force Plot — {case_label}", fontsize=9, y=1.02)
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Permutation importance plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_permutation_importance(
    imp_df: pd.DataFrame,
    model_name: str,
    color: str,
    save_path: Path,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of top-N permutation importances with std error bars."""
    df = imp_df.head(top_n).iloc[::-1]  # reverse for horizontal bar (most important at top)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        df["feature"],
        df["importance_mean"],
        xerr=df["importance_std"],
        color=color,
        alpha=0.85,
        error_kw={"elinewidth": 1.2, "capsize": 3},
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean RMSE increase when feature is shuffled (kWh)")
    ax.set_title(
        f"Permutation Importance — {model_name}\n"
        f"(top {top_n} features, test set, n_repeats=10)",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, save_path)


def plot_all_models_permutation(
    imp_dict: dict,
    feature_order: list[str],
    model_colours: dict,
    save_path: Path,
    top_n: int = 20,
) -> None:
    """2×3 faceted bar chart comparing permutation importance across all models.

    Parameters
    ----------
    imp_dict     : {model_name: DataFrame(feature, importance_mean, importance_std)}
    feature_order: feature names sorted by RF importance (top top_n only)
    model_colours: colour map from evaluation.py
    """
    model_names = list(imp_dict.keys())
    n_models    = len(model_names)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows), sharey=True)
    axes = axes.flatten()

    for i, name in enumerate(model_names):
        ax  = axes[i]
        df  = imp_dict[name].set_index("feature").reindex(feature_order).iloc[::-1]
        col = model_colours.get(name, "#888888")
        ax.barh(
            df.index,
            df["importance_mean"].fillna(0),
            xerr=df["importance_std"].fillna(0),
            color=col,
            alpha=0.85,
            error_kw={"elinewidth": 0.8, "capsize": 2},
        )
        ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("RMSE increase (kWh)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

    # Hide unused panels
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Permutation Importance — All Models (top 20 features sorted by RF rank)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Decision Tree visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_dt_tree(
    dt_model,
    feature_names: list[str],
    save_path: Path,
    max_depth: int = 4,
) -> None:
    """Visualize the first max_depth levels of the Decision Tree."""
    from sklearn.tree import plot_tree, export_graphviz

    fig, ax = plt.subplots(figsize=(28, 10))
    plot_tree(
        dt_model,
        max_depth=max_depth,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax,
        impurity=False,
        proportion=False,
        precision=2,
    )
    ax.set_title(
        f"Decision Tree — First {max_depth} Levels  "
        f"(45 features, test RMSE=14.44 kWh, R²=0.575)",
        fontsize=12,
    )
    _save(fig, save_path)

    # Also export DOT for the report appendix
    dot_path = save_path.parent / "phase10_dt_tree.dot"
    export_graphviz(
        dt_model,
        max_depth=max_depth,
        out_file=str(dot_path),
        feature_names=feature_names,
        filled=True,
        rounded=True,
    )
    logger.info("  Exported DT DOT file: %s", dot_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ElasticNet coefficient plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_elasticnet_coefficients(
    model,
    feature_names: list[str],
    save_path: Path,
) -> None:
    """Horizontal bar chart of ElasticNet standardized coefficients."""
    coef = np.asarray(model.coef_).ravel()
    coef_df = (
        pd.DataFrame({"feature": feature_names, "coefficient": coef})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=True)   # ascending for horizontal bar
    )

    colors = ["#e15759" if c > 0 else "#4e79a7" for c in coef_df["coefficient"]]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)

    intercept_kwh = float(np.expm1(model.intercept_[0]
                                   if hasattr(model.intercept_, "__len__")
                                   else model.intercept_))
    ax.set_xlabel("Standardized Coefficient (log-kWh per unit std)")
    ax.set_title(
        "ElasticNet — Standardized Coefficients\n"
        "(positive = higher consumption  |  negative = lower  |  target: log₁₊ₓ kWh)\n"
        f"Baseline prediction (all features at mean): {intercept_kwh:.1f} kWh/day",
        fontsize=10,
    )

    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#e15759", label="Increases consumption"),
            Patch(color="#4e79a7", label="Decreases consumption"),
        ],
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cross-model ranking plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_ranking_heatmap(
    ranking_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Heatmap of feature importance ranks across models.

    ranking_df: rows=features, columns=model names with rank values (1=most important).
    Accepts either a 'feature' column or feature names as the index.
    """
    df = ranking_df.copy()
    if "feature" in df.columns:
        df = df.set_index("feature")

    fig, ax = plt.subplots(figsize=(11, max(8, len(df) * 0.45)))
    mask = df.isnull()
    ranking_df = df   # reassign for heatmap call below
    sns.heatmap(
        ranking_df,
        cmap="RdYlGn_r",
        annot=True,
        fmt=".0f",
        linewidths=0.4,
        linecolor="white",
        ax=ax,
        mask=mask,
        cbar_kws={"label": "Rank (1 = most important)"},
    )
    ax.set_title(
        "Feature Importance Rank by Model\n"
        "(1 = most important  |  green = top rank  |  red = low rank)",
        fontsize=11,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    _save(fig, save_path)


def plot_spearman_heatmap(
    rho_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Heatmap of Spearman ρ between model feature-importance rankings."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        rho_df.astype(float),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Spearman ρ"},
    )
    ax.set_title(
        "Spearman ρ — Feature Importance Rank Correlation Across Models\n"
        "(ρ = 1 → identical rankings  |  ρ = 0 → uncorrelated)",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Accuracy–Interpretability tradeoff plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_interpretability_tradeoff(
    model_data: list[dict],
    baseline_rmse: float,
    save_path: Path,
) -> None:
    """Scatter plot of test RMSE vs. interpretability score.

    model_data items: {name, rmse, interp_score, color, training_seconds}
    interp_score: 1=black-box (ANN) … 5=fully transparent (ElasticNet)
    Bubble size is proportional to training time.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for m in model_data:
        size = max(100, min(800, m["training_seconds"] / 5))
        ax.scatter(
            m["interp_score"], m["rmse"],
            s=size,
            color=m["color"],
            alpha=0.85,
            zorder=5,
            edgecolors="white",
            linewidth=1.5,
        )
        ax.text(
            m["interp_score"] + 0.07,
            m["rmse"] + 0.1,
            m["name"],
            fontsize=9,
            va="bottom",
        )

    # Baseline reference
    ax.axhline(
        baseline_rmse,
        color="#888888",
        linestyle="--",
        linewidth=1.0,
        label=f"Per-HH Mean baseline (RMSE={baseline_rmse:.2f} kWh)",
    )

    ax.set_xlabel("Interpretability Score\n(1 = black-box → 5 = fully transparent)", fontsize=10)
    ax.set_ylabel("Test RMSE (kWh/day)", fontsize=10)
    ax.set_title(
        "Accuracy – Interpretability Tradeoff\n"
        "(HEAPO-Predict  |  test set Dec 2023 – Mar 2024  |  bubble size ∝ training time)",
        fontsize=11,
    )
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1\n(ANN)", "2\n(RF/XGB/LGBM)", "3\n–", "4\n(DT)", "5\n(ElasticNet)"])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
