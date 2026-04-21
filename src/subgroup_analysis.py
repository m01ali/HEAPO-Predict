"""
src/subgroup_analysis.py

Phase 11 — Subgroup and Bias Analysis helpers.

Public API
----------
build_subgroup_labels(df)               -> df with sg_* columns added
compute_subgroup_metrics(sub, pred_col, y_col, floor) -> dict
run_subgroup_metrics(df, sg_map, pred_cols, y_col, floor) -> DataFrame
mannwhitney_pairwise(df, sg_col, cat_a, cat_b, resid_col) -> dict
kruskal_wallis(df, sg_col, resid_col)   -> dict

plot_bias_heatmap(...)
plot_mae_grouped_bar(...)
plot_residual_boxplots(...)
plot_bias_vs_area(...)
plot_treatment_timeline(...)
plot_subgroup_rmse_table(...)
plot_composition_bar(...)
plot_track_b_bias_heatmap(...)
plot_track_b_residual_boxplot(...)
write_report(...)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal as scipy_kruskal
from scipy.stats import mannwhitneyu
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

FIG_DPI = 150

# ── Consistent colours across all Phase 11 figures ───────────────────────────
MODEL_COLOURS = {
    "ElasticNet": "#4e79a7",
    "DT":         "#f28e2b",
    "RF":         "#59a14f",
    "XGBoost":    "#e15759",
    "LightGBM":   "#76b7b2",
    "ANN":        "#edc948",
    "XGBoost B":  "#b07aa1",
}

# Map display name → prediction column name (as in phase9_test_predictions.parquet)
MODEL_PRED_COLS: Dict[str, str] = {
    "RF":         "pred_rf",
    "XGBoost":    "pred_xgb",
    "LightGBM":   "pred_lgbm",
    "DT":         "pred_dt",
    "ANN":        "pred_ann",
    "ElasticNet": "pred_elasticnet",
}

# Map display name → residual column (actual − predicted, positive = under-pred)
MODEL_RESID_COLS: Dict[str, str] = {
    "RF":         "residual_pred_rf",
    "XGBoost":    "residual_pred_xgb",
    "LightGBM":   "residual_pred_lgbm",
    "DT":         "residual_pred_dt",
    "ANN":        "residual_pred_ann",
    "ElasticNet": "residual_pred_elasticnet",
}

# Ordered living-area bucket categories for consistent axis ordering
AREA_ORDER = ["<100", "100-150", "150-200", "200-300", ">300"]

# Ordered building-age bucket categories
AGE_ORDER = ["pre-1970", "1970-1990", "1990-2010", "post-2010"]

MAPE_FLOOR = 0.5  # kWh — exclude near-zero days from sMAPE (from config)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Subgroup label engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_subgroup_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sg_* categorical columns to *df* using existing binary/encoded columns.

    Expects the following columns to be present (from test_full.parquet merged
    with phase9_test_predictions.parquet):
      hp_type_air_source, hp_type_ground_source, hp_type_unknown
      building_type_house, building_type_apartment
      heat_dist_floor, heat_dist_radiator, heat_dist_both, heat_dist_unknown
      has_pv, has_ev
      living_area_bucket          (already string-bucketed in Phase 6)
      Group                       (treatment / control)
      post_intervention           (0 / 1)
      month                       (integer 1-12)

    Returns the same DataFrame with new sg_* columns added in-place.
    """
    df = df.copy()

    # SG1 — HP type
    def _hp_type(row):
        if row.get("hp_type_air_source") == 1:    return "Air-Source"
        if row.get("hp_type_ground_source") == 1: return "Ground-Source"
        return "Unknown"
    df["sg_hp_type"] = df.apply(_hp_type, axis=1)

    # SG2 — Building type
    def _building_type(row):
        if row.get("building_type_house") == 1:     return "House"
        if row.get("building_type_apartment") == 1: return "Apartment"
        return "Unknown"
    df["sg_building_type"] = df.apply(_building_type, axis=1)

    # SG3 — Heat distribution
    def _heat_dist(row):
        if row.get("heat_dist_floor") == 1:    return "Floor"
        if row.get("heat_dist_radiator") == 1: return "Radiator"
        if row.get("heat_dist_both") == 1:     return "Both"
        return "Unknown"
    df["sg_heat_dist"] = df.apply(_heat_dist, axis=1)

    # SG4 — PV system
    df["sg_pv"] = df["has_pv"].map({1: "With PV", 0: "Without PV"}).fillna("Unknown")

    # SG5 — Living area bucket (already engineered in Phase 6)
    df["sg_area"] = df["living_area_bucket"].fillna("Unknown").astype(str)

    # SG6 — Group (treatment vs control)
    df["sg_group"] = df["Group"].fillna("Unknown").str.capitalize()

    # SG7 — Intervention status (treatment HHs only)
    df["sg_intervention"] = "Control (no visit)"
    mask_before = (df["Group"] == "treatment") & (df["post_intervention"] == 0)
    mask_after  = (df["Group"] == "treatment") & (df["post_intervention"] == 1)
    df.loc[mask_before, "sg_intervention"] = "Treatment (pre-visit)"
    df.loc[mask_after,  "sg_intervention"] = "Treatment (post-visit)"

    # SG8 — EV ownership
    df["sg_ev"] = df["has_ev"].map({1: "With EV", 0: "Without EV"}).fillna("Unknown")

    # SG9 — Month within test set (Dec/Jan/Feb/Mar)
    month_map = {12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}
    df["sg_month"] = pd.to_datetime(df["Date"]).dt.month.map(month_map).fillna("Other")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def _smape(y_true: np.ndarray, y_pred: np.ndarray, floor: float = MAPE_FLOOR) -> float:
    mask = y_true >= floor
    if mask.sum() == 0:
        return np.nan
    yt, yp = y_true[mask], y_pred[mask]
    return float(100 * np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp) + 1e-9)))


def compute_subgroup_metrics(
    sub: pd.DataFrame,
    pred_col: str,
    y_col: str = "kWh_received_Total",
    floor: float = MAPE_FLOOR,
    min_n: int = 30,
) -> Optional[dict]:
    """
    Compute bias and error metrics for one (subgroup, model) slice.
    Returns None if the slice has fewer than *min_n* rows.

    Residual convention (matches phase9): actual − predicted.
    Positive mean_bias → model under-predicts (too low).
    Negative mean_bias → model over-predicts (too high).
    """
    if len(sub) < min_n or pred_col not in sub.columns:
        return None
    y_true = sub[y_col].values.astype(float)
    y_pred = sub[pred_col].values.astype(float)
    resid  = y_true - y_pred
    return {
        "N":            int(len(sub)),
        "N_households": int(sub["Household_ID"].nunique()),
        "mean_kwh":     float(np.mean(y_true)),
        "mean_bias":    float(np.mean(resid)),        # + = under-pred
        "median_bias":  float(np.median(resid)),
        "MAE":          float(mean_absolute_error(y_true, y_pred)),
        "RMSE":         float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":           float(r2_score(y_true, y_pred)),
        "sMAPE":        _smape(y_true, y_pred, floor),
        "MedAE":        float(np.median(np.abs(resid))),
        "P10_bias":     float(np.percentile(resid, 10)),
        "P90_bias":     float(np.percentile(resid, 90)),
    }


def run_subgroup_metrics(
    df: pd.DataFrame,
    sg_cols_map: Dict[str, str],       # {dimension_name: column_name}
    pred_cols: Dict[str, str],         # {display_name: pred_col}
    y_col: str = "kWh_received_Total",
    floor: float = MAPE_FLOOR,
    min_n: int = 30,
) -> pd.DataFrame:
    """Iterate all (dimension × category × model) combinations and collect metrics."""
    rows = []
    for dim_name, sg_col in sg_cols_map.items():
        if sg_col not in df.columns:
            logger.warning("Subgroup column %s not found — skipping.", sg_col)
            continue
        for cat, sub in df.groupby(sg_col, dropna=False):
            cat_str = str(cat) if not pd.isna(cat) else "Unknown"
            for model_name, pred_col in pred_cols.items():
                m = compute_subgroup_metrics(sub, pred_col, y_col, floor, min_n)
                if m is not None:
                    rows.append({
                        "Dimension": dim_name,
                        "Category":  cat_str,
                        "Model":     model_name,
                        **m,
                    })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def mannwhitney_pairwise(
    df: pd.DataFrame,
    sg_col: str,
    cat_a: str,
    cat_b: str,
    resid_col: str,
    min_n: int = 30,
) -> Optional[dict]:
    """
    Two-sample Mann-Whitney U test on residuals between two subgroup categories.
    Returns None if either group has fewer than *min_n* observations.
    """
    a = df[df[sg_col] == cat_a][resid_col].dropna().values
    b = df[df[sg_col] == cat_b][resid_col].dropna().values
    if len(a) < min_n or len(b) < min_n:
        return None
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return {
        "sg_col":      sg_col,
        "cat_a":       cat_a,
        "cat_b":       cat_b,
        "n_a":         int(len(a)),
        "n_b":         int(len(b)),
        "median_a":    float(np.median(a)),
        "median_b":    float(np.median(b)),
        "delta_median": float(np.median(b) - np.median(a)),
        "stat":        float(stat),
        "p_value":     float(p),
    }


def kruskal_wallis(
    df: pd.DataFrame,
    sg_col: str,
    resid_col: str,
    min_n: int = 30,
) -> Optional[dict]:
    """Kruskal-Wallis H test across all categories of a subgroup column."""
    groups = []
    cats   = []
    for cat, sub in df.groupby(sg_col, dropna=False):
        vals = sub[resid_col].dropna().values
        if len(vals) >= min_n:
            groups.append(vals)
            cats.append(str(cat))
    if len(groups) < 2:
        return None
    stat, p = scipy_kruskal(*groups)
    return {
        "sg_col":      sg_col,
        "n_groups":    len(groups),
        "categories":  cats,
        "stat":        float(stat),
        "p_value":     float(p),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_composition_bar(composition_df: pd.DataFrame, save_path: Path) -> None:
    """
    Horizontal bar chart: for each subgroup dimension, show N_households per category.
    """
    # Exclude "Unknown" rows and overall
    cdf = composition_df[composition_df["Category"] != "Unknown"].copy()
    dims = cdf["Dimension"].unique()

    fig, axes = plt.subplots(len(dims), 1, figsize=(10, 2.2 * len(dims)))
    if len(dims) == 1:
        axes = [axes]

    palette = sns.color_palette("tab10", 6)
    for ax, dim in zip(axes, dims):
        sub = cdf[cdf["Dimension"] == dim].sort_values("N_households", ascending=True)
        bars = ax.barh(sub["Category"], sub["N_households"], color=palette[:len(sub)])
        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{int(row["N_households"])} HH ({row["Pct_rows"]:.1f}%)',
                    va="center", fontsize=8)
        ax.set_title(dim, fontsize=10, fontweight="bold")
        ax.set_xlabel("Unique households")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, sub["N_households"].max() * 1.3)

    plt.suptitle("Test Set Subgroup Composition (Dec 2023 – Mar 2024)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_bias_heatmap(
    metrics_df: pd.DataFrame,
    models: List[str],
    save_path: Path,
    title: str = "Mean Bias by Subgroup and Model (kWh)  [+ = under-prediction]",
) -> None:
    """
    Diverging heatmap: rows = subgroup categories, columns = models, cells = mean_bias.
    """
    # Filter to non-Unknown, non-Control-only rows; use key subgroups
    keep_dims = [
        "HP Type", "Building Type", "Heat Distribution",
        "PV System", "Living Area", "Group", "Intervention Status", "EV Ownership",
    ]
    plot_df = metrics_df[
        (metrics_df["Dimension"].isin(keep_dims)) &
        (~metrics_df["Category"].isin(["Unknown"]))
    ].copy()

    pivot = plot_df.pivot_table(
        index="Category", columns="Model", values="mean_bias", aggfunc="first"
    )
    # Keep only requested models that exist
    cols = [m for m in models if m in pivot.columns]
    pivot = pivot[cols]
    # Drop rows that are all NaN
    pivot = pivot.dropna(how="all")
    # Order columns
    pivot = pivot[cols]

    # Add overall row at bottom
    overall = {}
    for m in cols:
        if m in metrics_df.columns or True:
            sub = metrics_df[(metrics_df["Dimension"] == "Overall") & (metrics_df["Model"] == m)]
            if not sub.empty:
                overall[m] = sub["mean_bias"].values[0]
    if overall:
        overall_row = pd.DataFrame([overall], index=["── Overall ──"])
        pivot = pd.concat([pivot, overall_row])

    vmax = float(np.nanpercentile(np.abs(pivot.values.astype(float)), 95))
    vmax = max(vmax, 2.0)

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.8), max(6, len(pivot) * 0.45)))
    im = ax.imshow(pivot.values.astype(float), cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for r in range(len(pivot)):
        for c in range(len(cols)):
            val = pivot.values[r, c]
            if np.isnan(float(val if val is not None else np.nan)):
                continue
            val_f = float(val)
            text = f"{val_f:+.1f}"
            # Bold border for large bias
            weight = "bold" if abs(val_f) >= 5 else "normal"
            color = "white" if abs(val_f) > vmax * 0.6 else "black"
            ax.text(c, r, text, ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight)

    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cb.set_label("Mean bias (kWh)\n+ = under-prediction  − = over-prediction", fontsize=9)
    ax.set_title(title, fontsize=11, pad=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_mae_grouped_bar(
    metrics_df: pd.DataFrame,
    model: str,
    sg_dims: List[str],
    save_path: Path,
    overall_mae: float,
) -> None:
    """
    Grouped bar chart of MAE per subgroup category, for a single model.
    One subplot per subgroup dimension.
    """
    model_df = metrics_df[metrics_df["Model"] == model].copy()
    plot_dims = [d for d in sg_dims if d in model_df["Dimension"].values]
    n = len(plot_dims)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    color = MODEL_COLOURS.get(model, "#888888")
    for ax, dim in zip(axes, plot_dims):
        sub = model_df[
            (model_df["Dimension"] == dim) & (~model_df["Category"].isin(["Unknown"]))
        ].copy()
        # Order living area
        if dim == "Living Area":
            cat_order = [c for c in AREA_ORDER if c in sub["Category"].values]
            sub["Category"] = pd.Categorical(sub["Category"], categories=cat_order, ordered=True)
            sub = sub.sort_values("Category")
        else:
            sub = sub.sort_values("MAE", ascending=False)

        bars = ax.bar(sub["Category"], sub["MAE"], color=color, alpha=0.85, edgecolor="white")
        ax.axhline(overall_mae, color="black", ls="--", lw=1.2, label=f"Overall MAE = {overall_mae:.2f}")

        for bar, (_, row) in zip(bars, sub.iterrows()):
            pct_dev = 100 * (row["MAE"] - overall_mae) / overall_mae
            sign = "+" if pct_dev >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{sign}{pct_dev:.0f}%", ha="center", va="bottom", fontsize=8)

        ax.set_title(dim, fontsize=10, fontweight="bold")
        ax.set_ylabel("MAE (kWh)" if ax is axes[0] else "")
        ax.set_ylim(0, sub["MAE"].max() * 1.35)
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle(f"MAE by Subgroup — {model} (Test Dec 2023 – Mar 2024)", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_residual_boxplots(
    df: pd.DataFrame,
    sg_col: str,
    sg_label: str,
    resid_cols: Dict[str, str],   # {model_name: resid_col_name}
    save_path: Path,
    cat_order: Optional[List[str]] = None,
    exclude_cats: Optional[List[str]] = None,
) -> None:
    """
    One subplot per model: box+whisker of residuals by subgroup category.
    """
    models = list(resid_cols.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    exclude_cats = exclude_cats or ["Unknown"]
    plot_df = df[~df[sg_col].isin(exclude_cats)].copy()

    # Determine category order
    if cat_order is None:
        cats = sorted(plot_df[sg_col].dropna().unique())
    else:
        cats = [c for c in cat_order if c in plot_df[sg_col].values]

    for ax, model in zip(axes, models):
        rcol = resid_cols[model]
        if rcol not in plot_df.columns:
            continue
        data_by_cat = [plot_df[plot_df[sg_col] == c][rcol].dropna().values for c in cats]
        # Filter empty
        cats_plot = [c for c, d in zip(cats, data_by_cat) if len(d) >= 10]
        data_plot = [d for d in data_by_cat if len(d) >= 10]

        bp = ax.boxplot(data_plot, labels=cats_plot, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.5))
        color = MODEL_COLOURS.get(model, "#888888")
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(0, color="red", ls="--", lw=1.0, alpha=0.8)
        ax.set_title(model, fontsize=10, fontweight="bold", color=MODEL_COLOURS.get(model, "black"))
        ax.set_xlabel(sg_label)
        ax.set_ylabel("Residual (kWh)\nactual − predicted" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)

        for i, (cat, data) in enumerate(zip(cats_plot, data_plot), start=1):
            med = np.median(data)
            n_pts = len(data)
            sign = "+" if med >= 0 else ""
            ax.text(i, ax.get_ylim()[0] + 0.5,
                    f"n={n_pts}\n{sign}{med:.1f}",
                    ha="center", va="bottom", fontsize=7, color="dimgrey")

    fig.suptitle(f"Residuals by {sg_label} (Test Dec 2023 – Mar 2024)", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_bias_vs_area(
    df: pd.DataFrame,
    resid_col: str,
    model_name: str,
    save_path: Path,
) -> None:
    """
    Per-household scatter: mean living area vs. mean residual, coloured by HP type.
    """
    hh_level = (
        df.groupby("Household_ID")
        .agg(
            mean_area=("Survey_Building_LivingArea", "mean"),
            mean_bias=(resid_col, "mean"),
            hp_type=("sg_hp_type", "first"),
        )
        .reset_index()
        .dropna(subset=["mean_area", "mean_bias"])
    )

    hp_palette = {
        "Air-Source":    "#e15759",
        "Ground-Source": "#4e79a7",
        "Unknown":       "#aaaaaa",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for hp, grp in hh_level.groupby("hp_type"):
        ax.scatter(grp["mean_area"], grp["mean_bias"],
                   color=hp_palette.get(hp, "#aaaaaa"),
                   alpha=0.65, s=55, label=hp, edgecolors="white", lw=0.4)

    ax.axhline(0, color="black", ls="--", lw=1.0)
    for threshold in (150, 250):
        ax.axvline(threshold, color="grey", ls=":", lw=0.8)
        ax.text(threshold + 2, ax.get_ylim()[1] * 0.95,
                f"{threshold} m²", fontsize=8, color="grey")

    # LOWESS smoother
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sorted_area = hh_level.sort_values("mean_area")
        smoothed = lowess(sorted_area["mean_bias"], sorted_area["mean_area"], frac=0.4)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", lw=2, label="LOWESS trend")
    except ImportError:
        logger.warning("statsmodels not available — skipping LOWESS smoother.")

    ax.set_xlabel("Mean living area (m²)", fontsize=11)
    ax.set_ylabel(f"Per-household mean residual (kWh)\nactual − predicted  [{model_name}]", fontsize=11)
    ax.set_title(f"Per-Household Prediction Bias vs. Living Area — {model_name}", fontsize=12)
    ax.legend(title="HP type", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_treatment_timeline(
    df_val: pd.DataFrame,
    pred_col: str,
    model_name: str,
    save_path: Path,
    n_hh: int = 5,
) -> None:
    """
    For the N treatment households with the largest observed post-visit consumption
    reduction (on the val set), plot actual vs. predicted over time with visit marker.
    """
    if "sg_intervention" not in df_val.columns or "Group" not in df_val.columns:
        logger.warning("Missing sg_intervention/Group — skipping treatment timeline.")
        return
    if pred_col not in df_val.columns:
        logger.warning("Prediction column %s not found in val df — skipping timeline.", pred_col)
        return

    treat = df_val[df_val["Group"] == "treatment"].copy()
    treat["Date"] = pd.to_datetime(treat["Date"])

    # Compute per-HH consumption reduction (pre − post)
    hh_stats = []
    for hh, grp in treat.groupby("Household_ID"):
        pre  = grp[grp["post_intervention"] == 0]["kWh_received_Total"].mean()
        post = grp[grp["post_intervention"] == 1]["kWh_received_Total"].mean()
        if not np.isnan(pre) and not np.isnan(post):
            hh_stats.append({"Household_ID": hh, "reduction": pre - post})
    if not hh_stats:
        logger.warning("No treatment households with both pre and post data — skipping timeline.")
        return

    stats_df = pd.DataFrame(hh_stats).sort_values("reduction", ascending=False)
    top_hh = stats_df.head(n_hh)["Household_ID"].tolist()

    fig, axes = plt.subplots(len(top_hh), 1, figsize=(13, 3.5 * len(top_hh)))
    if len(top_hh) == 1:
        axes = [axes]

    for ax, hh in zip(axes, top_hh):
        hh_df = treat[treat["Household_ID"] == hh].sort_values("Date")

        # Find approximate visit date: last date of post_intervention==0
        pre_dates  = hh_df[hh_df["post_intervention"] == 0]["Date"]
        post_dates = hh_df[hh_df["post_intervention"] == 1]["Date"]
        if pre_dates.empty or post_dates.empty:
            visit_date = None
        else:
            visit_date = pre_dates.max()

        ax.plot(hh_df["Date"], hh_df["kWh_received_Total"],
                lw=1.5, color="#2c7bb6", label="Actual", alpha=0.9)
        ax.plot(hh_df["Date"], hh_df[pred_col],
                lw=1.5, color="#d7191c", ls="--", label=f"{model_name} prediction", alpha=0.9)

        if visit_date is not None:
            ax.axvline(visit_date, color="green", ls=":", lw=1.5, label="~Visit date")
            ax.text(visit_date, ax.get_ylim()[1] * 0.9, "Visit",
                    color="green", fontsize=8, ha="left")

        # Shade pre/post
        if visit_date is not None:
            ax.axvspan(hh_df["Date"].min(), visit_date,
                       alpha=0.05, color="blue", label="Pre-visit")
            ax.axvspan(visit_date, hh_df["Date"].max(),
                       alpha=0.05, color="orange", label="Post-visit")

        ax.set_title(f"HH {hh}", fontsize=9, fontweight="bold")
        ax.set_ylabel("kWh/day")
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=8, ncol=4)

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Treatment Households: Actual vs. {model_name} — Validation Set",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_subgroup_rmse_table(
    metrics_df: pd.DataFrame,
    models: List[str],
    save_path: Path,
    key_dims: Optional[List[str]] = None,
) -> None:
    """
    Styled table as a matplotlib figure: subgroup × model, values = RMSE (kWh).
    """
    if key_dims is None:
        key_dims = ["HP Type", "PV System", "Building Type", "Living Area", "Group"]

    filt = metrics_df[
        (metrics_df["Dimension"].isin(key_dims)) &
        (~metrics_df["Category"].isin(["Unknown"]))
    ].copy()

    pivot = filt.pivot_table(
        index=["Dimension", "Category"], columns="Model",
        values="RMSE", aggfunc="first"
    )
    cols = [m for m in models if m in pivot.columns]
    pivot = pivot[cols].reset_index()
    pivot.index = [f"{r['Dimension']} — {r['Category']}" for _, r in pivot.iterrows()]
    pivot = pivot.drop(columns=["Dimension", "Category"])

    n_rows = len(pivot)
    n_cols = len(cols)
    fig, ax = plt.subplots(figsize=(2.2 * n_cols + 2, 0.45 * n_rows + 1.5))
    ax.axis("off")

    vals = pivot.values.astype(float)
    cell_text = [[f"{v:.2f}" if not np.isnan(v) else "—" for v in row] for row in vals]

    # Colour map: green-to-red within each column (lower RMSE = greener)
    cell_colors = []
    for row in vals:
        row_colors = []
        for i, v in enumerate(row):
            col_vals = vals[:, i]
            col_min = np.nanmin(col_vals)
            col_max = np.nanmax(col_vals)
            if np.isnan(v) or col_max == col_min:
                row_colors.append("#f5f5f5")
            else:
                norm = (v - col_min) / (col_max - col_min)  # 0=best, 1=worst
                r = int(220 * norm + 100 * (1 - norm))
                g = int(100 * norm + 200 * (1 - norm))
                b = int(100 * norm + 100 * (1 - norm))
                row_colors.append(f"#{min(r,255):02x}{min(g,255):02x}{min(b,255):02x}")
        cell_colors.append(row_colors)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=pivot.index.tolist(),
        colLabels=cols,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax.set_title("RMSE (kWh) by Subgroup and Model\n(Test set Dec 2023 – Mar 2024)",
                 fontsize=11, pad=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_track_b_bias_heatmap(
    metrics_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Bias heatmap for Track B protocol subgroups (XGBoost B only)."""
    filt = metrics_df[
        (~metrics_df["Category"].isin(["Unknown", "nan"])) &
        (metrics_df["Model"] == "XGBoost B")
    ].copy()
    if filt.empty:
        logger.warning("No Track B metrics to plot — skipping heatmap.")
        return

    filt["Row"] = filt["Dimension"] + " — " + filt["Category"].astype(str)
    pivot = filt.set_index("Row")[["mean_bias"]].copy()

    vmax = float(np.nanpercentile(np.abs(pivot.values.astype(float)), 95))
    vmax = max(vmax, 2.0)

    fig, ax = plt.subplots(figsize=(4, max(4, 0.45 * len(pivot))))
    im = ax.imshow(pivot.values.astype(float), cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([0])
    ax.set_xticklabels(["XGBoost B"], fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    for r, val in enumerate(pivot.values):
        val_f = float(val[0])
        if not np.isnan(val_f):
            color = "white" if abs(val_f) > vmax * 0.6 else "black"
            ax.text(0, r, f"{val_f:+.1f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cb.set_label("Mean bias (kWh)", fontsize=9)
    ax.set_title("Track B Protocol Subgroups\nMean Bias — XGBoost B", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_track_b_residual_boxplot(
    df_b: pd.DataFrame,
    sg_col: str,
    sg_label: str,
    resid_col: str,
    save_path: Path,
    cat_order: Optional[List[str]] = None,
) -> None:
    """Box plots of XGBoost B residuals by a Track B protocol subgroup."""
    plot_df = df_b.dropna(subset=[sg_col, resid_col]).copy()
    plot_df[sg_col] = plot_df[sg_col].astype(str)
    cats = cat_order if cat_order else sorted(plot_df[sg_col].unique())
    cats = [c for c in cats if c in plot_df[sg_col].values]

    data_by_cat = [plot_df[plot_df[sg_col] == c][resid_col].dropna().values for c in cats]
    cats_plot = [c for c, d in zip(cats, data_by_cat) if len(d) >= 5]
    data_plot = [d for d in data_by_cat if len(d) >= 5]

    if not cats_plot:
        logger.warning("No data for Track B boxplot %s — skipping.", sg_col)
        return

    fig, ax = plt.subplots(figsize=(max(6, len(cats_plot) * 2), 5))
    bp = ax.boxplot(data_plot, labels=cats_plot, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(MODEL_COLOURS["XGBoost B"])
        patch.set_alpha(0.75)

    ax.axhline(0, color="red", ls="--", lw=1.0, alpha=0.8)
    for i, (cat, data) in enumerate(zip(cats_plot, data_plot), start=1):
        med = np.median(data)
        sign = "+" if med >= 0 else ""
        ax.text(i, ax.get_ylim()[0] * 0.95,
                f"n={len(data)}\n{sign}{med:.1f}",
                ha="center", va="top", fontsize=8, color="dimgrey")

    ax.set_xlabel(sg_label)
    ax.set_ylabel("Residual (kWh)  actual − predicted")
    ax.set_title(f"Track B Residuals by {sg_label} — XGBoost B", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    metrics_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    mw_df: pd.DataFrame,
    kw_results: List[dict],
    treatment_stats: dict,
    track_b_metrics: pd.DataFrame,
    save_path: Path,
    n_test_rows: int,
    n_test_hh: int,
) -> None:
    """Write the consolidated Phase 11 text report."""

    def _sep(char="=", width=70):
        return char * width

    lines = [
        _sep(),
        "HEAPO-Predict — Phase 11: Subgroup and Bias Analysis Report",
        _sep(),
        f"Test set : Dec 2023 – Mar 2024  |  N={n_test_rows:,} rows  |  Households={n_test_hh}",
        f"Track A models : RF, XGBoost, LightGBM, DT, ANN, ElasticNet",
        f"Track B models : XGBoost B, DT B, RF B",
        "",
    ]

    # ── Section 1: Composition (Track A only) ────────────────────────────────
    lines += [_sep(), "Section 1 — Subgroup Composition", _sep()]
    if composition_df is not None and not composition_df.empty:
        for dim in composition_df["Dimension"].unique():
            lines.append(f"\n  {dim}")
            sub = composition_df[composition_df["Dimension"] == dim]
            for _, row in sub.iterrows():
                lines.append(f"    {row['Category']:<35} {row['N_households']:>4} HH  "
                             f"{row['N_rows']:>7} rows  {row['Pct_rows']:>5.1f}%")
    else:
        lines.append("  Track A not selected — composition not computed.")

    # ── Section 2: Per-subgroup bias (Track A only) ───────────────────────────
    lines += ["", _sep(), "Section 2 — Per-Subgroup Bias (RF — focus model)", _sep()]
    if metrics_df is not None and not metrics_df.empty:
        rf_overall = metrics_df[(metrics_df["Model"] == "RF") & (metrics_df["Dimension"] == "Overall")]
        if not rf_overall.empty:
            lines.append(
                f"  Overall RF MAE = {rf_overall['MAE'].values[0]:.2f} kWh  "
                f"(RMSE = {rf_overall['RMSE'].values[0]:.2f} kWh)"
            )
        lines += [
            "",
            f"  {'Dimension':<20} {'Category':<25} {'N':>7} {'Mean bias':>10} {'MAE':>8} {'RMSE':>8} {'R2':>7}",
            f"  {'-'*20} {'-'*25} {'-'*7} {'-'*10} {'-'*8} {'-'*8} {'-'*7}",
        ]
        rf_df = metrics_df[
            (metrics_df["Model"] == "RF") & (metrics_df["Dimension"] != "Overall")
        ].copy()
        rf_df["abs_bias"] = rf_df["mean_bias"].abs()
        rf_df = rf_df.sort_values("abs_bias", ascending=False)
        for _, row in rf_df.iterrows():
            if row["Category"] in ("Unknown",):
                continue
            flag = " ◄" if row["abs_bias"] >= 2.0 else ""
            lines.append(
                f"  {row['Dimension']:<20} {row['Category']:<25} {row['N']:>7,} "
                f"{row['mean_bias']:>+9.2f}  {row['MAE']:>7.2f}  {row['RMSE']:>7.2f}  {row['R2']:>6.3f}{flag}"
            )
        lines += ["", "  ◄ = |mean_bias| > 2.0 kWh (> 27% of overall RF MAE of 7.47 kWh)"]
    else:
        lines.append("  Track A not selected — subgroup bias not computed.")

    # ── Section 3: Cross-model consistency (Track A only) ────────────────────
    lines += ["", _sep(), "Section 3 — Cross-Model Bias Consistency", _sep(),
              "  Mean bias per model (all test rows combined):", ""]
    if metrics_df is not None and not metrics_df.empty:
        overall = metrics_df[metrics_df["Dimension"] == "Overall"]
        for _, row in overall.sort_values("RMSE").iterrows():
            lines.append(f"    {row['Model']:<12} mean_bias={row['mean_bias']:>+6.2f} kWh  "
                         f"MAE={row['MAE']:.2f}  RMSE={row['RMSE']:.2f}  R2={row['R2']:.3f}")
    else:
        lines.append("  Track A not selected.")

    # ── Section 4: Treatment effect (Track A only) ────────────────────────────
    lines += ["", _sep(), "Section 4 — Treatment Effect Analysis", _sep()]
    if treatment_stats:
        for key, val in treatment_stats.items():
            lines.append(f"  {key:<45}: {val}")
    else:
        lines.append("  Track A not selected — treatment analysis not computed.")

    # ── Section 5: Statistical testing (Track A only) ────────────────────────
    lines += ["", _sep(), "Section 5 — Statistical Testing Summary", _sep()]
    _mw = mw_df if mw_df is not None else pd.DataFrame()
    n_tests = len(_mw)
    n_sig_raw  = int((_mw["p_value"]     < 0.05).sum()) if not _mw.empty else 0
    n_sig_bonf = int((_mw["p_bonferroni"] < 0.05).sum()) if not _mw.empty else 0
    lines += [
        f"  Mann-Whitney U tests run: {n_tests}",
        f"  Significant (p < 0.05, uncorrected): {n_sig_raw}",
        f"  Significant after Bonferroni correction: {n_sig_bonf}",
        "",
        f"  {'Subgroup':<30} {'A vs B':<40} {'Model':<10} {'Dmedian':>8} {'p (Bonf)':>10} {'Sig':>5}",
        f"  {'-'*30} {'-'*40} {'-'*10} {'-'*8} {'-'*10} {'-'*5}",
    ]
    if not _mw.empty:
        for _, row in _mw.sort_values("p_bonferroni").iterrows():
            sig = "*" if row["p_bonferroni"] < 0.05 else ""
            lines.append(
                f"  {row['sg_col']:<30} {row['cat_a'][:18]:<18} vs {row['cat_b'][:18]:<18}  "
                f"{row['Model']:<10} {row['delta_median']:>+7.2f}  {row['p_bonferroni']:>10.3e}  {sig:>5}"
            )
    lines += ["", "  Kruskal-Wallis H tests (multi-category subgroups, RF residuals):", ""]
    for res in kw_results:
        sig = "*" if res["p_value"] < 0.05 else ""
        cats_str = ", ".join(res["categories"])
        lines.append(f"    {res['sg_col']:<30} H={res['stat']:.2f}  p={res['p_value']:.3e}  "
                     f"groups=[{cats_str}]  {sig}")

    # ── Section 6: Track B protocol subgroups ────────────────────────────────
    lines += ["", _sep(), "Section 6 — Track B Protocol Subgroups (N<=5,475, models: XGBoost B / DT B / RF B)", _sep()]
    if track_b_metrics is not None and not track_b_metrics.empty:
        lines.append(f"  {'Dimension':<25} {'Category':<30} {'Model':<12} {'N':>6} {'Mean bias':>10} {'MAE':>8} {'RMSE':>8}")
        lines.append(f"  {'-'*25} {'-'*30} {'-'*12} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")
        for _, row in track_b_metrics.sort_values(["Dimension", "Model"]).iterrows():
            model_col = row["Model"] if "Model" in row.index else ""
            lines.append(
                f"  {row['Dimension']:<25} {row['Category']:<30} {model_col:<12} {row['N']:>6,} "
                f"{row['mean_bias']:>+9.2f}  {row['MAE']:>7.2f}  {row['RMSE']:>7.2f}"
            )
    else:
        lines.append("  Track B not selected or no metrics computed.")

    lines += [
        "",
        _sep(),
        "Section 7 — Recommendations",
        _sep(),
        "",
        "  1. Ground-Source vs Air-Source HP:",
        "     If RMSE is materially higher for one HP type, consider a stratified",
        "     model or an HP-type × HDD interaction feature to capture the different",
        "     thermal response of ground-coupled vs air-coupled systems.",
        "",
        "  2. PV Households (measurement limitation):",
        "     'With PV' bias partly reflects the self-consumption invisibility",
        "     (Section 2.1.2 of the HEAPO paper). On sunny days, net grid draw is",
        "     low not due to efficiency but due to self-consumption. Adding",
        "     kWh_returned_Total rolling stats as a PV generation proxy may help.",
        "",
        "  3. Large-Area Houses (>250 m²):",
        "     If systematic under-prediction, add Survey_Building_LivingArea²",
        "     as a quadratic term or log-transform the area feature.",
        "",
        "  4. Post-Intervention Treatment Households:",
        "     If model over-predicts post-visit (expects higher consumption than",
        "     actually occurs after HP optimisation), calibrate using a correction",
        "     factor derived from the training-set treatment HHs.",
        "",
        "  5. EV Households:",
        "     EV charging load is highly variable and poorly captured by daily",
        "     static features. Consider adding a rolling EV-load variability",
        "     feature (std of kWh_received_Total in last 7 days per HH).",
        _sep(),
    ]

    save_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written: %s", save_path)
