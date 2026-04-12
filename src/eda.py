"""
src/eda.py

Phase 5 — Exploratory Data Analysis module.
Called by scripts/05_eda.py.

All column names verified against features_full.parquet / features_protocol.parquet
produced by Phase 4. AffectsTimePoint values use spaces: 'before visit', 'after visit'.
"""

import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ── Global plot style ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e0e0e0",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
})

FIGURES_DIR = Path("outputs/figures")
TABLES_DIR  = Path("outputs/tables")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "kWh_received_Total"
FORBIDDEN_COLS = {
    "HeatPump_ElectricityConsumption_YearlyEstimated",
    "EXCLUDED_TARGET_PROXY_HeatPump_ElectricityConsumption_YearlyEstimated",
}
EXPECTED_SHAPES = {
    "features_full":     (913_620, 85),
    "features_protocol": (84_367, 171),
}

PALETTE = sns.color_palette("colorblind", 10)
SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50", "gap": "#E0E0E0"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", name)
    return path


def _annotate_n(ax, n: int, loc: str = "upper right", fontsize: int = 9):
    props = dict(facecolor="white", alpha=0.7, edgecolor="none")
    ax.annotate(f"n = {n:,}", xy=(0.97, 0.95) if loc == "upper right" else (0.03, 0.95),
                xycoords="axes fraction", ha="right" if loc == "upper right" else "left",
                va="top", fontsize=fontsize, bbox=props)


def _split_label(dates: pd.Series, train_end: str, val_end: str) -> pd.Series:
    """Assign 'train' / 'val' / 'test' label based on date."""
    tz = dates.dt.tz
    t_end = pd.Timestamp(train_end, tz=tz)
    v_end = pd.Timestamp(val_end,   tz=tz)
    labels = pd.Series("test", index=dates.index)
    labels[dates <= t_end] = "train"
    labels[(dates > t_end) & (dates <= v_end)] = "val"
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5.0 — Load artifacts
# ═══════════════════════════════════════════════════════════════════════════════

def task50_load(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Phase 4 parquets, assert shapes, guard forbidden columns."""
    logger.info("=" * 60)
    logger.info("TASK 5.0 — Loading Phase 4 artifacts")

    df_a = pd.read_parquet("data/processed/features_full.parquet")
    df_b = pd.read_parquet("data/processed/features_protocol.parquet")

    for name, df, expected in [
        ("features_full",     df_a, EXPECTED_SHAPES["features_full"]),
        ("features_protocol", df_b, EXPECTED_SHAPES["features_protocol"]),
    ]:
        assert df.shape == expected, (
            f"{name}: expected {expected}, got {df.shape}"
        )
        logger.info("%s: %s rows × %s cols  ✓", name, f"{df.shape[0]:,}", df.shape[1])

    for col in FORBIDDEN_COLS:
        assert col not in df_a.columns, f"Target proxy in Track A: {col}"
        assert col not in df_b.columns, f"Target proxy in Track B: {col}"

    assert df_a[TARGET].isna().sum() == 0, "Null target in Track A"
    assert df_b[TARGET].isna().sum() == 0, "Null target in Track B"

    # Weather-valid subset (rows where temperature is non-null)
    df_wx = df_a[df_a["Temperature_avg_daily"].notna()].copy()

    n_outlier = int(df_a["is_iqr_outlier"].sum())
    n_below   = int(df_a["below_min_days_threshold"].sum())
    n_hh_a    = df_a["Household_ID"].nunique()
    n_hh_b    = df_b["Household_ID"].nunique()
    n_hh_wx   = df_wx["Household_ID"].nunique()

    logger.info("Track A  : %s rows, %s households, %s cols",
                f"{len(df_a):,}", n_hh_a, df_a.shape[1])
    logger.info("Track B  : %s rows, %s households, %s cols",
                f"{len(df_b):,}", n_hh_b, df_b.shape[1])
    logger.info("Wx-valid : %s rows, %s households (%.1f%%)",
                f"{len(df_wx):,}", n_hh_wx, 100 * len(df_wx) / len(df_a))
    logger.info("IQR outliers  : %s rows (%.2f%%)", f"{n_outlier:,}",
                100 * n_outlier / len(df_a))
    logger.info("Below threshold HH: %s", n_below)

    return df_a, df_b, df_wx


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5.1 — Target variable analysis
# ═══════════════════════════════════════════════════════════════════════════════

def task51_target_analysis(df_a: pd.DataFrame, df_wx: pd.DataFrame, cfg: dict) -> dict:
    logger.info("=" * 60)
    logger.info("TASK 5.1 — Target variable analysis")

    results = {}
    target = df_a[TARGET]

    # ── 5.1.1 Raw histogram ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(target, bins=120, color=PALETTE[0], alpha=0.85, edgecolor="white", linewidth=0.3)
    for p, q, lbl in [(25, target.quantile(.25), "p25"),
                      (50, target.median(),      "p50"),
                      (75, target.quantile(.75), "p75"),
                      (99, target.quantile(.99), "p99")]:
        ax.axvline(q, color="#e53935", lw=1.5, ls="--", alpha=0.8)
        ax.text(q + 1, ax.get_ylim()[1] * 0.8, f"{lbl}\n{q:.1f}", fontsize=8,
                color="#e53935", va="top")
    ax.set_yscale("log")
    ax.set_xlabel("Daily electricity consumption (kWh)", fontsize=12)
    ax.set_ylabel("Count (log scale)", fontsize=12)
    ax.set_title("Distribution of daily heat pump electricity consumption\n"
                 f"All {len(target):,} household-days (Track A)", fontsize=13, fontweight="bold")
    _annotate_n(ax, len(target))
    fig.tight_layout()
    _save(fig, "05_target_histogram_all.png")

    stats = {
        "mean": float(target.mean()), "median": float(target.median()),
        "std":  float(target.std()),  "min":    float(target.min()),
        "max":  float(target.max()),  "p25":    float(target.quantile(.25)),
        "p75":  float(target.quantile(.75)), "p99": float(target.quantile(.99)),
        "skewness": float(target.skew()), "kurtosis": float(target.kurtosis()),
    }
    for k, v in stats.items():
        logger.info("  kWh_received_Total %-12s: %.4f", k, v)
    results["target_stats"] = stats

    # ── 5.1.2 Per-household mean ──────────────────────────────────────────────
    hh_means = df_a.groupby("Household_ID")[TARGET].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(hh_means, bins=60, color=PALETTE[1], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(hh_means.mean(), color="#e53935", lw=2, ls="--",
               label=f"Mean = {hh_means.mean():.1f} kWh")
    ax.axvline(hh_means.median(), color="#6a1b9a", lw=2, ls="-.",
               label=f"Median = {hh_means.median():.1f} kWh")
    ax.set_xlabel("Per-household mean daily consumption (kWh)", fontsize=12)
    ax.set_ylabel("Number of households", fontsize=12)
    ax.set_title(f"Per-household average daily consumption\n{len(hh_means):,} households",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _annotate_n(ax, len(hh_means))
    fig.tight_layout()
    _save(fig, "05_target_per_household_mean.png")
    low_consumers = int((hh_means < 5).sum())
    logger.info("  Households with mean < 5 kWh/day: %s (%.1f%%)",
                low_consumers, 100 * low_consumers / len(hh_means))
    results["hh_mean_stats"] = {"mean": float(hh_means.mean()), "std": float(hh_means.std()),
                                 "low_consumers_pct": 100 * low_consumers / len(hh_means)}

    # ── 5.1.3 Outlier overlay ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    normal  = df_a[~df_a["is_iqr_outlier"]][TARGET]
    outlier = df_a[ df_a["is_iqr_outlier"]][TARGET]
    axes[0].hist(normal,  bins=100, color=PALETTE[0], alpha=0.8, label="Normal",  edgecolor="white", linewidth=0.3)
    axes[0].hist(outlier, bins=30,  color="#e53935",  alpha=0.9, label=f"IQR outlier (n={len(outlier):,})", edgecolor="white", linewidth=0.3)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("kWh/day", fontsize=11); axes[0].set_ylabel("Count (log)", fontsize=11)
    axes[0].set_title("All rows: normal vs. outlier", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[1].hist(outlier, bins=30, color="#e53935", alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel("kWh/day", fontsize=11); axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title(f"IQR-outlier rows only  ({100*len(outlier)/len(df_a):.2f}% of data)",
                      fontsize=12, fontweight="bold")
    fig.suptitle("IQR-outlier rows in the target variable", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "05_target_histogram_outliers.png")
    results["n_outlier"] = len(outlier)

    # ── 5.1.4 Monthly seasonality boxplot ────────────────────────────────────
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(13, 6))
    data_by_month = [df_wx[df_wx["month"] == m][TARGET].values for m in range(1, 13)]
    bp = ax.boxplot(data_by_month, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
    cmap = plt.cm.RdYlBu_r
    months_heat = {10, 11, 12, 1, 2, 3, 4}
    for i, patch in enumerate(bp["boxes"], start=1):
        c = "#EF5350" if i in months_heat else "#42A5F5"
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_xticklabels(month_labels, fontsize=11)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Daily consumption (kWh)", fontsize=12)
    ax.set_title("Seasonal pattern: daily heat pump consumption by month\n"
                 "(weather-valid rows; red = heating season, blue = non-heating)",
                 fontsize=13, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#EF5350", alpha=0.75, label="Heating season (Oct–Apr)"),
                       Patch(facecolor="#42A5F5", alpha=0.75, label="Non-heating season")],
              fontsize=10)
    fig.tight_layout()
    _save(fig, "05_target_monthly_boxplot.png")

    # ── 5.1.5 Target vs. temperature (paper Figure 4 replication) ────────────
    seed = cfg.get("modeling", {}).get("random_seed", 42)
    n_samp = cfg.get("eda", {}).get("scatter_sample_size", 50_000)
    sample = df_wx.sample(min(n_samp, len(df_wx)), random_state=seed)

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
        frac = cfg.get("eda", {}).get("lowess_frac", 0.10)
        sm = sm_lowess(df_wx[TARGET].values, df_wx["Temperature_avg_daily"].values,
                       frac=frac, it=1, return_sorted=True)
        has_lowess = True
    except Exception:
        has_lowess = False
        logger.warning("statsmodels not available — skipping lowess smoother")

    fig, ax = plt.subplots(figsize=(11, 6))
    air    = sample[sample["hp_type_air_source"]   == 1]
    ground = sample[sample["hp_type_ground_source"] == 1]
    other  = sample[(sample["hp_type_air_source"] == 0) & (sample["hp_type_ground_source"] == 0)]
    ax.scatter(other["Temperature_avg_daily"],  other[TARGET],  s=4, alpha=0.2, color=PALETTE[2], label="Unknown HP type", rasterized=True)
    ax.scatter(ground["Temperature_avg_daily"], ground[TARGET], s=4, alpha=0.2, color=PALETTE[1], label="Ground-source HP", rasterized=True)
    ax.scatter(air["Temperature_avg_daily"],    air[TARGET],    s=4, alpha=0.2, color=PALETTE[0], label="Air-source HP",    rasterized=True)
    if has_lowess:
        ax.plot(sm[:, 0], sm[:, 1], color="black", lw=2.5, label="Lowess (full sample)", zorder=5)
    ax.set_xlabel("Daily mean outdoor temperature (°C)", fontsize=12)
    ax.set_ylabel("Daily consumption (kWh)", fontsize=12)
    ax.set_title("Daily heat pump consumption vs. outdoor temperature\n"
                 f"(sample n={len(sample):,}; replicates paper Figure 4)",
                 fontsize=13, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=10, markerscale=3)
    fig.tight_layout()
    _save(fig, "05_target_vs_temperature.png")

    # ── 5.1.6 Log-transform check ────────────────────────────────────────────
    log_target = np.log1p(df_a[TARGET])
    log_skew = float(log_target.skew())
    raw_skew = stats["skewness"]
    logger.info("  Skewness raw=%.3f  log1p=%.3f", raw_skew, log_skew)
    results["log_skew"] = log_skew

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(target, bins=100, color=PALETTE[0], alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[0].set_yscale("log")
    axes[0].set_title(f"Raw target  (skewness = {raw_skew:.2f})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("kWh/day", fontsize=11); axes[0].set_ylabel("Count (log)", fontsize=11)
    axes[1].hist(log_target, bins=100, color=PALETTE[3], alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[1].set_title(f"log1p(kWh)  (skewness = {log_skew:.2f})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("log1p(kWh/day)", fontsize=11); axes[1].set_ylabel("Count", fontsize=11)
    fig.suptitle("Raw vs. log-transformed target distribution", fontsize=14, fontweight="bold")
    if log_skew < 0.5:
        fig.text(0.5, -0.02, "⚠  log1p skewness < 0.5 — consider log-transform for linear models",
                 ha="center", fontsize=10, color="#c62828")
    fig.tight_layout()
    _save(fig, "05_target_log_histogram.png")

    logger.info("TASK 5.1 complete.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5.2 — AffectsTimePoint and treatment/control
# ═══════════════════════════════════════════════════════════════════════════════

def task52_affectstimepoint_and_groups(df_a: pd.DataFrame, df_wx: pd.DataFrame,
                                       cfg: dict) -> dict:
    logger.info("=" * 60)
    logger.info("TASK 5.2 — AffectsTimePoint & treatment/control")
    results = {}

    # ── 5.2.1 AffectsTimePoint bar chart ─────────────────────────────────────
    atp_counts = df_a["AffectsTimePoint"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(atp_counts.index, atp_counts.values,
                  color=[PALETTE[i] for i in range(len(atp_counts))],
                  edgecolor="white", linewidth=0.5, width=0.6)
    ax.bar_label(bars, labels=[f"{v:,}\n({100*v/len(df_a):.1f}%)" for v in atp_counts.values],
                 fontsize=9, padding=4)
    ax.set_xlabel("AffectsTimePoint category", fontsize=12)
    ax.set_ylabel("Number of household-days", fontsize=12)
    ax.set_title("Distribution of AffectsTimePoint across all household-days\n"
                 "(Track A)", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    _save(fig, "05_affects_timepoint_bar.png")
    logger.info("  AffectsTimePoint: %s", atp_counts.to_dict())
    results["atp_counts"] = atp_counts.to_dict()

    # ── 5.2.2 Treatment vs. control ──────────────────────────────────────────
    group_hh = df_a.groupby("Group")["Household_ID"].nunique()
    group_stats = df_a.groupby("Group")[TARGET].agg(["mean", "median", "std"])
    logger.info("  Group household counts: %s", group_hh.to_dict())
    logger.info("  Group target stats:\n%s", group_stats.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars = axes[0].bar(group_hh.index, group_hh.values,
                       color=[PALETTE[0], PALETTE[1]], edgecolor="white", width=0.5)
    axes[0].bar_label(bars, labels=[f"{v:,}" for v in group_hh.values], fontsize=11, padding=4)
    axes[0].set_title("Unique households per group", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Group", fontsize=11); axes[0].set_ylabel("Households", fontsize=11)

    groups = sorted(df_wx["Group"].dropna().unique())
    data_by_group = [df_wx[df_wx["Group"] == g][TARGET].values for g in groups]
    bp = axes[1].boxplot(data_by_group, patch_artist=True, showfliers=False,
                         medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], [PALETTE[0], PALETTE[1]]):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    axes[1].set_xticklabels(groups, fontsize=11)
    axes[1].set_title("Consumption distribution by group\n(weather-valid rows)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Group", fontsize=11); axes[1].set_ylabel("kWh/day", fontsize=11)
    for i, g in enumerate(groups, start=1):
        m = float(df_wx[df_wx["Group"] == g][TARGET].mean())
        axes[1].text(i, m + 1, f"μ={m:.1f}", ha="center", fontsize=9, color="#333")

    fig.suptitle("Treatment vs. control group breakdown", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_group_breakdown.png")
    results["group_stats"] = group_stats.to_dict()

    # ── 5.2.3 Temporal coverage timeline ─────────────────────────────────────
    train_end = cfg["splits"]["train_end"]
    val_end   = cfg["splits"]["val_end"]
    tz        = df_a["Date"].dt.tz

    df_a["_year_month"] = df_a["Date"].dt.to_period("M")
    monthly = df_a.groupby("_year_month").size().reset_index(name="rows")
    monthly["date"] = monthly["_year_month"].dt.to_timestamp()
    t_end = pd.Timestamp(train_end, tz=tz)
    v_end = pd.Timestamp(val_end,   tz=tz)

    def split_color(dt):
        dt_tz = pd.Timestamp(dt).tz_localize(tz) if dt.tzinfo is None else dt
        if dt_tz <= t_end: return SPLIT_COLORS["train"]
        if dt_tz <= v_end: return SPLIT_COLORS["val"]
        return SPLIT_COLORS["test"]

    fig, ax = plt.subplots(figsize=(14, 5))
    bar_colors = [split_color(d) for d in monthly["date"]]
    ax.bar(range(len(monthly)), monthly["rows"], color=bar_colors, edgecolor="none", width=0.9)
    ax.set_xticks(range(0, len(monthly), 6))
    ax.set_xticklabels([monthly["date"].iloc[i].strftime("%Y-%m")
                        for i in range(0, len(monthly), 6)], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Household-days per month", fontsize=12)
    ax.set_title("Data coverage timeline with train / val / test boundaries",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=SPLIT_COLORS["train"], label=f"Train (≤ {train_end})"),
                      Patch(color=SPLIT_COLORS["val"],   label=f"Val ({train_end}–{val_end})"),
                      Patch(color=SPLIT_COLORS["test"],  label=f"Test (> {val_end})")]
    ax.legend(handles=legend_handles, fontsize=10)
    fig.tight_layout()
    _save(fig, "05_temporal_coverage.png")
    df_a.drop(columns=["_year_month"], inplace=True)

    logger.info("TASK 5.2 complete.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5.3 — Univariate feature distributions
# ═══════════════════════════════════════════════════════════════════════════════

def task53_univariate_distributions(df_a: pd.DataFrame, df_wx: pd.DataFrame, cfg: dict):
    logger.info("=" * 60)
    logger.info("TASK 5.3 — Univariate feature distributions")

    # ── 5.3.1 Weather features (point-in-time) ───────────────────────────────
    wx_cols = ["Temperature_avg_daily", "Temperature_max_daily", "Temperature_min_daily",
               "temp_range_daily", "HDD_SIA_daily", "HDD_US_daily",
               "CDD_US_daily", "Humidity_avg_daily"]
    wx_labels = ["Avg Temp (°C)", "Max Temp (°C)", "Min Temp (°C)",
                 "Temp Range (°C)", "HDD SIA", "HDD US",
                 "CDD US", "Avg Humidity (%)"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col, lbl in zip(axes.flat, wx_cols, wx_labels):
        data = df_wx[col].dropna()
        ax.hist(data, bins=60, color=PALETTE[4], alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(lbl, fontsize=11, fontweight="bold")
        ax.set_xlabel(lbl, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)
        _annotate_n(ax, len(data), fontsize=8)
        mn, mx = data.mean(), data.std()
        ax.axvline(mn, color="#e53935", lw=1.5, ls="--")
        ax.text(0.02, 0.95, f"μ={mn:.1f}\nσ={mx:.1f}", transform=ax.transAxes,
                fontsize=7.5, va="top", color="#333")
        null_pct = 100 * df_a[col].isna().sum() / len(df_a)
        if null_pct > 0:
            ax.text(0.98, 0.04, f"null: {null_pct:.1f}%", transform=ax.transAxes,
                    fontsize=7.5, ha="right", color="#e53935")
        logger.info("  %-35s  mean=%.2f  null_pct=%.2f%%", col, mn, null_pct)
    fig.suptitle("Weather feature distributions (weather-valid rows)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_weather_feature_distributions.png")

    # ── 5.3.2 Rolling/lag weather features ───────────────────────────────────
    roll_cols  = ["temp_avg_lag_1d", "temp_avg_rolling_3d", "temp_avg_rolling_7d", "HDD_SIA_rolling_7d"]
    roll_lbls  = ["Temp avg lag 1d (°C)", "Temp avg rolling 3d (°C)",
                  "Temp avg rolling 7d (°C)", "HDD SIA rolling 7d"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, col, lbl in zip(axes, roll_cols, roll_lbls):
        data = df_a[col].dropna()
        ax.hist(data, bins=60, color=PALETTE[5], alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.set_xlabel(lbl, fontsize=9); ax.set_ylabel("Count", fontsize=9)
        null_cnt = df_a[col].isna().sum()
        ax.text(0.98, 0.95, f"null: {null_cnt:,}\n({100*null_cnt/len(df_a):.1f}%)",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="top", color="#e53935")
        logger.info("  %-30s  null=%s (%.2f%%)", col, f"{null_cnt:,}", 100*null_cnt/len(df_a))
    fig.suptitle("Rolling and lag weather features", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_rolling_weather_distributions.png")

    # ── 5.3.3 Temporal features ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    # day_of_week
    ax = axes[0, 0]
    dow = df_a["day_of_week"].value_counts().sort_index()
    ax.bar(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], dow.values,
           color=PALETTE[:7], edgecolor="white")
    ax.set_title("Day of week", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=9)
    # month
    ax = axes[0, 1]
    mc = df_a["month"].value_counts().sort_index()
    ax.bar(range(1, 13), mc.values, color=[("#EF5350" if m in {10,11,12,1,2,3,4} else "#42A5F5")
                                            for m in range(1,13)], edgecolor="white")
    ax.set_xticks(range(1,13)); ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.set_title("Month", fontsize=11, fontweight="bold"); ax.set_ylabel("Count", fontsize=9)
    # is_weekend
    ax = axes[0, 2]
    wk = df_a["is_weekend"].value_counts().sort_index()
    ax.bar(["Weekday (0)", "Weekend (1)"], wk.values, color=[PALETTE[0], PALETTE[1]], edgecolor="white", width=0.5)
    ax.bar_label(ax.containers[0], labels=[f"{v:,}" for v in wk.values], fontsize=9, padding=3)
    ax.set_title("Weekday vs. weekend", fontsize=11, fontweight="bold"); ax.set_ylabel("Count", fontsize=9)
    # season
    ax = axes[1, 0]
    sc = df_a["season"].value_counts()
    ax.bar(sc.index, sc.values, color=PALETTE[:4], edgecolor="white", width=0.6)
    ax.bar_label(ax.containers[0], labels=[f"{v:,}" for v in sc.values], fontsize=9, padding=3)
    ax.set_title("Season", fontsize=11, fontweight="bold"); ax.set_ylabel("Count", fontsize=9)
    # is_heating_season
    ax = axes[1, 1]
    hs = df_a["is_heating_season"].value_counts().sort_index()
    ax.bar(["Non-heating (0)", "Heating (1)"], hs.values, color=["#42A5F5", "#EF5350"], edgecolor="white", width=0.5)
    ax.bar_label(ax.containers[0], labels=[f"{v:,}" for v in hs.values], fontsize=9, padding=3)
    ax.set_title("Heating season flag", fontsize=11, fontweight="bold"); ax.set_ylabel("Count", fontsize=9)
    # day_of_year
    ax = axes[1, 2]
    ax.hist(df_a["day_of_year"], bins=73, color=PALETTE[6], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_title("Day of year", fontsize=11, fontweight="bold")
    ax.set_xlabel("Day (1–365)", fontsize=9); ax.set_ylabel("Count", fontsize=9)

    fig.suptitle("Temporal feature distributions (Track A)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_temporal_feature_distributions.png")

    # ── 5.3.4 Categorical household features ─────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    cat_info = [
        ("Survey_Building_Type",         "Building type"),
        ("Survey_HeatPump_Installation_Type", "HP installation type"),
        ("dhw_source",                   "DHW source (composite)"),
        ("heat_distribution",            "Heat distribution (composite)"),
        ("has_pv",                       "Has PV system"),
        ("has_ev",                       "Has electric vehicle"),
    ]
    for ax, (col, title) in zip(axes.flat, cat_info):
        counts = df_a[col].value_counts(dropna=False).head(8)
        idx_str = [str(x) if not pd.isna(x) else "NaN" for x in counts.index]
        bars = ax.barh(idx_str, counts.values, color=PALETTE[:len(counts)], edgecolor="white")
        ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values], fontsize=8, padding=3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Count", fontsize=9)
        ax.tick_params(labelsize=9)
    fig.suptitle("Categorical household feature distributions (Track A)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_categorical_feature_distributions.png")

    # ── 5.3.4b Household numeric features ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    area = df_a["Survey_Building_LivingArea"].dropna()
    axes[0].hist(area, bins=50, color=PALETTE[0], alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Living area (m²)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title(f"Building living area  (null: {df_a['Survey_Building_LivingArea'].isna().sum():,} rows)",
                      fontsize=12, fontweight="bold")
    axes[0].axvline(area.median(), color="#e53935", lw=2, ls="--",
                    label=f"Median = {area.median():.0f} m²")
    axes[0].legend(fontsize=10)
    _annotate_n(axes[0], len(area))

    res_cnt = df_a["Survey_Building_Residents"].value_counts().sort_index()
    axes[1].bar(res_cnt.index.astype(str), res_cnt.values, color=PALETTE[1], edgecolor="white", width=0.6)
    axes[1].bar_label(axes[1].containers[0], labels=[f"{v:,}" for v in res_cnt.values], fontsize=9, padding=3)
    axes[1].set_xlabel("Residents in household", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title(f"Number of residents  (null: {df_a['Survey_Building_Residents'].isna().sum():,} rows)",
                      fontsize=12, fontweight="bold")
    fig.suptitle("Household numeric feature distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_household_numeric_distributions.png")

    # ── 5.3.5 Reactive energy ─────────────────────────────────────────────────
    pf = df_a["power_factor_proxy"].dropna()
    null_cnt = df_a["power_factor_proxy"].isna().sum()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(pf, bins=80, color=PALETTE[7], alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Power factor proxy", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title(f"power_factor_proxy distribution\n(non-null rows: {len(pf):,})", fontsize=12, fontweight="bold")
    axes[1].bar(["Has reactive\nenergy", "No reactive\nenergy"],
                [len(pf), null_cnt], color=[PALETTE[0], PALETTE[3]], edgecolor="white", width=0.5)
    axes[1].bar_label(axes[1].containers[0],
                      labels=[f"{len(pf):,}\n({100*len(pf)/len(df_a):.1f}%)",
                               f"{null_cnt:,}\n({100*null_cnt/len(df_a):.1f}%)"],
                      fontsize=9, padding=4)
    axes[1].set_title("Reactive energy availability", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Rows", fontsize=11)
    fig.suptitle("Reactive energy feature (power_factor_proxy)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_reactive_energy_distribution.png")
    logger.info("  power_factor_proxy: %s non-null (%.1f%%),  null: %s",
                f"{len(pf):,}", 100*len(pf)/len(df_a), f"{null_cnt:,}")

    logger.info("TASK 5.3 complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5.4 — Bivariate: feature vs. target
# ═══════════════════════════════════════════════════════════════════════════════

def task54_bivariate_feature_target(df_a: pd.DataFrame, df_wx: pd.DataFrame, cfg: dict) -> dict:
    logger.info("=" * 60)
    logger.info("TASK 5.4 — Bivariate feature vs. target")
    results = {}
    seed = cfg.get("modeling", {}).get("random_seed", 42)
    n_samp = cfg.get("eda", {}).get("bivariate_sample_size", 20_000)

    # ── 5.4.1 Numeric features vs. target scatter ─────────────────────────────
    scatter_cols = [
        ("Temperature_avg_daily",   "Daily avg temperature (°C)", df_wx),
        ("HDD_SIA_daily",           "HDD SIA",                     df_wx),
        ("temp_avg_rolling_7d",     "Temp rolling 7d avg (°C)",   df_wx),
        ("Survey_Building_LivingArea", "Living area (m²)",         df_a),
        ("Survey_Building_Residents",  "Residents",                df_a),
        ("humidity_x_temp",         "Humidity × Temp",             df_wx),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (col, lbl, src) in zip(axes.flat, scatter_cols):
        data = src[[col, TARGET]].dropna()
        samp = data.sample(min(n_samp, len(data)), random_state=seed)
        ax.scatter(samp[col], samp[TARGET], s=5, alpha=0.25, color=PALETTE[0], rasterized=True)
        r, pval = sp_stats.pearsonr(data[col], data[TARGET])
        ax.set_xlabel(lbl, fontsize=10)
        ax.set_ylabel("kWh/day", fontsize=10)
        ax.set_title(col.replace("_", " "), fontsize=11, fontweight="bold")
        ax.text(0.97, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        _annotate_n(ax, len(data), loc="lower right", fontsize=8)
        logger.info("  %-35s  Pearson r = %.4f  (n=%s)", col, r, f"{len(data):,}")
        results[f"pearson_r_{col}"] = r
    fig.suptitle("Numeric features vs. daily consumption (scatter + Pearson r)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_bivariate_numeric_vs_target.png")

    # ── 5.4.2 Categorical features vs. target boxplots ────────────────────────
    cat_comparisons = [
        ("Survey_Building_Type",     "Building type"),
        ("Survey_HeatPump_Installation_Type", "HP installation type"),
        ("dhw_source",               "DHW source"),
        ("heat_distribution",        "Heat distribution"),
        ("has_pv",                   "Has PV (0/1)"),
        ("has_ev",                   "Has EV (0/1)"),
        ("season",                   "Season"),
        ("is_heating_season",        "Heating season (0/1)"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    anova_results = {}
    for ax, (col, title) in zip(axes.flat, cat_comparisons):
        cats = df_wx[col].dropna().unique()
        groups = [df_wx[df_wx[col] == c][TARGET].values for c in cats]
        labels = [str(c) for c in cats]
        bp = ax.boxplot(groups, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=2))
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("kWh/day", fontsize=9)
        try:
            fstat, pval = sp_stats.f_oneway(*[g for g in groups if len(g) > 1])
            sig = "✓" if pval < 0.05 else "—"
            ax.text(0.02, 0.97, f"ANOVA F={fstat:.1f}  p={pval:.3g}  {sig}",
                    transform=ax.transAxes, fontsize=7.5, va="top",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
            anova_results[col] = {"F": float(fstat), "p": float(pval)}
            logger.info("  ANOVA %-30s  F=%.2f  p=%.4g", col, fstat, pval)
        except Exception:
            pass
    fig.suptitle("Categorical features vs. daily consumption (boxplots + ANOVA)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_bivariate_categorical_vs_target.png")
    results["anova"] = anova_results

    # ── 5.4.3 Living area bucket vs. target ───────────────────────────────────
    buckets  = ["<100", "100-150", "150-200", "200-300", ">300"]
    existing = [b for b in buckets if b in df_wx["living_area_bucket"].values]
    groups   = [df_wx[df_wx["living_area_bucket"] == b][TARGET].values for b in existing]
    ns       = [len(g) for g in groups]
    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(groups, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticklabels([f"{b}\n(n={n:,})" for b, n in zip(existing, ns)], fontsize=10)
    ax.set_xlabel("Living area bucket (m²)", fontsize=12)
    ax.set_ylabel("Daily consumption (kWh)", fontsize=12)
    ax.set_title("Daily consumption by living area bucket\n(weather-valid rows)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_bivariate_area_bucket_vs_target.png")

    logger.info("TASK 5.4 complete.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5.5 — Correlation analysis and VIF
# ═══════════════════════════════════════════════════════════════════════════════

def task55_correlation_and_vif(df_wx: pd.DataFrame, cfg: dict) -> dict:
    logger.info("=" * 60)
    logger.info("TASK 5.5 — Correlation analysis & VIF")
    results = {}

    numeric_features = [
        "Temperature_avg_daily", "Temperature_max_daily", "Temperature_min_daily",
        "temp_range_daily", "HDD_SIA_daily", "HDD_US_daily", "CDD_US_daily",
        "Humidity_avg_daily", "Precipitation_total_daily",
        "temp_avg_lag_1d", "temp_avg_rolling_3d", "temp_avg_rolling_7d", "HDD_SIA_rolling_7d",
        "humidity_x_temp",
        "day_of_week", "month", "is_weekend", "day_of_year", "is_heating_season",
        "Survey_Building_LivingArea", "Survey_Building_Residents",
        "building_type_house", "building_type_apartment",
        "hp_type_air_source", "hp_type_ground_source",
        "dhw_hp", "dhw_ewh", "dhw_solar", "dhw_combined",
        "heat_dist_floor", "heat_dist_radiator",
        "has_pv", "has_ev", "has_dryer", "has_freezer",
    ]
    # Keep only columns that exist
    numeric_features = [c for c in numeric_features if c in df_wx.columns]
    all_cols = numeric_features + [TARGET]
    corr_df = df_wx[all_cols].dropna()
    corr_matrix = corr_df.corr(method="pearson")

    # Sort by |r| with target
    target_corrs = corr_matrix[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    top10 = target_corrs.head(10)
    logger.info("  Top 10 features by |Pearson r| with target:")
    for feat, r in top10.items():
        raw_r = corr_matrix[TARGET][feat]
        logger.info("    %-35s  r = %+.4f", feat, raw_r)
    results["top_corr"] = {k: float(corr_matrix[TARGET][k]) for k in top10.index}

    # ── Correlation heatmap ───────────────────────────────────────────────────
    # Order: target first, then sorted by |r| with target
    feat_order = [TARGET] + target_corrs.index.tolist()
    cm = corr_matrix.loc[feat_order, feat_order]
    fig, ax = plt.subplots(figsize=(18, 15))
    mask = np.zeros_like(cm, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(cm, ax=ax, mask=mask, annot=True, fmt=".2f", annot_kws={"size": 7},
                cmap="RdBu_r", vmin=-1, vmax=1, linewidths=0.3,
                cbar_kws={"shrink": 0.6, "label": "Pearson r"})
    ax.set_title("Feature correlation matrix (sorted by |r| with target)\n"
                 f"Lower triangle — weather-valid rows  (n={len(corr_df):,})",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    fig.tight_layout()
    _save(fig, "05_correlation_heatmap.png")

    # ── Multicollinearity flags ───────────────────────────────────────────────
    high_thresh = cfg.get("eda", {}).get("correlation_high_threshold", 0.85)
    multi_pairs = []
    cols_only = [c for c in feat_order if c != TARGET]
    for i, a in enumerate(cols_only):
        for b in cols_only[i+1:]:
            r = abs(corr_matrix.loc[a, b])
            if r > high_thresh:
                multi_pairs.append((a, b, float(corr_matrix.loc[a, b])))
                logger.warning("  Multicollinear: %-30s ↔ %-30s  r=%.3f", a, b, r)
    results["multicollinear_pairs"] = multi_pairs

    # ── VIF ───────────────────────────────────────────────────────────────────
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_cols = [c for c in numeric_features
                    if corr_df[c].std() > 0 and corr_df[c].notna().all()]
        # Use a random sample to speed up VIF computation
        vif_sample = corr_df[vif_cols].sample(min(20_000, len(corr_df)), random_state=42)
        X = vif_sample.values
        vif_vals = []
        for i in range(X.shape[1]):
            try:
                v = variance_inflation_factor(X, i)
                vif_vals.append((vif_cols[i], float(v)))
            except Exception:
                vif_vals.append((vif_cols[i], np.nan))
        vif_df = pd.DataFrame(vif_vals, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)
        vif_path = TABLES_DIR / "phase5_vif_table.txt"
        vif_df.to_string(buf=open(vif_path, "w"), index=False)
        logger.info("  VIF table saved → %s", vif_path)
        high_vif = vif_df[vif_df["VIF"] > cfg.get("eda", {}).get("vif_high_threshold", 10.0)]
        for _, row in high_vif.iterrows():
            logger.warning("  High VIF: %-35s  VIF=%.1f", row["feature"], row["VIF"])
        results["high_vif"] = high_vif[["feature", "VIF"]].to_dict("records")
    except ImportError:
        logger.warning("  statsmodels not available — VIF skipped")
        vif_df = pd.DataFrame()
        results["high_vif"] = []

    logger.info("TASK 5.5 complete.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5.6 — Temporal patterns
# ═══════════════════════════════════════════════════════════════════════════════

def task56_temporal_patterns(df_a: pd.DataFrame, df_wx: pd.DataFrame, cfg: dict) -> dict:
    logger.info("=" * 60)
    logger.info("TASK 5.6 — Temporal patterns")
    results = {}

    # ── 5.6.1 Weekly periodicity ──────────────────────────────────────────────
    dow_stats = df_wx.groupby("day_of_week")[TARGET].agg(["mean", "std"])
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(day_names, dow_stats["mean"], yerr=dow_stats["std"],
           color=PALETTE[:7], edgecolor="white", capsize=4, error_kw={"lw": 1.5})
    ax.set_xlabel("Day of week", fontsize=12)
    ax.set_ylabel("Mean daily consumption (kWh) ± 1 std", fontsize=12)
    ax.set_title("Weekly consumption pattern\n(weather-valid rows)", fontsize=13, fontweight="bold")
    try:
        groups_by_dow = [df_wx[df_wx["day_of_week"] == d][TARGET].values for d in range(7)]
        fstat, pval = sp_stats.f_oneway(*groups_by_dow)
        ax.text(0.02, 0.97, f"ANOVA  F={fstat:.2f}  p={pval:.3g}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        results["dow_anova"] = {"F": float(fstat), "p": float(pval)}
        logger.info("  Day-of-week ANOVA: F=%.2f, p=%.4g", fstat, pval)
    except Exception:
        pass
    fig.tight_layout()
    _save(fig, "05_temporal_day_of_week.png")

    # ── 5.6.2 Year-over-year seasonal overlay ─────────────────────────────────
    df_a["_year"]  = df_a["Date"].dt.year
    df_a["_calday"] = df_a["Date"].dt.dayofyear
    yearly = df_a.groupby(["_year", "_calday"])[TARGET].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    year_colors = plt.cm.tab10(np.linspace(0, 0.7, len(df_a["_year"].unique())))
    for color, (yr, grp) in zip(year_colors, yearly.groupby("_year")):
        smoothed = grp.set_index("_calday")[TARGET].rolling(7, center=True, min_periods=1).mean()
        ax.plot(smoothed.index, smoothed.values, lw=1.8, alpha=0.85, color=color, label=str(yr))
    ax.axvspan(1, 90,  alpha=0.06, color="#EF5350", label="Heating season approx.")
    ax.axvspan(274, 366, alpha=0.06, color="#EF5350")
    ax.set_xlabel("Day of year", fontsize=12)
    ax.set_ylabel("Mean daily consumption (kWh, 7d smooth)", fontsize=12)
    ax.set_title("Year-over-year seasonal consumption overlay\n(7-day rolling mean per calendar day)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, title="Year", title_fontsize=10)
    fig.tight_layout()
    _save(fig, "05_temporal_year_overlay.png")
    df_a.drop(columns=["_year", "_calday"], inplace=True)

    # ── 5.6.3 Household coverage heatmap (histogram version) ─────────────────
    df_a["_year_month"] = df_a["Date"].dt.to_period("M")
    hh_monthly = df_a.groupby(["Household_ID", "_year_month"]).size().reset_index(name="days")
    # Compute fraction of months with ≥ 20 days of coverage per household
    total_months = hh_monthly["_year_month"].nunique()
    hh_coverage = hh_monthly.groupby("Household_ID").size() / total_months

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(hh_coverage, bins=40, color=PALETTE[2], alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Fraction of months with data", fontsize=11)
    axes[0].set_ylabel("Number of households", fontsize=11)
    axes[0].set_title("Per-household data coverage (fraction of months)", fontsize=12, fontweight="bold")
    axes[0].axvline(hh_coverage.mean(), color="#e53935", lw=2, ls="--",
                    label=f"Mean = {hh_coverage.mean():.2f}")
    axes[0].legend(fontsize=9)
    _annotate_n(axes[0], len(hh_coverage))

    days_per_month = hh_monthly["days"]
    axes[1].hist(days_per_month, bins=35, color=PALETTE[3], alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel("Days present per household-month", fontsize=11)
    axes[1].set_ylabel("Count of household-months", fontsize=11)
    axes[1].set_title("Days available per household-month", fontsize=12, fontweight="bold")
    axes[1].axvline(28, color="#e53935", lw=2, ls="--", label="28-day reference")
    axes[1].legend(fontsize=9)

    fig.suptitle("Household data coverage", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_household_coverage_heatmap.png")
    df_a.drop(columns=["_year_month"], inplace=True)

    # ── 5.6.4 Pre/post split data volume ─────────────────────────────────────
    train_end = cfg["splits"]["train_end"]
    val_end   = cfg["splits"]["val_end"]
    tz = df_a["Date"].dt.tz
    t_end = pd.Timestamp(train_end, tz=tz)
    v_end = pd.Timestamp(val_end,   tz=tz)

    train_df = df_a[df_a["Date"] <= t_end]
    val_df   = df_a[(df_a["Date"] > t_end) & (df_a["Date"] <= v_end)]
    test_df  = df_a[df_a["Date"] > v_end]

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        logger.info("  %-5s  rows=%s  HH=%s  mean_kWh=%.2f  dates=[%s → %s]",
                    name, f"{len(split):,}", split["Household_ID"].nunique(),
                    split[TARGET].mean(),
                    split["Date"].min().strftime("%Y-%m-%d"),
                    split["Date"].max().strftime("%Y-%m-%d"))
    results["split_means"] = {
        "train": float(train_df[TARGET].mean()),
        "val":   float(val_df[TARGET].mean()),
        "test":  float(test_df[TARGET].mean()),
    }

    logger.info("TASK 5.6 complete.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5.7 — Missing data audit
# ═══════════════════════════════════════════════════════════════════════════════

def task57_missing_data(df_a: pd.DataFrame, df_b: pd.DataFrame, cfg: dict) -> dict:
    logger.info("=" * 60)
    logger.info("TASK 5.7 — Missing data audit")
    results = {}

    def _missing_barplot(df: pd.DataFrame, title: str, fname: str):
        null_frac = (df.isna().sum() / len(df)).sort_values(ascending=False)
        null_frac = null_frac[null_frac > 0]  # only show columns with any nulls
        if null_frac.empty:
            logger.info("  No nulls found in %s", fname)
            return

        colors = ["#e53935" if v > 0.30 else ("#FF9800" if v > 0.05 else "#4CAF50")
                  for v in null_frac.values]
        fig_h = max(5, min(30, len(null_frac) * 0.32))
        fig, ax = plt.subplots(figsize=(12, fig_h))
        bars = ax.barh(null_frac.index[::-1], null_frac.values[::-1] * 100,
                       color=colors[::-1], edgecolor="white", height=0.7)
        ax.set_xlabel("Missing (%)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axvline(5,  color="#FF9800", lw=1.5, ls="--", alpha=0.7, label="5%  threshold")
        ax.axvline(30, color="#e53935", lw=1.5, ls="--", alpha=0.7, label="30% threshold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 105)
        for bar, val in zip(bars, null_frac.values[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        _save(fig, fname)
        # Log high-null warnings
        for col, v in null_frac.items():
            if v > 0.60:
                logger.warning("  [DROP] %-40s  %.1f%% null — too sparse", col, v * 100)
            elif v > 0.30:
                logger.warning("  [WARN] %-40s  %.1f%% null — unexpected", col, v * 100)
        return null_frac

    nf_a = _missing_barplot(df_a, "Missing data by feature — Track A", "05_missing_data_track_a.png")
    nf_b = _missing_barplot(df_b, "Missing data by feature — Track B", "05_missing_data_track_b.png")

    # ── 5.7.3 Weather-null vs. consumption pattern ────────────────────────────
    null_wx  = df_a[df_a["Temperature_avg_daily"].isna()][TARGET]
    notnull_wx = df_a[df_a["Temperature_avg_daily"].notna()][TARGET]
    m_null, m_notnull = null_wx.mean(), notnull_wx.mean()
    logger.info("  Mean kWh (weather null):    %.2f kWh", m_null)
    logger.info("  Mean kWh (weather present): %.2f kWh", m_notnull)
    results["weather_null_mean"] = float(m_null)
    results["weather_present_mean"] = float(m_notnull)

    logger.info("TASK 5.7 complete.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5.8 — Subgroup comparisons
# ═══════════════════════════════════════════════════════════════════════════════

def task58_subgroup_comparisons(df_a: pd.DataFrame, df_wx: pd.DataFrame, cfg: dict) -> dict:
    logger.info("=" * 60)
    logger.info("TASK 5.8 — Subgroup comparisons")
    results = {}

    def _two_group_box(col, val0, val1, lbl0, lbl1, title, fname, src=None):
        src = src if src is not None else df_wx
        g0 = src[src[col] == val0][TARGET].dropna().values
        g1 = src[src[col] == val1][TARGET].dropna().values
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        bp = axes[0].boxplot([g0, g1], patch_artist=True, showfliers=False,
                             medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor(PALETTE[0]); bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_facecolor(PALETTE[1]); bp["boxes"][1].set_alpha(0.75)
        axes[0].set_xticklabels([f"{lbl0}\n(n={len(g0):,})", f"{lbl1}\n(n={len(g1):,})"], fontsize=10)
        axes[0].set_ylabel("kWh/day", fontsize=11)
        axes[0].set_title("Consumption distribution", fontsize=12, fontweight="bold")
        try:
            stat, pval = sp_stats.mannwhitneyu(g0, g1, alternative="two-sided")
            axes[0].text(0.02, 0.97, f"MWU p={pval:.3g}", transform=axes[0].transAxes,
                         fontsize=9, va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
            logger.info("  %-50s  MWU p=%.4g  μ₀=%.2f  μ₁=%.2f", title, pval, g0.mean(), g1.mean())
            results[f"mwu_{col}"] = {"p": float(pval), "mean0": float(g0.mean()), "mean1": float(g1.mean())}
        except Exception:
            pass
        axes[1].bar([lbl0, lbl1], [g0.mean(), g1.mean()],
                    color=[PALETTE[0], PALETTE[1]], edgecolor="white", width=0.5)
        axes[1].bar_label(axes[1].containers[0],
                          labels=[f"{g0.mean():.1f} kWh", f"{g1.mean():.1f} kWh"],
                          fontsize=10, padding=4)
        axes[1].set_ylabel("Mean kWh/day", fontsize=11)
        axes[1].set_title("Mean consumption comparison", fontsize=12, fontweight="bold")
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        _save(fig, fname)

    # ── 5.8.1 HP type ─────────────────────────────────────────────────────────
    hp_cats = ["air-source", "ground-source", "unknown"]
    groups  = [df_wx[df_wx["Survey_HeatPump_Installation_Type"] == c][TARGET].dropna().values
               for c in hp_cats]
    ns      = [len(g) for g in groups]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(groups, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticklabels([f"{c}\n(n={n:,})" for c, n in zip(hp_cats, ns)], fontsize=10)
    ax.set_ylabel("kWh/day", fontsize=11)
    ax.set_title("Daily consumption by heat pump type\n(weather-valid rows)", fontsize=13, fontweight="bold")
    try:
        fstat, pval = sp_stats.kruskal(*[g for g in groups if len(g) > 1])
        ax.text(0.02, 0.97, f"Kruskal-Wallis  H={fstat:.2f}  p={pval:.3g}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        logger.info("  HP type Kruskal-Wallis: H=%.2f p=%.4g", fstat, pval)
    except Exception:
        pass
    fig.tight_layout()
    _save(fig, "05_subgroup_hp_type.png")

    # ── 5.8.2 PV system ──────────────────────────────────────────────────────
    _two_group_box("has_pv", 0, 1, "No PV", "Has PV",
                   "Daily consumption: PV vs. no PV system", "05_subgroup_pv.png")

    # ── 5.8.3 Building type ───────────────────────────────────────────────────
    bt_cats = [c for c in ["house", "apartment"] if c in df_wx["Survey_Building_Type"].values]
    bt_groups = [df_wx[df_wx["Survey_Building_Type"] == c][TARGET].dropna().values for c in bt_cats]
    bt_null   = df_wx[df_wx["Survey_Building_Type"].isna()][TARGET].dropna().values
    all_groups = bt_groups + [bt_null]
    all_labels = bt_cats + ["NaN"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(all_groups, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticklabels([f"{l}\n(n={len(g):,})" for l, g in zip(all_labels, all_groups)], fontsize=10)
    ax.set_ylabel("kWh/day", fontsize=11)
    ax.set_title("Daily consumption by building type\n(weather-valid rows)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_subgroup_building_type.png")

    # ── 5.8.4 Below threshold households ─────────────────────────────────────
    g0 = df_a[df_a["below_min_days_threshold"] == 0][TARGET].dropna().values
    g1 = df_a[df_a["below_min_days_threshold"] == 1][TARGET].dropna().values
    try:
        from scipy.stats import ks_2samp
        ks_stat, ks_p = ks_2samp(g0, g1)
        logger.info("  KS test (below_threshold vs not): stat=%.4f  p=%.4g", ks_stat, ks_p)
        results["ks_below_threshold"] = {"stat": float(ks_stat), "p": float(ks_p)}
    except Exception:
        ks_p = np.nan
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot([g0, g1], patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=2))
    bp["boxes"][0].set_facecolor(PALETTE[0]); bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(PALETTE[3]); bp["boxes"][1].set_alpha(0.75)
    ax.set_xticklabels([f"≥180 days (n={len(g0):,})", f"<180 days (n={len(g1):,})"], fontsize=10)
    ax.set_ylabel("kWh/day", fontsize=11)
    ax.set_title("Consumption for households above/below 180-day threshold",
                 fontsize=13, fontweight="bold")
    if not np.isnan(ks_p):
        ax.text(0.02, 0.97, f"KS test p={ks_p:.3g}", transform=ax.transAxes,
                fontsize=9, va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.tight_layout()
    _save(fig, "05_subgroup_min_days_threshold.png")

    # ── 5.8.5 Treatment vs. control pre-visit ────────────────────────────────
    treat_before = df_a[(df_a["Group"] == "treatment") &
                        (df_a["AffectsTimePoint"] == "before visit")][TARGET].dropna()
    ctrl   = df_a[df_a["Group"] == "control"][TARGET].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot([ctrl.values, treat_before.values], patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=2))
    bp["boxes"][0].set_facecolor(PALETTE[0]); bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(PALETTE[1]); bp["boxes"][1].set_alpha(0.75)
    ax.set_xticklabels([f"Control\n(n={len(ctrl):,})",
                        f"Treatment\n(before visit)\n(n={len(treat_before):,})"], fontsize=10)
    ax.set_ylabel("kWh/day", fontsize=11)
    ax.set_title("Control group vs. treatment group pre-visit consumption",
                 fontsize=13, fontweight="bold")
    try:
        stat, pval = sp_stats.mannwhitneyu(ctrl.values, treat_before.values, alternative="two-sided")
        ax.text(0.02, 0.97, f"MWU p={pval:.3g}  |  μ ctrl={ctrl.mean():.2f}  μ treat={treat_before.mean():.2f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        logger.info("  Treatment pre-visit vs. control: MWU p=%.4g  ctrl_mean=%.2f  treat_mean=%.2f",
                    pval, ctrl.mean(), treat_before.mean())
    except Exception:
        pass
    fig.tight_layout()
    _save(fig, "05_subgroup_treatment_vs_control.png")

    # ── 5.8.6 EV ownership ────────────────────────────────────────────────────
    _two_group_box("has_ev", 0, 1, "No EV", "Has EV",
                   "Daily consumption: EV ownership effect\n"
                   "(note: kWh_received_Total includes EV charging)", "05_subgroup_ev.png")

    logger.info("TASK 5.8 complete.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5.9 — Track B protocol-enriched EDA
# ═══════════════════════════════════════════════════════════════════════════════

def task59_protocol_eda(df_b: pd.DataFrame, cfg: dict) -> dict:
    logger.info("=" * 60)
    logger.info("TASK 5.9 — Track B protocol-enriched EDA (%s rows, %s HH)",
                f"{len(df_b):,}", df_b["Household_ID"].nunique())
    results = {}
    seed = cfg.get("modeling", {}).get("random_seed", 42)
    n_samp = cfg.get("eda", {}).get("protocol_scatter_sample_size", 10_000)

    # ── 5.9.1 Building age distribution ──────────────────────────────────────
    ba = df_b["building_age"].dropna()
    bucket_order = ["pre-1970", "1970-1990", "1990-2010", "post-2010"]
    bucket_counts = df_b["building_age_bucket"].value_counts()
    bucket_counts = bucket_counts.reindex([b for b in bucket_order if b in bucket_counts.index])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(ba, bins=40, color=PALETTE[0], alpha=0.85, edgecolor="white", linewidth=0.3)
    for cut in [0, 20, 40, 54]:
        axes[0].axvline(cut, color="#e53935", lw=1.2, ls="--", alpha=0.6)
    axes[0].set_xlabel("Building age (years)", fontsize=11)
    axes[0].set_ylabel("Count (household-days)", fontsize=11)
    axes[0].set_title(f"Building age distribution\n(non-null rows: {len(ba):,})", fontsize=12, fontweight="bold")

    bars = axes[1].bar(bucket_counts.index, bucket_counts.values, color=PALETTE[:4], edgecolor="white", width=0.6)
    axes[1].bar_label(bars, labels=[f"{v:,}" for v in bucket_counts.values], fontsize=9, padding=3)
    axes[1].set_xlabel("Construction year bucket", fontsize=11)
    axes[1].set_ylabel("Count (household-days)", fontsize=11)
    axes[1].set_title("Building age bucket distribution", fontsize=12, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Building age — Track B protocol households", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_protocol_building_age.png")
    logger.info("  Building age per bucket: %s", bucket_counts.to_dict())

    # ── 5.9.2 Building age vs. target ─────────────────────────────────────────
    ba_data = df_b[["building_age", TARGET]].dropna()
    r_ba, _ = sp_stats.pearsonr(ba_data["building_age"], ba_data[TARGET])
    samp = ba_data.sample(min(n_samp, len(ba_data)), random_state=seed)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(samp["building_age"], samp[TARGET], s=8, alpha=0.3, color=PALETTE[0], rasterized=True)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
        frac = cfg.get("eda", {}).get("lowess_frac", 0.10)
        sm = sm_lowess(ba_data[TARGET].values, ba_data["building_age"].values, frac=frac, it=1, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], color="#e53935", lw=2.5, label="Lowess")
        ax.legend(fontsize=10)
    except Exception:
        pass
    ax.set_xlabel("Building age (years)", fontsize=12)
    ax.set_ylabel("Daily consumption (kWh)", fontsize=12)
    ax.set_title(f"Building age vs. consumption  (r = {r_ba:.3f})\n"
                 f"Track B, sample n={len(samp):,}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_protocol_building_age_vs_target.png")
    logger.info("  Building age Pearson r = %.4f", r_ba)
    results["building_age_r"] = float(r_ba)

    # ── 5.9.3 HP age distribution ─────────────────────────────────────────────
    hp_age = df_b["hp_age"].dropna()
    null_hp_age = df_b["hp_age"].isna().sum()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(hp_age, bins=40, color=PALETTE[1], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("HP age (years)", fontsize=12)
    ax.set_ylabel("Count (household-days)", fontsize=12)
    ax.set_title(f"Heat pump age distribution\n"
                 f"non-null rows: {len(hp_age):,}  |  null: {null_hp_age:,} ({100*null_hp_age/len(df_b):.1f}%)",
                 fontsize=13, fontweight="bold")
    ax.axvline(hp_age.median(), color="#e53935", lw=2, ls="--",
               label=f"Median = {hp_age.median():.0f} yrs")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "05_protocol_hp_age.png")
    logger.info("  HP age: median=%.1f yrs  null=%s (%.1f%%)",
                hp_age.median(), f"{null_hp_age:,}", 100*null_hp_age/len(df_b))

    # ── 5.9.4 Heating curve gradient analysis ─────────────────────────────────
    hc = df_b["heating_curve_gradient_full"].dropna()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].hist(hc, bins=50, color=PALETTE[2], alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[0].axvline(hc.mean(), color="#e53935", lw=2, ls="--",
                    label=f"Mean = {hc.mean():.3f}")
    axes[0].axvline(hc.median(), color="#6a1b9a", lw=2, ls="-.",
                    label=f"Median = {hc.median():.3f}")
    axes[0].set_xlabel("Heating curve gradient (full range)", fontsize=10)
    axes[0].set_ylabel("Count (household-days)", fontsize=10)
    axes[0].set_title("Heating curve gradient distribution\n(full range, °C/°C)", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)

    hct_groups = [
        df_b[df_b["heating_curve_too_high"] == False][TARGET].dropna().values,
        df_b[df_b["heating_curve_too_high"] == True][TARGET].dropna().values,
    ]
    bp = axes[1].boxplot(hct_groups, patch_artist=True, showfliers=False,
                         medianprops=dict(color="black", lw=2))
    bp["boxes"][0].set_facecolor(PALETTE[0]); bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor("#e53935");  bp["boxes"][1].set_alpha(0.75)
    axes[1].set_xticklabels([f"OK\n(n={len(hct_groups[0]):,})",
                              f"Too high\n(n={len(hct_groups[1]):,})"], fontsize=10)
    axes[1].set_ylabel("kWh/day", fontsize=10)
    axes[1].set_title("Consumption: heating curve\nflag OK vs. too high", fontsize=11, fontweight="bold")
    try:
        stat, pval = sp_stats.mannwhitneyu(hct_groups[0], hct_groups[1], alternative="two-sided")
        axes[1].text(0.02, 0.97, f"MWU p={pval:.3g}", transform=axes[1].transAxes,
                     fontsize=9, va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        logger.info("  Heating curve too_high MWU: p=%.4g  μ_ok=%.2f  μ_high=%.2f",
                    pval, hct_groups[0].mean(), hct_groups[1].mean())
        results["heating_curve_too_high_mwu"] = float(pval)
    except Exception:
        pass

    hh_hc = df_b.groupby("Household_ID").agg(
        hc_grad=("heating_curve_gradient_full", "first"),
        mean_kwh=(TARGET, "mean")).dropna()
    axes[2].scatter(hh_hc["hc_grad"], hh_hc["mean_kwh"], s=40, alpha=0.7, color=PALETTE[2], edgecolors="white", linewidths=0.5)
    if len(hh_hc) > 4:
        r_hc, _ = sp_stats.pearsonr(hh_hc["hc_grad"], hh_hc["mean_kwh"])
        axes[2].text(0.97, 0.95, f"r = {r_hc:.3f}", transform=axes[2].transAxes,
                     ha="right", va="top", fontsize=10,
                     bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        results["heating_curve_gradient_r"] = float(r_hc)
        logger.info("  Heating curve gradient vs. mean consumption: r=%.4f", r_hc)
    axes[2].set_xlabel("Heating curve gradient (full range)", fontsize=10)
    axes[2].set_ylabel("Mean kWh/day per household", fontsize=10)
    axes[2].set_title("Heating curve gradient vs.\nmean HH consumption  (1 pt per HH)", fontsize=11, fontweight="bold")

    fig.suptitle("Heating curve analysis — Track B", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_protocol_heating_curve.png")

    # ── 5.9.5 Issue flag prevalence ───────────────────────────────────────────
    issue_cols = {
        "heating_curve_too_high":    "Heating curve too high",
        "heating_limit_too_high":    "Heating limit too high",
        "night_setback_active_before": "Night setback active (before)",
        "descaling_needed":          "DHW descaling needed",
        "pipes_not_insulated":       "Pipes not insulated",
        "has_buffer_tank":           "Has buffer tank",
        "hp_internet_connection":    "HP internet connection",
    }
    # hp_correctly_planned: True = correctly planned (issue is when False)
    hh_b = df_b.groupby("Household_ID")[list(issue_cols.keys()) + ["hp_correctly_planned"]].first()
    hh_n = len(hh_b)
    prevalence = {}
    for col in issue_cols:
        if col in hh_b.columns:
            prevalence[issue_cols[col]] = float(hh_b[col].sum(skipna=True)) / hh_n
    if "hp_correctly_planned" in hh_b.columns:
        prevalence["HP NOT correctly planned"] = float((hh_b["hp_correctly_planned"] == False).sum()) / hh_n

    sorted_prev = dict(sorted(prevalence.items(), key=lambda x: x[1], reverse=True))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(list(sorted_prev.keys())[::-1],
                   [v * 100 for v in list(sorted_prev.values())[::-1]],
                   color=PALETTE[:len(sorted_prev)], edgecolor="white", height=0.65)
    ax.bar_label(bars, labels=[f"{v*100:.1f}%" for v in list(sorted_prev.values())[::-1]],
                 fontsize=9, padding=4)
    ax.set_xlabel("% of Track B households", fontsize=12)
    ax.set_title(f"HP issue flag prevalence — Track B ({hh_n} households)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 105)
    fig.tight_layout()
    _save(fig, "05_protocol_issue_flags.png")
    for label, v in sorted_prev.items():
        logger.info("  %-45s  %.1f%%", label, v * 100)
    results["issue_prevalence"] = {k: float(v) for k, v in sorted_prev.items()}

    # ── 5.9.6 Pre vs. post intervention ──────────────────────────────────────
    before = df_b[df_b["AffectsTimePoint"] == "before visit"].groupby("Household_ID")[TARGET].mean()
    after  = df_b[df_b["AffectsTimePoint"] == "after visit"].groupby("Household_ID")[TARGET].mean()
    paired = before.align(after, join="inner")
    before_p, after_p = paired[0], paired[1]
    n_paired = len(before_p)
    delta = after_p - before_p
    frac_reduced = float((delta < 0).sum()) / n_paired

    try:
        stat, pval = sp_stats.ttest_rel(before_p.values, after_p.values, alternative="greater")
    except Exception:
        stat, pval = np.nan, np.nan

    logger.info("  Pre/post: n_paired=%s  mean_before=%.2f  mean_after=%.2f  "
                "mean_delta=%.2f  frac_reduced=%.1f%%  paired_t_p=%.4g",
                n_paired, before_p.mean(), after_p.mean(), delta.mean(),
                frac_reduced * 100, pval)
    results["pre_post"] = {
        "n_paired": n_paired, "mean_before": float(before_p.mean()),
        "mean_after": float(after_p.mean()), "mean_delta": float(delta.mean()),
        "frac_reduced_pct": float(frac_reduced * 100), "paired_t_p": float(pval),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Connected scatter: before → after per HH
    axes[0].scatter(np.zeros(n_paired), before_p.values, s=20, color=PALETTE[0], alpha=0.6, zorder=3)
    axes[0].scatter(np.ones(n_paired),  after_p.values,  s=20, color=PALETTE[1], alpha=0.6, zorder=3)
    for b_val, a_val in zip(before_p.values, after_p.values):
        color = "#4CAF50" if a_val < b_val else "#e53935"
        axes[0].plot([0, 1], [b_val, a_val], lw=0.8, alpha=0.35, color=color)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Before visit", "After visit"], fontsize=11)
    axes[0].set_ylabel("Mean kWh/day per household", fontsize=11)
    axes[0].set_title(f"Per-household pre/post consumption\n"
                      f"(green = reduction, red = increase)  n={n_paired}",
                      fontsize=12, fontweight="bold")

    axes[1].hist(delta, bins=30, color=PALETTE[2], alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="black", lw=2, ls="-", alpha=0.5)
    axes[1].axvline(delta.mean(), color="#e53935", lw=2, ls="--",
                    label=f"Mean Δ = {delta.mean():.2f} kWh")
    axes[1].set_xlabel("Δ kWh/day (after − before)", fontsize=11)
    axes[1].set_ylabel("Count of households", fontsize=11)
    axes[1].set_title(f"Distribution of consumption change\n"
                      f"paired t-test p={pval:.4f}  |  "
                      f"{frac_reduced*100:.1f}% households reduced",
                      fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    fig.suptitle("Pre vs. post intervention consumption — Track B", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_protocol_pre_post_intervention.png")

    # ── 5.9.7 HP capacity per area ────────────────────────────────────────────
    cap_data = df_b[["hp_capacity_per_area", TARGET, "Household_ID"]].dropna()
    hh_cap = cap_data.groupby("Household_ID").agg(
        cap=("hp_capacity_per_area", "first"),
        mean_kwh=(TARGET, "mean")).dropna()
    r_cap, _ = sp_stats.pearsonr(hh_cap["cap"], hh_cap["mean_kwh"]) if len(hh_cap) > 4 else (np.nan, np.nan)
    p95 = hh_cap["cap"].quantile(0.95)
    n_extreme = int((hh_cap["cap"] > p95).sum())

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_cap = ["#e53935" if c > p95 else PALETTE[0] for c in hh_cap["cap"]]
    ax.scatter(hh_cap["cap"], hh_cap["mean_kwh"], s=40, c=colors_cap, alpha=0.75,
               edgecolors="white", linewidths=0.4)
    ax.axvline(p95, color="#e53935", lw=1.5, ls="--", alpha=0.7,
               label=f"p95 = {p95:.3f} kW/m²  ({n_extreme} HH above)")
    ax.set_xlabel("HP capacity per heated area (kW/m²)", fontsize=12)
    ax.set_ylabel("Mean kWh/day per household", fontsize=12)
    ax.set_title(f"HP capacity sizing vs. mean consumption  (r = {r_cap:.3f})\n"
                 f"1 point per household, n={len(hh_cap)}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "05_protocol_hp_capacity.png")
    logger.info("  HP capacity/area Pearson r = %.4f  p95 threshold = %.3f  above-p95 HH: %s",
                r_cap, p95, n_extreme)
    results["hp_capacity_r"] = float(r_cap) if not np.isnan(r_cap) else None

    logger.info("TASK 5.9 complete.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5.10 — EDA Summary Report
# ═══════════════════════════════════════════════════════════════════════════════

def task510_write_summary_report(df_a: pd.DataFrame, df_b: pd.DataFrame,
                                  df_wx: pd.DataFrame, all_results: dict, cfg: dict):
    logger.info("=" * 60)
    logger.info("TASK 5.10 — Writing EDA summary report")

    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st = all_results.get("target_stats", {})
    pp = all_results.get("pre_post", {})
    multi = all_results.get("multicollinear_pairs", [])
    high_vif = all_results.get("high_vif", [])
    anova = all_results.get("anova", {})
    split_means = all_results.get("split_means", {})
    top_corr = all_results.get("top_corr", {})

    lines = [
        "=" * 64,
        "HEAPO-Predict Phase 5 EDA Summary",
        f"Generated: {ts}",
        "=" * 64,
        "",
        "SECTION 1 — DATASET OVERVIEW",
        f"  Track A : {len(df_a):>10,} rows  |  {df_a['Household_ID'].nunique():,} HH  |  {df_a.shape[1]} cols",
        f"  Track B : {len(df_b):>10,} rows  |  {df_b['Household_ID'].nunique():,} HH  |  {df_b.shape[1]} cols",
        f"  Wx-valid: {len(df_wx):>10,} rows  ({100*len(df_wx)/len(df_a):.1f}% of Track A)",
        f"  IQR outlier rows : {int(df_a['is_iqr_outlier'].sum()):,} ({100*df_a['is_iqr_outlier'].mean():.2f}%)",
        f"  Below 180d HH    : {int(df_a['below_min_days_threshold'].sum()):,}",
        "",
        "SECTION 2 — TARGET VARIABLE (kWh_received_Total)",
        f"  mean     : {st.get('mean', float('nan')):.3f} kWh/day",
        f"  median   : {st.get('median', float('nan')):.3f} kWh/day",
        f"  std      : {st.get('std', float('nan')):.3f} kWh/day",
        f"  min/max  : {st.get('min', float('nan')):.2f} / {st.get('max', float('nan')):.2f}",
        f"  p25/p75  : {st.get('p25', float('nan')):.2f} / {st.get('p75', float('nan')):.2f}",
        f"  p99      : {st.get('p99', float('nan')):.2f}",
        f"  skewness : {st.get('skewness', float('nan')):.3f} (raw)  /  {all_results.get('log_skew', float('nan')):.3f} (log1p)",
    ]
    if all_results.get("log_skew", 999) < 0.5:
        lines.append("  [FLAG] log1p skewness < 0.5 — consider log-transform for linear models in Phase 7")

    lines += ["", "SECTION 3 — TOP CORRELATED FEATURES (|Pearson r| with target)"]
    for i, (feat, r) in enumerate(sorted(top_corr.items(), key=lambda x: abs(x[1]), reverse=True), 1):
        lines.append(f"  {i:2d}. {feat:<40s}  r = {r:+.4f}")

    lines += ["", "SECTION 4 — MULTICOLLINEARITY FLAGS (|r| > {})".format(
        cfg.get("eda", {}).get("correlation_high_threshold", 0.85))]
    if multi:
        for a, b, r in multi:
            rec = "drop one for linear models; both ok for trees"
            lines.append(f"  {a:<35s} ↔ {b:<35s}  r={r:.3f}  [{rec}]")
    else:
        lines.append("  None found above threshold")

    lines += ["", "SECTION 5 — HIGH-VIF FEATURES (VIF > {})".format(
        cfg.get("eda", {}).get("vif_high_threshold", 10.0))]
    if high_vif:
        for entry in high_vif:
            lines.append(f"  {entry['feature']:<40s}  VIF={entry['VIF']:.1f}")
    else:
        lines.append("  None found above threshold (or VIF not computed)")

    lines += ["", "SECTION 6 — MISSING DATA FLAGS"]
    null_frac_a = (df_a.isna().sum() / len(df_a)).sort_values(ascending=False)
    for col, v in null_frac_a[null_frac_a > 0.01].items():
        tag = "[DROP]" if v > 0.60 else ("[WARN]" if v > 0.30 else "[OK]  ")
        lines.append(f"  {tag} {col:<50s}  {v*100:.1f}% null")

    lines += ["", "SECTION 7 — SUBGROUP ANOVA/KW RESULTS"]
    for col, res in anova.items():
        sig = "significant" if res["p"] < 0.05 else "NOT significant"
        lines.append(f"  {col:<45s}  F={res['F']:.1f}  p={res['p']:.3g}  [{sig}]")

    lines += ["", "SECTION 8 — TREATMENT EFFECT (TRACK B PRE/POST)"]
    if pp:
        lines += [
            f"  Households with before+after data: {pp.get('n_paired', '?')}",
            f"  Mean before  : {pp.get('mean_before', float('nan')):.3f} kWh/day",
            f"  Mean after   : {pp.get('mean_after',  float('nan')):.3f} kWh/day",
            f"  Mean delta   : {pp.get('mean_delta',  float('nan')):+.3f} kWh/day",
            f"  Frac reduced : {pp.get('frac_reduced_pct', float('nan')):.1f}%",
            f"  Paired t-test p = {pp.get('paired_t_p', float('nan')):.4g}  "
            f"[{'significant' if pp.get('paired_t_p', 1) < 0.05 else 'NOT significant'}]",
        ]

    lines += ["", "SECTION 9 — PHASE 6 RECOMMENDATIONS"]
    recs = []
    if all_results.get("log_skew", 999) < 0.5:
        recs.append("[TARGET TRANSFORM]    log1p transform recommended for linear regression and ANN.")
    recs.append("[OUTLIER STRATEGY]    3,614 IQR-outlier rows (0.4%) retained. "
                "Consider winsorizing at p99 for linear models in Phase 7.")
    if multi:
        pairs_str = "; ".join(f"{a}↔{b}" for a, b, _ in multi[:3])
        recs.append(f"[MULTICOLLINEAR]      Drop one from each pair for linear models: {pairs_str}")
    recs.append("[IMPUTATION]          Weather nulls (2.9%): median impute per station. "
                "power_factor_proxy (45.3% null): impute=0 + has_reactive_energy flag. "
                "cop_rated (68%+ null in Track B): exclude from pooled Track B models.")
    recs.append("[DROP CANDIDATES]     kWh_received_HeatPump (97.4% null), "
                "Sunshine_duration_daily (structural missing at 3 stations). Already excluded from features.")
    recs.append("[SCALING NOTE]        Standardize all numeric features for LR and ANN. "
                "Tree models (RF, XGBoost) do not require scaling.")
    if split_means:
        test_m = split_means.get("test", 0); train_m = split_means.get("train", 0)
        if test_m > train_m * 1.15:
            recs.append(f"[SEASONAL SHIFT]      Test set mean={test_m:.2f} kWh vs. train={train_m:.2f} kWh "
                        f"(+{100*(test_m/train_m-1):.0f}%). Test covers heating season only — "
                        "expect higher MAPE at test time due to distributional shift.")
    if pp and pp.get("paired_t_p", 1) < 0.05:
        recs.append("[SUBGROUP NOTE]       Pre/post intervention effect is statistically significant. "
                    "AffectsTimePoint should be used as a stratification variable in subgroup analysis.")
    for i, rec in enumerate(recs, 1):
        lines.append(f"  {i}. {rec}")

    lines += ["", "SECTION 10 — OUTPUT FIGURES"]
    for f in sorted(FIGURES_DIR.glob("05_*.png")):
        lines.append(f"  {f.name}")

    lines += ["", "=" * 64]

    out_path = TABLES_DIR / "phase5_eda_summary.txt"
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    logger.info("EDA summary written → %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5.11 — Integrity checks
# ═══════════════════════════════════════════════════════════════════════════════

def task511_integrity_checks(df_a: pd.DataFrame, df_b: pd.DataFrame):
    logger.info("=" * 60)
    logger.info("TASK 5.11 — Integrity checks")

    assert df_a.shape == EXPECTED_SHAPES["features_full"],     "Track A shape changed!"
    assert df_b.shape == EXPECTED_SHAPES["features_protocol"], "Track B shape changed!"
    logger.info("  Shape invariance ✓")

    for col in FORBIDDEN_COLS:
        assert col not in df_a.columns
        assert col not in df_b.columns
    logger.info("  Forbidden columns absent ✓")

    summary_path = TABLES_DIR / "phase5_eda_summary.txt"
    assert summary_path.exists() and summary_path.stat().st_size > 0, "Summary report missing"
    logger.info("  Summary report exists ✓")

    figures = list(FIGURES_DIR.glob("05_*.png"))
    zero_byte = [f for f in figures if f.stat().st_size == 0]
    assert not zero_byte, f"Zero-byte figures: {zero_byte}"
    logger.info("  Figures: %s PNG files, none zero-byte ✓", len(figures))

    vif_path = TABLES_DIR / "phase5_vif_table.txt"
    if vif_path.exists():
        logger.info("  VIF table exists ✓")
    else:
        logger.warning("  VIF table not found — statsmodels may be unavailable")

    logger.info("TASK 5.11 — all integrity checks passed ✓")
