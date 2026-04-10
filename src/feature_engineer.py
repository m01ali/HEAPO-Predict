"""
src/feature_engineer.py

Feature engineering utilities for the HEAPO dataset.
Phase 4 implementation — Tasks 4.0 through 4.9.

Produces two analysis tracks with engineered features:
  Track A — Full-sample        : features_full.parquet     (all 1,272 households)
  Track B — Protocol-enriched  : features_protocol.parquet (152 treatment households)

All engineered columns are ADDITIVE — no original Phase 3 columns are dropped.

Column names verified against:
  - Table 1 (SMD), Table 4 (Metadata), Table 5 (Protocols), Table 6 (Weather)
  in Brudermueller et al. (2025), arXiv:2503.16993v1
  and against Phase 3 merged parquet column inventories.

Phase 5 (EDA) consumes features_full.parquet and features_protocol.parquet directly.
"""

import logging
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Expected shapes from Phase 3 outputs.
# A mismatch means Phase 3 was re-run with different parameters — fail loudly.
EXPECTED_PHASE3_SHAPES: Dict[str, Tuple[int, int]] = {
    "merged_full":     (913_620, 47),
    "merged_protocol": (84_367, 110),
}

# Target proxy columns that must NEVER appear in any output frame.
FORBIDDEN_COLUMNS: frozenset = frozenset({
    "HeatPump_ElectricityConsumption_YearlyEstimated",
    "EXCLUDED_TARGET_PROXY_HeatPump_ElectricityConsumption_YearlyEstimated",
})

# Season name from month integer
SEASON_MAP: Dict[int, str] = {
    12: "winter",  1: "winter",  2: "winter",
     3: "spring",  4: "spring",  5: "spring",
     6: "summer",  7: "summer",  8: "summer",
     9: "autumn", 10: "autumn", 11: "autumn",
}

# Season name → ordinal integer for encoded column
SEASON_INT_MAP: Dict[str, int] = {
    "winter": 0, "spring": 1, "summer": 2, "autumn": 3,
}

# Months classified as heating season
_HEATING_SEASON_MONTHS: frozenset = frozenset({10, 11, 12, 1, 2, 3, 4})

# Living area bucket bins (m²) and labels — right-inclusive intervals
_LIVING_AREA_BINS:   List = [0, 100, 150, 200, 300, 10_000]
_LIVING_AREA_LABELS: List = ["<100", "100-150", "150-200", "200-300", ">300"]
_LIVING_AREA_INT_MAP: Dict[str, int] = {
    "<100": 0, "100-150": 1, "150-200": 2, "200-300": 3, ">300": 4,
}

# Building age (construction year) bucket bins — right-inclusive intervals
_BUILDING_AGE_YEAR_BINS:   List = [0, 1970, 1990, 2010, 9_999]
_BUILDING_AGE_YEAR_LABELS: List = ["pre-1970", "1970-1990", "1990-2010", "post-2010"]
_BUILDING_AGE_INT_MAP: Dict[str, int] = {
    "pre-1970": 0, "1970-1990": 1, "1990-2010": 2, "post-2010": 3,
}

# Heating curve setpoint column names (all three outdoor temperature points)
_HC_COL_20: str = "HeatPump_HeatingCurveSetting_Outside20_BeforeVisit"
_HC_COL_0:  str = "HeatPump_HeatingCurveSetting_Outside0_BeforeVisit"
_HC_COL_M8: str = "HeatPump_HeatingCurveSetting_OutsideMinus8_BeforeVisit"

# Protocol binary flag aliases: engineered name → source column
_PROTOCOL_FLAG_MAP: Dict[str, str] = {
    "heating_curve_too_high":      "HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit",
    "heating_limit_too_high":      "HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit",
    "night_setback_active_before": "HeatPump_NightSetbackSetting_Activated_BeforeVisit",
    "night_setback_active_after":  "HeatPump_NightSetbackSetting_Activated_AfterVisit",
    "descaling_needed":            "DHW_Storage_LastDescaling_TooLongAgo",
    "pipes_not_insulated":         "HeatDistribution_Recommendation_InsulatePipes",
    "hp_correctly_planned":        "HeatPump_Installation_CorrectlyPlanned",
    "hp_internet_connection":      "HeatPump_Installation_InternetConnection",
    "has_buffer_tank":             "HeatDistribution_System_BufferTankAvailable",
}

# Admin/QC columns that are NOT model inputs.
# Retained in output parquets for filtering and reporting, but excluded from feature sets.
NON_FEATURE_COLS: frozenset = frozenset({
    "Household_ID", "Date", "Timestamp", "Group", "AffectsTimePoint",
    "Weather_ID", "is_iqr_outlier", "below_min_days_threshold",
    "hh_no_sunshine", "has_reactive_energy",
    "sunshine_available", "interpolated_flag", "temp_cross_station_flag",
    "living_area_extreme_flag",
    # 97.4% null — not usable as a feature (only dual-meter households)
    "kWh_received_HeatPump",
    # Sub-meter splits — very high null rate, not primary features
    "kWh_received_Other",
    "kvarh_received_capacitive_HeatPump",
    "kvarh_received_capacitive_Other",
    "kvarh_received_inductive_HeatPump",
    "kvarh_received_inductive_Other",
})


# ---------------------------------------------------------------------------
# Task 4.0 — Load Phase 3 Artifacts
# ---------------------------------------------------------------------------

def load_phase3_artifacts(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load merged_full.parquet and merged_protocol.parquet from Phase 3.

    Asserts expected shapes, target variable completeness, and absence of
    forbidden columns. Fails loudly on any violation.

    Returns:
        (track_a, track_b) DataFrames
    """
    def _load(filename: str) -> pd.DataFrame:
        path = processed_dir / filename
        df = pd.read_parquet(path)
        logger.info("Loaded %-35s (%d rows × %d cols)", filename, len(df), df.shape[1])
        return df

    track_a = _load("merged_full.parquet")
    track_b = _load("merged_protocol.parquet")

    # Shape assertions — catch any Phase 3 re-run with different parameters
    for name, df in [("merged_full", track_a), ("merged_protocol", track_b)]:
        expected = EXPECTED_PHASE3_SHAPES[name]
        actual   = df.shape
        if actual != expected:
            raise ValueError(
                f"Shape mismatch for '{name}': expected {expected}, got {actual}. "
                "Phase 3 may have been re-run with different parameters."
            )
        logger.info("Shape check ✓  %-25s %s", name, actual)

    # Target variable completeness
    for name, df in [("Track A", track_a), ("Track B", track_b)]:
        null_target = int(df["kWh_received_Total"].isnull().sum())
        if null_target > 0:
            raise ValueError(
                f"{name}: kWh_received_Total has {null_target} null values. "
                "Expected 0 after Phase 2/3 cleaning."
            )
    logger.info("Target variable check ✓  kWh_received_Total: 0 nulls in both tracks")

    # Forbidden column guard (defense-in-depth — also checked again after engineering)
    for name, df in [("Track A", track_a), ("Track B", track_b)]:
        leaked = FORBIDDEN_COLUMNS & set(df.columns)
        if leaked:
            raise ValueError(f"{name} input contains forbidden columns: {leaked}")
    logger.info("Forbidden column check ✓  target proxy absent from both input tracks")

    return track_a, track_b


# ---------------------------------------------------------------------------
# Task 4.1 — Temporal Features
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Add calendar/temporal features derived from the Timestamp column.

    New columns (all non-null — Timestamp is always present after Phase 2):
      day_of_week     : int8  0=Mon … 6=Sun
      month           : int8  1–12
      is_weekend      : int8  1 if Sat/Sun, 0 otherwise
      day_of_year     : int16 1–366
      season          : str   winter / spring / summer / autumn
      is_heating_season: int8 1 if month ∈ {Oct–Apr}, 0 otherwise

    Returns:
        (df_with_temporal_features, stats_dict)
    """
    df = df.copy()
    ts = df["Timestamp"]

    df["day_of_week"]        = ts.dt.dayofweek.astype("int8")
    df["month"]              = ts.dt.month.astype("int8")
    df["is_weekend"]         = (ts.dt.dayofweek >= 5).astype("int8")
    df["day_of_year"]        = ts.dt.dayofyear.astype("int16")
    df["season"]             = df["month"].map(SEASON_MAP)
    df["is_heating_season"]  = df["month"].isin(_HEATING_SEASON_MONTHS).astype("int8")

    new_cols = ["day_of_week", "month", "is_weekend", "day_of_year",
                "season", "is_heating_season"]
    null_total = int(df[new_cols].isnull().sum().sum())

    logger.info(
        "Temporal features added: %s  (total nulls: %d)",
        ", ".join(new_cols), null_total,
    )
    return df, {"temporal_null_total": null_total}


# ---------------------------------------------------------------------------
# Task 4.2 — Weather-Derived Features
# ---------------------------------------------------------------------------

def add_weather_features(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Add weather-derived features: point-in-time transforms and per-household
    rolling / lag features.

    CRITICAL: df is sorted by ['Household_ID', 'Date'] here and the sorted
    order is preserved for all downstream tasks. A global shift(1) would leak
    data across households — all rolling/lag uses groupby('Household_ID').

    Rolling/lag features are null for:
      - The first row per household (lag always null for the first record)
      - Rows in the weather gap (SMD extends to 2024-03-21, weather ends 2024-02-29)

    New columns:
      Point-in-time : temp_range_daily, HDD_SIA_daily, HDD_US_daily, CDD_US_daily,
                      humidity_x_temp
      Rolling / lag : temp_avg_lag_{n}d (per lag_days config)
                      temp_avg_rolling_{n}d (per rolling_windows_days config)
                      HDD_SIA_rolling_7d

    Returns:
        (df_sorted_with_weather_features, stats_dict)
    """
    df = df.copy()

    # Sort — REQUIRED before all groupby rolling/lag operations
    df = df.sort_values(["Household_ID", "Date"]).reset_index(drop=True)
    logger.info("Sorted by [Household_ID, Date]")

    # ------------------------------------------------------------------
    # 4.2.1 — Point-in-time weather features
    # ------------------------------------------------------------------
    df["temp_range_daily"] = (
        df["Temperature_max_daily"] - df["Temperature_min_daily"]
    )

    # Alias HDD/CDD columns to cleaner names (originals retained)
    df["HDD_SIA_daily"] = df["HeatingDegree_SIA_daily"]
    df["HDD_US_daily"]  = df["HeatingDegree_US_daily"]
    df["CDD_US_daily"]  = df["CoolingDegree_US_daily"]

    df["humidity_x_temp"] = df["Humidity_avg_daily"] * df["Temperature_avg_daily"]

    # ------------------------------------------------------------------
    # 4.2.2 — Per-household rolling and lag features
    # ------------------------------------------------------------------
    fe_cfg      = config.get("feature_engineering", {})
    rolling_cfg = fe_cfg.get("rolling_windows_days", [3, 7])
    lag_cfg     = fe_cfg.get("lag_days", [1])

    # Lag: previous N days' average temperature per household
    for lag in lag_cfg:
        col = f"temp_avg_lag_{lag}d"
        df[col] = (
            df.groupby("Household_ID", sort=False)["Temperature_avg_daily"].shift(lag)
        )
        logger.info("  %s: %d nulls (≥1 per HH from first-record lag + weather gap)",
                    col, df[col].isnull().sum())

    # Rolling mean: average temperature over N-day window per household
    for window in rolling_cfg:
        col = f"temp_avg_rolling_{window}d"
        df[col] = (
            df.groupby("Household_ID", sort=False)["Temperature_avg_daily"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        logger.info("  %s: %d nulls", col, df[col].isnull().sum())

    # Rolling sum: 7-day cumulative heating demand per household
    col = "HDD_SIA_rolling_7d"
    df[col] = (
        df.groupby("Household_ID", sort=False)["HeatingDegree_SIA_daily"]
        .transform(lambda x: x.rolling(7, min_periods=1).sum())
    )
    logger.info("  %s: %d nulls", col, df[col].isnull().sum())

    weather_gap_rows = int(df["Temperature_avg_daily"].isnull().sum())
    logger.info(
        "Weather features added (point-in-time: 5, rolling/lag: %d) | "
        "weather gap rows (null weather): %d",
        len(rolling_cfg) + len(lag_cfg) + 1,
        weather_gap_rows,
    )
    return df, {
        "weather_gap_rows":            weather_gap_rows,
        "weather_lag_1d_nulls":        int(df["temp_avg_lag_1d"].isnull().sum()),
        "weather_HDD_rolling_7d_nulls":int(df["HDD_SIA_rolling_7d"].isnull().sum()),
    }


# ---------------------------------------------------------------------------
# Task 4.3 — Household Static Features (Full-Sample Track A)
# ---------------------------------------------------------------------------

def add_household_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Encode metadata-derived static household features.

    Tasks 4.3.1–4.3.6:
      4.3.1  Building type OHE          (house / apartment)
      4.3.2  HP installation type OHE   (air-source / ground-source / unknown)
      4.3.3  DHW source composite + OHE
      4.3.4  Heat distribution composite + OHE
      4.3.5  Appliance flag aliases     (has_ev, has_dryer, has_freezer)
      4.3.6  Living area bucket         (ordinal string; encoded in Task 4.8)

    All engineered columns are strictly additive — originals untouched.

    Returns:
        (df_with_household_features, stats_dict)
    """
    df = df.copy()
    stats: dict = {}

    # ------------------------------------------------------------------
    # 4.3.1 — Building type OHE
    # ------------------------------------------------------------------
    btype = df["Survey_Building_Type"]
    unique_btypes = sorted(btype.dropna().unique().tolist())
    logger.info("Survey_Building_Type unique values: %s  (null rows: %d)",
                unique_btypes, btype.isnull().sum())

    df["building_type_house"]     = _ohe_col(btype, "house")
    df["building_type_apartment"] = _ohe_col(btype, "apartment")
    stats["building_type_null_rows"] = int(btype.isnull().sum())

    # ------------------------------------------------------------------
    # 4.3.2 — HP installation type OHE
    # Survey values per paper Table 4: "air-source", "ground-source"
    # Phase 2 set NaN → "unknown"
    # ------------------------------------------------------------------
    hptype = df["Survey_HeatPump_Installation_Type"]
    unique_hptypes = sorted(hptype.dropna().unique().tolist())
    logger.info("Survey_HeatPump_Installation_Type unique values: %s  (null rows: %d)",
                unique_hptypes, hptype.isnull().sum())

    df["hp_type_air_source"]    = _ohe_col(hptype, "air-source")
    df["hp_type_ground_source"] = _ohe_col(hptype, "ground-source")
    df["hp_type_unknown"]       = _ohe_col(hptype, "unknown")
    stats["hp_type_null_rows"] = int(hptype.isnull().sum())

    # ------------------------------------------------------------------
    # 4.3.3 — DHW source composite + OHE
    # ------------------------------------------------------------------
    dhw_hp  = df["Survey_DHW_Production_ByHeatPump"]
    dhw_ewh = df["Survey_DHW_Production_ByElectricWaterHeater"]
    dhw_sol = df["Survey_DHW_Production_BySolar"]

    hp_true  = _to_bool(dhw_hp)
    ewh_true = _to_bool(dhw_ewh)
    sol_true = _to_bool(dhw_sol)
    n_true   = hp_true.astype(int) + ewh_true.astype(int) + sol_true.astype(int)

    # All three source columns null → metadata missing for this household
    all_na = dhw_hp.isna() & dhw_ewh.isna() & dhw_sol.isna()

    df["dhw_source"] = np.select(
        [
            all_na,
            n_true > 1,
            hp_true  & (n_true == 1),
            ewh_true & (n_true == 1),
            sol_true & (n_true == 1),
        ],
        ["unknown", "combined", "heat_pump", "electric_water_heater", "solar"],
        default="unknown",
    )

    for cat, col in [
        ("heat_pump",             "dhw_hp"),
        ("electric_water_heater", "dhw_ewh"),
        ("solar",                 "dhw_solar"),
        ("combined",              "dhw_combined"),
        ("unknown",               "dhw_unknown"),
    ]:
        df[col] = (df["dhw_source"] == cat).astype("float64")

    stats["dhw_source_dist"] = df["dhw_source"].value_counts().to_dict()
    logger.info("DHW source distribution: %s", stats["dhw_source_dist"])

    # ------------------------------------------------------------------
    # 4.3.4 — Heat distribution composite + OHE
    # ------------------------------------------------------------------
    floor_col = "Survey_HeatDistribution_System_FloorHeating"
    rad_col   = "Survey_HeatDistribution_System_Radiator"

    floor_true = _to_bool(df[floor_col])
    rad_true   = _to_bool(df[rad_col])
    both_na    = df[floor_col].isna() & df[rad_col].isna()

    df["heat_distribution"] = np.select(
        [
            both_na,
            floor_true & rad_true,
            floor_true & ~rad_true,
            rad_true   & ~floor_true,
        ],
        ["unknown", "both", "floor", "radiators"],
        default="unknown",
    )

    for cat, col in [
        ("floor",    "heat_dist_floor"),
        ("radiators","heat_dist_radiator"),
        ("both",     "heat_dist_both"),
        ("unknown",  "heat_dist_unknown"),
    ]:
        df[col] = (df["heat_distribution"] == cat).astype("float64")

    stats["heat_dist_dist"] = df["heat_distribution"].value_counts().to_dict()
    logger.info("Heat distribution distribution: %s", stats["heat_dist_dist"])

    # ------------------------------------------------------------------
    # 4.3.5 — Appliance flag aliases (has_pv already int in Phase 2 output)
    # ------------------------------------------------------------------
    df["has_ev"]      = df["Survey_Installation_HasElectricVehicle"].fillna(False).astype("int8")
    df["has_dryer"]   = df["Survey_Installation_HasDryer"].fillna(False).astype("int8")
    df["has_freezer"] = df["Survey_Installation_HasFreezer"].fillna(False).astype("int8")

    stats["has_ev_true"]      = int(df["has_ev"].sum())
    stats["has_dryer_true"]   = int(df["has_dryer"].sum())
    stats["has_freezer_true"] = int(df["has_freezer"].sum())
    logger.info(
        "Appliance flags: has_ev=%d, has_dryer=%d, has_freezer=%d rows = True",
        stats["has_ev_true"], stats["has_dryer_true"], stats["has_freezer_true"],
    )

    # ------------------------------------------------------------------
    # 4.3.6 — Living area bucket (ordinal string; integer encoding in Task 4.8)
    # pd.cut with .astype(object) preserves NaN as actual NaN (not "nan" string)
    # ------------------------------------------------------------------
    df["living_area_bucket"] = pd.cut(
        df["Survey_Building_LivingArea"],
        bins=_LIVING_AREA_BINS,
        labels=_LIVING_AREA_LABELS,
        right=True,
    ).astype(object)

    stats["living_area_bucket_dist"] = (
        df["living_area_bucket"].value_counts(dropna=False).to_dict()
    )
    logger.info("Living area bucket distribution: %s", stats["living_area_bucket_dist"])

    return df, stats


# ---------------------------------------------------------------------------
# Task 4.4 — Protocol-Enriched Features (Track B only)
# ---------------------------------------------------------------------------

def add_protocol_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Engineer derived features from on-site inspection protocol data (Track B only).

    Tasks 4.4.1–4.4.8:
      4.4.1  Building age and age bucket
      4.4.2  HP age
      4.4.3  HP heating capacity per heated floor area
      4.4.4  Heating curve gradient features (4 features from 3 setpoints)
      4.4.5  Binary issue flag aliases (9 flags)
      4.4.6  HP location OHE
      4.4.7  Renovation composite score (0–4)
      4.4.8  post_intervention passthrough check

    All engineered columns strictly additive.

    Returns:
        (df_with_protocol_features, stats_dict)
    """
    df = df.copy()
    stats: dict = {}

    # ------------------------------------------------------------------
    # 4.4.1 — Building age and bucket
    # ------------------------------------------------------------------
    df["building_age"] = df["Visit_Year"] - df["Building_ConstructionYear"]

    # Flag impossible ages (construction year after visit year)
    df["building_age_error_flag"] = (df["building_age"] < 0).fillna(False)
    n_age_errors = int(df["building_age_error_flag"].sum())
    if n_age_errors > 0:
        logger.warning(
            "building_age: %d negative values flagged and set to NaN "
            "(construction year > visit year)",
            n_age_errors,
        )
        df.loc[df["building_age"] < 0, "building_age"] = np.nan

    # building_age_bucket — bin on construction year, not age
    df["building_age_bucket"] = pd.cut(
        df["Building_ConstructionYear"],
        bins=_BUILDING_AGE_YEAR_BINS,
        labels=_BUILDING_AGE_YEAR_LABELS,
        right=True,
    ).astype(object)

    stats["building_age_null_rows"]  = int(df["building_age"].isnull().sum())
    stats["building_age_error_rows"] = n_age_errors
    logger.info(
        "building_age: %d null rows (incl. %d error rows set to NaN)",
        stats["building_age_null_rows"], n_age_errors,
    )

    # ------------------------------------------------------------------
    # 4.4.2 — HP age
    # ------------------------------------------------------------------
    df["hp_age"] = df["Visit_Year"] - df["HeatPump_Installation_Year"]
    stats["hp_age_null_rows"] = int(df["hp_age"].isnull().sum())
    logger.info(
        "hp_age: %d null rows (%.1f%% of Track B) — HeatPump_Installation_Year missing",
        stats["hp_age_null_rows"],
        100 * stats["hp_age_null_rows"] / len(df),
    )

    # ------------------------------------------------------------------
    # 4.4.3 — HP heating capacity per heated floor area
    # Building_FloorAreaHeated_Total has 0 null rows per merge report.
    # Guard against zero area to avoid division-by-zero.
    # ------------------------------------------------------------------
    n_zero_area = int((df["Building_FloorAreaHeated_Total"] == 0).sum())
    if n_zero_area > 0:
        logger.warning(
            "Building_FloorAreaHeated_Total: %d zero values — replaced with NaN "
            "to prevent division-by-zero in hp_capacity_per_area",
            n_zero_area,
        )
    df["hp_capacity_per_area"] = (
        df["HeatPump_Installation_HeatingCapacity"]
        / df["Building_FloorAreaHeated_Total"].replace(0, np.nan)
    )
    stats["hp_capacity_per_area_null_rows"] = int(df["hp_capacity_per_area"].isnull().sum())
    logger.info("hp_capacity_per_area: %d null rows", stats["hp_capacity_per_area_null_rows"])

    # ------------------------------------------------------------------
    # 4.4.4 — Heating curve gradient features
    # Three supply temperature setpoints at outdoor temps: +20°C, 0°C, −8°C
    # Most discriminative protocol feature (40.98% of visits had this too high)
    # ------------------------------------------------------------------
    if all(c in df.columns for c in [_HC_COL_20, _HC_COL_0, _HC_COL_M8]):
        t20 = df[_HC_COL_20]
        t0  = df[_HC_COL_0]
        tm8 = df[_HC_COL_M8]

        # Upper segment: supply temp rise per 1°C outdoor drop (outdoor 20°C→0°C range)
        df["heating_curve_gradient_upper"] = (t0 - t20) / 20.0

        # Lower segment: supply temp rise per 1°C outdoor drop (outdoor 0°C→−8°C range)
        df["heating_curve_gradient_lower"] = (tm8 - t0) / 8.0

        # Full-range gradient (outdoor 20°C → −8°C)
        df["heating_curve_gradient_full"] = (tm8 - t20) / 28.0

        # Non-linearity: positive = curve steepens at low temperatures (expected correct config)
        df["heating_curve_nonlinearity"] = (
            df["heating_curve_gradient_lower"] - df["heating_curve_gradient_upper"]
        )

        stats["hc_gradient_null_rows"] = int(df["heating_curve_gradient_full"].isnull().sum())
        logger.info(
            "Heating curve gradient features: 4 features, %d null rows",
            stats["hc_gradient_null_rows"],
        )
    else:
        missing = [c for c in [_HC_COL_20, _HC_COL_0, _HC_COL_M8] if c not in df.columns]
        logger.warning("Heating curve gradient features SKIPPED — missing columns: %s", missing)

    # ------------------------------------------------------------------
    # 4.4.5 — Binary issue flag aliases
    # ------------------------------------------------------------------
    for alias, source_col in _PROTOCOL_FLAG_MAP.items():
        if source_col in df.columns:
            df[alias] = _bool_to_float(df[source_col])
            null_rows = int(df[alias].isnull().sum())
            null_pct  = 100 * null_rows / len(df)
            logger.info(
                "  %-35s ← %-55s  null: %d rows (%.1f%%)",
                alias, source_col, null_rows, null_pct,
            )
            stats[f"null_{alias}"] = null_rows
        else:
            logger.warning(
                "Source column '%s' not found for flag alias '%s' — skipped",
                source_col, alias,
            )

    # ------------------------------------------------------------------
    # 4.4.6 — HP location OHE
    # Expected values per paper Table 5: inside, outside, split
    # ------------------------------------------------------------------
    loc_col = "HeatPump_Installation_Location"
    if loc_col in df.columns:
        unique_locs = sorted(df[loc_col].dropna().unique().tolist())
        logger.info("HeatPump_Installation_Location unique values: %s", unique_locs)
        for loc_val in unique_locs:
            safe_name = loc_val.lower().replace(" ", "_").replace("-", "_")
            df[f"hp_location_{safe_name}"] = _ohe_col(df[loc_col], loc_val)
        stats["hp_location_null_rows"] = int(df[loc_col].isnull().sum())
        logger.info("  hp_location OHE: %d null rows in source", stats["hp_location_null_rows"])
    else:
        logger.warning("Column '%s' not found — HP location OHE skipped", loc_col)

    # ------------------------------------------------------------------
    # 4.4.7 — Renovation composite score (0–4)
    # Count of renovated building envelope components
    # ------------------------------------------------------------------
    ren_cols = [
        "Building_Renovated_Windows",
        "Building_Renovated_Roof",
        "Building_Renovated_Walls",
        "Building_Renovated_Floor",
    ]
    ren_present = [c for c in ren_cols if c in df.columns]
    if len(ren_present) == 4:
        df["renovation_score"] = sum(
            df[c].fillna(False).astype(int) for c in ren_present
        )
        stats["renovation_score_dist"] = df["renovation_score"].value_counts().sort_index().to_dict()
        logger.info("renovation_score distribution: %s", stats["renovation_score_dist"])
    else:
        missing_ren = [c for c in ren_cols if c not in df.columns]
        logger.warning(
            "renovation_score SKIPPED — missing columns: %s", missing_ren
        )

    # ------------------------------------------------------------------
    # 4.4.8 — post_intervention passthrough check
    # Already added in Phase 2; just verify presence and distribution
    # ------------------------------------------------------------------
    assert "post_intervention" in df.columns, (
        "post_intervention column missing from Track B — expected from Phase 2 cleaning."
    )
    n_post = int((df["post_intervention"] == 1).sum())
    logger.info(
        "post_intervention check ✓  %d rows = 1 (%.1f%% of Track B)",
        n_post, 100 * n_post / len(df),
    )

    return df, stats


# ---------------------------------------------------------------------------
# Task 4.5 — Reactive Energy Features (optional)
# ---------------------------------------------------------------------------

def add_reactive_energy_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Compute a power factor proxy from total active and reactive energy columns.

    Applied to all rows; naturally null where reactive energy columns are null
    (i.e., ~30% of rows where has_reactive_energy == 0).

    Guarded against division by zero (apparent_energy == 0).

    New column: power_factor_proxy  (float64, range [0, 1], null where reactive = null)

    Returns:
        (df_with_reactive_features, stats_dict)
    """
    df = df.copy()

    cap_col = "kvarh_received_capacitive_Total"
    ind_col = "kvarh_received_inductive_Total"
    kwh_col = "kWh_received_Total"

    # Total reactive energy magnitude (vector sum of capacitive + inductive)
    kvarh_total = np.sqrt(df[cap_col] ** 2 + df[ind_col] ** 2)

    # Apparent energy: phasor combination of active and reactive
    apparent_energy = np.sqrt(df[kwh_col] ** 2 + kvarh_total ** 2)

    # Avoid division by zero (kWh=0 should not exist after Phase 2, but guard anyway)
    apparent_safe = apparent_energy.replace(0.0, np.nan)

    df["power_factor_proxy"] = df[kwh_col] / apparent_safe

    # Clamp to [0, 1] — power factor is physically bounded
    df["power_factor_proxy"] = df["power_factor_proxy"].clip(0.0, 1.0)

    null_rows = int(df["power_factor_proxy"].isnull().sum())
    reactive_rows = int((df.get("has_reactive_energy", pd.Series(0, index=df.index)) == 1).sum())
    logger.info(
        "power_factor_proxy: %d null rows | has_reactive_energy=1: %d rows",
        null_rows, reactive_rows,
    )
    return df, {"pf_null_rows": null_rows, "reactive_rows": reactive_rows}


# ---------------------------------------------------------------------------
# Task 4.6 — Autoregressive Energy Features (optional, off by default)
# ---------------------------------------------------------------------------

def add_autoregressive_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Add autoregressive lag/rolling features of the target variable.

    WARNING: MUST use groupby(Household_ID) — never global shift.
    NOT included in the primary analysis; only for sensitivity analysis.
    Phase 6 MUST use strictly temporal train/test split when these are enabled.

    Assumes df is already sorted by [Household_ID, Date] from Task 4.2.

    New columns:
      kWh_total_lag_1d            : previous day's consumption per household
      kWh_total_rolling_7d_mean   : 7-day rolling mean per household
      kWh_total_rolling_30d_mean  : 30-day rolling mean per household

    Returns:
        (df_with_ar_features, stats_dict)
    """
    df = df.copy()

    df["kWh_total_lag_1d"] = (
        df.groupby("Household_ID", sort=False)["kWh_received_Total"].shift(1)
    )
    df["kWh_total_rolling_7d_mean"] = (
        df.groupby("Household_ID", sort=False)["kWh_received_Total"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["kWh_total_rolling_30d_mean"] = (
        df.groupby("Household_ID", sort=False)["kWh_received_Total"]
        .transform(lambda x: x.rolling(30, min_periods=1).mean())
    )

    lag_nulls = int(df["kWh_total_lag_1d"].isnull().sum())
    logger.info(
        "Autoregressive features added: kWh_total_lag_1d (%d nulls), "
        "kWh_total_rolling_7d_mean (%d nulls), kWh_total_rolling_30d_mean (%d nulls)",
        lag_nulls,
        int(df["kWh_total_rolling_7d_mean"].isnull().sum()),
        int(df["kWh_total_rolling_30d_mean"].isnull().sum()),
    )
    return df, {"ar_lag_1d_nulls": lag_nulls}


# ---------------------------------------------------------------------------
# Task 4.7 — Forbidden Column Guard
# ---------------------------------------------------------------------------

def check_forbidden_columns(df_full: pd.DataFrame, df_protocol: pd.DataFrame) -> None:
    """
    Assert that target proxy columns are absent from both output tracks.
    Defense-in-depth: this check also runs in Task 4.0 on the inputs.
    Raises ValueError on violation.
    """
    for name, df in [
        ("Track A (features_full)",     df_full),
        ("Track B (features_protocol)", df_protocol),
    ]:
        leaked = FORBIDDEN_COLUMNS & set(df.columns)
        if leaked:
            raise ValueError(
                f"TARGET LEAKAGE: forbidden columns found in {name}: {leaked}."
            )
    logger.info(
        "Forbidden column guard ✓  target proxy absent from both output tracks"
    )


# ---------------------------------------------------------------------------
# Task 4.8 — Categorical Encoding
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Encode string categorical features to integer ordinals for model readiness.
    Original string columns are retained alongside the encoded columns.

    Encoded columns:
      season               → season_encoded              (int8, 0=winter … 3=autumn)
      living_area_bucket   → living_area_bucket_encoded  (float64, 0=<100 … 4=>300, NaN if null)
      building_age_bucket  → building_age_bucket_encoded (float64, Track B only)

    All OHE float64 columns from Tasks 4.3/4.4 are already numeric — no action.

    Returns:
        (df_with_encoded_categoricals, stats_dict)
    """
    df = df.copy()
    stats: dict = {}

    # season → int8 (season is derived from month; always non-null)
    df["season_encoded"] = df["season"].map(SEASON_INT_MAP).astype("int8")
    stats["season_encoded_null"] = int(df["season_encoded"].isnull().sum())
    logger.info(
        "season_encoded distribution: %s",
        df["season_encoded"].value_counts().sort_index().to_dict(),
    )

    # living_area_bucket → float64 ordinal (NaN where source area is null)
    if "living_area_bucket" in df.columns:
        df["living_area_bucket_encoded"] = df["living_area_bucket"].map(_LIVING_AREA_INT_MAP)
        stats["living_area_bucket_encoded_null"] = int(
            df["living_area_bucket_encoded"].isnull().sum()
        )
        logger.info(
            "living_area_bucket_encoded: %d null rows",
            stats["living_area_bucket_encoded_null"],
        )

    # building_age_bucket → float64 ordinal (Track B only; NaN where construction year is null)
    if "building_age_bucket" in df.columns:
        df["building_age_bucket_encoded"] = df["building_age_bucket"].map(_BUILDING_AGE_INT_MAP)
        stats["building_age_bucket_encoded_null"] = int(
            df["building_age_bucket_encoded"].isnull().sum()
        )
        logger.info(
            "building_age_bucket_encoded: %d null rows",
            stats["building_age_bucket_encoded_null"],
        )

    return df, stats


# ---------------------------------------------------------------------------
# Task 4.9 — Integrity Checks
# ---------------------------------------------------------------------------

def run_integrity_checks(df_full: pd.DataFrame, df_protocol: pd.DataFrame) -> None:
    """
    Six post-engineering integrity checks.

    Check 1 : Track A row count is still 913,620 (no rows dropped during engineering)
    Check 2 : Track B row count is still 84,367
    Check 3 : kWh_received_Total has 0 nulls in both tracks
    Check 4 : No duplicate (Household_ID, Date) pairs in either track
    Check 5 : Per-household rolling correctness — first weather-valid row per
              household has null temp_avg_lag_1d
    Check 6 : Forbidden columns absent from both output tracks

    Raises ValueError on any hard failure; logs warnings for soft anomalies.
    """
    logger.info("Running post-engineering integrity checks...")
    pass_count = 0

    # Check 1 — Track A row count
    exp_a = EXPECTED_PHASE3_SHAPES["merged_full"][0]
    if len(df_full) != exp_a:
        raise ValueError(
            f"Check 1 FAIL: Track A has {len(df_full)} rows, expected {exp_a}"
        )
    logger.info("Check 1 ✓  Track A row count: %d", len(df_full))
    pass_count += 1

    # Check 2 — Track B row count
    exp_b = EXPECTED_PHASE3_SHAPES["merged_protocol"][0]
    if len(df_protocol) != exp_b:
        raise ValueError(
            f"Check 2 FAIL: Track B has {len(df_protocol)} rows, expected {exp_b}"
        )
    logger.info("Check 2 ✓  Track B row count: %d", len(df_protocol))
    pass_count += 1

    # Check 3 — target variable completeness
    for name, df in [("Track A", df_full), ("Track B", df_protocol)]:
        null_target = int(df["kWh_received_Total"].isnull().sum())
        if null_target > 0:
            raise ValueError(
                f"Check 3 FAIL: {name} kWh_received_Total has {null_target} null values"
            )
    logger.info("Check 3 ✓  kWh_received_Total: 0 nulls in both tracks")
    pass_count += 1

    # Check 4 — no duplicate (Household_ID, Date) pairs
    for name, df in [("Track A", df_full), ("Track B", df_protocol)]:
        dup_count = int(df.duplicated(subset=["Household_ID", "Date"]).sum())
        if dup_count > 0:
            raise ValueError(
                f"Check 4 FAIL: {name} has {dup_count} duplicate (Household_ID, Date) pairs"
            )
    logger.info("Check 4 ✓  No duplicate (Household_ID, Date) pairs in either track")
    pass_count += 1

    # Check 5 — rolling correctness: the actual first row (earliest date) per
    # household must have null temp_avg_lag_1d (no prior day to lag from).
    # Use .nth(0) not .first() — .first() returns the first NON-NULL value per
    # column, which would give a false positive here.
    if "temp_avg_lag_1d" in df_full.columns:
        first_rows = (
            df_full
            .sort_values(["Household_ID", "Date"])
            .groupby("Household_ID", sort=False)
            .nth(0)
        )
        # Restrict to households whose actual first row has valid weather
        # (households whose first date pre-dates weather coverage are excluded)
        first_with_weather = first_rows[first_rows["Temperature_avg_daily"].notna()]
        non_null_lag = int(first_with_weather["temp_avg_lag_1d"].notna().sum())
        if non_null_lag > 0:
            logger.warning(
                "Check 5 WARNING: %d households have non-null temp_avg_lag_1d on "
                "their actual first row — verify per-household rolling",
                non_null_lag,
            )
        else:
            logger.info(
                "Check 5 ✓  temp_avg_lag_1d: first row per household is null "
                "(correct per-household rolling confirmed)"
            )
        pass_count += 1

    # Check 6 — forbidden columns absent from outputs
    check_forbidden_columns(df_full, df_protocol)
    pass_count += 1

    logger.info("All %d integrity checks passed ✓", pass_count)


# ---------------------------------------------------------------------------
# Task 4.9 — Feature Catalog Report
# ---------------------------------------------------------------------------

def generate_feature_report(
    df_full: pd.DataFrame,
    df_protocol: pd.DataFrame,
    stats_a: dict,
    stats_b: dict,
) -> str:
    """
    Generate a human-readable feature catalog and Phase 4 summary report.

    Sections:
      1. Engineering summary (row/col counts, new columns added per track)
      2. Feature catalog table (name, source, dtype, track, missingness, notes)
      3. Missing value detail for key engineered features
      4. Excluded variable reminder

    Returns:
        report_text (str)
    """
    buf = StringIO()
    W   = 70

    def ln(text=""):
        print(text, file=buf)

    ln("=" * W)
    ln("PHASE 4 FEATURE ENGINEERING REPORT")
    ln("HEAPO-Predict — Daily HP Electricity Consumption")
    ln("=" * W)
    ln()

    # ------------------------------------------------------------------
    # Section 1 — Engineering summary
    # ------------------------------------------------------------------
    ln("=" * W)
    ln("ENGINEERING SUMMARY")
    ln("=" * W)
    ln(f"  Track A (features_full)     : {len(df_full):>9,} rows"
       f"  |  {df_full['Household_ID'].nunique():>5} households"
       f"  |  {df_full.shape[1]:>4} cols")
    ln(f"  Track B (features_protocol) : {len(df_protocol):>9,} rows"
       f"  |  {df_protocol['Household_ID'].nunique():>5} households"
       f"  |  {df_protocol.shape[1]:>4} cols")
    ln()
    orig_a = EXPECTED_PHASE3_SHAPES["merged_full"][1]
    orig_b = EXPECTED_PHASE3_SHAPES["merged_protocol"][1]
    ln(f"  Phase 3 cols (Track A) : {orig_a:>4}   →  New cols added: {df_full.shape[1] - orig_a}")
    ln(f"  Phase 3 cols (Track B) : {orig_b:>4}   →  New cols added: {df_protocol.shape[1] - orig_b}")
    ln()
    if stats_a.get("weather_gap_rows"):
        ln(f"  NOTE: {stats_a['weather_gap_rows']:,} rows (SMD > 2024-02-29) have null weather features.")
        ln("        These rows are retained; Phase 6 will exclude them from training/evaluation.")
    ln()

    # ------------------------------------------------------------------
    # Section 2 — Feature catalog
    # ------------------------------------------------------------------
    ln("=" * W)
    ln("FEATURE CATALOG")
    ln("=" * W)
    catalog = _build_feature_catalog(df_full, df_protocol)
    col_w = 50
    ln(f"  {'Feature':<{col_w}}  {'Trk':<4}  {'Dtype':<12}  {'Miss-A%':>8}  {'Miss-B%':>8}  Note")
    ln("  " + "-" * (col_w + 50))
    for e in catalog:
        ma = f"{e['miss_pct_a']:.1f}%" if e["miss_pct_a"] is not None else "  N/A"
        mb = f"{e['miss_pct_b']:.1f}%" if e["miss_pct_b"] is not None else "  N/A"
        ln(f"  {e['name']:<{col_w}}  {e['track']:<4}  {e['dtype']:<12}  "
           f"{ma:>8}  {mb:>8}  {e['note']}")
    ln()

    # ------------------------------------------------------------------
    # Section 3 — Missing value detail for key engineered features
    # ------------------------------------------------------------------
    ln("=" * W)
    ln("MISSING VALUE DETAIL — KEY ENGINEERED FEATURES")
    ln("=" * W)

    n_a = len(df_full)
    key_cols_a = [
        "temp_range_daily", "temp_avg_lag_1d", "temp_avg_rolling_3d",
        "temp_avg_rolling_7d", "HDD_SIA_rolling_7d", "humidity_x_temp",
        "building_type_house", "building_type_apartment",
        "hp_type_air_source", "hp_type_ground_source", "hp_type_unknown",
        "dhw_source", "heat_distribution",
        "has_ev", "has_dryer", "has_freezer",
        "power_factor_proxy", "living_area_bucket",
    ]
    ln(f"  Track A  (n = {n_a:,})")
    ln(f"  {'Column':<50}  {'Null rows':>12}  {'%':>8}")
    ln("  " + "-" * 75)
    for col in key_cols_a:
        if col in df_full.columns:
            n_null = int(df_full[col].isnull().sum())
            ln(f"  {col:<50}  {n_null:>12,}  {100*n_null/n_a:>7.1f}%")
    ln()

    n_b = len(df_protocol)
    key_cols_b = [
        "building_age", "hp_age", "hp_capacity_per_area",
        "heating_curve_gradient_full", "heating_curve_nonlinearity",
        "heating_curve_too_high", "heating_limit_too_high",
        "night_setback_active_before", "renovation_score",
    ]
    ln(f"  Track B  (n = {n_b:,})")
    ln(f"  {'Column':<50}  {'Null rows':>12}  {'%':>8}")
    ln("  " + "-" * 75)
    for col in key_cols_b:
        if col in df_protocol.columns:
            n_null = int(df_protocol[col].isnull().sum())
            ln(f"  {col:<50}  {n_null:>12,}  {100*n_null/n_b:>7.1f}%")
    ln()

    # ------------------------------------------------------------------
    # Section 4 — Excluded variable reminder
    # ------------------------------------------------------------------
    ln("=" * W)
    ln("EXCLUDED VARIABLES")
    ln("=" * W)
    ln("  HeatPump_ElectricityConsumption_YearlyEstimated")
    ln("  Reason : Near-direct proxy for the target (consultant's own annual kWh estimate).")
    ln("           Inclusion would constitute data leakage. Renamed with EXCLUDED_ prefix")
    ln("           in Phase 2 and absent from all Phase 3/4 outputs.")
    ln()
    ln("  kWh_received_HeatPump")
    ln("  Reason : Null for 97.4% of rows (1,239/1,272 households). Only dual-meter")
    ln("           households have a dedicated HP meter. Not usable as a general feature.")
    ln("           Retained in parquet for dual-meter subgroup analysis if needed.")
    ln()

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ohe_col(series: pd.Series, value: str) -> pd.Series:
    """
    One-hot encode: 1.0 where series == value, 0.0 otherwise, NaN where source is null.
    Returns float64.
    """
    mask_null = series.isna()
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[~mask_null] = (series[~mask_null] == value).astype("float64")
    return out


def _to_bool(series: pd.Series) -> pd.Series:
    """
    Convert a pandas nullable boolean (or object bool) to a regular bool Series.
    NA values become False (safe for counting purposes).
    """
    return series.fillna(False).astype(bool)


def _bool_to_float(series: pd.Series) -> pd.Series:
    """
    Convert a nullable boolean column to float64: True→1.0, False→0.0, NA→NaN.
    """
    mask_null = series.isna()
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[~mask_null] = series[~mask_null].astype(float)
    return out


def _build_feature_catalog(
    df_full: pd.DataFrame,
    df_protocol: pd.DataFrame,
) -> List[dict]:
    """
    Build catalog entries for every column in either track.
    Returns list of dicts sorted by (track, source, name).
    """
    def _classify_source(col: str) -> str:
        if col.startswith("Survey_") or col in {
            "living_area_bucket", "living_area_bucket_encoded",
            "building_type_house", "building_type_apartment",
            "hp_type_air_source", "hp_type_ground_source", "hp_type_unknown",
            "dhw_source", "dhw_hp", "dhw_ewh", "dhw_solar", "dhw_combined", "dhw_unknown",
            "heat_distribution", "heat_dist_floor", "heat_dist_radiator",
            "heat_dist_both", "heat_dist_unknown",
            "has_ev", "has_dryer", "has_freezer",
        }:
            return "Metadata"
        if col.startswith("kWh") or col.startswith("kvarh") or col in {
            "Household_ID", "Group", "AffectsTimePoint", "Timestamp", "Date",
            "is_iqr_outlier", "post_intervention", "below_min_days_threshold",
            "has_pv", "has_reactive_energy", "Weather_ID",
        }:
            return "SMD"
        if (
            col.startswith("Temperature") or col.startswith("HeatingDegree")
            or col.startswith("CoolingDegree") or col.startswith("Humidity")
            or col.startswith("Precipitation") or col.startswith("Sunshine")
            or col in {
                "sunshine_available", "interpolated_flag", "temp_cross_station_flag",
                "hh_no_sunshine", "temp_range_daily", "HDD_SIA_daily", "HDD_US_daily",
                "CDD_US_daily", "humidity_x_temp", "temp_avg_lag_1d",
                "temp_avg_rolling_3d", "temp_avg_rolling_7d", "HDD_SIA_rolling_7d",
            }
        ):
            return "Weather"
        if (
            col.startswith("Building") or col.startswith("HeatPump")
            or col.startswith("HeatDistribution") or col.startswith("DHW")
            or col.startswith("hp_location_")
            or col in {
                "Visit_Year",
                "building_age", "building_age_bucket", "building_age_bucket_encoded",
                "building_age_error_flag",
                "hp_age", "hp_capacity_per_area", "renovation_score",
                "heating_curve_gradient_upper", "heating_curve_gradient_lower",
                "heating_curve_gradient_full", "heating_curve_nonlinearity",
                "heating_curve_too_high", "heating_limit_too_high",
                "night_setback_active_before", "night_setback_active_after",
                "descaling_needed", "pipes_not_insulated",
                "hp_correctly_planned", "hp_internet_connection", "has_buffer_tank",
            }
        ):
            return "Protocol"
        if col in {
            "day_of_week", "month", "is_weekend", "day_of_year",
            "season", "season_encoded", "is_heating_season",
            "power_factor_proxy",
        } or col.startswith("kWh_total_"):
            return "Engineered"
        return "Other"

    all_cols_a = set(df_full.columns)
    all_cols_b = set(df_protocol.columns)
    all_cols   = sorted(all_cols_a | all_cols_b)

    n_a = len(df_full)
    n_b = len(df_protocol)
    entries = []

    for col in all_cols:
        in_a = col in all_cols_a
        in_b = col in all_cols_b
        track = "Both" if (in_a and in_b) else ("A" if in_a else "B")

        miss_pct_a: Optional[float] = None
        miss_pct_b: Optional[float] = None

        if in_a:
            null_a = int(df_full[col].isnull().sum())
            miss_pct_a = 100 * null_a / n_a if n_a > 0 else 0.0
        if in_b:
            null_b = int(df_protocol[col].isnull().sum())
            miss_pct_b = 100 * null_b / n_b if n_b > 0 else 0.0

        dtype = str(df_full[col].dtype) if in_a else str(df_protocol[col].dtype)

        note = ""
        if col in NON_FEATURE_COLS:
            note = "admin/flag — not a model input"
        elif col in FORBIDDEN_COLUMNS:
            note = "EXCLUDED — target proxy (leakage)"

        entries.append({
            "name":       col,
            "source":     _classify_source(col),
            "dtype":      dtype[:12],
            "track":      track,
            "miss_pct_a": miss_pct_a,
            "miss_pct_b": miss_pct_b,
            "note":       note,
        })

    return entries
