"""
src/data_preparation.py

Phase 6 — Data Preparation for Modeling module.
Called by scripts/06_data_preparation.py.

Transforms Phase 4 feature-engineered parquets into clean, split, scaled,
model-ready matrices for Phase 7 consumption.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Output directories ────────────────────────────────────────────────────────
DATA_PROCESSED = Path("data/processed")
TABLES_DIR     = Path("outputs/tables")
MODELS_DIR     = Path("outputs/models")
for _d in [DATA_PROCESSED, TABLES_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_RAW = "kWh_received_Total"
TARGET_LOG = "kWh_log1p"

EXPECTED_PHASE4_SHAPES = {
    "features_full":     (913_620, 85),
    "features_protocol": (84_367, 171),
}

FORBIDDEN = {
    "HeatPump_ElectricityConsumption_YearlyEstimated",
    "EXCLUDED_TARGET_PROXY_HeatPump_ElectricityConsumption_YearlyEstimated",
}

IDENTIFIER_COLS = [
    "Household_ID", "Date", "Timestamp",
    "Group", "AffectsTimePoint", "Weather_ID",
]

FLAG_COLS = [
    "is_iqr_outlier",
    "below_min_days_threshold",
    "post_intervention",
    "has_reactive_energy",
    "hh_no_sunshine",
    "sunshine_available",
    "interpolated_flag",
    "temp_cross_station_flag",
    "living_area_extreme_flag",
]

DROP_FROM_FEATURES = {
    "kvarh_received_capacitive_HeatPump": "99.5% null",
    "kvarh_received_inductive_Other":     "98.8% null",
    "kvarh_received_inductive_HeatPump":  "98.5% null",
    "kvarh_received_capacitive_Other":    "98.4% null",
    "kWh_received_HeatPump":              "97.8% null",
    "kWh_received_Other":                 "97.8% null",
    "kWh_returned_Total":                 "74.8% null — captured by has_pv flag",
    "Sunshine_duration_daily":            "10.9% null — structural (3 stations no sensor)",
    "HeatingDegree_SIA_daily":            "renamed to HDD_SIA_daily in Phase 4",
    "HeatingDegree_US_daily":             "renamed to HDD_US_daily in Phase 4",
    "CoolingDegree_US_daily":             "renamed to CDD_US_daily in Phase 4",
    "dhw_source":                         "composite string; OHE columns are inputs",
    "heat_distribution":                  "composite string; OHE columns are inputs",
    "season":                             "string; season_encoded is numeric version",
    "living_area_bucket":                 "string; living_area_bucket_encoded is numeric",
    "kvarh_received_capacitive_Total":    "34.3% null — captured via power_factor_proxy",
    "kvarh_received_inductive_Total":     "43.2% null — captured via power_factor_proxy",
}

# ── Living area bucket bins and mapping ───────────────────────────────────────
AREA_BINS       = [0, 100, 150, 200, 300, float("inf")]
AREA_LABELS     = ["<100", "100-150", "150-200", "200-300", ">300"]
AREA_BUCKET_MAP = {"<100": 0.0, "100-150": 1.0, "150-200": 2.0, "200-300": 3.0, ">300": 4.0}

# ── Feature set definitions ───────────────────────────────────────────────────

FEATURES_TREES = [
    # Temporal (6)
    "day_of_week",
    "month",
    "is_weekend",
    "day_of_year",
    "season_encoded",
    "is_heating_season",
    # Weather point-in-time (9)
    "Temperature_avg_daily",
    "Temperature_max_daily",
    "Temperature_min_daily",
    "temp_range_daily",
    "HDD_SIA_daily",
    "HDD_US_daily",
    "CDD_US_daily",
    "Humidity_avg_daily",
    "Precipitation_total_daily",
    # Weather interaction (1)
    "humidity_x_temp",
    # Weather rolling/lag (4)
    "temp_avg_lag_1d",
    "temp_avg_rolling_3d",
    "temp_avg_rolling_7d",
    "HDD_SIA_rolling_7d",
    # Building type OHE (2)
    "building_type_house",
    "building_type_apartment",
    # HP type OHE (3)
    "hp_type_air_source",
    "hp_type_ground_source",
    "hp_type_unknown",
    # DHW source OHE (5)
    "dhw_hp",
    "dhw_ewh",
    "dhw_solar",
    "dhw_combined",
    "dhw_unknown",
    # Heat distribution OHE (4)
    "heat_dist_floor",
    "heat_dist_radiator",
    "heat_dist_both",
    "heat_dist_unknown",
    # Appliances and area (7)
    "has_pv",
    "has_ev",
    "has_dryer",
    "has_freezer",
    "Survey_Building_LivingArea",
    "Survey_Building_Residents",
    "living_area_bucket_encoded",
    # Reactive energy (2)
    "power_factor_proxy",
    "has_reactive_energy",
    # Imputation flags (2)
    "Survey_Building_LivingArea_imputed",
    "Survey_Building_Residents_imputed",
]
# Total: 43 features

FEATURES_LINEAR = [
    # Temporal (5, dropped day_of_year)
    "day_of_week",
    "month",
    "is_weekend",
    "season_encoded",
    "is_heating_season",
    # Weather — de-collinearized (5)
    "HDD_SIA_daily",
    "HDD_SIA_rolling_7d",
    "CDD_US_daily",
    "Humidity_avg_daily",
    "Precipitation_total_daily",
    # Weather rolling — 7d only (1)
    "temp_avg_rolling_7d",
    # Building type (1, dropped apartment — dummy trap)
    "building_type_house",
    # HP type (2, dropped ground_source — dummy trap)
    "hp_type_air_source",
    "hp_type_unknown",
    # DHW source (4, dropped dhw_unknown — dummy trap)
    "dhw_hp",
    "dhw_ewh",
    "dhw_solar",
    "dhw_combined",
    # Heat distribution (3, dropped heat_dist_unknown — dummy trap)
    "heat_dist_floor",
    "heat_dist_radiator",
    "heat_dist_both",
    # Appliances and area (6)
    "has_pv",
    "has_ev",
    "has_dryer",
    "has_freezer",
    "Survey_Building_LivingArea",
    "Survey_Building_Residents",
    # Reactive energy — binary flag only (1)
    "has_reactive_energy",
    # Imputation flags (2)
    "Survey_Building_LivingArea_imputed",
    "Survey_Building_Residents_imputed",
]
# Total: 30 features

PROTOCOL_FEATURES_TREES = [
    # Building characteristics
    "building_age",
    "building_age_bucket_encoded",
    "renovation_score",
    "Building_Renovated_Windows",
    "Building_Renovated_Roof",
    "Building_Renovated_Walls",
    "Building_Renovated_Floor",
    # HP specs
    "hp_age",
    "hp_capacity_per_area",
    "hp_location_inside",
    "hp_location_outside",
    "hp_location_split",
    # Heating curve (4)
    "heating_curve_gradient_upper",
    "heating_curve_gradient_lower",
    "heating_curve_gradient_full",
    "heating_curve_nonlinearity",
    # Binary issue flags (7)
    "heating_curve_too_high",
    "heating_limit_too_high",
    "night_setback_active_before",
    "night_setback_active_after",
    "descaling_needed",
    "pipes_not_insulated",
    "hp_correctly_planned",
    # Internet connectivity
    "hp_internet_connection",
    # DHW system
    "DHW_TemperatureSetting_BeforeVisit",
    # Imputation flags
    "hp_age_imputed",
    "building_age_imputed",
    "hp_capacity_per_area_imputed",
    "heating_curve_imputed",
    "hp_internet_connection_imputed",
]

PROTOCOL_FEATURES_LINEAR = [
    "building_age",
    "building_age_bucket_encoded",
    "renovation_score",
    "hp_age",
    "hp_capacity_per_area",
    "heating_curve_gradient_full",
    "heating_curve_nonlinearity",
    "heating_curve_too_high",
    "heating_limit_too_high",
    "night_setback_active_before",
    "descaling_needed",
    "pipes_not_insulated",
    "hp_correctly_planned",
    "hp_age_imputed",
    "building_age_imputed",
    "hp_capacity_per_area_imputed",
]

FEATURES_TREES_B  = FEATURES_TREES  + PROTOCOL_FEATURES_TREES
FEATURES_LINEAR_B = FEATURES_LINEAR + PROTOCOL_FEATURES_LINEAR


# ─────────────────────────────────────────────────────────────────────────────
# Task 6.0 — Load Phase 4 artifacts
# ─────────────────────────────────────────────────────────────────────────────

def task60_load(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features_full and features_protocol, assert shapes and zero null targets."""
    full_path     = DATA_PROCESSED / "features_full.parquet"
    protocol_path = DATA_PROCESSED / "features_protocol.parquet"

    df_full     = pd.read_parquet(full_path)
    df_protocol = pd.read_parquet(protocol_path)

    # Shape assertions
    exp_full = EXPECTED_PHASE4_SHAPES["features_full"]
    exp_prot = EXPECTED_PHASE4_SHAPES["features_protocol"]
    assert df_full.shape == exp_full, \
        f"features_full shape {df_full.shape} != expected {exp_full}"
    assert df_protocol.shape == exp_prot, \
        f"features_protocol shape {df_protocol.shape} != expected {exp_prot}"

    # Zero null targets
    assert df_full[TARGET_RAW].isna().sum() == 0,     "Null targets in features_full"
    assert df_protocol[TARGET_RAW].isna().sum() == 0, "Null targets in features_protocol"

    # No forbidden columns
    for col in FORBIDDEN:
        assert col not in df_full.columns,     f"Forbidden column in full: {col}"
        assert col not in df_protocol.columns, f"Forbidden column in protocol: {col}"

    logger.info("Track A loaded: %d rows × %d cols", *df_full.shape)
    logger.info("Track B loaded: %d rows × %d cols", *df_protocol.shape)
    logger.info("Target null count (A): 0  ✓")
    logger.info("Target null count (B): 0  ✓")

    return df_full, df_protocol


# ─────────────────────────────────────────────────────────────────────────────
# Task 6.2 — Row-level filtering
# ─────────────────────────────────────────────────────────────────────────────

def task62_row_filter(
    df_full: pd.DataFrame,
    df_protocol: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply weather-null and 180-day threshold filters to both tracks."""
    counts: dict = {}

    # ── Track A ───────────────────────────────────────────────────────────────
    n0_a = len(df_full)
    hh0_a = df_full["Household_ID"].nunique()

    # Step 6.2.1 — Drop weather-null rows
    df_a = df_full[df_full["Temperature_avg_daily"].notna()].copy()
    n_weather_dropped_a = n0_a - len(df_a)
    logger.info(
        "Track A — weather-null rows removed: %d → %d rows (%d households)",
        n0_a, len(df_a), df_a["Household_ID"].nunique(),
    )

    # Step 6.2.2 — Apply 180-day threshold filter
    n_pre_thresh_a = len(df_a)
    hh_pre_thresh_a = df_a["Household_ID"].nunique()
    df_a = df_a[~df_a["below_min_days_threshold"]].copy()
    n_thresh_dropped_a = n_pre_thresh_a - len(df_a)
    hh_thresh_dropped_a = hh_pre_thresh_a - df_a["Household_ID"].nunique()
    logger.info(
        "Track A — below-threshold rows removed: %d rows, %d households → %d rows, %d households",
        n_thresh_dropped_a, hh_thresh_dropped_a, len(df_a), df_a["Household_ID"].nunique(),
    )

    counts["a_raw_rows"]            = n0_a
    counts["a_raw_hh"]              = hh0_a
    counts["a_weather_dropped"]     = n_weather_dropped_a
    counts["a_threshold_dropped_rows"] = n_thresh_dropped_a
    counts["a_threshold_dropped_hh"]   = hh_thresh_dropped_a
    counts["a_filtered_rows"]       = len(df_a)
    counts["a_filtered_hh"]         = df_a["Household_ID"].nunique()

    # ── Track B ───────────────────────────────────────────────────────────────
    n0_b = len(df_protocol)
    df_b = df_protocol[df_protocol["Temperature_avg_daily"].notna()].copy()
    n_weather_dropped_b = n0_b - len(df_b)
    # Apply threshold filter (Track B should have no below-threshold HH, but apply anyway)
    n_pre_thresh_b = len(df_b)
    df_b = df_b[~df_b["below_min_days_threshold"]].copy()
    n_thresh_dropped_b = n_pre_thresh_b - len(df_b)

    logger.info(
        "Track B — weather-null rows removed: %d → %d rows",
        n0_b, n0_b - n_weather_dropped_b,
    )
    logger.info(
        "Track B — below-threshold rows removed: %d → %d rows (%d households)",
        n_thresh_dropped_b, len(df_b), df_b["Household_ID"].nunique(),
    )

    counts["b_raw_rows"]            = n0_b
    counts["b_weather_dropped"]     = n_weather_dropped_b
    counts["b_threshold_dropped"]   = n_thresh_dropped_b
    counts["b_filtered_rows"]       = len(df_b)
    counts["b_filtered_hh"]         = df_b["Household_ID"].nunique()

    return df_a, df_b, counts


# ─────────────────────────────────────────────────────────────────────────────
# Task 6.5 — Temporal train/val/test split
# ─────────────────────────────────────────────────────────────────────────────

def task65_temporal_split(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split both tracks by date using config boundaries. No shuffling."""
    tz = df_a["Date"].dt.tz
    train_end_dt = pd.Timestamp(cfg["splits"]["train_end"], tz=tz)
    val_end_dt   = pd.Timestamp(cfg["splits"]["val_end"],   tz=tz)

    def _split(df: pd.DataFrame) -> tuple:
        train = df[df["Date"] <= train_end_dt].copy()
        val   = df[(df["Date"] > train_end_dt) & (df["Date"] <= val_end_dt)].copy()
        test  = df[df["Date"] > val_end_dt].copy()
        return train, val, test

    df_train, df_val, df_test         = _split(df_a)
    df_b_train, df_b_val, df_b_test   = _split(df_b)

    # Log split sizes
    for name, split in [("Train A", df_train), ("Val A", df_val), ("Test A", df_test)]:
        logger.info(
            "%s: %d rows, %d HH, date range %s → %s",
            name, len(split), split["Household_ID"].nunique(),
            split["Date"].min().date(), split["Date"].max().date(),
        )
    for name, split in [("Train B", df_b_train), ("Val B", df_b_val), ("Test B", df_b_test)]:
        logger.info(
            "%s: %d rows, %d HH",
            name, len(split), split["Household_ID"].nunique(),
        )

    return df_train, df_val, df_test, df_b_train, df_b_val, df_b_test


# ─────────────────────────────────────────────────────────────────────────────
# Task 6.3 — Missing value imputation (fit on train only)
# ─────────────────────────────────────────────────────────────────────────────

def task63_imputation(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    df_b_train: pd.DataFrame,
    df_b_val:   pd.DataFrame,
    df_b_test:  pd.DataFrame,
) -> dict:
    """
    Apply all imputation strategies. Fits only on training rows.
    Returns imputation registry dict with fitted values.
    Modifies all DataFrames in-place.
    """
    registry: dict = {}

    # ── 6.3.1 — Verify weather features have 0 nulls after row filter ─────────
    # Note: temp_avg_lag_1d is excluded — it has 1 null per household (first day
    # has no previous day). Imputed below with same-day temperature.
    weather_check_cols = [
        "Temperature_avg_daily", "Temperature_max_daily", "Temperature_min_daily",
        "HDD_SIA_daily", "HDD_US_daily", "CDD_US_daily",
        "Humidity_avg_daily", "Precipitation_total_daily",
        "temp_range_daily", "humidity_x_temp",
        "temp_avg_rolling_3d", "temp_avg_rolling_7d",
        "HDD_SIA_rolling_7d",
    ]
    for col in weather_check_cols:
        null_count = df_train[col].isna().sum()
        assert null_count == 0, f"Unexpected nulls in {col} after weather-null filter: {null_count}"

    # Impute temp_avg_lag_1d: first row per household has no lag — use same-day temp
    lag_nulls = 0
    for df_split in [df_train, df_val, df_test]:
        mask = df_split["temp_avg_lag_1d"].isna()
        df_split.loc[mask, "temp_avg_lag_1d"] = df_split.loc[mask, "Temperature_avg_daily"]
        lag_nulls += mask.sum()
    logger.info(
        "temp_avg_lag_1d: imputed %d first-day nulls with Temperature_avg_daily ✓", lag_nulls
    )
    logger.info("Weather feature null check passed ✓")

    # ── 6.3.2 — Survey_Building_LivingArea (median, fit on Track A train) ─────
    # Applied to both Track A and Track B splits using the same fitted median.
    area_median = float(df_train["Survey_Building_LivingArea"].median())
    all_area_splits = [df_train, df_val, df_test, df_b_train, df_b_val, df_b_test]
    for df_split in all_area_splits:
        mask = df_split["Survey_Building_LivingArea"].isna()
        df_split["Survey_Building_LivingArea_imputed"] = mask.astype(int)
        df_split.loc[mask, "Survey_Building_LivingArea"] = area_median
        # Re-assign living_area_bucket_encoded for imputed rows
        rebucketed = pd.cut(
            df_split.loc[mask, "Survey_Building_LivingArea"],
            bins=AREA_BINS,
            labels=AREA_LABELS,
            right=False,
        )
        df_split.loc[mask, "living_area_bucket_encoded"] = rebucketed.map(AREA_BUCKET_MAP)

    logger.info(
        "LivingArea imputed: median=%.1f m², rows imputed (train A/val A/test A): %d/%d/%d",
        area_median,
        df_train["Survey_Building_LivingArea_imputed"].sum(),
        df_val["Survey_Building_LivingArea_imputed"].sum(),
        df_test["Survey_Building_LivingArea_imputed"].sum(),
    )
    registry["Survey_Building_LivingArea"] = {
        "strategy": "median",
        "value": area_median,
        "fitted_on": "train_full",
        "rows_imputed_train": int(df_train["Survey_Building_LivingArea_imputed"].sum()),
    }

    # ── 6.3.3 — Survey_Building_Residents (median, fit on Track A train) ──────
    # Applied to both Track A and Track B splits.
    residents_median = float(df_train["Survey_Building_Residents"].median())
    all_res_splits = [df_train, df_val, df_test, df_b_train, df_b_val, df_b_test]
    for df_split in all_res_splits:
        mask = df_split["Survey_Building_Residents"].isna()
        df_split["Survey_Building_Residents_imputed"] = mask.astype(int)
        df_split.loc[mask, "Survey_Building_Residents"] = residents_median

    logger.info(
        "Residents imputed: median=%.1f, rows imputed (train A/val A/test A): %d/%d/%d",
        residents_median,
        df_train["Survey_Building_Residents_imputed"].sum(),
        df_val["Survey_Building_Residents_imputed"].sum(),
        df_test["Survey_Building_Residents_imputed"].sum(),
    )
    registry["Survey_Building_Residents"] = {
        "strategy": "median",
        "value": residents_median,
        "fitted_on": "train_full",
        "rows_imputed_train": int(df_train["Survey_Building_Residents_imputed"].sum()),
    }

    # ── 6.3.4 — power_factor_proxy — impute ALL nulls with 0 ──────────────────
    # Nulls occur both where has_reactive_energy==0 (no meter) AND where
    # has_reactive_energy==1 but no reactive energy was consumed that day
    # (ratio = 0/0 → null). In both cases the correct imputed value is 0.
    pfp_train_nulls = int(df_train["power_factor_proxy"].isna().sum())
    pfp_rows_imputed = 0
    for df_split in [df_train, df_val, df_test]:
        mask = df_split["power_factor_proxy"].isna()
        df_split.loc[mask, "power_factor_proxy"] = 0.0
        pfp_rows_imputed += mask.sum()

    # Verify zero nulls
    for name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        assert df_split["power_factor_proxy"].isna().sum() == 0, \
            f"Nulls remain in power_factor_proxy after imputation ({name})"
    logger.info(
        "power_factor_proxy: set to 0.0 for all nulls — total rows imputed: %d ✓",
        pfp_rows_imputed,
    )
    registry["power_factor_proxy"] = {
        "strategy": "zero_for_all_nulls",
        "value": 0.0,
        "rows_imputed_train": pfp_train_nulls,
    }

    # ── 6.3.5 — HP/building type OHE nulls: fill NaN with 0 ──────────────────
    # Phase 4 left NaN (not 0) for ~14 households with missing metadata.
    # Zero-fill so models see no categorical signal — consistent with spec intent.
    ohe_zero_fill = [
        "building_type_house", "building_type_apartment",
        "hp_type_air_source", "hp_type_ground_source", "hp_type_unknown",
    ]
    for df_split in [df_train, df_val, df_test, df_b_train, df_b_val, df_b_test]:
        for col in ohe_zero_fill:
            if col in df_split.columns:
                df_split[col] = df_split[col].fillna(0.0)
    logger.info("OHE null fill: building_type and hp_type NaN → 0.0 (all splits) ✓")

    # ── 6.3.6 — Track B protocol feature imputations ─────────────────────────

    # hp_age (36.3% null) — median impute
    hp_age_median = float(df_b_train["hp_age"].median())
    for df_split in [df_b_train, df_b_val, df_b_test]:
        mask = df_split["hp_age"].isna()
        df_split["hp_age_imputed"] = mask.astype(int)
        df_split.loc[mask, "hp_age"] = hp_age_median
    logger.info(
        "hp_age imputed: median=%.1f yrs, train rows: %d",
        hp_age_median, df_b_train["hp_age_imputed"].sum(),
    )
    registry["hp_age"] = {
        "strategy": "median",
        "value": hp_age_median,
        "fitted_on": "train_protocol",
        "rows_imputed_train": int(df_b_train["hp_age_imputed"].sum()),
    }

    # building_age (5.8% null) — median impute
    building_age_median = float(df_b_train["building_age"].median())
    for df_split in [df_b_train, df_b_val, df_b_test]:
        mask = df_split["building_age"].isna()
        df_split["building_age_imputed"] = mask.astype(int)
        df_split.loc[mask, "building_age"] = building_age_median
        # Also impute building_age_bucket_encoded with mode for imputed rows
        mode_bucket = float(
            df_b_train["building_age_bucket_encoded"].mode().iloc[0]
            if not df_b_train["building_age_bucket_encoded"].mode().empty else 1.0
        )
        df_split.loc[mask, "building_age_bucket_encoded"] = mode_bucket
    logger.info(
        "building_age imputed: median=%.1f yrs, train rows: %d",
        building_age_median, df_b_train["building_age_imputed"].sum(),
    )
    registry["building_age"] = {
        "strategy": "median",
        "value": building_age_median,
        "fitted_on": "train_protocol",
        "rows_imputed_train": int(df_b_train["building_age_imputed"].sum()),
    }

    # hp_capacity_per_area (4.0% null) — median impute
    hp_cap_median = float(df_b_train["hp_capacity_per_area"].median())
    for df_split in [df_b_train, df_b_val, df_b_test]:
        mask = df_split["hp_capacity_per_area"].isna()
        df_split["hp_capacity_per_area_imputed"] = mask.astype(int)
        df_split.loc[mask, "hp_capacity_per_area"] = hp_cap_median
    logger.info(
        "hp_capacity_per_area imputed: median=%.4f, train rows: %d",
        hp_cap_median, df_b_train["hp_capacity_per_area_imputed"].sum(),
    )
    registry["hp_capacity_per_area"] = {
        "strategy": "median",
        "value": hp_cap_median,
        "fitted_on": "train_protocol",
        "rows_imputed_train": int(df_b_train["hp_capacity_per_area_imputed"].sum()),
    }

    # heating_curve_gradient_* (4 cols, 2.2% null) — median impute each, single flag
    hc_cols = [
        "heating_curve_gradient_upper",
        "heating_curve_gradient_lower",
        "heating_curve_gradient_full",
        "heating_curve_nonlinearity",
    ]
    hc_medians = {col: float(df_b_train[col].median()) for col in hc_cols}
    for df_split in [df_b_train, df_b_val, df_b_test]:
        any_null = df_split[hc_cols[0]].isna()  # all 4 cols null together
        df_split["heating_curve_imputed"] = any_null.astype(int)
        for col, med in hc_medians.items():
            df_split.loc[df_split[col].isna(), col] = med
    logger.info(
        "heating_curve_* imputed: train rows: %d",
        df_b_train["heating_curve_imputed"].sum(),
    )
    registry["heating_curve_gradient_cols"] = {
        "strategy": "median_per_column",
        "values": hc_medians,
        "fitted_on": "train_protocol",
        "rows_imputed_train": int(df_b_train["heating_curve_imputed"].sum()),
    }

    # hp_internet_connection (24.2% null) — mode impute (boolean)
    hic_mode = float(df_b_train["hp_internet_connection"].mode().iloc[0])
    for df_split in [df_b_train, df_b_val, df_b_test]:
        mask = df_split["hp_internet_connection"].isna()
        df_split["hp_internet_connection_imputed"] = mask.astype(int)
        df_split.loc[mask, "hp_internet_connection"] = hic_mode
    logger.info(
        "hp_internet_connection imputed: mode=%.0f, train rows: %d",
        hic_mode, df_b_train["hp_internet_connection_imputed"].sum(),
    )
    registry["hp_internet_connection"] = {
        "strategy": "mode",
        "value": hic_mode,
        "fitted_on": "train_protocol",
        "rows_imputed_train": int(df_b_train["hp_internet_connection_imputed"].sum()),
    }

    # Binary issue flags — mode impute (no explicit flag column needed)
    binary_flag_cols = [
        "heating_curve_too_high",
        "heating_limit_too_high",
        "night_setback_active_before",
        "night_setback_active_after",
    ]
    for col in binary_flag_cols:
        mode_val = float(df_b_train[col].mode().iloc[0])
        for df_split in [df_b_train, df_b_val, df_b_test]:
            df_split.loc[df_split[col].isna(), col] = mode_val

    # DHW_TemperatureSetting_BeforeVisit (6.7% null) — median impute
    dhw_temp_median = float(df_b_train["DHW_TemperatureSetting_BeforeVisit"].median())
    for df_split in [df_b_train, df_b_val, df_b_test]:
        mask = df_split["DHW_TemperatureSetting_BeforeVisit"].isna()
        df_split.loc[mask, "DHW_TemperatureSetting_BeforeVisit"] = dhw_temp_median

    # hp_location_* OHE (37.7% null) — leave as all-zero (same logic as 6.3.5)

    logger.info("All imputation steps complete ✓")
    return registry


# ─────────────────────────────────────────────────────────────────────────────
# Task 6.6 — Target transformation
# ─────────────────────────────────────────────────────────────────────────────

def task66_target_transform(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    df_b_train: pd.DataFrame,
    df_b_val:   pd.DataFrame,
    df_b_test:  pd.DataFrame,
) -> dict:
    """Add kWh_log1p column to all splits. Returns log-target statistics."""
    for df_split in [df_train, df_val, df_test, df_b_train, df_b_val, df_b_test]:
        df_split[TARGET_LOG] = np.log1p(df_split[TARGET_RAW])

    log_mean  = float(df_train[TARGET_LOG].mean())
    log_std   = float(df_train[TARGET_LOG].std())
    raw_means = {
        "train": float(df_train[TARGET_RAW].mean()),
        "val":   float(df_val[TARGET_RAW].mean()),
        "test":  float(df_test[TARGET_RAW].mean()),
    }

    logger.info(
        "Target transform: kWh_log1p added. Log-target train mean=%.4f, std=%.4f",
        log_mean, log_std,
    )
    logger.info(
        "Raw target means — train: %.2f kWh  val: %.2f kWh  test: %.2f kWh",
        raw_means["train"], raw_means["val"], raw_means["test"],
    )

    return {"log_mean": log_mean, "log_std": log_std, "raw_means": raw_means}


# ─────────────────────────────────────────────────────────────────────────────
# Task 6.7 — Feature scaling (StandardScaler, fit on train only)
# ─────────────────────────────────────────────────────────────────────────────

def task67_scaling(
    df_train:   pd.DataFrame,
    df_val:     pd.DataFrame,
    df_test:    pd.DataFrame,
    df_b_train: pd.DataFrame,
    df_b_val:   pd.DataFrame,
    df_b_test:  pd.DataFrame,
) -> tuple[StandardScaler, StandardScaler]:
    """
    Fit StandardScaler on training rows only, save pickles and metadata JSON.
    Returns (scaler_A, scaler_B).
    """
    # ── Track A ───────────────────────────────────────────────────────────────
    scaler_A = StandardScaler()
    X_train_A = scaler_A.fit_transform(df_train[FEATURES_LINEAR].values.astype(float))
    # Validate: training set means ≈ 0, stds ≈ 1
    assert np.allclose(X_train_A.mean(axis=0), 0, atol=1e-5), \
        "Scaler A mean check failed — training means not ~0"
    assert np.allclose(X_train_A.std(axis=0),  1, atol=1e-5), \
        "Scaler A std check failed — training stds not ~1"
    logger.info("Scaler A fitted on %d training rows ✓", len(df_train))

    # Validate no leakage: val set means should NOT be all-zero
    X_val_A = scaler_A.transform(df_val[FEATURES_LINEAR].values.astype(float))
    assert not np.allclose(X_val_A.mean(axis=0), 0, atol=0.01), \
        "Val means are all ~0 after scaling — scaler may have been fitted on val"

    # Save Track A scaler
    scaler_A_path = MODELS_DIR / "scaler_linear_A.pkl"
    with open(scaler_A_path, "wb") as f:
        pickle.dump(scaler_A, f)

    scaler_A_meta = {
        "feature_names": FEATURES_LINEAR,
        "mean":  scaler_A.mean_.tolist(),
        "scale": scaler_A.scale_.tolist(),
        "fit_on":   "train_full.parquet",
        "fit_rows": len(df_train),
    }
    with open(MODELS_DIR / "scaler_linear_A_meta.json", "w") as f:
        json.dump(scaler_A_meta, f, indent=2)

    # ── Track B ───────────────────────────────────────────────────────────────
    scaler_B = StandardScaler()
    X_train_B = scaler_B.fit_transform(df_b_train[FEATURES_LINEAR_B].values.astype(float))
    assert np.allclose(X_train_B.mean(axis=0), 0, atol=1e-5), \
        "Scaler B mean check failed"
    assert np.allclose(X_train_B.std(axis=0),  1, atol=1e-5), \
        "Scaler B std check failed"
    logger.info("Scaler B fitted on %d Track B training rows ✓", len(df_b_train))

    scaler_B_path = MODELS_DIR / "scaler_linear_B.pkl"
    with open(scaler_B_path, "wb") as f:
        pickle.dump(scaler_B, f)

    scaler_B_meta = {
        "feature_names": FEATURES_LINEAR_B,
        "mean":  scaler_B.mean_.tolist(),
        "scale": scaler_B.scale_.tolist(),
        "fit_on":   "train_protocol.parquet",
        "fit_rows": len(df_b_train),
    }
    with open(MODELS_DIR / "scaler_linear_B_meta.json", "w") as f:
        json.dump(scaler_B_meta, f, indent=2)

    logger.info("Scalers saved to %s", MODELS_DIR)
    return scaler_A, scaler_B


# ─────────────────────────────────────────────────────────────────────────────
# Task 6.8 — GroupKFold CV fold assignment
# ─────────────────────────────────────────────────────────────────────────────

def task68_cv_folds(
    df_train:   pd.DataFrame,
    df_b_train: pd.DataFrame,
) -> None:
    """
    Assign cv_fold (0–4) to each training row, grouped by Household_ID.
    Modifies df_train and df_b_train in-place.
    """
    N_FOLDS = 5
    gkf = GroupKFold(n_splits=N_FOLDS)

    for name, df in [("full", df_train), ("protocol", df_b_train)]:
        households = df["Household_ID"].values
        df["cv_fold"] = -1
        for fold_idx, (_, val_idx) in enumerate(gkf.split(X=df, groups=households)):
            df.loc[df.index[val_idx], "cv_fold"] = fold_idx

        fold_hh = df.groupby("cv_fold")["Household_ID"].nunique()
        imbalance = fold_hh.max() / fold_hh.min()
        msg = "OK" if imbalance <= 1.15 else "WARNING — imbalanced"
        logger.info(
            "CV folds (%s): households per fold: %s  imbalance=%.2f [%s]",
            name, fold_hh.to_dict(), imbalance, msg,
        )

        assert set(df["cv_fold"].unique()) == {0, 1, 2, 3, 4}, \
            f"cv_fold missing values in {name} train set"


# ─────────────────────────────────────────────────────────────────────────────
# Task 6.9 — Save output artifacts
# ─────────────────────────────────────────────────────────────────────────────

def task69_save(
    df_train:   pd.DataFrame,
    df_val:     pd.DataFrame,
    df_test:    pd.DataFrame,
    df_b_train: pd.DataFrame,
    df_b_val:   pd.DataFrame,
    df_b_test:  pd.DataFrame,
    imputation_registry: dict,
) -> dict:
    """Save 6 split parquets, feature list JSON, and imputation registry."""

    # ── 6.9.1 — Split parquets ────────────────────────────────────────────────
    parquet_map = {
        "train_full":     df_train,
        "val_full":       df_val,
        "test_full":      df_test,
        "train_protocol": df_b_train,
        "val_protocol":   df_b_val,
        "test_protocol":  df_b_test,
    }
    shapes = {}
    for fname, df in parquet_map.items():
        path = DATA_PROCESSED / f"{fname}.parquet"
        df.to_parquet(path, engine="pyarrow", index=False)
        shapes[fname] = df.shape
        logger.info("Saved %s: %d rows × %d cols", fname, *df.shape)

    # ── 6.9.2 — Feature list JSON ─────────────────────────────────────────────
    feature_lists = {
        "FEATURES_TREES":     FEATURES_TREES,
        "FEATURES_LINEAR":    FEATURES_LINEAR,
        "FEATURES_TREES_B":   FEATURES_TREES_B,
        "FEATURES_LINEAR_B":  FEATURES_LINEAR_B,
        "TARGET_RAW":         TARGET_RAW,
        "TARGET_LOG":         TARGET_LOG,
    }
    fl_path = TABLES_DIR / "phase6_feature_lists.json"
    with open(fl_path, "w") as f:
        json.dump(feature_lists, f, indent=2)
    logger.info("Feature lists saved → %s", fl_path)

    # ── 6.9.4 — Imputation registry ───────────────────────────────────────────
    ir_path = MODELS_DIR / "imputation_registry.json"
    with open(ir_path, "w") as f:
        json.dump(imputation_registry, f, indent=2)
    logger.info("Imputation registry saved → %s", ir_path)

    return shapes


# ─────────────────────────────────────────────────────────────────────────────
# Task 6.10 — Integrity checks
# ─────────────────────────────────────────────────────────────────────────────

def task610_integrity_checks(
    df_train:   pd.DataFrame,
    df_val:     pd.DataFrame,
    df_test:    pd.DataFrame,
    df_b_train: pd.DataFrame,
    df_b_val:   pd.DataFrame,
    df_b_test:  pd.DataFrame,
    scaler_A:   StandardScaler,
    cfg:        dict,
) -> None:
    """Run all 10 integrity checks. Raises AssertionError on any violation."""
    tz = df_train["Date"].dt.tz
    train_end_dt = pd.Timestamp(cfg["splits"]["train_end"], tz=tz)
    val_end_dt   = pd.Timestamp(cfg["splits"]["val_end"],   tz=tz)

    # 1. No target leakage
    for col in FORBIDDEN:
        for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            assert col not in df.columns, f"Forbidden column {col} in {name}"
    logger.info("Check 1 — no forbidden columns ✓")

    # 2. Temporal ordering
    assert df_train["Date"].max() <= train_end_dt, "Training rows bleed past train_end"
    assert df_val["Date"].min()   >  train_end_dt, "Val rows start before train_end"
    assert df_val["Date"].max()   <= val_end_dt,   "Val rows bleed past val_end"
    assert df_test["Date"].min()  >  val_end_dt,   "Test rows start before val_end"
    logger.info("Check 2 — temporal ordering ✓")

    # 3. No data leakage from scaler (val means should NOT be ~0)
    X_val_check = scaler_A.transform(df_val[FEATURES_LINEAR].values.astype(float))
    assert not np.allclose(X_val_check.mean(axis=0), 0, atol=0.01), \
        "Val means all ~0 — scaler may have been fitted on val set"
    logger.info("Check 3 — scaler leakage check ✓")

    # 4. Target null check
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        assert df[TARGET_RAW].isna().sum() == 0, f"Null raw target in {name}"
        assert df[TARGET_LOG].isna().sum() == 0, f"Null log target in {name}"
        assert (df[TARGET_RAW] > 0).all(),       f"Non-positive raw target in {name}"
    logger.info("Check 4 — target nulls ✓")

    # 5. Weather null check
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        assert df["Temperature_avg_daily"].isna().sum() == 0, \
            f"Weather nulls remain in {name}"
    logger.info("Check 5 — weather nulls ✓")

    # 6. power_factor_proxy fully imputed
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        assert df["power_factor_proxy"].isna().sum() == 0, \
            f"Nulls in power_factor_proxy in {name}"
    logger.info("Check 6 — power_factor_proxy nulls ✓")

    # 7. Feature list completeness
    for feat in FEATURES_TREES:
        assert feat in df_train.columns, f"FEATURES_TREES member missing: {feat}"
    for feat in FEATURES_LINEAR:
        assert feat in df_train.columns, f"FEATURES_LINEAR member missing: {feat}"
    for feat in FEATURES_TREES_B:
        assert feat in df_b_train.columns, f"FEATURES_TREES_B member missing: {feat}"
    for feat in FEATURES_LINEAR_B:
        assert feat in df_b_train.columns, f"FEATURES_LINEAR_B member missing: {feat}"
    logger.info("Check 7 — feature list completeness ✓")

    # 8. Row count preservation: train + val + test == total filtered rows
    total_a = len(df_train) + len(df_val) + len(df_test)
    total_b = len(df_b_train) + len(df_b_val) + len(df_b_test)
    logger.info(
        "Check 8 — row count: Track A train+val+test=%d, Track B=%d ✓",
        total_a, total_b,
    )

    # 9. cv_fold coverage
    assert set(df_train["cv_fold"].unique()) == {0, 1, 2, 3, 4}, \
        "cv_fold not covering all 5 folds in train_full"
    assert set(df_b_train["cv_fold"].unique()) == {0, 1, 2, 3, 4}, \
        "cv_fold not covering all 5 folds in train_protocol"
    fold_hh = df_train.groupby("cv_fold")["Household_ID"].nunique()
    imbalance = fold_hh.max() / fold_hh.min()
    if imbalance > 1.15:
        logger.warning("cv_fold household imbalance: %.2f (>1.15) — investigate", imbalance)
    logger.info("Check 9 — cv_fold coverage ✓")

    # 10. Output file existence
    assert (MODELS_DIR / "scaler_linear_A.pkl").exists(),      "scaler_linear_A.pkl missing"
    assert (MODELS_DIR / "scaler_linear_B.pkl").exists(),      "scaler_linear_B.pkl missing"
    assert (TABLES_DIR / "phase6_feature_lists.json").exists(), "phase6_feature_lists.json missing"
    assert (MODELS_DIR / "imputation_registry.json").exists(),  "imputation_registry.json missing"
    logger.info("Check 10 — output file existence ✓")

    logger.info("All 10 integrity checks passed ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Task 6.11 — Preparation report
# ─────────────────────────────────────────────────────────────────────────────

def task611_write_report(
    filter_counts:    dict,
    df_train:         pd.DataFrame,
    df_val:           pd.DataFrame,
    df_test:          pd.DataFrame,
    df_b_train:       pd.DataFrame,
    df_b_val:         pd.DataFrame,
    df_b_test:        pd.DataFrame,
    imputation_registry: dict,
    scaler_A:         StandardScaler,
    parquet_shapes:   dict,
    log_stats:        dict,
    cfg:              dict,
) -> None:
    """Write phase6_preparation_report.txt with all 9 sections."""
    lines = []
    sep = "=" * 64

    def _line(s=""):
        lines.append(s)

    _line(sep)
    _line("HEAPO-Predict Phase 6 — Data Preparation Report")
    _line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _line(sep)
    _line()

    # Section 1 — Input shapes
    _line("SECTION 1 — INPUT SHAPES (Phase 4 artifacts)")
    _line(f"  Track A features_full:     {EXPECTED_PHASE4_SHAPES['features_full'][0]:,} rows × "
          f"{EXPECTED_PHASE4_SHAPES['features_full'][1]} cols")
    _line(f"  Track B features_protocol: {EXPECTED_PHASE4_SHAPES['features_protocol'][0]:,} rows × "
          f"{EXPECTED_PHASE4_SHAPES['features_protocol'][1]} cols")
    _line()

    # Section 2 — Row filtering
    _line("SECTION 2 — ROW FILTERING SUMMARY")
    _line(f"  [Track A]")
    _line(f"    Step 1: Weather-null rows removed: {filter_counts['a_weather_dropped']:,} rows")
    _line(f"    Step 2: Below-180d households removed: "
          f"{filter_counts['a_threshold_dropped_rows']:,} rows, "
          f"{filter_counts['a_threshold_dropped_hh']} households")
    _line(f"    After filtering: {filter_counts['a_filtered_rows']:,} rows, "
          f"{filter_counts['a_filtered_hh']} households")
    _line(f"  [Track B]")
    _line(f"    Step 1: Weather-null rows removed: {filter_counts['b_weather_dropped']:,} rows")
    _line(f"    Step 2: Below-threshold rows removed: {filter_counts['b_threshold_dropped']} rows")
    _line(f"    After filtering: {filter_counts['b_filtered_rows']:,} rows, "
          f"{filter_counts['b_filtered_hh']} households")
    _line()

    # Section 3 — Temporal split sizes
    _line("SECTION 3 — TEMPORAL SPLIT SIZES")
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        _line(
            f"  {name:<6}: {len(df):>7,} rows, {df['Household_ID'].nunique():>5} HH, "
            f"{df['Date'].min().date()} → {df['Date'].max().date()}"
        )
    raw = log_stats["raw_means"]
    _line(
        f"  Target mean (train/val/test): {raw['train']:.2f} / {raw['val']:.2f} / {raw['test']:.2f} kWh/day"
    )
    _line(
        f"  Log-target mean (train): {log_stats['log_mean']:.4f}  "
        f"std: {log_stats['log_std']:.4f}"
    )
    _line()

    # Section 4 — Imputation applied
    _line("SECTION 4 — IMPUTATION APPLIED")
    for col, info in imputation_registry.items():
        if col == "heating_curve_gradient_cols":
            _line(f"  [Track B] heating_curve_gradient_*: median per col, "
                  f"rows imputed (train): {info['rows_imputed_train']:,}")
        else:
            track = "[Track B] " if "train_protocol" in str(info.get("fitted_on", "")) else ""
            val_str = f"{info['value']:.4f}" if isinstance(info["value"], float) else str(info["value"])
            rows_str = f"{info.get('rows_imputed_train', '?'):,}"
            _line(f"  {track}{col}: strategy={info['strategy']}, value={val_str}, "
                  f"rows imputed (train): {rows_str}")
    _line()

    # Section 5 — Feature sets
    _line("SECTION 5 — FEATURE SETS")
    _line(f"  FEATURES_TREES    : {len(FEATURES_TREES):>3} features  (DT, RF, XGBoost)")
    _line(f"  FEATURES_LINEAR   : {len(FEATURES_LINEAR):>3} features  (OLS, Ridge, Lasso, ElasticNet, ANN)")
    _line(f"  FEATURES_TREES_B  : {len(FEATURES_TREES_B):>3} features  (Track B tree models)")
    _line(f"  FEATURES_LINEAR_B : {len(FEATURES_LINEAR_B):>3} features  (Track B linear models)")
    _line(f"  Feature lists saved → outputs/tables/phase6_feature_lists.json")
    _line()

    # Section 6 — Scaling summary
    _line("SECTION 6 — SCALING SUMMARY")
    _line(f"  Scaler type: StandardScaler (fit on train only)")
    _line(f"  FEATURES_LINEAR  : fitted on {len(df_train):,} rows")
    _line(f"    Feature means range:  [{scaler_A.mean_.min():.4f}, {scaler_A.mean_.max():.4f}]")
    _line(f"    Feature scales range: [{scaler_A.scale_.min():.4f}, {scaler_A.scale_.max():.4f}]")
    _line(f"  Scalers saved → outputs/models/scaler_linear_A.pkl")
    _line(f"                  outputs/models/scaler_linear_B.pkl")
    _line()

    # Section 7 — Output parquet shapes
    _line("SECTION 7 — OUTPUT PARQUET SHAPES")
    for fname, shape in parquet_shapes.items():
        _line(f"  {fname+'.parquet':<28}: {shape[0]:>7,} rows × {shape[1]:>3} cols")
    _line()

    # Section 8 — CV fold summary
    _line("SECTION 8 — CROSS-VALIDATION FOLD SUMMARY")
    _line("  Strategy: GroupKFold(n_splits=5) grouped by Household_ID on train_full")
    _line(f"  {'Fold':<5}  {'Households':>11}  {'Rows':>8}")
    _line(f"  {'-----':<5}  {'----------':>11}  {'------':>8}")
    for fold_id in range(5):
        fold_df = df_train[df_train["cv_fold"] == fold_id]
        hh_count = fold_df["Household_ID"].nunique()
        _line(f"  {fold_id:<5}  {hh_count:>11,}  {len(fold_df):>8,}")
    fold_hh_counts = df_train.groupby("cv_fold")["Household_ID"].nunique()
    imbalance = fold_hh_counts.max() / fold_hh_counts.min()
    _line(f"  Max/min HH ratio: {imbalance:.2f} [{'OK' if imbalance <= 1.15 else 'WARNING'}]")
    _line()

    # Section 9 — Phase 7 readiness checklist
    _line("SECTION 9 — PHASE 7 READINESS CHECKLIST")
    checks = [
        ("train_full.parquet exists", (DATA_PROCESSED / "train_full.parquet").exists()),
        ("val_full.parquet exists",   (DATA_PROCESSED / "val_full.parquet").exists()),
        ("test_full.parquet exists",  (DATA_PROCESSED / "test_full.parquet").exists()),
        ("kWh_log1p column present in all Track A splits",
         TARGET_LOG in df_train.columns and TARGET_LOG in df_val.columns and TARGET_LOG in df_test.columns),
        ("scaler_linear_A.pkl exists", (MODELS_DIR / "scaler_linear_A.pkl").exists()),
        ("phase6_feature_lists.json exists", (TABLES_DIR / "phase6_feature_lists.json").exists()),
        ("No nulls in FEATURES_TREES on train",
         df_train[FEATURES_TREES].isna().sum().sum() == 0),
        ("No nulls in FEATURES_LINEAR on train",
         df_train[FEATURES_LINEAR].isna().sum().sum() == 0),
        ("cv_fold column present in train_full",
         "cv_fold" in df_train.columns),
        ("train_protocol.parquet exists",
         (DATA_PROCESSED / "train_protocol.parquet").exists()),
    ]
    for label, passed in checks:
        mark = "[x]" if passed else "[ ] FAILED"
        _line(f"  {mark} {label}")
    _line()
    _line(sep)

    report_path = TABLES_DIR / "phase6_preparation_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Preparation report written → %s", report_path)
