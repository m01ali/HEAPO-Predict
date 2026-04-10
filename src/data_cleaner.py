"""
src/data_cleaner.py

Data cleaning utilities for the HEAPO dataset.
Phase 2 implementation — Tasks 2.1 through 2.4.

All cleaning decisions are grounded in the Phase 2 spec:
  spec/Phase-2-Data-Cleaning.md

Column names verified against:
  - Table 1 (SMD), Table 4 (Metadata), Table 5 (Protocols), Table 6 (Weather)
  in Brudermueller et al. (2025), arXiv:2503.16993v1
  and against actual parquet files produced by Phase 1.
"""

import logging
from io import StringIO
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Metadata binary columns stored as object dtype — confirmed in Phase 1 profiling
_BINARY_COLS_METADATA = [
    "Survey_HeatDistribution_System_FloorHeating",
    "Survey_HeatDistribution_System_Radiator",
    "Survey_DHW_Production_ByHeatPump",
    "Survey_DHW_Production_ByElectricWaterHeater",
    "Survey_DHW_Production_BySolar",
    "Survey_Installation_HasDryer",
    "Survey_Installation_HasFreezer",
    "Survey_Installation_HasElectricVehicle",
]

# Metadata checkbox columns: only True + NaN present (NaN = "not applicable", not missing)
_CHECKBOX_COLS_METADATA = [
    "Survey_DHW_Production_ByHeatPump",
    "Survey_DHW_Production_ByElectricWaterHeater",
    "Survey_DHW_Production_BySolar",
    "Survey_Installation_HasDryer",
    "Survey_Installation_HasFreezer",
    "Survey_Installation_HasElectricVehicle",
]

# Protocol columns confirmed as object dtype with boolean content — Phase 1 profiling
_PROTOCOL_OBJECT_BOOL_COLS = [
    # Building
    "Building_PVSystem_Available",
    "Building_ElectricVehicle_Available",        # CHECKBOX — filled NaN→False in step B
    "Building_FloorAreaHeated_AdditionalAreasPlanned",  # genuine binary; NaN kept
    # Heat pump installation
    "HeatPump_Installation_InternetConnection",
    # Heat distribution
    "HeatDistribution_System_Radiators",
    "HeatDistribution_System_FloorHeating",
    "HeatDistribution_System_ThermostaticValve",
    "HeatDistribution_System_BufferTankAvailable",
    # DHW
    "DHW_Production_ByHeatPumpBoiler",
    "DHW_Sterilization_Available",
    "DHW_Sterilization_Active",
    "DHW_TemperatureSetting_Changed",
    # HP condition / assessment
    "HeatPump_Clean",
    "HeatPump_BasicFunctionsOkay",
    "HeatPump_TechnicallyOkay",
    "HeatPump_Installation_CorrectlyPlanned",
    # Air-source specific
    "HeatPump_AirSource_AirDuctsDistanceOkay",
    "HeatPump_AirSource_AirDuctsFree",
    "HeatPump_AirSource_AirDuctsCleaningRequired",
    "HeatPump_AirSource_AirDuctsDrainOkay",
    "HeatPump_AirSource_EvaporatorClean",
    # Ground-source specific
    "HeatPump_GroundSource_BrineCircuit_AntiFreezeExists",
    "HeatPump_GroundSource_CurrentPressure_Okay",
    "HeatPump_GroundSource_CurrentTemperature_Okay",
    # Settings flags
    "HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit",
    "HeatPump_HeatingCurveSetting_Changed",
    "HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit",
    "HeatPump_HeatingLimitSetting_Changed",
    "HeatPump_NightSetbackSetting_Activated_BeforeVisit",
    "HeatPump_NightSetbackSetting_Activated_AfterVisit",
    # Circulation
    "HeatDistribution_Circulation_PumpStagePosition_Changed",
]

# Heating curve columns for sanity check
_HEATING_CURVE_COLS = [
    "HeatPump_HeatingCurveSetting_Outside20_BeforeVisit",
    "HeatPump_HeatingCurveSetting_Outside0_BeforeVisit",
    "HeatPump_HeatingCurveSetting_OutsideMinus8_BeforeVisit",
]

# Weather columns safe to linearly interpolate (gap transmission issues, NOT structural absence)
_INTERPOLATABLE_WEATHER_COLS = [
    "Temperature_max_daily",
    "Temperature_min_daily",
    "Temperature_avg_daily",
    "HeatingDegree_SIA_daily",
    "HeatingDegree_US_daily",
    "CoolingDegree_US_daily",
    "Humidity_avg_daily",
    "Precipitation_total_daily",
    # NOTE: Sunshine_duration_daily intentionally excluded — structural absence at 3 stations
]


# ---------------------------------------------------------------------------
# Task 2.1 — SMD Cleaning
# ---------------------------------------------------------------------------

def clean_smd(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Full SMD cleaning pipeline (Tasks 2.1.1 – 2.1.10).

    Steps:
      1.  Timezone conversion UTC → Europe/Zurich
      2.  Date column extraction
      3.  Duplicate row removal
      4.  Null target handling (per-household threshold)
      5.  Non-positive target removal
      6.  Hard cap outlier removal (>500 kWh)
      7.  Per-household IQR outlier flagging (retained, not dropped)
      8.  Exclude 'during visit' rows
      9.  Create post_intervention feature
      10. Flag below-min-days households
      11. Add has_pv flag
      12. Add has_reactive_energy flag

    Returns:
        (cleaned_df, stats_dict) where stats_dict holds counts for the report.
    """
    stats: dict = {"input_rows": len(df), "input_households": df["Household_ID"].nunique()}
    cfg = config.get("cleaning", {})
    hard_cap = cfg.get("smd_hard_cap_kwh", 500)
    null_threshold = cfg.get("null_fraction_threshold", 0.30)
    iqr_mult = cfg.get("iqr_multiplier", 3.0)
    min_days = config.get("data", {}).get("min_days_threshold", 180)

    # ------------------------------------------------------------------
    # 2.1.1 — Timezone conversion UTC → Europe/Zurich
    # ------------------------------------------------------------------
    df = df.copy()
    df["Timestamp"] = df["Timestamp"].dt.tz_convert("Europe/Zurich")
    logger.info("Timestamp converted to Europe/Zurich")

    # ------------------------------------------------------------------
    # 2.1.2 — Date column + duplicate detection
    # ------------------------------------------------------------------
    df["Date"] = df["Timestamp"].dt.normalize()  # tz-aware midnight; use for joins

    n_exact = df.duplicated().sum()
    if n_exact > 0:
        df = df[~df.duplicated()]
        logger.warning(f"Removed {n_exact} exact duplicate rows")
    else:
        logger.info("Exact duplicates: 0 ✓")

    n_ts_dup = df.duplicated(subset=["Household_ID", "Date"]).sum()
    if n_ts_dup > 0:
        affected = (
            df[df.duplicated(subset=["Household_ID", "Date"], keep=False)]["Household_ID"]
            .nunique()
        )
        df = df[~df.duplicated(subset=["Household_ID", "Date"], keep="first")]
        logger.warning(
            f"Removed {n_ts_dup} duplicate (Household_ID, Date) rows "
            f"across {affected} households"
        )
    else:
        logger.info("Duplicate (Household_ID, Date) pairs: 0 ✓")

    stats["exact_duplicates_removed"] = int(n_exact)
    stats["date_duplicates_removed"] = int(n_ts_dup)

    # ------------------------------------------------------------------
    # 2.1.3 — Null kWh_received_Total handling (per-household threshold)
    # ------------------------------------------------------------------
    null_rows_total = 0
    households_dropped = []
    households_rows_dropped = []

    null_counts = df.groupby("Household_ID")["kWh_received_Total"].apply(
        lambda x: x.isnull().sum()
    )
    household_sizes = df.groupby("Household_ID").size()
    null_fractions = null_counts / household_sizes

    affected_hids = null_fractions[null_fractions > 0].index.tolist()

    for hid in affected_hids:
        null_count = int(null_counts[hid])
        null_frac = float(null_fractions[hid])
        total = int(household_sizes[hid])

        if null_frac < null_threshold:
            idx_to_drop = df.loc[
                (df["Household_ID"] == hid) & df["kWh_received_Total"].isnull()
            ].index
            df = df.drop(index=idx_to_drop)
            null_rows_total += null_count
            households_rows_dropped.append(hid)
            logger.info(
                f"  HH {hid}: dropped {null_count} null target rows "
                f"({null_frac:.1%} of {total}) — household retained"
            )
        else:
            df = df[df["Household_ID"] != hid]
            null_rows_total += null_count
            households_dropped.append(hid)
            logger.warning(
                f"  HH {hid}: DROPPED ENTIRELY — "
                f"{null_count}/{total} rows ({null_frac:.1%}) null target"
            )

    logger.info(
        f"Null target: removed {null_rows_total} rows; "
        f"{len(households_dropped)} households dropped entirely; "
        f"{len(households_rows_dropped)} households had partial rows removed"
    )
    stats["null_target_rows_removed"] = null_rows_total
    stats["households_dropped_entirely"] = len(households_dropped)
    stats["households_dropped_list"] = households_dropped

    # ------------------------------------------------------------------
    # 2.1.4 — Remove non-positive target values
    # ------------------------------------------------------------------
    n_nonpositive = int((df["kWh_received_Total"] <= 0).sum())
    if n_nonpositive > 0:
        df = df[df["kWh_received_Total"] > 0]
        logger.info(f"Removed {n_nonpositive} rows with kWh_received_Total <= 0")
    else:
        logger.info("Non-positive target rows: 0 ✓")
    stats["nonpositive_rows_removed"] = n_nonpositive

    # ------------------------------------------------------------------
    # 2.1.5A — Hard cap: remove rows > hard_cap kWh
    # ------------------------------------------------------------------
    hard_cap_mask = df["kWh_received_Total"] > hard_cap
    n_hard_cap = int(hard_cap_mask.sum())
    if n_hard_cap > 0:
        removed_vals = df.loc[hard_cap_mask, ["Household_ID", "Date", "kWh_received_Total"]]
        logger.warning(
            f"Hard cap (>{hard_cap} kWh): removing {n_hard_cap} rows:\n"
            f"{removed_vals.to_string(index=False)}"
        )
        df = df[~hard_cap_mask]
    else:
        logger.info(f"Hard cap (>{hard_cap} kWh): 0 rows removed ✓")
    stats["hard_cap_rows_removed"] = n_hard_cap

    # ------------------------------------------------------------------
    # 2.1.5B — Per-household IQR outlier flagging (flag, do NOT drop)
    # ------------------------------------------------------------------
    df["is_iqr_outlier"] = _flag_iqr_outliers(df, iqr_mult)
    n_iqr = int(df["is_iqr_outlier"].sum())
    logger.info(
        f"IQR outlier flag ({iqr_mult}×IQR): {n_iqr} rows flagged across "
        f"{df.loc[df['is_iqr_outlier'], 'Household_ID'].nunique()} households "
        f"(retained in dataset)"
    )
    stats["iqr_outlier_flags"] = n_iqr

    # ------------------------------------------------------------------
    # 2.1.6 — Exclude 'during visit' rows
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df[df["AffectsTimePoint"] != "during visit"]
    n_during = n_before - len(df)
    if n_during != 56:
        logger.warning(f"Excluded {n_during} 'during visit' rows (expected 56)")
    else:
        logger.info(f"Excluded {n_during} 'during visit' rows ✓")
    stats["during_visit_rows_removed"] = n_during

    # ------------------------------------------------------------------
    # 2.1.7 — Create post_intervention feature
    # ------------------------------------------------------------------
    df["post_intervention"] = (df["AffectsTimePoint"] == "after visit").astype(int)
    dist = df["post_intervention"].value_counts().to_dict()
    logger.info(f"post_intervention distribution: {dist}")
    stats["post_intervention_1_count"] = int(dist.get(1, 0))

    # ------------------------------------------------------------------
    # 2.1.8 — Flag below-min-days households
    # ------------------------------------------------------------------
    days_per_hh = df.groupby("Household_ID")["Date"].nunique()
    below_ids = set(days_per_hh[days_per_hh < min_days].index)
    df["below_min_days_threshold"] = df["Household_ID"].isin(below_ids)
    n_below_hh = len(below_ids)
    n_below_rows = int(df["below_min_days_threshold"].sum())
    logger.info(
        f"below_min_days_threshold (<{min_days} days): "
        f"{n_below_hh} households ({n_below_rows} rows) flagged — retained"
    )
    stats["below_threshold_households"] = n_below_hh

    # ------------------------------------------------------------------
    # 2.1.9 — has_pv proxy feature
    # ------------------------------------------------------------------
    pv_ids = set(
        df.loc[
            df["kWh_returned_Total"].notna() & (df["kWh_returned_Total"] > 0), "Household_ID"
        ].unique()
    )
    df["has_pv"] = df["Household_ID"].isin(pv_ids).astype(int)
    logger.info(
        f"has_pv: {len(pv_ids)} households identified as PV "
        f"(via kWh_returned_Total > 0 proxy)"
    )
    stats["pv_households"] = len(pv_ids)

    # ------------------------------------------------------------------
    # 2.1.10 — has_reactive_energy flag
    # ------------------------------------------------------------------
    df["has_reactive_energy"] = df["kvarh_received_capacitive_Total"].notna().astype(int)
    n_reactive = int(df["has_reactive_energy"].sum())
    logger.info(
        f"has_reactive_energy: {n_reactive} rows "
        f"({n_reactive / len(df):.1%}) have reactive energy data"
    )
    stats["output_rows"] = len(df)
    stats["output_households"] = df["Household_ID"].nunique()
    return df, stats


def _flag_iqr_outliers(df: pd.DataFrame, multiplier: float) -> pd.Series:
    """Per-household IQR outlier flag for kWh_received_Total. Returns boolean Series."""

    def _per_hh(group: pd.DataFrame) -> pd.Series:
        vals = group["kWh_received_Total"]
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        return (vals < q1 - multiplier * iqr) | (vals > q3 + multiplier * iqr)

    return df.groupby("Household_ID", group_keys=False).apply(_per_hh)


# ---------------------------------------------------------------------------
# Task 2.2 — Metadata Cleaning
# ---------------------------------------------------------------------------

def clean_metadata(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Metadata cleaning pipeline (Tasks 2.2.1 – 2.2.6).

    Steps:
      1. Fix 'appartment' typo → 'apartment'
      2. Coerce 8 binary object columns to pandas 'boolean' dtype
      3. Fill 6 checkbox columns NaN → False
      4. Flag living area > 1000 m²
      5. Check building residents range
      6. Fill HP installation type NaN → 'unknown'
    """
    stats: dict = {"input_rows": len(df)}
    cfg = config.get("cleaning", {})
    area_threshold = cfg.get("living_area_flag_threshold_m2", 1000)

    df = df.copy()

    # 2.2.1 — Fix 'appartment' typo
    n_typo = int((df["Survey_Building_Type"] == "appartment").sum())
    df["Survey_Building_Type"] = df["Survey_Building_Type"].replace({"appartment": "apartment"})
    logger.info(f"Fixed 'appartment' → 'apartment' typo: {n_typo} rows")
    stats["appartment_typo_fixed"] = n_typo

    # 2.2.2 — Coerce binary object columns to pandas nullable boolean
    for col in _BINARY_COLS_METADATA:
        if col in df.columns:
            df[col] = df[col].astype("boolean")
    logger.info(f"Coerced {len(_BINARY_COLS_METADATA)} metadata binary columns → 'boolean' dtype")

    # 2.2.3 — Checkbox columns: NaN → False
    # These columns have zero False values in the data — NaN means "not applicable"
    for col in _CHECKBOX_COLS_METADATA:
        if col in df.columns:
            n_filled = int(df[col].isna().sum())
            df[col] = df[col].fillna(False)
            logger.info(f"  {col}: {n_filled} NaN → False")
    stats["checkbox_nans_filled"] = {
        col: int(df[col].isna().sum()) for col in _CHECKBOX_COLS_METADATA if col in df.columns
    }  # should all be 0 after fill

    # 2.2.4 — Living area range check
    extreme_area = df[df["Survey_Building_LivingArea"].notna() &
                      (df["Survey_Building_LivingArea"] > area_threshold)]
    if len(extreme_area) > 0:
        logger.warning(
            f"Survey_Building_LivingArea: {len(extreme_area)} households "
            f"with area > {area_threshold} m²:\n"
            f"{extreme_area[['Household_ID', 'Survey_Building_LivingArea']].to_string(index=False)}"
        )
    df["living_area_extreme_flag"] = (
        df["Survey_Building_LivingArea"].notna() &
        (df["Survey_Building_LivingArea"] > area_threshold)
    )
    stats["living_area_extreme_flags"] = int(df["living_area_extreme_flag"].sum())

    # 2.2.5 — Building residents range check
    extreme_res = df[df["Survey_Building_Residents"].notna() &
                     (df["Survey_Building_Residents"] > 10)]
    if len(extreme_res) > 0:
        logger.warning(
            f"Survey_Building_Residents: {len(extreme_res)} households with > 10 residents"
        )
    else:
        logger.info("Survey_Building_Residents: all values ≤ 10 ✓")
    stats["residents_extreme_flags"] = int(len(extreme_res))

    # 2.2.6 — HP installation type: fill NaN → 'unknown'
    n_hp_nan = int(df["Survey_HeatPump_Installation_Type"].isna().sum())
    df["Survey_HeatPump_Installation_Type"] = (
        df["Survey_HeatPump_Installation_Type"].fillna("unknown")
    )
    logger.info(f"Survey_HeatPump_Installation_Type: {n_hp_nan} NaN → 'unknown'")
    stats["hp_type_unknown_filled"] = n_hp_nan
    stats["output_rows"] = len(df)
    return df, stats


# ---------------------------------------------------------------------------
# Task 2.3 — Protocol Cleaning
# ---------------------------------------------------------------------------

def clean_protocols(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Protocol data cleaning pipeline (Tasks 2.3.1 – 2.3.8).

    Steps:
      1. Rename target proxy column (data leakage prevention)
      2. Building construction year range check
      3. HP installation year range check
      4. Heating curve temperature sanity + monotonicity check
      5. Ground-source structural missingness note
      6. Coerce object boolean columns + EV checkbox fill
      7. HP capacity and COP range checks
      8. HeatingPower string → numeric conversion
    """
    stats: dict = {"input_rows": len(df)}
    cfg = config.get("cleaning", {})

    df = df.copy()

    # 2.3.1 — Rename target proxy column (data leakage)
    target_proxy_col = "HeatPump_ElectricityConsumption_YearlyEstimated"
    renamed_col = f"EXCLUDED_TARGET_PROXY_{target_proxy_col}"
    if target_proxy_col in df.columns:
        df = df.rename(columns={target_proxy_col: renamed_col})
        logger.warning(
            f"LEAKAGE PREVENTION: '{target_proxy_col}' renamed to '{renamed_col}' — "
            f"this column is a direct proxy for the annual target variable and must "
            f"NEVER be used as a feature."
        )
        stats["target_proxy_renamed"] = True
    else:
        logger.info(f"Target proxy column '{target_proxy_col}' not found (already renamed?)")
        stats["target_proxy_renamed"] = False

    # 2.3.2 — Building construction year range check
    year_min = cfg.get("construction_year_min", 1900)
    year_max = cfg.get("construction_year_max", 2025)
    invalid_cy = df[
        df["Building_ConstructionYear"].notna()
        & (
            (df["Building_ConstructionYear"] < year_min)
            | (df["Building_ConstructionYear"] > year_max)
        )
    ]
    if len(invalid_cy) > 0:
        logger.warning(
            f"Building_ConstructionYear: {len(invalid_cy)} implausible values "
            f"(outside [{year_min}, {year_max}]) → set to NaN:\n"
            f"{invalid_cy[['Report_ID', 'Building_ConstructionYear']].to_string(index=False)}"
        )
        df.loc[invalid_cy.index, "Building_ConstructionYear"] = np.nan
    else:
        logger.info(f"Building_ConstructionYear: all values in [{year_min}, {year_max}] ✓")
    stats["construction_year_errors"] = len(invalid_cy)

    # 2.3.3 — HP installation year range check
    hp_yr_min = cfg.get("hp_install_year_min", 1980)
    hp_yr_max = cfg.get("hp_install_year_max", 2025)
    invalid_hp_yr = df[
        df["HeatPump_Installation_Year"].notna()
        & (
            (df["HeatPump_Installation_Year"] < hp_yr_min)
            | (df["HeatPump_Installation_Year"] > hp_yr_max)
        )
    ]
    if len(invalid_hp_yr) > 0:
        logger.warning(
            f"HeatPump_Installation_Year: {len(invalid_hp_yr)} implausible values → NaN"
        )
        df.loc[invalid_hp_yr.index, "HeatPump_Installation_Year"] = np.nan
    else:
        logger.info(f"HeatPump_Installation_Year: all values in [{hp_yr_min}, {hp_yr_max}] ✓")
    stats["hp_install_year_errors"] = len(invalid_hp_yr)

    # 2.3.4 — Heating curve temperature sanity + monotonicity
    hc_min = cfg.get("heating_curve_temp_min", 15)
    hc_max = cfg.get("heating_curve_temp_max", 70)
    total_range_flags = 0
    for col in _HEATING_CURVE_COLS:
        if col not in df.columns:
            continue
        invalid_hc = df[
            df[col].notna() & ((df[col] < hc_min) | (df[col] > hc_max))
        ]
        if len(invalid_hc) > 0:
            logger.warning(
                f"{col}: {len(invalid_hc)} values outside [{hc_min}, {hc_max}]°C "
                f"— flagged (not removed)"
            )
            df.loc[invalid_hc.index, col + "_range_flag"] = True
            total_range_flags += len(invalid_hc)

    col20 = "HeatPump_HeatingCurveSetting_Outside20_BeforeVisit"
    col0 = "HeatPump_HeatingCurveSetting_Outside0_BeforeVisit"
    colm8 = "HeatPump_HeatingCurveSetting_OutsideMinus8_BeforeVisit"
    if all(c in df.columns for c in [col20, col0, colm8]):
        three_mask = df[col20].notna() & df[col0].notna() & df[colm8].notna()
        non_mono = df[
            three_mask
            & ((df[col20] >= df[col0]) | (df[col0] >= df[colm8]))
        ]
        if len(non_mono) > 0:
            logger.warning(
                f"Heating curve monotonicity: {len(non_mono)} protocols "
                f"have non-monotonic settings — flagged"
            )
            df.loc[non_mono.index, "heating_curve_nonmonotonic_flag"] = True
        else:
            logger.info("Heating curve monotonicity: all valid ✓")
        stats["heating_curve_nonmonotonic"] = len(non_mono)
    stats["heating_curve_range_flags"] = total_range_flags

    # 2.3.5 — Ground-source structural missingness (documentation only)
    gs_cols = [c for c in df.columns if "GroundSource" in c]
    logger.info(
        f"Ground-source HP columns ({len(gs_cols)} total): structural missingness "
        f"(97–98% — air-source households have no brine/pressure data). Not imputed."
    )

    # 2.3.6A — Coerce object boolean columns to pandas 'boolean' dtype
    n_coerced = 0
    for col in _PROTOCOL_OBJECT_BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("boolean")
            n_coerced += 1
    logger.info(f"Coerced {n_coerced} protocol object-boolean columns → 'boolean' dtype")

    # 2.3.6B — EV checkbox: NaN → False (only True values exist; NaN = no EV)
    if "Building_ElectricVehicle_Available" in df.columns:
        n_ev = int(df["Building_ElectricVehicle_Available"].isna().sum())
        df["Building_ElectricVehicle_Available"] = (
            df["Building_ElectricVehicle_Available"].fillna(False)
        )
        logger.info(
            f"Building_ElectricVehicle_Available: {n_ev} NaN → False (checkbox semantics)"
        )
        stats["ev_checkbox_filled"] = n_ev

    # 2.3.7 — HP capacity and COP range checks
    cap_min = cfg.get("hp_capacity_min_kw", 2.0)
    cap_max = cfg.get("hp_capacity_max_kw", 60.0)
    invalid_cap = df[
        df["HeatPump_Installation_HeatingCapacity"].notna()
        & (
            (df["HeatPump_Installation_HeatingCapacity"] < cap_min)
            | (df["HeatPump_Installation_HeatingCapacity"] > cap_max)
        )
    ]
    if len(invalid_cap) > 0:
        logger.warning(
            f"HeatPump_Installation_HeatingCapacity: {len(invalid_cap)} "
            f"values outside [{cap_min}, {cap_max}] kW — flagged"
        )
    else:
        logger.info(
            f"HeatPump_Installation_HeatingCapacity: all values in [{cap_min}, {cap_max}] kW ✓"
        )
    stats["hp_capacity_flags"] = len(invalid_cap)

    cop_min = cfg.get("cop_min", 2.0)
    cop_max = cfg.get("cop_max", 6.0)
    if "HeatPump_Installation_Normpoint_COP" in df.columns:
        invalid_cop = df[
            df["HeatPump_Installation_Normpoint_COP"].notna()
            & (
                (df["HeatPump_Installation_Normpoint_COP"] < cop_min)
                | (df["HeatPump_Installation_Normpoint_COP"] > cop_max)
            )
        ]
        if len(invalid_cop) > 0:
            logger.warning(
                f"HeatPump_Installation_Normpoint_COP: {len(invalid_cop)} "
                f"values outside [{cop_min}, {cop_max}] — flagged"
            )
        else:
            logger.info(
                f"HeatPump_Installation_Normpoint_COP: all values in [{cop_min}, {cop_max}] ✓"
            )
        stats["cop_flags"] = len(invalid_cop)

    # 2.3.8 — HeatingPower: string → numeric (mixed content: numerics + 'A2/W35' codes)
    hp_col = "HeatPump_Installation_Normpoint_HeatingPower"
    if hp_col in df.columns:
        original_non_null = int(df[hp_col].notna().sum())
        df[hp_col] = pd.to_numeric(df[hp_col], errors="coerce")
        new_non_null = int(df[hp_col].notna().sum())
        n_coerced_nan = original_non_null - new_non_null
        if n_coerced_nan > 0:
            logger.warning(
                f"{hp_col}: {n_coerced_nan} non-numeric strings (e.g. 'A2/W35') "
                f"coerced to NaN — likely normpoint codes misplaced in wrong column"
            )
        else:
            logger.info(f"{hp_col}: all non-null values were numeric ✓")
        stats["heating_power_nonnumeric_coerced"] = n_coerced_nan

    stats["output_rows"] = len(df)
    return df, stats


# ---------------------------------------------------------------------------
# Task 2.4 — Weather Cleaning
# ---------------------------------------------------------------------------

def clean_weather(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Weather data cleaning pipeline (Tasks 2.4.1 – 2.4.6).

    Steps:
      1. Completeness check per station (log missing calendar days)
      2. Timestamp dtype: object → datetime64
      3. Sunshine structural absence: add sunshine_available flag
      4. Interpolate scattered temperature/precipitation gaps
      5. Cross-station temperature validation flag
      6. Range sanity checks
    """
    stats: dict = {"input_rows": len(df)}
    cfg = config.get("cleaning", {})
    no_sun_stations = cfg.get("no_sunshine_stations", ["HbsbG", "ceOxS", "sV3mR"])
    cross_station_thresh = cfg.get("weather_cross_station_temp_threshold", 5.0)

    df = df.copy()

    # 2.4.1 — Completeness check per station (before timestamp conversion)
    for sid, sdf in df.groupby("Weather_ID"):
        ts_vals = pd.to_datetime(sdf["Timestamp"])
        date_range = pd.date_range(ts_vals.min(), ts_vals.max(), freq="D")
        actual_dates = set(ts_vals.dt.date)
        expected_dates = set(date_range.date)
        missing = expected_dates - actual_dates
        if missing:
            logger.warning(
                f"Station {sid}: {len(missing)} missing calendar days — {sorted(missing)}"
            )
        else:
            logger.info(
                f"Station {sid}: {len(sdf)} rows, "
                f"{ts_vals.min().date()} → {ts_vals.max().date()}, "
                f"no missing calendar days ✓"
            )

    # 2.4.2 — Timestamp dtype conversion: object (datetime.date) → datetime64
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    logger.info(f"Weather Timestamp converted: object → {df['Timestamp'].dtype}")

    # 2.4.3 — Sunshine structural absence: add sunshine_available flag
    df["sunshine_available"] = ~df["Weather_ID"].isin(no_sun_stations)
    n_no_sun = int((~df["sunshine_available"]).sum())
    n_no_sun_hh_equiv = df.loc[~df["sunshine_available"], "Weather_ID"].nunique()
    logger.warning(
        f"Sunshine_duration_daily: STRUCTURALLY ABSENT at stations {no_sun_stations}. "
        f"{n_no_sun} station-days ({n_no_sun / len(df):.1%}) have no sunshine data. "
        f"Affects 107 of 1,298 SMD households (8.2%). NOT interpolated."
    )
    stats["no_sunshine_station_days"] = n_no_sun

    # 2.4.4 — Interpolate scattered temperature/precipitation gaps
    df = df.groupby("Weather_ID", group_keys=False).apply(_fill_weather_gaps)
    n_interpolated = int(df["interpolated_flag"].sum())
    logger.info(
        f"Weather gap interpolation: {n_interpolated} station-days had "
        f"missing values filled via linear interpolation "
        f"(Sunshine_duration_daily excluded)"
    )
    stats["weather_interpolated_rows"] = n_interpolated

    # 2.4.5 — Cross-station temperature validation
    daily_net_mean = (
        df.groupby("Timestamp")["Temperature_avg_daily"]
        .mean()
        .rename("_net_temp_mean")
    )
    df = df.merge(daily_net_mean, on="Timestamp", how="left")
    df["_temp_dev"] = (df["Temperature_avg_daily"] - df["_net_temp_mean"]).abs()
    extreme_flag = df["_temp_dev"] > cross_station_thresh
    n_cross = int(extreme_flag.sum())
    df["temp_cross_station_flag"] = extreme_flag
    df = df.drop(columns=["_net_temp_mean", "_temp_dev"])

    if n_cross > 0:
        logger.warning(
            f"Cross-station validation: {n_cross} station-days deviate "
            f">{cross_station_thresh}°C from network mean — flagged"
        )
    else:
        logger.info(
            f"Cross-station validation: all stations within {cross_station_thresh}°C "
            f"of network mean ✓"
        )
    stats["cross_station_flags"] = n_cross

    # 2.4.6 — Range sanity checks
    _run_weather_sanity_checks(df)

    stats["output_rows"] = len(df)
    return df, stats


def _fill_weather_gaps(station_df: pd.DataFrame) -> pd.DataFrame:
    """
    For a single station: linearly interpolate scattered gaps in temperature /
    precipitation columns. Marks interpolated rows with interpolated_flag=True.
    Sunshine_duration_daily is intentionally excluded (structural absence).
    """
    station_df = station_df.sort_values("Timestamp").copy()
    station_df["interpolated_flag"] = (
        station_df[_INTERPOLATABLE_WEATHER_COLS].isnull().any(axis=1)
    )
    station_df[_INTERPOLATABLE_WEATHER_COLS] = (
        station_df[_INTERPOLATABLE_WEATHER_COLS]
        .interpolate(method="linear", limit_direction="both")
    )
    return station_df


def _run_weather_sanity_checks(df: pd.DataFrame) -> None:
    """Log warnings for any out-of-range weather values."""
    # Temperature: -15°C to +40°C for Switzerland
    temp_oor = df[
        df["Temperature_avg_daily"].notna()
        & ((df["Temperature_avg_daily"] < -15) | (df["Temperature_avg_daily"] > 40))
    ]
    if len(temp_oor) > 0:
        logger.warning(f"Temperature_avg_daily out of [-15, +40]°C: {len(temp_oor)} rows")
    else:
        logger.info("Temperature_avg_daily range: all within [-15, +40]°C ✓")

    # Humidity: 0–100%
    hum_oor = df[
        df["Humidity_avg_daily"].notna()
        & ((df["Humidity_avg_daily"] < 0) | (df["Humidity_avg_daily"] > 100))
    ]
    if len(hum_oor) > 0:
        logger.warning(f"Humidity_avg_daily out of [0, 100]%: {len(hum_oor)} rows")
    else:
        logger.info("Humidity_avg_daily range: all within [0, 100]% ✓")

    # Sunshine: 0–16 hours (max physically possible at Swiss latitude)
    sun_oor = df[
        df["Sunshine_duration_daily"].notna()
        & ((df["Sunshine_duration_daily"] < 0) | (df["Sunshine_duration_daily"] > 16))
    ]
    if len(sun_oor) > 0:
        logger.warning(f"Sunshine_duration_daily out of [0, 16] hours: {len(sun_oor)} rows")
    else:
        logger.info("Sunshine_duration_daily range: all within [0, 16] hours ✓")


# ---------------------------------------------------------------------------
# Cleaning Report
# ---------------------------------------------------------------------------

def generate_cleaning_report(
    smd_stats: dict,
    meta_stats: dict,
    proto_stats: dict,
    weather_stats: dict,
) -> str:
    """
    Produce a human-readable cleaning report from the stats dicts returned
    by each clean_* function.
    """
    lines = []

    def _section(title: str) -> None:
        lines.append("=" * 70)
        lines.append(title)
        lines.append("=" * 70)

    def _row(label: str, value) -> None:
        lines.append(f"  {label:<50} {value}")

    _section("SMD DAILY CLEANING")
    _row("Input rows", f"{smd_stats.get('input_rows', '?'):,}")
    _row("Input households", f"{smd_stats.get('input_households', '?'):,}")
    _row("Exact duplicate rows removed", smd_stats.get("exact_duplicates_removed", 0))
    _row("(Household_ID, Date) duplicate rows removed", smd_stats.get("date_duplicates_removed", 0))
    _row("Null target rows removed", smd_stats.get("null_target_rows_removed", 0))
    _row("Households dropped entirely (null target)", smd_stats.get("households_dropped_entirely", 0))
    if smd_stats.get("households_dropped_list"):
        _row("  → Dropped household IDs", str(smd_stats["households_dropped_list"]))
    _row("Non-positive target rows removed", smd_stats.get("nonpositive_rows_removed", 0))
    _row("Hard cap (>500 kWh) rows removed", smd_stats.get("hard_cap_rows_removed", 0))
    _row("IQR outlier flags added (rows retained)", smd_stats.get("iqr_outlier_flags", 0))
    _row("'during visit' rows excluded", smd_stats.get("during_visit_rows_removed", 0))
    _row("post_intervention=1 rows", smd_stats.get("post_intervention_1_count", 0))
    _row("Households below 180-day threshold (flagged)", smd_stats.get("below_threshold_households", 0))
    _row("PV households identified (has_pv=1)", smd_stats.get("pv_households", 0))
    _row("Output rows", f"{smd_stats.get('output_rows', '?'):,}")
    _row("Output households", f"{smd_stats.get('output_households', '?'):,}")
    _row("New columns added", "Date, post_intervention, is_iqr_outlier,")
    _row("", "below_min_days_threshold, has_pv, has_reactive_energy")
    lines.append("")

    _section("METADATA CLEANING")
    _row("Input rows", f"{meta_stats.get('input_rows', '?'):,}")
    _row("'appartment' → 'apartment' typo fixes", meta_stats.get("appartment_typo_fixed", 0))
    _row("Binary object columns → 'boolean' dtype", 8)
    _row("Checkbox NaN → False fills", "(see per-column log)")
    _row("Living area extreme flags (>1000 m²)", meta_stats.get("living_area_extreme_flags", 0))
    _row("Residents above 10 (flags)", meta_stats.get("residents_extreme_flags", 0))
    _row("HP type NaN → 'unknown'", meta_stats.get("hp_type_unknown_filled", 0))
    _row("New column added", "living_area_extreme_flag")
    lines.append("")

    _section("PROTOCOL CLEANING")
    _row("Input rows", f"{proto_stats.get('input_rows', '?'):,}")
    _row("Target proxy column renamed", proto_stats.get("target_proxy_renamed", "?"))
    _row("Construction year errors → NaN", proto_stats.get("construction_year_errors", 0))
    _row("HP install year errors → NaN", proto_stats.get("hp_install_year_errors", 0))
    _row("Heating curve range flags", proto_stats.get("heating_curve_range_flags", 0))
    _row("Heating curve non-monotonic flags", proto_stats.get("heating_curve_nonmonotonic", 0))
    _row("EV checkbox NaN → False", proto_stats.get("ev_checkbox_filled", 0))
    _row("HeatingPower non-numeric → NaN", proto_stats.get("heating_power_nonnumeric_coerced", 0))
    _row("HP capacity out-of-range flags", proto_stats.get("hp_capacity_flags", 0))
    _row("COP out-of-range flags", proto_stats.get("cop_flags", 0))
    lines.append("")

    _section("WEATHER DAILY CLEANING")
    _row("Input rows", f"{weather_stats.get('input_rows', '?'):,}")
    _row("Timestamp dtype", "object → datetime64")
    _row("No-sunshine station-days flagged", weather_stats.get("no_sunshine_station_days", 0))
    _row("  (stations: HbsbG, ceOxS, sV3mR — 107 SMD households)", "")
    _row("Scattered gaps interpolated (temp/precip)", weather_stats.get("weather_interpolated_rows", 0))
    _row("Cross-station temperature flags (>5°C dev)", weather_stats.get("cross_station_flags", 0))
    _row("New columns added", "sunshine_available, interpolated_flag, temp_cross_station_flag")
    lines.append("")

    return "\n".join(lines)
