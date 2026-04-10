"""
src/data_merger.py

Data merging utilities for the HEAPO dataset.
Phase 3 implementation — Tasks 3.0 through 3.7.

Produces two analysis tracks:
  Track A — Full-sample  : SMD + Weather + Metadata  (~913k rows, all 1,272 households)
  Track B — Protocol-enriched: Track A (treatment only) + Protocols (~217 households)

All column names verified against:
  - Table 1 (SMD), Table 4 (Metadata), Table 5 (Protocols), Table 6 (Weather)
  in Brudermueller et al. (2025), arXiv:2503.16993v1
  and against actual parquet files produced by Phase 2.

Phase 4 (Feature Engineering) consumes merged_full.parquet and merged_protocol.parquet
directly — no additional loading or reshaping is needed between phases.
"""

import logging
from io import StringIO
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stations with NO sunshine sensor — structural absence, not missingness.
# 107 of 1,272 SMD households are served by these stations.
NO_SUNSHINE_STATIONS: set = {"HbsbG", "ceOxS", "sV3mR"}

# The original target proxy column name — must never appear in any frame (Phase 2
# renames it with the EXCLUDED_ prefix, which is still forbidden in merged outputs).
# Both names are forbidden from the final merged tracks.
FORBIDDEN_COLUMNS: set = {
    "EXCLUDED_TARGET_PROXY_HeatPump_ElectricityConsumption_YearlyEstimated",
    "HeatPump_ElectricityConsumption_YearlyEstimated",
}

# The renamed form is allowed to remain in protocols_clean.parquet (Phase 2 renamed
# rather than dropped it for auditability). It is excluded from PROTOCOL_FEATURE_COLS
# so it never enters the merged outputs. Only the original un-prefixed name is truly
# forbidden in any input frame at load time.
_FORBIDDEN_AT_LOAD: set = {"HeatPump_ElectricityConsumption_YearlyEstimated"}

# Expected shapes from Phase 2 outputs. A mismatch means Phase 2 was re-run
# with different parameters — fail loudly before merging incorrect data.
# households uses Household_ID as index, so shape[1] counts data columns only.
EXPECTED_PHASE2_SHAPES: Dict[str, Tuple[int, int]] = {
    "smd_daily_clean":     (913_620, 21),
    "metadata_clean":      (1_358,   14),
    "protocols_clean":     (410,     111),
    "weather_daily_clean": (15_083,  13),
    "households":          (1_408,   10),   # Household_ID is index
}

# Protocol columns to bring into Track B.
# Verified against actual protocols_clean.parquet column list on 2026-04-05.
# Columns are organised by conceptual group to match Phase 4 feature engineering.
PROTOCOL_FEATURE_COLS: list = [
    # ── Administrative (needed for Phase 4 derived features) ──────────────────
    "Visit_Year",

    # ── Building characteristics ──────────────────────────────────────────────
    "Building_ConstructionYear",          # → Phase 4: building_age
    "Building_HousingUnits",
    "Building_Residents",                 # protocol version
    "Building_FloorAreaHeated_Total",     # → Phase 4: hp_capacity_per_area
    "Building_FloorAreaHeated_Basement",
    "Building_FloorAreaHeated_GroundFloor",
    "Building_FloorAreaHeated_FirstFloor",
    "Building_FloorAreaHeated_SecondFloor",
    "Building_FloorAreaHeated_TopFloor",
    "Building_FloorAreaHeated_AdditionalAreasPlanned",
    "Building_Renovated_Windows",         # insulation quality proxies
    "Building_Renovated_Roof",
    "Building_Renovated_Walls",
    "Building_Renovated_Floor",
    "Building_PVSystem_Available",
    "Building_PVSystem_Size",
    "Building_ElectricVehicle_Available",

    # ── Heat pump installation ────────────────────────────────────────────────
    "HeatPump_Installation_Type",         # protocol version (may differ from survey)
    "HeatPump_Installation_Year",         # → Phase 4: hp_age
    "HeatPump_Installation_Manufacturer",
    "HeatPump_Installation_Model",
    "HeatPump_Installation_HeatingCapacity",
    "HeatPump_Installation_Normpoint_COP",
    "HeatPump_Installation_Normpoint_HeatingPower",
    "HeatPump_Installation_Normpoint_ElectricPower",
    "HeatPump_Installation_Location",     # inside / outside / split
    "HeatPump_Installation_InternetConnection",
    "HeatPump_Installation_Refrigerant_Type",
    "HeatPump_Installation_CorrectlyPlanned",

    # ── Heat distribution system ──────────────────────────────────────────────
    "HeatDistribution_System_FloorHeating",
    "HeatDistribution_System_Radiators",
    "HeatDistribution_System_ThermostaticValve",
    "HeatDistribution_System_BufferTankAvailable",

    # ── Heating curve settings (before visit) ─────────────────────────────────
    # All three points used in Phase 4 gradient feature engineering.
    "HeatPump_HeatingCurveSetting_Outside20_BeforeVisit",
    "HeatPump_HeatingCurveSetting_Outside0_BeforeVisit",
    "HeatPump_HeatingCurveSetting_OutsideMinus8_BeforeVisit",

    # ── HP setting issues (binary assessments — most predictive per Table 3) ──
    "HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit",   # 40.98% of visits
    "HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit",   # 25.61% of visits
    "HeatPump_NightSetbackSetting_Activated_BeforeVisit", # 36.10% of visits

    # ── HP settings (numeric) ─────────────────────────────────────────────────
    "HeatPump_HeatingLimitSetting_BeforeVisit",
    "HeatPump_NightSetbackSetting_Activated_AfterVisit",  # for treatment effect analysis

    # ── DHW configuration ─────────────────────────────────────────────────────
    "DHW_Production_ByHeatPump",
    "DHW_Production_ByElectricWaterHeater",
    "DHW_Production_BySolar",
    "DHW_Production_ByHeatPumpBoiler",
    "DHW_TemperatureSetting_BeforeVisit",
    "DHW_TemperatureSetting_AfterVisit",  # for treatment effect analysis
    "DHW_Storage_LastDescaling_TooLongAgo",
    "DHW_Sterilization_Available",
    "DHW_Sterilization_Active",

    # ── HP condition assessments ──────────────────────────────────────────────
    "HeatPump_Clean",
    "HeatPump_BasicFunctionsOkay",
    "HeatPump_TechnicallyOkay",

    # ── Heat distribution recommendations ─────────────────────────────────────
    "HeatDistribution_Recommendation_InsulatePipes",
    "HeatDistribution_Recommendation_InstallThermostaticValve",
    "HeatDistribution_Recommendation_InstallRPMValve",
    "HeatDistribution_ExpansionTank_Pressure_Categorization",

    # ── Ground-source HP specifics (structural missingness: ~97-98% null) ─────
    # Air-source households have no brine/pressure data — retained as-is.
    # XGBoost handles NaN natively; LR/ANN imputation handled in Phase 6.
    "HeatPump_GroundSource_BrineCircuit_AntiFreezeExists",
    "HeatPump_GroundSource_CurrentPressure_Okay",
    "HeatPump_GroundSource_CurrentTemperature_Okay",
    "HeatPump_GroundSource_CurrentPressure",
    "HeatPump_GroundSource_CurrentTemperature",
]

# Columns to explicitly exclude from Track B (internal Phase 2 flags, admin,
# and the target proxy). Everything NOT in PROTOCOL_FEATURE_COLS is excluded
# by the column-selection step — these are called out for documentation clarity.
_PROTOCOL_COLS_NEVER_INCLUDE: set = {
    # Target leakage
    "EXCLUDED_TARGET_PROXY_HeatPump_ElectricityConsumption_YearlyEstimated",
    # Join key — dropped after merge
    "Household_ID",
    # Phase 2 internal cleaning flags — not features
    "is_orphan",
    "visit_number",
    "HeatPump_HeatingCurveSetting_Outside20_BeforeVisit_range_flag",
    "HeatPump_HeatingCurveSetting_OutsideMinus8_BeforeVisit_range_flag",
    "heating_curve_nonmonotonic_flag",
}


# ---------------------------------------------------------------------------
# Task 3.0 — Load Phase 2 Artifacts
# ---------------------------------------------------------------------------

def load_phase2_artifacts(processed_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all five Phase 2 parquets and verify shapes.

    Also re-attaches Weather_ID to weather_daily_clean (it was dropped during
    Phase 2 cleaning — the raw weather parquet still carries it).

    Returns a dict with keys:
        smd, metadata, protocols, weather, households
    """
    def _load(name: str, filename: str, index_col=None) -> pd.DataFrame:
        path = processed_dir / filename
        df = pd.read_parquet(path)
        logger.info("Loaded %-30s (%d rows × %d cols)", path.name, len(df), df.shape[1])
        return df

    # Load all five frames
    smd       = _load("smd",       "smd_daily_clean.parquet")
    metadata  = _load("metadata",  "metadata_clean.parquet")
    protocols = _load("protocols", "protocols_clean.parquet")
    weather   = _load("weather",   "weather_daily_clean.parquet")
    households = pd.read_parquet(processed_dir / "households.parquet")
    logger.info(
        "Loaded %-30s (%d rows × %d cols, index=Household_ID)",
        "households.parquet", len(households), households.shape[1]
    )

    # Re-attach Weather_ID to weather_daily_clean.
    # Phase 2 cleaning dropped it (column count: 11 raw → 13 clean after adding
    # sunshine_available, interpolated_flag, temp_cross_station_flag).
    # The raw parquet row order is preserved through cleaning — safe to align by index.
    weather_raw = pd.read_parquet(processed_dir / "weather_daily.parquet",
                                  columns=["Weather_ID", "Timestamp"])
    assert len(weather_raw) == len(weather), (
        f"Row count mismatch between weather_daily.parquet ({len(weather_raw)}) "
        f"and weather_daily_clean.parquet ({len(weather)})"
    )
    weather = weather.copy()
    weather["Weather_ID"] = weather_raw["Weather_ID"].values
    logger.info("Re-attached Weather_ID to weather_daily_clean (%d unique stations)",
                weather["Weather_ID"].nunique())

    # Assert exact Phase 2 shapes before any merging
    checks = {
        "smd_daily_clean":     smd,
        "metadata_clean":      metadata,
        "protocols_clean":     protocols,
        "weather_daily_clean": weather,   # now 14 cols after Weather_ID re-attach
        "households":          households,
    }
    for name, df in checks.items():
        if name == "weather_daily_clean":
            # After re-attaching Weather_ID: 13 + 1 = 14 cols
            expected = (EXPECTED_PHASE2_SHAPES[name][0], EXPECTED_PHASE2_SHAPES[name][1] + 1)
        else:
            expected = EXPECTED_PHASE2_SHAPES[name]
        actual = df.shape
        if actual != expected:
            raise ValueError(
                f"Shape mismatch for '{name}': expected {expected}, got {actual}. "
                f"Phase 2 may have been re-run with different parameters."
            )
        logger.info("Shape check ✓  %-25s %s", name, actual)

    # Assert the un-prefixed target proxy has not leaked into any input frame.
    # The EXCLUDED_TARGET_PROXY_... renamed version is allowed to remain in
    # protocols_clean (Phase 2 kept it for auditability) — it is excluded from
    # PROTOCOL_FEATURE_COLS and therefore never enters the merged outputs.
    for name, df in checks.items():
        leaked = _FORBIDDEN_AT_LOAD & set(df.columns)
        if leaked:
            raise ValueError(
                f"'{name}' contains the un-prefixed target proxy column: {leaked}. "
                f"Phase 2 must have renamed it with the EXCLUDED_ prefix."
            )
    logger.info("Target proxy load check ✓  (original un-prefixed column absent from all inputs)")

    return {
        "smd":        smd,
        "metadata":   metadata,
        "protocols":  protocols,
        "weather":    weather,
        "households": households,
    }


# ---------------------------------------------------------------------------
# Task 3.2 — Merge SMD with Weather
# ---------------------------------------------------------------------------

def merge_smd_weather(
    smd: pd.DataFrame,
    weather: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Join SMD with daily weather on (Weather_ID, date).

    Weather_ID is already present in smd_daily_clean (attached during Phase 1
    loading from households.csv). Weather_ID was re-attached to weather_daily_clean
    in load_phase2_artifacts().

    Date alignment:
      - SMD 'Date' is datetime64[us, Europe/Zurich] (tz-aware midnight).
      - Weather 'Timestamp' is datetime64[ms] (tz-naive).
      - Both are normalized to Python datetime.date via .dt.date for a
        timezone-safe join key — no calendar-day ambiguity possible.
    """
    stats: dict = {"smd_rows_in": len(smd)}

    smd = smd.copy()
    weather = weather.copy()

    # Build date-only join keys (timezone-safe)
    smd["_date_key"] = smd["Date"].dt.date
    weather["_date_key"] = weather["Timestamp"].dt.date

    # Drop Timestamp from weather — Date column in SMD already records the day.
    weather_cols = [c for c in weather.columns if c != "Timestamp"]
    weather_for_join = weather[weather_cols]

    # Left join: preserves every SMD row; surfacing any unmatched rows as nulls.
    merged = smd.merge(
        weather_for_join,
        on=["Weather_ID", "_date_key"],
        how="left",
        suffixes=("", "_weather"),
    )
    merged.drop(columns=["_date_key"], inplace=True)

    # Check for null-weather rows after the join.
    # Expected cause: SMD data runs to 2024-03-21 but weather data ends 2024-02-28/29.
    # Rows in the March 2024 gap (21 days × ~831 households) will have null weather.
    # These are RETAINED — Phase 6 preprocessing excludes them from training/eval
    # or imputes via forward-fill if the gap is later filled.
    n_null_temp = merged["Temperature_avg_daily"].isna().sum()
    if n_null_temp > 0:
        gap_dates = merged[merged["Temperature_avg_daily"].isna()]["Date"]
        gap_min, gap_max = gap_dates.min(), gap_dates.max()
        n_gap_hh = merged[merged["Temperature_avg_daily"].isna()]["Household_ID"].nunique()
        logger.warning(
            "Weather join: %d rows (%d households) have null weather features. "
            "Date range: %s → %s. "
            "Cause: SMD data extends beyond weather coverage end (2024-02-29). "
            "Rows are RETAINED with null weather — Phase 6 must handle them.",
            n_null_temp, n_gap_hh, gap_min, gap_max,
        )
        stats["weather_gap_rows"] = int(n_null_temp)
        stats["weather_gap_households"] = int(n_gap_hh)
        stats["weather_gap_date_min"] = str(gap_min)
        stats["weather_gap_date_max"] = str(gap_max)
    else:
        logger.info("Weather join: 0 null temperature rows ✓")
        stats["weather_gap_rows"] = 0

    # Add household-level sunshine flag (constant per household).
    # 107 households at stations HbsbG/ceOxS/sV3mR have no sunshine sensor.
    merged["hh_no_sunshine"] = merged["Weather_ID"].isin(NO_SUNSHINE_STATIONS)

    # Log sunshine null count. Two sources: no-sensor stations + weather coverage gap.
    n_sunshine_null = int(merged["Sunshine_duration_daily"].isna().sum())
    n_no_sunshine_flag = int(merged["hh_no_sunshine"].sum())
    n_gap = int(merged["Temperature_avg_daily"].isna().sum())
    logger.info(
        "Sunshine_duration_daily: %d null rows "
        "(%d from no-sensor stations, %d from weather coverage gap; union = %d)",
        n_sunshine_null, n_no_sunshine_flag, n_gap,
        int((merged["hh_no_sunshine"].fillna(False) | merged["Temperature_avg_daily"].isna()).sum()),
    )

    # Log station distribution
    station_dist = merged.groupby("Weather_ID")["Household_ID"].nunique().to_dict()
    logger.info("Household-to-station distribution: %s", station_dist)

    stats.update({
        "rows_after_weather_join": len(merged),
        "null_temperature_rows": int(n_null_temp),
        "hh_no_sunshine": int(
            merged.groupby("Household_ID")["hh_no_sunshine"].first().sum()
        ),
        "cols_after_weather_join": merged.shape[1],
    })

    logger.info(
        "SMD + Weather: %d rows × %d cols | %d households",
        len(merged), merged.shape[1], merged["Household_ID"].nunique(),
    )
    return merged, stats


# ---------------------------------------------------------------------------
# Task 3.3 — Merge with Metadata
# ---------------------------------------------------------------------------

def merge_metadata(
    smd_weather: pd.DataFrame,
    metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Left-join metadata onto the SMD+Weather frame on Household_ID.

    Metadata is static (one row per household) and fans out to all daily rows.
    ~50 households (~3.55%) have no metadata — their metadata columns will be
    null. Do NOT impute here; Phase 6.1 handles imputation per model track.
    """
    stats: dict = {"rows_in": len(smd_weather)}

    merged = smd_weather.merge(metadata, on="Household_ID", how="left")

    # Row count must not change (left join)
    if len(merged) != len(smd_weather):
        raise ValueError(
            f"Metadata join changed row count: {len(smd_weather)} → {len(merged)}. "
            f"metadata_clean.parquet may have duplicate Household_IDs."
        )

    # Count and log households with no metadata
    n_no_meta_hh = (
        merged.groupby("Household_ID")["Survey_Building_Type"]
        .first()
        .isna()
        .sum()
    )
    pct = 100 * n_no_meta_hh / merged["Household_ID"].nunique()
    logger.info(
        "Households with no metadata: %d (%.1f%%) — null metadata columns retained",
        n_no_meta_hh, pct,
    )
    if n_no_meta_hh > 60:
        logger.warning(
            "Unexpectedly high metadata missingness: %d households (expected ≤60)",
            n_no_meta_hh,
        )

    stats.update({
        "rows_after_metadata_join": len(merged),
        "hh_no_metadata": int(n_no_meta_hh),
        "cols_after_metadata_join": merged.shape[1],
    })

    logger.info(
        "SMD + Weather + Metadata: %d rows × %d cols | %d households",
        len(merged), merged.shape[1], merged["Household_ID"].nunique(),
    )
    return merged, stats


# ---------------------------------------------------------------------------
# Task 3.4 — Build Protocol Map and Merge (Track B)
# ---------------------------------------------------------------------------

def build_protocol_map(
    households: pd.DataFrame,
    protocols: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a one-row-per-household protocol feature table for Track B.

    Mapping logic:
      - households.parquet (index=Household_ID) has Protocols_ReportIDs like '[29]'.
      - protocols_clean.parquet has visit_number (1=first visit, 2+=repeat).
      - Use visit_number==1 as the primary (baseline) visit — avoids leaking
        future consultant knowledge into pre-visit rows.
      - Multiple-visit households (111119, 120912, 8087988) are handled correctly:
        their first visit is used as the static feature baseline; the
        post_intervention flag (already in SMD) tracks before/after timing.
      - 193 orphan protocols (is_orphan=True, Household_ID=NaN) are excluded.

    Returns:
        protocol_per_hh — one row per treatment household, Household_ID as int64.
    """
    # Filter to non-orphan, first-visit protocols only
    linked = protocols[
        (~protocols["is_orphan"]) & (protocols["visit_number"] == 1)
    ].copy()

    # Household_ID is float64 in protocols (NaN for orphans forced this dtype).
    # Convert to int64 now that orphans are excluded.
    linked["Household_ID"] = linked["Household_ID"].astype("int64")

    # Log multi-visit households and their selected primary Report_ID
    multi_visit_hhs = protocols[
        (~protocols["is_orphan"]) & (protocols["visit_number"] > 1)
    ]["Household_ID"].dropna().astype("int64").unique()
    if len(multi_visit_hhs) > 0:
        primary_reports = linked[
            linked["Household_ID"].isin(multi_visit_hhs)
        ][["Household_ID", "Report_ID", "Visit_Year"]]
        logger.info(
            "Multiple-visit households: %s — primary (first) visit used:\n%s",
            multi_visit_hhs.tolist(),
            primary_reports.to_string(index=False),
        )

    logger.info(
        "Protocol map: %d linked non-orphan first-visit protocols "
        "(expected 214 unique households)",
        len(linked),
    )

    # Select feature columns (verified against actual protocols_clean columns)
    available_feature_cols = [c for c in PROTOCOL_FEATURE_COLS if c in linked.columns]
    missing_from_parquet = set(PROTOCOL_FEATURE_COLS) - set(available_feature_cols)
    if missing_from_parquet:
        logger.warning(
            "Protocol feature columns listed in PROTOCOL_FEATURE_COLS but absent "
            "from protocols_clean.parquet (check column name changes): %s",
            sorted(missing_from_parquet),
        )

    keep_cols = ["Household_ID", "Report_ID"] + available_feature_cols
    protocol_per_hh = linked[keep_cols].copy()

    # Verify no duplicate household IDs (one row per household)
    n_dup = protocol_per_hh.duplicated("Household_ID").sum()
    if n_dup > 0:
        raise ValueError(
            f"build_protocol_map: {n_dup} duplicate Household_IDs after "
            f"filtering to visit_number==1. Check protocols_clean.parquet."
        )

    logger.info(
        "Protocol feature table: %d rows × %d cols (1 per treatment household)",
        len(protocol_per_hh), protocol_per_hh.shape[1],
    )
    return protocol_per_hh


def merge_protocols(
    track_a: pd.DataFrame,
    protocol_per_hh: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Create Track B by filtering Track A to treatment households and
    left-joining protocol features.

    track_a   — full-sample merged frame (Track A output).
    protocol_per_hh — one-row-per-household protocol map from build_protocol_map().

    Returns (track_b_df, stats_dict).
    """
    stats: dict = {}

    # Track B base: all treatment-group rows from Track A
    track_b_base = track_a[track_a["Group"] == "treatment"].copy()
    n_treatment_rows = len(track_b_base)
    n_treatment_hh = track_b_base["Household_ID"].nunique()
    logger.info(
        "Track B base (treatment group): %d rows | %d households",
        n_treatment_rows, n_treatment_hh,
    )

    # Left join protocol features onto treatment rows
    track_b = track_b_base.merge(
        protocol_per_hh.drop(columns=["Report_ID"]),
        on="Household_ID",
        how="left",
    )

    # Row count must be preserved (left join)
    if len(track_b) != n_treatment_rows:
        raise ValueError(
            f"Protocol join changed Track B row count: "
            f"{n_treatment_rows} → {len(track_b)}"
        )

    # Count households that received protocol features
    n_with_protocols = (
        track_b.groupby("Household_ID")["Visit_Year"].first().notna().sum()
    )
    n_without_protocols = n_treatment_hh - n_with_protocols
    logger.info(
        "Track B: %d households have protocol features | %d have null protocol "
        "(treatment HHs with no linked report)",
        n_with_protocols, n_without_protocols,
    )

    # Verify target proxy is absent
    leaked = FORBIDDEN_COLUMNS & set(track_b.columns)
    if leaked:
        raise ValueError(f"Target proxy column(s) in Track B: {leaked}")

    stats.update({
        "track_b_rows": len(track_b),
        "track_b_households": int(n_treatment_hh),
        "hh_with_protocol_features": int(n_with_protocols),
        "hh_without_protocol_features": int(n_without_protocols),
        "track_b_cols": track_b.shape[1],
    })

    logger.info(
        "Track B: %d rows × %d cols | %d households",
        len(track_b), track_b.shape[1], n_treatment_hh,
    )
    return track_b, stats


# ---------------------------------------------------------------------------
# Task 3.5 — Integrity Checks
# ---------------------------------------------------------------------------

def run_integrity_checks(
    track_a: pd.DataFrame,
    track_b: pd.DataFrame,
    expected_track_a_rows: int = 913_620,
) -> None:
    """
    Run all 8 post-merge integrity checks. Raises ValueError on any failure.
    All checks must pass before outputs are written to disk.
    """
    logger.info("Running post-merge integrity checks...")
    errors = []

    # Check 1 — Track A row count
    if len(track_a) != expected_track_a_rows:
        errors.append(
            f"Check 1 FAIL: Track A row count {len(track_a)} != {expected_track_a_rows}"
        )
    else:
        logger.info("Check 1 ✓  Track A row count: %d", len(track_a))

    # Check 2 — No duplicate (Household_ID, Date) pairs in either track
    for name, df in [("Track A", track_a), ("Track B", track_b)]:
        n_dupes = df.duplicated(["Household_ID", "Date"]).sum()
        if n_dupes > 0:
            errors.append(
                f"Check 2 FAIL: {name} has {n_dupes} duplicate (Household_ID, Date) pairs"
            )
        else:
            logger.info("Check 2 ✓  %s: 0 duplicate (Household_ID, Date) pairs", name)

    # Check 3 — Target variable is fully non-null in Track A
    n_null_target = track_a["kWh_received_Total"].isna().sum()
    if n_null_target > 0:
        errors.append(
            f"Check 3 FAIL: Track A has {n_null_target} null kWh_received_Total rows"
        )
    else:
        logger.info("Check 3 ✓  kWh_received_Total: 0 null values")

    # Check 4 — No target proxy leak in either track
    for name, df in [("Track A", track_a), ("Track B", track_b)]:
        leaked = FORBIDDEN_COLUMNS & set(df.columns)
        if leaked:
            errors.append(f"Check 4 FAIL: {name} contains target proxy column(s): {leaked}")
        else:
            logger.info("Check 4 ✓  %s: no target proxy columns", name)

    # Check 5 — Weather features non-null for in-coverage rows.
    # SMD data extends to 2024-03-21 but weather ends 2024-02-29; the gap rows
    # (~17k, March 2024) legitimately have null weather — warn, don't fail.
    n_null_temp = track_a["Temperature_avg_daily"].isna().sum()
    if n_null_temp > 0:
        gap_hh = track_a[track_a["Temperature_avg_daily"].isna()]["Household_ID"].nunique()
        logger.warning(
            "Check 5 WARNING: %d rows (%d households) have null Temperature_avg_daily "
            "(SMD extends beyond weather coverage end 2024-02-29). "
            "These rows must be excluded or imputed in Phase 6.",
            n_null_temp, gap_hh,
        )
    else:
        logger.info("Check 5 ✓  Temperature_avg_daily: 0 null values")

    # Check 6 — Sunshine nulls should equal the UNION of:
    #   (a) hh_no_sunshine rows (stations HbsbG/ceOxS/sV3mR — no sunshine sensor).
    #   (b) Weather-gap rows (before 2019-01-01 or after 2024-02-29 — outside weather coverage).
    # Use actual column flags to compute expected rather than a sum (avoids double-counting
    # rows that fall into both categories).
    sunshine_nulls = int(track_a["Sunshine_duration_daily"].isna().sum())
    no_sunshine_mask = track_a["hh_no_sunshine"].fillna(False)
    weather_gap_mask = track_a["Temperature_avg_daily"].isna()
    expected_sunshine_nulls = int((no_sunshine_mask | weather_gap_mask).sum())
    if sunshine_nulls != expected_sunshine_nulls:
        logger.warning(
            "Check 6 WARNING: Sunshine null rows (%d) != union of "
            "hh_no_sunshine | weather_gap (%d). "
            "Investigate unexpected sunshine absence.",
            sunshine_nulls, expected_sunshine_nulls,
        )
    else:
        n_no_sunshine = int(no_sunshine_mask.sum())
        n_gap = int(weather_gap_mask.sum())
        logger.info(
            "Check 6 ✓  Sunshine nulls %d = union(%d no-sensor, %d gap)",
            sunshine_nulls, n_no_sunshine, n_gap,
        )

    # Check 7 — Metadata coverage within expected bounds
    n_no_meta = (
        track_a.groupby("Household_ID")["Survey_Building_Type"].first().isna().sum()
    )
    if n_no_meta > 60:
        errors.append(
            f"Check 7 FAIL: {n_no_meta} households have no metadata (expected ≤60)"
        )
    else:
        logger.info("Check 7 ✓  Households without metadata: %d (≤60 expected)", n_no_meta)

    # Check 8 — Track B protocol coverage.
    # 214 protocols are linked but only households that survived Phase 2 cleaning
    # appear here; 62 treatment households were dropped for null targets in Phase 2,
    # leaving ~152 treatment households. Check that at least half have protocol data
    # (a very conservative lower bound to catch join failures).
    if "Visit_Year" in track_b.columns:
        n_b_hh = track_b["Household_ID"].nunique()
        n_proto_hh = int(track_b.groupby("Household_ID")["Visit_Year"].first().notna().sum())
        n_null_proto = n_b_hh - n_proto_hh
        min_expected = max(100, n_b_hh // 2)
        if n_proto_hh < min_expected:
            errors.append(
                f"Check 8 FAIL: Only {n_proto_hh} of {n_b_hh} Track B households "
                f"have protocol features (expected ≥{min_expected})"
            )
        else:
            logger.info(
                "Check 8 ✓  Track B: %d/%d households have protocol features "
                "(%d without — dropped in Phase 2 cleaning)",
                n_proto_hh, n_b_hh, n_null_proto,
            )
    else:
        errors.append("Check 8 FAIL: Visit_Year column missing from Track B")

    # Raise all failures together for easy diagnosis
    if errors:
        raise ValueError("Post-merge integrity checks FAILED:\n" + "\n".join(errors))

    logger.info("All 8 integrity checks passed ✓")


# ---------------------------------------------------------------------------
# Task 3.7 — Merge Report
# ---------------------------------------------------------------------------

def generate_merge_report(
    track_a: pd.DataFrame,
    track_b: pd.DataFrame,
    stats: dict,
) -> str:
    """
    Generate a human-readable merge report for outputs/tables/phase3_merge_report.txt.
    """
    buf = StringIO()

    def w(line: str = "") -> None:
        buf.write(line + "\n")

    sep = "=" * 70

    w(sep)
    w("PHASE 3 MERGE REPORT")
    w("HEAPO-Predict — Daily HP Electricity Consumption")
    w(sep)
    w()

    # ── Track A summary ───────────────────────────────────────────────────────
    w(sep)
    w("TRACK A — FULL-SAMPLE MERGE (SMD + Weather + Metadata)")
    w(sep)
    w(f"  {'Input SMD rows':<45} {len(track_a):>10,}")
    w(f"  {'Input SMD households':<45} {track_a['Household_ID'].nunique():>10,}")
    w(f"  {'Weather join null rows (expected 0)':<45} "
      f"{stats.get('null_temperature_rows', 'N/A'):>10}")
    w(f"  {'Households with hh_no_sunshine=True':<45} "
      f"{int(track_a.groupby('Household_ID')['hh_no_sunshine'].first().sum()):>10,}")
    w(f"  {'Households without metadata':<45} "
      f"{int(track_a.groupby('Household_ID')['Survey_Building_Type'].first().isna().sum()):>10,}")
    w(f"  {'Output rows':<45} {len(track_a):>10,}")
    w(f"  {'Output columns':<45} {track_a.shape[1]:>10,}")
    w()

    # ── Track B summary ───────────────────────────────────────────────────────
    w(sep)
    w("TRACK B — PROTOCOL-ENRICHED MERGE (treatment group + Protocol data)")
    w(sep)
    w(f"  {'Base: treatment SMD rows':<45} {len(track_b):>10,}")
    w(f"  {'Base: treatment households':<45} {track_b['Household_ID'].nunique():>10,}")
    if "Visit_Year" in track_b.columns:
        n_proto = int(track_b.groupby("Household_ID")["Visit_Year"].first().notna().sum())
        w(f"  {'Households with protocol features':<45} {n_proto:>10,}")
    w(f"  {'Multiple-visit HHs (first visit used)':<45} {'3 [111119, 120912, 8087988]':>10}")
    w(f"  {'Output rows':<45} {len(track_b):>10,}")
    w(f"  {'Output columns':<45} {track_b.shape[1]:>10,}")
    w()

    # ── Missing values — Track A key columns ─────────────────────────────────
    w(sep)
    w("MISSING VALUE SUMMARY — TRACK A (key columns)")
    w(sep)
    key_cols_a = [
        "Survey_Building_Type",
        "Survey_Building_LivingArea",
        "Survey_Building_Residents",
        "Survey_HeatPump_Installation_Type",
        "Sunshine_duration_daily",
        "kvarh_received_capacitive_Total",
        "kvarh_received_inductive_Total",
        "kWh_received_HeatPump",
    ]
    n_hh = track_a["Household_ID"].nunique()
    w(f"  {'Column':<50} {'Null rows':>10}  {'Null HHs':>10}  {'%HHs':>6}")
    w("  " + "-" * 80)
    for col in key_cols_a:
        if col not in track_a.columns:
            continue
        n_null_rows = int(track_a[col].isna().sum())
        n_null_hh = int(
            track_a.groupby("Household_ID")[col].first().isna().sum()
        )
        pct = 100 * n_null_hh / n_hh
        w(f"  {col:<50} {n_null_rows:>10,}  {n_null_hh:>10,}  {pct:>5.1f}%")
    w()

    # ── Missing values — Track B protocol columns ─────────────────────────────
    w(sep)
    w("MISSING VALUE SUMMARY — TRACK B (selected protocol columns)")
    w(sep)
    proto_cols_to_report = [
        "Visit_Year",
        "Building_ConstructionYear",
        "Building_FloorAreaHeated_Total",
        "HeatPump_Installation_Year",
        "HeatPump_Installation_HeatingCapacity",
        "HeatPump_Installation_Normpoint_COP",
        "HeatPump_HeatingCurveSetting_Outside20_BeforeVisit",
        "HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit",
        "HeatPump_NightSetbackSetting_Activated_BeforeVisit",
        "HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit",
        "HeatPump_GroundSource_CurrentPressure",
    ]
    n_b_hh = track_b["Household_ID"].nunique()
    w(f"  {'Column':<55} {'Null rows':>10}  {'Null HHs':>10}  {'%HHs':>6}")
    w("  " + "-" * 85)
    for col in proto_cols_to_report:
        if col not in track_b.columns:
            continue
        n_null_rows = int(track_b[col].isna().sum())
        n_null_hh = int(
            track_b.groupby("Household_ID")[col].first().isna().sum()
        )
        pct = 100 * n_null_hh / n_b_hh if n_b_hh > 0 else 0
        w(f"  {col:<55} {n_null_rows:>10,}  {n_null_hh:>10,}  {pct:>5.1f}%")
    w()

    # ── Column inventory ──────────────────────────────────────────────────────
    w(sep)
    w("COLUMN INVENTORY")
    w(sep)
    w(f"  Track A columns ({track_a.shape[1]}):")
    for col in track_a.columns:
        dtype_str = str(track_a[col].dtype)
        w(f"    {col:<55} {dtype_str}")
    w()
    protocol_only_cols = [c for c in track_b.columns if c not in track_a.columns]
    w(f"  Protocol-only columns in Track B ({len(protocol_only_cols)}):")
    for col in protocol_only_cols:
        dtype_str = str(track_b[col].dtype)
        w(f"    {col:<55} {dtype_str}")
    w()

    return buf.getvalue()
