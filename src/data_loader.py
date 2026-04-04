"""
src/data_loader.py

Loading utilities for the HEAPO dataset.
Phase 1 implementation — Tasks 1.2 through 1.7.

All column names verified against:
  - Table 1 (SMD), Table 4 (Metadata), Table 5 (Protocols), Table 6 (Weather)
  in Brudermueller et al. (2025), arXiv:2503.16993v1
  and against actual files in heapo_data/ on 2026-04-04.
"""

import ast
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Station IDs verified by directory listing of heapo_data/weather_data/daily/
WEATHER_STATION_IDS = ["8jB", "Hg", "wDD", "z6I", "HbsbG", "ceOxS", "sV3mR", "MqO"]

# Expected columns per dataset (Tables 1, 4, 5, 6 of paper)
_SMD_DAILY_COLS = [
    "Household_ID", "Group", "AffectsTimePoint", "Timestamp",
    "kWh_received_Total", "kWh_received_HeatPump", "kWh_received_Other",
    "kWh_returned_Total",
    "kvarh_received_capacitive_Total", "kvarh_received_capacitive_HeatPump",
    "kvarh_received_capacitive_Other",
    "kvarh_received_inductive_Total", "kvarh_received_inductive_HeatPump",
    "kvarh_received_inductive_Other",
]

_WEATHER_DAILY_COLS = [
    "Weather_ID", "Timestamp",
    "Temperature_max_daily", "Temperature_min_daily", "Temperature_avg_daily",
    "HeatingDegree_SIA_daily", "HeatingDegree_US_daily", "CoolingDegree_US_daily",
    "Humidity_avg_daily", "Precipitation_total_daily", "Sunshine_duration_daily",
]

_METADATA_COLS = [
    "Household_ID",
    "Survey_Building_Type",                        # contains "appartment" typo — NOT fixed here
    "Survey_Building_LivingArea",
    "Survey_Building_Residents",
    "Survey_HeatPump_Installation_Type",
    "Survey_HeatDistribution_System_FloorHeating",
    "Survey_HeatDistribution_System_Radiator",
    "Survey_DHW_Production_ByHeatPump",
    "Survey_DHW_Production_ByElectricWaterHeater",
    "Survey_DHW_Production_BySolar",
    "Survey_Installation_HasDryer",
    "Survey_Installation_HasFreezer",
    "Survey_Installation_HasElectricVehicle",
]


# ---------------------------------------------------------------------------
# 1. Config helper
# ---------------------------------------------------------------------------

def load_config(path: str = "config/params.yaml") -> dict:
    """Load YAML config from the given path."""
    cfg_path = Path(path)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 2. Master mapping and overview (Task 1.2)
# ---------------------------------------------------------------------------

def load_master_mapping(data_path: str) -> pd.DataFrame:
    """
    Load households.csv — master table of all 1,408 households.

    Returns DataFrame indexed by Household_ID.
    Coerces Installation_HasPVSystem from object dtype to bool.
    """
    path = Path(data_path) / "meta_data" / "households.csv"
    df = pd.read_csv(path, sep=";")

    # Coerce Installation_HasPVSystem (stored as string "True"/"False") to bool
    df["Installation_HasPVSystem"] = df["Installation_HasPVSystem"].map(
        {"True": True, "False": False, True: True, False: False}
    ).astype("boolean")

    df = df.set_index("Household_ID")

    assert len(df) == 1408, f"Expected 1,408 households, got {len(df)}"
    logger.info("households: %d rows × %d cols", len(df), df.shape[1])
    return df


def load_daily_overview(data_path: str) -> pd.DataFrame:
    """
    Load smart_meter_data_daily_overview.csv.

    Returns DataFrame with 1,298 rows — the universe of valid daily-SMD household IDs.
    """
    path = (
        Path(data_path)
        / "smart_meter_data"
        / "overview"
        / "smart_meter_data_daily_overview.csv"
    )
    df = pd.read_csv(path, sep=";")

    assert len(df) == 1298, f"Expected 1,298 rows in daily overview, got {len(df)}"
    logger.info("daily_overview: %d rows × %d cols", len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# 3. Smart Meter Data — daily resolution (Task 1.3)
# ---------------------------------------------------------------------------

def _check_dual_meter(df: pd.DataFrame, hid: int) -> None:
    """
    Warn (do NOT raise) if kWh_received_HeatPump + kWh_received_Other
    deviates from kWh_received_Total by more than 0.01 kWh for any row.
    """
    mask = df["kWh_received_HeatPump"].notna() & df["kWh_received_Other"].notna()
    if not mask.any():
        return
    computed = df.loc[mask, "kWh_received_HeatPump"] + df.loc[mask, "kWh_received_Other"]
    discrepancy = (df.loc[mask, "kWh_received_Total"] - computed).abs()
    max_diff = discrepancy.max()
    if max_diff > 0.01:
        warnings.warn(
            f"Dual-meter mismatch for household {hid}: max diff = {max_diff:.4f} kWh",
            stacklevel=2,
        )


def load_smd_daily(
    data_path: str,
    household_ids: list[int],
    weather_id_map: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Load and concatenate daily SMD for all given household IDs.

    Runs dual-meter consistency check on each dual-meter household.
    Optionally joins Weather_ID from weather_id_map (Series indexed by Household_ID).
    Returns combined DataFrame; Timestamp is UTC-aware datetime64.

    Parameters
    ----------
    data_path : str
        Root path of the HEAPO dataset (heapo_data/).
    household_ids : list[int]
        IDs to load — must be the 1,298 from the daily overview.
    weather_id_map : pd.Series, optional
        Series mapping Household_ID → Weather_ID (from households DataFrame).
    """
    daily_dir = Path(data_path) / "smart_meter_data" / "daily"
    chunks: list[pd.DataFrame] = []
    missing: list[int] = []

    for hid in tqdm(household_ids, desc="Loading SMD daily", unit="hh"):
        csv_path = daily_dir / f"{hid}.csv"
        if not csv_path.exists():
            missing.append(hid)
            logger.warning("Missing daily file for household %d", hid)
            continue

        hdf = pd.read_csv(
            csv_path,
            sep=";",
            dtype={
                "Household_ID": "int64",
                "kWh_received_Total": "float64",
                "kWh_received_HeatPump": "float64",
                "kWh_received_Other": "float64",
                "kWh_returned_Total": "float64",
                "kvarh_received_capacitive_Total": "float64",
                "kvarh_received_capacitive_HeatPump": "float64",
                "kvarh_received_capacitive_Other": "float64",
                "kvarh_received_inductive_Total": "float64",
                "kvarh_received_inductive_HeatPump": "float64",
                "kvarh_received_inductive_Other": "float64",
            },
        )
        hdf["Timestamp"] = pd.to_datetime(hdf["Timestamp"], utc=True)
        _check_dual_meter(hdf, hid)
        chunks.append(hdf)

    if missing:
        logger.warning("%d household files not found: %s", len(missing), missing[:10])

    df = pd.concat(chunks, ignore_index=True)

    # Join Weather_ID from master mapping
    if weather_id_map is not None:
        df["Weather_ID"] = df["Household_ID"].map(weather_id_map)
    else:
        logger.warning("weather_id_map not provided — Weather_ID column not added")

    # Log AffectsTimePoint distribution
    atp_counts = df["AffectsTimePoint"].value_counts()
    logger.info("AffectsTimePoint distribution:\n%s", atp_counts.to_string())

    # Log Group distribution
    group_counts = df["Group"].value_counts()
    logger.info("Group distribution:\n%s", group_counts.to_string())

    n_hh = df["Household_ID"].nunique()
    logger.info(
        "smd_daily loaded: %d rows × %d cols — %d unique households",
        len(df), df.shape[1], n_hh,
    )
    return df


# ---------------------------------------------------------------------------
# 4. Metadata (Task 1.4)
# ---------------------------------------------------------------------------

def load_metadata(data_path: str) -> pd.DataFrame:
    """
    Load meta_data.csv — survey variables for up to 1,358 households.

    Returns DataFrame. Does NOT fix the 'appartment' typo — flagged for Phase 2.
    NOTE: Survey_Building_Type contains "appartment" (double-t). This is a known
    dataset typo per Table 4 of the paper. It is intentionally preserved here
    so the Phase 2 cleaning log can document the standardisation explicitly.
    """
    path = Path(data_path) / "meta_data" / "meta_data.csv"
    df = pd.read_csv(path, sep=";")

    # Log unique values of Survey_Building_Type — confirm the known typo
    if "Survey_Building_Type" in df.columns:
        unique_types = df["Survey_Building_Type"].unique()
        logger.info(
            "Survey_Building_Type unique values: %s  "
            "(NOTE: 'appartment' is a known typo — NOT fixed at load time)",
            unique_types,
        )

    logger.info("metadata: %d rows × %d cols", len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# 5. Protocol / Inspection Data (Task 1.5)
# ---------------------------------------------------------------------------

def load_protocols(data_path: str) -> pd.DataFrame:
    """
    Load protocols.csv (410 rows × 106 cols).

    Adds:
      - is_orphan (bool): True for the 193 rows where Household_ID is null
      - visit_number (int): per-household sequential visit count, sorted by Visit_Date

    Returns full DataFrame including orphans.

    NOTE: Paper states 196 null Household_ID rows; actual data has 193. Discrepancy logged.
    """
    path = Path(data_path) / "reports" / "protocols.csv"
    df = pd.read_csv(path, sep=";", low_memory=False)

    # Add orphan flag (before any date parsing that might fail on orphan rows)
    df["is_orphan"] = df["Household_ID"].isnull()
    n_orphans = df["is_orphan"].sum()

    if n_orphans != 196:
        logger.warning(
            "DISCREPANCY: paper states 196 orphan protocols (null Household_ID), "
            "actual data has %d. Proceeding with observed count.",
            n_orphans,
        )
    else:
        logger.info("Orphan protocol count matches paper: %d", n_orphans)

    # Parse Visit_Date to datetime
    if "Visit_Date" in df.columns:
        df["Visit_Date"] = pd.to_datetime(df["Visit_Date"], errors="coerce")

    # Add visit_number per household, sorted by Visit_Date ascending.
    # Orphans (null Household_ID) get visit_number = NaN.
    df["visit_number"] = np.nan
    linked = df[~df["is_orphan"]].copy()
    linked = linked.sort_values(["Household_ID", "Visit_Date"])
    linked["visit_number"] = linked.groupby("Household_ID").cumcount() + 1
    df.loc[linked.index, "visit_number"] = linked["visit_number"]
    df["visit_number"] = df["visit_number"].astype("Int64")  # nullable int

    # Log multiple-visit households
    multi_visit = linked[linked["visit_number"] > 1]["Household_ID"].unique()
    if len(multi_visit) > 0:
        logger.info(
            "Multiple-visit households (%d): %s",
            len(multi_visit), sorted(multi_visit.tolist()),
        )
    else:
        logger.warning("No multiple-visit households found — expected 3")

    logger.info(
        "protocols: %d rows × %d cols — %d orphans, %d linked",
        len(df), df.shape[1], n_orphans, (~df["is_orphan"]).sum(),
    )
    return df


# ---------------------------------------------------------------------------
# 6. Weather Data — daily resolution (Task 1.6)
# ---------------------------------------------------------------------------

def load_weather_daily(data_path: str) -> pd.DataFrame:
    """
    Load and concatenate daily weather data from all 8 stations.

    Returns stacked DataFrame with Weather_ID column.
    Timestamp parsed as date (not datetime — daily resolution).
    """
    weather_dir = Path(data_path) / "weather_data" / "daily"
    chunks: list[pd.DataFrame] = []

    for station_id in WEATHER_STATION_IDS:
        csv_path = weather_dir / f"{station_id}.csv"
        if not csv_path.exists():
            logger.warning("Weather file not found for station %s: %s", station_id, csv_path)
            continue

        wdf = pd.read_csv(csv_path, sep=";")
        # Parse Timestamp as date only (daily resolution — not datetime)
        wdf["Timestamp"] = pd.to_datetime(wdf["Timestamp"], utc=False).dt.date

        # Log date range and any missing columns
        expected_cols = set(_WEATHER_DAILY_COLS)
        actual_cols = set(wdf.columns)
        missing_cols = expected_cols - actual_cols
        if missing_cols:
            logger.warning("Station %s missing columns: %s", station_id, missing_cols)

        date_min = wdf["Timestamp"].min()
        date_max = wdf["Timestamp"].max()
        logger.info(
            "Station %s: %d rows, dates %s to %s",
            station_id, len(wdf), date_min, date_max,
        )
        chunks.append(wdf)

    df = pd.concat(chunks, ignore_index=True)

    unique_stations = df["Weather_ID"].nunique()
    logger.info(
        "weather_daily: %d rows × %d cols — %d unique stations",
        len(df), df.shape[1], unique_stations,
    )
    assert unique_stations == 8, f"Expected 8 weather stations, got {unique_stations}"
    return df


# ---------------------------------------------------------------------------
# 7. Data Profiling (Task 1.7)
# ---------------------------------------------------------------------------

def profile_dataset(df: pd.DataFrame, name: str, extra_info: str = "") -> str:
    """
    Generate a profiling summary for a single DataFrame.

    Returns a human-readable string. Also logs to logger.
    """
    lines: list[str] = []
    sep = "=" * 70

    lines.append(sep)
    lines.append(f"DATASET: {name}")
    lines.append(sep)

    # Shape and memory
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    lines.append(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    lines.append(f"Memory usage   : {mem_mb:.2f} MB")
    lines.append("")

    # Dtypes
    lines.append("--- Dtypes ---")
    for col, dtype in df.dtypes.items():
        lines.append(f"  {col:<55} {dtype}")
    lines.append("")

    # Missing values (only columns with >0 missing)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )
    if len(missing_df) > 0:
        lines.append(f"--- Missing Values ({len(missing_df)} columns with >0 missing) ---")
        for col, row in missing_df.iterrows():
            lines.append(
                f"  {col:<55} {int(row['missing_count']):>8,}  ({row['missing_pct']:.1f}%)"
            )
    else:
        lines.append("--- Missing Values: none ---")
    lines.append("")

    # Numeric distributions for key numeric columns
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        lines.append("--- Numeric Distributions ---")
        desc = df[num_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
        lines.append(desc.to_string())
        lines.append("")

    # Cardinality for object/string columns
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if cat_cols:
        lines.append("--- Categorical Cardinality ---")
        for col in cat_cols:
            n_unique = df[col].nunique()
            top = df[col].value_counts().head(5).to_dict()
            lines.append(f"  {col}: {n_unique} unique — top 5: {top}")
        lines.append("")

    # Extra dataset-specific info
    if extra_info:
        lines.append("--- Dataset-Specific Notes ---")
        lines.append(extra_info)
        lines.append("")

    return "\n".join(lines)


def _profile_smd_extra(df: pd.DataFrame, min_days: int) -> str:
    """Build the SMD-specific profiling block."""
    lines = []
    hh_days = df.groupby("Household_ID").size()
    lines.append(f"Total rows            : {len(df):,}")
    lines.append(f"Unique households     : {df['Household_ID'].nunique():,}")
    lines.append(f"Days per household    : min={hh_days.min()}, median={hh_days.median():.0f}, max={hh_days.max()}")
    below_thresh = (hh_days < min_days).sum()
    lines.append(f"Households below {min_days}-day threshold: {below_thresh}")
    lines.append(f"\nAffectsTimePoint distribution:\n{df['AffectsTimePoint'].value_counts().to_string()}")
    lines.append(f"\nGroup distribution:\n{df['Group'].value_counts().to_string()}")
    pv_hh = (df.groupby("Household_ID")["kWh_returned_Total"].max() > 0).sum()
    lines.append(f"\nHouseholds with kWh_returned_Total > 0 (PV proxy): {pv_hh}")
    dual_hh = df.groupby("Household_ID")["kWh_received_HeatPump"].apply(
        lambda s: s.notna().any()
    ).sum()
    lines.append(f"Dual-meter households (non-null kWh_received_HeatPump): {dual_hh}")
    return "\n".join(lines)


def _profile_metadata_extra(df: pd.DataFrame, smd_ids: set) -> str:
    lines = []
    meta_ids = set(df["Household_ID"].unique()) if "Household_ID" in df.columns else set()
    overlap = len(meta_ids & smd_ids)
    pct = overlap / len(smd_ids) * 100 if smd_ids else 0
    lines.append(f"SMD households with metadata: {overlap} / {len(smd_ids)} ({pct:.1f}%)")
    if "Survey_Building_Type" in df.columns:
        lines.append(f"\nSurvey_Building_Type values (note 'appartment' typo):\n"
                     f"{df['Survey_Building_Type'].value_counts().to_string()}")
    if "Survey_HeatPump_Installation_Type" in df.columns:
        lines.append(f"\nSurvey_HeatPump_Installation_Type:\n"
                     f"{df['Survey_HeatPump_Installation_Type'].value_counts().to_string()}")
    return "\n".join(lines)


def _profile_protocols_extra(df: pd.DataFrame) -> str:
    lines = []
    n_orphan = df["is_orphan"].sum() if "is_orphan" in df.columns else "N/A"
    n_linked = (~df["is_orphan"]).sum() if "is_orphan" in df.columns else "N/A"
    lines.append(f"Orphan reports (no Household_ID) : {n_orphan}")
    lines.append(f"Linked reports                   : {n_linked}")
    if "Visit_Year" in df.columns:
        lines.append(f"\nVisit_Year distribution:\n{df['Visit_Year'].value_counts().sort_index().to_string()}")
    if "visit_number" in df.columns:
        multi = df[df["visit_number"] > 1]["Household_ID"].dropna().unique()
        lines.append(f"\nMultiple-visit household IDs: {sorted(multi.tolist())}")
    return "\n".join(lines)


def _profile_weather_extra(df: pd.DataFrame) -> str:
    lines = []
    for station in df["Weather_ID"].unique():
        sub = df[df["Weather_ID"] == station]
        date_min = sub["Timestamp"].min()
        date_max = sub["Timestamp"].max()
        missing = sub.isnull().sum().sum()
        lines.append(f"  {station}: {len(sub):,} days, {date_min} to {date_max}, {missing} missing values")
    if "Temperature_avg_daily" in df.columns:
        t_min = df["Temperature_avg_daily"].min()
        t_max = df["Temperature_avg_daily"].max()
        lines.append(f"\nTemperature_avg_daily range: {t_min:.1f}°C to {t_max:.1f}°C")
    return "\n".join(lines)


def _profile_households_extra(df: pd.DataFrame) -> str:
    lines = []
    if "Group" in df.columns:
        lines.append(f"Group (treatment/control):\n{df['Group'].value_counts().to_string()}")
    if "Installation_HasPVSystem" in df.columns:
        lines.append(f"\nInstallation_HasPVSystem:\n{df['Installation_HasPVSystem'].value_counts().to_string()}")
    for col in ["SmartMeterData_Available_Daily", "SmartMeterData_Available_15min",
                "MetaData_Available", "Protocols_Available"]:
        if col in df.columns:
            lines.append(f"\n{col}:\n{df[col].value_counts().to_string()}")
    return "\n".join(lines)


def run_profiling(
    datasets: dict,
    min_days_threshold: int = 180,
) -> str:
    """
    Run profiling for all 5 datasets and return the full report as a string.

    Parameters
    ----------
    datasets : dict with keys 'households', 'smd_daily', 'metadata', 'protocols', 'weather_daily'
    min_days_threshold : int
        From config — households below this day count flagged in SMD profiling.
    """
    smd = datasets.get("smd_daily")
    smd_ids = set(smd["Household_ID"].unique()) if smd is not None else set()

    # households DataFrame may be indexed by Household_ID — reset for profiling
    hh_df = datasets.get("households")
    if hh_df is not None and hh_df.index.name == "Household_ID":
        hh_df_reset = hh_df.reset_index()
    else:
        hh_df_reset = hh_df

    sections = []

    if hh_df_reset is not None:
        extra = _profile_households_extra(hh_df_reset)
        sections.append(profile_dataset(hh_df_reset, "HOUSEHOLDS (master mapping)", extra))

    if smd is not None:
        extra = _profile_smd_extra(smd, min_days_threshold)
        sections.append(profile_dataset(smd, "SMD DAILY", extra))

    metadata = datasets.get("metadata")
    if metadata is not None:
        extra = _profile_metadata_extra(metadata, smd_ids)
        sections.append(profile_dataset(metadata, "METADATA (survey)", extra))

    protocols = datasets.get("protocols")
    if protocols is not None:
        extra = _profile_protocols_extra(protocols)
        sections.append(profile_dataset(protocols, "PROTOCOLS (inspection)", extra))

    weather = datasets.get("weather_daily")
    if weather is not None:
        extra = _profile_weather_extra(weather)
        sections.append(profile_dataset(weather, "WEATHER DAILY", extra))

    return "\n\n".join(sections)
