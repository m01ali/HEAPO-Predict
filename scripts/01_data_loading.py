#!/usr/bin/env python3
"""
scripts/01_data_loading.py

Phase 1 pipeline: load all HEAPO datasets, validate, profile, and save to parquet.

Run from project root:
    python scripts/01_data_loading.py

Outputs (data/processed/):
    smd_daily.parquet       — ~900k rows × 15 cols
    metadata.parquet        — survey variables
    protocols.parquet       — 410 rows with is_orphan + visit_number
    weather_daily.parquet   — 8 stations stacked
    households.parquet      — 1,408-row master mapping

    outputs/tables/phase1_profiling_report.txt
    outputs/logs/phase1_run.log     — full run log (all prints + warnings)
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    load_config,
    load_daily_overview,
    load_master_mapping,
    load_metadata,
    load_protocols,
    load_smd_daily,
    load_weather_daily,
    run_profiling,
)

# ---------------------------------------------------------------------------
# Tee — mirror stdout to a file so every print() is captured
# ---------------------------------------------------------------------------

class _Tee:
    """Write to both the original stream and a file simultaneously."""

    def __init__(self, original, file_handle):
        self._original = original
        self._fh = file_handle

    def write(self, data):
        self._original.write(data)
        self._fh.write(data)

    def flush(self):
        self._original.flush()
        self._fh.flush()

    def isatty(self):
        return self._original.isatty()


# ---------------------------------------------------------------------------
# Logging setup — console + file handler
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
LOG_DATEFMT = "%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
)
logger = logging.getLogger(__name__)

_log_file_handler: logging.FileHandler | None = None


def _setup_file_logging(log_path: Path) -> None:
    """Attach a FileHandler to the root logger so all log records go to disk."""
    global _log_file_handler
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    _log_file_handler.setLevel(logging.INFO)
    _log_file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    logging.getLogger().addHandler(_log_file_handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_parquet(df: pd.DataFrame, out_dir: Path, filename: str) -> None:
    path = out_dir / filename
    df.to_parquet(path, engine="pyarrow", index=True)
    size_mb = path.stat().st_size / 1024 ** 2
    logger.info("Saved %s  (%.1f MB)", path, size_mb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Config ────────────────────────────────────────────────────────────
    cfg = load_config("config/params.yaml")
    data_path = cfg["data"]["dataset_path"]
    min_days = cfg["data"]["min_days_threshold"]

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("outputs/tables")
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("outputs/logs/phase1_run.log")

    # Mirror all log records and print() output to the run log file
    _setup_file_logging(log_path)
    log_fh = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_fh)

    print("\n" + "=" * 70)
    print("  HEAPO-Predict  —  Phase 1 Data Loading")
    print("=" * 70)

    # ── 2. Master mapping and daily overview (Task 1.2) ───────────────────────
    print("\n[1/6] Loading master mapping and daily overview...")
    households = load_master_mapping(data_path)
    daily_overview = load_daily_overview(data_path)

    # Validate: all SMD household IDs are in master mapping
    smd_ids = set(daily_overview["Household_ID"].tolist())
    hh_ids = set(households.index.tolist())
    orphan_smd = smd_ids - hh_ids
    if orphan_smd:
        logger.warning(
            "%d household IDs in daily_overview not found in households.csv: %s",
            len(orphan_smd), sorted(orphan_smd)[:5],
        )
    else:
        logger.info("All 1,298 SMD household IDs are present in households.csv  ✓")

    # ── 3. SMD daily (Task 1.3) ───────────────────────────────────────────────
    print("\n[2/6] Loading smart meter data (daily, 1,298 files)...")
    weather_id_map = households["Weather_ID"]  # Series indexed by Household_ID
    smd_daily = load_smd_daily(
        data_path,
        household_ids=daily_overview["Household_ID"].tolist(),
        weather_id_map=weather_id_map,
    )

    # Validate Timestamp dtype (UTC-aware; pandas 3.x uses us precision, 2.x uses ns)
    ts_dtype = str(smd_daily["Timestamp"].dtype)
    assert "UTC" in ts_dtype and ts_dtype.startswith("datetime64"), (
        f"Timestamp dtype unexpected: {ts_dtype}"
    )
    n_null_total = smd_daily["kWh_received_Total"].isnull().sum()
    if n_null_total > 0:
        null_hh = smd_daily.loc[smd_daily["kWh_received_Total"].isnull(), "Household_ID"].unique()
        logger.warning(
            "kWh_received_Total has %d null rows across %d households: %s "
            "(spec states 'always present' — flag for Phase 2 cleaning)",
            n_null_total, len(null_hh), null_hh.tolist(),
        )
    assert "Weather_ID" in smd_daily.columns, "Weather_ID column missing from smd_daily"
    n_unique_hh = smd_daily["Household_ID"].nunique()
    assert n_unique_hh == 1298, f"Expected 1,298 households in smd_daily, got {n_unique_hh}"
    logger.info("smd_daily validation passed  ✓")

    # ── 4. Metadata (Task 1.4) ────────────────────────────────────────────────
    print("\n[3/6] Loading metadata...")
    metadata = load_metadata(data_path)
    assert len(metadata) <= 1408, f"metadata has {len(metadata)} rows > 1,408"
    logger.info("metadata validation passed  ✓")

    # ── 5. Protocols (Task 1.5) ───────────────────────────────────────────────
    print("\n[4/6] Loading protocols...")
    protocols = load_protocols(data_path)
    assert "is_orphan" in protocols.columns, "is_orphan column missing"
    assert "visit_number" in protocols.columns, "visit_number column missing"
    assert len(protocols) == 410, f"Expected 410 protocol rows, got {len(protocols)}"
    logger.info("protocols validation passed  ✓")

    # ── 6. Weather daily (Task 1.6) ───────────────────────────────────────────
    print("\n[5/6] Loading weather data (8 stations)...")
    weather_daily = load_weather_daily(data_path)
    assert weather_daily["Weather_ID"].nunique() == 8, (
        f"Expected 8 weather stations, got {weather_daily['Weather_ID'].nunique()}"
    )
    logger.info("weather_daily validation passed  ✓")

    # ── 7. Profiling (Task 1.7) ───────────────────────────────────────────────
    print("\n[6/6] Running data profiling...")
    report = run_profiling(
        {
            "households": households,
            "smd_daily": smd_daily,
            "metadata": metadata,
            "protocols": protocols,
            "weather_daily": weather_daily,
        },
        min_days_threshold=min_days,
    )

    report_path = report_dir / "phase1_profiling_report.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Profiling report saved to %s", report_path)
    print(report)

    # ── 8. Save to parquet ────────────────────────────────────────────────────
    print("\nSaving datasets to data/processed/ ...")
    _save_parquet(households, out_dir, "households.parquet")
    _save_parquet(smd_daily, out_dir, "smd_daily.parquet")
    _save_parquet(metadata, out_dir, "metadata.parquet")
    _save_parquet(protocols, out_dir, "protocols.parquet")
    _save_parquet(weather_daily, out_dir, "weather_daily.parquet")

    # ── 9. Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1 COMPLETE — Final Summary")
    print("=" * 70)
    print(f"  households      : {len(households):>10,} rows × {households.shape[1]} cols")
    print(f"  smd_daily       : {len(smd_daily):>10,} rows × {smd_daily.shape[1]} cols")
    print(f"  metadata        : {len(metadata):>10,} rows × {metadata.shape[1]} cols")
    print(f"  protocols       : {len(protocols):>10,} rows × {protocols.shape[1]} cols")
    print(f"  weather_daily   : {len(weather_daily):>10,} rows × {weather_daily.shape[1]} cols")
    print(f"\n  Unique SMD households    : {smd_daily['Household_ID'].nunique():,}")
    print(f"  Unique weather stations  : {weather_daily['Weather_ID'].nunique()}")
    print(f"  Orphan protocols         : {protocols['is_orphan'].sum()}")
    print(f"\n  Parquet files saved to   : {out_dir.resolve()}")
    print(f"  Profiling report saved to: {report_path.resolve()}")
    print(f"  Run log saved to         : {log_path.resolve()}")
    print("=" * 70 + "\n")

    # Restore stdout and close the log file handle
    sys.stdout = sys.stdout._original
    log_fh.close()


if __name__ == "__main__":
    main()
