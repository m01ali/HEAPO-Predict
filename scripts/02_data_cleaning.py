#!/usr/bin/env python3
"""
scripts/02_data_cleaning.py

Phase 2 pipeline: clean all HEAPO datasets and write cleaned parquets.

Run from project root:
    python scripts/02_data_cleaning.py

Inputs (data/processed/):
    smd_daily.parquet
    metadata.parquet
    protocols.parquet
    weather_daily.parquet

Outputs (data/processed/):
    smd_daily_clean.parquet     — cleaned SMD with Date, is_iqr_outlier, post_intervention, has_pv, has_reactive_energy
    metadata_clean.parquet      — cleaned survey metadata
    protocols_clean.parquet     — cleaned inspection protocols
    weather_daily_clean.parquet — cleaned weather with sunshine_available, interpolated_flag, temp_cross_station_flag

    outputs/tables/phase2_cleaning_report.txt
    outputs/logs/phase2_run.log
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_cleaner import (
    clean_metadata,
    clean_protocols,
    clean_smd,
    clean_weather,
    generate_cleaning_report,
)
from src.data_loader import load_config

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

def _load_parquet(processed_dir: Path, filename: str) -> pd.DataFrame:
    path = processed_dir / filename
    df = pd.read_parquet(path)
    logger.info("Loaded %s  (%d rows × %d cols)", path, len(df), df.shape[1])
    return df


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
    # Resolve all paths relative to the project root (parent of scripts/).
    # This makes the script runnable from any working directory.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    cfg = load_config(str(PROJECT_ROOT / "config/params.yaml"))

    processed_dir = PROJECT_ROOT / "data/processed"
    out_dir = PROJECT_ROOT / "data/processed"
    report_dir = PROJECT_ROOT / "outputs/tables"
    log_path = PROJECT_ROOT / "outputs/logs/phase2_run.log"

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    _setup_file_logging(log_path)

    logger.info("=" * 70)
    logger.info("Phase 2 — Data Cleaning")
    logger.info("=" * 70)

    # ── 2. Load raw parquets ─────────────────────────────────────────────────
    logger.info("\n--- Loading Phase 1 parquets ---")
    smd_raw = _load_parquet(processed_dir, "smd_daily.parquet")
    metadata_raw = _load_parquet(processed_dir, "metadata.parquet")
    protocols_raw = _load_parquet(processed_dir, "protocols.parquet")
    weather_raw = _load_parquet(processed_dir, "weather_daily.parquet")

    # ── 3. Clean SMD (Task 2.1) ───────────────────────────────────────────────
    logger.info("\n--- Task 2.1: Cleaning SMD ---")
    smd_clean, smd_stats = clean_smd(smd_raw, cfg)
    logger.info(
        "SMD clean: %d rows × %d cols | %d households",
        len(smd_clean),
        smd_clean.shape[1],
        smd_clean["Household_ID"].nunique(),
    )

    # ── 4. Clean Metadata (Task 2.2) ─────────────────────────────────────────
    logger.info("\n--- Task 2.2: Cleaning Metadata ---")
    metadata_clean, meta_stats = clean_metadata(metadata_raw, cfg)
    logger.info(
        "Metadata clean: %d rows × %d cols",
        len(metadata_clean),
        metadata_clean.shape[1],
    )

    # ── 5. Clean Protocols (Task 2.3) ────────────────────────────────────────
    logger.info("\n--- Task 2.3: Cleaning Protocols ---")
    protocols_clean, proto_stats = clean_protocols(protocols_raw, cfg)
    logger.info(
        "Protocols clean: %d rows × %d cols",
        len(protocols_clean),
        protocols_clean.shape[1],
    )

    # ── 6. Clean Weather (Task 2.4) ──────────────────────────────────────────
    logger.info("\n--- Task 2.4: Cleaning Weather ---")
    weather_clean, weather_stats = clean_weather(weather_raw, cfg)
    logger.info(
        "Weather clean: %d rows × %d cols",
        len(weather_clean),
        weather_clean.shape[1],
    )

    # ── 7. Save cleaned parquets ─────────────────────────────────────────────
    logger.info("\n--- Saving cleaned parquets ---")
    _save_parquet(smd_clean, out_dir, "smd_daily_clean.parquet")
    _save_parquet(metadata_clean, out_dir, "metadata_clean.parquet")
    _save_parquet(protocols_clean, out_dir, "protocols_clean.parquet")
    _save_parquet(weather_clean, out_dir, "weather_daily_clean.parquet")

    # ── 8. Generate and save cleaning report ─────────────────────────────────
    logger.info("\n--- Generating cleaning report ---")
    report_text = generate_cleaning_report(smd_stats, meta_stats, proto_stats, weather_stats)

    report_path = report_dir / "phase2_cleaning_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved cleaning report → %s", report_path)

    # ── 9. Final summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Phase 2 complete — summary")
    logger.info("=" * 70)
    logger.info(
        "SMD          : %9d rows  |  %4d households  |  %d cols",
        len(smd_clean),
        smd_clean["Household_ID"].nunique(),
        smd_clean.shape[1],
    )
    logger.info(
        "Metadata     : %9d rows  |  %d cols",
        len(metadata_clean),
        metadata_clean.shape[1],
    )
    logger.info(
        "Protocols    : %9d rows  |  %d cols",
        len(protocols_clean),
        protocols_clean.shape[1],
    )
    logger.info(
        "Weather      : %9d rows  |  %d cols",
        len(weather_clean),
        weather_clean.shape[1],
    )
    logger.info("Log          → %s", log_path)
    logger.info("Report       → %s", report_path)

    # Print to stdout for quick terminal feedback
    print("\nPhase 2 done.")
    print(f"  SMD        : {len(smd_clean):,} rows, {smd_clean['Household_ID'].nunique()} households")
    print(f"  Metadata   : {len(metadata_clean):,} rows")
    print(f"  Protocols  : {len(protocols_clean):,} rows")
    print(f"  Weather    : {len(weather_clean):,} rows")
    print(f"  Log        → {log_path}")
    print(f"  Report     → {report_path}")


if __name__ == "__main__":
    main()
