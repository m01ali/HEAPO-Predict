#!/usr/bin/env python3
"""
scripts/03_data_merging.py

Phase 3 pipeline: merge all cleaned HEAPO datasets and write two analysis-track parquets.

Run from project root:
    python scripts/03_data_merging.py

Inputs (data/processed/):
    smd_daily_clean.parquet
    metadata_clean.parquet
    protocols_clean.parquet
    weather_daily_clean.parquet
    weather_daily.parquet          ← raw (to recover Weather_ID dropped in Phase 2)
    households.parquet

Outputs (data/processed/):
    merged_full.parquet            ← Track A: all ~1,272 households, ~35 cols
    merged_protocol.parquet        ← Track B: ~217 treatment households, ~65 cols

    outputs/tables/phase3_merge_report.txt
    outputs/logs/phase3_run.log

Phase 4 (Feature Engineering) reads merged_full.parquet and merged_protocol.parquet
directly. No intermediate reshaping is needed between phases.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_config
from src.data_merger import (
    build_protocol_map,
    generate_merge_report,
    load_phase2_artifacts,
    merge_metadata,
    merge_protocols,
    merge_smd_weather,
    run_integrity_checks,
)

# ---------------------------------------------------------------------------
# Logging setup — mirrors Phase 2 pattern (console + file handler)
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
    df.to_parquet(path, engine="pyarrow", index=False)
    size_mb = path.stat().st_size / 1024 ** 2
    logger.info("Saved %-45s (%.1f MB)", str(path), size_mb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Config and paths ───────────────────────────────────────────────────
    # Resolve all paths relative to the project root (parent of scripts/).
    # This makes the script runnable from any working directory.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    cfg = load_config(str(PROJECT_ROOT / "config/params.yaml"))

    processed_dir = PROJECT_ROOT / "data/processed"
    out_dir       = PROJECT_ROOT / "data/processed"
    report_dir    = PROJECT_ROOT / "outputs/tables"
    log_path      = PROJECT_ROOT / "outputs/logs/phase3_run.log"

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    _setup_file_logging(log_path)

    logger.info("=" * 70)
    logger.info("Phase 3 — Data Merging")
    logger.info("=" * 70)

    # ── 2. Load and verify Phase 2 artifacts (Task 3.0) ──────────────────────
    logger.info("\n--- Task 3.0: Loading Phase 2 artifacts ---")
    artifacts = load_phase2_artifacts(processed_dir)
    smd       = artifacts["smd"]
    metadata  = artifacts["metadata"]
    protocols = artifacts["protocols"]
    weather   = artifacts["weather"]
    households = artifacts["households"]

    all_stats: dict = {}

    # ── 3. Merge SMD + Weather (Task 3.2) ─────────────────────────────────────
    logger.info("\n--- Task 3.2: Merging SMD with Weather ---")
    smd_weather, weather_stats = merge_smd_weather(smd, weather)
    all_stats.update(weather_stats)

    # ── 4. Merge with Metadata → Track A (Task 3.3) ──────────────────────────
    logger.info("\n--- Task 3.3: Merging with Metadata (Track A) ---")
    track_a, meta_stats = merge_metadata(smd_weather, metadata)
    all_stats.update(meta_stats)

    # ── 5. Save Track A ───────────────────────────────────────────────────────
    logger.info("\n--- Saving Track A ---")
    _save_parquet(track_a, out_dir, "merged_full.parquet")

    # ── 6. Build protocol map and merge → Track B (Task 3.4) ─────────────────
    logger.info("\n--- Task 3.4: Building protocol map ---")
    protocol_per_hh = build_protocol_map(households, protocols)

    logger.info("\n--- Task 3.4: Merging protocols (Track B) ---")
    track_b, proto_stats = merge_protocols(track_a, protocol_per_hh)
    all_stats.update(proto_stats)

    # ── 7. Save Track B ───────────────────────────────────────────────────────
    logger.info("\n--- Saving Track B ---")
    _save_parquet(track_b, out_dir, "merged_protocol.parquet")

    # ── 8. Integrity checks (Task 3.5) ────────────────────────────────────────
    logger.info("\n--- Task 3.5: Running integrity checks ---")
    run_integrity_checks(track_a, track_b)

    # ── 9. Generate and save merge report (Task 3.7) ─────────────────────────
    logger.info("\n--- Task 3.7: Generating merge report ---")
    report_text = generate_merge_report(track_a, track_b, all_stats)

    report_path = report_dir / "phase3_merge_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved merge report → %s", report_path)

    # ── 10. Final summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Phase 3 complete — summary")
    logger.info("=" * 70)
    logger.info(
        "Track A (merged_full)      : %9d rows | %4d households | %d cols",
        len(track_a), track_a["Household_ID"].nunique(), track_a.shape[1],
    )
    logger.info(
        "Track B (merged_protocol)  : %9d rows | %4d households | %d cols",
        len(track_b), track_b["Household_ID"].nunique(), track_b.shape[1],
    )
    logger.info("Log    → %s", log_path)
    logger.info("Report → %s", report_path)

    # Print to stdout for quick terminal feedback
    print("\nPhase 3 done.")
    print(f"  Track A : {len(track_a):>10,} rows | {track_a['Household_ID'].nunique()} households | {track_a.shape[1]} cols")
    print(f"  Track B : {len(track_b):>10,} rows | {track_b['Household_ID'].nunique()} households | {track_b.shape[1]} cols")
    print(f"  Log     → {log_path}")
    print(f"  Report  → {report_path}")


if __name__ == "__main__":
    main()
