#!/usr/bin/env python3
"""
scripts/04_feature_engineering.py

Phase 4 pipeline: engineer features from Phase 3 merged parquets.

Run from project root:
    python scripts/04_feature_engineering.py

Inputs (data/processed/):
    merged_full.parquet         ← Track A: 913,620 rows × 47 cols (1,272 households)
    merged_protocol.parquet     ← Track B:  84,367 rows × 110 cols (152 households)

Outputs (data/processed/):
    features_full.parquet       ← Track A with all engineered features
    features_protocol.parquet   ← Track B with all engineered features

    outputs/tables/phase4_feature_report.txt   ← feature catalog + integrity summary
    outputs/logs/phase4_run.log
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_config
from src.feature_engineer import (
    add_autoregressive_features,
    add_household_features,
    add_protocol_features,
    add_reactive_energy_features,
    add_temporal_features,
    add_weather_features,
    check_forbidden_columns,
    encode_categoricals,
    generate_feature_report,
    load_phase3_artifacts,
    run_integrity_checks,
)

# ---------------------------------------------------------------------------
# Logging setup — console + file handler (mirrors Phase 2/3 pattern)
# ---------------------------------------------------------------------------

LOG_FORMAT  = "%(asctime)s  %(levelname)-8s  %(message)s"
LOG_DATEFMT = "%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
)
logger = logging.getLogger(__name__)

_log_file_handler: logging.FileHandler | None = None


def _setup_file_logging(log_path: Path) -> None:
    """Attach a FileHandler to the root logger so all records go to disk."""
    global _log_file_handler
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    _log_file_handler.setLevel(logging.INFO)
    _log_file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    logging.getLogger().addHandler(_log_file_handler)


def _save_parquet(df: pd.DataFrame, out_dir: Path, filename: str) -> None:
    path = out_dir / filename
    df.to_parquet(path, engine="pyarrow", index=False)
    size_mb = path.stat().st_size / 1024 ** 2
    logger.info("Saved %-50s (%.1f MB)", str(path), size_mb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Config and paths ───────────────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    cfg = load_config(str(PROJECT_ROOT / "config/params.yaml"))
    fe_cfg = cfg.get("feature_engineering", {})

    processed_dir = PROJECT_ROOT / "data/processed"
    out_dir       = PROJECT_ROOT / "data/processed"
    report_dir    = PROJECT_ROOT / "outputs/tables"
    log_path      = PROJECT_ROOT / "outputs/logs/phase4_run.log"

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    _setup_file_logging(log_path)

    logger.info("=" * 70)
    logger.info("Phase 4 — Feature Engineering")
    logger.info("=" * 70)

    all_stats_a: dict = {}
    all_stats_b: dict = {}

    # ── Task 4.0: Load Phase 3 artifacts ─────────────────────────────────────
    logger.info("\n--- Task 4.0: Loading Phase 3 artifacts ---")
    track_a, track_b = load_phase3_artifacts(processed_dir)

    # ── Task 4.1: Temporal features (both tracks) ─────────────────────────────
    logger.info("\n--- Task 4.1: Adding temporal features ---")
    track_a, stats = add_temporal_features(track_a)
    all_stats_a.update(stats)

    track_b, stats = add_temporal_features(track_b)
    all_stats_b.update(stats)

    # ── Task 4.2: Weather-derived features (both tracks) ──────────────────────
    # NOTE: this step also sorts both frames by [Household_ID, Date].
    # All subsequent operations preserve that sort order.
    logger.info("\n--- Task 4.2: Adding weather-derived features ---")
    track_a, stats = add_weather_features(track_a, cfg)
    all_stats_a.update(stats)

    track_b, stats = add_weather_features(track_b, cfg)
    all_stats_b.update(stats)

    # ── Task 4.3: Household static features (both tracks) ─────────────────────
    logger.info("\n--- Task 4.3: Adding household static features ---")
    track_a, stats = add_household_features(track_a)
    all_stats_a.update(stats)

    track_b, stats = add_household_features(track_b)
    all_stats_b.update(stats)

    # ── Task 4.4: Protocol-enriched features (Track B only) ───────────────────
    logger.info("\n--- Task 4.4: Adding protocol-enriched features (Track B only) ---")
    track_b, stats = add_protocol_features(track_b)
    all_stats_b.update(stats)

    # ── Task 4.5: Reactive energy features (optional) ─────────────────────────
    if fe_cfg.get("include_reactive_energy", True):
        logger.info("\n--- Task 4.5: Adding reactive energy features ---")
        track_a, stats = add_reactive_energy_features(track_a)
        all_stats_a.update(stats)

        track_b, stats = add_reactive_energy_features(track_b)
        all_stats_b.update(stats)
    else:
        logger.info(
            "\n--- Task 4.5: Reactive energy features SKIPPED "
            "(feature_engineering.include_reactive_energy = false) ---"
        )

    # ── Task 4.6: Autoregressive features (optional, off by default) ──────────
    if fe_cfg.get("include_autoregressive", False):
        logger.info("\n--- Task 4.6: Adding autoregressive features ---")
        track_a, stats = add_autoregressive_features(track_a)
        all_stats_a.update(stats)

        track_b, stats = add_autoregressive_features(track_b)
        all_stats_b.update(stats)

        logger.warning(
            "Autoregressive features ENABLED — Phase 6 MUST use a strictly temporal "
            "train/test split. These features are for sensitivity analysis only."
        )
    else:
        logger.info(
            "\n--- Task 4.6: Autoregressive features SKIPPED "
            "(feature_engineering.include_autoregressive = false) ---"
        )

    # ── Task 4.7: Forbidden column guard ──────────────────────────────────────
    logger.info("\n--- Task 4.7: Forbidden column guard ---")
    check_forbidden_columns(track_a, track_b)

    # ── Task 4.8: Categorical encoding ────────────────────────────────────────
    logger.info("\n--- Task 4.8: Encoding categorical features ---")
    track_a, stats = encode_categoricals(track_a)
    all_stats_a.update(stats)

    track_b, stats = encode_categoricals(track_b)
    all_stats_b.update(stats)

    # ── Save outputs ──────────────────────────────────────────────────────────
    logger.info("\n--- Saving feature parquets ---")
    _save_parquet(track_a, out_dir, "features_full.parquet")
    _save_parquet(track_b, out_dir, "features_protocol.parquet")

    # ── Task 4.9: Integrity checks ────────────────────────────────────────────
    logger.info("\n--- Task 4.9: Running integrity checks ---")
    run_integrity_checks(track_a, track_b)

    # ── Task 4.9: Feature catalog report ──────────────────────────────────────
    logger.info("\n--- Task 4.9: Generating feature report ---")
    report_text = generate_feature_report(track_a, track_b, all_stats_a, all_stats_b)

    report_path = report_dir / "phase4_feature_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved feature report → %s", report_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Phase 4 complete — summary")
    logger.info("=" * 70)
    logger.info(
        "Track A (features_full)     : %9d rows | %4d households | %d cols",
        len(track_a), track_a["Household_ID"].nunique(), track_a.shape[1],
    )
    logger.info(
        "Track B (features_protocol) : %9d rows | %4d households | %d cols",
        len(track_b), track_b["Household_ID"].nunique(), track_b.shape[1],
    )
    logger.info("Log    → %s", log_path)
    logger.info("Report → %s", report_path)

    print("\nPhase 4 done.")
    print(f"  Track A : {len(track_a):>10,} rows"
          f" | {track_a['Household_ID'].nunique()} households"
          f" | {track_a.shape[1]} cols")
    print(f"  Track B : {len(track_b):>10,} rows"
          f" | {track_b['Household_ID'].nunique()} households"
          f" | {track_b.shape[1]} cols")
    print(f"  Log     → {log_path}")
    print(f"  Report  → {report_path}")


if __name__ == "__main__":
    main()
