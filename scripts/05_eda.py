"""
scripts/05_eda.py

Phase 5 — Exploratory Data Analysis orchestration.
Runs all EDA tasks in order and writes outputs to:
  outputs/figures/   — all PNG plots
  outputs/tables/    — phase5_eda_summary.txt, phase5_vif_table.txt
  outputs/logs/      — phase5_run.log
"""

import logging
import os
import sys
from pathlib import Path

# ── Ensure project root on sys.path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.data_loader import load_config
from src.eda import (
    task50_load,
    task51_target_analysis,
    task52_affectstimepoint_and_groups,
    task53_univariate_distributions,
    task54_bivariate_feature_target,
    task55_correlation_and_vif,
    task56_temporal_patterns,
    task57_missing_data,
    task58_subgroup_comparisons,
    task59_protocol_eda,
    task510_write_summary_report,
    task511_integrity_checks,
)


def setup_logging() -> logging.Logger:
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "phase5_run.log"

    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging to %s", log_path)
    return logger


def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("HEAPO-Predict — Phase 5: Exploratory Data Analysis")
    logger.info("=" * 60)

    cfg = load_config("config/params.yaml")

    # Task 5.0 — Load artifacts
    df_a, df_b, df_wx = task50_load(cfg)

    # Accumulate results for summary report
    all_results = {}

    # Task 5.1 — Target variable analysis
    r = task51_target_analysis(df_a, df_wx, cfg)
    all_results.update(r)

    # Task 5.2 — AffectsTimePoint and treatment/control
    r = task52_affectstimepoint_and_groups(df_a, df_wx, cfg)
    all_results.update(r)

    # Task 5.3 — Univariate distributions
    task53_univariate_distributions(df_a, df_wx, cfg)

    # Task 5.4 — Bivariate feature vs. target
    r = task54_bivariate_feature_target(df_a, df_wx, cfg)
    all_results.update(r)

    # Task 5.5 — Correlation and VIF
    r = task55_correlation_and_vif(df_wx, cfg)
    all_results.update(r)

    # Task 5.6 — Temporal patterns
    r = task56_temporal_patterns(df_a, df_wx, cfg)
    all_results.update(r)

    # Task 5.7 — Missing data audit
    r = task57_missing_data(df_a, df_b, cfg)
    all_results.update(r)

    # Task 5.8 — Subgroup comparisons
    r = task58_subgroup_comparisons(df_a, df_wx, cfg)
    all_results.update(r)

    # Task 5.9 — Track B protocol EDA
    r = task59_protocol_eda(df_b, cfg)
    all_results.update(r)

    # Task 5.10 — EDA summary report
    task510_write_summary_report(df_a, df_b, df_wx, all_results, cfg)

    # Task 5.11 — Integrity checks
    task511_integrity_checks(df_a, df_b)

    logger.info("=" * 60)
    logger.info("Phase 5 EDA complete. All outputs in outputs/figures/ and outputs/tables/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
