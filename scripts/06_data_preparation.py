"""
scripts/06_data_preparation.py

Phase 6 — Data Preparation for Modeling orchestration.
Runs all preparation tasks in order and writes outputs to:
  data/processed/   — 6 split parquets (train/val/test × full/protocol)
  outputs/models/   — scaler_linear_A.pkl, scaler_linear_B.pkl, imputation_registry.json
  outputs/tables/   — phase6_feature_lists.json, phase6_preparation_report.txt
  outputs/logs/     — phase6_run.log
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
from src.data_preparation import (
    task60_load,
    task62_row_filter,
    task63_imputation,
    task65_temporal_split,
    task66_target_transform,
    task67_scaling,
    task68_cv_folds,
    task69_save,
    task610_integrity_checks,
    task611_write_report,
)


def setup_logging() -> logging.Logger:
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "phase6_run.log"

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
    logger.info("HEAPO-Predict — Phase 6: Data Preparation for Modeling")
    logger.info("=" * 60)

    cfg = load_config("config/params.yaml")

    # Task 6.0 — Load Phase 4 artifacts
    df_full, df_protocol = task60_load(cfg)

    # Task 6.2 — Row-level filtering (before split to avoid asymmetric test reduction)
    logger.info("-" * 40)
    logger.info("Task 6.2 — Row filtering")
    df_a, df_b, filter_counts = task62_row_filter(df_full, df_protocol)

    # Task 6.5 — Temporal train/val/test split
    logger.info("-" * 40)
    logger.info("Task 6.5 — Temporal split")
    df_train, df_val, df_test, df_b_train, df_b_val, df_b_test = task65_temporal_split(
        df_a, df_b, cfg
    )

    # Task 6.3 — Missing value imputation (fit on train only, AFTER split)
    logger.info("-" * 40)
    logger.info("Task 6.3 — Imputation (fit on train only)")
    imputation_registry = task63_imputation(
        df_train, df_val, df_test,
        df_b_train, df_b_val, df_b_test,
    )

    # Task 6.6 — Target transformation
    logger.info("-" * 40)
    logger.info("Task 6.6 — Target transformation (log1p)")
    log_stats = task66_target_transform(
        df_train, df_val, df_test,
        df_b_train, df_b_val, df_b_test,
    )

    # Task 6.7 — Feature scaling
    logger.info("-" * 40)
    logger.info("Task 6.7 — Feature scaling (StandardScaler)")
    scaler_A, scaler_B = task67_scaling(
        df_train, df_val, df_test,
        df_b_train, df_b_val, df_b_test,
    )

    # Task 6.8 — GroupKFold CV fold assignment
    logger.info("-" * 40)
    logger.info("Task 6.8 — CV fold assignment (GroupKFold k=5)")
    task68_cv_folds(df_train, df_b_train)

    # Task 6.9 — Save output artifacts
    logger.info("-" * 40)
    logger.info("Task 6.9 — Saving artifacts")
    parquet_shapes = task69_save(
        df_train, df_val, df_test,
        df_b_train, df_b_val, df_b_test,
        imputation_registry,
    )

    # Task 6.10 — Integrity checks
    logger.info("-" * 40)
    logger.info("Task 6.10 — Integrity checks")
    task610_integrity_checks(
        df_train, df_val, df_test,
        df_b_train, df_b_val, df_b_test,
        scaler_A, cfg,
    )

    # Task 6.11 — Preparation report
    logger.info("-" * 40)
    logger.info("Task 6.11 — Writing preparation report")
    task611_write_report(
        filter_counts=filter_counts,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        df_b_train=df_b_train,
        df_b_val=df_b_val,
        df_b_test=df_b_test,
        imputation_registry=imputation_registry,
        scaler_A=scaler_A,
        parquet_shapes=parquet_shapes,
        log_stats=log_stats,
        cfg=cfg,
    )

    logger.info("=" * 60)
    logger.info("Phase 6 complete.")
    logger.info("  Parquets → data/processed/  (train/val/test × full/protocol)")
    logger.info("  Scalers  → outputs/models/")
    logger.info("  Report   → outputs/tables/phase6_preparation_report.txt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
