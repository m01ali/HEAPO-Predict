"""
Phase 13 — Final Reproducibility and Consistency Checks
========================================================
Audits all pipeline outputs for:
  CHECK 1: Output file existence (30 expected files)
  CHECK 2: Numerical consistency — key metrics match Phase 9 CSV values
  CHECK 3: Figure existence (21 report figures)
  CHECK 4: Data integrity — temporal boundaries, row counts, residual sign convention,
            feature list completeness
  CHECK 5: Model reproducibility — reload RF model, re-predict on 100 rows, compare
            with stored predictions (max absolute error < 1e-3 kWh)
  CHECK 6: Config completeness — 9 required keys present in config/params.yaml
  CHECK 7: Write outputs/tables/phase13_final_checks_report.txt

Exit code 0 if all checks pass, 1 if any check fails.
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "outputs" / "models"
REPORT_DIR = ROOT / "outputs" / "report"
LOG_DIR = ROOT / "outputs" / "logs"
CONFIG_PATH = ROOT / "config" / "params.yaml"
FEATURE_LISTS_PATH = TABLE_DIR / "phase6_feature_lists.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "phase13_run.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
results: dict[str, dict] = {}


def record(check_id: str, name: str, passed: bool, detail: str = "") -> None:
    results[check_id] = {"name": name, "passed": passed, "detail": detail}
    status = "PASS" if passed else "FAIL"
    log.info(f"[{status}] {check_id}: {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# CHECK 1: Output file existence
# ---------------------------------------------------------------------------

EXPECTED_FILES = [
    # Phase 6 feature lists
    TABLE_DIR / "phase6_feature_lists.json",
    # Phase 7 outputs
    TABLE_DIR / "phase7_val_predictions.parquet",
    TABLE_DIR / "phase7_training_report.txt",
    # Phase 8 outputs
    TABLE_DIR / "phase8_tuning_report.txt",
    TABLE_DIR / "phase8_val_predictions.parquet",
    TABLE_DIR / "phase8_test_predictions.parquet",
    # Phase 9 outputs
    TABLE_DIR / "phase9_metrics_test.csv",
    TABLE_DIR / "phase9_metrics_val.csv",
    TABLE_DIR / "phase9_metrics_seasonal.csv",
    TABLE_DIR / "phase9_metrics_cv.csv",
    TABLE_DIR / "phase9_ablation_metrics.csv",
    TABLE_DIR / "phase9_wilcoxon_matrix.csv",
    TABLE_DIR / "phase9_test_predictions.parquet",
    TABLE_DIR / "phase9_val_predictions.parquet",
    TABLE_DIR / "phase9_test_predictions_b.parquet",
    TABLE_DIR / "phase9_evaluation_report.txt",
    # Phase 10 outputs
    TABLE_DIR / "phase10_permutation_importance.csv",
    TABLE_DIR / "phase10_shap_mean_abs.csv",
    TABLE_DIR / "phase10_spearman_correlation.csv",
    TABLE_DIR / "phase10_feature_ranking_table.csv",
    TABLE_DIR / "phase10_interpretability_report.txt",
    # Phase 11 outputs
    TABLE_DIR / "phase11_subgroup_metrics.csv",
    TABLE_DIR / "phase11_mannwhitney_results.csv",
    TABLE_DIR / "phase11_subgroup_composition.csv",
    TABLE_DIR / "phase11_track_b_subgroup_metrics.csv",
    TABLE_DIR / "phase11_subgroup_report.txt",
    # Models
    MODEL_DIR / "model_rf_tuned.pkl",
    MODEL_DIR / "model_xgboost_b_tuned.pkl",
    MODEL_DIR / "scaler_linear_A.pkl",
    # Config
    CONFIG_PATH,
]


def check_1_file_existence() -> None:
    log.info("=" * 60)
    log.info("CHECK 1: Output file existence (30 expected files)")
    log.info("=" * 60)
    missing = [str(p.relative_to(ROOT)) for p in EXPECTED_FILES if not p.exists()]
    if missing:
        record(
            "CHECK_1",
            "Output file existence",
            False,
            f"{len(missing)} missing: {', '.join(missing)}",
        )
    else:
        record(
            "CHECK_1",
            "Output file existence",
            True,
            f"All {len(EXPECTED_FILES)} files present",
        )


# ---------------------------------------------------------------------------
# CHECK 2: Numerical consistency
# ---------------------------------------------------------------------------

# Tolerances — generous enough to survive float precision differences
METRIC_TOLERANCES = {
    # (Model, metric, expected_value, tolerance)
    ("RF", "RMSE"): (11.540, 0.01),
    ("XGBoost B", "RMSE"): (8.419, 0.01),
    ("ElasticNet", "R2"): (0.151, 0.01),
}


def check_2_numerical_consistency() -> None:
    log.info("=" * 60)
    log.info("CHECK 2: Numerical consistency (key test-set metrics)")
    log.info("=" * 60)
    try:
        metrics = pd.read_csv(TABLE_DIR / "phase9_metrics_test.csv")
        # Normalise column names
        metrics.columns = [c.strip() for c in metrics.columns]
        col_map = {c.upper(): c for c in metrics.columns}

        failures = []
        details = []

        for (model, metric_name), (expected, tol) in METRIC_TOLERANCES.items():
            # Find row
            row = metrics[metrics["Model"] == model]
            if row.empty:
                failures.append(f"Model '{model}' not found in metrics CSV")
                continue

            # Find column — try both 'R2' and 'R²'
            col_candidates = [metric_name, metric_name.replace("2", "²"), "R2", "RMSE", "MAE"]
            col = None
            for cand in [metric_name, metric_name.replace("2", "²")]:
                if cand in metrics.columns:
                    col = cand
                    break
                for c in metrics.columns:
                    if c.replace("²", "2").upper() == cand.upper():
                        col = c
                        break
                if col:
                    break

            if col is None:
                # Last-resort: check all columns case-insensitively
                for c in metrics.columns:
                    if c.upper() == metric_name.upper():
                        col = c
                        break

            if col is None:
                failures.append(f"Metric column '{metric_name}' not found")
                continue

            actual = float(row[col].iloc[0])
            diff = abs(actual - expected)
            ok = diff <= tol
            detail_str = f"{model} {metric_name}: expected={expected}, actual={actual:.6f}, diff={diff:.6f}"
            details.append(detail_str)
            log.info(f"  {'OK' if ok else 'MISMATCH'}: {detail_str}")
            if not ok:
                failures.append(detail_str)

        if failures:
            record("CHECK_2", "Numerical consistency", False, "; ".join(failures))
        else:
            record("CHECK_2", "Numerical consistency", True, "; ".join(details))

    except Exception as e:
        record("CHECK_2", "Numerical consistency", False, f"Exception: {e}")
        log.exception(e)


# ---------------------------------------------------------------------------
# CHECK 3: Figure existence (21 report figures)
# ---------------------------------------------------------------------------

REPORT_FIGURES = [
    # Phase 5 / EDA
    FIG_DIR / "05_target_monthly_boxplot.png",
    FIG_DIR / "05_subgroup_hp_type.png",
    # Phase 8 / tuning
    FIG_DIR / "phase8_optuna_rf.png",
    FIG_DIR / "phase8_optuna_xgboost.png",
    # Phase 9 / evaluation
    FIG_DIR / "phase9_predicted_vs_actual_rf.png",
    FIG_DIR / "phase9_predicted_vs_actual_xgboost.png",
    FIG_DIR / "phase9_cv_errorbar.png",
    FIG_DIR / "phase9_ablation_barplot.png",
    # Phase 10 / interpretability
    FIG_DIR / "phase10_shap_summary_beeswarm_xgboost.png",
    FIG_DIR / "phase10_shap_bar_rf.png",
    FIG_DIR / "phase10_spearman_correlation_heatmap.png",
    FIG_DIR / "phase10_feature_ranking_heatmap.png",
    FIG_DIR / "phase10_accuracy_interpretability_tradeoff.png",
    FIG_DIR / "phase10_dt_tree_structure.png",
    # Phase 11 / subgroups
    FIG_DIR / "phase11_bias_heatmap.png",
    FIG_DIR / "phase11_residuals_heat_dist.png",
    FIG_DIR / "phase11_residuals_pv.png",
    FIG_DIR / "phase11_residuals_ev.png",
    FIG_DIR / "phase11_track_b_residuals_building_age.png",
    FIG_DIR / "phase11_track_b_residuals_night_setback.png",
    FIG_DIR / "phase11_composition_bar.png",
]


def check_3_figure_existence() -> None:
    log.info("=" * 60)
    log.info("CHECK 3: Report figure existence (21 figures)")
    log.info("=" * 60)
    missing = [str(p.relative_to(ROOT)) for p in REPORT_FIGURES if not p.exists()]
    if missing:
        record(
            "CHECK_3",
            "Report figure existence",
            False,
            f"{len(missing)} missing: {', '.join(missing)}",
        )
    else:
        record(
            "CHECK_3",
            "Report figure existence",
            True,
            f"All {len(REPORT_FIGURES)} report figures present",
        )


# ---------------------------------------------------------------------------
# CHECK 4: Data integrity
# ---------------------------------------------------------------------------


def check_4_data_integrity() -> None:
    log.info("=" * 60)
    log.info("CHECK 4: Data integrity")
    log.info("=" * 60)

    sub_failures = []
    sub_details = []

    # --- 4a: Temporal boundary: test_min_date > train_max_date ---------------
    try:
        train_dates = pq.read_table(
            ROOT / "data" / "processed" / "train_full.parquet", columns=["Date"]
        ).to_pandas()
        test_dates = pq.read_table(
            ROOT / "data" / "processed" / "test_full.parquet", columns=["Date"]
        ).to_pandas()

        # Normalise to date only (strip tz)
        train_max = pd.to_datetime(train_dates["Date"]).dt.tz_localize(None).max().date()
        test_min = pd.to_datetime(test_dates["Date"]).dt.tz_localize(None).min().date()

        no_leakage = test_min > train_max
        detail = f"train_max={train_max}, test_min={test_min}"
        sub_details.append(f"4a temporal boundary: {detail}")
        if not no_leakage:
            sub_failures.append(f"DATA LEAKAGE — {detail}")
            log.error(f"  FAIL 4a: {detail}")
        else:
            log.info(f"  OK 4a: No leakage — {detail}")
    except Exception as e:
        sub_failures.append(f"4a exception: {e}")
        log.warning(f"  WARN 4a: {e}")

    # --- 4b: Row count match between test_full and phase9_test_predictions ---
    try:
        test_preds = pq.read_table(TABLE_DIR / "phase9_test_predictions.parquet").to_pandas()
        test_full_len = len(
            pq.read_table(ROOT / "data" / "processed" / "test_full.parquet", columns=["Date"]).to_pandas()
        )
        preds_len = len(test_preds)
        match = test_full_len == preds_len
        detail = f"test_full={test_full_len} rows, predictions={preds_len} rows"
        sub_details.append(f"4b row count: {detail}")
        if not match:
            sub_failures.append(f"Row count mismatch — {detail}")
            log.error(f"  FAIL 4b: {detail}")
        else:
            log.info(f"  OK 4b: {detail}")
    except Exception as e:
        sub_failures.append(f"4b exception: {e}")
        log.warning(f"  WARN 4b: {e}")

    # --- 4c: Residual sign convention: residual = actual - predicted ---------
    try:
        test_preds = pq.read_table(TABLE_DIR / "phase9_test_predictions.parquet").to_pandas()
        y_col = "kWh_received_Total"
        pred_col = "pred_rf"

        if y_col in test_preds.columns and pred_col in test_preds.columns:
            sample = test_preds.dropna(subset=[y_col, pred_col]).head(1000)
            computed_resid = sample[y_col] - sample[pred_col]

            # Check for a residual column
            resid_col = "residual_pred_rf"
            if resid_col in test_preds.columns:
                stored_resid = sample[resid_col]
                max_diff = (computed_resid - stored_resid).abs().max()
                ok = max_diff < 1e-6
                detail = f"max |computed - stored| residual = {max_diff:.2e}"
                sub_details.append(f"4c residual sign: {detail}")
                if not ok:
                    sub_failures.append(f"Residual sign convention mismatch — {detail}")
                    log.error(f"  FAIL 4c: {detail}")
                else:
                    log.info(f"  OK 4c: Residual = actual − predicted confirmed ({detail})")
            else:
                # Just check sign: sum of residuals should be near-zero for unbiased model
                mean_resid = computed_resid.mean()
                detail = f"mean residual (RF) = {mean_resid:.4f} kWh (no stored residual col to compare)"
                sub_details.append(f"4c residual check: {detail}")
                log.info(f"  OK 4c: {detail}")
        else:
            missing_cols = [c for c in [y_col, pred_col] if c not in test_preds.columns]
            sub_failures.append(f"4c missing columns: {missing_cols}")
            log.warning(f"  WARN 4c: Missing columns {missing_cols}")
    except Exception as e:
        sub_failures.append(f"4c exception: {e}")
        log.warning(f"  WARN 4c: {e}")

    # --- 4d: Feature lists vs parquet columns --------------------------------
    try:
        with open(FEATURE_LISTS_PATH) as fh:
            feat_lists = json.load(fh)

        features_trees = feat_lists.get("FEATURES_TREES", [])
        test_cols = set(
            pq.read_table(ROOT / "data" / "processed" / "test_full.parquet").schema.names
        )
        missing_feats = [f for f in features_trees if f not in test_cols]
        detail = f"{len(features_trees)} tree features; {len(missing_feats)} absent from test_full"
        sub_details.append(f"4d feature lists: {detail}")
        if missing_feats:
            sub_failures.append(f"Missing features in test_full: {missing_feats[:5]}...")
            log.error(f"  FAIL 4d: {detail} — first 5: {missing_feats[:5]}")
        else:
            log.info(f"  OK 4d: All {len(features_trees)} FEATURES_TREES present in test_full")
    except Exception as e:
        sub_failures.append(f"4d exception: {e}")
        log.warning(f"  WARN 4d: {e}")

    # --- Aggregate -----------------------------------------------------------
    if sub_failures:
        record("CHECK_4", "Data integrity", False, "; ".join(sub_failures))
    else:
        record("CHECK_4", "Data integrity", True, " | ".join(sub_details))


# ---------------------------------------------------------------------------
# CHECK 5: Model reproducibility
# ---------------------------------------------------------------------------

# Canonical 45-feature ordering used to train RF / XGBoost / LightGBM / DT.
# Derived from _ALL_51[:45] in scripts/09_evaluation.py — must match exactly.
_FEATURES_TREES_TRAIN_ORDER = [
    "kvarh_received_capacitive_Total", "kvarh_received_inductive_Total",
    "has_pv", "has_reactive_energy",
    "Temperature_max_daily", "Temperature_min_daily", "Temperature_avg_daily",
    "HeatingDegree_SIA_daily", "HeatingDegree_US_daily", "CoolingDegree_US_daily",
    "Humidity_avg_daily", "Precipitation_total_daily", "Sunshine_duration_daily",
    "Survey_Building_LivingArea", "Survey_Building_Residents",
    "day_of_week", "month", "is_weekend", "day_of_year", "is_heating_season",
    "temp_range_daily", "HDD_SIA_daily", "HDD_US_daily", "CDD_US_daily",
    "humidity_x_temp", "temp_avg_lag_1d", "temp_avg_rolling_3d",
    "temp_avg_rolling_7d", "HDD_SIA_rolling_7d",
    "building_type_house", "building_type_apartment",
    "hp_type_air_source", "hp_type_ground_source", "hp_type_unknown",
    "dhw_hp", "dhw_ewh", "dhw_solar", "dhw_combined", "dhw_unknown",
    "heat_dist_floor", "heat_dist_radiator", "heat_dist_both", "heat_dist_unknown",
    "has_ev", "has_dryer",
]


def check_5_model_reproducibility() -> None:
    log.info("=" * 60)
    log.info("CHECK 5: Model reproducibility (RF — 100 row re-prediction)")
    log.info("=" * 60)
    try:
        import joblib

        rf_path = MODEL_DIR / "model_rf_tuned.pkl"
        if not rf_path.exists():
            record("CHECK_5", "Model reproducibility", False, f"Model file not found: {rf_path}")
            return

        rf = joblib.load(rf_path)

        # Use the canonical training-order feature list (not phase6_feature_lists.json,
        # which uses a different ordering than what the model was trained with)
        features_trees = _FEATURES_TREES_TRAIN_ORDER

        # Load test data
        test_full = pq.read_table(
            ROOT / "data" / "processed" / "test_full.parquet"
        ).to_pandas()

        # Load stored predictions
        stored_preds = pq.read_table(TABLE_DIR / "phase9_test_predictions.parquet").to_pandas()

        # Take 100 rows deterministically
        np.random.seed(42)
        idx_sample = np.random.choice(len(test_full), size=min(100, len(test_full)), replace=False)
        idx_sample = np.sort(idx_sample)

        sample_full = test_full.iloc[idx_sample]
        sample_stored = stored_preds.iloc[idx_sample]

        X_sample = sample_full[features_trees].copy()
        # Coerce dtypes: any string/categorical columns that the model expects as numeric
        for col in X_sample.columns:
            if X_sample[col].dtype == object:
                X_sample[col] = pd.to_numeric(X_sample[col], errors="coerce").fillna(0)

        new_preds = rf.predict(X_sample.values)
        stored_vals = sample_stored["pred_rf"].values
        max_err = np.max(np.abs(new_preds - stored_vals))
        tol = 1e-3

        detail = f"max |new − stored| = {max_err:.2e} kWh (tolerance {tol})"
        if max_err <= tol:
            record("CHECK_5", "Model reproducibility", True, detail)
            log.info(f"  OK 5: {detail}")
        else:
            record("CHECK_5", "Model reproducibility", False, detail)
            log.error(f"  FAIL 5: {detail}")

    except Exception as e:
        record("CHECK_5", "Model reproducibility", False, f"Exception: {e}")
        log.exception(e)


# ---------------------------------------------------------------------------
# CHECK 6: Config completeness
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = [
    ("data", "dataset_path"),
    ("data", "zenodo_record_id"),
    ("data", "min_days_threshold"),
    ("splits", "train_end"),
    ("splits", "val_end"),
    ("splits", "test_end"),
    ("modeling", "random_seed"),
    ("evaluation", "mape_floor_kwh"),
    ("evaluation", "stat_test_alpha"),
]


def check_6_config_completeness() -> None:
    log.info("=" * 60)
    log.info("CHECK 6: Config completeness (9 required keys in config/params.yaml)")
    log.info("=" * 60)
    try:
        if not CONFIG_PATH.exists():
            record("CHECK_6", "Config completeness", False, f"Config file not found: {CONFIG_PATH}")
            return

        with open(CONFIG_PATH) as fh:
            cfg = yaml.safe_load(fh)

        missing = []
        present = []
        for section, key in REQUIRED_CONFIG_KEYS:
            if section not in cfg or key not in cfg[section]:
                missing.append(f"{section}.{key}")
            else:
                present.append(f"{section}.{key}={cfg[section][key]!r}")

        if missing:
            record(
                "CHECK_6",
                "Config completeness",
                False,
                f"Missing keys: {', '.join(missing)}",
            )
            log.error(f"  FAIL 6: Missing {missing}")
        else:
            record(
                "CHECK_6",
                "Config completeness",
                True,
                f"All {len(REQUIRED_CONFIG_KEYS)} required keys present",
            )
            log.info(f"  OK 6: {', '.join(present)}")
    except Exception as e:
        record("CHECK_6", "Config completeness", False, f"Exception: {e}")
        log.exception(e)


# ---------------------------------------------------------------------------
# CHECK 7: Write final report
# ---------------------------------------------------------------------------


def write_final_report() -> None:
    log.info("=" * 60)
    log.info("CHECK 7: Writing final checks report")
    log.info("=" * 60)

    total = len(results)
    passed = sum(1 for r in results.values() if r["passed"])
    failed = total - passed

    lines = [
        "=" * 70,
        "HEAPO-Predict — Phase 13: Final Reproducibility and Consistency Checks",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        f"SUMMARY: {passed}/{total} checks passed, {failed} failed",
        "",
        "-" * 70,
    ]

    for check_id, info in results.items():
        status = "PASS" if info["passed"] else "FAIL"
        lines.append(f"[{status}]  {check_id}: {info['name']}")
        if info["detail"]:
            # Wrap long detail lines
            detail = info["detail"]
            if len(detail) > 120:
                detail = detail[:117] + "..."
            lines.append(f"        {detail}")
        lines.append("")

    lines += [
        "-" * 70,
        "",
        "CHECKS LEGEND",
        "  CHECK_1  Output file existence         — 30 pipeline output files",
        "  CHECK_2  Numerical consistency         — RF RMSE, XGBoost B RMSE, ElasticNet R²",
        "  CHECK_3  Report figure existence       — 21 figures referenced in the report",
        "  CHECK_4  Data integrity                — temporal boundaries, row counts, residual",
        "                                           sign convention, feature list completeness",
        "  CHECK_5  Model reproducibility         — RF re-prediction on 100 test rows (<1e-3)",
        "  CHECK_6  Config completeness           — 9 required keys in config/params.yaml",
        "",
    ]

    if failed == 0:
        lines.append("All checks passed. Pipeline is reproducible and publication-ready.")
    else:
        lines.append(f"WARNING: {failed} check(s) failed. Review FAIL entries above before publication.")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    out_path = TABLE_DIR / "phase13_final_checks_report.txt"
    out_path.write_text(report_text, encoding="utf-8")
    log.info(f"  Report written to {out_path.relative_to(ROOT)}")

    # Also echo to stdout
    print("\n" + report_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    log.info("Phase 13 — Final Reproducibility and Consistency Checks")
    log.info(f"Root: {ROOT}")
    log.info(f"Timestamp: {datetime.now().isoformat()}")

    check_1_file_existence()
    check_2_numerical_consistency()
    check_3_figure_existence()
    check_4_data_integrity()
    check_5_model_reproducibility()
    check_6_config_completeness()
    write_final_report()

    all_passed = all(r["passed"] for r in results.values())
    exit_code = 0 if all_passed else 1
    log.info(f"Phase 13 complete — exit code {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
