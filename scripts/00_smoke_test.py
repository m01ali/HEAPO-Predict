#!/usr/bin/env python3
"""
scripts/00_smoke_test.py

Phase 0 smoke test. Verifies that the environment, dataset access,
and HEAPO dataloader are all working before any real work begins.

"""

import sys
from pathlib import Path


def check_config():
    import yaml

    cfg_path = Path("config/params.yaml")
    assert cfg_path.exists(), f"config not found: {cfg_path}"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert "data" in cfg and "dataset_path" in cfg["data"], "config missing data.dataset_path"
    assert "splits" in cfg, "config missing splits section"
    assert "modeling" in cfg, "config missing modeling section"
    assert "evaluation" in cfg, "config missing evaluation section"
    print("[SMOKE TEST] config/params.yaml loaded OK")
    return cfg


def check_dataset(cfg):
    data_path = Path(cfg["data"]["dataset_path"])
    assert data_path.exists(), (
        f"dataset_path '{data_path}' does not exist — "
        "run: unzip ~/Downloads/heapo_data.zip -d ."
    )
    for subfolder in ["meta_data", "smart_meter_data", "weather_data", "reports"]:
        assert (data_path / subfolder).exists(), f"missing subfolder in dataset: {subfolder}"
    print(f"[SMOKE TEST] dataset_path: {data_path} — exists, all 4 subfolders present")
    return data_path


def check_dataloader():
    from heapo import HEAPO  # noqa: F401
    print("[SMOKE TEST] HEAPO dataloader imported OK (heapo.HEAPO)")


def check_overview(data_path):
    import pandas as pd

    overview_path = (
        data_path / "smart_meter_data" / "overview" / "smart_meter_data_daily_overview.csv"
    )
    assert overview_path.exists(), f"daily overview not found: {overview_path}"
    df = pd.read_csv(overview_path, sep=";")
    assert "Household_ID" in df.columns, "Household_ID column missing from overview"
    n_hh = df["Household_ID"].nunique()
    print(f"[SMOKE TEST] overview loaded: {df.shape} — {n_hh} unique household IDs")
    return df


def check_single_household(data_path, overview_df):
    import pandas as pd

    first_id = overview_df["Household_ID"].iloc[0]
    hh_path = data_path / "smart_meter_data" / "daily" / f"{first_id}.csv"
    assert hh_path.exists(), f"daily file not found: {hh_path}"
    df = pd.read_csv(hh_path, sep=";", parse_dates=["Timestamp"])
    assert "kWh_received_Total" in df.columns, "kWh_received_Total column missing"
    assert "Timestamp" in df.columns, "Timestamp column missing"
    date_min = df["Timestamp"].min().date()
    date_max = df["Timestamp"].max().date()
    print(
        f"[SMOKE TEST] household {first_id} daily data: {df.shape}, "
        f"dates: {date_min} to {date_max}"
    )


def check_packages():
    packages = [
        "pandas",
        "numpy",
        "yaml",
        "sklearn",
        "xgboost",
        "lightgbm",
        "torch",
        "optuna",
        "shap",
        "matplotlib",
        "seaborn",
        "scipy",
        "statsmodels",
        "pyarrow",
    ]
    failed = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError as e:
            failed.append(f"{pkg}: {e}")
    if failed:
        print("[SMOKE TEST] PACKAGE IMPORT FAILURES:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    print("[SMOKE TEST] all core packages imported OK")


if __name__ == "__main__":
    cfg = check_config()
    data_path = check_dataset(cfg)
    check_dataloader()
    overview_df = check_overview(data_path)
    check_single_household(data_path, overview_df)
    check_packages()
    print("[SMOKE TEST] ALL CHECKS PASSED")
