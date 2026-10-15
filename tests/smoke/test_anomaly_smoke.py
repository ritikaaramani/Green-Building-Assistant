"""
tests/smoke/test_anomaly_smoke.py
==================================
Smoke tests: minimal assertions that the pipeline artifacts exist, are
non-empty, and are readable.  These run in seconds and are the first
check in CI.

Run with:
    conda run -n green-building pytest tests/smoke/test_anomaly_smoke.py -v
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/anomaly"

_ARTIFACTS = {
    "selected_buildings": DATA_DIR / "selected_buildings.csv",
    "features_v2": DATA_DIR / "anomaly_features_v2.csv",
    "predictions_v2": DATA_DIR / "anomaly_predictions_v2.csv",
    "model_disagreements": DATA_DIR / "model_disagreements.csv",
}

_OPTIONAL = {
    "predictions_v3": DATA_DIR / "anomaly_predictions_v3.csv",
    "report_v3": DATA_DIR / "anomaly_report_v3.json",
}

EXPECTED_BUILDINGS = [
    "Robin_education_Derrick",
    "Robin_education_Kiera",
    "Robin_lodging_Pricilla",
    "Robin_office_Maryann",
    "Robin_office_Serena",
]


@pytest.mark.parametrize("name,path", _ARTIFACTS.items())
def test_artifact_exists(name: str, path: Path):
    """Every required artifact file must exist and be non-empty."""
    if not path.exists():
        pytest.skip(f"Artifact not found (data not fetched): {path.name}")
    assert path.stat().st_size > 0, f"{name}: file is empty."


@pytest.mark.parametrize("name,path", _ARTIFACTS.items())
def test_artifact_readable_as_csv(name: str, path: Path):
    """Required CSV artifacts must be loadable without errors."""
    if not path.exists():
        pytest.skip(f"Artifact not found: {path.name}")
    if path.suffix != ".csv":
        pytest.skip("Not a CSV file.")
    df = pd.read_csv(path, nrows=10)
    assert len(df) > 0, f"{name}: CSV loaded but has no rows."


def test_features_v2_has_correct_buildings():
    path = DATA_DIR / "anomaly_features_v2.csv"
    if not path.exists():
        pytest.skip("features_v2 not found.")
    df = pd.read_csv(path, usecols=["building_id"])
    actual = set(df["building_id"].unique())
    expected = set(EXPECTED_BUILDINGS)
    assert actual == expected, f"Buildings mismatch. Expected {expected}, got {actual}."


def test_predictions_v2_has_anomaly_column():
    path = DATA_DIR / "anomaly_predictions_v2.csv"
    if not path.exists():
        pytest.skip("predictions_v2 not found.")
    df = pd.read_csv(path, nrows=5)
    assert "is_anomaly" in df.columns
    assert "anomaly_score" in df.columns


def test_predictions_v3_has_dual_flags():
    """V3-specific: both iforest and zscore columns must be present."""
    path = DATA_DIR / "anomaly_predictions_v3.csv"
    if not path.exists():
        pytest.skip("predictions_v3 not yet generated — run train_anomaly_detector_v3.py first.")
    df = pd.read_csv(path, nrows=5)
    assert "is_anomaly_iforest" in df.columns
    assert "is_anomaly_zscore" in df.columns
    assert "is_anomaly" in df.columns


def test_report_v3_is_valid_json():
    import json
    path = DATA_DIR / "anomaly_report_v3.json"
    if not path.exists():
        pytest.skip("report_v3 not yet generated — run train_anomaly_detector_v3.py first.")
    with open(path) as f:
        report = json.load(f)
    assert "model_version" in report
    assert report["model_version"] == "v3"
    assert "buildings" in report
    assert len(report["buildings"]) == len(EXPECTED_BUILDINGS)


def test_model_files_exist_after_training():
    models_dir = ROOT / "artifacts/models/anomaly"
    if not models_dir.exists():
        pytest.skip("Model artifacts dir not found — run train_anomaly_detector_v3.py first.")
    for building in EXPECTED_BUILDINGS:
        model_path = models_dir / f"{building}_iforest_v3.joblib"
        if not model_path.exists():
            pytest.skip(f"Model not yet saved: {model_path.name}")
        assert model_path.stat().st_size > 0, f"Empty model file: {model_path.name}"
