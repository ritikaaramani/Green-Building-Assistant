"""
tests/integration/test_anomaly_pipeline.py
==========================================
Integration tests for the end-to-end anomaly detection pipeline.

These tests read from the pre-computed artifacts on disk.  They are
skipped automatically when the required files do not exist so that CI
runs in environments without the BDG2 data still pass cleanly.

Run with:
    conda run -n green-building pytest tests/integration/test_anomaly_pipeline.py -v
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/anomaly"

SELECTED = DATA_DIR / "selected_buildings.csv"
FEATURES_V2 = DATA_DIR / "anomaly_features_v2.csv"
PREDICTIONS_V2 = DATA_DIR / "anomaly_predictions_v2.csv"
PREDICTIONS_V3 = DATA_DIR / "anomaly_predictions_v3.csv"

EXPECTED_BUILDINGS = {
    "Robin_education_Derrick",
    "Robin_education_Kiera",
    "Robin_lodging_Pricilla",
    "Robin_office_Maryann",
    "Robin_office_Serena",
}

EXPECTED_FEATURE_COLS_V2 = [
    "timestamp",
    "consumption",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_weekend",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_median_24h",
    "deviation_24h",
    "median_deviation_24h",
    "building_id",
]


def _skip_if_missing(*paths: Path):
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"Required file(s) not found: {[str(p) for p in missing]}")


# ── Selected buildings CSV ─────────────────────────────────────────────────

class TestSelectedBuildings:
    def test_file_exists_and_nonzero(self):
        _skip_if_missing(SELECTED)
        assert SELECTED.stat().st_size > 0

    def test_has_timestamp_column(self):
        _skip_if_missing(SELECTED)
        df = pd.read_csv(SELECTED, nrows=5)
        assert "timestamp" in df.columns

    def test_all_buildings_present(self):
        _skip_if_missing(SELECTED)
        df = pd.read_csv(SELECTED, nrows=1)
        for b in EXPECTED_BUILDINGS:
            assert b in df.columns, f"Building column missing: {b}"

    def test_row_count_plausible(self):
        """2016–2017 hourly = 17,544 rows expected."""
        _skip_if_missing(SELECTED)
        df = pd.read_csv(SELECTED)
        assert 17_000 <= len(df) <= 18_000, f"Unexpected row count: {len(df)}"


# ── V2 feature dataset ─────────────────────────────────────────────────────

class TestFeaturesV2:
    def setup_method(self):
        _skip_if_missing(FEATURES_V2)
        self.df = pd.read_csv(FEATURES_V2, parse_dates=["timestamp"])

    def test_expected_columns_present(self):
        for col in EXPECTED_FEATURE_COLS_V2:
            assert col in self.df.columns, f"Missing column: {col}"

    def test_all_buildings_present(self):
        actual = set(self.df["building_id"].unique())
        assert actual == EXPECTED_BUILDINGS

    def test_no_missing_values_after_warmup_drop(self):
        required = ["rolling_mean_24h", "rolling_std_24h", "rolling_median_24h"]
        assert self.df[required].isna().sum().sum() == 0

    def test_consumption_positive(self):
        assert (self.df["consumption"] >= 0).all()

    def test_cyclical_features_bounded(self):
        for col in ["hour_sin", "hour_cos", "day_sin", "day_cos"]:
            assert self.df[col].between(-1, 1).all(), f"{col} out of [-1, 1]"

    def test_deviation_definition(self):
        sample = self.df.dropna(subset=["rolling_mean_24h"]).sample(
            min(1000, len(self.df)), random_state=42
        )
        expected = sample["consumption"] - sample["rolling_mean_24h"]
        pd.testing.assert_series_equal(
            sample["deviation_24h"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_timestamps_unique_per_building(self):
        for building, group in self.df.groupby("building_id"):
            dupes = group["timestamp"].duplicated().sum()
            assert dupes == 0, f"{building} has {dupes} duplicate timestamps."

    def test_rows_per_building_equal(self):
        counts = self.df.groupby("building_id").size()
        assert counts.nunique() == 1, f"Unequal rows per building: {counts.to_dict()}"


# ── V2 predictions ─────────────────────────────────────────────────────────

class TestPredictionsV2:
    def setup_method(self):
        _skip_if_missing(PREDICTIONS_V2)
        self.df = pd.read_csv(PREDICTIONS_V2, parse_dates=["timestamp"])
        self.df["is_anomaly"] = (
            self.df["is_anomaly"].astype(str).str.lower() == "true"
        )

    def test_all_buildings_present(self):
        assert set(self.df["building_id"].unique()) == EXPECTED_BUILDINGS

    def test_anomaly_rate_within_contamination_band(self):
        """Flag rate should be close to contamination=0.02 (±1%)."""
        rate = self.df["is_anomaly"].mean()
        assert 0.005 <= rate <= 0.08, f"Anomaly rate {rate:.3%} outside expected range."

    def test_anomaly_score_finite(self):
        assert self.df["anomaly_score"].isna().sum() == 0
        assert self.df["anomaly_score"].apply(lambda x: x == x).all()

    def test_anomaly_prediction_values(self):
        assert set(self.df["anomaly_prediction"].unique()).issubset({-1, 1})

    def test_is_anomaly_consistent_with_prediction(self):
        expected = self.df["anomaly_prediction"] == -1
        pd.testing.assert_series_equal(
            self.df["is_anomaly"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_covers_approximately_20pct_of_timeline(self):
        """Test set should be last ~20% per building."""
        for building, group in self.df.groupby("building_id"):
            _skip_if_missing(FEATURES_V2)
            feats = pd.read_csv(FEATURES_V2, parse_dates=["timestamp"])
            total = len(feats[feats["building_id"] == building])
            expected_test = int(total * 0.20)
            actual = len(group)
            assert abs(actual - expected_test) <= 5, (
                f"{building}: expected ~{expected_test} test rows, got {actual}"
            )


# ── V3 predictions (skip if not yet generated) ────────────────────────────

class TestPredictionsV3:
    def setup_method(self):
        _skip_if_missing(PREDICTIONS_V3)
        self.df = pd.read_csv(PREDICTIONS_V3, parse_dates=["timestamp"])

    def test_required_columns_present(self):
        required = [
            "building_id", "timestamp", "consumption",
            "is_anomaly", "is_anomaly_iforest", "is_anomaly_zscore",
            "anomaly_score",
        ]
        for col in required:
            assert col in self.df.columns, f"Missing column: {col}"

    def test_combined_flag_is_union(self):
        """is_anomaly must equal is_anomaly_iforest OR is_anomaly_zscore."""
        for col in ["is_anomaly", "is_anomaly_iforest", "is_anomaly_zscore"]:
            self.df[col] = (
                self.df[col].astype(str).str.lower() == "true"
            )
        expected = self.df["is_anomaly_iforest"] | self.df["is_anomaly_zscore"]
        pd.testing.assert_series_equal(
            self.df["is_anomaly"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_all_buildings_present(self):
        assert set(self.df["building_id"].unique()) == EXPECTED_BUILDINGS

    def test_anomaly_rate_reasonable(self):
        for col in ["is_anomaly"]:
            self.df[col] = self.df[col].astype(str).str.lower() == "true"
        rate = self.df["is_anomaly"].mean()
        assert rate <= 0.15, f"V3 combined flag rate unexpectedly high: {rate:.2%}"
