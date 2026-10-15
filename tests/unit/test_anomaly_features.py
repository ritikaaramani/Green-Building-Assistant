"""
tests/unit/test_anomaly_features.py
=====================================
Unit tests for the anomaly feature engineering logic.

Tests are entirely in-memory: they generate synthetic DataFrames and
verify that the feature building functions produce the expected outputs.
No files are read from disk.

Run with:
    conda run -n green-building pytest tests/unit/test_anomaly_features.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Helpers that mirror the logic in build_anomaly_features_v2 ─────────────

def make_hourly_series(n_hours: int = 200, seed: int = 0) -> pd.DataFrame:
    """Create a synthetic hourly electricity consumption DataFrame."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2017-01-01", periods=n_hours, freq="h")
    consumption = rng.uniform(50, 300, size=n_hours).astype(float)
    return pd.DataFrame({"timestamp": timestamps, "consumption": consumption})


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df["timestamp"].dt.hour
    day = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * day / 7)
    df["day_cos"] = np.cos(2 * np.pi * day / 7)
    df["is_weekend"] = (day >= 5).astype(int)
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    df = df.copy()
    past = df["consumption"].shift(1)
    df["rolling_mean_24h"] = past.rolling(window, min_periods=window).mean()
    df["rolling_std_24h"] = past.rolling(window, min_periods=window).std()
    df["rolling_median_24h"] = past.rolling(window, min_periods=window).median()
    df["deviation_24h"] = df["consumption"] - df["rolling_mean_24h"]
    df["median_deviation_24h"] = df["consumption"] - df["rolling_median_24h"]
    return df


# ── Tests ─────────────────────────────────────────────────────────────────

class TestCyclicalFeatures:
    def test_hour_sin_cos_range(self):
        df = add_cyclical_features(make_hourly_series(48))
        assert df["hour_sin"].between(-1, 1).all()
        assert df["hour_cos"].between(-1, 1).all()

    def test_day_sin_cos_range(self):
        df = add_cyclical_features(make_hourly_series(200))
        assert df["day_sin"].between(-1, 1).all()
        assert df["day_cos"].between(-1, 1).all()

    def test_midnight_hour_sin_near_zero(self):
        """Midnight (hour=0) → sin=0, cos=1."""
        df = make_hourly_series(1)
        df["timestamp"] = pd.Timestamp("2017-01-01 00:00:00")
        df = add_cyclical_features(df)
        assert abs(df["hour_sin"].iloc[0]) < 1e-10
        assert abs(df["hour_cos"].iloc[0] - 1.0) < 1e-10

    def test_no_missing_cyclical_values(self):
        df = add_cyclical_features(make_hourly_series(100))
        for col in ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]:
            assert df[col].isna().sum() == 0, f"{col} has NaNs"

    def test_is_weekend_binary(self):
        df = add_cyclical_features(make_hourly_series(200))
        assert set(df["is_weekend"].unique()).issubset({0, 1})

    def test_cyclical_distance_midnight_to_23h(self):
        """Midnight and 23:00 should be closer in cyclical space than linear."""
        ts = pd.date_range("2017-01-01 00:00", periods=24, freq="h")
        df = pd.DataFrame({"timestamp": ts, "consumption": 1.0})
        df = add_cyclical_features(df)
        sin_0 = df[df["timestamp"].dt.hour == 0]["hour_sin"].iloc[0]
        cos_0 = df[df["timestamp"].dt.hour == 0]["hour_cos"].iloc[0]
        sin_23 = df[df["timestamp"].dt.hour == 23]["hour_sin"].iloc[0]
        cos_23 = df[df["timestamp"].dt.hour == 23]["hour_cos"].iloc[0]
        sin_12 = df[df["timestamp"].dt.hour == 12]["hour_sin"].iloc[0]
        cos_12 = df[df["timestamp"].dt.hour == 12]["hour_cos"].iloc[0]

        dist_0_to_23 = np.sqrt((sin_0 - sin_23) ** 2 + (cos_0 - cos_23) ** 2)
        dist_0_to_12 = np.sqrt((sin_0 - sin_12) ** 2 + (cos_0 - cos_12) ** 2)
        assert dist_0_to_23 < dist_0_to_12, (
            "Midnight and 23:00 should be closer in cyclical space than midnight and noon."
        )


class TestRollingFeatures:
    def test_rolling_mean_nan_warmup(self):
        """First 24 rows of rolling_mean_24h must be NaN (shift + min_periods=24)."""
        df = add_rolling_features(make_hourly_series(200))
        # shift(1) means row 0 is NaN; rolling window fills from row 24 onwards
        assert df["rolling_mean_24h"].iloc[:24].isna().all()

    def test_rolling_mean_non_nan_after_warmup(self):
        df = add_rolling_features(make_hourly_series(200))
        assert df["rolling_mean_24h"].iloc[24:].notna().all()

    def test_deviation_definition(self):
        """deviation_24h = consumption - rolling_mean_24h."""
        df = add_rolling_features(make_hourly_series(200))
        valid = df.dropna(subset=["rolling_mean_24h"])
        expected = valid["consumption"] - valid["rolling_mean_24h"]
        pd.testing.assert_series_equal(
            valid["deviation_24h"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_rolling_std_non_negative(self):
        df = add_rolling_features(make_hourly_series(200))
        valid = df.dropna(subset=["rolling_std_24h"])
        assert (valid["rolling_std_24h"] >= 0).all()

    def test_no_future_leakage(self):
        """Rolling features at row i must not use row i (shift(1) enforced)."""
        df = make_hourly_series(50)
        # Set a spike at row 30.
        df.loc[30, "consumption"] = 1e6
        df = add_rolling_features(df)
        # The spike must NOT appear in rolling_mean at row 30.
        # It should appear at row 31 (shifted by 1).
        mean_at_30 = df["rolling_mean_24h"].iloc[30]
        # row 30's rolling window covers rows 6-29 (shifted), none of which is 1e6
        if pd.notna(mean_at_30):
            assert mean_at_30 < 1e5, "Spike at row 30 leaked into its own rolling mean."


class TestMissingValueHandling:
    def test_interpolate_fills_interior_nans(self):
        df = make_hourly_series(100)
        df.loc[50, "consumption"] = np.nan
        df.loc[51, "consumption"] = np.nan
        df["consumption"] = df["consumption"].interpolate(
            method="linear", limit=2, limit_area="inside"
        )
        assert df["consumption"].iloc[50:52].notna().all()

    def test_interpolate_does_not_fill_leading_nans(self):
        df = make_hourly_series(100)
        df.loc[0, "consumption"] = np.nan
        original = df["consumption"].copy()
        df["consumption"] = df["consumption"].interpolate(
            method="linear", limit=2, limit_area="inside"
        )
        assert pd.isna(df["consumption"].iloc[0]), (
            "Leading NaN should not be filled with limit_area='inside'."
        )

    def test_large_gap_not_fully_filled(self):
        """limit=2 leaves the centre of a 5-NaN gap unfilled."""
        df = make_hourly_series(100)
        # Rows 40-44 inclusive = 5 consecutive NaNs.
        df.loc[40:44, "consumption"] = np.nan
        df["consumption"] = df["consumption"].interpolate(
            method="linear", limit=2, limit_area="inside"
        )
        # The centre row (42) must remain NaN: it is >2 steps from both ends.
        assert pd.isna(df["consumption"].iloc[42]), (
            "Centre of a 5-NaN gap must remain NaN with limit=2."
        )
        # The first two rows of the gap (40, 41) should be filled.
        assert pd.notna(df["consumption"].iloc[40])
        assert pd.notna(df["consumption"].iloc[41])


class TestDropnaAfterRolling:
    def test_dropna_removes_warmup_rows(self):
        df = make_hourly_series(200)
        df = add_rolling_features(df)
        required = ["rolling_mean_24h", "rolling_std_24h", "rolling_median_24h"]
        clean = df.dropna(subset=required)
        assert clean[required].isna().sum().sum() == 0
        # At most 24 warmup rows should have been dropped
        assert len(df) - len(clean) <= 24
