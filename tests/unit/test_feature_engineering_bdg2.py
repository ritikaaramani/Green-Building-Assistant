from pathlib import Path

import pandas as pd

from src.green_building.feature_engineering_bdg2 import _engineer_chunk, _prepare_metadata, _prepare_weather


def test_features_are_past_only_and_target_is_next_hour(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    weather_path = tmp_path / "weather.csv"
    pd.DataFrame({"building_id": ["B"], "site_id": ["S"], "sqm": [10.0], "primaryspaceusage": ["Office"]}).to_csv(metadata_path, index=False)
    pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=30, freq="h"), "site_id": ["S"] * 30, "airTemperature": [1.0] * 30, "cloudCoverage": [None] * 30, "dewTemperature": [1.0] * 30, "precipDepth1HR": [0.0] * 30, "precipDepth6HR": [0.0] * 30, "seaLvlPressure": [1000.0] * 30, "windDirection": [0.0] * 30, "windSpeed": [1.0] * 30}).to_csv(weather_path, index=False)
    electricity = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=30, freq="h"), "B": range(30)})

    features, target, _ = _engineer_chunk(electricity, ["B"], _prepare_metadata(metadata_path), _prepare_weather(weather_path))

    row = features.loc[features["prediction_timestamp"] == "2020-01-01 01:00:00"].iloc[0]
    assert row["electricity_lag_1h"] == 1
    assert target.loc[target["prediction_timestamp"] == "2020-01-01 01:00:00", "target_electricity_t_plus_1h"].iloc[0] == 2
    assert pd.isna(row["electricity_lag_24h"])
    assert pd.isna(row["electricity_rolling_mean_24h"])

    row = features.loc[features["prediction_timestamp"] == "2020-01-02 00:00:00"].iloc[0]
    assert row["electricity_lag_24h"] == 0