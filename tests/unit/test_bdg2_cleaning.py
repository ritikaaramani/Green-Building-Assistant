from pathlib import Path

import pandas as pd

from src.green_building.data.bdg2_cleaning import _clean_meter, _clean_weather


def test_clean_meter_coerces_values_and_removes_bad_time_rows(tmp_path: Path) -> None:
    source = tmp_path / "meter.csv"
    pd.DataFrame(
        {"timestamp": ["2020-01-01 01:00", "bad", "2020-01-01 01:00"], "Building_A": [0, "bad", -1]}
    ).to_csv(source, index=False)

    result = _clean_meter(source, tmp_path / "out.csv")
    cleaned = pd.read_csv(tmp_path / "out.csv")

    assert result["rows_removed"] == 2
    assert result["non_numeric_values_coerced_to_missing"] == 1
    assert result["negative_readings_observed"] == 1
    assert cleaned.shape == (1, 2)
    assert cleaned.loc[0, "Building_A"] == 0


def test_weather_preserves_precipitation_sentinel(tmp_path: Path) -> None:
    source = tmp_path / "weather.csv"
    pd.DataFrame(
        {"timestamp": ["2020-01-01 00:00"], "site_id": ["A"], "precipDepth1HR": [-1]}
    ).to_csv(source, index=False)

    result = _clean_weather(source, tmp_path / "out.csv")

    assert result["negative_precipitation_sentinel_values"] == 1
    assert pd.read_csv(tmp_path / "out.csv").loc[0, "precipDepth1HR"] == -1