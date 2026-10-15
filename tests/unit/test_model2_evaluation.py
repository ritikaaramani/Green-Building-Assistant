import pandas as pd

from src.green_building.models.energy_prediction.train_model2 import split_name


def test_evaluation_periods_use_test_boundary() -> None:
    timestamps = pd.Series(pd.to_datetime(["2017-06-30 23:00", "2017-07-01 00:00", "2017-12-31 23:00"]))
    result = split_name(timestamps)
    assert result.iloc[:2].tolist() == ["validation", "test"]
    assert result.iloc[2] == "test"