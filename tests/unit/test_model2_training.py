import pandas as pd

from src.green_building.models.energy_prediction.train_model2 import BOUNDARIES, split_name


def test_split_boundaries_are_chronological_and_half_open() -> None:
    timestamps = pd.Series(pd.to_datetime(["2016-12-31 23:00", "2017-01-01 00:00", "2017-06-30 23:00", "2017-07-01 00:00", "2017-12-31 23:00", "2018-01-01 00:00"]))
    result = split_name(timestamps)
    assert result.iloc[:5].tolist() == ["train", "validation", "validation", "test", "test"]
    assert pd.isna(result.iloc[5])
    assert BOUNDARIES["train"][1] == BOUNDARIES["validation"][0]
    assert BOUNDARIES["validation"][1] == BOUNDARIES["test"][0]