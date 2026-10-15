import json

from src.green_building.tracking.track_model2_mlflow import _flatten_metrics


def test_mlflow_metric_names_are_flat_and_complete(tmp_path) -> None:
    results = {"models": {"dummy": {"test": {"mae": 1, "rmse": 2, "r2": 3}}}}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    assert _flatten_metrics(results) == {"dummy_test_mae": 1.0, "dummy_test_rmse": 2.0, "dummy_test_r2": 3.0}