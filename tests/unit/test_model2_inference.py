from fastapi.testclient import TestClient

from src.green_building.models.energy_prediction.inference_api import app


client = TestClient(app)


def test_health_reports_training_schema() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["feature_columns"] == 18


def test_known_prediction_matches_saved_feature_schema() -> None:
    response = client.get("/predict", params={"building_id": "Panther_education_Janis", "timestamp": "2016-06-26 16:00:00"})
    assert response.status_code == 200
    body = response.json()
    assert body["feature_schema_matches_training"] is True
    assert body["actual_next_hour"] == 22.7624
    assert body["absolute_error"] is not None