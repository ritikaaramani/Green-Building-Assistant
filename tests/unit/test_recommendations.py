
import pytest

from green_building.recommendations import (
    calculate_priority,
    get_conditions,
    rank_interventions,
    explain_recommendation,
)


def test_priority_score():
    intervention = {
        "impact": 1.0,
        "urgency": 1.0,
        "feasibility": 1.0,
        "cost_effectiveness": 1.0,
    }
    assert calculate_priority(intervention) == 1.0


def test_invalid_score_rejected():
    intervention = {
        "impact": 1.5,
        "urgency": 0.5,
        "feasibility": 0.5,
        "cost_effectiveness": 0.5,
    }
    with pytest.raises(ValueError):
        calculate_priority(intervention)


def test_high_energy_condition():
    conditions = get_conditions({
        "predicted_energy_kwh": 145,
        "expected_energy_kwh": 100,
    })
    assert "high_energy" in conditions


def test_exactly_twenty_percent_above_baseline():
    conditions = get_conditions({
        "predicted_energy_kwh": 120,
        "expected_energy_kwh": 100,
    })
    assert "high_energy" not in conditions


def test_nighttime_anomaly_condition():
    conditions = get_conditions({"nighttime_anomaly": True})
    assert "nighttime_anomaly" in conditions


def test_recommendations_are_sorted():
    results = rank_interventions(
        {
            "nighttime_anomaly": True,
            "predicted_energy_kwh": 145,
            "expected_energy_kwh": 100,
            "hvac_available": True,
            "occupancy_control_available": True,
        },
        top_n=5,
    )

    scores = [item["priority_score"] for item in results]
    assert scores == sorted(scores, reverse=True)


def test_hvac_excluded_when_unavailable():
    results = rank_interventions(
        {
            "nighttime_anomaly": True,
            "hvac_available": False,
        },
        top_n=5,
    )

    assert all(item["id"] != "hvac_schedule" for item in results)


def test_empty_conditions_return_no_recommendations():
    assert rank_interventions({}) == []


def test_zero_expected_energy_rejected():
    with pytest.raises(ValueError):
        get_conditions({
            "predicted_energy_kwh": 100,
            "expected_energy_kwh": 0,
        })


def test_explanation_mentions_trigger():
    explanation = explain_recommendation({
        "matched_conditions": ["nighttime_anomaly"],
    })
    assert "nighttime" in explanation.lower()
