
"""Filter and rank green-building interventions."""

from numbers import Real

from .catalog import INTERVENTIONS

WEIGHTS = {
    "impact": 0.35,
    "urgency": 0.25,
    "feasibility": 0.20,
    "cost_effectiveness": 0.20,
}


def calculate_priority(intervention):
    """Calculate a weighted priority score between 0 and 1."""
    scores = {}

    for key in WEIGHTS:
        value = intervention[key]

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{key} must be between 0 and 1")

        scores[key] = value

    score = sum(
        WEIGHTS[key] * scores[key]
        for key in WEIGHTS
    )

    return round(score, 4)


def get_conditions(building_state):
    """Identify conditions that may justify interventions."""
    conditions = set()

    if building_state.get("nighttime_anomaly") is True:
        conditions.add("nighttime_anomaly")

    predicted = building_state.get("predicted_energy_kwh")
    expected = building_state.get("expected_energy_kwh")

    if predicted is not None or expected is not None:
        if predicted is None or expected is None:
            raise ValueError(
                "Provide both predicted_energy_kwh and expected_energy_kwh"
            )

        for name, value in (
            ("predicted_energy_kwh", predicted),
            ("expected_energy_kwh", expected),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or value < 0
            ):
                raise ValueError(f"{name} must be non-negative")

        if expected == 0:
            raise ValueError("expected_energy_kwh must be greater than zero")

        if predicted > expected * 1.20:
            conditions.add("high_energy")

    category = building_state.get("efficiency_category")

    if category in {
        "moderately_inefficient",
        "highly_inefficient",
    }:
        conditions.add("inefficient_building")

    return conditions


def rank_interventions(building_state, top_n=3):
    """Return applicable interventions in priority order."""
    if not isinstance(building_state, dict):
        raise ValueError("building_state must be a dictionary")

    if (
        isinstance(top_n, bool)
        or not isinstance(top_n, int)
        or top_n < 1
    ):
        raise ValueError("top_n must be a positive integer")

    conditions = get_conditions(building_state)
    recommendations = []

    for intervention in INTERVENTIONS:
        required = intervention["requires"]

        # Do not recommend actions requiring unavailable capabilities.
        if required and building_state.get(required) is not True:
            continue

        matched = conditions.intersection(intervention["conditions"])

        # Only include actions relevant to detected conditions.
        if not matched:
            continue

        result = intervention.copy()
        result["matched_conditions"] = sorted(matched)
        result["priority_score"] = calculate_priority(intervention)

        recommendations.append(result)

    recommendations.sort(
        key=lambda item: (-item["priority_score"], item["id"])
    )

    return recommendations[:top_n]
