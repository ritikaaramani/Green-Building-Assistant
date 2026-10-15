
"""Generate explanations for green-building recommendations."""

CONDITION_EXPLANATIONS = {
    "nighttime_anomaly": (
        "Unusual nighttime energy consumption warrants investigation."
    ),
    "high_energy": (
        "Predicted energy consumption is more than 20% "
        "above the expected baseline."
    ),
    "inefficient_building": (
        "The building efficiency assessment indicates "
        "potential improvement areas."
    ),
}


def explain_recommendation(recommendation):
    """Explain why an intervention was recommended."""
    matched_conditions = recommendation.get("matched_conditions", [])

    reasons = [
        CONDITION_EXPLANATIONS[condition]
        for condition in matched_conditions
        if condition in CONDITION_EXPLANATIONS
    ]

    if not reasons:
        return "No specific supporting condition was recorded."

    return " ".join(reasons)
