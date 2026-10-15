
"""Demonstrate recommendations using real BDG2 electricity data."""

from green_building.recommendations.energy_analysis import (
    analyze_building_energy,
)
from green_building.recommendations.ranker import rank_interventions
from green_building.recommendations.explain import explain_recommendation



DATA_PATH = "data/raw/bdg2/meters/cleaned/electricity_cleaned.csv"
BUILDING_ID = "Hog_education_Jewel"


def main():
    analysis = analyze_building_energy(DATA_PATH, BUILDING_ID)

    # Do not infer HVAC availability or building inefficiency from
    # electricity readings alone. Only use supported evidence.
    building_state = {
    "nighttime_anomaly": analysis["nighttime_anomaly"],
    "hvac_available": False,
    "occupancy_control_available": False,
    }

    recommendations = rank_interventions(building_state)

    print("\n=== BDG2 Energy Analysis ===")
    print("Building:", analysis["building_id"])
    print("Valid readings:", analysis["valid_readings"])
    print("Mean consumption:", analysis["mean_energy"])
    print("95th percentile:", analysis["p95_energy"])
    print("Nighttime anomaly:", analysis["nighttime_anomaly"])
    print("Total anomalies:", analysis["anomaly_count"])
    print("Anomaly rate:", analysis["anomaly_rate_pct"], "%")
    print("Nighttime anomalies:", analysis["nighttime_anomaly_count"])
    print(
        "Nighttime anomaly rate:",
        analysis["nighttime_anomaly_rate_pct"],
        "%",
    )

    print("\n=== Ranked Interventions ===")
    if not recommendations:
        print(
            "No intervention was triggered by the currently "
            "supported conditions. This does not prove the "
            "building is energy-efficient."
        )

    for index, item in enumerate(recommendations, start=1):
        print(f"\n{index}. {item['name']}")
        print("   Priority score:", item["priority_score"])
        print("   Reason:", explain_recommendation(item))


if __name__ == "__main__":
    main()
