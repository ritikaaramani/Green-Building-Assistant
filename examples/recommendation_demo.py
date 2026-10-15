
from green_building.recommendations import (
    rank_interventions,
    explain_recommendation,
)

building_state = {
    "efficiency_category": "moderately_inefficient",
    "predicted_energy_kwh": 145.0,
    "expected_energy_kwh": 100.0,
    "nighttime_anomaly": True,
    "hvac_available": True,
    "occupancy_control_available": True,
}

recommendations = rank_interventions(building_state, top_n=5)

print("\nAI-BASED GREEN BUILDING ASSISTANT")
print("=" * 45)
print("Scenario: High predicted energy and nighttime anomaly")

for rank, item in enumerate(recommendations, start=1):
    print(f"\nRank {rank}: {item['name']}")
    print(f"Priority score: {item['priority_score']:.2f}")
    print(f"Reason: {explain_recommendation(item)}")
