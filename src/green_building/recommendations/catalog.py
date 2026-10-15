
"""Catalogue of candidate green-building interventions.

Scores are initial assumptions, not measured energy savings.
"""

INTERVENTIONS = [
    {
        "id": "hvac_schedule",
        "name": "Optimize HVAC schedules",
        "description": "Review HVAC operation outside occupied hours.",
        "conditions": ["nighttime_anomaly", "high_energy"],
        "impact": 0.80,
        "urgency": 0.70,
        "feasibility": 0.90,
        "cost_effectiveness": 0.90,
        "requires": "hvac_available",
    },
    {
        "id": "equipment_inspection",
        "name": "Inspect building equipment",
        "description": "Investigate unusual energy consumption.",
        "conditions": ["nighttime_anomaly"],
        "impact": 0.70,
        "urgency": 1.00,
        "feasibility": 0.80,
        "cost_effectiveness": 0.70,
        "requires": None,
    },
    {
        "id": "occupancy_control",
        "name": "Improve occupancy-based controls",
        "description": "Reduce unnecessary operation in unoccupied areas.",
        "conditions": ["nighttime_anomaly", "high_energy"],
        "impact": 0.75,
        "urgency": 0.65,
        "feasibility": 0.75,
        "cost_effectiveness": 0.80,
        "requires": "occupancy_control_available",
    },
    {
        "id": "lighting_upgrade",
        "name": "Upgrade lighting to LEDs",
        "description": "Evaluate replacing inefficient lighting.",
        "conditions": ["inefficient_building"],
        "impact": 0.65,
        "urgency": 0.40,
        "feasibility": 0.75,
        "cost_effectiveness": 0.70,
        "requires": None,
    },
    {
        "id": "insulation_review",
        "name": "Review building insulation",
        "description": "Investigate potential thermal-efficiency improvements.",
        "conditions": ["inefficient_building", "high_energy"],
        "impact": 0.80,
        "urgency": 0.50,
        "feasibility": 0.45,
        "cost_effectiveness": 0.50,
        "requires": None,
    },
]
