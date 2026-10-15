
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/anomaly"

v1 = pd.read_csv(
    DATA_DIR / "anomaly_predictions.csv",
    parse_dates=["timestamp"]
)

v2 = pd.read_csv(
    DATA_DIR / "anomaly_predictions_v2.csv",
    parse_dates=["timestamp"]
)

keys = ["building_id", "timestamp"]

comparison = v1[keys + ["consumption", "is_anomaly"]].merge(
    v2[keys + ["is_anomaly"]],
    on=keys,
    how="inner",
    suffixes=("_v1", "_v2"),
    validate="one_to_one"
)

# Convert anomaly labels safely, including CSV boolean values.
for col in ["is_anomaly_v1", "is_anomaly_v2"]:
    comparison[col] = (
        comparison[col].astype(str).str.lower() == "true"
    )

disagreements = comparison[
    comparison["is_anomaly_v1"] != comparison["is_anomaly_v2"]
].copy()

disagreements["model_difference"] = disagreements.apply(
    lambda row: (
        "V1 only" if row["is_anomaly_v1"] else "V2 only"
    ),
    axis=1
)

print("Total disagreements:", len(disagreements))

for building in disagreements["building_id"].unique():
    print(f"\n--- {building} ---")

    building_data = disagreements[
        disagreements["building_id"] == building
    ]

    print(
        building_data[
            ["timestamp", "consumption", "model_difference"]
        ].head(10).to_string(index=False)
    )

# Save all disagreements for detailed inspection.
output = DATA_DIR / "model_disagreements.csv"
disagreements.to_csv(output, index=False)

print(f"\nSaved disagreement report: {output}")
