
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data/processed/anomaly/anomaly_predictions.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])

print("Total evaluation readings:", len(df))
print("Total flagged anomalies:", int(df["is_anomaly"].sum()))

for building, group in df.groupby("building_id"):
    anomalies = group[group["is_anomaly"]].copy()
    anomalies = anomalies.sort_values("anomaly_score")

    print(f"\n{'=' * 65}")
    print("BUILDING:", building)
    print("Flagged anomalies:", len(anomalies))

    if anomalies.empty:
        print("No anomalies flagged.")
        continue

    print("\n10 most anomalous readings:")
    columns = [
        "timestamp",
        "consumption",
        "rolling_mean_24h",
        "rolling_std_24h",
        "deviation_24h",
        "anomaly_score",
    ]

    print(
        anomalies[columns]
        .head(10)
        .round(3)
        .to_string(index=False)
    )

print("\nInspection complete.")
