
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/anomaly"

v1 = pd.read_csv(DATA_DIR / "anomaly_predictions.csv",
                 parse_dates=["timestamp"])
v2 = pd.read_csv(DATA_DIR / "anomaly_predictions_v2.csv",
                 parse_dates=["timestamp"])

keys = ["building_id", "timestamp"]

comparison = v1[keys + ["is_anomaly"]].merge(
    v2[keys + ["is_anomaly"]],
    on=keys,
    how="inner",
    suffixes=("_v1", "_v2"),
    validate="one_to_one",
)

comparison["is_anomaly_v1"] = (
    comparison["is_anomaly_v1"].astype(bool)
)
comparison["is_anomaly_v2"] = (
    comparison["is_anomaly_v2"].astype(bool)
)

print("Common evaluation readings:", len(comparison))
print("\nResults on identical timestamps:")

for building, group in comparison.groupby("building_id"):
    a = group["is_anomaly_v1"]
    b = group["is_anomaly_v2"]

    print(f"\n{building}")
    print("V1 flagged:", int(a.sum()))
    print("V2 flagged:", int(b.sum()))
    print("Flagged by both:", int((a & b).sum()))
    print("V1 only:", int((a & ~b).sum()))
    print("V2 only:", int((~a & b).sum()))

print("\nOverall")
print("V1 flagged:", int(comparison["is_anomaly_v1"].sum()))
print("V2 flagged:", int(comparison["is_anomaly_v2"].sum()))
print(
    "Agreement:",
    f"{(comparison['is_anomaly_v1'] == comparison['is_anomaly_v2']).mean() * 100:.2f}%",
)
