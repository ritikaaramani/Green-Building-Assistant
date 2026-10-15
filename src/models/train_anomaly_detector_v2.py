
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/anomaly/anomaly_features_v2.csv"
OUTPUT_FILE = ROOT / "data/processed/anomaly/anomaly_predictions_v2.csv"

FEATURE_COLUMNS = [
    "consumption",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_weekend",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_median_24h",
    "deviation_24h",
    "median_deviation_24h",
]


def train_detector():
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    results = []

    for building, group in df.groupby("building_id", sort=True):
        group = group.sort_values("timestamp").copy()

        split = int(len(group) * 0.8)
        train = group.iloc[:split]
        test = group.iloc[split:].copy()

        model = IsolationForest(
            n_estimators=100,
            contamination=0.02,
            random_state=42,
            n_jobs=1,
        )

        model.fit(train[FEATURE_COLUMNS])

        test["anomaly_prediction"] = model.predict(
            test[FEATURE_COLUMNS]
        )
        test["anomaly_score"] = model.decision_function(
            test[FEATURE_COLUMNS]
        )
        test["is_anomaly"] = (
            test["anomaly_prediction"] == -1
        )

        count = int(test["is_anomaly"].sum())
        rate = test["is_anomaly"].mean() * 100

        print(f"\nBuilding: {building}")
        print("Evaluation rows:", len(test))
        print("Flagged anomalies:", count)
        print(f"Flagged percentage: {rate:.2f}%")

        results.append(test)

    predictions = pd.concat(results, ignore_index=True)
    predictions.to_csv(OUTPUT_FILE, index=False)

    print("\nTotal evaluation rows:", len(predictions))
    print("Total flagged:", int(predictions["is_anomaly"].sum()))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    train_detector()
