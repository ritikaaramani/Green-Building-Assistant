
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data/processed/anomaly/anomaly_features.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data/processed/anomaly"

FEATURE_COLUMNS = [
    "consumption",
    "hour",
    "day_of_week",
    "is_weekend",
    "rolling_mean_24h",
    "rolling_std_24h",
    "deviation_24h",
]


def train_detector():
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    results = []

    for building, group in df.groupby("building_id", sort=True):
        group = group.sort_values("timestamp").copy()

        X = group[FEATURE_COLUMNS]

        # Train on the earlier 80%; evaluate on the later 20%.
        split = int(len(group) * 0.8)
        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        model = make_pipeline(
            StandardScaler(),
            IsolationForest(
                n_estimators=100,
                contamination=0.02,
                random_state=42,
                n_jobs=1,
            ),
        )

        model.fit(X_train)

        # -1 means anomaly; 1 means normal.
        predictions = model.predict(X_test)
        scores = model.decision_function(X_test)

        evaluated = group.iloc[split:].copy()
        evaluated["anomaly_prediction"] = predictions
        evaluated["anomaly_score"] = scores
        evaluated["is_anomaly"] = predictions == -1

        results.append(evaluated)

        print(f"\nBuilding: {building}")
        print("Training rows:", len(X_train))
        print("Evaluation rows:", len(X_test))
        print("Flagged anomalies:", int((predictions == -1).sum()))
        print(
            "Flagged percentage:",
            f"{(predictions == -1).mean() * 100:.2f}%",
        )

    output_file = OUTPUT_DIR / "anomaly_predictions.csv"
    pd.concat(results, ignore_index=True).to_csv(
        output_file,
        index=False,
    )

    print("\nPredictions saved to:", output_file)


if __name__ == "__main__":
    train_detector()
