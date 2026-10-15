
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data/processed/anomaly/selected_buildings.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data/processed/anomaly"

BUILDINGS = [
    "Robin_education_Kiera",
    "Robin_lodging_Pricilla",
    "Robin_office_Maryann",
    "Robin_office_Serena",
    "Robin_education_Derrick",
]


def build_features():
    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
    )

    results = []

    for building in BUILDINGS:
        series = df[["timestamp", building]].copy()
        series = series.rename(columns={building: "consumption"})

        # Handle the few missing readings by time interpolation.
        series["consumption"] = series["consumption"].interpolate(
            method="linear",
            limit=2,
            limit_area="inside",
        )

        # Remove rows where consumption is still missing.
        series = series.dropna(subset=["consumption"]).copy()

        # Time-based features.
        series["hour"] = series["timestamp"].dt.hour
        series["day_of_week"] = series["timestamp"].dt.dayofweek
        series["is_weekend"] = (
            series["day_of_week"] >= 5
        ).astype(int)

        # Rolling statistics use past readings only.
        shifted = series["consumption"].shift(1)

        series["rolling_mean_24h"] = shifted.rolling(
            window=24,
            min_periods=12,
        ).mean()

        series["rolling_std_24h"] = shifted.rolling(
            window=24,
            min_periods=12,
        ).std()

        # Deviation from recent consumption.
        series["deviation_24h"] = (
            series["consumption"] - series["rolling_mean_24h"]
        )

        series["building_id"] = building

        results.append(series)

    features = pd.concat(results, ignore_index=True)

    # Initial rolling windows do not have enough past observations.
    features = features.dropna(
        subset=["rolling_mean_24h", "rolling_std_24h"]
    )

    output_file = OUTPUT_DIR / "anomaly_features.csv"
    features.to_csv(output_file, index=False)

    print("Feature dataset shape:", features.shape)
    print("\nColumns:")
    print(features.columns.tolist())
    print("\nMissing values:")
    print(features.isna().sum())
    print("\nRows per building:")
    print(features["building_id"].value_counts())
    print("\nSaved to:", output_file)


if __name__ == "__main__":
    build_features()
