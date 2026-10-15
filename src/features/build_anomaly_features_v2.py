
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/anomaly/selected_buildings.csv"
OUTPUT_FILE = ROOT / "data/processed/anomaly/anomaly_features_v2.csv"

BUILDINGS = [
    "Robin_education_Kiera",
    "Robin_lodging_Pricilla",
    "Robin_office_Maryann",
    "Robin_office_Serena",
    "Robin_education_Derrick",
]


def build_features():
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    results = []

    for building in BUILDINGS:
        group = df[["timestamp", building]].copy()
        group = group.rename(columns={building: "consumption"})
        group = group.sort_values("timestamp")

        # Interpolate only the two known internal missing readings.
        group["consumption"] = group["consumption"].interpolate(
            method="linear",
            limit=2,
            limit_area="inside",
        )
        group = group.dropna(subset=["consumption"]).copy()

        # Cyclical time features: midnight is close to 23:00.
        hour = group["timestamp"].dt.hour
        day = group["timestamp"].dt.dayofweek

        group["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        group["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        group["day_sin"] = np.sin(2 * np.pi * day / 7)
        group["day_cos"] = np.cos(2 * np.pi * day / 7)
        group["is_weekend"] = (day >= 5).astype(int)

        # Past readings only: no future-data leakage.
        past = group["consumption"].shift(1)

        group["rolling_mean_24h"] = past.rolling(
            24, min_periods=24
        ).mean()

        group["rolling_std_24h"] = past.rolling(
            24, min_periods=24
        ).std()

        group["rolling_median_24h"] = past.rolling(
            24, min_periods=24
        ).median()

        group["deviation_24h"] = (
            group["consumption"] - group["rolling_mean_24h"]
        )

        group["median_deviation_24h"] = (
            group["consumption"] - group["rolling_median_24h"]
        )

        group["building_id"] = building
        results.append(group)

    features = pd.concat(results, ignore_index=True)

    required = [
        "rolling_mean_24h",
        "rolling_std_24h",
        "rolling_median_24h",
    ]
    features = features.dropna(subset=required)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_FILE, index=False)

    print("Feature dataset shape:", features.shape)
    print("Buildings:", features["building_id"].nunique())
    print("Missing values:", int(features.isna().sum().sum()))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    build_features()
