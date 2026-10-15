
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = (
    PROJECT_ROOT
    / "data/raw/bdg2/meters/cleaned/electricity_cleaned.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data/processed/anomaly"
OUTPUT_FILE = OUTPUT_DIR / "selected_buildings.csv"

BUILDINGS = [
    "Robin_education_Kiera",
    "Robin_lodging_Pricilla",
    "Robin_office_Maryann",
    "Robin_office_Serena",
    "Robin_education_Derrick",
]


def prepare_data():
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        RAW_DATA,
        usecols=["timestamp"] + BUILDINGS,
        parse_dates=["timestamp"],
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    df[BUILDINGS] = df[BUILDINGS].replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    print("Dataset shape:", df.shape)
    print(
        "Date range:",
        df["timestamp"].min(),
        "to",
        df["timestamp"].max(),
    )

    print("\nMissing values:")
    print(df[BUILDINGS].isna().sum())

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved to:", OUTPUT_FILE)
    print(
        "Output size:",
        f"{OUTPUT_FILE.stat().st_size / (1024**2):.2f} MB",
    )


if __name__ == "__main__":
    prepare_data()
