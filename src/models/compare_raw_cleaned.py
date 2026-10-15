
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/raw/bdg2/meters"

CASES = [
    ("Robin_education_Kiera", "2017-11-20 15:00:00"),
    ("Robin_office_Maryann", "2017-08-26 07:00:00"),
]

def main():
    for building, time_text in CASES:
        target = pd.Timestamp(time_text)

        print(f"\n{'=' * 65}")
        print("Building:", building)
        print("Target:", target)

        for filename in [
            "raw/electricity.csv",
            "cleaned/electricity_cleaned.csv",
        ]:
            path = DATA_DIR / filename

            df = pd.read_csv(
                path,
                usecols=["timestamp", building],
                parse_dates=["timestamp"],
            )

            window = df[
                df["timestamp"].between(
                    target - pd.Timedelta(hours=3),
                    target + pd.Timedelta(hours=3),
                )
            ]

            print(f"\n{filename}")
            print(window.to_string(index=False))

if __name__ == "__main__":
    main()
