
from pathlib import Path
import pandas as pd

BASE = Path("data/raw/bdg2")

files = {
    "electricity": BASE / "meters/cleaned/electricity_cleaned.csv",
    "metadata": BASE / "metadata/metadata.csv",
}

for name, path in files.items():
    print(f"\n{'=' * 60}")
    print(f"{name.upper()}: {path}")
    print("=" * 60)

    df = pd.read_csv(path, nrows=5)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.to_string(index=False))

    print("\nData types:")
    print(df.dtypes)
