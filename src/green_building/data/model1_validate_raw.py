"""
Model 1 - Building Efficiency Assessment
Step 5: Raw-data validation for CBECS 2018.

Reads the raw CSV (read-only), checks its structure and quality,
and saves a text report + a per-column summary CSV.

Run from the repo root:
    python src/green_building/data/model1_validate_raw.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---- Paths (relative to the repo root, no machine-specific paths) ----
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = REPO_ROOT / "data" / "raw" / "model1_cbecs2018" / "cbecs2018_final_public.csv"
OUT_DIR = REPO_ROOT / "reports" / "model1" / "validation"

# Columns we expect to need, taken from the EIA User's Guide.
# We CHECK that they exist; we do not assume it.
KEY_COLUMNS = {
    "PUBID": "Building ID",
    "PBA": "Principal building activity (building type code)",
    "PUBCLIM": "Climate zone group",
    "SQFT": "Floor area (square feet)",
    "MFBTU": "Annual major fuel consumption (thousand Btu)",
    "ELCNS": "Annual electricity consumption (kWh)",
    "FINALWT": "Survey weight",
    "YRCONC": "Year of construction (category)",
    "WKHRS": "Weekly operating hours",
    "NWKER": "Number of workers",
    "NFLOOR": "Number of floors",
    "REGION": "Census region",
}

lines = []  # everything printed is also saved to the report


def report(text=""):
    print(text)
    lines.append(str(text))


def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV}\nRun this from the repo root.")

    df = pd.read_csv(RAW_CSV, low_memory=False)

    report("=" * 60)
    report("MODEL 1 - RAW DATA VALIDATION (CBECS 2018)")
    report("=" * 60)
    report(f"File: {RAW_CSV.relative_to(REPO_ROOT)}")
    report(f"Size: {RAW_CSV.stat().st_size / 1e6:.2f} MB")
    report(f"Rows (buildings): {df.shape[0]}")
    report(f"Columns: {df.shape[1]}")

    # 1. Data types
    report("\n--- Column data types ---")
    report(df.dtypes.value_counts().to_string())

    # 2. Duplicates
    report("\n--- Duplicates ---")
    report(f"Fully duplicated rows: {df.duplicated().sum()}")
    if "PUBID" in df.columns:
        report(f"Duplicated PUBID values: {df['PUBID'].duplicated().sum()}")

    # 3. Key columns: present? missing values? range?
    report("\n--- Key columns ---")
    missing_keys = [c for c in KEY_COLUMNS if c not in df.columns]
    report(f"Missing key columns: {missing_keys if missing_keys else 'none'}")
    for col, meaning in KEY_COLUMNS.items():
        if col not in df.columns:
            continue
        s = df[col]
        info = f"{col:8s} | {meaning:48s} | missing={s.isna().sum():5d}"
        if pd.api.types.is_numeric_dtype(s):
            info += f" | min={s.min():.4g} max={s.max():.4g} mean={s.mean():.4g}"
        report(info)

    # 4. Categories needed for peer groups
    for col in ["PBA", "PUBCLIM"]:
        if col in df.columns:
            report(f"\n--- Value counts: {col} ---")
            report(df[col].value_counts(dropna=False).sort_index().to_string())

    # 5. Energy and floor-area sanity checks
    report("\n--- Energy / area sanity checks ---")
    for col in ["MFBTU", "SQFT"]:
        if col in df.columns:
            s = df[col]
            report(f"{col}: missing={s.isna().sum()}, zero={(s == 0).sum()}, negative={(s < 0).sum()}")

    # 6. Negative and infinite values across ALL numeric columns
    num = df.select_dtypes(include=[np.number])
    neg_cols = (num < 0).sum()
    neg_cols = neg_cols[neg_cols > 0]
    report("\n--- Negative values (all numeric columns) ---")
    report(neg_cols.to_string() if len(neg_cols) else "none")
    report(f"Infinite values in dataset: {int(np.isinf(num).sum().sum())}")

    # 7. Overall missingness
    miss_pct = df.isna().mean() * 100
    report("\n--- Missingness overview ---")
    report(f"Columns with no missing values: {(miss_pct == 0).sum()}")
    report(f"Columns >50% missing: {(miss_pct > 50).sum()}")
    report(f"Columns 100% missing: {(miss_pct == 100).sum()}")
    report("Note: in CBECS a blank often means 'not applicable' (e.g. no gas used), not an error.")

    # 8. Possible target-leakage columns (energy use / cost)
    leak = [c for c in df.columns
            if c.endswith(("CNS", "BTU", "EXP")) or c in ("MFBTU", "MFEXP")]
    report("\n--- Possible leakage columns (energy consumption / expenditure) ---")
    report(f"Count: {len(leak)}")
    report(", ".join(leak[:40]) + (" ..." if len(leak) > 40 else ""))
    report("These must NOT be used as model inputs if labels come from energy use.")

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "raw_validation_report.txt").write_text("\n".join(lines))
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": miss_pct.round(2),
        "n_unique": df.nunique(),
    })
    summary.to_csv(OUT_DIR / "raw_column_summary.csv", index_label="column")
    print(f"\nSaved: {(OUT_DIR / 'raw_validation_report.txt').relative_to(REPO_ROOT)}")
    print(f"Saved: {(OUT_DIR / 'raw_column_summary.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()