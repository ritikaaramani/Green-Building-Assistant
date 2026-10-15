"""
src/data/load_data.py
=====================
Reusable functions for loading and validating the UCI Energy Efficiency
dataset (ENB2012).

Primary dataset
---------------
ENB2012_data.xlsx / ENB2012_data.csv
  768 rows × 10 columns (X1–X8 features, Y1 heating load, Y2 cooling load)
  Source: https://archive.ics.uci.edu/dataset/242/energy+efficiency

The energydata_complete.csv dataset (Appliances Energy Prediction, 19 735 rows,
29 columns) has a completely different schema and prediction target (appliance
energy consumption). It is NOT combined with ENB2012 here.

Usage
-----
    from src.data.load_data import load_energy_efficiency, validate_dataframe

    df = load_energy_efficiency("dataset-sai/ENB2012_data.xlsx")
    report = validate_dataframe(df)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "COLUMN_RENAME_MAP",
    "FEATURE_COLUMNS",
    "TARGET_COLUMNS",
    "load_energy_efficiency",
    "validate_dataframe",
    "validation_report",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mapping from compact UCI column names to human-readable names.
COLUMN_RENAME_MAP: Dict[str, str] = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
    "Y1": "heating_load",
    "Y2": "cooling_load",
}

#: Feature column names (after renaming).
FEATURE_COLUMNS: list[str] = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
]

#: Target column names (after renaming).
TARGET_COLUMNS: list[str] = ["heating_load", "cooling_load"]

#: All expected columns after renaming.
_ALL_EXPECTED_COLS: list[str] = FEATURE_COLUMNS + TARGET_COLUMNS

#: Valid orientation values in the dataset (2, 3, 4, 5 = N, E, S, W).
_VALID_ORIENTATIONS = {2, 3, 4, 5}

#: Valid glazing area distribution values (0 = uniform, 1–5 = skewed).
_VALID_GLAZING_DIST = {0, 1, 2, 3, 4, 5}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_energy_efficiency(
    path: str | os.PathLike,
    *,
    rename_columns: bool = True,
) -> pd.DataFrame:
    """Load the UCI Energy Efficiency (ENB2012) dataset.

    Supports both Excel (.xlsx / .xls) and CSV (.csv) formats.  The function
    reads the file, optionally renames the compact X1–X8/Y1/Y2 headers to
    descriptive names, and returns a clean DataFrame.

    Parameters
    ----------
    path:
        Absolute or relative path to the raw data file.
    rename_columns:
        If *True* (default) the X/Y column names are replaced with the
        human-readable names defined in ``COLUMN_RENAME_MAP``.

    Returns
    -------
    pd.DataFrame
        Loaded (and optionally renamed) dataset.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is not supported, or if the file does not
        contain the expected ENB2012 column set.
    """
    path = Path(path)

    # ── 1. Existence check ─────────────────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path.resolve()}\n"
            "Check that the path is correct and the file has been placed in "
            "the project's dataset-sai/ folder."
        )

    # ── 2. Read file ───────────────────────────────────────────────────────
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        logger.info("Reading Excel file: %s", path)
        df = pd.read_excel(path, engine="openpyxl")
    elif suffix == ".csv":
        logger.info("Reading CSV file: %s", path)
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            "Provide a .xlsx, .xls, or .csv file."
        )

    logger.info("Loaded %d rows × %d columns from '%s'", *df.shape, path.name)

    # ── 3. Column validation (before rename) ───────────────────────────────
    _validate_raw_columns(df, path)

    # ── 4. Optional rename ─────────────────────────────────────────────────
    if rename_columns:
        df = df.rename(columns=COLUMN_RENAME_MAP)
        logger.info("Columns renamed to human-readable names.")

    return df


def _validate_raw_columns(df: pd.DataFrame, path: Path) -> None:
    """Check that the raw DataFrame has exactly the ENB2012 column set."""
    expected_raw = set(COLUMN_RENAME_MAP.keys())   # {"X1", …, "Y2"}
    found = set(df.columns)

    missing = expected_raw - found
    extra = found - expected_raw

    if missing or extra:
        raise ValueError(
            f"Unexpected column layout in '{path.name}'.\n"
            f"  Expected : {sorted(expected_raw)}\n"
            f"  Found    : {sorted(found)}\n"
            f"  Missing  : {sorted(missing)}\n"
            f"  Extra    : {sorted(extra)}\n"
            "Hint: Make sure you are loading the ENB2012 dataset, "
            "not energydata_complete.csv or another file."
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Run quality checks on a loaded (renamed) ENB2012 DataFrame.

    Checks performed
    ----------------
    * All expected columns are present.
    * Every column has a numeric dtype.
    * Missing-value counts per column.
    * Duplicate-row count.
    * Range checks for orientation and glazing_area_distribution.
    * Non-negative check for heating_load and cooling_load.

    Parameters
    ----------
    df:
        DataFrame as returned by :func:`load_energy_efficiency` with
        ``rename_columns=True``.

    Returns
    -------
    dict
        A dictionary with keys:
        ``"n_rows"``, ``"n_cols"``,
        ``"missing_per_column"``, ``"total_missing"``,
        ``"duplicate_rows"``, ``"dtype_errors"``,
        ``"range_errors"``, ``"is_valid"``.

    Raises
    ------
    ValueError
        If required columns are absent (schema mismatch).
    """
    # ── Schema check ───────────────────────────────────────────────────────
    missing_cols = [c for c in _ALL_EXPECTED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"DataFrame is missing expected columns: {missing_cols}.\n"
            "Did you load the DataFrame with rename_columns=True?"
        )

    report: Dict[str, Any] = {}

    # ── Dimensions ─────────────────────────────────────────────────────────
    report["n_rows"] = len(df)
    report["n_cols"] = len(df.columns)

    # ── Missing values ─────────────────────────────────────────────────────
    missing_per_col = df[_ALL_EXPECTED_COLS].isnull().sum().to_dict()
    report["missing_per_column"] = missing_per_col
    report["total_missing"] = sum(missing_per_col.values())

    # ── Duplicates ─────────────────────────────────────────────────────────
    report["duplicate_rows"] = int(df.duplicated().sum())

    # ── dtype checks ───────────────────────────────────────────────────────
    dtype_errors: Dict[str, str] = {}
    for col in _ALL_EXPECTED_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            dtype_errors[col] = str(df[col].dtype)
    report["dtype_errors"] = dtype_errors

    # ── Range / domain checks ──────────────────────────────────────────────
    range_errors: Dict[str, str] = {}

    # Orientation must be one of {2, 3, 4, 5}
    bad_orient = df["orientation"][~df["orientation"].isin(_VALID_ORIENTATIONS)]
    if not bad_orient.empty:
        range_errors["orientation"] = (
            f"{len(bad_orient)} rows with invalid values: "
            f"{sorted(bad_orient.unique().tolist())}"
        )

    # Glazing area distribution must be one of {0, 1, 2, 3, 4, 5}
    bad_glazing_dist = df["glazing_area_distribution"][
        ~df["glazing_area_distribution"].isin(_VALID_GLAZING_DIST)
    ]
    if not bad_glazing_dist.empty:
        range_errors["glazing_area_distribution"] = (
            f"{len(bad_glazing_dist)} rows with invalid values: "
            f"{sorted(bad_glazing_dist.unique().tolist())}"
        )

    # Targets must be non-negative (physical energy loads)
    for target in TARGET_COLUMNS:
        neg_mask = df[target] < 0
        if neg_mask.any():
            range_errors[target] = (
                f"{neg_mask.sum()} rows with negative values."
            )

    report["range_errors"] = range_errors

    # ── Overall validity flag ──────────────────────────────────────────────
    report["is_valid"] = (
        report["total_missing"] == 0
        and len(dtype_errors) == 0
        and len(range_errors) == 0
    )

    return report


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def validation_report(df: pd.DataFrame) -> str:
    """Return a formatted validation report string for *df*.

    Parameters
    ----------
    df:
        DataFrame as returned by :func:`load_energy_efficiency`.

    Returns
    -------
    str
        Multi-line plain-text report suitable for printing or logging.
    """
    r = validate_dataframe(df)
    sep = "=" * 60
    lines = [
        sep,
        "  ENB2012 Energy Efficiency — Validation Report",
        sep,
        f"  Rows          : {r['n_rows']}",
        f"  Columns       : {r['n_cols']}",
        f"  Total missing : {r['total_missing']}",
        f"  Duplicate rows: {r['duplicate_rows']}",
        "",
        "  Missing values per column:",
    ]
    for col, cnt in r["missing_per_column"].items():
        flag = "  ⚠" if cnt > 0 else "  ✓"
        lines.append(f"    {flag} {col:<30s}: {cnt}")

    if r["dtype_errors"]:
        lines += ["", "  ⚠ dtype errors (expected numeric):"]
        for col, dtype in r["dtype_errors"].items():
            lines.append(f"    {col}: {dtype}")
    else:
        lines.append("\n  ✓ All columns have numeric dtypes.")

    if r["range_errors"]:
        lines += ["", "  ⚠ Range / domain errors:"]
        for col, msg in r["range_errors"].items():
            lines.append(f"    {col}: {msg}")
    else:
        lines.append("  ✓ All range checks passed.")

    lines += [
        "",
        f"  Overall status: {'✓ VALID' if r['is_valid'] else '✗ ISSUES FOUND'}",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m src.data.load_data <path_to_dataset>")
        sys.exit(1)

    data_path = sys.argv[1]
    print(f"\nLoading: {data_path}")
    dataframe = load_energy_efficiency(data_path)
    print(validation_report(dataframe))
