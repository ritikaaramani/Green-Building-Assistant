"""
tests/test_data.py
==================
Unit tests for src.data.load_data and src.data.clean_data.

Run with:
    conda run -n green-building pytest tests/test_data.py -v

The tests use the real ENB2012 Excel file located in dataset-sai/.
If that file is unavailable the test suite skips automatically rather
than failing, so CI runs without the raw data also pass cleanly.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Make sure the repo root is importable ──────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.load_data import (
    COLUMN_RENAME_MAP,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    load_energy_efficiency,
    validate_dataframe,
    validation_report,
)
from src.data.clean_data import run_cleaning_pipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

XLSX_PATH = REPO_ROOT / "dataset-sai" / "ENB2012_data.xlsx"
CSV_PATH = REPO_ROOT / "dataset-sai" / "ENB2012_data.csv"

_real_data_available = XLSX_PATH.exists()

skip_if_no_data = pytest.mark.skipif(
    not _real_data_available,
    reason=f"Real dataset not found at {XLSX_PATH}",
)


@pytest.fixture(scope="module")
def df_xlsx():
    """Load the ENB2012 Excel file once per test module."""
    return load_energy_efficiency(XLSX_PATH)


@pytest.fixture
def df_minimal_valid():
    """A tiny perfectly-valid ENB2012-shaped DataFrame for fast tests."""
    return pd.DataFrame(
        {
            "relative_compactness": [0.98, 0.74, 0.62],
            "surface_area": [514.5, 612.0, 808.5],
            "wall_area": [294.0, 318.5, 367.5],
            "roof_area": [110.25, 147.0, 220.5],
            "overall_height": [7.0, 5.25, 3.5],
            "orientation": [2, 3, 4],
            "glazing_area": [0.0, 0.1, 0.25],
            "glazing_area_distribution": [0, 1, 3],
            "heating_load": [15.55, 21.33, 10.89],
            "cooling_load": [21.33, 28.11, 14.44],
        }
    )


@pytest.fixture
def raw_xlsx_path():
    return XLSX_PATH


# ---------------------------------------------------------------------------
# 1. File-loading tests
# ---------------------------------------------------------------------------

class TestLoadEnergyEfficiency:

    @skip_if_no_data
    def test_load_xlsx_returns_dataframe(self, df_xlsx):
        """Loading the Excel file must return a non-empty DataFrame."""
        assert isinstance(df_xlsx, pd.DataFrame)
        assert len(df_xlsx) > 0

    @skip_if_no_data
    def test_xlsx_row_count(self, df_xlsx):
        """ENB2012 must have exactly 768 rows."""
        assert len(df_xlsx) == 768, (
            f"Expected 768 rows, got {len(df_xlsx)}"
        )

    @skip_if_no_data
    def test_xlsx_column_count(self, df_xlsx):
        """ENB2012 must have exactly 10 columns."""
        assert df_xlsx.shape[1] == 10

    @skip_if_no_data
    def test_column_names_renamed_correctly(self, df_xlsx):
        """All X1–X8/Y1/Y2 columns must be renamed to descriptive names."""
        expected = set(FEATURE_COLUMNS + TARGET_COLUMNS)
        assert set(df_xlsx.columns) == expected, (
            f"Column mismatch.\nExpected: {sorted(expected)}\n"
            f"Got     : {sorted(df_xlsx.columns)}"
        )

    @skip_if_no_data
    def test_feature_columns_present(self, df_xlsx):
        """Every feature column must be present."""
        for col in FEATURE_COLUMNS:
            assert col in df_xlsx.columns, f"Feature column missing: {col}"

    @skip_if_no_data
    def test_target_columns_present(self, df_xlsx):
        """Both target columns (heating_load, cooling_load) must be present."""
        for col in TARGET_COLUMNS:
            assert col in df_xlsx.columns, f"Target column missing: {col}"

    @skip_if_no_data
    def test_all_columns_numeric(self, df_xlsx):
        """Every column must be numeric (float or int)."""
        non_numeric = [
            c for c in df_xlsx.columns
            if not pd.api.types.is_numeric_dtype(df_xlsx[c])
        ]
        assert non_numeric == [], (
            f"Non-numeric columns found: {non_numeric}"
        )

    @skip_if_no_data
    def test_csv_load_same_shape(self):
        """ENB2012 CSV must load with the same shape as the Excel version."""
        if not CSV_PATH.exists():
            pytest.skip(f"CSV not found at {CSV_PATH}")
        df_csv = load_energy_efficiency(CSV_PATH)
        assert df_csv.shape[0] == 768
        assert df_csv.shape[1] == 10

    def test_load_nonexistent_file_raises(self):
        """Loading a missing file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_energy_efficiency("this/does/not/exist.xlsx")

    def test_load_unsupported_extension_raises(self):
        """Loading an unsupported file type must raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file extension"):
                load_energy_efficiency(tmp)
        finally:
            os.unlink(tmp)

    def test_load_wrong_schema_raises(self):
        """A CSV with wrong column names must raise ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("col_a,col_b,col_c\n1,2,3\n")
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="Unexpected column layout"):
                load_energy_efficiency(tmp)
        finally:
            os.unlink(tmp)

    def test_rename_columns_false_keeps_x_names(self):
        """With rename_columns=False the original X1/Y1 names must be kept."""
        if not XLSX_PATH.exists():
            pytest.skip("Real dataset not found")
        df = load_energy_efficiency(XLSX_PATH, rename_columns=False)
        assert "X1" in df.columns
        assert "Y1" in df.columns
        assert "relative_compactness" not in df.columns


# ---------------------------------------------------------------------------
# 2. Validation tests
# ---------------------------------------------------------------------------

class TestValidateDataframe:

    def test_valid_frame_passes(self, df_minimal_valid):
        """A clean minimal DataFrame must pass validation."""
        result = validate_dataframe(df_minimal_valid)
        assert result["is_valid"] is True
        assert result["total_missing"] == 0
        assert result["duplicate_rows"] == 0
        assert result["dtype_errors"] == {}
        assert result["range_errors"] == {}

    def test_missing_values_detected(self, df_minimal_valid):
        """Injected NaN must be flagged in the validation report."""
        df = df_minimal_valid.copy()
        df.loc[0, "heating_load"] = np.nan
        result = validate_dataframe(df)
        assert result["total_missing"] == 1
        assert result["missing_per_column"]["heating_load"] == 1
        assert result["is_valid"] is False

    def test_duplicate_rows_detected(self, df_minimal_valid):
        """Appending an exact duplicate row must increment duplicate count."""
        df = pd.concat(
            [df_minimal_valid, df_minimal_valid.iloc[[0]]], ignore_index=True
        )
        result = validate_dataframe(df)
        assert result["duplicate_rows"] == 1

    def test_invalid_orientation_detected(self, df_minimal_valid):
        """Orientation value 99 (not in {2,3,4,5}) must appear in range_errors."""
        df = df_minimal_valid.copy()
        df.loc[0, "orientation"] = 99
        result = validate_dataframe(df)
        assert "orientation" in result["range_errors"]
        assert result["is_valid"] is False

    def test_invalid_glazing_dist_detected(self, df_minimal_valid):
        """Glazing distribution value 9 must appear in range_errors."""
        df = df_minimal_valid.copy()
        df.loc[1, "glazing_area_distribution"] = 9
        result = validate_dataframe(df)
        assert "glazing_area_distribution" in result["range_errors"]
        assert result["is_valid"] is False

    def test_negative_target_detected(self, df_minimal_valid):
        """A negative heating_load must be flagged as a range error."""
        df = df_minimal_valid.copy()
        df.loc[2, "heating_load"] = -5.0
        result = validate_dataframe(df)
        assert "heating_load" in result["range_errors"]
        assert result["is_valid"] is False

    def test_non_numeric_column_detected(self, df_minimal_valid):
        """Casting a numeric column to str must appear in dtype_errors."""
        df = df_minimal_valid.copy()
        df["surface_area"] = df["surface_area"].astype(str)
        result = validate_dataframe(df)
        assert "surface_area" in result["dtype_errors"]
        assert result["is_valid"] is False

    def test_missing_column_raises(self, df_minimal_valid):
        """A DataFrame missing an expected column must raise ValueError."""
        df = df_minimal_valid.drop(columns=["cooling_load"])
        with pytest.raises(ValueError, match="missing expected columns"):
            validate_dataframe(df)

    @skip_if_no_data
    def test_real_dataset_passes_validation(self, df_xlsx):
        """The real ENB2012 Excel dataset must pass all validation checks."""
        result = validate_dataframe(df_xlsx)
        assert result["is_valid"] is True, (
            f"Real dataset failed validation:\n"
            f"  missing: {result['total_missing']}\n"
            f"  dtype errors: {result['dtype_errors']}\n"
            f"  range errors: {result['range_errors']}"
        )

    def test_validation_report_is_string(self, df_minimal_valid):
        """validation_report() must return a non-empty string."""
        report = validation_report(df_minimal_valid)
        assert isinstance(report, str)
        assert len(report) > 0


# ---------------------------------------------------------------------------
# 3. Cleaning-pipeline tests
# ---------------------------------------------------------------------------

class TestCleaningPipeline:

    @skip_if_no_data
    def test_pipeline_runs_and_produces_output(self, tmp_path):
        """The cleaning pipeline must complete without error."""
        out_csv = tmp_path / "clean.csv"
        rep_txt = tmp_path / "report.txt"
        result = run_cleaning_pipeline(
            raw_path=XLSX_PATH,
            output_path=out_csv,
            report_path=rep_txt,
        )
        assert out_csv.exists(), "Cleaned CSV not written."
        assert rep_txt.exists(), "Cleaning report not written."
        assert "df_clean" in result
        assert "report_text" in result

    @skip_if_no_data
    def test_cleaned_csv_has_correct_columns(self, tmp_path):
        """Cleaned CSV must contain all 10 renamed columns."""
        out_csv = tmp_path / "clean.csv"
        result = run_cleaning_pipeline(
            raw_path=XLSX_PATH,
            output_path=out_csv,
            report_path=tmp_path / "r.txt",
        )
        df_clean = pd.read_csv(out_csv)
        expected = set(FEATURE_COLUMNS + TARGET_COLUMNS)
        assert set(df_clean.columns) == expected

    @skip_if_no_data
    def test_cleaned_csv_has_no_missing_values(self, tmp_path):
        """Cleaned CSV must have zero missing values."""
        out_csv = tmp_path / "clean.csv"
        result = run_cleaning_pipeline(
            raw_path=XLSX_PATH,
            output_path=out_csv,
            report_path=tmp_path / "r.txt",
        )
        df_clean = pd.read_csv(out_csv)
        assert df_clean.isnull().sum().sum() == 0

    @skip_if_no_data
    def test_raw_file_unchanged_after_pipeline(self, tmp_path):
        """The raw Excel file must not be modified by the pipeline."""
        import hashlib

        def md5(p: Path) -> str:
            h = hashlib.md5()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()

        checksum_before = md5(XLSX_PATH)
        run_cleaning_pipeline(
            raw_path=XLSX_PATH,
            output_path=tmp_path / "clean.csv",
            report_path=tmp_path / "r.txt",
        )
        checksum_after = md5(XLSX_PATH)
        assert checksum_before == checksum_after, (
            "Raw file was modified by the cleaning pipeline!"
        )

    @skip_if_no_data
    def test_decisions_list_populated(self, tmp_path):
        """Pipeline must return at least one documented cleaning decision."""
        result = run_cleaning_pipeline(
            raw_path=XLSX_PATH,
            output_path=tmp_path / "clean.csv",
            report_path=tmp_path / "r.txt",
        )
        assert isinstance(result["decisions"], list)
        assert len(result["decisions"]) >= 1

    def test_pipeline_raises_on_missing_raw_file(self, tmp_path):
        """Pipeline must raise FileNotFoundError for a missing raw file."""
        with pytest.raises(FileNotFoundError):
            run_cleaning_pipeline(
                raw_path="no_such_file.xlsx",
                output_path=tmp_path / "clean.csv",
                report_path=tmp_path / "r.txt",
            )
