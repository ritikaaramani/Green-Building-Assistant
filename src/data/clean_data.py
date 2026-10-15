"""
src/data/clean_data.py
======================
Cleaning workflow for the UCI Energy Efficiency (ENB2012) dataset.

Design principles
-----------------
* The original raw file is **never modified or overwritten**.
* Every cleaning step is documented and justified.
* No rows or columns are silently dropped.
* A before/after report is printed and saved alongside the cleaned file.
* The output is written to ``data/processed/energy_efficiency_clean.csv``.

Usage (CLI)
-----------
    conda run -n green-building python -m src.data.clean_data

Usage (import)
--------------
    from src.data.clean_data import run_cleaning_pipeline

    summary = run_cleaning_pipeline(
        raw_path="dataset-sai/ENB2012_data.xlsx",
        output_path="data/processed/energy_efficiency_clean.csv",
    )
    print(summary["report_text"])
"""

from __future__ import annotations

import logging
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.data.load_data import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    load_energy_efficiency,
    validate_dataframe,
)

__all__ = ["run_cleaning_pipeline"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------
_DEFAULT_RAW = Path("dataset-sai/ENB2012_data.xlsx")
_DEFAULT_OUT = Path("data/processed/energy_efficiency_clean.csv")
_DEFAULT_REPORT = Path("data/processed/cleaning_report.txt")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_cleaning_pipeline(
    raw_path: str | os.PathLike = _DEFAULT_RAW,
    output_path: str | os.PathLike = _DEFAULT_OUT,
    report_path: str | os.PathLike = _DEFAULT_REPORT,
) -> Dict[str, Any]:
    """Execute the full cleaning pipeline for the ENB2012 dataset.

    Steps performed (each is logged and documented)
    ------------------------------------------------
    1. Load raw data with :func:`~src.data.load_data.load_energy_efficiency`.
    2. Snapshot before-state metrics.
    3. Drop fully-empty rows — rows where **every** cell is NaN (none expected
       in ENB2012, but the step is retained for reproducibility on re-runs
       with modified source files).
    4. Remove exact structural duplicates — rows where all 10 column values are
       identical.  Physical building simulations can produce legitimately
       identical parameter combinations, so each removed row is counted and
       reported.  Non-identical near-duplicates and outliers are deliberately
       NOT removed at this stage; that decision belongs to the EDA/feature
       phase.
    5. Validate the cleaned frame and assert no remaining issues.
    6. Write ``data/processed/energy_efficiency_clean.csv``.
    7. Write a plain-text cleaning report.

    Parameters
    ----------
    raw_path:
        Path to the raw ENB2012 file (.xlsx or .csv).
    output_path:
        Destination for the cleaned CSV.
    report_path:
        Destination for the plain-text cleaning report.

    Returns
    -------
    dict with keys:
        ``"df_clean"``    — the cleaned DataFrame
        ``"report_text"`` — multi-line report string
        ``"decisions"``   — list of (step, description, rows_affected) tuples
    """
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    # ── 1. Load ─────────────────────────────────────────────────────────────
    logger.info("Loading raw data from '%s' …", raw_path)
    df = load_energy_efficiency(raw_path, rename_columns=True)

    before = _snapshot(df)
    decisions: List[Tuple[str, str, int]] = []

    # ── 3. Drop fully-empty rows ────────────────────────────────────────────
    fully_empty_mask = df.isnull().all(axis=1)
    n_fully_empty = int(fully_empty_mask.sum())
    if n_fully_empty > 0:
        df = df[~fully_empty_mask].reset_index(drop=True)
        logger.info("Dropped %d fully-empty rows.", n_fully_empty)
    decisions.append((
        "Drop fully-empty rows",
        "Rows where every column is NaN carry no information.  "
        f"{n_fully_empty} row(s) removed.",
        n_fully_empty,
    ))

    # ── 4. Remove exact structural duplicates ───────────────────────────────
    dup_mask = df.duplicated()
    n_dups = int(dup_mask.sum())
    if n_dups > 0:
        df = df[~dup_mask].reset_index(drop=True)
        logger.info("Dropped %d exact duplicate rows.", n_dups)
    decisions.append((
        "Remove exact structural duplicates",
        "Rows identical across all 10 columns are redundant.  "
        f"{n_dups} duplicate row(s) removed.  "
        "Near-duplicates and statistical outliers are intentionally "
        "kept — that analysis is deferred to the EDA phase.",
        n_dups,
    ))

    # ── 5. Validate cleaned frame ────────────────────────────────────────────
    after = _snapshot(df)
    vr = validate_dataframe(df)

    if not vr["is_valid"]:
        issues = []
        if vr["total_missing"] > 0:
            issues.append(f"missing values: {vr['total_missing']}")
        if vr["dtype_errors"]:
            issues.append(f"dtype errors: {vr['dtype_errors']}")
        if vr["range_errors"]:
            issues.append(f"range errors: {vr['range_errors']}")
        raise RuntimeError(
            "Cleaned DataFrame still has issues after the pipeline: "
            + "; ".join(issues)
        )

    # ── 6. Write cleaned CSV ─────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Cleaned dataset saved → %s", output_path)

    # ── 7. Build and write report ────────────────────────────────────────────
    report_text = _build_report(raw_path, output_path, before, after, decisions)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Cleaning report saved → %s", report_path)

    print(report_text)
    return {"df_clean": df, "report_text": report_text, "decisions": decisions}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _snapshot(df: pd.DataFrame) -> Dict[str, Any]:
    """Capture quality metrics for before/after comparison."""
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "total_missing": int(df.isnull().sum().sum()),
        "missing_per_col": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def _build_report(
    raw_path: Path,
    output_path: Path,
    before: Dict[str, Any],
    after: Dict[str, Any],
    decisions: List[Tuple[str, str, int]],
) -> str:
    """Compose the human-readable cleaning report."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sep = "=" * 64

    lines = [
        sep,
        "  ENB2012 Energy Efficiency — Cleaning Report",
        f"  Generated : {ts}",
        sep,
        "",
        f"  Raw source  : {raw_path}",
        f"  Output file : {output_path}",
        "",
        "  ┌─────────────────────────┬──────────┬──────────┐",
        "  │ Metric                  │  Before  │  After   │",
        "  ├─────────────────────────┼──────────┼──────────┤",
        f"  │ Row count               │ {before['n_rows']:>8d} │ {after['n_rows']:>8d} │",
        f"  │ Column count            │ {before['n_cols']:>8d} │ {after['n_cols']:>8d} │",
        f"  │ Total missing values    │ {before['total_missing']:>8d} │ {after['total_missing']:>8d} │",
        f"  │ Duplicate rows          │ {before['duplicate_rows']:>8d} │ {after['duplicate_rows']:>8d} │",
        "  └─────────────────────────┴──────────┴──────────┘",
        "",
        "  Cleaning decisions (documented):",
        "  " + "-" * 60,
    ]

    for i, (step, rationale, affected) in enumerate(decisions, 1):
        wrapped = textwrap.fill(rationale, width=56,
                                initial_indent="      ",
                                subsequent_indent="      ")
        lines += [
            f"  {i}. {step}",
            f"     Rows affected: {affected}",
            f"     Rationale:",
            wrapped,
            "",
        ]

    rows_removed = before["n_rows"] - after["n_rows"]
    lines += [
        "  " + "-" * 60,
        f"  Total rows removed : {rows_removed}",
        f"  Total rows retained: {after['n_rows']}",
        "",
        "  Columns kept (all 10 original columns retained):",
        "    Features : " + ", ".join(FEATURE_COLUMNS),
        "    Targets  : " + ", ".join(TARGET_COLUMNS),
        "",
        "  ✓ Cleaned dataset written to processed/",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    raw = sys.argv[1] if len(sys.argv) > 1 else str(_DEFAULT_RAW)
    out = sys.argv[2] if len(sys.argv) > 2 else str(_DEFAULT_OUT)
    rep = sys.argv[3] if len(sys.argv) > 3 else str(_DEFAULT_REPORT)

    run_cleaning_pipeline(raw_path=raw, output_path=out, report_path=rep)
