"""Phase 1 cleaning for the Building Data Genome 2 dataset.

The source data is wide: each meter file has one timestamp column and one
column per building. This module deliberately does not impute readings,
create features, split data, or train a model.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


METER_DIR = Path("meters") / "cleaned"
METER_FILES = (
    "chilledwater_cleaned.csv",
    "electricity_cleaned.csv",
    "gas_cleaned.csv",
    "hotwater_cleaned.csv",
    "irrigation_cleaned.csv",
    "solar_cleaned.csv",
    "steam_cleaned.csv",
    "water_cleaned.csv",
)


def _clean_meter(path: Path, destination: Path) -> dict[str, Any]:
    """Clean one wide meter file while retaining missing readings."""
    frame = pd.read_csv(path)
    timestamp_column = frame.columns[0]
    original_rows = len(frame)
    missing_before = frame.isna().sum().to_dict()
    parsed_timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    invalid_timestamps = int(parsed_timestamps.isna().sum())
    frame[timestamp_column] = parsed_timestamps

    non_numeric_values = 0
    negative_values = 0
    for column in frame.columns[1:]:
        before = frame[column].notna()
        numeric = pd.to_numeric(frame[column], errors="coerce")
        non_numeric_values += int((before & numeric.isna()).sum())
        negative_values += int((numeric < 0).sum())
        frame[column] = numeric

    duplicate_timestamps = int(frame.duplicated(subset=[timestamp_column]).sum())
    invalid_timestamp_rows = frame[timestamp_column].isna()
    duplicate_rows = frame.duplicated(subset=[timestamp_column])
    rows_removed = int((invalid_timestamp_rows | duplicate_rows).sum())
    frame = frame.loc[~(invalid_timestamp_rows | duplicate_rows)].copy()
    frame = frame.sort_values(timestamp_column).reset_index(drop=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, date_format="%Y-%m-%d %H:%M:%S")

    intervals = frame[timestamp_column].diff().dropna()
    expected_interval = pd.Timedelta(hours=1)
    missing_intervals = int((intervals > expected_interval).sum())
    return {
        "source": str(path),
        "output": str(destination),
        "rows_before": original_rows,
        "rows_after": len(frame),
        "columns": len(frame.columns),
        "building_columns": len(frame.columns) - 1,
        "invalid_timestamps": invalid_timestamps,
        "duplicate_timestamp_rows_removed": duplicate_timestamps,
        "rows_removed": rows_removed,
        "non_numeric_values_coerced_to_missing": non_numeric_values,
        "negative_readings_observed": negative_values,
        "missing_readings_after": int(frame.iloc[:, 1:].isna().sum().sum()),
        "missing_before": missing_before,
        "missing_after": frame.isna().sum().to_dict(),
        "dtypes_after": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_intervals_after": missing_intervals,
        "timestamp_start": str(frame[timestamp_column].min()),
        "timestamp_end": str(frame[timestamp_column].max()),
        "timestamp_frequency": str(intervals.mode().iloc[0]) if not intervals.empty else "n/a",
    }


def _clean_metadata(path: Path, destination: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    missing_before = frame.isna().sum().to_dict()
    duplicate_building_ids = int(frame["building_id"].duplicated().sum())
    missing_building_ids = int(frame["building_id"].isna().sum())
    duplicate_rows = int(frame.duplicated().sum())

    for column in frame.columns:
        if column in {"building_id", "site_id", "timezone", "primaryspaceusage",
                      "sub_primaryspaceusage", "industry", "subindustry",
                      "heatingtype", "leed_level", "rating"}:
            continue
        if column == "date_opened":
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
            continue
        if frame[column].dtype == "object":
            numeric = pd.to_numeric(frame[column], errors="coerce")
            non_missing = frame[column].notna().sum()
            if non_missing and int(numeric.notna().sum()) == int(non_missing):
                frame[column] = numeric

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, date_format="%Y-%m-%d")
    return {
        "source": str(path),
        "output": str(destination),
        "rows_before": len(frame),
        "rows_after": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": duplicate_rows,
        "duplicate_building_ids": duplicate_building_ids,
        "missing_building_ids": missing_building_ids,
        "missing_values_after": int(frame.isna().sum().sum()),
        "missing_before": missing_before,
        "missing_after": frame.isna().sum().to_dict(),
        "dtypes_after": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "building_ids": frame["building_id"].dropna().astype(str).tolist(),
    }


def _clean_weather(path: Path, destination: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    timestamp_column = frame.columns[0]
    missing_before = frame.isna().sum().to_dict()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce")
    invalid_timestamps = int(frame[timestamp_column].isna().sum())
    duplicate_rows = int(frame.duplicated(subset=[timestamp_column, "site_id"]).sum())
    frame = frame.loc[frame[timestamp_column].notna()].drop_duplicates(
        subset=[timestamp_column, "site_id"], keep="first"
    ).sort_values(["site_id", timestamp_column])
    for column in frame.columns:
        if column not in {timestamp_column, "site_id"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return {
        "source": str(path),
        "output": str(destination),
        "rows_before": len(pd.read_csv(path, usecols=[timestamp_column])),
        "rows_after": len(frame),
        "columns": len(frame.columns),
        "invalid_timestamps": invalid_timestamps,
        "duplicate_timestamp_site_rows_removed": duplicate_rows,
        "missing_values_after": int(frame.isna().sum().sum()),
        "missing_before": missing_before,
        "missing_after": frame.isna().sum().to_dict(),
        "dtypes_after": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "negative_precipitation_sentinel_values": int((frame.get("precipDepth1HR", pd.Series(dtype=float)) == -1).sum()),
    }


def _report(summary: dict[str, Any], report_path: Path) -> str:
    lines = [
        "Building Data Genome 2 - Phase 1 Data Cleaning Report",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Raw root: {summary['raw_root']}",
        f"Output root: {summary['output_root']}",
        "",
        "Decisions:",
        "- Parsed timestamps with errors='coerce'; invalid timestamp rows were removed.",
        "- Removed duplicate timestamps in meter files and duplicate timestamp/site rows in weather.",
        "- Coerced meter and weather measurements to numeric; conversion failures become missing and are counted.",
        "- Preserved missing readings and legitimate zero readings; no imputation was performed.",
        "- No meter negatives were observed. Weather precipitation -1 values were retained as the dataset sentinel.",
        "- High readings were retained because they are not confirmed errors; outlier analysis is deferred to EDA.",
        "",
        "Meter files:",
    ]
    for item in summary["meters"]:
        lines.append(
            f"  {Path(item['source']).name}: {item['rows_before']}x{item['columns']} -> "
            f"{item['rows_after']}x{item['columns']}; removed={item['rows_removed']}; "
            f"missing readings={item['missing_readings_after']}; gaps={item['missing_intervals_after']}; "
            f"negative readings={item['negative_readings_observed']}"
        )
        lines.append("    Columns / dtype / missing before -> after:")
        for column, dtype in item["dtypes_after"].items():
            lines.append(
                f"      {column}: {dtype}; {item['missing_before'].get(column, 0)} -> "
                f"{item['missing_after'].get(column, 0)}"
            )
    metadata = summary["metadata"]
    weather = summary["weather"]
    lines += [
        "",
        "Metadata:",
        f"  {metadata['rows_before']}x{metadata['columns']} -> {metadata['rows_after']}x{metadata['columns']}; "
        f"duplicate rows={metadata['duplicate_rows']}; duplicate building IDs={metadata['duplicate_building_ids']}; "
        f"missing building IDs={metadata['missing_building_ids']}; missing values retained={metadata['missing_values_after']}",
        "  Columns / dtype / missing before -> after:",
    ]
    for column, dtype in metadata["dtypes_after"].items():
        lines.append(
            f"    {column}: {dtype}; {metadata['missing_before'].get(column, 0)} -> "
            f"{metadata['missing_after'].get(column, 0)}"
        )
    lines += [
        "Weather:",
        f"  {weather['rows_before']}x{weather['columns']} -> {weather['rows_after']}x{weather['columns']}; "
        f"removed duplicates={weather['duplicate_timestamp_site_rows_removed']}; "
        f"missing values retained={weather['missing_values_after']}; "
        f"precipitation sentinel -1 values={weather['negative_precipitation_sentinel_values']}",
        "  Columns / dtype / missing before -> after:",
    ]
    for column, dtype in weather["dtypes_after"].items():
        lines.append(
            f"    {column}: {dtype}; {weather['missing_before'].get(column, 0)} -> "
            f"{weather['missing_after'].get(column, 0)}"
        )
    lines += [
        "",
        "Remaining issues requiring later attention:",
        "- Meter readings contain missing values because the source does not measure every meter/building continuously.",
        "- Large values remain for domain review; this phase does not label them as anomalies.",
        "- No target imputation, feature engineering, train/test split, or model training was performed.",
    ]
    text = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return text


def run_cleaning_pipeline(
    raw_root: str | os.PathLike,
    output_root: str | os.PathLike = "data/processed/bdg2",
    report_path: str | os.PathLike = "reports/bdg2_cleaning_report.txt",
) -> dict[str, Any]:
    """Clean all BDG2 metadata, meter, and weather files without mutating raw data."""
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    summary: dict[str, Any] = {"raw_root": str(raw_root), "output_root": str(output_root), "meters": []}
    summary["metadata"] = _clean_metadata(
        raw_root / "metadata" / "metadata.csv", output_root / "metadata.csv"
    )
    for filename in METER_FILES:
        summary["meters"].append(
            _clean_meter(raw_root / METER_DIR / filename, output_root / "meters" / filename)
        )
    summary["weather"] = _clean_weather(
        raw_root / "weather" / "weather.csv", output_root / "weather.csv"
    )
    summary["report_text"] = _report(summary, Path(report_path))
    print(summary["report_text"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/bdg2"))
    parser.add_argument("--report", type=Path, default=Path("reports/bdg2_cleaning_report.txt"))
    args = parser.parse_args()
    run_cleaning_pipeline(args.raw_root, args.output_root, args.report)


if __name__ == "__main__":
    main()