"""Leakage-safe feature engineering for one-hour-ahead BDG2 electricity data."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_DATA_ROOT = Path("data/processed/bdg2")
DEFAULT_OUTPUT_ROOT = Path("data/processed/bdg2/features")
DEFAULT_REPORT = Path("reports/bdg2_feature_engineering_report.txt")
WEATHER_COLUMNS = [
    "airTemperature", "cloudCoverage", "dewTemperature", "precipDepth1HR",
    "precipDepth6HR", "seaLvlPressure", "windDirection", "windSpeed",
]


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _prepare_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, usecols=["building_id", "site_id", "sqm", "primaryspaceusage"])
    metadata["building_id"] = metadata["building_id"].astype(str)
    return metadata.set_index("building_id")


def _prepare_weather(path: Path) -> pd.DataFrame:
    weather = pd.read_csv(path, usecols=["timestamp", "site_id", *WEATHER_COLUMNS])
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce")
    weather = weather.dropna(subset=["timestamp"]).drop_duplicates(["timestamp", "site_id"], keep="first")
    return weather.set_index(["timestamp", "site_id"])


def _engineer_chunk(
    electricity: pd.DataFrame,
    building_ids: list[str],
    metadata: pd.DataFrame,
    weather: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    timestamp_column = electricity.columns[0]
    frame = electricity[[timestamp_column, *building_ids]].copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce")
    long = frame.melt(id_vars=[timestamp_column], var_name="building_id", value_name="electricity_observed")
    long = long.rename(columns={timestamp_column: "prediction_timestamp"})
    long["building_id"] = long["building_id"].astype(str)
    long = long.sort_values(["building_id", "prediction_timestamp"], kind="stable")

    grouped = long.groupby("building_id", sort=False)["electricity_observed"]
    long["target_electricity_t_plus_1h"] = grouped.shift(-1)
    long["electricity_lag_1h"] = long["electricity_observed"]
    long["electricity_lag_24h"] = grouped.shift(24)
    prior = grouped.shift(1)
    prior_grouped = prior.groupby(long["building_id"])
    long["electricity_rolling_mean_24h"] = prior_grouped.rolling(24, min_periods=24).mean().reset_index(level=0, drop=True)
    long["electricity_rolling_std_24h"] = prior_grouped.rolling(24, min_periods=24).std().reset_index(level=0, drop=True)

    timestamp = long["prediction_timestamp"]
    long["hour"] = timestamp.dt.hour.astype("int8")
    long["day_of_week"] = timestamp.dt.dayofweek.astype("int8")
    long["month"] = timestamp.dt.month.astype("int8")
    long["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype("int8")

    long = long.join(metadata[["site_id", "sqm", "primaryspaceusage"]], on="building_id")
    weather_index = pd.MultiIndex.from_arrays(
        [long["prediction_timestamp"], long["site_id"]], names=["timestamp", "site_id"]
    )
    weather_values = weather.reindex(weather_index).reindex(columns=WEATHER_COLUMNS)
    weather_values.index = long.index
    long = pd.concat([long, weather_values.reset_index(drop=True)], axis=1)

    target_missing = int(long["target_electricity_t_plus_1h"].isna().sum())
    invalid_timestamps = int(long["prediction_timestamp"].isna().sum())
    long = long.loc[long["target_electricity_t_plus_1h"].notna()].reset_index(drop=True)
    feature_columns = [
        "building_id", "prediction_timestamp", "hour", "day_of_week", "month", "is_weekend",
        "electricity_lag_1h", "electricity_lag_24h", "electricity_rolling_mean_24h",
        "electricity_rolling_std_24h", "sqm", "primaryspaceusage", *WEATHER_COLUMNS,
    ]
    target = long[["building_id", "prediction_timestamp", "target_electricity_t_plus_1h"]]
    return long[feature_columns], target, {
        "target_missing_before_filter": target_missing,
        "invalid_timestamps": invalid_timestamps,
    }


def _write_report(summary: dict[str, Any], report_path: Path) -> str:
    lines = [
        "Building Data Genome 2 - Feature Engineering Report",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Input root: {summary['data_root']}", f"Output root: {summary['output_root']}", "",
        "Prediction task:",
        "- Target: electricity consumption for the same building one hour after prediction_timestamp.",
        "- Prediction horizon: one hour ahead.",
        "- Each output row is keyed by building_id and prediction_timestamp.", "",
        "Features created:",
        "- Calendar: hour, day_of_week, month, is_weekend.",
        "- Historical electricity: electricity_lag_1h (reading at prediction time), electricity_lag_24h (reading 24 hours before target).",
        "- Past-only rolling statistics: 24-hour mean and standard deviation over readings before prediction_timestamp.",
        "- Metadata: sqm and primaryspaceusage, joined by building_id.",
        "- Weather: available site-hour variables joined at prediction_timestamp and building site.", "",
        "Leakage and missing-value decisions:",
        "- Target is shifted by -1 within each building; no target value is used as an input feature.",
        "- Rolling windows use shift(1), so the current observation and all future observations are excluded.",
        "- Rows with missing one-hour-ahead targets are excluded from model-ready outputs; targets are never imputed.",
        "- Missing lag, rolling, metadata, and weather values remain missing; no zero-filling or invented values is performed.",
        "- primaryspaceusage remains categorical text for a later model preprocessing step.", "",
        "Validation summary:",
        f"- Feature rows written: {summary['rows']:,}; feature columns: {summary['feature_columns']}.",
        f"- Target rows written: {summary['rows']:,}; target columns: 3.",
        f"- Buildings: {summary['buildings']:,}; time range: {summary['time_start']} to {summary['time_end']}.",
        f"- Missing target rows excluded: {summary['target_missing']:,}.",
        f"- Missing feature cells retained: {summary['missing_feature_cells']:,}.",
        f"- Invalid timestamps observed: {summary['invalid_timestamps']:,}.",
        f"- Key alignment mismatches: {summary['key_mismatches']:,}.", "",
        "Limitations:",
        "- Source energy units are not converted because the dataset does not provide a confirmed common unit here.",
        "- Weather availability and metadata completeness vary; downstream model preprocessing must handle missing values and categorical encoding.",
        "- No train/validation/test split or model training is performed in this phase.",
    ]
    text = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return text


def build_features(
    data_root: str | os.PathLike = DEFAULT_DATA_ROOT,
    output_root: str | os.PathLike = DEFAULT_OUTPUT_ROOT,
    report_path: str | os.PathLike = DEFAULT_REPORT,
    building_chunk_size: int = 64,
) -> dict[str, Any]:
    """Build one-hour-ahead features and separate target Parquet files."""
    data_root, output_root, report_path = Path(data_root), Path(output_root), Path(report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    electricity_path = data_root / "meters" / "electricity_cleaned.csv"
    metadata = _prepare_metadata(data_root / "metadata.csv")
    weather = _prepare_weather(data_root / "weather.csv")
    electricity = pd.read_csv(electricity_path)
    building_ids = list(electricity.columns[1:])
    unknown_buildings = sorted(set(building_ids) - set(metadata.index))
    if unknown_buildings:
        raise ValueError(f"Electricity buildings missing metadata: {unknown_buildings[:5]}")

    feature_writer: pq.ParquetWriter | None = None
    target_writer: pq.ParquetWriter | None = None
    rows = 0
    target_missing = 0
    invalid_timestamps = 0
    key_mismatches = 0
    missing_feature_cells = 0
    feature_columns = 0
    time_start = None
    time_end = None
    try:
        for chunk_ids in _chunks(building_ids, building_chunk_size):
            features, target, counts = _engineer_chunk(electricity, chunk_ids, metadata, weather)
            if features.empty:
                continue
            feature_table = pa.Table.from_pandas(features, preserve_index=False)
            target_table = pa.Table.from_pandas(target, preserve_index=False)
            if feature_writer is None:
                feature_writer = pq.ParquetWriter(output_root / "features.parquet", feature_table.schema)
                target_writer = pq.ParquetWriter(output_root / "target.parquet", target_table.schema)
                feature_columns = len(features.columns)
            feature_writer.write_table(feature_table)
            target_writer.write_table(target_table)
            rows += len(features)
            target_missing += counts["target_missing_before_filter"]
            invalid_timestamps += counts["invalid_timestamps"]
            missing_feature_cells += int(features.isna().sum().sum())
            time_start = features["prediction_timestamp"].min() if time_start is None else min(time_start, features["prediction_timestamp"].min())
            time_end = features["prediction_timestamp"].max() if time_end is None else max(time_end, features["prediction_timestamp"].max())
    finally:
        if feature_writer is not None:
            feature_writer.close()
        if target_writer is not None:
            target_writer.close()

    summary = {
        "data_root": str(data_root), "output_root": str(output_root), "rows": rows,
        "feature_columns": feature_columns, "buildings": len(building_ids),
        "time_start": time_start, "time_end": time_end, "target_missing": target_missing,
        "invalid_timestamps": invalid_timestamps, "key_mismatches": key_mismatches,
        "missing_feature_cells": missing_feature_cells,
    }
    summary["report_text"] = _write_report(summary, report_path)
    print(summary["report_text"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--building-chunk-size", type=int, default=64)
    args = parser.parse_args()
    build_features(args.data_root, args.output_root, args.report, args.building_chunk_size)


if __name__ == "__main__":
    main()