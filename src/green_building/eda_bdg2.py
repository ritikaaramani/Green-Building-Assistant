"""Exploratory analysis for the cleaned Building Data Genome 2 data.

This module creates descriptive summaries and plots only. Its aggregations are
for analysis and are not persisted as model features.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_DATA_ROOT = Path("data/processed/bdg2")
DEFAULT_REPORT = Path("reports/bdg2_eda_report.txt")
DEFAULT_FIGURES = Path("reports/figures")


def _meter_files(data_root: Path) -> list[Path]:
    return sorted((data_root / "meters").glob("*.csv"))


def _read_meter(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    frame = pd.read_csv(path)
    timestamp = frame.columns[0]
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce")
    return frame, timestamp, list(frame.columns[1:])


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _format_number(value: float) -> str:
    return f"{value:,.3f}" if abs(value) < 1000 else f"{value:,.0f}"


def _metadata_analysis(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include="number")
    stats = numeric.describe().T
    return {
        "frame": frame,
        "shape": frame.shape,
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing": frame.isna().sum().to_dict(),
        "missing_pct": (frame.isna().mean() * 100).round(2).to_dict(),
        "numeric_stats": stats,
        "unique_buildings": int(frame["building_id"].nunique()),
        "unique_sites": int(frame["site_id"].nunique()),
    }


def _meter_analysis(path: Path, figures: Path) -> dict[str, Any]:
    frame, timestamp, buildings = _read_meter(path)
    values = frame[buildings]
    total = values.sum(axis=1, skipna=True)
    long_stats = values.describe().T
    missing = values.isna().sum().sum()
    intervals = frame[timestamp].sort_values().diff().dropna()

    return {
        "name": path.stem,
        "frame": frame,
        "timestamp": timestamp,
        "buildings": buildings,
        "values": values,
        "total": pd.Series(total.to_numpy(), index=frame[timestamp]),
        "stats": long_stats,
        "missing": int(missing),
        "missing_pct": float(missing / values.size * 100),
        "time_start": frame[timestamp].min(),
        "time_end": frame[timestamp].max(),
        "interval_counts": intervals.value_counts().head(5),
    }


def _create_plots(
    electricity: dict[str, Any],
    meter_totals: pd.DataFrame,
    metadata: pd.DataFrame,
    weather: pd.DataFrame,
    figures: Path,
) -> list[str]:
    created: list[str] = []
    values = electricity["values"].stack().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(values, bins=60, color="#1f77b4", edgecolor="white")
    axes[0].set_title("Electricity reading distribution")
    axes[0].set_xlabel("Reading (source units)")
    axes[0].set_ylabel("Reading count")
    axes[1].boxplot(values, vert=False, showfliers=False)
    axes[1].set_title("Electricity readings (central range)")
    axes[1].set_xlabel("Reading (source units)")
    path = figures / "electricity_distribution.png"
    _savefig(fig, path)
    created.append(str(path))

    hourly = electricity["total"].groupby(electricity["total"].index.hour).mean()
    weekly = electricity["total"].groupby(electricity["total"].index.dayofweek).mean()
    monthly = electricity["total"].groupby(electricity["total"].index.month).mean()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    axes[0].plot(hourly.index, hourly.values, marker="o")
    axes[0].set(title="Mean electricity by hour", xlabel="Hour of day", ylabel="Mean reading")
    axes[1].plot(weekly.index, weekly.values, marker="o", color="#ff7f0e")
    axes[1].set(title="Mean electricity by weekday", xlabel="Day (0=Monday)", ylabel="Mean reading")
    axes[2].plot(monthly.index, monthly.values, marker="o", color="#2ca02c")
    axes[2].set(title="Mean electricity by month", xlabel="Month", ylabel="Mean reading")
    path = figures / "electricity_temporal_patterns.png"
    _savefig(fig, path)
    created.append(str(path))

    top = electricity["stats"].sort_values("mean", ascending=False).head(10)
    low = electricity["stats"].loc[electricity["stats"]["count"] > 0].sort_values("mean").head(10)
    comparison = pd.concat([top, low]).drop_duplicates()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(comparison.index, comparison["mean"], color="#9467bd")
    ax.set(title="Buildings with highest and lowest mean electricity", xlabel="Mean reading", ylabel="Building")
    path = figures / "building_electricity_comparison.png"
    _savefig(fig, path)
    created.append(str(path))

    building = electricity["stats"][["mean", "50%", "min", "max", "std", "count"]].copy()
    building = building.rename(columns={"50%": "median"})
    building.index.name = "building_id"
    joined = building.join(metadata.set_index("building_id")[["primaryspaceusage", "sqm"]], how="left")
    grouped = joined.dropna(subset=["primaryspaceusage"]).groupby("primaryspaceusage")["mean"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    grouped.plot.barh(ax=ax, color="#17becf")
    ax.set(title="Mean building electricity by primary space use", xlabel="Mean building reading", ylabel="Primary space use")
    path = figures / "electricity_by_building_type.png"
    _savefig(fig, path)
    created.append(str(path))

    corr_frame = meter_totals.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr_frame, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr_frame.columns)), corr_frame.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_frame.index)), corr_frame.index)
    ax.set_title("Correlation of hourly electricity total and weather variables")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    path = figures / "electricity_weather_correlation.png"
    _savefig(fig, path)
    created.append(str(path))
    return created


def _build_report(
    data_root: Path,
    metadata_info: dict[str, Any],
    analyses: list[dict[str, Any]],
    figures: list[str],
    weather: pd.DataFrame,
) -> str:
    electricity = next(item for item in analyses if item["name"].startswith("electricity"))
    total = electricity["total"]
    hourly = total.groupby(total.index.hour).mean()
    weekly = total.groupby(total.index.dayofweek).mean()
    monthly = total.groupby(total.index.month).mean()
    building_stats = electricity["stats"]
    metadata = metadata_info["frame"].set_index("building_id")
    joined = building_stats.join(metadata[["primaryspaceusage", "sqm"]], how="left")
    by_type = joined.groupby("primaryspaceusage")["mean"].mean().sort_values(ascending=False)
    area_corr = joined[["mean", "sqm"]].corr().iloc[0, 1]
    all_missing = sum(item["missing"] for item in analyses)
    meter_lines = []
    for item in analyses:
        meter_lines.append(
            f"  {item['name']}: {len(item['frame']):,} timestamps, {len(item['buildings']):,} buildings, "
            f"missing readings={item['missing']:,} ({item['missing_pct']:.2f}%), "
            f"range={item['time_start']} to {item['time_end']}"
        )
    lines = [
        "Building Data Genome 2 - Exploratory Data Analysis Report",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Cleaned data root: {data_root}",
        "Scope: descriptive EDA only; no feature engineering, imputation, splitting, or model training.",
        "",
        "1. Dataset overview",
        f"Metadata dimensions: {metadata_info['shape'][0]:,} rows x {metadata_info['shape'][1]} columns.",
        f"Unique buildings: {metadata_info['unique_buildings']:,}; unique sites: {metadata_info['unique_sites']:,}.",
        f"Metadata missing cells: {sum(metadata_info['missing'].values()):,}.",
        "Metadata dtypes: " + ", ".join(f"{k}={v}" for k, v in metadata_info["dtypes"].items()),
        "Numerical metadata summary:",
    ]
    for column, row in metadata_info["numeric_stats"].iterrows():
        lines.append(
            f"  {column}: count={row['count']:.0f}, mean={_format_number(row['mean'])}, "
            f"std={_format_number(row['std'])}, min={_format_number(row['min'])}, max={_format_number(row['max'])}"
        )
    lines += ["", "Meter overview:"] + meter_lines
    lines += [
        "",
        "2. Electricity consumption distribution",
        f"Readings: {int(building_stats['count'].sum()):,} non-missing values across {len(building_stats):,} buildings.",
        f"Reading mean={_format_number(building_stats['mean'].mean())}, median of building medians={_format_number(building_stats['50%'].median())}, "
        f"minimum={_format_number(building_stats['min'].min())}, maximum={_format_number(building_stats['max'].max())}.",
        f"Largest building maximum: {_format_number(building_stats['max'].max())}; these high values were retained for domain review, not removed as outliers.",
        f"Total missing values across all meter tables: {all_missing:,}; missingness is concentrated in meters not installed for every building.",
        "",
        "3. Temporal analysis",
        f"Electricity covers {electricity['time_start']} through {electricity['time_end']} at the modal interval {electricity['interval_counts'].index[0] if len(electricity['interval_counts']) else 'unavailable'}.",
        f"Mean total electricity is lowest at hour {int(hourly.idxmin())} ({_format_number(hourly.min())}) and highest at hour {int(hourly.idxmax())} ({_format_number(hourly.max())}).",
        f"Mean total electricity is lowest on weekday index {int(weekly.idxmin())} and highest on weekday index {int(weekly.idxmax())}.",
        f"Mean total electricity is lowest in month {int(monthly.idxmin())} and highest in month {int(monthly.idxmax())}; these recurring differences support investigating calendar and seasonal variables later.",
        "",
        "4. Building-level analysis",
        f"Highest mean electricity building: {building_stats['mean'].idxmax()} ({_format_number(building_stats['mean'].max())}); lowest non-empty mean: {building_stats['mean'].idxmin()} ({_format_number(building_stats['mean'].min())}).",
        "Mean electricity by primary space use, ordered high to low:",
    ]
    lines.extend(f"  {category}: {_format_number(value)}" for category, value in by_type.items())
    lines += [
        f"Correlation between building mean electricity and available sqm: {area_corr:.3f}; this is an association, not evidence of causation.",
        "",
        "5. Correlations and relationships",
        f"Weather rows available: {len(weather):,}; numeric weather columns: {', '.join(weather.select_dtypes(include='number').columns)}.",
        "The weather correlation figure compares hourly electricity totals with available weather variables and excludes identifiers.",
        "Correlation should not be interpreted as causation; site, timezone, occupancy, and missing meter coverage can confound these relationships.",
        "",
        "6. Data quality findings and limitations",
        "- Meter tables are wide and meter coverage differs by energy type; absence of a building column means that meter is unavailable, not necessarily zero.",
        "- Missing readings remain in the cleaned files and were not imputed in EDA.",
        "- Energy units are inherited from the source files and are reported as source units because no universal conversion was confirmed.",
        "- Metadata contains missing values in several fields, limiting type and area comparisons.",
        "- This analysis uses electricity as the primary consumption series; other meters are compared only through descriptive totals.",
        "",
        "7. Recommended next-phase investigations",
        "- Add calendar variables such as hour, weekday, month, and holiday indicators during feature engineering.",
        "- Investigate weather variables, building area, primary space use, and site/timezone context as candidate predictors.",
        "- Decide a missing-value strategy by meter coverage and prediction target; do not treat unavailable meter columns as zero without justification.",
        "- Review extreme readings with domain context before applying any outlier policy.",
        "",
        "Generated figures:",
    ]
    lines.extend(f"- {figure}" for figure in figures)
    return "\n".join(lines) + "\n"


def run_eda(
    data_root: str | os.PathLike = DEFAULT_DATA_ROOT,
    report_path: str | os.PathLike = DEFAULT_REPORT,
    figures_dir: str | os.PathLike = DEFAULT_FIGURES,
) -> dict[str, Any]:
    """Run reproducible descriptive EDA over the cleaned BDG2 dataset."""
    data_root, report_path, figures_dir = Path(data_root), Path(report_path), Path(figures_dir)
    metadata_info = _metadata_analysis(data_root / "metadata.csv")
    analyses = [_meter_analysis(path, Path(figures_dir)) for path in _meter_files(data_root)]
    if not analyses:
        raise FileNotFoundError(f"No cleaned meter CSV files found under {data_root / 'meters'}")
    electricity = next((item for item in analyses if item["name"].startswith("electricity")), analyses[0])
    weather = pd.read_csv(data_root / "weather.csv")
    weather[weather.columns[0]] = pd.to_datetime(weather[weather.columns[0]], errors="coerce")
    weather_numeric = weather.select_dtypes(include="number")
    weather_hourly = weather.assign(timestamp=weather[weather.columns[0]]).groupby("timestamp")[weather_numeric.columns].mean()
    electricity_hourly = electricity["total"].groupby(level=0).sum().rename("electricity")
    joined_weather = electricity_hourly.to_frame().join(weather_hourly, how="inner")
    figures = _create_plots(electricity, joined_weather, metadata_info["frame"], weather, Path(figures_dir))
    report = _build_report(data_root, metadata_info, analyses, figures, weather)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    return {"report_text": report, "figures": figures, "analyses": analyses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    run_eda(args.data_root, args.report, args.figures_dir)


if __name__ == "__main__":
    main()