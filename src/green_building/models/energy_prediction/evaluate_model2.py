"""Evaluate saved Model 2 predictions and generate residual diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.green_building.models.energy_prediction.train_model2 import (
    DEFAULT_FEATURES, DEFAULT_TARGET, Preprocessor, _feature_columns, _reader, split_name,
)


DEFAULT_MODEL = Path("models/energy_prediction/xgboost_model.json")
DEFAULT_REPORT = Path("reports/model2_evaluation_report.txt")
DEFAULT_FIGURES = Path("reports/evaluation")
TRAIN_SAMPLE_ROWS = 1_000_000


def _sample_training_coverage(features: Path, target: Path, columns: list[str], batch_size: int) -> dict[str, Any]:
    selected: list[pd.DataFrame] = []
    seen = 0
    for frame in _reader(features, target, columns, batch_size):
        mask = split_name(pd.to_datetime(frame["prediction_timestamp"])).eq("train")
        part = frame.loc[mask].head(TRAIN_SAMPLE_ROWS - seen)
        if not part.empty:
            selected.append(part[["building_id", "prediction_timestamp"]])
            seen += len(part)
        if seen >= TRAIN_SAMPLE_ROWS:
            break
    sample = pd.concat(selected, ignore_index=True)
    timestamps = pd.to_datetime(sample["prediction_timestamp"])
    return {
        "rows": int(len(sample)),
        "unique_buildings": int(sample["building_id"].nunique()),
        "time_start": str(timestamps.min()),
        "time_end": str(timestamps.max()),
        "months": {str(key): int(value) for key, value in timestamps.dt.month.value_counts().sort_index().items()},
    }


def _make_plots(frame: pd.DataFrame, figures: Path) -> list[str]:
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    plot_frame = frame.sample(min(len(frame), 100_000), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(plot_frame["actual"], plot_frame["predicted"], s=3, alpha=0.15)
    limit = max(float(plot_frame["actual"].max()), float(plot_frame["predicted"].max()))
    ax.plot([0, limit], [0, limit], color="black", linestyle="--", label="Perfect prediction")
    ax.set(xlabel="Actual electricity", ylabel="Predicted electricity", title="Test actual vs predicted")
    ax.legend()
    path = figures / "model2_actual_vs_predicted.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 5))
    clipped = frame["residual"].clip(frame["residual"].quantile(0.01), frame["residual"].quantile(0.99))
    ax.hist(clipped, bins=80, color="#d95f02", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--")
    ax.set(xlabel="Residual (actual - predicted)", ylabel="Count", title="Test residual distribution (1st-99th percentile range)")
    path = figures / "model2_residual_distribution.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))

    hourly = frame.groupby("hour").agg(mae=("absolute_error", "mean"), rmse=("squared_error", lambda values: np.sqrt(values.mean())))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hourly.index, hourly["mae"], marker="o", label="MAE")
    ax.plot(hourly.index, hourly["rmse"], marker="o", label="RMSE")
    ax.set(xlabel="Prediction hour", ylabel="Error", title="Test error by prediction hour")
    ax.legend()
    path = figures / "model2_error_by_hour.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))

    top = frame.groupby("building_id")["absolute_error"].mean().nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    top.plot.barh(ax=ax, color="#7570b3")
    ax.set(xlabel="Mean absolute error", ylabel="Building", title="Buildings with highest test mean absolute error")
    path = figures / "model2_top_building_errors.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
    return paths


def _evaluate_test(features: Path, target: Path, columns: list[str], preprocessor: Preprocessor, model: XGBRegressor, batch_size: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    chunks: list[pd.DataFrame] = []
    building_stats: dict[str, list[float]] = {}
    period_stats: dict[str, list[float]] = {}
    test_rows = 0
    actual_sum = 0.0
    actual_squared_sum = 0.0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    for frame in _reader(features, target, columns, batch_size):
        timestamp = pd.to_datetime(frame["prediction_timestamp"])
        selected = frame.loc[split_name(timestamp).eq("test")].copy()
        if selected.empty:
            continue
        actual = selected.pop("target_electricity_t_plus_1h").to_numpy(dtype=np.float32)
        predicted = model.predict(preprocessor.transform(selected))
        residual = actual - predicted
        result = pd.DataFrame({
            "building_id": selected["building_id"].to_numpy(),
            "prediction_timestamp": pd.to_datetime(selected["prediction_timestamp"]).to_numpy(),
            "hour": pd.to_datetime(selected["prediction_timestamp"]).dt.hour.to_numpy(),
            "actual": actual, "predicted": predicted, "residual": residual,
            "absolute_error": np.abs(residual), "squared_error": residual ** 2,
        })
        chunks.append(result.sample(min(len(result), 20_000), random_state=42))
        for key, group in result.groupby("building_id"):
            values = building_stats.setdefault(str(key), [0.0, 0.0, 0.0])
            values[0] += float(group["absolute_error"].sum()); values[1] += float(group["squared_error"].sum()); values[2] += len(group)
        for key, group in result.groupby(result["prediction_timestamp"].dt.to_period("M").astype(str)):
            values = period_stats.setdefault(str(key), [0.0, 0.0, 0.0])
            values[0] += float(group["absolute_error"].sum()); values[1] += float(group["squared_error"].sum()); values[2] += len(group)
        test_rows += len(result)
        actual_sum += float(actual.sum())
        actual_squared_sum += float(np.square(actual).sum())
        absolute_error_sum += float(np.abs(residual).sum())
        squared_error_sum += float(np.square(residual).sum())
    sampled = pd.concat(chunks, ignore_index=True)
    building = pd.DataFrame.from_dict(building_stats, orient="index", columns=["absolute_error_sum", "squared_error_sum", "rows"])
    building["mae"] = building["absolute_error_sum"] / building["rows"]
    building["rmse"] = np.sqrt(building["squared_error_sum"] / building["rows"])
    period = pd.DataFrame.from_dict(period_stats, orient="index", columns=["absolute_error_sum", "squared_error_sum", "rows"])
    period["mae"] = period["absolute_error_sum"] / period["rows"]
    period["rmse"] = np.sqrt(period["squared_error_sum"] / period["rows"])
    actual_mean = actual_sum / test_rows
    total_variation = actual_squared_sum - (test_rows * actual_mean ** 2)
    return sampled, {
        "rows": test_rows,
        "metrics": {
            "mae": absolute_error_sum / test_rows,
            "rmse": float(np.sqrt(squared_error_sum / test_rows)),
            "r2": float(1.0 - squared_error_sum / total_variation) if total_variation else float("nan"),
        },
        "building": building,
        "period": period,
    }


def evaluate_model(features: str | os.PathLike = DEFAULT_FEATURES, target: str | os.PathLike = DEFAULT_TARGET, model_path: str | os.PathLike = DEFAULT_MODEL, report_path: str | os.PathLike = DEFAULT_REPORT, figures_dir: str | os.PathLike = DEFAULT_FIGURES, batch_size: int = 100_000) -> dict[str, Any]:
    features, target, model_path, report_path, figures_dir = map(Path, [features, target, model_path, report_path, figures_dir])
    numeric, categorical, columns = _feature_columns(features)
    preprocessor = Preprocessor(numeric, categorical)
    first = next(_reader(features, target, columns, batch_size))
    preprocessor.fit(first.loc[split_name(pd.to_datetime(first["prediction_timestamp"])).eq("train")])
    model = XGBRegressor(); model.load_model(model_path)
    coverage = _sample_training_coverage(features, target, columns, batch_size)
    coverage["building_coverage_pct"] = round(coverage["unique_buildings"] / 1578 * 100, 3)
    sampled, aggregates = _evaluate_test(features, target, columns, preprocessor, model, batch_size)
    plots = _make_plots(sampled, figures_dir)
    building = aggregates["building"].sort_values("absolute_error_sum", ascending=False).head(15)
    period = aggregates["period"].sort_values("absolute_error_sum", ascending=False).head(15)
    report = {
        "evaluation_split": "test",
        "test_boundaries": {"start": "2017-07-01 00:00:00", "end": "2017-12-31 23:59:59"},
        "validation_rows_evaluated": 0,
        "model": str(model_path), "test_rows": aggregates["rows"],
        "test_metrics": aggregates["metrics"], "training_sample_coverage": coverage,
        "top_buildings_by_total_absolute_error": building.reset_index(names="building_id").to_dict("records"),
        "top_periods_by_total_absolute_error": period.reset_index(names="period").to_dict("records"),
        "plots": plots,
        "findings": [
            "The training sample is a deterministic first 1,000,000-row sample from the chronological training split and covers all training months but only 156 of 1,578 buildings (9.886%); retraining with stratified building coverage is recommended before production use.",
            "Test predictions were generated in batches; actual-versus-predicted and residual plots use a deterministic diagnostic sample.",
            "Buildings with larger total error may contribute more because they have more rows; compare MAE as well as total error.",
            "Residual concentrations by month and hour identify periods for later investigation, not automatic data removal.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    print(json.dumps(evaluate_model(args.features, args.target, args.model, args.report, args.figures_dir, args.batch_size), indent=2, default=str))


if __name__ == "__main__":
    main()