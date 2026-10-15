"""Train and evaluate Model 2 without modifying feature Parquet files."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DEFAULT_FEATURES = Path("data/processed/bdg2/features/features.parquet")
DEFAULT_TARGET = Path("data/processed/bdg2/features/target.parquet")
DEFAULT_REPORT = Path("reports/model2_results.json")
DEFAULT_MODEL_DIR = Path("models/energy_prediction")
DROP_COLUMNS = {"building_id", "prediction_timestamp", "precipDepth6HR"}
TARGET_COLUMN = "target_electricity_t_plus_1h"
BOUNDARIES = {
    "train": (pd.Timestamp("2016-01-01"), pd.Timestamp("2017-01-01")),
    "validation": (pd.Timestamp("2017-01-01"), pd.Timestamp("2017-07-01")),
    "test": (pd.Timestamp("2017-07-01"), pd.Timestamp("2018-01-01")),
}


def split_name(timestamp: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=timestamp.index, dtype="string")
    for name, (start, end) in BOUNDARIES.items():
        result.loc[timestamp.ge(start) & timestamp.lt(end)] = name
    return result


def _reader(features: Path, target: Path, columns: list[str], batch_size: int) -> Iterator[pd.DataFrame]:
    feature_batches = ds.dataset(features, format="parquet").scanner(columns=columns, batch_size=batch_size).to_batches()
    target_batches = ds.dataset(target, format="parquet").scanner(columns=["building_id", "prediction_timestamp", TARGET_COLUMN], batch_size=batch_size).to_batches()
    for feature_batch, target_batch in zip(feature_batches, target_batches):
        frame = feature_batch.to_pandas()
        target_frame = target_batch.to_pandas()
        keys = ["building_id", "prediction_timestamp"]
        if not frame[keys].reset_index(drop=True).equals(target_frame[keys].reset_index(drop=True)):
            raise ValueError("Feature and target Parquet row keys are not aligned.")
        frame[TARGET_COLUMN] = target_frame[TARGET_COLUMN].to_numpy()
        yield frame


def _feature_columns(features: Path) -> tuple[list[str], list[str], list[str]]:
    columns = ds.dataset(features, format="parquet").schema.names
    categorical = [column for column in ["primaryspaceusage"] if column in columns and column not in DROP_COLUMNS]
    numeric = [column for column in columns if column not in DROP_COLUMNS and column not in categorical]
    return numeric, categorical, ["building_id", "prediction_timestamp", *numeric, *categorical]


class Preprocessor:
    def __init__(self, numeric: list[str], categorical: list[str]) -> None:
        self.numeric, self.categorical = numeric, categorical
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)

    def fit(self, frame: pd.DataFrame) -> "Preprocessor":
        if self.categorical:
            self.encoder.fit(frame[self.categorical].fillna("Missing").astype(str))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = frame[self.numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if not self.categorical:
            return numeric
        encoded = self.encoder.transform(frame[self.categorical].fillna("Missing").astype(str))
        return np.column_stack([numeric, encoded]).astype(np.float32, copy=False)

    @property
    def output_names(self) -> list[str]:
        names = list(self.numeric)
        if self.categorical:
            names.extend(self.encoder.get_feature_names_out(self.categorical).tolist())
        return names


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {"mae": float(mean_absolute_error(actual, predicted)), "rmse": float(np.sqrt(mean_squared_error(actual, predicted))), "r2": float(r2_score(actual, predicted))}


def _collect_training_sample(features: Path, target: Path, columns: list[str], preprocessor: Preprocessor, max_rows: int, batch_size: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    frames: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    seen = 0
    missing_targets = 0
    for frame in _reader(features, target, columns, batch_size):
        timestamps = pd.to_datetime(frame["prediction_timestamp"])
        train_mask = split_name(timestamps).eq("train")
        usable = train_mask & frame[TARGET_COLUMN].notna()
        missing_targets += int(train_mask.sum() - usable.sum())
        selected = frame.loc[usable].head(max_rows - seen)
        if selected.empty:
            continue
        targets.append(selected[TARGET_COLUMN].to_numpy(dtype=np.float32))
        frames.append(selected.drop(columns=[TARGET_COLUMN]))
        seen += len(selected)
        if seen >= max_rows:
            break
    sample = pd.concat(frames, ignore_index=True)
    return preprocessor.transform(sample), np.concatenate(targets), {"train_rows": seen, "missing_train_targets": missing_targets}


def _evaluate(model: Any, features: Path, target: Path, columns: list[str], preprocessor: Preprocessor, split: str, batch_size: int) -> dict[str, float | int]:
    actual: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    for frame in _reader(features, target, columns, batch_size):
        timestamps = pd.to_datetime(frame["prediction_timestamp"])
        selected = frame.loc[split_name(timestamps).eq(split) & frame[TARGET_COLUMN].notna()]
        if selected.empty:
            continue
        actual.append(selected[TARGET_COLUMN].to_numpy(dtype=np.float32))
        predicted.append(model.predict(preprocessor.transform(selected.drop(columns=[TARGET_COLUMN]))))
    y_true = np.concatenate(actual)
    return {"rows": int(len(y_true)), **_metrics(y_true, np.concatenate(predicted))}


def train_and_evaluate(features: str | os.PathLike = DEFAULT_FEATURES, target: str | os.PathLike = DEFAULT_TARGET, report_path: str | os.PathLike = DEFAULT_REPORT, model_dir: str | os.PathLike = DEFAULT_MODEL_DIR, max_train_rows: int = 1_000_000, batch_size: int = 100_000) -> dict[str, Any]:
    features, target, report_path, model_dir = map(Path, [features, target, report_path, model_dir])
    numeric, categorical, columns = _feature_columns(features)
    preprocessor = Preprocessor(numeric, categorical)
    first = next(_reader(features, target, columns, batch_size))
    train_first = first.loc[split_name(pd.to_datetime(first["prediction_timestamp"])).eq("train")]
    preprocessor.fit(train_first)
    x_train, y_train, sampling = _collect_training_sample(features, target, columns, preprocessor, max_train_rows, batch_size)
    dummy = DummyRegressor(strategy="mean").fit(x_train, y_train)
    xgb = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", tree_method="hist", n_jobs=max(1, (os.cpu_count() or 2) - 1), random_state=42).fit(x_train, y_train)
    model_dir.mkdir(parents=True, exist_ok=True)
    xgb.save_model(model_dir / "xgboost_model.json")
    pd.to_pickle(dummy, model_dir / "dummy_regressor.pkl")
    (model_dir / "feature_columns.json").write_text(json.dumps(preprocessor.output_names, indent=2), encoding="utf-8")
    results: dict[str, Any] = {"generated": datetime.now(timezone.utc).isoformat(), "features": str(features), "target": str(target), "dropped_features": ["precipDepth6HR"], "numeric_features": numeric, "categorical_features": categorical, "preprocessing": "numeric NaNs preserved for XGBoost; primaryspaceusage missing values encoded as Missing; encoder fit on training data only", "max_train_rows": max_train_rows, "training": sampling, "parameters": {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "random_state": 42}, "models": {}, "mlflow": "not configured: mlflow is not installed"}
    for name, model in [("dummy", dummy), ("xgboost", xgb)]:
        results["models"][name] = {split: _evaluate(model, features, target, columns, preprocessor, split, batch_size) for split in ["validation", "test"]}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    print(json.dumps(train_and_evaluate(
        features=args.features,
        target=args.target,
        report_path=args.report,
        model_dir=args.model_dir,
        max_train_rows=args.max_train_rows,
        batch_size=args.batch_size,
    ), indent=2, default=str))


if __name__ == "__main__":
    main()