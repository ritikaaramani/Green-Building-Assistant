"""Minimal local inference API for the Phase 1 energy prediction demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[4]
FEATURES_PATH = ROOT / "data/processed/bdg2/features/features.parquet"
TARGET_PATH = ROOT / "data/processed/bdg2/features/target.parquet"
MODEL_PATH = ROOT / "models/energy_prediction/xgboost_model.json"
DUMMY_PATH = ROOT / "models/energy_prediction/dummy_regressor.pkl"
COLUMNS_PATH = ROOT / "models/energy_prediction/feature_columns.json"
KEY_COLUMNS = ["building_id", "prediction_timestamp"]
NUMERIC_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend", "electricity_lag_1h",
    "electricity_lag_24h", "electricity_rolling_mean_24h", "electricity_rolling_std_24h",
    "sqm", "airTemperature", "cloudCoverage", "dewTemperature", "precipDepth1HR",
    "seaLvlPressure", "windDirection", "windSpeed",
]


def _load_model() -> tuple[XGBRegressor, Any, list[str]]:
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    dummy = pd.read_pickle(DUMMY_PATH)
    feature_columns = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
    if model.n_features_in_ != len(feature_columns):
        raise RuntimeError("Saved model and feature_columns.json have incompatible schemas.")
    return model, dummy, feature_columns


MODEL, DUMMY, TRANSFORMED_COLUMNS = _load_model()
FEATURE_DATASET = ds.dataset(FEATURES_PATH, format="parquet")
TARGET_DATASET = ds.dataset(TARGET_PATH, format="parquet")

app = FastAPI(title="Green Building Assistant Phase 1 Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def _available_buildings() -> list[str]:
    values: set[str] = set()
    for batch in FEATURE_DATASET.scanner(columns=["building_id"], batch_size=250_000).to_batches():
        values.update(batch.column("building_id").to_pylist())
    return sorted(values)


def _row(building_id: str, timestamp: str) -> pd.DataFrame:
    parsed = pd.Timestamp(timestamp)
    table = FEATURE_DATASET.to_table(
        columns=[*KEY_COLUMNS, *NUMERIC_COLUMNS, "primaryspaceusage"],
        filter=(ds.field("building_id") == building_id) & (ds.field("prediction_timestamp") == parsed.to_datetime64()),
    )
    frame = table.to_pandas()
    if frame.empty:
        raise HTTPException(status_code=404, detail="No engineered feature row exists for that building and timestamp.")
    return frame.iloc[[0]].copy()


def _encoded_input(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce").copy()
    usage = frame["primaryspaceusage"].fillna("Missing").astype(str).iloc[0]
    encoded_columns = [column for column in TRANSFORMED_COLUMNS if column.startswith("primaryspaceusage_")]
    for column in encoded_columns:
        category = column.removeprefix("primaryspaceusage_")
        result[column] = float(usage == category)
    result = result.reindex(columns=TRANSFORMED_COLUMNS)
    if list(result.columns) != TRANSFORMED_COLUMNS:
        raise RuntimeError("Inference feature schema does not match training schema.")
    return result


def _actual(building_id: str, timestamp: str) -> float | None:
    parsed = pd.Timestamp(timestamp)
    table = TARGET_DATASET.to_table(
        columns=[*KEY_COLUMNS, "target_electricity_t_plus_1h"],
        filter=(ds.field("building_id") == building_id) & (ds.field("prediction_timestamp") == parsed.to_datetime64()),
    )
    if not table.num_rows:
        return None
    value = table.column("target_electricity_t_plus_1h")[0].as_py()
    return None if value is None else float(value)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "feature_columns": len(TRANSFORMED_COLUMNS), "model": str(MODEL_PATH)}


@app.get("/buildings")
def buildings() -> dict[str, list[str]]:
    return {"buildings": _available_buildings()}


@app.get("/timestamps")
def timestamps(building_id: str = Query(..., min_length=1)) -> dict[str, list[str]]:
    table = FEATURE_DATASET.to_table(columns=[*KEY_COLUMNS], filter=ds.field("building_id") == building_id)
    if not table.num_rows:
        raise HTTPException(status_code=404, detail="Building is not available in the engineered dataset.")
    return {"timestamps": [pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S") for value in table.column("prediction_timestamp").to_pylist()]}


@app.get("/predict")
def predict(building_id: str = Query(..., min_length=1), timestamp: str = Query(..., min_length=1)) -> dict[str, Any]:
    try:
        frame = _row(building_id, timestamp)
        features = _encoded_input(frame)
        xgb_prediction = float(MODEL.predict(features)[0])
        dummy_prediction = float(DUMMY.predict(features)[0])
        actual = _actual(building_id, timestamp)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "building_id": building_id,
        "prediction_timestamp": pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
        "target_timestamp": (pd.Timestamp(timestamp) + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "xgboost_prediction": xgb_prediction,
        "dummy_prediction": dummy_prediction,
        "actual_next_hour": actual,
        "absolute_error": None if actual is None else abs(xgb_prediction - actual),
        "feature_columns": list(features.columns),
        "feature_schema_matches_training": list(features.columns) == TRANSFORMED_COLUMNS,
        "unit_note": "Original dataset energy units; standardization has not been confirmed.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)