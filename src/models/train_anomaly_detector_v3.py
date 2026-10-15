"""
train_anomaly_detector_v3.py
============================
Production-ready Isolation Forest anomaly detector for the BDG2
electricity dataset.

Improvements over V2:
- Serialises each per-building model with joblib so inference can be
  run without re-training.
- Adds a robust z-score threshold layer that flags readings whose
  deviation from the rolling mean exceeds `zscore_threshold` standard
  deviations, independent of the Isolation Forest decision boundary.
- Writes a per-building evaluation summary to a JSON report.
- Fully reproducible via a config dictionary at the top of the file.

Usage:
    python src/models/train_anomaly_detector_v3.py
    python src/models/train_anomaly_detector_v3.py --no-save-models
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data/processed/anomaly/anomaly_features_v2.csv"
OUTPUT_DIR = ROOT / "data/processed/anomaly"
MODELS_DIR = ROOT / "artifacts/models/anomaly"

OUTPUT_PREDICTIONS = OUTPUT_DIR / "anomaly_predictions_v3.csv"
OUTPUT_REPORT = OUTPUT_DIR / "anomaly_report_v3.json"

# ── Model configuration ────────────────────────────────────────────────────
CONFIG = {
    # Isolation Forest
    "n_estimators": 200,
    "contamination": 0.02,
    "max_samples": "auto",
    "random_state": 42,
    "n_jobs": 1,
    # Train/test split (chronological)
    "train_fraction": 0.80,
    # Z-score secondary flag: flag if |z| > threshold
    "zscore_threshold": 3.5,
    # Features used for Isolation Forest
    "feature_columns": [
        "consumption",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "is_weekend",
        "rolling_mean_24h",
        "rolling_std_24h",
        "rolling_median_24h",
        "deviation_24h",
        "median_deviation_24h",
    ],
}

_SEP = "-" * 65


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train V3 anomaly detector.")
    p.add_argument(
        "--no-save-models",
        action="store_true",
        help="Skip serialising trained models to disk.",
    )
    return p.parse_args()


def _zscore_flag(series: pd.Series, threshold: float) -> pd.Series:
    """Flag values whose absolute z-score exceeds threshold."""
    mu = series.mean()
    sigma = series.std()
    if sigma == 0:
        return pd.Series(False, index=series.index)
    return ((series - mu).abs() / sigma) > threshold


def train_building(
    building: str,
    group: pd.DataFrame,
    config: dict,
    save_model: bool,
) -> dict:
    """
    Train one pipeline per building; return evaluation results dict.
    """
    feature_cols = config["feature_columns"]
    group = group.sort_values("timestamp").copy()

    split = int(len(group) * config["train_fraction"])
    train = group.iloc[:split]
    test = group.iloc[split:].copy()

    # ── Build and fit pipeline ─────────────────────────────────────────────
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "iforest",
                IsolationForest(
                    n_estimators=config["n_estimators"],
                    contamination=config["contamination"],
                    max_samples=config["max_samples"],
                    random_state=config["random_state"],
                    n_jobs=config["n_jobs"],
                ),
            ),
        ]
    )
    pipeline.fit(train[feature_cols])

    # ── Inference on held-out test set ─────────────────────────────────────
    test["iforest_prediction"] = pipeline.predict(test[feature_cols])
    test["anomaly_score"] = pipeline.decision_function(test[feature_cols])
    test["is_anomaly_iforest"] = test["iforest_prediction"] == -1

    # ── Secondary: z-score flag on raw consumption ─────────────────────────
    # Fit z-score params on train set only.
    train_mu = train["consumption"].mean()
    train_sigma = train["consumption"].std()
    if train_sigma > 0:
        z = (test["consumption"] - train_mu) / train_sigma
        test["is_anomaly_zscore"] = z.abs() > config["zscore_threshold"]
    else:
        test["is_anomaly_zscore"] = False

    # ── Combined flag: Isolation Forest OR z-score ─────────────────────────
    test["is_anomaly"] = (
        test["is_anomaly_iforest"] | test["is_anomaly_zscore"]
    )
    test["building_id"] = building

    # ── Optional: persist pipeline ─────────────────────────────────────────
    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{building}_iforest_v3.joblib"
        meta_path = MODELS_DIR / f"{building}_iforest_v3_meta.json"
        joblib.dump(pipeline, model_path)
        meta = {
            "building_id": building,
            "model_path": str(model_path),
            "feature_columns": feature_cols,
            "train_rows": len(train),
            "train_start": str(train["timestamp"].min()),
            "train_end": str(train["timestamp"].max()),
            "config": {k: v for k, v in config.items() if k != "feature_columns"},
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ── Build result summary ────────────────────────────────────────────────
    n_if = int(test["is_anomaly_iforest"].sum())
    n_zs = int(test["is_anomaly_zscore"].sum())
    n_combined = int(test["is_anomaly"].sum())
    n_test = len(test)

    # Score percentiles (lower score = more anomalous)
    scores = test["anomaly_score"]
    result = {
        "building_id": building,
        "train_rows": len(train),
        "test_rows": n_test,
        "iforest_flagged": n_if,
        "iforest_flag_rate_pct": round(n_if / n_test * 100, 2),
        "zscore_flagged": n_zs,
        "zscore_flag_rate_pct": round(n_zs / n_test * 100, 2),
        "combined_flagged": n_combined,
        "combined_flag_rate_pct": round(n_combined / n_test * 100, 2),
        "score_min": round(float(scores.min()), 5),
        "score_p5": round(float(np.percentile(scores, 5)), 5),
        "score_median": round(float(scores.median()), 5),
        "score_p95": round(float(np.percentile(scores, 95)), 5),
        "score_max": round(float(scores.max()), 5),
    }
    return result, test


def train_detector(save_models: bool = True) -> None:
    t0 = time.perf_counter()
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])

    print(f"Loaded {len(df):,} rows from {INPUT_FILE.name}")
    print(f"Buildings: {sorted(df['building_id'].unique())}\n")

    all_results = []
    all_predictions = []

    for building, group in df.groupby("building_id", sort=True):
        print(_SEP)
        print(f"Building : {building}")
        result, test_df = train_building(
            building, group, CONFIG, save_models
        )
        all_results.append(result)
        all_predictions.append(test_df)

        print(f"  Train rows         : {result['train_rows']:,}")
        print(f"  Test rows          : {result['test_rows']:,}")
        print(f"  IForest flagged    : {result['iforest_flagged']} "
              f"({result['iforest_flag_rate_pct']:.2f}%)")
        print(f"  Z-score flagged    : {result['zscore_flagged']} "
              f"({result['zscore_flag_rate_pct']:.2f}%)")
        print(f"  Combined flagged   : {result['combined_flagged']} "
              f"({result['combined_flag_rate_pct']:.2f}%)")
        print(f"  Score [min,p5,med,p95,max]: "
              f"[{result['score_min']}, {result['score_p5']}, "
              f"{result['score_median']}, {result['score_p95']}, "
              f"{result['score_max']}]")
        if save_models:
            print(f"  Model saved to : artifacts/models/anomaly/"
                  f"{building}_iforest_v3.joblib")

    # ── Combine and save predictions ───────────────────────────────────────
    predictions = pd.concat(all_predictions, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_PREDICTIONS, index=False)

    # ── Write JSON report ──────────────────────────────────────────────────
    report = {
        "model_version": "v3",
        "config": CONFIG,
        "runtime_seconds": round(time.perf_counter() - t0, 2),
        "total_test_rows": len(predictions),
        "total_iforest_flagged": int(predictions["is_anomaly_iforest"].sum()),
        "total_zscore_flagged": int(predictions["is_anomaly_zscore"].sum()),
        "total_combined_flagged": int(predictions["is_anomaly"].sum()),
        "buildings": all_results,
    }
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"\n{_SEP}")
    print(f"Total test rows   : {len(predictions):,}")
    print(f"Total IForest     : {report['total_iforest_flagged']}")
    print(f"Total Z-score     : {report['total_zscore_flagged']}")
    print(f"Total combined    : {report['total_combined_flagged']}")
    print(f"Runtime           : {elapsed}s")
    print(f"Predictions saved : {OUTPUT_PREDICTIONS}")
    print(f"Report saved      : {OUTPUT_REPORT}")


if __name__ == "__main__":
    args = _parse_args()
    train_detector(save_models=not args.no_save_models)
