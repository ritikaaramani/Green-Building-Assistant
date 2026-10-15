"""
inspect_anomaly_windows.py
==========================
For each flagged anomaly in the V2 predictions, print a ±N-hour
context window from the feature dataset so patterns (spikes, drops,
sustained shifts) can be evaluated before deciding whether they are
true or spurious anomalies.

Usage:
    python src/models/inspect_anomaly_windows.py
    python src/models/inspect_anomaly_windows.py --window 12 --top 5
    python src/models/inspect_anomaly_windows.py --building Robin_office_Maryann
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_FILE = ROOT / "data/processed/anomaly/anomaly_predictions_v2.csv"
FEATURES_FILE = ROOT / "data/processed/anomaly/anomaly_features_v2.csv"

CONTEXT_COLUMNS = [
    "timestamp",
    "consumption",
    "rolling_mean_24h",
    "rolling_std_24h",
    "deviation_24h",
    "anomaly_score",
    "is_anomaly",
]

_SEPARATOR = "=" * 72


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect anomaly context windows.")
    p.add_argument(
        "--window",
        type=int,
        default=6,
        help="Hours of context on each side of an anomaly (default: 6).",
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of most-anomalous readings to inspect per building "
             "(default: 10; 0 = all flagged).",
    )
    p.add_argument(
        "--building",
        type=str,
        default=None,
        help="Restrict output to a single building_id.",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Only inspect anomalies whose anomaly_score is below this "
             "threshold (lower = more anomalous).",
    )
    return p.parse_args()


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = pd.read_csv(PREDICTIONS_FILE, parse_dates=["timestamp"])
    preds["is_anomaly"] = preds["is_anomaly"].astype(str).str.lower() == "true"

    feats = pd.read_csv(FEATURES_FILE, parse_dates=["timestamp"])
    return preds, feats


def _context_window(
    feats_building: pd.DataFrame,
    target_ts: pd.Timestamp,
    window_hours: int,
    anomaly_score: float,
) -> pd.DataFrame:
    """Return rows within ±window_hours of target_ts, with is_anomaly column."""
    lo = target_ts - pd.Timedelta(hours=window_hours)
    hi = target_ts + pd.Timedelta(hours=window_hours)
    mask = feats_building["timestamp"].between(lo, hi)
    window = feats_building[mask].copy()
    window["is_anomaly"] = window["timestamp"] == target_ts
    # Fill anomaly_score only for the target row.
    window["anomaly_score"] = window["timestamp"].apply(
        lambda t: round(anomaly_score, 5) if t == target_ts else float("nan")
    )
    return window[CONTEXT_COLUMNS]


def inspect_windows(
    window_hours: int = 6,
    top_n: int = 10,
    building_filter: str | None = None,
    score_threshold: float | None = None,
) -> None:
    preds, feats = _load_data()

    anomalies = preds[preds["is_anomaly"]].copy()
    anomalies = anomalies.sort_values("anomaly_score")  # lowest = most anomalous

    if score_threshold is not None:
        anomalies = anomalies[anomalies["anomaly_score"] < score_threshold]

    buildings = (
        [building_filter]
        if building_filter
        else sorted(anomalies["building_id"].unique())
    )

    total_shown = 0

    for building in buildings:
        b_anomalies = anomalies[anomalies["building_id"] == building]
        b_feats = feats[feats["building_id"] == building].sort_values("timestamp")

        if b_anomalies.empty:
            print(f"\n{_SEPARATOR}")
            print(f"BUILDING: {building}  — no flagged anomalies.")
            continue

        selected = b_anomalies if top_n == 0 else b_anomalies.head(top_n)

        print(f"\n{_SEPARATOR}")
        print(f"BUILDING : {building}")
        print(f"Flagged  : {len(b_anomalies)}  |  Showing: {len(selected)}")
        print(f"Window   : ±{window_hours} h around each flagged reading")

        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            ts = row["timestamp"]
            score = row["anomaly_score"]
            print(f"\n  [{rank}]  {ts}   score={score:.5f}")
            window_df = _context_window(b_feats, ts, window_hours, score)
            formatted = window_df.copy()
            numeric_cols = [
                "consumption", "rolling_mean_24h",
                "rolling_std_24h", "deviation_24h", "anomaly_score",
            ]
            formatted[numeric_cols] = (
                formatted[numeric_cols].map(
                    lambda x: f"{x:.3f}" if pd.notna(x) else ""
                )
            )
            formatted["is_anomaly"] = formatted["is_anomaly"].map(
                {True: ">>> ANOMALY <<<", False: ""}
            )
            print(formatted.to_string(index=False))

        total_shown += len(selected)

    print(f"\n{_SEPARATOR}")
    print(f"Total readings shown: {total_shown}")


if __name__ == "__main__":
    args = _parse_args()
    inspect_windows(
        window_hours=args.window,
        top_n=args.top,
        building_filter=args.building,
        score_threshold=args.score_threshold,
    )
