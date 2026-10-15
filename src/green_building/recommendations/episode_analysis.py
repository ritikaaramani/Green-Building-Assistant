
"""Summarize consecutive energy-anomaly episodes."""

import pandas as pd


def summarize_anomaly_episodes(
    data: pd.DataFrame,
    building_id: str,
) -> pd.DataFrame:
    """Group consecutive hourly anomalies and summarize their evidence.

    Required columns:
        timestamp, anomaly, baseline, threshold, and building_id.

    Evidence levels are heuristic categories, not probabilities.
    """
    required = {
        "timestamp",
        "anomaly",
        "baseline",
        "threshold",
        building_id,
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = data.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["anomaly"] = df["anomaly"].astype(bool)

    flagged = df[df["anomaly"]].copy()
    columns = [
        "start",
        "end",
        "flagged_readings",
        "mean_consumption",
        "max_consumption",
        "mean_baseline",
        "mean_excess_baseline",
        "mean_excess_threshold",
        "mean_excess_pct",
        "duration_hours",
        "evidence_level",
    ]

    if flagged.empty:
        return pd.DataFrame(columns=columns)

    previous_anomaly = df["anomaly"].shift(1, fill_value=False)
    previous_timestamp = df["timestamp"].shift(1)
    gap = (
        df["timestamp"] - previous_timestamp
    ) > pd.Timedelta(hours=1)

    df["new_episode"] = df["anomaly"] & (~previous_anomaly | gap)
    df["episode_id"] = df["new_episode"].cumsum()

    flagged = df[df["anomaly"]].copy()
    flagged["excess_above_baseline"] = (
        flagged[building_id] - flagged["baseline"]
    )
    flagged["excess_above_threshold"] = (
        flagged[building_id] - flagged["threshold"]
    )
    flagged["excess_pct_of_baseline"] = (
        flagged["excess_above_baseline"]
        / flagged["baseline"].abs().clip(lower=1e-9)
        * 100
    )

    episodes = flagged.groupby("episode_id").agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        flagged_readings=("timestamp", "size"),
        mean_consumption=(building_id, "mean"),
        max_consumption=(building_id, "max"),
        mean_baseline=("baseline", "mean"),
        mean_excess_baseline=("excess_above_baseline", "mean"),
        mean_excess_threshold=("excess_above_threshold", "mean"),
        mean_excess_pct=("excess_pct_of_baseline", "mean"),
    )

    episodes["duration_hours"] = (
        (episodes["end"] - episodes["start"]).dt.total_seconds()
        / 3600
        + 1
    )

    episodes["evidence_level"] = "low"

    medium = (
        (episodes["duration_hours"] >= 2)
        & (episodes["mean_excess_threshold"] > 0)
    )
    high = (
        (episodes["duration_hours"] >= 3)
        & (episodes["mean_excess_pct"] >= 20)
        & (episodes["mean_excess_threshold"] > 0)
    )

    episodes.loc[medium, "evidence_level"] = "medium"
    episodes.loc[high, "evidence_level"] = "high"

    return episodes[columns].reset_index(drop=True)
