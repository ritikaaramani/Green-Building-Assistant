
"""Analyze BDG2 electricity data using time-of-week baselines."""

import pandas as pd


def analyze_building_energy(
    csv_path,
    building_id,
    nighttime_hours=(0, 5),
    baseline_quantile=0.95,
):
    """Analyze consumption and flag preliminary historical anomalies."""
    if not 0 < baseline_quantile < 1:
        raise ValueError("baseline_quantile must be between 0 and 1")

    df = pd.read_csv(
        csv_path,
        usecols=["timestamp", building_id],
        parse_dates=["timestamp"],
    )
    df[building_id] = pd.to_numeric(df[building_id], errors="coerce")
    df = df.dropna(subset=["timestamp", building_id]).copy()

    if df.empty:
        raise ValueError(f"No valid readings for {building_id}")

    if (df[building_id] < 0).any():
        raise ValueError("Negative readings need investigation")

    df = df.sort_values("timestamp").reset_index(drop=True)
    values = df[building_id]

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_of_week"] = df["day_of_week"] * 24 + df["hour"]

    # Compare each reading with readings for the same weekday and hour.
    groups = df.groupby("hour_of_week")[building_id]
    df["profile_median"] = groups.transform("median")

    absolute_deviation = (values - df["profile_median"]).abs()
    df["mad"] = absolute_deviation.groupby(
        df["hour_of_week"]
    ).transform("median")

    # Robust threshold: at least 20% above the typical profile,
    # or three robust standard deviations above it, whichever is larger.
    robust_sigma = 1.4826 * df["mad"]
    robust_limit = 3 * robust_sigma
    relative_limit = df["profile_median"].abs() * 0.20

    df["threshold"] = (
        df["profile_median"]
        + pd.concat(
            [robust_limit, relative_limit],
            axis=1,
        ).max(axis=1)
    )

    df["is_anomaly"] = values > df["threshold"]

    nighttime = df[df["hour"].between(*nighttime_hours)]
    nighttime_count = int(nighttime["is_anomaly"].sum())
    nighttime_rate = (
        float(nighttime["is_anomaly"].mean())
        if not nighttime.empty
        else 0.0
    )


    # Require at least 3 flagged nighttime readings and a 2% rate.
    # These are preliminary thresholds that require validation.
    nighttime_anomaly = (
        nighttime_count >= 3 and nighttime_rate >= 0.02
    )


    # Keep the original summary fields for compatibility.
    rolling_mean = (
        df.set_index("timestamp")[building_id]
        .rolling("7D", min_periods=24)
        .mean()
    )

    return {
        "building_id": building_id,
        "valid_readings": int(len(df)),
        "mean_energy": round(float(values.mean()), 3),
        "median_energy": round(float(values.median()), 3),
        "max_energy": round(float(values.max()), 3),
        "p95_energy": round(float(values.quantile(baseline_quantile)), 3),
        "nighttime_anomaly": bool(nighttime_anomaly),
        "nighttime_anomaly_count": nighttime_count,
        "nighttime_anomaly_rate_pct": round(nighttime_rate * 100, 2),
        "anomaly_count": int(df["is_anomaly"].sum()),
        "anomaly_rate_pct": round(float(df["is_anomaly"].mean()) * 100, 2),
        "rolling_mean_available": bool(rolling_mean.notna().any()),
        "hourly_profile": (
            df.groupby("hour")[building_id]
            .mean()
            .round(3)
            .to_dict()
        ),
    }
