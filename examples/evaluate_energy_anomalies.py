"""Evaluate causal, rolling energy anomaly detection on a later period."""

import numpy as np
import pandas as pd

from green_building.recommendations.episode_analysis import (
    summarize_anomaly_episodes,
)


DATA_PATH = "data/raw/bdg2/meters/cleaned/electricity_cleaned.csv"
BUILDING_ID = "Hog_education_Jewel"
SPLIT_DATE = "2017-07-01"

# Use the previous 8 occurrences of the same weekday and hour.
# Require at least 4 previous observations before flagging anomalies.
WINDOW = 8
MIN_PERIODS = 4
RELATIVE_THRESHOLD = 0.20
ROBUST_SIGMA_MULTIPLIER = 3.0


def rolling_mad(window):
    """Return a robust estimate of spread for a historical window."""
    median = np.median(window)
    return 1.4826 * np.median(np.abs(window - median))


def main():
    df = pd.read_csv(
        DATA_PATH,
        usecols=["timestamp", BUILDING_ID],
        parse_dates=["timestamp"],
    )

    df[BUILDING_ID] = pd.to_numeric(
        df[BUILDING_ID], errors="coerce"
    )
    df = df.dropna(
        subset=["timestamp", BUILDING_ID]
    ).sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid readings found")

    if (df[BUILDING_ID] < 0).any():
        raise ValueError("Negative readings need investigation")

    # Keep the original chronological train/test split.
    split_timestamp = pd.Timestamp(SPLIT_DATE)
    train = df[df["timestamp"] < split_timestamp].copy()
    test = df[df["timestamp"] >= split_timestamp].copy()

    if train.empty or test.empty:
        raise ValueError("Both training and test periods must contain data")

    # Group readings by weekday and hour.
    df["hour_of_week"] = (
        df["timestamp"].dt.dayofweek * 24
        + df["timestamp"].dt.hour
    )

    # Causal rolling baseline: every prediction uses earlier observations
    # from the same weekday-hour group, never the current or future reading.
    grouped = df.groupby("hour_of_week")[BUILDING_ID]

    df["baseline"] = grouped.transform(
        lambda series: (
            series.shift(1)
            .rolling(WINDOW, min_periods=MIN_PERIODS)
            .median()
        )
    )

    df["spread"] = grouped.transform(
        lambda series: (
            series.shift(1)
            .rolling(WINDOW, min_periods=MIN_PERIODS)
            .apply(rolling_mad, raw=True)
        )
    )

    df["spread"] *= 1.0  # MAD is scaled inside rolling_mad.

    robust_margin = ROBUST_SIGMA_MULTIPLIER * df["spread"]
    relative_margin = RELATIVE_THRESHOLD * df["baseline"].abs()

    df["threshold"] = df["baseline"] + pd.concat(
        [robust_margin, relative_margin],
        axis=1,
    ).max(axis=1)

    df["anomaly"] = (
        df[BUILDING_ID] > df["threshold"]
    )

    # Evaluate only the later period, excluding readings without enough
    # historical observations to establish a baseline.
    evaluated = df[
        (df["timestamp"] >= split_timestamp)
        & df["baseline"].notna()
        & df["threshold"].notna()
    ].copy()

    print("Building:", BUILDING_ID)
    print(
        "Training period:",
        train["timestamp"].min(),
        "to",
        train["timestamp"].max(),
    )
    print("Training readings:", len(train))
    print(
        "Test period:",
        test["timestamp"].min(),
        "to",
        test["timestamp"].max(),
    )
    print("Evaluated test readings:", len(evaluated))
    print("Flagged test anomalies:", int(evaluated["anomaly"].sum()))
    print(
        "Test anomaly rate (%):",
        round(evaluated["anomaly"].mean() * 100, 3),
    )

    print("\nMonthly anomaly report:")
    evaluated["month"] = evaluated["timestamp"].dt.to_period("M")

    monthly = evaluated.groupby("month").agg(
        readings=("anomaly", "size"),
        anomalies=("anomaly", "sum"),
        mean_consumption=(BUILDING_ID, "mean"),
    )

    monthly["anomaly_rate_pct"] = (
        monthly["anomalies"] / monthly["readings"] * 100
    )

    numeric_monthly = monthly.select_dtypes(include="number")
    monthly[numeric_monthly.columns] = numeric_monthly.round(3)
    print(monthly.to_string())

    print("\nFlagged test readings (first 25):")
    flagged = evaluated.loc[
        evaluated["anomaly"],
        ["timestamp", BUILDING_ID, "baseline", "threshold"],
    ].copy()

    flagged["excess_above_baseline"] = (
        flagged[BUILDING_ID] - flagged["baseline"]
    )

    flagged = flagged.head(25)
    numeric_columns = flagged.select_dtypes(include="number").columns
    flagged[numeric_columns] = flagged[numeric_columns].round(3)
    print(flagged.to_string(index=False))

    # Descriptive comparison only: the test period may have different
    # weather, occupancy, schedules, or operating conditions.
    train["hour"] = train["timestamp"].dt.hour
    test["hour"] = test["timestamp"].dt.hour

    comparison = pd.DataFrame({
        "training_mean": train.groupby("hour")[BUILDING_ID].mean(),
        "test_mean": test.groupby("hour")[BUILDING_ID].mean(),
    })

    comparison["difference"] = (
        comparison["test_mean"] - comparison["training_mean"]
    )

    print("\nTraining vs test average by hour:")
    print(comparison.round(2).to_string())



    print("\nAnomaly episode analysis:")

    episodes = summarize_anomaly_episodes(
        evaluated,
        BUILDING_ID,
    )

    if episodes.empty:
        print("No anomaly episodes detected.")
    else:
        print("Number of episodes:", len(episodes))
        print(
            "Total flagged readings:",
            int(episodes["flagged_readings"].sum()),
        )
        print(
            "Longest episode (hours):",
            int(episodes["duration_hours"].max()),
        )

        print("\nEvidence-level counts:")
        print(
            episodes["evidence_level"]
            .value_counts()
            .to_string()
        )

        print("\nTop 10 episodes by duration:")
        display = episodes.sort_values(
            ["duration_hours", "mean_excess_threshold"],
            ascending=False,
        ).head(10).copy()

        numeric_columns = display.select_dtypes(
            include="number"
        ).columns
        display[numeric_columns] = display[numeric_columns].round(2)

        print(display.to_string(index=False))


if __name__ == "__main__":
    main()
