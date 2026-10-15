
import numpy as np
import pandas as pd

from green_building.recommendations.energy_analysis import (
    analyze_building_energy,
)

from green_building.recommendations.episode_analysis import (
    summarize_anomaly_episodes,
)


def test_detects_repeated_nighttime_spikes(tmp_path):
    timestamps = pd.date_range(
        "2025-01-06",
        periods=24 * 28,
        freq="h",
    )

    values = np.full(len(timestamps), 50.0)

    # Introduce repeated spikes at 2 AM across multiple weeks.
    for i, timestamp in enumerate(timestamps):
        if timestamp.hour == 2 and timestamp.day % 3 == 0:
            values[i] = 150.0

    path = tmp_path / "electricity.csv"
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "test_building": values,
        }
    ).to_csv(path, index=False)

    result = analyze_building_energy(
        str(path),
        "test_building",
    )

    assert result["valid_readings"] == len(timestamps)
    assert result["anomaly_count"] > 0
    assert result["nighttime_anomaly"] is True


def test_stable_consumption_has_no_anomalies(tmp_path):
    timestamps = pd.date_range(
        "2025-01-06",
        periods=24 * 28,
        freq="h",
    )

    path = tmp_path / "stable.csv"
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "test_building": np.full(len(timestamps), 50.0),
        }
    ).to_csv(path, index=False)

    result = analyze_building_energy(
        str(path),
        "test_building",
    )

    assert result["anomaly_count"] == 0
    assert result["nighttime_anomaly"] is False


def test_groups_consecutive_anomalies_into_one_episode():
    data = pd.DataFrame({
        "timestamp": pd.date_range(
            "2025-01-06 10:00",
            periods=4,
            freq="h",
        ),
        "test_building": [80.0, 90.0, 85.0, 50.0],
        "baseline": [50.0, 50.0, 50.0, 50.0],
        "threshold": [60.0, 60.0, 60.0, 60.0],
        "anomaly": [True, True, True, False],
    })

    episodes = summarize_anomaly_episodes(data, "test_building")

    assert len(episodes) == 1
    assert episodes.iloc[0]["flagged_readings"] == 3
    assert episodes.iloc[0]["duration_hours"] == 3
    assert episodes.iloc[0]["evidence_level"] == "high"


def test_separates_anomalies_with_a_normal_reading():
    data = pd.DataFrame({
        "timestamp": pd.date_range(
            "2025-01-06 10:00",
            periods=3,
            freq="h",
        ),
        "test_building": [80.0, 50.0, 90.0],
        "baseline": [50.0, 50.0, 50.0],
        "threshold": [60.0, 60.0, 60.0],
        "anomaly": [True, False, True],
    })

    episodes = summarize_anomaly_episodes(data, "test_building")

    assert len(episodes) == 2
    assert episodes["flagged_readings"].tolist() == [1, 1]


def test_empty_anomaly_input_returns_empty_summary():
    data = pd.DataFrame({
        "timestamp": pd.date_range(
            "2025-01-06 10:00",
            periods=3,
            freq="h",
        ),
        "test_building": [50.0, 51.0, 49.0],
        "baseline": [50.0, 50.0, 50.0],
        "threshold": [60.0, 60.0, 60.0],
        "anomaly": [False, False, False],
    })

    episodes = summarize_anomaly_episodes(data, "test_building")

    assert episodes.empty
