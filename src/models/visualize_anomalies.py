"""
src/models/visualize_anomalies.py
==================================
Generate all Model 3 anomaly detection figures.

Outputs (saved to reports/figures/model3/):
  - per_building_<building>.png     : time-series consumption with flagged
                                      anomalies from V1, V2, and V3 overlaid
  - v1_v2_comparison.png            : side-by-side bar chart of flag counts
  - anomaly_score_distributions.png : per-building score histograms (V3)
  - flag_rate_summary.png           : stacked bar of IForest vs z-score flags

Usage:
    python src/models/visualize_anomalies.py
    python src/models/visualize_anomalies.py --no-show   # save only, no GUI
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe on headless/CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/anomaly"
FIGURES_DIR = ROOT / "reports/figures/model3"

BUILDINGS = [
    "Robin_education_Derrick",
    "Robin_education_Kiera",
    "Robin_lodging_Pricilla",
    "Robin_office_Maryann",
    "Robin_office_Serena",
]

BUILDING_LABELS = {
    "Robin_education_Derrick": "Education — Derrick",
    "Robin_education_Kiera": "Education — Kiera",
    "Robin_lodging_Pricilla": "Lodging — Pricilla",
    "Robin_office_Maryann": "Office — Maryann",
    "Robin_office_Serena": "Office — Serena",
}

# Colour palette
C_NORMAL = "#4C72B0"
C_V1 = "#DD8452"
C_V2 = "#55A868"
C_V3_IF = "#C44E52"
C_V3_ZS = "#8172B3"
C_ZSCORE = "#8172B3"

ALPHA_SCATTER = 0.75
POINT_SIZE = 28


def _load_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _read(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        for col in df.columns:
            if col.startswith("is_anomaly"):
                df[col] = df[col].astype(str).str.lower() == "true"
        return df

    v1 = _read(DATA_DIR / "anomaly_predictions.csv")
    v2 = _read(DATA_DIR / "anomaly_predictions_v2.csv")
    v3 = _read(DATA_DIR / "anomaly_predictions_v3.csv")
    return v1, v2, v3


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#FAFAFA",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CCCCCC",
            "axes.labelcolor": "#333333",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.6,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
        }
    )


# ── Plot 1: Per-building time series ──────────────────────────────────────

def plot_per_building(
    v1: pd.DataFrame, v2: pd.DataFrame, v3: pd.DataFrame
) -> list[Path]:
    saved = []
    for building in BUILDINGS:
        label = BUILDING_LABELS[building]
        b1 = v1[v1["building_id"] == building].sort_values("timestamp")
        b2 = v2[v2["building_id"] == building].sort_values("timestamp")
        b3 = v3[v3["building_id"] == building].sort_values("timestamp")

        fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
        fig.suptitle(
            f"Anomaly Detection — {label}\n(Evaluation period: last 20% of 2016–2017)",
            fontsize=14, fontweight="bold", y=0.98,
        )

        for ax, df, version, color in zip(
            axes,
            [b1, b2, b3],
            ["V1 (IForest, linear features)", "V2 (IForest, cyclical+median features)",
             "V3 (IForest + Z-score, contamination=0.02)"],
            [C_V1, C_V2, C_V3_IF],
        ):
            normal = df[~df["is_anomaly"]]
            anomaly = df[df["is_anomaly"]]

            ax.plot(
                normal["timestamp"], normal["consumption"],
                color=C_NORMAL, linewidth=0.6, alpha=0.7, label="Normal",
            )
            ax.scatter(
                anomaly["timestamp"], anomaly["consumption"],
                color=color, s=POINT_SIZE, zorder=5, alpha=ALPHA_SCATTER,
                label=f"Flagged ({len(anomaly)})",
            )

            # Add rolling mean if available
            if "rolling_mean_24h" in df.columns:
                ax.plot(
                    df["timestamp"], df["rolling_mean_24h"],
                    color="#999999", linewidth=0.8, linestyle="--",
                    alpha=0.6, label="24h rolling mean",
                )

            ax.set_ylabel("kWh", fontsize=9)
            ax.set_title(version, fontsize=10, pad=3)
            ax.legend(loc="upper right", framealpha=0.85, ncol=3)
            ax.grid(True, axis="y")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())

        axes[-1].set_xlabel("Date", fontsize=9)
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out = FIGURES_DIR / f"per_building_{building}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
        print(f"  Saved: {out.name}")
    return saved


# ── Plot 2: V1 vs V2 flag count comparison ────────────────────────────────

def plot_v1_v2_comparison(v1: pd.DataFrame, v2: pd.DataFrame) -> Path:
    labels = [BUILDING_LABELS[b] for b in BUILDINGS]
    counts_v1 = [int(v1[v1["building_id"] == b]["is_anomaly"].sum()) for b in BUILDINGS]
    counts_v2 = [int(v2[v2["building_id"] == b]["is_anomaly"].sum()) for b in BUILDINGS]

    x = np.arange(len(BUILDINGS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width / 2, counts_v1, width, label="V1", color=C_V1, alpha=0.88)
    bars2 = ax.bar(x + width / 2, counts_v2, width, label="V2", color=C_V2, alpha=0.88)

    ax.bar_label(bars1, padding=3, fontsize=8)
    ax.bar_label(bars2, padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Flagged anomaly count (evaluation set)")
    ax.set_title(
        "V1 vs V2 — Anomaly Flag Counts per Building\n"
        "(no ground-truth labels; counts reflect model sensitivity, not accuracy)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linewidth=0.5)
    ax.set_ylim(0, max(max(counts_v1), max(counts_v2)) * 1.2)

    plt.tight_layout()
    out = FIGURES_DIR / "v1_v2_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


# ── Plot 3: Anomaly score distributions (V3) ─────────────────────────────

def plot_score_distributions(v3: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), sharey=False)
    fig.suptitle(
        "V3 Isolation Forest — Anomaly Score Distributions per Building\n"
        "(lower score = more isolated = more anomalous; threshold ≈ 0)",
        fontsize=12, fontweight="bold",
    )

    for ax, building in zip(axes, BUILDINGS):
        df = v3[v3["building_id"] == building].copy()
        df["is_anomaly_iforest"] = (
            df["is_anomaly_iforest"].astype(str).str.lower() == "true"
        )
        normal = df[~df["is_anomaly_iforest"]]["anomaly_score"]
        flagged = df[df["is_anomaly_iforest"]]["anomaly_score"]

        bins = np.linspace(df["anomaly_score"].min(), df["anomaly_score"].max(), 35)
        ax.hist(normal, bins=bins, color=C_NORMAL, alpha=0.7, label="Normal")
        ax.hist(flagged, bins=bins, color=C_V3_IF, alpha=0.85, label=f"Flagged ({len(flagged)})")
        ax.axvline(0, color="#333333", linewidth=1, linestyle="--", label="Score=0")

        ax.set_title(BUILDING_LABELS[building].replace(" — ", "\n"), fontsize=8.5)
        ax.set_xlabel("Anomaly score", fontsize=8)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(True, axis="y", linewidth=0.5)

    plt.tight_layout()
    out = FIGURES_DIR / "anomaly_score_distributions.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


# ── Plot 4: Flag rate summary — IForest vs z-score breakdown (V3) ─────────

def plot_flag_rate_summary(v3: pd.DataFrame) -> Path:
    for col in ["is_anomaly_iforest", "is_anomaly_zscore", "is_anomaly"]:
        v3[col] = v3[col].astype(str).str.lower() == "true"

    labels = [BUILDING_LABELS[b] for b in BUILDINGS]
    if_only, zs_only, both_flags = [], [], []
    for b in BUILDINGS:
        df = v3[v3["building_id"] == b]
        both = int((df["is_anomaly_iforest"] & df["is_anomaly_zscore"]).sum())
        if_o = int((df["is_anomaly_iforest"] & ~df["is_anomaly_zscore"]).sum())
        zs_o = int((~df["is_anomaly_iforest"] & df["is_anomaly_zscore"]).sum())
        if_only.append(if_o)
        zs_only.append(zs_o)
        both_flags.append(both)

    x = np.arange(len(BUILDINGS))
    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    p1 = ax.bar(x, if_only, width, label="IForest only", color=C_V3_IF, alpha=0.85)
    p2 = ax.bar(x, zs_only, width, bottom=if_only, label="Z-score only", color=C_ZSCORE, alpha=0.85)
    p3 = ax.bar(
        x, both_flags, width,
        bottom=[i + z for i, z in zip(if_only, zs_only)],
        label="Both flagged", color="#2CA02C", alpha=0.85,
    )

    ax.bar_label(p1, label_type="center", fontsize=8, fmt="%d")
    ax.bar_label(p2, label_type="center", fontsize=8, fmt="%d")
    ax.bar_label(p3, label_type="center", fontsize=8, fmt="%d")

    totals = [i + z + b for i, z, b in zip(if_only, zs_only, both_flags)]
    for xi, total in enumerate(totals):
        ax.text(xi, total + 2, str(total), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Flagged readings (evaluation set)")
    ax.set_title(
        "V3 Flag Attribution — Isolation Forest vs Z-score Contribution\n"
        "(contamination=0.02, z-score threshold=3.5σ)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linewidth=0.5)

    plt.tight_layout()
    out = FIGURES_DIR / "flag_rate_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    _set_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading prediction files...")
    v1, v2, v3 = _load_predictions()

    print("\n[1/4] Per-building time series plots...")
    plot_per_building(v1, v2, v3)

    print("\n[2/4] V1 vs V2 comparison bar chart...")
    plot_v1_v2_comparison(v1, v2)

    print("\n[3/4] V3 anomaly score distributions...")
    plot_score_distributions(v3)

    print("\n[4/4] V3 flag rate summary (IForest vs z-score)...")
    plot_flag_rate_summary(v3)

    saved = list(FIGURES_DIR.glob("*.png"))
    print(f"\nDone. {len(saved)} figures in {FIGURES_DIR}")
    for f in sorted(saved):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
