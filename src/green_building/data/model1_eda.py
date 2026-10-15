"""
Model 1 - Building Efficiency Assessment
Step 7: Exploratory Data Analysis (EDA) on CBECS 2018.

Read-only: the raw file is NOT modified. EUI is calculated here only
to explore it; the labelled dataset is created later, in its own step.

Run from the repo root:
    python src/green_building/data/model1_eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = REPO_ROOT / "data" / "raw" / "model1_cbecs2018" / "cbecs2018_final_public.csv"
FIG_DIR = REPO_ROOT / "reports" / "model1" / "figures"
EDA_DIR = REPO_ROOT / "reports" / "model1" / "eda"

# Code meanings confirmed from the official codebook (Step 6)
PBA_NAMES = {1: "Vacant", 2: "Office", 4: "Laboratory", 5: "Warehouse (non-refrig.)",
             6: "Food sales", 7: "Public order/safety", 8: "Outpatient health",
             11: "Refrigerated warehouse", 12: "Religious worship", 13: "Public assembly",
             14: "Education", 15: "Food service", 16: "Inpatient health", 17: "Nursing",
             18: "Lodging", 23: "Strip shopping center", 24: "Enclosed mall",
             25: "Retail (non-mall)", 26: "Service", 91: "Other"}
CLIM_NAMES = {1: "Cold/very cold", 2: "Cool", 3: "Mixed mild", 4: "Warm",
              5: "Hot/very hot", 7: "Withheld"}

COLOR = "#2f6db5"      # one hue for all charts (no rainbow)
MIN_PEERS = 30         # a peer group smaller than this is too small to compare fairly

lines = []


def report(text=""):
    print(text)
    lines.append(str(text))


def weighted_median(values, weights):
    """Median that uses survey weights, so it represents all U.S. buildings."""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cum = np.cumsum(w)
    return v[np.searchsorted(cum, cum[-1] / 2)]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    report(f"  saved figure: reports/model1/figures/{name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    cols = ["PUBID", "PBA", "PUBCLIM", "SQFT", "MFBTU", "FINALWT",
            "YRCONC", "WKHRS", "NWKER", "NFLOOR"]
    df = pd.read_csv(RAW_CSV, usecols=cols)
    df["TYPE"] = df["PBA"].map(PBA_NAMES)
    df["CLIMATE"] = df["PUBCLIM"].map(CLIM_NAMES)

    report("=" * 60)
    report("MODEL 1 - EDA (CBECS 2018)")
    report("=" * 60)
    report(f"Buildings loaded: {len(df)}")

    # 1. Who are the 79 buildings with no energy data?
    report("\n--- 1. Buildings with missing MFBTU, by type ---")
    missing = df[df["MFBTU"].isna()]
    report(f"Total missing: {len(missing)}")
    report(missing["TYPE"].value_counts().to_string())

    # 2. EUI = annual energy (kBtu) / floor area (sq ft)
    e = df[df["MFBTU"].notna()].copy()
    e["EUI"] = e["MFBTU"] / e["SQFT"]
    report("\n--- 2. EUI (kBtu per sq ft per year), buildings with energy data ---")
    report(f"Buildings: {len(e)}")
    q = e["EUI"].quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
    for k, v in q.items():
        report(f"  {int(k * 100):3d}th percentile: {v:10.2f}")
    report(f"  Weighted median (all U.S. buildings): {weighted_median(e['EUI'], e['FINALWT']):.2f}")

    # 3. EUI by building type
    report("\n--- 3. EUI by building type ---")
    by_type = (e.groupby("TYPE")
                 .apply(lambda g: pd.Series({
                     "n": len(g),
                     "median_EUI": g["EUI"].median(),
                     "weighted_median_EUI": weighted_median(g["EUI"], g["FINALWT"]),
                     "p25": g["EUI"].quantile(0.25),
                     "p75": g["EUI"].quantile(0.75)}), include_groups=False)
                 .sort_values("median_EUI"))
    by_type["n"] = by_type["n"].astype(int)
    report(by_type.round(1).to_string())
    by_type.round(2).to_csv(EDA_DIR / "eui_by_building_type.csv")

    # 4. EUI by climate
    report("\n--- 4. EUI by climate zone ---")
    by_clim = e.groupby("CLIMATE")["EUI"].agg(["count", "median"]).round(1)
    report(by_clim.to_string())

    # 5. Peer-group sizes (building type x climate)
    report(f"\n--- 5. Peer groups (type x climate); small = fewer than {MIN_PEERS} buildings ---")
    peers = e.groupby(["TYPE", "CLIMATE"]).size().rename("n").reset_index()
    small = peers[peers["n"] < MIN_PEERS]
    report(f"Total peer groups: {len(peers)}")
    report(f"Small groups: {len(small)} covering {small['n'].sum()} buildings")
    report(f"Building types with < {MIN_PEERS} buildings in total: "
           f"{(e['TYPE'].value_counts() < MIN_PEERS).sum()}")
    peers.to_csv(EDA_DIR / "peer_group_sizes.csv", index=False)

    # 6. Buildings open 0 hours a week
    report("\n--- 6. Buildings with WKHRS = 0 (not open), by type ---")
    report(e.loc[e["WKHRS"] == 0, "TYPE"].value_counts().to_string() or "none")

    # 7. Outliers (kept for now - only reported)
    lo, hi = e["EUI"].quantile([0.01, 0.99])
    report(f"\n--- 7. Extreme EUI (below 1st pct {lo:.1f} or above 99th pct {hi:.1f}) ---")
    report(f"Below 1st percentile: {(e['EUI'] < lo).sum()}   Above 99th percentile: {(e['EUI'] > hi).sum()}")
    report("Top 10 highest EUI:")
    report(e.nlargest(10, "EUI")[["PUBID", "TYPE", "SQFT", "WKHRS", "EUI"]].round(1).to_string(index=False))

    # 8. Which characteristics move with EUI? (Spearman = rank correlation, robust to skew)
    report("\n--- 8. Spearman correlation with EUI ---")
    e["NFLOOR_ORD"] = e["NFLOOR"].replace({994: 10, 995: 11})  # 10 = 10-14 floors, 11 = 15+
    for c in ["WKHRS", "NWKER", "SQFT", "YRCONC", "NFLOOR_ORD"]:
        report(f"  {c:11s}: {e['EUI'].corr(e[c], method='spearman'):+.3f}")

    # ---------------- Figures ----------------
    report("\n--- Figures ---")

    counts = df["TYPE"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(counts.index, counts.values, color=COLOR, height=0.7)
    style(ax, "Buildings per type (sample counts)", "Number of buildings", "")
    save(fig, "fig1_building_types.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["SQFT"], bins=np.logspace(3, np.log10(df["SQFT"].max()), 40), color=COLOR)
    ax.set_xscale("log")
    style(ax, "Floor area distribution", "Floor area (sq ft, log scale)", "Buildings")
    save(fig, "fig2_floor_area.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(e["EUI"], bins=np.logspace(np.log10(e["EUI"].min()), np.log10(e["EUI"].max()), 50), color=COLOR)
    ax.set_xscale("log")
    style(ax, "EUI distribution", "EUI (kBtu / sq ft / year, log scale)", "Buildings")
    save(fig, "fig3_eui_distribution.png")

    order = by_type.index.tolist()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.boxplot([e.loc[e["TYPE"] == t, "EUI"] for t in order], orientation="horizontal", tick_labels=order,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#c9daf0", edgecolor=COLOR),
               medianprops=dict(color=COLOR, linewidth=2))
    ax.set_xscale("log")
    style(ax, "EUI by building type (outliers hidden)", "EUI (kBtu / sq ft / year, log scale)", "")
    save(fig, "fig4_eui_by_type.png")

    corder = [CLIM_NAMES[k] for k in sorted(CLIM_NAMES) if CLIM_NAMES[k] in set(e["CLIMATE"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([e.loc[e["CLIMATE"] == c, "EUI"] for c in corder], tick_labels=corder,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#c9daf0", edgecolor=COLOR),
               medianprops=dict(color=COLOR, linewidth=2))
    ax.set_yscale("log")
    style(ax, "EUI by climate zone (outliers hidden)", "", "EUI (log scale)")
    save(fig, "fig5_eui_by_climate.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(e["WKHRS"], e["EUI"], s=8, alpha=0.3, color=COLOR, linewidths=0)
    ax.set_yscale("log")
    style(ax, "EUI vs weekly operating hours", "Hours open per week", "EUI (log scale)")
    save(fig, "fig6_eui_vs_hours.png")

    (EDA_DIR / "eda_summary.txt").write_text("\n".join(lines))
    print("\nSaved: reports/model1/eda/eda_summary.txt")


if __name__ == "__main__":
    main()
