"""
Model 1 - Building Efficiency Assessment
Step 9: Feature engineering + efficiency labels.
Step 12c (Experiment 2): extra features chosen from the screening in Step 12b.
Option C: binary target = clearly High vs clearly Low efficiency (within building type).

Input  : data/interim/model1/cbecs2018_clean.csv
Outputs: data/processed/model1/model1_dataset.csv   (model-ready)
         data/processed/model1/feature_spec.json    (which columns are features/target)
         reports/model1/features/                   (thresholds, class balance, report)
         docs/model1_data_dictionary.md             (every feature + target explained)

LABEL:
  EUI = MFBTU (kBtu) / SQFT (sq ft)  ->  kBtu per sq ft per year
  Within each building type (PBA), buildings are split into thirds by EUI,
  using survey weights (FINALWT):
     EFFICIENCY_CLASS: 0 = High, 1 = Moderate, 2 = Low efficiency (kept for reference)
  OPTION C target (EFFICIENCY_BINARY):
     0 = High efficiency (lowest third), 1 = Low efficiency (highest third),
     blank = Moderate (middle third) -> excluded from training and testing.
  This is a RELATIVE ranking against similar CBECS 2018 buildings,
  NOT an official certification or ENERGY STAR score.

FEATURES: building characteristics only. No energy use or cost column is
allowed as a feature (checked automatically).

Run from the repo root:
    python src/green_building/features/model1_build_features.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
IN_CSV = REPO_ROOT / "data" / "interim" / "model1" / "cbecs2018_clean.csv"
CODEBOOK = REPO_ROOT / "data" / "raw" / "model1_cbecs2018" / "2018microdata_codebook.xlsx"
OUT_DIR = REPO_ROOT / "data" / "processed" / "model1"
REPORT_DIR = REPO_ROOT / "reports" / "model1" / "features"
DICT_MD = REPO_ROOT / "docs" / "model1_data_dictionary.md"

LABEL_NAMES = {0: "High efficiency", 1: "Moderate efficiency", 2: "Low efficiency"}
BINARY_NAMES = {0: "High efficiency", 1: "Low efficiency"}          # OPTION C

# Candidate features (building characteristics). Any not found is reported and skipped.
CATEGORICAL = ["PBA", "PUBCLIM", "REGION", "CENDIV", "WLCNS", "RFCNS", "BLDSHP",
               "OWNTYPE", "MAINHT", "MAINCL", "RENOV",
               # Exp 2: detailed type, fuels, controls/maintenance
               "PBAPLUS", "NGUSED", "NGWATR", "EMCS", "MAINT", "VAV"]
NUMERIC = ["SQFT", "NFLOOR", "YRCONC", "WKHRS", "NWKER", "GLSSPC", "FLCEILHT",
           "NELVTR", "BASEMNT", "HEATP", "COOLP", "LTOHRP",
           # Exp 2: off-hours lighting, climate, equipment
           "LTNHRP", "HDD65", "CDD65", "PCTERMN", "LAPTPN",
           "RFGOPN", "RFGWIN", "RFGVNN", "FDPREPSF"]
# Engineered features created below
ENGINEERED = ["LOG_SQFT", "WORKERS_PER_1000SQFT", "OPEN_24_7",
              "PCS_PER_1000SQFT", "KITCHEN_SHARE"]

# Names that signal energy use/cost -> must never be features
ALLOWED_LOOKALIKES = {"WLCNS", "RFCNS"}   # wall/roof construction material (codebook)

lines = []


def report(text=""):
    print(text)
    lines.append(str(text))


def weighted_quantile(values, weights, q):
    """Quantile q (0-1) of values, where each value counts as many times as its weight."""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cum, v))


def codebook_info():
    """Return ({variable: label}, {variable: values/codes text}) from the codebook."""
    raw = pd.read_excel(CODEBOOK, header=None, dtype=str)
    clean = raw.map(lambda c: " ".join(str(c).split()) if pd.notna(c) else c)
    hdr = clean.index[clean.apply(lambda r: "Variable name" in r.values, axis=1)][0]
    head = clean.loc[hdr].tolist()
    body = clean.loc[hdr + 1:]
    names = body[head.index("Variable name")]
    labels = dict(zip(names, body[head.index("Label")]))
    values = dict(zip(names, body[head.index("Values/Format codes")].astype(str)))
    return labels, values


def has_special_codes(values_text):
    """True if the codebook lists 99x codes (e.g. '995=More than 30') for this variable."""
    t = str(values_text).replace(" ", "")
    return any(f"{k}=" in t for k in (994, 995, 996, 997))


def looks_like_energy(col):
    return col not in ALLOWED_LOOKALIKES and col.endswith(("BTU", "CNS", "EXP"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV, low_memory=False)
    labels, values = codebook_info()

    report("=" * 60)
    report("MODEL 1 - FEATURES + LABELS  (Experiment 2 features, Option C target)")
    report("=" * 60)
    report(f"Input: {len(df)} buildings x {df.shape[1]} columns")

    leftover = [c for c in df.columns if looks_like_energy(c) and c != "MFBTU"]
    report(f"\nEnergy-looking columns still in clean data (besides MFBTU): {leftover or 'none'}")

    needed = ["PUBID", "PBA", "SQFT", "MFBTU", "FINALWT"] + [
        c for c in CATEGORICAL + NUMERIC if c in df.columns]
    df = df[list(dict.fromkeys(needed))].copy()

    # ---------- 1. TARGET ----------
    df["EUI"] = df["MFBTU"] / df["SQFT"]
    thresholds = []
    df["EFFICIENCY_CLASS"] = -1
    for pba, g in df.groupby("PBA"):
        t1 = weighted_quantile(g["EUI"], g["FINALWT"], 1 / 3)
        t2 = weighted_quantile(g["EUI"], g["FINALWT"], 2 / 3)
        cls = np.where(g["EUI"] <= t1, 0, np.where(g["EUI"] <= t2, 1, 2))
        df.loc[g.index, "EFFICIENCY_CLASS"] = cls
        thresholds.append({"PBA": pba, "n": len(g), "high_if_EUI_at_most": round(t1, 2),
                           "low_if_EUI_above": round(t2, 2)})
    df["EFFICIENCY_LABEL"] = df["EFFICIENCY_CLASS"].map(LABEL_NAMES)
    # OPTION C: 0 = High, 1 = Low, blank (NaN) = Moderate -> left out of train/test
    df["EFFICIENCY_BINARY"] = df["EFFICIENCY_CLASS"].map({0: 0, 2: 1})
    thr = pd.DataFrame(thresholds)
    thr.to_csv(REPORT_DIR / "label_thresholds_by_type.csv", index=False)
    report("\n--- Label thresholds per building type (kBtu/sqft/yr, weighted tertiles) ---")
    report(thr.to_string(index=False))

    # ---------- 2. FEATURES ----------
    cat = [c for c in CATEGORICAL if c in df.columns]
    num = [c for c in NUMERIC if c in df.columns]
    missing = [c for c in CATEGORICAL + NUMERIC if c not in df.columns]
    report(f"\nCandidate features not found in data (skipped): {missing or 'none'}")

    # 99x special codes -> top-coded category (max real value + 1).
    # ONLY for columns whose codebook actually lists 99x codes.
    for c in num:
        if not has_special_codes(values.get(c, "")):
            continue
        codes = df[c] >= 994
        if codes.any():
            top = df.loc[~codes, c].max() + 1
            report(f"  {c}: {codes.sum()} special-code values (>=994) -> top-coded as {top}")
            df.loc[codes, c] = top

    df["LOG_SQFT"] = np.log10(df["SQFT"])
    df["WORKERS_PER_1000SQFT"] = df["NWKER"] / df["SQFT"] * 1000
    df["OPEN_24_7"] = (df["WKHRS"] >= 168).astype(int)
    # Exp 2: computer density and kitchen share of floor area
    df["PCS_PER_1000SQFT"] = (df[["PCTERMN", "LAPTPN"]].sum(axis=1, min_count=1)
                              / df["SQFT"] * 1000)
    df["KITCHEN_SHARE"] = df["FDPREPSF"] / df["SQFT"]
    num = num + ENGINEERED
    features = cat + num

    # ---------- 3. LEAKAGE GUARDS ----------
    forbidden = {"MFBTU", "EUI", "EFFICIENCY_CLASS", "EFFICIENCY_LABEL",
                 "EFFICIENCY_BINARY", "FINALWT"}                       # OPTION C: + binary
    bad = [f for f in features if f in forbidden or looks_like_energy(f)]
    if bad:
        raise SystemExit(f"LEAKAGE: these features are not allowed: {bad}")
    report("\nLeakage guard: PASS (no energy use/cost, EUI, label or weight in features)")

    # ---------- 4. SAVE ----------
    keep = ["PUBID", "FINALWT", "EUI", "EFFICIENCY_CLASS", "EFFICIENCY_LABEL",
            "EFFICIENCY_BINARY"] + features                              # OPTION C
    out = df[keep]
    out.to_csv(OUT_DIR / "model1_dataset.csv", index=False)
    spec = {"id": "PUBID", "weight": "FINALWT",
            "target": "EFFICIENCY_BINARY",                               # OPTION C
            "target_names": BINARY_NAMES,                                # OPTION C
            "categorical_features": cat, "numeric_features": num,
            "not_features": ["EUI", "EFFICIENCY_CLASS", "EFFICIENCY_LABEL",
                             "EFFICIENCY_BINARY", "FINALWT", "PUBID"],
            "label_method": "Option C: weighted EUI tertiles within building type (PBA); "
                            "lowest third = High (0), highest third = Low (1), "
                            "middle third (Moderate) excluded",
            "feature_set": "experiment_2"}
    (OUT_DIR / "feature_spec.json").write_text(json.dumps(spec, indent=2))

    # ---------- 5. CHECKS + CLASS BALANCE ----------
    report("\n--- 3-class distribution (reference) ---")
    dist = out["EFFICIENCY_LABEL"].value_counts()
    report((dist.astype(str) + "  (" + (dist / len(out) * 100).round(1).astype(str) + "%)").to_string())
    report("\n--- OPTION C target (buildings used for training/testing) ---")
    bdist = out["EFFICIENCY_BINARY"].map(BINARY_NAMES).value_counts()
    report(bdist.to_string())
    report(f"Moderate buildings excluded: {out['EFFICIENCY_BINARY'].isna().sum()}")
    report("\n--- Missing values in features (%, blanks = 'not applicable') ---")
    miss = (out[features].isna().mean() * 100).round(1)
    report(miss[miss > 0].sort_values(ascending=False).to_string() if (miss > 0).any() else "none")

    report("\n--- Numeric feature ranges (check for odd codes) ---")
    report(out[num].describe().T[["count", "min", "50%", "max"]].round(2).to_string())

    checks = {
        "every building labelled": (out["EFFICIENCY_CLASS"] >= 0).all(),
        "3 classes present": out["EFFICIENCY_CLASS"].nunique() == 3,
        "binary target has 2 classes": out["EFFICIENCY_BINARY"].nunique() == 2,     # OPTION C
        "binary blank only for Moderate": (out["EFFICIENCY_BINARY"].isna()
                                           == (out["EFFICIENCY_CLASS"] == 1)).all(),  # OPTION C
        "no infinite values": not np.isinf(out.select_dtypes("number")).any().any(),
        "row count unchanged": len(out) == len(df),
    }
    report("\n--- Validation ---")
    for k, ok in checks.items():
        report(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit("Validation FAILED")

    # ---------- 6. DATA DICTIONARY ----------
    eng_desc = {"LOG_SQFT": "log10 of floor area (reduces skew)",
                "WORKERS_PER_1000SQFT": "employees on main shift per 1,000 sq ft (occupancy density)",
                "OPEN_24_7": "1 if open 168 hours/week, else 0",
                "PCS_PER_1000SQFT": "desktop + laptop computers per 1,000 sq ft",
                "KITCHEN_SHARE": "commercial kitchen area / total floor area"}
    moderate_pct = round(out["EFFICIENCY_BINARY"].isna().mean() * 100, 1)
    rows = ["| Column | Role | Type | Meaning | Missing % |", "|---|---|---|---|---|"]
    rows.append(f"| EFFICIENCY_BINARY | **Target (Option C)** | binary | 0 = High efficiency, "
                f"1 = Low efficiency; blank = Moderate (excluded) | {moderate_pct} |")
    rows.append("| EFFICIENCY_CLASS | Label source only (NOT a feature) | categorical | 0 = High, "
                "1 = Moderate, 2 = Low (weighted EUI tertile within building type) | 0 |")
    rows.append("| EUI | Label source only (NOT a feature) | numeric | MFBTU / SQFT, kBtu/sq ft/yr | 0 |")
    rows.append("| FINALWT | Weight only (NOT a feature) | numeric | CBECS survey weight | 0 |")
    for f in features:
        meaning = eng_desc.get(f, labels.get(f, ""))
        rows.append(f"| {f} | Feature | {'categorical' if f in cat else 'numeric'} | {meaning} | {miss[f]} |")
    DICT_MD.write_text("# Model 1 — Data Dictionary\n\n"
                       "Source: U.S. EIA, 2018 CBECS public-use microdata (Dec 2022 release).\n"
                       "Dataset: data/processed/model1/model1_dataset.csv "
                       "(Experiment 2 features, Option C target)\n\n" + "\n".join(rows) + "\n")

    report(f"\nFeatures: {len(features)} ({len(cat)} categorical, {len(num)} numeric)")
    report(f"Saved: data/processed/model1/model1_dataset.csv ({out.shape[0]} x {out.shape[1]})")
    report("Saved: data/processed/model1/feature_spec.json")
    report("Saved: docs/model1_data_dictionary.md")
    (REPORT_DIR / "features_report.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
