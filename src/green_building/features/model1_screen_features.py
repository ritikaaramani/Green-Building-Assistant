"""
Model 1 - Building Efficiency Assessment
Step 12b: Screen extra candidate features from the cleaned CBECS data.

Why: Experiment 1 used 26 hand-picked features (macro-F1 ~0.53).
The cleaned data still has ~500 building-characteristic columns
(equipment, lighting, controls, fuels used ...). This ranks them.

Rules:
  - Uses TRAIN buildings only (same saved split) -> test set stays untouched.
  - Energy use / cost columns were removed in Step 8 and are re-checked here.
  - Nothing in the dataset is changed - this only writes a report.

Run from the repo root:
    python src/green_building/features/model1_screen_features.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

REPO_ROOT = Path(__file__).resolve().parents[3]
CLEAN_CSV = REPO_ROOT / "data" / "interim" / "model1" / "cbecs2018_clean.csv"
CODEBOOK = REPO_ROOT / "data" / "raw" / "model1_cbecs2018" / "2018microdata_codebook.xlsx"
DATASET = REPO_ROOT / "data" / "processed" / "model1" / "model1_dataset.csv"
SPEC = REPO_ROOT / "data" / "processed" / "model1" / "feature_spec.json"
SPLIT = REPO_ROOT / "reports" / "model1" / "models" / "train_test_split.csv"
OUT_CSV = REPO_ROOT / "reports" / "model1" / "features" / "candidate_features.csv"

NOT_CANDIDATES = {"PUBID", "FINALWT", "MFBTU"}   # id, survey weight, label input
LEAK_WORDS = ["consumption", "expenditure", "thous btu", "kwh", "cubic feet", "gallons", "cost"]


def codebook_labels():
    """Return {variable: label} from the official codebook."""
    raw = pd.read_excel(CODEBOOK, header=None, dtype=str)
    clean = raw.map(lambda c: " ".join(str(c).split()) if pd.notna(c) else c)
    hdr = clean.index[clean.apply(lambda r: "Variable name" in r.values, axis=1)][0]
    head = clean.loc[hdr].tolist()
    body = clean.loc[hdr + 1:]
    return dict(zip(body[head.index("Variable name")], body[head.index("Label")]))


def main():
    spec = json.loads(SPEC.read_text())
    current = set(spec["categorical_features"] + spec["numeric_features"])
    labels = codebook_labels()

    clean = pd.read_csv(CLEAN_CSV, low_memory=False)
    target = pd.read_csv(DATASET, usecols=["PUBID", "EUI", "EFFICIENCY_CLASS"])
    split = pd.read_csv(SPLIT)
    train_ids = set(split.loc[split["split"] == "train", "PUBID"])

    df = clean.merge(target, on="PUBID", how="inner")
    df = df[df["PUBID"].isin(train_ids)]
    print(f"Training buildings used for screening: {len(df)}")

    cols = [c for c in clean.columns
            if c not in NOT_CANDIDATES
            and pd.api.types.is_numeric_dtype(clean[c])
            and not any(w in str(labels.get(c, "")).lower() for w in LEAK_WORDS)]

    X = df[cols].fillna(-1)                     # -1 = "not applicable" (only for scoring)
    y = df["EFFICIENCY_CLASS"]
    n_unique = df[cols].nunique()
    discrete = (n_unique <= 20).values           # few values -> treat as categories
    print(f"Scoring {len(cols)} columns (about 1 minute)...")
    mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=42)
    rho = df[cols].corrwith(df["EUI"], method="spearman")

    out = pd.DataFrame({
        "column": cols,
        "in_model_v1": ["yes" if c in current else "" for c in cols],
        "label": [labels.get(c, "") for c in cols],
        "missing_pct": (df[cols].isna().mean() * 100).round(1).values,
        "n_unique": n_unique.values,
        "mutual_info": mi.round(4),
        "spearman_EUI": rho.round(3).values,
    }).sort_values("mutual_info", ascending=False)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 220)
    show = out.head(40).copy()
    show["label"] = show["label"].astype(str).str.slice(0, 55)
    print("\nTop 40 columns by mutual information with the efficiency class (train only):")
    print(show.to_string(index=False))
    print(f"\nColumns already in model v1 inside the top 40: "
          f"{(show['in_model_v1'] == 'yes').sum()}")
    print(f"Saved full ranking: {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()