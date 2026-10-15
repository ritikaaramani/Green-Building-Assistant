"""
Model 1 - Building Efficiency Assessment
Step 8: Data cleaning for CBECS 2018.

Input  : data/raw/model1_cbecs2018/cbecs2018_final_public.csv   (never modified)
Output : data/interim/model1/cbecs2018_clean.csv
Reports: reports/model1/cleaning/

Every row or column removed is written to a report with its reason.
Missing values are NOT filled here: most blanks in CBECS mean
"not applicable" and are handled later, inside the model pipeline.

Run from the repo root:
    python src/green_building/data/model1_clean.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw" / "model1_cbecs2018"
RAW_CSV = RAW_DIR / "cbecs2018_final_public.csv"
CODEBOOK = RAW_DIR / "2018microdata_codebook.xlsx"
OUT_CSV = REPO_ROOT / "data" / "interim" / "model1" / "cbecs2018_clean.csv"
REPORT_DIR = REPO_ROOT / "reports" / "model1" / "cleaning"

# ---- Decisions agreed after EDA (Step 7) ----
EXCLUDE_PBA = {1: "D2: vacant building - efficiency not meaningful",
               11: "D4: refrigerated warehouse - only 23 buildings, too few to compare"}
LOW_EUI_THRESHOLD = 1.0   # D6: kBtu/sqft/yr; buildings below this are REPORTED for review
DROP_LOW_EUI = False      # D6: set to True only after reviewing low_eui_buildings.csv

# Columns needed later that must survive cleaning
KEEP_ALWAYS = ["PUBID", "PBA", "SQFT", "MFBTU", "FINALWT", "PUBCLIM"]

# Words in a codebook label that mean "energy used or paid for" -> leakage
LEAK_WORDS = ["consumption", "expenditure", "thous btu", "kwh", "cubic feet",
              "gallons", "cost"]

lines = []


def report(text=""):
    print(text)
    lines.append(str(text))


def load_codebook_labels():
    """Return {variable name: label} from the official codebook."""
    raw = pd.read_excel(CODEBOOK, header=None, dtype=str)
    clean = raw.map(lambda c: " ".join(str(c).split()) if pd.notna(c) else c)
    header_row = clean.index[clean.apply(lambda r: "Variable name" in r.values, axis=1)][0]
    header = clean.loc[header_row].tolist()
    name_col, label_col = header.index("Variable name"), header.index("Label")
    values_col = header.index("Values/Format codes") if "Values/Format codes" in header else None
    body = raw.loc[header_row + 1:]
    labels = dict(zip(body[name_col].astype(str).str.strip(), body[label_col].astype(str)))
    values = (dict(zip(body[name_col].astype(str).str.strip(), body[values_col].astype(str)))
              if values_col is not None else {})
    return labels, values


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV, low_memory=False)
    labels, values = load_codebook_labels()
    start_rows, start_cols = df.shape

    report("=" * 60)
    report("MODEL 1 - DATA CLEANING (CBECS 2018)")
    report("=" * 60)
    report(f"Start: {start_rows} rows x {start_cols} columns")

    excluded = []   # (PUBID, reason)
    dropped = []    # (column, reason, codebook label)

    # ---------- ROWS ----------
    # D1: no energy data -> target (EUI) cannot be computed. Never impute the target.
    mask = df["MFBTU"].isna()
    excluded += [(p, "D1: MFBTU missing ('Not applicable') - EUI cannot be computed")
                 for p in df.loc[mask, "PUBID"]]
    df = df[~mask]
    report(f"\nD1 removed {mask.sum()} buildings with no MFBTU")

    # D2 + D4: building types excluded from benchmarking
    for code, reason in EXCLUDE_PBA.items():
        mask = df["PBA"] == code
        excluded += [(p, reason) for p in df.loc[mask, "PUBID"]]
        df = df[~mask]
        report(f"{reason.split(':')[0]} removed {mask.sum()} buildings (PBA={code})")

    # D6: very low EUI - report for review, drop only if switched on
    eui = df["MFBTU"] / df["SQFT"]
    low = df.loc[eui < LOW_EUI_THRESHOLD, ["PUBID", "PBA", "SQFT", "WKHRS", "MFBTU"]].copy()
    low["EUI"] = eui[low.index].round(3)
    low.sort_values("EUI").to_csv(REPORT_DIR / "low_eui_buildings.csv", index=False)
    report(f"\nD6: {len(low)} buildings with EUI < {LOW_EUI_THRESHOLD} kBtu/sqft/yr "
           f"-> listed in low_eui_buildings.csv")
    if DROP_LOW_EUI:
        excluded += [(p, f"D6: EUI < {LOW_EUI_THRESHOLD}") for p in low["PUBID"]]
        df = df[~df["PUBID"].isin(low["PUBID"])]
        report(f"D6 removed {len(low)} buildings")
    else:
        report("D6: kept for now (DROP_LOW_EUI = False) - review the list first")

    # ---------- COLUMNS ----------
    # Replicate weights: only for survey error estimates, not building features
    rep_w = [c for c in df.columns if c.startswith("FINALWT") and c != "FINALWT"]
    dropped += [(c, "replicate survey weight (not a building feature)", labels.get(c, "")) for c in rep_w]

    # Imputation flags: ZXXX tells whether XXX was imputed - metadata, not a feature
    z_flags = [c for c in df.columns if c.startswith("Z") and c[1:] in df.columns]
    dropped += [(c, "imputation flag (metadata, not a feature)", labels.get(c, "")) for c in z_flags]

    # Leakage: energy used or paid for (except MFBTU, kept only to build the label)
    leak = [c for c in df.columns if c not in KEEP_ALWAYS and c not in rep_w and c not in z_flags
            and any(w in labels.get(c, "").lower() for w in LEAK_WORDS)]
    dropped += [(c, "LEAKAGE: energy consumption/expenditure", labels.get(c, "")) for c in leak]

    # Completely empty columns (after row removals)
    done = set(rep_w) | set(z_flags) | set(leak)
    empty = [c for c in df.columns if c not in done and c not in KEEP_ALWAYS and df[c].isna().all()]
    dropped += [(c, "100% empty", labels.get(c, "")) for c in empty]

    # Constant columns (one value only -> no information)
    done |= set(empty)
    const = [c for c in df.columns if c not in done and c not in KEEP_ALWAYS
             and df[c].nunique(dropna=False) == 1]
    dropped += [(c, "constant (single value)", labels.get(c, "")) for c in const]

    drop_cols = [c for c, _, _ in dropped]
    df = df.drop(columns=drop_cols)
    report(f"\nColumns dropped: {len(drop_cols)}")
    for reason in ["replicate", "imputation", "LEAKAGE", "100% empty", "constant"]:
        n = sum(1 for _, r, _ in dropped if r.startswith(reason))
        report(f"  {reason:12s}: {n}")

    # ---------- SPECIAL CODES ----------
    # NFLOOR: 994 = 10-14 floors, 995 = 15+ floors (codebook) -> ordered scale
    n_codes = df["NFLOOR"].isin([994, 995]).sum()
    df["NFLOOR"] = df["NFLOOR"].replace({994: 10, 995: 11})
    report(f"\nNFLOOR: recoded {n_codes} special codes (994->10 '10-14 floors', 995->11 '15+ floors')")

    # Other columns whose codebook lists 99x special codes - reported, handled in Step 9
    special = [c for c in df.columns if c != "NFLOOR"
               and any(f"{k}=" in str(values.get(c, "")).replace(" ", "") for k in (994, 995, 996, 997))]
    report(f"Other columns with 99x special codes (handle in feature step): {len(special)}")
    report("  " + ", ".join(special[:30]) + (" ..." if len(special) > 30 else ""))

    # ---------- VALIDATION ----------
    report("\n--- Validation checks ---")
    checks = {
        "no missing MFBTU": df["MFBTU"].notna().all(),
        "no vacant / refrigerated warehouse": not df["PBA"].isin(EXCLUDE_PBA.keys()).any(),
        "SQFT > 0 everywhere": (df["SQFT"] > 0).all(),
        "PUBID unique": df["PUBID"].is_unique,
        "no NFLOOR special codes": not df["NFLOOR"].isin([994, 995]).any(),
        "no infinite values": not np.isinf(df.select_dtypes("number")).any().any(),
        "rows add up": len(df) + len(excluded) == start_rows,
        "no leakage columns left": not any(c in df.columns for c in leak),
    }
    for name, ok in checks.items():
        report(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise SystemExit("A validation check FAILED - clean file NOT written.")

    # ---------- SAVE ----------
    df.to_csv(OUT_CSV, index=False)
    pd.DataFrame(excluded, columns=["PUBID", "reason"]).to_csv(REPORT_DIR / "excluded_buildings.csv", index=False)
    pd.DataFrame(dropped, columns=["column", "reason", "codebook_label"]).to_csv(
        REPORT_DIR / "dropped_columns.csv", index=False)

    report(f"\nEnd: {df.shape[0]} rows x {df.shape[1]} columns")
    report(f"Buildings per type after cleaning:")
    report(df["PBA"].value_counts().sort_index().to_string())
    report(f"\nSaved: {OUT_CSV.relative_to(REPO_ROOT)}")
    (REPORT_DIR / "cleaning_report.txt").write_text("\n".join(lines))
    print(f"Saved reports in: {REPORT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()