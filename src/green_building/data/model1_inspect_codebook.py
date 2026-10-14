"""
Model 1 - Building Efficiency Assessment
Step 6: Inspect the official CBECS 2018 codebook.

Purpose: confirm what the codes in our key columns mean
(e.g. PBA building types, PUBCLIM = 7, NFLOOR = 994/995)
instead of guessing. Read-only: the codebook is not modified.

Run from the repo root:
    python src/green_building/data/model1_inspect_codebook.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CODEBOOK = REPO_ROOT / "data" / "raw" / "model1_cbecs2018" / "2018microdata_codebook.xlsx"
OUT_FILE = REPO_ROOT / "reports" / "model1" / "validation" / "codebook_lookup.txt"

# Variables whose meaning/codes we need to confirm
LOOKUP = ["PBA", "PUBCLIM", "NFLOOR", "YRCONC", "WKHRS", "NWKER", "SQFT",
          "MFBTU", "ELCNS", "FINALWT", "WLCNS", "RFCNS"]

lines = []


def report(text=""):
    print(text)
    lines.append(str(text))


def main():
    if not CODEBOOK.exists():
        raise FileNotFoundError(f"Codebook not found: {CODEBOOK}")

    sheets = pd.read_excel(CODEBOOK, sheet_name=None, header=None, dtype=str)

    # 1. Show the codebook's layout so we understand its structure
    report("=" * 60)
    report("CBECS 2018 CODEBOOK - STRUCTURE")
    report("=" * 60)
    for name, sheet in sheets.items():
        report(f"\nSheet '{name}': {sheet.shape[0]} rows x {sheet.shape[1]} columns")
        report("First 4 rows:")
        for _, row in sheet.head(4).iterrows():
            cells = [str(c).strip() for c in row if pd.notna(c)]
            report("  " + " | ".join(cells)[:300])

    # 2. Find each variable we care about and print its full entry
    report("\n" + "=" * 60)
    report("LOOKUP OF KEY VARIABLES")
    report("=" * 60)
    for var in LOOKUP:
        found = False
        for name, sheet in sheets.items():
            # a row matches if any cell is exactly the variable name
            mask = sheet.apply(lambda r: any(str(c).strip() == var for c in r), axis=1)
            for _, row in sheet[mask].iterrows():
                cells = [str(c).strip() for c in row if pd.notna(c)]
                report(f"\n[{var}] (sheet '{name}')")
                for cell in cells:
                    report("   " + cell.replace("\n", "\n   "))
                found = True
        if not found:
            report(f"\n[{var}] NOT FOUND in codebook")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(lines))
    print(f"\nSaved: {OUT_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()