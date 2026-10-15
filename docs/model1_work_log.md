# Model 1 — Building Efficiency Assessment: Work Log

- **Owner:** Ansh Sai Santra
- **Branch:** model1-building-efficiency
- **Model:** XGBoost Classifier (compared against a Logistic Regression baseline)
- **Goal:** Classify a building's energy efficiency as High / Moderate / Low relative to comparable buildings.

---

## Planned pipeline

1. Dataset selection
2. Download and record dataset
3. Store and version raw data
4. Validate raw data
5. EDA
6. Cleaning
7. Feature engineering and label definition
8. Model-ready dataset (versioned separately)
9. Baseline model: Logistic Regression
10. Main model: XGBoost Classifier
11. Compare baseline vs XGBoost
12. Experiment tracking, final documentation, push

---

## Step 1 — Dataset selection

- **Date:** 2026-10-14
- **Status:** Done
- **Objective:** Choose a dataset suitable for building-level efficiency classification.
- **Candidates considered:**
  - Building Data Genome Project 2 (BDG2): hourly meter data, 1,636 buildings, 19 sites.
  - CBECS 2018 (U.S. EIA): 6,436 surveyed commercial buildings with annual energy use.
- **Decision:** CBECS 2018 public-use microdata.
- **Justification:**
  - It has one row per building, with annual total energy, floor area, building type, climate zone and operating characteristics.
  - BDG2 is mainly electricity-only for most buildings and is missing many building characteristics.
  - CBECS is public domain (EIA asks for a source credit).
- **Files to download:**
  - cbecs2018_final_public.csv
  - 2018microdata_codebook.xlsx
  - Users Guide to the 2018 CBECS Public Use Microdata File.pdf
- **Official page:** https://www.eia.gov/consumption/commercial/data/2018/index.php?view=microdata
- **Next step:** Create working branch and log file.

---

## Step 2 — Create working branch and work log

- **Date:** 2026-10-14
- **Status:** Done
- **Objective:** Keep all Model 1 work separate from teammates' work.
- **Commands executed:**
  git checkout main
  git pull origin main
  git checkout -b model1-building-efficiency
- **Files created:** docs/model1_work_log.md
- **Output observed:** Push succeeded. New branch `model1-building-efficiency` created on origin and set to track `origin/model1-building-efficiency`.
- **Git commit hash:** 793f4b7
- **Next step:** Download CBECS 2018 files and place them in the raw data folder.

---

## Step 3 — Obtain and verify raw dataset

- **Date:** 2026-10-14
- **Status:** Done
- **Objective:** Store the official CBECS 2018 files unchanged in a dedicated raw folder and record checksums.
- **Source:** U.S. Energy Information Administration (EIA), 2018 CBECS Public Use Microdata
  - Page: https://www.eia.gov/consumption/commercial/data/2018/index.php?view=microdata
  - Release: Nov 2021; CSV and codebook revised Dec 2022; User's Guide revised Jan 2023
  - Licence: Public domain; attribution requested ("Source: U.S. Energy Information Administration, 2018 CBECS")
- **Download date:** 2026-10-14
- **Location:** data/raw/model1_cbecs2018/
- **Dataset decision:** CBECS 2018 alone is sufficient. BDG2 not used (hourly, mostly electricity-only meter data from 19 sites, no labels, sparse building characteristics). Baseline (Logistic Regression) and XGBoost will both use CBECS.
- **Commands executed:**
  mkdir -p data/raw/model1_cbecs2018
  mv ~/Downloads/cbecs2018_final_public.csv .
  mv ~/Downloads/2018microdata_codebook.xlsx .
  curl -L -o cbecs2018_users_guide.pdf "<EIA User's Guide URL>"
  ls -lh
  head -c 300 cbecs2018_final_public.csv
  wc -l cbecs2018_final_public.csv
  shasum -a 256 *
- **Errors encountered:** User's Guide PDF was not in ~/Downloads (`zsh: no matches found`).
- **Resolution:** Downloaded it directly from EIA with curl.
- **Files:**

  | File | Size | SHA-256 |
  |---|---|---|
  | cbecs2018_final_public.csv | 19M | 3e37c81aab2938e63fae47bec837270ba696699aceef904d1326dd25970e7247 |
  | 2018microdata_codebook.xlsx | 88K | 5ee2ccb1ae8a2c4196cdb5e3ffd2be11e877d13cae3474f550306f93e92bd26e |
  | cbecs2018_users_guide.pdf | 1.6M | 815738ac9925d11584c6f817b42e891e6e1f32335d6ddd20a6fb76e5bd874a47 |

- **Observed:**
  - CSV header begins: PUBID, REGION, CENDIV, PBA, PUBCLIM, SQFT, SQFTC, WLCNS, RFCNS, ..., NFLOOR, ..., YRCONC, ...
  - Key columns confirmed present: PBA (building activity), PUBCLIM (climate), SQFT (floor area), NFLOOR, YRCONC.
  - `wc -l` = 6438 lines. Expected 6,436 records + 1 header = 6,437. Difference to be checked in Python during raw-data validation (possible trailing blank line or embedded line break).
- **Rule:** Files in data/raw/model1_cbecs2018/ are never edited.
- **Git commit hash:** (committed together with DVC tracking in Step 4)
- **Next step:** Version raw data with DVC.

---

## Step 4 — Version raw data with DVC

- **Date:** 2026-10-14
- **Status:** Done
- **Objective:** Version the raw CBECS 2018 files without committing large data to Git.
- **Why:** Git stores only a small pointer file (.dvc); DVC stores the data. Raw and processed data will be versioned separately.
- **Commands executed:**
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install dvc
  dvc add data/raw/model1_cbecs2018
- **Environment:** Python <version>, DVC <version>
- **Files created:**
  - data/raw/model1_cbecs2018.dvc (pointer file, committed to Git)
  - data/raw/.gitignore (makes Git ignore the actual data folder)
- **Observed (from the .dvc file):**
  - md5: be531a35b19eeb724795cd1def4c053a.dir
  - size: 21597835 bytes
  - nfiles: 3
- **Validation:** `git status` showed only data/raw/.gitignore, data/raw/model1_cbecs2018.dvc and the work log. The CSV was not listed, confirming Git ignores the raw data.
- **Limitation:** No DVC remote is configured in the repo yet, so the data exists only in my local DVC cache; `dvc push` is not yet possible. Data remains reproducible from the official EIA URL + SHA-256 checksums in Step 3.
- **Git commit hash:** 5ebb4fc
- **Next step:** Raw-data validation in Python (schema, true row count, missing values, key columns).

---

## Step 5 — Raw-data validation

- **Date:** 2026-10-14
- **Status:** Done
- **Objective:** Check the actual structure and quality of the raw CBECS 2018 CSV before any cleaning.
- **Script:** src/green_building/data/model1_validate_raw.py (reads raw data only; never modifies it)
- **Command executed:** python src/green_building/data/model1_validate_raw.py
- **Libraries installed:** pandas 3.0.6, numpy 2.5.4, openpyxl 3.1.5 (Python 3.14, .venv)
- **Outputs created:**
  - reports/model1/validation/raw_validation_report.txt
  - reports/model1/validation/raw_column_summary.csv
- **Observed results:**
  - 6,436 rows (buildings) × 1,249 columns; file 19.85 MB.
  - Earlier `wc -l` = 6438 was a line-break count; pandas parsed exactly 6,436 records (matches EIA). Resolved.
  - All columns numeric (637 float64, 612 int64); categorical variables are stored as numeric codes.
  - Duplicate rows: 0. Duplicate PUBID: 0.
  - All 12 key columns present: PUBID, PBA, PUBCLIM, SQFT, MFBTU, ELCNS, FINALWT, YRCONC, WKHRS, NWKER, NFLOOR, REGION.
  - SQFT: 1,001 – 2.1M sq ft, no missing/zero/negative values; strongly right-skewed.
  - MFBTU and ELCNS: 79 missing each; no zero or negative values.
  - FINALWT mean 919.5 → 6,436 × 919.5 ≈ 5.92M buildings, consistent with EIA's stated population.
  - PBA: 20 building-type codes; largest Office (2) = 1,329, Education (14) = 936; smallest Refrigerated warehouse (11) = 23, Enclosed mall (24) = 35; Vacant (1) = 117.
  - PUBCLIM: codes 1,2,3,4,5,7 (no 6); code 7 = 852 buildings, likely "withheld for confidentiality" (to confirm in codebook).
  - NFLOOR: max = 995, mean = 87.67 → contains special codes (e.g. 994/995), not real floor counts.
  - WKHRS: 0–168 (168 = 24/7). NWKER: 0–7,500.
  - Negative values: none. Infinite values: none.
  - Missingness: 764 columns complete; 317 columns >50% missing; 22 columns 100% missing.
  - Possible leakage columns flagged: 64 (energy consumption/expenditure).
- **Issues found and planned handling:**
  1. 79 buildings with missing MFBTU → exclude (target cannot be computed; target is never imputed).
  2. Skip-pattern blanks ("not applicable") → keep as information with missing indicators; no zero/mean filling. Any imputation fitted on training data only.
  3. 22 fully empty columns → drop.
  4. Special codes (NFLOOR 994/995, possibly PUBCLIM 7) → recode using the codebook.
  5. Small building-type groups (PBA 11, 24) → merge or handle during labelling; decide after EDA.
  6. Vacant buildings (PBA 1) → candidate for exclusion; decide after EDA.
  7. Non-feature columns (Z* imputation flags, FINALWT1–151 replicate weights) → not used as inputs. FINALWT kept only for weighted statistics.
- **Correction:** The leakage check used name suffixes (CNS/BTU/EXP) and produced false positives: WLCNS (wall construction material) and RFCNS (roof construction material) are building characteristics, not energy use. The final leakage list will be built from the codebook, not name patterns.
- **Git commit hash:** c446f73
- **Next step:** Inspect the codebook to confirm code meanings (PBA, PUBCLIM, NFLOOR special codes) and build the exact leakage/non-feature column list.

---

## Step 6 — Codebook inspection

- **Date:** 2026-10-14
- **Status:** Done
- **Objective:** Confirm the official meaning of codes in key columns before cleaning, instead of assuming.
- **Script:** src/green_building/data/model1_inspect_codebook.py (read-only)
- **Command executed:** python src/green_building/data/model1_inspect_codebook.py
- **Output:** reports/model1/validation/codebook_lookup.txt
- **Error encountered:** First run printed nothing — the script file had not been saved.
- **Resolution:** Pasted code, saved the file (Cmd+S), re-ran successfully.
- **Codebook structure:** 1 sheet ("2018 CBECS microdata codebook"), 1,252 rows × 6 columns (Variable order, Variable name, Variable type, Label, Values/Format codes, Question text).
- **Confirmed meanings:**
  - PBA: 20 building-activity codes (1=Vacant, 2=Office, 4=Laboratory, 5=Nonrefrigerated warehouse, 6=Food sales, 7=Public order and safety, 8=Outpatient health care, 11=Refrigerated warehouse, 12=Religious worship, 13=Public assembly, 14=Education, 15=Food service, 16=Inpatient health care, 17=Nursing, 18=Lodging, 23=Strip shopping center, 24=Enclosed mall, 25=Retail other than mall, 26=Service, 91=Other).
  - PUBCLIM: ASHRAE climate zone (third-party data). 1=Cold/very cold, 2=Cool, 3=Mixed mild, 4=Warm, 5=Hot/very hot, 7=Withheld for confidentiality.
  - NFLOOR: 1–9 = actual floors; 994 = 10–14 floors; 995 = 15 or more floors.
  - YRCONC: 2=Before 1946, 3=1946–59, 4=1960–69, 5=1970–79, 6=1980–89, 7=1990–99, 8=2000–2012, 9=2013–2018.
  - WKHRS: total hours open per week (0–168). NWKER: employees during main shift (0–7,500).
  - SQFT: gross floor area (1,001–2,100,000), incl. basements, hallways, indoor parking.
  - MFBTU: derived annual major fuels consumption, thousand Btu (kBtu); Missing = "Not applicable".
  - ELCNS: annual electricity consumption (kWh); Missing = "Not applicable".
  - FINALWT: nonresponse-adjusted survey weight.
  - WLCNS: wall construction material. RFCNS: roof construction material.
- **Key finding (units):** MFBTU (kBtu) ÷ SQFT (sq ft) = EUI in kBtu/sq ft/year — same unit as ENERGY STAR. No conversion needed.
- **Decisions:**
  1. PUBCLIM = 7 is "Withheld", not a climate zone → keep as a separate "unknown" category.
  2. NFLOOR 994/995 → recode to ordered categories (1–9, 10–14, 15+).
  3. WLCNS, RFCNS confirmed as building characteristics → valid features (corrects Step 5 suffix-based leakage flag).
  4. Final leakage list will be built from codebook labels (consumption/expenditure), not name patterns.
  5. 79 buildings with MFBTU = "Not applicable" → excluded (target cannot be computed).
- **Git commit hash:** a2fb3d6 
- **Next step:** Exploratory Data Analysis (EDA).

---

## Step 7 — Exploratory Data Analysis (EDA)

- **Date:** 2026-10-14
- **Status:** Done
- **Objective:** Understand EUI distribution, differences across building types and climates, peer-group sizes, outliers and feature relationships before cleaning and labelling.
- **Script:** src/green_building/data/model1_eda.py (read-only; EUI computed in memory only)
- **Commands executed:**
  pip install matplotlib
  pip install scipy
  python src/green_building/data/model1_eda.py
- **Error encountered:** ModuleNotFoundError: No module named 'scipy' (needed by pandas for Spearman correlation).
- **Resolution:** pip install scipy; re-ran successfully.
- **Outputs:**
  - reports/model1/eda/eda_summary.txt
  - reports/model1/eda/eui_by_building_type.csv
  - reports/model1/eda/peer_group_sizes.csv
  - reports/model1/figures/fig1_building_types.png … fig6_eui_vs_hours.png
- **EUI definition used:** EUI = MFBTU (kBtu) / SQFT (sq ft) → kBtu/sq ft/year.
- **Observed results:**
  1. 79 buildings with missing MFBTU: Warehouse 41, Vacant 32, Other 4, Retail 1, Service 1.
  2. EUI over 6,357 buildings: min 0.01, 1st pct 3.37, 25th 37.78, median 64.18, 75th 104.58, 99th 454.72, max 1,710.84. Weighted median (FINALWT) = 47.46. Strongly right-skewed.
  3. Median EUI by type ranges from 19.5 (Vacant) and 25.8 (Warehouse) to 216.4 (Food sales) and 245.4 (Food service) — ~10× difference.
  4. Median EUI by climate: Cold 66.4, Mixed mild 65.2, Cool 63.5, Hot 52.8, Warm 51.4, Withheld 97.3.
  5. Peer groups (type × climate): 101 groups; 46 have < 30 buildings (607 buildings). By type only, 1 type has < 30 buildings (Refrigerated warehouse, n = 23).
  6. WKHRS = 0: Vacant 58, Warehouse 14, Service 7, Public assembly 6, Other 5, Public order/safety 2.
  7. 64 buildings below 1st pct (3.4) and 64 above 99th pct (454.7). Highest EUIs are small food-service buildings and large 24/7 "Other" buildings — plausible, not errors.
  8. Spearman correlation with EUI: WKHRS +0.463, NWKER +0.291, NFLOOR_ORD +0.192, SQFT +0.134, YRCONC +0.025.
- **Interpretation:**
  - EUI depends strongly on building type → efficiency must be benchmarked within building type (same principle as ENERGY STAR).
  - Climate effect is smaller than type effect; type × climate groups are too small → climate used as a model feature, not a peer-group dimension.
  - Weighted and unweighted medians differ → label thresholds should use survey weights (FINALWT).
  - Building characteristics correlate with EUI → predicting efficiency tiers without energy data is feasible. Correlations are across all types and partly reflect type differences.
  - "Withheld" climate has high EUI → kept as its own category.
- **Proposed cleaning decisions (to confirm in Step 8):**
  D1 exclude 79 buildings without MFBTU; D2 exclude Vacant buildings; D3 peer group = building type only; D4 exclude Refrigerated warehouse (n=23, too small, cannot be merged with non-refrigerated warehouse); D5 keep extreme high EUI; D6 inspect extremely low EUI.
- **Git commit hash:** <fill in next step>
- **Next step:** Data cleaning.