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
- **Git commit hash:** c2aa65f
- **Next step:** Data cleaning.

---

## Step 8 — Data cleaning

- **Date:** 2026-10-15
- **Status:** Done
- **Objective:** Remove buildings and columns that cannot be used, fix special codes, and record every removal with its reason. Raw data unchanged.
- **Script:** src/green_building/data/model1_clean.py
- **Command executed:** python src/green_building/data/model1_clean.py
- **Error encountered:** First run failed with "No such file or directory" — script file had not been created.
- **Resolution:** Created file with `touch`, pasted code, saved, re-ran successfully.
- **Input:** data/raw/model1_cbecs2018/cbecs2018_final_public.csv (6,436 rows × 1,249 columns)
- **Output:** data/interim/model1/cbecs2018_clean.csv (6,249 rows × 537 columns) — to be versioned with DVC in Step 10, not committed to Git.
- **Reports:** reports/model1/cleaning/
  - cleaning_report.txt — full run log
  - excluded_buildings.csv — every removed building + reason
  - dropped_columns.csv — every removed column + reason + codebook label
  - low_eui_buildings.csv — buildings with EUI < 1 kBtu/sq ft/yr
- **Rows removed (187):**
  - D1: 79 buildings with MFBTU missing ("Not applicable") — target cannot be computed; target never imputed.
  - D2: 85 vacant buildings (PBA = 1) — efficiency not meaningful.
  - D4: 23 refrigerated warehouses (PBA = 11) — too few to benchmark; EUI ~3.6× non-refrigerated warehouses, so not merged.
- **D6 (EUI < 1):** 6 buildings found and listed; **kept**. They are 0.1% of the data and labels are rank-based within building type, so they cannot distort thresholds.
- **Columns removed (712):**
  - 151 replicate weights (FINALWT1–151) — survey error estimation only.
  - 477 imputation flags (Z-prefixed) — metadata, not features.
  - 49 leakage columns — identified from codebook labels (consumption, expenditure, thous Btu, kWh, cubic feet, gallons, cost).
  - 22 fully empty columns.
  - 13 constant columns.
- **Kept on purpose:** MFBTU and SQFT (to compute the label in Step 9), FINALWT (weighted thresholds), PUBCLIM incl. code 7 = Withheld. None of these are model inputs except PUBCLIM.
- **Special codes:** NFLOOR — 549 values recoded (994 → 10 "10–14 floors", 995 → 11 "15+ floors"). 19 other columns contain 99x codes (BASEMNT, FLCEILHT, NELVTR, NESLTR, RWSEAT, PBSEAT, HCBED, NRSBED, LODGRM, NOCC, NOCCAT, XRAYN, RFGRSN, RFGCOMPN, SERVERN, LGOFFDEVN, SMOFFDEVN, WBOARDSN, TVVIDEON) → handled in Step 9 for any used as features.
- **Missing values:** NOT filled. Blanks are mostly survey skip patterns ("not applicable"); handled inside the model pipeline using training data only.
- **Validation (all PASS):** no missing MFBTU; no vacant/refrigerated warehouse; SQFT > 0; PUBID unique; no NFLOOR special codes; no infinite values; rows add up (6,249 + 187 = 6,436); no leakage columns left.
- **Buildings per type after cleaning (PBA: n):** 2:1329, 4:69, 5:712, 6:91, 7:105, 8:217, 12:271, 13:481, 14:936, 15:218, 16:276, 17:114, 18:418, 23:208, 24:35, 25:338, 26:349, 91:82. Smallest group = 35 (Enclosed mall).
- **Pending check:** name-based scan for remaining *BTU/*CNS/*EXP columns will be enforced automatically in the Step 9 script (expected only WLCNS, RFCNS, MFBTU).
- **Git commit hash:** a264a29 
- **Next step:** Feature engineering and efficiency label definition.

---

## Step 9 — Feature engineering and efficiency labels

- **Date:** 2026-10-15
- **Status:** Done
- **Objective:** Define the efficiency target defensibly, select building-characteristic features, and create the model-ready dataset without target leakage.
- **Script:** src/green_building/features/model1_build_features.py
- **Command executed:** python src/green_building/features/model1_build_features.py
- **Input:** data/interim/model1/cbecs2018_clean.csv (6,249 × 537)
- **Outputs:**
  - data/processed/model1/model1_dataset.csv (6,249 × 31) — versioned with DVC in Step 10
  - data/processed/model1/feature_spec.json — lists features, target, id, weight
  - reports/model1/features/label_thresholds_by_type.csv
  - reports/model1/features/features_report.txt
  - docs/model1_data_dictionary.md
- **Target definition:**
  - EUI = MFBTU (kBtu) / SQFT (sq ft) → kBtu/sq ft/year (same unit as ENERGY STAR; no conversion).
  - Within each building type (PBA), weighted EUI tertiles using FINALWT:
    0 = High efficiency (lowest third), 1 = Moderate, 2 = Low efficiency (highest third).
  - Benchmark source: CBECS 2018 itself (relative peer comparison, ENERGY STAR-style principle).
  - Limitation: relative ranking against CBECS 2018 peers of the same type — NOT an official certification or ENERGY STAR score. EUI does not adjust for occupancy, climate or operating hours; those are model inputs instead.
- **Example thresholds (kBtu/sq ft/yr):** Office (PBA 2): High ≤ 41.42, Low > 65.51; Warehouse (5): ≤ 13.21 / > 32.39; Education (14): ≤ 36.99 / > 64.58; Food service (15): ≤ 174.44 / > 322.59. Full table in label_thresholds_by_type.csv.
- **Features (26):**
  - Categorical (11): PBA, PUBCLIM, REGION, CENDIV, WLCNS, RFCNS, BLDSHP, OWNTYPE, MAINHT, MAINCL, RENOV
  - Numeric (12): SQFT, NFLOOR, YRCONC, WKHRS, NWKER, GLSSPC, FLCEILHT, NELVTR, BASEMNT, HEATP, COOLP, LTOHRP
  - Engineered (3): LOG_SQFT, WORKERS_PER_1000SQFT, OPEN_24_7
- **Special codes:** top-coded only where the codebook documents 99x codes — FLCEILHT (14 values → 51), NELVTR (38 → 31), BASEMNT (25 → 5).
- **Error found and fixed:** First version treated every value ≥ 994 as a special code, which wrongly capped 308 NWKER values (real worker counts up to 7,500) at 991. Fixed so only codebook-documented 99x variables are recoded. Also removed pandas PerformanceWarnings by keeping only needed columns.
- **Leakage prevention:** Automatic guard confirms no energy consumption/expenditure column, MFBTU, EUI, label or FINALWT is a feature. Check for remaining *BTU/*CNS/*EXP columns in clean data: none (closes Step 8 pending check). SQFT is kept as a feature — it is a building characteristic and cannot reveal EUI without energy data.
- **Class distribution:**
  - Sample counts: Low 2,588 (41.4%), Moderate 2,182 (34.9%), High 1,479 (23.7%)
  - Weighted share (U.S. buildings): High 33.3%, Moderate 33.4%, Low 33.4% — confirms correct tertiles.
  - Reason for sample imbalance: survey over-samples large buildings, which tend to have higher EUI. Handled in training with class weights and macro-F1.
- **Missing values in features:** NELVTR 53.9%, BASEMNT 39.7%, RENOV 6.3%, MAINCL 5.6%, COOLP 5.6%, MAINHT 5.1%, HEATP 5.1%, LTOHRP 0.5%. Mostly "not applicable" skip patterns → handled in the model pipeline (imputation + missing indicator, fitted on training data only).
- **Validation (all PASS):** every building labelled; 3 classes present; no infinite values; row count unchanged.
- **Git commit hash:** 23bc65a 
- **Next step:** Version the interim and processed datasets with DVC.

---

## Step 10 — Version interim and processed data with DVC

- **Date:** 2026-10-15
- **Status:** Done
- **Objective:** Version the cleaned and model-ready datasets separately from raw data, so every model can be traced to the exact data it was trained on.
- **Commands executed:**
  dvc add data/interim/model1
  dvc add data/processed/model1
- **Files created/modified:**
  - data/interim/model1.dvc (new pointer file)
  - data/interim/.gitignore (new — makes Git ignore the data folder)
  - data/processed/model1.dvc (new pointer file)
  - data/processed/.gitignore (modified — one line `/model1` appended; teammate's existing entry untouched)
- **Observed:**
  - Interim: md5 <md5>, size <size>, nfiles <n>
  - Processed: md5 <md5>, size <size>, nfiles <n>
- **Data lineage:**
  raw (Step 4) → model1_clean.py → interim (this step) → model1_build_features.py → processed (this step)
- **Limitation:** No DVC remote configured yet — data is in my local DVC cache only. All datasets can be regenerated from the official EIA download (Step 3 checksums) by running the scripts in order.
- **Git commit hash:** 4de5426 
- **Next step:** Train the baseline model (Logistic Regression).

---

## Step 11 — Baseline model (Logistic Regression)

- **Date:** 2026-10-15
- **Status:** Done
- **Objective:** Train a simple, interpretable baseline that XGBoost must beat, using the same split and metrics.
- **Scripts:**
  - src/green_building/models/efficiency/model1_common.py (shared: data loading, split, metrics, confusion plot)
  - src/green_building/models/efficiency/model1_train_baseline.py
- **Commands executed:**
  pip install scikit-learn
  python src/green_building/models/efficiency/model1_train_baseline.py
- **Data:** data/processed/model1/model1_dataset.csv (6,249 buildings, 26 features)
- **Split:** 80/20 stratified, random_state = 42 → Train 4,999 / Test 1,250. One row = one building (PUBID unique), so test buildings are unseen. Split saved to reports/model1/models/train_test_split.csv and re-used for XGBoost.
- **Pipeline:**
  - Numeric: median imputation + missing-value indicator → StandardScaler
  - Categorical: constant imputation (-1 = not applicable) → One-hot encoding
  - All preprocessing fitted on training data only (inside the pipeline) → no test-set leakage.
  - LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
- **Evaluation strategy:** 5-fold stratified CV on train (macro-F1) + single held-out test set. Main metric: macro-F1 (classes are imbalanced in the sample).
- **Results (test set, 1,250 buildings):**

  | Model | Accuracy | Balanced acc. | Macro-F1 |
  |---|---|---|---|
  | Majority-class floor | 0.414 | — | 0.195 |
  | Logistic Regression | 0.541 | 0.538 | 0.528 |

  - 5-fold CV macro-F1 (train): 0.519 ± 0.009
  - Per class (precision / recall / F1): High 0.435 / 0.578 / 0.496 (n=296); Moderate 0.474 / 0.390 / 0.428 (n=436); Low 0.673 / 0.647 / 0.659 (n=518)
  - Confusion matrix (rows actual High/Moderate/Low; cols predicted): [171, 83, 42], [145, 170, 121], [77, 106, 335]
- **Observations:**
  - Clearly above majority-class floor (0.195) and random guessing (~0.33) → building characteristics carry real signal about efficiency tier.
  - CV ≈ test and small std → stable, not over-fitted.
  - Low efficiency easiest (F1 0.659); Moderate hardest (F1 0.428) — it borders both other classes.
  - Most errors are adjacent-class; severe High↔Low errors = 119/1,250 (9.5%).
  - Limitation: ~46% of buildings misclassified — many drivers of energy use (equipment condition, behaviour, maintenance) are not in CBECS.
- **Outputs:**
  - artifacts/models/model1/baseline_logreg.joblib (model — versioned with DVC in Step 13)
  - reports/model1/models/baseline_metrics.json
  - reports/model1/models/baseline_classification_report.txt
  - reports/model1/models/train_test_split.csv
  - reports/model1/figures/fig7_confusion_baseline.png
- **Git commit hash:** 6c94c73i
- **Next step:** Train XGBoost Classifier on the same split; target to beat = macro-F1 0.528.

 ## Step 12 — Main model: XGBoost Classifier (Experiment 1).
**Date:** 2026-10-15
**Script:** `src/green_building/models/efficiency/model1_train_xgboost.py`

**Setup**
- Same features (26), same saved train/test split (4,999 / 1,250), same metrics as the baseline (shared `model1_common.py`).
- Numeric features passed through unchanged (XGBoost handles NaN natively; trees need no scaling). Categorical: missing → -1, then one-hot.
- Class imbalance: balanced sample weights.
- Tuning: RandomizedSearchCV, 20 settings × 5-fold stratified CV on TRAIN only, scoring = macro-F1. Test set used once at the end.
- Feature importance: permutation importance on the test set (drop in macro-F1 when a feature is shuffled, 10 repeats).

**Issue + fix**
- `XGBoostError: libomp.dylib could not be loaded` on macOS → fixed with `brew install libomp` (23.1.3).
- Earlier gain-based importance summed over one-hot columns and inflated multi-level categoricals → replaced with permutation importance on the original 26 features.

**Best settings:** max_depth 3, n_estimators 500, learning_rate 0.1, subsample 0.7, colsample_bytree 0.8, min_child_weight 1, reg_lambda 1.

**Results (test set)**
| Metric | Baseline (LogReg) | XGBoost |
|---|---|---|
| CV macro-F1 (train) | 0.519 ± 0.009 | 0.526 ± 0.014 |
| Accuracy | 0.541 | 0.543 |
| Balanced accuracy | 0.538 | 0.534 |
| Macro-F1 | 0.528 | 0.530 |
| F1 High / Moderate / Low | 0.496 / 0.428 / 0.659 | 0.479 / 0.440 / 0.670 |
| Severe errors (High↔Low) | 119 | 100 |

Confusion matrix (rows = actual High/Moderate/Low): [155, 99, 42], [138, 185, 113], [58, 121, 339]

Top features (permutation importance): PBA 0.060, WKHRS 0.050, MAINHT 0.040, WORKERS_PER_1000SQFT 0.029, LTOHRP 0.010.

**Interpretation**
- XGBoost ≈ baseline: the +0.002 macro-F1 gain is far smaller than the CV std (±0.014), so it is not a meaningful improvement. The only real gain is 19 fewer severe High↔Low errors.
- Tuning chose the shallowest trees (depth 3), which suggests simple relationships and that deeper trees overfit.
- Likely performance ceiling: features describe the building, not its equipment condition, controls or occupant behaviour. Moderate is the hardest class (it borders both others).
- Feature importance is consistent with the EDA (operating hours strongest), and the engineered occupancy-density feature ranked 4th.
- Formal comparison and final model choice in Step 13.

**Outputs:** `reports/model1/models/xgboost_metrics.json`, `xgboost_classification_report.txt`, `xgboost_feature_importance.csv`, `reports/model1/figures/fig8_confusion_xgboost.png`, `fig9_xgboost_feature_importance.png`. Model saved at `artifacts/models/model1/xgboost.joblib` (not in Git; versioned with DVC in Step 13).

**Commit:** ded6942

## Step 12b — Screening of unused columns (to improve Experiment 1)
**Date:** 2026-10-15
**Script:** `src/green_building/features/model1_screen_features.py`
**Output:** `reports/model1/features/candidate_features.csv`

**Why:** Experiment 1 (26 features) reached only macro-F1 ≈ 0.53 for both models. The cleaned data still had ~534 numeric building-characteristic columns that were never tried.

**Method**
- Training buildings only (4,999, same saved split) so the test set stays untouched.
- Energy use/cost columns excluded again by codebook label words.
- Score 1: mutual information with the efficiency class (missing = -1 for scoring only).
- Score 2: Spearman correlation with EUI (direction of the effect).

**Results**
- 6 of the top 40 columns were already in the model (MAINHT, WKHRS, NWKER, FLCEILHT, PBA, MAINCL).
- Highest mutual information only 0.06 (maximum ≈ 1.1 for 3 balanced classes): no single column is a strong predictor.
- Effects match domain knowledge: building automation (EMCS, −0.24) and regular HVAC maintenance (MAINT, −0.18) go with lower EUI; lights on when closed (LTNHRP, +0.37) and open refrigerated cases (RFGOPN, +0.18) go with higher EUI.
- `ConstantInputWarning`: some columns are constant within the training set (harmless).

**Selected:** LTNHRP, PBAPLUS, HDD65, CDD65, NGUSED, NGWATR, EMCS, MAINT, VAV, PCTERMN, LAPTPN, RFGOPN, RFGWIN, RFGVNN, FDPREPSF + engineered PCS_PER_1000SQFT, KITCHEN_SHARE.

**Rejected:** category duplicates (NWKERC, WKHRSC, PCTRMC, LAPTPC, LNHRPC, LOHRPC); *_NG equipment flags (repeat NGUSED); NGHT1, ELHT1, BOILER (repeat MAINHT); EMCSHT, EMCSCL (sub-parts of EMCS, 51% missing); PKLTN, RGSTRN (weak link, many blanks).

**Leakage check:** all selected columns describe type, equipment, controls, fuels or climate — not energy used or paid for.

## Step 12c — Experiment 2 dataset: new features + Option C target
**Date:** 2026-10-15
**Script:** `src/green_building/features/model1_build_features.py`

- Added the 15 columns from Step 12b + engineered PCS_PER_1000SQFT, KITCHEN_SHARE → 43 features (17 categorical, 26 numeric).
- New target **EFFICIENCY_BINARY (Option C):** 0 = High efficiency (lowest EUI third of its building type), 1 = Low efficiency (highest third); Moderate (middle third) left blank and excluded. 3-class thresholds unchanged.
- Counts: Low 2,588, High 1,479, Moderate excluded 2,182 (of 6,249).
- Special codes top-coded: FLCEILHT (14), NELVTR (38), BASEMNT (25). Numeric ranges checked, no leftover codes.
- Leakage guard PASS; 6/6 validation checks PASS. Output 6,249 × 49; re-versioned with DVC.
- Shared split unchanged; `get_split` removes Moderate buildings from both sides after the split (train 4,999 → 3,253, test 1,250 → 814).

## Step 12e — Label design comparison (training CV only)
**Date:** 2026-10-15
**Script:** `src/green_building/models/efficiency/model1_label_options.py`
**Output:** `reports/model1/models/label_options_cv.csv`

5-fold CV on the 4,999 training buildings, same XGBoost settings, test set not used.

| Option | Classes | Majority floor | CV accuracy | CV macro-F1 |
|---|---|---|---|---|
| 0 current (3 classes within type) | 3 | 0.414 | 0.545 | 0.535 |
| B within type, 2 classes | 2 | 0.595 | 0.728 | 0.720 |
| **C within type, High vs Low** | 2 | 0.636 | **0.788** | **0.773** |
| A absolute, 3 classes | 3 | 0.462 | 0.659 | 0.646 |
| AB absolute, 2 classes | 2 | 0.660 | 0.792 | 0.773 |
| AC absolute, lowest vs highest third | 2 | 0.701 | 0.882 | 0.861 |

**Decision:** Option C — keeps the meaning "efficient compared with similar buildings". AC scores higher but predicts energy intensity, not efficiency vs peers. New features alone barely moved the 3-class label (0.535) → the label design was the main lever.

## Step 12d — Baseline + XGBoost retrained on Option C (Experiment 2)
**Date:** 2026-10-15
- Experiment 1 results archived in `reports/model1/experiments/exp1/`.
- XGBoost script: objective now chosen automatically (binary for 2 classes).

| Model (test, 814 buildings) | Accuracy | Balanced acc. | Macro-F1 | CV macro-F1 |
|---|---|---|---|---|
| Majority floor | 0.636 | 0.500 | 0.389 | — |
| Logistic Regression | 0.794 | 0.784 | 0.780 | 0.783 ± 0.010 |
| XGBoost | 0.794 | 0.776 | 0.776 | 0.781 ± 0.018 |

- Confusion (rows actual High/Low): LogReg [221, 75], [93, 425]; XGBoost [210, 86], [82, 436].
- XGBoost best: max_depth 6, n_estimators 500, learning_rate 0.03, subsample 0.85, colsample_bytree 0.8, min_child_weight 5, reg_lambda 10.
- Top permutation importance: PBA 0.047, NGUSED 0.028, WKHRS 0.027, SQFT 0.014, MAINHT 0.009.
- XGBoost ties the baseline again → the limit is the information in the data, not the algorithm.

## Step 12f — Improvement methods (training CV only)
**Date:** 2026-10-15
Exploration on training buildings only (5-fold CV), run on a copy of the processed dataset.

| Method | CV accuracy | CV balanced | CV macro-F1 |
|---|---|---|---|
| LogReg (balanced) | 0.794 | 0.792 | 0.783 |
| XGBoost (balanced) | 0.791 | 0.781 | 0.777 |
| LogReg, no class balancing | 0.804 | 0.779 | 0.784 |
| LogReg + peer-relative features | 0.797 | 0.795 | 0.786 |
| XGBoost shallower / HistGradientBoosting | 0.790 | 0.772–0.784 | 0.772–0.778 |
| Ensemble LogReg + XGBoost | 0.800 | 0.796 | 0.788 |
| **Ensemble + peer-relative features** | **0.804** | **0.800** | **0.793** |
| XGBoost / Ensemble on all 539 building columns | 0.800–0.802 | 0.796–0.797 | 0.789–0.790 |

- Peer-relative feature = log(1 + value) − log(1 + median of the same building type); medians learned on training data only.
- All 539 non-leaky columns did not beat the 43 selected features → ≈ 0.80 is the CBECS ceiling for Option C.
- Selective prediction (answer only when confident), out-of-fold: threshold 0.65 → 79% coverage, 0.861 accuracy; 0.70 → 72%, 0.877; 0.75 → 63%, 0.903.
- "No class balancing" rejected: higher accuracy only by favouring the larger class (balanced accuracy drops).
- **Decision:** Ensemble (LogReg + XGBoost) + peer-relative features, confidence threshold 0.65.

## Step 12g — FINAL model
**Date:** 2026-10-16
**Script:** `src/green_building/models/efficiency/model1_train_final.py`
**Outputs:** `reports/model1/models/final_metrics.json`, `final_classification_report.txt`, `final_confidence_table_cv.csv`; figures `fig11_confusion_final.png`, `fig12_confidence_coverage.png`; model `artifacts/models/model1/final_ensemble.joblib` (versioned with DVC in Step 13).

- Soft-vote ensemble of Logistic Regression + XGBoost, 43 features + 15 peer-relative features, Option C label, confidence threshold 0.65. All choices made on training CV; test set used once.
- CV on train: accuracy 0.808 ± 0.010, balanced 0.803, macro-F1 0.796 ± 0.009.

| Test set (814 clear-case buildings) | Accuracy | Balanced acc. | Macro-F1 |
|---|---|---|---|
| Majority floor | 0.636 | 0.500 | 0.389 |
| Baseline LogReg | 0.794 | 0.784 | 0.780 |
| **Final ensemble** | **0.808** | **0.792** | **0.793** |
| **Final, confident only (≥ 0.65)** | **0.873** | 0.859 | 0.861 |

- Confident mode: verdict for 632 / 814 buildings (77.6%); 182 marked "Uncertain".
- Confusion (all, rows actual High/Low): [217, 79], [77, 441].
- Sanity check: truly borderline (Moderate) test buildings are marked "Uncertain" 34.2% of the time vs 22.4% for clear cases → the uncertainty output behaves sensibly.
- Test results match CV (0.808 vs 0.808; 0.873 vs 0.861) → no overfitting.

**Rejected:** two scripts generated by an AI coding agent (`model1_train_exp2.py`, `model1_train_exp3.py`) dropped 9 features based on permutation importance measured on the **test set** and compared models on the test set (test-set leakage). Exp2 (XGBoost + RandomForest + ExtraTrees) scored lower (accuracy 0.807, balanced 0.778, macro-F1 0.785). Exp3 (stacking of 4 models, 337 MB model file) scored accuracy 0.814, balanced 0.801, macro-F1 0.800 — within noise of the final model (+0.006) and not trustworthy, because its features and model choice were made using the test set. Neither was committed; their scripts, reports and model files were deleted on 2026-10-16.

**Commits:** 9834fa5 (Steps 12b, 12c, 12e — data, features, label), 5c7dc52 (Steps 12d, 12f, 12g — models, final ensemble)

## Step 13 — Model comparison + model versioning with DVC
**Date:** 2026-10-16
**Script:** `src/green_building/models/efficiency/model1_compare_models.py` (reads saved metric files only — no training, no new test-set access)
**Outputs:** `reports/model1/models/model_comparison.csv`, `model_comparison.md`, `reports/model1/figures/fig13_model_comparison.png`

| Experiment | Model | Accuracy | Balanced acc. | Macro-F1 | CV macro-F1 | Test buildings |
|---|---|---|---|---|---|---|
| Exp 1: 3 classes | Majority floor | 0.414 | 0.333 | 0.195 | — | 1,250 |
| Exp 1: 3 classes | Logistic Regression (baseline) | 0.541 | 0.538 | 0.528 | 0.519 | 1,250 |
| Exp 1: 3 classes | XGBoost | 0.543 | 0.534 | 0.530 | 0.526 | 1,250 |
| Final label: High vs Low | Majority floor | 0.636 | 0.500 | 0.389 | — | 814 |
| Final label: High vs Low | Logistic Regression (baseline) | 0.794 | 0.784 | 0.780 | 0.783 | 814 |
| Final label: High vs Low | XGBoost | 0.794 | 0.776 | 0.776 | 0.781 | 814 |
| Final label: High vs Low | **Final ensemble** | **0.808** | **0.792** | **0.793** | 0.796 | 814 |
| Final label: High vs Low | **Final ensemble, confident only (77.6% coverage)** | **0.873** | 0.859 | 0.861 | — | 632 |

**Conclusions**
- Biggest gain came from the label design (macro-F1 0.53 → 0.78), not from the algorithm.
- XGBoost alone never beat the Logistic Regression baseline; the ensemble beats it by +1.4 points (accuracy) / +1.3 points (macro-F1), consistent in CV and test.
- Confident mode adds the largest practical gain: 0.873 accuracy on 77.6% of buildings, the rest flagged "Uncertain".
- **Selected model for deployment:** `final_ensemble.joblib` with confidence threshold 0.65.

**Model versioning:** `dvc add artifacts/models/model1` → `artifacts/models/model1.dvc` (3 files, 2.2 MB: `baseline_logreg.joblib`, `xgboost.joblib`, `final_ensemble.joblib`). Model files are in the DVC cache, not in Git. No DVC remote configured yet (to decide with the team).

**Commit:** 5f220b8

## Step 14 — Experiment tracking with MLflow
**Date:** 2026-10-16
**Scripts:** `src/green_building/models/efficiency/model1_mlflow_utils.py`, `model1_mlflow_log_history.py`; MLflow logging added to `model1_train_final.py`
**Tool:** MLflow 3.17.1, local SQLite store in `artifacts/mlflow/` (ignored by Git via its own `.gitignore`)

**What was done**
- Experiment `model1-building-efficiency` created.
- History logged from saved result files (no retraining, no new test-set access), each run tagged with the Git commit that produced it:
  - `exp1_baseline_logreg` (6c94c73), `exp1_xgboost` (ded6942) — Experiment 1, 3 classes
  - `label_options_cv` + 6 child runs (9834fa5) — label comparison on training CV
  - `optionC_baseline_logreg`, `optionC_xgboost`, `final_ensemble` (5c7dc52)
- `model1_train_final.py` now logs every run automatically (parameters, CV + test metrics, confident-mode metrics, figures, report and the model file). Live run `final_ensemble_train` (commit d22f97f) reproduced the same results: test accuracy 0.808, macro-F1 0.793, confident accuracy 0.873 at 77.6% coverage.
- Total: 13 runs. Re-running the history script replaces the old history runs (no duplicates).

**Dashboard:** `mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --port 5001` → http://127.0.0.1:5001 → switch to **Model training** view.
- Port 5001 used because macOS AirPlay Receiver occupies port 5000 (gave HTTP 403).

**Commit:** 4259172

## Step 15 — Documentation, requirements, merge with main, reproducibility check
**Date:** 2026-10-16

**Documentation:** `docs/model1_README.md` — overview, data, label, features, results, how to run each step, MLOps practices, how to use the saved model. Kept in its own file so the team README is not modified. `requirements-model1.txt` pins the exact versions used (Python 3.14; pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, xgboost 3.4.1, dvc 3.67.1, mlflow 3.17.1, …). Commit 73e6bbc.

**Merge with main:** merged `origin/main` (0ae5f37 — teammate commit adding the DagsHub DVC remote and BDG2 tracking). One add/add conflict in `data/raw/.gitignore`, resolved by keeping both entries (`/model1_cbecs2018`, `/bdg2`). Branch is 0 commits behind main. Commit 0d47c45.

**DVC:** final model re-saved by the MLflow-tracked run → `artifacts/models/model1` re-versioned (commit 2367c6e). `dvc status` for all four Model 1 `.dvc` files: up to date.

**Reproducibility check** — fresh environment created from `requirements-model1.txt` (Linux, Python 3.14.8), pipeline re-run from the raw CBECS file:

| Output | Result |
|---|---|
| Cleaned data (`cbecs2018_clean.csv`) | Byte-for-byte identical |
| Labels (3-class and High/Low) | Identical |
| Features | Identical except LOG_SQFT in 119 rows, max difference 8.9e-16 (OS math-library rounding of log10) |
| Baseline (test) | Identical: accuracy 0.7936, macro-F1 0.7798, same confusion matrix |
| Final ensemble (test, all) | 0.811 / macro-F1 0.795 vs original 0.808 / 0.793 — 2 of 814 predictions differ |
| Final ensemble (confident) | 0.873 on 630 buildings vs 0.873 on 632 |
| Re-run on the original Mac | Exactly identical (0.808 / 0.873) |

**Conclusion:** fully reproducible on the same machine; across operating systems, XGBoost's multi-threaded training and floating-point rounding shift ±2 test predictions (±0.002), which does not change any conclusion.

**Pending (needs the owner's accounts):** `git push`, `dvc push` of Model 1 data/models to the team's DagsHub remote (collaborator access + token), Pull Request into `main`.

**Commit:** <fill in after committing>
