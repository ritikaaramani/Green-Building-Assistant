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
- **Git commit hash:** <fill in at start of next step>
- **Next step:** Raw-data validation in Python (schema, true row count, missing values, key columns).