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
- **Output observed:** <paste output of `git branch` here>
- **Git commit hash:** <fill after commit>
- **Next step:** Download CBECS 2018 files and place them in the raw data folder.
