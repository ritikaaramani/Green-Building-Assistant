# Model 1 — Building Efficiency Assessment

Part of the **AI-Based Green Building Assistant**. Model 1 tells a user whether a commercial
building is **clearly efficient or clearly inefficient compared with similar buildings**, and
says **"Uncertain"** when the evidence is not strong enough.

| | |
|---|---|
| **Task** | Binary classification: High efficiency (0) vs Low efficiency (1), with an "Uncertain" output |
| **Final model** | Soft-vote ensemble of Logistic Regression + XGBoost, with peer-relative features |
| **Baseline** | Logistic Regression (same data, split, features and label) |
| **Data** | U.S. EIA — 2018 Commercial Buildings Energy Consumption Survey (CBECS), public-use microdata |
| **Test accuracy** | **0.808** on all clear-case buildings (baseline 0.794, majority floor 0.636) |
| **Confident mode** | **0.873** accuracy on the 77.6% of buildings where the model is confident (threshold 0.65) |
| **Owner / branch** | Saakshi — `model1-building-efficiency` |
| **Full history** | [`docs/model1_work_log.md`](model1_work_log.md) — every step, decision, result and commit |

---

## 1. Data

- **Source:** CBECS 2018 public-use microdata (U.S. Energy Information Administration; public domain, attribution requested). 6,436 buildings × 1,249 columns, plus the official codebook and user guide.
- **Location:** `data/raw/model1_cbecs2018/` (tracked with DVC, never modified).
- **Cleaning** (`src/green_building/data/model1_clean.py`): removed buildings with no energy data (79), vacant buildings (85) and refrigerated warehouses (23, too few to compare) → **6,249 buildings**. Dropped replicate weights, imputation flags, empty/constant columns and **all energy-use/cost columns (leakage)**.
- **Energy Use Intensity:** EUI = total energy (kBtu) ÷ floor area (sq ft), in kBtu/sq ft/year.

## 2. Label (target)

Within each **building type** (office, school, restaurant …), buildings are ranked by EUI using the
CBECS survey weights and split into thirds:

| EUI third within its building type | Class |
|---|---|
| Lowest third | **0 = High efficiency** |
| Middle third | Moderate — **excluded** from training/testing (borderline) |
| Highest third | **1 = Low efficiency** |

This is a **relative benchmark against similar U.S. buildings**, not an official certification or ENERGY STAR score.
Why this design: the original 3-class label (Experiment 1) plateaued at macro-F1 ≈ 0.53 for every model; six label
designs were compared with **training-set cross-validation only**, and this one gave the best results while keeping the
meaning "efficient compared with similar buildings" (see work log, Step 12e).

## 3. Features

43 building characteristics + 15 peer-relative features (no energy-use or cost column is allowed — checked automatically):

- **Type & location:** building activity (PBA, PBAPLUS), climate zone, region, heating/cooling degree days
- **Size & structure:** floor area, floors, year built, wall/roof material, glass %, ceiling height, elevators, basement
- **Operations:** weekly hours, workers, % lit when open / closed, 24/7 flag, workers per 1,000 sq ft
- **Systems & controls:** main heating/cooling equipment, % heated/cooled, natural gas use, building automation (EMCS), HVAC maintenance, VAV
- **Equipment:** computers, laptops, refrigeration, vending machines, kitchen area (and per-sq-ft versions)
- **Peer-relative:** e.g. *hours open compared with the median building of the same type* (medians learned on training data only)

Full list with meanings: [`docs/model1_data_dictionary.md`](model1_data_dictionary.md).

## 4. Results (test set, used once)

| Model | Accuracy | Balanced acc. | Macro-F1 | CV macro-F1 (train) |
|---|---|---|---|---|
| Majority floor (always "Low") | 0.636 | 0.500 | 0.389 | — |
| Logistic Regression — **baseline** | 0.794 | 0.784 | 0.780 | 0.783 |
| XGBoost | 0.794 | 0.776 | 0.776 | 0.781 |
| **Final ensemble** | **0.808** | **0.792** | **0.793** | 0.796 |
| **Final ensemble, confident only (77.6% of buildings)** | **0.873** | 0.859 | 0.861 | — |

- 814 test buildings (296 High, 518 Low); the same saved 80/20 split is used by every model.
- Test results match cross-validation → no overfitting.
- Truly borderline (Moderate) buildings are flagged "Uncertain" more often (34%) than clear ones (22%).
- Experiment 1 (3 classes) for reference: baseline 0.541 / XGBoost 0.543 accuracy.
- Chart: `reports/model1/figures/fig13_model_comparison.png`; table: `reports/model1/models/model_comparison.md`.

**Limitations:** CBECS describes buildings, not how well they are operated or maintained, so ≈ 0.80 is the ceiling for
"all buildings" (all 539 usable columns were tested and did not beat the 43 selected). Labels are relative to 2018
U.S. buildings of the same type.

## 5. How to run (from the repository root)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-model1.txt          # macOS also needs: brew install libomp
dvc pull                                       # download data and models from the DVC remote
```

Pipeline, in order:

| # | Step | Command |
|---|---|---|
| 1 | Validate raw data | `python src/green_building/data/model1_validate_raw.py` |
| 2 | EDA (figures 1–6) | `python src/green_building/data/model1_eda.py` |
| 3 | Clean | `python src/green_building/data/model1_clean.py` |
| 4 | Features + label | `python src/green_building/features/model1_build_features.py` |
| 5 | Baseline | `python src/green_building/models/efficiency/model1_train_baseline.py` |
| 6 | XGBoost | `python src/green_building/models/efficiency/model1_train_xgboost.py` |
| 7 | **Final model** | `python src/green_building/models/efficiency/model1_train_final.py` |
| 8 | Comparison table + chart | `python src/green_building/models/efficiency/model1_compare_models.py` |

Analysis scripts (optional): `model1_screen_features.py` (feature screening), `model1_label_options.py` (label comparison).

## 6. MLOps

| Practice | How |
|---|---|
| Version control | All work on branch `model1-building-efficiency`, one commit per step, logged in `docs/model1_work_log.md` |
| Data & model versioning | **DVC**: `data/raw/model1_cbecs2018.dvc`, `data/interim/model1.dvc`, `data/processed/model1.dvc`, `artifacts/models/model1.dvc` |
| Experiment tracking | **MLflow** (local store `artifacts/mlflow/`): 13 runs with parameters, metrics, figures, models and Git commit tags. History: `model1_mlflow_log_history.py`; the final training script logs automatically. Dashboard: `mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --port 5001` → "Model training" view |
| Reproducibility | Fixed random seed (42), saved train/test split, pinned package versions |
| Leakage control | Energy columns removed by codebook label; automatic leakage guard; all model choices made on training cross-validation, test set used once |

## 7. Using the trained model

```python
import joblib, pandas as pd
bundle = joblib.load("artifacts/models/model1/final_ensemble.joblib")
model, threshold = bundle["model"], bundle["confidence_threshold"]
X = pd.DataFrame([...])[bundle["features"]]          # same 43 columns as the dataset
p_low = model.predict_proba(X)[:, 1]                  # probability of "Low efficiency"
confidence = abs(p_low - 0.5) + 0.5
verdict = ["Uncertain" if c < threshold else ("Low efficiency" if p >= 0.5 else "High efficiency")
           for p, c in zip(p_low, confidence)]
```
(Run from `src/green_building/models/efficiency/` or add it to `sys.path`, because the model uses the `PeerRelative` transformer from `model1_common.py`.)
