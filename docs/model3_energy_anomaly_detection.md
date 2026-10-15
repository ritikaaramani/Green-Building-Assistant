# Model 3: Energy Anomaly Detection

## Overview

Model 3 detects anomalous electricity consumption readings in the
Building Data Genome Project 2 (BDG2) dataset using per-building
Isolation Forest models with a secondary z-score layer.
It is entirely unsupervised — no ground-truth anomaly labels exist
in the current dataset.

---

## Dataset

**Source:** Building Data Genome Project 2 (BDG2)
**File:** `data/raw/bdg2/meters/cleaned/electricity_cleaned.csv`
**Meter type:** Electricity (kWh), hourly readings
**Period:** 2016-01-01 00:00 → 2017-12-31 23:00 (17,544 hourly readings)
**Buildings used (5 of 1,636):**

| Building ID | Type |
|---|---|
| `Robin_education_Kiera` | Education |
| `Robin_education_Derrick` | Education |
| `Robin_lodging_Pricilla` | Lodging |
| `Robin_office_Maryann` | Office |
| `Robin_office_Serena` | Office |

**Selection rationale:** The five buildings span three occupancy types
from the "Robin" site cluster, providing diversity in consumption
patterns while keeping runtime tractable.

**Missing values:** `Robin_lodging_Pricilla` has 2 internally missing
readings. All others are complete.

---

## Feature Engineering

### V1 features (`anomaly_features.csv`)

| Feature | Description |
|---|---|
| `consumption` | Raw kWh reading |
| `hour` | Integer hour of day (0–23) |
| `day_of_week` | Integer day (0=Monday, 6=Sunday) |
| `is_weekend` | Binary (1 if day ≥ 5) |
| `rolling_mean_24h` | Mean of past 24 readings (shift(1) applied) |
| `rolling_std_24h` | Std dev of past 24 readings |
| `deviation_24h` | `consumption − rolling_mean_24h` |

### V2 features (`anomaly_features_v2.csv`)

Replaces linear hour/day integers with **cyclical encoding** and adds
rolling median:

| Feature | Formula |
|---|---|
| `hour_sin / hour_cos` | `sin/cos(2π × hour / 24)` |
| `day_sin / day_cos` | `sin/cos(2π × day / 7)` |
| `rolling_median_24h` | Median of past 24 readings |
| `median_deviation_24h` | `consumption − rolling_median_24h` |

**Cyclical encoding rationale:** Hour 23 and hour 0 are one hour apart
in time but 23 units apart in linear space. Sine/cosine encoding
preserves the circular distance, improving isolation of overnight
patterns.

### Missing value treatment

Interior gaps of ≤ 2 consecutive readings are filled with linear
interpolation (`limit=2, limit_area='inside'`). Leading/trailing NaNs
and gaps > 2 are left as NaN and dropped before training.

### Leakage prevention (verified)

All rolling statistics use `shift(1)` before the rolling window,
ensuring that the current reading's value is never included in its own
rolling mean/std/median. This was verified numerically:
`rolling_mean_24h` at row 24 equals `mean(consumption[0:23])` exactly.

---

## Model Methodology

### Architecture

One independent pipeline per building — buildings have very different
consumption scales and occupancy patterns. A global model would be
dominated by high-consumption buildings.

### V3 pipeline (production)

```
StandardScaler → IsolationForest
```

**Isolation Forest hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | More trees than V1/V2 (100) for lower variance |
| `contamination` | 0.02 | Expected ~2% anomaly rate; sets the score threshold |
| `max_samples` | `"auto"` | Uses `min(256, n_samples)` per tree |
| `random_state` | 42 | Reproducibility |

**StandardScaler:** Fitted on the training set only. Applied to both
training and test sets. Prevents scale from dominating isolation.

### Secondary z-score layer

For each building, `μ` and `σ` of `consumption` are computed on the
**training set only** (no look-ahead). A test reading is additionally
flagged if:

```
|consumption − μ_train| / σ_train > 3.5
```

The threshold of 3.5σ was chosen to catch statistical outliers that
Isolation Forest may under-flag (e.g. sustained plateau anomalies).

### Combination rule

```
is_anomaly = is_anomaly_iforest OR is_anomaly_zscore
```

This was numerically verified to hold for every row in the evaluation
set (17,520 rows, 5 buildings, exact match = True).

### Chronological split (verified)

Each building is split 80/20 by time position (not random):

| Split boundary | Date |
|---|---|
| Train end | 2017-08-07 |
| Test start | 2017-08-08 |

No timestamp appears in both train and test. This was verified for all
five buildings.

---

## Evaluation Results

> **Important caveat:** No ground-truth anomaly labels exist. The counts
> below reflect model sensitivity at `contamination=0.02`, not detection
> accuracy. Flagged readings should be inspected rather than assumed correct.

### V1 vs V2 vs V3 flag counts (evaluation set, 3,504 rows per building)

| Building | V1 Flagged | V2 Flagged | V3 IForest | V3 Z-score | V3 Combined |
|---|---|---|---|---|---|
| Robin_education_Derrick | 19 | 72 | 85 | 1 | **85** |
| Robin_education_Kiera | 381 | 233 | 216 | 0 | **216** |
| Robin_lodging_Pricilla | 23 | 12 | 9 | 0 | **9** |
| Robin_office_Maryann | 82 | 74 | 65 | 66 | **105** |
| Robin_office_Serena | 2 | 29 | 26 | 0 | **26** |
| **Total** | **507** | **420** | **401** | **67** | **441** |

### V1 vs V2 agreement on identical timestamps

| Metric | Value |
|---|---|
| Common evaluation readings | 17,520 |
| Agreement | 97.89% |
| Both flag | 279 |
| V1 only | 228 |
| V2 only | 141 |

### V3 anomaly score statistics (lower = more anomalous)

| Building | Anomalous mean score | Normal mean score | Separation |
|---|---|---|---|
| Robin_education_Derrick | -0.0088 | 0.0795 | 0.0883 |
| Robin_education_Kiera | -0.0196 | 0.0586 | 0.0782 |
| Robin_lodging_Pricilla | -0.0093 | 0.1142 | 0.1235 |
| Robin_office_Maryann | -0.0148 | 0.1865 | 0.2013 |
| Robin_office_Serena | -0.0146 | 0.1203 | 0.1349 |

Score separation is positive for all buildings, confirming the model
assigns systematically lower scores to flagged readings.

### Notable patterns from window inspection

- **Robin_office_Maryann:** Consumption drops to near-zero (4.5 kWh)
  for extended periods — possible meter fault or building shutdown.
  Both IForest and z-score flag these.
- **Robin_education_Kiera:** Isolated low-consumption overnight readings
  flagged by V1; V2 reduces these flags due to cyclical features.
- **Robin_office_Serena:** Rapid ramp-up from ~1 kWh baseline to 60–90
  kWh flagged — consistent with sudden occupancy changes.

---

## Generated Figures

All figures are in `reports/figures/model3/`.

| File | Description |
|---|---|
| `per_building_Robin_*.png` | Three-panel time series (V1/V2/V3) per building |
| `v1_v2_comparison.png` | Side-by-side bar chart of V1 vs V2 flag counts |
| `anomaly_score_distributions.png` | Per-building score histograms (V3) |
| `flag_rate_summary.png` | V3 IForest vs z-score flag attribution |

---

## Reproduction Commands

Run all commands from the repository root with `green-building` active.

```bash
conda activate green-building
cd /path/to/Green-Building-Assistant

# 1. Select buildings from raw BDG2 data
python src/data/prepare_anomaly_data.py

# 2. Build V2 features (cyclical encoding, rolling statistics)
python src/features/build_anomaly_features_v2.py

# 3. Train V3 production model (saves models + predictions + report)
python src/models/train_anomaly_detector_v3.py

# 4. Inspect anomaly context windows
python src/models/inspect_anomaly_windows.py --top 5

# 5. Generate all figures
python src/models/visualize_anomalies.py

# 6. Run full test suite
python -m pytest tests/ -q
```

**Expected runtimes on MacBook Air M2 (8 GB):**

| Script | Runtime |
|---|---|
| `prepare_anomaly_data.py` | ~3 s |
| `build_anomaly_features_v2.py` | ~5 s |
| `train_anomaly_detector_v3.py` | ~2 s |
| `visualize_anomalies.py` | ~60–90 s |

---

## Model Artifacts

| Artifact | Path |
|---|---|
| Per-building models | `artifacts/models/anomaly/<building>_iforest_v3.joblib` |
| Model metadata | `artifacts/models/anomaly/<building>_iforest_v3_meta.json` |
| V3 predictions | `data/processed/anomaly/anomaly_predictions_v3.csv` |
| V3 evaluation report | `data/processed/anomaly/anomaly_report_v3.json` |

Models are serialised with `joblib` and can be loaded for inference:

```python
import joblib, pandas as pd

model = joblib.load("artifacts/models/anomaly/Robin_office_Maryann_iforest_v3.joblib")

# Feature columns must match training exactly (see CONFIG in train_anomaly_detector_v3.py)
score = model.decision_function(X_new)  # lower = more anomalous
prediction = model.predict(X_new)       # -1 = anomaly, 1 = normal
```

---

## Limitations

1. **No ground-truth labels.** Anomaly counts reflect model sensitivity
   at `contamination=0.02`, not accuracy. A reading flagged by all three
   model versions may still be a valid operational state.

2. **Unsupervised threshold selection.** `contamination=0.02` was chosen
   as a reasonable prior (~2% expected anomaly rate for building energy
   data) but was not validated against domain knowledge for these specific
   buildings.

3. **Five buildings only.** The pipeline was developed and validated on
   5 of 1,636 BDG2 buildings. Generalisation to other building types,
   climates, or meter resolutions is unknown.

4. **No retraining schedule.** Models are trained on 2016–2017 data.
   Consumption patterns may shift seasonally or due to building changes.
   Models should be periodically retrained.

5. **Z-score threshold (3.5σ).** Calibrated heuristically. For
   `Robin_office_Maryann`, the z-score layer flags 66 readings that
   IForest misses — suggesting that building's anomalies are extreme
   statistical outliers that the tree ensemble underweights.

6. **Hourly resolution.** Sub-hourly or daily anomaly patterns are not
   captured. The 24-hour rolling window may miss very short events.

---

## Team Integration

### Running on a new machine

```bash
git clone https://github.com/ritikaaramani/Green-Building-Assistant
cd Green-Building-Assistant
conda activate green-building

# Fetch BDG2 data via DVC (requires DagsHub credentials)
dvc remote modify --local dagshub access_key_id  <YOUR_DAGSHUB_TOKEN>
dvc remote modify --local dagshub secret_access_key <YOUR_DAGSHUB_TOKEN>
dvc pull

# Then run reproduction commands above
```

### Adding a new building

1. Add the building column name to `BUILDINGS` in
   `src/data/prepare_anomaly_data.py` and rerun the script.
2. Rerun `build_anomaly_features_v2.py`.
3. Rerun `train_anomaly_detector_v3.py` — it trains one model per
   building automatically.

### Inference on new data

```python
from pathlib import Path
import joblib, pandas as pd

# Load a saved model
model = joblib.load(
    Path("artifacts/models/anomaly/Robin_office_Maryann_iforest_v3.joblib")
)

# Prepare features using the same pipeline as build_anomaly_features_v2.py
# then call:
predictions = model.predict(X_new[FEATURE_COLUMNS])  # -1 = anomaly
scores = model.decision_function(X_new[FEATURE_COLUMNS])
```

### DVC tracking

The following files are tracked by DVC:
- `dataset-sai/ENB2012_data.xlsx` (Model 1 raw data)
- `data/processed/energy_efficiency_clean.csv` (Model 1 cleaned data)

BDG2 processed outputs (`data/processed/anomaly/`) are currently not
DVC-tracked. To add them:

```bash
dvc add data/processed/anomaly/anomaly_predictions_v3.csv
git add data/processed/anomaly/anomaly_predictions_v3.csv.dvc .gitignore
```

---

## Branch

`feature/model3-energy-anomaly`

Last updated: 2026-10-15
