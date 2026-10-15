# Phase 1 Completion Report

## Requirements and Evidence

| Requirement | Evidence | Status |
| --- | --- | --- |
| Dataset versioning | `data/raw/bdg2.dvc` and `data/processed/bdg2.dvc` point to DVC directory outputs; current processed BDG2 output is 12 files | Local setup verified; pointer update remains uncommitted |
| Data cleaning | `src/green_building/data/bdg2_cleaning.py`, cleaning report, and unit tests | Complete |
| EDA | `src/green_building/eda_bdg2.py`, EDA report, and saved figures | Complete |
| Feature engineering | One-hour target, calendar/lags/rolling/metadata/weather features in Parquet; feature report | Complete |
| Model 2 baseline | DummyRegressor results in `reports/model2_results.json` | Complete |
| Model 2 XGBoost | Saved `models/energy_prediction/xgboost_model.json` and metrics | Complete |
| Test evaluation | Test-only report and four diagnostic plots under `reports/evaluation/` | Complete |
| Experiment tracking | `src/green_building/tracking/track_model2_mlflow.py` logged existing results without retraining | Complete |

## Existing Model Results

XGBoost test metrics are MAE `23.7885`, RMSE `152.9273`, and R2 `0.6946` over 6,457,408 rows. The DummyRegressor baseline test metrics are MAE `149.4197`, RMSE `276.9903`, and R2 `-0.0018`.

## Reproducibility

- Training and evaluation scripts use deterministic random state `42` and explicit chronological boundaries.
- Parquet data is read in batches; no raw, cleaned, or feature-engineered dataset is modified.
- Model, feature-column, result, and evaluation artifacts are saved under `models/` and `reports/`.
- Full test suite result before tracking additions: `18 passed, 15 skipped`.
- MLflow Skinny `3.1.4` uses the local file store `mlruns/` and logged two runs from existing results; it did not retrain.
- Tracking run IDs: DummyRegressor `592dedac005a4868be636de94ba23134`; XGBoost `3ad4721806e547aa8dabf28a9e4e757c`.

## Limitations and Missing Setup

- Full MLflow could not install because its `greenlet` dependency requires Microsoft C++ Build Tools on this Python 3.9 environment; `mlflow-skinny` is used for local tracking instead.
- No remote MLflow tracking server or model registry is configured; this is outside Phase 1 scope.
- The model used a deterministic one-million-row training sample covering only 156 of 1,578 buildings. Stratified building coverage is recommended before production use.
- DVC reports unrelated legacy missing outputs for the old ENB2012 dataset. The current processed BDG2 pointer is also a local uncommitted change after adding feature Parquet files; it must be committed and DVC-pushed later for a fresh checkout to reproduce those outputs. No commit or push was performed in this phase.
- No API, deployment, monitoring, Docker, CI/CD, or model registry work was added.