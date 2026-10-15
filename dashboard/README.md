# Phase 1 Prediction Demo

The dashboard reads the existing evaluation payload and calls a local FastAPI
inference service for the Live Energy Prediction section. It does not retrain
the model or modify any dataset.

From the repository root, start the API in one terminal:

```powershell
python -m src.green_building.models.energy_prediction.inference_api
```

Start the static dashboard in a second terminal:

```powershell
python -m http.server 8080
```

Open `http://localhost:8080/dashboard/` in a browser.

The demo uses the existing `xgboost_model.json`, `dummy_regressor.pkl`,
`feature_columns.json`, and processed feature/target Parquet files. The model
was trained on a deterministic 1,000,000-row sample from 2016 covering 156 of
1,578 buildings. Predictions use only the selected engineered row at the
prediction timestamp; actual next-hour consumption is shown when available.

Verified example:

- Building: `Panther_education_Janis`
- Prediction timestamp: `2016-06-26 16:00:00`
- XGBoost prediction: `28.074270`
- Actual next-hour consumption: `22.762400`
- Absolute error: `5.311870`
- Training schema match: yes, 18 transformed columns