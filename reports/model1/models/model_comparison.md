# Model 1 — Model comparison (test set)

| Experiment | Model | Accuracy | Balanced acc. | Macro-F1 | CV macro-F1 (train) | Test buildings | Coverage |
|---|---|---|---|---|---|---|---|
| Exp 1: 3 classes | Majority floor | 0.414 | 0.333 | 0.195 | — | 1250 | 100.0% |
| Exp 1: 3 classes | Logistic Regression (baseline) | 0.541 | 0.538 | 0.528 | 0.519 | 1250 | 100.0% |
| Exp 1: 3 classes | XGBoost | 0.543 | 0.534 | 0.530 | 0.526 | 1250 | 100.0% |
| Final label: High vs Low | Majority floor | 0.636 | 0.500 | 0.389 | — | 814 | 100.0% |
| Final label: High vs Low | Logistic Regression (baseline) | 0.794 | 0.784 | 0.780 | 0.783 | 814 | 100.0% |
| Final label: High vs Low | XGBoost | 0.794 | 0.776 | 0.776 | 0.781 | 814 | 100.0% |
| Final label: High vs Low | Final ensemble | 0.808 | 0.792 | 0.793 | 0.796 | 814 | 100.0% |
| Final label: High vs Low | Final ensemble, confident only | 0.873 | 0.859 | 0.861 | — | 632 | 77.6% |

- Same saved train/test split for every model; the test set was used once per model.
- Majority floor = always predicting the most common class.
- Confident only: the final ensemble gives a verdict only when its probability is ≥ 0.65; the remaining 182 buildings are labelled 'Uncertain'.
