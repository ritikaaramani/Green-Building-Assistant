"""
Model 1 - Building Efficiency Assessment
Step 14: Log the experiment history into MLflow.

The experiments in Steps 11-13 were run before MLflow was set up. This script
records them as MLflow runs FROM THE SAVED RESULT FILES (no retraining, no new
test-set access), so every experiment can be compared in one dashboard.
Each run is tagged with the Git commit that produced its results.

Runs created (experiment "model1-building-efficiency"):
  exp1_baseline_logreg, exp1_xgboost          - Experiment 1 (3-class label)
  label_options_cv (+ 6 child runs)           - Step 12e label comparison (training CV)
  optionC_baseline_logreg, optionC_xgboost    - Option C label
  final_ensemble                              - Step 12g final model (+ model file)

Re-running is safe: previously logged history runs are removed first.

Run from the repo root:
    python src/green_building/models/efficiency/model1_mlflow_log_history.py
Then open the dashboard:
    mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --port 5001
"""

import json
from pathlib import Path

import mlflow
import pandas as pd

from model1_mlflow_utils import REPO_ROOT, log_files, setup_mlflow

R = REPO_ROOT / "reports" / "model1"
MODELS, EXP1, FIGS = R / "models", R / "experiments" / "exp1", R / "figures"
ARTIFACT_MODELS = REPO_ROOT / "artifacts" / "models" / "model1"
SOURCE_TAG = "logged_from_saved_results"

LABEL_3CLASS = "3 classes: weighted EUI tertiles within building type (PBA)"
LABEL_C = "Option C: High vs Low efficiency within building type (Moderate excluded)"


def load(path):
    return json.loads(Path(path).read_text())


def test_metrics(m, prefix="test_"):
    return {f"{prefix}accuracy": m["accuracy"], f"{prefix}balanced_accuracy": m["balanced_accuracy"],
            f"{prefix}macro_f1": m["macro_f1"], f"{prefix}macro_precision": m["macro_precision"],
            f"{prefix}macro_recall": m["macro_recall"]}


def per_class(m):
    return {f"test_f1_{k.split()[0].lower()}": v for k, v in m["per_class_f1"].items()}


def start(name, commit, stage, extra_tags=None):
    tags = {"source": SOURCE_TAG, "git_commit": commit, "stage": stage, "model_owner": "Model 1"}
    tags.update(extra_tags or {})
    return mlflow.start_run(run_name=name, tags=tags)


def remove_old_history():
    old = mlflow.search_runs(filter_string=f"tags.source = '{SOURCE_TAG}'", output_format="list")
    for run in old:
        mlflow.delete_run(run.info.run_id)
    if old:
        print(f"Removed {len(old)} previously logged history runs (re-logging fresh).")


def main():
    setup_mlflow()
    remove_old_history()

    # ---------- Experiment 1 (3-class label, 26 features, 1,250 test buildings) ----------
    b1, x1 = load(EXP1 / "baseline_metrics.json"), load(EXP1 / "xgboost_metrics.json")
    with start("exp1_baseline_logreg", "6c94c73", "experiment_1"):
        mlflow.log_params({"label": LABEL_3CLASS, "model": "LogisticRegression", "class_weight": "balanced",
                           "n_features": 26, "test_buildings": 1250,
                           "preprocessing": "median impute + missing flags + scaling; one-hot categoricals"})
        mlflow.log_metrics({**test_metrics(b1["baseline"]), **per_class(b1["baseline"]),
                            "cv_macro_f1_mean": b1["baseline"]["cv_macro_f1_mean"],
                            "cv_macro_f1_std": b1["baseline"]["cv_macro_f1_std"],
                            **test_metrics(b1["majority_class"], "floor_")})
        log_files([EXP1 / "baseline_metrics.json", EXP1 / "baseline_classification_report.txt",
                   EXP1 / "fig7_confusion_baseline.png"])
    with start("exp1_xgboost", "ded6942", "experiment_1"):
        mlflow.log_params({"label": LABEL_3CLASS, "model": "XGBoost", "n_features": 26, "test_buildings": 1250,
                           "tuning": "RandomizedSearchCV 20 x 5-fold, macro-F1",
                           **{f"xgb_{k}": v for k, v in x1["best_params"].items()}})
        mlflow.log_metrics({**test_metrics(x1), **per_class(x1),
                            "cv_macro_f1_mean": x1["cv_macro_f1_mean"], "cv_macro_f1_std": x1["cv_macro_f1_std"]})
        log_files([EXP1 / "xgboost_metrics.json", EXP1 / "xgboost_classification_report.txt",
                   EXP1 / "xgboost_feature_importance.csv", EXP1 / "fig8_confusion_xgboost.png",
                   EXP1 / "fig9_xgboost_feature_importance.png"])

    # ---------- Step 12e: label options (training CV only) ----------
    opts = pd.read_csv(MODELS / "label_options_cv.csv")
    with start("label_options_cv", "9834fa5", "label_selection",
               {"note": "5-fold CV on training buildings only; test set not used"}) as parent:
        mlflow.log_params({"n_options": len(opts), "chosen_option": "C", "model": "XGBoost (fixed settings)"})
        log_files([MODELS / "label_options_cv.csv"])
        for r in opts.itertuples():
            with mlflow.start_run(run_name=f"label_{r.option.split()[0]}", nested=True,
                                  tags={"source": SOURCE_TAG, "git_commit": "9834fa5",
                                        "stage": "label_selection", "parent": parent.info.run_id}):
                mlflow.log_params({"option": r.option.strip(), "classes": int(r.classes),
                                   "buildings": int(r.buildings)})
                mlflow.log_metrics({"cv_accuracy": r.cv_accuracy, "cv_accuracy_std": r.cv_acc_std,
                                    "cv_macro_f1": r.cv_macro_f1, "majority_floor": r.majority_floor})

    # ---------- Option C: baseline + XGBoost (43 features, 814 test buildings) ----------
    bc, xc = load(MODELS / "baseline_metrics.json"), load(MODELS / "xgboost_metrics.json")
    with start("optionC_baseline_logreg", "5c7dc52", "experiment_2"):
        mlflow.log_params({"label": LABEL_C, "model": "LogisticRegression", "class_weight": "balanced",
                           "n_features": 43, "test_buildings": 814})
        mlflow.log_metrics({**test_metrics(bc["baseline"]), **per_class(bc["baseline"]),
                            "cv_macro_f1_mean": bc["baseline"]["cv_macro_f1_mean"],
                            "cv_macro_f1_std": bc["baseline"]["cv_macro_f1_std"],
                            **test_metrics(bc["majority_class"], "floor_")})
        log_files([MODELS / "baseline_metrics.json", MODELS / "baseline_classification_report.txt",
                   FIGS / "fig7_confusion_baseline.png"])
    with start("optionC_xgboost", "5c7dc52", "experiment_2"):
        mlflow.log_params({"label": LABEL_C, "model": "XGBoost", "n_features": 43, "test_buildings": 814,
                           "tuning": "RandomizedSearchCV 20 x 5-fold, macro-F1",
                           **{f"xgb_{k}": v for k, v in xc["best_params"].items()}})
        mlflow.log_metrics({**test_metrics(xc), **per_class(xc),
                            "cv_macro_f1_mean": xc["cv_macro_f1_mean"], "cv_macro_f1_std": xc["cv_macro_f1_std"]})
        log_files([MODELS / "xgboost_metrics.json", MODELS / "xgboost_classification_report.txt",
                   MODELS / "xgboost_feature_importance.csv", FIGS / "fig8_confusion_xgboost.png",
                   FIGS / "fig9_xgboost_feature_importance.png"])

    # ---------- Final ensemble ----------
    f = load(MODELS / "final_metrics.json")
    t, c, cv = f["test_all_buildings"], f["test_confident_only"], f["cv_train"]
    with start("final_ensemble", "5c7dc52", "final", {"selected_for_deployment": "true"}):
        mlflow.log_params({"label": LABEL_C, "model": "soft-vote LogisticRegression + XGBoost",
                           "n_features": 43, "n_peer_relative_features": 15, "test_buildings": 814,
                           "confidence_threshold": f["confidence_threshold"],
                           **{f"xgb_{k}": v for k, v in f["xgb_params"].items()}})
        mlflow.log_metrics({**test_metrics(t), **per_class(t),
                            "cv_accuracy_mean": cv["accuracy_mean"], "cv_accuracy_std": cv["accuracy_std"],
                            "cv_balanced_accuracy_mean": cv["balanced_accuracy_mean"],
                            "cv_macro_f1_mean": cv["macro_f1_mean"], "cv_macro_f1_std": cv["macro_f1_std"],
                            "confident_coverage": c["coverage"], "confident_accuracy": c["accuracy"],
                            "confident_balanced_accuracy": c["balanced_accuracy"],
                            "confident_macro_f1": c["macro_f1"], "confident_uncertain_count": c["uncertain"],
                            "moderate_marked_uncertain": f["moderate_test_buildings_marked_uncertain"],
                            "clear_marked_uncertain": f["clear_test_buildings_marked_uncertain"]})
        log_files([MODELS / "final_metrics.json", MODELS / "final_classification_report.txt",
                   MODELS / "final_confidence_table_cv.csv", MODELS / "model_comparison.md",
                   FIGS / "fig11_confusion_final.png", FIGS / "fig12_confidence_coverage.png",
                   FIGS / "fig13_model_comparison.png"])
        log_files([ARTIFACT_MODELS / "final_ensemble.joblib"], folder="model")

    runs = mlflow.search_runs(filter_string=f"tags.source = '{SOURCE_TAG}'")
    print(f"\nLogged {len(runs)} runs to MLflow experiment 'model1-building-efficiency':")
    show = runs[["tags.mlflow.runName", "tags.stage", "tags.git_commit"]].copy()
    for col in ["metrics.test_accuracy", "metrics.test_macro_f1", "metrics.cv_macro_f1"]:
        if col in runs:
            show[col.split(".")[1]] = runs[col].round(3)
    print(show.iloc[::-1].to_string(index=False))
    print("\nOpen the dashboard:\n  mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --port 5001\n"
          "  then visit http://127.0.0.1:5001")


if __name__ == "__main__":
    main()
