"""
Model 1 - Building Efficiency Assessment
Step 12: Main model = XGBoost Classifier.

Uses EXACTLY the same train/test split, features and metrics as the baseline
(via model1_common.py), so the comparison is fair.

Steps:
  1. Tune a few XGBoost settings with 5-fold cross-validation on the TRAIN set only.
  2. Refit the best settings on the full train set.
  3. Evaluate once on the untouched TEST set.
  4. Save model, metrics, confusion matrix and permutation feature importance.

Run from the repo root:
    python src/green_building/models/efficiency/model1_train_xgboost.py
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from model1_common import (FIG_DIR, MODEL_DIR, RANDOM_STATE, REPO_ROOT, REPORT_DIR, evaluate,
                           get_split, load_data, plot_confusion)

# Settings tried during tuning (20 random combinations x 5 folds)
PARAM_SPACE = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [3, 4, 5, 6],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__min_child_weight": [1, 3, 5],
    "model__reg_lambda": [1, 5, 10],
}


def build_pipeline(cat, num):
    # XGBoost handles missing numbers by itself (it learns which way to send NaN),
    # so numeric features pass through unchanged - no scaling needed for trees.
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=-1)),  # -1 = "not applicable"
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([("num", "passthrough", num), ("cat", categorical, cat)])
    model = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                          tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("prep", pre), ("model", model)])


def feature_importance(pipe, X_test, y_test):
    """Permutation importance on the TEST set: shuffle one feature at a time and
    measure how much macro-F1 drops. Big drop = the model relies on that feature.
    Works on the original 26 features (not one-hot pieces), so categories with
    many levels are not unfairly inflated."""
    result = permutation_importance(pipe, X_test, y_test, scoring="f1_macro", n_repeats=10,
                                    random_state=RANDOM_STATE, n_jobs=1)
    imp = pd.DataFrame({"importance": result.importances_mean, "std": result.importances_std},
                       index=X_test.columns)
    return imp.sort_values("importance", ascending=False)


def main():
    df, spec = load_data()
    cat, num = spec["categorical_features"], spec["numeric_features"]
    names = {int(k): v for k, v in spec["target_names"].items()}
    train, test = get_split(df, spec)
    X_train, y_train = train[cat + num], train[spec["target"]]
    X_test, y_test = test[cat + num], test[spec["target"]]
    print(f"Train: {len(train)} buildings   Test: {len(test)} buildings   Features: {len(cat + num)}")

    # Class imbalance: give each class equal total weight during training
    weights = compute_sample_weight("balanced", y_train)

    # 1. Tune on TRAIN only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(build_pipeline(cat, num), PARAM_SPACE, n_iter=20, cv=cv,
                                scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=1, verbose=1)
    print("\nTuning XGBoost (20 settings x 5 folds = 100 fits, ~1-3 minutes)...")
    search.fit(X_train, y_train, model__sample_weight=weights)
    best = search.best_estimator_   # already refit on the full train set
    cv_std = search.cv_results_["std_test_score"][search.best_index_]

    # 2. Evaluate ONCE on the untouched test set
    metrics, text = evaluate("xgboost", y_test, best.predict(X_test), names)
    metrics["cv_macro_f1_mean"] = float(search.best_score_)
    metrics["cv_macro_f1_std"] = float(cv_std)
    metrics["best_params"] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

    # 3. Feature importance
    imp = feature_importance(best, X_test, y_test)

    # Save everything
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_DIR / "xgboost.joblib")
    (REPORT_DIR / "xgboost_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (REPORT_DIR / "xgboost_classification_report.txt").write_text(text)
    imp.round(4).to_csv(REPORT_DIR / "xgboost_feature_importance.csv")
    plot_confusion(metrics["confusion_matrix"], names, "XGBoost - test set", "fig8_confusion_xgboost.png")

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top["importance"], xerr=top["std"], color="#2f6db5", height=0.7,
            error_kw=dict(ecolor="#888888", lw=1))
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_title("XGBoost permutation importance (test set, top 15)", loc="left", fontsize=11)
    ax.set_xlabel("Drop in macro-F1 when the feature is shuffled")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_xgboost_feature_importance.png", dpi=150)
    plt.close(fig)

    # Print results
    base = json.loads((REPORT_DIR / "baseline_metrics.json").read_text())["baseline"]
    print("\nBest settings:", metrics["best_params"])
    print(f"\n=== XGBoost ===")
    print(f"5-fold CV macro-F1 on train: {metrics['cv_macro_f1_mean']:.3f} +/- {metrics['cv_macro_f1_std']:.3f}")
    print(f"Test accuracy          : {metrics['accuracy']:.3f}   (baseline {base['accuracy']:.3f})")
    print(f"Test balanced accuracy : {metrics['balanced_accuracy']:.3f}   (baseline {base['balanced_accuracy']:.3f})")
    print(f"Test macro-F1          : {metrics['macro_f1']:.3f}   (baseline {base['macro_f1']:.3f})")
    print("\nPer-class report (test set):")
    print(text)
    print("Confusion matrix (rows = actual, columns = predicted):")
    for row in metrics["confusion_matrix"]:
        print("  ", row)
    print("\nTop 10 features (drop in test macro-F1 when shuffled):")
    print(imp.head(10).round(3).to_string())
    print(f"\nSaved model : {(MODEL_DIR / 'xgboost.joblib').relative_to(REPO_ROOT)}")
    print("Saved report: reports/model1/models/xgboost_metrics.json")
    print("Saved figures: fig8_confusion_xgboost.png, fig9_xgboost_feature_importance.png")


if __name__ == "__main__":
    main()