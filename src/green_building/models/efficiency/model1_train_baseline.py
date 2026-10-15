"""
Model 1 - Building Efficiency Assessment
Step 11: Baseline model = Logistic Regression.

Why a baseline? It gives a simple, honest reference point. XGBoost is only
worth its extra complexity if it clearly beats this. We also report a
"majority class" dummy model = the score you get by always guessing the
most common class (the absolute floor).

Run from the repo root:
    python src/green_building/models/efficiency/model1_train_baseline.py
"""

import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model1_common import (MODEL_DIR, RANDOM_STATE, REPO_ROOT, REPORT_DIR, evaluate,
                           get_split, load_data, plot_confusion)


def build_pipeline(cat, num):
    # Missing values are filled INSIDE the pipeline, so the fill values are learned
    # from the training data only (no information leaks from the test set).
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),  # + "was missing" flag
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=-1)),     # -1 = "not applicable"
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([("num", numeric, num), ("cat", categorical, cat)])
    model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
    return Pipeline([("prep", pre), ("model", model)])


def main():
    df, spec = load_data()
    cat, num = spec["categorical_features"], spec["numeric_features"]
    names = {int(k): v for k, v in spec["target_names"].items()}
    train, test = get_split(df, spec)
    X_train, y_train = train[cat + num], train[spec["target"]]
    X_test, y_test = test[cat + num], test[spec["target"]]
    print(f"Train: {len(train)} buildings   Test: {len(test)} buildings   Features: {len(cat + num)}")

    # 1. Floor: always predict the most common class
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_metrics, _ = evaluate("majority_class", y_test, dummy.predict(X_test), names)

    # 2. Baseline: Logistic Regression
    pipe = build_pipeline(cat, num)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    pipe.fit(X_train, y_train)
    metrics, text = evaluate("logistic_regression", y_test, pipe.predict(X_test), names)
    metrics["cv_macro_f1_mean"] = float(cv_scores.mean())
    metrics["cv_macro_f1_std"] = float(cv_scores.std())

    # Save everything
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "baseline_logreg.joblib")
    (REPORT_DIR / "baseline_metrics.json").write_text(
        json.dumps({"baseline": metrics, "majority_class": dummy_metrics}, indent=2))
    (REPORT_DIR / "baseline_classification_report.txt").write_text(text)
    plot_confusion(metrics["confusion_matrix"], names,
                   "Baseline (Logistic Regression) - test set", "fig7_confusion_baseline.png")

    print("\n=== Majority-class floor (always guesses the most common class) ===")
    print(f"accuracy {dummy_metrics['accuracy']:.3f}   macro-F1 {dummy_metrics['macro_f1']:.3f}")
    print("\n=== Baseline: Logistic Regression ===")
    print(f"5-fold CV macro-F1 on train: {metrics['cv_macro_f1_mean']:.3f} +/- {metrics['cv_macro_f1_std']:.3f}")
    print(f"Test accuracy          : {metrics['accuracy']:.3f}")
    print(f"Test balanced accuracy : {metrics['balanced_accuracy']:.3f}")
    print(f"Test macro-F1          : {metrics['macro_f1']:.3f}")
    print("\nPer-class report (test set):")
    print(text)
    print("Confusion matrix (rows = actual, columns = predicted):")
    for row in metrics["confusion_matrix"]:
        print("  ", row)
    print(f"\nSaved model : {(MODEL_DIR / 'baseline_logreg.joblib').relative_to(REPO_ROOT)}")
    print(f"Saved report: {(REPORT_DIR / 'baseline_metrics.json').relative_to(REPO_ROOT)}")
    print("Saved figure: reports/model1/figures/fig7_confusion_baseline.png")


if __name__ == "__main__":
    main()