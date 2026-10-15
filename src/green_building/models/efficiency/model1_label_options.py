"""
Model 1 - Building Efficiency Assessment
Step 12e: Compare different LABEL designs using the TRAINING buildings only.

Why: with the original label (3 classes within building type) both models stop
at ~0.54. Here we test other honest label designs with 5-fold cross-validation
on the training set. The TEST set is NOT used, so the final evaluation stays fair.

Options tested:
  0  current      : 3 classes, EUI thirds within building type (Experiment 1 label)
  B  within_2cls  : 2 classes, above/below median EUI within building type
  C  within_ends  : within building type, only clearly High vs clearly Low (middle third removed)
  A  absolute_3cls: 3 classes, EUI thirds across ALL buildings (energy intensity)
  AB absolute_2cls: 2 classes, above/below median EUI across all buildings
  AC absolute_ends: across all buildings, only lowest vs highest third

Run from the repo root:
    python src/green_building/models/efficiency/model1_label_options.py
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from model1_common import RANDOM_STATE, REPO_ROOT, REPORT_DIR, get_split, load_data
from model1_train_xgboost import build_pipeline

# Fixed XGBoost settings (best from Experiment 1) - same for every option, so it's fair
XGB_SETTINGS = {"model__n_estimators": 500, "model__max_depth": 3, "model__learning_rate": 0.1,
                "model__subsample": 0.7, "model__colsample_bytree": 0.8}


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cum, v))


def make_labels(train):
    """All label options. New thresholds use TRAINING buildings only. NaN = building left out."""
    eui, w = train["EUI"], train["FINALWT"]
    within_med = pd.Series(np.nan, index=train.index)
    for _, g in train.groupby("PBA"):
        within_med.loc[g.index] = weighted_quantile(g["EUI"], g["FINALWT"], 0.5)
    t1, t2 = weighted_quantile(eui, w, 1 / 3), weighted_quantile(eui, w, 2 / 3)
    med = weighted_quantile(eui, w, 0.5)
    absolute_3 = pd.Series(np.where(eui <= t1, 0, np.where(eui <= t2, 1, 2)), index=train.index)
    current = train["EFFICIENCY_CLASS"]
    return {
        "0  current (3 cls, within type)": current,
        "B  within type, 2 classes": (eui > within_med).astype(int),
        "C  within type, High vs Low only": current.map({0: 0, 2: 1}),          # middle -> NaN
        "A  absolute, 3 classes": absolute_3,
        "AB absolute, 2 classes": (eui > med).astype(int),
        "AC absolute, lowest vs highest third": absolute_3.map({0: 0, 2: 1}),   # middle -> NaN
    }


def cv_score(X, y, cat, num):
    """5-fold CV on training data: accuracy, macro-F1 and the majority-class floor."""
    two_class = y.nunique() == 2
    params = dict(XGB_SETTINGS,
                  model__objective="binary:logistic" if two_class else "multi:softprob",
                  model__eval_metric="logloss" if two_class else "mlogloss")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    acc, f1, floor = [], [], []
    for tr, va in skf.split(X, y):
        pipe = clone(build_pipeline(cat, num)).set_params(**params)
        weights = compute_sample_weight("balanced", y.iloc[tr])
        pipe.fit(X.iloc[tr], y.iloc[tr], model__sample_weight=weights)
        pred = pipe.predict(X.iloc[va])
        acc.append(accuracy_score(y.iloc[va], pred))
        f1.append(f1_score(y.iloc[va], pred, average="macro"))
        floor.append((y.iloc[va] == y.iloc[tr].mode()[0]).mean())
    return np.mean(acc), np.std(acc), np.mean(f1), np.mean(floor)


def main():
    df, spec = load_data()
    cat, num = spec["categorical_features"], spec["numeric_features"]
    train, _ = get_split(df, spec, drop_missing_target=False)   # all train buildings; test ignored
    print(f"Feature set: {spec.get('feature_set', 'experiment_1')}  ({len(cat + num)} features)")
    print(f"Training buildings: {len(train)}  (test set NOT used)\n")

    rows = []
    for name, label in make_labels(train).items():
        keep = label.notna()
        X, y = train.loc[keep, cat + num], label[keep].astype(int)
        print(f"Running {name} ...")
        acc, acc_sd, f1, floor = cv_score(X, y, cat, num)
        rows.append({"option": name, "classes": y.nunique(), "buildings": int(keep.sum()),
                     "majority_floor": round(floor, 3), "cv_accuracy": round(acc, 3),
                     "cv_acc_std": round(acc_sd, 3), "cv_macro_f1": round(f1, 3)})

    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "label_options_cv.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== Label options: 5-fold CV on TRAINING buildings (XGBoost, same settings) ===")
    print(out.to_string(index=False))
    print(f"\nSaved: {(REPORT_DIR / 'label_options_cv.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
