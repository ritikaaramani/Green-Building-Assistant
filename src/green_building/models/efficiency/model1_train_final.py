"""
Model 1 - Building Efficiency Assessment
Step 12g: FINAL model.

Decisions (all made on TRAINING data / cross-validation, never on the test set):
  - Label     : Option C - clearly High (0) vs clearly Low (1) efficiency within
                building type; Moderate buildings excluded            (Step 12e)
  - Model     : soft-vote ensemble of Logistic Regression + XGBoost   (Step 12f)
  - Features  : 43 Experiment-2 features + 15 peer-relative features
                (value compared with the median of the same building type)
  - Confidence: the model gives a verdict only if its probability for the
                chosen class is >= 0.65; otherwise "Uncertain"        (Step 12f)

Steps:
  1. 5-fold CV on TRAIN: out-of-fold probabilities -> CV metrics and the
     coverage/accuracy table for different confidence thresholds.
  2. Fit on the full TRAIN set.
  3. Evaluate ONCE on the TEST set: all buildings (threshold 0.5) and
     confident-only (threshold 0.65, with coverage).
  4. Save model, metrics, report, figures.
  5. Log the run to MLflow (Step 14) if MLflow is installed.

Run from the repo root:
    python src/green_building/models/efficiency/model1_train_final.py
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from model1_common import (FIG_DIR, MODEL_DIR, RANDOM_STATE, REPO_ROOT, REPORT_DIR, PeerRelative,
                           evaluate, get_split, load_data, plot_confusion)

CONFIDENCE = 0.65          # chosen in Step 12f from training CV (79% coverage, ~0.86 accuracy)
CV_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# Columns compared with the median of the same building type (PBA)
PEER_COLS = ["WKHRS", "NWKER", "SQFT", "LTOHRP", "LTNHRP", "WORKERS_PER_1000SQFT",
             "PCS_PER_1000SQFT", "HEATP", "COOLP", "NFLOOR", "YRCONC", "HDD65", "CDD65",
             "GLSSPC", "FLCEILHT"]

# XGBoost settings = best from the Option C tuning in Step 12d (training CV)
XGB_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.85,
                  colsample_bytree=0.8, min_child_weight=5, reg_lambda=10)


def build_model(cat, num, peer, pos_weight):
    rel = [c + "_REL" for c in peer]
    lr_prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), num + rel),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    xgb_prep = ColumnTransformer([
        ("num", "passthrough", num + rel),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
    logreg = Pipeline([("peer", PeerRelative(peer)), ("prep", lr_prep),
                       ("model", LogisticRegression(max_iter=4000, class_weight="balanced",
                                                    random_state=RANDOM_STATE))])
    xgb = Pipeline([("peer", PeerRelative(peer)), ("prep", xgb_prep),
                    ("model", XGBClassifier(tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
                                            scale_pos_weight=pos_weight, **XGB_PARAMS))])
    # Soft vote = average the two models' probabilities
    return VotingClassifier([("logreg", logreg), ("xgboost", xgb)], voting="soft")


def selective(y_true, proba_low, threshold):
    """Metrics when the model only answers if confident (max probability >= threshold)."""
    conf = np.maximum(proba_low, 1 - proba_low)
    answered = conf >= threshold
    pred = (proba_low[answered] >= 0.5).astype(int)
    yt = np.asarray(y_true)[answered]
    return {"threshold": threshold, "coverage": float(answered.mean()),
            "answered": int(answered.sum()), "uncertain": int((~answered).sum()),
            "accuracy": float(accuracy_score(yt, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
            "macro_f1": float(f1_score(yt, pred, average="macro"))}


def main():
    df, spec = load_data()
    cat, num = spec["categorical_features"], spec["numeric_features"]
    peer = [c for c in PEER_COLS if c in num]
    names = {int(k): v for k, v in spec["target_names"].items()}
    train, test = get_split(df, spec)
    X_train, y_train = train[cat + num], train[spec["target"]]
    X_test, y_test = test[cat + num], test[spec["target"]]
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()   # balances High vs Low for XGBoost
    print(f"Train: {len(train)}   Test: {len(test)}   Features: {len(cat + num)} + {len(peer)} peer-relative")
    print(f"Confidence threshold: {CONFIDENCE}")

    model = build_model(cat, num, peer, pos_weight)

    # ---------- 1. Cross-validation on TRAIN only ----------
    print("\n[1/3] 5-fold cross-validation on training buildings ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y_train))
    fold_acc, fold_bal, fold_f1 = [], [], []
    for tr, va in cv.split(X_train, y_train):
        m = clone(model).fit(X_train.iloc[tr], y_train.iloc[tr])
        oof[va] = m.predict_proba(X_train.iloc[va])[:, 1]
        pred = (oof[va] >= 0.5).astype(int)
        fold_acc.append(accuracy_score(y_train.iloc[va], pred))
        fold_bal.append(balanced_accuracy_score(y_train.iloc[va], pred))
        fold_f1.append(f1_score(y_train.iloc[va], pred, average="macro"))
    cv_summary = {"accuracy_mean": float(np.mean(fold_acc)), "accuracy_std": float(np.std(fold_acc)),
                  "balanced_accuracy_mean": float(np.mean(fold_bal)),
                  "macro_f1_mean": float(np.mean(fold_f1)), "macro_f1_std": float(np.std(fold_f1))}
    cv_table = pd.DataFrame([selective(y_train, oof, t) for t in CV_THRESHOLDS])

    # ---------- 2. Fit on full TRAIN ----------
    print("[2/3] Training final model on all training buildings ...")
    model.fit(X_train, y_train)

    # ---------- 3. Evaluate ONCE on TEST ----------
    print("[3/3] Evaluating once on the test set ...")
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_all = (proba_test >= 0.5).astype(int)
    metrics, report_text = evaluate("final_ensemble", y_test, pred_all, names)
    test_conf = selective(y_test, proba_test, CONFIDENCE)

    # Moderate test buildings (not used for scoring): how often does the model say "Uncertain"?
    _, test_all = get_split(df, spec, drop_missing_target=False)
    moderate = test_all[test_all["EFFICIENCY_CLASS"] == 1]
    p_mod = model.predict_proba(moderate[cat + num])[:, 1]
    moderate_uncertain = float((np.maximum(p_mod, 1 - p_mod) < CONFIDENCE).mean())
    clear_uncertain = 1 - test_conf["coverage"]

    # ---------- Save ----------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "confidence_threshold": CONFIDENCE, "features": cat + num,
                 "peer_cols": peer, "target_names": names}, MODEL_DIR / "final_ensemble.joblib")
    out = {"label": spec["label_method"], "model": "soft-vote LogisticRegression + XGBoost, peer-relative features",
           "xgb_params": XGB_PARAMS, "confidence_threshold": CONFIDENCE,
           "cv_train": cv_summary, "test_all_buildings": metrics, "test_confident_only": test_conf,
           "moderate_test_buildings_marked_uncertain": moderate_uncertain,
           "clear_test_buildings_marked_uncertain": clear_uncertain}
    (REPORT_DIR / "final_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    (REPORT_DIR / "final_classification_report.txt").write_text(report_text)
    cv_table.round(4).to_csv(REPORT_DIR / "final_confidence_table_cv.csv", index=False)
    plot_confusion(metrics["confusion_matrix"], names, "Final ensemble - test set",
                   "fig11_confusion_final.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(cv_table["coverage"] * 100, cv_table["accuracy"], marker="o", color="#2f6db5")
    for _, r in cv_table.iterrows():
        ax.annotate(f"{r['threshold']:.2f}", (r["coverage"] * 100, r["accuracy"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8, color="#555555")
    ax.scatter([test_conf["coverage"] * 100], [test_conf["accuracy"]], color="#d1495b", zorder=3,
               label=f"Test set at threshold {CONFIDENCE}")
    ax.set_xlabel("Buildings given a verdict (%)")
    ax.set_ylabel("Accuracy on those buildings")
    ax.set_title("Confidence vs coverage (training CV; labels = threshold)", loc="left", fontsize=11)
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig12_confidence_coverage.png", dpi=150)
    plt.close(fig)

    # ---------- Print ----------
    base = json.loads((REPORT_DIR / "baseline_metrics.json").read_text())
    floor = base["majority_class"]
    print("\n=== 5-fold CV on TRAIN (final ensemble) ===")
    print(f"accuracy {cv_summary['accuracy_mean']:.3f} +/- {cv_summary['accuracy_std']:.3f}   "
          f"balanced {cv_summary['balanced_accuracy_mean']:.3f}   macro-F1 {cv_summary['macro_f1_mean']:.3f}")
    print("\nConfidence table (training CV):")
    print(cv_table[["threshold", "coverage", "accuracy", "balanced_accuracy"]].round(3).to_string(index=False))

    print("\n=== TEST SET - all clear-case buildings (threshold 0.5) ===")
    print(f"{'':24s}{'accuracy':>10}{'balanced':>10}{'macro-F1':>10}")
    print(f"{'Majority floor':24s}{floor['accuracy']:>10.3f}{floor['balanced_accuracy']:>10.3f}{floor['macro_f1']:>10.3f}")
    print(f"{'Baseline (LogReg)':24s}{base['baseline']['accuracy']:>10.3f}"
          f"{base['baseline']['balanced_accuracy']:>10.3f}{base['baseline']['macro_f1']:>10.3f}")
    print(f"{'FINAL ensemble':24s}{metrics['accuracy']:>10.3f}{metrics['balanced_accuracy']:>10.3f}"
          f"{metrics['macro_f1']:>10.3f}")
    print("\n" + report_text)
    print("Confusion matrix (rows = actual, columns = predicted):")
    for row in metrics["confusion_matrix"]:
        print("  ", row)

    print(f"\n=== TEST SET - confident predictions only (threshold {CONFIDENCE}) ===")
    print(f"Verdict given for {test_conf['answered']} of {len(y_test)} buildings "
          f"({test_conf['coverage']*100:.1f}%), {test_conf['uncertain']} marked 'Uncertain'")
    print(f"accuracy {test_conf['accuracy']:.3f}   balanced {test_conf['balanced_accuracy']:.3f}   "
          f"macro-F1 {test_conf['macro_f1']:.3f}")
    print(f"\nSanity check - share marked 'Uncertain':")
    print(f"  Moderate test buildings (true borderline): {moderate_uncertain*100:.1f}%")
    print(f"  Clear High/Low test buildings          : {clear_uncertain*100:.1f}%")

    print(f"\nSaved model : {(MODEL_DIR / 'final_ensemble.joblib').relative_to(REPO_ROOT)}")
    print("Saved report: reports/model1/models/final_metrics.json, final_classification_report.txt,"
          " final_confidence_table_cv.csv")
    print("Saved figures: fig11_confusion_final.png, fig12_confidence_coverage.png")

    # ---------- 5. MLflow tracking (Step 14) ----------
    try:
        import mlflow
        from model1_mlflow_utils import git_commit, log_files, setup_mlflow
    except ImportError:
        print("\n(MLflow not installed - run not tracked)")
        return
    setup_mlflow()
    tags = {"source": "training_script", "git_commit": git_commit(), "stage": "final"}
    with mlflow.start_run(run_name="final_ensemble_train", tags=tags) as run:
        mlflow.log_params({"label": spec["label_method"], "model": "soft-vote LogisticRegression + XGBoost",
                           "n_features": len(cat + num), "n_peer_relative_features": len(peer),
                           "train_buildings": len(train), "test_buildings": len(test),
                           "confidence_threshold": CONFIDENCE,
                           **{f"xgb_{k}": v for k, v in XGB_PARAMS.items()}})
        mlflow.log_metrics({"test_accuracy": metrics["accuracy"],
                            "test_balanced_accuracy": metrics["balanced_accuracy"],
                            "test_macro_f1": metrics["macro_f1"],
                            "cv_accuracy_mean": cv_summary["accuracy_mean"],
                            "cv_accuracy_std": cv_summary["accuracy_std"],
                            "cv_macro_f1_mean": cv_summary["macro_f1_mean"],
                            "cv_macro_f1_std": cv_summary["macro_f1_std"],
                            "confident_coverage": test_conf["coverage"],
                            "confident_accuracy": test_conf["accuracy"],
                            "confident_balanced_accuracy": test_conf["balanced_accuracy"],
                            "confident_macro_f1": test_conf["macro_f1"],
                            "moderate_marked_uncertain": moderate_uncertain})
        log_files([REPORT_DIR / "final_metrics.json", REPORT_DIR / "final_classification_report.txt",
                   REPORT_DIR / "final_confidence_table_cv.csv", FIG_DIR / "fig11_confusion_final.png",
                   FIG_DIR / "fig12_confidence_coverage.png"])
        log_files([MODEL_DIR / "final_ensemble.joblib"], folder="model")
    print(f"\nLogged to MLflow: run 'final_ensemble_train' ({run.info.run_id[:8]}), "
          f"commit {tags['git_commit']}")


if __name__ == "__main__":
    main()
