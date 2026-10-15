"""
Model 1 - shared helpers used by BOTH the baseline and the XGBoost script,
so both models see exactly the same data split and are scored the same way.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "processed" / "model1"
REPORT_DIR = REPO_ROOT / "reports" / "model1" / "models"
FIG_DIR = REPO_ROOT / "reports" / "model1" / "figures"
MODEL_DIR = REPO_ROOT / "artifacts" / "models" / "model1"
SPLIT_FILE = REPORT_DIR / "train_test_split.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data():
    """Return (dataframe, feature spec)."""
    spec = json.loads((DATA_DIR / "feature_spec.json").read_text())
    df = pd.read_csv(DATA_DIR / "model1_dataset.csv")
    return df, spec


def get_split(df, spec, drop_missing_target=True):
    """Same 80/20 split for every model and every experiment. Created once, then re-used.

    Each row is a different building (PUBID is unique), so the test set
    contains buildings the model has never seen. Stratified on the 3-class
    label so each class keeps the same proportion in train and test.

    Option C: buildings with a blank target (Moderate) are removed from BOTH
    sides AFTER the split, so the test buildings are a subset of the original
    test set (nothing moves between train and test).
    """
    ids = df[spec["id"]]
    if SPLIT_FILE.exists():
        split = pd.read_csv(SPLIT_FILE)
        print(f"Re-using saved split: {SPLIT_FILE.relative_to(REPO_ROOT)}")
    else:
        train_ids, _ = train_test_split(ids, test_size=TEST_SIZE, stratify=df["EFFICIENCY_CLASS"],
                                        random_state=RANDOM_STATE)
        split = pd.DataFrame({"PUBID": ids, "split": np.where(ids.isin(train_ids), "train", "test")})
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        split.to_csv(SPLIT_FILE, index=False)
        print(f"Created new split: {SPLIT_FILE.relative_to(REPO_ROOT)}")
    is_train = df[spec["id"]].map(dict(zip(split["PUBID"], split["split"]))) == "train"
    assert not set(df.loc[is_train, spec["id"]]) & set(df.loc[~is_train, spec["id"]]), "train/test overlap!"
    train, test = df[is_train], df[~is_train]

    if drop_missing_target:
        target = spec["target"]
        n_before = (len(train), len(test))
        train = train[train[target].notna()].astype({target: int})
        test = test[test[target].notna()].astype({target: int})
        if (len(train), len(test)) != n_before:
            print(f"Removed buildings with no target (Moderate): "
                  f"train {n_before[0]} -> {len(train)}, test {n_before[1]} -> {len(test)}")
    return train, test


def evaluate(name, y_true, y_pred, target_names):
    """All metrics in one dictionary + a readable report."""
    labels = sorted(target_names)
    names = [target_names[k] for k in labels]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class_f1": dict(zip(names, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).round(4).tolist())),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    text = classification_report(y_true, y_pred, labels=labels, target_names=names, digits=3, zero_division=0)
    return metrics, text


def plot_confusion(cm, target_names, title, filename):
    names = [target_names[k] for k in sorted(target_names)]
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(names)), names, rotation=20)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


class PeerRelative(BaseEstimator, TransformerMixin):
    """Peer-relative features (used by the final model, Step 12g).

    Adds <col>_REL = log(1 + value) - log(1 + median of the same building type).
    Example: WKHRS_REL > 0 means "open longer than a typical building of this type".
    The medians are learned in fit() from TRAINING buildings only, so no
    information leaks from the test set. Matches the label, which also compares
    each building with its own type.
    """

    def __init__(self, cols=None, group="PBA"):
        self.cols = cols
        self.group = group

    def fit(self, X, y=None):
        self.med_ = X.groupby(self.group)[self.cols].median()
        self.global_ = X[self.cols].median()          # fallback for unseen types
        return self

    def transform(self, X):
        X = X.copy()
        med = self.med_.reindex(X[self.group]).set_index(X.index).fillna(self.global_)
        for c in self.cols:
            X[c + "_REL"] = np.log1p(X[c].clip(lower=0)) - np.log1p(med[c].clip(lower=0))
        return X
