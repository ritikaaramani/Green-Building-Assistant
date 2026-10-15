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


def get_split(df, spec):
    """Same 80/20 split for every model. Created once, then re-used from file.

    Each row is a different building (PUBID is unique), so the test set
    contains buildings the model has never seen. Stratified = each class
    keeps the same proportion in train and test.
    """
    ids, y = df[spec["id"]], df[spec["target"]]
    if SPLIT_FILE.exists():
        split = pd.read_csv(SPLIT_FILE)
        print(f"Re-using saved split: {SPLIT_FILE.relative_to(REPO_ROOT)}")
    else:
        train_ids, _ = train_test_split(ids, test_size=TEST_SIZE, stratify=y,
                                        random_state=RANDOM_STATE)
        split = pd.DataFrame({"PUBID": ids, "split": np.where(ids.isin(train_ids), "train", "test")})
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        split.to_csv(SPLIT_FILE, index=False)
        print(f"Created new split: {SPLIT_FILE.relative_to(REPO_ROOT)}")
    is_train = df[spec["id"]].map(dict(zip(split["PUBID"], split["split"]))) == "train"
    assert not set(df.loc[is_train, spec["id"]]) & set(df.loc[~is_train, spec["id"]]), "train/test overlap!"
    return df[is_train], df[~is_train]


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