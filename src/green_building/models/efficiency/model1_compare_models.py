"""
Model 1 - Building Efficiency Assessment
Step 13: Model comparison (table + chart) from the saved metric files.

Reads only saved results - no training, no test-set access:
  reports/model1/experiments/exp1/*_metrics.json   (Experiment 1: 3-class label)
  reports/model1/models/baseline_metrics.json      (Option C: floor + Logistic Regression)
  reports/model1/models/xgboost_metrics.json       (Option C: XGBoost)
  reports/model1/models/final_metrics.json         (Option C: final ensemble)

Outputs:
  reports/model1/models/model_comparison.csv
  reports/model1/models/model_comparison.md
  reports/model1/figures/fig13_model_comparison.png

Run from the repo root:
    python src/green_building/models/efficiency/model1_compare_models.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
MODELS = REPO_ROOT / "reports" / "model1" / "models"
EXP1 = REPO_ROOT / "reports" / "model1" / "experiments" / "exp1"
FIG = REPO_ROOT / "reports" / "model1" / "figures" / "fig13_model_comparison.png"

METRICS = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated categorical slots 1-3


def load(path):
    return json.loads(Path(path).read_text())


def row(experiment, model, m, n_test, cv=None, coverage=1.0, note=""):
    return {"experiment": experiment, "model": model,
            "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"], "cv_macro_f1": cv, "test_buildings": n_test,
            "coverage": coverage, "note": note}


def main():
    e1_base, e1_xgb = load(EXP1 / "baseline_metrics.json"), load(EXP1 / "xgboost_metrics.json")
    c_base, c_xgb = load(MODELS / "baseline_metrics.json"), load(MODELS / "xgboost_metrics.json")
    final = load(MODELS / "final_metrics.json")
    n1 = int(np.sum(e1_base["baseline"]["confusion_matrix"]))
    nc = int(np.sum(c_base["baseline"]["confusion_matrix"]))
    conf = final["test_confident_only"]

    e1, c = "Exp 1: 3 classes", "Final label: High vs Low"
    rows = [
        row(e1, "Majority floor", e1_base["majority_class"], n1),
        row(e1, "Logistic Regression (baseline)", e1_base["baseline"], n1, e1_base["baseline"].get("cv_macro_f1_mean")),
        row(e1, "XGBoost", e1_xgb, n1, e1_xgb.get("cv_macro_f1_mean")),
        row(c, "Majority floor", c_base["majority_class"], nc),
        row(c, "Logistic Regression (baseline)", c_base["baseline"], nc, c_base["baseline"].get("cv_macro_f1_mean")),
        row(c, "XGBoost", c_xgb, nc, c_xgb.get("cv_macro_f1_mean")),
        row(c, "Final ensemble", final["test_all_buildings"], nc, final["cv_train"]["macro_f1_mean"]),
        row(c, "Final ensemble, confident only", conf, conf["answered"], None, conf["coverage"],
            f"threshold {conf['threshold']}; {conf['uncertain']} buildings marked Uncertain"),
    ]
    df = pd.DataFrame(rows)
    df.round(4).to_csv(MODELS / "model_comparison.csv", index=False)

    # Markdown table for the report / README
    lines = ["# Model 1 — Model comparison (test set)", "",
             "| Experiment | Model | Accuracy | Balanced acc. | Macro-F1 | CV macro-F1 (train) | Test buildings | Coverage |",
             "|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        cv = f"{r.cv_macro_f1:.3f}" if pd.notna(r.cv_macro_f1) else "—"
        lines.append(f"| {r.experiment} | {r.model} | {r.accuracy:.3f} | {r.balanced_accuracy:.3f} | "
                     f"{r.macro_f1:.3f} | {cv} | {r.test_buildings} | {r.coverage*100:.1f}% |")
    lines += ["", "- Same saved train/test split for every model; the test set was used once per model.",
              "- Majority floor = always predicting the most common class.",
              f"- Confident only: the final ensemble gives a verdict only when its probability is ≥ "
              f"{conf['threshold']}; the remaining {conf['uncertain']} buildings are labelled 'Uncertain'."]
    (MODELS / "model_comparison.md").write_text("\n".join(lines) + "\n")

    # Chart: two panels (one per label design), grouped bars = 3 metrics
    short = {"Majority floor": "Majority\nfloor", "Logistic Regression (baseline)": "Baseline\n(LogReg)",
             "XGBoost": "XGBoost", "Final ensemble": "Final\nensemble",
             "Final ensemble, confident only": f"Final,\nconfident\n({conf['coverage']*100:.0f}% covered)"}
    panels = [(e1, f"Experiment 1 — 3 classes ({n1} test buildings)"),
              (c, f"Final label — High vs Low ({nc} test buildings)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True,
                             gridspec_kw={"width_ratios": [3, 5]})
    width = 0.26
    for ax, (exp, title) in zip(axes, panels):
        sub = df[df["experiment"] == exp].reset_index(drop=True)
        x = np.arange(len(sub))
        for i, ((key, label), color) in enumerate(zip(METRICS, COLORS)):
            bars = ax.bar(x + (i - 1) * width, sub[key], width, color=color, label=label,
                          edgecolor="white", linewidth=1.5)
            if key == "macro_f1":                      # label the primary metric only
                for b, v in zip(bars, sub[key]):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=8, color="#333333")
        ax.set_xticks(x, [short[m] for m in sub["model"]], fontsize=9)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#bdbdb8")
        ax.spines["bottom"].set_color("#bdbdb8")
        ax.tick_params(colors="#444444")
    axes[0].set_ylabel("Score on the test set")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    fig.suptitle("Model 1 — model comparison (numbers above bars = macro-F1)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 200)
    show = df[["experiment", "model", "accuracy", "balanced_accuracy", "macro_f1", "cv_macro_f1",
               "test_buildings", "coverage"]].round(3)
    print(show.to_string(index=False))
    print(f"\nSaved: {(MODELS / 'model_comparison.csv').relative_to(REPO_ROOT)}")
    print(f"Saved: {(MODELS / 'model_comparison.md').relative_to(REPO_ROOT)}")
    print(f"Saved: {FIG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
