"""Log existing Model 2 results and artifacts to a local MLflow experiment.

This module intentionally does not retrain models. It reads the existing JSON
results and logs the saved model files and reports as run artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS = Path("reports/model2_results.json")
DEFAULT_EVALUATION = Path("reports/model2_evaluation_report.txt")
DEFAULT_MODEL_DIR = Path("models/energy_prediction")
DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_EXPERIMENT = "green-building-model2"


def _flatten_metrics(results: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for model_name, splits in results["models"].items():
        for split_name, values in splits.items():
            for metric_name in ("mae", "rmse", "r2"):
                metrics[f"{model_name}_{split_name}_{metric_name}"] = float(values[metric_name])
    return metrics


def track_existing_results(
    results_path: str | Path = DEFAULT_RESULTS,
    evaluation_path: str | Path = DEFAULT_EVALUATION,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> dict[str, Any]:
    """Create two MLflow runs from existing Model 2 results and artifacts."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow is not installed. Install requirements-mlflow.txt first.") from exc

    results_path, evaluation_path, model_dir = map(Path, [results_path, evaluation_path, model_dir])
    results = json.loads(results_path.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    common_params = {
        "train_rows": results["training"]["train_rows"],
        "max_train_rows": results["max_train_rows"],
        "train_start": "2016-01-01",
        "train_end_exclusive": "2017-01-01",
        "validation_start": "2017-01-01",
        "validation_end_exclusive": "2017-07-01",
        "test_start": "2017-07-01",
        "test_end_exclusive": "2018-01-01",
        "dropped_feature": "precipDepth6HR",
        "categorical_feature": "primaryspaceusage",
        "tracking_source": str(results_path),
    }
    runs: dict[str, str] = {}
    for model_name, split_results in results["models"].items():
        parameters = dict(common_params)
        parameters["model"] = model_name
        if model_name == "xgboost":
            parameters.update({key: value for key, value in results["parameters"].items()})
        with mlflow.start_run(run_name=f"model2-{model_name}") as run:
            mlflow.log_params(parameters)
            metrics = {
                f"{split}_{metric}": float(value)
                for split, values in split_results.items()
                for metric, value in values.items()
                if metric in {"mae", "rmse", "r2"}
            }
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"phase": "phase-1", "dataset": "BDG2", "retrained": "false"})
            for artifact in model_dir.iterdir():
                if artifact.is_file():
                    mlflow.log_artifact(str(artifact), artifact_path="models")
            mlflow.log_artifact(str(results_path), artifact_path="reports")
            if evaluation_path.exists():
                mlflow.log_artifact(str(evaluation_path), artifact_path="reports")
            runs[model_name] = run.info.run_id
    return {"tracking_uri": tracking_uri, "experiment": experiment_name, "runs": runs, "metrics": _flatten_metrics(results)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    print(json.dumps(track_existing_results(args.results, args.evaluation, args.model_dir, args.tracking_uri, args.experiment), indent=2))


if __name__ == "__main__":
    main()