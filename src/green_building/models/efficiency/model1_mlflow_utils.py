"""
Model 1 - shared MLflow settings (Step 14).

All Model 1 tracking data lives in artifacts/mlflow/ (ignored by Git):
  artifacts/mlflow/mlflow.db       run metadata (parameters, metrics, tags)
  artifacts/mlflow/artifacts/      files attached to runs (figures, reports, models)

Open the dashboard from the repo root:
    mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --port 5001
then go to http://127.0.0.1:5001 in your browser.
"""

import subprocess
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[4]
TRACK_DIR = REPO_ROOT / "artifacts" / "mlflow"
EXPERIMENT = "model1-building-efficiency"


def setup_mlflow():
    """Point MLflow at the local Model 1 store and select the Model 1 experiment."""
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    gitignore = TRACK_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# MLflow tracking data is local (not in Git)\n*\n!.gitignore\n")
    mlflow.set_tracking_uri(f"sqlite:///{TRACK_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(TRACK_DIR / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def git_commit():
    """Short hash of the current Git commit (recorded on every run for reproducibility)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def log_files(paths, folder=None):
    """Attach existing files to the active run (missing files are skipped with a note)."""
    for p in paths:
        p = Path(p)
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path=folder)
        else:
            print(f"  (skipped missing file: {p.relative_to(REPO_ROOT)})")
