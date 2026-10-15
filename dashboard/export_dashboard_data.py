"""Export the existing Model 2 results into a browser-friendly dashboard payload."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "model2_results.json"
DESTINATION = Path(__file__).resolve().parent / "model2_dashboard_data.json"


def export() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload = {
        "generated": source["generated"],
        "source": "reports/model2_results.json",
        "training": source["training"],
        "parameters": source["parameters"],
        "models": source["models"],
        "split": {
            "train": "2016-01-01 to 2016-12-31",
            "validation": "2017-01-01 to 2017-06-30",
            "test": "2017-07-01 to 2017-12-31",
        },
        "unit_note": "Metrics use the dataset's original energy units; standardization has not been confirmed.",
    }
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    export()