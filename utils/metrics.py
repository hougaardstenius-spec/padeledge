import os
import json
from typing import Dict, Any, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")


def load_metrics_summary() -> Optional[Dict[str, Any]]:
    """
    Forsøger at læse models/metrics.json hvis den findes.
    Forventet format (du kan let tilpasse train_shot_model.py):
    {
      "timestamp": "...",
      "accuracy": 0.92,
      "macro_f1": 0.90,
      "per_class": {
          "bandeja": {"precision": 0.91, "recall": 0.88, "f1": 0.89},
          ...
      }
    }
    """
    if not os.path.exists(METRICS_PATH):
        return None

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
