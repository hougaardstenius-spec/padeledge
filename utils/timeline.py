# utils/timeline.py
import numpy as np

SHOT_COLORS = {
    "bandeja": "#2E86C1",
    "vibora": "#D68910",
    "derecha": "#27AE60",
    "reves": "#8E44AD",
    "smash": "#C0392B",
    "unknown": "#7F8C8D"
}

def build_timeline(predictions, timestamps):
    """
    Build structured timeline event objects.

    Args:
        predictions (list[str])
        timestamps (list[float])
    Returns:
        list[dict]
    """
    timeline = []

    for shot, t in zip(predictions, timestamps):
        timeline.append({
            "time": float(t),
            "shot": shot,
            "color": SHOT_COLORS.get(shot, SHOT_COLORS["unknown"])
        })

    return timeline
