"""
Production-ready shot detector module.

Handles:
- Lazy model loading (thread-safe)
- Input validation
- Stable prediction interface
- Clear error handling without crashing
"""

import joblib
import numpy as np
import threading
from typing import List, Optional, Union

MODEL_PATH = "models/shot_classifier.pkl"

# Thread lock for safe lazy loading
_model = None
_model_lock = threading.Lock()


# ----------------------------
# Internal model loader
# ----------------------------
def _load_model():
    """
    Loads the ML model (lazy, thread-safe).
    """
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:  # Double-check lock
                try:
                    _model = joblib.load(MODEL_PATH)
                except Exception as e:
                    raise RuntimeError(f"Failed to load model: {e}")

    return _model


# ----------------------------
# Public API
# ----------------------------
def predict_shot(keypoints: Union[List, np.ndarray]) -> Optional[str]:
    """
    Predicts shot class based on extracted keypoints.

    Args:
        keypoints (list or np.ndarray):
            Pose keypoints as list-of-lists or (N, 3)/(N, 4) array.

    Returns:
        str or None:
            The predicted shot label, or None if prediction fails.
    """

    # --- Validate input ---
    if keypoints is None:
        return None

    try:
        arr = np.array(keypoints, dtype=float)
    except Exception:
        return None

    if arr.size == 0:
        return None

    # Flatten to ML model format
    feature_vec = arr.flatten().reshape(1, -1)

    # --- Predict ---
    try:
        model = _load_model()
        pred = model.predict(feature_vec)[0]
        return str(pred)
    except Exception:
        return None


def predict_proba(keypoints: Union[List, np.ndarray]) -> Optional[dict]:
    """
    Returns class probabilities instead of a single prediction.

    Args:
        keypoints: pose keypoints

    Returns:
        dict or None:
            Example:
            {
                "bandeja": 0.72,
                "vibora": 0.20,
                "smash": 0.08
            }
    """

    if keypoints is None:
        return None

    try:
        arr = np.array(keypoints, dtype=float)
    except Exception:
        return None

    if arr.size == 0:
        return None

    feature_vec = arr.flatten().reshape(1, -1)

    try:
        model = _load_model()
        proba = model.predict_proba(feature_vec)[0]
        classes = model.classes_

        return {cls: float(p) for cls, p in zip(classes, proba)}

    except Exception:
        return None


# ----------------------------
# Utility
# ----------------------------
def is_model_loaded() -> bool:
    """
    Returns True if model is already in memory.
    Helpful for debugging or monitoring.
    """
    return _model is not None
