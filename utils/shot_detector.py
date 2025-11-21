import os
import joblib
import numpy as np
import time
import subprocess
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "shot_classifier.pkl")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "scripts", "train_shot_model.py")
LOG_PATH = os.path.join(BASE_DIR, "models", "auto_retrain.log")


def is_model_valid(model_path: str) -> bool:
    """Checks if model file exists, is non-empty, and can be loaded."""
    if not os.path.exists(model_path):
        return False

    if os.path.getsize(model_path) < 4096:  # 4KB minimum
        return False

    try:
        joblib.load(model_path)
        return True
    except Exception:
        return False


def auto_retrain():
    """Runs training script automatically if model is missing or corrupted."""
    log_msg = f"[{time.ctime()}] AUTO-RETRAIN triggered...\n"
    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT],
            capture_output=True,
            text=True,
            cwd=BASE_DIR,
        )
        log_msg += "STDOUT:\n" + (result.stdout or "") + "\n"
        log_msg += "STDERR:\n" + (result.stderr or "") + "\n"
    except Exception as e:
        log_msg += f"ERROR during retraining: {e}\n"

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_msg)

    return is_model_valid(MODEL_PATH)


class ShotDetector:
    def __init__(self):
        """Loads model safely — retrains automatically if missing or broken."""
        if not is_model_valid(MODEL_PATH):
            print("⚠️ Model not valid — retraining...")
            ok = auto_retrain()
            if not ok:
                raise RuntimeError(
                    f"❌ Auto-retrain failed. Check log at: {LOG_PATH}"
                )

        self.model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded: {MODEL_PATH}")

    def predict(self, feature_vector):
        """Predicts a single shot label."""
        fv = np.array(feature_vector).flatten().reshape(1, -1)
        return self.model.predict(fv)[0]

    def analyze(self, video_path: str):
        """
        Placeholder — your real analyzer in training_dashboard V2.
        """
        raise NotImplementedError("You said you'd implement analyze() separately.")
