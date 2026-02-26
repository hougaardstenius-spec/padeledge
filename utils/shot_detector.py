import os
import joblib
import numpy as np
import time
import subprocess
import sys
import cv2

from utils.video_processor import (
    extract_keypoints_from_video,
    summarize_feature_sequence,
    MODEL_FRAMES,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.getenv(
    "PADELEDGE_MODEL_PATH", os.path.join(BASE_DIR, "models", "shot_classifier.pkl")
)
TRAIN_SCRIPT = os.getenv(
    "PADELEDGE_TRAIN_SCRIPT", os.path.join(BASE_DIR, "scripts", "train_shot_model.py")
)
LOG_PATH = os.getenv(
    "PADELEDGE_AUTO_RETRAIN_LOG", os.path.join(BASE_DIR, "models", "auto_retrain.log")
)


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
        Baseline analyzer using sliding windows over motion features.
        Returns: (predicted_labels, timestamps_sec, representative_keypoints)
        """
        keypoint_seq = extract_keypoints_from_video(video_path)
        if keypoint_seq is None or len(keypoint_seq) == 0:
            return [], [], []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        n_frames = len(keypoint_seq)
        window_frames = max(int(fps * 0.8), 8)
        stride = max(window_frames // 2, 4)

        preds = []
        timestamps = []
        rep_keypoints = []

        if n_frames <= window_frames:
            features = summarize_feature_sequence(
                keypoint_seq, target_frames=MODEL_FRAMES
            )
            if features is not None:
                preds.append(self.predict(features))
                timestamps.append(float(n_frames / 2.0 / fps))
                rep_keypoints.append(keypoint_seq[n_frames // 2])
            return preds, timestamps, rep_keypoints

        for start in range(0, n_frames - window_frames + 1, stride):
            end = start + window_frames
            window_seq = keypoint_seq[start:end]
            features = summarize_feature_sequence(
                window_seq, target_frames=MODEL_FRAMES
            )
            if features is None:
                continue

            pred = self.predict(features)
            mid = start + (window_frames // 2)
            timestamp = float(mid / fps)

            if preds and preds[-1] == pred:
                # Merge consecutive identical windows to reduce duplicate events.
                timestamps[-1] = timestamp
                rep_keypoints[-1] = keypoint_seq[mid]
            else:
                preds.append(pred)
                timestamps.append(timestamp)
                rep_keypoints.append(keypoint_seq[mid])

        return preds, timestamps, rep_keypoints
