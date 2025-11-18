# utils/shot_detector.py
import joblib
import numpy as np
from utils.video_processor import extract_keypoints_from_video

MODEL_PATH = "models/shot_classifier.pkl"

class ShotDetector:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def analyze(self, video_path):
        """
        Returns:
            predictions (list[str])
            timestamps (list[float])
            keypoint_sequences (list[np.array])
        """
        keypoints, timestamps = extract_keypoints_from_video(video_path, return_timestamps=True)

        if keypoints is None or len(keypoints) == 0:
            return [], [], []

        X = [kp.flatten() for kp in keypoints]
        preds = self.model.predict(X)

        return preds, timestamps, keypoints
