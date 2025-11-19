# utils/shot_detector.py
import os
import joblib
import numpy as np
from utils.video_processor import extract_keypoints_from_video

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "shot_classifier.pkl"))

class ShotDetector:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model missing at {MODEL_PATH}. Train model first (scripts/train_model.py).")
        self.model = joblib.load(MODEL_PATH)

    def analyze(self, video_path, max_frames=120):
        """
        Extract keypoints + predict per-sampled-frame.
        Returns:
            preds: list of labels (len = number of frames used)
            timestamps: list of floats (seconds)
            keypoint_seqs: list/np.array (frames, landmarks*3)
        """
        kp_seq, timestamps = extract_keypoints_from_video(video_path, max_frames=max_frames, return_timestamps=True)
        if kp_seq is None:
            return [], [], []
        # predict frame-by-frame (or in batch)
        X = [kp.flatten() for kp in kp_seq]
        preds = self.model.predict(X).tolist()
        return preds, timestamps.tolist(), kp_seq
