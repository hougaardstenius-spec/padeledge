import numpy as np
import joblib
from utils.video_processor import extract_keypoints_from_video

MODEL_PATH = "models/shot_classifier.pkl"

# Load model once
shot_model = joblib.load(MODEL_PATH)

def detect_shots(keypoints, segment_size=25):
    """
    Detect shots by sliding window across keypoints.
    Returns list of: {type, timestamp_frame}
    """

    results = []
    total_frames = len(keypoints)

    for i in range(0, total_frames - segment_size, segment_size):
        segment = keypoints[i:i + segment_size]
        feature_vec = segment.flatten().reshape(1, -1)

        pred = shot_model.predict(feature_vec)[0]

        results.append({
            "type": pred,
            "frame": i
        })

    return results
