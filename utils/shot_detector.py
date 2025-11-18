import joblib
import numpy as np

MODEL_PATH = "models/shot_classifier.pkl"
shot_model = joblib.load(MODEL_PATH)


def detect_shots(keypoints, segment_size=25):
    """
    Sliding window detection.
    Returns list of events:
      { "type": <label>, "frame": <frame number> }
    """

    results = []

    num_frames = len(keypoints)

    for i in range(0, num_frames - segment_size, segment_size):
        segment = keypoints[i : i + segment_size]
        feature_vec = segment.flatten().reshape(1, -1)

        pred = shot_model.predict(feature_vec)[0]

        results.append({
            "type": pred,
            "frame": i
        })

    return results
