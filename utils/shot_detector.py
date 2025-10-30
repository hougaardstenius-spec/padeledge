import numpy as np
from utils.stroke_classifier import classify_strokes

def detect_shots(keypoints):
    # Simple segmentation (every N frames = 1 shot)
    window = 5
    shots = []
    for i in range(0, len(keypoints), window):
        sample = np.mean(keypoints[i:i+window], axis=0).reshape(1, -1)
        stroke = classify_strokes(sample)[0]
        shots.append(stroke)
    return shots
