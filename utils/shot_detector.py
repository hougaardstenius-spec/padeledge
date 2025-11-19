# --------------------------------------------
# IMPORTS
# --------------------------------------------
import os
import joblib
import numpy as np

# (andre imports du måske har)
# from utils.video_processor import extract_keypoints_from_video


# --------------------------------------------
# MODEL PATH SETUP (indsæt denne del)
# --------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", 
    "models", 
    "shot_classifier.pkl"
)

MODEL_PATH = os.path.abspath(MODEL_PATH)


# --------------------------------------------
# SHOT DETECTOR CLASS
# --------------------------------------------
class ShotDetector:
    def __init__(self):
        # Fail-safe hvis modellen mangler
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"❌ Model file not found at: {MODEL_PATH}\n"
                f"Make sure 'models/shot_classifier.pkl' exists and is included in your repository."
            )

        # Load modellen
        self.model = joblib.load(MODEL_PATH)

    def predict(self, feature_vector):
        return self.model.predict([feature_vector])[0]
