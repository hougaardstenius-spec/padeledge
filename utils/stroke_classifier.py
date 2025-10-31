import os
import joblib
import numpy as np
from utils.synthetic_data import train_synthetic_model

MODEL_PATH = "utils/padel_model.pkl"

# Automatically train if the model file doesn't exist
if not os.path.exists(MODEL_PATH):
    print("⚠️ Model file missing — training new synthetic model...")
    train_synthetic_model()
else:
    print("✅ Loaded existing padel_model.pkl")

# Load the trained model
model = joblib.load(MODEL_PATH)

def classify_strokes(features):
    """Predicts the stroke type based on input features."""
    if features is None or len(features) == 0:
        return ["unknown"]
    try:
        preds = model.predict(np.array(features))
        return preds
    except Exception as e:
        print(f"⚠️ Error during classification: {e}")
        return ["unknown"]
