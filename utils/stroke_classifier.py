import os
import numpy as np
import joblib

# --------------------------------------------------------
# Define all padel stroke types and their synthetic patterns
# --------------------------------------------------------
STROKES = {
    # Groundstrokes
    "forehand":        {"angle_mean": 45,  "angle_std": 8,  "dist1": 0.3,  "dist2": 0.4},
    "backhand":        {"angle_mean": 135, "angle_std": 10, "dist1": 0.35, "dist2": 0.45},
    "chiquita":        {"angle_mean": 55,  "angle_std": 6,  "dist1": 0.25, "dist2": 0.35},
    "lob":             {"angle_mean": 80,  "angle_std": 7,  "dist1": 0.2,  "dist2": 0.25},

    # Volleys
    "forehand_volley": {"angle_mean": 30,  "angle_std": 6,  "dist1": 0.25, "dist2": 0.3},
    "backhand_volley": {"angle_mean": 150, "angle_std": 6,  "dist1": 0.25, "dist2": 0.35},
    "chancletazo":     {"angle_mean": 40,  "angle_std": 8,  "dist1": 0.2,  "dist2": 0.25},
    "volley_lob":      {"angle_mean": 70,  "angle_std": 7,  "dist1": 0.22, "dist2": 0.28},

    # Overheads
    "bandeja":         {"angle_mean": 75,  "angle_std": 5,  "dist1": 0.2,  "dist2": 0.3},
    "vibora":          {"angle_mean": 85,  "angle_std": 5,  "dist1": 0.22, "dist2": 0.32},
    "rulo":            {"angle_mean": 95,  "angle_std": 6,  "dist1": 0.25, "dist2": 0.35},
    "gancho":          {"angle_mean": 110, "angle_std": 8,  "dist1": 0.25, "dist2": 0.38},
    "smash":           {"angle_mean": 20,  "angle_std": 4,  "dist1": 0.28, "dist2": 0.4},

    # After backglass
    "bajada":          {"angle_mean": 60,  "angle_std": 6,  "dist1": 0.3,  "dist2": 0.35},
    "cuchilla":        {"angle_mean": 130, "angle_std": 6,  "dist1": 0.27, "dist2": 0.33},
}

# --------------------------------------------------------
# Generate synthetic dataset (for quick testing/training)
# --------------------------------------------------------
def generate_synthetic_data(samples_per_stroke=80, save_dir="data/synthetic"):
    os.makedirs(save_dir, exist_ok=True)
    X, y = [], []

    for stroke, params in STROKES.items():
        for _ in range(samples_per_stroke):
            arm_angle = np.random.normal(params["angle_mean"], params["angle_std"])
            wrist_elbow_dist = np.random.normal(params["dist1"], 0.02)
            shoulder_hip_dist = np.random.normal(params["dist2"], 0.02)
            X.append([arm_angle, wrist_elbow_dist, shoulder_hip_dist])
            y.append(stroke)

    X, y = np.array(X), np.array(y)
    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)
    print(f"✅ Generated {len(y)} samples for {len(STROKES)} stroke types.")
    return X, y


# --------------------------------------------------------
# Train model on synthetic data
# --------------------------------------------------------
def train_synthetic_model():
    from sklearn.ensemble import RandomForestClassifier

    X, y = generate_synthetic_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    os.makedirs("utils", exist_ok=True)
    joblib.dump(model, "utils/padel_model.pkl")
    print("✅ Trained model saved as utils/padel_model.pkl")
    print(f"Stroke classes: {sorted(set(y))}")


# --------------------------------------------------------
# Load model and classify strokes
# --------------------------------------------------------
def classify_strokes(input_data):
    if not os.path.exists("utils/padel_model.pkl"):
        raise FileNotFoundError("❌ Model file missing! Train it first using train_synthetic_model().")

    model = joblib.load("utils/padel_model.pkl")
    predictions = model.predict(input_data)
    return predictions


# --------------------------------------------------------
# Optional: run this file directly to retrain the model
# --------------------------------------------------------
if __name__ == "__main__":
    train_synthetic_model()
