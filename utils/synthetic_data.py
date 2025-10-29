import os
import numpy as np
import joblib

# Each stroke type will have its own characteristic pattern of angles & distances
STROKES = {
    "forehand":        {"angle_mean": 45,  "angle_std": 8,  "dist1": 0.3, "dist2": 0.4},
    "backhand":        {"angle_mean": 135, "angle_std": 10, "dist1": 0.35, "dist2": 0.45},
    "volley_forehand": {"angle_mean": 30,  "angle_std": 7,  "dist1": 0.25, "dist2": 0.3},
    "volley_backhand": {"angle_mean": 150, "angle_std": 7,  "dist1": 0.25, "dist2": 0.35},
    "bandeja":         {"angle_mean": 70,  "angle_std": 5,  "dist1": 0.2, "dist2": 0.3},
    "vibora":          {"angle_mean": 80,  "angle_std": 5,  "dist1": 0.22, "dist2": 0.32},
    "rulo":            {"angle_mean": 100, "angle_std": 7,  "dist1": 0.25, "dist2": 0.35},
    "smash":           {"angle_mean": 20,  "angle_std": 4,  "dist1": 0.28, "dist2": 0.4},
    "bajada":          {"angle_mean": 60,  "angle_std": 6,  "dist1": 0.3, "dist2": 0.35},
}

def generate_synthetic_data(samples_per_stroke=50, save_dir="data/synthetic"):
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
    print(f"✅ Generated synthetic dataset with {len(y)} samples.")
    return X, y

def train_synthetic_model():
    """Trains a model directly on synthetic data."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = generate_synthetic_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    os.makedirs("utils", exist_ok=True)
    joblib.dump(model, "utils/padel_model.pkl")
    print("✅ AI model trained on synthetic data and saved as utils/padel_model.pkl")
