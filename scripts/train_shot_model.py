import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.video_processor import extract_keypoints_from_video
from utils.shot_detector import ShotDetector

DATA_DIR = "data/samples"
MODEL_PATH = "models/shot_classifier.pkl"
SHOT_DETECTOR_PATH = "utils/shot_detector.py"


def find_training_videos():
    """
    Scans through data/samples/<shot> and returns:
      - files: list of video paths
      - labels: list of corresponding shot labels
    """
    files = []
    labels = []

    for root, dirs, _ in os.walk(DATA_DIR):
        for d in dirs:
            class_dir = os.path.join(root, d)
            label = d.lower()

            video_files = glob.glob(os.path.join(class_dir, "*.mp4")) + \
                          glob.glob(os.path.join(class_dir, "*.mov")) + \
                          glob.glob(os.path.join(class_dir, "*.avi"))

            for vf in video_files:
                files.append(vf)
                labels.append(label)

    return files, labels


def load_training_data():
    print("📂 Scanning training folders...")
    files, labels = find_training_videos()
    print(f"Found {len(files)} samples.")

    X = []
    y = []

    for filepath, label in zip(files, labels):
        print(f"➡ Processing {filepath} ({label})")

        keypoints = extract_keypoints_from_video(filepath)

        if keypoints is None:
            print(f"⚠️ No keypoints extracted from {filepath}")
            continue

        feature_vec = keypoints.flatten()
        X.append(feature_vec)
        y.append(label)

    return np.array(X), np.array(y)


def update_shot_detector():
    """
    Automatically updates utils/shot_detector.py so it loads the new model path.
    """
    print("🛠 Updating utils/shot_detector.py...")

    new_loader_code = f"""
# AUTO-GENERATED — DO NOT EDIT MANUALLY
import joblib
import numpy as np

MODEL_PATH = "{MODEL_PATH}"

# Lazy loaded model
_model = None

def load_model():
    global _model
    if _model is None:
        print("Loading model from:", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_shot(keypoints):
    model = load_model()
    feature_vec = np.array(keypoints).flatten().reshape(1, -1)
    return model.predict(feature_vec)[0]
"""

    with open(SHOT_DETECTOR_PATH, "w", encoding="utf-8") as f:
        f.write(new_loader_code)

    print("✅ utils/shot_detector.py updated automatically.")


def main():
    print("📥 Loading training data...")
    X, y = load_training_data()

    print("📊 Training RandomForest model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

    # NEW ✓ auto-update shot detector
    update_shot_detector()


if __name__ == "__main__":
    main()
