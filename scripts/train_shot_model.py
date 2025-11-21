import sys
import os
import glob
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ensure /app is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.video_processor import extract_keypoints_from_video

# Absolute paths (IMPORTANT for Docker)
BASE_DIR = "/app"
DATA_DIR = os.path.join(BASE_DIR, "data", "samples")
MODEL_PATH = os.path.join(BASE_DIR, "models", "shot_classifier.pkl")
SHOT_DETECTOR_PATH = os.path.join(BASE_DIR, "utils", "shot_detector.py")
SHOT_DETECTOR_BACKUP = SHOT_DETECTOR_PATH + ".backup"


def find_training_videos():
    """Scan folder structure for labeled training data"""
    files = []
    labels = []

    for root, dirs, _ in os.walk(DATA_DIR):
        for d in dirs:
            class_dir = os.path.join(root, d)
            label = d.lower()

            video_files = (
                glob.glob(os.path.join(class_dir, "*.mp4")) +
                glob.glob(os.path.join(class_dir, "*.mov")) +
                glob.glob(os.path.join(class_dir, "*.avi"))
            )

            for vf in video_files:
                files.append(vf)
                labels.append(label)

    return files, labels


def load_training_data():
    print("📂 Scanning training folders...")
    files, labels = find_training_videos()
    print(f"Found {len(files)} labeled video samples.")

    X, y = [], []

    for path, label in zip(files, labels):
        print(f"➡ Processing {path} ({label})")

        keypoints = extract_keypoints_from_video(path)
        if keypoints is None:
            print(f"⚠️ WARNING: No keypoints extracted → skipping {path}")
            continue

        X.append(keypoints.flatten())
        y.append(label)

    return np.array(X), np.array(y)


def update_shot_detector():
    """Safely update shot_detector.py so app loads latest model."""
    print("🛠 Updating utils/shot_detector.py safely...")

    # Backup original file
    if not os.path.exists(SHOT_DETECTOR_BACKUP):
        os.rename(SHOT_DETECTOR_PATH, SHOT_DETECTOR_BACKUP)

    new_code = f'''
# AUTO-GENERATED — DO NOT EDIT
import os
import joblib
import numpy as np

MODEL_PATH = "{MODEL_PATH}"

_model = None

def load_model():
    global _model
    if _model is None:
        print("Loading model:", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_shot(keypoints):
    model = load_model()
    vec = np.array(keypoints).flatten().reshape(1, -1)
    return model.predict(vec)[0]
'''

    with open(SHOT_DETECTOR_PATH, "w", encoding="utf-8") as f:
        f.write(new_code)

    print("✅ shot_detector.py updated.")


def main():
    print("📥 Loading training data...")
    X, y = load_training_data()

    if len(X) == 0:
        print("❌ ERROR: No training data found.")
        return

    print("📊 Training RandomForest model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

    update_shot_detector()
    print("🎉 Training complete.")

# Sicherheits-check: ensure file is not empty
if os.path.getsize(MODEL_PATH) < 4096:
    raise RuntimeError("❌ Model file too small — training likely failed")

if __name__ == "__main__":
    main()
