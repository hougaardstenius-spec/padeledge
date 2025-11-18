import os
import glob
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils.video_processor import extract_keypoints_from_video

DATA_DIR = "data/samples"
MODEL_PATH = "models/shot_classifier.pkl"


def find_training_videos():
    """
    Scans through data/samples/<category>/<subcat>*/video.mp4
    Returns:
      files: list of filepaths
      labels: list of corresponding shot labels
    """
    files = []
    labels = []

    for root, dirs, _ in os.walk(DATA_DIR):
        for d in dirs:
            class_dir = os.path.join(root, d)
            # final label = last folder name (bandeja, vibora, etc.)
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


def main():
    print("📥 Loading training data...")
    X, y = load_training_data()

    print("📊 Training ML model...")
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


if __name__ == "__main__":
    main()
