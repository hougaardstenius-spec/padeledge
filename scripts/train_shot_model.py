import sys
import os
import glob
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add root path so utils imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.video_processor import extract_keypoints_from_video
from utils.shot_detector import ShotDetector

DATA_DIR = "/app/data/samples"
MODEL_PATH = "/app/models/shot_classifier.pkl"


def find_training_videos():
    files = []
    labels = []

    for root, dirs, _ in os.walk(DATA_DIR):
        for d in dirs:
            class_dir = os.path.join(root, d)
            label = d.lower()

            video_files = glob.glob(os.path.join(class_dir, "*.mp4"))
            video_files += glob.glob(os.path.join(class_dir, "*.mov"))
            video_files += glob.glob(os.path.join(class_dir, "*.avi"))

            for vf in video_files:
                files.append(vf)
                labels.append(label)

    return files, labels


def load_training_data(verbose=True):
    if verbose:
        print("📂 Scanning training data folder:", DATA_DIR)

    files, labels = find_training_videos()

    if verbose:
        print(f"Found {len(files)} videos")

    X = []
    y = []

    for vf, label in zip(files, labels):
        if verbose:
            print(f"➡ Extracting keypoints from: {vf}")

        keypoints = extract_keypoints_from_video(vf)

        if keypoints is None:
            if verbose:
                print(f"⚠ No keypoints detected — skipping")
            continue

        X.append(keypoints.flatten())
        y.append(label)

    return np.array(X), np.array(y)


def train_model(verbose=True):
    X, y = load_training_data(verbose=verbose)

    if len(X) == 0:
        raise RuntimeError("❌ No training data found! Aborting training.")

    if verbose:
        print("📊 Training model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds)

    if verbose:
        print("\n=== CLASSIFICATION REPORT ===\n")
        print(report)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    if verbose:
        print(f"✅ Model saved to: {MODEL_PATH}")

    return True


def main(verbose=False):
    try:
        train_model(verbose=verbose)
    except Exception as e:
        print("❌ TRAINING ERROR:", str(e))
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    main(verbose=args.verbose)
