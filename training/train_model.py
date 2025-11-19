# scripts/train_model.py
import os
import glob
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils.video_processor import extract_keypoints_from_video

DATA_DIR = "data/samples"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "shot_classifier.pkl")
FEATURE_CACHE = "datasets/features.npz"

def find_videos():
    files = []
    labels = []
    for root, dirs, _ in os.walk(DATA_DIR):
        for d in dirs:
            class_dir = os.path.join(root, d)
            # label is last folder name
            label = d.lower()
            vids = glob.glob(os.path.join(class_dir, "*.mp4")) + \
                   glob.glob(os.path.join(class_dir, "*.mov")) + \
                   glob.glob(os.path.join(class_dir, "*.avi"))
            for v in vids:
                files.append(v)
                labels.append(label)
    return files, labels

def build_dataset(rebuild=False):
    if os.path.exists(FEATURE_CACHE) and not rebuild:
        print("Loading cached features:", FEATURE_CACHE)
        npz = np.load(FEATURE_CACHE, allow_pickle=True)
        return npz["X"], npz["y"]
    files, labels = find_videos()
    X, y = [], []
    for fp, lab in zip(files, labels):
        print("Processing:", fp)
        kp = extract_keypoints_from_video(fp, max_frames=120)
        if kp is None:
            print("No keypoints for", fp, "- skipping")
            continue
        # flatten to 1D
        feat = kp.flatten()
        X.append(feat)
        y.append(lab)
    X = np.array(X)
    y = np.array(y)
    os.makedirs(os.path.dirname(FEATURE_CACHE), exist_ok=True)
    np.savez_compressed(FEATURE_CACHE, X=X, y=y)
    return X, y

def train_and_save(X, y):
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Classification report:\n", classification_report(y_test, preds))
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Saved model to:", MODEL_PATH)

def main():
    X, y = build_dataset(rebuild=False)
    if len(X) == 0:
        raise SystemExit("No training samples found under data/samples/. Add videos before training.")
    train_and_save(X, y)

if __name__ == "__main__":
    main()
