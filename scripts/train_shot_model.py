import sys
import os
import glob
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add root path so utils imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.video_processor import extract_clip_features, MODEL_FRAMES

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.getenv("PADELEDGE_DATA_DIR", os.path.join(BASE_DIR, "data", "samples"))
MODEL_PATH = os.getenv("PADELEDGE_MODEL_PATH", os.path.join(BASE_DIR, "models", "shot_classifier.pkl"))
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
ARCHIVE_DIR = os.path.join(BASE_DIR, "models", "archive")


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
            print(f"➡ Extracting fixed-length features from: {vf}")

        features = extract_clip_features(vf, target_frames=MODEL_FRAMES)

        if features is None:
            if verbose:
                print("⚠ No frame features detected — skipping")
            continue

        X.append(features)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


def _archive_existing_model_if_present():
    if not os.path.exists(MODEL_PATH):
        return None

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_path = os.path.join(ARCHIVE_DIR, f"shot_classifier_{ts}.pkl")
    os.replace(MODEL_PATH, archived_path)
    return archived_path


def _build_metrics(y_true, preds, trained_on_full_data=False):
    if y_true is None or preds is None:
        return {
            "timestamp": datetime.now().isoformat(),
            "dummy": True,
            "reason": "No validation split available",
        }

    report = classification_report(y_true, preds, output_dict=True, zero_division=0)
    per_class = {}
    for label, payload in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        if isinstance(payload, dict):
            per_class[label] = {
                "precision": payload.get("precision", 0.0),
                "recall": payload.get("recall", 0.0),
                "f1": payload.get("f1-score", 0.0),
                "support": int(payload.get("support", 0)),
            }

    return {
        "timestamp": datetime.now().isoformat(),
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "per_class": per_class,
        "trained_on_full_data": trained_on_full_data,
        "feature_frames": MODEL_FRAMES,
        "feature_dim": None,
    }


def train_model(verbose=True):
    X, y = load_training_data(verbose=verbose)

    if len(X) == 0:
        raise RuntimeError("❌ No training data found! Aborting training.")

    if verbose:
        print("📊 Training model...")

    labels, counts = np.unique(y, return_counts=True)
    can_stratify = len(labels) >= 2 and np.min(counts) >= 2 and len(X) >= 5

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1
    )

    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics_payload = _build_metrics(y_test, preds, trained_on_full_data=False)
        if verbose:
            print("\n=== CLASSIFICATION REPORT ===\n")
            print(classification_report(y_test, preds, zero_division=0))
    else:
        # Fallback for very small datasets where stratified split is invalid.
        model.fit(X, y)
        metrics_payload = {
            "timestamp": datetime.now().isoformat(),
            "dummy": True,
            "reason": "Dataset too small for stratified validation split",
            "num_samples": int(len(X)),
            "num_classes": int(len(labels)),
            "feature_frames": MODEL_FRAMES,
            "feature_dim": int(X.shape[1]) if X.ndim == 2 else None,
        }
        if verbose:
            print("⚠ Trained on full dataset (no holdout metrics available).")

    metrics_payload["feature_dim"] = int(X.shape[1]) if X.ndim == 2 else None

    _archive_existing_model_if_present()
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"✅ Model saved to: {MODEL_PATH}")
        print(f"✅ Metrics saved to: {METRICS_PATH}")

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
