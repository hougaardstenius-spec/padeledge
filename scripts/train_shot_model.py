import sys
import os
import glob
import time
import json
import shutil
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sørg for at /app (eller projektrod) er på PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.video_processor import extract_keypoints_from_video

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "samples")

MODELS_DIR = os.path.join(BASE_DIR, "models")
LATEST_DIR = os.path.join(MODELS_DIR, "latest")
ARCHIVE_DIR = os.path.join(MODELS_DIR, "archive")

# “klassisk” model path til bagudkompatibilitet
MODEL_PATH = os.path.join(MODELS_DIR, "shot_classifier.pkl")
LATEST_MODEL_PATH = os.path.join(LATEST_DIR, "shot_classifier.pkl")

METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def find_training_videos():
    """Scan data/samples/<category>/<shot_type> for videoer."""
    files = []
    labels = []

    if not os.path.isdir(DATA_DIR):
        print(f"❌ DATA_DIR findes ikke: {DATA_DIR}")
        return files, labels

    for root, dirs, _ in os.walk(DATA_DIR):
        for d in dirs:
            class_dir = os.path.join(root, d)
            label = d.lower()

            video_files = (
                glob.glob(os.path.join(class_dir, "*.mp4"))
                + glob.glob(os.path.join(class_dir, "*.mov"))
                + glob.glob(os.path.join(class_dir, "*.avi"))
            )

            for vf in video_files:
                files.append(vf)
                labels.append(label)

    return files, labels


def load_training_data():
    print("📂 Scanner træningsmapper...")
    files, labels = find_training_videos()
    print(f"➡ Fundet {len(files)} videoklip med labels.")

    X, y = [], []

    for path, label in zip(files, labels):
        print(f"🎥 {path}  [{label}]")
        keypoints = extract_keypoints_from_video(path)

        if keypoints is None:
            print(f"⚠️ Ingen keypoints fra {path} – skipper.")
            continue

        X.append(keypoints.flatten())
        y.append(label)

    return np.array(X), np.array(y)


def archive_old_models():
    """Flyt eksisterende modeller til models/archive/ med timestamp."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    candidates = []

    if os.path.exists(LATEST_MODEL_PATH):
        candidates.append(LATEST_MODEL_PATH)
    if os.path.exists(MODEL_PATH) and MODEL_PATH != LATEST_MODEL_PATH:
        candidates.append(MODEL_PATH)

    for path in candidates:
        try:
            base = os.path.basename(path)
            archive_name = f"shot_classifier_{ts}_{base}"
            archive_path = os.path.join(ARCHIVE_DIR, archive_name)
            print(f"📦 Arkiverer {path} → {archive_path}")
            shutil.move(path, archive_path)
        except Exception as e:
            print(f"⚠️ Kunne ikke arkivere {path}: {e}")


def save_metrics(y_test, preds):
    """Gem metrics til models/metrics.json, så dashboardet kan læse dem."""
    try:
        report = classification_report(y_test, preds, output_dict=True)
    except Exception as e:
        print(f"⚠️ Kunne ikke generere classification_report: {e}")
        return

    metrics = {
        "timestamp": time.time(),
        "accuracy": report.get("accuracy"),
        "per_class": {},
    }

    for label, stats in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        metrics["per_class"][label] = {
            "precision": stats.get("precision"),
            "recall": stats.get("recall"),
            "f1": stats.get("f1-score") or stats.get("f1"),
            "support": stats.get("support"),
        }

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"📊 Metrics gemt til {METRICS_PATH}")


def main():
    print("📥 Loader træningsdata...")
    X, y = load_training_data()

    if X.size == 0:
        print("❌ Ingen feature-vektorer fundet. Afbryder træning.")
        return

    print("📊 Træner RandomForest model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, preds))

    # Gem metrics
    save_metrics(y_test, preds)

    # Forbered mapper
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LATEST_DIR, exist_ok=True)

    # Arkivér gamle modeller
    archive_old_models()

    # Gem ny model til latest + "root" path
    print(f"💾 Gemmer model til {LATEST_MODEL_PATH}")
    joblib.dump(model, LATEST_MODEL_PATH)

    print(f"📎 Kopierer model til {MODEL_PATH}")
    shutil.copy2(LATEST_MODEL_PATH, MODEL_PATH)

    # Safety check – sikr at filen ikke er tom
    if os.path.getsize(MODEL_PATH) < 4096:
        raise RuntimeError("❌ Model-fil er for lille – træning fejlede sandsynligvis.")

    print("🎉 Træning færdig og model gemt.")


if __name__ == "__main__":
    main()
