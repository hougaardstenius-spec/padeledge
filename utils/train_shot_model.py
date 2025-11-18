import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from utils.video_processor import extract_keypoints_from_video

TRAINING_CSV = "data/training_labels.csv"
MODEL_PATH = "models/shot_classifier.pkl"

def load_training_data():
    df = pd.read_csv(TRAINING_CSV)
    X = []
    y = []

    for idx, row in df.iterrows():
        filepath = row["filepath"]
        label = row["label"]

        print(f"Processing {filepath}...")

        keypoints = extract_keypoints_from_video(filepath)

        if keypoints is None or len(keypoints) == 0:
            print(f"⚠️ Warning: No keypoints for {filepath}")
            continue

        feature_vec = np.array(keypoints).flatten()
        X.append(feature_vec)
        y.append(label)

    return np.array(X), np.array(y)


def main():
    print("📥 Loading training dataset...")
    X, y = load_training_data()

    print(f"Dataset loaded: {X.shape} samples")
    print("Training model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\n✅ Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
