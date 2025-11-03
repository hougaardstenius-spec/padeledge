import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_FILE = "data/features.npy"
MODEL_FILE = "utils/padel_model.pkl"

def train_model():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Run prepare_dataset.py first!")

    data = np.load(DATA_FILE, allow_pickle=True).item()
    X, y = np.array(data["X"]), np.array(data["y"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model trained with {acc*100:.2f}% accuracy")
    print(classification_report(y_test, preds))

    os.makedirs("utils", exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"💾 Saved trained model to {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
