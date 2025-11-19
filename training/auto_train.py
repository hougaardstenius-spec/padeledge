# scripts/auto_train.py
import os
import json
from time import sleep
from pathlib import Path
from scripts.train_model import find_videos, build_dataset, train_and_save, FEATURE_CACHE, MODEL_DIR

MARKER = "scripts/.trained_files.json"

def load_trained():
    if os.path.exists(MARKER):
        with open(MARKER, "r") as f:
            return set(json.load(f))
    return set()

def save_trained(files):
    with open(MARKER, "w") as f:
        json.dump(list(files), f)

def main():
    trained = load_trained()
    vids, _ = find_videos()
    vids_set = set(vids)
    new = vids_set - trained
    if new:
        print("New videos detected:", len(new))
        # Rebuild dataset from scratch (or we could append incremental)
        X, y = build_dataset(rebuild=True)
        if len(X) > 0:
            train_and_save(X, y)
            save_trained(vids_set)
            print("Retrain done.")
    else:
        print("No new videos detected. Nothing to do.")

if __name__ == "__main__":
    main()
