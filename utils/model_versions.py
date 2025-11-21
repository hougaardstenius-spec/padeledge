import os
import glob
from datetime import datetime
from typing import List, Dict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LATEST_DIR = os.path.join(MODELS_DIR, "latest")
ARCHIVE_DIR = os.path.join(MODELS_DIR, "archive")


def _format_bytes(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def get_current_model_overview() -> Dict:
    info = {
        "exists": False,
        "path": None,
        "modified": None,
        "size": None,
        "latest_path": None,
        "latest_modified": None,
        "archive_count": 0,
    }

    classic_path = os.path.join(MODELS_DIR, "shot_classifier.pkl")
    latest_path = os.path.join(LATEST_DIR, "shot_classifier.pkl")

    effective_path = latest_path if os.path.exists(latest_path) else classic_path

    if os.path.exists(effective_path):
        info["exists"] = True
        info["path"] = effective_path
        info["modified"] = datetime.fromtimestamp(os.path.getmtime(effective_path))
        info["size"] = _format_bytes(os.path.getsize(effective_path))

    if os.path.exists(latest_path):
        info["latest_path"] = latest_path
        info["latest_modified"] = datetime.fromtimestamp(os.path.getmtime(latest_path))

    if os.path.isdir(ARCHIVE_DIR):
        archive_files = glob.glob(os.path.join(ARCHIVE_DIR, "shot_classifier_*.pkl"))
        info["archive_count"] = len(archive_files)

    return info


def list_model_versions() -> List[Dict]:
    versions = []
    if not os.path.isdir(ARCHIVE_DIR):
        return versions

    for path in sorted(glob.glob(os.path.join(ARCHIVE_DIR, "shot_classifier_*.pkl"))):
        ts = datetime.fromtimestamp(os.path.getmtime(path))
        size = _format_bytes(os.path.getsize(path))
        name = os.path.basename(path)
        versions.append(
            {
                "name": name,
                "path": path,
                "modified": ts,
                "size": size,
            }
        )

    return versions
