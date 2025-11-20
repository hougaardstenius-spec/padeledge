import os
import glob
from typing import List, Dict, Optional

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "samples")
UNCERTAIN_DIR = os.path.join(BASE_DIR, "data", "uncertain")


def _list_video_files(folder: str) -> List[str]:
    files = []
    for ext in ("*.mp4", "*.mov", "*.avi"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    return files


def get_dataset_overview() -> pd.DataFrame:
    """
    Scanner data/samples/<category>/<shot_type> for videoer og returnerer en tabel.
    """
    rows = []

    if not os.path.isdir(DATA_DIR):
        return pd.DataFrame(columns=["Category", "Shot Type", "Num Clips"])

    for category in sorted(os.listdir(DATA_DIR)):
        cat_path = os.path.join(DATA_DIR, category)
        if not os.path.isdir(cat_path):
            continue

        for shot_type in sorted(os.listdir(cat_path)):
            shot_path = os.path.join(cat_path, shot_type)
            if not os.path.isdir(shot_path):
                continue

            files = _list_video_files(shot_path)
            rows.append(
                {
                    "Category": category,
                    "Shot Type": shot_type,
                    "Num Clips": len(files),
                    "Folder": shot_path,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Category", "Shot Type", "Num Clips", "Folder"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["Category", "Shot Type"]).reset_index(drop=True)
    return df


def list_sample_videos(category: str, shot_type: str, limit: int = 10) -> List[str]:
    """
    Returnerer op til 'limit' videoer for en given kategori/shot-type.
    """
    shot_path = os.path.join(DATA_DIR, category, shot_type)
    if not os.path.isdir(shot_path):
        return []

    files = _list_video_files(shot_path)
    return sorted(files)[:limit]


def save_labeled_clip(
    temp_path: str,
    category: str,
    shot_type: str,
    filename: Optional[str] = None,
) -> str:
    """
    Flytter en midlertidig upload til den rigtige træningsmappe:
    data/samples/<category>/<shot_type>/<filename>
    """
    if filename is None:
        filename = os.path.basename(temp_path)

    target_dir = os.path.join(DATA_DIR, category, shot_type)
    os.makedirs(target_dir, exist_ok=True)

    target_path = os.path.join(target_dir, filename)
    os.replace(temp_path, target_path)
    return target_path


def list_uncertain_clips() -> List[str]:
    """
    Returnerer videoer i data/uncertain (til active learning).
    """
    if not os.path.isdir(UNCERTAIN_DIR):
        return []

    files = _list_video_files(UNCERTAIN_DIR)
    return sorted(files)
