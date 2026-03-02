import argparse
import csv
import sys
from pathlib import Path


ALLOWED_SHOT_TYPES = {
    "bandeja",
    "vibora",
    "smash",
    "forehand",
    "backhand",
    "volley",
    "bajada",
    "other",
}


def _normalize_shot(value: str) -> str:
    value = (value or "").strip().lower()
    return value if value in ALLOWED_SHOT_TYPES else "other"


def _expected_shot_from_clip_path(clip_path: str) -> str:
    parts = Path(clip_path).as_posix().split("/")
    folder_shot = parts[-2] if len(parts) >= 2 else "other"
    return _normalize_shot(folder_shot)


def find_conflicts(labels_path: Path):
    conflicts = []
    with labels_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_no, row in enumerate(reader, start=2):
            clip_path = (row.get("clip_path") or "").strip()
            actual = _normalize_shot(row.get("shot_type"))
            expected = _expected_shot_from_clip_path(clip_path)
            if actual != expected:
                conflicts.append(
                    {
                        "row": row_no,
                        "clip_path": clip_path,
                        "label_shot_type": actual,
                        "expected_from_path": expected,
                    }
                )
    return conflicts


def main():
    parser = argparse.ArgumentParser(
        description="Find label conflicts between labels.csv shot_type and folder-derived shot type."
    )
    parser.add_argument(
        "--labels",
        default="data/labels.csv",
        help="Path to labels csv (default: data/labels.csv)",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"labels file not found: {labels_path}")
        sys.exit(1)

    conflicts = find_conflicts(labels_path)
    print(f"labels: {labels_path}")
    print(f"conflicts: {len(conflicts)}")
    for c in conflicts:
        print(
            f"row {c['row']}: clip={c['clip_path']} label={c['label_shot_type']} expected={c['expected_from_path']}"
        )

    if conflicts:
        sys.exit(1)
    print("No label conflicts found.")
    sys.exit(0)


if __name__ == "__main__":
    main()

