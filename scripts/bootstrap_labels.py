import argparse
import csv
import os
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".avi"}
OUTPUT_COLUMNS = [
    "match_id",
    "player_id",
    "view",
    "clip_path",
    "shot_type",
    "quality",
    "confidence_labeler",
    "outcome",
    "notes",
]

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


def _normalize_shot(folder_name: str) -> str:
    value = folder_name.strip().lower()
    return value if value in ALLOWED_SHOT_TYPES else "other"


def _iter_video_paths(samples_root: Path):
    for path in sorted(samples_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            yield path


def bootstrap_labels(
    samples_root: Path,
    output_csv: Path,
    default_match_id: str,
    default_player_id: str,
    default_view: str = "end_to_end",
    overwrite: bool = False,
):
    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"{output_csv} already exists. Use --overwrite to replace it."
        )

    rows = []
    for video_path in _iter_video_paths(samples_root):
        rel = video_path.relative_to(samples_root).as_posix()
        # Expecting <category>/<shot_type>/<file>; fallback to parent folder name.
        parts = rel.split("/")
        folder_shot = parts[-2] if len(parts) >= 2 else video_path.parent.name
        shot_type = _normalize_shot(folder_shot)
        rows.append(
            {
                "match_id": default_match_id,
                "player_id": default_player_id,
                "view": default_view,
                "clip_path": rel,
                "shot_type": shot_type,
                "quality": "ok",
                "confidence_labeler": "2",
                "outcome": "",
                "notes": "",
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap labels.csv from clips found under data/samples."
    )
    parser.add_argument(
        "--samples-root",
        default="data/samples",
        help="Root folder containing training clips",
    )
    parser.add_argument(
        "--output",
        default="data/labels.csv",
        help="Output labels CSV path",
    )
    parser.add_argument(
        "--default-match-id",
        default="match_unknown",
        help="Default match_id value",
    )
    parser.add_argument(
        "--default-player-id",
        default="player_unknown",
        help="Default player_id value",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it already exists",
    )
    args = parser.parse_args()

    samples_root = Path(args.samples_root)
    output = Path(args.output)
    if not samples_root.exists():
        raise FileNotFoundError(f"samples root not found: {samples_root}")

    count = bootstrap_labels(
        samples_root=samples_root,
        output_csv=output,
        default_match_id=args.default_match_id,
        default_player_id=args.default_player_id,
        overwrite=args.overwrite,
    )
    print(f"Wrote {count} rows to {output}")


if __name__ == "__main__":
    main()

