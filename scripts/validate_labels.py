import argparse
import csv
import os
import sys
from collections import Counter


REQUIRED_COLUMNS = [
    "match_id",
    "player_id",
    "view",
    "clip_path",
    "shot_type",
    "quality",
    "confidence_labeler",
]

OPTIONAL_COLUMNS = [
    "outcome",
    "notes",
]

ALLOWED_VIEWS = {"end_to_end"}
ALLOWED_QUALITY = {"good", "ok", "bad_visibility"}
ALLOWED_OUTCOMES = {"winner", "forced_error", "unforced_error", "neutral"}
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


def _error(errors, row_no, message):
    errors.append(f"row {row_no}: {message}")


def _normalize(value):
    return (value or "").strip()


def validate_csv(labels_path: str, samples_root: str, allow_missing_files: bool = False):
    errors = []
    warnings = []

    if not os.path.exists(labels_path):
        return [f"labels file not found: {labels_path}"], warnings, {}

    with open(labels_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        missing = [c for c in REQUIRED_COLUMNS if c not in headers]
        if missing:
            errors.append(f"missing required columns: {', '.join(missing)}")
            return errors, warnings, {}

        unknown_cols = [c for c in headers if c not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
        if unknown_cols:
            warnings.append(f"unknown columns present: {', '.join(unknown_cols)}")

        rows = list(reader)

    seen_clip_paths = set()
    shot_counter = Counter()
    quality_counter = Counter()
    outcome_counter = Counter()

    for i, row in enumerate(rows, start=2):
        match_id = _normalize(row.get("match_id"))
        player_id = _normalize(row.get("player_id"))
        view = _normalize(row.get("view")).lower()
        clip_path = _normalize(row.get("clip_path"))
        shot_type = _normalize(row.get("shot_type")).lower()
        quality = _normalize(row.get("quality")).lower()
        conf_raw = _normalize(row.get("confidence_labeler"))
        outcome = _normalize(row.get("outcome")).lower()

        if not match_id:
            _error(errors, i, "match_id is required")
        if not player_id:
            _error(errors, i, "player_id is required")

        if view not in ALLOWED_VIEWS:
            _error(errors, i, f"invalid view '{view}' (allowed: {sorted(ALLOWED_VIEWS)})")

        if not clip_path:
            _error(errors, i, "clip_path is required")
        else:
            if clip_path in seen_clip_paths:
                _error(errors, i, f"duplicate clip_path '{clip_path}'")
            seen_clip_paths.add(clip_path)

            if not allow_missing_files:
                full_path = clip_path
                if not os.path.isabs(clip_path):
                    full_path = os.path.join(samples_root, clip_path)
                if not os.path.exists(full_path):
                    _error(errors, i, f"clip file not found: {full_path}")

        if shot_type not in ALLOWED_SHOT_TYPES:
            _error(
                errors,
                i,
                f"invalid shot_type '{shot_type}' (allowed: {sorted(ALLOWED_SHOT_TYPES)})",
            )
        else:
            shot_counter[shot_type] += 1

        if quality not in ALLOWED_QUALITY:
            _error(
                errors,
                i,
                f"invalid quality '{quality}' (allowed: {sorted(ALLOWED_QUALITY)})",
            )
        else:
            quality_counter[quality] += 1

        if not conf_raw:
            _error(errors, i, "confidence_labeler is required")
        else:
            try:
                conf = int(conf_raw)
                if conf < 1 or conf > 3:
                    _error(errors, i, "confidence_labeler must be in range 1..3")
            except ValueError:
                _error(errors, i, "confidence_labeler must be an integer")

        if outcome:
            if outcome not in ALLOWED_OUTCOMES:
                _error(
                    errors,
                    i,
                    f"invalid outcome '{outcome}' (allowed: {sorted(ALLOWED_OUTCOMES)})",
                )
            else:
                outcome_counter[outcome] += 1

    stats = {
        "rows": len(rows),
        "unique_clip_paths": len(seen_clip_paths),
        "shot_counts": dict(shot_counter),
        "quality_counts": dict(quality_counter),
        "outcome_counts": dict(outcome_counter),
    }

    if len(rows) == 0:
        warnings.append("no label rows found (header-only file)")

    if shot_counter:
        min_cls = min(shot_counter.values())
        max_cls = max(shot_counter.values())
        if max_cls > 3 * max(min_cls, 1):
            warnings.append(
                f"class imbalance warning: max/min ratio={max_cls}/{min_cls} (>3x)"
            )

    return errors, warnings, stats


def main():
    parser = argparse.ArgumentParser(
        description="Validate labels CSV against Padeledge labeling protocol."
    )
    parser.add_argument(
        "--labels",
        default="data/labels.csv",
        help="Path to labels csv (default: data/labels.csv)",
    )
    parser.add_argument(
        "--samples-root",
        default="data/samples",
        help="Base directory used to resolve relative clip_path values",
    )
    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help="Skip existence checks for clip_path files",
    )
    args = parser.parse_args()

    errors, warnings, stats = validate_csv(
        labels_path=args.labels,
        samples_root=args.samples_root,
        allow_missing_files=args.allow_missing_files,
    )

    print(f"labels: {args.labels}")
    print(f"samples_root: {args.samples_root}")
    print(f"rows: {stats.get('rows', 0)}")

    if stats.get("shot_counts"):
        print(f"shot_counts: {stats['shot_counts']}")
    if stats.get("quality_counts"):
        print(f"quality_counts: {stats['quality_counts']}")
    if stats.get("outcome_counts"):
        print(f"outcome_counts: {stats['outcome_counts']}")

    for warning in warnings:
        print(f"WARNING: {warning}")

    if errors:
        print("\nValidation failed:")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)

    print("Validation passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()

