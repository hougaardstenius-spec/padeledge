import argparse
import csv
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


VIDEO_EXTS = {".mp4", ".mov", ".avi"}
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
ALLOWED_VIEWS = {"end_to_end"}

DEFAULT_ALLOWED_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    "vimeo.com",
    "www.vimeo.com",
}

MANIFEST_COLUMNS = [
    "url",
    "source_name",
    "license",
    "rights_confirmed",
    "approved_for_training",
    "view",
    "expected_shot_type",
    "match_id",
    "player_id",
    "notes",
]

REVIEW_COLUMNS = [
    "candidate_id",
    "status",
    "reason",
    "url",
    "domain",
    "download_path",
    "view",
    "expected_shot_type",
    "match_id",
    "player_id",
    "source_name",
    "license",
    "notes",
    "ingested_at",
]


def _truthy(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y"}


def _normalize_shot(value: str) -> str:
    value = (value or "").strip().lower()
    if not value:
        return "other"
    return value if value in ALLOWED_SHOT_TYPES else "other"


def _normalize_view(value: str) -> str:
    value = (value or "").strip().lower()
    return value or "end_to_end"


def _safe_name(value: str) -> str:
    keep = []
    for ch in (value or ""):
        keep.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(keep).strip("._") or "candidate"


def _extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower().split(":")[0]


def _load_allowed_domains(path: Path | None) -> set[str]:
    allowed = set(DEFAULT_ALLOWED_DOMAINS)
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip().lower()
                if not raw or raw.startswith("#"):
                    continue
                allowed.add(raw)
    return allowed


def _validate_row(row: dict, row_no: int, allowed_domains: set[str]):
    errors = []

    for col in MANIFEST_COLUMNS:
        if col not in row:
            errors.append(f"missing column '{col}' in manifest header")
            return errors

    url = (row.get("url") or "").strip()
    source_name = (row.get("source_name") or "").strip()
    license_name = (row.get("license") or "").strip()
    view = _normalize_view(row.get("view"))

    if not url:
        errors.append("url is required")
    else:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            errors.append("url must start with http/https")
        domain = _extract_domain(url)
        if domain not in allowed_domains:
            errors.append(f"domain not in whitelist: {domain}")

    if not source_name:
        errors.append("source_name is required")
    if not license_name:
        errors.append("license is required")
    if not _truthy(row.get("rights_confirmed")):
        errors.append("rights_confirmed must be yes/true/1")
    if not _truthy(row.get("approved_for_training")):
        errors.append("approved_for_training must be yes/true/1")
    if view not in ALLOWED_VIEWS:
        errors.append(f"view must be one of {sorted(ALLOWED_VIEWS)}")

    return errors


def _download_direct(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        with output_path.open("wb") as f:
            shutil.copyfileobj(response, f)
    return output_path


def _download_with_ytdlp(url: str, output_pattern: Path):
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "mp4/best",
        "-o",
        str(output_pattern),
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return None, f"yt-dlp failed: {err[-300:]}"

    parent = output_pattern.parent
    stem = output_pattern.stem.replace("%(ext)s", "")
    candidates = sorted(parent.glob(f"{stem}*"))
    if not candidates:
        return None, "yt-dlp completed but no file found"
    return candidates[-1], ""


def ingest_manifest(
    manifest_path: Path,
    output_root: Path,
    review_csv: Path,
    allowed_domains: set[str],
    dry_run: bool = False,
):
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    rows_out = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            row_no = idx + 1
            url = (row.get("url") or "").strip()
            domain = _extract_domain(url) if url else ""
            shot_type = _normalize_shot(row.get("expected_shot_type"))
            view = _normalize_view(row.get("view"))
            candidate_id = f"c{idx:05d}_{_safe_name(Path(urlparse(url).path).stem)}"

            result = {
                "candidate_id": candidate_id,
                "status": "rejected",
                "reason": "",
                "url": url,
                "domain": domain,
                "download_path": "",
                "view": view,
                "expected_shot_type": shot_type,
                "match_id": (row.get("match_id") or "").strip(),
                "player_id": (row.get("player_id") or "").strip(),
                "source_name": (row.get("source_name") or "").strip(),
                "license": (row.get("license") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
                "ingested_at": datetime.utcnow().isoformat() + "Z",
            }

            errors = _validate_row(row, row_no=row_no, allowed_domains=allowed_domains)
            if errors:
                result["reason"] = "; ".join(errors)
                rows_out.append(result)
                continue

            if dry_run:
                result["status"] = "accepted_dry_run"
                result["reason"] = "validated only (dry-run)"
                rows_out.append(result)
                continue

            domain_dir = output_root / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            path_obj = Path(urlparse(url).path)
            ext = path_obj.suffix.lower()
            if ext in VIDEO_EXTS:
                out_path = domain_dir / f"{candidate_id}{ext}"
                try:
                    downloaded = _download_direct(url, out_path)
                    result["status"] = "downloaded"
                    result["download_path"] = str(downloaded)
                    result["reason"] = "direct download"
                except Exception as e:
                    result["status"] = "failed"
                    result["reason"] = f"direct download failed: {e}"
            else:
                out_pattern = domain_dir / f"{candidate_id}.%(ext)s"
                try:
                    downloaded, err = _download_with_ytdlp(url, out_pattern)
                except FileNotFoundError:
                    downloaded, err = None, "yt-dlp not installed"

                if downloaded is None:
                    result["status"] = "failed"
                    result["reason"] = err
                else:
                    result["status"] = "downloaded"
                    result["download_path"] = str(downloaded)
                    result["reason"] = "yt-dlp download"

            rows_out.append(result)

    review_csv.parent.mkdir(parents=True, exist_ok=True)
    with review_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    counts = {"downloaded": 0, "accepted_dry_run": 0, "rejected": 0, "failed": 0}
    for r in rows_out:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return rows_out, counts


def main():
    parser = argparse.ArgumentParser(
        description="Controlled web ingest pipeline with whitelist + rights checks."
    )
    parser.add_argument(
        "--manifest",
        default="data/ingest/source_manifest.csv",
        help="Input source manifest CSV path",
    )
    parser.add_argument(
        "--output-root",
        default="data/ingest/raw",
        help="Where downloaded raw clips are stored",
    )
    parser.add_argument(
        "--review-csv",
        default="data/ingest/review_candidates.csv",
        help="Output CSV with ingest status for manual review",
    )
    parser.add_argument(
        "--allowed-domains-file",
        default="data/ingest/allowed_domains.txt",
        help="Path to allowed domains text file (one per line)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and produce review CSV without downloading files",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any rows are rejected/failed",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest)
    output_root = Path(args.output_root)
    review_csv = Path(args.review_csv)
    allowed_domains_file = Path(args.allowed_domains_file)

    allowed_domains = _load_allowed_domains(allowed_domains_file)
    rows, counts = ingest_manifest(
        manifest_path=manifest,
        output_root=output_root,
        review_csv=review_csv,
        allowed_domains=allowed_domains,
        dry_run=args.dry_run,
    )

    print(f"manifest: {manifest}")
    print(f"review_csv: {review_csv}")
    print(f"total_rows: {len(rows)}")
    print(f"counts: {counts}")

    has_issues = counts.get("rejected", 0) > 0 or counts.get("failed", 0) > 0
    if args.strict and has_issues:
        print("Strict mode failed due to rejected/failed rows.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

