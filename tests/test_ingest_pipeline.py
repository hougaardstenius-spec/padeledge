import csv
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
INGEST = BASE_DIR / "scripts" / "ingest_pipeline.py"


def test_ingest_pipeline_dry_run_writes_review_csv(tmp_path):
    manifest = tmp_path / "source_manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "url,source_name,license,rights_confirmed,approved_for_training,view,expected_shot_type,match_id,player_id,notes",
                "https://example.com/clip1.mp4,Example Source,CC-BY,yes,yes,end_to_end,bandeja,m1,p1,ok",
                "https://bad-domain.net/clip2.mp4,Bad Source,CC-BY,yes,yes,end_to_end,vibora,m2,p2,blocked domain",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    domains = tmp_path / "allowed_domains.txt"
    domains.write_text("example.com\n", encoding="utf-8")
    review = tmp_path / "review_candidates.csv"
    raw = tmp_path / "raw"

    proc = subprocess.run(
        [
            sys.executable,
            str(INGEST),
            "--manifest",
            str(manifest),
            "--output-root",
            str(raw),
            "--review-csv",
            str(review),
            "--allowed-domains-file",
            str(domains),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert review.exists()

    with review.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    statuses = {r["status"] for r in rows}
    assert "accepted_dry_run" in statuses
    assert "rejected" in statuses


def test_ingest_pipeline_strict_fails_on_rejections(tmp_path):
    manifest = tmp_path / "source_manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "url,source_name,license,rights_confirmed,approved_for_training,view,expected_shot_type,match_id,player_id,notes",
                "https://example.com/clip1.mp4,Example Source,CC-BY,no,yes,end_to_end,bandeja,m1,p1,rights missing",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    domains = tmp_path / "allowed_domains.txt"
    domains.write_text("example.com\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(INGEST),
            "--manifest",
            str(manifest),
            "--review-csv",
            str(tmp_path / "review.csv"),
            "--allowed-domains-file",
            str(domains),
            "--dry-run",
            "--strict",
        ],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )
    assert proc.returncode != 0
    assert "Strict mode failed" in proc.stdout

