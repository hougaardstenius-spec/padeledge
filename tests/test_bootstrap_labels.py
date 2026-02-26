import csv
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
BOOTSTRAP = BASE_DIR / "scripts" / "bootstrap_labels.py"
VALIDATOR = BASE_DIR / "scripts" / "validate_labels.py"


def test_bootstrap_labels_generates_valid_csv(tmp_path):
    samples_root = tmp_path / "samples"
    (samples_root / "overhead" / "bandeja").mkdir(parents=True)
    (samples_root / "overhead" / "mystery_shot").mkdir(parents=True)
    (samples_root / "overhead" / "bandeja" / "clip1.mp4").write_bytes(b"fake")
    (samples_root / "overhead" / "mystery_shot" / "clip2.mov").write_bytes(b"fake")
    out_csv = tmp_path / "labels.csv"

    proc = subprocess.run(
        [
            sys.executable,
            str(BOOTSTRAP),
            "--samples-root",
            str(samples_root),
            "--output",
            str(out_csv),
            "--default-match-id",
            "mtest",
            "--default-player-id",
            "ptest",
        ],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert out_csv.exists()

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["match_id"] == "mtest"
    assert rows[0]["player_id"] == "ptest"
    assert rows[0]["view"] == "end_to_end"
    assert rows[0]["quality"] == "ok"
    assert rows[0]["confidence_labeler"] == "2"
    # Unknown folder names are normalized to 'other'
    assert any(r["shot_type"] == "other" for r in rows)

    val = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--labels",
            str(out_csv),
            "--samples-root",
            str(samples_root),
        ],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )
    assert val.returncode == 0, val.stdout + "\n" + val.stderr

