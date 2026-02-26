import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
VALIDATOR = BASE_DIR / "scripts" / "validate_labels.py"
SAMPLE_CLIP = (
    BASE_DIR / "data" / "samples" / "overhead" / "bandeja" / "Bandeja 2.mp4"
)


def test_validate_labels_passes_with_valid_row(tmp_path):
    csv_path = tmp_path / "labels.csv"
    rel_clip = SAMPLE_CLIP.relative_to(BASE_DIR / "data" / "samples")
    csv_path.write_text(
        "\n".join(
            [
                "match_id,player_id,view,clip_path,shot_type,quality,confidence_labeler,outcome,notes",
                f"m1,p1,end_to_end,{rel_clip},bandeja,good,3,winner,clean contact",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--labels",
            str(csv_path),
            "--samples-root",
            str(BASE_DIR / "data" / "samples"),
        ],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "Validation passed." in proc.stdout


def test_validate_labels_fails_on_invalid_values(tmp_path):
    csv_path = tmp_path / "labels_invalid.csv"
    csv_path.write_text(
        "\n".join(
            [
                "match_id,player_id,view,clip_path,shot_type,quality,confidence_labeler,outcome,notes",
                "m1,p1,side_view,missing.mp4,unknown_shot,bad,7,bad_outcome,broken row",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--labels",
            str(csv_path),
            "--samples-root",
            str(BASE_DIR / "data" / "samples"),
        ],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )
    assert proc.returncode != 0
    assert "Validation failed:" in proc.stdout

