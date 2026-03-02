import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
CHECKER = BASE_DIR / "scripts" / "check_label_conflicts.py"


def test_check_label_conflicts_detects_mismatch(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "\n".join(
            [
                "match_id,player_id,view,clip_path,shot_type,quality,confidence_labeler,outcome,notes",
                "m1,p1,end_to_end,overhead/bandeja/a.mp4,vibora,good,3,,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(CHECKER), "--labels", str(labels)],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )
    assert proc.returncode != 0
    assert "conflicts: 1" in proc.stdout


def test_check_label_conflicts_passes_when_consistent(tmp_path):
    labels = tmp_path / "labels_ok.csv"
    labels.write_text(
        "\n".join(
            [
                "match_id,player_id,view,clip_path,shot_type,quality,confidence_labeler,outcome,notes",
                "m1,p1,end_to_end,overhead/bandeja/a.mp4,bandeja,good,3,,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(CHECKER), "--labels", str(labels)],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "No label conflicts found." in proc.stdout

