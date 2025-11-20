import os
import sys
import subprocess
from typing import Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_SCRIPT = os.path.join(BASE_DIR, "scripts", "train_shot_model.py")
TRAIN_LOG = os.path.join(BASE_DIR, "models", "train_last.log")


def run_training_now() -> str:
    """
    Kører train_shot_model.py synkront med samme Python som Streamlit.
    Returnerer samlet log (stdout + stderr) som tekst og gemmer den i models/train_last.log.
    """
    if not os.path.exists(TRAIN_SCRIPT):
        return f"❌ Træningsscript ikke fundet: {TRAIN_SCRIPT}"

    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT],
            capture_output=True,
            text=True,
            cwd=BASE_DIR,
        )
    except Exception as e:
        return f"❌ Fejl ved kørsel af træningsscript: {e}"

    output_parts = []
    if result.stdout:
        output_parts.append("STDOUT:\n" + result.stdout)
    if result.stderr:
        output_parts.append("\nSTDERR:\n" + result.stderr)

    if not output_parts:
        output_parts.append("⚠️ Træningsscriptet kørte uden output.")

    combined = "\n".join(output_parts)

    # gem også til fil
    os.makedirs(os.path.dirname(TRAIN_LOG), exist_ok=True)
    with open(TRAIN_LOG, "w", encoding="utf-8") as f:
        f.write(combined)

    return combined


def load_training_log() -> str:
    """
    Læser seneste træningslog (fra manual træning via dashboardet).
    """
    if not os.path.exists(TRAIN_LOG):
        return "Ingen træningslog fundet endnu. Kør en manuel træning først."

    try:
        with open(TRAIN_LOG, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"❌ Fejl ved læsning af logfil: {e}"
