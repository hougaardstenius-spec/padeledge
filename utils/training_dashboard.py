import os
import glob
import sys
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "samples")
MODEL_PATH = os.path.join(BASE_DIR, "models", "shot_classifier.pkl")
LATEST_DIR = os.path.join(BASE_DIR, "models", "latest")
ARCHIVE_DIR = os.path.join(BASE_DIR, "models", "archive")


def _format_bytes(num_bytes: int) -> str:
    """Human readable filesize."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def get_dataset_stats(base_dir: str = DATA_DIR) -> pd.DataFrame:
    """Scan data/samples og returner antal videoer pr. shot-type."""
    rows = []

    if not os.path.isdir(base_dir):
        return pd.DataFrame(columns=["Category", "Shot Type", "Num Clips"])

    # forventer struktur: data/samples/<category>/<shot_type>/*.mp4
    for category in sorted(os.listdir(base_dir)):
        cat_path = os.path.join(base_dir, category)
        if not os.path.isdir(cat_path):
            continue

        for shot_type in sorted(os.listdir(cat_path)):
            shot_path = os.path.join(cat_path, shot_type)
            if not os.path.isdir(shot_path):
                continue

            video_files = []
            for ext in ("*.mp4", "*.mov", "*.avi"):
                video_files.extend(glob.glob(os.path.join(shot_path, ext)))

            rows.append({
                "Category": category,
                "Shot Type": shot_type,
                "Num Clips": len(video_files),
            })

    if not rows:
        return pd.DataFrame(columns=["Category", "Shot Type", "Num Clips"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["Category", "Shot Type"]).reset_index(drop=True)
    return df


def get_model_info() -> dict:
    """Returner info om nuværende model + evt. latest/archives."""
    info = {
        "exists": False,
        "path": MODEL_PATH,
        "modified": None,
        "size": None,
        "archive_count": 0,
        "latest_path": None,
        "latest_modified": None,
    }

    # Klassisk model path
    if os.path.exists(MODEL_PATH):
        info["exists"] = True
        info["modified"] = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        info["size"] = _format_bytes(os.path.getsize(MODEL_PATH))

    # Latest-folder
    latest_model = os.path.join(LATEST_DIR, "shot_classifier.pkl")
    if os.path.exists(latest_model):
        info["latest_path"] = latest_model
        info["latest_modified"] = datetime.fromtimestamp(os.path.getmtime(latest_model))

    # Archive count
    if os.path.isdir(ARCHIVE_DIR):
        archive_files = glob.glob(os.path.join(ARCHIVE_DIR, "shot_classifier_*.pkl"))
        info["archive_count"] = len(archive_files)

    return info


def run_training_script() -> str:
    """
    Kør træningsscriptet synkront og returnér log-output som tekst.
    Kører via det samme Python-interpreter som Streamlit (sys.executable).
    """
    st.info("⏳ Starter træning... dette kan tage lidt tid afhængigt af antal videoer.")
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "scripts", "train_shot_model.py")],
            capture_output=True,
            text=True,
            cwd=BASE_DIR,
        )
    except Exception as e:
        return f"❌ Fejl ved kørsel af træningsscript: {e}"

    output = []
    if result.stdout:
        output.append("STDOUT:\n" + result.stdout)
    if result.stderr:
        output.append("\nSTDERR:\n" + result.stderr)

    if not output:
        output.append("⚠️ Træningsscriptet kørte uden output.")

    return "\n".join(output)


def render_training_dashboard():
    """Hovedentry: kaldes fra streamlit_app.py for at vise Training Dashboard."""
    st.title("🧠 Training Dashboard V1")
    st.write(
        "Overblik over træningsdata, modelstatus og mulighed for at trigge en ny træning."
    )

    # ---------- MODEL INFO ----------
    st.subheader("📦 Modelstatus")

    info = get_model_info()
    cols = st.columns(3)

    with cols[0]:
        st.metric(
            "Model findes",
            "Ja ✅" if info["exists"] else "Nej ❌",
        )

    with cols[1]:
        if info["modified"]:
            st.metric(
                "Sidst opdateret (models/shot_classifier.pkl)",
                info["modified"].strftime("%Y-%m-%d %H:%M"),
            )
        else:
            st.metric("Sidst opdateret", "—")

    with cols[2]:
        st.metric(
            "Arkiverede versioner",
            str(info["archive_count"]),
        )

    if info["size"]:
        st.caption(f"Modelstørrelse: {info['size']}")

    if info["latest_modified"]:
        st.caption(
            f"Latest-model sidst opdateret: {info['latest_modified'].strftime('%Y-%m-%d %H:%M')} "
            f"({info['latest_path']})"
        )

    st.markdown("---")

    # ---------- DATASET STATISTICS ----------
    st.subheader("📊 Dataset – oversigt over træningsklip")

    df = get_dataset_stats()
    if df.empty:
        st.warning(
            "Ingen træningsdata fundet i `data/samples`. "
            "Læg videoer i fx `data/samples/overhead/bandeja/*.mp4`."
        )
    else:
        st.dataframe(df, use_container_width=True)

        # Simple opsummering
        total_clips = int(df["Num Clips"].sum())
        num_shots = df["Shot Type"].nunique()
        num_categories = df["Category"].nunique()

        c1, c2, c3 = st.columns(3)
        c1.metric("Antal klip", total_clips)
        c2.metric("Shot-typer", num_shots)
        c3.metric("Kategorier", num_categories)

    st.markdown("---")

    # ---------- TRAIN NOW ----------
    st.subheader("🚀 Træn model nu")

    st.write(
        "Klik på knappen nedenfor for at køre træningsscriptet `scripts/train_shot_model.py`. "
        "Dette vil læse alle videoer i `data/samples`, træne modellen og opdatere `models/shot_classifier.pkl`."
    )

    if st.button("🔁 Kør træning nu", type="primary"):
        with st.spinner("Træner model..."):
            logs = run_training_script()
        st.success("Træning gennemført (eller forsøgt) – se log nedenfor.")
        st.text_area(
            "Træningslog",
            logs,
            height=300,
        )
    else:
        st.info("Tryk på knappen for at starte en manuel træning.")
