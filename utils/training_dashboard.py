import os
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.dataset_manager import get_dataset_overview, list_sample_videos
from utils.model_versions import get_current_model_overview, list_model_versions
from utils.metrics import load_metrics_summary
from utils.training_api import run_training_now, load_training_log
from utils.labeling_ui import render_labeling_ui


def _render_overview():
    st.subheader("📌 Overblik")

    model_info = get_current_model_overview()
    metrics = load_metrics_summary()
    df = get_dataset_overview()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Model tilgængelig", "Ja ✅" if model_info["exists"] else "Nej ❌")
        if model_info["path"]:
            st.caption(f"Path: `{model_info['path']}`")

    with c2:
        if model_info["modified"]:
            st.metric(
                "Sidst opdateret",
                model_info["modified"].strftime("%Y-%m-%d %H:%M"),
            )
        else:
            st.metric("Sidst opdateret", "—")

        st.metric("Arkiverede versioner", model_info["archive_count"])

    with c3:
        total_clips = int(df["Num Clips"].sum()) if not df.empty else 0
        n_shots = df["Shot Type"].nunique() if not df.empty else 0
        st.metric("Træningsklip i dataset", total_clips)
        st.metric("Shot-typer", n_shots)

    st.markdown("---")

    st.subheader("📊 Model metrics (seneste træning)")
    if not metrics:
        st.info(
            "Ingen metrics fundet endnu. "
            "Du kan udvide træningsscriptet til at skrive `models/metrics.json`."
        )
    else:
        acc = metrics.get("accuracy", None)
        if acc is not None:
            st.metric("Accuracy", f"{acc:.3f}")

        per_class = metrics.get("per_class", {})
        if per_class:
            rows = []
            for label, m in per_class.items():
                rows.append(
                    {
                        "Label": label,
                        "Precision": m.get("precision"),
                        "Recall": m.get("recall"),
                        "F1": m.get("f1-score") or m.get("f1"),
                    }
                )
            mdf = pd.DataFrame(rows)
            st.dataframe(mdf, use_container_width=True)
        else:
            st.write("Ingen per-klasse metrics tilgængelige.")


def _render_dataset_tab():
    st.subheader("📂 Dataset Explorer")
    df = get_dataset_overview()
    if df.empty:
        st.warning(
            "Ingen træningsdata fundet i `data/samples`. "
            "Tilføj klip via 'Labeling'-fanen eller direkte i filsystemet."
        )
        return

    st.dataframe(df[["Category", "Shot Type", "Num Clips"]], use_container_width=True)

    st.markdown("### Eksempelklip")
    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox(
            "Vælg kategori",
            sorted(df["Category"].unique()),
        )
    with col2:
        subset = df[df["Category"] == category]
        shot_type = st.selectbox(
            "Vælg shot-type",
            sorted(subset["Shot Type"].unique()),
        )

    sample_paths = list_sample_videos(category, shot_type, limit=6)
    if not sample_paths:
        st.info("Ingen eksempler fundet for denne kombination.")
    else:
        cols = st.columns(3)
        for i, path in enumerate(sample_paths):
            with cols[i % 3]:
                st.caption(os.path.basename(path))
                st.video(path)


def _render_versions_tab():
    st.subheader("🧬 Model-versioner")
    versions = list_model_versions()
    if not versions:
        st.info("Ingen arkiverede modeller fundet i `models/archive/`.")
        return

    rows = []
    for v in versions:
        rows.append(
            {
                "Name": v["name"],
                "Modified": v["modified"].strftime("%Y-%m-%d %H:%M"),
                "Size": v["size"],
                "Path": v["path"],
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.caption(
        "Hver gang du træner og arkiverer, bør en ny fil blive tilføjet i `models/archive/`."
    )


def _render_training_tab():
    st.subheader("🚀 Manuel træning")

    st.write(
        "Klik på knappen for at køre træningsscriptet `scripts/train_shot_model.py` direkte "
        "fra appen. Det vil bruge alle videoer i `data/samples` og opdatere modellen."
    )

    if st.button("🔁 Kør træning nu", type="primary"):
        with st.spinner("Træner model... hold øje med loggen nedenfor."):
            logs = run_training_now()
        st.success("Træning gennemført (eller forsøgt). Se log nedenfor.")
        st.text_area("Træningslog (seneste run)", logs, height=300)
    else:
        st.info("Ingen manuel træning kørt i denne session endnu.")

    st.markdown("### Seneste log fra manuel træning")
    log_text = load_training_log()
    st.text_area("Tidligere log", log_text, height=250)


def _render_active_learning_tab():
    st.subheader("🧠 Active Learning (V2 placeholder)")
    st.write(
        "Her kan du på sigt vise klip, hvor modellen er usikker, og lade brugeren "
        "label dem manuelt. For nu kan du bruge 'Labeling'-fanen til at flytte og label "
        "klip fra fx `data/uncertain`."
    )
    st.info(
        "Når du senere tilføjer logik i din inferens til at gemme usikre klip i "
        "`data/uncertain`, vil de automatisk dukke op i 'Labeling'-fanen."
    )


def render_training_dashboard():
    """
    ENTRY POINT: kaldt fra streamlit_app.py
    """
    st.title("🧠 Training Dashboard V2")
    st.write(
        "Overblik over træningsdata, modelversioner, manuelle træningskørsler og "
        "et enkelt labeling-UI til at udvide dit dataset."
    )

    tabs = st.tabs(
        [
            "Overview",
            "Dataset",
            "Labeling",
            "Versions",
            "Training",
            "Active Learning",
        ]
    )

    with tabs[0]:
        _render_overview()
    with tabs[1]:
        _render_dataset_tab()
    with tabs[2]:
        render_labeling_ui()
    with tabs[3]:
        _render_versions_tab()
    with tabs[4]:
        _render_training_tab()
    with tabs[5]:
        _render_active_learning_tab()
