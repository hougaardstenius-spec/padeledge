import os
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.dataset_manager import get_dataset_overview, list_sample_videos
from utils.model_versions import get_current_model_overview, list_model_versions
from utils.metrics import load_metrics_summary
from utils.training_api import run_training_now, load_training_log
from utils.labeling_ui import render_labeling_ui

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUTO_RETRAIN_LOG = os.path.join(BASE_DIR, "models", "auto_retrain.log")


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
            "Træningsscriptet skriver til `models/metrics.json`, når træning er kørt succesfuldt."
        )
    else:
        if metrics.get("dummy"):
            st.warning("⚠️ Seneste model er en dummy-model (ingen rigtig træning).")

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
                        "F1": m.get("f1"),
                        "Support": m.get("support"),
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
            "Ingen træningsdata fundet i `data/samples`.\n"
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


def _render_health_tab():
    st.subheader("🩺 Dataset Health")

    df = get_dataset_overview()
    if df.empty:
        st.warning("Ingen træningsdata tilgængelig. Kan ikke beregne health.")
        return

    total_clips = int(df["Num Clips"].sum())
    st.write(f"Samlet antal klip i dataset: **{total_clips}**")

    # Class imbalance
    min_clips = int(df["Num Clips"].min())
    max_clips = int(df["Num Clips"].max())

    st.write(f"Mindste antal klip i en shot-type: **{min_clips}**")
    st.write(f"Største antal klip i en shot-type: **{max_clips}**")

    if min_clips == 0:
        st.error("❌ Der findes shot-typer med 0 klip. Modellen kan ikke lære dem.")
    elif min_clips < 5:
        st.warning("⚠️ Nogle shot-typer har meget få klip (<5). Modellen bliver ustabil dér.")

    if max_clips > 0 and min_clips > 0 and max_clips / max(min_clips, 1) > 10:
        st.warning(
            "⚠️ Dataset er stærkt ubalanceret (nogle klasser har >10x flere klip end andre)."
        )

    st.markdown("### Fordeling pr. shot-type")
    st.bar_chart(df.set_index("Shot Type")["Num Clips"])


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
        "Hver gang du træner, arkiveres tidligere modeller automatisk i `models/archive/`."
    )


def _load_auto_retrain_log() -> str:
    if not os.path.exists(AUTO_RETRAIN_LOG):
        return "Ingen auto-retrain logfil fundet endnu."
    try:
        with open(AUTO_RETRAIN_LOG, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Fejl ved læsning af auto-retrain log: {e}"


def _render_training_tab():
    st.subheader("🚀 Manuel træning")

    st.write(
        "Klik på knappen for at køre træningsscriptet `scripts/train_shot_model.py` direkte "
        "fra appen. Det vil bruge alle videoer i `data/samples`, træne modellen, "
        "opdatere `models/shot_classifier.pkl` og skrive metrics + arkiver."
    )

    if st.button("🔁 Kør træning nu", type="primary"):
        with st.spinner("Træner model... hold øje med loggen nedenfor."):
            logs = run_training_now()
        st.success("Træning gennemført (eller forsøgt). Se log nedenfor.")
        st.text_area("Træningslog (seneste run)", logs, height=300)

        st.markdown("### Opdaterede metrics")
        metrics = load_metrics_summary()
        if metrics:
            st.json(metrics)
        else:
            st.info("Ingen metrics kunne læses efter træning.")
    else:
        st.info("Ingen manuel træning kørt i denne session endnu.")

    st.markdown("### Seneste log fra manuel træning")
    log_text = load_training_log()
    st.text_area("Tidligere log", log_text, height=250)

    st.markdown("---")
    st.subheader("🤖 Auto-retrain log (fra ShotDetector)")

    auto_log = _load_auto_retrain_log()
    st.text_area("Auto-retrain log", auto_log, height=250)


def _render_active_learning_tab():
    st.subheader("🧠 Active Learning (V2 placeholder)")
    st.write(
        "Her kan du på sigt vise klip, hvor modellen er usikker, og lade brugeren "
        "label dem manuelt. Lige nu kan du bruge 'Labeling'-fanen til at label klip "
        "fra fx `data/uncertain`."
    )
    st.info(
        "Når inferens-logik senere gemmer usikre klip i `data/uncertain`, vil de "
        "dukke op under 'Labeling' → 'Review uncertain clips'."
    )


def render_training_dashboard():
    """
    ENTRY POINT: kaldt fra streamlit_app.py
    """
    st.title("🧠 Training Dashboard V2")
    st.write(
        "Overblik over træningsdata, modelversioner, manuelle træningskørsler og "
        "et labeling-UI til at udvide dit dataset."
    )

    tabs = st.tabs(
        [
            "Overview",
            "Dataset",
            "Health",
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
        _render_health_tab()
    with tabs[3]:
        render_labeling_ui()
    with tabs[4]:
        _render_versions_tab()
    with tabs[5]:
        _render_training_tab()
    with tabs[6]:
        _render_active_learning_tab()
