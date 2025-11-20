# streamlit_app.py
import os
import streamlit as st

# --- Match Analyzer Utils ---
from utils.shot_detector import ShotDetector
from utils.timeline import build_timeline
from utils.thumbnails import extract_thumbnail
from utils.heatmap import generate_heatmap_xy
from utils.feedback import generate_feedback

# --- Training Dashboard ---
from utils.training_dashboard import render_training_dashboard

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="PadelEdge Pro",
    layout="wide",
    page_icon="🎾"
)

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Menu",
    ["Match Analyzer", "Training Dashboard"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("PadelEdge Pro • AI Shot Recognition")

# =========================================================
#  PAGE 1 — MATCH ANALYZER
# =========================================================
if page == "Match Analyzer":

    st.title("🎾 PadelEdge – Pro Shot Analysis")
    st.write("Upload en video for at analysere slag, positioner og få AI feedback.")

    # ------------------------------------
    # Video upload
    # ------------------------------------
    uploaded = st.file_uploader(
        "Upload video (mp4/mov/avi)",
        type=["mp4", "mov", "avi"]
    )
    if not uploaded:
        st.info("Upload en video til venstre panel for at starte.")
        st.stop()

    # Save upload
    os.makedirs("data/uploads", exist_ok=True)
    video_path = os.path.join("data", "uploads", uploaded.name)
    with open(video_path, "wb") as f:
        f.write(uploaded.read())

    st.video(video_path)
    st.info("Kører analyse — dette kan tage ét øjeblik...")

    # ------------------------------------
    # Run shot analysis
    # ------------------------------------
    detector = ShotDetector()
    preds, timestamps, keypoints = detector.analyze(video_path)

    if not preds:
        st.error("Ingen slag fundet.")
        st.stop()

    # ------------------------------------
    # Shot Timeline
    # ------------------------------------
    st.subheader("📌 Shot Timeline")
    events = build_timeline(preds, timestamps)

    cols = st.columns(3)
    for i, ev in enumerate(events):
        with cols[i % 3]:
            st.markdown(f"**{ev['shot'].title()}** — {ev['time']:.2f}s")
            thumb = extract_thumbnail(video_path, ev["time"])
            if thumb:
                st.image(thumb, width=220)
            st.markdown("---")

    # ------------------------------------
    # Impact Heatmap
    # ------------------------------------
    st.subheader("🔥 Impact Heatmap")
    heat = generate_heatmap_xy(keypoints)
    if heat:
        st.image(heat, use_column_width=True)
    else:
        st.info("Ingen impact-data til heatmap.")

    # ------------------------------------
    # AI Feedback
    # ------------------------------------
    st.subheader("🧠 AI Coaching Feedback")

    for i, (shot, kp) in enumerate(zip(preds, keypoints)):
        st.markdown(f"### {i+1}. {shot.title()}")
        msgs = generate_feedback(shot, kp)
        for m in msgs:
            st.markdown(f"- {m}")
        st.markdown("---")

# =========================================================
#  PAGE 2 — TRAINING DASHBOARD
# =========================================================
elif page == "Training Dashboard":
    render_training_dashboard()
