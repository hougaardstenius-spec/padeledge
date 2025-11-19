# streamlit_app.py
import os
import streamlit as st
from utils.shot_detector import ShotDetector
from utils.timeline import build_timeline
from utils.thumbnails import extract_thumbnail
from utils.heatmap import generate_heatmap_xy
from utils.feedback import generate_feedback

st.set_page_config(page_title="PadelEdge Pro", layout="wide")
st.title("PadelEdge – Pro Shot Analysis")

uploaded = st.file_uploader("Upload video (mp4/mov/avi)", type=["mp4","mov","avi"])
if not uploaded:
    st.info("Upload en video til venstre panel for at starte.")
    st.stop()

os.makedirs("data/uploads", exist_ok=True)
video_path = os.path.join("data", "uploads", uploaded.name)
with open(video_path, "wb") as f:
    f.write(uploaded.read())

st.video(video_path)
st.info("Kører analyse — dette kan tage ét øjeblik...")

detector = ShotDetector()
preds, timestamps, keypoints = detector.analyze(video_path)

if not preds:
    st.error("Ingen slag fundet.")
    st.stop()

events = build_timeline(preds, timestamps)
st.subheader("📌 Shot Timeline")
for ev in events:
    st.write(f"{ev['time']:.2f}s — {ev['shot'].title()}")
    thumb = extract_thumbnail(video_path, ev['time'])
    if thumb:
        st.image(thumb, width=220)

st.subheader("🔥 Impact Heatmap")
heat = generate_heatmap_xy(keypoints)
if heat:
    st.image(heat, use_column_width=True)
else:
    st.info("Ingen impact-data til heatmap.")

st.subheader("🧠 AI Coaching Feedback")
for i, (p, kp) in enumerate(zip(preds, keypoints)):
    msgs = generate_feedback(p, kp)
    st.markdown(f"**{i+1}. {p.title()}**")
    for m in msgs:
        st.markdown(f"- {m}")
