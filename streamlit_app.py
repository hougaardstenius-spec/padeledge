import streamlit as st
import os
from utils.shot_detector import ShotDetector
from utils.timeline import build_timeline
from utils.thumbnails import extract_thumbnail
from utils.heatmap import generate_heatmap_xy
from utils.feedback import generate_feedback

import cv2

st.set_page_config(
    page_title="Padeledge – Pro Padel Shot Analysis",
    page_icon="🎾",
    layout="wide"
)

# --- LOAD MODEL ---
detector = ShotDetector()

# --- SIDEBAR ---
st.sidebar.title("🎾 Padeledge Pro")
uploaded_video = st.sidebar.file_uploader("Upload din padelvideo", type=["mp4","mov","avi"])

if uploaded_video:
    video_path = f"data/uploads/{uploaded_video.name}"
    os.makedirs("data/uploads", exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.sidebar.success("Video uploaded ✔")
else:
    st.sidebar.info("Upload a video to begin")
    st.stop()

# --- MAIN CONTENT ---
st.title("Padel Shot Analysis Dashboard")

st.video(video_path)


# --- PROCESSING ---
with st.spinner("Analyserer video…"):
    preds, timestamps, keypoints = detector.analyze(video_path)

if len(preds) == 0:
    st.error("Kunne ikke finde slag i videoen.")
    st.stop()

timeline = build_timeline(preds, timestamps)


# --- TIMELINE SECTION ---
st.subheader("📌 Shot Timeline")

cols = st.columns(6)

timeline_data = []
thumb_paths = []

for i, event in enumerate(timeline):
    thumb = extract_thumbnail(video_path, event["time"])
    thumb_paths.append(thumb)

    timeline_data.append({
        "shot": event["shot"],
        "time": event["time"],
        "thumb": thumb,
        "color": event["color"]
    })


for item in timeline_data:
    with st.container(border=True):
        st.image(item["thumb"], width=90)
        st.markdown(f"**{item['shot'].title()}**")
        st.markdown(f"⏱ {item['time']:.2f} sec")


# --- HEATMAP ---
st.subheader("🔥 Heatmap of Shot Positions")

heatmap_path = generate_heatmap_xy(keypoints, "data/heatmaps/latest.png")
st.image(heatmap_path, caption="Shot Impact Heatmap")


# --- FEEDBACK ---
st.subheader("💡 AI Coaching Feedback")

for shot, kp_frame in zip(preds, keypoints):
    fb = generate_feedback(shot, kp_frame)
    with st.container(border=True):
        st.markdown(f"### {shot.title()}")
        for line in fb:
            st.markdown(f"- {line}")
