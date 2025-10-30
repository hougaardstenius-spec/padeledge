import streamlit as st
from utils.video_processor import extract_keypoints_from_video
from utils.shot_detector import detect_shots
from utils.match_analyzer import analyze_match

st.title("🎾 Padel Match Analyzer")

uploaded_file = st.file_uploader("Upload your match video", type=["mp4", "mov", "avi"])
if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("temp_video.mp4")
    st.write("Analyzing video... please wait ⏳")

    keypoints = extract_keypoints_from_video("temp_video.mp4")
    shots = detect_shots(keypoints)
    summary = analyze_match(shots)

    st.success("✅ Analysis complete!")
    st.subheader("📊 Match Summary")
    st.write(f"Winners: {summary['winners']}")
    st.write(f"Forced Errors: {summary['forced_errors']}")
    st.write(f"Unforced Errors: {summary['unforced_errors']}")
    st.write(f"Total Points: {summary['points_played']}")
    st.write(f"Score Estimate: {summary['score']}")
