import base64
import streamlit as st
import pandas as pd
import numpy as np
from utils.video_processor import extract_keypoints_from_video
from utils.shot_detector import detect_shots
from utils.match_analyzer import analyze_match

# --------------------------------------------
# Page Setup
# --------------------------------------------
st.set_page_config(page_title="🎾 PadelEdge AI – Beta", layout="wide")

# --------------------------------------------
# Custom Background Function
# --------------------------------------------
def add_background(image_file):
    """Adds a full-page background image."""
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: rgba(0,0,0,0);
    }}
    [data-testid="stSidebar"] {{
        background: rgba(255,255,255,0.85);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ✅ Add your PNG background (file should be in same folder as this script)
add_background("background.png")

# --------------------------------------------
# Custom CSS for Overlay Dashboard
# --------------------------------------------
st.markdown("""
<style>
.dashboard-card {
    background: rgba(0, 0, 0, 0.6);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1rem;
    color: white;
    box-shadow: 0 0 25px rgba(0,0,0,0.3);
}
h1, h2, h3 {
    color: #00ffcc;
}
.metric-label {
    font-size: 1.2rem;
    font-weight: 500;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #00ffcc;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------
# Title and Upload Section
# --------------------------------------------
st.title("🎾 PadelEdge AI – Match Analyzer (Beta)")
st.write("Upload a short Padel video to test the AI-driven stroke recognition system.")

uploaded_file = st.file_uploader("📤 Upload your match video", type=["mp4", "mov", "avi"])

# --------------------------------------------
# Video Analysis Workflow
# --------------------------------------------
if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("temp_video.mp4")
    st.write("Analyzing match... please wait ⏳")

    keypoints = extract_keypoints_from_video("temp_video.mp4")
    shots = detect_shots(keypoints)
    summary = analyze_match(shots)

    # Display results inside a transparent dashboard overlay
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.header("📊 Match Summary")

    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="metric-label">Winners</div><div class="metric-value">{summary["winners"]}</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-label">Forced Errors</div><div class="metric-value">{summary["forced_errors"]}</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-label">Unforced Errors</div><div class="metric-value">{summary["unforced_errors"]}</div>', unsafe_allow_html=True)

    st.subheader("🏆 Estimated Score")
    st.markdown(f"<h2 style='text-align:center;'>{summary['score']}</h2>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close overlay card

else:
    st.info("Please upload a video to start your analysis 🎥")
