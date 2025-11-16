import base64
import streamlit as st
import pandas as pd
import numpy as np
import time
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
def add_background(image_file: str):
    """Adds a full-page background image using base64 encoding."""
    try:
        with open(image_file, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()

        css = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """

        st.markdown(css, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error(f"❌ Background image not found: {image_file}")
    except Exception as e:
        st.error(f"❌ Error loading background: {e}")

# Inject background
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
# Video Analysis Workflow with Progress Bar
# --------------------------------------------
if uploaded_file:

    # Save temporary video
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("temp_video.mp4")
    st.markdown("### ⏳ Analyzing match... please wait")

    progress_text = st.empty()
    progress_bar = st.progress(0)

    # STEP 1: Extract features
    progress_text.text("🔍 Extracting player movements from video...")
    time.sleep(1)
    keypoints = extract_keypoints_from_video("temp_video.mp4")
    progress_bar.progress(30)

    # STEP 2: Identify strokes
    progress_text.text("🎯 Identifying stroke types (AI model in action)...")
    shot_counter_placeholder = st.empty()

    shots = detect_shots(keypoints)
    unique_shots, counts = np.unique(shots, return_counts=True)

    detected_summary = {}
    for i, stroke in enumerate(unique_shots):
        detected_summary[stroke] = counts[i]
        time.sleep(0.3)

        total = sum(detected_summary.values())
        shot_counter_placeholder.markdown(
            f"**Detected {total} strokes so far:** " +
            ", ".join([f"{k}: {v}" for k, v in detected_summary.items()])
        )

        progress_bar.progress(40 + int((i / len(unique_shots)) * 25))

    # STEP 3: Analyze match
    progress_text.text("📊 Compiling match report and statistics...")
    time.sleep(1)
    summary = analyze_match(shots)
    progress_bar.progress(100)

    # STEP 4: Display results
    progress_text.text("✅ Analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()

    # Dashboard block
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.header("📊 Match Summary")

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f'<div class="metric-label">Winners</div><div class="metric-value">{summary["winners"]}</div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        f'<div class="metric-label">Forced Errors</div><div class="metric-value">{summary["forced_errors"]}</div>',
        unsafe_allow_html=True,
    )
    col3.markdown(
        f'<div class="metric-label">Unforced Errors</div><div class="metric-value">{summary["unforced_errors"]}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("🏆 Estimated Score")
    st.markdown(f"<h2 style='text-align:center;'>{summary['score']}</h2>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Please upload a video to start your analysis 🎥")
