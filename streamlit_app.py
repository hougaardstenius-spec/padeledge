import base64
import streamlit as st
import pandas as pd
import numpy as np
import time

from utils.video_processor import extract_keypoints_from_video
from utils.shot_detector import detect_shots
from utils.match_analyzer import analyze_match


# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="PadelEdge AI – Beta",
    layout="wide",
    page_icon="🎾"
)


# -------------------------------------------------------
# ADD BACKGROUND IMAGE
# -------------------------------------------------------
def add_background(image_file: str):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background: url("data:image/png;base64,{encoded}") no-repeat center center;
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except:
        pass


add_background("background.png")


# -------------------------------------------------------
# CUSTOM PRO CSS
# -------------------------------------------------------
st.markdown("""
<style>
:root {
    --accent: #00ffcc;
    --accent2: #00c3ff;
    --text-light: #e6eef6;
}

/* Global text */
h1, h2, h3, h4 {
    color: var(--accent);
    font-weight: 700;
}

/* LOGO SECTION */
.padel-logo-wrap {
    display: flex;
    align-items: center;
    gap: 1.4rem;
    padding: 1rem 0 2.5rem 0;
}

/* Glass cards */
.padel-card {
    background: rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(8px);
    border-radius: 22px;
    padding: 2rem;
    margin-top: 1rem;
    box-shadow: 0 0 25px rgba(0,0,0,0.3);
}

/* Upload area */
.padel-upload {
    padding: 2rem;
    text-align: center;
    border: 2px dashed rgba(255,255,255,0.25);
    border-radius: 18px;
    background: rgba(0,0,0,0.4);
}

/* Metrics */
.metric-box {
    background: rgba(255,255,255,0.08);
    padding: 1.2rem;
    border-radius: 16px;
    text-align: center;
    color: white;
}
.metric-label {
    font-size: 1.1rem;
    opacity: 0.85;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
}

/* Sticky top progress */
.sticky-progress {
    position: sticky;
    top: 0;
    z-index: 999;
    padding: 0.7rem;
    backdrop-filter: blur(10px);
    background: rgba(0,0,0,0.7);
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

/* Centered video */
.video-container {
    display: flex;
    justify-content: center;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)



# -------------------------------------------------------
# HEADER SECTION
# -------------------------------------------------------
logo_col, title_col = st.columns([1,5])
with logo_col:
    st.image("Designer-5.png", width=120)

with title_col:
    st.markdown("""
    <div class="padel-logo-wrap">
        <div>
            <h1>PadelEdge AI Match Analyzer</h1>
            <h3 style="margin-top:-10px; color:#e6eef6">
                Data Driven Padel Performance
            </h3>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------
# UPLOAD UI
# -------------------------------------------------------
st.markdown("<div class='padel-card'>", unsafe_allow_html=True)
st.markdown("<h2>📤 Upload Your Match Video</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a Padel Video",
    type=["mp4", "mov", "avi"],
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)



# -------------------------------------------------------
# ANALYSIS WORKFLOW
# -------------------------------------------------------
if uploaded_file:

    # Sticky progress top bar
    st.markdown("<div class='sticky-progress'>", unsafe_allow_html=True)
    progress_text = st.empty()
    progress_bar = st.progress(0)
    st.markdown("</div>", unsafe_allow_html=True)

    # Store temp video
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.markdown("<div class='video-container'>", unsafe_allow_html=True)
    st.video("temp_video.mp4")
    st.markdown("</div>", unsafe_allow_html=True)

    # STEP 1
    progress_text.text("🔍 Extracting keypoints from video...")
    keypoints = extract_keypoints_from_video("temp_video.mp4")
    time.sleep(0.6)
    progress_bar.progress(30)

    # STEP 2
    progress_text.text("🎯 Detecting stroke types...")
    shots = detect_shots(keypoints)
    unique_shots, counts = np.unique(shots, return_counts=True)
    time.sleep(0.6)
    progress_bar.progress(65)

    # STEP 3
    progress_text.text("📊 Computing match statistics...")
    summary = analyze_match(shots)
    time.sleep(0.6)
    progress_bar.progress(100)

    progress_text.text("✅ Analysis complete!")


    # ------------------------------------------------
    # RESULTS UI
    # ------------------------------------------------
    st.markdown("<div class='padel-card'>", unsafe_allow_html=True)
    st.header("📊 Match Summary")

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Winners</div>
            <div class="metric-value">{summary['winners']}</div>
        </div>
    """, unsafe_allow_html=True)
    c2.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Forced Errors</div>
            <div class="metric-value">{summary['forced_errors']}</div>
        </div>
    """, unsafe_allow_html=True)
    c3.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Unforced Errors</div>
            <div class="metric-value">{summary['unforced_errors']}</div>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("🏆 Estimated Score")
    st.markdown(
        f"<h2 style='text-align:center; color:var(--accent)'>{summary['score']}</h2>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


else:
    st.info("Upload a video to begin 🎥")
