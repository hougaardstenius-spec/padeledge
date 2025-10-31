import base64
import streamlit as st
from utils.video_processor import extract_keypoints_from_video
from utils.shot_detector import detect_shots
from utils.match_analyzer import analyze_match

st.set_page_config(page_title="🎾 PadelEdge Beta", layout="wide")

# ------------------------------------------------------
# Custom Background
# ------------------------------------------------------
def add_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);  /* Transparent header */
    }}
    [data-testid="stSidebar"] {{
        background: rgba(255,255,255,0.85);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ✅ Apply background image
add_background("background.png")

# ------------------------------------------------------
# Main App
# ------------------------------------------------------
st.title("🎾 PadelEdge AI – Match Analyzer (Beta)")
uploaded_file = st.file_uploader("Upload your match video", type=["mp4", "mov", "avi"])
if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("temp_video.mp4")
    st.write("Analyzing match... please wait ⏳")

    keypoints = extract_keypoints_from_video("temp_video.mp4")
    shots = detect_shots(keypoints)
    summary = analyze_match(shots)

    st.success("✅ Analysis complete!")
    st.subheader("📊 Match Summary")
    st.write(f"Winners: {summary['winners']}")
    st.write(f"Forced Errors: {summary['forced_errors']}")
    st.write(f"Unforced Errors: {summary['unforced_errors']}")
    st.write(f"Score Estimate: {summary['score']}")
else:
    st.info("Upload a short Padel video to get started 🎥")
