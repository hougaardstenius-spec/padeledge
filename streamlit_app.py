import streamlit as st
from utils.stroke_classifier import classify_strokes

st.set_page_config(page_title="PadelEdge – Beta AI", layout="wide")

st.title("🎾 PadelEdge Beta – AI Stroke Analyzer")
st.write("Upload a short padel clip to test early AI-based stroke detection.")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file:
    st.video(uploaded_file)
    with st.spinner("Analyzing your strokes..."):
        results = classify_strokes(uploaded_file)
    st.success(results.get("status", "Done!"))
    st.write("### Breakdown:")
    st.json(results["details"])
else:
    st.info("Please upload a video to start.")
