import streamlit as st
from utils.stroke_classifier import classify_strokes

st.set_page_config(page_title="Padeledge Beta", page_icon="🎾", layout="wide")

st.title("🎾 PadeledgeBeta")
st.markdown("""
Welcome to **Padeledge** — your AI-powered padel performance tracker.  
Upload a short clip from your match or practice to analyze your shots!
""")

uploaded_file = st.file_uploader("📤 Upload your padel video (MP4 only)", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.info("Analyzing your video... please wait ⏳")

    # Simulate classification
    results = classify_strokes(uploaded_file)

    st.success("✅ Analysis complete!")
    st.markdown("### 📊 Stroke Summary:")
    for stroke, count in results.items():
        st.write(f"- **{stroke}**: {count} detected")

st.markdown("---")
st.caption("Padeledge © 2025 — beta version")