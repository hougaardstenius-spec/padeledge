# streamlit_app.py — PadelEdge (Pro UI redesign)
import base64
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from utils.video_processor import extract_keypoints_from_video
from utils.shot_detector import detect_shots
from utils.match_analyzer import analyze_match

# -------------------------
# Config
# -------------------------
PAGE_TITLE = "🎾 PadelEdge Pro – Match Analyzer"
ACCENT_COLOR = "#00ffcc"           # neon cyan/turquoise
ACCENT_COLOR_2 = "#00C853"         # neon green (for metrics)
GLASS_BG = "rgba(255,255,255,0.04)"  # subtle glass
LOGO_CANDIDATES = ["Designer-5.png", "background.png", "logo.png", "padel_logo.png"]

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="collapsed")

# -------------------------
# Helpers: background + logo loader
# -------------------------
def add_background(image_file: str):
    """Adds a full-page background image using base64. Silently fails if file not found."""
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
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        # don't break app if background missing
        pass

def load_logo():
    """Try a set of candidate filenames for logo. Return data URI or None."""
    for candidate in LOGO_CANDIDATES:
        if os.path.exists(candidate):
            try:
                with open(candidate, "rb") as f:
                    b = f.read()
                    encoded = base64.b64encode(b).decode()
                    return f"data:image/png;base64,{encoded}"
            except Exception:
                continue
    return None

# Try to add a subtle background (optional)
add_background("background.png")

# -------------------------
# Global CSS (glass + neon + sticky)
# -------------------------
st.markdown(
    f"""
    <style>
    :root {{
        --accent: {ACCENT_COLOR};
        --accent-2: {ACCENT_COLOR_2};
        --glass: {GLASS_BG};
    }}

    /* body font + smoothing */
    html, body, [data-testid="stAppViewContainer"] {{
        -webkit-font-smoothing: antialiased;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        color: #e6eef6;
    }}

    /* Top sticky progress bar */
    .padel-sticky-top {{
        position: sticky;
        top: 0;
        z-index: 9999;
        backdrop-filter: blur(6px);
        padding: 0.6rem 1rem;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }}

    .padel-hero {{
        display:flex;
        align-items:center;
        gap:1rem;
    }}

    /* Glass card */
    .padel-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
        border-radius: 14px;
        padding: 1.25rem;
        box-shadow: 0 6px 30px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.04);
    }}

    .padel-hero-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--accent);
        margin: 0;
    }}

    .padel-hero-sub {{
        margin: 0;
        color: rgba(230,238,246,0.75);
        font-size: 0.95rem;
    }}

    .padel-upload-wrap {{
        display:flex;
        gap:1rem;
        align-items:center;
        justify-content:center;
        padding: 1rem;
        border-radius: 12px;
        background: rgba(255,255,255,0.01);
        border: 1px dashed rgba(255,255,255,0.03);
    }}

    .padel-metric {{
        background: var(--glass);
        border-radius: 12px;
        padding: 1rem;
        text-align:center;
    }}

    .metric-value {{
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--accent-2);
    }}

    .metric-label {{
        font-size: 0.9rem;
        color: rgba(230,238,246,0.8);
    }}

    /* small responsive tweaks */
    @media (max-width: 800px) {{
        .padel-hero {{
            flex-direction: column;
            text-align:center;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Utility: sticky top progress UI
# -------------------------
def show_sticky_progress(progress: int, label: str = ""):
    """Renders a slim sticky progress bar & label at the top of the page."""
    safe_label = st.session_state.get("status_text", label)
    # Render a compact top area with the progress and label
    st.markdown(
        f"""
        <div class="padel-sticky-top padel-card">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem;">
                <div style="display:flex; align-items:center; gap:1rem;">
                    <div style="font-weight:700; color:var(--accent);">PADELEDGE</div>
                    <div style="color:rgba(230,238,246,0.8); font-size:0.95rem;">{safe_label}</div>
                </div>
                <div style="min-width:220px; max-width:520px;">
                    <div style="height:10px; width:100%; background: rgba(255,255,255,0.06); border-radius:6px;">
                        <div style="height:100%; width:{progress}%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); border-radius:6px;"></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# Header / Hero Area
# -------------------------
logo_data_uri = load_logo()

header_cols = st.columns([1, 4, 1])
with header_cols[1]:
    st.markdown(
        '<div class="padel-card padel-hero" style="align-items:center;">'
        + (f'<img src="{logo_data_uri}" alt="logo" style="height:72px; border-radius:10px; box-shadow:0 8px 30px rgba(2,6,23,0.6);"/>' if logo_data_uri else "")
        + f'<div style="margin-left:0.6rem;">'
        + f'<div class="padel-hero-title">PADELEDGE — Data Driven Padel Performance</div>'
        + f'<div class="padel-hero-sub">Upload en kort kampvideo og få AI-drevet stroke analysis, taktiske indsigter og score-estimat.</div>'
        + '</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)

# Initialize session state for progress / status
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "status_text" not in st.session_state:
    st.session_state.status_text = ""

# Show sticky top (initially hidden progress 0)
show_sticky_progress(st.session_state.progress, st.session_state.status_text)

# -------------------------
# Upload area (centered)
# -------------------------
st.markdown(
    """
    <div class="padel-card" style="max-width:1100px; margin: 0 auto;">
        <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;">
            <div style="flex:1; min-width: 280px;">
                <h2 style="margin:0; color:var(--accent);">Upload din kampvideo</h2>
                <p style="margin:4px 0 0 0; color: rgba(230,238,246,0.75);">Under 90 sekunder anbefales. Vi tager os af resten — se realtime feedback nedenfor.</p>
            </div>
            <div style="min-width:320px;">
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"], key="uploader")

st.markdown(
    """
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar: only visible AFTER upload (user chose option C)
# -------------------------
def render_sidebar(summary=None):
    st.sidebar.markdown(f"<div style='padding:8px 12px; border-radius:8px; background:{GLASS_BG};'>"
                        f"<h3 style='color:var(--accent); margin:0;'>Dashboard</h3>"
                        f"<small style='color:rgba(230,238,246,0.7)'>Kampdetaljer</small></div>", unsafe_allow_html=True)
    if summary is None:
        st.sidebar.info("Ingen resultater endnu — upload en video for at få indsigt.")
        return

    # compact metrics
    st.sidebar.markdown("<br/>", unsafe_allow_html=True)
    st.sidebar.metric("Winners", summary.get("winners", 0))
    st.sidebar.metric("Forced Errors", summary.get("forced_errors", 0))
    st.sidebar.metric("Unforced Errors", summary.get("unforced_errors", 0))
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Score estimate:** {summary.get('score','-')}")
    st.sidebar.markdown("<br/>", unsafe_allow_html=True)
    st.sidebar.markdown("Download: [CSV resultater](#) (kommer snart)", unsafe_allow_html=True)

# -------------------------
# Main processing logic
# -------------------------
def process_video_and_render(temp_path):
    """Orkestrerer extraction -> shot detection -> analysis -> render results"""
    try:
        # Reset progress & status
        st.session_state.progress = 5
        st.session_state.status_text = "Starter ekstraktion..."
        show_sticky_progress(st.session_state.progress)

        # STEP 1: extract keypoints
        time.sleep(0.8)
        keypoints = extract_keypoints_from_video(temp_path)  # external util
        st.session_state.progress = 30
        st.session_state.status_text = "Ekstraktion færdig — kører stroke-detektion..."
        show_sticky_progress(st.session_state.progress)

        # STEP 2: detect shots
        time.sleep(0.6)
        shots = detect_shots(keypoints)  # external util
        st.session_state.progress = 55
        st.session_state.status_text = "Stroke-klassificering færdig — analyserer..."
        show_sticky_progress(st.session_state.progress)

        # incremental shot display
        unique_shots, counts = np.unique(shots, return_counts=True)
        detected_summary = {}
        shot_holder = st.empty()
        for i, stroke in enumerate(unique_shots):
            detected_summary[stroke] = int(counts[i])
            total = sum(detected_summary.values())
            shot_holder.markdown(
                f"**{total} strokes registreret:** " + ", ".join([f"{k}: {v}" for k, v in detected_summary.items()])
            )
            st.session_state.progress = 55 + int(((i + 1) / len(unique_shots)) * 15)
            st.session_state.status_text = f"Identificerer {stroke}..."
            show_sticky_progress(st.session_state.progress)
            time.sleep(0.25)

        # STEP 3: analyze match
        time.sleep(0.6)
        summary = analyze_match(shots)  # external util returns dict with winners, forced_errors, unforced_errors, score
        st.session_state.progress = 95
        st.session_state.status_text = "Færdiggør rapport..."
        show_sticky_progress(st.session_state.progress)
        time.sleep(0.6)

        st.session_state.progress = 100
        st.session_state.status_text = "Færdig"
        show_sticky_progress(st.session_state.progress)

        # Render results (main column)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="padel-card">', unsafe_allow_html=True)
        st.header("📊 Match Summary")
        st.write("Taktiske indsigter og nøglemetrikker fundet af AI'en")

        # 3 metric cards
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.markdown(f'<div class="padel-metric"><div class="metric-value">🏆 {summary.get("winners",0)}</div><div class="metric-label">Winners</div></div>', unsafe_allow_html=True)
        mcol2.markdown(f'<div class="padel-metric"><div class="metric-value">🎯 {summary.get("forced_errors",0)}</div><div class="metric-label">Forced Errors</div></div>', unsafe_allow_html=True)
        mcol3.markdown(f'<div class="padel-metric"><div class="metric-value">⚡ {summary.get("unforced_errors",0)}</div><div class="metric-label">Unforced Errors</div></div>', unsafe_allow_html=True)

        st.subheader("🏆 Score Estimate")
        st.markdown(f"<h2 style='text-align:center; color:var(--accent);'>{summary.get('score','-')}</h2>", unsafe_allow_html=True)

        # render sidebar now that we have summary
        render_sidebar(summary)

        st.markdown("</div>", unsafe_allow_html=True)

        return summary

    except Exception as e:
        st.error(f"Der opstod en fejl under behandling: {e}")
        st.session_state.status_text = "Fejl"
        show_sticky_progress(st.session_state.progress)
        return None

# -------------------------
# Upload handling flow
# -------------------------
if uploaded_file is not None:
    # Save temp file
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Show preview & start processing button
    preview_cols = st.columns([2, 1])
    with preview_cols[0]:
        st.video(temp_path)
    with preview_cols[1]:
        st.markdown('<div class="padel-card">', unsafe_allow_html=True)
        st.markdown("**Klar til analyse**")
        if st.button("Start analyse", key="start_btn"):
            # Reset progress
            st.session_state.progress = 0
            st.session_state.status_text = "Forbereder..."
            show_sticky_progress(st.session_state.progress, st.session_state.status_text)
            summary = process_video_and_render(temp_path)
            # optionally save summary to csv or session_state
            st.session_state.latest_summary = summary
        else:
            st.info("Klik 'Start analyse' for at lade AI'en analysere videoen.")
        st.markdown("</div>", unsafe_allow_html=True)

    # If we already have last summary, render sidebar
    if st.session_state.get("latest_summary"):
        render_sidebar(st.session_state.latest_summary)

else:
    # no file uploaded -> show tips and examples
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        '<div class="padel-card" style="max-width:1100px; margin: 0 auto;">'
        '<h3 style="color:var(--accent); margin-bottom:4px;">Tips til bedste resultater</h3>'
        '<ul style="color:rgba(230,238,246,0.8); margin-top:6px;">'
        '<li>Hold kamera over netniveau for god oversigt</li>'
        '<li>Undgå kraftig sol bagved spillerne</li>'
        '<li>Upload korte klip (30–90s) for hurtigere analyse</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True,
    )
