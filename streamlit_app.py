# streamlit_app.py — PadelEdge (Premium sport-tech + Shot Timeline)
import base64
import os
import time
import json

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import altair as alt

from utils.video_processor import extract_keypoints_from_video
from utils.shot_detector import detect_shots
from utils.match_analyzer import analyze_match

# -------------------------
# Config / Constants
# -------------------------
PAGE_TITLE = "🎾 PadelEdge Pro – Match Analyzer"
ACCENT_COLOR = "#00ffcc"           # neon cyan/turquoise
ACCENT_COLOR_2 = "#00C853"         # neon green (for metrics)
GLASS_BG = "rgba(255,255,255,0.04)"  # subtle glass
LOGO_CANDIDATES = ["Designer-5.png", "padel_logo.png", "logo.png", "logo.svg"]

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="collapsed")

# -------------------------
# Helper functions
# -------------------------
def add_background(image_file: str):
    """Adds a full-page background image using base64. Silent fail if not found."""
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
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
        pass


def load_logo():
    """Try candidate filenames and return a data URI or None."""
    for candidate in LOGO_CANDIDATES:
        if os.path.exists(candidate):
            try:
                with open(candidate, "rb") as f:
                    b = f.read()
                return "data:image/png;base64," + base64.b64encode(b).decode()
            except Exception:
                continue
    return None


def show_sticky_progress(progress: int = 0, label: str = ""):
    """Renders a slim sticky progress bar & label at the top of the page."""
    st.markdown(
        f"""
        <div style="position:sticky; top:0; z-index:9999; backdrop-filter: blur(6px); padding:8px 12px;
                    background: linear-gradient(180deg, rgba(0,0,0,0.55), rgba(0,0,0,0.35)); 
                    border-bottom:1px solid rgba(255,255,255,0.02);">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem;">
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="font-weight:800; color:{ACCENT_COLOR}; letter-spacing:1px;">PADELEDGE</div>
                    <div style="color:rgba(230,238,246,0.8); font-size:0.95rem;">{label}</div>
                </div>
                <div style="min-width:240px; max-width:520px;">
                    <div style="height:10px; width:100%; background: rgba(255,255,255,0.06); border-radius:6px;">
                        <div style="height:100%; width:{int(progress)}%; background: linear-gradient(90deg, {ACCENT_COLOR}, {ACCENT_COLOR_2}); border-radius:6px;"></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Basic styling (glass + neon)
# -------------------------
st.markdown(
    f"""
    <style>
    :root {{
        --accent: {ACCENT_COLOR};
        --accent-2: {ACCENT_COLOR_2};
        --glass: {GLASS_BG};
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        -webkit-font-smoothing: antialiased;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        color: #e6eef6;
        background-color: #05060a;
    }}

    .padel-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 6px 30px rgba(2,6,23,0.7);
        border: 1px solid rgba(255,255,255,0.03);
        margin-bottom: 12px;
    }}

    .padel-hero-title {{
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--accent);
        margin: 0;
    }}

    .padel-hero-sub {{
        margin: 0;
        color: rgba(230,238,246,0.78);
        font-size: 0.95rem;
    }}

    .padel-upload {{
        padding: 18px;
        text-align: center;
        border: 1px dashed rgba(255,255,255,0.05);
        border-radius: 12px;
        background: rgba(255,255,255,0.01);
    }}

    .metric-box {{
        background: rgba(255,255,255,0.02);
        padding: 12px;
        border-radius: 12px;
        text-align: center;
    }}

    .metric-value {{
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--accent-2);
    }}

    .insight-box {{
        background: rgba(0,255,180,0.04);
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid var(--accent-2);
    }}

    /* responsive tweaks */
    @media (max-width: 800px) {{
        .padel-hero-title {{ font-size:1.25rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Try background + logo
# -------------------------
add_background("background.png")
logo_data = load_logo()

# -------------------------
# HEADER / HERO
# -------------------------
header_cols = st.columns([1, 4, 1])
with header_cols[1]:
    left_html = ""
    if logo_data:
        left_html += f'<img src="{logo_data}" style="height:72px; border-radius:8px; margin-right:12px;" />'
    left_html += f'<div style="display:inline-block; vertical-align:middle;"><div class="padel-hero-title">PADELEDGE — Data Driven Padel Performance</div><div class="padel-hero-sub">Upload en kort kampvideo og få AI-analyse af slag, score-estimat og taktiske indsigter.</div></div>'
    st.markdown(f'<div class="padel-card" style="display:flex; align-items:center; gap:12px;">{left_html}</div>', unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -------------------------
# Session state init
# -------------------------
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "status_text" not in st.session_state:
    st.session_state.status_text = ""
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "uploaded_video" not in st.session_state:
    st.session_state.uploaded_video = None

# show sticky area (initially empty)
show_sticky_progress(st.session_state.get("progress", 0), st.session_state.get("status_text", ""))

# -------------------------
# UPLOAD UI
# -------------------------
st.markdown('<div class="padel-card">', unsafe_allow_html=True)
st.markdown('<h2 style="margin:0; color:var(--accent);">Upload din kampvideo</h2>', unsafe_allow_html=True)
st.markdown('<p style="margin:6px 0 0 0; color:rgba(230,238,246,0.78);">Under 90 sekunder anbefales. Vi analyserer slag, positioner og genererer et dashboard.</p>', unsafe_allow_html=True)
st.markdown('<div class="padel-upload" style="margin-top:12px;">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload video (mp4, mov, avi)", type=["mp4", "mov", "avi"], key="uploader")

st.markdown("</div></div>", unsafe_allow_html=True)

# -------------------------
# Processing pipeline
# -------------------------
def process_video(temp_path: str):
    """Performs extraction -> shot detection -> analysis and returns summary dict."""
    try:
        # STEP 1
        st.session_state.progress = 5
        st.session_state.status_text = "Starter ekstraktion..."
        show_sticky_progress(st.session_state.progress, st.session_state.status_text)
        time.sleep(0.6)

        keypoints = extract_keypoints_from_video(temp_path)
        st.session_state.progress = 30
        st.session_state.status_text = "Ekstraktion færdig — kører stroke-detektion..."
        show_sticky_progress(st.session_state.progress, st.session_state.status_text)
        time.sleep(0.4)

        # STEP 2
        shots_raw = detect_shots(keypoints)
        # shots_raw can be: list of labels OR list of dicts with 'type' and 'time'.
        st.session_state.progress = 55
        st.session_state.status_text = "Stroke-klassificering færdig — analyserer..."
        show_sticky_progress(st.session_state.progress, st.session_state.status_text)
        time.sleep(0.4)

        # Normalize shots into list of dicts with 'type' and 'time'
        shots = []
        # Attempt to detect timestamps if provided by detect_shots
        if isinstance(shots_raw, list) and len(shots_raw) > 0 and isinstance(shots_raw[0], dict) and 'time' in shots_raw[0]:
            shots = shots_raw
        elif isinstance(shots_raw, (list, np.ndarray)):
            # fallback: create equally spaced timestamps (1s apart)
            for i, label in enumerate(shots_raw):
                shots.append({'type': str(label), 'time': float(i + 1)})
        else:
            # unknown format -> return empty
            shots = []

        # STEP 3: Analyze match
        # The analyze_match function should accept the normalized shots; if not, pass labels
        try:
            summary = analyze_match(shots)
        except Exception:
            # Fallback: call analyze_match with labels only
            labels = [s['type'] for s in shots]
            summary = analyze_match(labels)

        # Ensure shot timestamps are in the summary for timeline use
        if 'shots_timestamps' not in summary:
            summary['shots_timestamps'] = shots

        st.session_state.progress = 95
        st.session_state.status_text = "Færdiggør rapport..."
        show_sticky_progress(st.session_state.progress, st.session_state.status_text)
        time.sleep(0.4)

        st.session_state.progress = 100
        st.session_state.status_text = "Færdig"
        show_sticky_progress(st.session_state.progress, st.session_state.status_text)
        time.sleep(0.2)

        return summary

    except Exception as e:
        st.error(f"Fejl under behandling: {e}")
        st.session_state.status_text = "Fejl"
        show_sticky_progress(st.session_state.progress, st.session_state.status_text)
        return None


# -------------------------
# Timeline HTML builder
# -------------------------
def build_timeline_component(video_path: str, shots_timestamps: list):
    """Returns an HTML string containing a video player + clickable timeline.
    shots_timestamps: list of {'type':str, 'time':float}
    """
    # Build JSON for events
    events_json = json.dumps(shots_timestamps)

    # HTML with inline JS that sets video.currentTime when clicking a marker
    html = f"""
    <div style='display:flex; gap:18px; flex-direction:column;'>
      <video id='padel-video' width='100%' controls crossorigin playsinline>
        <source src='{video_path}' type='video/mp4'>
        Your browser does not support the video tag.
      </video>

      <div id='timeline' style='display:flex; gap:8px; flex-wrap:wrap; padding-top:8px; align-items:center;'>
      </div>
    </div>
    <script>
    const events = {events_json};
    const timeline = document.getElementById('timeline');
    const video = document.getElementById('padel-video');

    function fmt(s) {{
        const mm = Math.floor(s/60).toString().padStart(2,'0');
        const ss = Math.floor(s%60).toString().padStart(2,'0');
        return mm + ':' + ss;
    }}

    events.forEach((ev, idx) => {{
        const btn = document.createElement('button');
        btn.style.padding = '6px 10px';
        btn.style.borderRadius = '8px';
        btn.style.border = '1px solid rgba(255,255,255,0.06)';
        btn.style.background = 'linear-gradient(90deg, rgba(0,255,180,0.06), rgba(0,195,255,0.04))';
        btn.style.color = '#e9fbf8';
        btn.style.cursor = 'pointer';
        btn.style.fontSize = '13px';
        btn.innerText = `${{fmt(ev.time)}} — ${{ev.type}}`;
        btn.onclick = () => {{
            try {{ video.currentTime = ev.time; video.play(); }} catch(e) {{ console.error(e); }}
        }};
        timeline.appendChild(btn);
    }});
    </script>
    """
    return html


# -------------------------
# Dashboard rendering (after analysis)
# -------------------------

def render_dashboard(summary: dict):
    """Renders the premium dashboard for a given summary (stored in session_state)."""
    if summary is None:
        st.warning("Ingen analyse resultater tilgængelige.")
        return

    # top summary cards
    st.markdown("<div class='padel-card'>", unsafe_allow_html=True)
    st.subheader("📊 Match Summary - Overview")

    total_shots = summary.get("total_shots", sum([v.get("count", 0) for v in summary.get("shot_types", {}).values()]))

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-box"><div class="metric-value">{total_shots}</div><div class="metric-label">Total Shots</div></div>', unsafe_allow_html=True)
    fore_acc = summary.get("shot_types", {}).get("forehand", {}).get("accuracy", 0)
    back_acc = summary.get("shot_types", {}).get("backhand", {}).get("accuracy", 0)
    smash_acc = summary.get("shot_types", {}).get("smash", {}).get("accuracy", 0)
    c2.markdown(f'<div class="metric-box"><div class="metric-value">{fore_acc*100:.1f}%</div><div class="metric-label">Forehand Accuracy</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><div class="metric-value">{back_acc*100:.1f}%</div><div class="metric-label">Backhand Accuracy</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-box"><div class="metric-value">{smash_acc*100:.1f}%</div><div class="metric-label">Smash Efficiency</div></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # video & timeline
    st.markdown("<div class='padel-card'>", unsafe_allow_html=True)
    st.subheader("🎥 Video & Shot Timeline")

    video_path = st.session_state.get("uploaded_video") or "temp_video.mp4"
    # If running on Streamlit Cloud, the relative path usually works; else use full path
    # Build timeline HTML and embed as a component
    shots_ts = summary.get('shots_timestamps', [])
    # Ensure times are floats
    try:
        shots_ts = [{'type':str(s.get('type', s.get('label', 'shot'))), 'time': float(s.get('time', i+1))} if isinstance(s, dict) else {'type': str(s), 'time': float(i+1)} for i, s in enumerate(shots_ts)]
    except Exception:
        # fallback: build from shot_types counts
        shots_ts = []
        idx = 0
        for k, v in summary.get('shot_types', {}).items():
            cnt = int(v.get('count', 0))
            for j in range(cnt):
                shots_ts.append({'type': k, 'time': float(idx + 1)})
                idx += 1

    timeline_html = build_timeline_component(video_path, shots_ts)
    components.html(timeline_html, height=420, scrolling=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # shot distribution chart (Altair)
    st.markdown("<div class='padel-card'>", unsafe_allow_html=True)
    st.subheader("📈 Shot Distribution")
    shot_types = summary.get("shot_types", {})
    if shot_types:
        df = pd.DataFrame([
            {"shot": k.capitalize(), "count": v.get("count", 0), "accuracy": v.get("accuracy", 0)}
            for k, v in shot_types.items()
        ])
        bar = alt.Chart(df).mark_bar().encode(
            x=alt.X("shot:N", sort="-y"),
            y="count:Q",
            tooltip=["shot", "count", alt.Tooltip("accuracy:Q", format=".2f")]
        ).properties(height=300)
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("Ingen slagdata fundet i analysen.")
    st.markdown("</div>", unsafe_allow_html=True)

    # AI insights + downloads
    st.markdown("<div class='padel-card'>", unsafe_allow_html=True)
    st.subheader("🤖 AI Insights")
    ai_insights = summary.get("ai_insights", []) or []
    if ai_insights:
        for insight in ai_insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    else:
        st.info("Ingen AI-insights genereret for denne video.")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader("⬇️ Downloads")
    colA, colB = st.columns(2)
    colA.download_button("Download JSON", data=json.dumps(summary, ensure_ascii=False, indent=2).encode('utf-8'), file_name="analysis.json")
    # Flatten CSV (simple representation)
    try:
        csv_df = []
        for k, v in shot_types.items():
            row = {"shot": k, "count": v.get("count", 0), "accuracy": v.get("accuracy", 0)}
            csv_df.append(row)
        csv_df = pd.DataFrame(csv_df)
        colB.download_button("Download CSV", data=csv_df.to_csv(index=False).encode("utf-8"), file_name="analysis.csv")
    except Exception:
        colB.info("CSV-download ikke tilgængelig.")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Main flow: upload & trigger
# -------------------------

if 'uploader' in st.session_state:
    # keep
    pass

if uploaded_file is not None:
    # save temp video
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # preview + start button
    preview_cols = st.columns([2, 1])
    with preview_cols[0]:
        st.video(temp_path)
    with preview_cols[1]:
        st.markdown('<div class="padel-card">', unsafe_allow_html=True)
        st.markdown("<strong>Klar til analyse</strong>")
        st.markdown("<p style='margin-top:6px; color:rgba(230,238,246,0.8)'>Klik for at starte AI-analysen. Dette kan tage lidt tid afhængig af videoens længde.</p>", unsafe_allow_html=True)
        if st.button("Start analyse", key="start"):
            # Reset progress
            st.session_state.progress = 0
            st.session_state.status_text = "Forbereder..."
            show_sticky_progress(st.session_state.progress, st.session_state.status_text)

            # Process
            summary = process_video(temp_path)
            if summary:
                # Save to session so dashboard can read it
                st.session_state.analysis_results = summary
                st.session_state.uploaded_video = temp_path
                # Render dashboard immediately
                render_dashboard(summary)
        else:
            st.info("Tryk 'Start analyse' for at analysere videoen.")
        st.markdown("</div>", unsafe_allow_html=True)

    # If analysis already exists in session, show dashboard
    if st.session_state.get("analysis_results") and st.session_state.get("uploaded_video"):
        # Provide option to re-open dashboard (useful after refresh)
        if st.button("Vis seneste dashboard", key="show_dashboard"):
            render_dashboard(st.session_state.analysis_results)

else:
    # no upload -> tips / examples
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="padel-card" style="max-width:1100px; margin: 0 auto;">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:var(--accent); margin-bottom:6px;'>Tips til bedste resultater</h3>", unsafe_allow_html=True)
    st.markdown("<ul style='color:rgba(230,238,246,0.85);'><li>Optag fra net-højde for bedre oversigt.</li><li>Undgå stærkt modlys og hurtige panoreringer.</li><li>Kortere klip (30–90s) giver hurtigere og mere præcis analyse.</li></ul>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
