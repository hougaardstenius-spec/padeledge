# utils/timeline.py
SHOT_COLORS = {
    "bandeja": "#00ffb4",
    "vibora": "#00c3ff",
    "smash": "#ff6b6b",
    "volley": "#ffd166",
    "forehand": "#a78bfa",
    "backhand": "#7dd3fc",
    "other": "#9ca3af"
}

def build_timeline(preds, timestamps):
    events = []
    for p, t in zip(preds, timestamps):
        events.append({
            "shot": p,
            "time": float(t),
            "color": SHOT_COLORS.get(p.lower(), SHOT_COLORS["other"])
        })
    return events
