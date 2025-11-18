import numpy as np

def analyze_match(shots):
    summary = {
        "Bandeja": 0,
        "Vibora": 0,
        "Smash": 0,
        "Volley": 0,
        "Other": 0,
    }

    for s in shots:
        shot_type = s["type"]
        if shot_type in summary:
            summary[shot_type] += 1
        else:
            summary["Other"] += 1

    # Placeholder scoring logic (customize later)
    score = f"{summary['Bandeja'] + summary['Smash']} – {summary['Vibora'] + summary['Volley']}"

    return summary, score
