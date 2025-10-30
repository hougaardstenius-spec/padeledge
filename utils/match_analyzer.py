def analyze_match(shots):
    winners = forced_errors = unforced_errors = 0
    total_points = 0

    for i, stroke in enumerate(shots):
        # Super simple logic — you can refine later:
        if "smash" in stroke or "gancho" in stroke:
            winners += 1
        elif "volley" in stroke or "bandeja" in stroke:
            forced_errors += 1
        else:
            unforced_errors += 1
        total_points += 1

    return {
        "winners": winners,
        "forced_errors": forced_errors,
        "unforced_errors": unforced_errors,
        "points_played": total_points,
        "score": f"{winners} - {forced_errors + unforced_errors}"
    }
