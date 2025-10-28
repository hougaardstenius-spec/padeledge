import random

def classify_strokes(video_file):
    """Mock classifier — returns fake stroke data for testing."""
    strokes = [
        "Forehand Groundstroke", "Backhand Groundstroke",
        "Forehand Volley", "Backhand Volley",
        "Bandeja Overhead", "Vibora Overhead",
        "Rulo Overhead", "Smash Overhead",
        "Bajada (After Back Glass)"
    ]
    return {stroke: random.randint(0, 10) for stroke in strokes}
