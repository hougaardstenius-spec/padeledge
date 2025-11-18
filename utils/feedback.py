# utils/feedback.py
import numpy as np

def elbow_height_feedback(kp_frame):
    """
    Simple biomechanical heuristic:
    If elbow is too low compared to shoulder, advise raising.
    """
    # Mediapipe: shoulder=12 or 11, elbow=14 or 13
    r_shoulder_y = kp_frame[12 * 3 + 1]
    r_elbow_y = kp_frame[14 * 3 + 1]

    if r_elbow_y > r_shoulder_y + 0.10:
        return "Din albue hænger for lavt – løft den 10 cm."
    return None


def generate_feedback(shot_type, keypoints):
    """
    Generate human-like actionable coaching feedback.
    """

    feedback_msgs = []

    if shot_type == "bandeja":
        elbow_msg = elbow_height_feedback(keypoints)
        if elbow_msg:
            feedback_msgs.append(elbow_msg)

    if shot_type == "vibora":
        feedback_msgs.append(
            "Prøv at give mere sidespin ved at rotere håndleddet lidt mere."
        )

    if shot_type == "smash":
        feedback_msgs.append(
            "Du kunne få mere power med mere hofterotation."
        )

    if not feedback_msgs:
        feedback_msgs.append("Godt arbejde! Fortsæt sådan.")

    return feedback_msgs
