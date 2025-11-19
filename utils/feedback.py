# utils/feedback.py
def generate_feedback(shot_type, kp_frame):
    """
    kp_frame: 1D array (landmarks*3) for the frame of interest
    """
    messages = []
    if kp_frame is None:
        return ["Ingen keypoint-data for dette slag."]
    try:
        # example: elbow vs shoulder (right)
        r_sh_y = kp_frame[12*3 + 1]
        r_el_y = kp_frame[14*3 + 1]
        # if elbow significantly lower (larger y), give feedback
        if r_el_y > r_sh_y + 0.08:
            messages.append("Din bandeja er for flad — løft albuen ca. 10 cm højere ved kontakt.")
    except Exception:
        pass

    if not messages:
        messages.append("Godt slag — arbejd videre med timing og konsistens.")
    return messages
