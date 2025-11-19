# utils/video_processor.py
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_keypoints_from_video(filepath, max_frames=120, frame_step=3, return_timestamps=False):
    """
    Extract pose keypoints using MediaPipe Pose.
    Returns: np.array shape (N_frames, N_landmarks*3) or (kp_seq, timestamps)
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return (None, None) if return_timestamps else None

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    keypoints = []
    timestamps = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    sampled = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            kp = []
            for lm in res.pose_landmarks.landmark:
                kp.extend([lm.x, lm.y, lm.z])
            keypoints.append(kp)
            timestamps.append(frame_idx / fps)
            sampled += 1
        frame_idx += 1
        if sampled >= max_frames:
            break

    cap.release()
    pose.close()

    if len(keypoints) == 0:
        return (None, None) if return_timestamps else None

    # pad with last frame if needed
    while len(keypoints) < max_frames:
        keypoints.append(keypoints[-1])
        timestamps.append(timestamps[-1] if timestamps else 0.0)

    arr = np.array(keypoints)
    if return_timestamps:
        return arr, np.array(timestamps[:arr.shape[0]])
    return arr
