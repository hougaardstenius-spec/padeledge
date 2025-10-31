import numpy as np

# ----------------------------------------------------------
# Try importing MediaPipe + OpenCV; if not available (e.g., Streamlit Cloud),
# automatically switch to mock mode with fake keypoints.
# ----------------------------------------------------------
try:
    import cv2
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    USE_MEDIAPIPE = True
    print("✅ MediaPipe detected — running in REAL video mode.")
except Exception as e:
    USE_MEDIAPIPE = False
    print("⚠️ MediaPipe not available — running in MOCK mode.")


def extract_keypoints_from_video(video_path, frame_skip=3):
    """
    Extracts pose keypoints using MediaPipe if available,
    otherwise generates mock keypoints (safe for Streamlit Cloud).
    """
    if not USE_MEDIAPIPE:
        print("⚠️ Using simulated keypoints — Streamlit Cloud mock mode.")
        # Generate fake data (100 samples, 3 features)
        return np.random.normal(loc=0.5, scale=0.1, size=(100, 3))

    # ✅ Local Mode: Use real video analysis
    try:
        cap = cv2.VideoCapture(video_path)
        pose = mp_pose.Pose()
        keypoints = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    angles = calc_angles(lm)
                    keypoints.append(angles)
            frame_idx += 1

        cap.release()
        return np.array(keypoints)

    except Exception as e:
        print(f"⚠️ OpenCV video processing failed ({e}). Switching to mock mode.")
        return np.random.normal(loc=0.5, scale=0.1, size=(100, 3))


def calc_angles(lm):
    """Helper function to compute pose-based features."""
    mp_pose = getattr(__import__('mediapipe').solutions, 'pose')

    shoulder = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                         lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
    elbow = np.array([lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                      lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y])
    wrist = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                      lm[mp_pose.PoseLandmark.RIGHT_WRIST].y])
    hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x,
                    lm[mp_pose.PoseLandmark.RIGHT_HIP].y])

    arm_vector = elbow - shoulder
    angle = np.degrees(np.arctan2(arm_vector[1], arm_vector[0]))
    dist1 = np.linalg.norm(wrist - elbow)
    dist2 = np.linalg.norm(shoulder - hip)
    return [angle, dist1, dist2]
