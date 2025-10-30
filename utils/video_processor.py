import numpy as np

# ⚠️ MediaPipe temporarily disabled on Streamlit Cloud
# This dummy function just simulates video keypoint extraction
def extract_keypoints_from_video(video_path, frame_skip=3):
    # Simulate keypoints as random data
    keypoints = np.random.normal(size=(100, 3))
    return keypoints

def extract_keypoints_from_video(video_path, frame_skip=3):
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
                # Example: use shoulder, elbow, wrist, hip keypoints
                lm = results.pose_landmarks.landmark
                angles = calc_angles(lm)
                keypoints.append(angles)
        frame_idx += 1

    cap.release()
    return np.array(keypoints)

def calc_angles(lm):
    # Example: arm angle, wrist-elbow distance, shoulder-hip distance
    shoulder = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                         lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
    elbow = np.array([lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                      lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y])
    wrist = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                      lm[mp_pose.PoseLandmark.RIGHT_WRIST].y])
    hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x,
                    lm[mp_pose.PoseLandmark.RIGHT_HIP].y])

    arm_vector = elbow - shoulder
    wrist_vector = wrist - elbow
    angle = np.degrees(np.arctan2(arm_vector[1], arm_vector[0]))

    dist1 = np.linalg.norm(wrist - elbow)
    dist2 = np.linalg.norm(shoulder - hip)
    return [angle, dist1, dist2]
