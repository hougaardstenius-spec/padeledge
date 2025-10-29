import cv2
import mediapipe as mp
import numpy as np
import tempfile
import joblib

mp_pose = mp.solutions.pose

def classify_strokes(video_file):
    # Load trained model
    try:
        model = joblib.load("utils/padel_model.pkl")
    except:
        return {"error": "No trained model found. Please train one first."}

    # Save video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    features = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
                shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

                arm_angle = np.degrees(np.arctan2(
                    elbow.y - shoulder.y, elbow.x - shoulder.x))
                wrist_elbow_dist = np.linalg.norm([
                    wrist.x - elbow.x, wrist.y - elbow.y])
                shoulder_hip_dist = np.linalg.norm([
                    shoulder.x - hip.x, shoulder.y - hip.y])

                features.append([arm_angle, wrist_elbow_dist, shoulder_hip_dist])

    cap.release()
    if not features:
        return {"error": "No pose landmarks detected."}

    avg_feat = np.array(features).mean(axis=0).reshape(1, -1)
    prediction = model.predict(avg_feat)[0]

    return {
        "predicted_stroke": prediction,
        "status": "✅ AI-based classification successful!"
    }
