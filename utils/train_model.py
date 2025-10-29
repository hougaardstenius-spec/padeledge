import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

mp_pose = mp.solutions.pose

def extract_pose_features(video_path):
    """Extracts key pose angles for AI training."""
    cap = cv2.VideoCapture(video_path)
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

                # Compute simple angles and distances
                arm_angle = np.degrees(np.arctan2(
                    elbow.y - shoulder.y, elbow.x - shoulder.x))
                wrist_elbow_dist = np.linalg.norm([
                    wrist.x - elbow.x, wrist.y - elbow.y])
                shoulder_hip_dist = np.linalg.norm([
                    shoulder.x - hip.x, shoulder.y - hip.y])

                features.append([arm_angle, wrist_elbow_dist, shoulder_hip_dist])
        cap.release()
    return np.array(features).mean(axis=0) if len(features) > 0 else [0, 0, 0]

def train_model(data_dir="data/samples"):
    """Train AI model on labeled padel videos."""
    X, y = [], []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for file in tqdm(os.listdir(label_dir), desc=f"Processing {label}"):
            if file.endswith((".mp4", ".mov", ".avi")):
                video_path = os.path.join(label_dir, file)
                feat = extract_pose_features(video_path)
                X.append(feat)
                y.append(label)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    os.makedirs("utils", exist_ok=True)
    joblib.dump(model, "utils/padel_model.pkl")
    print("✅ Model trained and saved to utils/padel_model.pkl")
