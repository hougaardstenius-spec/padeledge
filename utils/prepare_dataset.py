import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            features.append([
                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                lm[mp_pose.PoseLandmark.RIGHT_WRIST].x
            ])
    cap.release()
    return np.mean(features, axis=0)  # average per video

def build_dataset(video_dir="data/raw_videos", out_file="data/features.npy"):
    X, y = [], []
    for label in os.listdir(video_dir):
        folder = os.path.join(video_dir, label)
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            feats = extract_features(path)
            X.append(feats)
            y.append(label)
    np.save(out_file, {"X": X, "y": y})
    print(f"✅ Dataset saved to {out_file} with {len(y)} samples.")
