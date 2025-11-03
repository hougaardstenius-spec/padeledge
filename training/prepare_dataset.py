import os
import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_features(video_path):
    """
    Extracts average pose features from a video using MediaPipe.
    Returns a vector of distances and angles that represent the player's stroke.
    """
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    frames_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            shoulder = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                 lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
            elbow = np.array([lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                              lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y])
            wrist = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                              lm[mp_pose.PoseLandmark.RIGHT_WRIST].y])
            hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x,
                            lm[mp_pose.PoseLandmark.RIGHT_HIP].y])

            # Calculate angles and distances
            arm_vec = elbow - shoulder
            angle = np.degrees(np.arctan2(arm_vec[1], arm_vec[0]))
            dist1 = np.linalg.norm(wrist - elbow)
            dist2 = np.linalg.norm(shoulder - hip)
            frames_features.append([angle, dist1, dist2])

    cap.release()

    if len(frames_features) == 0:
        print(f"⚠️ No features found in {video_path}")
        return np.zeros(3)

    return np.mean(frames_features, axis=0)


def build_dataset(video_dir="data/raw_videos", out_file="data/features.npy"):
    """
    Loops through all videos by stroke type and extracts features.
    Saves X (features) and y (labels) as a NumPy dictionary.
    """
    X, y = [], []
    for label in os.listdir(video_dir):
        label_folder = os.path.join(video_dir, label)
        if not os.path.isdir(label_folder):
            continue

        for fname in os.listdir(label_folder):
            if not fname.lower().endswith((".mp4", ".mov", ".avi")):
                continue
            path = os.path.join(label_folder, fname)
            print(f"🎾 Processing {path} ...")
            features = extract_features(path)
            X.append(features)
            y.append(label)

    np.save(out_file, {"X": X, "y": y})
    print(f"✅ Saved {len(y)} samples to {out_file}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    build_dataset()
