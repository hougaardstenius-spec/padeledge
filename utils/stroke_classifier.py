import cv2
import mediapipe as mp
import tempfile
import math

mp_pose = mp.solutions.pose

def classify_strokes(video_file):
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    # Load video
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        return {"error": "Could not open video."}

    stroke_counts = {
        "forehand": 0,
        "backhand": 0,
        "volley_forehand": 0,
        "volley_backhand": 0,
        "bandeja": 0,
        "vibora": 0,
        "rulo": 0,
        "smash": 0,
        "bajada": 0
    }

    frame_count = 0
    prev_wrist_y = None
    prev_elbow_y = None

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                wrist_y = wrist.y
                elbow_y = elbow.y
                shoulder_y = shoulder.y

                # Detect motion direction
                if prev_wrist_y and abs(wrist_y - prev_wrist_y) > 0.1:
                    motion = "up" if wrist_y < prev_wrist_y else "down"

                    # Heuristic classification
                    if motion == "up" and wrist_y < shoulder_y:
                        stroke_counts["smash"] += 1
                    elif motion == "down" and wrist_y < shoulder_y and elbow_y < shoulder_y:
                        stroke_counts["bandeja"] += 1
                    elif motion == "down" and wrist_y > shoulder_y:
                        stroke_counts["forehand"] += 1
                    elif motion == "up" and wrist_y > shoulder_y:
                        stroke_counts["backhand"] += 1
                    # more heuristics can be added later...

                prev_wrist_y = wrist_y
                prev_elbow_y = elbow_y

        cap.release()

    total = sum(stroke_counts.values())
    return {
        "total_strokes": total,
        "details": stroke_counts,
        "status": "✅ Stroke classification complete (beta heuristic mode)"
    }
