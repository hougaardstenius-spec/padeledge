# utils/thumbnails.py
import cv2
import os

def extract_thumbnail(video_path, timestamp, save_folder="data/thumbnails"):
    os.makedirs(save_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_index = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    thumb_path = os.path.join(save_folder, f"{timestamp:.2f}.jpg")
    cv2.imwrite(thumb_path, frame)

    cap.release()
    return thumb_path
