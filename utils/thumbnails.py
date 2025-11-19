# utils/thumbnails.py
import os, cv2

def extract_thumbnail(video_path, timestamp_sec, save_folder="data/thumbnails"):
    os.makedirs(save_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_no = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    out_path = os.path.join(save_folder, f"thumb_{int(timestamp_sec*100)}.jpg")
    cv2.imwrite(out_path, frame)
    cap.release()
    return out_path
