import cv2
import numpy as np

MODEL_FRAMES = 30


def extract_keypoints_from_video(video_path: str):
    """
    Mediapipe-free version.
    Extracts extremely lightweight motion+pose proxy features.

    This function:
    - Reads video frames
    - Converts to grayscale
    - Computes frame differences
    - Extracts simple movement statistics per frame
    - Returns a (num_frames, feature_dim) numpy array
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video:", video_path)
        return None

    prev = None
    feature_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed + consistency
        frame = cv2.resize(frame, (256, 144))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype("float32")

        if prev is None:
            prev = gray
            continue

        diff = cv2.absdiff(gray, prev)

        # Feature extraction (simple motion descriptors)
        mean_motion = np.mean(diff)
        max_motion = np.max(diff)
        std_motion = np.std(diff)

        # Histogram-based motion
        hist = cv2.calcHist([diff.astype("uint8")], [0], None, [16], [0, 256])
        hist = hist.flatten()

        feature_vector = np.concatenate([[mean_motion, max_motion, std_motion], hist])
        feature_list.append(feature_vector)

        prev = gray

    cap.release()

    if len(feature_list) == 0:
        return None

    return np.array(feature_list)


def summarize_feature_sequence(feature_seq: np.ndarray, target_frames: int = MODEL_FRAMES):
    """
    Converts a variable-length frame feature sequence to a fixed-size vector.
    """
    if feature_seq is None or len(feature_seq) == 0:
        return None

    arr = np.asarray(feature_seq, dtype=np.float32)
    if arr.ndim != 2:
        return None

    n_frames, feat_dim = arr.shape

    if n_frames == target_frames:
        sampled = arr
    else:
        # Linear interpolation indices keep dimensionality fixed across clip lengths.
        src_idx = np.arange(n_frames, dtype=np.float32)
        dst_idx = np.linspace(0, n_frames - 1, num=target_frames, dtype=np.float32)
        sampled = np.empty((target_frames, feat_dim), dtype=np.float32)
        for d in range(feat_dim):
            sampled[:, d] = np.interp(dst_idx, src_idx, arr[:, d])

    return sampled.flatten()


def extract_clip_features(video_path: str, target_frames: int = MODEL_FRAMES):
    """
    Convenience helper for training/inference from a video file path.
    """
    seq = extract_keypoints_from_video(video_path)
    return summarize_feature_sequence(seq, target_frames=target_frames)
