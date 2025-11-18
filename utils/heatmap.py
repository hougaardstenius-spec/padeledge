# utils/heatmap.py
import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate_heatmap_xy(keypoint_sequences, save_path="data/heatmap.png"):
    """
    Creates a 2D heatmap of shot impact positions
    using the wrist keypoint (x,y).

    Args:
        keypoint_sequences: list of np.arrays (frames, keypoints)
        save_path: output file
    Returns:
        save_path
    """
    xs, ys = [], []

    for kp in keypoint_sequences:
        if kp is None or kp.size == 0:
            continue

        # Wrist index: MediaPipe Pose → right wrist = 16
        wrist_x = kp[:, 16 * 3]
        wrist_y = kp[:, 16 * 3 + 1]

        xs.extend(wrist_x)
        ys.extend(wrist_y)

    xs = np.array(xs)
    ys = np.array(ys)

    plt.figure(figsize=(5, 4))
    plt.hist2d(xs, ys, bins=35)
    plt.colorbar()
    plt.title("Shot Impact Heatmap")
    plt.savefig(save_path, dpi=140)
    plt.close()

    return save_path
