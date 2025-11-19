# utils/heatmap.py
import numpy as np
import matplotlib.pyplot as plt

def generate_heatmap_xy(keypoint_sequences, out_path="data/heatmaps/latest.png"):
    """
    keypoint_sequences: np.array (frames, landmarks*3)
    We use right wrist index (MediaPipe: 16) -> x=idx*3, y=idx*3+1
    """
    if keypoint_sequences is None or len(keypoint_sequences) == 0:
        return None
    xs, ys = [], []
    for kp in keypoint_sequences:
        try:
            x = kp[16*3]
            y = kp[16*3 + 1]
            xs.append(x)
            ys.append(y)
        except Exception:
            continue
    if len(xs) == 0:
        return None
    plt.figure(figsize=(5,4))
    plt.hist2d(xs, ys, bins=40, cmap='inferno')
    plt.colorbar()
    plt.gca().invert_yaxis()  # optional depending on coord system
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path
