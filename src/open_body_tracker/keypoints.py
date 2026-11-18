"""23 keypoints: the 17 COCO body points followed by 6 foot points (COCO-WholeBody order)."""
NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel",
)
NUM_KEYPOINTS = len(NAMES)
# Index of the mirrored keypoint (used for horizontal-flip augmentation).
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 20, 21, 22, 17, 18, 19]
HIPS = (11, 12)
# Body 12 points used for the 3D metrics (shoulders, elbows, wrists, hips, knees, ankles).
BODY12 = list(range(5, 17))
EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (17, 18), (16, 20), (16, 22), (20, 21),
    (0, 1), (0, 2), (1, 3), (2, 4),
]
