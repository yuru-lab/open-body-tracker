"""Hand pass for hand models (v0.5.0): 21 points per hand, 2D in image pixels and 3D in metres.

Per frame, after the person model: the person model's rough hand points (its 42 hand rows) give each hand a square
crop of 2.0 x the hand's extent, resized to 140 x 140. The hand model (obt-hand-*.onnx, the same backbone with its own
head) sees right hands only, so a left hand crop is mirrored before and its output mirrored back. Its 3D is relative to
the wrist in the hand crop's ray frame; it is turned into camera axes as the body is (open_body_tracker.geometry) and
placed at the body's wrist, so the hands share the body's origin.

A hand is skipped when the body's wrist score is under `min_wrist_score` (the wrist is out of view or hidden). The
hand points' own scores are the SimCC peak logits: low for blurred or tiny hands, but they do not tell hidden fingers
from visible ones (the model always predicts every finger).
"""
import cv2
import numpy as np

from .geometry import ray_to_camera
from .smoothing import OneEuro

SIDES = ("right", "left")  # order of the person model's hand rows
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINTS = {"thumb": ("tip", "ip", "mcp", "cmc")}  # per finger, tip first; the others: tip, dip, pip, mcp
NAMES = tuple(f"{f}_{j}" for f in FINGERS for j in JOINTS.get(f, ("tip", "dip", "pip", "mcp"))) + ("wrist",)
WRIST = len(NAMES) - 1
EDGES = [(4 * f + k, 4 * f + k + 1) for f in range(5) for k in range(3)] + [(4 * f + 3, WRIST) for f in range(5)]
BODY_WRIST = {"right": 10, "left": 9}  # COCO rows of the person model
INPUT, CONTEXT = 140, 2.0
CENTRE = (INPUT / 2, INPUT / 2)


def hand_affine(points):
    """(21, 2) image points -> 2x3 affine from the image to the 140 x 140 hand crop (square, 2.0 x the extent)."""
    lo, hi = points.min(0), points.max(0)
    extent = max(float((hi - lo).max()), 8.0)
    scale = INPUT / (extent * CONTEXT)
    cx, cy = (lo + hi) / 2
    return np.array([[scale, 0, INPUT / 2 - scale * cx], [0, scale, INPUT / 2 - scale * cy]], np.float32)


class HandPass:
    def __init__(self, session, smoothing=True, min_wrist_score=3.0):
        """session: an onnxruntime session of an obt-hand-*.onnx file."""
        self.session, self.min_wrist_score = session, min_wrist_score
        self.filters = {side: dict(xy=OneEuro(0.65, 0.006), xyz=OneEuro(0.65, 0.5)) for side in SIDES} if smoothing else None

    def reset(self):
        if self.filters:
            for side in self.filters.values():
                for f in side.values():
                    f.reset()

    def __call__(self, rgb, hand_points, body_score, body_xyz, focal=None, now=0.0):
        """hand_points (2, 21, 2) image px from the person model, body_score (23,), body_xyz (23, 3) metres (the pose's
        final origin and axes) -> {"right": dict or None, "left": dict or None}; each dict has xy (21, 2) image px,
        score (21,), xyz (21, 3) metres in the body's frame (wrist at the body's wrist) and box [x, y, w, h]."""
        height, width = rgb.shape[:2]
        out, crops, jobs = {side: None for side in SIDES}, [], []
        for i, side in enumerate(SIDES):
            if body_score[BODY_WRIST[side]] < self.min_wrist_score:
                if self.filters:
                    for f in self.filters[side].values():
                        f.reset()
                continue
            matrix = hand_affine(hand_points[i])
            flags = cv2.INTER_AREA if matrix[0, 0] < 1 else cv2.INTER_LINEAR
            patch = cv2.warpAffine(rgb, matrix, (INPUT, INPUT), flags=flags, borderValue=(0, 0, 0))
            crops.append(patch[:, ::-1] if side == "left" else patch)
            jobs.append((side, matrix))
        if not crops:
            return out
        batch = np.stack(crops).transpose(0, 3, 1, 2).astype(np.float32)
        xy, score, xyz = self.session.run(["xy", "score", "xyz"], {"crop": batch})
        for (side, matrix), p, s, z in zip(jobs, xy, score, xyz):
            p, z = p.copy(), z.copy()
            if side == "left":  # mirrored back
                p[:, 0] = INPUT - 1 - p[:, 0]
                z[:, 0] = -z[:, 0]
            inverse = cv2.invertAffineTransform(matrix)
            p = p @ inverse[:, :2].T + inverse[:, 2]
            z = ray_to_camera(z, inverse, width, height, focal, centre=CENTRE)
            if self.filters:
                p, z = self.filters[side]["xy"](p, now), self.filters[side]["xyz"](z, now)
            x0, y0 = inverse[:, 2]
            box = [float(x0), float(y0), INPUT * float(inverse[0, 0]), INPUT * float(inverse[1, 1])]
            out[side] = dict(xy=p, score=s, xyz=z + body_xyz[BODY_WRIST[side]], box=box)
        return out
