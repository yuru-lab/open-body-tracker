"""Flip test: run the crop and its mirror image, average the classifier outputs, then decode.

This is MMPose's SimCC flip test extended to the X/Y/Z heads: the mirror's x and X bins are reversed (both axes are
symmetric around the crop centre / the hips) and left/right points are swapped. It roughly doubles the pose-model
time; on the benchmarks in docs/MODELS.md it lowers the 3D error by about 1 mm and the frame-to-frame jitter by about 3.5 mm.
"""
import numpy as np

from .keypoints import FLIP

LOGITS = ["logits_x", "logits_y", "logits_X", "logits_Y", "logits_Z"]
SIMCC_SPLIT = 2.0
CENTRES = np.linspace(-1.5, 1.5, 300, dtype=np.float32)  # metric bins (1 cm, ±1.5 m)


def mirror(crops):
    """(N, 3, H, W) -> the horizontally mirrored crops."""
    return np.ascontiguousarray(crops[..., ::-1])


def average(plain, mirrored):
    """Average the logits of the crops (x, y, X, Y, Z) with those of their mirrors, mapped back."""
    index = np.array(FLIP)
    back = [mirrored[0][:, index, ::-1], mirrored[1][:, index], mirrored[2][:, index, ::-1],
            mirrored[3][:, index], mirrored[4][:, index]]
    return [(a + b) / 2 for a, b in zip(plain, back)]


def decode(logits):
    """Logits -> crop-pixel xy (N,23,2), confidence (N,23), metres xyz (N,23,3); same as open_body_tracker.model.decode."""
    x, y, X, Y, Z = logits
    xy = np.stack([x.argmax(-1), y.argmax(-1)], -1).astype(np.float32) / SIMCC_SPLIT
    score = np.minimum(x.max(-1), y.max(-1))

    def expectation(v):
        e = np.exp(v - v.max(-1, keepdims=True))
        return (e / e.sum(-1, keepdims=True) * CENTRES).sum(-1)

    return xy, score, np.stack([expectation(v) for v in (X, Y, Z)], -1).astype(np.float32)


def run(session, crops):
    """crops (N, 3, 256, 192) float32 RGB 0-255 -> flip-tested (xy, score, xyz)."""
    n = len(crops)
    outputs = session.run(LOGITS, {"crop": np.concatenate([crops, mirror(crops)])})
    return decode(average([o[:n] for o in outputs], [o[n:] for o in outputs]))
