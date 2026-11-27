"""Flip test: run the crop and its mirror image and average the two predictions.

Models up to v0.3.0 average the classifier outputs, then decode: MMPose's SimCC flip test extended to the X/Y/Z heads
(the mirror's x and X bins are reversed, both axes being symmetric around the crop centre / the hips, and left/right
points are swapped). v0.4.0 models (ONNX metadata obt.flip = "outputs") average the decoded outputs instead, because
their 3D goes through the rigid skeleton: the mirror's x becomes 191 - x, its X becomes -X (exact in the ray frame),
left/right swapped. Either way it roughly doubles the pose-model time; docs/MODELS.md lists what it gains.
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


def average_outputs(plain, mirrored, width=192):
    """(xy, score, xyz) of the crops and of their mirrors -> the average, mirror mapped back."""
    index = np.array(FLIP)
    xy, score, xyz = (v[:, index].copy() for v in mirrored)
    xy[..., 0] = width - 1 - xy[..., 0]
    xyz[..., 0] = -xyz[..., 0]
    return tuple((a + b) / 2 for a, b in zip(plain, (xy, score, xyz)))


def averages_outputs(session):
    return session.get_modelmeta().custom_metadata_map.get("obt.flip") == "outputs"


def run(session, crops):
    """crops (N, 3, 256, 192) float32 RGB 0-255 -> flip-tested (xy, score, xyz)."""
    n = len(crops)
    both = {"crop": np.concatenate([crops, mirror(crops)])}
    if averages_outputs(session):
        outputs = session.run(["xy", "score", "xyz"], both)
        return average_outputs([o[:n] for o in outputs], [o[n:] for o in outputs])
    outputs = session.run(LOGITS, both)
    return decode(average([o[:n] for o in outputs], [o[n:] for o in outputs]))
