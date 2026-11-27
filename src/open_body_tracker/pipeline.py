"""Frame-to-pose pipeline: detect (or track) one person, crop, run the pose model, map back to the image."""
import time

import numpy as np
import onnxruntime as ort

from . import flip as flip_test
from .anchor import BodyAnchor
from .geometry import ray_to_camera
from .keypoints import HIPS
from .preprocess import crop
from .smoothing import OneEuro


class BoxTracker:
    """Reuse a box made from the previous pose; ask the detector again every `refresh` seconds or when the pose is weak."""

    def __init__(self, refresh=1.0, min_points=8, min_score=0.3, margin=0.15):
        self.refresh, self.min_points, self.min_score, self.margin = refresh, min_points, min_score, margin
        self.box, self.detected_at = None, -1e9

    def needs_detection(self, now):
        return self.box is None or now - self.detected_at >= self.refresh

    def detected(self, box, now):
        self.box, self.detected_at = box, now

    def update(self, xy, score, width, height):
        good = xy[score >= self.min_score]
        if len(good) < self.min_points:
            self.box = None
            return
        (left, top), (right, bottom) = good.min(0), good.max(0)
        mx, my = (right - left) * self.margin, (bottom - top) * self.margin
        left, top = max(0., left - mx), max(0., top - my)
        right, bottom = min(float(width), right + mx), min(float(height), bottom + my)
        self.box = [left, top, right - left, bottom - top] if min(right - left, bottom - top) >= 8 else None


class PosePipeline:
    def __init__(self, model_path, detector, providers=None, track=True, smoothing=True, flip=False, anchor=False,
                 focal=None):
        """anchor=True: the 3D origin follows the visible body (hips, else shoulders, else face; see anchor.py)
        instead of always being the hips, which are guessed when out of view.
        focal: the camera's focal length in pixels, used by v0.4.0 models to turn their 3D from the crop's ray frame
        into camera axes (None: a 65-degree horizontal field of view is assumed; see geometry.py)."""
        self.session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
        self.ray_frame = self.session.get_modelmeta().custom_metadata_map.get("obt.frame") == "ray"
        self.focal = focal
        if flip and not set(flip_test.LOGITS) <= {o.name for o in self.session.get_outputs()}:
            raise ValueError(f"{model_path} has no logit outputs; the flip test needs an obt-body-v0.x.x model (not obt-h3c.onnx)")
        self.flip = flip
        self.detector = detector
        self.tracker = BoxTracker() if track else None
        self.filters = dict(xy=OneEuro(0.65, 0.006), xyz=OneEuro(0.65, 0.5)) if smoothing else None
        self.anchor = BodyAnchor() if anchor else None

    def __call__(self, rgb, now=None):
        """-> dict(box, xy (23,2) image px, score (23,), xyz (23,3) metres, hip-centred) or None.
        With anchor=True, xyz is relative to the blended origin and "origin" holds its (hips, shoulders, face) weights."""
        now = time.monotonic() if now is None else now
        height, width = rgb.shape[:2]
        if self.tracker is None or self.tracker.needs_detection(now):
            box = self.detector(rgb)
            if self.tracker is not None:
                self.tracker.detected(box, now)
            source = "detector"
        else:
            box, source = self.tracker.box, "tracker"
        if box is None:
            if self.filters:
                for f in self.filters.values():
                    f.reset()
            if self.anchor is not None:
                self.anchor.reset()
            return None
        patch, inverse = crop(rgb, box)
        crops = patch.transpose(2, 0, 1)[None].astype(np.float32)
        if self.flip:
            xy, score, xyz = flip_test.run(self.session, crops)
        else:
            xy, score, xyz = self.session.run(["xy", "score", "xyz"], {"crop": crops})
        xy = xy[0] @ inverse[:, :2].T + inverse[:, 2]
        score, xyz = score[0], xyz[0]
        if self.ray_frame:
            xyz = ray_to_camera(xyz, inverse, width, height, self.focal)
        xyz = xyz - xyz[list(HIPS)].mean(0)
        if self.tracker is not None:
            self.tracker.update(xy, score, width, height)
        if self.anchor is not None:
            xyz = self.anchor(xyz, score, now)
        if self.filters:
            xy, xyz = self.filters["xy"](xy, now), self.filters["xyz"](xyz, now)
        pose = dict(box=box, box_source=source, xy=xy, score=score, xyz=xyz)
        if self.anchor is not None:
            pose["origin"] = dict(zip(("hips", "shoulders", "face"), self.anchor.weights.tolist()))
        return pose
