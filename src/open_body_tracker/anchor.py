"""Origin of the 3D pose that follows what the camera sees: hips, else shoulders, else face.

The model always predicts all 23 points relative to the hips, also when the hips are out of view (shoulders-up framing
or a desk in front). Those guessed hips move, and every other point moves with them. This moves the origin, smoothly,
to the shoulders and then the face as the hips and then the shoulders stop being visible.

- Visibility per point comes from the 2D score (about 6 for points in view, under 2 for points out of view): 0 at or
  below `low`, 1 at or above `high`. It falls with time constant `fall` and recovers with the slower `rise`, so a
  point on the edge of the view does not flip the origin back and forth.
- When the origin moves, the output keeps the position it had (bumpless hand-off) and only then drifts to the new
  origin at no more than `drift` m/s. Switching origins outright jumped the body by up to about 20 cm.
"""
import numpy as np

from .keypoints import HIPS

SHOULDERS = (5, 6)
FACE = (0, 1, 2, 3, 4)


class BodyAnchor:
    def __init__(self, low=3.0, high=5.0, fall=0.1, rise=0.3, drift=0.05):
        self.low, self.high, self.fall, self.rise, self.drift = low, high, fall, rise, drift
        self.reset()

    def reset(self):
        self.visible = self.time = self.weights = None
        self.offset = np.zeros(3, np.float32)

    def _visibility(self, score, seconds):
        target = np.clip((score - self.low) / (self.high - self.low), 0, 1)
        if self.visible is None:
            self.visible = target
        else:
            dt = max(seconds - self.time, 0.0)
            tau = np.where(target < self.visible, self.fall, self.rise)
            self.visible = self.visible + np.minimum(dt / tau, 1) * (target - self.visible)
        self.time = seconds
        hips, shoulders = self.visible[list(HIPS)].min(), self.visible[list(SHOULDERS)].min()
        return np.array([hips, (1 - hips) * shoulders, (1 - hips) * (1 - shoulders)], np.float32)

    @staticmethod
    def _origin(xyz, weights):
        parts = np.stack([xyz[list(HIPS)].mean(0), xyz[list(SHOULDERS)].mean(0), xyz[list(FACE)].mean(0)])
        return weights @ parts

    def __call__(self, xyz, score, seconds):
        """xyz (23, 3) metres, any origin; score (23,) 2D confidence -> (23, 3) relative to the blended origin."""
        dt = 0.0 if self.time is None else max(seconds - self.time, 0.0)
        previous = self.weights
        self.weights = self._visibility(score, seconds)
        origin = self._origin(xyz, self.weights)
        if previous is not None:
            self.offset = self.offset + origin - self._origin(xyz, previous)
            size = float(np.linalg.norm(self.offset))
            if size > 0:
                self.offset = self.offset * (max(size - self.drift * dt, 0.0) / size)
        return xyz - origin + self.offset
