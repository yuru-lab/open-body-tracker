"""One Euro filter (Casiez et al., CHI 2012) on fixed-shape arrays."""
import numpy as np


class OneEuro:
    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.value = self.speed = self.time = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1 / (1 + 1 / (2 * np.pi * cutoff) / dt)

    def __call__(self, value, seconds):
        if self.value is None:
            self.value, self.speed, self.time = value.copy(), np.zeros_like(value), seconds
            return self.value
        dt = max(seconds - self.time, 1e-3)
        self.time = seconds
        a = self._alpha(self.d_cutoff, dt)
        self.speed = a * (value - self.value) / dt + (1 - a) * self.speed
        a = self._alpha(self.min_cutoff + self.beta * np.abs(self.speed), dt)
        self.value = a * value + (1 - a) * self.value
        return self.value
