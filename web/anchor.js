// Origin of the 3D pose that follows what the camera sees: hips, else shoulders, else face.
// Same as src/open_body_tracker/anchor.py (see there for the reasoning and the measurements).
// - Visibility per point from the 2D score: 0 at or below `low`, 1 at or above `high`; falls with time constant
//   `fall` and recovers with the slower `rise`, so a point on the edge of the view does not flip the origin.
// - When the origin moves, the output keeps its position (bumpless hand-off) and drifts to the new origin at no
//   more than `drift` m/s.
const HIPS = [11, 12], SHOULDERS = [5, 6], FACE = [0, 1, 2, 3, 4];

const mean = (xyz, ids) => [0, 1, 2].map((i) => ids.reduce((sum, k) => sum + xyz[k][i], 0) / ids.length);

export class BodyAnchor {
  constructor({ low = 3, high = 5, fall = 0.1, rise = 0.3, drift = 0.05 } = {}) {
    Object.assign(this, { low, high, fall, rise, drift });
    this.reset();
  }

  reset() { this.visible = null; this.time = null; this.weights = null; this.offset = [0, 0, 0]; }

  #visibility(score, seconds) {
    const target = score.map((s) => Math.min(1, Math.max(0, (s - this.low) / (this.high - this.low))));
    if (this.visible === null) this.visible = target;
    else {
      const dt = Math.max(seconds - this.time, 0);
      this.visible = this.visible.map((v, k) => v + Math.min(dt / (target[k] < v ? this.fall : this.rise), 1) * (target[k] - v));
    }
    this.time = seconds;
    const hips = Math.min(...HIPS.map((k) => this.visible[k])), shoulders = Math.min(...SHOULDERS.map((k) => this.visible[k]));
    return [hips, (1 - hips) * shoulders, (1 - hips) * (1 - shoulders)];
  }

  static origin(xyz, weights) {
    const parts = [mean(xyz, HIPS), mean(xyz, SHOULDERS), mean(xyz, FACE)];
    return [0, 1, 2].map((i) => weights[0] * parts[0][i] + weights[1] * parts[1][i] + weights[2] * parts[2][i]);
  }

  // xyz: 23 [x, y, z] in metres (any origin), score: 23 2D confidences, seconds: frame time -> 23 [x, y, z].
  apply(xyz, score, seconds) {
    const dt = this.time === null ? 0 : Math.max(seconds - this.time, 0);
    const previous = this.weights;
    this.weights = this.#visibility(score, seconds);
    const origin = BodyAnchor.origin(xyz, this.weights);
    if (previous) {
      const before = BodyAnchor.origin(xyz, previous);
      this.offset = this.offset.map((v, i) => v + origin[i] - before[i]);
      const size = Math.hypot(...this.offset);
      if (size > 0) this.offset = this.offset.map((v) => v * Math.max(size - this.drift * dt, 0) / size);
    }
    return xyz.map((p) => p.map((v, i) => v - origin[i] + this.offset[i]));
  }
}
