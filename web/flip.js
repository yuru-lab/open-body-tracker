// Flip test, the same computation as src/open_body_tracker/flip.py: average the classifier outputs of a crop and of its
// mirror image (mirror: x and X bins reversed, left/right points swapped), then decode.
const FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 20, 21, 22, 17, 18, 19];
const SIMCC_SPLIT = 2, METRIC_BINS = 300, METRIC_RANGE = 1.5;

// (3, H, W) planar crop -> (2, 3, H, W): the crop and its horizontal mirror.
export function mirrorBatch(input, plane, width, height) {
  const both = new Float32Array(2 * input.length);
  both.set(input);
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < height; y++) {
      const row = c * plane + y * width;
      for (let x = 0; x < width; x++) both[3 * plane + row + x] = input[row + width - 1 - x];
    }
  }
  return both;
}

function averaged(tensor, reverse) {
  const data = tensor.data, bins = tensor.dims[2], half = 23 * bins, result = new Float32Array(half);
  for (let k = 0; k < 23; k++) {
    for (let b = 0; b < bins; b++) {
      result[k * bins + b] = (data[k * bins + b] + data[half + FLIP[k] * bins + (reverse ? bins - 1 - b : b)]) / 2;
    }
  }
  return [result, bins];
}

function peak(values, offset, bins) {
  let best = 0;
  for (let b = 1; b < bins; b++) if (values[offset + b] > values[offset + best]) best = b;
  return best;
}

// outputs of a batch-2 run (logits_x/y/X/Y/Z) -> { xy (46), score (23), xyz (69) } in crop pixels / metres.
export function flipDecode(out) {
  const [x, xBins] = averaged(out.logits_x, true), [y, yBins] = averaged(out.logits_y, false);
  const metric = [averaged(out.logits_X, true)[0], averaged(out.logits_Y, false)[0], averaged(out.logits_Z, false)[0]];
  const xy = new Float32Array(46), score = new Float32Array(23), xyz = new Float32Array(69);
  for (let k = 0; k < 23; k++) {
    const ix = peak(x, k * xBins, xBins), iy = peak(y, k * yBins, yBins);
    xy[2 * k] = ix / SIMCC_SPLIT; xy[2 * k + 1] = iy / SIMCC_SPLIT;
    score[k] = Math.min(x[k * xBins + ix], y[k * yBins + iy]);
    metric.forEach((values, axis) => {  // softmax expectation over the bin centres
      const offset = k * METRIC_BINS, top = values[offset + peak(values, offset, METRIC_BINS)];
      let sum = 0, mean = 0;
      for (let b = 0; b < METRIC_BINS; b++) {
        const e = Math.exp(values[offset + b] - top);
        sum += e; mean += e * (-METRIC_RANGE + 2 * METRIC_RANGE * b / (METRIC_BINS - 1));
      }
      xyz[3 * k + axis] = mean / sum;
    });
  }
  return { xy, score, xyz };
}
