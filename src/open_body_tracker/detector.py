"""Person detector: official Megvii YOLOX ONNX (nano or tiny, COCO, Apache-2.0). Returns the best person box."""
import cv2
import numpy as np
import onnxruntime as ort


class PersonDetector:
    def __init__(self, path, threshold=0.3, providers=None, threads=2):
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(str(path), options, providers=providers or ["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.size = self.input.shape[2]
        self.threshold = threshold
        grids, strides = [], []
        for stride in (8, 16, 32):
            cells = self.size // stride
            xv, yv = np.meshgrid(np.arange(cells), np.arange(cells))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((cells * cells, 1), stride))
        self.grid, self.stride = np.concatenate(grids), np.concatenate(strides)

    def __call__(self, rgb):
        """rgb (H, W, 3) uint8 -> [x, y, w, h] of the highest-scoring person, or None."""
        height, width = rgb.shape[:2]
        ratio = min(self.size / width, self.size / height)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)  # official YOLOX preprocessing: BGR, top-left letterbox
        resized = cv2.resize(rgb[:, :, ::-1], (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_LINEAR)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        raw = self.session.run(None, {self.input.name: canvas.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        scores = raw[:, 4] * raw[:, 5]  # objectness x person class
        best = int(scores.argmax())
        if scores[best] <= self.threshold:
            return None
        cx, cy = (raw[best, :2] + self.grid[best]) * self.stride[best, 0] / ratio
        w, h = np.exp(raw[best, 2:4]) * self.stride[best, 0] / ratio
        left, top = max(0., cx - w / 2), max(0., cy - h / 2)
        right, bottom = min(float(width), cx + w / 2), min(float(height), cy + h / 2)
        if right <= left or bottom <= top:
            return None
        return [float(left), float(top), float(right - left), float(bottom - top)]
