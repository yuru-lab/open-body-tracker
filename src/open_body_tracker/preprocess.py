"""Person crop and normalization shared by the Python demo, evaluation and training."""
import cv2
import numpy as np

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # ImageNet, RGB 0-255
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
CROP_SIZE = (192, 256)  # width, height
PADDING = 1.25


def crop_affine(box, size=CROP_SIZE, padding=PADDING):
    """box [x, y, w, h] in pixels -> 2x3 affine matrix from the image to the crop (aspect kept, 25% margin)."""
    x, y, w, h = map(float, box)
    out_w, out_h = size
    if min(w, h) <= 0:
        raise ValueError("empty box")
    w = max(w, h * out_w / out_h) * padding
    centre_x, centre_y, scale = x + box[2] / 2, y + box[3] / 2, out_w / w
    return np.array([[scale, 0, out_w / 2 - centre_x * scale], [0, scale, out_h / 2 - centre_y * scale]], np.float32)


def crop(rgb, box, size=CROP_SIZE):
    """-> (uint8 crop HxWx3, inverse affine 2x3 mapping crop pixels back to the image)."""
    matrix = crop_affine(box, size)
    return cv2.warpAffine(rgb, matrix, size, flags=cv2.INTER_LINEAR), cv2.invertAffineTransform(matrix)


def normalize(crops):
    """uint8 (N,H,W,3) RGB -> float32 (N,3,H,W)."""
    return np.ascontiguousarray(((crops.astype(np.float32) - MEAN) / STD).transpose(0, 3, 1, 2))
