"""Training augmentation for 256x192 person crops.

Same transforms, defaults and random-number order as the MMPose pipeline used for the released model
(RandomFlip -> RandomHalfBody -> RandomBBoxTransform -> TopdownAffine -> PhotometricDistortion ->
Albumentation[Blur, MedianBlur, CoarseDropout]). Random numbers come from numpy's and Python's global generators,
seeded per update by the caller. Adapted from MMPose (Apache-2.0, OpenMMLab).
"""
import math

import albumentations
import cv2
import numpy as np
from scipy.stats import truncnorm

WIDTH, HEIGHT = 192, 256
# Three invisible probe points ride along with the keypoints; they reveal the final 2D warp (for the 3D labels).
PROBES = np.array([[96., 128.], [196., 128.], [96., 228.]], np.float32)

_ALBU = albumentations.Compose([
    albumentations.Blur(p=0.1),
    albumentations.MedianBlur(p=0.1),
    albumentations.CoarseDropout(max_holes=1, max_height=0.4, max_width=0.4, min_holes=1, min_height=0.2, min_width=0.2, p=0.5),
])


def _warp_matrix(center, scale, rotation_deg):
    """MMPose get_warp_matrix (fixed aspect ratio, no shift)."""
    angle = np.deg2rad(rotation_deg)
    sn, cs = np.sin(angle), np.cos(angle)
    src_dir = np.array([[cs, -sn], [sn, cs]]) @ np.array([scale[0] * -0.5, 0.])
    src = np.zeros((3, 2), np.float32)
    dst = np.zeros((3, 2), np.float32)
    src[0] = center
    src[1] = center + src_dir
    dst[0] = [WIDTH * 0.5, HEIGHT * 0.5]
    dst[1] = np.array([WIDTH * 0.5, HEIGHT * 0.5]) + np.array([WIDTH * -0.5, 0.])
    for points in (src, dst):
        direction = points[0] - points[1]
        points[2] = points[1] + np.r_[-direction[1], direction[0]]
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _convert(img, alpha=1., beta=0.):
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def augment(rgb, xy, valid, flip_index, lower_ids, xyz=None):
    """rgb (256,192,3) uint8 crop; xy (K,2) crop px; valid (K,) bool; xyz (K,3) metres or None.

    Returns (rgb, xy, valid, xyz) after augmentation; xyz is flipped/rotated like the image (depth unchanged).
    """
    count = len(xy)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    points = np.concatenate([xy, PROBES]).astype(np.float32)
    visible = np.concatenate([valid, np.zeros(3, bool)]).astype(np.float32)
    flip_index = list(flip_index) + [count, count + 1, count + 2]
    center = np.array([96., 128.], np.float32)
    scale = np.array([192., 256.], np.float32)

    # RandomFlip(prob=0.5, horizontal)
    flipped = np.random.choice(["horizontal", None], p=[0.5, 0.5]) is not None
    if flipped:
        img = np.flip(img, axis=1)
        center = np.array([WIDTH - center[0] - 1, center[1]], np.float32)
        points, visible = points[flip_index], visible[flip_index]
        points[:, 0] = WIDTH - 1 - points[:, 0]

    # RandomHalfBody(min_total=9, min_upper=2, min_lower=3, padding=1.5, prob=0.3, upper_prioritized=0.7)
    if visible.sum() >= 9 and not np.random.rand() > 0.3:
        upper = [i for i in range(11) if visible[i] > 0]
        lower = [i for i in lower_ids if visible[i] > 0]
        prefer_upper = np.random.rand() < 0.7
        ids = None
        if len(upper) < 2 and len(lower) < 3:
            ids = None
        elif len(lower) < 3:
            ids = upper
        elif len(upper) < 2:
            ids = lower
        else:
            ids = upper if prefer_upper else lower
        if ids is not None:
            chosen = points[ids]
            center = chosen.mean(axis=0)[:2]
            scale = np.array([*(chosen.max(axis=0) - chosen.min(axis=0))], dtype=center.dtype) * 1.5

    # RandomBBoxTransform(shift 0.16 @ 0.3, scale 0.5-1.5 @ 1.0, rotation 90 @ 0.6)
    v = truncnorm.rvs(-1., 1., size=(1, 4)).astype(np.float32)
    offset = np.where(np.random.rand(1, 1) < 0.3, v[:, :2] * 0.16, 0.)[0]
    factor = np.where(np.random.rand(1, 1) < 1.0, v[:, 2:3] * 0.5 + 1.0, 1.)[0]
    rotation = np.where(np.random.rand(1) < 0.6, v[:, 3] * 90., 0.)[0]
    center = center + offset * scale
    scale = scale * factor

    # TopdownAffine(192x256): fix aspect ratio, warp image and points
    w, h = np.hsplit(scale[None], [1])
    scale = np.where(w > h * 0.75, np.hstack([w, w / 0.75]), np.hstack([h * 0.75, h]))[0]
    matrix = _warp_matrix(center, scale, rotation)
    img = cv2.warpAffine(img, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
    points = cv2.transform(points[None], matrix)[0]

    # PhotometricDistortion (MMPose defaults)
    contrast_mode, brightness_flag, contrast_flag = np.random.randint(2), np.random.randint(2), np.random.randint(2)
    hsv_mode, swap_flag = np.random.randint(4), np.random.randint(2)
    brightness = np.random.uniform(-32, 32)
    contrast = np.random.uniform(0.5, 1.5)
    saturation = np.random.uniform(0.5, 1.5)
    hue = np.random.randint(-18, 18)
    order = np.random.permutation(3)
    if brightness_flag:
        img = _convert(img, beta=brightness)
    if contrast_mode == 1 and contrast_flag:
        img = _convert(img, alpha=contrast)
    if hsv_mode:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if hsv_mode in (1, 3):
            img[:, :, 1] = _convert(img[:, :, 1], alpha=saturation)
        if hsv_mode in (2, 3):
            img[:, :, 0] = img[:, :, 0].astype(int) + hue
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    if contrast_mode == 1 and contrast_flag:
        img = _convert(img, alpha=contrast)
    if swap_flag:
        img = img[..., order]

    img = _ALBU(image=img)["image"]

    if xyz is not None:
        warp = np.stack([(points[count + 1] - points[count]) / 100., (points[count + 2] - points[count]) / 100.], axis=1).astype(np.float64)
        rotation_2d = warp / math.sqrt(abs(np.linalg.det(warp)))
        xyz = xyz[flip_index[:count]] if flipped else xyz
        xyz = np.concatenate([xyz[:, :2] @ rotation_2d.T, xyz[:, 2:]], axis=1).astype(np.float32)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), points[:count].astype(np.float32), visible[:count], xyz
