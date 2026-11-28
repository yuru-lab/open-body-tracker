"""Ray frame -> camera axes for v0.4.0 models.

From v0.4.0 the 3D output is in the crop's ray frame: the camera turned (by the smallest rotation) to look straight
through the crop centre. The crop alone shows the person as that turned camera sees them, so the network does not need
to know where in the image the crop came from. To get camera axes (x right, y down, z forward), turn it back; this needs
the camera's focal length in pixels. When it is unknown, DEFAULT_FOV (horizontal, degrees) is assumed: a person near
the image centre barely changes, one at the image edge turns by up to half the field of view.
"""
import math

import numpy as np

DEFAULT_FOV = 65.0
CROP_CENTRE = (96.0, 128.0)  # crop pixels (x, y)


def default_focal(width, fov=DEFAULT_FOV):
    """Focal length in pixels of a camera with the given horizontal field of view."""
    return width / 2 / math.tan(math.radians(fov) / 2)


def rotation_to_z(ray):
    """(3,) -> 3x3 R with R @ unit(ray) = +z, the smallest such rotation."""
    r = np.asarray(ray, np.float64)
    r = r / np.linalg.norm(r)
    v = np.cross(r, [0.0, 0.0, 1.0])
    s, c = np.linalg.norm(v), r[2]
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def ray_to_camera(xyz, inverse, width, height, focal=None, principal=None, centre=CROP_CENTRE):
    """(..., 3) ray-frame points of one crop -> camera axes. inverse: the crop -> image 2x3 affine
    (open_body_tracker.preprocess.crop); focal in pixels (None: DEFAULT_FOV); principal point (None: image centre);
    centre: the crop centre in crop pixels (person crop by default; hand crops: open_body_tracker.hands.CENTRE)."""
    focal = focal or default_focal(width)
    cx, cy = principal or (width / 2, height / 2)
    u, v = np.asarray(centre) @ np.asarray(inverse)[:, :2].T + np.asarray(inverse)[:, 2]
    rotation = rotation_to_z([(u - cx) / focal, (v - cy) / focal, 1.0])
    return (np.asarray(xyz) @ rotation).astype(np.float32)  # x_cam = R^T x_ray, row vectors
