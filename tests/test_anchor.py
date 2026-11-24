import numpy as np
import pytest

from open_body_tracker.anchor import FACE, SHOULDERS, BodyAnchor
from open_body_tracker.keypoints import HIPS

FPS = 25.0


def standing_pose():
    """Hips at the origin, shoulders 0.5 m and face 0.7 m above them (y points down)."""
    xyz = np.zeros((23, 3), np.float32)
    xyz[list(SHOULDERS), 1] = -0.5
    xyz[list(FACE), 1] = -0.7
    xyz[list(SHOULDERS), 0] = (-0.2, 0.2)
    xyz[list(HIPS), 0] = (-0.1, 0.1)
    return xyz


def scores(hips=6.0, shoulders=6.0):
    score = np.full(23, 6.0, np.float32)
    score[list(HIPS)], score[list(SHOULDERS)] = hips, shoulders
    return score


def test_visible_hips_keep_the_hip_origin():
    anchor, xyz = BodyAnchor(), standing_pose()
    for t in range(10):
        out = anchor(xyz, scores(), t / FPS)
        np.testing.assert_allclose(out, xyz, atol=1e-6)
    assert anchor.weights.tolist() == [1, 0, 0]


def test_hips_leaving_view_moves_origin_without_a_jump():
    anchor, xyz = BodyAnchor(), standing_pose()
    outputs = [anchor(xyz, scores(), t / FPS) for t in range(5)]
    outputs += [anchor(xyz, scores(hips=1.5), t / FPS) for t in range(5, 5 + 4 * int(FPS))]
    steps = np.abs(np.diff(np.stack(outputs), axis=0)).max(axis=(1, 2))
    assert steps.max() <= 0.05 / FPS + 1e-6          # only the slow drift, never the 0.5 m hip-to-shoulder jump
    assert anchor.weights == pytest.approx([0, 1, 0], abs=1e-6)
    shoulder_centred = xyz - xyz[list(SHOULDERS)].mean(0)
    np.testing.assert_allclose(outputs[-1], shoulder_centred, atol=0.35)  # 4 s at 0.05 m/s: 0.2 m of 0.5 m drifted
    assert np.linalg.norm(outputs[-1][0] - outputs[4][0]) == pytest.approx(0.05 * (4 * FPS - 1) / FPS, abs=0.01)


def test_shoulders_out_of_view_fall_back_to_the_face():
    anchor = BodyAnchor(drift=1e9)                    # no hand-off memory: the origin itself is visible
    xyz = standing_pose()
    for t in range(int(FPS)):
        out = anchor(xyz, scores(hips=0.0, shoulders=0.0), t / FPS)
    np.testing.assert_allclose(out[list(FACE)].mean(0), 0, atol=1e-6)


def test_edge_flicker_is_damped():
    anchor, xyz = BodyAnchor(), standing_pose()
    anchor(xyz, scores(), 0.0)
    weights = []
    for t in range(1, 50):
        anchor(xyz, scores(hips=6.0 if t % 2 else 1.0), t / FPS)
        weights.append(anchor.weights[0])
    settled = np.array(weights[int(FPS):])
    assert np.abs(np.diff(settled)).max() < 0.3       # a raw 0/1 switch would swing by 1 every frame
    assert settled.max() < 0.5                        # flickering hips count as mostly not visible


def test_reset_forgets_the_offset():
    anchor, xyz = BodyAnchor(), standing_pose()
    anchor(xyz, scores(), 0.0)
    anchor(xyz, scores(hips=0.0), 1.0)
    assert np.linalg.norm(anchor.offset) > 0
    anchor.reset()
    out = anchor(xyz, scores(hips=0.0), 2.0)
    np.testing.assert_allclose(out, xyz - xyz[list(SHOULDERS)].mean(0), atol=1e-6)
