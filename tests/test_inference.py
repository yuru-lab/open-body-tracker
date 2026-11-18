from pathlib import Path

import numpy as np
import pytest

from open_body_tracker.keypoints import EDGES, FLIP, NAMES, NUM_KEYPOINTS
from open_body_tracker.pipeline import BoxTracker
from open_body_tracker.preprocess import crop, crop_affine

MODELS = Path(__file__).resolve().parents[1] / "models"


def test_keypoint_tables():
    assert NUM_KEYPOINTS == 23 and len(FLIP) == 23 and sorted(FLIP) == list(range(23))
    assert all(NAMES[FLIP[i]].replace("left", "X").replace("right", "X") == NAMES[i].replace("left", "X").replace("right", "X") for i in range(23))
    assert all(0 <= a < 23 and 0 <= b < 23 for a, b in EDGES)


def test_crop_inverse_maps_crop_centre_to_box_centre():
    rgb = np.zeros((480, 640, 3), np.uint8)
    patch, inverse = crop(rgb, [100, 50, 200, 300])
    assert patch.shape == (256, 192, 3)
    np.testing.assert_allclose(np.array([96, 128]) @ inverse[:, :2].T + inverse[:, 2], [200, 200], atol=1e-4)
    assert crop_affine([0, 0, 192, 256])[0, 0] == pytest.approx(1 / 1.25)


def test_tracker_box_from_confident_points():
    tracker = BoxTracker()
    assert tracker.needs_detection(0.0)
    tracker.detected([0, 0, 10, 10], 0.0)
    assert not tracker.needs_detection(0.5) and tracker.needs_detection(1.0)
    xy = np.stack([np.linspace(100, 200, 23), np.linspace(50, 450, 23)], 1)
    tracker.update(xy, np.ones(23), 640, 480)
    assert tracker.box == pytest.approx([85, 0, 130, 480])
    tracker.update(xy, np.zeros(23), 640, 480)
    assert tracker.box is None


@pytest.mark.skipif(not (MODELS / "obt-h3c.safetensors").exists() or not (MODELS / "obt-h3c.onnx").exists(),
                    reason="needs models/obt-h3c.safetensors and models/obt-h3c.onnx")
def test_onnx_matches_pytorch():
    torch = pytest.importorskip("torch")
    import onnxruntime as ort
    from open_body_tracker.model import decode
    from open_body_tracker.preprocess import normalize
    from open_body_tracker.weights import load_model
    crops = np.random.default_rng(0).integers(0, 256, (2, 256, 192, 3), dtype=np.uint8)
    with torch.inference_mode():
        xy, score, xyz = (v.numpy() for v in decode(load_model(MODELS / "obt-h3c.safetensors")(torch.from_numpy(normalize(crops)))))
    session = ort.InferenceSession(str(MODELS / "obt-h3c.onnx"), providers=["CPUExecutionProvider"])
    oxy, oscore, oxyz = session.run(None, {"crop": crops.transpose(0, 3, 1, 2).astype(np.float32)})
    assert np.mean(np.abs(oxy - xy) < 1) > 0.99
    assert np.abs(oxyz - xyz).max() < 1e-3
