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


@pytest.mark.parametrize("name", ["obt-body-v1", "obt-body-v2"])
def test_onnx_matches_pytorch(name):
    if not (MODELS / f"{name}.safetensors").exists() or not (MODELS / f"{name}.onnx").exists():
        pytest.skip(f"needs models/{name}.safetensors and models/{name}.onnx")
    torch = pytest.importorskip("torch")
    import onnxruntime as ort
    from open_body_tracker.model import decode
    from open_body_tracker.preprocess import normalize
    from open_body_tracker.weights import load_model
    crops = np.random.default_rng(0).integers(0, 256, (2, 256, 192, 3), dtype=np.uint8)
    with torch.inference_mode():
        xy, score, xyz = (v.numpy() for v in decode(load_model(MODELS / f"{name}.safetensors")(torch.from_numpy(normalize(crops)))))
    session = ort.InferenceSession(str(MODELS / f"{name}.onnx"), providers=["CPUExecutionProvider"])
    oxy, oscore, oxyz = session.run(["xy", "score", "xyz"], {"crop": crops.transpose(0, 3, 1, 2).astype(np.float32)})
    assert np.mean(np.abs(oxy - xy) < 1) > 0.99
    assert np.abs(oxyz - xyz).max() < 1e-3


def test_flip_test_matches_logit_average():
    """open_body_tracker.flip (numpy, on the ONNX logits) == averaging the PyTorch logits of the crop and its mirror."""
    if not (MODELS / "obt-body-v2.safetensors").exists() or not (MODELS / "obt-body-v2.onnx").exists():
        pytest.skip("needs models/obt-body-v2.safetensors and models/obt-body-v2.onnx")
    torch = pytest.importorskip("torch")
    import onnxruntime as ort
    from open_body_tracker import flip
    from open_body_tracker.model import decode
    from open_body_tracker.preprocess import normalize
    from open_body_tracker.weights import load_model
    crops = np.random.default_rng(1).integers(0, 256, (2, 256, 192, 3), dtype=np.uint8)
    model, index = load_model(MODELS / "obt-body-v2.safetensors"), torch.tensor(FLIP)
    with torch.inference_mode():
        image = torch.from_numpy(normalize(crops))
        plain, mirrored = model(image), model(image.flip(-1))
        back = [mirrored[0][:, index].flip(-1), mirrored[1][:, index], mirrored[2][:, index].flip(-1),
                mirrored[3][:, index], mirrored[4][:, index]]
        xy, _, xyz = (v.numpy() for v in decode(tuple((a + b) / 2 for a, b in zip(plain, back))))
    session = ort.InferenceSession(str(MODELS / "obt-body-v2.onnx"), providers=["CPUExecutionProvider"])
    fxy, _, fxyz = flip.run(session, crops.transpose(0, 3, 1, 2).astype(np.float32))
    assert np.mean(np.abs(fxy - xy) < 1) > 0.99
    assert np.abs(fxyz - xyz).max() < 1e-3
