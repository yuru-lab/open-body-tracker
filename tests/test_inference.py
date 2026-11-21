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


@pytest.mark.parametrize("name", ["obt-body-v0.1.0", "obt-body-v0.3.0"])
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
    if not (MODELS / "obt-body-v0.3.0.safetensors").exists() or not (MODELS / "obt-body-v0.3.0.onnx").exists():
        pytest.skip("needs models/obt-body-v0.3.0.safetensors and models/obt-body-v0.3.0.onnx")
    torch = pytest.importorskip("torch")
    import onnxruntime as ort
    from open_body_tracker import flip
    from open_body_tracker.model import decode
    from open_body_tracker.preprocess import normalize
    from open_body_tracker.weights import load_model
    crops = np.random.default_rng(1).integers(0, 256, (2, 256, 192, 3), dtype=np.uint8)
    model, index = load_model(MODELS / "obt-body-v0.3.0.safetensors"), torch.tensor(FLIP)
    with torch.inference_mode():
        image = torch.from_numpy(normalize(crops))
        plain, mirrored = model(image), model(image.flip(-1))
        back = [mirrored[0][:, index].flip(-1), mirrored[1][:, index], mirrored[2][:, index].flip(-1),
                mirrored[3][:, index], mirrored[4][:, index]]
        xy, _, xyz = (v.numpy() for v in decode(tuple((a + b) / 2 for a, b in zip(plain, back))))
    session = ort.InferenceSession(str(MODELS / "obt-body-v0.3.0.onnx"), providers=["CPUExecutionProvider"])
    fxy, _, fxyz = flip.run(session, crops.transpose(0, 3, 1, 2).astype(np.float32))
    assert np.mean(np.abs(fxy - xy) < 1) > 0.99
    assert np.abs(fxyz - xyz).max() < 1e-3


def test_model_names():
    from open_body_tracker.hub import FILES, MODELS, model_file
    assert model_file() == "obt-body-v0.3.0.onnx" and model_file("v2", fp16=True) == "obt-body-v0.2.0-fp16.onnx"
    assert model_file("v1") == "obt-body-v0.1.0.onnx" and model_file("v0.3.0-int8") == "obt-body-v0.3.0-int8.onnx"
    with pytest.raises(ValueError):
        model_file("v0.3.0-int8", fp16=True)
    assert all(model_file(v) in FILES for v in MODELS)
    # renamed files are the same bytes as their earlier names
    assert FILES["obt-body-v0.3.0.onnx"][1] == FILES["obt-body-v0.2.0.onnx"][1] == FILES["obt-body-v2.onnx"][1]
    assert FILES["obt-body-v0.1.0-fp16.onnx"][1] == FILES["obt-body-v1-fp16.onnx"][1]


def test_int8_close_to_float32():
    """The INT8 CPU model stays close to the float32 model it was made from (scripts/quantize_onnx.py)."""
    if not (MODELS / "obt-body-v0.3.0-int8.onnx").exists() or not (MODELS / "obt-body-v0.3.0.onnx").exists():
        pytest.skip("needs models/obt-body-v0.3.0.onnx and models/obt-body-v0.3.0-int8.onnx")
    import onnxruntime as ort
    from open_body_tracker import flip
    crops = np.random.default_rng(0).uniform(0, 255, (2, 3, 256, 192)).astype(np.float32)
    names = ["xy", "score", "xyz"]
    full = ort.InferenceSession(str(MODELS / "obt-body-v0.3.0.onnx"), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(MODELS / "obt-body-v0.3.0-int8.onnx"), providers=["CPUExecutionProvider"])
    (xy, _, xyz), (qxy, _, qxyz) = full.run(names, {"crop": crops}), int8.run(names, {"crop": crops})
    assert np.abs(qxy - xy).mean() < 3 and np.abs(qxyz - xyz).mean() < 0.03
    assert all(np.isfinite(v).all() for v in flip.run(int8, crops))  # the logit outputs are kept, so the flip test works
