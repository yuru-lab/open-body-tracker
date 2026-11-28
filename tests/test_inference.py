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


def test_ray_to_camera():
    from open_body_tracker.geometry import default_focal, ray_to_camera, rotation_to_z
    points = np.random.default_rng(2).normal(size=(23, 3))
    _, centred = crop(np.zeros((480, 640, 3), np.uint8), [270, 90, 100, 300])  # crop centre = image centre
    np.testing.assert_allclose(ray_to_camera(points, centred, 640, 480), points, atol=1e-6)
    _, inverse = crop(np.zeros((480, 640, 3), np.uint8), [500, 100, 100, 300])  # crop centre at (550, 250)
    focal = default_focal(640)
    ray = np.array([(550 - 320) / focal, (250 - 240) / focal, 1.0])
    np.testing.assert_allclose(rotation_to_z(ray) @ (ray / np.linalg.norm(ray)), [0, 0, 1], atol=1e-12)
    # the ray-frame forward axis becomes the ray through the crop centre
    np.testing.assert_allclose(ray_to_camera(np.array([0, 0, 1.0]), inverse, 640, 480), ray / np.linalg.norm(ray), atol=1e-6)


@pytest.mark.parametrize("name", ["obt-body-v0.1.0", "obt-body-v0.3.0", "obt-body-v0.4.0-small", "obt-body-v0.4.0", "obt-body-v0.5.0-small"])
def test_onnx_matches_pytorch(name):
    if not (MODELS / f"{name}.safetensors").exists() or not (MODELS / f"{name}.onnx").exists():
        pytest.skip(f"needs models/{name}.safetensors and models/{name}.onnx")
    torch = pytest.importorskip("torch")
    import onnxruntime as ort
    from open_body_tracker.preprocess import normalize
    from open_body_tracker.weights import load_model
    crops = np.random.default_rng(0).integers(0, 256, (2, 256, 192, 3), dtype=np.uint8)
    with torch.inference_mode():
        model = load_model(MODELS / f"{name}.safetensors")
        xy, score, xyz = (v.numpy() for v in model.decode(model(torch.from_numpy(normalize(crops)))))
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
    from open_body_tracker.hub import FILES, HAND_MODELS, MODELS, hand_model_file, model_file
    assert model_file() == "obt-body-v0.5.0-small.onnx" and model_file("v0.4.0", fp16=True) == "obt-body-v0.4.0-fp16.onnx"
    assert hand_model_file() == "obt-hand-v0.5.0-small.onnx" and hand_model_file(fp16=True) == "obt-hand-v0.5.0-small-fp16.onnx"
    assert all(v in MODELS and hand_model_file(v) in FILES for v in HAND_MODELS)
    with pytest.raises(ValueError):
        hand_model_file("v0.4.0-small")
    assert model_file("v2", fp16=True) == "obt-body-v0.2.0-fp16.onnx" and model_file("v0.4.0-small-int8") == "obt-body-v0.4.0-small-int8.onnx"
    assert model_file("v1") == "obt-body-v0.1.0.onnx" and model_file("v0.3.0-int8") == "obt-body-v0.3.0-int8.onnx"
    for int8 in ("v0.3.0-int8", "v0.4.0-small-int8", "v0.5.0-small-int8"):
        with pytest.raises(ValueError):
            model_file(int8, fp16=True)
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


def test_flip_test_averages_outputs_for_v040():
    """v0.4.0: open_body_tracker.flip == averaging the decoded PyTorch outputs of the crop and its mirror."""
    name = "obt-body-v0.4.0-small"
    if not (MODELS / f"{name}.safetensors").exists() or not (MODELS / f"{name}.onnx").exists():
        pytest.skip(f"needs models/{name}.safetensors and models/{name}.onnx")
    torch = pytest.importorskip("torch")
    import onnxruntime as ort
    from open_body_tracker import flip
    from open_body_tracker.preprocess import normalize
    from open_body_tracker.weights import load_model
    crops = np.random.default_rng(1).integers(0, 256, (2, 256, 192, 3), dtype=np.uint8)
    model, index = load_model(MODELS / f"{name}.safetensors"), torch.tensor(FLIP)
    with torch.inference_mode():
        image = torch.from_numpy(normalize(crops))
        pxy, _, pxyz = model.decode(model(image))
        mxy, _, mxyz = model.decode(model(image.flip(-1)))
        mxy, mxyz = mxy[:, index].clone(), mxyz[:, index].clone()
        mxy[..., 0], mxyz[..., 0] = 191 - mxy[..., 0], -mxyz[..., 0]
        xy, xyz = ((pxy + mxy) / 2).numpy(), ((pxyz + mxyz) / 2).numpy()
    session = ort.InferenceSession(str(MODELS / f"{name}.onnx"), providers=["CPUExecutionProvider"])
    assert flip.averages_outputs(session)
    fxy, _, fxyz = flip.run(session, crops.transpose(0, 3, 1, 2).astype(np.float32))
    assert np.mean(np.abs(fxy - xy) < 1) > 0.99
    assert np.abs(fxyz - xyz).max() < 1e-3


def test_hand_crop_geometry():
    """The hand crop is a square of 2.0 x the hand's extent around its box centre, 140 x 140 pixels."""
    from open_body_tracker.hands import EDGES, INPUT, NAMES, WRIST, hand_affine
    assert len(NAMES) == 21 and NAMES[WRIST] == "wrist" and NAMES[0] == "thumb_tip" and NAMES[19] == "pinky_mcp"
    assert all(0 <= a < 21 and 0 <= b < 21 for a, b in EDGES) and len(EDGES) == 20
    points = np.array([[100.0, 200.0], [150.0, 220.0], [120.0, 260.0]])
    matrix = hand_affine(np.vstack([points, points.mean(0)[None].repeat(18, 0)]))
    np.testing.assert_allclose(np.array([125, 230]) @ matrix[:, :2].T + matrix[:, 2], [INPUT / 2, INPUT / 2], atol=1e-4)
    assert matrix[0, 0] == pytest.approx(INPUT / (2.0 * 60))


def test_hand_model_matches_pytorch():
    """obt-hand ONNX == the PyTorch hand pass; the body rows of a hand model == the same weights without hand rows."""
    weights, hand = MODELS / "obt-body-v0.5.0-small.safetensors", MODELS / "obt-hand-v0.5.0-small.onnx"
    if not weights.exists() or not hand.exists():
        pytest.skip("needs models/obt-body-v0.5.0-small.safetensors and models/obt-hand-v0.5.0-small.onnx")
    torch = pytest.importorskip("torch")
    import dataclasses
    import onnxruntime as ort
    from open_body_tracker.model import BodyModel, decode_hand
    from open_body_tracker.preprocess import normalize
    from open_body_tracker.weights import load_model
    model = load_model(weights)
    crops = np.random.default_rng(3).integers(0, 256, (2, 140, 140, 3), dtype=np.uint8)
    with torch.inference_mode():
        xy, _, xyz = (v.numpy() for v in decode_hand(model.hands(torch.from_numpy(normalize(crops)))))
    session = ort.InferenceSession(str(hand), providers=["CPUExecutionProvider"])
    oxy, _, oxyz = session.run(["xy", "score", "xyz"], {"crop": crops.transpose(0, 3, 1, 2).astype(np.float32)})
    assert np.mean(np.abs(oxy - xy) < 1) > 0.99 and np.abs(oxyz - xyz).max() < 1e-3
    assert np.abs(xyz[:, -1]).max() < 1e-6  # wrist-relative
    body = BodyModel(config=dataclasses.replace(model.config, hands=False))
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("hand2.")}
    for key in [k for k in state if k.startswith(("head.final_layer.", "head.final_layer2.")) and state[k].shape[:1] == (65,)]:
        state[key] = state[key][:23]
    body.load_state_dict(state, strict=False)
    body.eval()
    people = torch.from_numpy(normalize(np.random.default_rng(4).integers(0, 256, (2, 256, 192, 3), dtype=np.uint8)))
    with torch.inference_mode():
        a, b = model.decode(model(people)), body.decode(body(people))
    assert (a[0] == b[0]).all() and (a[2] - b[2]).abs().max() < 1e-5


def test_pipeline_with_hands():
    """End to end on a blank frame with a fixed person box: both hands come back with 21 points each."""
    body, hand = MODELS / "obt-body-v0.5.0-small.onnx", MODELS / "obt-hand-v0.5.0-small.onnx"
    if not body.exists() or not hand.exists():
        pytest.skip("needs models/obt-body-v0.5.0-small.onnx and models/obt-hand-v0.5.0-small.onnx")
    from open_body_tracker.pipeline import PosePipeline
    pipeline = PosePipeline(body, lambda rgb: [200, 40, 240, 400], track=False, hand_model=hand)
    pipeline.hands.min_wrist_score = -1e9  # a blank frame has no confident wrist; force both hands through
    rgb = np.random.default_rng(5).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    pose = pipeline(rgb, now=0.0)
    for side in ("right", "left"):
        h = pose["hands"][side]
        assert h["xy"].shape == (21, 2) and h["score"].shape == (21,) and h["xyz"].shape == (21, 3)
        np.testing.assert_allclose(h["xyz"][-1], pose["xyz"][{"right": 10, "left": 9}[side]], atol=1e-5)
    with pytest.raises(ValueError):
        PosePipeline(MODELS / "obt-body-v0.4.0-small.onnx", lambda rgb: None, hand_model=hand)
