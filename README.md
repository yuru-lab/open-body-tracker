# open-body-tracker

[日本語](README.ja.md)

Lightweight full-body **3D pose from a single RGB camera**: 23 keypoints (body + feet) in 2D and in metres,
for one person, in real time on a consumer GPU or in the browser with WebGPU.

**[Try the browser demo](https://yuru-lab.github.io/open-body-tracker/)**. It runs locally and your video never leaves the device.

- Weights: [GitHub Releases](https://github.com/yuru-lab/open-body-tracker/releases) · [Hugging Face](https://huggingface.co/Yurup/open-body-tracker)
- Accuracy, training data and limitations: [MODEL_CARD.md](MODEL_CARD.md) · [DATA.md](DATA.md)

| | obt-h3c |
|---|---|
| COCO val2017 keypoint AP (2D, GT boxes) | 71.7 |
| ASPset-510 held-out PA-MPJPE (3D, body 12 points) | 67.9 mm |
| Model size | 95.6 M params · 191 MB (fp16 ONNX) |
| Speed (one person) | ~10 ms on an RTX 3060 Ti |

## How it works

```
camera frame ──► YOLOX-nano / tiny ──► person box ──► 256×192 crop ──► obt-h3c ──► 23 × (x, y) + 23 × (X, Y, Z)
                 (person detector)       ▲                                │
                                         └──── box from the previous pose ┘   (detector re-runs every 1 s)
```

## Quick start

Requires [mise](https://mise.jdx.dev/) (it installs Python, uv and Node) or Python 3.12 + [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/yuru-lab/open-body-tracker
cd open-body-tracker
mise install          # Python 3.12, uv, Node
mise run setup        # uv sync

mise run demo         # webcam demo; models are downloaded to ./models on first run
uv run obt-demo --image photo.jpg            # writes photo.pose.jpg
uv run obt-demo --detector tiny              # fewer false detections, slower
uv sync --extra gpu && uv run obt-demo --gpu # CUDA
```

Browser demo locally: `mise run web`, then open http://localhost:8000. Put the ONNX files in `web/models/`
to skip the download.

The default person detector is YOLOX-nano (fast). If it mistakes chairs or other objects for a person, switch to YOLOX-tiny.

## Use it in your code

```python
import cv2
from open_body_tracker.detector import PersonDetector
from open_body_tracker.hub import fetch
from open_body_tracker.pipeline import PosePipeline

pipeline = PosePipeline(fetch("obt-h3c.onnx"), PersonDetector(fetch("yolox_nano.onnx")))
pose = pipeline(cv2.imread("photo.jpg")[:, :, ::-1].copy())  # RGB
# pose["xy"]    (23, 2) image pixels
# pose["score"] (23,)   confidence
# pose["xyz"]   (23, 3) metres, hip-centred, camera axes (x right, y down, z forward)
```

Keypoint order: `open_body_tracker/keypoints.py` (COCO 17 + left/right big toe, small toe, heel).

## Repository layout

```
src/open_body_tracker/   model (PyTorch), preprocessing, detector, tracking pipeline, demo
web/                     browser demo (ONNX Runtime Web, WebGPU)
scripts/                 checkpoint conversion and ONNX export
docs/REPRODUCE.md        training recipe and reproduction
```

## Reproducibility

The training recipe is documented now. The training code, the image id lists and the SAM 3D Body teacher labels
(so that you do not need to run SAM 3D Body) will be released together after a full re-training run confirms that
the cleaned-up code reproduces the released accuracy. Status: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## License

- Code: [Apache-2.0](LICENSE)
- Model weights and teacher labels: [SAM License](licenses/SAM-LICENSE.txt), because the model was trained on SAM 3D Body outputs. Redistribute with the license; its use restrictions apply.
- Person detector: YOLOX (Apache-2.0, Megvii)
- Third-party notices: [NOTICE](NOTICE)

This is research software. It is not a medical or measurement device.
