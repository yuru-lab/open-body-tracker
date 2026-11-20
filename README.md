# open-body-tracker

[日本語](README.ja.md)

Lightweight full-body **3D pose from a single RGB camera**: 23 keypoints (body + feet) in 2D and in metres,
for one person, in real time on a consumer GPU or in the browser with WebGPU.

**[Try the browser demo](https://yuru-lab.github.io/open-body-tracker/)**. It runs locally and your video never leaves the device.

![obt-body-v1 on an ASPset-510 clip: 2D skeleton on the video, 3D prediction next to the reference](docs/sample-aspset.gif)

<sub>A person from [ASPset-510](https://github.com/anibali/aspset-510) (CC0) who was never used in training.
Grey: ASPset multi-view reference 3D. Colour: obt-body-v1 (the first release), one camera, each frame on its own,
no smoothing. The clip was chosen by a fixed rule (front camera, error within ±8 mm of the held-out average, most
varied pose): 64 mm PA-MPJPE here, 67.9 mm on average. Harder poses such as crouching or a back view reach about 86 mm.</sub>

- Weights: [GitHub Releases](https://github.com/yuru-lab/open-body-tracker/releases) · [Hugging Face](https://huggingface.co/Yurup/open-body-tracker)
- Model versions and benchmarks: [docs/MODELS.md](docs/MODELS.md) · accuracy, training data and limitations:
  [MODEL_CARD.md](MODEL_CARD.md) · [DATA.md](DATA.md)

| | **obt-body-v2** (default) | obt-body-v1 |
|---|---:|---:|
| COCO val2017 keypoint AP (2D, GT boxes, clean subset) | 71.6 | 71.7 |
| ASPset-510 held-out subjects, PA-MPJPE (3D, body 12 points) ¹ | 53.7 mm | 67.9 mm |
| Harmony4D test (other scenes, two people in contact), PA-MPJPE | 71.3 mm | 76.0 mm |
| Model size | 95.6 M params · 191 MB (fp16 ONNX) | same |
| Speed (one person) | ~6–10 ms on an RTX 3060 Ti | same |

<sub>¹ v2 was also trained on 7 other people filmed in the ASPset studio, so its gain there is larger than you
should expect elsewhere; Harmony4D is the better guide. Details, intervals and the flip test: [docs/MODELS.md](docs/MODELS.md).</sub>

## How it works

```
camera frame ──► YOLOX-nano / tiny ──► person box ──► 256×192 crop ──► obt-body ──► 23 × (x, y) + 23 × (X, Y, Z)
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
uv run obt-demo --flip                       # flip test: a little more accurate and steadier, ~2x slower
uv run obt-demo --model-version v1           # the previous model
uv sync --extra gpu && uv run obt-demo --gpu # CUDA
```

Browser demo locally: `mise run web`, then open http://localhost:8000. Put the ONNX files in `web/models/`
to skip the download.

The default person detector is YOLOX-nano (fast). If it mistakes chairs or other objects for a person, switch to YOLOX-tiny.

## Use it in your code

```python
import cv2
from open_body_tracker.detector import PersonDetector
from open_body_tracker.hub import fetch, model_file
from open_body_tracker.pipeline import PosePipeline

pipeline = PosePipeline(fetch(model_file("v2")), PersonDetector(fetch("yolox_nano.onnx")))  # flip=True: flip test
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
docs/MODELS.md           model versions and benchmarks
docs/REPRODUCE.md        training recipe and reproduction
```

## Reproducibility

The training code is in [`training/`](training/) and the labels, including the SAM 3D Body teacher outputs (so you
do not need to run SAM 3D Body), are on Hugging Face:
[`Yurup/open-body-tracker-labels`](https://huggingface.co/datasets/Yurup/open-body-tracker-labels). A full
re-training run with exactly this code and these labels reproduced the released accuracy (COCO AP 71.6 vs 71.7,
ASPset PA-MPJPE 67.6 vs 67.9 mm). obt-body-v2 adds one step that prepares ASPset-510 (CC0) crops; its training
batches and first losses match the release run bit for bit. Steps and details: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## License

- Code: [Apache-2.0](LICENSE)
- Model weights and teacher labels: [SAM License](licenses/SAM-LICENSE.txt), because the model was trained on SAM 3D Body outputs. Redistribute with the license; its use restrictions apply.
- obt-body-v2 was also trained on ASPset-510 (CC0)
- Person detector: YOLOX (Apache-2.0, Megvii)
- Third-party notices: [NOTICE](NOTICE)

This is research software. It is not a medical or measurement device.
