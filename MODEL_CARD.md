# Model card: obt-body

obt-body-v0.3.0 (default; same weights as v0.2.0) and obt-body-v0.1.0 share the architecture, input and outputs.
v0.2.0 / v0.3.0 add 3D training data from ASPset-510. obt-body-v0.3.0-int8 is v0.3.0 quantized for CPUs.
Earlier file names: `obt-body-v2.*` = v0.2.0, `obt-body-v1.*` and `obt-h3c.*` = v0.1.0. Version history and all
benchmarks: [docs/MODELS.md](docs/MODELS.md).

| | |
|---|---|
| Task | Single-person, single-RGB-camera full-body pose: 23 keypoints in 2D and root-relative metric 3D |
| Keypoints | COCO 17 body points + 6 foot points (big toe, small toe, heel per foot) |
| Input | 256×192 person crop (from a person detector box, 25% margin) |
| Architecture | DINOv2 ViT-B/14 backbone → 2-scale neck → RTMW-style SimCC head (2D) + 3 × 300-bin classifiers for X/Y/Z (1 cm bins, ±1.5 m around the hip midpoint) |
| Parameters | 95.6 M |
| Files | `obt-body-v0.3.0.safetensors` (PyTorch), `obt-body-v0.3.0.onnx` (fp32, 382 MB), `obt-body-v0.3.0-fp16.onnx` (fp16 weights, 191 MB), `obt-body-v0.3.0-int8.onnx` (INT8 for CPU, 142 MB); the same without int8 for v0.2.0 and v0.1.0; `obt-h3c.*` = v0.1.0 under its first name (no logit outputs) |
| License | **Apache-2.0** (all versions; see [docs/LICENSING.md](docs/LICENSING.md)) |

## Training

Two stages, both starting from public weights only:

1. **2D (90 epochs, 42,390 updates of 32 crops)** — from DINOv2 ViT-B/14. Each update mixes
   16 ordinary + 4 hard COCO 2014 person crops (human keypoint labels) and 12 Open Images V7 person crops
   labelled by SAM 3D Body (teacher). Only images published under **CC BY 2.0** are used; CC BY-SA images
   were excluded (verified: zero CC BY-SA image ids in all updates).
2. **3D (30 epochs, 14,130 updates)** — adds the 6 foot points and the metric X/Y/Z heads, trained on the
   SAM 3D Body 3D outputs of the same crops, while keeping the 2D labels.
   **v0.2.0 / v0.3.0 only:** 4 of the 12 teacher crops of every update are replaced by ASPset-510 (CC0) crops of 7 training
   subjects (43,201 frames, one camera per recording) with their multi-view 3D (body 12 points, hip-centred; no 2D
   label). The ASPset subjects used for evaluation (official test split) are not among them.

The final checkpoint (EMA weights at the last update) was fixed **before** the final evaluation; it was not
picked among intermediate checkpoints. See [DATA.md](DATA.md) and [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Evaluation

Final evaluations, each run once on data not used for training or model selection:

| Benchmark | Metric | obt-body-v0.3.0 | obt-body-v0.1.0 | Reference |
|---|---|---:|---:|---:|
| COCO val2017, 2D, ground-truth boxes (images that overlap training removed, 4,737 images) | keypoint AP | **71.6** | 71.7 | RTMW-m (official): 74.1 |
| ASPset-510, 3 held-out subjects, 3,877 frames ¹ | PA-MPJPE (body 12 points) | **53.7 mm** [51.6, 55.8] | 67.9 mm [65.5, 70.4] | SAM 3D Body teacher: 46.7 mm |
| | MPJPE (hip-aligned) | 75.4 mm | 92.7 mm | teacher: 67.7 mm |
| | left/right depth order correct | 99.1 % | 97.8 % | teacher: 99.3 % |
| Harmony4D test, 3 sequences, 26,474 person crops (other scenes, fisheye cameras, two people in contact) | PA-MPJPE | **71.3 mm** | 76.0 mm | teacher: 34.6 mm; MediaPipe Full: 103.3 mm |
| | MPJPE (hip-aligned) | 127.0 mm | 130.3 mm | teacher: 70.6 mm |

¹ v0.2.0 / v0.3.0 were trained on 7 other ASPset subjects filmed in the same studio, so this gain is larger than in other scenes.

Intervals are 95% bootstrap intervals over recordings. ASPset reference 3D comes from a multi-camera
markerless reconstruction, not motion capture. The optional flip test (crop + mirror, averaged) improves v0.3.0 by about
1 mm in 3D and 3.5 mm in jitter at twice the cost; see [docs/MODELS.md](docs/MODELS.md).

Speed (one 256×192 crop, batch 1, same for every version): 5.6 ms (fp16) / 8 ms (fp32) on an RTX 3060 Ti with
PyTorch; ~100 ms on a Core i5-12600 CPU with ONNX Runtime (4 threads), 46 ms with the INT8 model.

INT8 model against float32 (paired): ASPset PA-MPJPE +0.18 mm, Harmony4D PA-MPJPE +0.52 mm, jitter +1.3 mm,
COCO AP 71.7 (±0). The head and two MLP layers stay float32; see [docs/MODELS.md](docs/MODELS.md).
Person detector on CPU (640×480, 2 threads): YOLOX-nano ~9 ms, YOLOX-tiny ~29 ms.

Person detector on COCO val2017 (top-1 person box, IoU ≥ 0.5, images with one large person):
YOLOX-nano 90.8 %, YOLOX-tiny 94.0 %; false detections on images without people: 3.6 % / 2.6 %.
In live use YOLOX-nano sometimes locks onto chairs; switch to YOLOX-tiny if that happens.

## Limitations

- One person at a time; the highest-scoring person box is used.
- 3D is relative to the hips (no global position) and is estimated from one view: depth errors are larger
  than in-image errors, and hands/face are not modelled beyond the COCO face points.
- Trained on still photos; there is no temporal model (the demos use a One Euro filter).
- Not a certified measurement device. Evaluate it for your own use case.

## License and use

The weights are licensed under Apache-2.0, all versions included (until 2026-11-21 they were distributed under the
SAM License). Part of the training labels were predicted by [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)
(Meta); the weights contain no SAM 3D Body code or weights. Why we consider Apache-2.0 appropriate, what is not
settled, and the licence history: [docs/LICENSING.md](docs/LICENSING.md). The training images keep their own
licences (CC BY 2.0; ASPset-510: CC0; see DATA.md).
