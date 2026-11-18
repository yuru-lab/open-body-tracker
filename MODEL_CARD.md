# Model card: obt-h3c

| | |
|---|---|
| Task | Single-person, single-RGB-camera full-body pose: 23 keypoints in 2D and root-relative metric 3D |
| Keypoints | COCO 17 body points + 6 foot points (big toe, small toe, heel per foot) |
| Input | 256×192 person crop (from a person detector box, 25% margin) |
| Architecture | DINOv2 ViT-B/14 backbone → 2-scale neck → RTMW-style SimCC head (2D) + 3 × 300-bin classifiers for X/Y/Z (1 cm bins, ±1.5 m around the hip midpoint) |
| Parameters | 95.6 M |
| Files | `obt-h3c.safetensors` (PyTorch), `obt-h3c.onnx` (fp32, 382 MB), `obt-h3c-fp16.onnx` (fp16 weights, 191 MB) |
| License | **SAM License** (`licenses/SAM-LICENSE.txt`) — the model was trained on SAM 3D Body outputs |

## Training

Two stages, both starting from public weights only:

1. **2D (90 epochs, 42,390 updates of 32 crops)** — from DINOv2 ViT-B/14. Each update mixes
   16 ordinary + 4 hard COCO 2014 person crops (human keypoint labels) and 12 Open Images V7 person crops
   labelled by SAM 3D Body (teacher). Only images published under **CC BY 2.0** are used; CC BY-SA images
   were excluded (verified: zero CC BY-SA image ids in all updates).
2. **3D (30 epochs, 14,130 updates)** — adds the 6 foot points and the metric X/Y/Z heads, trained on the
   SAM 3D Body 3D outputs of the same crops, while keeping the 2D labels.

The final checkpoint (EMA weights at the last update) was fixed **before** the final evaluation; it was not
picked among intermediate checkpoints. See [DATA.md](DATA.md) and [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Evaluation

Final evaluations, run once on data not used for training or model selection:

| Benchmark | Metric | obt-h3c | Reference |
|---|---|---:|---:|
| COCO val2017, 2D, ground-truth boxes (images that overlap training removed, 4,737 images) | keypoint AP | **71.7** | RTMW-m (official): 74.1 |
| ASPset-510, 3 held-out subjects, 3,877 frames | PA-MPJPE (body 12 points) | **67.9 mm** [65.5, 70.4] | SAM 3D Body teacher: 46.7 mm |
| | MPJPE (hip-aligned) | 92.7 mm | teacher: 67.7 mm |
| | left/right depth order correct | 97.8 % | teacher: 99.3 % |

Intervals are 95% bootstrap intervals over recordings. ASPset reference 3D comes from a multi-camera
markerless reconstruction, not motion capture.

Speed (one 256×192 crop, batch 1): ~10 ms on an RTX 3060 Ti (PyTorch, CUDA), ~200 ms on CPU.
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

The weights are distributed under the SAM License (Meta). Redistribution must include the license, and the
license's use restrictions apply (e.g. no military, warfare, weapons or ITAR-controlled uses). If you publish
research results obtained with this model, acknowledge the use of SAM Materials. The training images remain
under their own licences (CC BY 2.0, see DATA.md).
