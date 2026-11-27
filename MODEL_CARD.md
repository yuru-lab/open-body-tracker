# Model card: obt-body

obt-body-v0.4.0-small (default, ViT-S) and obt-body-v0.4.0 (ViT-B) are the current models; each has an INT8 file for
CPUs. They change the architecture (ray-frame 3D, rigid skeleton, finer patch grid) but keep the input and outputs
of v0.1.0 – v0.3.0. Earlier versions stay available; version history and all benchmarks: [docs/MODELS.md](docs/MODELS.md).

| | |
|---|---|
| Task | Single-person, single-RGB-camera full-body pose: 23 keypoints in 2D and root-relative metric 3D |
| Keypoints | COCO 17 body points + 6 foot points (big toe, small toe, heel per foot) |
| Input | 256×192 person crop (from a person detector box, 25% margin) |
| Architecture | DINOv2 ViT-S/14 (v0.4.0-small) or ViT-B/14 (v0.4.0) on a 26×20 patch grid → 2-scale neck → RTMW-style SimCC head (2D) + 3 × 300-bin classifiers for X/Y/Z (1 cm bins, ±1.5 m) + a 4-number body-shape code. The 3D joints are rebuilt as a rigid skeleton (10 bone lengths from the shape code, left = right; bone directions from the X/Y/Z heads) in the crop's ray frame, then turned into camera axes with the camera's focal length (assumed when unknown) |
| Parameters | 28.9 M (v0.4.0-small), 95.9 M (v0.4.0); v0.1.0 – v0.3.0: 95.6 M |
| Files | `obt-body-v0.4.0-small.safetensors` (PyTorch), `.onnx` (fp32, 116 MB), `-fp16.onnx` (58 MB), `-int8.onnx` (53 MB); `obt-body-v0.4.0.safetensors`, `.onnx` (384 MB), `-fp16.onnx` (192 MB), `-int8.onnx` (140 MB); earlier versions as before |
| License | **Apache-2.0** (all versions; see [docs/LICENSING.md](docs/LICENSING.md)) |

## Training (v0.4.0)

Every update draws 32 crops: 16 ordinary + 4 hard COCO 2014 person crops (human 2D labels) and 12 crops with 3D
labels, shared between Open Images V7 crops labelled by SAM 3D Body (teacher), ASPset-510 (CC0, the 7 training
subjects of v0.2.0) and Harmony4D (MIT; multi-view 3D of 11 training-split sessions, 02–04 and 07–14, all cameras).
Only COCO and Open Images photos published under **CC BY 2.0** are used. 3D targets are in the ray frame of the
(augmented) crop; the rigid skeleton has its own losses on joints and bone lengths.

- **obt-body-v0.4.0** (ViT-B): from the 2D stage of v0.1.0 (positions resized to the 26×20 grid), then 90 epochs
  (42,390 updates) of 3D training with 4 teacher + 4 ASPset + 4 Harmony4D crops per update (Harmony4D every 10th
  frame, 204,027 crops).
- **obt-body-v0.4.0-small** (ViT-S): 2D stage of 90 epochs from DINOv2 ViT-S/14; 90 epochs of 3D training as above,
  also matching the outputs of an internal ViT-B obt model trained on the same sources (distillation); then 30 more
  epochs with denser Harmony4D (every 4th frame, 504,943 crops; 3 teacher + 3 ASPset + 6 Harmony4D crops per update).

The released weights are the EMA weights of the last update; no checkpoint was picked among intermediate ones.
**Model selection:** the recipe (data mix, extra fine-tuning, model sizes) was chosen on ASPset held-out and Harmony4D
01/05/15, so those numbers are development results for v0.4.0. Harmony4D 06/16 stayed closed until both models were
fixed and were then scored once. The training code of v0.4.0 is not yet in this repository; see
[docs/REPRODUCE.md](docs/REPRODUCE.md) for v0.1.0 – v0.3.0.

## Evaluation

| Benchmark | Metric | v0.4.0-small | v0.4.0 | v0.3.0 | Reference |
|---|---|---:|---:|---:|---:|
| Harmony4D test 06_sword3 + 16_mma5, 48,280 crops (scored once) | PA-MPJPE (body 12 points) | 49.4 mm [48.7, 50.2] | **45.2 mm** [44.5, 45.9] | — | InstantHMR medium-commercial: 47.1 mm [46.2, 48.0] |
| | MPJPE (hip-aligned) | 62.5 mm | **57.7 mm** | — | InstantHMR: 67.0 mm |
| | jitter (acceleration error) | 41.5 mm | **38.5 mm** | — | InstantHMR: 43.7 mm |
| Harmony4D test 01 + 05 + 15, 26,474 crops | PA-MPJPE | 52.8 mm [50.9, 54.8] | **48.9 mm** [47.0, 51.1] | 71.3 mm | teacher: 34.6 mm |
| | MPJPE | 73.7 mm | **66.9 mm** | 127.0 mm | teacher: 70.6 mm |
| ASPset-510, 3 held-out subjects, 3,877 frames ¹ | PA-MPJPE | 50.3 mm [48.2, 52.3] | **48.1 mm** [46.1, 50.1] | 53.7 mm | teacher: 46.7 mm |
| | MPJPE | 71.7 mm | **68.0 mm** | 75.4 mm | teacher: 67.7 mm |
| COCO val2017, 2D, ground-truth boxes, clean (4,737 images) | keypoint AP | 69.9 | **72.8** | 71.6 | RTMW-m (official): 74.1 |

¹ Trained on 7 other ASPset subjects filmed in the same studio, so this is closer to home than the other sets.

Intervals are 95% bootstrap intervals over recordings (× camera for Harmony4D). ASPset reference 3D comes from a
multi-camera markerless reconstruction, Harmony4D's from multi-view triangulation plus a fitted body model; neither
is marker-based motion capture. The optional flip test (crop + mirror, averaged) improves v0.4.0 by about 0.5 mm in
3D and 2.4 mm in jitter at twice the cost.

Speed (one 256×192 crop, batch 1): v0.4.0-small 5.2 ms (fp16) / 7.8 ms (fp32), v0.4.0 5.5 / 12.7 ms, on an
RTX 5080 with PyTorch; on a Core i5-12600 CPU with ONNX Runtime (4 threads) 82 / 248 ms in fp32 and 48 / 120 ms with
the INT8 files (v0.3.0 measured alongside: 3.7 / 7.8 ms GPU, 111 / 47 ms CPU). INT8 against float32: ASPset
PA-MPJPE +0.2 mm for both. Person detector on CPU (640×480, 2 threads): YOLOX-nano ~9 ms, YOLOX-tiny ~29 ms.

Person detector on COCO val2017 (top-1 person box, IoU ≥ 0.5, images with one large person):
YOLOX-nano 90.8 %, YOLOX-tiny 94.0 %; false detections on images without people: 3.6 % / 2.6 %.
In live use YOLOX-nano sometimes locks onto chairs; switch to YOLOX-tiny if that happens.

## Limitations

- One person at a time; the highest-scoring person box is used.
- 3D is relative to the hips (no global position) and is estimated from one view: depth errors are larger
  than in-image errors, and hands/face are not modelled beyond the COCO face points.
- Trained on still photos and single video frames; there is no temporal model (the demos use a One Euro filter).
- v0.4.0: the 3D orientation in camera axes depends on the focal length; with the default assumption (65°
  horizontal field of view) a person at the image edge may be turned by a few degrees. Bone lengths are fixed per
  frame by the shape code, which is estimated from the crop and is not a body measurement.
- Not a certified measurement device. Evaluate it for your own use case.

## License and use

The weights are licensed under Apache-2.0, all versions included (until 2026-11-21 they were distributed under the
SAM License). Part of the training labels were predicted by [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)
(Meta); the weights contain no SAM 3D Body code or weights. Why we consider Apache-2.0 appropriate, what is not
settled, and the licence history: [docs/LICENSING.md](docs/LICENSING.md). The training images keep their own
licences (CC BY 2.0; ASPset-510: CC0; Harmony4D: MIT; see DATA.md).
