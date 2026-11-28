# Model card: obt-body and obt-hand

obt-body-v0.5.0-small (default, ViT-S) with obt-hand-v0.5.0-small (beta, 21 points per hand), and obt-body-v0.4.0
(ViT-B, the most accurate body model, no hands) are the current models; each has an INT8 file for CPUs.
obt-body-v0.5.0-small is obt-body-v0.4.0-small trained further together with the hands. v0.4.0 changed the
architecture (ray-frame 3D, rigid skeleton, finer patch grid) but kept the input and outputs of v0.1.0 – v0.3.0.
Earlier versions stay available; version history and all benchmarks: [docs/MODELS.md](docs/MODELS.md).

| | |
|---|---|
| Task | Single-person, single-RGB-camera full-body pose: 23 keypoints in 2D and root-relative metric 3D; with obt-hand (v0.5.0, beta) also 21 points per hand in 2D and in 3D relative to the wrist |
| Keypoints | COCO 17 body points + 6 foot points (big toe, small toe, heel per foot); hands: 4 points per finger and the wrist |
| Input | 256×192 person crop (from a person detector box, 25% margin); hands: a 140×140 crop per hand (2.0 × the extent of the person model's rough hand points; left hands mirrored) |
| Architecture | DINOv2 ViT-S/14 (v0.4.0-small) or ViT-B/14 (v0.4.0) on a 26×20 patch grid → 2-scale neck → RTMW-style SimCC head (2D) + 3 × 300-bin classifiers for X/Y/Z (1 cm bins, ±1.5 m) + a 4-number body-shape code. The 3D joints are rebuilt as a rigid skeleton (10 bone lengths from the shape code, left = right; bone directions from the X/Y/Z heads) in the crop's ray frame, then turned into camera axes with the camera's focal length (assumed when unknown) |
| Hand pass (v0.5.0) | The person model's head has 42 more rows (rough hand points, used to place the hand crops; masked so the 23 body rows are unaffected). obt-hand runs the same ViT-S/14 on the hand crop (10×10 patches) with its own neck and 21-point head: SimCC 2D + 3 × 300-bin classifiers for wrist-relative 3D (±0.25 m, 1.7 mm bins) in the hand crop's ray frame |
| Parameters | 31.6 M (v0.5.0-small person model) + 6.9 M (hand neck and head; the hand file also stores the shared backbone), 28.9 M (v0.4.0-small), 95.9 M (v0.4.0); v0.1.0 – v0.3.0: 95.6 M |
| Files | `obt-body-v0.5.0-small.safetensors` (PyTorch, person model and hand pass), `.onnx` (fp32, 127 MB), `-fp16.onnx` (64 MB), `-int8.onnx` (64 MB); `obt-hand-v0.5.0-small.onnx` (115 MB), `-fp16.onnx` (58 MB), `-int8.onnx` (54 MB); `obt-body-v0.4.0-small.safetensors`, `.onnx` (116 MB), `-fp16.onnx` (58 MB), `-int8.onnx` (53 MB); `obt-body-v0.4.0.safetensors`, `.onnx` (384 MB), `-fp16.onnx` (192 MB), `-int8.onnx` (140 MB); earlier versions as before |
| License | **Apache-2.0** (all versions; see [docs/LICENSING.md](docs/LICENSING.md)) |

## Training (v0.5.0)

obt-body-v0.5.0-small starts from obt-body-v0.4.0-small and was trained in two steps. In both, every update draws 32
person crops (12 ordinary + 4 hard COCO, 8 Open Images teacher crops, 2 ASPset, 6 Harmony4D from every 4th frame),
and the body also learns from v0.4.0-small's outputs (distillation):

1. 30 epochs (14,130 updates): the 42 hand rows of the person model learn SAM 3D Body's hand points on the Open Images
   teacher crops.
2. 40 epochs (18,840 updates): the hand pass is added; every update also draws 64 hand crops (hand loss weight 2).
   Hand crops: 82,508 from the Open Images person photos (placed by the teacher's hand points) and 24,156 cut around
   Open Images "Human hand" boxes (from 23,047 more photos), the latter 23% of the draws. Labels:
   the hand decoder of SAM 3D Body run on each hand crop. A crop is kept only when SAM's fit is good and MediaPipe Hand
   Landmarker finds a hand in the same place (its mean point distance below 0.3 × the hand's size); for the hand-box
   crops, which have no left/right label, SAM is run as both a right and a left hand and the one closer to
   MediaPipe's points is kept when MediaPipe's handedness agrees. MediaPipe outputs are used only for these checks,
   not as targets. Augmentation: scale, shift, rotation (±40°), low resolution, colour.

As in v0.4.0, only photos published under CC BY 2.0 are used, the released weights are the EMA weights of the last
update, and the hand recipe was chosen on hand results we cannot publish (see [Limitations](#limitations)). The
training code of v0.5.0 is not in this repository.

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

| Benchmark | Metric | v0.5.0-small | v0.4.0-small | v0.4.0 | v0.3.0 | Reference |
|---|---|---:|---:|---:|---:|---:|
| Harmony4D test 06_sword3 + 16_mma5, 48,280 crops (scored once per release) | PA-MPJPE (body 12 points) | 48.8 mm [48.0, 49.5] | 49.4 mm [48.7, 50.2] | **45.2 mm** [44.5, 45.9] | — | InstantHMR medium-commercial: 47.1 mm [46.2, 48.0] |
| | MPJPE (hip-aligned) | 61.4 mm | 62.5 mm | **57.7 mm** | — | InstantHMR: 67.0 mm |
| | jitter (acceleration error) | 41.9 mm | 41.5 mm | **38.5 mm** | — | InstantHMR: 43.7 mm |
| Harmony4D test 01 + 05 + 15, 26,474 crops | PA-MPJPE | 52.1 mm [50.2, 54.2] | 52.8 mm [50.9, 54.8] | **48.9 mm** [47.0, 51.1] | 71.3 mm | teacher: 34.6 mm |
| | MPJPE | 72.3 mm | 73.7 mm | **66.9 mm** | 127.0 mm | teacher: 70.6 mm |
| ASPset-510, 3 held-out subjects, 3,877 frames ¹ | PA-MPJPE | 50.1 mm [48.1, 52.1] | 50.3 mm [48.2, 52.3] | **48.1 mm** [46.1, 50.1] | 53.7 mm | teacher: 46.7 mm |
| | MPJPE | 71.0 mm | 71.7 mm | **68.0 mm** | 75.4 mm | teacher: 67.7 mm |
| COCO val2017, 2D, ground-truth boxes, clean (4,737 images) | keypoint AP | 69.1 | 69.9 | **72.8** | 71.6 | RTMW-m (official): 74.1 |

¹ Trained on 7 other ASPset subjects filmed in the same studio, so this is closer to home than the other sets.

Intervals are 95% bootstrap intervals over recordings (× camera for Harmony4D). ASPset reference 3D comes from a
multi-camera markerless reconstruction, Harmony4D's from multi-view triangulation plus a fitted body model; neither
is marker-based motion capture. The optional flip test (crop + mirror, averaged) improves v0.4.0 by about 0.5 mm in
3D and 2.4 mm in jitter at twice the cost.

Hands (v0.5.0): no accuracy numbers yet (see Limitations). Speed for both hands (batch 2): 28 ms fp32 / 17 ms INT8 on
the Core i5-12600 CPU, 3.2 ms fp16 on the RTX 5080 (PyTorch); TensorRT fp16 on an RTX 3060 Ti: 0.9 ms per hand
crop, 1.7 ms for the person crop. v0.5.0-small person model: CPU 82 / 52 ms (fp32 / INT8; v0.4.0-small measured
alongside: 78 / 46 ms), GPU 4.7 / 6.9 ms (fp16 / fp32, median of 4 runs; v0.4.0-small in the same runs: 4.7 / 7.3 ms).

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
  than in-image errors. The face is not modelled beyond the COCO face points.
- Hands (v0.5.0) are a beta. We have not found a public hand benchmark whose licence lets us publish results, so no
  hand accuracy is reported; the hand recipe was chosen on results we cannot publish. Each hand is seen only through
  its 140×140 crop: far-away hands are rough, and when the person model misplaces a hand the crop misses it. Every
  finger is always predicted, hidden or not; a hand is dropped only when the body's wrist score is low. The hand
  labels come from SAM 3D Body and were checked automatically, not by people.
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
licences (CC BY 2.0; ASPset-510: CC0; Harmony4D: MIT; see DATA.md). The hand labels were also predicted by SAM 3D
Body (its hand decoder) and checked with MediaPipe Hand Landmarker (Apache-2.0); neither is included.
