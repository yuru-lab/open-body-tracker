# Reproducing the model

Everything needed to train `obt-h3c` from public data is in this repository and the label release
[`Yurup/open-body-tracker-labels`](https://huggingface.co/datasets/Yurup/open-body-tracker-labels).
SAM 3D Body does **not** need to be run: its outputs are part of the label release.

> **Verified (2026-11-20):** a full re-training run with exactly this code and the published labels, on one
> RTX 3060 Ti, reproduces the released accuracy within run-to-run noise:
>
> | | released `obt-h3c` | re-trained |
> |---|---:|---:|
> | COCO val2017 AP, clean / full | 71.7 / 72.7 | 71.6 / 73.0 |
> | ASPset-510 held-out PA-MPJPE | 67.9 mm | 67.6 mm |
> | ASPset-510 held-out MPJPE | 92.7 mm | 92.6 mm |
>
> Paired difference on the same 3,877 ASPset frames (re-trained − released): PA-MPJPE −0.24 mm
> [−0.78, +0.33], MPJPE −0.09 mm [−0.74, +0.63] (95% bootstrap intervals over the 90 recordings).
> Wall time: stage 1 about 11 h, stage 2 about 3.7 h.

## 1. Labels and photos

```sh
uv sync --extra train
uv run hf download Yurup/open-body-tracker-labels --repo-type dataset --local-dir labels
uv run python training/prepare_data.py --labels labels --out data      # downloads ~110k photos, ~40 GB of crops
```

`prepare_data.py` downloads each photo from its official location (COCO and the Open Images S3 bucket), checks its
SHA-256, cuts the fixed 256×192 crop and compares the crop with the SHA-256 of the crop used for the release.
On the release machine all 265,463 crops were bit-identical (OpenCV 5.0.0, pinned in `pyproject.toml`).

| File | Content | Licence |
|---|---|---|
| `coco2014.json` | 15,044 COCO 2014 person crops (14,955 ordinary + 89 hard poses): photo, crop affine, 17 keypoints | annotations CC BY 4.0; photos CC BY 2.0 only |
| `teacher.npz` | 250,419 Open Images V7 person crops: crop affine, 23 keypoints (2D) and 23 root-relative 3D joints from SAM 3D Body | SAM License |
| `openimages.jsonl` | 106,308 Open Images photos: URL, SHA-256, licence, author, title | photos CC BY 2.0 |

## 2. Training

```sh
uv run python training/train.py --stage 2d --out runs/stage1                            # ~11 h on an RTX 3060 Ti
uv run python training/train.py --stage 3d --init runs/stage1/final.pt --out runs/stage2 # ~3.5 h
uv run python scripts/convert_checkpoint.py runs/stage2/final.pt models/my-model.safetensors
uv run python scripts/export_onnx.py models/my-model.safetensors models/my-model.onnx
```

Interrupted runs continue with `--resume` (checkpoint every 1,000 updates). Peak memory: about 7.5 GB of GPU memory
(activation checkpointing, batch 32) and a few GB of RAM; the crops are memory-mapped.

| | Stage 1: 2D | Stage 2: 3D |
|---|---|---|
| Start | DINOv2 ViT-B/14 (official) + seeded head | stage 1 EMA weights; foot rows copied from the ankles; seeded X/Y/Z heads |
| Batch | 32 = 16 COCO + 4 hard COCO + 12 teacher crops | same; 3D loss on teacher crops only |
| Length | 90 epochs × 471 updates = 42,390 | 30 epochs × 471 = 14,130 |
| Optimiser | AdamW, weight decay 0.05, layer-wise lr decay 0.75 (backbone top ×0.1), grad-norm clip 35 | same |
| Learning rate | warm-up 500 updates, peak 5e-4 until half, cosine to 2.5e-5 | peak 2.5e-4, cosine to 1.25e-5 |
| Loss | SimCC KL divergence (label softmax β=10, σ = 4.9 / 5.66 px) | + same KL loss on X/Y/Z, Gaussian labels σ = 3 cm, weight 1 |
| Augmentation | flip, half-body, scale 0.5–1.5, rotation ±90°, shift, photometric, blur, median blur, coarse dropout | same; 3D labels flipped/rotated with the image |
| Kept | EMA (momentum 2e-4) weights of the last update | same |

How close is the code to the release run? Checked against the original (MMPose-based) training code:

- the initial weights of both stages are bit-identical (same seed),
- the augmented batches (images, 2D and 3D labels) of both stages are bit-identical,
- the SimCC/3D labels and the loss are identical,
- the losses of the first 10 updates are identical; later updates drift slightly (different PyTorch version), so the
  final weights are not bit-identical and the accuracy is expected to match within run-to-run noise.

## 3. Evaluation

```sh
uv run python training/evaluate_coco.py --weights models/my-model.safetensors --coco data/coco2017 --download
```

reports COCO val2017 keypoint AP with ground-truth boxes; "clean" excludes the 263 val2017 photos that appear in the
training archives (`training/coco_val2017_exclude.json`). The released model scores 71.7 (clean) / 72.7 (full).
