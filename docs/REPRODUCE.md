# Reproducing the model

> **Status:** the training code and data-preparation scripts are being cleaned up for release, and a full
> re-training run is being used to check that the published recipe reproduces the released accuracy.
> This page will be updated with the scripts and the result of that run.

What is already available:

- `scripts/convert_checkpoint.py` — training checkpoint → `.safetensors`
- `scripts/export_onnx.py` — `.safetensors` → ONNX (fp32 or fp16), with crop normalisation and decoding inside the graph
- `tests/test_inference.py` — checks ONNX against PyTorch

Recipe (summary of what the scripts will run):

| | Stage 1: 2D | Stage 2: 3D |
|---|---|---|
| Start | DINOv2 ViT-B/14 (official) + random head | Stage 1 EMA weights; foot rows copied from the ankles; X/Y/Z heads random |
| Batch | 32 = 16 COCO + 4 hard COCO + 12 teacher crops | same mix; 3D loss on teacher crops |
| Length | 90 epochs × 471 updates = 42,390 | 30 epochs × 471 = 14,130 |
| Optimiser | AdamW, weight decay 0.05, layer-wise lr decay 0.75 (backbone top ×0.1), grad-clip 35 | same |
| Schedule | linear warm-up 500 updates, constant to half, cosine to 2.5e-5 | peak 2.5e-4, same shape, cosine to 1.25e-5 |
| Loss | SimCC KL divergence (label softmax β=10, σ = 4.9 / 5.66 px) | + same KL loss on X/Y/Z with Gaussian labels (σ = 3 cm), weight 1 |
| Augmentation | flip, half-body, scale 0.5–1.5, rotation ±90°, shift, photometric, blur, coarse dropout | same |
| Kept | EMA (momentum 2e-4) weights of the last update | same |

Hardware used for the release: one RTX 3060 Ti (8 GB), about 11 h (stage 1) + 4 h (stage 2).
