# Data

This repository contains **no images**. Training and evaluation data come from public sources that you
download yourself.

| Dataset | Used for | Licence (what we rely on) |
|---|---|---|
| COCO 2014 train/val, person keypoints | 2D training (≈15,000 person crops, incl. a small set of hard poses) | Annotations CC BY 4.0. **Only images marked CC BY 2.0** in the COCO metadata are used. |
| Open Images V7 (train, validation, test), person boxes | Teacher crops (250,419 person crops from 106,308 photos) | Images marked CC BY 2.0 in the official image list; boxes CC BY 4.0 |
| SAM 3D Body outputs on the Open Images crops | 2D, foot and 3D teacher labels | not distributed; SAM 3D Body itself: SAM License (Meta) |
| DINOv2 ViT-B/14 weights (v0.1.0 – v0.4.0), ViT-S/14 weights (v0.4.0-small) | Backbone initialisation | Apache-2.0 |
| COCO val2017 | Final 2D evaluation only | as COCO |
| ASPset-510 | **v0.2.0 and later:** 3D training on 7 subjects (43,201 frames, `training/prepare_aspset.py`). All versions: 3D development (2 other subjects) and evaluation (the 3 official test subjects) | CC0 |
| Harmony4D, training split (11 sessions: 02–04, 07–14) | **v0.4.0 only:** 3D training with its multi-view 3D, all cameras (every 10th frame; v0.4.0-small also every 4th frame) | MIT (dataset card) |
| Harmony4D, test split (01, 05, 15; 06, 16) | 3D evaluation only (docs/MODELS.md); never used for training | MIT (dataset card) |

Not used for the released weights: CC BY-SA images, synthetic renders, Human-Art, or any data with
non-commercial terms.

Attribution: COCO Consortium (cocodataset.org); Open Images (Google, storage.googleapis.com/openimages);
individual photographs belong to their Flickr authors under the licence stated in each dataset's metadata.
