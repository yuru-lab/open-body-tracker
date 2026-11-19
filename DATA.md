# Data

This repository contains **no images**. Training and evaluation data come from public sources that you
download yourself.

| Dataset | Used for | Licence (what we rely on) |
|---|---|---|
| COCO 2014 train/val, person keypoints | 2D training (≈15,000 person crops, incl. a small set of hard poses) | Annotations CC BY 4.0. **Only images marked CC BY 2.0** in the COCO metadata are used. |
| Open Images V7 (train, validation, test), person boxes | Teacher crops (250,419 person crops from 106,308 photos) | Images marked CC BY 2.0 in the official image list; boxes CC BY 4.0 |
| SAM 3D Body outputs on the Open Images crops | 2D, foot and 3D teacher labels | SAM License (Meta) — released with this project as keypoint files only |
| DINOv2 ViT-B/14 weights | Backbone initialisation | Apache-2.0 |
| COCO val2017 | Final 2D evaluation only | as COCO |
| ASPset-510 | 3D development (2 subjects) and final evaluation (3 held-out subjects) only | CC0 |

Not used for the released weights: CC BY-SA images, synthetic renders, Human-Art, or any data with
non-commercial terms.

Attribution: COCO Consortium (cocodataset.org); Open Images (Google, storage.googleapis.com/openimages);
individual photographs belong to their Flickr authors under the licence stated in each dataset's metadata.
