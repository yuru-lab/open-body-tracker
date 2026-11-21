# Licensing

Short version: the code, the model weights and the SAM 3D Body teacher labels are all Apache-2.0. Annotations
and metadata we adapted from COCO and Open Images stay CC BY 4.0. Photos are not distributed.

| What | Where | Licence |
|---|---|---|
| Source code | this repository | [Apache-2.0](../LICENSE) |
| Model weights, every version (`obt-body-v*.*`, and the earlier names `obt-body-v1/v2.*`, `obt-h3c.*`) | [GitHub Releases](https://github.com/yuru-lab/open-body-tracker/releases), [Hugging Face](https://huggingface.co/Yurup/open-body-tracker) | [Apache-2.0](../LICENSE) |
| Teacher labels `teacher.npz`: 2D and 3D keypoints predicted by SAM 3D Body | [labels on Hugging Face](https://huggingface.co/datasets/Yurup/open-body-tracker-labels) | Apache-2.0, except the crop boxes (next row) |
| Crop boxes in `teacher.npz` and the photo list `openimages.jsonl` (from Open Images V7 boxes and metadata) | same | CC BY 4.0 (Google LLC) |
| `coco2014.json` (adapted COCO 2014 person keypoints) | same | CC BY 4.0 (COCO Consortium) |
| Training and evaluation photos | not distributed; `training/prepare_data.py` downloads them from the original sources | each photo's own licence (only CC BY 2.0 photos are used; see [DATA.md](../DATA.md)) |
| YOLOX person detectors (unmodified mirrors) | Hugging Face | Apache-2.0 (Megvii), [licenses/YOLOX-LICENSE.txt](../licenses/YOLOX-LICENSE.txt) |

Third-party work the model builds on, and the credits we keep, are listed in [NOTICE](../NOTICE).

## Why the weights and labels are Apache-2.0

The model was trained in part on outputs of [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) (Meta),
which is distributed under the SAM License. We are grateful to Meta for releasing it. We read the SAM License
(version of 19 November 2025, [licenses/SAM-LICENSE.txt](../licenses/SAM-LICENSE.txt)) as follows:

- "SAM Materials" are the documentation, models, software, code and weights **distributed by Meta**. Redistribution
  terms (Section 1.b.i) apply to SAM Materials and derivative works of them.
- The License treats outputs separately from SAM Materials ("the SAM Materials and any output and results",
  Sections 3 and 5.b) and has no clause that restricts what outputs may be used for, including training other models.
- Our weights contain no SAM Materials: no SAM 3D Body code, weights or architecture. They are DINOv2 ViT-B/14
  weights (Apache-2.0) trained further, plus a head re-implemented from MMPose (Apache-2.0). SAM 3D Body was only
  run to produce keypoint labels.
- The teacher labels are such outputs: keypoint coordinates predicted for Open Images photos.

We therefore distribute the weights and the labels under Apache-2.0 and do not pass on the SAM License's
use restrictions.

**Not settled.** "Derivative works" is not defined in the SAM License, and whether a model trained on another
model's outputs is a derivative work has not been settled by courts. This is our reading, not legal advice. If
this matters for your use, ask your own lawyer.

What still applies to us, the authors, because we used SAM 3D Body: research publications that use it must
acknowledge SAM Materials (Section 1.b.ii), and our own use of SAM 3D Body follows its terms.
The SAM License text is kept in `licenses/` for that reason; it is **not** the licence of this project's files.

## History

| Date | Change |
|---|---|
| 2026-11-18 (v0.1.0), 2026-11-20 (v0.2.0) | Weights and teacher labels released under the SAM License, as a conservative first choice. |
| 2026-11-21 (v0.3.0) | Weights (all versions, including the files first released in v0.1.0 and v0.2.0) and teacher labels relicensed to Apache-2.0, for the reasons above. COCO- and Open Images-derived parts unchanged (CC BY 4.0). Copies obtained earlier under the SAM License may still be used under it; the Apache-2.0 grant is offered in addition. |
