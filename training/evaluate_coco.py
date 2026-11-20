"""COCO val2017 keypoint AP (17 body points) with ground-truth person boxes, as reported in MODEL_CARD.md.

    uv run --extra train python training/evaluate_coco.py --weights models/obt-body-v2.safetensors --coco data/coco2017 --download

"clean" excludes the 263 val2017 photos that also appear in the training archives (training/coco_val2017_exclude.json).
Protocol: every non-crowd person box, 25% margin, 256x192 crop, no flip test; detection score = mean keypoint score.
"""
import argparse
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import cv2
import numpy as np
import torch

from open_body_tracker.model import decode
from open_body_tracker.preprocess import crop, normalize
from open_body_tracker.weights import load_model

URLS = {"val2017": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"}


def download(coco):
    coco.mkdir(parents=True, exist_ok=True)
    if not (coco / "annotations/person_keypoints_val2017.json").exists():
        print("downloading", URLS["annotations"])
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URLS["annotations"]).read())).extract("annotations/person_keypoints_val2017.json", coco)
    if not (coco / "val2017").is_dir():
        print("downloading", URLS["val2017"])
        path = coco / "val2017.zip"
        urllib.request.urlretrieve(URLS["val2017"], path)
        zipfile.ZipFile(path).extractall(coco)
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--coco", type=Path, default=Path("data/coco2017"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if args.download:
        download(args.coco)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ground = COCO(str(args.coco / "annotations/person_keypoints_val2017.json"))
    persons = [a for a in ground.dataset["annotations"]
               if a["category_id"] == 1 and not a["iscrowd"] and a["bbox"][2] > 1 and a["bbox"][3] > 1]
    by_image = {}
    for annotation in persons:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    detections, batch, owners = [], [], []

    def flush():
        with torch.inference_mode():
            xy, score, _ = decode(model(torch.from_numpy(normalize(np.stack([b[0] for b in batch]))).to(device)))
        for (patch, inverse), image_id, points, scores in zip(batch, owners, xy.cpu().numpy().astype(np.float64), score.cpu().numpy()):
            points = points[:17] @ inverse[:, :2].T + inverse[:, 2]
            detections.append(dict(image_id=image_id, category_id=1, score=float(np.clip(scores[:17], 0, None).mean()),
                                   keypoints=np.concatenate([points, np.ones((17, 1))], 1).reshape(-1).tolist()))
        batch.clear()
        owners.clear()

    for count, image_id in enumerate(sorted(by_image)):
        rgb = cv2.cvtColor(cv2.imread(str(args.coco / "val2017" / ground.imgs[image_id]["file_name"])), cv2.COLOR_BGR2RGB)
        for annotation in by_image[image_id]:
            batch.append(crop(rgb, annotation["bbox"]))
            owners.append(image_id)
            if len(batch) == 64:
                flush()
        if count % 500 == 0:
            print(f"{count}/{len(by_image)} images", flush=True)
    if batch:
        flush()

    excluded = set(json.loads((Path(__file__).parent / "coco_val2017_exclude.json").read_text())["image_ids"])
    all_ids = sorted(img["id"] for img in ground.dataset["images"])
    prediction = ground.loadRes(detections)
    results = {}
    for subset, ids in (("clean", [i for i in all_ids if i not in excluded]), ("full", all_ids)):
        evaluator = COCOeval(ground, prediction, "keypoints")
        evaluator.params.imgIds = ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        results[subset] = dict(zip(["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"], map(float, evaluator.stats)))
    print(json.dumps({k: round(v["AP"] * 100, 1) for k, v in results.items()}))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
