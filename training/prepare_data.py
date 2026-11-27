"""Download the training photos and cut the fixed 256x192 crops listed in the label files.

    uv run --extra train python training/prepare_data.py --labels labels --out data

Inputs (label files, not distributed; see docs/REPRODUCE.md):
    labels/coco2014.json      COCO 2014 crops (human keypoints) and their source photos
    labels/teacher.npz        Open Images V7 crops with SAM 3D Body teacher labels
    labels/openimages.jsonl   Open Images source photos (URL, SHA-256, licence, author)
Outputs:
    data/coco_crops.npy, data/teacher_crops.npy   uint8 (N, 256, 192, 3) RGB, in label order
    data/photos/...                                downloaded photos (can be deleted afterwards)
Every photo is checked against its SHA-256, and every crop against the SHA-256 of the crop used for the release.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request

import cv2
import numpy as np

SIZE = (192, 256)


def fetch(url, path, sha256, sources, retries=3):
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
        return path
    for source in sources:  # optional local mirrors, looked up by file name
        candidate = source / path.name
        if candidate.exists() and hashlib.sha256(candidate.read_bytes()).hexdigest() == sha256:
            return candidate
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "open-body-tracker"}), timeout=60).read()
            if hashlib.sha256(data).hexdigest() != sha256:
                raise ValueError("SHA-256 mismatch")
            partial = path.with_name(f"{path.name}.{threading.get_ident()}.part")
            partial.write_bytes(data)
            partial.replace(path)  # atomic: other threads never read a half-written photo
            return path
        except Exception as error:  # noqa: BLE001 - report after the last attempt
            if attempt == retries - 1:
                raise RuntimeError(f"{url}: {error}") from error
            time.sleep(2 ** attempt)


def cut(photo, affine, area_when_shrinking):
    image = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    matrix = np.asarray(affine, np.float64)
    flags = cv2.INTER_AREA if area_when_shrinking and matrix[0, 0] < 1 else cv2.INTER_LINEAR
    crop = cv2.warpAffine(image, matrix, SIZE, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def build(name, jobs, out, workers):
    """jobs: list of (photo_url, photo_path, photo_sha256, affine, crop_sha256, area_when_shrinking, sources)."""
    pack = np.lib.format.open_memmap(out / f"{name}_crops.npy", mode="w+", dtype=np.uint8, shape=(len(jobs), 256, 192, 3))
    matched = 0

    def one(index):
        url, path, sha, affine, crop_sha, area, sources = jobs[index]
        crop = cut(fetch(url, path, sha, sources), affine, area)
        pack[index] = crop
        return hashlib.sha256(crop.tobytes()).hexdigest() == crop_sha

    start = time.monotonic()
    with ThreadPoolExecutor(workers) as pool:
        for done, ok in enumerate(pool.map(one, range(len(jobs))), 1):
            matched += ok
            if done % 5000 == 0 or done == len(jobs):
                print(f"{name}: {done}/{len(jobs)} crops, {matched} identical to the release crops, {time.monotonic() - start:.0f} s", flush=True)
    pack.flush()
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=Path("labels"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only the first N crops of each set (for a quick check)")
    parser.add_argument("--mirror", type=Path, action="append", default=[],
                        help="directory that already holds some photos (looked up by file name); may be repeated")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    photos = args.out / "photos"

    coco = json.loads((args.labels / "coco2014.json").read_text())
    jobs = []
    for crop in coco["crops"][:args.limit]:
        image = coco["images"][crop["image"]]
        jobs.append((image["url"], photos / "coco" / image["file"], image["sha256"], crop["affine"], crop["crop_sha256"], False, args.mirror))
    coco_ok = build("coco", jobs, args.out, args.workers)

    meta = {row["key"]: row for row in map(json.loads, (args.labels / "openimages.jsonl").read_text().splitlines())}
    with np.load(args.labels / "teacher.npz") as labels:
        ids, affine, crop_sha = labels["id"][:args.limit], labels["affine"][:args.limit], labels["crop_sha256"][:args.limit]
    jobs = []
    for identifier, matrix, sha in zip(ids, affine, crop_sha):
        row = meta[str(identifier).rsplit("/", 1)[0]]
        jobs.append((row["url"], photos / "openimages" / f"{row['key']}.jpg", row["sha256"], matrix, sha, True, args.mirror))
    teacher_ok = build("teacher", jobs, args.out, args.workers)
    print(json.dumps(dict(coco=dict(crops=len(coco["crops"][:args.limit]), identical=coco_ok),
                          teacher=dict(crops=len(jobs), identical=teacher_ok))))


if __name__ == "__main__":
    main()
