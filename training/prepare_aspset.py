"""Prepare the ASPset-510 (CC0) crops and 3D labels used by obt-body-v2 (training/train.py --stage 3d --aspset).

    uv run python training/prepare_aspset.py --out data            # downloads ~24 GB from the Internet Archive
    uv run python training/prepare_aspset.py --out data --archives /path/with/aspset510_v1_trainval-*.tar.gz

Only the 7 training actors listed in training/aspset_train.json are read (one camera per recording). Frames are
decoded with OpenCV and cut into 256x192 crops around ASPset's own person boxes; the 3D labels are ASPset's
multi-view joints, moved into the camera frame and centred on the hip midpoint (body 12 points only). Each recording's
crops and labels are checked against the SHA-256 of the release run (OpenCV 5.0.0, pinned in pyproject.toml).

Writes data/aspset_crops.npy (43,201 x 256 x 192 x 3 uint8, ~6.4 GB) and data/aspset_xyz.npy (43,201 x 23 x 3).
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import tarfile
import tempfile
import urllib.request

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE_URL = "https://archive.org/download/aspset510/"
ARCHIVES = {  # name -> MD5 published by the Internet Archive
    "aspset510_v1_trainval-joints_3d.tar.gz": "f848dc688c3d8536063515fff5349b98",
    "aspset510_v1_trainval-cameras.tar.gz": "1dca82253524fa6f379616851c8e9356",
    "aspset510_v1_trainval-boxes.tar.gz": "9816529acd1bacf21ad527412251e21e",
    "aspset510_v1_trainval-videos.tar.gz": "917a7b4724d7fb8e1db80783466e339d",
}
PREFIX = "ASPset-510/trainval"
C3D_LABELS = ["right_ankle", "right_knee", "right_hip", "right_wrist", "right_elbow", "right_shoulder",
              "left_ankle", "left_knee", "left_hip", "left_wrist", "left_elbow", "left_shoulder",
              "head_top", "head", "neck", "spine", "pelvis"]
# ASPset joint for each of the COCO body points 5..16 (shoulders, elbows, wrists, hips, knees, ankles)
MAPPING = [11, 5, 10, 4, 9, 3, 8, 2, 7, 1, 6, 0]
BODY12 = list(range(5, 17))
BODY12_IN_BODY12 = [6, 7]  # hips inside the body-12 slice
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def archive(directory, name):
    path = directory / name
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"downloading {ARCHIVE_URL + name}")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(ARCHIVE_URL + name, partial)
        partial.rename(path)
    if md5(path) != ARCHIVES[name]:
        raise RuntimeError(f"{path}: MD5 mismatch")
    return path


def read_members(path, wanted):
    with tarfile.open(path) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar if m.isfile() and m.name in wanted}


def c3d_points(raw):
    """ASPset's C3D files: 17 points, float, 50 Hz, millimetres. -> (frames, 17, 3) metres, (frames, 17) valid."""
    block, magic, points, analog, first, last, _, scale, data_start, _, rate = struct.unpack_from("<BBHHHHHfHHf", raw)
    assert magic == 80 and points == 17 and analog == 0 and first == 1 and scale == -1 and rate == 50
    groups, parameters, pos = {}, {}, (block - 1) * 512 + 4
    while raw[pos]:
        length, group = struct.unpack_from("<bb", raw, pos)
        pos += 2
        name = raw[pos:pos + abs(length)].decode("ascii")
        pos += abs(length)
        offset = struct.unpack_from("<h", raw, pos)[0]
        end = pos + offset
        pos += 2
        if group < 0:
            groups[-group] = name
        else:
            dtype, ndim = struct.unpack_from("<bB", raw, pos)
            pos += 2
            dims = list(raw[pos:pos + ndim])
            pos += ndim
            parameters[(group, name)] = (dtype, dims, raw[pos:pos + abs(dtype) * int(np.prod(dims))])
        pos = end
    point = next(k for k, v in groups.items() if v == "POINT")
    _, dims, labels = parameters[(point, "LABELS")]
    assert [labels[i * dims[0]:(i + 1) * dims[0]].decode().strip() for i in range(17)] == C3D_LABELS
    assert parameters[(point, "UNITS")][2].decode().strip() == "mm"
    values = np.frombuffer(raw, dtype="<f4", count=last * 17 * 4, offset=(data_start - 1) * 512).reshape(last, 17, 4).copy()
    return values[:, :, :3] / 1000, values[:, :, 3] >= 0


def labels(files, actor, recording, camera):
    """-> camera-frame, hip-centred COCO-17 layout xyz (frames, 17, 3) float32, valid, ASPset boxes (x1, y1, x2, y2)."""
    xyz, valid = c3d_points(files[f"{PREFIX}/joints_3d/{actor}/{recording}.c3d"])
    calibration = json.loads(files[f"{PREFIX}/cameras/{actor}/{actor}-{camera}.json"])
    extrinsic = np.array(calibration["extrinsic_matrix"]).reshape(4, 4)
    homogeneous = np.concatenate((xyz * 1000, np.ones((*xyz.shape[:2], 1))), -1)
    camera_xyz = (homogeneous @ extrinsic.T)[:, :, :3] / 1000
    target, mask = np.zeros_like(camera_xyz), np.zeros_like(valid)
    target[:, 5:], mask[:, 5:] = camera_xyz[:, MAPPING], valid[:, MAPPING]
    target = target - target[..., 11:13, :].mean(-2, keepdims=True)
    boxes = np.loadtxt(io.BytesIO(files[f"{PREFIX}/boxes/{actor}/{recording}-{camera}.csv"]), delimiter=",", skiprows=1)
    assert len(boxes) == len(target)
    return target.astype(np.float32), mask, boxes


def crop_affine(box):
    """xyxy box -> 2x3 affine to the 256x192 crop (centre, 192:256 aspect, 25% margin), float64 as in the release run."""
    box = np.asarray(box, dtype=np.float64)
    centre = (box[:2] + box[2:]) / 2
    width = max(box[2] - box[0], (box[3] - box[1]) * 192 / 256) * 1.25
    height = width * 256 / 192
    scale = 192 / width
    return np.array([[scale, 0, (width / 2 - centre[0]) * scale], [0, scale, (height / 2 - centre[1]) * scale]], np.float64)


def training_xyz(target):
    """The 23-point 3D label used in training: body 12 re-centred on the hips in float64, other points zero."""
    full = target.astype(np.float64)
    body = full[:, BODY12]
    full[:, BODY12] = body - body[:, BODY12_IN_BODY12].mean(1, keepdims=True)
    out = np.zeros((len(target), 23, 3), np.float32)
    out[:, BODY12] = full.astype(np.float32)[:, BODY12]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--archives", type=Path, default=Path("data/aspset-archives"),
                        help="where the ASPset .tar.gz files are (downloaded there if missing)")
    parser.add_argument("--videos", type=Path, help="a folder that already holds the <recording>-<camera>.mkv files")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    spec = json.loads((HERE / "aspset_train.json").read_text())
    records = spec["recordings"]
    wanted = {f"{PREFIX}/joints_3d/{r['actor']}/{r['recording']}.c3d" for r in records}
    wanted |= {f"{PREFIX}/cameras/{r['actor']}/{r['actor']}-{r['camera']}.json" for r in records}
    wanted |= {f"{PREFIX}/boxes/{r['actor']}/{r['recording']}-{r['camera']}.csv" for r in records}
    files = {}
    for name in ARCHIVES:
        if name.endswith("videos.tar.gz"):
            continue
        files |= read_members(archive(args.archives, name), wanted)
    assert wanted <= files.keys()

    videos, scratch = args.videos, None
    if videos is None:
        scratch = Path(tempfile.mkdtemp(dir=args.out if args.out.exists() else None))
        videos = scratch
        members = {f"{PREFIX}/videos/{r['actor']}/{r['recording']}-{r['camera']}.mkv" for r in records}
        print("extracting the 209 training videos (streams through the 24 GB archive once)")
        with tarfile.open(archive(args.archives, "aspset510_v1_trainval-videos.tar.gz"), "r|gz") as tar:
            for member in tar:
                if member.isfile() and member.name in members:
                    with tar.extractfile(member) as source, open(videos / Path(member.name).name, "wb") as target:
                        shutil.copyfileobj(source, target)

    args.out.mkdir(parents=True, exist_ok=True)
    crops = np.lib.format.open_memmap(args.out / "aspset_crops.npy.part", mode="w+", dtype=np.uint8,
                                      shape=(spec["frames"], 256, 192, 3))
    xyz_all = np.zeros((spec["frames"], 23, 3), np.float32)
    whole_crops, whole_xyz, row = hashlib.sha256(), hashlib.sha256(), 0
    for number, r in enumerate(records, 1):
        target, mask, boxes = labels(files, r["actor"], r["recording"], r["camera"])
        assert len(target) == r["frames"] and mask[:, BODY12].all()
        capture = cv2.VideoCapture(str(videos / f"{r['recording']}-{r['camera']}.mkv"))
        assert capture.isOpened()
        part = np.empty((r["frames"], 256, 192, 3), np.uint8)
        for index in range(r["frames"]):
            ok, image = capture.read()
            assert ok and image.shape == (2160, 3840, 3)
            part[index] = cv2.cvtColor(cv2.warpAffine(image, crop_affine(boxes[index]), (192, 256)), cv2.COLOR_BGR2RGB)
        assert not capture.read()[0], "extra video frames"
        capture.release()
        xyz = training_xyz(target)
        for value, key in ((part, "crops_sha256"), (xyz, "xyz_sha256")):
            if hashlib.sha256(value.tobytes()).hexdigest() != r[key]:
                raise RuntimeError(f"{r['recording']}-{r['camera']}: {key} differs from the release run")
        crops[row:row + len(part)] = part
        xyz_all[row:row + len(part)] = xyz
        whole_crops.update(part.tobytes())
        whole_xyz.update(xyz.tobytes())
        row += len(part)
        print(f"{number}/{len(records)} {r['recording']}-{r['camera']}: {len(part)} frames, identical", flush=True)
    assert row == spec["frames"]
    assert whole_crops.hexdigest() == spec["crops_sha256"] and whole_xyz.hexdigest() == spec["xyz_sha256"]
    crops.flush()
    del crops
    (args.out / "aspset_crops.npy.part").rename(args.out / "aspset_crops.npy")
    np.save(args.out / "aspset_xyz.npy", xyz_all)
    if scratch is not None:
        shutil.rmtree(scratch)
    print(f"wrote {args.out / 'aspset_crops.npy'} and {args.out / 'aspset_xyz.npy'} ({row} frames, all bit-identical)")


if __name__ == "__main__":
    main()
