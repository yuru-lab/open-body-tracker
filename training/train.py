"""Train obt-h3c in two stages.

    # stage 1: 2D, 17 COCO points, from DINOv2 ViT-B/14 (about 11 h on an RTX 3060 Ti)
    uv run --extra train python training/train.py --stage 2d --out runs/stage1
    # stage 2: adds 6 foot points and metric 3D (about 4 h)
    uv run --extra train python training/train.py --stage 3d --init runs/stage1/final.pt --out runs/stage2
    # release weights
    uv run --extra train python scripts/convert_checkpoint.py runs/stage2/final.pt models/my-model.safetensors

Every update draws 32 crops: 16 ordinary + 4 hard COCO crops (human labels) and 12 Open Images crops (SAM 3D Body
teacher labels). Sampling and augmentation are keyed by (seed, update), so an interrupted run resumes exactly
(--resume). Only the EMA weights of the last update are kept for release; no checkpoint is selected by score.
"""
import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import urllib.request

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment import augment  # noqa: E402
from open_body_tracker.keypoints import FLIP  # noqa: E402
from open_body_tracker.model import CROP_HEIGHT, CROP_WIDTH, DEPTH, METRIC_BINS, METRIC_RANGE, SIMCC_SPLIT, BodyModel  # noqa: E402
from open_body_tracker.preprocess import normalize  # noqa: E402

DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
DINOV2_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
FLIP17 = FLIP[:17]
ANKLE_FOR_FOOT = [15, 15, 15, 16, 16, 16]
STAGES = {  # epochs of 471 updates, peak learning rate
    "2d": dict(epochs=90, peak_lr=5e-4),
    "3d": dict(epochs=30, peak_lr=2.5e-4),
}


# ---------------------------------------------------------------- data

class Data:
    """Crops (from prepare_data.py) and labels (from the label release), in label order."""

    def __init__(self, labels, data, stage):
        self.stage = stage
        coco = json.loads((labels / "coco2014.json").read_text())
        self.coco_xy = np.array([c["xy"] for c in coco["crops"]], np.float32)
        self.coco_valid = np.array([c["valid"] for c in coco["crops"]], bool)
        pools = [c["pool"] for c in coco["crops"]]
        self.pools = {name: [i for i, p in enumerate(pools) if p == name] for name in ("ordinary", "hard")}
        with np.load(labels / "teacher.npz") as teacher:
            self.teacher = {k: teacher[k] for k in ("xy", "valid", "xyz")}
        self.coco_crops = np.load(data / "coco_crops.npy", mmap_mode="r")
        self.teacher_crops = np.load(data / "teacher_crops.npy", mmap_mode="r")
        assert len(self.coco_crops) == len(self.coco_xy) and len(self.teacher_crops) == len(self.teacher["xy"])
        # One reference epoch = all human-labelled crops once (471 updates).
        self.epoch = (len(pools) + 31) // 32

    def sample(self, source, index):
        """-> (rgb, xy, valid, xyz or None) before augmentation."""
        if source == "teacher":
            rgb = np.array(self.teacher_crops[index])
            xy, valid, xyz = self.teacher["xy"][index], self.teacher["valid"][index], self.teacher["xyz"][index]
            return (rgb, xy[:17], valid[:17], None) if self.stage == "2d" else (rgb, xy, valid, xyz.copy())
        row = self.pools[source][index]
        rgb, xy, valid = np.array(self.coco_crops[row]), self.coco_xy[row], self.coco_valid[row]
        if self.stage == "2d":
            return rgb, xy, valid, None
        return (rgb, np.concatenate([xy, np.zeros((6, 2), np.float32)]), np.concatenate([valid, np.zeros(6, bool)]),
                np.zeros((23, 3), np.float32))

    def batch(self, seed, update):
        """16 ordinary + 4 hard + 12 teacher crops, keyed by (seed, update)."""
        stream = lambda number: np.random.default_rng(np.random.SeedSequence((seed, update, number)))  # noqa: E731
        slots = (("ordinary", stream(1).integers(len(self.pools["ordinary"]), size=16)),
                 ("hard", stream(2).integers(len(self.pools["hard"]), size=4)),
                 ("teacher", stream(7).integers(len(self.teacher_crops), size=12)))
        augment_seed = int(stream(4).integers(2 ** 32))
        random.seed(augment_seed)
        np.random.seed(augment_seed)
        flip, lower = (FLIP17, list(range(11, 17))) if self.stage == "2d" else (FLIP, list(range(11, 23)))
        items, has3d = [], []
        for source, indices in slots:
            for index in indices:
                rgb, xy, valid, xyz = self.sample(source, int(index))
                items.append(augment(rgb, xy, valid, flip, lower, xyz))
                has3d.append(source == "teacher")
        rgb, xy, valid = (np.stack([item[k] for item in items]) for k in range(3))
        xyz = None if self.stage == "2d" else np.stack([item[3] for item in items])
        return rgb, xy, valid, xyz, np.array(has3d)


# ---------------------------------------------------------------- targets and loss (MMPose SimCC / KLDiscretLoss)

def simcc_targets(xy, valid, sigma=(4.9, 5.66)):
    """Gaussian SimCC labels (not normalised) and weights; points whose label window leaves the crop get weight 0."""
    n, k, _ = xy.shape
    width, height = int(round(CROP_WIDTH * SIMCC_SPLIT)), int(round(CROP_HEIGHT * SIMCC_SPLIT))
    mu = np.around(xy * SIMCC_SPLIT).astype(np.int64)
    weights = valid.astype(np.float32).copy()
    tx, ty = np.zeros((n, k, width), np.float32), np.zeros((n, k, height), np.float32)
    x, y = np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    radius = np.array(sigma) * 3
    for i in range(n):
        for j in range(k):
            if weights[i, j] < 0.5:
                continue
            left, top = mu[i, j] - radius
            right, bottom = mu[i, j] + radius + 1
            if left >= width or top >= height or right < 0 or bottom < 0:
                weights[i, j] = 0
                continue
            tx[i, j] = np.exp(-((x - mu[i, j, 0]) ** 2) / (2 * sigma[0] ** 2))
            ty[i, j] = np.exp(-((y - mu[i, j, 1]) ** 2) / (2 * sigma[1] ** 2))
    return tx, ty, weights


CENTRES = np.linspace(-METRIC_RANGE, METRIC_RANGE, METRIC_BINS, dtype=np.float32)


def metric_targets(values, sigma=0.03):
    """values (N,K) metres -> Gaussian labels on the 300 fixed 1 cm bins."""
    clipped = np.clip(values, -METRIC_RANGE, METRIC_RANGE)[..., None]
    return np.exp(-(CENTRES - clipped) ** 2 / (2 * sigma ** 2)).astype(np.float32)


def kl_loss(predictions, targets, weights, label_beta=10.):
    """MMPose KLDiscretLoss(beta=1, label_softmax=True, label_beta=10): summed over the batch, averaged over keypoints."""
    keypoints = predictions[0].shape[1]
    total = 0
    for prediction, target in zip(predictions, targets):
        log_p = torch.log_softmax(prediction.reshape(-1, prediction.shape[-1]), dim=1)
        labels = torch.softmax(target.reshape(-1, target.shape[-1]) * label_beta, dim=1)
        loss = torch.nn.functional.kl_div(log_p, labels, reduction="none").mean(dim=1)
        total = total + (loss * weights.reshape(-1)).sum()
    return total / keypoints


# ---------------------------------------------------------------- optimisation

def layer_scale(name, top=0.1, decay=0.75):
    """Learning-rate multiplier: 1 outside the backbone, layer-wise decay inside it."""
    if not name.startswith("backbone."):
        return 1.
    part = name.split(".")
    layer = int(part[2]) + 1 if part[1] == "blocks" else DEPTH + 1 if part[1] == "norm" else 0
    return top * decay ** (DEPTH + 1 - layer)


def make_optimizer(model):
    groups = []
    for name, parameter in model.named_parameters():
        plain = parameter.ndim <= 1 or name in ("backbone.cls_token", "backbone.pos_embed")
        groups.append(dict(params=[parameter], weight_decay=0. if plain else .05, lr_scale=layer_scale(name), name=name))
    groups.sort(key=lambda group: -group["lr_scale"])
    return torch.optim.AdamW(groups, lr=5e-4, weight_decay=.05)


def learning_rate(index, total, peak, minimum, warmup=500):
    """Linear warm-up, constant until half, then cosine to the minimum at the last update."""
    if index < warmup:
        fraction = index / (warmup - 1)
        return peak * (1e-5 + (1. - 1e-5) * fraction)
    if index <= total // 2:
        return peak
    fraction = (index - total // 2) / (total - 1 - total // 2)
    return minimum + (peak - minimum) * (1. + math.cos(math.pi * fraction)) / 2.


class EMA:
    """MMPose ExpMomentumEMA(momentum=2e-4, gamma=2000) over all floating-point weights and buffers."""

    def __init__(self, model, momentum=2e-4, gamma=2000):
        import copy
        self.module = copy.deepcopy(model).requires_grad_(False)
        self.momentum, self.gamma, self.steps = momentum, gamma, 0

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for key, average in self.module.state_dict().items():
            if self.steps == 0:
                average.copy_(source[key])
            elif average.dtype.is_floating_point:
                m = (1 - self.momentum) * math.exp(-float(1 + self.steps) / self.gamma) + self.momentum
                average.mul_(1 - m).add_(source[key], alpha=m)
        self.steps += 1


# ---------------------------------------------------------------- model setup

def dinov2_weights(cache=Path("models")):
    path = cache / "dinov2_vitb14_pretrain.pth"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DINOV2_URL}")
        urllib.request.urlretrieve(DINOV2_URL, path.with_suffix(".part"))
        path.with_suffix(".part").rename(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == DINOV2_SHA256, "unexpected DINOv2 checkpoint"
    return torch.load(path, map_location="cpu", weights_only=True)


def build_model(stage, init, seed):
    if stage == "2d":
        model = BodyModel(keypoints=17, metric=False, checkpointing=True)
        model.backbone.load_dinov2(dinov2_weights())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model.neck = type(model.neck)()
            model.head = type(model.head)(17, False)
            model.head.init_weights()
        return model
    model = BodyModel(keypoints=23, metric=True, checkpointing=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        model.head = type(model.head)(23, True)
        model.head.init_weights()
    state = torch.load(init, map_location="cpu", weights_only=True)["ema"]
    own = model.state_dict()
    for key, value in state.items():
        if value.shape != own[key].shape:  # 17 -> 23 rows: feet start as copies of the ankles
            assert key.startswith(("head.final_layer.", "head.final_layer2.")) and value.shape[0] == 17
            value = torch.cat([value, value[ANKLE_FOR_FOOT]], 0)
        own[key] = value
    missing = set(own) - set(state)
    assert missing == {f"head.cls_{axis}.weight" for axis in "XYZ"}, missing
    model.load_state_dict(own)
    return model


# ---------------------------------------------------------------- training

def train_step(model, ema, optimizer, batch, stage, lr, metric_weight=1.):
    rgb, xy, valid, xyz, has3d = batch
    tx, ty, weights = simcc_targets(xy, valid)
    device = next(model.parameters()).device
    tensor = partial(torch.as_tensor, device=device)
    for group in optimizer.param_groups:
        group["lr"] = lr * group["lr_scale"]
    model.train()
    optimizer.zero_grad()
    outputs = model(tensor(normalize(rgb)))
    loss2d = kl_loss(outputs[:2], (tensor(tx), tensor(ty)), tensor(weights))
    loss3d = torch.zeros((), device=device)
    if stage == "3d":
        labels = [tensor(metric_targets(xyz[..., axis])) for axis in range(3)]
        mask = tensor(np.repeat(has3d.astype(np.float32)[:, None], xyz.shape[1], 1))
        loss3d = kl_loss(outputs[2:], labels, mask)
    loss = loss2d + metric_weight * loss3d
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 35, error_if_nonfinite=True)
    optimizer.step()
    ema.update(model)
    return float(loss2d.detach()), float(loss3d.detach())


def save(path, model, ema, optimizer, update, args):
    partial_path = path.with_suffix(".partial")
    torch.save(dict(model=model.state_dict(), ema=ema.module.state_dict(), ema_steps=ema.steps,
                    optimizer=optimizer.state_dict(), update=update, args=vars(args) | dict(init=str(args.init), labels=str(args.labels),
                                                                                             data=str(args.data), out=str(args.out))),
               partial_path)
    partial_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--init", type=Path, help="stage-1 final.pt (required for --stage 3d)")
    parser.add_argument("--labels", type=Path, default=Path("labels"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int, help="stop early (for a smoke test)")
    args = parser.parse_args()
    assert args.stage == "2d" or args.init, "--stage 3d needs --init"

    # Deterministic kernels: the same machine and versions give the same run.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(1)

    data = Data(args.labels, args.data, args.stage)
    total = data.epoch * STAGES[args.stage]["epochs"]
    peak = STAGES[args.stage]["peak_lr"]
    minimum = peak / 20
    model = build_model(args.stage, args.init, args.seed).cuda()
    ema = EMA(model)
    optimizer = make_optimizer(model)
    first = 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume and (args.out / "latest.pt").exists():
        state = torch.load(args.out / "latest.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        ema.module.load_state_dict(state["ema"])
        ema.steps = state["ema_steps"]
        optimizer.load_state_dict(state["optimizer"])
        first = state["update"]
        print(f"resumed at update {first}")
    stop = min(total, args.max_updates or total)
    print(f"stage {args.stage}: {total} updates ({data.epoch} per epoch), {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters")
    start = time.monotonic()
    log = (args.out / "log.jsonl").open("a")
    for index in range(first, stop):
        lr = learning_rate(index, total, peak, minimum)
        loss2d, loss3d = train_step(model, ema, optimizer, data.batch(args.seed, index), args.stage, lr)
        if not (math.isfinite(loss2d) and math.isfinite(loss3d)):
            raise FloatingPointError(f"non-finite loss at update {index + 1}")
        update = index + 1
        log.write(json.dumps(dict(update=update, loss2d=loss2d, loss3d=loss3d, lr=lr)) + "\n")
        if update % 100 == 0:
            log.flush()
            rate = (update - first) / (time.monotonic() - start)
            print(f"update {update}/{total}  loss2d {loss2d:.4f}  loss3d {loss3d:.4f}  lr {lr:.2e}  "
                  f"{rate:.2f} it/s  eta {(stop - update) / rate / 3600:.1f} h", flush=True)
        if update % 1000 == 0 and update < total:
            save(args.out / "latest.pt", model, ema, optimizer, update, args)
    if stop == total:
        save(args.out / "final.pt", model, ema, optimizer, total, args)
        print(f"done: {args.out / 'final.pt'}")
    else:
        save(args.out / "latest.pt", model, ema, optimizer, stop, args)


if __name__ == "__main__":
    main()
