"""Webcam or image demo: 2D skeleton on the frame and a rotating 3D view next to it.

    uv run obt-demo                      # webcam 0, YOLOX-nano, CPU
    uv run obt-demo --detector tiny --gpu
    uv run obt-demo --image person.jpg   # writes person.pose.jpg
    uv run obt-demo --model-version v0.3.0-int8   # about 2x faster on CPUs; v0.1.0 = the first model
    uv run obt-demo --flip                         # flip test (about 2x slower, a little steadier)
    uv run obt-demo --anchor                       # 3D origin follows the visible body (shoulders-up framing)
"""
import argparse
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .detector import PersonDetector
from .hub import ALIASES, DEFAULT_MODEL, MODELS, fetch, model_file
from .keypoints import EDGES
from .pipeline import PosePipeline

LEFT, RIGHT, CENTRE = (235, 140, 40), (40, 140, 245), (200, 200, 200)  # BGR: blue = left, orange = right
DRAW_SCORE = 3.0  # joints in view score about 6, joints out of view under 2


def colour(a, b):
    left = {1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19}
    right = {2, 4, 6, 8, 10, 12, 14, 16, 20, 21, 22}
    return LEFT if a in left and b in left else RIGHT if a in right and b in right else CENTRE


def draw_2d(frame, pose):
    x, y, w, h = map(int, pose["box"])
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 200, 80) if pose["box_source"] == "tracker" else (0, 220, 255), 1)
    for a, b in EDGES:
        if min(pose["score"][a], pose["score"][b]) >= DRAW_SCORE:
            cv2.line(frame, tuple(map(int, pose["xy"][a])), tuple(map(int, pose["xy"][b])), colour(a, b), 2, cv2.LINE_AA)


def draw_3d(size, pose, angle):
    view = np.full((size, size, 3), 24, np.uint8)
    c, s = math.cos(angle), math.sin(angle)
    xyz = pose["xyz"] @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], np.float32).T  # rotate about the vertical axis
    points = (xyz[:, :2] * size / 2.4 + size / 2).astype(int)  # 2.4 m across the view; y points down (camera frame)
    cv2.line(view, (0, int(size / 2 + 1.0 * size / 2.4)), (size, int(size / 2 + 1.0 * size / 2.4)), (60, 60, 60), 1)
    for a, b in EDGES:
        if a >= 5 and b >= 5:  # body and feet only
            cv2.line(view, tuple(points[a]), tuple(points[b]), colour(a, b), 2, cv2.LINE_AA)
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--detector", choices=["nano", "tiny"], default="nano")
    parser.add_argument("--gpu", action="store_true", help="run the pose model with CUDA (needs onnxruntime-gpu)")
    parser.add_argument("--no-track", action="store_true", help="run the detector on every frame")
    parser.add_argument("--model-version", choices=[*MODELS, *ALIASES], default=DEFAULT_MODEL,
                        help=f"pose model to download (default {DEFAULT_MODEL}; see docs/MODELS.md)")
    parser.add_argument("--model", type=Path, help="pose model ONNX file (overrides --model-version)")
    parser.add_argument("--flip", action="store_true", help="flip test: a little more accurate and steadier, about 2x slower")
    parser.add_argument("--anchor", action="store_true",
                        help="3D origin follows the visible body (hips, else shoulders, else face): steadier when only "
                             "the upper body is in view")
    args = parser.parse_args()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"]
    detector = PersonDetector(fetch(f"yolox_{args.detector}.onnx"))
    pipeline = PosePipeline(args.model or fetch(model_file(args.model_version)), detector, providers,
                            track=not args.no_track and not args.image, flip=args.flip, anchor=args.anchor)

    if args.image:
        frame = cv2.imread(str(args.image))
        pose = pipeline(frame[:, :, ::-1].copy())
        if pose is None:
            raise SystemExit("no person found")
        draw_2d(frame, pose)
        side = cv2.resize(draw_3d(480, pose, math.radians(30)), (frame.shape[0], frame.shape[0]))
        output = args.image.with_suffix(".pose.jpg")
        cv2.imwrite(str(output), np.hstack([frame, side]))
        print(f"wrote {output}")
        return

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    angle, last = math.radians(30), time.monotonic()
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        pose = pipeline(frame[:, :, ::-1].copy())
        now = time.monotonic()
        fps, last = 1 / max(now - last, 1e-6), now
        side = np.full((frame.shape[0], frame.shape[0], 3), 24, np.uint8)
        if pose is not None:
            draw_2d(frame, pose)
            side = draw_3d(frame.shape[0], pose, angle)
        cv2.putText(frame, f"{fps:.0f} FPS  [q] quit  [a/d] rotate", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("open-body-tracker", np.hstack([frame, side]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        angle += {ord("a"): -0.1, ord("d"): 0.1}.get(key, 0)
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
