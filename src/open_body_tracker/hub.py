"""Download the released models into ./models (pose model from Hugging Face, YOLOX from the official release)."""
import hashlib
from pathlib import Path
import urllib.request

HF_REPO = "open-body-tracker/open-body-tracker"  # TODO: set to the published Hugging Face repository
HF = f"https://huggingface.co/{HF_REPO}/resolve/main/"
YOLOX = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"

FILES = {
    "obt-h3c.onnx": (HF + "obt-h3c.onnx", None),
    "obt-h3c-fp16.onnx": (HF + "obt-h3c-fp16.onnx", None),
    "yolox_nano.onnx": (YOLOX + "yolox_nano.onnx", "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"),
    "yolox_tiny.onnx": (YOLOX + "yolox_tiny.onnx", "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"),
}


def fetch(name, directory="models"):
    path = Path(directory) / name
    url, digest = FILES[name]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)
    if digest and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"{path}: SHA-256 mismatch")
    return path
