"""Download the released models into ./models (pose model from Hugging Face, YOLOX from the official release)."""
import hashlib
from pathlib import Path
import urllib.request

HF_REPO = "Yurup/open-body-tracker"
HF = f"https://huggingface.co/{HF_REPO}/resolve/main/"
YOLOX = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"

# Pose models. Every release names its models after its own tag (obt-body-<tag>[-small][-int8]), even when the weights
# did not change; docs/MODELS.md lists which releases share weights.
MODELS = {
    "v0.3.0": "obt-body-v0.3.0",            # default
    "v0.3.0-int8": "obt-body-v0.3.0-int8",  # the same weights quantized for CPUs (no fp16 file)
    "v0.2.0": "obt-body-v0.2.0",            # same weights as v0.3.0
    "v0.1.0": "obt-body-v0.1.0",
}
DEFAULT_MODEL = "v0.3.0"
# Names used before this rule. obt-body-v1/v2.* and obt-h3c.* (v0.1.0's first name; no logit outputs, so no flip test)
# stay on Hugging Face for old code.
ALIASES = {"v2": "v0.2.0", "v1": "v0.1.0"}

FILES = {
    "obt-body-v0.3.0.onnx": (HF + "obt-body-v0.3.0.onnx", "e046ec966cc357c7db37319721978563f0f6319e4e5e4d40fb55d4452aee0096"),
    "obt-body-v0.3.0-fp16.onnx": (HF + "obt-body-v0.3.0-fp16.onnx", "1cee6d22621a3bd9f26112fa4257e993dc92c4bfd1dc84c986a6ce0849554b7b"),
    "obt-body-v0.3.0-int8.onnx": (HF + "obt-body-v0.3.0-int8.onnx", "e79f32b650a4362e21934d824cf802899e799ac5d2070a3582f925eed4b71819"),
    "obt-body-v0.2.0.onnx": (HF + "obt-body-v0.2.0.onnx", "e046ec966cc357c7db37319721978563f0f6319e4e5e4d40fb55d4452aee0096"),
    "obt-body-v0.2.0-fp16.onnx": (HF + "obt-body-v0.2.0-fp16.onnx", "1cee6d22621a3bd9f26112fa4257e993dc92c4bfd1dc84c986a6ce0849554b7b"),
    "obt-body-v0.1.0.onnx": (HF + "obt-body-v0.1.0.onnx", "93b4c024df6933b3d62f01884aa1bee0e730f26686c4bb43e9b552dc669098f3"),
    "obt-body-v0.1.0-fp16.onnx": (HF + "obt-body-v0.1.0-fp16.onnx", "5ea98e361d182b25f0bd28847fa387c67e1757aca2e9b62b216019f5f41d7acb"),
    "obt-body-v2.onnx": (HF + "obt-body-v2.onnx", "e046ec966cc357c7db37319721978563f0f6319e4e5e4d40fb55d4452aee0096"),
    "obt-body-v2-fp16.onnx": (HF + "obt-body-v2-fp16.onnx", "1cee6d22621a3bd9f26112fa4257e993dc92c4bfd1dc84c986a6ce0849554b7b"),
    "obt-body-v1.onnx": (HF + "obt-body-v1.onnx", "93b4c024df6933b3d62f01884aa1bee0e730f26686c4bb43e9b552dc669098f3"),
    "obt-body-v1-fp16.onnx": (HF + "obt-body-v1-fp16.onnx", "5ea98e361d182b25f0bd28847fa387c67e1757aca2e9b62b216019f5f41d7acb"),
    "obt-h3c.onnx": (HF + "obt-h3c.onnx", "f71d29b5bea52fc4a853f2aff626dd81a4cffae8374b84ea349cf79703e8a57d"),
    "obt-h3c-fp16.onnx": (HF + "obt-h3c-fp16.onnx", "21ab7573354a21e2b65986bd5c93be2ee2b663bcd29eca2ad2c7e1a76bda265b"),
    "yolox_nano.onnx": (YOLOX + "yolox_nano.onnx", "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"),
    "yolox_tiny.onnx": (YOLOX + "yolox_tiny.onnx", "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"),
}


def model_file(version=DEFAULT_MODEL, fp16=False):
    """"v0.3.0" / "v0.3.0-int8" / "v0.2.0" / "v0.1.0" (or the old "v2" / "v1") -> the ONNX file name, e.g. obt-body-v0.3.0.onnx."""
    version = ALIASES.get(version, version)
    if version not in MODELS:
        raise ValueError(f"unknown model version {version!r}; choose from {', '.join(MODELS)}")
    if fp16 and version.endswith("-int8"):
        raise ValueError("the INT8 model has no fp16 file")
    return MODELS[version] + ("-fp16" if fp16 else "") + ".onnx"


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
