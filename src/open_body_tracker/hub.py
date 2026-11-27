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
    "v0.4.0-small": "obt-body-v0.4.0-small",            # default: ViT-S
    "v0.4.0-small-int8": "obt-body-v0.4.0-small-int8",  # the same weights quantized for CPUs (no fp16 file)
    "v0.4.0": "obt-body-v0.4.0",                        # ViT-B: more accurate, slower
    "v0.4.0-int8": "obt-body-v0.4.0-int8",
    "v0.3.0": "obt-body-v0.3.0",
    "v0.3.0-int8": "obt-body-v0.3.0-int8",  # the same weights quantized for CPUs (no fp16 file)
    "v0.2.0": "obt-body-v0.2.0",            # same weights as v0.3.0
    "v0.1.0": "obt-body-v0.1.0",
}
DEFAULT_MODEL = "v0.4.0-small"
# Names used before this rule. obt-body-v1/v2.* and obt-h3c.* (v0.1.0's first name; no logit outputs, so no flip test)
# stay on Hugging Face for old code.
ALIASES = {"v2": "v0.2.0", "v1": "v0.1.0"}

FILES = {
    "obt-body-v0.4.0-small.onnx": (HF + "obt-body-v0.4.0-small.onnx", "e7e835be37247c10acce3b534628741b9510c2d2e66bc67431d07123b0c80f48"),
    "obt-body-v0.4.0-small-fp16.onnx": (HF + "obt-body-v0.4.0-small-fp16.onnx", "cf4f1d71fe12737ea4e8a5cc15abf9308823e2514360b52a02648751db7a63f8"),
    "obt-body-v0.4.0-small-int8.onnx": (HF + "obt-body-v0.4.0-small-int8.onnx", "3ac1bf56c490a08be7f1207a80c3cd5f22193cab3a319825f4fbb12439bce223"),
    "obt-body-v0.4.0.onnx": (HF + "obt-body-v0.4.0.onnx", "e4aa4e65ce04f30268f53c2065bfb14cf07a278e14419a59ab3bb22cb75a1ce8"),
    "obt-body-v0.4.0-fp16.onnx": (HF + "obt-body-v0.4.0-fp16.onnx", "58d2a6a420c8ead1eec09d7a4ef10bc2ee8a1722e4310f4516ac97959caac2ce"),
    "obt-body-v0.4.0-int8.onnx": (HF + "obt-body-v0.4.0-int8.onnx", "efccf16f31cba81b65e9cde28f3e19397044ea7738dc348fd01f547272e7906c"),
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
    """A key of MODELS (e.g. "v0.4.0-small", "v0.4.0", "v0.3.0-int8") or the old "v2" / "v1" -> the ONNX file name,
    e.g. obt-body-v0.4.0-small.onnx."""
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
