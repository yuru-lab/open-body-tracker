"""Load the release weights (.safetensors) into BodyModel."""
from pathlib import Path

from .model import BodyModel, ModelConfig


def load_model(path, device="cpu"):
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), "pt") as f:
        config = ModelConfig.from_metadata(f.metadata())
    model = BodyModel(config=config)
    missing, unexpected = model.load_state_dict(load_file(Path(path)), strict=False)
    assert not unexpected and all(k.endswith("num_batches_tracked") for k in missing), (missing, unexpected)
    return model.to(device).eval().requires_grad_(False)
