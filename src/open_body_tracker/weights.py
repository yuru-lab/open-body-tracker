"""Load the release weights (.safetensors) into BodyModel."""
from pathlib import Path

from .model import BodyModel


def load_model(path, device="cpu"):
    from safetensors.torch import load_file
    model = BodyModel()
    missing, unexpected = model.load_state_dict(load_file(Path(path)), strict=False)
    assert not unexpected and all(k.endswith("num_batches_tracked") for k in missing), (missing, unexpected)
    return model.to(device).eval().requires_grad_(False)
