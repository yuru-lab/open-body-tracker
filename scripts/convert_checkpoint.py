"""Convert a training checkpoint (.pt) into release weights (.safetensors, EMA weights only)."""
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from open_body_tracker.model import BodyModel, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw", action="store_true", help="use the raw weights instead of the EMA weights")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "ema" in saved:  # training/train.py
        state = saved["model" if args.raw else "ema"]
    else:  # research checkpoints of the release run
        state = {k.removeprefix("module."): v for k, v in saved["state_dict" if args.raw else "ema_state_dict"].items() if k != "steps"}
    state = {k: v for k, v in state.items() if not k.endswith("num_batches_tracked")}
    flags = saved.get("args", {})  # v0.4.0 training runs record backbone, grid, rigid and ray
    config = ModelConfig(flags.get("backbone", "b"), tuple(flags.get("grid") or (18, 14)), bool(flags.get("rigid")),
                         "ray" if flags.get("ray") else "camera") if "rigid" in flags else ModelConfig()
    model = BodyModel(config=config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all(k.endswith("num_batches_tracked") for k in missing), (missing, unexpected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items()}, args.output, metadata={"updates": str(saved.get("update", saved.get("updates", ""))), **(config.metadata() if config.rigid else {})})
    print(f"wrote {args.output} ({sum(v.numel() for v in state.values()) / 1e6:.1f} M values)")


if __name__ == "__main__":
    main()
