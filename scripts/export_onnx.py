"""Export the release weights to ONNX with preprocessing and decoding inside the graph.

Input  "crop":  float32 (N, 3, 256, 192), RGB 0-255 (a person crop, see open_body_tracker.preprocess.crop)
Outputs "xy":   float32 (N, 23, 2)  keypoints in crop pixels
        "score": float32 (N, 23)    2D confidence (SimCC peak logit)
        "xyz":  float32 (N, 23, 3)  metres relative to the model's root; subtract the hip midpoint for display
        "logits_x", "logits_y", "logits_X", "logits_Y", "logits_Z": the raw classifier outputs (N, 23, bins), used
        by the flip test (open_body_tracker.flip), which averages them with those of the mirrored crop
"""
import argparse
from pathlib import Path

import torch
from torch import nn

from open_body_tracker.flip import LOGITS
from open_body_tracker.model import decode
from open_body_tracker.preprocess import MEAN, STD
from open_body_tracker.weights import load_model


class Exported(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, crop):
        logits = self.model((crop - self.mean) / self.std)
        return (*decode(logits), *logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fp16", action="store_true", help="store weights in float16 (inputs/outputs stay float32)")
    args = parser.parse_args()
    model = Exported(load_model(args.weights))
    example = torch.rand(1, 3, 256, 192) * 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = ["xy", "score", "xyz", *LOGITS]
    torch.onnx.export(model, (example,), args.output, input_names=["crop"], output_names=outputs,
                      dynamic_axes={name: {0: "n"} for name in ["crop", *outputs]},
                      opset_version=17, dynamo=False)
    if args.fp16:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
        converted = convert_float_to_float16(onnx.load(str(args.output)), keep_io_types=True)
        onnx.save(converted, str(args.output))
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
