"""Export the release weights to ONNX with preprocessing and decoding inside the graph.

Input  "crop":  float32 (N, 3, 256, 192), RGB 0-255 (a person crop, see open_body_tracker.preprocess.crop)
Outputs "xy":   float32 (N, 23, 2)  keypoints in crop pixels
        "score": float32 (N, 23)    2D confidence (SimCC peak logit)
        "xyz":  float32 (N, 23, 3)  metres relative to the model's root; subtract the hip midpoint for display
                                    (v0.4.0: hip midpoint at 0, in the crop's ray frame, see open_body_tracker.geometry)
        "logits_x", "logits_y", "logits_X", "logits_Y", "logits_Z": the raw classifier outputs (N, 23, bins), used
        by the flip test of models up to v0.3.0 (open_body_tracker.flip), which averages them with those of the
        mirrored crop; v0.4.0 models average the decoded outputs instead (their 3D goes through the rigid skeleton)
Metadata (v0.4.0): obt.frame = "ray", obt.flip = "outputs", and the model configuration.
Hand models (v0.5.0) add to this graph
        "hand_xy":    float32 (N, 2, 21, 2) rough hand points in crop pixels (right hand, then left), for placing the
                      hand crops
        "hand_score": float32 (N, 2, 21)
and --hand-output writes the hand pass as a second graph (the same backbone weights, stored again):
Input  "crop":  float32 (N, 3, 140, 140), RGB 0-255, a RIGHT hand crop (mirror a left hand crop first, see
                open_body_tracker.hands)
Outputs "xy" (N, 21, 2) hand-crop pixels, "score" (N, 21), "xyz" (N, 21, 3) metres relative to the wrist (last point)
        in the hand crop's ray frame. Metadata: obt.frame = "ray", obt.hand = "right".
"""
import argparse
from pathlib import Path

import torch
from torch import nn

from open_body_tracker.flip import LOGITS
from open_body_tracker.model import HAND_INPUT, decode_hand
from open_body_tracker.preprocess import MEAN, STD
from open_body_tracker.weights import load_model


class Exported(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, crop):
        outputs = self.model((crop - self.mean) / self.std)
        logits = tuple(o[:, :self.model.body_points] for o in outputs[:5])
        hands = self.model.hand_points(outputs) if self.model.config.hands else ()
        return (*self.model.decode(outputs), *logits, *hands)


class ExportedHand(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, crop):
        return decode_hand(self.model.hands((crop - self.mean) / self.std))


def to_fp16(path):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    onnx.save(convert_float_to_float16(onnx.load(str(path)), keep_io_types=True), str(path))


def add_metadata(path, meta):
    import onnx
    proto = onnx.load(str(path))
    for key, value in meta.items():
        proto.metadata_props.add(key=key, value=value)
    onnx.save(proto, str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fp16", action="store_true", help="store weights in float16 (inputs/outputs stay float32)")
    parser.add_argument("--hand-output", type=Path, help="hand models: also write the hand pass graph here")
    args = parser.parse_args()
    model = Exported(load_model(args.weights))
    config = model.model.config
    example = torch.rand(1, 3, 256, 192) * 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = ["xy", "score", "xyz", *LOGITS, *(["hand_xy", "hand_score"] if config.hands else [])]
    torch.onnx.export(model, (example,), args.output, input_names=["crop"], output_names=outputs,
                      dynamic_axes={name: {0: "n"} for name in ["crop", *outputs]},
                      opset_version=17, dynamo=False)
    if config.rigid:
        add_metadata(args.output, {"obt.frame": config.frame, "obt.flip": "outputs",
                                   **{f"obt.{k}": v for k, v in config.metadata().items()}})
    if args.fp16:
        to_fp16(args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.0f} MB)")
    if args.hand_output:
        if not config.hands:
            raise SystemExit(f"{args.weights} has no hand pass")
        hand = ExportedHand(model.model)
        names = ["xy", "score", "xyz"]
        torch.onnx.export(hand, (torch.rand(1, 3, HAND_INPUT, HAND_INPUT) * 255,), args.hand_output, input_names=["crop"],
                          output_names=names, dynamic_axes={name: {0: "n"} for name in ["crop", *names]},
                          opset_version=17, dynamo=False)
        add_metadata(args.hand_output, {"obt.frame": "ray", "obt.hand": "right", "obt.input": str(HAND_INPUT)})
        if args.fp16:
            to_fp16(args.hand_output)
        print(f"wrote {args.hand_output} ({args.hand_output.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
