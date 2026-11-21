"""Make the CPU INT8 model (obt-body-v0.3.0-int8.onnx) from obt-body-v0.3.0.onnx.

    uv run python scripts/quantize_onnx.py models/obt-body-v0.3.0.onnx models/obt-body-v0.3.0-int8.onnx

Dynamic INT8 quantization of the weight MatMuls (weights int8 per output channel with reduce_range, activations
uint8 per batch). Kept in float32:
- the output head (the 2D and 3D classifiers are sensitive and cheap),
- the second MLP layer of backbone blocks 8 and 11, whose inputs carry very large DINOv2 activations
  (block 8 alone moves the 3D output by ~6 cm when quantized).
reduce_range keeps the int8 products from saturating on CPUs without VNNI.
"""
import argparse
import hashlib
from pathlib import Path
import tempfile

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

FLOAT_BLOCKS = (8, 11)


def float_nodes(model):
    names = [n.name for n in model.graph.node if n.op_type == "MatMul"]
    keep = [n for n in names if "/head/" in n]
    keep += [n for n in names for b in FLOAT_BLOCKS if f"/blocks.{b}/mlp/fc2/" in n]
    assert len(keep) == len([n for n in names if "/head/" in n]) + len(FLOAT_BLOCKS), "unexpected graph"
    return keep


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        prepared = Path(scratch) / "prepared.onnx"
        quant_pre_process(str(args.source), str(prepared), skip_symbolic_shape=True)
        quantize_dynamic(prepared, args.target, weight_type=QuantType.QInt8, per_channel=True, reduce_range=True,
                         op_types_to_quantize=["MatMul"], nodes_to_exclude=float_nodes(onnx.load(prepared)))
    print(args.target, hashlib.sha256(args.target.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
