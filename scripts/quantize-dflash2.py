#!/usr/bin/env python3
"""Convert the published BF16 DFlash2 companion to MLX affine Q4."""

import argparse
import json
import os
from pathlib import Path

import mlx.core as mx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--group-size", type=int, default=64, choices=(32, 64, 128))
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        parser.error("source and output must differ")
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")

    config_path = source / "config.json"
    weights_path = source / "model.safetensors"
    config = json.loads(config_path.read_text())
    if config.get("architectures") != ["DFlash2DraftModel"]:
        parser.error("source is not a DFlash2DraftModel checkpoint")
    if "quantization" in config:
        parser.error("source checkpoint is already quantized")

    weights = mx.load(str(weights_path))
    converted = {}
    quantized = 0
    for name, value in weights.items():
        if name.endswith(".weight") and value.ndim == 2:
            if value.shape[-1] % args.group_size:
                parser.error(f"{name} input width is not divisible by the group size")
            packed, scales, biases = mx.quantize(
                value, group_size=args.group_size, bits=4, mode="affine"
            )
            prefix = name.removesuffix(".weight")
            converted[name] = packed
            converted[f"{prefix}.scales"] = scales
            converted[f"{prefix}.biases"] = biases
            quantized += 1
        else:
            converted[name] = value

    if quantized != 47:
        parser.error(f"expected 47 DFlash2 projections, found {quantized}")

    output.mkdir(parents=True, exist_ok=True)
    temporary_weights = output / "model.incomplete.safetensors"
    mx.save_safetensors(
        str(temporary_weights), converted,
        metadata={"quantization": f"affine-q4-group{args.group_size}"},
    )
    os.replace(temporary_weights, output / "model.safetensors")
    config["quantization"] = {"group_size": args.group_size, "bits": 4}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Quantized {quantized} projections into {output}")


if __name__ == "__main__":
    main()
