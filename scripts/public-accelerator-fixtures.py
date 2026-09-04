#!/usr/bin/env python3
"""Generate small, untrained Core ML fixtures; never download model weights."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=128, choices=[128, 2048])
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if output == repository or repository in output.parents:
        parser.error("--output must be outside the repository")
    output.mkdir(parents=True, exist_ok=True)

    # Keep converter scratch outside the checkout and remove it when finished.
    with tempfile.TemporaryDirectory(prefix="public-accelerator-") as scratch:
        tempfile.tempdir = scratch
        generate(output, args.tokens)


def generate(output, tokens):
    rng = np.random.default_rng(42)
    channels = 512
    shape = (1, channels, 1, tokens)
    x = rng.normal(0, 0.25, shape).astype(np.float32)
    w1, w2 = [
        rng.normal(0, 1 / np.sqrt(channels), (channels, channels, 1, 1)).astype(np.float16)
        for _ in range(2)
    ]

    @mb.program(
        input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)],
        opset_version=ct.target.macOS14,
    )
    def projection(x):
        h = mb.relu(x=mb.conv(x=x, weight=w1, pad_type="valid"))
        return mb.conv(x=h, weight=w2, pad_type="valid", name="y")

    start = time.perf_counter()
    model = ct.convert(
        projection,
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    model.save(str(output / "projection-fp16.mlpackage"))
    conversion_ms = (time.perf_counter() - start) * 1000

    # INT4 per-block weights require macOS 15. The activation graph stays FP16.
    int4_source = ct.convert(
        projection,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    start = time.perf_counter()
    int4 = ct.optimize.coreml.linear_quantize_weights(
        int4_source,
        config=ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=64
            )
        ),
    )
    int4.save(str(output / "projection-int4.mlpackage"))
    quantization_ms = (time.perf_counter() - start) * 1000

    state_shape = (1, 4, 32, 128)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=state_shape, dtype=types.fp16),
            mb.StateTensorSpec(shape=state_shape, dtype=types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def accumulator(x, cache):
        y = mb.add(x=x, y=mb.read_state(input=cache), name="y")
        mb.coreml_update_state(state=cache, value=y)
        return y

    stateful = ct.convert(
        accumulator, minimum_deployment_target=ct.target.macOS15, skip_model_load=True
    )
    stateful.save(str(output / "accumulator.mlpackage"))

    @mb.program(
        input_specs=[mb.StateTensorSpec(shape=(1,), dtype=types.fp32)],
        opset_version=ct.target.macOS15,
    )
    def fp32_state(cache):
        return mb.read_state(input=cache)

    try:
        ct.convert(fp32_state, minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
        fp32_state_result = "accepted"
    except ValueError as error:
        fp32_state_result = str(error)

    xf = x.astype(np.float16).astype(np.float32).reshape(channels, tokens)
    reference = w2[:, :, 0, 0].astype(np.float32) @ np.maximum(
        w1[:, :, 0, 0].astype(np.float32) @ xf, 0
    )
    for name, values in [
        ("input-f32", x), ("reference-f32", reference), ("weight1-f16", w1), ("weight2-f16", w2)
    ]:
        values.tofile(output / (name + ".bin"))
    metadata = {
        "untrained_synthetic_graph": True,
        "coremltools": ct.__version__,
        "numpy": np.__version__,
        "seed": 42,
        "shape": list(shape),
        "state_shape": list(state_shape),
        "conversion_ms": conversion_ms,
        "int4_quantization_ms": quantization_ms,
        "fp32_state_conversion": fp32_state_result,
        "package_bytes": {
            p.name: sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            for p in output.glob("*.mlpackage")
        },
    }
    compiled = output / "compiled"
    compiled.mkdir(exist_ok=True)
    metadata["compilation_ms"] = {}
    for package in sorted(output.glob("*.mlpackage")):
        start = time.perf_counter()
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(package), str(compiled)], check=True
        )
        metadata["compilation_ms"][package.name] = (time.perf_counter() - start) * 1000
    (output / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
