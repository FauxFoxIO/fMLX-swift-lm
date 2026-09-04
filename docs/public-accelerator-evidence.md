# Public Core ML accelerator evidence

Public Core ML can execute a converted projection graph on the Neural Engine and
can copy state across a public API boundary. It does **not** provide a drop-in ANE
prefill path for this repository's local Qwen3.8-27B-4bit checkpoint. No trained
Qwen Core ML inference, cache handoff, or end-to-end speedup is established here.
The production runtime should retain its MLX path.

## Checkpoint and conversion boundary

Read-only inspection used the model under
`~/Library/Application Support/Mirage/Flow/OnDeviceModels/models/mlx-community/Qwen3.8-27B-4bit`.
The Hugging Face cache directory contained a ref without downloaded snapshot files.
Despite its directory name, `config.json` specifies `qwen3_5`, 64 text layers:
48 linear-attention GatedDelta layers and 16 full-attention layers. Hidden width
is 5,120; full attention has four KV heads of dimension 256. The three safetensor
shards contain 16,054,262,240 tensor bytes, with 498 U32 and 1,682 BF16 tensors.
For example, `linear_attn.in_proj_qkv.weight` in layer zero is packed U32
`[10240, 640]`; its BF16 scales and biases are `[10240, 80]`. Quantization is MLX
affine 4-bit with group size 64.

Core ML Tools accepts supported PyTorch/TensorFlow graphs and directly authored
MIL programs. It does not consume this Swift model or MLX safetensors as an
executable model. A converter must implement the architecture, unpack or translate
the weight representation, preserve RoPE/masking/gating, and match every layer's
state semantics. Merely selecting `cpuAndNeuralEngine` cannot do that work.
[Supported conversion formats](https://apple.github.io/coremltools/docs-guides/source/target-conversion-formats.html).

The repository's `Qwen35GatedDeltaNet.zeroStates` uses FP32 recurrent state of
shape `[batch, 48, 128, 128]` per recurrent layer. It also retains a convolution
history of `[batch, 3, 10240]`. Attention state alone is therefore insufficient
to continue decoding. A complete bridge must carry both recurrent tensors, all
attention K/V tensors, positions, valid lengths and cache metadata. At batch one,
uncompressed recurrent state is 144 MiB; two-byte convolution history adds
2.8125 MiB. Two-byte full-attention K/V adds 64 KiB per prompt token, or 128 MiB
at 2,048 tokens. These are shape-derived payload sizes, not measured transfer
latencies or total process memory.

The pinned Core ML Tools 9.0 exporter rejects FP32 state with
`State only support fp16 dtype`. The fixture generator executes this failing
conversion and records its result. Changing this model's recurrent state to
FP16 would change its numerical contract. Keeping FP32 recurrence as explicit
tensor inputs/outputs is an alternative graph design, but its ANE placement,
copy cost and trained-model accuracy remain unverified.
[Exporter restriction](https://github.com/apple/coremltools/blob/9.0/coremltools/converters/mil/backend/mil/load.py#L736-L740).

Core ML per-block INT4 and MLX affine 4-bit are distinct encodings. The probe's
symmetric INT4 quantization is a new quantization of its own FP16 weights; it is
not an import of the Qwen weights. Apple recommends evaluating compression on
the actual hardware and notes that per-block INT4 is particularly useful for
Mac GPU models. It must not be assumed to imply ANE execution or unchanged
accuracy. [Compression guidance](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html).

## Public APIs and availability

The probe uses `MLModel`, `MLMultiArray`, and the public compute-unit policies
`cpuOnly`, `cpuAndGPU`, `cpuAndNeuralEngine`, and `all`. These are allowed-device
sets, not instructions forcing a particular accelerator. On macOS 14.4 and
later, `MLComputePlan` reports anticipated supported/preferred devices per
operation. A preferred device is compiler-plan evidence, not an Instruments
hardware trace. [Compute-unit policy](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine),
[compute plan](https://developer.apple.com/documentation/coreml/mlcomputeplan).

`MLState`, stateful prediction, and the probe's per-block INT4 fixture require
macOS 15. The Swift test retains a macOS 14 baseline and gates these paths. State
access uses `withMultiArray`: its buffer is valid only within the closure and
may have a different address next time. Predictions sharing a state must be
serialized, and state must not be accessed while prediction is in flight. This
precludes retaining a state pointer as a permanent MLX cache view. The probe
copies within the closure, respects strides, then resumes inference from an
independent state object. It adds no production cache adapter.
[Stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html),
[state access](https://developer.apple.com/documentation/coreml/mlstate).

MLX exposes host-array copies and conditional no-copy Metal views with its own
lifetime requirements. Neither establishes a shared persistent Core ML state
allocation. The measured bridge therefore uses evaluated MLX arrays, a copied
FP16 host buffer, `MLMultiArray`, and a copied MLX result. Pending MLX model work,
Qwen cache layout conversion and quantization are outside that measurement.

## Measured public graph

Measured September 3, 2026 on an M4 Max Mac Studio, 128 GB, macOS 27.0
`26A5421a`, Xcode `27A5252f`, Swift 6.4. Dependencies: Python 3.11,
Core ML Tools 9.0, NumPy 2.4.6. No trained model or additional weight download
was used. Beta OS/compiler results need remeasurement on deployment hardware.

The untrained graph is `W2 × ReLU(W1 × X)` with two seeded 512-by-512 FP16
matrices. Its Core ML implementation uses 1-by-1 convolutions in
`[1, 512, 1, sequence]` layout. This follows Apple's documented ANE-friendly
projection layout, but omits attention, recurrence, normalization, sampling and
tokenization. The reference uses FP32 multiplication of the same FP16-rounded
weights and input. [Apple's transformer layout guidance](https://machinelearning.apple.com/research/neural-engine-transformers).

The standalone Swift executable was compiled with `-O`, Swift 6 language mode,
and a macOS 14 deployment target. It measures host FP32-to-FP16 conversion,
input allocation/copy, synchronous prediction, and copying FP16 output to a host
array. Five warmups precede 20 samples. Timings below are medians in milliseconds;
independent component medians need not sum to the median total.

At sequence 2,048:

| Weights | Allowed units | Preferred projection device | Input | Prediction | Output | Total | Relative RMSE |
|---|---|---|---:|---:|---:|---:|---:|
| FP16 | CPU | CPU | 0.772 | 0.571 | 0.032 | 1.377 | 0.00408 |
| FP16 | CPU + GPU | GPU | 0.787 | 0.918 | 0.032 | 1.744 | 0.000294 |
| FP16 | CPU + ANE | ANE | 0.782 | 0.333 | 0.033 | 1.152 | 0.000659 |
| FP16 | All | ANE | 0.788 | 0.314 | 0.031 | 1.130 | 0.000659 |
| INT4 | CPU | CPU | 0.780 | 0.563 | 0.030 | 1.359 | 0.1510 |
| INT4 | CPU + GPU | GPU | 0.760 | 1.112 | 0.033 | 1.859 | 0.1510 |
| INT4 | CPU + ANE | CPU | 0.757 | 0.603 | 0.031 | 1.398 | 0.1510 |
| INT4 | All | GPU | 0.791 | 0.949 | 0.031 | 1.762 | 0.1510 |

At sequence 128, every policy preferred CPU and total times were 0.103–0.109 ms.
The INT4 convolution operations did not list ANE among their supported devices
at either shape. The 2,048-position FP16 graph demonstrates a useful ANE-selected
compute path, but its measured host boundary costs more than prediction itself.
This does not establish an MLX or Qwen speedup. INT4's approximately 15% relative
error here is a synthetic quantization result, not an acceptable LLM quality
criterion.

For the 2,048-position fixture, MIL-to-package conversion took 256.2 ms and INT4
quantization/saving took 72.3 ms (excluding the separate macOS 15 source graph
conversion). `coremlcompiler` took 190.7 ms for FP16 and 120.3 ms for INT4.
Package sizes were 1,051,878 and 282,092 bytes respectively. Per-policy loading
took 11.7–58.5 ms. First-call totals ranged from 1.7–255.8 ms; the largest was
FP16 CPU+GPU. These are observed single-process timings with existing system
caches, not guaranteed cold-install costs. Conversion and compilation are not
included in warm totals.

The separate accumulator graph verified a 32 KiB FP16 state update, export,
import, continued prediction, and independence of the original state under all
four policies. This demonstrates public state transport, not Qwen KV/recurrent
state correctness. The table above measures the standalone host boundary; the
separate XCTest results below include MLX transport.

## Measured MLX boundary

`PublicAcceleratorProbeTests.testPublicCoreMLComputeAndStateTransfer` passed with
the same 2,048-position fixtures. The test measures an evaluated resident MLX
FP32 input, its conversion and copy into Core ML FP16, synchronous prediction,
then copying the result into an evaluated MLX FP16 array. No outstanding upstream
MLX model computation is included. Five warmups precede 20 samples; all times
below are medians in milliseconds from that XCTest invocation.

| Weights | Allowed Core ML units | Preferred projection device | MLX input conversion/copy | Prediction | Copy to MLX | Total |
|---|---|---|---:|---:|---:|---:|
| FP16 | CPU | CPU | 0.283 | 0.497 | 0.064 | 0.839 |
| FP16 | CPU + GPU | GPU | 0.287 | 0.695 | 0.063 | 1.126 |
| FP16 | CPU + ANE | ANE | 0.269 | 0.337 | 0.063 | 0.710 |
| FP16 | All | ANE | 0.343 | 0.339 | 0.063 | 0.853 |
| INT4 | CPU | CPU | 0.270 | 0.500 | 0.064 | 0.900 |
| INT4 | CPU + GPU | GPU | 0.282 | 0.766 | 0.065 | 1.115 |
| INT4 | CPU + ANE | CPU | 0.287 | 0.478 | 0.063 | 0.871 |
| INT4 | All | GPU | 0.394 | 0.850 | 0.062 | 1.403 |

The same XCTest measured the equivalent two-matrix/ReLU computation with resident,
already-converted MLX FP16 input and weights at **0.626 ms**, including evaluation
but excluding the later host copy used for its accuracy check. Its relative RMSE
was 0.000294. Core ML's FP16 ANE-selected prediction had relative RMSE 0.000659;
INT4 remained approximately 0.151. The Core ML copied route's best FP16 median
total, 0.710 ms, did not beat the resident MLX baseline in this run, even though
its prediction portion took 0.337 ms. The routes start at different precision
boundaries: Core ML includes an FP32-to-FP16 conversion, while the resident MLX
baseline starts with FP16. These measurements establish costs for those explicit
boundaries, not a general accelerator ranking or trained-model speedup.

Core ML load times in this XCTest were 13.2–50.7 ms, with first-call totals of
1.72–146.0 ms. The separate 32 KiB accumulator state round trip through MLX
passed continuation and independence checks under all policies. Its one-shot
export/import measurements were 0.0731 ms (CPU), 0.0509 ms (CPU+GPU), 0.0505 ms
(CPU+ANE), and 0.0670 ms (all); these are not warmed medians or full Qwen cache
transfer measurements.

All 13 `[PUBLIC_ACCELERATOR]` JSON records were preserved from the passing
`test-expanded.log` before the build log was reused. The durable machine-readable
report is `/Users/ethan/Documents/fMLX-runtime-01a069c9/public-accelerator-results.jsonl`.
It retains the full compute plans, precision errors, first-call/load timings and
state results. The JSON does not record Xcode build configuration, so it should
not be used to compare Swift host overhead against the separately optimized
standalone executable without checking that configuration.

## Reproduction

Use an external cache with its ownership manifest. This conversation uses
`~/Library/Caches/CodexBuilds/fmlx-01a069c9-7a4f-swift6.4-27A5252f-macos/public-accelerator`.
Converter scratch is created in the macOS temporary directory and removed on
exit. Fixture generation requires a host/toolchain that can compile macOS 15
models. The scripts write no build artifacts or dependency caches in the repo.

```sh
PROBE_CACHE="$HOME/Library/Caches/CodexBuilds/fmlx-01a069c9-7a4f-swift6.4-27A5252f-macos/public-accelerator"
python3.11 -m venv "$PROBE_CACHE/venv"
"$PROBE_CACHE/venv/bin/python" -m pip install --cache-dir "$PROBE_CACHE/pip-cache" coremltools==9.0 numpy==2.4.6
PYTHONDONTWRITEBYTECODE=1 "$PROBE_CACHE/venv/bin/python" scripts/public-accelerator-fixtures.py --tokens 2048 --output "$PROBE_CACHE/fixtures-2048"
xcrun swiftc -O -swift-version 6 -D PUBLIC_ACCELERATOR_STANDALONE -parse-as-library -target arm64-apple-macos14.0 -module-cache-path "$PROBE_CACHE/ModuleCache" Tests/MLXLMTests/PublicAcceleratorProbeTests.swift -o "$PROBE_CACHE/probe"
"$PROBE_CACHE/probe" "$PROBE_CACHE/fixtures-2048"
```

For the MLX boundary, run the repository's Xcode test command with the existing
external derived-data and package-cache arguments, disable test parallelism, and
select `MLXLMTests/PublicAcceleratorProbeTests/testPublicCoreMLComputeAndStateTransfer`.
Set `TEST_RUNNER_MLX_PUBLIC_ACCELERATOR_FIXTURES` to the absolute fixture directory
so the test host receives `MLX_PUBLIC_ACCELERATOR_FIXTURES`. Without that variable,
the XCTest explicitly skips. Use a release configuration for performance
comparison; no hardware-dependent timing threshold is imposed.

The remaining production blocker is an equivalent, converted trained graph with
a complete, precision-preserving hybrid cache contract. Neither this successful
synthetic ANE plan nor copying an arbitrary tensor satisfies that contract.
