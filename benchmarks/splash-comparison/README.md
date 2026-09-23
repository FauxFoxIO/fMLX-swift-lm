# Splash comparison protocol

This directory defines the matched, same-machine benchmark used to compare fMLX
with Splash. It is intentionally separate from correctness tests: a Debug
`xctest` run, a different checkpoint, or a result copied from another machine is
not a comparable performance result.

## Required evidence

Before collecting samples, copy `session.example.json` outside the repository and
fill every placeholder. Record the exact target and draft weight hashes, engine
revisions, Metal/backend versions, hardware, OS, toolchain, build configuration,
and commands. Splash's native DFlash state must be established from its current
source or backend protocol; command-line flag names alone are not evidence that it
was enabled. The accelerated Splash result is the comparison baseline. Record
fMLX ordinary decoding as a separate ablation. Splash's native server requires
both target and draft directories and exposes no non-speculative mode, so it is
not labeled or fabricated as an ordinary-decoding ablation.

`protocol.json` fixes the workload cells, sampler, timing boundaries, repetitions,
equivalence rule, and noise policy before measurement. The same fully rendered
prompt token IDs must reach both engines. Record their SHA-256 and count for every
sample. If an engine cannot accept token IDs directly, capture its tokenization and
require byte-for-byte equality before comparing throughput.

Run one engine at a time. Complete a warmup and measured trial for one member of a
pair, unload it, then run the paired member. Alternate the pair order as declared
in the protocol. Do not have both engines submitting GPU work concurrently. Keep
raw logs and normalized JSON Lines in the reusable conversation cache; that cache
is disposable at nightly cleanup, so retain it for the duration of qualification.

The tracked fMLX fixture is an example for `InstalledQwenBenchmarkTests`. Replace
the path placeholders in an out-of-repository copy. It is not a benchmark result.
Build and run the package tests in Release mode:

```sh
xcodebuild build-for-testing \
  -scheme mlx-swift-lm-Package \
  -configuration Release \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -derivedDataPath "$FMLX_RELEASE_DERIVED_DATA"

FMLX_INSTALLED_QWEN_BENCHMARK_CONFIG="$FMLX_BENCHMARK_CONFIG" \
  xcrun xctest \
  "$FMLX_RELEASE_DERIVED_DATA/Build/Products/Release/FMLXTextTests.xctest"
```

The existing fixture covers cold and warm-prefix single-request cells and a
same-prompt fanout. The canonical protocol also requires an appended-turn warm
prefix and four different-prefix requests. A result set is incomplete until the
harness records those cells; substituting four copies of one prompt for the
different-prefix cell is invalid.

## Normalized samples and ledger

Normalize each invocation as one JSON object matching
`sample.schema.json`. Preserve all unrounded measurements. Set `warmup` explicitly;
warmups remain in the raw evidence but are excluded from summaries. A stopped,
fallback, mismatched-token, or thermally invalid run remains recorded with
`valid: false` and a reason rather than being deleted.

For one request, `ttftMilliseconds` is that request's TTFT. For fanout it is the
maximum request TTFT (tail latency); the per-request values remain in
`requestTTFTMilliseconds`. Fanout throughput is total generated tokens divided by
the declared fanout boundary. `requestDecodeTokensPerSecond` preserves each
request's decode rate.

Create the machine-readable ledger with:

```sh
python3 scripts/summarize-splash-comparison.py \
  benchmarks/splash-comparison/protocol.json \
  /path/to/session.json \
  /path/to/samples.jsonl \
  > /path/to/ledger.json
```

The command validates the protocol/session relationship, required cells, sample
counts, pairing, token parity, and raw metrics. It reports medians, p10/p90 spread,
median absolute deviation, fMLX/Splash ratios, and a per-cell outcome. `match`
means fMLX is within the predeclared 5% equivalence margin; `exceed` means its
median is at least Splash's median. A noisy or incomplete cell is
`inconclusive`, never a pass. Commit a final ledger only when its absolute paths
and machine-specific private details are suitable for the repository; otherwise
keep the raw ledger in the build cache and report its digest and location.

Peak MLX allocation is not process memory. Samples record both process resident
memory and MLX allocation/cached memory when available. An absent metric is
`null`, not zero. iPhone results require a physical-device session and must not be
inferred from a Mac run or a successful iOS build.
