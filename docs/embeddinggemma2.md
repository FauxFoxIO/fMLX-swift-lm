# Native EmbeddingGemma 2

`MLXVLM` exposes a separate local embedding loader and container. It does not
register this bidirectional encoder with the causal language-model factories.
The container serializes preprocessing, loading, inference, and unloading.

```swift
import Foundation
import MLXVLM

let embeddings = try await EmbeddingGemma2Loader.loadContainer(
    from: installedSnapshotDirectory,
    artifact: .init(
        identifier: "mlx-community/embeddinggemma-2-bf16",
        revision: installedSnapshotCommit),
    precision: .bfloat16,
    limits: .init(maximumCheckpointBytes: 4 * 1024 * 1024 * 1024,
                  maximumTokens: 8192))

let query = try await embeddings.embed(
    text: "Which planet is known as the Red Planet?",
    purpose: .query(), dimension: .d256)

let document = try await embeddings.embed(
    [.text("Mars: "), .image(.file(imageURL)),
     .text("\nObservation: "), .audio(.file(audioURL)),
     .video(.file(videoURL))],
    purpose: .document(title: "Mars observations"), dimension: .d256)

let vector: [Float] = document.values
await embeddings.unload([.vision, .audio])
```

The installer supplies the actual immutable converted-checkpoint revision.
`EmbeddingGemma2Artifact.referenceRevision` identifies the upstream Google
configuration revision `914f7f89142e33e77833254d9c9b90c3cef7303b` separately.
Results include both identities, requested dimension, and the final token count.
The container returns only finite, unit-normalized vectors at 768, 512, 256, or
128 dimensions. It rejects a zero or non-finite result.

Inputs are ordered `[EmbeddingGemma2Content]` elements. Adjacent text strings are
concatenated literally; include desired spaces or newlines. Images accept local
files or packed RGB bytes. Audio accepts local files or interleaved PCM with an
explicit sample rate and channel count. Video accepts local files or ordered
timestamped frames. Video uses its visual tower; supply the soundtrack as an
audio element if it belongs in the same embedding.

The native tokenizer is loaded from the same directory. An additional
`loadContainer(from:using:artifact:precision:limits:)` overload accepts an existing
`MLXLMCommon.TokenizerLoader`. Tokenizer-level truncation and padding must be
disabled. No chat template is applied. Default prompts are
`task: search result | query: ` and `title: none | text: `; custom task/title values
are supported. Prompt and media delimiter tokens participate in mean pooling.

## Installed assets and limits

Install the complete checkpoint, including `config.json`, `tokenizer.json`, its
tokenizer metadata, and all indexed safetensors shards (or `model.safetensors`).
Do not install a text-only conversion. The loader validates the text, vision,
audio, and projection namespaces and shapes, including tensors for unloaded
towers. Keep the snapshot immutable for the container's lifetime.

The BF16 checkpoint's tensor headers declare 1,488,743,984 bytes: 542,005,296 text,
335,515,648 vision including projection, and 611,223,040 audio including
projection. These are tensor-storage sizes, not measured process memory. The
default complete-checkpoint limit is 4 GiB. FP32 materialization expands BF16
weights; attention, media preprocessing, and intermediate activations require
additional memory. Quantized MLX variants use their checkpoint quantization
metadata and preserve non-quantized BF16/FP32 towers. FP16 weights are rejected.

Text is materialized at load time. Vision and audio remain unloaded until a
matching request. `unload` releases selected components; later requests reload
them. The MLX allocator may retain reusable buffers after release.

The input limit is at most 8,192 tokens including prompts, BOS/EOS policy,
delimiters, and exact media feature counts. A caller can choose a lower limit.
Over-budget units throw instead of being truncated. Long files must be split
into bounded contiguous units by the consuming product.

- Images resize to sides divisible by 48. Supported soft-token budgets are
  70/140/280/560/1120; the default is 280. Real placeholders equal the actual
  pooled grid size. Patches pad to budget × 9 with coordinates `(-1, -1)`.
- Audio mixes to mono and resamples to 16 kHz. The frontend uses 128 HTK mel bins,
  a periodic 320-sample Hann window, 160-sample hop, 512-point magnitude FFT,
  160 prepended zeros, and `log(mel + 0.001)`. For `N` actual samples, valid mel
  frames are `floor((N - 1) / 160)` and placeholders are
  `ceil(ceil(validFrames / 2) / 2)`. Waveform padding to a multiple of 128 carries
  an explicit validity mask. There is no 11- or 30-second cutoff; fewer than
  161 samples fail because they produce no valid mel frame.
- Video samples at one frame per second, with a maximum of 32 frames selected
  uniformly on overflow. Each frame uses budget 140 and has no prompt timestamp.
  Timestamped arrays use a frame at or before each whole-second offset; files
  use nominal frame-rate positions within their track time range.

## Reference and validation status

The text encoder, modality projections, and audio Conformer are ported from
[MLX-VLM revision 3d87e884](https://github.com/Blaizzy/mlx-vlm/tree/3d87e88402f307efbf68e568971aa887ee7d9ed0).
Vision reuses the existing Gemma4 transformer primitives with a dedicated padded
patch wrapper. Architecture and preprocessing were checked against the official
[EmbeddingGemma 2 model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py)
and [Gemma4 processors](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4).
Source attribution and license texts are retained in `LICENSES/`.

On October 11, 2026, MLXVLM compiled for macOS arm64 Debug with Xcode 27.1.
Five focused Swift Testing checks passed against the pinned BF16 artifacts,
including six real inference cases, dimension truncation, media preprocessing,
modality unloading/reloading, and oversized-input rejection. Matching fused
normalization corrected a BF16 regression that failed before the change.

Against the pinned Python MLX model, prepared text/query/image/audio/video/mixed
inputs had maximum component error 2.14e-7 with FP32 materialization. BF16 had
maximum component error 0.002419 and minimum cosine 0.999784. These comparisons
share Swift token IDs and prepared media tensors; they establish model tensor
parity for bounded fixtures, not independent tokenizer or processor parity.

Verified artifacts total 1.521 GB. The focused native process measured peak
footprint 2.96 GB with BF16 and 5.55 GB with FP32; MLX allocation peaks were
1.94 GB and 3.66 GB respectively. These are isolated fixture measurements,
not application-wide or sustained-corpus resource guarantees.

On October 15, 2026, `Tests/EmbeddingGemma2CheckpointTests` checked the container end
to end against Transformers 5.20 `EmbeddingGemma2Model` in float32 on CPU
(`google/embeddinggemma-2` at `914f7f89`), including its tokenizer, prompts, and
image and audio processors. Six cases cover a query, a document, code, an image, an
audio clip, and interleaved text, image, and audio. Token counts match exactly in
every case. The `mlx-community/embeddinggemma-2-bf16` export run in FP32 reaches
minimum cosine 0.99999999999 and maximum component error 3.6e-7; in BF16, 0.99993 and
0.0015; `mlx-community/embeddinggemma-2-8bit`, 0.99977 and 0.0030. The 256-dimension
prefix matches after renormalization. The Google snapshot loads directly and gives the
same BF16 results. `tools/decision_models/generate_fixtures.py embeddinggemma` writes
the reference.

Apple file decoding and resampling, intermediate activations, video against the
reference, 4-bit variants, and exactly 8,192-token inference remain unverified. Apple
media decoding and FP32 DSP can differ from Python decoding and NumPy double
intermediates. The focused check used no newly installed Python packages.
Codex assisted implementation and these validation checks.
