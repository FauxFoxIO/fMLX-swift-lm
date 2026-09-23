# Bonsai PTQ1 and DFlash2

The native text loader recognizes the strict Prism Hadamard Qwen 3.5 schema used
by Bonsai 2. Published MLX affine 2-bit artifacts keep their existing path. The
`scripts/convert-prism-ptq1.py` converter accepts the pinned Prism GGUF reader and
preserves every 128-weight `PTQ1_0` block as 28 bytes in safetensors. Native Metal
kernels decode those blocks directly for projections, embeddings, and tied output
heads; no affine copy is materialized.

The artifact contract is fail-closed: schema version, Qwen dimensions, component
set, tensor namespace, grouped GDN layout, transform manifest, all 402 transformed
module records, and explicit Hadamard signs must agree before weights are used.
The compact artifact is text-only because the published PTQ1 GGUF has no vision
tensors. The published affine artifact retains text and vision support.

The trained `z-lab/Qwen3.8-27B-DFlash2` drafter is supported as a standalone
companion for the matching Qwen 3.8 27B target. It captures target layers
5/19/33/47/61, keeps its 2,047-token rotating history, applies the published
position-dependent sliding mask, and samples from the selector's top-16
distribution without applying target top-p/top-k filters a second time.

For sustained decoding, quantize the companion's 47 matrix projections to MLX
affine Q4 before loading it. The codebooks, norms, and convolution kernels stay
in BF16. This matches the draft representation used by the accelerated Splash
baseline while preserving target-verified output:

```sh
python3 scripts/quantize-dflash2.py /path/to/Qwen3.8-27B-DFlash2 /path/to/DFlash2-Q4
```

The converter requires the Python `mlx` package. The output is a standard MLX
checkpoint with an explicit group-size-64, four-bit quantization declaration.

On the local M4 Max qualification fixture, four-position greedy DFlash verification
matched ordinary output exactly. A 512-token prompt followed by 64 generated tokens
measured about 26.7 versus 22.2 tokens/s for ordinary decoding, with the same
`7aaa89f1ccc5d2ec` token hash. Eight-position verification changed the greedy
output on the same target, so the public drafter advertises a maximum block size
of four. The available capture establishes the divergent token, not its logit
margin or underlying arithmetic cause. This is a correctness cap, not a statement
about the checkpoint's trained block size.

The matched 512-prompt/256-output Splash comparison used five measured Release
runs after warmup. fMLX DFlash had a 32.555 tokens/s median and the paired ordinary
path had a 27.596 tokens/s median; all runs produced the same
`df356ec5d4df1d6d` token hash. Splash's median was 34.181 tokens/s, making the
parity-safe fMLX result 95.24% of Splash. A separate Q8 paged-KV probe reached
33.719 tokens/s but diverged from ordinary output at token 57, so Q8 KV is not
enabled for this qualified path.

Concurrent speculative verification uses a two-position shape because the real
27B target's wider batched projection changed the greedy output. Four
simultaneous requests then matched ordinary decoding exactly and used 35 batched
target forwards instead of 63 ordinary batched forwards. On this fixture its
aggregate rate was lower (about 14.3 versus 18.1 tokens/s), so no concurrent
DFlash speedup is claimed.

For the converted Bonsai PTQ1 artifact, an eight-token greedy run produced the
same token hash as the published MLX affine checkpoint. The compact run used a
lower observed MLX peak (about 6.37 GB versus 8.45 GB) but was slower on this
first native kernel implementation (about 6.75 versus 17.0 decode tokens/s).
These are single-machine qualification measurements, not universal performance
claims. No physical-iPhone execution was available; `NativeTextModel` performs a
conservative checkpoint-plus-runtime memory admission check before loading so an
oversized 27B artifact fails without first pressuring unified memory.
