# Qwen3.8-27B text compatibility

The inspected `mlx-community/Qwen3.8-27B-4bit` checkpoint uses the existing
`Qwen35Model` text architecture. No attention activation change is required.
This is an architecture and tensor-layout finding; trained inference remains a
separate acceptance check. It does not establish support for other Qwen3.8 models
or image/video input through the text-only loader.

## Gate semantics

`output_gate_type: "swish"` configures the **Gated DeltaNet output gate**.
vLLM maps `swish` to `silu` when constructing its gated RMS normalization.
The existing `Qwen3NextRMSNormGated` used by `Qwen35GatedDeltaNet` already applies
this operation, with the gate multiplication in float32.
[vLLM GDN implementation](https://github.com/vllm-project/vllm/blob/d6bce42983bc0b2095ad6422dbf1399e219ae572/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L471)

`attn_output_gate: true` independently enables the **full-attention sigmoid
gate**. vLLM and SGLang both split the projected query and gate within each head,
then multiply the attention result by sigmoid before its output projection.
Changing `Qwen35Attention.mergeHeadsAndProject` to SiLU would break this model.
[vLLM attention](https://github.com/vllm-project/vllm/blob/d6bce42983bc0b2095ad6422dbf1399e219ae572/vllm/model_executor/models/qwen3_next.py#L424),
[SGLang attention](https://github.com/sgl-project/sglang/blob/9ed2721c6d98b99f12bf7067883e4a137a0cd62c/python/sglang/srt/models/qwen3_5.py#L1365)

The Swift configuration currently ignores these two extra keys. Their values in
this checkpoint match the implementation's existing behavior. This finding does
not cover a checkpoint that disables the attention gate or selects a different
GDN gate. No external implementation code was copied for this audit.

## Architecture and saved tensors

The official configuration declares `model_type: qwen3_5`, architecture
`Qwen3_5ForConditionalGeneration`, and a dense text model with 64 layers:
three GDN layers followed by one full-attention layer, repeated 16 times.
The explicit `layer_types` list agrees with `full_attention_interval: 4`.
The dimensions and RoPE settings also match the existing Qwen3.5-27B architecture.
[Qwen3.8 configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json),
[Qwen3.5 configuration](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json)

The local checkpoint was inspected on 2026-09-03 at:

```text
/Users/ethan/Library/Application Support/Mirage/Flow/OnDeviceModels/models/mlx-community/Qwen3.8-27B-4bit
```

The initial architecture audit read JSON and the three safetensors headers.
The later qualification loaded the trained tensors and executed decoding. All 2,180 indexed tensor names matched the union of the shard headers:
1,847 under `language_model` and 333 under `vision_tower`. The conversion README
identifies mlx-vlm 0.6.8 and Qwen/Qwen3.8-27B as its source.

| Component | Checkpoint layout and interpretation |
| --- | --- |
| Hidden/FFN/vocabulary | 5,120 / 17,408 / 248,320; untied LM head is present |
| Full attention | 24 query heads, 4 KV heads, head dimension 256 |
| Query projection | Packed U32 `[12288, 640]`; 4-bit unpacking gives `[12288, 5120]`, including both query and gate |
| Key/value projections | Packed U32 `[1024, 640]` each |
| Attention output | Packed U32 `[5120, 768]`, representing `[5120, 6144]` |
| GDN | 16 key heads, 48 value heads, both dimensions 128; repeat ratio 3 |
| GDN input projections | QKV `[10240, 640]`, Z `[6144, 640]`, A/B `[48, 640]`, packed U32 |
| GDN convolution | BF16 `[10240, 4, 1]`, already in MLX layout |
| GDN recurrent state | Float32, as requested by `mamba_ssm_dtype`; existing Swift zero state uses float32 |
| Quantization | Affine 4-bit, group size 64; scales/biases agree with unpacked input dimensions |
| RoPE | Base 10,000,000, factor 0.25, giving 64 rotary dimensions of each 256-dimensional head |
| Normalization | RMS epsilon 1e-6; converted norms must retain their saved scale without another `+1` |

The convolution layout identifies a converted MLX checkpoint. Existing sanitize
logic correctly avoids adding `1` again to its converted normalization weights.
GDN's gated norm uses ordinary multiplicative weights. This agrees with the
Python MLX conversion convention.
[MLX-LM normalization conversion](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py#L318),
[MLX-VLM gated norm](https://github.com/Blaizzy/mlx-vlm/blob/5c9b5f52adfeab35b5ece0bb2d6e4d44541d9e32/mlx_vlm/models/qwen3_5/language.py#L49)

For text-only input, temporal/height/width RoPE positions coincide. Interleaving
the configured `[11, 11, 10]` sections therefore gives the same frequencies as
ordinary partial RoPE; the existing text implementation is compatible. Visual
position handling belongs to the VLM path and was outside this audit.

## MTP availability

The configuration declares one MTP layer and shared embeddings, but the saved
checkpoint has **no MTP head weights**. No tensor name in any shard contains
`mtp`, `draft`, `enorm`, `hnorm`, or `eh_proj`, and the language-model layers run
only from 0 through 63. The model card's statement that MTP was trained does not
make a head available in this converted artifact. Paired MTP decoding requires a
separate compatible trained head; ordinary target decoding is unaffected.

## Acceptance coverage

`Qwen38CompatibilityTests` covers the sigmoid/SiLU distinction, 64-dimensional
text RoPE at a nonzero position, unchanged converted norms, float32 recurrent
state, and compiled continuation versus full prefill using a small dense model
with the checkpoint's metadata. The tiny fixture retains the 3:1 GDN head ratio.
Cache checks cover attention offsets and GDN state shapes/dtype separately;
the recurrent cache does not maintain an attention-style token offset.
These checks require no downloads. They do not establish trained output quality
or throughput. Run them with the project's macOS `xcodebuild test` command and
the conversation's external DerivedData/cache paths.

## Verified MTP companion

The existing target can use `mlx-community/Qwen3.8-27B-MTP-4bit` at revision
`b643c01b6d3b094e325edb6ebd832e16c486c575`. Its safetensors header contains 31 MTP
parameters, including fusion, attention, MLP and normalization tensors. Required
weights/config/index total 238,939,914 bytes; redundant tokenizer files are unnecessary.
The tokenizer digest matches the local target. The approved download was verified
against SHA-256 `76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6`.
Two concurrent greedy 32-token outputs matched the ordinary target iterator exactly.
The head accepted 28 of 33 proposed tokens across those two requests; cancelling
another request during prefill preserved the following request’s exact output.
This establishes the tested pairing, not general task accuracy or a speedup.

The standalone conversion saves bare keys (`fc.weight`, `layers.0.*`). The Swift
sanitizer now adds the `mtp.` module namespace for a recognized standalone head,
retaining quantization metadata and already-converted norms. The runtime uses the
existing greedy, block-size-two rollback contract; checkpoint `block_size: 3` does
not override that limit.

Sources: [companion](https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit),
[pinned model card](https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit/blob/b643c01b6d3b094e325edb6ebd832e16c486c575/README.md).
