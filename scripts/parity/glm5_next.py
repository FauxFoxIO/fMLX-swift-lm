"""Tiny GLM-5.3-Flash (`glm5_next`) text fixtures from transformers, plus its MTP layer.

Writes two fixtures:

- `glm5_next`: float32 weights in Hugging Face naming (`model.language_model.*`, per-expert MLPs,
  separate `q/k/v_conv1d`, `hc_attn_*` / `hc_ffn_*`, MTP layer at index `num_hidden_layers`, and a
  stray vision tensor the loader must skip). The config nests the text model under `text_config`
  with `deepseek_sparse_attention` layer types and a `linear_attn_config`, like the released
  checkpoint. It has KDA and sparse-attention layers, a shared indexer layer, hyper-connections,
  and a sequence long enough that the pooled indexer drops pools.
- `glm5_next_mixed`: the same model in the mlx-community layout (same names, per-expert affine
  quantization with float16 scales, 5-bit routed `down_proj` and 6-bit shared experts declared
  under `switch_mlp` names, unquantized KDA, indexer, `kv_b_proj`, embeddings, and head). Its
  reference logits come from transformers running on the dequantized weights.

transformers does not implement the MTP layer, so its reference follows SGLang's NextN layer:
`eh_proj(cat(enorm(embed), hnorm(hidden)))`, one plain-residual decoder layer with sparse
attention and the MoE, then `shared_head.norm` and the shared `lm_head`. `hidden` is the main
model's final (post-norm) hidden state.
"""

import copy
import json
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
import torch
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextAttention,
    Glm5NextTextIndexer,
    Glm5NextTextModel,
    Glm5NextTextMoE,
    Glm5NextTextRMSNorm,
)

ROOT = Path(__file__).resolve().parents[2] / "Tests/MLXLMTests/Resources/Parity"

LAYER_TYPES = ["linear_attention", "deepseek_sparse_attention", "deepseek_sparse_attention",
               "linear_attention"]
text_config = {
    "model_type": "glm5_next_text",
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "q_lora_rank": 32,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 0,
    "qk_head_dim": 16,
    "head_dim": 0,
    "v_head_dim": 16,
    "index_n_heads": 8,
    "index_head_dim": 16,
    "index_topk": 8,
    "index_kpool": 4,
    "index_kpool_always_select_tail": True,
    "index_kpool_compress": True,
    "index_share_for_mtp_iteration": True,
    "indexer_types": ["full", "full", "shared", "full"],
    "layer_types": LAYER_TYPES,
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "first_k_dense_replace": 1,
    "linear_attn_config": {
        "num_heads": 2,
        "head_dim": 32,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [0, 3],
        "full_attn_layers": [1, 2],
    },
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "num_nextn_predict_layers": 1,
    "hc_mult": 4,
    "hc_eps": 1e-6,
    "hc_sinkhorn_iters": 20,
    "mhc": True,
    "mla_use_nope": True,
    "swiglu_limit": 0.75,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 256,
    "attention_bias": False,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "pad_token_id": 0,
}
config = {
    "architectures": ["Glm5NextForConditionalGeneration"],
    "model_type": "glm5_next",
    "text_config": text_config,
    "vision_config": {"model_type": "glm5_next_vision", "depth": 1, "hidden_size": 8},
    "tie_word_embeddings": False,
}
IDS = [1, 5, 9, 2, 77, 31, 8, 100, 3, 64, 12, 90, 45, 7, 19, 120, 33, 58, 4, 71]
NUM_LAYERS = text_config["num_hidden_layers"]


def make_config(**overrides):
    values = {k: v for k, v in text_config.items() if k != "model_type"}
    values["layer_types"] = [
        "linear_attention" if t == "linear_attention" else "indexed_attention"
        for t in values["layer_types"]
    ]
    values.update(overrides)
    hf = Glm5NextTextConfig(**values)
    hf._attn_implementation = "eager"
    return hf


def randomize(module, generator):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("A_log"):
                p.copy_(0.1 * torch.randn(p.shape, generator=generator))
            elif name.endswith("dt_bias"):
                # Spreads the decay across (0, 1) instead of pinning it near exp(-5).
                p.copy_(-3 + torch.randn(p.shape, generator=generator))
            elif p.ndim == 1:
                base = 0.0 if name.endswith("bias") else 1.0
                p.copy_(base + 0.1 * torch.randn(p.shape, generator=generator))
            else:
                p.copy_(torch.randn(p.shape, generator=generator) / math.sqrt(p.shape[-1]))
        for name, b in module.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(0.1 * torch.randn(b.shape, generator=generator))


def pool_scores(indexer, hidden_states, q_resid, attention_mask):
    """The indexer's masked pool scores and pool budget, recomputed as `forward` does."""
    b, s, _ = hidden_states.shape
    q = indexer.wq_b(q_resid).view(b, s, -1, indexer.head_dim)
    k = indexer.k_norm(indexer.wk(hidden_states)).view(b, s, -1, indexer.head_dim).squeeze(2)
    gates = torch.nn.functional.linear(hidden_states, indexer.index_kpool_compress_gate)
    packed = torch.cat([k, gates, attention_mask.to(k.dtype)[..., None]], dim=-1)
    valid_keys = packed[..., -1].bool()
    visible = indexer.get_visible_tokens(valid_keys=valid_keys, q_length=s, current_length=s)
    pool_keys, pool_indices, pool_valid = indexer.get_pooled_states(packed_states=packed)
    scores = torch.relu(q.float() @ pool_keys.transpose(-1, -2).float().unsqueeze(1)
                        * indexer.softmax_scale)
    weights = indexer.weights_proj(hidden_states).float() * indexer.n_heads**-0.5
    index_scores = (weights.unsqueeze(-2) @ scores).squeeze(-2)
    pool_end = pool_indices[..., -1].clamp(0, s - 1)
    pool_visible = visible.gather(-1, pool_end[:, None, :].expand(b, s, -1))
    candidates = pool_visible & pool_valid[:, None]
    select_k = min(indexer.index_topk // indexer.index_kpool, index_scores.shape[-1])
    return index_scores, candidates, select_k


class BoundaryGaps:
    """Records the smallest gap at the top-k pool boundary so ties cannot make the fixture
    ambiguous, and how many rows actually dropped a visible pool."""

    def __init__(self):
        self.smallest, self.sparse_rows = float("inf"), 0

    def __call__(self, indexer, args, kwargs, output):
        hidden, q_resid, mask = kwargs["hidden_states"], kwargs["q_resid"], kwargs["attention_mask"]
        scores, candidates, k = pool_scores(indexer, hidden, q_resid, mask)
        for row, valid in zip(scores.flatten(0, -2), candidates.flatten(0, -2)):
            visible = row[valid]
            if visible.numel() > k:
                self.sparse_rows += 1
                ranked = visible.sort(descending=True).values
                self.smallest = min(self.smallest, (ranked[k - 1] - ranked[k]).item())


def watch(modules):
    gaps = BoundaryGaps()
    for module in modules:
        if isinstance(module, Glm5NextTextIndexer):
            module.register_forward_hook(gaps, with_kwargs=True)
    return gaps


def hub_name(name):
    """transformers' module names -> the released checkpoint's names."""
    for source, target in (
        ("self_attn.forget_gate.", "self_attn."),
        ("attn_hc.fn", "hc_attn_fn"), ("attn_hc.base", "hc_attn_base"),
        ("attn_hc.scale", "hc_attn_scale"), ("ffn_hc.fn", "hc_ffn_fn"),
        ("ffn_hc.base", "hc_ffn_base"), ("ffn_hc.scale", "hc_ffn_scale"),
    ):
        name = name.replace(source, target)
    return name


def hub_tensors(prefix, state):
    out = {}
    for name, value in state.items():
        name = hub_name(name)
        if name.endswith("mlp.experts.gate_up_proj"):
            base = name.removesuffix("gate_up_proj")
            gate, up = value.chunk(2, dim=1)
            for e in range(value.shape[0]):
                out[f"{prefix}{base}{e}.gate_proj.weight"] = gate[e]
                out[f"{prefix}{base}{e}.up_proj.weight"] = up[e]
        elif name.endswith("mlp.experts.down_proj"):
            base = name.removesuffix("down_proj")
            for e in range(value.shape[0]):
                out[f"{prefix}{base}{e}.down_proj.weight"] = value[e]
        elif name.endswith("self_attn.conv1d.weight"):
            base = name.removesuffix("conv1d.weight")
            for part, chunk in zip(("q", "k", "v"), value.chunk(3, dim=0)):
                out[f"{prefix}{base}{part}_conv1d.weight"] = chunk
        else:
            out[f"{prefix}{name}"] = value
    return out


def hf_checkpoint(model, lm_head, mtp):
    out = hub_tensors("model.language_model.", model.state_dict())
    out["lm_head.weight"] = lm_head.weight
    layer = f"model.language_model.layers.{NUM_LAYERS}."
    block = {f"self_attn.{k}": v for k, v in mtp["attn"].state_dict().items()}
    block |= {f"mlp.{k}": v for k, v in mtp["mlp"].state_dict().items()}
    block |= {
        "input_layernorm.weight": mtp["input_norm"].weight,
        "post_attention_layernorm.weight": mtp["post_norm"].weight,
        "eh_proj.weight": mtp["eh_proj"].weight,
        "enorm.weight": mtp["enorm"].weight,
        "hnorm.weight": mtp["hnorm"].weight,
        "shared_head.norm.weight": mtp["head_norm"].weight,
    }
    out |= hub_tensors(layer, block)
    out["model.visual.patch_embed.proj.weight"] = torch.ones(2, 3)
    return {k: v.detach().float().contiguous() for k, v in out.items()}


def save(name, cfg, weights, expected):
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    mx.save_safetensors(str(out / "expected.safetensors"), expected)
    size = sum(p.stat().st_size for p in out.iterdir())
    print(f"wrote {out} ({size} bytes)")
    assert size < 1_000_000, "fixtures must stay under 1 MB"


def to_mx(t):
    return mx.array(t.detach().float().numpy())


def forward(model, lm_head, ids):
    mask = torch.ones_like(ids, dtype=torch.bool)
    hidden = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    return lm_head(hidden), hidden


torch.manual_seed(0)
generator = torch.Generator().manual_seed(0)
hf_config = make_config()
model = Glm5NextTextModel(hf_config).eval()
lm_head = torch.nn.Linear(32, 128, bias=False)
randomize(model, generator)
randomize(lm_head, generator)

mtp_config = make_config(
    num_hidden_layers=1, layer_types=["indexed_attention"], indexer_types=["full"],
    mlp_layer_types=["sparse"])
mtp = torch.nn.ModuleDict(
    {
        "enorm": Glm5NextTextRMSNorm(32, eps=1e-5),
        "hnorm": Glm5NextTextRMSNorm(32, eps=1e-5),
        "eh_proj": torch.nn.Linear(64, 32, bias=False),
        "input_norm": Glm5NextTextRMSNorm(32, eps=1e-5),
        "attn": Glm5NextTextAttention(mtp_config, 0),
        "post_norm": Glm5NextTextRMSNorm(32, eps=1e-5),
        "mlp": Glm5NextTextMoE(mtp_config),
        "head_norm": Glm5NextTextRMSNorm(32, eps=1e-5),
    }
).eval()
randomize(mtp, generator)

ids = torch.tensor([IDS])
length = ids.shape[1]
gaps = watch(list(model.modules()) + list(mtp.modules()))
with torch.no_grad():
    logits, hidden = forward(model, lm_head, ids)

    dense = Glm5NextTextModel(make_config(index_topk=4 * length)).eval()
    dense.load_state_dict(model.state_dict())
    sparse_effect = (forward(dense, lm_head, ids)[0] - logits).abs().max().item()
    print(f"max |sparse - dense| logits: {sparse_effect:.4f}")
    assert sparse_effect > 1e-2, "pooled sparse selection must change the logits"

    unclamped = Glm5NextTextModel(make_config(swiglu_limit=1e6)).eval()
    unclamped.load_state_dict(model.state_dict())
    clamp_effect = (forward(unclamped, lm_head, ids)[0] - logits).abs().max().item()
    print(f"max |clamped - unclamped| logits: {clamp_effect:.4f}")
    assert clamp_effect > 1e-2, "the SwiGLU limit must change the logits"

    # MTP: token i+1 pairs with the main hidden at position i.
    next_token = int(logits[0, -1].argmax())
    mtp_ids = torch.cat([ids[:, 1:], torch.tensor([[next_token]])], dim=1)
    embeds = model.embed_tokens(mtp_ids)
    h = mtp["eh_proj"](torch.cat([mtp["enorm"](embeds), mtp["hnorm"](hidden)], dim=-1))
    attn_out, _, _ = mtp["attn"](
        mtp["input_norm"](h), attention_mask=torch.ones_like(mtp_ids, dtype=torch.bool))
    h = h + attn_out
    h = h + mtp["mlp"](mtp["post_norm"](h))
    mtp_logits = lm_head(mtp["head_norm"](h))
print(f"smallest top-k boundary gap: {gaps.smallest:.2e} over {gaps.sparse_rows} sparse rows")
assert gaps.sparse_rows > 0, "the sequence must be long enough to drop pools"
assert gaps.smallest > 1e-3, "index score ties make the top-k selection ambiguous"

weights = hf_checkpoint(model, lm_head, mtp)
save(
    "glm5_next",
    config,
    {k: to_mx(v) for k, v in weights.items()},
    {
        "input_ids": mx.array(ids.numpy().astype(np.int32)),
        "logits": to_mx(logits),
        "hidden": to_mx(hidden),
        "mtp_input_ids": mx.array(mtp_ids.numpy().astype(np.int32)),
        "mtp_logits": to_mx(mtp_logits),
    },
)

# mlx-community layout: same names, per-expert affine quantization with float16 scales.
GROUP = 32
quant = {"group_size": GROUP, "bits": 4}
mlx_weights = {}
per_layer = {}
main = {
    k: to_mx(v) for k, v in weights.items()
    if not k.startswith(f"model.language_model.layers.{NUM_LAYERS}.")
}
dequantized = {}


def quantize(name, w, bits):
    path = name.removesuffix(".weight")
    wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=bits)
    scales, biases = scales.astype(mx.float16), biases.astype(mx.float16)
    mlx_weights[f"{path}.weight"] = wq
    mlx_weights[f"{path}.scales"] = scales
    mlx_weights[f"{path}.biases"] = biases
    return mx.dequantize(wq, scales.astype(mx.float32), biases.astype(mx.float32),
                         group_size=GROUP, bits=bits)


QUANTIZED = ("mlp.", "self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa")
for name, w in main.items():
    layer = name.split(".")[3] if name.startswith("model.language_model.layers.") else None
    sparse_attention = layer is not None and LAYER_TYPES[int(layer)] != "linear_attention"
    quantizable = (
        w.ndim == 2 and not name.endswith("mlp.gate.weight")
        and (any(f".{p}" in name for p in QUANTIZED[:1])
             or any(p in name for p in QUANTIZED[1:])
             or (sparse_attention and name.endswith("self_attn.o_proj.weight")))
    )
    if not quantizable:
        mlx_weights[name] = w
        dequantized[name] = w
        continue
    bits = 4
    canonical = name.replace("model.language_model.", "model.").removesuffix(".weight")
    if ".mlp.experts." in name and name.endswith("down_proj.weight"):
        bits = 5
        switch = canonical.split(".mlp.")[0] + ".mlp.switch_mlp.down_proj"
        per_layer[switch] = {"group_size": GROUP, "bits": 5}
    elif ".mlp.shared_experts." in name:
        bits = 6
        per_layer[canonical] = {"group_size": GROUP, "bits": 6}
    elif name.endswith("mlp.down_proj.weight"):
        bits = 5
        per_layer[canonical] = {"group_size": GROUP, "bits": 5}
    dequantized[name] = quantize(name, w, bits)

reference = Glm5NextTextModel(hf_config).eval()
state = {}
prefix = "model.language_model."
for name, value in model.state_dict().items():
    hub = prefix + hub_name(name)
    if name.endswith("mlp.experts.gate_up_proj"):
        base = hub.removesuffix("gate_up_proj")
        n = value.shape[0]
        merged = mx.stack([
            mx.concatenate([dequantized[f"{base}{e}.gate_proj.weight"],
                            dequantized[f"{base}{e}.up_proj.weight"]], axis=0)
            for e in range(n)])
        state[name] = torch.tensor(np.array(merged))
    elif name.endswith("mlp.experts.down_proj"):
        base = hub.removesuffix("down_proj")
        merged = mx.stack([dequantized[f"{base}{e}.down_proj.weight"]
                           for e in range(value.shape[0])])
        state[name] = torch.tensor(np.array(merged))
    elif name.endswith("self_attn.conv1d.weight"):
        base = hub.removesuffix("conv1d.weight")
        merged = mx.concatenate([dequantized[f"{base}{p}_conv1d.weight"] for p in "qkv"])
        state[name] = torch.tensor(np.array(merged))
    else:
        state[name] = torch.tensor(np.array(dequantized[hub]))
reference.load_state_dict(state)
gaps = watch(reference.modules())
with torch.no_grad():
    mixed_logits, _ = forward(reference, lm_head, ids)
assert gaps.smallest > 1e-3, "index score ties make the top-k selection ambiguous"

mixed_config = copy.deepcopy(config)
mixed_config["quantization"] = {**quant, **per_layer}
mixed_config["quantization_config"] = {**quant, **per_layer}
save(
    "glm5_next_mixed",
    mixed_config,
    mlx_weights,
    {"input_ids": mx.array(ids.numpy().astype(np.int32)), "logits": to_mx(mixed_logits)},
)
