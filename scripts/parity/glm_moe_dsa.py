"""Tiny GLM-MoE-DSA (GLM-5.3) fixtures from transformers, plus a DeepSeek-style MTP layer.

Writes two fixtures:

- `glm_moe_dsa`: float32 weights in Hugging Face naming (per-expert MLPs, `kv_b_proj`, MTP layer
  at index `num_hidden_layers`). The sequence is longer than `index_topk` and includes shared
  indexer layers, so the sparse selection changes the logits.
- `glm_moe_dsa_mixed`: the same model in mlx-community naming (stacked experts, absorbed
  `embed_q`/`unembed_out`, no MTP) with mixed 4/5-bit affine quantization. Its reference logits
  come from transformers running on the dequantized weights.

The MTP reference follows transformers' `MtpLayer`: `eh_proj(cat(enorm(embed), hnorm(hidden)))`,
one decoder layer, then `shared_head.norm` and the shared `lm_head`. `hidden` is the main model's
final (post-norm) hidden state, as transformers' `MTPCandidateGenerator` passes it.
"""

import copy
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaIndexer,
    GlmMoeDsaRMSNorm,
    apply_rotary_pos_emb_interleave,
)

ROOT = Path(__file__).resolve().parents[2] / "Tests/MLXLMTests/Resources/Parity"

config = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "q_lora_rank": 32,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 8,
    "qk_head_dim": 40,
    "head_dim": 8,
    "v_head_dim": 32,
    "index_n_heads": 8,
    "index_head_dim": 16,
    "index_topk": 5,
    "indexer_types": ["full", "shared", "full", "shared"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "first_k_dense_replace": 1,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "moe_layer_freq": 1,
    "num_nextn_predict_layers": 1,
    "rms_norm_eps": 1e-5,
    "rope_interleave": True,
    "indexer_rope_interleave": True,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "max_position_embeddings": 256,
    "attention_bias": False,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
}
IDS = [1, 5, 9, 2, 77, 31, 8, 100, 3, 64, 12, 90, 45, 7]


def make_config(**overrides):
    values = {k: v for k, v in config.items() if k not in ("architectures", "model_type")}
    values.update(overrides)
    hf = GlmMoeDsaConfig(**values)
    hf._attn_implementation = "eager"
    return hf


def randomize(module, generator):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("e_score_correction_bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=generator))
            elif p.ndim == 1:
                base = 0.0 if name.endswith("bias") else 1.0
                p.copy_(base + 0.1 * torch.randn(p.shape, generator=generator))
            else:
                p.copy_(torch.randn(p.shape, generator=generator) / math.sqrt(p.shape[-1]))
        for name, b in module.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(0.1 * torch.randn(b.shape, generator=generator))


def index_scores(indexer, hidden_states, q_resid, position_embeddings, attention_mask):
    """The indexer's pre-top-k scores, recomputed exactly as `GlmMoeDsaIndexer.forward`."""
    b, s, _ = hidden_states.shape
    rope, rest = indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim
    q = indexer.wq_b(q_resid).view(b, s, indexer.n_heads, indexer.head_dim)
    q_rot, q_pass = torch.split(q, [rope, rest], dim=-1)
    k = indexer.k_norm(indexer.wk(hidden_states)).unsqueeze(2)
    k_rot, k_pass = torch.split(k, [rope, rest], dim=-1)
    cos, sin = position_embeddings
    q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)
    scores = torch.relu(q.float() @ k.transpose(-1, -2).float().unsqueeze(1) * indexer.softmax_scale)
    weights = indexer.weights_proj(hidden_states).float() * indexer.n_heads**-0.5
    return (weights.unsqueeze(-2) @ scores).squeeze(-2) + attention_mask


class BoundaryGaps:
    """Records the smallest gap at the top-k boundary so ties cannot make the fixture ambiguous."""

    def __init__(self, topk):
        self.topk, self.smallest = topk, float("inf")

    def __call__(self, indexer, args, kwargs, output):
        scores = index_scores(indexer, *args[:4]).flatten(0, -2)
        for row in scores:
            valid = row[row > torch.finfo(torch.float32).min / 2]
            if valid.numel() > self.topk:
                ranked = valid.sort(descending=True).values
                self.smallest = min(self.smallest, (ranked[self.topk - 1] - ranked[self.topk]).item())


def causal_mask(length):
    mask = torch.full((length, length), torch.finfo(torch.float32).min).triu(1)
    return mask[None, None]


def hf_checkpoint(model, mtp, num_layers):
    """Hub naming: per-expert MLP weights, MTP layer at `model.layers.<num_layers>`."""
    out = {}
    state = model.state_dict()
    for name, value in state.items():
        if ".mlp.experts.gate_up_proj" in name:
            prefix = name.replace("gate_up_proj", "")
            gate, up = value.chunk(2, dim=1)
            for e in range(value.shape[0]):
                out[f"{prefix}{e}.gate_proj.weight"] = gate[e]
                out[f"{prefix}{e}.up_proj.weight"] = up[e]
        elif ".mlp.experts.down_proj" in name:
            prefix = name.replace("down_proj", "")
            for e in range(value.shape[0]):
                out[f"{prefix}{e}.down_proj.weight"] = value[e]
        else:
            out[name] = value
    layer = f"model.layers.{num_layers}"
    for name, value in mtp["block"].state_dict().items():
        if name.startswith("mlp.experts.gate_up_proj"):
            gate, up = value.chunk(2, dim=1)
            for e in range(value.shape[0]):
                out[f"{layer}.mlp.experts.{e}.gate_proj.weight"] = gate[e]
                out[f"{layer}.mlp.experts.{e}.up_proj.weight"] = up[e]
        elif name.startswith("mlp.experts.down_proj"):
            for e in range(value.shape[0]):
                out[f"{layer}.mlp.experts.{e}.down_proj.weight"] = value[e]
        else:
            out[f"{layer}.{name}"] = value
    out[f"{layer}.eh_proj.weight"] = mtp["eh_proj"].weight
    out[f"{layer}.enorm.weight"] = mtp["enorm"].weight
    out[f"{layer}.hnorm.weight"] = mtp["hnorm"].weight
    out[f"{layer}.shared_head.norm.weight"] = mtp["post_norm"].weight
    return {k: v.detach().float().contiguous() for k, v in out.items()}


def save(name, cfg, weights, expected):
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    mx.save_safetensors(str(out / "expected.safetensors"), expected)
    size = sum(p.stat().st_size for p in out.iterdir())
    print(f"wrote {out} ({size} bytes)")


def to_mx(t):
    return mx.array(t.detach().float().numpy())


torch.manual_seed(0)
generator = torch.Generator().manual_seed(0)
hf_config = make_config()
model = GlmMoeDsaForCausalLM(hf_config).eval()
randomize(model, generator)

mtp_config = make_config(num_hidden_layers=1, indexer_types=["full"], mlp_layer_types=["sparse"])
mtp = torch.nn.ModuleDict(
    {
        "enorm": GlmMoeDsaRMSNorm(32, eps=1e-5),
        "hnorm": GlmMoeDsaRMSNorm(32, eps=1e-5),
        "eh_proj": torch.nn.Linear(64, 32, bias=False),
        "block": GlmMoeDsaDecoderLayer(mtp_config, 0),
        "post_norm": GlmMoeDsaRMSNorm(32, eps=1e-5),
    }
).eval()
randomize(mtp, generator)

ids = torch.tensor([IDS])
length = ids.shape[1]
gaps = BoundaryGaps(config["index_topk"])
for module in list(model.modules()) + list(mtp.modules()):
    if isinstance(module, GlmMoeDsaIndexer):
        module.register_forward_hook(gaps, with_kwargs=True)
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
    logits = out.logits
    hidden = out.hidden_states[-1]  # post-norm final hidden

    dense = GlmMoeDsaForCausalLM(make_config(index_topk=length)).eval()
    dense.load_state_dict(model.state_dict())
    sparse_effect = (dense(ids).logits - logits).abs().max().item()
    print(f"max |sparse - dense| logits: {sparse_effect:.4f}")
    assert sparse_effect > 1e-2, "sparse selection must change the logits"

    # MTP: token i+1 pairs with the main hidden at position i.
    next_token = int(logits[0, -1].argmax())
    mtp_ids = torch.cat([ids[:, 1:], torch.tensor([[next_token]])], dim=1)
    embeds = model.model.embed_tokens(mtp_ids)
    h = mtp["eh_proj"](torch.cat([mtp["enorm"](embeds), mtp["hnorm"](hidden)], dim=-1))
    position_ids = torch.arange(length)[None]
    rope = model.model.rotary_emb(h, position_ids=position_ids)
    h, _ = mtp["block"](
        h, attention_mask=causal_mask(length), position_embeddings=rope, position_ids=position_ids
    )
    mtp_logits = model.lm_head(mtp["post_norm"](h))
print(f"smallest top-k boundary gap: {gaps.smallest:.2e}")
assert gaps.smallest > 1e-3, "index score ties make the top-k selection ambiguous"

weights = hf_checkpoint(model, mtp, config["num_hidden_layers"])
save(
    "glm_moe_dsa",
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

# mlx-community layout with mixed-precision quantization.
GROUP = 32
quant = {"group_size": GROUP, "bits": 4, "mode": "affine"}
mlx_weights = {}
per_layer = {}
nope, v_dim, heads = config["qk_nope_head_dim"], config["v_head_dim"], config["num_attention_heads"]


def quantize(path, w, bits):
    wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=bits)
    mlx_weights[f"{path}.weight"] = wq
    mlx_weights[f"{path}.scales"] = scales
    mlx_weights[f"{path}.biases"] = biases
    if bits != quant["bits"]:
        per_layer[path] = {"group_size": GROUP, "bits": bits, "mode": "affine"}
    return mx.dequantize(wq, scales, biases, group_size=GROUP, bits=bits)


main = {k: to_mx(v) for k, v in weights.items() if not k.startswith("model.layers.4.")}
dequantized = {}
for name, w in main.items():
    path = name.removesuffix(".weight")
    if name.endswith("kv_b_proj.weight"):
        v = w.reshape(heads, nope + v_dim, -1)
        attn = path.removesuffix(".kv_b_proj")
        wk = quantize(f"{attn}.embed_q", mx.contiguous(v[:, :nope, :].swapaxes(-1, -2)), 5)
        wv = quantize(f"{attn}.unembed_out", mx.contiguous(v[:, nope:, :]), 5)
        dequantized[name] = mx.concatenate([wk.swapaxes(-1, -2), wv], axis=1).reshape(w.shape)
    elif ".mlp.experts." in name:
        continue
    elif w.ndim == 2 and not name.endswith("mlp.gate.weight"):
        bits = 4 if name in ("model.embed_tokens.weight", "lm_head.weight") else 5
        dequantized[name] = quantize(path, w, bits)
    else:
        mlx_weights[name] = w
        dequantized[name] = w
for layer in range(config["num_hidden_layers"]):
    prefix = f"model.layers.{layer}.mlp"
    if f"{prefix}.experts.0.gate_proj.weight" not in main:
        continue
    for proj, bits in (("gate_proj", 4), ("up_proj", 4), ("down_proj", 5)):
        stacked = mx.stack(
            [main[f"{prefix}.experts.{e}.{proj}.weight"] for e in range(config["n_routed_experts"])]
        )
        deq = quantize(f"{prefix}.switch_mlp.{proj}", stacked, bits)
        for e in range(config["n_routed_experts"]):
            dequantized[f"{prefix}.experts.{e}.{proj}.weight"] = deq[e]

reference = GlmMoeDsaForCausalLM(hf_config).eval()
state = {}
for name, value in model.state_dict().items():
    if ".mlp.experts.gate_up_proj" in name:
        prefix = name.removesuffix("gate_up_proj")
        gate = [dequantized[f"{prefix}{e}.gate_proj.weight"] for e in range(4)]
        up = [dequantized[f"{prefix}{e}.up_proj.weight"] for e in range(4)]
        merged = mx.stack([mx.concatenate([g, u], axis=0) for g, u in zip(gate, up)])
        state[name] = torch.tensor(np.array(merged))
    elif ".mlp.experts.down_proj" in name:
        prefix = name.removesuffix("down_proj")
        merged = mx.stack([dequantized[f"{prefix}{e}.down_proj.weight"] for e in range(4)])
        state[name] = torch.tensor(np.array(merged))
    else:
        state[name] = torch.tensor(np.array(dequantized[name]))
reference.load_state_dict(state)
gaps = BoundaryGaps(config["index_topk"])
for module in reference.modules():
    if isinstance(module, GlmMoeDsaIndexer):
        module.register_forward_hook(gaps, with_kwargs=True)
with torch.no_grad():
    mixed_logits = reference(ids).logits
assert gaps.smallest > 1e-3, "index score ties make the top-k selection ambiguous"

mixed_config = copy.deepcopy(config)
mixed_config["num_nextn_predict_layers"] = 0
mixed_config["quantization"] = {**quant, **per_layer}
mixed_config["quantization_config"] = {**quant, **per_layer}
save(
    "glm_moe_dsa_mixed",
    mixed_config,
    mlx_weights,
    {"input_ids": mx.array(ids.numpy().astype(np.int32)), "logits": to_mx(mixed_logits)},
)
