"""Tiny Muse Spark Open fixtures from mlx-lm's `muse_spark_open`.

Writes three fixtures that share one set of random weights:

- `muse_spark_meta`: Meta's reference names and config (`model_type` `muse_spark_open`).
- `muse_spark_hf`: the transformers names and config (`model_type` `muse_spark`), with Q and K
  permuted to the split-half RoPE layout as the transformers conversion does.
- `muse_spark_nvfp4`: Meta's NVFP4 layout, with the experts of the inner layers in nvfp4 with
  a per-expert global scale and the first and last layers' experts in bf16.

The layers alternate sliding (RoPE) and full (NoPE) attention. The prompt is longer than the
sliding window.
"""

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models import muse_spark_open as msp

ROOT = Path(__file__).resolve().parents[2] / "Tests/MLXLMTests/Resources/Parity"

HIDDEN = 64
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
LAYERS = 4
VOCAB = 128
EXPERTS = 8
TOP_K = 3
EXPERT_DIM = 32
EXPERT_HIDDEN = 48
WINDOW = 4
ROPE_THETA = 10000.0
SOFT_CAP = 4.0
BASE_WIDTH = 16
RMS_EPS = 1e-5
POST_EPS = 1e-8
ROUTE_SCALE = 1.5

meta_config = {
    "model_type": "muse_spark_open",
    "hidden_size": HIDDEN,
    "num_attention_heads": HEADS,
    "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "num_hidden_layers": LAYERS,
    "vocab_size": VOCAB,
    "rms_norm_eps": RMS_EPS,
    "rms_post_norm_eps": POST_EPS,
    "rope_theta": ROPE_THETA,
    "every_n_layers_nope": 2,
    "global_attn_cfg": [WINDOW, 0],
    "output_soft_cap_temp": SOFT_CAP,
    "ffn_dim_multiplier": 2.0,
    "max_position_embeddings": 1024,
    "moe": {
        "is_enabled": True,
        "num_experts": EXPERTS,
        "topk": TOP_K,
        "route_scale": ROUTE_SCALE,
        "router_score_division_eps": 1e-15,
        "router_in_fp32": True,
        "compression_factor": HIDDEN // EXPERT_DIM,
        "override_routed_hidden_dim": EXPERT_HIDDEN,
        "n_dense_layers_before_moe": 0,
        "interleave_moe_layer_step": 1,
    },
    "metap": {
        "use_metap": True,
        "router_has_one_inf_dim": True,
        "base_width": BASE_WIDTH,
        "metap_mode": "sp",
        "m_emb": 1.0,
    },
}

layer_types = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(LAYERS)]
hf_text_config = {
    "model_type": "muse_spark_text",
    "hidden_size": HIDDEN,
    "intermediate_size": 2 * HIDDEN,
    "num_attention_heads": HEADS,
    "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "num_hidden_layers": LAYERS,
    "vocab_size": VOCAB,
    "rms_norm_eps": RMS_EPS,
    "post_norm_eps": POST_EPS,
    "rope_parameters": {"rope_type": "default", "rope_theta": ROPE_THETA},
    "layer_rope_theta": [ROPE_THETA if t == "sliding_attention" else 0 for t in layer_types],
    "layer_types": layer_types,
    "mlp_layer_types": ["sparse"] * LAYERS,
    "sliding_window": WINDOW,
    "final_logit_softcapping": SOFT_CAP,
    "output_multiplier": (HIDDEN / BASE_WIDTH) ** -0.5,
    "qk_scale_factor": 43.7840518911 / HEAD_DIM**0.5,
    "residual_gate_temperature": 0.3,
    "num_local_experts": EXPERTS,
    "num_experts_per_tok": TOP_K,
    "num_shared_experts": 0,
    "moe_hidden_size": EXPERT_DIM,
    "moe_intermediate_size": EXPERT_HIDDEN,
    "routed_scaling_factor": ROUTE_SCALE,
    "max_position_embeddings": 1024,
    "tie_word_embeddings": False,
}
hf_config = {
    "architectures": ["MuseSparkForConditionalGeneration"],
    "model_type": "muse_spark",
    "text_config": hf_text_config,
}


def normal(*shape, scale=1.0, dtype=mx.bfloat16):
    return (mx.random.normal(shape) * scale).astype(dtype)


def make_raw(seed):
    """Weights in Meta's reference names, before mlx-lm's sanitize."""
    mx.random.seed(seed)
    raw = {
        "model.embed_tokens.weight": normal(VOCAB, HIDDEN),
        "model.norm.weight": (1 + normal(HIDDEN, scale=0.1, dtype=mx.float32)).astype(mx.bfloat16),
        "lm_head.weight": normal(VOCAB, HIDDEN, scale=HIDDEN**-0.5 * 3),
    }
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        s = HIDDEN**-0.5
        raw |= {
            p + "self_attn.pre_qkv_proj_norm.weight": normal(HIDDEN, scale=0.1),
            p + "self_attn.q_proj.weight": normal(HEADS * HEAD_DIM, HIDDEN, scale=s),
            p + "self_attn.k_proj.weight": normal(KV_HEADS * HEAD_DIM, HIDDEN, scale=s),
            p + "self_attn.v_proj.weight": normal(KV_HEADS * HEAD_DIM, HIDDEN, scale=s),
            p + "self_attn.o_gate_proj.weight": normal(HEADS * HEAD_DIM, HIDDEN, scale=s),
            p + "self_attn.o_proj.weight": normal(HIDDEN, HEADS * HEAD_DIM, scale=s),
            p + "pre_ffn_norm.weight": normal(HIDDEN, scale=0.1),
            p + "post_ffn_norm.weight": normal(HIDDEN, scale=0.1),
            p + "add_attn.gate": normal(HIDDEN, scale=0.5, dtype=mx.float32),
            p + "add_ffn.gate": normal(HIDDEN, scale=0.5, dtype=mx.float32),
            # Meta stores the router as (hidden, experts).
            p + "feed_forward.router.weight": normal(HIDDEN, EXPERTS, scale=4 * s, dtype=mx.float32),
            p + "feed_forward.expert_bias": normal(EXPERTS, scale=0.3, dtype=mx.float32),
            p + "feed_forward.pooler.weight": normal(EXPERT_DIM, HIDDEN, scale=s),
            p + "feed_forward.unpooler.weight": normal(HIDDEN, EXPERT_DIM, scale=EXPERT_DIM**-0.5),
            p + "feed_forward.pre_expert_norm.weight": normal(EXPERT_DIM, scale=0.1),
            p + "feed_forward.post_expert_norm.weight": normal(EXPERT_DIM, scale=0.1),
            p + "feed_forward.experts.w13_weight": normal(
                EXPERTS, 2 * EXPERT_HIDDEN, EXPERT_DIM, scale=EXPERT_DIM**-0.5
            ),
            p + "feed_forward.experts.w2_weight": normal(
                EXPERTS, EXPERT_DIM, EXPERT_HIDDEN, scale=EXPERT_HIDDEN**-0.5
            ),
        }

    return raw


ids = mx.array([[1, 5, 9, 2, 77, 31, 8, 100, 4, 64, 3, 27]], dtype=mx.int32)


def build_model(raw):
    model = msp.Model(msp.ModelArgs.from_dict(meta_config))
    model.load_weights(list(model.sanitize(dict(raw)).items()), strict=True)
    return model


def routing_margin(model):
    """Smallest gap between the last chosen and the first unchosen biased expert score.

    Top-k selection is discontinuous, so a fixture with near ties fails on rounding noise.
    """
    gaps = []
    routing = msp.expert_routing

    def probe(logits, bias, *, top_k, multiplier, **kwargs):
        biased = mx.sort(mx.sigmoid(logits * multiplier) + bias, axis=-1)
        gaps.append((biased[..., -top_k] - biased[..., -top_k - 1]).min())
        return routing(logits, bias, top_k=top_k, multiplier=multiplier, **kwargs)

    msp.expert_routing = probe
    try:
        mx.eval(model(ids))
    finally:
        msp.expert_routing = routing
    return min(g.item() for g in gaps)


def quantize_experts(model):
    """Nvfp4 experts with a global scale in the inner layers, as Meta's NVFP4 build has."""
    for i in range(1, LAYERS - 1):
        switch = model.model.layers[i].feed_forward.switch_mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer = getattr(switch, name)
            setattr(switch, name, layer.to_quantized(16, 4, "nvfp4", global_scale=True))
    return model


def run(model):
    full = model(ids).astype(mx.float32)
    # The same tokens one at a time through the cache must agree with the full pass.
    cache = model.make_cache()
    steps = [model(ids[:, :5], cache=cache)]
    for t in range(5, ids.shape[1]):
        steps.append(model(ids[:, t : t + 1], cache=cache))
    stepped = mx.concatenate(steps, axis=1).astype(mx.float32)
    diff = mx.abs(stepped - full).max().item()
    print(f"mlx-lm cached decode vs prefill: {diff:.5f}")
    assert diff < 0.1
    return full


def write(name, config, weights, logits):
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    mx.save_safetensors(
        str(out / "expected.safetensors"), {"input_ids": ids, "logits": logits}
    )
    print(f"wrote {out}: max |logit| {mx.abs(logits).max().item():.3f}")


def permute_for_rope(w, heads):
    """Interleaved-pair RoPE rows to split-half rows, as the transformers conversion does."""
    rows, cols = w.shape
    return w.reshape(heads, rows // heads // 2, 2, cols).swapaxes(1, 2).reshape(rows, cols)


def to_hf(weights):
    renames = {
        "self_attn.pre_qkv_proj_norm.weight": "input_layernorm.weight",
        "self_attn.v_proj.weight": "self_attn.v_proj.weight",
        "self_attn.o_proj.weight": "self_attn.o_proj.weight",
        "self_attn.o_gate_proj.weight": "self_attn.gate_proj.weight",
        "pre_ffn_norm.weight": "pre_feedforward_layernorm.weight",
        "post_ffn_norm.weight": "post_feedforward_layernorm.weight",
        "add_attn.gate": "post_attention_residual_gate.gate",
        "add_ffn.gate": "post_feedforward_residual_gate.gate",
        "feed_forward.expert_bias": "mlp.gate.e_score_correction_bias",
        "feed_forward.experts.w13_weight": "mlp.experts.gate_up_proj",
        "feed_forward.experts.w2_weight": "mlp.experts.down_proj",
        "feed_forward.pre_expert_norm.weight": "mlp.pre_expert_norm.weight",
        "feed_forward.post_expert_norm.weight": "mlp.experts.post_expert_norm.weight",
        "feed_forward.pooler.weight": "mlp.pre_expert_proj.weight",
        "feed_forward.unpooler.weight": "mlp.post_expert_proj.weight",
    }
    out = {
        "model.language_model.embed_tokens.weight": weights["model.embed_tokens.weight"],
        "model.language_model.norm.weight": weights["model.norm.weight"],
        "lm_head.weight": weights["lm_head.weight"],
    }
    for i in range(LAYERS):
        src, dst = f"model.layers.{i}.", f"model.language_model.layers.{i}."
        for k, v in renames.items():
            out[dst + v] = weights[src + k]
        out[dst + "self_attn.q_proj.weight"] = permute_for_rope(
            weights[src + "self_attn.q_proj.weight"], HEADS
        )
        out[dst + "self_attn.k_proj.weight"] = permute_for_rope(
            weights[src + "self_attn.k_proj.weight"], KV_HEADS
        )
        out[dst + "mlp.gate.weight"] = weights[src + "feed_forward.router.weight"].T
    return out


# Pick weights whose routing is far from ties in both references.
for seed in range(100):
    raw = make_raw(seed)
    margins = (
        routing_margin(build_model(raw)),
        routing_margin(quantize_experts(build_model(raw))),
    )
    if min(margins) > 0.01:
        break
else:
    raise RuntimeError("no seed with a clear routing margin")
print(f"seed {seed}: routing margins {margins[0]:.4f} (bf16) {margins[1]:.4f} (nvfp4)")

logits = run(build_model(raw))
write("muse_spark_meta", meta_config, raw, logits)
write("muse_spark_hf", hf_config, to_hf(raw), logits)

# Meta's NVFP4 checkpoint layout: packed uint8 weights, e4m3 block scales, and the global
# scale stored divided by 6 * 448.
model = quantize_experts(build_model(raw))
nvfp4 = dict(raw)
for i in range(1, LAYERS - 1):
    switch = model.model.layers[i].feed_forward.switch_mlp
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
    p = f"model.layers.{i}.feed_forward.experts."
    scale_2 = msp._NVFP4_SCALE_2_TO_GLOBAL
    nvfp4[p + "w13_weight"] = mx.concatenate(
        [gate.weight.view(mx.uint8), up.weight.view(mx.uint8)], axis=1
    )
    nvfp4[p + "w13_weight_scale"] = mx.concatenate([gate.scales, up.scales], axis=1)
    nvfp4[p + "w13_weight_scale_2"] = (
        mx.stack([gate.global_scale, up.global_scale], axis=1) / scale_2
    )
    nvfp4[p + "w2_weight"] = down.weight.view(mx.uint8)
    nvfp4[p + "w2_weight_scale"] = down.scales
    nvfp4[p + "w2_weight_scale_2"] = down.global_scale / scale_2
    # Activation scales, which the MLX path ignores.
    nvfp4[p + "w13_input_scale"] = mx.ones((EXPERTS, 2), dtype=mx.float32)
    nvfp4[p + "w2_input_scale"] = mx.ones((EXPERTS,), dtype=mx.float32)
assert any("global_scale" in k for k, _ in tree_flatten(model.parameters()))
write("muse_spark_nvfp4", meta_config, nvfp4, run(model))
