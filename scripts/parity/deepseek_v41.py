"""Tiny DeepSeek-V4.1 (`deepseek_v41`) fixture in DeepSeek's native checkpoint format.

The checkpoint uses the published layout: FP8 E4M3 weights with UE8M0 scales per 32x32 tile, FP4
E2M1 routed experts packed two per byte with one UE8M0 scale per 32 inputs, Engram tables in FP8
with per-row scales, and the vision, MTP and `bias_vl` tensors a text model drops.

The reference is mlx-lm's `deepseek_v41` model, which repacks the FP8/FP4 weights as MLX
mxfp8/mxfp4 without requantizing. mlx-lm rejects Engram layers, so this script adds Engram as in
DeepSeek's `inference/engram.py` and `inference/model.py`: hash the n-grams ending at each
position, look up one row per (n-gram size, head), and gate the projected value into each
hyper-connection stream.

The config exercises every attention path: window-only layers, two ratio-2 KV sources with
their own indexers and a layer that reuses them, a ratio-1 source that also picks candidate
blocks, and an index-only layer that reads the source's index keys and candidates. The window is
shorter than the sequence and `index_topk` is smaller than the compressed length, so eviction
and sparse selection both change the logits.

Pass `--deepseek-inference <dir>` (DeepSeek's `inference/` directory) to also check the n-gram
hashes against `engram.NgramHashState`.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models import deepseek_v41 as ds
from safetensors.torch import save_file
from tokenizers import Tokenizer, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepseek_v41_engram_tables import (  # noqa: E402
    build_compressed_token_map,
    compute_hash_multipliers,
)

ROOT = Path(__file__).resolve().parents[2] / "Tests/MLXLMTests/Resources/Parity"
NAME = "deepseek_v41"

# Token texts with case, accent and whitespace variants, so the compressed map merges some ids.
WORDS = ["the", "The", " THE", "café", "Cafe", "CAFÉ", "naïve", "naive", " ", "  ", "\n", "\t "]
VOCAB = [f"<s{i}>" for i in range(4)] + WORDS + [f"w{i}" for i in range(48)]
IDS = [0, 5, 9, 2, 17, 31, 8, 40, 3, 22, 12, 50, 45, 7, 4, 6, 33, 61, 18, 27, 9, 4, 5, 11]

text = {
    "model_type": "deepseek_v41_text",
    "vocab_size": len(VOCAB),
    "hidden_size": 32,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 10,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "qk_rope_head_dim": 16,
    "q_lora_rank": 32,
    "o_lora_rank": 32,
    "o_groups": 2,
    "hidden_act": "silu",
    "swiglu_limit": 1.0,
    "rms_norm_eps": 1e-20,
    "attention_bias": False,
    "tie_word_embeddings": False,
    "max_position_embeddings": 1024,
    "rope_theta": 10000,
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 4,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 16,
    },
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.5,
    "sliding_window": 4,
    # Trailing entries for the DSpark stages, as the published config has.
    "compress_ratios": [0, 0, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0],
    "compress_rope_theta": 160000,
    "kv_source_layer_ids": [2, 4, 5],
    "index_source_layer_ids": [2, 4, 5, 7],
    "index_n_heads": 8,
    "index_head_dim": 32,
    "index_topk": 3,
    "candidate_source_layer_id": 5,
    "candidate_topk_blocks": 3,
    "candidate_block_size": 2,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "engram_layer_ids": [1, 4],
    "engram_max_ngram_size": 3,
    "engram_vocab_size": 50,
    "engram_n_heads": 2,
    "engram_head_dim": 32,
    "engram_pad_token_id": 2,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_noise_token_id": len(VOCAB) - 1,
    "dspark_target_layer_ids": [7, 8, 9],
    "dspark_markov_rank": 16,
    "dspark_n_routed_experts": 4,
    "dspark_num_experts_per_tok": 2,
}

FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


# MARK: - Native quantization


def pow2_scale(amax, maximum):
    return np.exp2(np.ceil(np.log2(np.maximum(amax, 1e-30) / maximum))).astype(np.float32)


def fp8_tiles(w):
    """FP8 E4M3 with one UE8M0 scale per 32x32 tile. Returns (fp8, scale, dequantized)."""
    out, inp = w.shape
    tiles = w.reshape(out // 32, 32, inp // 32, 32)
    scale = pow2_scale(np.abs(tiles).max(axis=(1, 3)), 448.0)
    q = torch.tensor(tiles / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    deq = (q.float().numpy() * scale[:, None, :, None]).reshape(out, inp)
    q = q.reshape(out, inp)
    return q, torch.tensor(scale).to(torch.float8_e8m0fnu), deq


def fp8_rows(w):
    """FP8 E4M3 with one UE8M0 scale per row and 32 columns, as Engram tables use."""
    rows, dim = w.shape
    blocks = w.reshape(rows, dim // 32, 32)
    scale = pow2_scale(np.abs(blocks).max(axis=-1), 448.0)
    q = torch.tensor(blocks / scale[..., None]).to(torch.float8_e4m3fn)
    deq = (q.float().numpy() * scale[..., None]).reshape(rows, dim)
    return q.reshape(rows, dim), torch.tensor(scale).to(torch.float8_e8m0fnu), deq


def fp4_rows(w):
    """FP4 E2M1 packed two per byte (even element in the low nibble), UE8M0 per 32 inputs."""
    out, inp = w.shape
    blocks = w.reshape(out, inp // 32, 32)
    scale = pow2_scale(np.abs(blocks).max(axis=-1), 6.0)
    v = np.clip(blocks / scale[..., None], -6, 6)
    code = np.abs(np.abs(v)[..., None] - FP4_VALUES).argmin(axis=-1).astype(np.uint8)
    code |= (v < 0).astype(np.uint8) << 3
    deq = (np.sign(v) * FP4_VALUES[code & 7] * scale[..., None]).reshape(out, inp)
    code = code.reshape(out, inp)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return (
        torch.tensor(packed.view(np.int8)),
        torch.tensor(scale).to(torch.float8_e8m0fnu),
        deq.astype(np.float32),
    )


# MARK: - Engram reference (DeepSeek inference/engram.py)


def next_primes(args):
    """The bucket modulus of each (layer, n-gram size, head): primes drawn in order, never reused."""
    seen, layers = set(), []
    for _ in args["engram_layer_ids"]:
        per_ngram = []
        for _ in range(args["engram_max_ngram_size"] - 1):
            sizes, current = [], args["engram_vocab_size"] - 1
            for _ in range(args["engram_n_heads"]):
                current += 1
                while current in seen or any(current % d == 0 for d in range(2, int(current**0.5) + 1)):
                    current += 1
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        layers.append(per_ngram)
    return np.array(layers, dtype=np.int64)


def engram_hashes(ids, token_map, multipliers, primes, pad):
    """Hash ids [B, L, layers, (max_ngram - 1) * heads] of the n-grams ending at each position."""
    batch, length = ids.shape
    n = multipliers.shape[1]
    compressed = token_map[ids]
    tokens = []
    for shift in range(n):
        source = np.zeros_like(compressed)
        source[:, shift:] = compressed[:, : length - shift]
        blocked = np.arange(length)[None] < shift
        tokens.append(np.where(blocked, pad, source))
    tokens = np.stack(tokens, axis=-1).astype(np.int64)
    products = tokens[:, :, None, :] * multipliers
    flat = primes.reshape(primes.shape[0], -1)
    offsets = np.concatenate([np.zeros((flat.shape[0], 1), np.int64), np.cumsum(flat, 1)[:, :-1]], 1)
    rolling, hashes = products[..., 0], []
    for i in range(1, n):
        rolling = np.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling[..., None] % primes[:, i - 1])
    return np.concatenate(hashes, axis=-1) + offsets


class Engram(nn.Module):
    def __init__(self, args, table):
        super().__init__()
        self._table = table
        self.hc, self.dim, self.eps = args.hc_mult, args.hidden_size, args.rms_norm_eps
        cols = (text["engram_max_ngram_size"] - 1) * text["engram_n_heads"]
        self.wkv = nn.Linear(cols * text["engram_head_dim"], self.dim * (self.hc + 1), bias=False)
        self.q_weight = mx.ones((self.hc, self.dim))
        self.k_weight = mx.ones((self.hc, self.dim))

    def __call__(self, streams, hash_ids):
        batch, length = hash_ids.shape[:2]
        kv = self.wkv(self._table[hash_ids].reshape(batch, length, -1))
        key = kv[..., : self.hc * self.dim].reshape(batch, length, self.hc, self.dim)
        value = kv[..., self.hc * self.dim :]
        key = key.astype(mx.float32)
        h = streams.astype(mx.float32)
        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)
        rstd = mx.rsqrt(mx.mean(h * h, -1) + self.eps) * mx.rsqrt(mx.mean(key * key, -1) + self.eps)
        dot = mx.sum(h * weight * key, -1) * rstd * self.dim**-0.5
        magnitude = mx.sqrt(mx.maximum(mx.abs(dot), 1e-6))
        gate = mx.sigmoid(mx.where(dot < 0, -magnitude, magnitude))
        out = h + gate[..., None] * value.astype(mx.float32)[:, :, None, :]
        return out.astype(streams.dtype)


def forward(model, engrams, ids, hashes, cache, captures=None):
    """mlx-lm's TextModel.__call__ without pipelining, with Engram before its layers.

    `captures` collects the stream mean at the input of each DSpark target layer.
    """
    inner, args = model.model, model.args
    h = inner.embed_tokens(ids)
    batch, length, _ = h.shape
    start_pos = cache[0].offset
    previous = 0 if cache[0].window is None else cache[0].window.shape[1]
    window_idx = ds._window_indices(previous=previous, length=length, window=args.sliding_window)
    window_idx = mx.broadcast_to(window_idx, (batch, *window_idx.shape))
    streams = mx.broadcast_to(h[:, :, None, :], (batch, length, args.hc_mult, args.hidden_size))
    pre_mix = mx.zeros((batch, length, args.hc_mult), dtype=mx.float32)
    pre_mix[..., 0] = 1.0
    shared = {}
    for i, (layer, layer_cache) in enumerate(zip(inner.layers, cache)):
        if i in engrams:
            slot = text["engram_layer_ids"].index(i)
            streams = engrams[i](streams, hashes[:, :, slot])
        if captures is not None and i in text["dspark_target_layer_ids"]:
            captures.append(mx.mean(streams, axis=2))
        streams, pre_mix = layer(
            streams, pre_mix, start_pos=start_pos, window_idx=window_idx, cache=layer_cache,
            shared=shared,
        )
    return model.lm_head(inner.norm(ds._collapse(streams, pre_mix)))


# MARK: - DSpark reference (DeepSeek inference/model.py)


def dspark_attention(attn, x, start, window):
    """DSparkAttention for a draft block at positions `start...`: every block query sees the
    window of target KV and the whole block."""
    batch, length, _ = x.shape
    rd = attn.rope_dim
    q = attn.q_b_proj(attn.q_a_norm(attn.q_a_proj(x))).reshape(batch, length, attn.num_heads, -1)
    q = ds._apply_rope(q, attn.rope, rope_dim=rd, offset=start)
    kv = ds._apply_rope(attn.kv_norm(attn.kv_proj(x)), attn.rope, rope_dim=rd, offset=start)
    keys = mx.concatenate([window, ds._fake_quant_fp8(kv, 32)], axis=1)[:, None].astype(mx.float32)
    out = mx.fast.scaled_dot_product_attention(
        q.transpose(0, 2, 1, 3).astype(mx.float32), keys, keys, scale=attn.head_dim**-0.5,
        sinks=attn.sinks.astype(mx.float32),
    )
    out = out.transpose(0, 2, 1, 3).astype(q.dtype)
    out = ds._apply_rope(out, attn.rope, rope_dim=rd, offset=start, inverse=True)
    out = attn.o_a_proj(out.reshape(batch * length, attn.o_groups, 1, -1), attn._group_ids)
    return attn.o_b_proj(out.reshape(batch, length, -1))


class DSpark:
    """The DSpark block drafter stored under `mtp.*`: `main_proj` of the target layers' stream
    means seeds each stage's window KV; a block of the bonus token plus noise tokens runs
    through the stages; the shared head plus a Markov bigram bias proposes each next token."""

    def __init__(self, config, target, loaded):
        cfg = json.loads(json.dumps(config))
        cfg["text_config"]["n_routed_experts"] = text["dspark_n_routed_experts"]
        cfg["text_config"]["num_experts_per_tok"] = text["dspark_num_experts_per_tok"]
        args = ds.ModelArgs.from_dict(cfg)
        base = text["num_hidden_layers"]
        self.target, self.args = target, args
        self.stages = [ds.DecoderLayer(args, base + s) for s in range(text["num_nextn_predict_layers"])]
        own = ("main_proj", "main_norm", "norm", "markov_head", "confidence_head")
        stage_weights = {
            "layers." + k.split(".", 1)[1]: v
            for k, v in loaded.items()
            if k.split(".")[2] not in own
        }
        weights = ds.Model.sanitize(SimpleNamespace(args=args), stage_weights)
        for s, stage in enumerate(self.stages):
            prefix = f"model.layers.{s}."
            params = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}

            def predicate(path, module, params=params):
                if f"{path}.scales" not in params:
                    return False
                bits = 4 if ".switch_mlp." in path else 8
                return {"group_size": 32, "bits": bits, "mode": f"mxfp{bits}"}

            nn.quantize(stage, class_predicate=predicate)
            stage.load_weights(list(params.items()), strict=True)
        last = len(self.stages) - 1
        self.main_proj = nn.Linear(3 * args.hidden_size, args.hidden_size, bias=False).to_quantized(
            group_size=32, bits=8, mode="mxfp8"
        )
        w, s = ds._repack(loaded["mtp.0.main_proj.weight"], loaded["mtp.0.main_proj.scale"])
        self.main_proj.load_weights([("weight", w), ("scales", s)], strict=True)
        self.main_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.main_norm.weight = loaded["mtp.0.main_norm.weight"]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.norm.weight = loaded[f"mtp.{last}.norm.weight"]
        self.markov_embed = loaded[f"mtp.{last}.markov_head.embed.weight"]
        self.markov_head = loaded[f"mtp.{last}.markov_head.head.weight"]
        self.windows = [None] * len(self.stages)

    def seed(self, hidden, start):
        """Appends the KV of target positions `start...` to every stage's window."""
        main_x = self.main_norm(self.main_proj(hidden))
        for s, stage in enumerate(self.stages):
            attn = stage.self_attn
            kv = ds._apply_rope(attn.kv_norm(attn.kv_proj(main_x)), attn.rope, rope_dim=attn.rope_dim, offset=start)
            kv = ds._fake_quant_fp8(kv, 32)
            window = kv if self.windows[s] is None else mx.concatenate([self.windows[s], kv], axis=1)
            self.windows[s] = window[:, -self.args.sliding_window :]

    def draft(self, bonus, start):
        """Draft logits (with the Markov bias) and greedy tokens for the block at `start`."""
        args = self.args
        ids = mx.array([[bonus] + [text["dspark_noise_token_id"]] * (text["dspark_block_size"] - 1)])
        h = self.target.model.embed_tokens(ids)
        streams = mx.broadcast_to(h[:, :, None, :], (1, ids.shape[1], args.hc_mult, args.hidden_size))
        pre_mix = mx.zeros((1, ids.shape[1], args.hc_mult), dtype=mx.float32)
        pre_mix[..., 0] = 1.0
        for stage, window in zip(self.stages, self.windows):
            attn_pre, attn_post, attn_comb = stage.attn_hc(streams)
            x = stage.input_layernorm(ds._collapse(streams, pre_mix))
            x = dspark_attention(stage.self_attn, x, start, window)
            streams = ds._expand(x, streams, attn_post, attn_comb)
            ffn_pre, ffn_post, ffn_comb = stage.ffn_hc(streams)
            x = stage.mlp(stage.post_attention_layernorm(ds._collapse(streams, attn_pre)))
            streams = ds._expand(x, streams, ffn_post, ffn_comb)
            pre_mix = ffn_pre
        logits = self.target.lm_head(self.norm(ds._collapse(streams, pre_mix))).astype(mx.float32)
        tokens, rows = [bonus], []
        for i in range(ids.shape[1]):
            bias = self.markov_embed[mx.array([tokens[-1]])] @ self.markov_head.T
            row = logits[:, i] + bias
            rows.append(row)
            tokens.append(mx.argmax(row, axis=-1).item())
        return mx.stack(rows, axis=1), tokens[1:]


# MARK: - Top-k tie guard


class TopKGaps:
    """Records the smallest score gap at every top-k boundary, so ties cannot flip a selection."""

    def __init__(self):
        self.smallest = float("inf")
        self.original = mx.argpartition

    def __call__(self, a, kth, axis=-1):
        scores = -np.array(a.astype(mx.float32)).reshape(-1, a.shape[-1])
        for row in scores:
            valid = np.sort(row[row > -np.inf])[::-1]
            if valid.size > kth + 1 and np.isfinite(valid[kth]):
                self.smallest = min(self.smallest, float(valid[kth] - valid[kth + 1]))
        return self.original(a, kth, axis=axis)


def rounding_sensitivity(run, logits, trials=4, eps=3e-6):
    """The largest logit change when every FP8/FP4 rounding input moves by ~`eps` relative.

    Rounding is constant unless a value sits near a rounding boundary, so a stable fixture
    shows no change. Kernel noise between implementations is about 1e-6 relative.
    """
    originals = (ds._fake_quant_fp8, ds._fake_quant_fp4)
    key = mx.random.key(1234)
    worst = 0.0

    def jittered(function):
        def wrapped(x, *args, **kwargs):
            nonlocal key
            key, sub = mx.random.split(key)
            noise = 1 + eps * mx.random.normal(x.shape, key=sub)
            return function((x * noise).astype(x.dtype), *args, **kwargs)

        return wrapped

    for _ in range(trials):
        ds._fake_quant_fp8, ds._fake_quant_fp4 = (jittered(f) for f in originals)
        try:
            other = run()
        finally:
            ds._fake_quant_fp8, ds._fake_quant_fp4 = originals
        worst = max(worst, mx.abs(other - logits).max().item())
    return worst


# MARK: - Fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek-inference", type=Path)
    options = parser.parse_args()

    tokenizer = Tokenizer(models.WordLevel({t: i for i, t in enumerate(VOCAB)}, unk_token="<s1>"))
    token_map, compressed_size = build_compressed_token_map(tokenizer)
    text["engram_compressed_vocab_size"] = compressed_size
    multipliers = compute_hash_multipliers(
        text["engram_layer_ids"], text["engram_max_ngram_size"], compressed_size
    )
    primes = next_primes(text)
    text["engram_num_embeddings"] = [int(p.sum()) for p in primes]
    token_map = np.array(token_map, dtype=np.int64)
    pad = int(token_map[text["engram_pad_token_id"]])
    ids = np.array([IDS], dtype=np.int64)
    hashes = engram_hashes(ids, token_map, multipliers, primes, pad)
    print(f"compressed vocab {len(VOCAB)} -> {compressed_size}, tables {text['engram_num_embeddings']}")

    if options.deepseek_inference:
        check_against_deepseek(options.deepseek_inference, tokenizer, ids, hashes)

    for seed in range(64):
        if write_fixture(seed, token_map, multipliers, ids, hashes):
            return
    raise SystemExit("no seed in range avoids top-k ties")


def write_fixture(seed, token_map, multipliers, ids, hashes):
    """Writes the fixture for `seed`; returns False when a top-k boundary is too close to call."""
    rng = np.random.default_rng(seed)
    native = {}  # DeepSeek names -> torch tensors
    dense = {}  # DeepSeek names -> dequantized float32, for the Engram reference

    def normal(*shape, scale=1.0):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def matrix(out, inp):
        # Uneven tile magnitudes give every tile its own scale.
        tiles = np.exp2(rng.integers(-2, 3, (out // 32, inp // 32))).astype(np.float32)
        return normal(out, inp, scale=inp**-0.5) * np.repeat(np.repeat(tiles, 32, 0), 32, 1)

    def plain(name, value):
        native[name] = torch.tensor(value)
        dense[name] = value

    def fp8(name, value):
        q, s, deq = fp8_tiles(value)
        native[f"{name}.weight"], native[f"{name}.scale"] = q, s
        dense[f"{name}.weight"] = deq

    def fp4(name, value):
        q, s, deq = fp4_rows(value)
        native[f"{name}.weight"], native[f"{name}.scale"] = q, s
        dense[f"{name}.weight"] = deq

    d, hc = text["hidden_size"], text["hc_mult"]
    heads, hd = text["num_attention_heads"], text["head_dim"]
    inter, experts = text["moe_intermediate_size"], text["n_routed_experts"]
    mix = (2 + hc) * hc
    plain("embed.weight", normal(len(VOCAB), d))
    plain("head.weight", normal(len(VOCAB), d, scale=d**-0.5))
    plain("norm.weight", 1 + normal(d, scale=0.1))
    for i in range(text["num_hidden_layers"]):
        p = f"layers.{i}"
        plain(f"{p}.attn_norm.weight", 1 + normal(d, scale=0.1))
        plain(f"{p}.ffn_norm.weight", 1 + normal(d, scale=0.1))
        for kind in ("attn", "ffn"):
            plain(f"{p}.hc_{kind}_fn", normal(mix, hc * d, scale=(hc * d) ** -0.5))
            plain(f"{p}.hc_{kind}_base", normal(mix, scale=0.5))
            plain(f"{p}.hc_{kind}_scale", 1 + normal(3, scale=0.2))
        a = f"{p}.attn"
        plain(f"{a}.attn_sink", normal(heads))
        fp8(f"{a}.wq_a", matrix(text["q_lora_rank"], d))
        plain(f"{a}.q_norm.weight", 1 + normal(text["q_lora_rank"], scale=0.1))
        fp8(f"{a}.wq_b", matrix(heads * hd, text["q_lora_rank"]))
        fp8(f"{a}.wkv", matrix(hd, d))
        plain(f"{a}.kv_norm.weight", 1 + normal(hd, scale=0.1))
        fp8(f"{a}.wo_a", matrix(text["o_groups"] * text["o_lora_rank"], heads * hd // text["o_groups"]))
        fp8(f"{a}.wo_b", matrix(d, text["o_groups"] * text["o_lora_rank"]))
        ratio = text["compress_ratios"][i]
        if i in text["kv_source_layer_ids"]:
            plain(f"{a}.compressor.wkv.weight", normal(hd, d, scale=d**-0.5))
            if ratio > 1:
                plain(f"{a}.compressor.wgate.weight", normal(hd, d, scale=d**-0.5))
            plain(f"{a}.compressor.norm.weight", 1 + normal(hd, scale=0.1))
        if i in text["index_source_layer_ids"]:
            idx = f"{a}.indexer"
            fp8(f"{idx}.wq_b", matrix(text["index_n_heads"] * text["index_head_dim"], text["q_lora_rank"]))
            plain(f"{idx}.weights_proj.weight", normal(text["index_n_heads"], d, scale=d**-0.5))
            if i in text["kv_source_layer_ids"]:
                plain(f"{idx}.wk.weight", normal(text["index_head_dim"], hd, scale=hd**-0.5))
                plain(f"{idx}.k_norm.weight", 1 + normal(text["index_head_dim"], scale=0.1))
        f = f"{p}.ffn"
        plain(f"{f}.gate.weight", normal(experts, d, scale=d**-0.5))
        plain(f"{f}.gate.bias", normal(experts, scale=0.1))
        plain(f"{f}.gate.bias_vl", normal(experts, scale=5.0))
        for e in range(experts):
            fp4(f"{f}.experts.{e}.w1", matrix(inter, d))
            fp4(f"{f}.experts.{e}.w2", matrix(d, inter))
            fp4(f"{f}.experts.{e}.w3", matrix(inter, d))
        fp8(f"{f}.shared_experts.w1", matrix(inter, d))
        fp8(f"{f}.shared_experts.w2", matrix(d, inter))
        fp8(f"{f}.shared_experts.w3", matrix(inter, d))
        if i in text["engram_layer_ids"]:
            slot = text["engram_layer_ids"].index(i)
            rows = text["engram_num_embeddings"][slot]
            q, s, deq = fp8_rows(normal(rows, text["engram_head_dim"]))
            native[f"{p}.engram.embed.weight"], native[f"{p}.engram.embed.scale"] = q, s
            dense[f"{p}.engram.embed.weight"] = deq
            cols = (text["engram_max_ngram_size"] - 1) * text["engram_n_heads"]
            fp8(f"{p}.engram.wkv", matrix(d * (hc + 1), cols * text["engram_head_dim"]))
            plain(f"{p}.engram.q_weight", 1 + normal(hc, d, scale=0.3))
            plain(f"{p}.engram.k_weight", 1 + normal(hc, d, scale=0.3))
    # DSpark stages: window-only attention, their own MoE, and the drafter heads.
    draft_experts, rank = text["dspark_n_routed_experts"], text["dspark_markov_rank"]
    stages = text["num_nextn_predict_layers"]
    for s in range(stages):
        p = f"mtp.{s}"
        plain(f"{p}.attn_norm.weight", 1 + normal(d, scale=0.1))
        plain(f"{p}.ffn_norm.weight", 1 + normal(d, scale=0.1))
        for kind in ("attn", "ffn"):
            plain(f"{p}.hc_{kind}_fn", normal(mix, hc * d, scale=(hc * d) ** -0.5))
            plain(f"{p}.hc_{kind}_base", normal(mix, scale=0.5))
            plain(f"{p}.hc_{kind}_scale", 1 + normal(3, scale=0.2))
        a = f"{p}.attn"
        plain(f"{a}.attn_sink", normal(heads))
        fp8(f"{a}.wq_a", matrix(text["q_lora_rank"], d))
        plain(f"{a}.q_norm.weight", 1 + normal(text["q_lora_rank"], scale=0.1))
        fp8(f"{a}.wq_b", matrix(heads * hd, text["q_lora_rank"]))
        fp8(f"{a}.wkv", matrix(hd, d))
        plain(f"{a}.kv_norm.weight", 1 + normal(hd, scale=0.1))
        fp8(f"{a}.wo_a", matrix(text["o_groups"] * text["o_lora_rank"], heads * hd // text["o_groups"]))
        fp8(f"{a}.wo_b", matrix(d, text["o_groups"] * text["o_lora_rank"]))
        f = f"{p}.ffn"
        plain(f"{f}.gate.weight", normal(draft_experts, d, scale=d**-0.5))
        plain(f"{f}.gate.bias", normal(draft_experts, scale=0.1))
        plain(f"{f}.gate.bias_vl", normal(draft_experts, scale=5.0))
        for e in range(draft_experts):
            fp4(f"{f}.experts.{e}.w1", matrix(inter, d))
            fp4(f"{f}.experts.{e}.w2", matrix(d, inter))
            fp4(f"{f}.experts.{e}.w3", matrix(inter, d))
        fp8(f"{f}.shared_experts.w1", matrix(inter, d))
        fp8(f"{f}.shared_experts.w2", matrix(d, inter))
        fp8(f"{f}.shared_experts.w3", matrix(inter, d))
    fp8("mtp.0.main_proj", matrix(d, 3 * d))
    plain("mtp.0.main_norm.weight", 1 + normal(d, scale=0.1))
    last = f"mtp.{stages - 1}"
    plain(f"{last}.norm.weight", 1 + normal(d, scale=0.1))
    plain(f"{last}.markov_head.embed.weight", normal(len(VOCAB), rank))
    plain(f"{last}.markov_head.head.weight", normal(len(VOCAB), rank, scale=rank**-0.5))
    plain(f"{last}.confidence_head.proj.weight", normal(1, d + rank, scale=0.1))
    draft = {k: native.pop(k) for k in [k for k in native if k.startswith("mtp.")]}

    # Tensors a text model drops.
    plain("mtp.0.attn_norm.weight", np.ones(d, np.float32))
    plain("vision.norm.weight", np.ones(8, np.float32))
    plain("image_start", np.ones(d, np.float32))

    out = ROOT / NAME
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in native.items()}, str(out / "model.safetensors"))
    save_file(
        {"token_map": torch.tensor(token_map, dtype=torch.int32),
         "multipliers": torch.tensor(multipliers)},
        str(out / "deepseek_v41_engram.safetensors"),
    )
    config = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        "text_config": text,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    draft_out = ROOT / f"{NAME}_dspark"
    draft_out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in draft.items()}, str(draft_out / "model.safetensors"))
    (draft_out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    # Reference: mlx-lm loads the native file, repacks it, and runs mxfp8/mxfp4 layers.
    ds.ModelArgs.__post_init__ = lambda self: None
    args = ds.ModelArgs.from_dict(config)
    model = ds.Model(args)
    loaded = mx.load(str(out / "model.safetensors"))
    weights = model.sanitize({k: v for k, v in loaded.items()})
    assert not any(".engram." in k or "bias_vl" in k or "mtp" in k for k in weights)

    def predicate(path, module):
        if f"{path}.scales" not in weights:
            return False
        bits = 4 if ".switch_mlp." in path else 8
        return {"group_size": 32, "bits": bits, "mode": f"mxfp{bits}"}

    nn.quantize(model, class_predicate=predicate)
    model.load_weights(list(weights.items()), strict=True)

    engrams = {}
    for i in text["engram_layer_ids"]:
        p = f"layers.{i}.engram"
        e = Engram(args, mx.array(dense[f"{p}.embed.weight"]))
        nn.quantize(e, group_size=32, bits=8, mode="mxfp8", class_predicate=lambda _, m: isinstance(m, nn.Linear))
        w, s = ds._repack(loaded[f"{p}.wkv.weight"], loaded[f"{p}.wkv.scale"])
        e.load_weights(
            [("wkv.weight", w), ("wkv.scales", s), ("q_weight", loaded[f"{p}.q_weight"]),
             ("k_weight", loaded[f"{p}.k_weight"])],
            strict=True,
        )
        engrams[i] = e

    dspark = DSpark(config, model, mx.load(str(draft_out / "model.safetensors")))

    def run_dspark():
        captures = []
        forward(model, engrams, x, h, model.make_cache(), captures=captures)
        hidden = mx.concatenate(captures, axis=-1)
        dspark.windows = [None] * len(dspark.stages)
        # Seed in two chunks, as an incremental prompt prefill does.
        dspark.seed(hidden[:, :10], 0)
        dspark.seed(hidden[:, 10:], 10)
        draft_logits, draft_tokens = dspark.draft(bonus, len(IDS))
        return hidden, draft_logits, draft_tokens

    gaps = TopKGaps()
    mx.argpartition = gaps
    try:
        x = mx.array(ids.astype(np.int32))
        h = mx.array(hashes)
        logits = forward(model, engrams, x, h, model.make_cache())
        bonus = mx.argmax(logits[0, -1]).item()
        dspark_hidden, dspark_logits, dspark_tokens = run_dspark()
        # The reference itself must agree with chunked prefill and decode.
        cache = model.make_cache()
        rows = []
        for s, e in [(0, 5), (5, 9), (9, 14)] + [(t, t + 1) for t in range(14, len(IDS))]:
            rows.append(forward(model, engrams, x[:, s:e], h[:, s:e], cache))
        chunked = mx.concatenate(rows, axis=1)
    finally:
        mx.argpartition = gaps.original
    drift = mx.abs(chunked - logits).max().item()
    print(f"reference chunked vs full: {drift:.2e}")
    assert drift < 1e-4
    print(f"seed {seed}: smallest top-k boundary gap {gaps.smallest:.2e}")
    if gaps.smallest <= 1e-3:
        return False
    jitter = rounding_sensitivity(lambda: forward(model, engrams, x, h, model.make_cache()), logits)
    jitter = max(jitter, rounding_sensitivity(lambda: run_dspark()[1], dspark_logits))
    print(f"seed {seed}: logit change from jittered FP8/FP4 rounding inputs {jitter:.2e}")
    if jitter > 1e-5:
        return False

    # Sparse selection and Engram must both change the logits.
    def effect(topk, engram_layers):
        indexers = [l.self_attn.indexer for l in model.model.layers if "indexer" in l.self_attn]
        for indexer in indexers:
            indexer.index_topk = topk
        other = forward(model, engram_layers, x, h, model.make_cache())
        for indexer in indexers:
            indexer.index_topk = text["index_topk"]
        return mx.abs(other - logits).max().item()

    sparse = effect(len(IDS), engrams)
    engram = effect(text["index_topk"], {})
    print(f"max |dense indexer - sparse|: {sparse:.3f}, max |no engram - engram|: {engram:.3f}")
    assert sparse > 1e-2 and engram > 1e-2

    mx.save_safetensors(
        str(out / "expected.safetensors"),
        {
            "input_ids": x,
            "logits": logits.astype(mx.float32),
            "engram_hashes": mx.array(hashes.astype(np.int32)),
        },
    )
    mx.save_safetensors(
        str(draft_out / "expected.safetensors"),
        {
            "input_ids": x,
            "hidden": dspark_hidden.astype(mx.float32),
            "bonus": mx.array([bonus], dtype=mx.int32),
            "draft_logits": dspark_logits.astype(mx.float32),
            "draft_tokens": mx.array([dspark_tokens], dtype=mx.int32),
        },
    )
    for directory in (out, draft_out):
        size = sum(f.stat().st_size for f in directory.iterdir())
        print(f"wrote {directory} ({size} bytes)")
        assert size < 1_000_000
    print(f"bonus {bonus}, DSpark greedy draft {dspark_tokens}")
    return True


def check_against_deepseek(inference, tokenizer, ids, hashes):
    """Compare the n-gram hashes with DeepSeek's NgramHashState, prefill and decode."""
    sys.path.insert(0, str(inference))
    import engram as reference

    class Wrapped:
        backend_tokenizer = tokenizer

        def __len__(self):
            return tokenizer.get_vocab_size(with_added_tokens=True)

    args = SimpleNamespace(
        **text, engram_pad_id=text["engram_pad_token_id"], max_batch_size=1, max_seq_len=64
    )
    layout = reference.EngramLayout.from_args(args)
    state = reference.NgramHashState(args, layout, Wrapped())
    full = state(torch.tensor(ids), 0).numpy()
    assert np.array_equal(full, hashes), "prefill hashes differ from DeepSeek's"
    split = 10
    head = state(torch.tensor(ids[:, :split]), 0).numpy()
    tail = [state(torch.tensor(ids[:, t : t + 1]), t).numpy() for t in range(split, ids.shape[1])]
    assert np.array_equal(np.concatenate([head, *tail], axis=1), hashes)
    print("n-gram hashes match DeepSeek's NgramHashState")


if __name__ == "__main__":
    main()
