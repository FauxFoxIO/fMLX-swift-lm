"""Two-rank tensor-parallel references for the Muse Spark fixtures, from mlx-lm's `shard()`.

Muse runs its branches in bf16, and a sharded layer sums bf16 partial products. On these tiny
fixtures that rounding moves near-tied expert choices, so sharded logits differ from the
unsharded ones by more than rounding. This writes what mlx-lm's own sharding produces, to
`Resources/Parity/<name>/expected_tp2.safetensors`: `logits` [1, L, vocab] for `input_ids`, and
`greedy`, 12 greedy tokens after the prompt.

Run on one machine with two ranks over the ring backend:

    mlx.launch --backend ring -n 2 --python python scripts/parity/muse_spark_tp.py
"""

import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.models import muse_spark_open as msp

ROOT = Path(__file__).resolve().parents[2] / "Tests/MLXLMTests/Resources/Parity"
FIXTURES = ["muse_spark_meta", "muse_spark_nvfp4_tp"]
GREEDY_TOKENS = 12


def load(directory):
    config = json.loads((directory / "config.json").read_text())
    model = msp.Model(msp.ModelArgs.from_dict(config))
    weights = mx.load(str(directory / "model.safetensors"))
    if any(k.endswith("w13_weight_scale") for k in weights):
        # As muse_spark.py: nvfp4 experts with a global scale in the inner layers.
        for i in range(1, config["num_hidden_layers"] - 1):
            switch = model.model.layers[i].feed_forward.switch_mlp
            for name in ("gate_proj", "up_proj", "down_proj"):
                layer = getattr(switch, name)
                setattr(switch, name, layer.to_quantized(16, 4, "nvfp4", global_scale=True))
    model.load_weights(list(model.sanitize(dict(weights)).items()), strict=True)
    return model


def greedy(model, ids):
    cache = model.make_cache()
    logits = model(ids, cache=cache)
    tokens = []
    for _ in range(GREEDY_TOKENS):
        token = logits[:, -1].argmax(axis=-1)
        tokens.append(token.item())
        logits = model(token[:, None], cache=cache)
    return mx.array([tokens], dtype=mx.int32)


group = mx.distributed.init(strict=True)
assert group.size() == 2, "run with two ranks"
for name in FIXTURES:
    directory = ROOT / name
    ids = mx.load(str(directory / "expected.safetensors"))["input_ids"]
    model = load(directory)
    model.shard(group)
    logits = model(ids).astype(mx.float32)
    tokens = greedy(model, ids)
    # Every rank holds the same gathered logits.
    leader = mx.distributed.all_sum(
        mx.where(group.rank() == 0, logits, mx.zeros_like(logits)), group=group
    )
    assert mx.array_equal(leader, logits).item(), "ranks disagree"
    if group.rank() == 0:
        out = directory / "expected_tp2.safetensors"
        mx.save_safetensors(str(out), {"logits": logits, "greedy": tokens})
        print(f"wrote {out}: greedy {tokens.tolist()[0]}")
