"""Tiny Qwen3 fixture from mlx-lm; checks the parity harness itself."""

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3

NAME = "qwen3"
OUT = Path(__file__).resolve().parents[2] / "Tests/MLXLMTests/Resources/Parity" / NAME

config = {
    "model_type": "qwen3",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-6,
    "vocab_size": 128,
    "max_position_embeddings": 256,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}

mx.random.seed(0)
model = qwen3.Model(qwen3.ModelArgs.from_dict(config))
weights = {k: v * 4 if v.ndim == 1 else v for k, v in tree_flatten(model.parameters())}
model.load_weights(list(weights.items()))
ids = mx.array([[1, 5, 9, 2, 77, 31, 8, 100]], dtype=mx.int32)
logits = model(ids).astype(mx.float32)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
mx.save_safetensors(str(OUT / "model.safetensors"), weights)
mx.save_safetensors(str(OUT / "expected.safetensors"), {"input_ids": ids, "logits": logits})
print(f"wrote {OUT}")
