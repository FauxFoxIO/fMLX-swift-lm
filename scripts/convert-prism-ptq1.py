#!/usr/bin/env python3
"""Convert a Prism PTQ1_0 GGUF into compact MLX safetensors.

Requires numpy, ml-dtypes, safetensors, and the gguf Python package from Prism's
PTQ1-enabled llama.cpp fork on PYTHONPATH.
The output keeps every PTQ1 block byte-for-byte; it does not transcode to MLX
affine quantization.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import ml_dtypes
from gguf import GGUFReader
from safetensors.numpy import save_file


STEMS = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
}
GLOBALS = {
    "output.weight": "language_model.lm_head.weight",
    "output_norm.weight": "language_model.model.norm.weight",
    "token_embd.weight": "language_model.model.embed_tokens.weight",
}


def target_name(name: str) -> tuple[str, str]:
    if name in GLOBALS:
        return GLOBALS[name], name
    if name.startswith("blk."):
        _, layer, stem = name.split(".", 2)
        if stem not in STEMS:
            raise ValueError(f"unmapped GGUF tensor: {name}")
        return f"language_model.model.layers.{layer}.{STEMS[stem]}", stem
    raise ValueError(f"unmapped GGUF tensor: {name}")


def value_permutation(value_heads: int, key_heads: int, unit: int) -> np.ndarray:
    return (
        np.arange(value_heads * unit)
        .reshape(value_heads // key_heads, key_heads, unit)
        .transpose(1, 0, 2)
        .reshape(-1)
    )


def reorder(array: np.ndarray, stem: str, value_heads: int, key_heads: int,
            key_dim: int, value_dim: int) -> np.ndarray:
    if value_heads == key_heads:
        return array
    if stem in ("attn_qkv.weight", "ssm_conv1d.weight"):
        permutation = value_permutation(value_heads, key_heads, value_dim)
        boundary = 2 * key_heads * key_dim
        return np.concatenate([array[:boundary], array[boundary:][permutation]], axis=0)
    if stem == "attn_gate.weight":
        permutation = value_permutation(value_heads, key_heads, value_dim)
        return array[permutation]
    if stem in ("ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias"):
        permutation = value_permutation(value_heads, key_heads, 1)
        return array[permutation]
    return array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("template", type=Path, help="Published Prism MLX artifact directory")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(str(args.gguf))
    fields = {name: field.contents() for name, field in reader.fields.items()}
    if fields.get("general.architecture") != "qwen35":
        raise ValueError("expected a qwen35 GGUF")
    get = lambda name: fields["qwen35." + name]
    value_heads = int(get("ssm.time_step_rank"))
    key_heads = int(get("ssm.group_count"))
    value_dim = int(get("ssm.inner_size")) // value_heads
    key_dim = int(get("ssm.state_size"))

    tensors = {}
    for tensor in reader.tensors:
        destination, stem = target_name(tensor.name)
        kind = tensor.tensor_type.name
        shape = tuple(int(size) for size in tensor.shape[::-1])
        if kind == "PTQ1_0":
            if len(shape) != 2 or shape[1] % 128:
                raise ValueError(f"invalid PTQ1 matrix shape: {tensor.name} {shape}")
            rows, width = shape
            raw = np.frombuffer(tensor.data.tobytes(), dtype=np.uint8)
            expected = rows * (width // 128) * 28
            if raw.size != expected:
                raise ValueError(f"invalid PTQ1 byte count: {tensor.name}")
            array = raw.reshape(rows, width // 128, 28).copy()
            array = reorder(array, stem, value_heads, key_heads, key_dim, value_dim)
        elif kind in ("F16", "F32", "BF16"):
            if kind == "BF16":
                array = np.frombuffer(tensor.data.tobytes(), dtype="<u2").view(
                    ml_dtypes.bfloat16
                ).reshape(shape).copy()
            else:
                array = np.asarray(tensor.data).copy()
            array = reorder(array, stem, value_heads, key_heads, key_dim, value_dim)
            if stem == "ssm_a":
                if not (array < 0).all():
                    raise ValueError("stored SSM A must be negative")
                array = np.log(-array)
            if stem == "ssm_conv1d.weight":
                array = array[..., None]
        else:
            raise ValueError(f"unsupported tensor type {kind}: {tensor.name}")
        tensors[destination] = np.ascontiguousarray(array)

    config = json.loads((args.template / "config.json").read_text())
    quantization = config.pop("quantization")
    quantization["mode"] = "ptq1_0"
    config["compact_quantization"] = quantization
    config["components"]["vision"] = False
    temporary = args.output / "model.safetensors.incomplete"
    save_file(tensors, temporary)
    temporary.replace(args.output / "model.safetensors")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ("hadamard.json", "tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja", "generation_config.json"):
        source = args.template / name
        if source.exists():
            shutil.copy2(source, args.output / name)


if __name__ == "__main__":
    main()
