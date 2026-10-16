"""Write the Engram hash tables that DeepSeek-V4.1 derives from its tokenizer.

Engram hashes n-grams of "compressed" token ids: tokens that normalize alike (case, accents,
whitespace) share an id. The map and the per-layer hash multipliers come from the tokenizer
and NumPy's PCG64 generator, so they are computed here once and stored beside the checkpoint:

    python scripts/deepseek_v41_engram_tables.py /path/to/DeepSeek-V4.1-Flash

writes `deepseek_v41_engram.safetensors` with `token_map` (int32, one entry per token id) and
`multipliers` (int64, [engram layers, engram_max_ngram_size]). fMLX computes the bucket primes
itself and checks them against `engram_num_embeddings`.

This follows `inference/engram.py` in DeepSeek's Hugging Face repository
(`build_compressed_token_map` and `compute_hash_multipliers`). Needs `numpy`, `tokenizers`, and
`safetensors`.
"""

import argparse
import json
from pathlib import Path

import numpy as np

FILE_NAME = "deepseek_v41_engram.safetensors"


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id to an id shared by all tokens that normalize to the same text.

    `tokenizer` is a `tokenizers.Tokenizer`. Returns the map and the number of distinct ids.
    """
    from tokenizers import Regex, normalizers

    # A private-use character keeps a single-space token from stripping to "".
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    key_to_id: dict[str, int] = {}
    size = tokenizer.get_vocab_size(with_added_tokens=True)
    lookup = [0] * size
    for token_id in range(size):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            # A partial UTF-8 byte token: key it by its raw form.
            key = tokenizer.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = key_to_id.setdefault(key, len(key_to_id))
    return lookup, len(key_to_id)


def compute_hash_multipliers(layer_ids, max_ngram_size: int, compressed_vocab_size: int):
    """One odd multiplier per (engram layer, look-back), from a per-layer seeded generator.

    The bound keeps `token_id * multiplier` inside int64.
    """
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(values * 2 + 1)
    return np.stack(rows).astype(np.int64)


def text_config(config: dict) -> dict:
    return config.get("text_config", config)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="directory with config.json and tokenizer.json")
    parser.add_argument("--output", type=Path, help=f"default: <model>/{FILE_NAME}")
    args = parser.parse_args()

    from safetensors.numpy import save_file
    from tokenizers import Tokenizer

    config = text_config(json.loads((args.model / "config.json").read_text()))
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    token_map, size = build_compressed_token_map(tokenizer)
    expected = config["engram_compressed_vocab_size"]
    if size != expected:
        raise SystemExit(f"compressed vocab has {size} ids, config declares {expected}")
    multipliers = compute_hash_multipliers(
        config["engram_layer_ids"], config["engram_max_ngram_size"], size
    )
    output = args.output or args.model / FILE_NAME
    save_file(
        {"token_map": np.array(token_map, dtype=np.int32), "multipliers": multipliers},
        str(output),
    )
    print(f"wrote {output}: {len(token_map)} tokens -> {size} ids")


if __name__ == "__main__":
    main()
