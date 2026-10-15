#!/usr/bin/env python3
"""Reference fixtures for the MLXDecision tests.

Run with an environment that has torch, torchvision, transformers, gliner2 and,
for the real-checkpoint references, mlx-vlm. See README.md for the pinned
revisions and the exact commands.

Subcommands:

  synthetic        Tiny random models and prompt/schema encodings. Needs the
                   d1-omni reference sources (`--d1-source`, a directory holding
                   modeling_d1.py, encoder.py, vision.py, audio.py, prompt.py).
                   Writes `tiny-d1-omni/`, `tiny-deberta/`, and JSON fixtures.
  d1               Real-checkpoint d1-omni references through mlx-vlm.
  gliner           Real-checkpoint GLiNER2.5-Decide references through gliner2.
  embeddinggemma   Real-checkpoint EmbeddingGemma 2 references through mlx-vlm.
  lfm2vl           LFM2-VL image layouts, crops, markup, and a tiny random
                   Lfm2VlForConditionalGeneration through Transformers (the
                   d1-3B image path). Writes `Tests/LFM2VLTests/Resources`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Shared deterministic inputs. The Swift tests rebuild these from the same
# formulas and check the checksums recorded here.


def synthetic_rgb(width: int, height: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    channels = [(x * 3 + y * 5 + c * 70 + (x * y) % 37) % 256 for c in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def synthetic_pcm(count: int) -> np.ndarray:
    state = 12345
    out = np.zeros(count, dtype=np.int16)
    for n in range(count):
        t = n / 16000.0
        state = (1103515245 * state + 12345) % 2147483648
        noise = (state / 2147483648.0 - 0.5) * 800.0
        value = (
            6000.0 * math.sin(2 * math.pi * 220.0 * t)
            + 3000.0 * math.sin(2 * math.pi * (300.0 + 400.0 * t) * t)
            + noise
        )
        out[n] = int(max(-32768, min(32767, round(value))))
    return out


def checksum(values) -> int:
    flat = np.asarray(values).astype(np.int64).ravel()
    weights = (np.arange(flat.size) % 9973) + 1
    return int((flat * weights).sum() % (2**61 - 1))


def summary(array) -> dict:
    a = np.asarray(array, dtype=np.float64).ravel()
    n = a.size
    indices = sorted({(k * 7919) % n for k in range(48)}) if n else []
    return {
        "count": int(n),
        "sum": float(a.sum()),
        "abs_sum": float(np.abs(a).sum()),
        "indices": indices,
        "values": [float(a[i]) for i in indices],
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")



def normalized_question(name, q):
    """An ordered, JSON-safe form of a Decision Index question for the Swift tests."""
    kind, crit = q["type"], q.get("criteria")
    out = {"id": name, "kind": kind, "instructions": q["instructions"]}
    if kind == "choice":
        pairs = crit.items() if isinstance(crit, dict) else ((k, None) for k in crit)
        out["choices"] = [[k, v] for k, v in pairs]
    elif kind == "score":
        out["levels"] = list(crit)
    else:
        c = crit or {}
        out["when_true"] = c.get("true", c.get("yes"))
        out["when_false"] = c.get("false", c.get("no"))
        out["has_criteria"] = bool(crit)
    return out


# ---------------------------------------------------------------------------
# Fake tokenizers shared with the Swift tests.

D1_SPECIAL = {
    "<|startoftext|>": 1,
    "<|mask|>": 2,
    "<|reserved_7|>": 3,
    "<|reserved_8|>": 4,
    "<|reserved_9|>": 5,
    "<|reserved_10|>": 6,
    "<|reserved_11|>": 7,
}


class FakeD1Tokenizer:
    """Every Unicode scalar becomes one ID in [10, 310)."""

    bos_token_id = 1

    def convert_tokens_to_ids(self, token):
        return D1_SPECIAL[token]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + ord(c) % 300 for c in text]}


GLINER_SPECIAL = [
    "[SEP_STRUCT]",
    "[SEP_TEXT]",
    "[P]",
    "[C]",
    "[E]",
    "[R]",
    "[L]",
    "[EXAMPLE]",
    "[OUTPUT]",
    "[DESCRIPTION]",
]


class FakeGLiNERTokenizer:
    """Pieces of up to four scalars; the first piece of every call gets `▁`."""

    unk_token = "[UNK]"

    def add_special_tokens(self, mapping):
        return 0

    def tokenize(self, text):
        if text in GLINER_SPECIAL:
            return [text]
        words = text.split()
        pieces = []
        for word in words:
            chunks = [word[i : i + 4] for i in range(0, len(word), 4)]
            pieces.append("▁" + chunks[0])
            pieces.extend(chunks[1:])
        return pieces

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._id(tokens)
        return [self._id(t) for t in tokens]

    @staticmethod
    def _id(token):
        if token in GLINER_SPECIAL:
            return 1000 + GLINER_SPECIAL.index(token)
        return 10 + sum((i + 1) * ord(c) for i, c in enumerate(token)) % 900


# ---------------------------------------------------------------------------
# Synthetic fixtures.

TINY_D1_CONFIG = {
    "architectures": ["D1OmniModel"],
    "model_type": "d1_omni",
    "max_length": 16384,
    "image_text_length": 896,
    "audio_text_length": 15360,
    "head_layers": 2,
    "projector_hidden_size": 48,
    "temperatures": {
        "choice:2": 1.5,
        "choice:3-5": 1.25,
        "noul:2": 1.75,
        "score:3-5": 0.8,
        "choice": 1.0,
        "score": 1.0,
        "noul": 1.0,
    },
    "bos_token_id": 1,
    "pad_token_id": 0,
    "text_config": {
        "vocab_size": 320,
        "hidden_size": 64,
        "intermediate_size": 192,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "layer_types": ["conv", "full_attention", "conv"],
        "norm_eps": 1e-05,
        "conv_L_cache": 3,
        "block_ffn_dim_multiplier": 1.0,
        "block_multiple_of": 32,
        "max_position_embeddings": 128000,
        "rope_theta": 10000.0,
    },
    "vision_config": {
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 32,
        "intermediate_size": 64,
        "layer_norm_eps": 1e-06,
        "model_type": "siglip2_vision_model",
        "num_attention_heads": 2,
        "num_channels": 3,
        "num_hidden_layers": 2,
        "num_patches": 16,
        "patch_size": 16,
        "vision_use_head": False,
    },
    "audio_config": {
        "feat_in": 128,
        "n_layers": 2,
        "d_model": 32,
        "subsampling_conv_channels": 8,
        "ff_expansion_factor": 2,
        "n_heads": 2,
        "conv_kernel_size": 9,
        "residual_width": 16,
    },
}


def randomize(module, torch, seed: int) -> None:
    """Replace every float tensor so no parameter keeps its trivial init."""
    norms = (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)
    scale_names = set()
    for name, child in module.named_modules():
        if isinstance(child, norms) or type(child).__name__ == "RMSNorm":
            scale_names.add(f"{name}.weight" if name else "weight")
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in module.state_dict().items():
            if not tensor.is_floating_point():
                continue
            noise = torch.randn(tensor.shape, generator=generator) * 0.2
            if name.endswith("running_var"):
                value = 0.5 + noise.abs()
            elif name.endswith("running_mean"):
                value = noise * 0.5
            elif name in scale_names:
                value = 1.0 + noise
            else:
                value = noise
            tensor.copy_(value.to(tensor.dtype))


def synthetic(args) -> None:
    import torch
    from PIL import Image
    from safetensors.torch import save_file

    out = Path(args.output)
    sys.path.insert(0, str(Path(args.d1_source).resolve().parent))
    package = Path(args.d1_source).name
    modeling = importlib.import_module(f"{package}.modeling_d1")
    prompt = importlib.import_module(f"{package}.prompt")
    vision = importlib.import_module(f"{package}.vision")
    audio = importlib.import_module(f"{package}.audio")

    torch.manual_seed(0)
    config = modeling.D1OmniConfig(**TINY_D1_CONFIG)
    model = modeling.D1OmniModel(config).eval()
    randomize(model, torch, seed=7)
    tiny = out / "tiny-d1-omni"
    tiny.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(tiny / "model.safetensors"), metadata={"format": "pt"})
    write_json(tiny / "config.json", TINY_D1_CONFIG)

    def logits_for(rows):
        # The reference `_forward`, stopping before the softmax.
        device, dtype = model.device, model.encoder.embed_tokens.weight.dtype
        seqs, lengths, offsets = [], [], []
        for prefix, ids, *_ in rows:
            text = model.encoder.embed_tokens(torch.tensor(ids))
            seqs.append(text if prefix is None else torch.cat([prefix[0].to(dtype), text]))
            offsets.append(0 if prefix is None else prefix.shape[1])
            lengths.append(len(ids))
        h = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        total = torch.tensor([len(s) for s in seqs])
        pad = torch.arange(h.shape[1])[None] < total[:, None]
        h = model.encoder(h, pad, torch.tensor(offsets))
        text = torch.nn.utils.rnn.pad_sequence(
            [h[i, o : o + n] for i, (o, n) in enumerate(zip(offsets, lengths))], batch_first=True
        )
        k = max(len(r[2]) for r in rows)
        text_pad = torch.arange(text.shape[1])[None] < torch.tensor(lengths)[:, None]
        marker_pos = torch.zeros(len(rows), k, dtype=torch.long)
        marker_mask = torch.zeros(len(rows), k, dtype=torch.bool)
        for i, r in enumerate(rows):
            marker_pos[i, : len(r[2])] = torch.tensor(r[2])
            marker_mask[i, : len(r[2])] = True
        qtype = torch.tensor([prompt.QTYPES[r[3].type] for r in rows])
        return model.head(text, text_pad, marker_pos, marker_mask, qtype)

    def probabilities(rows):
        with torch.no_grad():
            return model._forward(rows)

    rng = np.random.default_rng(3)
    text_rows = []
    specs = [
        (9, [3, 6], "choice", {"a": "", "b": ""}),
        (14, [2, 5, 9, 12], "score", ["0", "1", "2", "3"]),
        (6, [1, 4], "noul", None),
        (11, [4, 7, 10], "choice", {"x": "", "y": "", "z": ""}),
    ]
    for length, markers, kind, criteria in specs:
        ids = [int(v) for v in rng.integers(8, 320, size=length)]
        text_rows.append((ids, markers, kind, criteria))

    def rows_of(entries, prefix=None, calibrate=True):
        return [
            (prefix, ids, markers, prompt.Question(kind, "q", criteria), calibrate)
            for ids, markers, kind, criteria in entries
        ]

    with torch.no_grad():
        text_logits = logits_for(rows_of(text_rows))
    text_fixture = {
        "rows": [
            {"ids": ids, "markers": m, "type": kind, "options": len(m)}
            for ids, m, kind, _ in text_rows
        ],
        "logits": [row[: len(m)].tolist() for row, (_, m, _, _) in zip(text_logits, text_rows)],
        "probabilities": probabilities(rows_of(text_rows)),
    }

    image_cases = []
    for width, height in [(100, 60), (1400, 1000), (40, 900)]:
        pixels = synthetic_rgb(width, height)
        image = Image.fromarray(pixels, "RGB")
        plan = vision.layout(width, height)
        from torchvision.transforms.v2 import functional as tvf

        x = tvf.pil_to_tensor(image)
        crops = []
        if plan["tiled"]:
            gw, gh = plan["grid"]
            big = tvf.resize(
                x, [gh * 512, gw * 512], interpolation=tvf.InterpolationMode.BILINEAR, antialias=True
            )
            crops = [
                big[:, r * 512 : (r + 1) * 512, c * 512 : (c + 1) * 512]
                for r in range(gh)
                for c in range(gw)
            ]
        crops.append(
            tvf.resize(
                x, list(plan["thumbnail"]), interpolation=tvf.InterpolationMode.BILINEAR, antialias=True
            )
        )
        with torch.no_grad():
            prefix = model.vision([image])
            rows = rows_of(text_rows[:2], prefix=prefix, calibrate=False)
            logits = logits_for(rows)
        image_cases.append(
            {
                "width": width,
                "height": height,
                "pixel_checksum": checksum(pixels),
                "grid": list(plan["grid"]),
                "tiled": plan["tiled"],
                "thumbnail": list(plan["thumbnail"]),
                "crops": [
                    {"height": int(c.shape[1]), "width": int(c.shape[2]), "checksum": checksum(c.permute(1, 2, 0).numpy())}
                    for c in crops
                ],
                "prefix_shape": list(prefix.shape[1:]),
                "prefix": summary(prefix.numpy()),
                "logits": [row[: len(r[2])].tolist() for row, r in zip(logits, rows)],
                "probabilities": probabilities(rows),
            }
        )

    audio_cases = []
    for count in [20800, 4000]:
        pcm = synthetic_pcm(count)
        with torch.no_grad():
            mel, frames = model.audio.frontend(audio.waveform(pcm))
            encoded, lengths = model.audio.encoder(mel, frames)
            prefix = model.audio(pcm)
            rows = rows_of(text_rows[2:], prefix=prefix, calibrate=False)
            logits = logits_for(rows)
        audio_cases.append(
            {
                "samples": count,
                "pcm_checksum": checksum(pcm),
                "mel_shape": list(mel.shape[1:]),
                "frames": int(frames[0]),
                "mel": summary(mel.numpy()),
                "encoded_shape": list(encoded.shape[1:]),
                "encoded_length": int(lengths[0]),
                "encoded": summary(encoded.numpy()),
                "prefix_shape": list(prefix.shape[1:]),
                "prefix": summary(prefix.numpy()),
                "logits": [row[: len(r[2])].tolist() for row, r in zip(logits, rows)],
                "probabilities": probabilities(rows),
            }
        )
    # End to end through the reference API with the fake tokenizer.
    model.__dict__["tokenizer"] = FakeD1Tokenizer()
    end_to_end = []
    e2e_questions = [
        {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        {"type": "choice", "instructions": "Which team?",
         "criteria": {"billing": "Charges", "technical": "", "fraud": "Suspected unauthorised use"}},
        {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "Today", "Now"]},
        {"type": "noul", "instructions": "Holds?", "criteria": {"true": "it holds"}},
    ]
    for state, media in [
        ("I was charged twice this month, please refund one of them.", None),
        (None, ("image", 100, 60)),
        ("caption", ("image", 40, 900)),
        (None, ("audio", 20800)),
    ]:
        images = audio_clip = None
        if media and media[0] == "image":
            images = [Image.fromarray(synthetic_rgb(media[1], media[2]), "RGB")]
        elif media:
            audio_clip = synthetic_pcm(media[1])
        with torch.no_grad():
            probs, read = model.probabilities_batch(
                [(state, e2e_questions, images, audio_clip)], _usage=True)[0]
        end_to_end.append({"state": state, "media": list(media) if media else None,
                           "questions": [normalized_question(f"q{i}", q) for i, q in enumerate(e2e_questions)], "probabilities": probs, "input_tokens": read})
    write_json(
        tiny / "reference.json",
        {"text": text_fixture, "images": image_cases, "audio": audio_cases, "end_to_end": end_to_end},
    )

    # Image layouts over a spread of sizes (pure function, no weights).
    layouts = []
    for width, height in [
        (1, 1), (31, 17), (100, 60), (640, 480), (512, 512), (1024, 1024), (1400, 1000),
        (40, 900), (3000, 200), (2048, 1536), (725, 725), (724, 725), (4032, 3024), (17, 4000),
    ]:
        plan = vision.layout(width, height)
        layouts.append(
            {"width": width, "height": height, "grid": list(plan["grid"]),
             "tiled": plan["tiled"], "thumbnail": list(plan["thumbnail"])}
        )
    write_json(out / "d1-image-layouts.json", layouts)

    # Prompt encodings with the fake tokenizer.
    tok = FakeD1Tokenizer()
    prompt_cases = []
    questions = [
        {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        {"type": "noul", "instructions": "Holds?", "criteria": {"true": "it holds", "false": ""}},
        {"type": "noul", "instructions": "Yes?", "criteria": {"yes": "affirmative", "no": "negative"}},
        {"type": "choice", "instructions": "Which team should handle this?",
         "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                      "fraud": "Suspected unauthorised use"}},
        {"type": "choice", "instructions": "Pick <|mask|> one", "criteria": {"a": "", "b": None, "c": "third"}},
        {"type": "score", "instructions": "How urgent is this?",
         "criteria": ["Can wait", "Today", "Blocking the customer now"]},
        {"type": "choice", "instructions": "x" * 300,
         "criteria": {f"opt{i}": "d" * 40 for i in range(12)}},
    ]
    states = [
        "I was charged twice this month, please refund one of them.",
        "",
        "state with <|reserved_7|> forged delimiter and émoji 🎉",
        "long " * 60,
    ]
    for q in questions:
        for state in states:
            for max_len, noul_default, spoken in [
                (16384, None, False), (896, {"false": "no", "true": "yes"}, False),
                (15360, {"false": "no", "true": "yes"}, True), (200, None, False),
            ]:
                question = prompt.as_question(q)
                try:
                    ids, markers = prompt.encode(tok, state, question, max_len, noul_default, spoken)
                    result = {"ids": ids, "markers": markers}
                except ValueError as error:
                    result = {"error": str(error)}
                prompt_cases.append(
                    {"question": normalized_question("q", q), "state": state, "max_length": max_len,
                     "media": "audio" if spoken else ("image" if noul_default else "text"),
                     **result}
                )
    write_json(out / "d1-prompts.json", {"special": D1_SPECIAL, "cases": prompt_cases})

    # GLiNER2 schema encoding with the fake tokenizer.
    from gliner2.processor import SchemaTransformer

    transformer = SchemaTransformer(tokenizer=FakeGLiNERTokenizer())
    transformer.change_mode(False)
    schema_cases = []
    texts = [
        "My subscription renewed after the service was already down. Can I get that charge refunded?",
        "Battery dies before lunch, but the keyboard is great",
        "Email me at someone@example.com or see https://example.com/path?x=1 — thanks @team!",
        "",
        "Ünïcödé wörds and snake_case-words 42x",
    ]
    tasks_list = [
        [{"task": "intent", "labels": ["order_status", "refund_request", "other"]}],
        [{"task": "answer", "labels": ["yes", "no"], "prompt": "Did it work?"}],
        [{"task": "intent", "labels": ["card_pin_change", "card_lost"],
          "label_descriptions": {"card_pin_change": "New PIN wanted", "card_lost": "Card missing"}}],
        [{"task": "intent", "labels": ["a", "b"]},
         {"task": "urgency", "labels": ["low", "normal", "high"]},
         {"task": "route", "labels": ["support", "billing"], "prompt": "Who owns it?"}],
    ]
    for text in texts:
        for tasks in tasks_list:
            schema = {"classifications": [dict(t, multi_label=False, cls_threshold=0.5) for t in tasks]}
            normalized = transformer._normalize_text(text)
            record = transformer._transform_record({"text": normalized, "schema": json.loads(json.dumps(schema))})
            schema_cases.append(
                {"text": text, "tasks": tasks, "normalized": normalized,
                 "schema_tokens": record.schema_tokens_list, "text_tokens": record.text_tokens,
                 "input_ids": record.input_ids,
                 "text_word_first_positions": record.text_word_first_positions,
                 "schema_special_positions": record.schema_special_positions}
            )
    write_json(out / "gliner2-schemas.json", {"cases": schema_cases})

    # Tiny DeBERTa-v2 encoder and classifier in the GLiNER2 checkpoint layout.
    from transformers import DebertaV2Config, DebertaV2Model

    deberta_config = {
        "model_type": "deberta-v2", "vocab_size": 300, "hidden_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "intermediate_size": 128, "hidden_act": "gelu",
        "max_position_embeddings": 32, "max_relative_positions": -1, "position_buckets": 8,
        "relative_attention": True, "pos_att_type": ["p2c", "c2p"], "share_att_key": True,
        "position_biased_input": False, "norm_rel_ebd": "layer_norm", "type_vocab_size": 0,
        "layer_norm_eps": 1e-7, "pad_token_id": 0,
    }
    torch.manual_seed(0)
    encoder = DebertaV2Model(DebertaV2Config(**deberta_config)).eval()
    randomize(encoder, torch, seed=11)
    classifier = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))
    randomize(classifier, torch, seed=13)
    weights = {f"encoder.{k}": v.contiguous() for k, v in encoder.state_dict().items()}
    weights.update({f"classifier.{k}": v.contiguous() for k, v in classifier.state_dict().items()})
    weights["count_pred.0.weight"] = torch.zeros(4, 64)
    tiny_deberta = out / "tiny-deberta"
    tiny_deberta.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(tiny_deberta / "model.safetensors"))
    gliner_config = {
        "architectures": ["Gliner2ForSchemaExtraction"], "model_type": "gliner2",
        "architecture": "span", "token_pooling": "first", "counting_layer": "count_lstm",
        "encoder_config": deberta_config,
    }
    write_json(tiny_deberta / "config.json", gliner_config)
    sequences = [[int(v) for v in rng.integers(1, 300, size=n)] for n in (40, 13)]
    longest = max(map(len, sequences))
    ids = torch.tensor([s + [0] * (longest - len(s)) for s in sequences])
    mask = torch.tensor([[1] * len(s) + [0] * (longest - len(s)) for s in sequences])
    with torch.no_grad():
        hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        scores = classifier(hidden).squeeze(-1)
    write_json(
        tiny_deberta / "reference.json",
        {
            "sequences": sequences,
            "hidden": [summary(hidden[i, : len(s)].numpy()) for i, s in enumerate(sequences)],
            "first_hidden": [hidden[i, 0, :8].tolist() for i in range(len(sequences))],
            "classifier": [scores[i, : len(s)].tolist() for i, s in enumerate(sequences)],
        },
    )


# ---------------------------------------------------------------------------
# Real-checkpoint references.

D1_TEXT_CASES = [
    ("I was charged twice this month, please refund one of them.", {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                              "fraud": "Suspected unauthorised use"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["Can wait", "Today", "Blocking the customer now"]},
    }),
    ('{"page": "Release notes", "selection": "Ship the canvas build", "speaker": "user"}', {
        "intent": {"type": "choice", "instructions": "What does the user want?",
                   "criteria": {"navigate": "Open a different page", "edit": "Change the selected content",
                                "ask": "Ask a question about the content", "run": "Start an agent task"}},
        "durable": {"type": "noul", "instructions": "Should this be saved for later?",
                    "criteria": {"true": "worth keeping", "false": "throwaway"}},
    }),
    ("The deploy left resource tags inconsistent across staging. Production checkout is unaffected.", {
        "severity": {"type": "score", "instructions": "How severe is this incident?",
                     "criteria": ["info", "low", "medium", "high", "critical"]},
        "page": {"type": "noul", "instructions": "Should someone be paged now?"},
    }),
]


def yes_first(questions, probabilities):
    """MLX-VLM reads a noul as [false, true]; the fixtures use [yes, no] like the Swift API."""
    return [p[::-1] if q["type"] == "noul" else p for q, p in zip(questions, probabilities)]


def d1(args) -> None:
    import mlx.core as mx
    from PIL import Image
    from mlx_vlm import load
    from mlx_vlm.models.d1_omni.prompt import as_question, encode

    results = {"model": args.model_id, "revision": args.revision, "text": [], "images": [], "audio": []}
    model, processor = load(args.model)
    tok = getattr(processor, "tokenizer", processor)
    for state, questions in D1_TEXT_CASES:
        parsed = [as_question(q) for q in questions.values()]
        [(probs, read)] = model.probabilities(tok, [(state, parsed)])
        probs = yes_first(list(questions.values()), probs)
        encoded = [encode(tok, state, q, model.config.max_length) for q in parsed]
        results["text"].append({
            "state": state, "questions": [normalized_question(n, q) for n, q in questions.items()],
            "probabilities": probs, "input_tokens": read,
            "ids": [ids for ids, _ in encoded], "markers": [m for _, m in encoded]})
    if model.config.text_config.hidden_size and not getattr(model.config, "quantization", None):
        # The same export upcast to float32 isolates port differences from float16 rounding.
        float32_model, _ = load(args.model)
        float32_model.set_dtype(mx.float32)
        results["text_float32"] = []
        for state, questions in D1_TEXT_CASES:
            parsed = [as_question(q) for q in questions.values()]
            [(probs, _)] = float32_model.probabilities(tok, [(state, parsed)])
            results["text_float32"].append(yes_first(list(questions.values()), probs))
        del float32_model
    image_questions = {
        "count": {"type": "choice", "instructions": "How many shapes are there?",
                  "criteria": {"one": "One", "two": "Two", "more": "Three or more"}},
        "striped": {"type": "noul", "instructions": "Is the picture striped?"},
    }
    for width, height in [(320, 240), (1400, 1000)]:
        image = Image.fromarray(synthetic_rgb(width, height), "RGB")
        parsed = [as_question(q) for q in image_questions.values()]
        [(probs, read)] = model.probabilities(tok, [(None, parsed, [image])])
        probs = yes_first(list(image_questions.values()), probs)
        prefix = model.vision([image])
        results["images"].append({
            "width": width, "height": height,
            "questions": [normalized_question(n, q) for n, q in image_questions.items()],
            "probabilities": probs, "input_tokens": read,
            "prefix_shape": list(prefix.shape), "prefix": summary(np.array(prefix.astype(mx.float32)))})
    audio_questions = {
        "kind": {"type": "choice", "instructions": "What kind of utterance is this?",
                 "criteria": {"request": "A request to do something", "question": "A question asking for information",
                              "noise": "No speech, only sounds"}},
        "speech": {"type": "noul", "instructions": "Is anyone speaking in this clip?"},
    }
    for count in [20800]:
        pcm = synthetic_pcm(count)
        parsed = [as_question(q) for q in audio_questions.values()]
        [(probs, read)] = model.probabilities(tok, [(None, parsed, None, pcm)])
        probs = yes_first(list(audio_questions.values()), probs)
        prefix = model.audio(pcm)[0]
        results["audio"].append({
            "samples": count, "questions": [normalized_question(n, q) for n, q in audio_questions.items()],
            "probabilities": probs,
            "input_tokens": read, "prefix_shape": list(prefix.shape),
            "prefix": summary(np.array(prefix.astype(mx.float32)))})
    write_json(Path(args.output), results)


GLINER_CASES = [
    ("My subscription renewed on April 15 for ¥5,400 after the service was already down. Can I get that charge refunded?",
     [{"task": "intent", "labels": ["order_status", "refund_request", "cancel_subscription", "update_payment",
                                    "login_problem", "shipping_delay", "bug_report", "speak_to_human", "other"]}]),
    ("This is the third time I have explained the same missing refund. Stop the bot and get me a person.",
     [{"task": "handoff", "labels": ["yes", "no"]}]),
    ("The treaty was signed in Paris in 1992. It entered into force the following year, after the last signatory ratified it.",
     [{"task": "answer", "labels": ["yes", "no"], "prompt": "Did the treaty enter into force in 1992?"}]),
    ("Please reset the card PIN. The new one never arrived and the old one is locked after three tries.",
     [{"task": "intent", "labels": ["card_pin_change", "card_lost", "balance_inquiry"],
       "label_descriptions": {"card_pin_change": "The customer wants a new PIN or the current PIN replaced",
                              "card_lost": "The physical card is missing",
                              "balance_inquiry": "The customer wants the current balance"}}]),
    ("From: compliance@group.example\nSubject: Protocol update — action required today\n\nPlease confirm the new retention rule is applied before Friday's audit.",
     [{"task": "intent", "labels": ["fyi", "request", "approval", "complaint", "newsletter", "security_alert"]},
      {"task": "urgency", "labels": ["low", "normal", "high", "critical"]},
      {"task": "route", "labels": ["support", "billing", "legal", "security", "finance", "archive"]}]),
    ("I finished it in two nights. The ending is earned, the middle drags, and I would still hand it to a friend.",
     [{"task": "rating", "labels": [str(i) for i in range(11)]}]),
]


def gliner(args) -> None:
    from gliner2 import AutoExtractor
    from gliner2.inference import runtime

    model = AutoExtractor.from_pretrained(args.model)
    model.eval()
    captured = []
    original = runtime.__dict__.get("ExtractorRuntimeMixin")
    target = None
    for cls in type(model).__mro__:
        if "_decode_classification_probabilities" in cls.__dict__:
            target = cls
            break
    decode = target.__dict__["_decode_classification_probabilities"]

    def capture(results, schema_name, cls_config, probs):
        captured.append((schema_name, probs.detach().float().tolist()))
        return decode.__func__(results, schema_name, cls_config, probs)

    setattr(target, "_decode_classification_probabilities", staticmethod(capture))
    processor = model.processor
    out = {"model": args.model_id, "revision": args.revision, "cases": []}
    for text, tasks in GLINER_CASES:
        captured.clear()
        spec = {}
        for t in tasks:
            cfg = {"labels": t["labels"]}
            if "label_descriptions" in t:
                cfg["labels"] = t["label_descriptions"]
            if "prompt" in t:
                cfg["prompt"] = t["prompt"]
            spec[t["task"]] = cfg
        model.classify_text(text, spec)
        schema = {"classifications": []}
        for t in tasks:
            entry = {"task": t["task"], "labels": t["labels"], "multi_label": False, "cls_threshold": 0.5}
            if "prompt" in t:
                entry["prompt"] = t["prompt"]
            if "label_descriptions" in t:
                entry["label_descriptions"] = t["label_descriptions"]
            schema["classifications"].append(entry)
        record = processor._transform_record({"text": processor._normalize_text(text), "schema": schema})
        out["cases"].append({
            "text": text, "tasks": tasks, "input_ids": record.input_ids,
            "probabilities": {name: probs for name, probs in captured}})
    write_json(Path(args.output), out)


EMBEDDING_GEMMA2_CASES = [
    ("query", [("text", "how do I rotate an API key")]),
    ("document", [("text", "Rotate keys from the account settings page, then update every client.")]),
    ("document", [("text", 'func greet(name: String) -> String { "Hello, \\(name)" }')]),
    ("document", [("image",)]),
    ("document", [("audio",)]),
    ("document", [("text", "Mars: "), ("image",), ("text", "\nObservation: "), ("audio",)]),
]


def embeddinggemma(args) -> None:
    """Transformers' EmbeddingGemma 2 in float32: processor, tokenizer, model, mean pool."""
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32).eval()
    image = Image.fromarray(synthetic_rgb(320, 240), "RGB")
    pcm = synthetic_pcm(20800).astype(np.float32) / 32768.0
    prompts = {"query": "task: search result | query: ", "document": "title: none | text: "}
    cases = []
    for purpose, content in EMBEDDING_GEMMA2_CASES:
        text = prompts[purpose]
        images, audio = [], []
        for item in content:
            if item[0] == "text":
                text += item[1]
            elif item[0] == "image":
                text += processor.image_token
                images.append(image)
            else:
                text += processor.audio_token
                audio.append(pcm)
        kwargs = {}
        if images:
            kwargs["images"] = [images]
        if audio:
            kwargs["audio"] = audio
        inputs = processor(text=[text], return_tensors="pt", **kwargs)
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"][..., None].to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        values = torch.nn.functional.normalize(pooled, dim=-1)[0]
        cases.append({
            "purpose": purpose, "content": [list(item) for item in content],
            "token_count": int(inputs["input_ids"].shape[1]),
            "input_ids": inputs["input_ids"][0].tolist(),
            "values": values.tolist()})
    write_json(Path(args.output), {
        "model": args.model_id, "revision": args.revision, "image": [320, 240],
        "audio_samples": 20800, "cases": cases})


# ---------------------------------------------------------------------------
# LFM2-VL (d1-3B's backbone): Transformers' image processor, markup, and model.

LFM2VL_IMAGE_SIZES = [
    (5, 4), (17, 33), (300, 200), (512, 512), (640, 512), (700, 1100), (1100, 700),
    (1024, 1024), (1280, 1024), (2000, 300), (300, 2000), (1500, 1499), (3000, 900),
]
LFM2VL_RATIO_LIMITS = [(2, 10), (1, 1), (1, 6), (2, 4), (4, 4), (3, 12), (2, 16), (1, 30)]
LFM2VL_TOKENS = {"<image>": 10, "<|image_start|>": 11, "<|image_end|>": 12, "<|img_thumbnail|>": 13}
for _r in range(1, 11):
    for _c in range(1, 11):
        LFM2VL_TOKENS[f"<|img_row_{_r}_col_{_c}|>"] = 100 + 10 * (_r - 1) + (_c - 1)
TINY_LFM2VL_CONFIG = {
    "model_type": "lfm2_vl",
    "downsample_factor": 2,
    "image_token_id": 10,
    "projector_bias": True,
    "projector_hidden_act": "gelu",
    "projector_hidden_size": 48,
    "projector_use_layernorm": False,
    "tie_word_embeddings": True,
    "text_config": {
        "model_type": "lfm2", "vocab_size": 256, "hidden_size": 64, "intermediate_size": 96,
        "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "layer_types": ["conv", "full_attention", "conv", "full_attention"],
        "block_auto_adjust_ff_dim": False, "conv_L_cache": 3, "conv_bias": False,
        "norm_eps": 1e-5, "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
        "tie_word_embeddings": True,
    },
    "vision_config": {
        "model_type": "siglip2_vision_model", "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_channels": 3, "num_patches": 64,
        "patch_size": 16, "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6,
        "vision_use_head": False,
    },
}
# The image settings of LiquidAI/d1-3B (processor_config.json at 051bcc4).
D1_3B_IMAGE_PROCESSOR = {
    "do_image_splitting": True, "downsample_factor": 2, "encoder_patch_size": 16,
    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "max_image_tokens": 256,
    "max_num_patches": 1024, "max_pixels_tolerance": 2.0, "max_tiles": 10, "min_image_tokens": 64,
    "min_tiles": 2, "resample": 3, "tile_size": 512, "use_thumbnail": True,
    "return_row_col_info": True,
}


def lfm2vl(args) -> None:
    import types

    import torch
    from PIL import Image
    from safetensors.torch import save_file
    from transformers import Lfm2VlConfig, Lfm2VlForConditionalGeneration
    from transformers.models.lfm2_vl.image_processing_lfm2_vl import Lfm2VlImageProcessor
    from transformers.models.lfm2_vl.processing_lfm2_vl import Lfm2VlProcessor

    out = Path(args.output)

    def processor(**overrides):
        return Lfm2VlImageProcessor(**{**D1_3B_IMAGE_PROCESSOR, **overrides})

    def markup(image_processor, outputs, index):
        stub = types.SimpleNamespace(
            image_processor=image_processor, image_token="<image>",
            image_start_token="<|image_start|>", image_end_token="<|image_end|>",
            image_thumbnail_token="<|img_thumbnail|>")
        stub._build_image_tokens = types.MethodType(Lfm2VlProcessor._build_image_tokens, stub)
        stub._compute_tokens_per_tile = types.MethodType(Lfm2VlProcessor._compute_tokens_per_tile, stub)
        stub._compute_tokens_for_image = types.MethodType(Lfm2VlProcessor._compute_tokens_for_image, stub)
        stub._get_image_num_tokens = types.MethodType(Lfm2VlProcessor._get_image_num_tokens, stub)
        return Lfm2VlProcessor.replace_image_token(
            stub, outputs, image_idx=index, use_image_special_tokens=True)

    def crops_of(outputs):
        pixels = outputs["pixel_values"]
        masks = outputs["pixel_attention_mask"]
        shapes = outputs["spatial_shapes"].tolist()
        result = []
        for i, (h, w) in enumerate(shapes):
            n = int(masks[i].sum())
            values = torch.round(pixels[i, :n] * 127.5 + 127.5).to(torch.int64)
            result.append({"rows": h, "columns": w, "patches": n, "checksum": checksum(values.numpy())})
        return result

    ratios = []
    for low, high in LFM2VL_RATIO_LIMITS:
        ratios.append({"min_tiles": low, "max_tiles": high,
                       "ratios": [list(r) for r in processor()._target_ratios(low, high)]})

    layouts = []
    for resample in (3, 2, 1):
        image_processor = processor(resample=resample)
        for width, height in LFM2VL_IMAGE_SIZES:
            image = Image.fromarray(synthetic_rgb(width, height))
            outputs = image_processor([image], return_tensors="pt")
            layouts.append({
                "width": width, "height": height, "resample": resample,
                "rows": int(outputs["image_rows"][0]), "columns": int(outputs["image_cols"][0]),
                "size": [int(x) for x in outputs["image_sizes"][0]], "crops": crops_of(outputs),
                "markup": markup(image_processor, outputs, 0),
            })
    unsplit = processor(do_image_splitting=False)
    for width, height in [(1100, 700), (3000, 900)]:
        outputs = unsplit([Image.fromarray(synthetic_rgb(width, height))], return_tensors="pt")
        layouts.append({
            "width": width, "height": height, "resample": 3, "do_image_splitting": False,
            "rows": int(outputs["image_rows"][0]), "columns": int(outputs["image_cols"][0]),
            "size": [int(x) for x in outputs["image_sizes"][0]], "crops": crops_of(outputs),
            "markup": markup(unsplit, outputs, 0),
        })

    # A tiny random model over a prompt with a single-crop image and a tiled one.
    torch.manual_seed(0)
    config = Lfm2VlConfig(**TINY_LFM2VL_CONFIG)
    model = Lfm2VlForConditionalGeneration(config).eval()
    randomize(model, torch, seed=11)
    tiny = out / "tiny-lfm2-vl"
    tiny.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, str(tiny / "model.safetensors"), metadata={"format": "pt"})
    write_json(tiny / "config.json", TINY_LFM2VL_CONFIG)

    sizes = [(300, 200), (1100, 700)]
    image_processor = processor()
    images = [Image.fromarray(synthetic_rgb(w, h)) for w, h in sizes]
    outputs = image_processor(images, return_tensors="pt")
    words = re.compile(r"<\|[a-z0-9_]+\|>|<image>")
    ids = [1, 40, 41, 42]
    for index in range(len(images)):
        ids += [LFM2VL_TOKENS[t] for t in words.findall(markup(image_processor, outputs, index))]
        ids += [43 + index, 50]
    ids += [60, 61, 62]
    with torch.no_grad():
        logits = model(
            input_ids=torch.tensor([ids]), pixel_values=outputs["pixel_values"],
            pixel_attention_mask=outputs["pixel_attention_mask"],
            spatial_shapes=outputs["spatial_shapes"]).logits[0]
    write_json(out / "lfm2-vl.json", {
        "transformers": __import__("transformers").__version__,
        "tokens": LFM2VL_TOKENS, "ratios": ratios, "layouts": layouts,
        "tiny": {"images": [list(s) for s in sizes], "input_ids": ids,
                 "last": logits[-1].tolist(), "logits": summary(logits.numpy())},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--d1-source", required=True)
    s.add_argument("--output", required=True)
    for name in ("d1", "gliner", "embeddinggemma"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="local snapshot directory")
        p.add_argument("--model-id", required=True)
        p.add_argument("--revision", required=True)
        p.add_argument("--output", required=True)
    v = sub.add_parser("lfm2vl")
    v.add_argument("--output", required=True)
    args = parser.parse_args()
    {"synthetic": synthetic, "d1": d1, "gliner": gliner, "embeddinggemma": embeddinggemma,
     "lfm2vl": lfm2vl}[args.command](args)


if __name__ == "__main__":
    main()
