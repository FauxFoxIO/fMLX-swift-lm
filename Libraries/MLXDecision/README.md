# MLXDecision

Native MLX decision models. A decision model answers typed questions about a
state with a probability for every outcome, in one forward pass, without
generating text.

| Model | Checkpoints | Inputs | Loader |
|---|---|---|---|
| Liquid d1-omni-600M (`d1_omni`) | `LiquidAI/d1-omni-600M`, `-MLX-fp16`, `-MLX-8bit` | text, images, 16 kHz audio | `D1OmniLoader` |
| Fastino GLiNER2.5-Decide (`gliner2`, DeBERTa-v3) | `fastino/GLiNER2.5-Decide` | text | `GLiNER2Loader` |
| Liquid d1-2B (`lfm3_vl`) and its fine-tunes | merged checkpoints and their `fmlx-d1` exports | rendered rows, images | `D1LFM3Model` |

d1-omni and GLiNER2 conform to `DecisionModel`:

```swift
import MLXDecision

let model = try await D1OmniLoader.load(
    from: snapshotDirectory,
    artifact: DecisionArtifact(identifier: "LiquidAI/d1-omni-600M-MLX-fp16", revision: snapshotRevision))

let result = try await model.decide(
    DecisionRequest(
        state: "I was charged twice this month, please refund one of them.",
        questions: [
            DecisionQuestion(id: "refund", instructions: "Is the customer asking for a refund?", kind: .yesNo()),
            DecisionQuestion(
                id: "team", instructions: "Which team should handle this?",
                kind: .choice([
                    DecisionOption("billing", description: "Charges, refunds, invoices"),
                    DecisionOption("technical", description: "App or site faults"),
                ])),
            DecisionQuestion(
                id: "urgency", instructions: "How urgent is this?",
                kind: .score(levels: ["Can wait", "Today", "Blocking the customer now"])),
        ]))

result.distributions[1].top          // "billing"
result.distributions[0].probabilities  // [P(yes), P(no)]
```

States are text. For JSON, build a `DecisionJSON` value, such as
`.object([.init("page", .string("Release notes"))])`; its `serialized` text is
exactly what Python's `json.dumps(state, ensure_ascii=False)` writes, the form
d1-omni was trained on.

Loaders read a local snapshot and never download. The installer supplies the
artifact identity and revision.

## d1-omni

- Text, JSON (serialized by the caller), images, or one audio clip per
  request. A request carries images or audio, not both.
- Images are tiled as LFM2-VL tiles them (up to ten 512 px tiles plus a
  thumbnail) and resized exactly as torchvision's uint8 antialiased bilinear
  resize does. Files and encoded images keep their stored 8-bit samples, without
  color management or alpha compositing, as PIL's `convert("RGB")` reads them,
  and are turned upright by EXIF orientation. Media decode before the model's
  lock; the media prefix length is checked before an encoder runs.
- Audio is mixed to mono, resampled to 16 kHz, cut to 30 s, and padded to
  0.5 s. Every 80 ms becomes one prefix position.
- Text answers are divided by the per-type temperatures in `config.json`.
  Image and audio answers are the model's softmax as trained.
- The trunk and decision head load with the model. The vision and audio towers
  load with the first request that needs them (or at load with `preload`), and
  `unload(_:)` releases them.
- `maximumLength` lowers the 16,384-position row limit (media included) for
  tighter memory budgets; states are cut on the right to fit, as the reference
  cuts them.
- `DecisionPrecision.float32` upcasts a float16 export. Quantized exports load
  the layers that ship `.scales` as quantized layers.

## GLiNER2

- `GLiNER2Model.classify(_:tasks:)` mirrors `classify_text`: every task of a
  call reads one joint sequence. Single-label tasks return a softmax,
  multi-label tasks a sigmoid per label.
- As a `DecisionModel`, a choice becomes a task whose labels are the option
  names (descriptions become label descriptions), a score uses its levels as
  labels, and a yes/no question uses `yes` and `no`. The question ID is the task
  name and the instructions are the prompt.
- Text words past `maximumTokens` are dropped; the schema always stays whole.

## d1-2B

Liquid d1-2B is LFM3 text (24 layers: short convolutions, full attention with
d1's `set` mask, and sliding-window attention) with the LFM3-VL vision tower.
`D1LFM3Model` runs it as a stateless trunk for a host that renders d1's prompt
itself, as C0 in Mirage does: one forward per row, no KV cache, and the
final-norm hidden state at each answer slot.

```swift
let model = try D1LFM3Model.load(directory: exportDirectory)  // memory-mapped
let sequence = D1LFM3Sequence(
    ids: ids, stateLength: stateLength, groups: questionGroups, slots: answerSlots)
let images = [D1LFM3Image(pixels: patches, rows: 26, columns: 38)]
let logits = try await model.optionLogits(sequence, images: images, readout: readout)
// readout[k][option] = the option's token forms; softmax(logits[k] / model.temperature!)
```

- `groups` gives each position's question; the state (`stateLength` positions)
  sees and is seen by every position. Images fill the `<|image|>` placeholders
  in order; `pixels` are the processor's patches in merge-block order.
- `hidden(_:images:)` returns the hidden states instead. `optionLogits` reads
  the tied head from the model's own embedding table, so the host keeps no
  separate copy.
- Rows run at their own length; there are no buckets. Right padding
  (`D1LFM3Engine.hidden(_:features:paddedLength:)`) is exact.
- `fmlx-d1 convert` writes exports from the merged PyTorch checkpoint, in one
  floating-point type, with per-tensor quantization from a recipe. Quantized
  exports are MLX affine (`QuantizedLinear`, `QuantizedEmbedding`); the
  scales and minimums come from llama.cpp's importance-weighted search, with an
  importance matrix from `fmlx-d1 imatrix`. `fmlx-d1 sensitivity` measures
  each tensor group's error at 4 bits, to choose which tensors keep more bits.
- Exports load memory-mapped by default, so their weights stay file-backed.

## d1-3B

Liquid d1-3B is an LFM2-VL checkpoint (`lfm2_vl`) that answers through the
language model's next-token logits, so it runs on `MLXVLM.LFM2VL` rather than in
this library. Render d1's own prompt (its `prompt.py`, with one `<image>` token
per picture where the chat template would place it), then pass the token ids
and pictures to `LFM2VLProcessor.prepare(tokens:images:processing:)`. It lays
out, resizes, and marks up each picture exactly as Transformers'
`Lfm2VlProcessor` does for d1-3B. d1's reference first scales pictures above
1 MP down with PIL bicubic; callers do that before `prepare`. Read each option's
best token log-probability at the last position and softmax over the options.
`Tests/LFM2VLTests` checks this path against Transformers and, with a local
`LiquidAI/d1-3B-MLX-8bit` snapshot, against MLX-VLM.

## Parity

`Tests/MLXDecisionTests` checks the Swift code against the reference
implementations: the d1-omni PyTorch code (tiny random model, every modality),
Transformers DeBERTa-v2, the GLiNER2 schema processor, and the published
checkpoints when their snapshots are present. `tools/decision_models` holds the
fixture generator.

See `LICENSES/DecisionModels-NOTICE.md` for attribution.
