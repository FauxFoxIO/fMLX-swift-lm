# Model-specific text processing in fMLX

The `FMLXText` product owns the concrete local tokenizer integration. It uses
[Swift Tokenizers 0.7.3](https://github.com/DePasqualeOrg/swift-tokenizers/tree/0.7.3) as an engine, not as a universal vocabulary. Each
`CheckpointTextProcessor` loads the selected checkpoint's `tokenizer.json`,
tokenizer configuration and chat-template sidecars. No Python process, remote
tokenizer service, model download or generic chat-template fallback is used.

```swift
import FMLXText
import MLXLMCommon

let text = try await CheckpointTextProcessor.load(directory: checkpointDirectory)
let tokens = try text.prepareChat(
    messages: [["role": "user", "content": "Hello"]],
    additionalContext: ["enable_thinking": false]
)
try text.validateContext(promptTokenCount: tokens.count, maximumOutputTokens: 128)
```

Only pass template options the selected model supports. Structured messages retain
tool-call IDs, function arguments and tool results; tool schemas pass directly to
the checkpoint's template. Named `tool_use`/`default` selection and sidecar
precedence follow the tokenizer implementation. `tokens.count` is the exact input
count including the rendered conversation and tool framing. No truncation occurs.
Reserved context keys cannot replace messages, tools or checkpoint special tokens.

`generation_config.json` EOS IDs override model-config EOS IDs, then tokenizer EOS
and explicitly supplied extra EOS tokens are merged. Unknown/out-of-range token
IDs fail. Model vocabularies may be padded beyond the tokenizer vocabulary, but
tokenizer IDs cannot exceed the model embeddings. The model's text configuration
supplies its context window; an optional caller capacity can lower that limit.

## Native runtime pairing

`NativeTextModel.load(directory:modelRevision:configuration:)` loads audited native
weights and their matching text processor together. `prepareRequest` applies that
model's template and carries its stop IDs and prefix identity into a
`ConcurrentTextRuntime.Request`. Request state remains owned by the existing native
scheduler. Use one `makeDecoder()` result per generation; the tokenizer itself is
immutable and shareable, but incremental decoding state is not.

`generateText(request:tools:)` provides the paired text-level stream for first-party
consumers. It owns one checkpoint decoder per generation, separates reasoning,
parses the checkpoint's tool-call format against the supplied tool names, and
reports usage and the terminal stop reason. Consumer cancellation cancels the
underlying scheduler request. A bounded event buffer fails an unresponsive
consumer rather than silently dropping text. Mirage's daemon uses this adapter
for its in-process Responses transport; callers do not need an HTTP tokenizer or
another mutable decoding singleton.

The model revision must identify immutable checkpoint weights, not a branch or
mutable directory name. Tokenizer/template identity hashes include the engine
version and checkpoint assets. Changed text assets invalidate cache reuse, and
changes detected during loading are rejected. The bundled loader does not apply
LoRA adapters or attach an MTP head. Those advanced callers can use the text
processor with the existing raw runtime, supplying their correct adapter/drafter
and cache identities.

This is a text-only integration. It does not turn an unsupported architecture
into a supported native model or add multimodal preprocessing. Missing
`tokenizer.json`, missing chat templates, incompatible vocabularies, and tokenizer
padding/truncation settings fail explicitly. Checkpoint string-stop metadata is
exposed; the raw request convenience rejects it until a text-stream matcher is
provided rather than silently ignoring it. Existing `MLXLMCommon.Tokenizer`
integrations remain source-compatible; this product keeps the new backend's
throwing encode/decode behavior instead of hiding failures in a nonthrowing adapter.

## Verification

`FMLXTextTests` covers checkpoint-dependent token IDs, chat/template selection,
structured tools, EOS precedence, Unicode decoding, independent concurrent
preparation, prefix identity changes, exact context accounting and missing/invalid
assets. A tiny real MLX checkpoint covers the bundled native load/request path.
The optional `FMLX_TOKENIZER_CHECKPOINT` environment variable exercises installed
checkpoint files without downloading weights; `FMLX_EXPECTED_RAW_TOKENS` accepts a
JSON token array from an independent reference implementation for the probe text.

On September 3, 2026, all nine focused tests passed. The installed
Qwen3-0.6B-4bit and Qwen3.8-27B-4bit checkpoints both matched Python tokenizers
0.23.2 exactly on the fixed multilingual/code probe (18 raw tokens, with different
IDs for each checkpoint). Their model-specific chat templates produced 30 tokens;
EOS resolved to `[151645]` and `[248044, 248046]`, respectively. Whole and
incremental Unicode decoding matched the original input. This verifies tokenizer
integration, not the complete Mirage backend or model-quality/performance claims.

Build cache for this integration:
`/Users/ethan/Library/Caches/CodexBuilds/01a06956-422a-7bf1-8ff1-4ca2d74f9b86/fmlx-swift-lm-xcode-27A5252f-macos`.
DerivedData, SourcePackages, module caches and SwiftPM scratch output stay there.
For direct `xcrun xctest` runs, run from this external directory and set
`LLVM_PROFILE_FILE` there too; otherwise the instrumented bundle writes
`default.profraw` into the working directory. The first probe's generated profile
was moved into this cache, and subsequent probes explicitly redirected profiling.
