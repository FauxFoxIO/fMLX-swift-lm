# Native concurrent text runtime

`MLXLMCommon.ConcurrentTextRuntime` owns one model and its request state.
`NativeInferenceRuntime` routes model IDs and shares admission accounting across
models. These are additive native Swift APIs; no transport, Foundation Models,
chat policy, package product change, or deployment-floor increase is required.

## Supported execution

| Path | Behavior | Limits |
| --- | --- | --- |
| Llama/Mistral, Qwen3 | Chunked prefill and continuous batched decode | Simple or affine 4/8-bit attention KV |
| Qwen3.5/3.8 text and hybrid wrappers | Batched projections with independent attention and packed recurrent rows | Exact Mamba caches; multimodal inputs excluded |
| Qwen MTP | Per-request target/drafter state, bounded shifted-prompt prefill, existing rollback machinery | Matching trained head, greedy sampling, block size two; unquantized target KV |
| Prefix snapshots | Immutable hot copies and optional persistent restart restore | Simple, affine quantized, Mamba; MTP requests start cold |
| Public Core ML/ANE | Isolated feasibility and transfer probe | No production acceleration claim; see public-accelerator-evidence.md |

Batched decode evaluates projections and MLPs across rows, then each row's
attention against its own native cache. It supports unequal context lengths
without attention padding; it is not one fused attention kernel. Prefill chunks
and MTP rounds interleave between requests. GPU work settles before actor state
changes. Cancellation cannot interrupt a synchronous Metal forward.

The per-slot batch-view design follows Osaurus AI's MIT-licensed `BatchKVCache`
at vMLX commit `4546a5d720e7013adffdbddd728c6106e4f9e637`. No package fork or full
engine transplant is required. Upstream `e3d4a20e9e20e7b8ab39aded7bbfad4ae22c9438`
already supplies cache copies, batch-position RoPE, speculative rounds and hybrid
rollback; the new layer schedules and combines those facilities.

## Embedding

Load and configure weights before transferring exclusive model ownership. The
`sending` parameters prohibit retaining an unsynchronized second model owner.
`NativeTextModelLoader.load(directory:)` loads audited local text models;
`loadMTP(directory:)` loads a standalone preconverted Qwen MTP head. Neither
method downloads weights. The caller must select the trained head matching its
target revision, not merely a matching architecture name.

```swift
let generation = try await runtime.generate(.init(
    tokens: promptTokens, maxTokens: 128, stopTokenIDs: eosTokenIDs,
    priority: .interactive, prefixTokenCount: stablePrefixTokens.count,
    cacheIdentity: runtime.identity
))
for try await event in generation.events {
    switch event {
    case .token(let token): consume(token)
    case .fallback(let reason): report(reason)
    default: break
    }
}
```

Raw callers own tokenization, chat templates, stop IDs and detokenization.
The owned [FMLXText product](checkpoint-text.md) supplies checkpoint-specific text
processing and a paired native loader without putting tokenizer policy inside
the scheduler or requiring Mirage/Bright Eyes to implement it themselves.
`Request.speculative` defaults to true when a compatible drafter is configured.
Capabilities and execution/fallback events disclose the selected path. Nonzero
temperature and quantized target KV use ordinary decoding. MTP prefix reuse is
disabled because target-only snapshots cannot reconstruct the drafter's shifted
prompt state. The separate trained head's bare parameter keys are normalized to
`mtp.*`; already-converted normalization values are preserved.

Call `cancel(id)` when breaking stream iteration early. Consumer cancellation
also propagates through stream termination. Buffers are bounded; overflow fails
that request with `consumerTooSlow`. A row's consumer failure does not terminate
other batched rows. `shutdown()` settles requests and releases model ownership.

For multiple models, construct `NativeInferenceRuntime`, then call
`load(id:estimatedResidentBytes:loader:)`. Its loader reservation must cover model
weights, the configured hot-prefix budget, and loading workspace. The reservation
is reduced to resident weights and prefix capacity after loading. Use `generate`,
`cancel`, `capabilities`, `clearCaches`, and `unload` with the model ID. Loading the
same ID twice is rejected. Resource status exposes resident and request bytes.

## Admission and cache contracts

Each active request reserves `(prompt + maximum output + 255)` times an FP32 KV
upper bound, recurrent state/rollback storage, optional drafter KV, and configured
working memory. The 255-token allowance covers allocation granularity. Configure
working memory from measurements; this accounting does not impose an OS or Metal
allocation limit. Weights and the hot-prefix budget are counted once per model.

Admission is bounded by slots, queue capacity, and local/global bytes. Waiting
requests retain their order; interactive requests may use reserved headroom while
background admission waits. `interactiveReservedSlots` protects per-model slots;
`interactiveHeadroomBytes` protects service-wide bytes. Active work is not evicted.
Weighted service gives interactive requests three scheduling turns per background
turn, not three times the GPU time. Models registered in one native service also
share a turn arbiter, so only one bounded model forward runs at a time across
models. Cancelling a waiting turn releases it without cancelling a peer.

Prefix identity includes immutable model, tokenizer, template, adapter and cache
layout revisions. A mismatch forces cold generation. Snapshots never alias a
request's mutable cache. Requests must leave a prompt suffix after their prefix.
The persistent store validates identity/layout, tensor metadata, byte lengths and
SHA-256 integrity before reconstructing arrays. Corruption becomes a cold miss.
Disk limits and LRU eviction apply per identity/layout namespace. Files publish
atomically; unrelated files remain untouched. Cache paths must have no symlinked
components. On macOS, canonicalize temporary directories with POSIX `realpath`.
Persistence and snapshot copies are synchronous and can delay a scheduler turn.

## Evidence

Focused fixtures cover unequal-length batched Llama and Qwen hybrid decoding,
affine 4/8-bit KV, shared-prefix branches, MTP incremental prefill and cancellation,
stream overflow, model routing/admission, and persistent corruption/restart cases.
Synthetic fixtures establish behavior, not trained-model quality or speedup.
See `runtime-completion-ledger.md` for current trained-model qualification status.

`ConcurrentTextRuntimeBenchmark` is opt-in using `FMLX_BENCHMARK_FIXTURE`. It
loads only local weights. The JSON fixture supplies tokenized long, conversation,
and short prompts, content identities, budgets, prefix length and trial count.
Optional `longOutputTokens` and `conversationOutputTokens` tune output lengths.
It compares active limits one and four, cold/warm prefixes, and interactive arrivals
during prefill/decode. `testTrainedModelParity` compares batched output against the
existing `TokenIterator`. `scripts/summarize-concurrent-benchmark.py` excludes trial
zero and reports p50/p95 TTFT, aggregate throughput and MLX memory. These are
matched scheduler configurations. `testExistingIteratorSerializedBaseline` also
measures the existing iterator with a serialized queue and the same token budgets.
Its prefill progress callback reports graph submission; runtime prefill events
report completed chunks, so that arrival milestone is not exactly identical.
Decode arrivals use the first returned token in both paths. No external-engine
speedup is claimed.

`testTrainedMTPParity` checks the separate head against ordinary greedy output.
`testTrainedMTPPerformance` alternates ordinary/MTP order with both weights resident,
excludes trial zero in analysis, and records acceptance, TTFT, throughput and memory.
The optional `secondaryFixture` points to a separately tokenized trained model for
`testTrainedMultiModelQualification`, including arrivals during prefill/decode,
cancellation, active unload, exact output parity and reservation release.

Build artifacts use the conversation cache outside the repository. Durable logs,
fixtures and reports are under `/Users/ethan/Documents/fMLX-runtime-01a069c9`.
This implementation and its documentation were AI-assisted.
