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
| Qwen MTP / DFlash2 | Per-request target/drafter state, row-batched target verification, stochastic ratio verification, bounded shifted-prompt prefill, existing rollback machinery | Matching trained head; unquantized target KV; trained DFlash2 is capped at parity-verified block size four |
| Prompt lookup | Greedy prompt-derived candidates verified by the target without a draft model | Opt-in; simple attention or checkpoint-capable Qwen hybrid KV, without quantization or streamed experts; hybrid rewind is capped at three drafts |
| Prefix snapshots | Immutable hot copies and optional persistent restart restore | Simple, affine quantized, Mamba, rotating draft KV, and paired target/drafter state |
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
`Request.promptLookup` is a separate opt-in. On supported cache layouts, an
explicit lookup request takes precedence over the trained drafter and uses
`Configuration.promptLookupDraftTokens` proposals per verification. Hybrid Qwen
targets checkpoint recurrent state without retaining drafter tensors. Unsupported
models and non-greedy requests report a fallback and keep their ordinary or MTP
path. The default remains off, and an acceptance rate alone does not establish a
speedup: compare matched edit prompts against ordinary and trained-draft decode.
Capabilities and execution/fallback events disclose the selected path. Quantized
target KV uses ordinary decoding. Qwen MTP and DFlash2 reuse paired target/drafter
snapshots at existing chunk boundaries, including across runtime restarts when
persistent storage is configured. The drafter needs one
lookahead token inside the allowed prefix, so a 128-token chunk size and 2048-token
prefix permit 1920 tokens of reuse. Other drafters must explicitly support this
state-copy contract. Paired persistent entries use a separate layout namespace
and record the processed target frontier independently from the lookahead token. The separate trained
head's bare parameter keys are normalized to
`mtp.*`; already-converted normalization values are preserved.

When two or more compatible speculative requests reach a verification boundary
with the same block width, the runtime evaluates their target rows in one forward.
Attention remains row-local and recurrent rollback checkpoints are split back to
each request before acceptance. Concurrent verification is limited to two positions:
wider real-model batch shapes changed the greedy result even though the
corresponding single-row shape preserved parity. Shared-target-KV drafters remain
interleaved.

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
Ordinary and MTP snapshots share one hot-cache byte budget and LRU, but never
restore each other's state. A cache miss or insufficient prefix budget stays cold.
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

`InstalledPromptLookupComparisonTests` compares ordinary and prompt-lookup greedy
decode on the same rendered tokens, alternating order after a warmup with no
prefix cache and a 96-token output cap. On a Mac Studio M4 Max (128 GB,
macOS 27.0), pinned `mlx-community/Qwen3-0.6B-4bit` at
`73e3e38d981303bc594367cd910ea6eb48349da8` (weight SHA-256
`392e8d466d56100ada00eb82031fb854297fc9e389b7d303eba3af114e87bce2`)
matched output exactly in all seven Release trials per workload at verification
width eight. Across six measured
pairs, median decode rates were 348.8 versus 967.3 tokens/s for a copy-style
request (the model changed one interpolation), 349.6 versus 866.6 for a rename
request, and 352.3 versus 354.4 for new code.
Those are 2.77×, 2.48×, and 1.01× of ordinary decode respectively.
Lookup accepted 264 of 294 proposals on the copy request and 246 of 294 on
the rename request.
The rename reply copied the original source without performing the requested
edit; it does not establish successful-edit speed. Prompt lookup proposed no
tokens on the new-code control. First-token times were similar after warmup.
These results qualify one model, device and set of prompts, not other models,
batch shapes, concurrent traffic or iPhone execution. Flow leaves lookup off by
default.

`InstalledHybridPromptLookupTests` exercised a successful 67-token Swift rename
with Flow's Qwen3.8-27B target at `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`
and matching MTP companion at `b643c01b6d3b094e325edb6ebd832e16c486c575`.
The target's three weight shards have SHA-256 digests
`6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d`,
`83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670`, and
`31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a`;
the companion is `76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6`.
The target and companion stayed resident for all three modes. On the same M4 Max,
Release trials rotated mode order. At width two, one warmup preceded six measured
pairs, with exact output in all seven trials. Median decode rates were 28.1
tokens/s ordinary, 42.6 lookup and 54.6 MTP; lookup accepted 144/180 proposals
and MTP accepted 306/306. First-token medians were 521, 522 and 533 ms. A separate
run configured at width four (capped to three by the hybrid target) had one
warmup and two measured pairs: lookup reached 43.2 tokens/s
versus 54.0 for MTP, accepting 37/51 versus 51/51 proposals per request. Output
again matched exactly.
This is evidence for hybrid lookup correctness on one device and prompt, and for
preferring the trained draft on this request. A two-request Release check matched
each request's sequential output while ordinary steps reached 80 width-two
batches. Cancelling a lookup request after ten tokens left its peer's complete
output unchanged and no active request state. This covers one concurrent
scenario, not aggregate-throughput gains or iPhone output parity.

On the same pinned target and companion, a second Release workload asked for an
exact copy of the supplied Swift source. It returned the 62-token source exactly
in all seven order-rotated trials. With one warmup and six measured pairs,
ordinary decode reached a median 28.2 tokens/s, lookup width three 57.1, and
trained MTP 53.3. Lookup accepted 46/48 proposals per trial; MTP accepted
48/48. Lookup was 7.0% faster than MTP on this copy request while remaining
slower on the rename above. Both modes produced the same token sequence as
ordinary decode. These two workloads justify making the Flow lookup opt-in
available even when a trained drafter is installed, without claiming a single
mode wins across edits.
A width-seven copy probe also preserved exact output, but measured 48.7
tokens/s in its one measured pair. It uses the generic recurrent checkpoint
path; the production cap remains three proposals for this hybrid target.
With one proposal, the same copy output also matched exactly, but lookup
reached 49.2 tokens/s versus 53.7 for MTP across two measured pairs. The
three-proposal result above is specific to this pinned target and workload.

For a new Swift function on the same target, prompt lookup proposed no tokens.
One warmup and three measured Release pairs produced identical 80-token outputs:
ordinary and lookup each reached a median 26.3 tokens/s, while the trained MTP
head reached 49.6. A prototype that switched between lookup and MTP per round
also preserved exact output. It retained MTP-like speed on this new-code request,
but was slower than pure lookup on the measured copy trial and slower than pure
MTP on the measured rename trial. It was removed; keeping its drafter cache in
sync after each lookup round was one added cost, but that cost was not isolated
from the rest of the hybrid path.

An experimental one-pass eight-position GDN checkpoint kernel matched chained
recurrence bitwise and preserved real-model copy output. It lifted a width-seven
lookup probe from about 48.7 to 52.3 tokens/s across separate runs, while an
adjacent width-three control reached 55.7. Seven recurrent checkpoints for this
48-GDN-layer target reserve roughly 1 GB of state, so the three-proposal cap
remains. These probes point to multi-row quantized projection and checkpoint
memory costs as the next areas to measure, rather than proving either is the
sole cause of the gap.

The pinned target contains 15.13 GB of non-vision safetensor data. At 28.2
ordinary tokens/s, one read of those bytes per token would imply about 427
GB/s; the local 40-core M4 Max is [specified at 546 GB/s](https://support.apple.com/en-au/122211), giving an optimistic
one-read ceiling near 36 tokens/s. File bytes are not measured GPU traffic, so
this is only a roofline estimate. A Metal System Trace of one copy comparison
showed the test process active on the GPU for 92.5% of a five-second inference
window. That window includes several decode modes and does not provide
per-shader timings. It suggests limited host-idle headroom on this target,
while leaving quantized projection and recurrent costs to profile directly.

The prompt-candidate index now stops scanning once it finds the longest possible
ngram with a full requested continuation. In a Release CPU microbenchmark with a
32,768-token repeated prompt and 1,000 lookups varying the continuation limit
from four to eleven, search time fell from 73.1 ms to 0.063 ms total. This is a
worst-case lookup measurement, not an end-to-end generation speedup. Two more
Qwen3.8 copy trials after the change preserved exact output and 46/48 lookup
acceptance.

An experimental Qwen3.8 affine-Q4 GDN projection kernel tiled four or eight input
rows over one weight read. Across six warmed direct Release runs on the pinned
checkpoint, the four-row projection median fell from 0.423 to 0.352 ms and the
eight-row median from 0.578 to 0.504 ms. All compared outputs met the focused
BF16 tolerance of `max(0.125, 3% of the MLX reference)`. An initial full-model
A/B used MTP block sizes three and seven, but the candidate was gated to exactly
four or eight rows. That A/B never activated the kernel and establishes no
end-to-end speed or greedy-token result for it. A corrected block-four A/B
logged 48 specialized GDN dispatches per installed-model run. Exact greedy
tokens and MTP acceptance matched the off path. After one anomalously slow off
copy run, alternating process medians gave only about 0.7% faster copy and 2.1%
faster new code than the later off control. Those small differences do not
establish a material end-to-end gain, so the production route and experimental
kernel were removed. The installed Qwen3.8 MTP
head and native rewind contract cap full-model verification at four rows; eight
rows have projection-only evidence until those contracts change.

## Smaller text models and Woof

On the same M4 Max, `InstalledQwenBenchmarkTests` ran a 128-token chat prompt
and 128-token greedy reply in Release, with one warmup and two measured trials
per model. Median ordinary decode rates were 106.0 tokens/s for
`ConwayResearch/Underdog-Woof-4B-1.1` at
`cf5f8db5409258e73303b78e112051fc443cb02b`, 109.3 for
`mlx-community/Qwen3.5-4B-4bit` at
`0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`, 220.1 for
`mlx-community/Qwen3.5-2B-4bit` at
`674aaa7240b91e8012fcad5d791b7dfe5ba90207`, and 312.3 for
`mlx-community/Qwen3.5-0.8B-4bit` at
`da28692b5f139cb0ec58a356b437486b7dac7462`. These are speed
measurements, not evidence that the smaller models can complete the same agent
tasks. The 4B checkpoints contain 2.367 GB of non-vision tensors each; the 2B
and 0.8B checkpoints contain 1.059 and 0.424 GB. Vision tensors in the Qwen
downloads are not part of text decode.

The same ordinary benchmark measured 176.2 tokens/s for
`openbmb/MiniCPM5-2B-MLX` on the 128-token prompt. Its published model card
reports stronger agent evaluations than several 4B models, but this local run
measures only speed and correct loading through the Llama-shaped model path.
Its `<function name="...">` tool-call format needs a parser before Flow can use
it as a tool-calling explorer.

The pinned Woof checkpoint also ran an installed prompt-lookup benchmark at
width four with three alternating-order trials per workload. The 62-token copy
measured 119–121 ordinary versus 254–266 lookup tokens/s, accepting 46/48
proposals. A successful 62-token rename measured 121–122 versus 202–203,
accepting 38/51. New Swift code measured 121–124 versus 120–123, with no
lookup proposals. All nine paired trials had exact greedy token parity and no
fallback. The generated new-code function mishandled inputs zero and one; these
speed results do not establish model quality. Woof's published Husky rates were
measured on an M5 Max with a separate, unpublished trained Flash draft and
different prompts, so these M4 Max trials are not a same-hardware comparison.
The named Woof model preset uses the existing Qwen3.5 loading path. Flow keeps
lookup opt-in and selects the measured width four only for this pinned Woof
revision.

With the Qwen3.5-4B Q4 target and its BF16 MTP companion, paired greedy trials
measured about 122 ordinary versus 171 tokens/s with speculation. The Q4 MTP
companion also reached about 171 tokens/s. Both produced the same 128 tokens as
ordinary decoding; the BF16 head accepted 88 of 117 proposals in one measured
trial. A research-only pairing of Woof with the Qwen3.5-4B BF16 head measured
about 123 ordinary versus 170 tokens/s, also with exact greedy output and
87/119 accepted proposals in one trial. The Woof publisher does not provide
this head or document Woof's training lineage, so this pairing is not enabled
automatically. These short trials do not qualify other prompts or devices.

Build artifacts and disposable benchmark logs use the conversation cache outside
the repository.
This implementation and its documentation were AI-assisted.
