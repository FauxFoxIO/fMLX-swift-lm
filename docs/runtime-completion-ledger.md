# Runtime completion ledger

This ledger records the implemented runtime and bounded qualification. Performance
claims apply only to the recorded fixtures, device and configurations.

| Deliverable | Implementation | Verification / remaining work |
| --- | --- | --- |
| Shared weights, isolated requests, bounded streams, cancellation | Implemented | Focused behavioral fixtures passed |
| Continuous batched decode | Batched projections, row-native attention and recurrent packing | Unequal contexts and native/4-bit/8-bit KV parity passed on tiny Llama/hybrid Qwen; trained Qwen native-KV parity passed |
| Multi-model admission and interactive headroom | Shared resource actor and native model registry | Routing, headroom, weighted turns and cancellation passed; trained Qwen3.8/Qwen3 peers preserve exact output through prefill/decode cancellation and active unload |
| Persistent prefix lifecycle | Versioned validated tensor archives and hot snapshots | 14 restart/corruption/eviction/isolation tests passed; both trained models restored native/4-bit/8-bit KV 128-token prefixes with exact parity to each mode’s cold output; disk cap is per namespace |
| Concrete trained model qualification | Local Qwen3.8-27B-4bit loads and runs | Two batched 32-token outputs match existing TokenIterator exactly; trained Qwen3-0.6B also matches both 32-token outputs with 30 batched forwards |
| Scheduled MTP | Incremental target/drafter prefill, existing verified rounds/rollback, explicit fallback | Tiny ordinary/MTP parity, actual speculative-round telemetry, stop handling and hybrid cancellation passed; trained companion passed two concurrent 32-token greedy comparisons and prefill cancellation; measured 23.00 vs 20.30 tokens/s on one 64-token conversation fixture |
| Public ANE path | Public Core ML fixture export, placement and MLX transfer probes | Completed: 0.710 ms total Core ML/ANE vs 0.626 ms resident MLX; no production speedup or trained-state bridge claimed |
| Typed embedding contract | Native model loading/routing, request/cancel/stream/cache lifecycle and capabilities | No transport or chat policy; local loaders cover audited target/head formats |
| Matched mixed-workload evidence | Trained local harness, active limits one/four, cold/warm, prefill/decode arrivals | 168 rounds completed across eight cells; trial zero excluded (20 measured/cell). Interleaving-only control is recorded separately |

Exact MTP alternative: `mlx-community/Qwen3.8-27B-MTP-4bit`, revision
`b643c01b6d3b094e325edb6ebd832e16c486c575`, required download 238,939,914 bytes.
Metadata/header inspection verified 31 tensors and matching tokenizer. The current
local target omitted these tensors; the separately converted head retained them.
The approved target companion and conventional-attention checkpoint have now been
downloaded at their pinned revisions and their weight digests recorded.

Build cache (reused):
`/Users/ethan/Library/Caches/CodexBuilds/fmlx-01a069c9-7a4f-swift6.4-27A5252f-macos`.
Durable reports: `/Users/ethan/Documents/fMLX-runtime-01a069c9`.

The main benchmark shows lower interactive latency with concurrent admission but
lower aggregate throughput and higher active memory in these mixed workloads.
For cold-prefix prefill arrivals, short-request p95 TTFT was 4.92 s with one
active request versus 1.00 s with four and batching; median aggregate throughput
was 14.30 versus 11.62 tokens/s. The interleaving-only control shows that the
throughput choice depends on arrival phase. No universal batching speedup is claimed.

Full results: `/Users/ethan/Documents/fMLX-runtime-01a069c9/runtime-results.md`.
The completed work is committed locally; the exact commit is reported in the task
and durable results. No pushes or pull requests are part of this task.

The interleaving-only control completed 22 rounds (10 measured per arrival phase).
The additional conventional-attention qualification candidate is
`mlx-community/Qwen3-0.6B-4bit` at
`73e3e38d981303bc594367cd910ea6eb48349da8`: 351,386,061 bytes for the repository.
The approved download contains the trained conventional-attention fixture used for
batched, persistent-prefix and cross-model qualification, all completed.

The direct existing-`TokenIterator` serialized reference also completed 22 rounds
(10 measured per arrival phase). Its later run was variable; the report does not
attribute a clean throughput advantage from these unisolated sequential runs.
`testTrainedMTPParity` now executes the approved companion and passes. Exact
downloads, pinned revisions and weight digests are in the durable
`pending-model-downloads.json` manifest.

Trained MTP measurements alternate ordinary/speculative order with both weights
resident (ten measured samples per mode). Acceptance was 280/340, with median
TTFT 196 ms versus 186 ms ordinary. This is a fixture-specific throughput result.
Trained cross-model checks compare each output to its own TokenIterator baseline;
they cover both arrival phases, cancellation and unloading with a surviving peer.

The trained conventional-attention matrix completed 88 rounds (ten measured plus
one excluded warmup per cell). For cold prefill arrivals, short TTFT p95 fell
from 1012 to 80 ms and median aggregate throughput rose from 72.33 to 78.74 tokens/s
with concurrent batching. Its 4-bit KV continuation differs from native KV; each
persistent-restore comparison uses the same quantization mode as its cold baseline.
