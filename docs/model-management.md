# Native model management

`FMLXModelManagement` is a macOS 14-compatible model catalog and download layer. It has no
Python process, mounted runtime image, or HTTP inference server. Applications choose their model
and cache roots and keep inference in-process through `FMLXText`.

## Surface

- `FMLXModelStore.installedModels()` discovers existing `owner/model` directories. A checkpoint
  must have a supported `model_type`, `tokenizer.json`, and every shard named by its weight index.
- `search(query:)` and `HuggingFaceModelHub.repository` use public Hugging Face metadata. Search
  results are restricted to fMLX's native scheduled text architectures.
- `download(repositoryID:)` resolves a branch/tag to an immutable commit, persists a transaction,
  resumes partial files with HTTP Range, checks declared byte counts and LFS SHA-256 values, then
  atomically moves the validated staging directory into the installed catalog.
- `cancelDownload(id:)`, `retryDownload(id:)`, and `recoverDownloads()` preserve partial files.
  Recovery resumes work that was pending, downloading, or validating when the service stopped.
- `cacheInfo()`, `clearCache()`, and `deleteModel(repositoryID:)` operate only inside the configured
  roots and reject symbolic-link catalog entries.

Model files with JSON, safetensors, Jinja, tokenizer text/model, README, and license extensions are
retained. This covers fMLX text checkpoints and nested auxiliary weights while preserving upstream
license notices. Unrelated repository assets and Python/server files are intentionally omitted.

## oMLX parity used by Mirage

| Mirage capability | First-party owner |
| --- | --- |
| Runtime availability/lifecycle | fMLX is linked in-process; availability is platform support |
| Installed catalog and metadata | `FMLXModelStore` |
| Repository search and metadata | `HuggingFaceModelHub` |
| Progress, durable errors, cancellation, retry, restart recovery | download transactions |
| Resumable transfer, validation, atomic install | `ModelFileTransfer` + `FMLXModelStore` |
| Delete, cache inspect, cache clear | `FMLXModelStore` |
| Load/unload, residency, admission, cancellation | `FMLXText.NativeTextModel` and `ConcurrentTextRuntime` |
| Checkpoint-specific tokenization | `CheckpointTextProcessor` |
| Persistent prefix cache | `RuntimePersistentCacheConfiguration` |
| Embedded Qwen 3.5/3.6 MTP | `NativeTextModelLoader.loadEmbeddedMTP` |

The scheduled runtime currently accepts Llama/Mistral, Qwen 3, and Qwen 3.5-family text models.
Search and installation reject other architectures rather than exposing a model that inference
cannot load. Multimodal tensors may be present in a combined Qwen checkpoint, but this surface runs
its text model only; image/video request processing remains outside `FMLXText`.

The combined `OsaurusAI/Qwen3.6-35B-A3B-MXFP4-MTP` checkpoint declares
`qwen3_5_moe`, `text_config.mtp_num_hidden_layers = 1`, and embedded `mtp.*` weights. Its normal
target load filters MTP tensors, while `loadEmbeddedMTP` loads those same tensors into the matching
drafter and gives both models to `ConcurrentTextRuntime` for verified speculative decoding.
