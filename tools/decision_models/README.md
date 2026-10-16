# Decision model fixtures

`generate_fixtures.py` writes the reference data in
`Tests/MLXDecisionTests/Resources`. Model weights are never committed.

## Environment

A Python 3.14 virtual environment with `torch` 2.14, `torchvision` 0.29,
`transformers` 5.20, `gliner2` at `b86d35c` (installed with its `local`
extra), and `mlx-vlm` at `4623cf9`. The d1-omni reference sources come from
`LiquidAI/d1-omni-600M` at `02b55d7`: put `modeling_d1.py`, `encoder.py`,
`vision.py`, `audio.py`, and `prompt.py` in a directory with an empty
`__init__.py`. Keep downloaded code and checkpoints in their own directories
and run Python with `-I`.

## Commands

```sh
python -I generate_fixtures.py synthetic --d1-source path/to/d1ref \
    --output Tests/MLXDecisionTests/Resources
python -I generate_fixtures.py d1 --model <snapshot of LiquidAI/d1-omni-600M-MLX-fp16> \
    --model-id LiquidAI/d1-omni-600M-MLX-fp16 --revision <snapshot sha> \
    --output Tests/MLXDecisionTests/Resources/d1-omni-mlx-fp16.json
python -I generate_fixtures.py d1 --model <snapshot of LiquidAI/d1-omni-600M-MLX-8bit> \
    --model-id LiquidAI/d1-omni-600M-MLX-8bit --revision <snapshot sha> \
    --output Tests/MLXDecisionTests/Resources/d1-omni-mlx-8bit.json
python -I generate_fixtures.py gliner --model <snapshot of fastino/GLiNER2.5-Decide> \
    --model-id fastino/GLiNER2.5-Decide --revision <snapshot sha> \
    --output Tests/MLXDecisionTests/Resources/gliner2.5-decide.json
python -I generate_fixtures.py lfm2vl --output Tests/LFM2VLTests/Resources
python -I generate_fixtures.py d1-3b --model <snapshot of LiquidAI/d1-3B-MLX-8bit> \
    --model-id LiquidAI/d1-3B-MLX-8bit --revision 9409cbf3bd5067e1a9b17437b1970f6a0c299ccd \
    --output Tests/LFM2VLTests/Resources/d1-3b-mlx-8bit.json
python -I generate_fixtures.py embeddinggemma --model <snapshot of google/embeddinggemma-2> \
    --model-id google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b \
    --output Tests/EmbeddingGemma2CheckpointTests/Resources/embeddinggemma-2.json
```

`lfm2vl` records Transformers' LFM2-VL tile ratios, layouts, crop bytes, and
image markup, and the logits of a tiny random `Lfm2VlForConditionalGeneration`.
`d1-3b` runs d1-3B's MLX export through MLX-VLM on the crops of Transformers'
`Lfm2VlImageProcessor`, which the PyTorch reference uses (MLX-VLM's own
processor resizes with PIL).

`synthetic` builds a tiny random d1-omni and DeBERTa-v2 encoder, runs them
through the PyTorch references, and records prompt and schema encodings made
with fake tokenizers that the Swift tests reimplement. The other commands run
the published checkpoints through MLX-VLM (d1-omni), the `gliner2` package
(GLiNER2.5-Decide), and Transformers in float32 (EmbeddingGemma 2, including its
processor).

The Swift checkpoint tests run when these variables name local snapshots; with
`xcodebuild`, prefix each with `TEST_RUNNER_`:

- `MLX_DECISION_D1_OMNI_FP16`
- `MLX_DECISION_D1_OMNI_8BIT`
- `MLX_DECISION_GLINER2_DECIDE`
- `MLX_EMBEDDING_GEMMA2_BF16` (a BF16 export or the Google snapshot)
- `MLX_EMBEDDING_GEMMA2_8BIT`
- `MLX_LFM2VL_D1_3B_8BIT`

## Running the tests

Run them with the package scheme. `-only-testing` limits the run to these
targets, though `xcodebuild` still builds every test target:

```sh
TEST_RUNNER_MLX_DECISION_D1_OMNI_FP16=<snapshot> \
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
    -skipPackagePluginValidation -only-testing:MLXDecisionTests \
    -only-testing:LFM2VLTests -only-testing:EmbeddingGemma2CheckpointTests
```
