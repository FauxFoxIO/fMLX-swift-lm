# Decision model source attribution

The native Swift decision models in `Libraries/MLXDecision` were written with AI
assistance and ported from the following upstream work. The repository's existing
MIT license and attribution remain unchanged.

- Liquid AI d1-omni-600M reference model code (`modeling_d1.py`, `encoder.py`,
  `vision.py`, `audio.py`, `prompt.py`), copyright Liquid AI, Inc., LFM Open
  License v1.0, revision `02b55d7076f15129e59ab3f94783f32c4b088674`:
  https://huggingface.co/LiquidAI/d1-omni-600M/tree/02b55d7076f15129e59ab3f94783f32c4b088674
  The Swift port changes these files: it rewrites them for MLX Swift, loads both
  the PyTorch checkpoint and its MLX exports, and replaces PIL, torchvision, and
  NumPy preprocessing with native image and audio code.
- MLX-VLM, copyright © 2025 Prince Canuma, MIT license, implementation revision
  `4623cf9d2633c5ee2b1ac125eeae51423447c41f`. The `d1_omni`, `lfm2_vl` vision,
  `nemotron_h_nano_omni` Parakeet audio, `gliner2_5` DeBERTa, and interpolation
  ports:
  https://github.com/Blaizzy/mlx-vlm/tree/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/models
- GLiNER2, copyright Fastino AI, Apache License 2.0, revision
  `b86d35c02f2c1a4d9840ea50197bd1b6d13b031e`. Schema transformer, word splitter,
  and the span extractor's classification path:
  https://github.com/fastino-ai/GLiNER2/tree/b86d35c02f2c1a4d9840ea50197bd1b6d13b031e
- Hugging Face Transformers, copyright the Hugging Face team, Apache License
  2.0. DeBERTa-v2 disentangled attention and SigLIP2 vision embeddings:
  https://github.com/huggingface/transformers/tree/main/src/transformers/models/deberta_v2
  https://github.com/huggingface/transformers/tree/main/src/transformers/models/siglip2
- torchvision antialiased bilinear uint8 resize and PyTorch interpolation, copyright
  the PyTorch contributors, BSD license, as MLX-VLM reproduces them.

The license texts are retained alongside this notice as
`DecisionModels-LFM-Open-License-1.0.txt`, `DecisionModels-GLiNER2-Apache-2.0.txt`,
`EmbeddingGemma2-MLX-VLM-MIT.txt`, `EmbeddingGemma2-Transformers-Apache-2.0.txt`,
and `EmbeddingGemma2-PyTorch-BSD.txt`.

Model weights are installed separately and keep their own licenses: d1-omni
weights under the LFM Open License v1.0 (commercial use limited to entities
below the license's revenue threshold), GLiNER2.5-Decide weights under Apache
License 2.0.
