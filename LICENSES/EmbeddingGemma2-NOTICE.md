# EmbeddingGemma 2 source attribution

The native Swift EmbeddingGemma 2 implementation was written with AI assistance and
ported from the following upstream work. The repository's existing MIT license
and attribution remain unchanged.

- Google EmbeddingGemma 2 model architecture and configuration, source revision
  `914f7f89142e33e77833254d9c9b90c3cef7303b`:
  https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b
- MLX-VLM, copyright © 2025 Prince Canuma, MIT license, implementation revision
  `3d87e88402f307efbf68e568971aa887ee7d9ed0`. Text encoder, multimodal projection,
  and audio Conformer port:
  https://github.com/Blaizzy/mlx-vlm/tree/3d87e88402f307efbf68e568971aa887ee7d9ed0/mlx_vlm/models/embedding_gemma2
  https://github.com/Blaizzy/mlx-vlm/tree/3d87e88402f307efbf68e568971aa887ee7d9ed0/mlx_vlm/models/gemma4
- Hugging Face Transformers, copyright © 2026 Google LLC and the HuggingFace Team,
  Apache License 2.0. EmbeddingGemma 2 model, Gemma4 image and audio preprocessing,
  and EmbeddingGemma 2 video sampling:
  https://github.com/huggingface/transformers/tree/main/src/transformers/models/embedding_gemma2
  https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4
- PyTorch bicubic interpolation coefficient quantization, copyright the PyTorch
  contributors, BSD license:
  https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/UpSampleKernel.cpp

The license texts are retained alongside this notice as
`EmbeddingGemma2-MLX-VLM-MIT.txt`, `EmbeddingGemma2-Transformers-Apache-2.0.txt`,
and `EmbeddingGemma2-PyTorch-BSD.txt`.
Model weights are installed separately and retain their own upstream license.

The Swift implementation adapts these algorithms, weight names, and architecture
to a separate embedding API, with native media decoding, strict complete-checkpoint
validation, and lazy modality components. It does not use causal generation.
