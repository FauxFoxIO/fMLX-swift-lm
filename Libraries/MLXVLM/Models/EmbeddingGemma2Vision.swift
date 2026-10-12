// Copyright © 2026 Faux Fox, Inc.
// Portions © 2025 Prince Canuma (MIT), © 2026 Google LLC and Hugging Face (Apache-2.0).
// See LICENSES/EmbeddingGemma2-NOTICE.md for source attribution and license texts.
// Native Swift port of Google/Hugging Face Gemma4 and MLX-VLM vision processing.
// https://github.com/Blaizzy/mlx-vlm/tree/3d87e88402f307efbf68e568971aa887ee7d9ed0/mlx_vlm/models/gemma4

import Foundation
import MLX
import MLXNN

private final class EmbeddingGemma2PatchEmbedder: Module {
    @ModuleInfo(key: "input_proj") var projection: Linear
    @ParameterInfo(key: "position_embedding_table") var positions: MLXArray

    init(_ c: Gemma4VisionConfiguration) {
        _projection.wrappedValue = Linear(3 * c.patchSize * c.patchSize, c.hiddenSize, bias: false)
        _positions.wrappedValue = MLXArray.ones([2, c.positionEmbeddingSize, c.hiddenSize])
        super.init()
    }

    func callAsFunction(_ patches: MLXArray, coordinates: MLXArray, valid: MLXArray) -> MLXArray {
        let b = patches.dim(0)
        let t = patches.dim(1)
        let d = positions.dim(2)
        let safe = maximum(coordinates, 0).asType(.int32)
        let x = take(positions[0], safe[0..., 0..., 0].flattened(), axis: 0).reshaped(b, t, d)
        let y = take(positions[1], safe[0..., 0..., 1].flattened(), axis: 0).reshaped(b, t, d)
        let positionEmbedding = MLX.where(
            valid.expandedDimensions(axis: -1), x + y, MLXArray(0, dtype: x.dtype))
        let dtype = (projection as? QuantizedLinear)?.scales.dtype ?? projection.weight.dtype
        return projection((2 * (patches - 0.5)).asType(dtype)) + positionEmbedding
    }
}

final class EmbeddingGemma2VisionEncoder: Module {
    let config: Gemma4VisionConfiguration
    @ModuleInfo(key: "patch_embedder") private var patchEmbedder: EmbeddingGemma2PatchEmbedder
    @ModuleInfo var encoder: Gemma4VisionTransformerModel
    @ParameterInfo(key: "std_bias") var standardBias: MLXArray?
    @ParameterInfo(key: "std_scale") var standardScale: MLXArray?

    init(_ c: Gemma4VisionConfiguration) {
        config = c
        _patchEmbedder.wrappedValue = EmbeddingGemma2PatchEmbedder(c)
        _encoder.wrappedValue = Gemma4VisionTransformerModel(config: c)
        if c.standardize {
            _standardBias.wrappedValue = MLXArray.zeros([c.hiddenSize])
            _standardScale.wrappedValue = MLXArray.ones([c.hiddenSize])
        }
        super.init()
    }

    func callAsFunction(_ input: EmbeddingGemma2PreparedImage) throws -> MLXArray {
        let p = input.positions
        guard input.patches.ndim == 3, p.shape == [1, input.patches.dim(1), 2],
            input.patches.dim(2) == 3 * config.patchSize * config.patchSize,
            input.patches.dim(1) % (config.poolingKernelSize * config.poolingKernelSize) == 0
        else { throw EmbeddingGemma2Error.invalidInput("invalid vision patch shape") }
        let valid = logicalNot(all(p .== -1, axis: -1))
        let h = patchEmbedder(input.patches, coordinates: p, valid: valid)
        let both = logicalAnd(valid.expandedDimensions(axis: 1), valid.expandedDimensions(axis: 2))
        // Finite mask fill avoids all-masked padded query rows producing NaNs.
        let mask = MLX.where(both, MLXArray(0, dtype: h.dtype), MLXArray(-1e4, dtype: h.dtype))
            .expandedDimensions(axis: 1)
        let encoded = encoder(h, positions: p, mask: mask)
        let zeroed = MLX.where(
            valid.expandedDimensions(axis: -1), encoded, MLXArray(0, dtype: encoded.dtype))
        let kernel = config.poolingKernelSize
        let length = input.patches.dim(1) / (kernel * kernel)
        let safe = maximum(p, 0)
        let width = max(safe[0..., 0..., 0], axis: -1, keepDims: true) + 1
        let cells = floor(safe.asType(.float32) / Float(kernel)).asType(.int32)
        let index = cells[0..., 0..., 0] + (width / kernel) * cells[0..., 0..., 1]
        let weights =
            (index.expandedDimensions(axis: -1) .== MLXArray(0 ..< length)).asType(.float32)
            / Float(kernel * kernel)
        var pooled = einsum("bLl,bLd->bld", weights, zeroed).asType(encoded.dtype)
        pooled = pooled * pow(Float(config.hiddenSize), 0.5)
        guard input.tokenCount > 0, input.tokenCount <= length else {
            throw EmbeddingGemma2Error.invalidInput("invalid pooled vision count")
        }
        let occupied = any(weights .!= 0, axis: 1)
        let actual = occupied.sum().item(Int.self)
        guard actual == input.tokenCount else {
            throw EmbeddingGemma2Error.featureCount(
                modality: "vision", expected: input.tokenCount, actual: actual)
        }
        var result = pooled[0, ..<input.tokenCount, 0...]
        if let standardBias, let standardScale { result = (result - standardBias) * standardScale }
        return result
    }
}

final class EmbeddingGemma2MultimodalProjection: Module, UnaryLayer {
    @ModuleInfo(key: "embedding_projection") var projection: Linear
    @ModuleInfo(key: "embedding_pre_projection_norm") var norm: EmbeddingGemma2RMSNorm

    init(inputSize: Int, hiddenSize: Int, epsilon: Float) {
        _projection.wrappedValue = Linear(inputSize, hiddenSize, bias: false)
        _norm.wrappedValue = EmbeddingGemma2RMSNorm(inputSize, epsilon: epsilon, scale: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { projection(norm(x)) }
}
