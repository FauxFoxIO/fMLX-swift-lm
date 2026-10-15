// Copyright © 2026 Faux Fox, Inc.
// Port of the d1-omni reference `vision.py` (LiquidAI/d1-omni-600M), the Hugging Face
// Transformers SigLIP2 vision tower, and MLX-VLM `lfm2_vl/vision.py`.

import Foundation
import MLX
import MLXNN

final class D1OmniVisionAttention: Module {
    let heads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dimensions: Int, heads: Int) {
        self.heads = heads
        scale = pow(Float(dimensions / heads), -0.5)
        _qProj.wrappedValue = Linear(dimensions, dimensions)
        _kProj.wrappedValue = Linear(dimensions, dimensions)
        _vProj.wrappedValue = Linear(dimensions, dimensions)
        _outProj.wrappedValue = Linear(dimensions, dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        func split(_ y: MLXArray) -> MLXArray { y.reshaped(b, l, heads, -1).transposed(0, 2, 1, 3) }
        let y = MLXFast.scaledDotProductAttention(
            queries: split(qProj(x)), keys: split(kProj(x)), values: split(vProj(x)),
            scale: scale, mask: .none)
        return outProj(y.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class D1OmniVisionMLP: Module, UnaryLayer {
    let tanhApproximation: Bool
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(_ config: D1OmniConfiguration.Vision) {
        tanhApproximation = config.hiddenAct.contains("tanh")
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = fc1(x)
        return fc2(tanhApproximation ? geluApproximate(h) : gelu(h))
    }
}

final class D1OmniVisionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: D1OmniVisionAttention
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo var mlp: D1OmniVisionMLP
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm

    init(_ config: D1OmniConfiguration.Vision) {
        _attention.wrappedValue = D1OmniVisionAttention(
            dimensions: config.hiddenSize, heads: config.attentionHeads)
        _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _mlp.wrappedValue = D1OmniVisionMLP(config)
        _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(norm1(x))
        return h + mlp(norm2(h))
    }
}

final class D1OmniVisionEncoder: Module {
    @ModuleInfo var layers: [D1OmniVisionLayer]

    init(_ config: D1OmniConfiguration.Vision) {
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in D1OmniVisionLayer(config) }
    }
}

final class D1OmniVisionEmbeddings: Module {
    let gridSize: Int

    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(_ config: D1OmniConfiguration.Vision) {
        gridSize = Int(Double(config.patches).squareRoot())
        _patchEmbedding.wrappedValue = Linear(
            config.channels * config.patchSize * config.patchSize, config.hiddenSize)
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.patches, dimensions: config.hiddenSize)
    }

    /// `patches` [N, H*W, C*P*P] for N crops that share one patch grid.
    func callAsFunction(_ patches: MLXArray, height: Int, width: Int) -> MLXArray {
        let dtype = patchEmbedding.weight.dtype
        let embedded = patchEmbedding(patches.asType(dtype))
        let table = positionEmbedding(MLXArray(Int32(0) ..< Int32(gridSize * gridSize)))
        let grid = table.asType(.float32).reshaped(gridSize, gridSize, -1)
        let ry = MLXArray(Self.bilinearMatrix(input: gridSize, output: height), [height, gridSize])
        let rx = MLXArray(Self.bilinearMatrix(input: gridSize, output: width), [width, gridSize])
        let channels = grid.dim(-1)
        var resized = matmul(ry, grid.reshaped(gridSize, gridSize * channels))
        resized = resized.reshaped(height, gridSize, channels).transposed(0, 2, 1)
        resized = matmul(resized, rx.transposed()).transposed(0, 2, 1)
        return embedded + resized.reshaped(1, height * width, channels).asType(dtype)
    }

    /// Row-major [output, input] weights of `F.interpolate(mode="bilinear",
    /// align_corners=False, antialias=True)` along one axis.
    static func bilinearMatrix(input: Int, output: Int) -> [Float] {
        var matrix = [Float](repeating: 0, count: input * output)
        let scale = Double(input) / Double(output)
        for o in 0 ..< output {
            var src = (Double(o) + 0.5) * scale - 0.5
            if scale > 1 {
                let low = max((src - scale + 1).rounded(.down), 0)
                let high = min((src + scale).rounded(.down), Double(input - 1))
                var weights = [(Int, Double)]()
                var position = low
                while position <= high {
                    weights.append((Int(position), max(0, 1 - abs(position - src) / scale)))
                    position += 1
                }
                let total = weights.reduce(0) { $0 + $1.1 }
                for (index, weight) in weights {
                    matrix[o * input + index] += Float(weight / total)
                }
            } else {
                src = max(src, 0)
                let base = src.rounded(.down)
                let fraction = src - base
                let low = min(Int(base), input - 1)
                let high = min(Int(base) + 1, input - 1)
                matrix[o * input + low] += Float(1 - fraction)
                matrix[o * input + high] += Float(fraction)
            }
        }
        return matrix
    }
}

final class D1OmniVisionTower: Module {
    @ModuleInfo var embeddings: D1OmniVisionEmbeddings
    @ModuleInfo var encoder: D1OmniVisionEncoder
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

    init(_ config: D1OmniConfiguration.Vision) {
        _embeddings.wrappedValue = D1OmniVisionEmbeddings(config)
        _encoder.wrappedValue = D1OmniVisionEncoder(config)
        _postLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ patches: MLXArray, height: Int, width: Int) -> MLXArray {
        var h = embeddings(patches, height: height, width: width)
        for layer in encoder.layers {
            h = layer(h)
        }
        return postLayerNorm(h)
    }
}

/// LFM2-VL's 2x2 pixel unshuffle and two-layer GELU MLP into the trunk's width.
final class D1OmniProjector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(visionSize: Int, hidden: Int, output: Int) {
        _linear1.wrappedValue = Linear(visionSize * 4, hidden)
        _linear2.wrappedValue = Linear(hidden, output)
    }

    /// `x` [N, H, W, C] with even H and W.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (n, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        var y = x.reshaped(n, h, w / 2, c * 2).transposed(0, 2, 1, 3)
        y = y.reshaped(n, w / 2, h / 2, c * 4).transposed(0, 2, 1, 3)
        y = linear2(gelu(linear1(y)))
        return y.reshaped(n, -1, y.dim(-1))
    }
}

final class D1OmniVision: Module {
    @ModuleInfo var tower: D1OmniVisionTower
    @ModuleInfo var projector: D1OmniProjector

    init(_ config: D1OmniConfiguration) {
        _tower.wrappedValue = D1OmniVisionTower(config.vision)
        _projector.wrappedValue = D1OmniProjector(
            visionSize: config.vision.hiddenSize, hidden: config.projectorHiddenSize,
            output: config.text.hiddenSize)
    }

    /// Prefix embeddings [P, D]: every image's tiles then thumbnail, images in order.
    func callAsFunction(_ images: [D1OmniPreparedImage]) -> MLXArray {
        var parts = [MLXArray]()
        for image in images {
            var index = 0
            while index < image.crops.count {
                // Tiles share one grid, so they run as one batch.
                let crop = image.crops[index]
                var end = index + 1
                while end < image.crops.count, image.crops[end].height == crop.height,
                    image.crops[end].width == crop.width
                {
                    end += 1
                }
                let batch = stacked(image.crops[index ..< end].map(\.patches))
                let hidden = tower(batch, height: crop.height, width: crop.width)
                let grid = hidden.reshaped(end - index, crop.height, crop.width, -1)
                let projected = projector(grid)
                parts.append(projected.reshaped(-1, projected.dim(-1)))
                index = end
            }
        }
        return concatenated(parts, axis: 0)
    }
}
