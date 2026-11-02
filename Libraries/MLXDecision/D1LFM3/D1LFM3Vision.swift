// Copyright © 2026 Faux Fox, Inc.
// Port of the LFM3-VL vision tower and projector (Liquid d1-2B `modeling_lfm3_vl.py`).

import Foundation
import MLX
import MLXNN

// One frame per image. Patches arrive in the processor's merge-block order: each run of
// `merge × merge` rows is one block, so the projector reads four consecutive rows as one token.

final class D1LFM3VisionAttention: Module {
    let heads: Int
    let keyValueHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_layernorm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kNorm: RMSNorm

    init(_ config: D1LFM3Configuration.Vision) {
        heads = config.attentionHeads
        keyValueHeads = config.keyValueHeads
        headDim = config.hiddenSize / config.attentionHeads
        scale = pow(Float(headDim), -0.5)
        let d = config.hiddenSize
        _qProj.wrappedValue = Linear(d, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(d, keyValueHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(d, keyValueHeads * headDim, bias: false)
        _outProj.wrappedValue = Linear(heads * headDim, d, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
    }

    /// `x` [N, D]; `cos`, `sin` [N, headDim] float32 axial tables.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let n = x.dim(0)
        func rotate(_ t: MLXArray) -> MLXArray {
            let t32 = t.asType(.float32)
            let half = headDim / 2
            let rotated = concatenated(
                [-t32[.ellipsis, half...], t32[.ellipsis, ..<half]], axis: -1)
            return
                (t32 * cos.expandedDimensions(axis: 1) + rotated * sin.expandedDimensions(axis: 1))
                .asType(t.dtype)
        }
        let q = rotate(qNorm(qProj(x).reshaped(n, heads, headDim)))
        let k = rotate(kNorm(kProj(x).reshaped(n, keyValueHeads, headDim)))
        let v = vProj(x).reshaped(n, keyValueHeads, headDim)
        let y = MLXFast.scaledDotProductAttention(
            queries: q.transposed(1, 0, 2)[.newAxis], keys: k.transposed(1, 0, 2)[.newAxis],
            values: v.transposed(1, 0, 2)[.newAxis], scale: scale, mask: .none)
        return outProj(y[0].transposed(1, 0, 2).reshaped(n, -1))
    }
}

/// `out(C * conv(B * V))` with a depthwise 3×3 convolution over the patch grid.
final class D1LFM3VisionConv: Module {
    let merge: Int

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "spatial_conv") var spatialConv: Conv2d

    init(_ config: D1LFM3Configuration.Vision) {
        merge = config.mergeSize
        let d = config.hiddenSize
        _inProj.wrappedValue = Linear(d, 3 * d, bias: false)
        _outProj.wrappedValue = Linear(d, d, bias: false)
        _spatialConv.wrappedValue = Conv2d(
            inputChannels: d, outputChannels: d, kernelSize: .init(config.convolutionKernel),
            padding: .init(config.convolutionKernel / 2), groups: d, bias: false)
    }

    func callAsFunction(_ x: MLXArray, rows: Int, columns: Int) -> MLXArray {
        let n = x.dim(0)
        let d = x.dim(-1)
        let parts = split(inProj(x), parts: 3, axis: -1)
        let gated = parts[0] * parts[2]
        let m = merge
        let frame = gated.reshaped(rows / m, columns / m, m, m, d).transposed(0, 2, 1, 3, 4)
            .reshaped(1, rows, columns, d)
        let convolved = spatialConv(frame).reshaped(rows / m, m, columns / m, m, d)
            .transposed(0, 2, 1, 3, 4).reshaped(n, d)
        return outProj(parts[1] * convolved)
    }
}

final class D1LFM3VisionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: D1LFM3VisionAttention?
    @ModuleInfo var conv: D1LFM3VisionConv?
    @ModuleInfo(key: "feed_forward") var feedForward: D1LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ config: D1LFM3Configuration.Vision, type: String) throws {
        switch type {
        case "conv": _conv.wrappedValue = D1LFM3VisionConv(config)
        case "full_attention": _attention.wrappedValue = D1LFM3VisionAttention(config)
        default: throw DecisionError.invalidConfiguration("unknown LFM3-VL vision layer \(type)")
        }
        let d = config.hiddenSize
        _feedForward.wrappedValue = D1LFM3MLP(dimensions: d, hidden: config.intermediateSize)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: d, eps: config.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: d, eps: config.normEps)
    }
}

final class D1LFM3VisionTower: Module {
    let configuration: D1LFM3Configuration.Vision

    @ModuleInfo var patchifier: Linear
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
    @ModuleInfo var layers: [D1LFM3VisionLayer]

    init(_ config: D1LFM3Configuration.Vision) throws {
        configuration = config
        _patchifier.wrappedValue = Linear(config.patchValues, config.hiddenSize, bias: false)
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _layers.wrappedValue = try config.layerTypes.map { try D1LFM3VisionLayer(config, type: $0) }
    }

    /// The axial RoPE tables of one frame, patches in merge-block order: rows over the first half of
    /// the rotated pairs, columns over the second.
    func rope(rows: Int, columns: Int) -> (MLXArray, MLXArray) {
        let m = configuration.mergeSize
        let headDim = configuration.hiddenSize / configuration.attentionHeads
        let spatial = headDim / 2
        let inverse = stride(from: 0, to: spatial, by: 2).map {
            1 / pow(configuration.rope.theta, Float($0) / Float(spatial))
        }
        var table = [Float]()
        table.reserveCapacity(rows * columns * headDim)
        for blockRow in 0 ..< rows / m {
            for blockColumn in 0 ..< columns / m {
                for i in 0 ..< m {
                    for j in 0 ..< m {
                        let h = Float(blockRow * m + i)
                        let w = Float(blockColumn * m + j)
                        let freqs = inverse.map { h * $0 } + inverse.map { w * $0 }
                        table += freqs + freqs
                    }
                }
            }
        }
        let emb = MLXArray(table, [rows * columns, headDim])
        return (MLX.cos(emb), MLX.sin(emb))
    }

    func callAsFunction(_ pixels: MLXArray, rows: Int, columns: Int) -> MLXArray {
        let (cos, sin) = rope(rows: rows, columns: columns)
        var h = embeddingNorm(patchifier(pixels))
        for layer in layers {
            let x = layer.operatorNorm(h)
            if let attention = layer.attention {
                h = h + attention(x, cos: cos, sin: sin)
            } else if let conv = layer.conv {
                h = h + conv(x, rows: rows, columns: columns)
            }
            h = h + layer.feedForward(layer.ffnNorm(h))
        }
        return h
    }
}

final class D1LFM3Projector: Module {
    let merge: Int

    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: D1LFM3Configuration, vision: D1LFM3Configuration.Vision) {
        merge = vision.mergeSize
        let width = vision.hiddenSize * merge * merge
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: vision.hiddenSize, eps: config.projectorLayerNormEps)
        _linear1.wrappedValue = Linear(width, config.projectorHiddenSize)
        _linear2.wrappedValue = Linear(config.projectorHiddenSize, config.text.hiddenSize)
    }

    func callAsFunction(_ h: MLXArray) -> MLXArray {
        let normed = MLXFast.layerNorm(
            h.asType(.float32), weight: layerNorm.weight?.asType(.float32),
            bias: layerNorm.bias?.asType(.float32), eps: layerNorm.eps
        ).asType(h.dtype)
        let merged = normed.reshaped(-1, h.dim(-1) * merge * merge)
        return linear2(gelu(linear1(merged)))
    }
}
