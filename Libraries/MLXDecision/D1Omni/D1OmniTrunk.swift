// Copyright © 2026 Faux Fox, Inc.
// Port of the d1-omni reference `encoder.py` (LiquidAI/d1-omni-600M) and MLX-VLM `d1_omni/encoder.py`.

import Foundation
import MLX
import MLXNN

// The trunk is LFM2.5-Encoder: LFM2 blocks with centred 3-tap convolutions and no causal mask.
// A media prefix sits in front of the text. Prefix queries never see text keys, and the
// convolution never lets the last prefix position read the first text position.

final class D1OmniAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_layernorm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kNorm: RMSNorm

    init(_ config: D1OmniConfiguration.Text) {
        heads = config.attentionHeads
        headDim = config.hiddenSize / config.attentionHeads
        scale = pow(Float(headDim), -0.5)
        let d = config.hiddenSize
        _qProj.wrappedValue = Linear(d, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(d, config.keyValueHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(d, config.keyValueHeads * headDim, bias: false)
        _outProj.wrappedValue = Linear(heads * headDim, d, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(b, l, -1, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, l, -1, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(b, l, -1, headDim).transposed(0, 2, 1, 3)
        q = q * cos + rotateHalf(q) * sin
        k = k * cos + rotateHalf(k) * sin
        // Full heads, as the reference repeats them; a fused GQA kernel cannot take a [B, 1, L, L] mask.
        let groups = heads / k.dim(1)
        k = repeated(k, count: groups, axis: 1)
        let v = repeated(values, count: groups, axis: 1)
        let y = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .array(mask))
        return outProj(y.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

/// `out(C * conv(B * x))` with a centred 3-tap depthwise convolution.
final class D1OmniShortConv: Module {
    @ModuleInfo var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: D1OmniConfiguration.Text) {
        let d = config.hiddenSize
        _conv.wrappedValue = Conv1d(
            inputChannels: d, outputChannels: d, kernelSize: config.convolutionCache,
            groups: d, bias: false)
        _inProj.wrappedValue = Linear(d, 3 * d, bias: false)
        _outProj.wrappedValue = Linear(d, d, bias: false)
    }

    func callAsFunction(_ x: MLXArray, pad: MLXArray, keepRight: MLXArray) -> MLXArray {
        let parts = split(inProj(x * pad[.ellipsis, .newAxis]), parts: 3, axis: -1)
        let (b, c, u) = (parts[0], parts[1], parts[2])
        let bx = b * u
        let length = bx.dim(1)
        let padded = MLX.padded(bx, widths: [0, [1, 1], 0])
        let w = conv.weight[0..., 0..., 0]
        var y = padded[0..., 0 ..< length] * w[0..., 0]
        y = y + padded[0..., 1 ..< (length + 1)] * w[0..., 1]
        y = y + padded[0..., 2 ..< (length + 2)] * keepRight[.ellipsis, .newAxis] * w[0..., 2]
        return outProj(c * y)
    }
}

final class D1OmniMLP: Module, UnaryLayer {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(_ config: D1OmniConfiguration.Text) {
        let hidden = config.feedForwardSize
        _w1.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
        _w2.wrappedValue = Linear(hidden, config.hiddenSize, bias: false)
        _w3.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

final class D1OmniLayer: Module {
    let isAttention: Bool
    @ModuleInfo(key: "self_attn") var attention: D1OmniAttention?
    @ModuleInfo var conv: D1OmniShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: D1OmniMLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ config: D1OmniConfiguration.Text, kind: String) {
        isAttention = kind == "full_attention"
        if isAttention {
            _attention.wrappedValue = D1OmniAttention(config)
        } else {
            _conv.wrappedValue = D1OmniShortConv(config)
        }
        _feedForward.wrappedValue = D1OmniMLP(config)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }
}

final class D1OmniTrunk: Module {
    let ropeTheta: Float
    let headDim: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [D1OmniLayer]
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ config: D1OmniConfiguration.Text) {
        ropeTheta = config.ropeTheta
        headDim = config.hiddenSize / config.attentionHeads
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = config.layerTypes.map { D1OmniLayer(config, kind: $0) }
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }

    func rope(length: Int, dtype: DType) -> (MLXArray, MLXArray) {
        let exponent =
            MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) }) / Float(headDim)
        let inverse = 1 / pow(MLXArray(ropeTheta), exponent)
        let freqs = MLXArray(0 ..< length).asType(.float32)[0..., .newAxis] * inverse[.newAxis]
        let emb = concatenated([freqs, freqs], axis: -1)
        return (MLX.cos(emb).asType(dtype), MLX.sin(emb).asType(dtype))
    }

    /// `h` [B, L, D] embeddings, `pad` [B, L] true on real positions, `prefix` media lengths.
    func callAsFunction(_ input: MLXArray, pad: MLXArray, prefix: [Int]) -> MLXArray {
        var h = input
        let length = h.dim(1)
        let t = MLXArray(Int32(0) ..< Int32(length))[.newAxis]
        let prefixes = MLXArray(prefix.map(Int32.init))[0..., .newAxis]
        var mask = pad[0..., .newAxis, .newAxis, 0...]
        if prefix.contains(where: { $0 > 0 }) {
            let blocked = logicalAnd(
                (t .< prefixes).expandedDimensions(axis: 2),
                (t .>= prefixes).expandedDimensions(axis: 1))
            mask = logicalAnd(mask, logicalNot(blocked).expandedDimensions(axis: 1))
        }
        let keepRight = (t .!= (prefixes - 1)).asType(h.dtype)
        let padf = pad.asType(h.dtype)
        let (cos, sin) = rope(length: length, dtype: h.dtype)
        for layer in layers {
            var x = layer.operatorNorm(h)
            if let attention = layer.attention {
                x = attention(x, cos: cos, sin: sin, mask: mask)
            } else if let conv = layer.conv {
                x = conv(x, pad: padf, keepRight: keepRight)
            }
            h = h + x
            h = h + layer.feedForward(layer.ffnNorm(h))
        }
        return embeddingNorm(h)
    }
}

/// A pre-norm `nn.TransformerEncoderLayer` (ReLU feed-forward, one head per 64 channels).
final class D1OmniHeadLayer: Module {
    let heads: Int

    @ModuleInfo var attention: D1OmniHeadAttention
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(width: Int) {
        heads = width / 64
        _attention.wrappedValue = D1OmniHeadAttention(width: width, heads: width / 64)
        _norm1.wrappedValue = LayerNorm(dimensions: width)
        _norm2.wrappedValue = LayerNorm(dimensions: width)
        _linear1.wrappedValue = Linear(width, 4 * width)
        _linear2.wrappedValue = Linear(4 * width, width)
    }

    func callAsFunction(_ h: MLXArray, mask: MLXArray) -> MLXArray {
        let x = h + attention(norm1(h), mask: mask)
        return x + linear2(relu(linear1(norm2(x))))
    }
}

final class D1OmniHeadAttention: Module {
    let heads: Int
    let scale: Float

    @ModuleInfo(key: "query_proj") var queryProj: Linear
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(width: Int, heads: Int) {
        self.heads = heads
        scale = pow(Float(width / heads), -0.5)
        _queryProj.wrappedValue = Linear(width, width)
        _keyProj.wrappedValue = Linear(width, width)
        _valueProj.wrappedValue = Linear(width, width)
        _outProj.wrappedValue = Linear(width, width)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        func split(_ y: MLXArray) -> MLXArray { y.reshaped(b, l, heads, -1).transposed(0, 2, 1, 3) }
        let y = MLXFast.scaledDotProductAttention(
            queries: split(queryProj(x)), keys: split(keyProj(x)), values: split(valueProj(x)),
            scale: scale, mask: .array(mask))
        return outProj(y.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class D1OmniHeadStack: Module {
    @ModuleInfo var layers: [D1OmniHeadLayer]

    init(width: Int, count: Int) {
        _layers.wrappedValue = (0 ..< count).map { _ in D1OmniHeadLayer(width: width) }
    }
}

/// Adds a question-type embedding, runs the head layers over the text positions, and scores
/// the hidden state at each option marker with one shared MLP.
final class D1OmniDecisionHead: Module {
    @ModuleInfo(key: "type_emb") var typeEmbedding: Embedding
    @ModuleInfo var head: D1OmniHeadStack
    @ModuleInfo var scorer: Sequential

    init(width: Int, layers: Int) {
        _typeEmbedding.wrappedValue = Embedding(embeddingCount: 3, dimensions: width)
        _head.wrappedValue = D1OmniHeadStack(width: width, count: layers)
        _scorer.wrappedValue = Sequential(layers: [
            LayerNorm(dimensions: width), Linear(width, width), GELU(), Linear(width, 1),
        ])
    }

    /// `text` [B, T, D], `pad` [B, T] bool, `markers` [B, K] int32, `markerMask` [B, K] bool.
    func callAsFunction(
        _ text: MLXArray, pad: MLXArray, markers: MLXArray, markerMask: MLXArray, types: MLXArray
    ) -> MLXArray {
        var h = text + typeEmbedding(types)[0..., .newAxis, 0...]
        let mask = pad[0..., .newAxis, .newAxis, 0...]
        for layer in head.layers {
            h = layer(h, mask: mask)
        }
        let index = broadcast(
            markers[.ellipsis, .newAxis], to: [markers.dim(0), markers.dim(1), h.dim(-1)])
        let gathered = takeAlong(h, index, axis: 1)
        let logits = scorer(gathered).squeezed(axis: -1).asType(.float32)
        return MLX.where(markerMask, logits, MLXArray(Float(-1e4)))
    }
}
