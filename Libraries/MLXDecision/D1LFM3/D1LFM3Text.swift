// Copyright © 2026 Faux Fox, Inc.
// Port of the LFM3 text model (Liquid d1-2B `modeling_lfm3.py`) as a stateless decision trunk.

import Foundation
import MLX
import MLXNN

// One forward over the whole row: no KV cache and no decode. Full-attention layers read d1's `set`
// mask, sliding layers a causal window. The trunk returns final-norm hidden states at the answer slots.

final class D1LFM3Attention: Module {
    let heads: Int
    let keyValueHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: D1LFM3Configuration.Text) {
        heads = config.attentionHeads
        keyValueHeads = config.keyValueHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        let d = config.hiddenSize
        _qProj.wrappedValue = Linear(d, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(d, keyValueHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(d, keyValueHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, d, bias: false)
        _gateProj.wrappedValue = Linear(d, heads, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
    }

    /// `x` [1, L, D]; `freqs` the layer type's RoPE wavelengths; `mask` [L, L] bool.
    func callAsFunction(_ x: MLXArray, freqs: MLXArray, mask: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(b, l, heads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, l, keyValueHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, keyValueHeads, headDim).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(
            q, dimensions: headDim, traditional: false, base: nil, scale: 1, offset: 0, freqs: freqs
        )
        k = MLXFast.RoPE(
            k, dimensions: headDim, traditional: false, base: nil, scale: 1, offset: 0, freqs: freqs
        )
        var y = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .array(mask))
        let gate = sigmoid(gateProj(x)).transposed(0, 2, 1)[.ellipsis, .newAxis]
        y = (y * gate).transposed(0, 2, 1, 3).reshaped(b, l, -1)
        return oProj(y)
    }
}

/// `out(C * conv(B * x))` with a causal depthwise convolution.
final class D1LFM3ShortConv: Module {
    let width: Int

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    /// Tap-major taps [k, D]: tap `k - 1` reads the current position.
    @ParameterInfo(key: "conv_weight") var convWeight: MLXArray

    init(_ config: D1LFM3Configuration.Text) {
        let d = config.hiddenSize
        width = config.convolutionCache
        _inProj.wrappedValue = Linear(d, 3 * d, bias: false)
        _outProj.wrappedValue = Linear(d, d, bias: false)
        _convWeight.wrappedValue = MLXArray.zeros([width, d])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outProj(D1LFM3Kernels.shortConvolution(inProj(x), weight: convWeight))
    }
}

final class D1LFM3MLP: Module, UnaryLayer {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(dimensions: Int, hidden: Int) {
        _w1.wrappedValue = Linear(dimensions, hidden, bias: false)
        _w2.wrappedValue = Linear(hidden, dimensions, bias: false)
        _w3.wrappedValue = Linear(dimensions, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

final class D1LFM3Layer: Module {
    enum Kind { case conv, full, sliding }
    let kind: Kind

    @ModuleInfo(key: "self_attn") var attention: D1LFM3Attention?
    @ModuleInfo var conv: D1LFM3ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: D1LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm

    init(_ config: D1LFM3Configuration.Text, type: String) throws {
        switch type {
        case "conv": kind = .conv
        case "full_attention": kind = .full
        case "sliding_attention": kind = .sliding
        default: throw DecisionError.invalidConfiguration("unknown LFM3 layer type \(type)")
        }
        if kind == .conv {
            _conv.wrappedValue = D1LFM3ShortConv(config)
        } else {
            _attention.wrappedValue = D1LFM3Attention(config)
        }
        let d = config.hiddenSize
        _feedForward.wrappedValue = D1LFM3MLP(dimensions: d, hidden: config.intermediateSize)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: d, eps: config.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: d, eps: config.normEps)
        _postOperatorNorm.wrappedValue = RMSNorm(dimensions: d, eps: config.normEps)
        _postFFNNorm.wrappedValue = RMSNorm(dimensions: d, eps: config.normEps)
    }

    func mix(_ h: MLXArray, context: D1LFM3Text.Context) -> MLXArray {
        let x = operatorNorm(h)
        let y: MLXArray
        switch kind {
        case .conv: y = conv!(x)
        case .full: y = attention!(x, freqs: context.fullFreqs, mask: context.fullMask)
        case .sliding: y = attention!(x, freqs: context.slidingFreqs, mask: context.slidingMask)
        }
        return h + postOperatorNorm(y)
    }

    func feed(_ h: MLXArray) -> MLXArray {
        h + postFFNNorm(feedForward(ffnNorm(h)))
    }
}

final class D1LFM3Text: Module {
    let configuration: D1LFM3Configuration.Text

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [D1LFM3Layer]
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    struct Context {
        let fullFreqs: MLXArray
        let slidingFreqs: MLXArray
        let fullMask: MLXArray
        let slidingMask: MLXArray
    }

    init(_ config: D1LFM3Configuration.Text) throws {
        configuration = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = try config.layerTypes.map { try D1LFM3Layer(config, type: $0) }
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }

    /// `Lfm3RotaryEmbedding`'s frequencies as `MLXFast.RoPE` wavelengths: the first `factor × dim / 2`
    /// follow the full head dimension's progression, and the rest never rotate.
    func wavelengths(_ type: String) -> MLXArray {
        let rope = configuration.rope[type]
        let theta = rope?.theta ?? 10000
        let half = configuration.headDim / 2
        let rotated = Int((rope?.partialRotaryFactor ?? 1) * Float(configuration.headDim)) / 2
        return MLXArray(
            (0 ..< half).map { i -> Float in
                i < rotated ? pow(theta, Float(2 * i) / Float(configuration.headDim)) : .infinity
            })
    }

    /// d1 `set` attention for full layers and a causal window for sliding layers, both [L, L] bool.
    /// `groups` [L] holds each position's question; the state (`< stateLength`) sees and is seen by all.
    /// Keys at or past `real` (right padding) are never seen.
    func masks(length: Int, real: Int, stateLength: Int, groups: MLXArray) -> (MLXArray, MLXArray) {
        let q = MLXArray(Int32(0) ..< Int32(length))[0..., .newAxis]
        let k = MLXArray(Int32(0) ..< Int32(length))[.newAxis]
        let state = Int32(stateLength)
        var full = (q .< state) .|| (k .< state) .|| (groups[0..., .newAxis] .== groups[.newAxis])
        if real < length {
            full = full .&& (k .< Int32(real))
        }
        let window = Int32(configuration.slidingWindow)
        let sliding = (k .<= q) .&& (k .> (q - window))
        return (full, sliding)
    }

    /// `h` [1, L, D] input embeddings; returns the final-norm hidden states at `slots` [K].
    func callAsFunction(
        _ input: MLXArray, real: Int, stateLength: Int, groups: MLXArray, slots: MLXArray
    ) -> MLXArray {
        let (full, sliding) = masks(
            length: input.dim(1), real: real, stateLength: stateLength, groups: groups)
        let context = Context(
            fullFreqs: wavelengths("full_attention"),
            slidingFreqs: wavelengths("sliding_attention"),
            fullMask: full, slidingMask: sliding)
        var h = input
        for (index, layer) in layers.enumerated() {
            h = layer.mix(h, context: context)
            // Only the answer slots are read after the last operator, so its feed-forward runs on them alone.
            if index == layers.count - 1 {
                h = h[0..., slots]
            }
            h = layer.feed(h)
        }
        return embeddingNorm(h[0])
    }
}
