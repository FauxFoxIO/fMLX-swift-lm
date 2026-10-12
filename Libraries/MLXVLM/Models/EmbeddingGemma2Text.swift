// Copyright © 2026 Faux Fox, Inc.
// Portions © 2025 Prince Canuma (MIT), © 2026 Google LLC and Hugging Face (Apache-2.0).
// See LICENSES/EmbeddingGemma2-NOTICE.md for source attribution and license texts.
// Native Swift port of Google/Hugging Face EmbeddingGemma 2 and MLX-VLM.
// https://github.com/Blaizzy/mlx-vlm/tree/3d87e88402f307efbf68e568971aa887ee7d9ed0/mlx_vlm/models/embedding_gemma2

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct EmbeddingGemma2TextConfiguration: Decodable, Sendable {
    struct Layer: Decodable, Sendable {
        let headDim: Int?
        let heads: Int?
        let kvHeads: Int?
        enum CodingKeys: String, CodingKey {
            case headDim = "head_dim"
            case heads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
        }
    }

    struct Rotary: Decodable, Sendable {
        let theta: Float
        let type: String
        enum CodingKeys: String, CodingKey {
            case theta = "rope_theta"
            case type = "rope_type"
        }
    }

    let hiddenSize: Int
    let intermediateSize: Int
    let hiddenLayers: Int
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let perLayerWidth: Int
    let embeddingDim: Int
    let vocabularySize: Int
    let epsilon: Float
    let slidingWindow: Int
    let padTokenId: Int
    let bosTokenId: Int
    let eosTokenId: Int
    let attentionBias: Bool
    let activation: String
    let layerTypes: [String]
    let perLayer: [String: Layer]
    let rotary: [String: Rotary]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case heads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case perLayerWidth = "hidden_size_per_layer_input"
        case embeddingDim = "embedding_dim"
        case vocabularySize = "vocab_size"
        case epsilon = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case padTokenId = "pad_token_id"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case attentionBias = "attention_bias"
        case activation = "hidden_activation"
        case layerTypes = "layer_types"
        case perLayer = "per_layer_config"
        case rotary = "rope_parameters"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        heads = try c.decode(Int.self, forKey: .heads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        perLayerWidth = try c.decode(Int.self, forKey: .perLayerWidth)
        embeddingDim = try c.decode(Int.self, forKey: .embeddingDim)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        epsilon = try c.decode(Float.self, forKey: .epsilon)
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 0
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 2
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 1
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        activation = try c.decodeIfPresent(String.self, forKey: .activation) ?? "gelu_pytorch_tanh"
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        perLayer = try c.decode([String: Layer].self, forKey: .perLayer)
        rotary = try c.decode([String: Rotary].self, forKey: .rotary)
    }

    func validate() throws {
        guard hiddenSize > 0, intermediateSize > 0, hiddenLayers > 0,
            heads > 0, kvHeads > 0, headDim > 0, perLayerWidth > 0,
            embeddingDim == 768, vocabularySize > 0, epsilon.isFinite, epsilon > 0,
            slidingWindow > 0, layerTypes.count == hiddenLayers,
            activation == "gelu_pytorch_tanh"
        else { throw EmbeddingGemma2Error.invalidConfiguration("unsupported text topology") }
        for i in 0 ..< hiddenLayers {
            let kind = layerTypes[i]
            let layer = perLayer[String(format: "%02d", i)]
            let h = layer?.heads ?? heads
            let k = layer?.kvHeads ?? kvHeads
            let d = layer?.headDim ?? headDim
            guard ["sliding_attention", "full_attention"].contains(kind),
                h > 0, k > 0, h % k == 0, d > 0, d % 2 == 0,
                let rope = rotary[kind], rope.type == "default", rope.theta.isFinite,
                rope.theta > 0
            else {
                throw EmbeddingGemma2Error.invalidConfiguration("unsupported attention layer \(i)")
            }
        }
        guard
            [padTokenId, bosTokenId, eosTokenId].allSatisfy({ (0 ..< vocabularySize).contains($0) })
        else {
            throw EmbeddingGemma2Error.invalidConfiguration("text token IDs are outside vocabulary")
        }
    }
}

/// Match the pinned MLX text and projection normalization, including BF16 rounding.
final class EmbeddingGemma2RMSNorm: Module, UnaryLayer {
    let epsilon: Float
    @ParameterInfo var weight: MLXArray?

    init(_ dimensions: Int, epsilon: Float, scale: Bool = true) {
        self.epsilon = epsilon
        self._weight.wrappedValue = scale ? MLXArray.ones([dimensions]) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight ?? MLXArray.mlxNone, eps: epsilon)
    }
}

private final class EmbeddingGemma2Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let base: Float
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var out: Linear
    @ModuleInfo(key: "q_norm") var qNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: EmbeddingGemma2RMSNorm

    init(_ c: EmbeddingGemma2TextConfiguration, layer: Int) {
        let override = c.perLayer[String(format: "%02d", layer)]
        heads = override?.heads ?? c.heads
        kvHeads = override?.kvHeads ?? c.kvHeads
        headDim = override?.headDim ?? c.headDim
        base = c.rotary[c.layerTypes[layer]]?.theta ?? 10_000
        _q.wrappedValue = Linear(c.hiddenSize, heads * headDim, bias: c.attentionBias)
        _k.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: c.attentionBias)
        _v.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: c.attentionBias)
        _out.wrappedValue = Linear(heads * headDim, c.hiddenSize, bias: c.attentionBias)
        _qNorm.wrappedValue = EmbeddingGemma2RMSNorm(headDim, epsilon: c.epsilon)
        _kNorm.wrappedValue = EmbeddingGemma2RMSNorm(headDim, epsilon: c.epsilon)
        _vNorm.wrappedValue = EmbeddingGemma2RMSNorm(headDim, epsilon: c.epsilon, scale: false)
        super.init()
    }

    private func rotate(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let exponents =
            MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32) / Float(headDim)
        let frequencies = 1 / MLX.pow(MLXArray(base), exponents)
        let angles = positions.asType(.float32).expandedDimensions(axis: -1) * frequencies
        let doubled = concatenated([angles, angles], axis: -1).expandedDimensions(axis: 1)
        let cosine = cos(doubled).asType(x.dtype)
        let sine = sin(doubled).asType(x.dtype)
        let half = headDim / 2
        let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        return x * cosine + rotated * sine
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, positions: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let queries = rotate(
            qNorm(q(x).reshaped(b, t, heads, headDim)).transposed(0, 2, 1, 3), positions: positions)
        let keys = rotate(
            kNorm(k(x).reshaped(b, t, kvHeads, headDim)).transposed(0, 2, 1, 3),
            positions: positions)
        let values = vNorm(v(x).reshaped(b, t, kvHeads, headDim)).transposed(0, 2, 1, 3)
        return out(
            MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: 1, mask: .array(mask)
            ).transposed(0, 2, 1, 3).reshaped(b, t, heads * headDim))
    }
}

private final class EmbeddingGemma2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ c: EmbeddingGemma2TextConfiguration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(geluApproximate(gate(x)) * up(x)) }
}

private final class EmbeddingGemma2PLE: Module, UnaryLayer {
    let count: Int
    let width: Int
    let scale: Float
    @ModuleInfo(key: "per_layer_model_projection") var projection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var norm: EmbeddingGemma2RMSNorm

    init(_ c: EmbeddingGemma2TextConfiguration) {
        count = c.hiddenLayers
        width = c.perLayerWidth
        scale = pow(Float(c.hiddenSize), -0.5)
        _projection.wrappedValue = Linear(c.hiddenSize, count * width, bias: false)
        _norm.wrappedValue = EmbeddingGemma2RMSNorm(width, epsilon: c.epsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm((projection(x) * scale).reshaped(x.dim(0), x.dim(1), count, width))
    }
}

private final class EmbeddingGemma2PLEBlock: Module {
    @ModuleInfo(key: "per_layer_input_gate") var gate: Linear
    @ModuleInfo(key: "per_layer_projection") var projection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var norm: EmbeddingGemma2RMSNorm

    init(_ c: EmbeddingGemma2TextConfiguration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.perLayerWidth, bias: false)
        _projection.wrappedValue = Linear(c.perLayerWidth, c.hiddenSize, bias: false)
        _norm.wrappedValue = EmbeddingGemma2RMSNorm(c.hiddenSize, epsilon: c.epsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, input: MLXArray) -> MLXArray {
        x + norm(projection(geluApproximate(gate(x)) * input))
    }
}

private final class EmbeddingGemma2EncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: EmbeddingGemma2Attention
    @ModuleInfo var mlp: EmbeddingGemma2MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var attentionNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFFN: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFFN: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "ple_block") var ple: EmbeddingGemma2PLEBlock
    @ParameterInfo(key: "layer_scalar") var scalar: MLXArray

    init(_ c: EmbeddingGemma2TextConfiguration, layer: Int) {
        _attention.wrappedValue = EmbeddingGemma2Attention(c, layer: layer)
        _mlp.wrappedValue = EmbeddingGemma2MLP(c)
        _inputNorm.wrappedValue = EmbeddingGemma2RMSNorm(c.hiddenSize, epsilon: c.epsilon)
        _attentionNorm.wrappedValue = EmbeddingGemma2RMSNorm(c.hiddenSize, epsilon: c.epsilon)
        _preFFN.wrappedValue = EmbeddingGemma2RMSNorm(c.hiddenSize, epsilon: c.epsilon)
        _postFFN.wrappedValue = EmbeddingGemma2RMSNorm(c.hiddenSize, epsilon: c.epsilon)
        _ple.wrappedValue = EmbeddingGemma2PLEBlock(c)
        _scalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, positions: MLXArray, perLayer: MLXArray)
        -> MLXArray
    {
        let h = x + attentionNorm(attention(inputNorm(x), mask: mask, positions: positions))
        let f = h + postFFN(mlp(preFFN(h)))
        return ple(f, input: perLayer) * scalar
    }
}

final class EmbeddingGemma2TextEncoder: Module {
    let config: EmbeddingGemma2TextConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo private var ple: EmbeddingGemma2PLE
    @ModuleInfo private var layers: [EmbeddingGemma2EncoderLayer]
    @ModuleInfo var norm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "embedding_projection") var projection: Linear

    init(_ c: EmbeddingGemma2TextConfiguration) {
        config = c
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _ple.wrappedValue = EmbeddingGemma2PLE(c)
        _layers.wrappedValue = (0 ..< c.hiddenLayers).map {
            EmbeddingGemma2EncoderLayer(c, layer: $0)
        }
        _norm.wrappedValue = EmbeddingGemma2RMSNorm(c.hiddenSize, epsilon: c.epsilon)
        _projection.wrappedValue = Linear(c.hiddenSize, c.embeddingDim, bias: false)
        super.init()
    }

    func callAsFunction(_ embeddings: MLXArray, validMask: MLXArray) throws -> MLXArray {
        let length = embeddings.dim(1)
        let positions = MLXArray(0 ..< length).expandedDimensions(axis: 0)
        let fullMask = validMask.asType(.bool).expandedDimensions(axes: [1, 2])
        let p = MLXArray(0 ..< length)
        let local =
            abs(p.expandedDimensions(axis: 1) - p.expandedDimensions(axis: 0))
            .<= config.slidingWindow
        let slidingMask = fullMask .&& local
        let perLayer = ple(embeddings)
        var h = embeddings
        for (i, layer) in layers.enumerated() {
            try Task.checkCancellation()
            h = layer(
                h, mask: config.layerTypes[i] == "sliding_attention" ? slidingMask : fullMask,
                positions: positions, perLayer: perLayer[0..., 0..., i, 0...])
            // Bound the lazy graph; the bidirectional encoder cannot use a causal KV cache.
            try checkedEval(h)
        }
        return projection(norm(h))
    }
}
