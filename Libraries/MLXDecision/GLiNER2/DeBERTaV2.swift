// Copyright © 2026 Faux Fox, Inc.
// Port of Hugging Face Transformers `deberta_v2` (Apache-2.0) as MLX-VLM `gliner2_5/deberta.py`
// writes it: disentangled attention with shared keys and log-bucketed relative positions.

import Foundation
import MLX
import MLXNN

public struct DeBERTaV2Configuration: Codable, Sendable {
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let intermediateSize: Int
    public let hiddenAct: String
    public let maxPositionEmbeddings: Int
    public let maxRelativePositions: Int
    public let positionBuckets: Int
    public let relativeAttention: Bool
    public let positionAttentionTypes: [String]
    public let shareAttentionKey: Bool
    public let positionBiasedInput: Bool
    public let normalizeRelativeEmbeddings: String?
    public let typeVocabularySize: Int
    public let layerNormEps: Float
    public let embeddingSize: Int?
    public let convolutionKernelSize: Int?

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case maxRelativePositions = "max_relative_positions"
        case positionBuckets = "position_buckets"
        case relativeAttention = "relative_attention"
        case positionAttentionTypes = "pos_att_type"
        case shareAttentionKey = "share_att_key"
        case positionBiasedInput = "position_biased_input"
        case normalizeRelativeEmbeddings = "norm_rel_ebd"
        case typeVocabularySize = "type_vocab_size"
        case layerNormEps = "layer_norm_eps"
        case embeddingSize = "embedding_size"
        case convolutionKernelSize = "conv_kernel_size"
    }

    /// Transformers replaces a non-positive maximum with `max_position_embeddings`.
    var relativeSpan: Int {
        maxRelativePositions < 1 ? maxPositionEmbeddings : maxRelativePositions
    }
    /// Rows of the relative-embedding table attention reads.
    var positionEmbeddingSize: Int { positionBuckets > 0 ? positionBuckets : relativeSpan }

    func validate() throws {
        guard relativeAttention, shareAttentionKey, !positionBiasedInput, typeVocabularySize == 0,
            Set(positionAttentionTypes).isSubset(of: ["c2p", "p2c"]),
            (normalizeRelativeEmbeddings ?? "none").contains("layer_norm"),
            embeddingSize == nil || embeddingSize == hiddenSize,
            (convolutionKernelSize ?? 0) == 0, hiddenAct == "gelu",
            hiddenSize % attentionHeads == 0
        else {
            throw DecisionError.invalidConfiguration(
                "only DeBERTa-v3 encoders (relative attention, shared keys, no conv layer) are supported"
            )
        }
    }

    /// `make_log_bucket_position` over `build_relative_position`, as Int32 [L, L].
    func relativePositions(length: Int) -> [Int32] {
        var result = [Int32](repeating: 0, count: length * length)
        let bucket = positionBuckets
        let maximum = relativeSpan
        for q in 0 ..< length {
            for k in 0 ..< length {
                let relative = q - k
                guard bucket > 0, maximum > 0 else {
                    result[q * length + k] = Int32(relative)
                    continue
                }
                let middle = bucket / 2
                let absolute = relative < middle && relative > -middle ? middle - 1 : abs(relative)
                if absolute <= middle {
                    result[q * length + k] = Int32(relative)
                } else {
                    // Transformers computes this in float32.
                    let ratio = Float(absolute) / Float(middle)
                    let scale = log(Float(Double(maximum - 1) / Double(middle)))
                    let logPosition =
                        (log(ratio) / scale * Float(middle - 1)).rounded(.up) + Float(middle)
                    result[q * length + k] = Int32(logPosition) * Int32(relative.signum())
                }
            }
        }
        return result
    }
}

final class DeBERTaV2Embeddings: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(_ config: DeBERTaV2Configuration) {
        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ ids: MLXArray, mask: MLXArray) -> MLXArray {
        let states = layerNorm(wordEmbeddings(ids))
        return states * mask[.ellipsis, .newAxis].asType(states.dtype)
    }
}

final class DeBERTaV2SelfAttention: Module {
    let heads: Int
    let headDim: Int
    let content2Position: Bool
    let position2Content: Bool

    @ModuleInfo(key: "query_proj") var queryProj: Linear
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear

    init(_ config: DeBERTaV2Configuration) {
        heads = config.attentionHeads
        headDim = config.hiddenSize / config.attentionHeads
        content2Position = config.positionAttentionTypes.contains("c2p")
        position2Content = config.positionAttentionTypes.contains("p2c")
        _queryProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _keyProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _valueProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    }

    private func split(_ x: MLXArray) -> MLXArray {
        x.reshaped(x.dim(0), x.dim(1), heads, headDim).transposed(0, 2, 1, 3)
    }

    /// `relative` [1, S, D] normalized relative embeddings; `c2p`/`p2c` gather indices [1, 1, L, L].
    func callAsFunction(
        _ states: MLXArray, keep: MLXArray, relative: MLXArray, c2p: MLXArray?, p2c: MLXArray?
    ) -> MLXArray {
        let (b, l) = (states.dim(0), states.dim(1))
        let query = split(queryProj(states))
        let key = split(keyProj(states))
        let value = split(valueProj(states))
        let factor = 1 + (content2Position ? 1 : 0) + (position2Content ? 1 : 0)
        let scale = Float(Double(headDim * factor).squareRoot())
        var scores = matmul(query, key.transposed(0, 1, 3, 2) / scale)
        let positionQuery = split(queryProj(relative))
        let positionKey = split(keyProj(relative))
        let gatherShape = [b, heads, l, l]
        if let c2p {
            let c2pScores = matmul(query, positionKey.transposed(0, 1, 3, 2))
            scores =
                scores + takeAlong(c2pScores, broadcast(c2p, to: gatherShape), axis: -1) / scale
        }
        if let p2c {
            let p2cScores = matmul(key, positionQuery.transposed(0, 1, 3, 2))
            scores =
                scores
                + takeAlong(p2cScores, broadcast(p2c, to: gatherShape), axis: -1)
                .transposed(0, 1, 3, 2) / scale
        }
        scores = MLX.where(keep, scores, MLXArray(Float(-1e4), dtype: scores.dtype))
        let probabilities = softmax(scores.asType(.float32), axis: -1).asType(value.dtype)
        return matmul(probabilities, value).transposed(0, 2, 1, 3).reshaped(b, l, -1)
    }
}

final class DeBERTaV2Output: Module {
    @ModuleInfo var dense: Linear
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(input: Int, output: Int, eps: Float) {
        _dense.wrappedValue = Linear(input, output)
        _layerNorm.wrappedValue = LayerNorm(dimensions: output, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, residual: MLXArray) -> MLXArray {
        layerNorm(dense(x) + residual)
    }
}

final class DeBERTaV2Attention: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: DeBERTaV2SelfAttention
    @ModuleInfo var output: DeBERTaV2Output

    init(_ config: DeBERTaV2Configuration) {
        _selfAttention.wrappedValue = DeBERTaV2SelfAttention(config)
        _output.wrappedValue = DeBERTaV2Output(
            input: config.hiddenSize, output: config.hiddenSize, eps: config.layerNormEps)
    }
}

final class DeBERTaV2Intermediate: Module {
    @ModuleInfo var dense: Linear

    init(_ config: DeBERTaV2Configuration) {
        _dense.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
    }
}

final class DeBERTaV2Layer: Module {
    @ModuleInfo var attention: DeBERTaV2Attention
    @ModuleInfo var intermediate: DeBERTaV2Intermediate
    @ModuleInfo var output: DeBERTaV2Output

    init(_ config: DeBERTaV2Configuration) {
        _attention.wrappedValue = DeBERTaV2Attention(config)
        _intermediate.wrappedValue = DeBERTaV2Intermediate(config)
        _output.wrappedValue = DeBERTaV2Output(
            input: config.intermediateSize, output: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(
        _ states: MLXArray, keep: MLXArray, relative: MLXArray, c2p: MLXArray?, p2c: MLXArray?
    ) -> MLXArray {
        let attended = attention.output(
            attention.selfAttention(states, keep: keep, relative: relative, c2p: c2p, p2c: p2c),
            residual: states)
        return output(gelu(intermediate.dense(attended)), residual: attended)
    }
}

final class DeBERTaV2Encoder: Module {
    let configuration: DeBERTaV2Configuration

    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "rel_embeddings") var relativeEmbeddings: Embedding
    @ModuleInfo var layers: [DeBERTaV2Layer]

    init(_ config: DeBERTaV2Configuration) {
        configuration = config
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        let tableSize =
            config.positionBuckets > 0 ? config.positionBuckets * 2 : config.relativeSpan * 2
        _relativeEmbeddings.wrappedValue = Embedding(
            embeddingCount: tableSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in DeBERTaV2Layer(config) }
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
        let length = input.dim(1)
        let span = configuration.positionEmbeddingSize
        let table = relativeEmbeddings(MLXArray(Int32(0) ..< Int32(2 * span)))
        let relative = layerNorm(table)[.newAxis]
        let positions = MLXArray(configuration.relativePositions(length: length), [length, length])
        let c2p =
            configuration.positionAttentionTypes.contains("c2p")
            ? clip(positions + Int32(span), min: 0, max: 2 * span - 1)[.newAxis, .newAxis] : nil
        let p2c =
            configuration.positionAttentionTypes.contains("p2c")
            ? clip(-positions + Int32(span), min: 0, max: 2 * span - 1)[.newAxis, .newAxis] : nil
        let keep = mask[0..., .newAxis, 0..., .newAxis] .&& mask[0..., .newAxis, .newAxis, 0...]
        var states = input
        for layer in layers {
            states = layer(states, keep: keep, relative: relative, c2p: c2p, p2c: p2c)
        }
        return states
    }
}

final class DeBERTaV2Model: Module {
    @ModuleInfo var embeddings: DeBERTaV2Embeddings
    @ModuleInfo var encoder: DeBERTaV2Encoder

    init(_ config: DeBERTaV2Configuration) {
        _embeddings.wrappedValue = DeBERTaV2Embeddings(config)
        _encoder.wrappedValue = DeBERTaV2Encoder(config)
    }

    /// `ids` [B, L] int32, `mask` [B, L] bool -> last hidden states [B, L, D].
    func callAsFunction(_ ids: MLXArray, mask: MLXArray) -> MLXArray {
        encoder(embeddings(ids, mask: mask), mask: mask)
    }
}
