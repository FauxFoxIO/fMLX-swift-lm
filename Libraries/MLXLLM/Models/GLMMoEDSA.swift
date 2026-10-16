//
//  GLMMoEDSA.swift
//  LLM
//
//  GLM-5.x (`glm_moe_dsa`): absorbed MLA attention with DeepSeek sparse attention (DSA).
//  Layers marked "shared" in `indexer_types` reuse the previous layer's top-k selection.
//  Port of transformers' `models/glm_moe_dsa/modeling_glm_moe_dsa.py`.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct GLMMoEDSAConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var nSharedExperts: Int
    var nRoutedExperts: Int
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var qkNopeHeadDim: Int
    var vHeadDim: Int
    var normTopkProb: Bool
    var nGroup: Int
    var topkGroup: Int
    var numExpertsPerTok: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int
    var attentionBias: Bool
    var indexHeads: Int
    var indexHeadDim: Int
    var indexTopK: Int
    var ropeInterleave: Bool
    var indexerRopeInterleave: Bool
    var indexerTypes: [String]
    var mlpLayerTypes: [String]
    var numNextnPredictLayers: Int
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case firstKDenseReplace = "first_k_dense_replace"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case indexHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopK = "index_topk"
        case indexTopKFreq = "index_topk_freq"
        case indexSkipTopKOffset = "index_skip_topk_offset"
        case indexTopKPattern = "index_topk_pattern"
        case ropeInterleave = "rope_interleave"
        case indexerRopeInterleave = "indexer_rope_interleave"
        case indexerTypes = "indexer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case tieWordEmbeddings = "tie_word_embeddings"
        case topkMethod = "topk_method"
        case scoringFunc = "scoring_func"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        nRoutedExperts = try c.decode(Int.self, forKey: .nRoutedExperts)
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qLoraRank = try c.decode(Int.self, forKey: .qLoraRank)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 202_752
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        indexHeads = try c.decodeIfPresent(Int.self, forKey: .indexHeads) ?? 32
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexTopK = try c.decodeIfPresent(Int.self, forKey: .indexTopK) ?? 2048
        ropeInterleave = try c.decodeIfPresent(Bool.self, forKey: .ropeInterleave) ?? true
        indexerRopeInterleave =
            try c.decodeIfPresent(Bool.self, forKey: .indexerRopeInterleave) ?? true
        numNextnPredictLayers =
            try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        let topkMethod = try c.decodeIfPresent(String.self, forKey: .topkMethod) ?? "noaux_tc"
        let scoringFunc = try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        guard topkMethod == "noaux_tc", scoringFunc == "sigmoid" else {
            throw DecodingError.dataCorruptedError(
                forKey: .topkMethod, in: c,
                debugDescription: "glm_moe_dsa supports noaux_tc sigmoid routing only")
        }

        // `rope_parameters` (transformers v5) supersedes `rope_theta` / `rope_scaling`.
        let ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        let legacyTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = ropeParameters?["rope_theta"]?.asFloat() ?? legacyTheta ?? 10_000
        if let ropeParameters, case .string(let type) = ropeParameters["rope_type"],
            type != "default"
        {
            ropeScaling = ropeParameters
        } else {
            ropeScaling = try c.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeScaling)
        }

        if let types = try c.decodeIfPresent([String].self, forKey: .indexerTypes) {
            indexerTypes = types
        } else if let pattern = try? c.decodeIfPresent(String.self, forKey: .indexTopKPattern) {
            indexerTypes = pattern.map { $0 == "S" ? "shared" : "full" }
        } else if let pattern = try? c.decodeIfPresent([String].self, forKey: .indexTopKPattern) {
            indexerTypes = pattern
        } else {
            let freq = max(try c.decodeIfPresent(Int.self, forKey: .indexTopKFreq) ?? 1, 1)
            let offset = try c.decodeIfPresent(Int.self, forKey: .indexSkipTopKOffset) ?? 2
            indexerTypes = (0 ..< hiddenLayers).map {
                max($0 - offset + 1, 0) % freq == 0 ? "full" : "shared"
            }
        }
        if let types = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes) {
            mlpLayerTypes = types
        } else {
            let dense = min(
                try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 3, hiddenLayers)
            mlpLayerTypes =
                Array(repeating: "dense", count: dense)
                + Array(repeating: "sparse", count: hiddenLayers - dense)
        }
        guard indexerTypes.count >= hiddenLayers, mlpLayerTypes.count >= hiddenLayers,
            indexerTypes.first != "shared"
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .indexerTypes, in: c,
                debugDescription: "indexer_types / mlp_layer_types do not cover every layer")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(nSharedExperts, forKey: .nSharedExperts)
        try c.encode(nRoutedExperts, forKey: .nRoutedExperts)
        try c.encode(routedScalingFactor, forKey: .routedScalingFactor)
        try c.encode(kvLoraRank, forKey: .kvLoraRank)
        try c.encode(qLoraRank, forKey: .qLoraRank)
        try c.encode(qkRopeHeadDim, forKey: .qkRopeHeadDim)
        try c.encode(qkNopeHeadDim, forKey: .qkNopeHeadDim)
        try c.encode(vHeadDim, forKey: .vHeadDim)
        try c.encode(normTopkProb, forKey: .normTopkProb)
        try c.encode(nGroup, forKey: .nGroup)
        try c.encode(topkGroup, forKey: .topkGroup)
        try c.encode(numExpertsPerTok, forKey: .numExpertsPerTok)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encode(indexHeads, forKey: .indexHeads)
        try c.encode(indexHeadDim, forKey: .indexHeadDim)
        try c.encode(indexTopK, forKey: .indexTopK)
        try c.encode(ropeInterleave, forKey: .ropeInterleave)
        try c.encode(indexerRopeInterleave, forKey: .indexerRopeInterleave)
        try c.encode(indexerTypes, forKey: .indexerTypes)
        try c.encode(mlpLayerTypes, forKey: .mlpLayerTypes)
        try c.encode(numNextnPredictLayers, forKey: .numNextnPredictLayers)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
    }

    func makeRope(traditional: Bool) -> RoPELayer {
        initializeRope(
            dims: qkRopeHeadDim, base: ropeTheta, traditional: traditional,
            scalingConfig: ropeScaling, maxPositionEmbeddings: maxPositionEmbeddings)
    }
}

// MARK: - Sparse attention

/// Top-k token selection shared by a full indexer layer with the shared layers after it.
struct GLMMoEDSASelection {
    /// Selected key positions `[B, L, topK]`, or `nil` when every causal position is kept.
    let indices: MLXArray?
    /// Boolean attention mask broadcastable to `[B, H, L, T]`, or `nil` for none.
    let allowed: MLXArray?
}

/// Lightning indexer: scores every cached key for each query and keeps the top-k.
final class GLMMoEDSAIndexer: Module {
    let heads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear

    init(_ config: GLMMoEDSAConfiguration) {
        heads = config.indexHeads
        headDim = config.indexHeadDim
        // Folds the q.k softmax scale and the 1/sqrt(heads) weight scale; relu is positive-homogeneous.
        scale = pow(Float(config.indexHeadDim), -0.5) * pow(Float(config.indexHeads), -0.5)
        // RoPE rotates the leading `qk_rope_head_dim` features and passes the rest through.
        rope = config.makeRope(traditional: config.indexerRopeInterleave)
        _wqB.wrappedValue = Linear(config.qLoraRank, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _kNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: 1e-6)
        _weightsProj.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        super.init()
    }

    /// Indexer keys for new positions, `[B, 1, L, headDim]`.
    func keys(_ x: MLXArray, offset: RoPEOffset?) -> MLXArray {
        let k = kNorm(wk(x)).expandedDimensions(axis: 1)
        return applyRotaryPosition(rope, to: k, offset: offset)
    }

    /// Unmasked index scores `[B, L, T]` in float32 against all cached indexer keys.
    func scores(_ x: MLXArray, qResid: MLXArray, keys: MLXArray, offset: RoPEOffset?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = wqB(qResid).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        q = applyRotaryPosition(rope, to: q, offset: offset).asType(.float32)
        let k = keys.asType(.float32).swappedAxes(-1, -2)
        let w = (weightsProj(x).asType(.float32) * scale).transposed(0, 2, 1)
            .expandedDimensions(axis: -1)
        if l == 1 {
            return (relu(matmul(q, k)) * w).sum(axis: 1)
        }
        // Accumulate per head so prefill holds one [B, L, T] buffer instead of heads of them.
        var total = relu(matmul(q[0..., ..<1], k)) * w[0..., ..<1]
        for h in 1 ..< heads {
            total = total + relu(matmul(q[0..., h ..< (h + 1)], k)) * w[0..., h ..< (h + 1)]
        }
        return total.squeezed(axis: 1)
    }
}

final class GLMMoEDSAAttention: Module {
    let numHeads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let qHeadDim: Int
    let indexTopK: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "indexer") var indexer: GLMMoEDSAIndexer?

    init(_ config: GLMMoEDSAConfiguration, hasIndexer: Bool) {
        numHeads = config.attentionHeads
        qkNopeHeadDim = config.qkNopeHeadDim
        qkRopeHeadDim = config.qkRopeHeadDim
        kvLoraRank = config.kvLoraRank
        qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        indexTopK = config.indexTopK

        var scale = pow(Float(qHeadDim), -0.5)
        if let scaling = config.ropeScaling,
            let mscaleAllDim = scaling["mscale_all_dim"]?.asFloat(), mscaleAllDim != 0,
            let factor = scaling["factor"]?.asFloat(), factor > 1
        {
            let s = 0.1 * mscaleAllDim * log(factor) + 1
            scale *= s * s
        }
        self.scale = scale
        rope = config.makeRope(traditional: config.ropeInterleave)

        _qAProj.wrappedValue = Linear(
            config.hiddenSize, config.qLoraRank, bias: config.attentionBias)
        _qALayerNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: 1e-6)
        _qBProj.wrappedValue = Linear(config.qLoraRank, numHeads * qHeadDim, bias: false)
        _kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, kvLoraRank + qkRopeHeadDim, bias: config.attentionBias)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: 1e-6)
        _embedQ.wrappedValue = MultiLinear(
            inputDims: qkNopeHeadDim, outputDims: kvLoraRank, numHeads: numHeads)
        _unembedOut.wrappedValue = MultiLinear(
            inputDims: kvLoraRank, outputDims: config.vHeadDim, numHeads: numHeads)
        _oProj.wrappedValue = Linear(
            numHeads * config.vHeadDim, config.hiddenSize, bias: config.attentionBias)
        _indexer.wrappedValue = hasIndexer ? GLMMoEDSAIndexer(config) : nil
        super.init()
    }

    private func multiLinear(_ module: Module, _ x: MLXArray) -> MLXArray {
        switch module {
        case let m as MultiLinear: m(x)
        case let m as QuantizedMultiLinear: m(x)
        default: fatalError("expected MultiLinear, got \(type(of: module))")
        }
    }

    /// Returns the attention output and the top-k selection this layer used.
    ///
    /// The cache holds the normalized KV latent as keys and `[k_pe | indexer key]` as values,
    /// so attention never materializes per-head keys or values.
    func callAsFunction(
        _ x: MLXArray, cache: KVCache?, sharedSelection: GLMMoEDSASelection?
    ) -> (MLXArray, GLMMoEDSASelection) {
        let (b, l) = (x.dim(0), x.dim(1))
        precondition(
            !(cache is QuantizedKVCacheProtocol), "glm_moe_dsa needs an unquantized KV cache")

        let qResid = qALayerNorm(qAProj(x))
        let q = qBProj(qResid).reshaped(b, l, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let qParts = split(q, indices: [qkNopeHeadDim], axis: -1)
        let kv = split(kvAProjWithMqa(x), indices: [kvLoraRank], axis: -1)

        let ropeOffset = cache?.ropeOffset
        let offset = cache?.offset ?? 0
        let qPe = applyRotaryPosition(rope, to: qParts[1], offset: ropeOffset)
        var latent = kvALayerNorm(kv[0]).expandedDimensions(axis: 1)
        let kPe = applyRotaryPosition(
            rope, to: kv[1].reshaped(b, 1, l, qkRopeHeadDim), offset: ropeOffset)

        var values = kPe
        if let indexer {
            values = concatenated([kPe, indexer.keys(x, offset: ropeOffset)], axis: -1)
        }
        if let cache {
            (latent, values) = cache.update(keys: latent, values: values)
        }
        let t = latent.dim(2)
        let keysPe = values[.ellipsis, ..<qkRopeHeadDim]

        let selection: GLMMoEDSASelection
        if let indexer {
            let causal: MLXArray? =
                l > 1
                ? MLXArray(0 ..< t)[.newAxis, 0...] .<= (MLXArray(0 ..< l) + offset)[0..., .newAxis]
                : nil
            if t > indexTopK {
                var scores = indexer.scores(
                    x, qResid: qResid, keys: values[.ellipsis, qkRopeHeadDim...],
                    offset: ropeOffset)
                if let causal {
                    scores = MLX.where(causal, scores, MLXArray(-Float.infinity))
                }
                let indices = argPartition(-scores, kth: indexTopK - 1, axis: -1)[
                    .ellipsis, ..<indexTopK]
                var allowed: MLXArray? = nil
                if l > 1 {
                    let picked = putAlong(
                        MLXArray.zeros([b, l, t], dtype: .bool), indices,
                        values: MLXArray(true), axis: -1)
                    allowed = (picked .&& causal!).expandedDimensions(axis: 1)
                }
                selection = GLMMoEDSASelection(indices: indices, allowed: allowed)
            } else {
                selection = GLMMoEDSASelection(indices: nil, allowed: causal)
            }
        } else {
            guard let sharedSelection else {
                fatalError("a shared DSA layer needs a preceding full indexer layer")
            }
            selection = sharedSelection
        }

        let qLatent = multiLinear(embedQ, qParts[0])
        var output: MLXArray
        if l == 1, let indices = selection.indices {
            // Decode: gather the selected latents so attention reads only top-k rows.
            let rows = indices.expandedDimensions(axis: -1)
            let selectedLatent = takeAlong(
                latent, broadcast(rows, to: [b, 1, indexTopK, kvLoraRank]), axis: 2)
            let selectedPe = takeAlong(
                keysPe, broadcast(rows, to: [b, 1, indexTopK, qkRopeHeadDim]), axis: 2)
            let peScores = matmul(qPe * scale, selectedPe.swappedAxes(-1, -2))
            output = MLXFast.scaledDotProductAttention(
                queries: qLatent, keys: selectedLatent, values: selectedLatent, scale: scale,
                mask: .array(peScores))
        } else {
            var peScores = matmul(qPe * scale, keysPe.swappedAxes(-1, -2))
            if let allowed = selection.allowed {
                peScores = MLX.where(
                    allowed, peScores, MLXArray(-Float.infinity).asType(peScores.dtype))
            }
            output = MLXFast.scaledDotProductAttention(
                queries: qLatent, keys: latent, values: latent, scale: scale,
                mask: .array(peScores))
        }
        output = multiLinear(unembedOut, output)
        output = output.transposed(0, 2, 1, 3).reshaped(b, l, -1)
        return (oProj(output), selection)
    }
}

// MARK: - MLP and MoE

final class GLMMoEDSAMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// `noaux_tc` router: selects on `sigmoid + e_score_correction_bias`, weights by `sigmoid`.
final class GLMMoEDSAGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float
    let nGroup: Int
    let topkGroup: Int

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: GLMMoEDSAConfiguration) {
        topK = config.numExpertsPerTok
        normTopkProb = config.normTopkProb
        routedScalingFactor = config.routedScalingFactor
        nGroup = config.nGroup
        topkGroup = config.topkGroup
        _weight.wrappedValue = zeros([config.nRoutedExperts, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = zeros([config.nRoutedExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // The reference router runs in float32 (`moe_router_dtype`).
        let logits = matmul(x.asType(.float32), weight.asType(.float32).T)
        if nGroup == 1 {
            return sigmoidRouterTopK(
                logits: logits, bias: eScoreCorrectionBias, k: topK, normalize: normTopkProb,
                scalingFactor: routedScalingFactor)
        }
        let scores = sigmoid(logits)
        var selection = scores + eScoreCorrectionBias.asType(.float32)
        if nGroup > 1 {
            let grouped = unflatten(selection, axis: -1, shape: [nGroup, -1])
            let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1)
            let dropped = argPartition(groupScores, kth: nGroup - topkGroup - 1, axis: -1)[
                .ellipsis, ..<(nGroup - topkGroup)]
            let masked = putAlong(
                grouped, dropped.expandedDimensions(axis: -1),
                values: MLXArray(-Float.infinity), axis: -2)
            selection = flattened(masked, start: -2, end: -1)
        }
        let indices = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, indices, axis: -1)
        if normTopkProb {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, weights * routedScalingFactor)
    }
}

final class GLMMoEDSAMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: GLMMoEDSAGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLMMoEDSAMLP?

    init(_ config: GLMMoEDSAConfiguration) {
        _gate.wrappedValue = GLMMoEDSAGate(config)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts)
        if config.nSharedExperts > 0 {
            _sharedExperts.wrappedValue = GLMMoEDSAMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * config.nSharedExperts)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMLP(x, indices)
        y = weightedExpertSum(y, scores).asType(y.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

// MARK: - Decoder

final class GLMMoEDSADecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: GLMMoEDSAAttention
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: GLMMoEDSAConfiguration, hasIndexer: Bool, sparseMLP: Bool) {
        _attention.wrappedValue = GLMMoEDSAAttention(config, hasIndexer: hasIndexer)
        _mlp.wrappedValue =
            sparseMLP
            ? GLMMoEDSAMoE(config)
            : GLMMoEDSAMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, cache: KVCache?, sharedSelection: GLMMoEDSASelection?
    ) -> (MLXArray, GLMMoEDSASelection) {
        let (r, selection) = attention(
            inputLayerNorm(x), cache: cache, sharedSelection: sharedSelection)
        let h = x + r
        let mlpOut = (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        return (h + mlpOut, selection)
    }
}

public final class GLMMoEDSAModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [GLMMoEDSADecoderLayer]
    let norm: RMSNorm

    init(_ config: GLMMoEDSAConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = (0 ..< config.hiddenLayers).map { i in
            GLMMoEDSADecoderLayer(
                config, hasIndexer: config.indexerTypes[i] != "shared",
                sparseMLP: config.mlpLayerTypes[i] == "sparse")
        }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// Final normalized hidden states.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        var selection: GLMMoEDSASelection? = nil
        for (i, layer) in layers.enumerated() {
            (h, selection) = layer(h, cache: cache?[i], sharedSelection: selection)
        }
        return norm(h)
    }
}

public final class GLMMoEDSAModel: Module, LLMModel {
    public let vocabularySize: Int
    public let model: GLMMoEDSAModelInner
    let configuration: GLMMoEDSAConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ configuration: GLMMoEDSAConfiguration) {
        self.configuration = configuration
        vocabularySize = configuration.vocabularySize
        model = GLMMoEDSAModelInner(configuration)
        _lmHead.wrappedValue = Linear(
            configuration.hiddenSize, configuration.vocabularySize, bias: false)
        super.init()
    }

    /// One unbounded cache per layer. DSA selects over absolute positions, so the cache is
    /// never windowed; KV quantization is not supported.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        model.layers.map { _ in KVCacheSimple() }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        forward(input, cache: cache, state: state, lastTokenOnly: false)
    }

    public func nextTokenLogits(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        forward(input, cache: cache, state: state, lastTokenOnly: true)
    }

    private func forward(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?, lastTokenOnly: Bool
    ) -> LMOutput {
        let hidden = model(input.tokens, cache: cache)
        let emit = state?[mtpEmitFlagKey] ?? false
        let finalOnly =
            emit ? (state?[mtpFinalPrefillLogitsOnlyKey] ?? false) : lastTokenOnly
        let logits = lmHead(finalOnly ? hidden[0..., (-1)..., 0...] : hidden)
        guard emit else { return LMOutput(logits: logits) }
        var out = state ?? LMOutput.State()
        out[mtpFinalPrefillLogitsOnlyKey] = nil
        out[mtpLastHiddenStatesKey] = hidden
        // The MTP head keeps its own cache, so no target K/V is shared.
        out[mtpSharedKVSourceIndicesKey] = [:]
        return LMOutput(logits: logits, state: out)
    }

    public var excludedCheckpointNamespaces: [String] {
        let first = configuration.hiddenLayers
        return (first ..< first + configuration.numNextnPredictLayers).map {
            "model.layers.\($0)"
        }
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        let mtpLayers = Set(excludedCheckpointNamespaces)
        var checkpoint = try checkpoint.mapNames { name in
            let parts = name.split(separator: ".", maxSplits: 3)
            if parts.count > 3, mtpLayers.contains(parts[0 ..< 3].joined(separator: ".")) {
                return nil
            }
            return name.contains("rotary_emb.inv_freq") ? nil : name
        }
        try GLMMoEDSACheckpoint.normalize(
            &checkpoint, configuration: configuration,
            layerPrefixes: (0 ..< configuration.hiddenLayers).map { "model.layers.\($0)" })
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(ModelCheckpoint(weights: weights)).weights
    }
}

extension GLMMoEDSAModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}

extension GLMMoEDSAModel {
    public var toolCallFormat: ToolCallFormat? { .glm4 }
}

// MARK: - Checkpoint conversion

/// Converts Hugging Face (FP8 block-quantized) and mlx-community checkpoints to module names.
enum GLMMoEDSACheckpoint {
    static let fp8BlockSize = 128

    /// Dequantizes FP8 weights, stacks per-expert MLPs, splits `kv_b_proj` into the absorbed
    /// `embed_q` / `unembed_out` projections, and keeps router weights in float32.
    static func normalize(
        _ checkpoint: inout ModelCheckpoint, configuration: GLMMoEDSAConfiguration,
        layerPrefixes: [String]
    ) throws {
        var weights = checkpoint.weights
        var quantization = checkpoint.perLayerQuantization

        for key in weights.keys where key.hasSuffix(".weight_scale_inv") {
            let weightKey = String(key.dropLast("_scale_inv".count))
            guard let scaleInv = weights.removeValue(forKey: key),
                let weight = weights[weightKey]
            else { continue }
            weights[weightKey] = dequantizeFP8(weight, scaleInv: scaleInv)
        }

        for prefix in layerPrefixes {
            stackExperts(
                &weights, &quantization, prefix: "\(prefix).mlp",
                count: configuration.nRoutedExperts)
            try splitKVB(
                &weights, &quantization, prefix: "\(prefix).self_attn",
                configuration: configuration)
            if let gate = weights["\(prefix).mlp.gate.weight"] {
                weights["\(prefix).mlp.gate.weight"] = gate.asType(.float32)
            }
        }
        checkpoint.weights = weights
        checkpoint.perLayerQuantization = quantization
    }

    /// Block-wise FP8 dequantization. MLX loads `F8_E4M3` tensors as `uint8`.
    static func dequantizeFP8(_ weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
        let w = weight.dtype == .uint8 ? fromFP8(weight, dtype: .bfloat16) : weight
        let bs = fp8BlockSize
        let (m, n) = (w.dim(0), w.dim(1))
        let padBottom = (bs - m % bs) % bs
        let padSide = (bs - n % bs) % bs
        var p = padded(w, widths: [.init((0, padBottom)), .init((0, padSide))])
        p = p.reshaped((m + padBottom) / bs, bs, (n + padSide) / bs, bs)
        p = p * scaleInv[0..., .newAxis, 0..., .newAxis]
        return p.reshaped(m + padBottom, n + padSide)[..<m, ..<n].asType(.bfloat16)
    }

    private static func stackExperts(
        _ weights: inout [String: MLXArray],
        _ quantization: inout BaseConfiguration.PerLayerQuantization?,
        prefix: String, count: Int
    ) {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            for part in ["weight", "scales", "biases"]
            where weights["\(prefix).experts.0.\(proj).\(part)"] != nil {
                let parts = (0 ..< count).compactMap {
                    weights.removeValue(forKey: "\(prefix).experts.\($0).\(proj).\(part)")
                }
                weights["\(prefix).switch_mlp.\(proj).\(part)"] = stacked(parts)
            }
            if let option = quantization?.perLayerQuantization["\(prefix).experts.0.\(proj)"] {
                quantization?.perLayerQuantization["\(prefix).switch_mlp.\(proj)"] = option
            }
        }
    }

    private static func splitKVB(
        _ weights: inout [String: MLXArray],
        _ quantization: inout BaseConfiguration.PerLayerQuantization?,
        prefix: String, configuration: GLMMoEDSAConfiguration
    ) throws {
        guard var w = weights.removeValue(forKey: "\(prefix).kv_b_proj.weight") else { return }
        var option: BaseConfiguration.Quantization? = nil
        if let scales = weights.removeValue(forKey: "\(prefix).kv_b_proj.scales") {
            let biases = weights.removeValue(forKey: "\(prefix).kv_b_proj.biases")
            if let declared = quantization?.quantization(layer: "\(prefix).kv_b_proj") {
                option = declared
            } else if biases != nil {
                // Undeclared settings: infer affine bits and group size from the shapes.
                let rank = configuration.kvLoraRank
                option = .init(groupSize: rank / scales.dim(-1), bits: w.dim(-1) * 32 / rank)
            }
            guard let option else {
                throw GLMMoEDSACheckpointError.unknownQuantization("\(prefix).kv_b_proj")
            }
            w = dequantized(
                w, scales: scales, biases: biases, groupSize: option.groupSize,
                bits: option.bits, mode: option.mode)
        }

        let nope = configuration.qkNopeHeadDim
        let v = w.reshaped(configuration.attentionHeads, nope + configuration.vHeadDim, -1)
        let parts = [
            ("embed_q", contiguous(v[0..., ..<nope, 0...].swappedAxes(-1, -2))),
            ("unembed_out", contiguous(v[0..., nope..., 0...])),
        ]
        for (name, part) in parts {
            guard let option else {
                weights["\(prefix).\(name).weight"] = part
                continue
            }
            let (q, scales, biases) = quantized(
                part, groupSize: option.groupSize, bits: option.bits, mode: option.mode)
            weights["\(prefix).\(name).weight"] = q
            weights["\(prefix).\(name).scales"] = scales
            weights["\(prefix).\(name).biases"] = biases
            quantization?.perLayerQuantization["\(prefix).\(name)"] = .quantize(option)
        }
    }
}

enum GLMMoEDSACheckpointError: LocalizedError {
    case unknownQuantization(String)

    var errorDescription: String? {
        switch self {
        case .unknownQuantization(let layer):
            "Quantized '\(layer)' has no declared quantization settings."
        }
    }
}
