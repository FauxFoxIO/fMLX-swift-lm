//
//  GLM5Next.swift
//  LLM
//
//  GLM-5.3-Flash (`glm5_next`), text only: Kimi Delta Attention layers, no-RoPE MLA layers with
//  a pooled-key DeepSeek sparse-attention indexer, manifold-constrained hyper-connections, and
//  a sigmoid-routed MoE. Port of transformers' `models/glm5_next/modeling_glm5_next.py`.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct GLM5NextConfiguration: Codable, Sendable {
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
    var qkNopeHeadDim: Int
    var vHeadDim: Int
    var normTopkProb: Bool
    var numExpertsPerTok: Int
    var rmsNormEps: Float
    var attentionBias: Bool
    var indexHeads: Int
    var indexHeadDim: Int
    var indexTopK: Int
    var indexKPool: Int
    var indexSelectTail: Bool
    var indexerTypes: [String]
    var mlpLayerTypes: [String]
    var linearLayers: [Bool]
    var linearHeads: Int
    var linearHeadDim: Int
    var linearConvKernel: Int
    var linearLowerBound: Float?
    var hcMult: Int
    var hcEps: Float
    var hcSinkhornIters: Int
    var swigluLimit: Float?
    var numNextnPredictLayers: Int
    var tieWordEmbeddings: Bool

    enum OuterKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

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
        case numExpertsPerTok = "num_experts_per_tok"
        case firstKDenseReplace = "first_k_dense_replace"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case indexHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopK = "index_topk"
        case indexKPool = "index_kpool"
        case indexSelectTail = "index_kpool_always_select_tail"
        case indexTopKFreq = "index_topk_freq"
        case indexSkipTopKOffset = "index_skip_topk_offset"
        case indexTopKPattern = "index_topk_pattern"
        case indexerTypes = "indexer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case layerTypes = "layer_types"
        case linearAttnConfig = "linear_attn_config"
        case linearHeads = "linear_num_heads"
        case linearHeadDim = "linear_head_dim"
        case linearConvKernel = "linear_conv_kernel_dim"
        case linearLowerBound = "linear_lower_bound"
        case hcMult = "hc_mult"
        case hcEps = "hc_eps"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case swigluLimit = "swiglu_limit"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case tieWordEmbeddings = "tie_word_embeddings"
        case topkMethod = "topk_method"
        case scoringFunc = "scoring_func"
    }

    struct LinearAttentionConfig: Codable {
        var numHeads: Int?
        var headDim: Int?
        var shortConvKernelSize: Int?
        var gateLowerBound: Float?
        var safeGate: Bool?
        var kdaLayers: [Int]?

        enum CodingKeys: String, CodingKey {
            case numHeads = "num_heads"
            case headDim = "head_dim"
            case shortConvKernelSize = "short_conv_kernel_size"
            case gateLowerBound = "gate_lower_bound"
            case safeGate = "safe_gate"
            case kdaLayers = "kda_layers"
        }
    }

    /// Reads the text model from `text_config` when present, else from the top level.
    public init(from decoder: Decoder) throws {
        let outer = try decoder.container(keyedBy: OuterKeys.self)
        let c =
            outer.contains(.textConfig)
            ? try outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            : try decoder.container(keyedBy: CodingKeys.self)

        modelType = try outer.decodeIfPresent(String.self, forKey: .modelType) ?? "glm5_next"
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
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        indexHeads = try c.decodeIfPresent(Int.self, forKey: .indexHeads) ?? 32
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexTopK = try c.decodeIfPresent(Int.self, forKey: .indexTopK) ?? 2048
        indexKPool = try c.decodeIfPresent(Int.self, forKey: .indexKPool) ?? 16
        indexSelectTail = try c.decodeIfPresent(Bool.self, forKey: .indexSelectTail) ?? true
        hcMult = try c.decodeIfPresent(Int.self, forKey: .hcMult) ?? 4
        hcEps = try c.decodeIfPresent(Float.self, forKey: .hcEps) ?? 1e-6
        hcSinkhornIters = try c.decodeIfPresent(Int.self, forKey: .hcSinkhornIters) ?? 20
        swigluLimit = try c.decodeIfPresent(Float.self, forKey: .swigluLimit)
        numNextnPredictLayers =
            try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? outer.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        let linear = try c.decodeIfPresent(LinearAttentionConfig.self, forKey: .linearAttnConfig)
        linearHeads =
            try linear?.numHeads ?? c.decodeIfPresent(Int.self, forKey: .linearHeads) ?? 64
        linearHeadDim =
            try linear?.headDim ?? c.decodeIfPresent(Int.self, forKey: .linearHeadDim) ?? 128
        linearConvKernel =
            try linear?.shortConvKernelSize
            ?? c.decodeIfPresent(Int.self, forKey: .linearConvKernel) ?? 4
        if c.contains(.linearLowerBound) {
            linearLowerBound = try c.decodeIfPresent(Float.self, forKey: .linearLowerBound)
        } else {
            linearLowerBound = -5
        }
        if let linear {
            linearLowerBound = linear.gateLowerBound ?? linearLowerBound
            if linear.safeGate ?? true, linearLowerBound == nil {
                linearLowerBound = -5
            }
        }

        let layers = hiddenLayers
        if let types = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            linearLayers = types.map { $0 == "linear_attention" }
        } else if let kda = linear?.kdaLayers {
            linearLayers = (0 ..< layers).map { kda.contains($0) }
        } else {
            linearLayers = (0 ..< layers).map { $0 % 4 != 3 }
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
            indexerTypes = (0 ..< layers).map {
                max($0 - offset + 1, 0) % freq == 0 ? "full" : "shared"
            }
        }
        if let types = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes) {
            mlpLayerTypes = types
        } else {
            let dense = min(
                try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 3, layers)
            mlpLayerTypes =
                Array(repeating: "dense", count: dense)
                + Array(repeating: "sparse", count: layers - dense)
        }

        func invalid(_ key: CodingKeys, _ message: String) -> DecodingError {
            DecodingError.dataCorruptedError(forKey: key, in: c, debugDescription: message)
        }
        let topkMethod = try c.decodeIfPresent(String.self, forKey: .topkMethod) ?? "noaux_tc"
        let scoringFunc = try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        let nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        guard topkMethod == "noaux_tc", scoringFunc == "sigmoid", nGroup == 1 else {
            throw invalid(.topkMethod, "glm5_next supports single-group noaux_tc sigmoid routing")
        }
        guard try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 0 == 0 else {
            throw invalid(.qkRopeHeadDim, "glm5_next sparse attention layers use no RoPE")
        }
        guard linearLayers.count >= layers, indexerTypes.count >= layers,
            mlpLayerTypes.count >= layers
        else {
            throw invalid(.layerTypes, "layer type lists do not cover every layer")
        }
        guard indexKPool > 0, indexTopK % indexKPool == 0 else {
            throw invalid(.indexKPool, "index_topk must be a multiple of index_kpool")
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
        try c.encode(0, forKey: .qkRopeHeadDim)
        try c.encode(qkNopeHeadDim, forKey: .qkNopeHeadDim)
        try c.encode(vHeadDim, forKey: .vHeadDim)
        try c.encode(normTopkProb, forKey: .normTopkProb)
        try c.encode(numExpertsPerTok, forKey: .numExpertsPerTok)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encode(indexHeads, forKey: .indexHeads)
        try c.encode(indexHeadDim, forKey: .indexHeadDim)
        try c.encode(indexTopK, forKey: .indexTopK)
        try c.encode(indexKPool, forKey: .indexKPool)
        try c.encode(indexSelectTail, forKey: .indexSelectTail)
        try c.encode(indexerTypes, forKey: .indexerTypes)
        try c.encode(mlpLayerTypes, forKey: .mlpLayerTypes)
        try c.encode(
            linearLayers.map { $0 ? "linear_attention" : "indexed_attention" },
            forKey: .layerTypes)
        try c.encode(linearHeads, forKey: .linearHeads)
        try c.encode(linearHeadDim, forKey: .linearHeadDim)
        try c.encode(linearConvKernel, forKey: .linearConvKernel)
        try c.encode(linearLowerBound, forKey: .linearLowerBound)
        try c.encode(hcMult, forKey: .hcMult)
        try c.encode(hcEps, forKey: .hcEps)
        try c.encode(hcSinkhornIters, forKey: .hcSinkhornIters)
        try c.encodeIfPresent(swigluLimit, forKey: .swigluLimit)
        try c.encode(numNextnPredictLayers, forKey: .numNextnPredictLayers)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
    }
}

// MARK: - Hyper-connections

/// Computes the pre, post, and Sinkhorn-balanced comb weights of one hyper-connection.
final class GLM5NextHyperMixer: @unchecked Sendable {
    let mult: Int
    let rmsNormEps: Float
    private let weights: @Sendable ([MLXArray]) -> [MLXArray]

    init(_ config: GLM5NextConfiguration) {
        let n = config.hcMult
        let eps = config.hcEps
        let iterations = config.hcSinkhornIters
        mult = n
        rmsNormEps = config.rmsNormEps
        weights = compile { args in
            let (mixes, scale, base) = (args[0], args[1], args[2])
            let pre = sigmoid(mixes[.ellipsis, ..<n] * scale[0] + base[..<n]) + eps
            let post = sigmoid(mixes[.ellipsis, n ..< (2 * n)] * scale[1] + base[n ..< (2 * n)]) * 2
            let combLogits = mixes[.ellipsis, (2 * n)...] * scale[2] + base[(2 * n)...]
            var comb =
                softmax(unflatten(combLogits, axis: -1, shape: [n, n]), axis: -1) + eps
            comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
            for _ in 1 ..< max(iterations, 1) {
                comb = comb / (comb.sum(axis: -1, keepDims: true) + eps)
                comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
            }
            return [pre, post, comb]
        }
    }

    /// Returns `(pre, post, comb)` in float32 for `mixes` `[..., (2 + N) * N]`.
    func callAsFunction(mixes: MLXArray, scale: MLXArray, base: MLXArray) -> (
        MLXArray, MLXArray, MLXArray
    ) {
        let out = weights([mixes, scale, base])
        return (out[0], out[1], out[2])
    }
}

/// One hyper-connection: collapses `hc_mult` residual streams into a block input and
/// returns the weights that expand the block output back into streams.
final class GLM5NextHyperConnection: Module {
    @ParameterInfo(key: "fn") var fn: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(_ config: GLM5NextConfiguration) {
        let n = config.hcMult
        _fn.wrappedValue = zeros([(2 + n) * n, n * config.hiddenSize])
        _base.wrappedValue = zeros([(2 + n) * n])
        _scale.wrappedValue = ones([3])
        super.init()
    }

    /// `streams` `[B, L, N, D]` -> `(post [B, L, N], comb [B, L, N, N], collapsed [B, L, D])`.
    func callAsFunction(_ streams: MLXArray, mixer: GLM5NextHyperMixer) -> (
        MLXArray, MLXArray, MLXArray
    ) {
        let (b, l) = (streams.dim(0), streams.dim(1))
        let flat = streams.reshaped(b, l, -1).asType(.float32)
        let normed = MLXFast.rmsNorm(flat, weight: MLXArray.mlxNone, eps: mixer.rmsNormEps)
        let mixes = matmul(normed, fn.asType(.float32).T)
        let (pre, post, comb) = mixer(
            mixes: mixes, scale: scale.asType(.float32), base: base.asType(.float32))
        let collapsed = (pre.expandedDimensions(axis: -1) * streams).sum(axis: 2)
        return (post, comb, collapsed.asType(streams.dtype))
    }

    /// `post * output + comb^T @ residual`, the new residual streams.
    static func expand(
        _ output: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
    ) -> MLXArray {
        let dtype = residual.dtype
        return post.asType(dtype).expandedDimensions(axis: -1) * output.expandedDimensions(axis: -2)
            + matmul(comb.asType(dtype).swappedAxes(-1, -2), residual)
    }
}

// MARK: - Kimi Delta Attention

final class GLM5NextLinearAttention: Module {
    /// Heads and `heads * headDim` on this rank.
    var heads: Int
    let headDim: Int
    var projectionDim: Int
    let convKernel: Int
    let lowerBound: Float?
    let normEps: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "f_a_proj") var fAProj: Linear
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

    init(_ config: GLM5NextConfiguration) {
        heads = config.linearHeads
        headDim = config.linearHeadDim
        projectionDim = heads * headDim
        convKernel = config.linearConvKernel
        lowerBound = config.linearLowerBound
        normEps = config.rmsNormEps
        let hidden = config.hiddenSize
        _qProj.wrappedValue = Linear(hidden, projectionDim, bias: false)
        _kProj.wrappedValue = Linear(hidden, projectionDim, bias: false)
        _vProj.wrappedValue = Linear(hidden, projectionDim, bias: false)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: 3 * projectionDim, outputChannels: 3 * projectionDim,
            kernelSize: convKernel, stride: 1, padding: 0, dilation: 1, groups: 3 * projectionDim,
            bias: false)
        _fAProj.wrappedValue = Linear(hidden, headDim, bias: false)
        _fBProj.wrappedValue = Linear(headDim, projectionDim, bias: false)
        _bProj.wrappedValue = Linear(hidden, heads, bias: false)
        _gAProj.wrappedValue = Linear(hidden, headDim, bias: false)
        _gBProj.wrappedValue = Linear(headDim, projectionDim, bias: false)
        _oNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _oProj.wrappedValue = Linear(projectionDim, hidden, bias: false)
        _aLog.wrappedValue = zeros([heads])
        _dtBias.wrappedValue = zeros([projectionDim])
        super.init()
    }

    /// Depthwise causal conv plus SiLU, in float32 like the reference.
    private func shortConv(_ convInput: MLXArray, length: Int, dtype: DType) -> MLXArray {
        let weight = conv1d.weight.asType(.float32)
        let out: MLXArray
        if length == 1 {
            var acc = convInput[0..., 0, 0...].asType(.float32) * weight[0..., 0, 0]
            for tap in 1 ..< convKernel {
                acc = acc + convInput[0..., tap, 0...].asType(.float32) * weight[0..., tap, 0]
            }
            out = acc.expandedDimensions(axis: 1)
        } else {
            out = MLX.conv1d(
                convInput.asType(.float32), weight, groups: 3 * projectionDim)
        }
        return silu(out).asType(dtype)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: MambaCache?, checkpointIndices: [Int] = []
    ) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        var qkv = concatenated([qProj(x), kProj(x), vProj(x)], axis: -1)
        if let mask {
            qkv = qkv * mask[.ellipsis, .newAxis].asType(qkv.dtype)
        }
        let convState =
            cache?[0] ?? MLXArray.zeros([b, convKernel - 1, 3 * projectionDim], dtype: qkv.dtype)
        let convInput = concatenated([convState, qkv], axis: 1)
        let mixed = shortConv(convInput, length: t, dtype: x.dtype)
        let parts = split(mixed, indices: [projectionDim, 2 * projectionDim], axis: -1)
        let q = parts[0].reshaped(b, t, heads, headDim)
        let k = parts[1].reshaped(b, t, heads, headDim)
        let v = parts[2].reshaped(b, t, heads, headDim)

        let decay = kimiDeltaDecay(
            f: fBProj(fAProj(x)).reshaped(b, t, heads, headDim),
            aLog: aLog.reshaped(heads, 1), dtBias: dtBias.reshaped(heads, headDim),
            lowerBound: lowerBound)
        let beta = sigmoid(bProj(x).asType(.float32))

        let splits = checkpointIndices.filter { $0 > 0 && $0 < t }
        let core: MLXArray
        if let cache {
            let state: MLXArray
            if !splits.isEmpty, mask == nil {
                let result = kimiDeltaUpdateWithCheckpoints(
                    q: q, k: k, v: v, decay: decay, beta: beta, state: cache[1],
                    useKernel: !training)
                core = result.output
                state = result.state
                for split in Set(splits).sorted() {
                    cache.saveSpeculativeCheckpoint(
                        convState: contiguous(convInput[0..., split ..< (split + convKernel - 1)]),
                        recurrentState: result.checkpoints[split - 1],
                        advancedBy: split, rewinding: t - split)
                }
            } else {
                let result = kimiDeltaUpdate(
                    q: q, k: k, v: v, decay: decay, beta: beta, state: cache[1], mask: mask,
                    useKernel: !training)
                core = result.0
                state = result.1
            }
            cache[0] = contiguous(convInput[0..., (-(convKernel - 1))..., 0...])
            cache[1] = state
            cache.advance(t)
        } else {
            core =
                kimiDeltaUpdate(
                    q: q, k: k, v: v, decay: decay, beta: beta, mask: mask, useKernel: !training
                ).0
        }

        let gate = gBProj(gAProj(x)).reshaped(b, t, heads, headDim)
        let normed = MLXFast.rmsNorm(
            core.asType(.float32), weight: oNorm.weight.asType(.float32), eps: normEps)
        let out = (normed * sigmoid(gate.asType(.float32))).asType(core.dtype)
        return oProj(out.reshaped(b, t, -1))
    }
}

// MARK: - Sparse attention

/// The keys a query attends to, shared by a full indexer layer with a following shared layer.
enum GLM5NextSelection {
    /// Every causal position.
    case dense
    /// A boolean mask `[B, 1, L, T]` for a multi-token query.
    case masked(MLXArray)
    /// Gathered key rows `[B, N]` and their boolean mask `[B, 1, 1, N]` for one query.
    case gathered(indices: MLXArray, mask: MLXArray)
}

/// DSA indexer over pooled keys. It scores complete pools of `index_kpool` keys, keeps the top
/// `index_topk / index_kpool` pools, and adds the query's incomplete trailing pool.
final class GLM5NextIndexer: Module {
    let heads: Int
    let headDim: Int
    let pool: Int
    let selectedPools: Int
    let selectTail: Bool
    let scale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ParameterInfo(key: "index_kpool_compress_ape") var compressAPE: MLXArray
    @ParameterInfo(key: "index_kpool_compress_gate") var compressGate: MLXArray

    init(_ config: GLM5NextConfiguration) {
        heads = config.indexHeads
        headDim = config.indexHeadDim
        pool = config.indexKPool
        selectedPools = config.indexTopK / config.indexKPool
        selectTail = config.indexSelectTail
        // Folds the q.k softmax scale and the 1/sqrt(heads) weight scale; relu is positive-homogeneous.
        scale = pow(Float(headDim), -0.5) * pow(Float(heads), -0.5)
        _wqB.wrappedValue = Linear(config.qLoraRank, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _kNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: 1e-6)
        _weightsProj.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        _compressAPE.wrappedValue = zeros([pool, headDim])
        _compressGate.wrappedValue = ones([headDim, config.hiddenSize])
        super.init()
    }

    /// Cache rows `[B, 1, L, 2 * headDim]`: the indexer key and its pooling gate logits.
    func cacheEntries(_ x: MLXArray) -> MLXArray {
        concatenated([kNorm(wk(x)), matmul(x, compressGate.T)], axis: -1)
            .expandedDimensions(axis: 1)
    }

    /// Softmax-weighted keys of the first `pools` complete pools, `[B, P, headDim]`.
    func pooledKeys(_ entries: MLXArray, pools: Int) -> MLXArray {
        let b = entries.dim(0)
        let grouped = entries[0..., 0, ..<(pools * pool), 0...].reshaped(b, pools, pool, -1)
        let keys = grouped[.ellipsis, ..<headDim]
        let logits = grouped[.ellipsis, headDim...].asType(.float32) + compressAPE.asType(.float32)
        let weights = softmax(logits, axis: 2).asType(keys.dtype)
        return (weights * keys).sum(axis: 2)
    }

    /// Pool scores `[B, L, P]` in float32.
    func scores(_ x: MLXArray, qResid: MLXArray, pooled: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let q = wqB(qResid).reshaped(b, l, heads, headDim).asType(.float32)
        let k = pooled.asType(.float32).expandedDimensions(axis: 1).swappedAxes(-1, -2)
        let w = weightsProj(x).asType(.float32) * scale
        if l == 1 {
            return (relu(matmul(q, k)) * w.expandedDimensions(axis: -1)).sum(axis: 2)
        }
        // Accumulate per head so prefill holds one [B, L, P] buffer.
        let kT = k.squeezed(axis: 1)
        var total = relu(matmul(q[0..., 0..., 0], kT)) * w[0..., 0..., 0 ..< 1]
        for h in 1 ..< heads {
            total = total + relu(matmul(q[0..., 0..., h], kT)) * w[0..., 0..., h ..< (h + 1)]
        }
        return total
    }

    /// The selection for queries at positions `offset ..< offset + L` over `T` cached keys.
    func select(
        _ x: MLXArray, qResid: MLXArray, entries: MLXArray, offset: Int
    ) -> GLM5NextSelection {
        let (b, l) = (x.dim(0), x.dim(1))
        let t = entries.dim(2)
        let pools = t / pool
        // With every complete pool selected the result is plain causal attention.
        guard pools > selectedPools else { return .dense }

        let positions = MLXArray(offset ..< (offset + l)).asType(.int32)
        let visiblePools = floorDivide(positions + 1, pool)
        let valid =
            MLXArray(0 ..< pools).asType(.int32)[.newAxis, 0...]
            .< visiblePools[0..., .newAxis]
        let poolScores = MLX.where(
            valid, scores(x, qResid: qResid, pooled: pooledKeys(entries, pools: pools)),
            MLXArray(-Float.infinity))
        let picked = argPartition(-poolScores, kth: selectedPools - 1, axis: -1)[
            .ellipsis, ..<selectedPools]
        let tailWidth = selectTail ? pool - 1 : 0

        if l == 1 {
            // The last query sees every complete pool, so each picked pool is valid.
            let offsets = MLXArray(0 ..< pool).asType(.int32)
            var indices =
                (picked.asType(.int32) * pool).expandedDimensions(axis: -1) + offsets
            indices = indices.reshaped(b, selectedPools * pool)
            var mask = MLXArray.ones([1, selectedPools * pool], dtype: .bool)
            if tailWidth > 0 {
                let tail = MLXArray((t - tailWidth) ..< t).asType(.int32)
                indices = concatenated(
                    [indices, broadcast(tail[.newAxis, 0...], to: [b, tailWidth])], axis: 1)
                mask = concatenated(
                    [mask, (tail .>= MLXArray(Int32(pool * (t / pool))))[.newAxis, 0...]],
                    axis: 1)
            }
            let n = indices.dim(1)
            return .gathered(
                indices: indices, mask: broadcast(mask, to: [b, n]).reshaped(b, 1, 1, n))
        }

        let pickedPools =
            putAlong(
                MLXArray.zeros([b, l, pools], dtype: .bool), picked, values: MLXArray(true),
                axis: -1) .&& valid
        var tokens = repeated(pickedPools, count: pool, axis: -1)
        if t > pools * pool {
            tokens = padded(
                tokens, widths: [.init((0, 0)), .init((0, 0)), .init((0, t - pools * pool))],
                value: MLXArray(false))
        }
        if tailWidth > 0 {
            let keys = MLXArray(0 ..< t).asType(.int32)[.newAxis, 0...]
            let tailStart = (visiblePools * pool)[0..., .newAxis]
            let tail = (keys .>= tailStart) .&& (keys .<= positions[0..., .newAxis])
            tokens = tokens .|| tail
        }
        return .masked(tokens.expandedDimensions(axis: 1))
    }
}

/// Absorbed MLA without RoPE. The cache stores the normalized KV latent as keys and the
/// indexer rows as values.
final class GLM5NextSparseAttention: Module {
    /// Heads on this rank.
    var numHeads: Int
    let qkNopeHeadDim: Int
    let kvLoraRank: Int
    let scale: Float

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "indexer") var indexer: GLM5NextIndexer?

    init(_ config: GLM5NextConfiguration, hasIndexer: Bool) {
        numHeads = config.attentionHeads
        qkNopeHeadDim = config.qkNopeHeadDim
        kvLoraRank = config.kvLoraRank
        scale = pow(Float(config.qkNopeHeadDim), -0.5)
        _qAProj.wrappedValue = Linear(
            config.hiddenSize, config.qLoraRank, bias: config.attentionBias)
        _qALayerNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        _qBProj.wrappedValue = Linear(config.qLoraRank, numHeads * qkNopeHeadDim, bias: false)
        _kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, kvLoraRank, bias: config.attentionBias)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.rmsNormEps)
        _embedQ.wrappedValue = MultiLinear(
            inputDims: qkNopeHeadDim, outputDims: kvLoraRank, numHeads: numHeads)
        _unembedOut.wrappedValue = MultiLinear(
            inputDims: kvLoraRank, outputDims: config.vHeadDim, numHeads: numHeads)
        _oProj.wrappedValue = Linear(
            numHeads * config.vHeadDim, config.hiddenSize, bias: config.attentionBias)
        _indexer.wrappedValue = hasIndexer ? GLM5NextIndexer(config) : nil
        super.init()
    }

    private func multiLinear(_ module: Module, _ x: MLXArray) -> MLXArray {
        switch module {
        case let m as MultiLinear: m(x)
        case let m as QuantizedMultiLinear: m(x)
        default: fatalError("expected MultiLinear, got \(type(of: module))")
        }
    }

    func callAsFunction(
        _ x: MLXArray, cache: KVCache?, sharedSelection: GLM5NextSelection?
    ) -> (MLXArray, GLM5NextSelection) {
        let (b, l) = (x.dim(0), x.dim(1))
        precondition(
            !(cache is QuantizedKVCacheProtocol), "glm5_next needs an unquantized KV cache")

        let qResid = qALayerNorm(qAProj(x))
        let q = qBProj(qResid).reshaped(b, l, numHeads, qkNopeHeadDim).transposed(0, 2, 1, 3)
        var latent = kvALayerNorm(kvAProjWithMqa(x)).expandedDimensions(axis: 1)
        var entries =
            indexer?.cacheEntries(x) ?? MLXArray.zeros([b, 1, l, 1], dtype: latent.dtype)
        if let cache {
            (latent, entries) = cache.update(keys: latent, values: entries)
        }
        let t = latent.dim(2)
        let offset = t - l

        let selection: GLM5NextSelection
        if let indexer {
            selection = indexer.select(x, qResid: qResid, entries: entries, offset: offset)
        } else {
            guard let sharedSelection else {
                fatalError("a shared DSA layer needs a preceding full indexer layer")
            }
            selection = sharedSelection
        }

        let qLatent = multiLinear(embedQ, q)
        var output: MLXArray
        switch selection {
        case .dense:
            var mask = MLXFast.ScaledDotProductAttentionMaskMode.none
            if l > 1 {
                let causal =
                    MLXArray(0 ..< t)[.newAxis, 0...]
                    .<= (MLXArray(0 ..< l) + offset)[0..., .newAxis]
                mask = .array(causal)
            }
            output = MLXFast.scaledDotProductAttention(
                queries: qLatent, keys: latent, values: latent, scale: scale, mask: mask)
        case .masked(let allowed):
            output = MLXFast.scaledDotProductAttention(
                queries: qLatent, keys: latent, values: latent, scale: scale,
                mask: .array(allowed))
        case .gathered(let indices, let mask):
            let rows = indices.reshaped(b, 1, -1, 1)
            let selected = takeAlong(
                latent, broadcast(rows, to: [b, 1, indices.dim(1), kvLoraRank]), axis: 2)
            output = MLXFast.scaledDotProductAttention(
                queries: qLatent, keys: selected, values: selected, scale: scale,
                mask: .array(mask))
        }
        output = multiLinear(unembedOut, output)
        output = output.transposed(0, 2, 1, 3).reshaped(b, l, -1)
        return (oProj(output), selection)
    }
}

// MARK: - MLP and MoE

private func clampedSwiGLU(limit: Float?) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    guard let limit else { return compiledSiluProduct }
    return compile(shapeless: true) { gate, up in
        silu(minimum(gate, MLXArray(limit))) * clip(up, min: MLXArray(-limit), max: MLXArray(limit))
    }
}

final class GLM5NextMLP: Module, UnaryLayer {
    let activation: @Sendable (MLXArray, MLXArray) -> MLXArray

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int, swigluLimit: Float?) {
        activation = clampedSwiGLU(limit: swigluLimit)
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(activation(gateProj(x), upProj(x)))
    }
}

/// Stacked routed experts with the clamped SwiGLU.
final class GLM5NextSwitchGLU: Module {
    let activation: @Sendable (MLXArray, MLXArray) -> MLXArray

    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    init(inputDims: Int, hiddenDims: Int, numExperts: Int, swigluLimit: Float?) {
        activation = clampedSwiGLU(limit: swigluLimit)
        _gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: false)
        _upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: false)
        _downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: false)
        super.init()
    }

    /// Expert outputs `[..., K, D]` for `indices` `[..., K]`.
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        var idx = indices
        var inverseOrder: MLXArray? = nil
        let doSort = indices.size >= 64
        if doSort {
            var order = MLXArray()
            (x, idx, order) = gatherSort(x: x, indices: indices)
            inverseOrder = order
        }
        let hidden = activation(
            gateProj(x, idx, sortedIndices: doSort), upProj(x, idx, sortedIndices: doSort))
        x = downProj(hidden, idx, sortedIndices: doSort)
        if let inverseOrder {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }
        return MLX.squeezed(x, axis: -2)
    }
}

final class GLM5NextGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: GLM5NextConfiguration) {
        topK = config.numExpertsPerTok
        normTopkProb = config.normTopkProb
        routedScalingFactor = config.routedScalingFactor
        _weight.wrappedValue = zeros([config.nRoutedExperts, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = zeros([config.nRoutedExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = matmul(x.asType(.float32), weight.asType(.float32).T)
        return sigmoidRouterTopK(
            logits: logits, bias: eScoreCorrectionBias, k: topK, normalize: normTopkProb,
            scalingFactor: routedScalingFactor)
    }
}

final class GLM5NextMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: GLM5NextGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: GLM5NextSwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM5NextMLP?

    /// Set when the experts are sharded; the combined output is summed across it.
    var shardingGroup: MLXDistributed.Group?

    init(_ config: GLM5NextConfiguration) {
        _gate.wrappedValue = GLM5NextGate(config)
        _switchMLP.wrappedValue = GLM5NextSwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts, swigluLimit: config.swigluLimit)
        if config.nSharedExperts > 0 {
            _sharedExperts.wrappedValue = GLM5NextMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
                swigluLimit: config.swigluLimit)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        let routed = switchMLP(x, indices)
        var y = weightedExpertSum(routed, scores).asType(routed.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        if let shardingGroup {
            return MLXDistributed.allSum(y, group: shardingGroup)
        }
        return y
    }
}

// MARK: - Decoder

final class GLM5NextDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var attention: Module
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attentionHC: GLM5NextHyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: GLM5NextHyperConnection

    init(_ config: GLM5NextConfiguration, index: Int) {
        isLinear = config.linearLayers[index]
        _attention.wrappedValue =
            isLinear
            ? GLM5NextLinearAttention(config)
            : GLM5NextSparseAttention(config, hasIndexer: config.indexerTypes[index] != "shared")
        _mlp.wrappedValue =
            config.mlpLayerTypes[index] == "sparse"
            ? GLM5NextMoE(config)
            : GLM5NextMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize,
                swigluLimit: config.swigluLimit)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attentionHC.wrappedValue = GLM5NextHyperConnection(config)
        _ffnHC.wrappedValue = GLM5NextHyperConnection(config)
        super.init()
    }

    /// Advances the `[B, L, N, D]` residual streams. Returns the sparse selection this layer
    /// used, or `nil` for a linear-attention layer.
    func callAsFunction(
        _ streams: MLXArray, cache: KVCache?, ssmMask: MLXArray?,
        previousSelection: GLM5NextSelection?, mixer: GLM5NextHyperMixer,
        checkpointIndices: [Int]
    ) -> (MLXArray, GLM5NextSelection?) {
        var (post, comb, x) = attentionHC(streams, mixer: mixer)
        var selection: GLM5NextSelection? = nil
        let attentionOut: MLXArray
        switch attention {
        case let linear as GLM5NextLinearAttention:
            attentionOut = linear(
                inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache,
                checkpointIndices: checkpointIndices)
        case let sparse as GLM5NextSparseAttention:
            let (out, used) = sparse(
                inputLayerNorm(x), cache: cache, sharedSelection: previousSelection)
            attentionOut = out
            selection = used
        default:
            fatalError("unexpected attention module \(type(of: attention))")
        }
        var h = GLM5NextHyperConnection.expand(
            attentionOut, residual: streams, post: post, comb: comb)

        (post, comb, x) = ffnHC(h, mixer: mixer)
        let mlpOut = (mlp as! UnaryLayer)(postAttentionLayerNorm(x))
        h = GLM5NextHyperConnection.expand(mlpOut, residual: h, post: post, comb: comb)
        return (h, selection)
    }
}

public final class GLM5NextModelInner: Module {
    let mixer: GLM5NextHyperMixer
    let hcMult: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [GLM5NextDecoderLayer]
    let norm: RMSNorm

    init(_ config: GLM5NextConfiguration) {
        mixer = GLM5NextHyperMixer(config)
        hcMult = config.hcMult
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = (0 ..< config.hiddenLayers).map { GLM5NextDecoderLayer(config, index: $0) }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// Final normalized hidden states `[B, L, D]`.
    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, checkpointIndices: [Int] = []
    ) -> MLXArray {
        let h = embedTokens(inputs)
        var streams = broadcast(
            h.expandedDimensions(axis: 2), to: [h.dim(0), h.dim(1), hcMult, h.dim(2)])
        let linearIndex = layers.firstIndex { $0.isLinear }
        let ssmMask = linearIndex.flatMap {
            createSSMMask(h: h, cache: cache?[$0] as? MambaCache)
        }
        var selection: GLM5NextSelection? = nil
        for (i, layer) in layers.enumerated() {
            (streams, selection) = layer(
                streams, cache: cache?[i], ssmMask: ssmMask, previousSelection: selection,
                mixer: mixer, checkpointIndices: checkpointIndices)
        }
        return norm(streams.mean(axis: 2))
    }
}

public final class GLM5NextModel: Module, LLMModel {
    public let vocabularySize: Int
    public let model: GLM5NextModelInner
    let configuration: GLM5NextConfiguration
    public private(set) var tensorParallelGroup: MLXDistributed.Group?

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ configuration: GLM5NextConfiguration) {
        self.configuration = configuration
        vocabularySize = configuration.vocabularySize
        model = GLM5NextModelInner(configuration)
        _lmHead.wrappedValue = Linear(
            configuration.hiddenSize, configuration.vocabularySize, bias: false)
        super.init()
    }

    /// A recurrent cache per KDA layer and an unbounded latent cache per sparse layer. DSA
    /// selects over absolute positions, so the attention cache is never windowed.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        model.layers.map { $0.isLinear ? MambaCache() : KVCacheSimple() }
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
        let emit = state?[mtpEmitFlagKey] ?? false
        let checkpointOnly = state?[speculativeCheckpointOnlyKey] ?? false
        var checkpoints = state?[mtpCacheCheckpointIndicesKey] ?? []
        if let index = state?[mtpCacheCheckpointIndexKey] {
            checkpoints.append(index)
        }
        let hidden = model(input.tokens, cache: cache, checkpointIndices: checkpoints)
        let finalOnly =
            emit
            ? (state?[mtpFinalPrefillLogitsOnlyKey] ?? false)
            : lastTokenOnly && !checkpointOnly
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
        GLM5NextCheckpoint.excludedNamespaces(configuration, keepingMTP: false)
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        let mtpLayers =
            (configuration.hiddenLayers
            ..< configuration.hiddenLayers + max(configuration.numNextnPredictLayers, 1))
            .map { "model.layers.\($0)" }
        var checkpoint = try checkpoint.mapNames { name in
            guard let mapped = GLM5NextCheckpoint.canonicalName(name) else { return nil }
            if mtpLayers.contains(where: { mapped.hasPrefix($0 + ".") }) { return nil }
            return mapped
        }
        try GLM5NextCheckpoint.normalize(
            &checkpoint, configuration: configuration,
            layerPrefixes: (0 ..< configuration.hiddenLayers).map { "model.layers.\($0)" })
        if configuration.tieWordEmbeddings {
            checkpoint.weights["lm_head.weight"] = nil
        }
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(ModelCheckpoint(weights: weights)).weights
    }
}

extension GLM5NextModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}

extension GLM5NextModel {
    public var toolCallFormat: ToolCallFormat? { .glm4 }
}

/// Speculative verification checkpoints each KDA layer's state inside the verify pass and
/// rewinds it in place, like Qwen3.5's gated-delta layers.
extension GLM5NextModel: PromptLookupHybridModel {
    public var maximumNativeTargetCacheRewind: Int { 7 }
}

// MARK: - Checkpoint conversion

/// Converts Hugging Face (FP8 block-quantized) and mlx-community checkpoints to module names.
enum GLM5NextCheckpoint {
    /// Vision tower and MTP namespaces in source checkpoint naming.
    static func excludedNamespaces(_ configuration: GLM5NextConfiguration, keepingMTP: Bool)
        -> [String]
    {
        var namespaces = ["model.visual", "visual", "model.vision_tower", "vision_tower"]
        if !keepingMTP {
            let mtp = configuration.hiddenLayers
            for prefix in [
                "model.language_model.layers", "model.layers", "language_model.model.layers",
            ] {
                namespaces += (mtp ..< mtp + max(configuration.numNextnPredictLayers, 1)).map {
                    "\(prefix).\($0)"
                }
            }
        }
        return namespaces
    }

    /// Maps a source tensor name to `model.*` / `lm_head.*` naming, or `nil` to drop it.
    static func canonicalName(_ name: String) -> String? {
        var name = name
        for (source, target) in [
            ("model.language_model.", "model."), ("language_model.model.", "model."),
            ("language_model.lm_head.", "lm_head."), ("language_model.", "model."),
        ] where name.hasPrefix(source) {
            name = target + name.dropFirst(source.count)
            break
        }
        if name.hasPrefix("model.visual.") || name.hasPrefix("visual.")
            || name.contains("rotary_emb.inv_freq")
        {
            return nil
        }
        for (source, target) in [
            (".hc_attn_fn", ".attn_hc.fn"), (".hc_attn_base", ".attn_hc.base"),
            (".hc_attn_scale", ".attn_hc.scale"), (".hc_ffn_fn", ".ffn_hc.fn"),
            (".hc_ffn_base", ".ffn_hc.base"), (".hc_ffn_scale", ".ffn_hc.scale"),
        ] where name.hasSuffix(source) {
            return String(name.dropLast(source.count)) + target
        }
        return name
    }

    static func normalize(
        _ checkpoint: inout ModelCheckpoint, configuration: GLM5NextConfiguration,
        layerPrefixes: [String]
    ) throws {
        var weights = checkpoint.weights
        var quantization = checkpoint.perLayerQuantization

        for key in weights.keys where key.hasSuffix(".weight_scale_inv") {
            let weightKey = String(key.dropLast("_scale_inv".count))
            guard let scaleInv = weights.removeValue(forKey: key),
                let weight = weights[weightKey]
            else { continue }
            weights[weightKey] = GLMMoEDSACheckpoint.dequantizeFP8(weight, scaleInv: scaleInv)
        }

        for prefix in layerPrefixes {
            stackExperts(
                &weights, &quantization, prefix: "\(prefix).mlp",
                count: configuration.nRoutedExperts)
            try splitKVB(
                &weights, &quantization, prefix: "\(prefix).self_attn",
                configuration: configuration)
            fuseShortConvs(&weights, prefix: "\(prefix).self_attn")
            let hyperConnections = ["attn_hc", "ffn_hc"].flatMap { hc in
                ["fn", "base", "scale"].map { "\(prefix).\(hc).\($0)" }
            }
            let float32Keys = [
                "\(prefix).mlp.gate.weight", "\(prefix).mlp.gate.e_score_correction_bias",
                "\(prefix).self_attn.A_log", "\(prefix).self_attn.dt_bias",
            ]
            for key in float32Keys + hyperConnections {
                weights[key] = weights[key]?.asType(.float32)
            }
        }

        // Some conversions store float16 scales for a bfloat16 model. Matching the activation
        // dtype keeps quantized matmuls from promoting every activation to float32.
        let modelType = weights["model.embed_tokens.weight"]?.dtype ?? .bfloat16
        if modelType == .bfloat16 {
            for (key, value) in weights
            where value.dtype == .float16 && (key.hasSuffix(".scales") || key.hasSuffix(".biases"))
            {
                weights[key] = value.asType(.bfloat16)
            }
        }
        checkpoint.weights = weights
        checkpoint.perLayerQuantization = quantization
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
            if let option = quantization?.perLayerQuantization["\(prefix).experts.0.\(proj)"],
                quantization?.perLayerQuantization["\(prefix).switch_mlp.\(proj)"] == nil
            {
                quantization?.perLayerQuantization["\(prefix).switch_mlp.\(proj)"] = option
            }
        }
    }

    /// Concatenates `q_conv1d`, `k_conv1d`, `v_conv1d` into one depthwise `[3C, K, 1]` weight.
    private static func fuseShortConvs(_ weights: inout [String: MLXArray], prefix: String) {
        func depthwise(_ w: MLXArray) -> MLXArray {
            switch w.ndim {
            case 2: w.expandedDimensions(axis: -1)
            case 3 where w.dim(1) == 1: w.swappedAxes(1, 2)
            default: w
            }
        }
        let names = ["q_conv1d", "k_conv1d", "v_conv1d"].map { "\(prefix).\($0).weight" }
        if names.allSatisfy({ weights[$0] != nil }) {
            weights["\(prefix).conv1d.weight"] = concatenated(
                names.map { depthwise(weights.removeValue(forKey: $0)!) }, axis: 0)
        } else if let fused = weights["\(prefix).conv1d.weight"] {
            weights["\(prefix).conv1d.weight"] = depthwise(fused)
        }
    }

    private static func splitKVB(
        _ weights: inout [String: MLXArray],
        _ quantization: inout BaseConfiguration.PerLayerQuantization?,
        prefix: String, configuration: GLM5NextConfiguration
    ) throws {
        guard var w = weights.removeValue(forKey: "\(prefix).kv_b_proj.weight") else { return }
        var option: BaseConfiguration.Quantization? = nil
        if let scales = weights.removeValue(forKey: "\(prefix).kv_b_proj.scales") {
            let biases = weights.removeValue(forKey: "\(prefix).kv_b_proj.biases")
            if let declared = quantization?.quantization(layer: "\(prefix).kv_b_proj") {
                option = declared
            } else if biases != nil {
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

// MARK: - Tensor parallelism

extension GLM5NextLinearAttention {
    /// Shards Kimi Delta Attention by heads. Each rank keeps its heads' projections, short-conv
    /// channels, decay, gate and recurrent state; `o_proj` sums across `group`.
    func shard(group: MLXDistributed.Group) throws {
        let size = group.size
        guard heads % size == 0 else {
            throw TensorParallelError.indivisible(
                name: "linear attention heads", dimension: heads, groupSize: size)
        }
        try shardLinears(
            in: self, group: group,
            [
                "q_proj": .allToSharded, "k_proj": .allToSharded, "v_proj": .allToSharded,
                "f_b_proj": .allToSharded, "b_proj": .allToSharded, "g_b_proj": .allToSharded,
                "o_proj": .shardedToAll,
            ])
        // The depthwise conv runs over [q | k | v] channels, so keep this rank's third of each.
        try shardInPlace(conv1d, predicate: { _, _ in (0, .count(3)) }, group: group)
        try shardInPlace(
            self,
            predicate: { path, _ in
                path == "A_log" || path == "dt_bias" ? (0, .count(1)) : nil
            }, group: group)
        heads /= size
        projectionDim /= size
    }
}

extension GLM5NextSparseAttention {
    /// Shards absorbed MLA by heads. The query and KV latents and the indexer stay whole, so
    /// every rank selects the same keys; `o_proj` sums across `group`.
    func shard(group: MLXDistributed.Group) throws {
        let size = group.size
        guard numHeads % size == 0 else {
            throw TensorParallelError.indivisible(
                name: "attention heads", dimension: numHeads, groupSize: size)
        }
        try shardLinears(
            in: self, group: group, ["q_b_proj": .allToSharded, "o_proj": .shardedToAll])
        let byHead: ShardingPredicate = { _, _ in (0, .count(1)) }
        try shardInPlace(embedQ, predicate: byHead, group: group)
        try shardInPlace(unembedOut, predicate: byHead, group: group)
        numHeads /= size
    }
}

extension GLM5NextMoE {
    /// Shards every routed and shared expert in place; the block sums its output once.
    func shard(group: MLXDistributed.Group) throws {
        try switchMLP.gateProj.shard(.allToSharded, group: group)
        try switchMLP.upProj.shard(.allToSharded, group: group)
        try switchMLP.downProj.shard(.shardedToAll, group: group)
        if let shared = sharedExperts {
            try shardInPlace(shared.gateProj, sharding: .allToSharded, group: group)
            try shardInPlace(shared.upProj, sharding: .allToSharded, group: group)
            try shardInPlace(shared.downProj, sharding: .shardedToAll, group: group)
        }
        shardingGroup = group
    }
}

extension GLM5NextDecoderLayer {
    /// Hyper-connections, norms and the router stay whole. Attention and the MLP each sum
    /// across `group` once.
    func shard(group: MLXDistributed.Group) throws {
        switch attention {
        case let linear as GLM5NextLinearAttention: try linear.shard(group: group)
        case let sparse as GLM5NextSparseAttention: try sparse.shard(group: group)
        default:
            throw TensorParallelError.modelNotShardable(String(describing: type(of: attention)))
        }
        switch mlp {
        case let moe as GLM5NextMoE:
            try moe.shard(group: group)
        case let dense as GLM5NextMLP:
            try shardLinears(
                in: dense, group: group,
                [
                    "gate_proj": .allToSharded, "up_proj": .allToSharded,
                    "down_proj": .shardedToAll,
                ])
        default:
            throw TensorParallelError.modelNotShardable(String(describing: type(of: mlp)))
        }
    }
}

extension GLM5NextModel: ShardableModel {
    /// Embeddings, norms and the LM head stay whole on every rank.
    public func shard(group: MLXDistributed.Group) throws {
        for layer in model.layers {
            try layer.shard(group: group)
        }
        tensorParallelGroup = group
    }
}
