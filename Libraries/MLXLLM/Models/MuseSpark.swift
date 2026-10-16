// Copyright © 2026 Apple Inc.

// Text model of Meta's Muse Spark Open. Port of mlx-lm's `muse_spark_open`, with the
// transformers `muse_spark` names and config as well.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Muse Spark Open text configuration.
///
/// Decodes Meta's reference config (`muse_spark_open`), the transformers config
/// (`muse_spark` with `text_config`), and a flat transformers text config (`muse_spark_text`).
public struct MuseSparkConfiguration: Codable, Sendable, ModelConfigurationValidating {
    public var modelType: String
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var vocabularySize: Int
    public var rmsNormEps: Float
    public var postNormEps: Float
    /// RoPE base per layer; 0 marks a NoPE layer.
    public var layerRopeTheta: [Float]
    public var layerTypes: [String]
    public var slidingWindow: Int?
    public var softCap: Float
    /// Scales the router logits and the output logits.
    public var outputMultiplier: Float
    public var embeddingMultiplier: Float
    /// Applied to the normed queries, on top of `1 / sqrt(head_dim)`.
    public var qkScaleFactor: Float
    public var residualGateTemperature: Float
    public var numExperts: Int
    public var numExpertsPerToken: Int
    /// Width of the latent the experts run in.
    public var moeHiddenSize: Int
    public var moeIntermediateSize: Int
    public var routedScalingFactor: Float
    public var routerScoreEps: Float
    public var useFP32ForOutput: Bool
    var unsupported: [String] = []

    enum FlatKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case postNormEps = "post_norm_eps"
        case ropeParameters = "rope_parameters"
        case ropeTheta = "rope_theta"
        case layerRopeTheta = "layer_rope_theta"
        case layerTypes = "layer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case slidingWindow = "sliding_window"
        case softCap = "final_logit_softcapping"
        case outputMultiplier = "output_multiplier"
        case embeddingMultiplier = "embedding_multiplier"
        case qkScaleFactor = "qk_scale_factor"
        case residualGateTemperature = "residual_gate_temperature"
        case numExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case moeHiddenSize = "moe_hidden_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case routedScalingFactor = "routed_scaling_factor"
        case routerScoreEps = "router_score_eps"
        case useFP32ForOutput = "use_fp32_for_output"
        // Meta's reference config.
        case rmsPostNormEps = "rms_post_norm_eps"
        case everyNLayersNope = "every_n_layers_nope"
        case globalAttnCfg = "global_attn_cfg"
        case outputSoftCapTemp = "output_soft_cap_temp"
        case ffnDimMultiplier = "ffn_dim_multiplier"
        case moe
        case metap
        case numericsArgs = "numerics_args"
    }

    private struct RopeParameters: Codable {
        var ropeTheta: Float?
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
    }

    private struct MetaMoE: Codable {
        var isEnabled: Bool?
        var numExperts: Int
        var topk: Int
        var routeScale: Float?
        var routerScoreDivisionEps: Float?
        var routerInFP32: Bool?
        var compressionFactor: Int?
        var overrideRoutedHiddenDim: Int?
        var nDenseLayersBeforeMoE: Int?
        var interleaveMoELayerStep: Int?
        var useSharedExpert: Bool?

        enum CodingKeys: String, CodingKey {
            case isEnabled = "is_enabled"
            case numExperts = "num_experts"
            case topk
            case routeScale = "route_scale"
            case routerScoreDivisionEps = "router_score_division_eps"
            case routerInFP32 = "router_in_fp32"
            case compressionFactor = "compression_factor"
            case overrideRoutedHiddenDim = "override_routed_hidden_dim"
            case nDenseLayersBeforeMoE = "n_dense_layers_before_moe"
            case interleaveMoELayerStep = "interleave_moe_layer_step"
            case useSharedExpert = "use_shared_expert"
        }
    }

    private struct MetaMetaP: Codable {
        var useMetaP: Bool?
        var routerHasOneInfDim: Bool?
        var baseWidth: Float?
        var metapMode: String?
        var mEmb: Float?

        enum CodingKeys: String, CodingKey {
            case useMetaP = "use_metap"
            case routerHasOneInfDim = "router_has_one_inf_dim"
            case baseWidth = "base_width"
            case metapMode = "metap_mode"
            case mEmb = "m_emb"
        }
    }

    private struct MetaNumerics: Codable {
        var useFP32ForOutput: Bool?
        var outDTypePerModuleOverrides: String?

        enum CodingKeys: String, CodingKey {
            case useFP32ForOutput = "use_fp32_for_output"
            case outDTypePerModuleOverrides = "out_dtype_per_module_overrides"
        }
    }

    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: FlatKeys.self)
        let modelType = try root.decodeIfPresent(String.self, forKey: .modelType) ?? "muse_spark"
        if root.contains(.everyNLayersNope) {
            try self.init(meta: root, modelType: modelType)
        } else if root.contains(.textConfig) {
            let text = try root.nestedContainer(keyedBy: FlatKeys.self, forKey: .textConfig)
            try self.init(transformers: text, modelType: modelType)
        } else {
            try self.init(transformers: root, modelType: modelType)
        }
    }

    private init(transformers c: KeyedDecodingContainer<FlatKeys>, modelType: String) throws {
        self.modelType = modelType
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / attentionHeads
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        postNormEps = try c.decodeIfPresent(Float.self, forKey: .postNormEps) ?? 1e-8

        let ropeTheta =
            try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)?.ropeTheta
            ?? c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        let layers = hiddenLayers
        // As transformers: every 4th layer from the last is full attention without RoPE.
        let isNoPE = { (i: Int) in (layers - 1 - i) % 4 == 0 }
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< layers).map { isNoPE($0) ? "full_attention" : "sliding_attention" }
        layerRopeTheta =
            try c.decodeIfPresent([Float].self, forKey: .layerRopeTheta)
            ?? (0 ..< layers).map { isNoPE($0) ? 0 : ropeTheta }
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 2048
        softCap = try c.decodeIfPresent(Float.self, forKey: .softCap) ?? 20
        outputMultiplier = try c.decodeIfPresent(Float.self, forKey: .outputMultiplier) ?? 1
        embeddingMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .embeddingMultiplier) ?? 1
        qkScaleFactor =
            try c.decodeIfPresent(Float.self, forKey: .qkScaleFactor)
            ?? Float(43.7840518911 / Double(headDim).squareRoot())
        residualGateTemperature =
            try c.decodeIfPresent(Float.self, forKey: .residualGateTemperature) ?? 0.3
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        numExpertsPerToken = try c.decode(Int.self, forKey: .numExpertsPerToken)
        moeHiddenSize = try c.decodeIfPresent(Int.self, forKey: .moeHiddenSize) ?? hiddenSize
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        routerScoreEps = try c.decodeIfPresent(Float.self, forKey: .routerScoreEps) ?? 1e-15
        useFP32ForOutput = try c.decodeIfPresent(Bool.self, forKey: .useFP32ForOutput) ?? false

        if let mlp = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes),
            mlp.contains(where: { $0 != "sparse" })
        {
            unsupported.append("dense feed-forward layers")
        }
        if let shared = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts), shared != 0 {
            unsupported.append("shared experts")
        }
    }

    private init(meta c: KeyedDecodingContainer<FlatKeys>, modelType: String) throws {
        self.modelType = modelType
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        postNormEps = try c.decode(Float.self, forKey: .rmsPostNormEps)
        softCap = try c.decode(Float.self, forKey: .outputSoftCapTemp)
        qkScaleFactor = Float(43.7840518911 / Double(headDim).squareRoot())
        residualGateTemperature = 0.3

        // As mlx-lm's layer_schedule: NoPE every n layers counted from the last, and the
        // window pattern aligned so that it ends on the last layer.
        let ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        let everyN = try c.decode(Int.self, forKey: .everyNLayersNope)
        let windows = try c.decode([Int].self, forKey: .globalAttnCfg)
        let layers = hiddenLayers
        let offset = everyN - (layers % everyN)
        layerRopeTheta = (0 ..< layers).map { (layers - $0 - 1) % everyN != 0 ? ropeTheta : 0 }
        let layerWindows = (0 ..< layers).map { windows[($0 + offset) % windows.count] }
        layerTypes = layerWindows.map { $0 > 0 ? "sliding_attention" : "full_attention" }
        let distinct = Set(layerWindows.filter { $0 > 0 })
        if distinct.count > 1 {
            unsupported.append("several sliding window sizes \(distinct.sorted())")
        }
        slidingWindow = distinct.first

        let moe = try c.decode(MetaMoE.self, forKey: .moe)
        numExperts = moe.numExperts
        numExpertsPerToken = moe.topk
        routedScalingFactor = moe.routeScale ?? 1
        routerScoreEps = moe.routerScoreDivisionEps ?? 0
        moeHiddenSize = hiddenSize / (moe.compressionFactor ?? 1)
        let ffnMultiplier = try c.decodeIfPresent(Float.self, forKey: .ffnDimMultiplier) ?? 2
        moeIntermediateSize =
            moe.overrideRoutedHiddenDim ?? Int(Float(hiddenSize) * ffnMultiplier)
        if moe.isEnabled != true || (moe.nDenseLayersBeforeMoE ?? 0) != 0
            || (moe.interleaveMoELayerStep ?? 1) != 1
        {
            unsupported.append("dense feed-forward layers")
        }
        if moe.useSharedExpert == true {
            unsupported.append("shared experts")
        }
        if moe.routerInFP32 != true {
            unsupported.append("a router that is not fp32")
        }

        let metap = try c.decodeIfPresent(MetaMetaP.self, forKey: .metap)
        let useMetaP = metap?.useMetaP ?? false
        if useMetaP, metap?.routerHasOneInfDim == true {
            let widthMultiplier = Float(hiddenSize) / (metap?.baseWidth ?? 256)
            outputMultiplier =
                metap?.metapMode == "mup" ? 1 / widthMultiplier : 1 / widthMultiplier.squareRoot()
        } else {
            outputMultiplier = 1
        }
        embeddingMultiplier = useMetaP ? (metap?.mEmb ?? 1) : 1

        let numerics = try c.decodeIfPresent(MetaNumerics.self, forKey: .numericsArgs)
        let overrides = (numerics?.outDTypePerModuleOverrides ?? "").split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces).split(separator: ":") }
        useFP32ForOutput =
            (try c.decodeIfPresent(Bool.self, forKey: .useFP32ForOutput) ?? false)
            || numerics?.useFP32ForOutput == true
            || overrides.contains {
                $0.count == 3 && $0[0] == "output" && $0[2].lowercased() == "fp32"
            }
    }

    /// Encodes the flat transformers text config, which decodes back to the same values.
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: FlatKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(postNormEps, forKey: .postNormEps)
        try c.encode(layerRopeTheta, forKey: .layerRopeTheta)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encodeIfPresent(slidingWindow, forKey: .slidingWindow)
        try c.encode(softCap, forKey: .softCap)
        try c.encode(outputMultiplier, forKey: .outputMultiplier)
        try c.encode(embeddingMultiplier, forKey: .embeddingMultiplier)
        try c.encode(qkScaleFactor, forKey: .qkScaleFactor)
        try c.encode(residualGateTemperature, forKey: .residualGateTemperature)
        try c.encode(numExperts, forKey: .numExperts)
        try c.encode(numExpertsPerToken, forKey: .numExpertsPerToken)
        try c.encode(moeHiddenSize, forKey: .moeHiddenSize)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(routedScalingFactor, forKey: .routedScalingFactor)
        try c.encode(routerScoreEps, forKey: .routerScoreEps)
        try c.encode(useFP32ForOutput, forKey: .useFP32ForOutput)
    }

    public func validateModelConfiguration() throws {
        if let first = unsupported.first {
            throw MuseSparkError.unsupportedConfiguration(first)
        }
        guard layerTypes.count == hiddenLayers, layerRopeTheta.count == hiddenLayers else {
            throw MuseSparkError.unsupportedConfiguration(
                "layer_types and layer_rope_theta must list every layer")
        }
        if layerTypes.contains("sliding_attention"), slidingWindow == nil {
            throw MuseSparkError.unsupportedConfiguration("sliding layers without a window")
        }
    }
}

public enum MuseSparkError: LocalizedError {
    case unsupportedConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedConfiguration(let detail):
            "Muse Spark: unsupported configuration: \(detail)."
        }
    }
}

// MARK: - Numerics

// The reference runs the residual stream in fp32 and the branches in bf16.
private let museComputeType = DType.bfloat16

/// `silu(gate) * up` in fp32, rounded once, as mlx-lm's `precise_swiglu`.
private let museSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) {
    gate, up in
    (MLXNN.silu(gate.asType(.float32)) * up.asType(.float32)).asType(up.dtype)
}

/// Weights of the stream and of the branch for a fixed residual gate.
func museResidualGateCoefficients(_ gate: MLXArray, temperature: Float) -> (MLXArray, MLXArray) {
    let w = gate.asType(.float32) / temperature
    let beta = sigmoid(w)
    let alpha = sqrt(maximum(sigmoid(-w) * (1 + beta), MLXArray(Float(1e-3))))
    return (alpha, beta)
}

/// Weightless RMS norm in fp32 of `x`.
private func museRMSNormFloat(_ x: MLXArray, eps: Float) -> MLXArray {
    let f = x.asType(.float32)
    return f * rsqrt(mean(f * f, axis: -1, keepDims: true) + eps)
}

/// Post-norms the branch and adds it to the fp32 stream with the gate weights.
///
/// The optional `postWeight` scales the branch before the norm, rounded to its dtype.
private func museNormAdd(
    branch: MLXArray, stream: MLXArray, alpha: MLXArray, beta: MLXArray,
    postWeight: MLXArray?, eps: Float
) -> MLXArray {
    var v = branch
    if let postWeight {
        v = postWeight.asType(branch.dtype) * branch
    }
    let normed = MLXFast.rmsNorm(v, weight: MLXArray.mlxNone, eps: eps)
    return alpha * stream + beta * normed.asType(.float32)
}

// MARK: - Attention

final class MuseSparkAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let queryScale: Float
    let eps: Float
    let rope: RoPELayer?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_gate_proj") var gateProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: MuseSparkConfiguration, layer: Int) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        queryScale = config.qkScaleFactor
        eps = config.rmsNormEps

        let dim = config.hiddenSize
        _qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _gateProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, dim, bias: false)

        // The weights hold Q and K in the interleaved-pair RoPE layout.
        let theta = config.layerRopeTheta[layer]
        rope =
            theta == 0
            ? nil
            : initializeRope(
                dims: headDim, base: theta, traditional: true, scalingConfig: nil,
                maxPositionEmbeddings: nil)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))

        // Weightless QK norm; queries also take the tuned scale, rounded once.
        var queries = MLXFast.rmsNorm(
            qProj(x).reshaped(b, l, heads, headDim), weight: MLXArray.mlxNone, eps: eps)
        queries = (queries.asType(.float32) * queryScale).asType(queries.dtype)
        var keys = MLXFast.rmsNorm(
            kProj(x).reshaped(b, l, kvHeads, headDim), weight: MLXArray.mlxNone, eps: eps)
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3)

        // Per-head output norm times the sigmoid gate, in fp32 and rounded once.
        let gate = gateProj(x).reshaped(b, l, heads, headDim)
        let gated = museRMSNormFloat(output, eps: eps) * sigmoid(gate.asType(.float32))
        return oProj(gated.asType(output.dtype).reshaped(b, l, -1))
    }
}

// MARK: - MoE

final class MuseSparkMoE: Module {
    let topK: Int
    let routerMultiplier: Float
    let routeScale: Float
    let scoreEps: Float
    let postExpertEps: Float

    /// Router weight `[experts, hidden]`, kept in fp32.
    @ParameterInfo(key: "router") var router: MLXArray
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray

    @ModuleInfo var pooler: Linear
    @ModuleInfo(key: "pre_expert_norm") var preExpertNorm: RMSNorm
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "post_expert_norm") var postExpertNorm: RMSNorm
    @ModuleInfo var unpooler: Linear

    init(_ config: MuseSparkConfiguration) {
        topK = config.numExpertsPerToken
        routerMultiplier = config.outputMultiplier
        routeScale = config.routedScalingFactor
        scoreEps = config.routerScoreEps
        postExpertEps = config.postNormEps

        let dim = config.hiddenSize
        let latent = config.moeHiddenSize
        _router.wrappedValue = MLXArray.zeros([config.numExperts, dim])
        _expertBias.wrappedValue = MLXArray.zeros([config.numExperts])
        _pooler.wrappedValue = Linear(dim, latent, bias: false)
        _preExpertNorm.wrappedValue = RMSNorm(dimensions: latent, eps: config.rmsNormEps)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: latent, hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts, activationProduct: museSwiGLU)
        // The weight scales each expert output before its norm.
        _postExpertNorm.wrappedValue = RMSNorm(dimensions: latent, eps: config.postNormEps)
        _unpooler.wrappedValue = Linear(latent, dim, bias: false)
        super.init()
    }

    /// Experts `[..., k]` and their weights, ordered by biased score as mlx-lm's kernel.
    func route(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = matmul(x.asType(.float32), router.asType(.float32).T)
        let scores = sigmoid(logits * routerMultiplier)
        let biased = scores + expertBias.asType(.float32)
        var indices = argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let order = argSort(-takeAlong(biased, indices, axis: -1), axis: -1)
        indices = takeAlong(indices, order, axis: -1)
        let selected = takeAlong(scores, indices, axis: -1)
        let weights = selected / (selected.sum(axis: -1, keepDims: true) + scoreEps) * routeScale
        return (indices, weights)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = route(x)
        let h = switchMLP(preExpertNorm(pooler(x)), indices)

        // Norm each expert output, then add them in order, rounding after each add.
        let v = h.asType(.float32) * postExpertNorm.weight.asType(.float32)
        let inv = rsqrt(mean(v * v, axis: -1, keepDims: true) + postExpertEps)
        let roundedWeights = weights.asType(h.dtype).asType(.float32)
        let terms = v * inv * expandedDimensions(roundedWeights, axis: -1)
        var combined = terms[.ellipsis, 0, 0...].asType(h.dtype)
        for k in 1 ..< topK {
            combined = (combined.asType(.float32) + terms[.ellipsis, k, 0...]).asType(h.dtype)
        }
        return unpooler(combined)
    }
}

// MARK: - Decoder

final class MuseSparkDecoderLayer: Module {
    let usesSliding: Bool
    let postNormEps: Float

    @ModuleInfo(key: "pre_attn_norm") var preAttnNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttention: MuseSparkAttention
    @ModuleInfo(key: "pre_ffn_norm") var preFFNNorm: RMSNorm
    @ModuleInfo(key: "feed_forward") var feedForward: MuseSparkMoE
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm

    @ParameterInfo(key: "add_attn_alpha") var addAttnAlpha: MLXArray
    @ParameterInfo(key: "add_attn_beta") var addAttnBeta: MLXArray
    @ParameterInfo(key: "add_ffn_alpha") var addFFNAlpha: MLXArray
    @ParameterInfo(key: "add_ffn_beta") var addFFNBeta: MLXArray

    init(_ config: MuseSparkConfiguration, layer: Int) {
        usesSliding = config.layerTypes[layer] == "sliding_attention"
        postNormEps = config.postNormEps
        let dim = config.hiddenSize
        _preAttnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: config.rmsNormEps)
        _selfAttention.wrappedValue = MuseSparkAttention(config, layer: layer)
        _preFFNNorm.wrappedValue = RMSNorm(dimensions: dim, eps: config.rmsNormEps)
        _feedForward.wrappedValue = MuseSparkMoE(config)
        _postFFNNorm.wrappedValue = RMSNorm(dimensions: dim, eps: config.postNormEps)

        // Loading updates parameters in place, so each needs its own array.
        func coefficients() -> (MLXArray, MLXArray) {
            museResidualGateCoefficients(
                MLXArray.zeros([dim]), temperature: config.residualGateTemperature)
        }
        (_addAttnAlpha.wrappedValue, _addAttnBeta.wrappedValue) = coefficients()
        (_addFFNAlpha.wrappedValue, _addFFNBeta.wrappedValue) = coefficients()
        super.init()
    }

    /// `h` is the fp32 residual stream.
    func callAsFunction(
        _ h: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let attentionInput = preAttnNorm(h).asType(museComputeType)
        var r = selfAttention(attentionInput, mask: mask, cache: cache)
        let h = museNormAdd(
            branch: r, stream: h, alpha: addAttnAlpha, beta: addAttnBeta, postWeight: nil,
            eps: postNormEps)
        r = feedForward(preFFNNorm(h).asType(r.dtype))
        return museNormAdd(
            branch: r, stream: h, alpha: addFFNAlpha, beta: addFFNBeta,
            postWeight: postFFNNorm.weight, eps: postFFNNorm.eps)
    }
}

final class MuseSparkModelInner: Module {
    let config: MuseSparkConfiguration
    let fullAttentionIndex: Int
    let slidingAttentionIndex: Int?

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MuseSparkDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: MuseSparkConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = (0 ..< config.hiddenLayers).map { MuseSparkDecoderLayer(config, layer: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        fullAttentionIndex = layers.firstIndex { !$0.usesSliding } ?? 0
        slidingAttentionIndex = layers.firstIndex { $0.usesSliding }
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = MLXFast.rmsNorm(
            embedTokens(inputs), weight: MLXArray.mlxNone, eps: config.rmsNormEps)
        if config.embeddingMultiplier != 1 {
            h = h * config.embeddingMultiplier
        }
        h = h.asType(.float32)

        let fullMask = createAttentionMask(h: h, cache: cache?[fullAttentionIndex])
        var slidingMask = fullMask
        if let slidingAttentionIndex {
            slidingMask = createAttentionMask(
                h: h, cache: cache?[slidingAttentionIndex], windowSize: config.slidingWindow)
        }
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: layer.usesSliding ? slidingMask : fullMask, cache: cache?[i])
        }
        return norm(h).asType(museComputeType)
    }
}

// MARK: - Model

public final class MuseSparkModel: Module, LLMModel, KVCacheDimensionProvider,
    WeightTensorSelecting
{
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: MuseSparkConfiguration

    @ModuleInfo(key: "model") var model: MuseSparkModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ configuration: MuseSparkConfiguration) {
        self.configuration = configuration
        vocabularySize = configuration.vocabularySize
        kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        _model.wrappedValue = MuseSparkModelInner(configuration)
        _lmHead.wrappedValue = Linear(
            configuration.hiddenSize, configuration.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        // An fp32 head (use_fp32_for_output) takes an fp32 input.
        if !(lmHead is Quantized) {
            out = out.asType(lmHead.weight.dtype)
        }
        let logits = lmHead(out).asType(.float32) * configuration.outputMultiplier
        let cap = configuration.softCap
        return tanh(logits / cap) * cap
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layers.map {
            try makeHybridAttentionKVCache(
                parameters: parameters, slidingWindow: configuration.slidingWindow,
                usesSlidingWindow: $0.usesSliding)
        }
    }

    public var loraLayers: [Module] { model.layers }

    // MARK: Checkpoints

    public func shouldLoadWeightTensor(named name: String) -> Bool {
        MuseSparkCheckpoint.mapName(name) != nil
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        let checkpoint = try MuseSparkCheckpoint.prepare(checkpoint, configuration: configuration)
        let perLayer = checkpoint.perLayerQuantization
        quantizeGlobalScaleSwitchLinears(in: self, weights: checkpoint.weights) { path in
            if let q = perLayer?.quantization(layer: path), q.mode == .nvfp4 {
                return q.asTuple
            }
            return (16, 4, .nvfp4)
        }
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try MuseSparkCheckpoint.prepare(
            ModelCheckpoint(weights: weights), configuration: configuration
        ).weights
    }
}

// MARK: - Checkpoint names

/// Maps Meta's reference names, the transformers names, and mlx-lm's converted names to
/// this model's names, and normalizes the values.
enum MuseSparkCheckpoint {
    private static let textPrefixes = [
        "model.embed_tokens", "model.layers.", "model.norm", "lm_head",
    ]

    private static let droppedLeaves: Set<String> = [
        "w13_input_scale", "w2_input_scale", "gate_up_proj_input_scale", "down_proj_input_scale",
        "q_scale", "k_scale", "v_scale", "num_tokens_per_expert", "inv_freq",
    ]

    // Applied in order to the name with a trailing dot.
    private static let renames: [(String, String)] = [
        (".mlp.experts.gate_up_proj_weight_scale_2.", ".feed_forward.experts.w13_weight_scale_2."),
        (".mlp.experts.gate_up_proj_weight_scale.", ".feed_forward.experts.w13_weight_scale."),
        (".mlp.experts.gate_up_proj.", ".feed_forward.experts.w13_weight."),
        (".mlp.experts.down_proj_weight_scale_2.", ".feed_forward.experts.w2_weight_scale_2."),
        (".mlp.experts.down_proj_weight_scale.", ".feed_forward.experts.w2_weight_scale."),
        (".mlp.experts.down_proj.", ".feed_forward.experts.w2_weight."),
        (".mlp.experts.post_expert_norm.", ".feed_forward.post_expert_norm."),
        (".mlp.gate.weight.", ".feed_forward.router."),
        (".mlp.gate.e_score_correction_bias.", ".feed_forward.expert_bias."),
        (".mlp.pre_expert_proj.", ".feed_forward.pooler."),
        (".mlp.post_expert_proj.", ".feed_forward.unpooler."),
        (".mlp.", ".feed_forward."),
        (".input_layernorm.", ".pre_attn_norm."),
        (".self_attn.pre_qkv_proj_norm.", ".pre_attn_norm."),
        (".self_attn.gate_proj.", ".self_attn.o_gate_proj."),
        (".pre_feedforward_layernorm.", ".pre_ffn_norm."),
        (".post_feedforward_layernorm.", ".post_ffn_norm."),
        (".post_attention_residual_gate.gate.", ".add_attn.gate."),
        (".post_feedforward_residual_gate.gate.", ".add_ffn.gate."),
        (".add_attn_gate.", ".add_attn.gate."),
        (".add_ffn_gate.", ".add_ffn.gate."),
    ]

    /// This model's name for a tensor or layer, or `nil` for one the text model does not use.
    static func mapName(_ name: String) -> String? {
        var name = name
        for (prefix, replacement) in [
            ("language_model.model.", "model."), ("model.language_model.", "model."),
            ("language_model.", ""),
        ] where name.hasPrefix(prefix) {
            name = replacement + name.dropFirst(prefix.count)
            break
        }
        guard textPrefixes.contains(where: { name.hasPrefix($0) }),
            let leaf = name.split(separator: ".").last, !droppedLeaves.contains(String(leaf)),
            !name.hasSuffix("embed_norm.weight")
        else { return nil }

        var dotted = name + "."
        for (from, to) in renames where dotted.contains(from) {
            dotted = dotted.replacingOccurrences(of: from, with: to)
        }
        return String(dotted.dropLast())
    }

    static func prepare(
        _ checkpoint: ModelCheckpoint, configuration: MuseSparkConfiguration
    ) throws -> ModelCheckpoint {
        let names = checkpoint.weights.keys
        // Checkpoints in this model's layout already hold norm gains and gate weights.
        let converted = names.contains { $0.hasSuffix(".feed_forward.router") }
        // The transformers conversion permutes Q and K to the split-half RoPE layout.
        let splitHalfRoPE =
            !converted
            && names.contains { $0.contains(".input_layernorm.") || $0.contains(".mlp.") }

        var checkpoint = try checkpoint.mapNames(mapName)
        if var perLayer = checkpoint.perLayerQuantization {
            for key in perLayer.perLayerQuantization.keys
            where key.hasSuffix(".self_attn.qkvg_proj") {
                let attention = key.dropLast("qkvg_proj".count)
                let option = perLayer.perLayerQuantization.removeValue(forKey: key)
                for projection in ["q_proj", "k_proj", "v_proj", "o_gate_proj"] {
                    perLayer.perLayerQuantization[attention + projection] = option
                }
            }
            checkpoint.perLayerQuantization = perLayer
        }
        checkpoint.weights = try sanitize(
            checkpoint.weights, configuration: configuration, converted: converted,
            splitHalfRoPE: splitHalfRoPE)
        return checkpoint
    }

    private static let zeroCenteredNorms = [
        ".pre_attn_norm.weight", ".pre_ffn_norm.weight", ".post_ffn_norm.weight",
        ".feed_forward.pre_expert_norm.weight", ".feed_forward.post_expert_norm.weight",
    ]

    /// The e4m3 and e2m1 maxima: Meta stores the nvfp4 global scale divided by 6 * 448.
    private static let nvfp4ScaleToGlobal: Float = 6 * 448

    private static func sanitize(
        _ weights: [String: MLXArray], configuration c: MuseSparkConfiguration,
        converted: Bool, splitHalfRoPE: Bool
    ) throws -> [String: MLXArray] {
        var out = [String: MLXArray]()
        func packed(_ v: MLXArray) -> MLXArray { v.dtype == .uint8 ? v.view(dtype: .uint32) : v }
        func blockScales(_ v: MLXArray) -> MLXArray {
            v.dtype == .uint8 ? v : v.view(dtype: .uint8)
        }

        for (key, value) in weights {
            guard let range = key.range(of: #"^model\.layers\.\d+\."#, options: .regularExpression)
            else {
                out[key] = value
                continue
            }
            let prefix = String(key[range])
            let suffix = String(key[range.upperBound...])
            let experts = prefix + "feed_forward.switch_mlp."

            switch suffix {
            case "add_attn.gate", "add_ffn.gate":
                let name = suffix == "add_attn.gate" ? "add_attn" : "add_ffn"
                let (alpha, beta) = museResidualGateCoefficients(
                    value, temperature: c.residualGateTemperature)
                out[prefix + name + "_alpha"] = alpha
                out[prefix + name + "_beta"] = beta
            case "feed_forward.router.weight":
                // Meta stores the router as [hidden, experts].
                out[prefix + "feed_forward.router"] = value.T
            case "feed_forward.experts.w13_weight":
                let parts = split(value, parts: 2, axis: 1)
                out[experts + "gate_proj.weight"] = packed(parts[0])
                out[experts + "up_proj.weight"] = packed(parts[1])
            case "feed_forward.experts.w13_weight_scale":
                let parts = split(value, parts: 2, axis: 1)
                out[experts + "gate_proj.scales"] = blockScales(parts[0])
                out[experts + "up_proj.scales"] = blockScales(parts[1])
            case "feed_forward.experts.w13_weight_scale_2":
                let scale = value.asType(.float32).reshaped(-1, 2) * nvfp4ScaleToGlobal
                out[experts + "gate_proj.global_scale"] = scale[0..., 0]
                out[experts + "up_proj.global_scale"] = scale[0..., 1]
            case "feed_forward.experts.w2_weight":
                out[experts + "down_proj.weight"] = packed(value)
            case "feed_forward.experts.w2_weight_scale":
                out[experts + "down_proj.scales"] = blockScales(value)
            case "feed_forward.experts.w2_weight_scale_2":
                out[experts + "down_proj.global_scale"] =
                    value.asType(.float32).reshaped(-1) * nvfp4ScaleToGlobal
            default:
                if suffix.hasPrefix("self_attn.qkvg_proj.") {
                    // mlx-lm fuses the projections; this model keeps them apart.
                    let parameter = suffix.dropFirst("self_attn.qkvg_proj.".count)
                    let q = c.attentionHeads * c.headDim
                    let kv = c.kvHeads * c.headDim
                    let parts = split(value, indices: [q, q + kv, q + 2 * kv], axis: 0)
                    for (projection, part) in zip(
                        ["q_proj", "k_proj", "v_proj", "o_gate_proj"], parts)
                    {
                        out[prefix + "self_attn.\(projection).\(parameter)"] = part
                    }
                } else if splitHalfRoPE, suffix.hasPrefix("self_attn.q_proj.") {
                    out[key] = interleavedRoPERows(value, heads: c.attentionHeads)
                } else if splitHalfRoPE, suffix.hasPrefix("self_attn.k_proj.") {
                    out[key] = interleavedRoPERows(value, heads: c.kvHeads)
                } else if !converted, zeroCenteredNorms.contains(where: { key.hasSuffix($0) }) {
                    out[key] = (1 + value.asType(.float32)).asType(.bfloat16)
                } else {
                    out[key] = value
                }
            }
        }

        if c.useFP32ForOutput, out["lm_head.scales"] == nil, let head = out["lm_head.weight"] {
            out["lm_head.weight"] = head.asType(.float32)
        }
        return out
    }

    /// Undoes the transformers split-half permutation of Q or K rows (weights, scales, biases).
    static func interleavedRoPERows(_ value: MLXArray, heads: Int) -> MLXArray {
        let rows = value.dim(0)
        let columns = value.dim(1)
        return value.reshaped(heads, 2, rows / heads / 2, columns).swappedAxes(1, 2)
            .reshaped(rows, columns)
    }
}

// MARK: - Conversion

extension MuseSparkModel {
    /// Conversion recipe for running the 816B model on two 256 GB Macs.
    ///
    /// Routed experts take `expertBits`; the experts of the first and last layers, which
    /// Meta's NVFP4 build keeps in bf16, take `boundaryExpertBits` (`nil` keeps them in bf16).
    /// Attention and latent projections take `denseBits`, the embedding and head take
    /// `embeddingBits`. Norms, the router, its bias and the residual gates are parameters,
    /// not layers, and keep their precision. With the defaults the model is about 367 GB.
    public static func conversionQuantizationPredicate(
        hiddenLayers: Int = 62,
        expertBits: Int = 3,
        boundaryExpertBits: Int? = 4,
        denseBits: Int = 4,
        embeddingBits: Int = 8,
        groupSize: Int = 64
    ) -> ModelConversionQuantizationPredicate {
        let boundary: Set<Int> = [0, hiddenLayers - 1]
        return { path, module in
            func quantize(_ bits: Int) -> ModelConversionQuantizationDecision {
                .quantize(.init(bits: bits, groupSize: groupSize, mode: .affine))
            }
            if path == "model.embed_tokens" || path == "lm_head" {
                return quantize(embeddingBits)
            }
            let parts = path.split(separator: ".")
            if parts.count == 6, parts[0] == "model", parts[1] == "layers",
                parts[3] == "feed_forward", parts[4] == "switch_mlp", let layer = Int(parts[2])
            {
                if boundary.contains(layer) {
                    return boundaryExpertBits.map(quantize) ?? .skip
                }
                return quantize(expertBits)
            }
            return module is Linear ? quantize(denseBits) : .skip
        }
    }
}
