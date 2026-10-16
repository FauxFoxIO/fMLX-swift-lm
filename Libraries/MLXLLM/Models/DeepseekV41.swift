//
//  DeepseekV41.swift
//  LLM
//
//  DeepSeek-V4.1 Flash (`deepseek_v41`) text model: single-KV-head latent attention over a
//  sliding window plus indexer-selected compressed KV, hyper-connections with Sinkhorn mixing,
//  sqrt-softplus routed MoE, and Engram n-gram memory.
//  Port of mlx-lm `models/deepseek_v41.py`; Engram follows DeepSeek's `inference/model.py`.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct DeepseekV41Configuration: Codable, Sendable {
    public var vocabularySize: Int
    public var hiddenSize: Int
    public var moeIntermediateSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var headDim: Int
    public var qkRopeHeadDim: Int
    public var qLoraRank: Int
    public var oLoraRank: Int
    public var oGroups: Int
    public var swigluLimit: Float
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var ropeScaling: [String: StringOrNumber]?
    public var maxPositionEmbeddings: Int
    public var nRoutedExperts: Int
    public var nSharedExperts: Int
    public var numExpertsPerTok: Int
    public var scoringFunc: String
    public var gateTemp: Float
    public var normTopkProb: Bool
    public var routedScalingFactor: Float
    public var slidingWindow: Int
    public var compressRatios: [Int]
    public var compressRopeTheta: Float
    public var kvSourceLayerIDs: [Int]
    public var indexSourceLayerIDs: [Int]
    public var candidateSourceLayerID: Int
    public var candidateTopkBlocks: Int
    public var candidateBlockSize: Int
    public var indexHeads: Int
    public var indexHeadDim: Int
    public var indexTopK: Int
    public var hcMult: Int
    public var hcSinkhornIters: Int
    public var hcEps: Float
    public var engramLayerIDs: [Int]
    public var engramNumEmbeddings: [Int]
    public var engramMaxNgramSize: Int
    public var engramVocabSize: Int
    public var engramHeads: Int
    public var engramHeadDim: Int
    public var engramPadTokenID: Int
    public var engramCompressedVocabSize: Int
    public var numNextnPredictLayers: Int
    public var tieWordEmbeddings: Bool
    public var dsparkBlockSize: Int
    public var dsparkNoiseTokenID: Int
    public var dsparkTargetLayerIDs: [Int]
    public var dsparkMarkovRank: Int
    public var dsparkRoutedExperts: Int
    public var dsparkExpertsPerTok: Int

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qLoraRank = "q_lora_rank"
        case oLoraRank = "o_lora_rank"
        case oGroups = "o_groups"
        case swigluLimit = "swiglu_limit"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case scoringFunc = "scoring_func"
        case gateTemp = "gate_temp"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case slidingWindow = "sliding_window"
        case compressRatios = "compress_ratios"
        case compressRopeTheta = "compress_rope_theta"
        case kvSourceLayerIDs = "kv_source_layer_ids"
        case indexSourceLayerIDs = "index_source_layer_ids"
        case candidateSourceLayerID = "candidate_source_layer_id"
        case candidateTopkBlocks = "candidate_topk_blocks"
        case candidateBlockSize = "candidate_block_size"
        case indexHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopK = "index_topk"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case engramLayerIDs = "engram_layer_ids"
        case engramNumEmbeddings = "engram_num_embeddings"
        case engramMaxNgramSize = "engram_max_ngram_size"
        case engramVocabSize = "engram_vocab_size"
        case engramHeads = "engram_n_heads"
        case engramHeadDim = "engram_head_dim"
        case engramPadTokenID = "engram_pad_token_id"
        case engramCompressedVocabSize = "engram_compressed_vocab_size"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case tieWordEmbeddings = "tie_word_embeddings"
        case dsparkBlockSize = "dspark_block_size"
        case dsparkNoiseTokenID = "dspark_noise_token_id"
        case dsparkTargetLayerIDs = "dspark_target_layer_ids"
        case dsparkMarkovRank = "dspark_markov_rank"
        case dsparkRoutedExperts = "dspark_n_routed_experts"
        case dsparkExpertsPerTok = "dspark_num_experts_per_tok"
    }

    /// Decodes the text model from the root of `config.json` or from its `text_config`.
    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: CodingKeys.self)
        let c =
            root.contains(.textConfig)
            ? try root.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig) : root
        func value<T: Decodable>(_ key: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: key) ?? fallback
        }
        vocabularySize = try value(.vocabularySize, 129_280)
        hiddenSize = try value(.hiddenSize, 5120)
        moeIntermediateSize = try value(.moeIntermediateSize, 2304)
        hiddenLayers = try value(.hiddenLayers, 40)
        attentionHeads = try value(.attentionHeads, 64)
        headDim = try value(.headDim, 512)
        qkRopeHeadDim = try value(.qkRopeHeadDim, 64)
        qLoraRank = try value(.qLoraRank, 1280)
        oLoraRank = try value(.oLoraRank, 1024)
        oGroups = try value(.oGroups, 8)
        swigluLimit = try value(.swigluLimit, 10)
        rmsNormEps = try value(.rmsNormEps, 1e-20)
        ropeTheta = try value(.ropeTheta, 10_000)
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        maxPositionEmbeddings = try value(.maxPositionEmbeddings, 1_048_576)
        nRoutedExperts = try value(.nRoutedExperts, 384)
        nSharedExperts = try value(.nSharedExperts, 1)
        numExpertsPerTok = try value(.numExpertsPerTok, 6)
        scoringFunc = try value(.scoringFunc, "sqrtsoftplus")
        gateTemp = try value(.gateTemp, 1)
        normTopkProb = try value(.normTopkProb, true)
        routedScalingFactor = try value(.routedScalingFactor, 1.5)
        slidingWindow = try value(.slidingWindow, 128)
        compressRatios = try value(.compressRatios, [])
        compressRopeTheta = try value(.compressRopeTheta, 160_000)
        kvSourceLayerIDs = try value(.kvSourceLayerIDs, [])
        indexSourceLayerIDs = try value(.indexSourceLayerIDs, [])
        candidateSourceLayerID = try value(.candidateSourceLayerID, -1)
        candidateTopkBlocks = try value(.candidateTopkBlocks, 2048)
        candidateBlockSize = try value(.candidateBlockSize, 8)
        indexHeads = try value(.indexHeads, 32)
        indexHeadDim = try value(.indexHeadDim, 128)
        indexTopK = try value(.indexTopK, 512)
        hcMult = try value(.hcMult, 4)
        hcSinkhornIters = try value(.hcSinkhornIters, 20)
        hcEps = try value(.hcEps, 1e-6)
        engramLayerIDs = try value(.engramLayerIDs, [])
        engramNumEmbeddings = try value(.engramNumEmbeddings, [])
        engramMaxNgramSize = try value(.engramMaxNgramSize, 1)
        engramVocabSize = try value(.engramVocabSize, 0)
        engramHeads = try value(.engramHeads, 0)
        engramHeadDim = try value(.engramHeadDim, 0)
        engramPadTokenID = try value(.engramPadTokenID, 2)
        engramCompressedVocabSize = try value(.engramCompressedVocabSize, 0)
        numNextnPredictLayers = try value(.numNextnPredictLayers, 0)
        tieWordEmbeddings = try value(.tieWordEmbeddings, false)
        dsparkBlockSize = try value(.dsparkBlockSize, 0)
        dsparkNoiseTokenID = try value(.dsparkNoiseTokenID, 0)
        dsparkTargetLayerIDs = try value(.dsparkTargetLayerIDs, [])
        dsparkMarkovRank = try value(.dsparkMarkovRank, 256)
        dsparkRoutedExperts = try value(.dsparkRoutedExperts, 0)
        dsparkExpertsPerTok = try value(.dsparkExpertsPerTok, 0)

        guard ["sqrtsoftplus", "sigmoid", "softmax"].contains(scoringFunc) else {
            throw DecodingError.dataCorruptedError(
                forKey: .scoringFunc, in: c,
                debugDescription: "unsupported scoring_func '\(scoringFunc)'")
        }
        guard nSharedExperts == 1, attentionHeads % oGroups == 0,
            engramLayerIDs.count == engramNumEmbeddings.count
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .nSharedExperts, in: c,
                debugDescription: "deepseek_v41 needs one shared expert, heads divisible by "
                    + "o_groups, and one engram_num_embeddings entry per Engram layer")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(qkRopeHeadDim, forKey: .qkRopeHeadDim)
        try c.encode(qLoraRank, forKey: .qLoraRank)
        try c.encode(oLoraRank, forKey: .oLoraRank)
        try c.encode(oGroups, forKey: .oGroups)
        try c.encode(swigluLimit, forKey: .swigluLimit)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(nRoutedExperts, forKey: .nRoutedExperts)
        try c.encode(nSharedExperts, forKey: .nSharedExperts)
        try c.encode(numExpertsPerTok, forKey: .numExpertsPerTok)
        try c.encode(scoringFunc, forKey: .scoringFunc)
        try c.encode(gateTemp, forKey: .gateTemp)
        try c.encode(normTopkProb, forKey: .normTopkProb)
        try c.encode(routedScalingFactor, forKey: .routedScalingFactor)
        try c.encode(slidingWindow, forKey: .slidingWindow)
        try c.encode(compressRatios, forKey: .compressRatios)
        try c.encode(compressRopeTheta, forKey: .compressRopeTheta)
        try c.encode(kvSourceLayerIDs, forKey: .kvSourceLayerIDs)
        try c.encode(indexSourceLayerIDs, forKey: .indexSourceLayerIDs)
        try c.encode(candidateSourceLayerID, forKey: .candidateSourceLayerID)
        try c.encode(candidateTopkBlocks, forKey: .candidateTopkBlocks)
        try c.encode(candidateBlockSize, forKey: .candidateBlockSize)
        try c.encode(indexHeads, forKey: .indexHeads)
        try c.encode(indexHeadDim, forKey: .indexHeadDim)
        try c.encode(indexTopK, forKey: .indexTopK)
        try c.encode(hcMult, forKey: .hcMult)
        try c.encode(hcSinkhornIters, forKey: .hcSinkhornIters)
        try c.encode(hcEps, forKey: .hcEps)
        try c.encode(engramLayerIDs, forKey: .engramLayerIDs)
        try c.encode(engramNumEmbeddings, forKey: .engramNumEmbeddings)
        try c.encode(engramMaxNgramSize, forKey: .engramMaxNgramSize)
        try c.encode(engramVocabSize, forKey: .engramVocabSize)
        try c.encode(engramHeads, forKey: .engramHeads)
        try c.encode(engramHeadDim, forKey: .engramHeadDim)
        try c.encode(engramPadTokenID, forKey: .engramPadTokenID)
        try c.encode(engramCompressedVocabSize, forKey: .engramCompressedVocabSize)
        try c.encode(numNextnPredictLayers, forKey: .numNextnPredictLayers)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(dsparkBlockSize, forKey: .dsparkBlockSize)
        try c.encode(dsparkNoiseTokenID, forKey: .dsparkNoiseTokenID)
        try c.encode(dsparkTargetLayerIDs, forKey: .dsparkTargetLayerIDs)
        try c.encode(dsparkMarkovRank, forKey: .dsparkMarkovRank)
        try c.encode(dsparkRoutedExperts, forKey: .dsparkRoutedExperts)
        try c.encode(dsparkExpertsPerTok, forKey: .dsparkExpertsPerTok)
    }

    /// Compression ratio of `layer`: 0 for window-only attention.
    func compressRatio(_ layer: Int) -> Int {
        layer < compressRatios.count ? compressRatios[layer] : 0
    }

    /// Window-only layers use the base theta without YaRN; compressed layers use the
    /// compressed theta with YaRN and no attention scaling.
    func makeRope(layer: Int) -> DeepseekV41RoPE {
        guard compressRatio(layer) > 0 else {
            return DeepseekV41RoPE(dims: qkRopeHeadDim, base: ropeTheta)
        }
        let scaling = ropeScaling ?? [:]
        return DeepseekV41RoPE(
            dims: qkRopeHeadDim, base: compressRopeTheta,
            factor: scaling["factor"]?.asFloat() ?? 1,
            originalMaxPositions: Int(
                scaling["original_max_position_embeddings"]?.asFloat()
                    ?? Float(maxPositionEmbeddings)),
            betaFast: scaling["beta_fast"]?.asFloat() ?? 32,
            betaSlow: scaling["beta_slow"]?.asFloat() ?? 1)
    }
}

// MARK: - Simulated low-precision rounding

/// The reference keeps these activations in FP8/FP4. The cache and indexer contents depend
/// on that rounding, so it is reproduced exactly.
enum DeepseekV41Rounding {
    private static let fp4Values: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

    /// FP8 E4M3 round trip with a power-of-two scale per block.
    static func fp8(_ x: MLXArray, block: Int = 32) -> MLXArray {
        let shape = x.shape
        guard shape[shape.count - 1] % block == 0 else { return x }
        let blocks = x.asType(.float32).reshaped(Array(shape.dropLast()) + [-1, block])
        let amax = maximum(abs(blocks).max(axis: -1, keepDims: true), MLXArray(Float(1e-4)))
        let scale = pow(MLXArray(Float(2)), ceil(log2(amax / 448)))
        let q = clip(blocks / scale, min: -448, max: 448)
        return (fromFP8(toFP8(q), dtype: .float32) * scale).reshaped(shape).asType(x.dtype)
    }

    static func roundFP4(_ x: MLXArray) -> MLXArray {
        let magnitude = abs(x)
        var index = MLXArray.zeros(magnitude.shape, dtype: .int32)
        for threshold: Float in [0.25, 1.25, 2.5, 5] {
            index = index + (magnitude .> threshold).asType(.int32)
        }
        for threshold: Float in [0.75, 1.75, 3.5] {
            index = index + (magnitude .>= threshold).asType(.int32)
        }
        return sign(x) * MLXArray(fp4Values).take(index)
    }

    /// FP4 E2M1 round trip with a power-of-two or E4M3 scale per block.
    static func fp4(_ x: MLXArray, block: Int, e4m3Scale: Bool = false) -> MLXArray {
        let shape = x.shape
        guard shape[shape.count - 1] % block == 0 else { return x }
        let blocks = x.asType(.float32).reshaped(Array(shape.dropLast()) + [-1, block])
        let floor: Float = e4m3Scale ? 6 * pow(2, -9) : 6 * pow(2, -126)
        let amax = maximum(abs(blocks).max(axis: -1, keepDims: true), MLXArray(floor))
        let scale =
            e4m3Scale
            ? fromFP8(toFP8(amax / 6), dtype: .float32)
            : pow(MLXArray(Float(2)), ceil(log2(amax / 6)))
        let q = roundFP4(clip(blocks / scale, min: -6, max: 6)) * scale
        return q.reshaped(shape).asType(x.dtype)
    }
}

// MARK: - Rotary embedding

/// Interleaved RoPE on the trailing `dims` features, with mlx-lm's YaRN frequencies.
struct DeepseekV41RoPE {
    let dims: Int
    let freqs: MLXArray
    let inverseFreqs: MLXArray

    init(
        dims: Int, base: Float, factor: Float = 1, originalMaxPositions: Int = 4096,
        betaFast: Float = 32, betaSlow: Float = 1
    ) {
        func correctionDim(_ rotations: Float) -> Double {
            Double(dims) * log(Double(originalMaxPositions) / (Double(rotations) * 2 * .pi))
                / (2 * log(Double(base)))
        }
        let low = max(Int(floor(correctionDim(betaFast))), 0)
        let high = min(Int(ceil(correctionDim(betaSlow))), dims - 1)
        let span = low == high ? Float(high) + 0.001 - Float(low) : Float(high - low)
        let ramp = clip(
            (MLXArray(0 ..< (dims / 2)).asType(.float32) - Float(low)) / span, min: 0, max: 1)
        let mask = 1 - ramp
        let extra = pow(
            MLXArray(base),
            MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32) / Float(dims))
        let inter = factor * extra
        self.dims = dims
        freqs = (inter * extra) / (inter * mask + extra * (1 - mask))
        inverseFreqs = -freqs
    }

    /// Rotates the trailing features of `[B, L, D]` or `[B, L, H, D]` at positions
    /// `(offset + i) * scale`; `inverse` undoes the rotation.
    func apply(_ x: MLXArray, offset: Int, scale: Float = 1, inverse: Bool = false) -> MLXArray {
        let split = x.dim(-1) - dims
        var tail = x[.ellipsis, split...].asType(.float32)
        let freqs = inverse ? inverseFreqs : freqs
        if x.ndim == 4 {
            tail = tail.transposed(0, 2, 1, 3)
        }
        tail = MLXFast.RoPE(
            tail, dimensions: dims, traditional: true, base: nil, scale: scale, offset: offset,
            freqs: freqs)
        if x.ndim == 4 {
            tail = tail.transposed(0, 2, 1, 3)
        }
        return concatenated([x[.ellipsis, ..<split], tail.asType(x.dtype)], axis: -1)
    }
}

// MARK: - Hyper-connections

private let sinkhornSource = """
    uint n = thread_position_in_grid.x;
    if (n >= comb_shape[0]) {
        return;
    }
    float m[HC * HC];
    for (int k = 0; k < HC * HC; k++) {
        m[k] = comb[n * HC * HC + k];
    }
    float e = eps[0];
    for (int t = 0; t < 2 * ITERS - 1; t++) {
        bool rows = t % 2 == 1;
        for (int a = 0; a < HC; a++) {
            float s = 0.0f;
            for (int b = 0; b < HC; b++) {
                s += rows ? m[a * HC + b] : m[b * HC + a];
            }
            float d = s + e;
            for (int b = 0; b < HC; b++) {
                int k = rows ? a * HC + b : b * HC + a;
                m[k] = m[k] / d;
            }
        }
    }
    for (int k = 0; k < HC * HC; k++) {
        out[n * HC * HC + k] = m[k];
    }
    """

private final class SinkhornKernel: Sendable {
    static let shared = SinkhornKernel()
    let kernel = MLXFast.metalKernel(
        name: "deepseek_v41_sinkhorn", inputNames: ["comb", "eps"], outputNames: ["out"],
        source: sinkhornSource)
}

/// Normalizes the columns, then the rows and the columns `iters - 1` more times.
func deepseekV41Sinkhorn(_ comb: MLXArray, iters: Int, eps: Float, fused: Bool = true)
    -> MLXArray
{
    guard fused, iters > 0, comb.size > 0, Device.defaultDevice().deviceType == .gpu else {
        var comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
        for _ in 1 ..< max(iters, 1) {
            comb = comb / (comb.sum(axis: -1, keepDims: true) + eps)
            comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
        }
        return comb
    }
    let hc = comb.dim(-1)
    let flat = comb.asType(.float32).reshaped(-1, hc, hc)
    let rows = flat.dim(0)
    let out = SinkhornKernel.shared.kernel(
        [flat, MLXArray([eps])], template: [("HC", hc), ("ITERS", iters)],
        grid: (rows, 1, 1), threadGroup: (min(rows, 256), 1, 1),
        outputShapes: [flat.shape], outputDTypes: [.float32])
    return out[0].reshaped(comb.shape)
}

/// Mixing coefficients for one sublayer, derived from the `hc_mult` residual streams.
final class DeepseekV41HyperConnection: Module {
    let hc: Int
    let iters: Int
    let hcEps: Float
    let normEps: Float
    let _scaleIDs: MLXArray
    let _ones: MLXArray

    @ParameterInfo(key: "fn") var fn: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var mixScale: MLXArray

    init(_ config: DeepseekV41Configuration) {
        hc = config.hcMult
        iters = config.hcSinkhornIters
        hcEps = config.hcEps
        normEps = config.rmsNormEps
        _scaleIDs = MLXArray(
            Array(repeating: Int32(0), count: hc) + Array(repeating: 1, count: hc)
                + Array(repeating: 2, count: hc * hc))
        _ones = MLXArray.ones([hc * config.hiddenSize], dtype: .float32)
        let mix = (2 + hc) * hc
        _fn.wrappedValue = MLXArray.zeros([mix, hc * config.hiddenSize], dtype: .float32)
        _base.wrappedValue = MLXArray.zeros([mix], dtype: .float32)
        _mixScale.wrappedValue = MLXArray.ones([3], dtype: .float32)
        super.init()
    }

    /// `(pre, post, comb)` for streams `[B, L, hc, D]`, all float32.
    func callAsFunction(_ streams: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let flat = streams.reshaped(streams.dim(0), streams.dim(1), -1).asType(.float32)
        let normed = MLXFast.rmsNorm(flat, weight: _ones, eps: normEps)
        let mix = matmul(normed, fn.T) * mixScale.take(_scaleIDs) + base
        let parts = split(mix, indices: [hc, 2 * hc], axis: -1)
        let pre = sigmoid(parts[0]) + hcEps
        let post = 2 * sigmoid(parts[1])
        var comb = parts[2].reshaped(Array(parts[2].shape.dropLast()) + [hc, hc])
        comb = softmax(comb, axis: -1) + hcEps
        return (pre, post, deepseekV41Sinkhorn(comb, iters: iters, eps: hcEps))
    }
}

/// Collapses `[B, L, hc, D]` streams to `[B, L, D]` with weights `[B, L, hc]`.
func deepseekV41Collapse(_ streams: MLXArray, _ mix: MLXArray) -> MLXArray {
    (mix.expandedDimensions(axis: -1) * streams.asType(.float32)).sum(axis: 2)
        .asType(streams.dtype)
}

/// Adds a sublayer output to each stream and mixes the residual streams through `comb`.
func deepseekV41Expand(_ x: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray)
    -> MLXArray
{
    let mixed = (comb.expandedDimensions(axis: -1) * residual.expandedDimensions(axis: -2))
        .sum(axis: 2)
    let out = post.expandedDimensions(axis: -1) * x.expandedDimensions(axis: -2) + mixed
    return out.asType(residual.dtype)
}

// MARK: - Cache

/// Rows appended along axis 1 into a buffer that grows in steps.
struct DeepseekV41Rows {
    private(set) var storage: MLXArray?
    private(set) var count = 0
    static let step = 256

    var array: MLXArray? {
        guard let storage, count > 0 else { return nil }
        return count == storage.dim(1) ? storage : storage[0..., ..<count]
    }

    mutating func append(_ rows: MLXArray) {
        let n = rows.dim(1)
        if storage == nil || count + n > storage!.dim(1) {
            let capacity = (count + n + Self.step - 1) / Self.step * Self.step
            var grown = MLXArray.zeros([rows.dim(0), capacity, rows.dim(2)], dtype: rows.dtype)
            if let storage, count > 0 {
                grown[0..., ..<count] = storage[0..., ..<count]
            }
            storage = grown
        }
        storage![0..., count ..< (count + n)] = rows
        count += n
    }

    mutating func truncate(_ newCount: Int) {
        count = min(count, max(newCount, 0))
    }

    mutating func replace(with array: MLXArray?) {
        storage = array
        count = array?.dim(1) ?? 0
    }
}

/// Cache for one decoder layer: the sliding window, the compressed KV and index keys this
/// layer produces (KV source layers only), and the recent compressor projections.
///
/// Positions can be rewound by up to ``rewindLimit``: the window, the compressor inputs and
/// the Engram history keep that many extra positions.
public final class DeepseekV41Cache: KVCache {
    public static let defaultRewindLimit = 64

    public let windowSize: Int
    public let rewindLimit: Int
    let ratio: Int

    public private(set) var offset = 0
    var window: MLXArray?
    var compressKV = DeepseekV41Rows()
    var indexK = DeepseekV41Rows()
    var rawKV: MLXArray?
    var rawGate: MLXArray?
    var rawStart = 0
    /// Compressed token ids of the latest positions, per batch row (first layer only).
    var engramHistory: [[Int32]] = []

    public init(windowSize: Int, ratio: Int, rewindLimit: Int = DeepseekV41Cache.defaultRewindLimit)
    {
        self.windowSize = windowSize
        self.ratio = ratio
        self.rewindLimit = rewindLimit
    }

    public var maxSize: Int? { nil }

    /// Window rows held before this step.
    var windowCount: Int { window?.dim(1) ?? 0 }

    public func innerState() -> [MLXArray] {
        [window, compressKV.array, indexK.array, rawKV, rawGate].compactMap { $0 }
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("DeepseekV41Cache holds latent attention state, not per-head keys and values")
    }

    /// Returns the window with `kv` appended and advances the offset.
    func updateWindow(_ kv: MLXArray) -> MLXArray {
        let all = window.map { concatenated([$0, kv], axis: 1) } ?? kv
        let keep = windowSize + rewindLimit
        window = all.dim(1) > keep ? all[0..., (all.dim(1) - keep)...] : all
        offset += kv.dim(1)
        return all
    }

    /// Appends compressor projections for positions starting at `start` and returns those of
    /// the open group onward, which starts at `groupStart`.
    func pendingRows(kv: MLXArray, gate: MLXArray, start: Int, groupStart: Int) -> (
        MLXArray, MLXArray
    ) {
        let allKV = rawKV.map { concatenated([$0, kv], axis: 1) } ?? kv
        let allGate = rawGate.map { concatenated([$0, gate], axis: 1) } ?? gate
        let first = rawKV == nil ? start : rawStart
        let from = groupStart - first
        let keep = min(allKV.dim(1), rewindLimit + ratio)
        let drop = allKV.dim(1) - keep
        rawKV = drop > 0 ? allKV[0..., drop...] : allKV
        rawGate = drop > 0 ? allGate[0..., drop...] : allGate
        rawStart = first + drop
        return (allKV[0..., from...], allGate[0..., from...])
    }

    public var state: [MLXArray] {
        get {
            var arrays = innerState()
            if !engramHistory.isEmpty {
                let width = engramHistory[0].count
                arrays.append(
                    MLXArray(engramHistory.flatMap { $0 }, [engramHistory.count, width]))
            }
            return arrays
        }
        set {
            pendingState = newValue
            restore()
        }
    }

    public var metaState: [String] {
        get {
            var flags = ""
            if window != nil { flags.append("w") }
            if compressKV.array != nil { flags.append("c") }
            if indexK.array != nil { flags.append("i") }
            if rawKV != nil { flags.append("r") }
            if rawGate != nil { flags.append("g") }
            if !engramHistory.isEmpty { flags.append("h") }
            return [String(offset), String(rawStart), flags]
        }
        set {
            guard newValue.count == 3, let offset = Int(newValue[0]),
                let rawStart = Int(newValue[1])
            else {
                fatalError("DeepseekV41Cache metaState has \(newValue.count) entries, expected 3")
            }
            self.offset = offset
            self.rawStart = rawStart
            pendingFlags = newValue[2]
            restore()
        }
    }

    private var pendingState: [MLXArray]?
    private var pendingFlags: String?

    /// Applies a restored state once both the arrays and the flags naming them are set.
    private func restore() {
        guard let state = pendingState, let flags = pendingFlags else { return }
        pendingState = nil
        pendingFlags = nil
        var arrays = state[...]
        func next(_ flag: Character) -> MLXArray? {
            flags.contains(flag) ? arrays.popFirst() : nil
        }
        window = next("w")
        compressKV.replace(with: next("c"))
        indexK.replace(with: next("i"))
        rawKV = next("r")
        rawGate = next("g")
        engramHistory = []
        if let history = next("h") {
            let values = history.asArray(Int32.self)
            let width = history.dim(1)
            engramHistory = (0 ..< history.dim(0)).map {
                Array(values[($0 * width) ..< (($0 + 1) * width)])
            }
        }
    }

    public var isTrimmable: Bool { true }

    public var maxTrimCount: Int { min(offset, rewindLimit) }

    public func isTrimmable(after positions: Int) -> Bool { true }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let n = min(max(n, 0), maxTrimCount)
        guard n > 0 else { return 0 }
        let newOffset = offset - n
        if let window {
            self.window = window[0..., ..<(window.dim(1) - n)]
        }
        if ratio > 0 {
            compressKV.truncate(newOffset / ratio)
            indexK.truncate(newOffset / ratio)
        }
        if let rawKV, let rawGate {
            let keep = max(newOffset - rawStart, 0)
            self.rawKV = rawKV[0..., ..<keep]
            self.rawGate = rawGate[0..., ..<keep]
        }
        engramHistory = engramHistory.map { Array($0.dropLast(min(n, $0.count))) }
        offset = newOffset
        return n
    }

    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        .none
    }

    public func copy() -> any KVCache {
        let new = DeepseekV41Cache(windowSize: windowSize, ratio: ratio, rewindLimit: rewindLimit)
        new.offset = offset
        new.window = window.map { $0[.ellipsis] }
        new.compressKV.replace(with: compressKV.array.map { $0[.ellipsis] })
        new.indexK.replace(with: indexK.array.map { $0[.ellipsis] })
        new.rawKV = rawKV.map { $0[.ellipsis] }
        new.rawGate = rawGate.map { $0[.ellipsis] }
        new.rawStart = rawStart
        new.engramHistory = engramHistory
        return new
    }
}

/// State one layer hands to the layers after it in the same forward pass.
final class DeepseekV41SharedState {
    var compressKV: MLXArray?
    var indexK: MLXArray?
    var topK: MLXArray?
    var candidates: MLXArray?
}

// MARK: - Attention

final class DeepseekV41Compressor: Module {
    let ratio: Int

    @ModuleInfo(key: "kv_proj") var kvProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear?
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm

    init(_ config: DeepseekV41Configuration, layer: Int) {
        ratio = config.compressRatio(layer)
        _kvProj.wrappedValue = Linear(config.hiddenSize, config.headDim, bias: false)
        if ratio > 1 {
            _gateProj.wrappedValue = Linear(config.hiddenSize, config.headDim, bias: false)
        }
        _kvNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// Pools each complete group of `ratio` tokens into one latent. Returns the latents
    /// (`nil` when no group completes) and the index of the first one.
    func callAsFunction(_ x: MLXArray, startPos: Int, cache: DeepseekV41Cache) -> (MLXArray?, Int) {
        guard ratio > 1, let gateProj else {
            return (kvNorm(kvProj(x)), startPos)
        }
        let h = x.asType(.float32)
        let groupStart = startPos / ratio * ratio
        let (kv, gate) = cache.pendingRows(
            kv: kvProj(h), gate: gateProj(h), start: startPos, groupStart: groupStart)
        let full = kv.dim(1) - kv.dim(1) % ratio
        let first = groupStart / ratio
        guard full > 0 else { return (nil, first) }
        let (b, d) = (kv.dim(0), kv.dim(2))
        let groups = kv[0..., ..<full].reshaped(b, -1, ratio, d)
        let weights = softmax(gate[0..., ..<full].reshaped(b, -1, ratio, d), axis: 2)
        let latent = (groups * weights).sum(axis: 2)
        return (kvNorm(latent.asType(x.dtype)), first)
    }
}

final class DeepseekV41Indexer: Module {
    let ratio: Int
    let isCandidateSource: Bool
    let candidateTopkBlocks: Int
    let candidateBlockSize: Int
    let heads: Int
    let headDim: Int
    let topK: Int

    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    init(_ config: DeepseekV41Configuration, layer: Int) {
        ratio = config.compressRatio(layer)
        isCandidateSource = layer == config.candidateSourceLayerID
        candidateTopkBlocks = config.candidateTopkBlocks
        candidateBlockSize = config.candidateBlockSize
        heads = config.indexHeads
        headDim = config.indexHeadDim
        topK = config.indexTopK
        _qBProj.wrappedValue = Linear(config.qLoraRank, heads * headDim, bias: false)
        _weightsProj.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        // Only a layer that compresses its own KV makes index keys.
        if config.kvSourceLayerIDs.contains(layer) {
            _kProj.wrappedValue = Linear(config.headDim, config.indexHeadDim, bias: false)
            _kNorm.wrappedValue = RMSNorm(dimensions: config.indexHeadDim, eps: config.rmsNormEps)
        }
        super.init()
    }

    /// Index keys for new compressed latents.
    func keys(_ latent: MLXArray, rope: DeepseekV41RoPE, first: Int) -> MLXArray {
        guard let kProj, let kNorm else {
            fatalError("only a KV source layer's indexer makes index keys")
        }
        let k = rope.apply(kNorm(kProj(latent)), offset: first, scale: Float(ratio))
        return DeepseekV41Rounding.fp4(k, block: 32)
    }

    /// The top-k compressed entries of each query (`-1` for none) and the candidate mask:
    /// the one this layer selects, or the given one.
    func callAsFunction(
        _ x: MLXArray, qResidual: MLXArray, indexK: MLXArray?, candidates: MLXArray?,
        rope: DeepseekV41RoPE, startPos: Int
    ) -> (MLXArray, MLXArray?) {
        let (b, l) = (x.dim(0), x.dim(1))
        guard let indexK, indexK.dim(1) > 0 else {
            return (MLXArray.zeros([b, l, 0], dtype: .int32), candidates)
        }
        let n = indexK.dim(1)
        var q = qBProj(qResidual).reshaped(b, l, heads, headDim)
        q = DeepseekV41Rounding.fp4(rope.apply(q, offset: startPos), block: 32).asType(.float32)
        let keys = indexK.asType(.float32).expandedDimensions(axis: 1).swappedAxes(-1, -2)
        let weights = (weightsProj(x).asType(.float32) * pow(Float(heads), -0.5))
            .expandedDimensions(axis: -1)
        let keyScale = pow(Float(headDim), -0.5)
        var scores: MLXArray
        if l * heads * n <= 1 << 24 {
            scores = (maximum(matmul(q, keys), 0) * keyScale * weights).sum(axis: 2)
        } else {
            // Accumulate per head so long prefill holds one [B, L, N] buffer.
            scores = MLXArray.zeros([b, l, n], dtype: .float32)
            for h in 0 ..< heads {
                let qh = q[0..., 0..., h ..< (h + 1)]
                scores =
                    scores
                    + (maximum(matmul(qh, keys), 0) * keyScale
                    * weights[0..., 0..., h ..< (h + 1)]).squeezed(axis: 2)
            }
        }
        // A query sees the groups that end at or before it.
        let lens = MLXArray((0 ..< l).map { Int32((startPos + 1 + $0) / ratio) })
            .expandedDimensions(axis: -1)
        let visible = MLXArray(0 ..< Int32(n)).expandedDimensions(axis: 0) .< lens
        scores = MLX.where(visible.expandedDimensions(axis: 0), scores, -Float.infinity)

        var candidates = candidates
        if isCandidateSource && candidateTopkBlocks > 0 {
            let block = candidateBlockSize
            let pad = (block - n % block) % block
            let padded =
                pad > 0
                ? MLX.padded(
                    scores, widths: [.init((0, 0)), .init((0, 0)), .init((0, pad))],
                    value: MLXArray(-Float.infinity)) : scores
            var blockScores = padded.reshaped(b, l, -1, block).max(axis: -1)
            let blocks = blockScores.dim(-1)
            // Always keep the newest, partly filled block.
            let last = MLX.where(lens .> 0, floorDivide(lens - 1, block), MLXArray(Int32(-1)))
            blockScores = MLX.where(
                MLXArray(0 ..< Int32(blocks)) .== last, Float.infinity, blockScores)
            let kBlocks = min(candidateTopkBlocks, blocks)
            let chosen = argPartition(-blockScores, kth: kBlocks - 1, axis: -1)[
                .ellipsis, ..<kBlocks]
            let reachable = takeAlong(blockScores, chosen, axis: -1) .> -Float.infinity
            let picked = putAlong(
                MLXArray.zeros(blockScores.shape, dtype: .bool), chosen, values: reachable,
                axis: -1)
            candidates = repeated(picked, count: block, axis: -1)[.ellipsis, ..<n]
        } else if let candidates {
            scores = MLX.where(candidates, scores, -Float.infinity)
        }

        let k = min(topK, n)
        let chosen = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k].asType(.int32)
        return (MLX.where(chosen .< lens.expandedDimensions(axis: 0), chosen, -1), candidates)
    }
}

/// Window positions each query attends to, `-1` for none. The window holds `previous` rows,
/// then the new ones.
func deepseekV41WindowIndices(previous: Int, length: Int, window: Int) -> MLXArray {
    let width = min(window, previous + length)
    var indices = [Int32]()
    indices.reserveCapacity(length * width)
    for i in 0 ..< length {
        let end = previous + i
        let start = max(end - window + 1, 0)
        for j in 0 ..< width {
            indices.append(start + j > end ? -1 : Int32(start + j))
        }
    }
    return MLXArray(indices, [length, width])
}

/// Each query attends to the rows its indices pick from each KV array, plus a learned sink.
/// `q` is `[B, L, H, D]`, each KV array `[B, N, D]`, and its indices `[B, L, k]`.
func deepseekV41SparseAttention(
    _ q: MLXArray, kvs: [MLXArray], indices: [MLXArray], sinks: MLXArray, scale: Float
) -> MLXArray {
    let (b, l, h, d) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
    let rows = zip(kvs, indices).map { kv, idx in
        let n = kv.dim(1)
        let base = (MLXArray(0 ..< Int32(b)) * Int32(n)).reshaped(b, 1, 1)
        let flat = kv.reshaped(b * n, d)
        return flat.take((maximum(idx, 0) + base).flattened(), axis: 0)
            .reshaped(idx.shape + [d])
    }
    // Each query has its own keys, so each query is one batch entry.
    let keys = concatenated(rows, axis: 2).reshaped(b * l, 1, -1, d).asType(.float32)
    let mask = (concatenated(indices, axis: -1) .>= 0).reshaped(b * l, 1, 1, -1)
    // SDPA would round the scores to the input dtype at this head size, so use float32.
    let out = MLXFast.scaledDotProductAttention(
        queries: q.reshaped(b * l, h, 1, d).asType(.float32), keys: keys, values: keys,
        scale: scale, mask: .array(mask), sinks: sinks.asType(.float32))
    return out.reshaped(b, l, h, d).asType(q.dtype)
}

final class DeepseekV41Attention: Module {
    let numHeads: Int
    let headDim: Int
    let oGroups: Int
    let ratio: Int
    let isKVSource: Bool
    let isIndexSource: Bool
    let rope: DeepseekV41RoPE
    let _groupIDs: MLXArray

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_norm") var qANorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_proj") var kvProj: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    @ModuleInfo(key: "o_a_proj") var oAProj: SwitchLinear
    @ModuleInfo(key: "o_b_proj") var oBProj: Linear
    @ParameterInfo(key: "sinks") var sinks: MLXArray
    @ModuleInfo(key: "compressor") var compressor: DeepseekV41Compressor?
    @ModuleInfo(key: "indexer") var indexer: DeepseekV41Indexer?

    init(_ config: DeepseekV41Configuration, layer: Int) {
        numHeads = config.attentionHeads
        headDim = config.headDim
        oGroups = config.oGroups
        ratio = config.compressRatio(layer)
        isKVSource = layer < config.hiddenLayers && config.kvSourceLayerIDs.contains(layer)
        isIndexSource = layer < config.hiddenLayers && config.indexSourceLayerIDs.contains(layer)
        rope = config.makeRope(layer: layer)
        _groupIDs = MLXArray(0 ..< Int32(config.oGroups)).expandedDimensions(axis: 0)

        _qAProj.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: false)
        _qANorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        _qBProj.wrappedValue = Linear(config.qLoraRank, numHeads * headDim, bias: false)
        _kvProj.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _kvNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        // wo_a is block diagonal: one projection per output group.
        _oAProj.wrappedValue = SwitchLinear(
            inputDims: numHeads * headDim / oGroups, outputDims: config.oLoraRank,
            numExperts: oGroups, bias: false)
        _oBProj.wrappedValue = Linear(oGroups * config.oLoraRank, config.hiddenSize, bias: false)
        _sinks.wrappedValue = MLXArray.zeros([numHeads], dtype: .float32)
        if isKVSource {
            _compressor.wrappedValue = DeepseekV41Compressor(config, layer: layer)
        }
        if isIndexSource {
            _indexer.wrappedValue = DeepseekV41Indexer(config, layer: layer)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, startPos: Int, windowIndices: MLXArray, cache: DeepseekV41Cache,
        shared: DeepseekV41SharedState
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let qResidual = qANorm(qAProj(x))
        let q = rope.apply(
            qBProj(qResidual).reshaped(b, l, numHeads, headDim), offset: startPos)

        let kv = DeepseekV41Rounding.fp8(rope.apply(kvNorm(kvProj(x)), offset: startPos))
        var kvs = [cache.updateWindow(kv)]
        var indices = [windowIndices]
        if ratio > 0 {
            if isKVSource {
                compress(x, startPos: startPos, cache: cache, shared: shared)
            }
            if let indexer {
                let (topK, candidates) = indexer(
                    x, qResidual: qResidual, indexK: shared.indexK,
                    candidates: shared.candidates, rope: rope, startPos: startPos)
                shared.topK = topK
                if indexer.isCandidateSource {
                    shared.candidates = candidates
                }
            }
            if let compressed = shared.compressKV, let topK = shared.topK, topK.dim(-1) > 0 {
                kvs.append(compressed)
                indices.append(topK)
            }
        }

        var output = deepseekV41SparseAttention(
            q, kvs: kvs, indices: indices, sinks: sinks, scale: pow(Float(headDim), -0.5))
        output = rope.apply(output, offset: startPos, inverse: true)
        output = oAProj(output.reshaped(b * l, oGroups, 1, -1), _groupIDs)
        return oBProj(output.reshaped(b, l, -1))
    }

    /// Window KV of target positions `start...` from their `main_x` rows.
    func windowKV(_ x: MLXArray, start: Int) -> MLXArray {
        DeepseekV41Rounding.fp8(rope.apply(kvNorm(kvProj(x)), offset: start))
    }

    /// DSpark draft attention: each query of the block at positions `start...` attends to the
    /// window of target KV and to the whole block.
    func dspark(_ x: MLXArray, start: Int, window: MLXArray?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let q = rope.apply(
            qBProj(qANorm(qAProj(x))).reshaped(b, l, numHeads, headDim), offset: start)
        let block = windowKV(x, start: start)
        let keys = (window.map { concatenated([$0, block], axis: 1) } ?? block)
            .expandedDimensions(axis: 1).asType(.float32)
        var output = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3).asType(.float32), keys: keys, values: keys,
            scale: pow(Float(headDim), -0.5), mask: .none, sinks: sinks.asType(.float32))
        output = output.transposed(0, 2, 1, 3).asType(q.dtype)
        output = rope.apply(output, offset: start, inverse: true)
        output = oAProj(output.reshaped(b * l, oGroups, 1, -1), _groupIDs)
        return oBProj(output.reshaped(b, l, -1))
    }

    /// Adds new compressed KV and index keys to this layer's cache and shares them.
    private func compress(
        _ x: MLXArray, startPos: Int, cache: DeepseekV41Cache, shared: DeepseekV41SharedState
    ) {
        guard let compressor else { return }
        let (latent, first) = compressor(x, startPos: startPos, cache: cache)
        if let latent {
            let kv = rope.apply(latent, offset: first, scale: Float(ratio))
            cache.compressKV.append(DeepseekV41Rounding.fp4(kv, block: 16, e4m3Scale: true))
            if let indexer {
                cache.indexK.append(indexer.keys(latent, rope: rope, first: first))
            }
        }
        shared.compressKV = cache.compressKV.array
        if indexer != nil {
            shared.indexK = cache.indexK.array
        }
    }
}

// MARK: - MoE

/// SwiGLU with the up branch clamped on both sides and the gate branch from above.
func deepseekV41SwiGLU(up: MLXArray, gate: MLXArray, limit: Float) -> MLXArray {
    var gate = gate.asType(.float32)
    var up = up.asType(.float32)
    if limit > 0 {
        gate = minimum(gate, limit)
        up = clip(up, min: -limit, max: limit)
    }
    return silu(gate) * up
}

final class DeepseekV41MLP: Module, UnaryLayer {
    let limit: Float

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: DeepseekV41Configuration) {
        limit = config.swigluLimit
        _gateProj.wrappedValue = Linear(
            config.hiddenSize, config.moeIntermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.moeIntermediateSize, bias: false)
        _downProj.wrappedValue = Linear(
            config.moeIntermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(deepseekV41SwiGLU(up: upProj(x), gate: gateProj(x), limit: limit).asType(x.dtype))
    }
}

/// Routed experts with the clamped SwiGLU.
final class DeepseekV41SwitchGLU: Module {
    let limit: Float

    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    init(_ config: DeepseekV41Configuration) {
        limit = config.swigluLimit
        let (d, h, e) = (config.hiddenSize, config.moeIntermediateSize, config.nRoutedExperts)
        _gateProj.wrappedValue = SwitchLinear(
            inputDims: d, outputDims: h, numExperts: e, bias: false)
        _upProj.wrappedValue = SwitchLinear(inputDims: d, outputDims: h, numExperts: e, bias: false)
        _downProj.wrappedValue = SwitchLinear(
            inputDims: h, outputDims: d, numExperts: e, bias: false)
        super.init()
    }

    /// Expert outputs `[N, k, D]` for tokens `[N, D]` and indices `[N, k]`.
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = expandedDimensions(x, axes: [-2, -3])
        let sort = indices.size >= 64
        var idx = indices
        var inverse = MLXArray()
        if sort {
            (x, idx, inverse) = gatherSort(x: x, indices: indices)
        }
        let up = upProj(x, idx, sortedIndices: sort)
        let gate = gateProj(x, idx, sortedIndices: sort)
        let activated = deepseekV41SwiGLU(up: up, gate: gate, limit: limit).asType(up.dtype)
        var y = downProj(activated, idx, sortedIndices: sort)
        if sort {
            y = scatterUnsort(x: y, invOrder: inverse, shape: indices.shape)
        }
        return y.squeezed(axis: -2)
    }
}

final class DeepseekV41Experts: Module {
    @ModuleInfo(key: "switch_mlp") var switchMLP: DeepseekV41SwitchGLU

    init(_ config: DeepseekV41Configuration) {
        _switchMLP.wrappedValue = DeepseekV41SwitchGLU(config)
        super.init()
    }
}

/// `noaux_tc` router: selects on `score + e_score_correction_bias`, weights by `score`.
final class DeepseekV41Router: Module {
    let topK: Int
    let scoringFunc: String
    let gateTemp: Float
    let normTopkProb: Bool
    let routedScalingFactor: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var bias: MLXArray

    init(_ config: DeepseekV41Configuration) {
        topK = config.numExpertsPerTok
        scoringFunc = config.scoringFunc
        gateTemp = config.gateTemp
        normTopkProb = config.normTopkProb
        routedScalingFactor = config.routedScalingFactor
        _weight.wrappedValue = MLXArray.zeros([config.nRoutedExperts, config.hiddenSize])
        _bias.wrappedValue = MLXArray.zeros([config.nRoutedExperts], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = matmul(x.asType(.float32), weight.asType(.float32).T) / gateTemp
        let scores: MLXArray =
            switch scoringFunc {
            case "sigmoid": sigmoid(logits)
            case "softmax": softmax(logits, axis: -1)
            default: sqrt(softplus(logits))
            }
        let biased = scores + bias.asType(.float32)
        let indices = argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, indices, axis: -1)
        if normTopkProb && topK > 1 {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, weights * routedScalingFactor)
    }
}

final class DeepseekV41MoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: DeepseekV41Router
    @ModuleInfo(key: "experts") var experts: DeepseekV41Experts
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV41MLP

    init(_ config: DeepseekV41Configuration) {
        _gate.wrappedValue = DeepseekV41Router(config)
        _experts.wrappedValue = DeepseekV41Experts(config)
        _sharedExperts.wrappedValue = DeepseekV41MLP(config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let flat = x.reshaped(-1, shape[shape.count - 1])
        let (indices, weights) = gate(flat)
        let routed =
            (experts.switchMLP(flat, indices).asType(.float32)
            * weights.expandedDimensions(axis: -1).asType(.float32)).sum(axis: 1)
        let out = routed + sharedExperts(flat).asType(.float32)
        return out.reshaped(shape).asType(x.dtype)
    }
}

// MARK: - Decoder

final class DeepseekV41DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DeepseekV41Attention
    @ModuleInfo(key: "mlp") var mlp: DeepseekV41MoE
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attentionHC: DeepseekV41HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: DeepseekV41HyperConnection
    @ModuleInfo(key: "engram") var engram: DeepseekV41Engram?

    init(_ config: DeepseekV41Configuration, layer: Int) {
        _attention.wrappedValue = DeepseekV41Attention(config, layer: layer)
        _mlp.wrappedValue = DeepseekV41MoE(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attentionHC.wrappedValue = DeepseekV41HyperConnection(config)
        _ffnHC.wrappedValue = DeepseekV41HyperConnection(config)
        if config.engramLayerIDs.contains(layer) {
            _engram.wrappedValue = DeepseekV41Engram(config)
        }
        super.init()
    }

    /// A DSpark stage: the same block with the draft attention of ``DeepseekV41Attention``.
    func dspark(
        _ streams: MLXArray, preMix: MLXArray, start: Int, window: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let (attentionPre, attentionPost, attentionComb) = attentionHC(streams)
        var h = inputLayerNorm(deepseekV41Collapse(streams, preMix))
        h = attention.dspark(h, start: start, window: window)
        var streams = deepseekV41Expand(
            h, residual: streams, post: attentionPost, comb: attentionComb)
        let (ffnPre, ffnPost, ffnComb) = ffnHC(streams)
        h = mlp(postAttentionLayerNorm(deepseekV41Collapse(streams, attentionPre)))
        streams = deepseekV41Expand(h, residual: streams, post: ffnPost, comb: ffnComb)
        return (streams, ffnPre)
    }

    /// Each sublayer's mixing coefficients collapse the streams for the next sublayer:
    /// attention uses the previous layer's `pre`, and the FFN uses this attention's.
    func callAsFunction(
        _ streams: MLXArray, preMix: MLXArray, startPos: Int, windowIndices: MLXArray,
        cache: DeepseekV41Cache, shared: DeepseekV41SharedState
    ) -> (MLXArray, MLXArray) {
        let (attentionPre, attentionPost, attentionComb) = attentionHC(streams)
        var h = inputLayerNorm(deepseekV41Collapse(streams, preMix))
        h = attention(
            h, startPos: startPos, windowIndices: windowIndices, cache: cache, shared: shared)
        var streams = deepseekV41Expand(
            h, residual: streams, post: attentionPost, comb: attentionComb)

        let (ffnPre, ffnPost, ffnComb) = ffnHC(streams)
        h = mlp(postAttentionLayerNorm(deepseekV41Collapse(streams, attentionPre)))
        streams = deepseekV41Expand(h, residual: streams, post: ffnPost, comb: ffnComb)
        return (streams, ffnPre)
    }
}

public final class DeepseekV41ModelInner: Module {
    let configuration: DeepseekV41Configuration
    let memory: DeepseekV41EngramMemory

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [DeepseekV41DecoderLayer]
    let norm: RMSNorm

    init(_ configuration: DeepseekV41Configuration, memory: DeepseekV41EngramMemory) {
        self.configuration = configuration
        self.memory = memory
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize, dimensions: configuration.hiddenSize)
        layers = (0 ..< configuration.hiddenLayers).map {
            DeepseekV41DecoderLayer(configuration, layer: $0)
        }
        norm = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        super.init()
    }

    func makeCache() -> [DeepseekV41Cache] {
        (0 ..< configuration.hiddenLayers).map {
            DeepseekV41Cache(
                windowSize: configuration.slidingWindow,
                ratio: configuration.compressRatio($0))
        }
    }

    /// Final normalized hidden states.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(inputs, cache: cache, capturing: []).hidden
    }

    /// Final normalized hidden states, plus the stream mean at the input of each layer in
    /// `capturing` (what the DSpark drafter reads).
    func forward(_ inputs: MLXArray, cache: [KVCache]?, capturing: [Int]) -> (
        hidden: MLXArray, captures: [MLXArray]
    ) {
        let inputs = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        let caches: [DeepseekV41Cache] =
            cache?.map {
                guard let cache = $0 as? DeepseekV41Cache else {
                    fatalError("deepseek_v41 needs DeepseekV41Cache, got \(type(of: $0))")
                }
                return cache
            } ?? makeCache()
        let h = embedTokens(inputs)
        let (b, l) = (h.dim(0), h.dim(1))
        let (hc, d) = (configuration.hcMult, configuration.hiddenSize)
        let startPos = caches[0].offset
        // Every layer's window holds the same rows, so all use the same indices.
        let window = deepseekV41WindowIndices(
            previous: caches[0].windowCount, length: l, window: configuration.slidingWindow)
        let windowIndices = broadcast(window.expandedDimensions(axis: 0), to: [b] + window.shape)

        let engramRows = memory.lookup(inputs, cache: caches[0])
        var streams = broadcast(h.expandedDimensions(axis: 2), to: [b, l, hc, d])
        var oneHot = [Float](repeating: 0, count: hc)
        oneHot[0] = 1
        var preMix = broadcast(MLXArray(oneHot), to: [b, l, hc])
        let shared = DeepseekV41SharedState()
        var captures = [MLXArray]()
        for (i, layer) in layers.enumerated() {
            if let engram = layer.engram, let rows = engramRows[i] {
                streams = engram(streams, rows: rows)
            }
            if capturing.contains(i) {
                captures.append(streams.mean(axis: 2))
            }
            (streams, preMix) = layer(
                streams, preMix: preMix, startPos: startPos, windowIndices: windowIndices,
                cache: caches[i], shared: shared)
        }
        return (norm(deepseekV41Collapse(streams, preMix)), captures)
    }
}

public final class DeepseekV41Model: Module, LLMModel, ModelArtifactPreparing {
    public let vocabularySize: Int
    public let model: DeepseekV41ModelInner
    let configuration: DeepseekV41Configuration
    /// Engram hash state and tables, loaded beside the checkpoint by ``prepareArtifact(in:)``.
    public let engramMemory: DeepseekV41EngramMemory

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ configuration: DeepseekV41Configuration) {
        self.configuration = configuration
        vocabularySize = configuration.vocabularySize
        engramMemory = DeepseekV41EngramMemory(configuration)
        model = DeepseekV41ModelInner(configuration, memory: engramMemory)
        _lmHead.wrappedValue = Linear(
            configuration.hiddenSize, configuration.vocabularySize, bias: false)
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        model.makeCache()
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

    /// With ``mtpEmitFlagKey`` set, also publishes the concatenated stream means at the input
    /// of the layers in ``mtpHiddenLayerIDsKey`` for the DSpark drafter.
    private func forward(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?, lastTokenOnly: Bool
    ) -> LMOutput {
        let emit = state?[mtpEmitFlagKey] ?? false
        let layerIDs = emit ? (state?[mtpHiddenLayerIDsKey] ?? []) : []
        let (hidden, captures) = model.forward(input.tokens, cache: cache, capturing: layerIDs)
        let finalOnly = emit ? (state?[mtpFinalPrefillLogitsOnlyKey] ?? false) : lastTokenOnly
        let logits = lmHead(finalOnly ? hidden[0..., (-1)..., 0...] : hidden)
        guard emit else { return LMOutput(logits: logits) }
        var out = state ?? LMOutput.State()
        out[mtpFinalPrefillLogitsOnlyKey] = nil
        out[mtpLastHiddenStatesKey] =
            captures.isEmpty ? hidden : concatenated(captures, axis: -1)
        // The drafter keeps its own window, so no target K/V is shared.
        out[mtpSharedKVSourceIndicesKey] = [:]
        return LMOutput(logits: logits, state: out)
    }

    public func prepareArtifact(in modelDirectory: URL) throws {
        try engramMemory.load(directory: modelDirectory)
    }

    public var excludedCheckpointNamespaces: [String] {
        DeepseekV41Checkpoint.excludedNamespaces(configuration)
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        try DeepseekV41Checkpoint.prepare(checkpoint, configuration: configuration)
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(ModelCheckpoint(weights: weights)).weights
    }
}

extension DeepseekV41Model: MTPTargetModel {
    public var mtpTargetArchitectureID: String {
        "deepseek_v41:\(configuration.hiddenSize):\(configuration.vocabularySize)"
    }
    public var mtpEmbedding: Embedding { model.embedTokens }
    public var mtpOutputHead: Linear? { lmHead }
}

extension DeepseekV41Model: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}
