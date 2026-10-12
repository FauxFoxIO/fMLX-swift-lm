// Copyright © 2026 Faux Fox, Inc.
// Portions © 2025 Prince Canuma (MIT), © 2026 Google LLC and Hugging Face (Apache-2.0).
// See LICENSES/EmbeddingGemma2-NOTICE.md for source attribution and license texts.
// Port of MLX-VLM's Gemma 4 audio tower, revision 3d87e88402f307efbf68e568971aa887ee7d9ed0.
// Derived from Hugging Face Transformers (Apache-2.0) and MLX-VLM (MIT).
// https://github.com/Blaizzy/mlx-vlm/blob/3d87e88402f307efbf68e568971aa887ee7d9ed0/mlx_vlm/models/gemma4/audio.py
// https://www.apache.org/licenses/LICENSE-2.0

import Foundation
import MLX
import MLXNN

struct EmbeddingGemma2AudioConfiguration: Codable, Sendable {
    var modelType: String = "gemma4_audio"
    var hiddenSize: Int = 1024
    var numHiddenLayers: Int = 12
    var numAttentionHeads: Int = 8
    var hiddenAct: String = "silu"
    var subsamplingConvChannels: [Int] = [128, 32]
    var convKernelSize: Int = 5
    var attentionChunkSize: Int = 12
    var attentionContextLeft: Int = 13
    var attentionContextRight: Int = 0
    var attentionLogitCap: Float = 50
    var attentionInvalidLogitsValue: Float = -1e9
    var gradientClipping: Float = 1e10
    var residualWeight: Float = 0.5
    var rmsNormEps: Float = 1e-6
    var outputProjDims: Int? = 1536
    var useClippedLinears: Bool = true
    var chunkSizeFeedForward: Int = 0
    var initializerRange: Float = 0.02
    var dtype: String = "bfloat16"

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case hiddenAct = "hidden_act"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case convKernelSize = "conv_kernel_size"
        case attentionChunkSize = "attention_chunk_size"
        case attentionContextLeft = "attention_context_left"
        case attentionContextRight = "attention_context_right"
        case attentionLogitCap = "attention_logit_cap"
        case attentionInvalidLogitsValue = "attention_invalid_logits_value"
        case gradientClipping = "gradient_clipping"
        case residualWeight = "residual_weight"
        case rmsNormEps = "rms_norm_eps"
        case outputProjDims = "output_proj_dims"
        case useClippedLinears = "use_clipped_linears"
        case chunkSizeFeedForward = "chunk_size_feed_forward"
        case initializerRange = "initializer_range"
        case dtype
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        numHiddenLayers =
            try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? numHiddenLayers
        numAttentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? numAttentionHeads
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? hiddenAct
        subsamplingConvChannels =
            try c.decodeIfPresent([Int].self, forKey: .subsamplingConvChannels)
            ?? subsamplingConvChannels
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? convKernelSize
        attentionChunkSize =
            try c.decodeIfPresent(Int.self, forKey: .attentionChunkSize) ?? attentionChunkSize
        attentionContextLeft =
            try c.decodeIfPresent(Int.self, forKey: .attentionContextLeft) ?? attentionContextLeft
        attentionContextRight =
            try c.decodeIfPresent(Int.self, forKey: .attentionContextRight) ?? attentionContextRight
        attentionLogitCap =
            try c.decodeIfPresent(Float.self, forKey: .attentionLogitCap) ?? attentionLogitCap
        attentionInvalidLogitsValue =
            try c.decodeIfPresent(Float.self, forKey: .attentionInvalidLogitsValue)
            ?? attentionInvalidLogitsValue
        gradientClipping =
            try c.decodeIfPresent(Float.self, forKey: .gradientClipping) ?? gradientClipping
        residualWeight =
            try c.decodeIfPresent(Float.self, forKey: .residualWeight) ?? residualWeight
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? rmsNormEps
        if c.contains(.outputProjDims) {
            outputProjDims = try c.decodeIfPresent(Int.self, forKey: .outputProjDims)
        }
        useClippedLinears =
            try c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? useClippedLinears
        chunkSizeFeedForward =
            try c.decodeIfPresent(Int.self, forKey: .chunkSizeFeedForward) ?? chunkSizeFeedForward
        initializerRange =
            try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? initializerRange
        dtype = try c.decodeIfPresent(String.self, forKey: .dtype) ?? dtype
    }
}

private final class AudioClippableLinear: Module, UnaryLayer {
    @ModuleInfo var linear: Linear
    @ParameterInfo(key: "input_min") var inputMin: MLXArray
    @ParameterInfo(key: "input_max") var inputMax: MLXArray
    @ParameterInfo(key: "output_min") var outputMin: MLXArray
    @ParameterInfo(key: "output_max") var outputMax: MLXArray

    init(_ inputDimensions: Int, _ outputDimensions: Int) {
        // The pinned audio tower always retains clipping buffers, regardless of the config flag.
        _linear.wrappedValue = Linear(inputDimensions, outputDimensions, bias: false)
        _inputMin.wrappedValue = MLXArray(-Float.infinity)
        _inputMax.wrappedValue = MLXArray(Float.infinity)
        _outputMin.wrappedValue = MLXArray(-Float.infinity)
        _outputMax.wrappedValue = MLXArray(Float.infinity)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        clip(linear(clip(x, min: inputMin, max: inputMax)), min: outputMin, max: outputMax)
    }
}

private final class AudioSubsamplingConvBlock: Module {
    @ModuleInfo var conv: Conv2d
    @ModuleInfo var norm: LayerNorm

    init(_ config: EmbeddingGemma2AudioConfiguration, index: Int) {
        let inputChannels = index == 0 ? 1 : config.subsamplingConvChannels[index - 1]
        let outputChannels = config.subsamplingConvChannels[index]
        _conv.wrappedValue = Conv2d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: 3, stride: 2, padding: 0, bias: false)
        _norm.wrappedValue = LayerNorm(
            dimensions: outputChannels, eps: config.rmsNormEps, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, validMask: MLXArray) -> (MLXArray, MLXArray) {
        var x = MLX.where(validMask[0..., 0..., .newAxis, .newAxis], x, MLXArray(0, dtype: x.dtype))
        x = padded(x, widths: [0, 1, 1, 0])
        x = conv(x)
        let outputMask = validMask[0..., stride(from: 0, to: validMask.dim(1), by: 2)][
            0..., ..<x.dim(1)]
        return (relu(norm(x)), outputMask)
    }
}

private final class AudioSubsampleConvProjection: Module {
    @ModuleInfo var layer0: AudioSubsamplingConvBlock
    @ModuleInfo var layer1: AudioSubsamplingConvBlock
    @ModuleInfo(key: "input_proj_linear") var inputProjLinear: Linear

    init(_ config: EmbeddingGemma2AudioConfiguration) {
        _layer0.wrappedValue = AudioSubsamplingConvBlock(config, index: 0)
        _layer1.wrappedValue = AudioSubsamplingConvBlock(config, index: 1)
        var frequency = 128
        for _ in 0 ..< 2 {
            frequency = (frequency + 2 - 3) / 2 + 1
        }
        _inputProjLinear.wrappedValue = Linear(
            frequency * config.subsamplingConvChannels[1], config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ features: MLXArray, validMask: MLXArray) -> (MLXArray, MLXArray) {
        let (x0, mask0) = layer0(features.expandedDimensions(axis: -1), validMask: validMask)
        let (x1, mask1) = layer1(x0, validMask: mask0)
        let x = x1.reshaped(x1.dim(0), x1.dim(1), x1.dim(2) * x1.dim(3))
        return (inputProjLinear(x), mask1)
    }
}

private final class AudioConformerFeedForward: Module, UnaryLayer {
    let gradientClipping: Float
    let residualWeight: Float
    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: RMSNorm
    @ModuleInfo(key: "ffw_layer_1") var ffwLayer1: AudioClippableLinear
    @ModuleInfo(key: "ffw_layer_2") var ffwLayer2: AudioClippableLinear
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: RMSNorm

    init(_ config: EmbeddingGemma2AudioConfiguration) {
        gradientClipping = config.gradientClipping
        residualWeight = config.residualWeight
        // The pinned tower uses AudioRMSNorm's default epsilon here and in block norms.
        _preLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
        _ffwLayer1.wrappedValue = AudioClippableLinear(config.hiddenSize, config.hiddenSize * 4)
        _ffwLayer2.wrappedValue = AudioClippableLinear(config.hiddenSize * 4, config.hiddenSize)
        _postLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = clip(x, min: -gradientClipping, max: gradientClipping)
        h = preLayerNorm(h)
        h = ffwLayer2(silu(ffwLayer1(h)))
        h = clip(h, min: -gradientClipping, max: gradientClipping)
        return x + postLayerNorm(h) * residualWeight
    }
}

private final class AudioChunkAttention: Module {
    let numHeads: Int
    let headDim: Int
    let chunkSize: Int
    let maxPastHorizon: Int
    let maxFutureHorizon: Int
    let contextSize: Int
    let invalidLogitsValue: Float
    let softcap: Float
    let qScale: Float
    let kScale: Float
    private let _invTimescales: MLXArray

    @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear
    @ParameterInfo(key: "per_dim_scale") var perDimScale: MLXArray
    @ModuleInfo(key: "q_proj") var qProj: AudioClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: AudioClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: AudioClippableLinear
    @ModuleInfo var post: AudioClippableLinear

    init(_ config: EmbeddingGemma2AudioConfiguration) {
        numHeads = config.numAttentionHeads
        headDim = config.hiddenSize / config.numAttentionHeads
        chunkSize = config.attentionChunkSize
        maxPastHorizon = max(0, config.attentionContextLeft - 1)
        maxFutureHorizon = config.attentionContextRight
        contextSize = chunkSize + maxPastHorizon + maxFutureHorizon
        invalidLogitsValue = config.attentionInvalidLogitsValue
        softcap = config.attentionLogitCap
        qScale = Float(pow(Double(headDim), -0.5) / log(2.0))
        kScale = Float(log(1 + exp(1.0)) / log(2.0))

        let numTimescales = config.hiddenSize / 2
        let increment = Float(log(10_000.0) / Double(max(numTimescales - 1, 1)))
        _invTimescales = exp(MLXArray(0 ..< numTimescales).asType(.float32) * -increment)
            .reshaped(1, 1, numTimescales)
        _relativeKProj.wrappedValue = Linear(
            config.hiddenSize, numHeads * headDim, bias: false)
        _perDimScale.wrappedValue = MLXArray.zeros([headDim])
        _qProj.wrappedValue = AudioClippableLinear(config.hiddenSize, numHeads * headDim)
        _kProj.wrappedValue = AudioClippableLinear(config.hiddenSize, numHeads * headDim)
        _vProj.wrappedValue = AudioClippableLinear(config.hiddenSize, numHeads * headDim)
        _post.wrappedValue = AudioClippableLinear(config.hiddenSize, config.hiddenSize)
        super.init()
    }

    private func padTime(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        var widths = Array(repeating: IntOrPair(0), count: x.ndim)
        widths[1] = IntOrPair((left, right))
        return padded(x, widths: widths)
    }

    private func convertToBlocks(_ x: MLXArray) -> MLXArray {
        let blocks = (x.dim(1) + chunkSize - 1) / chunkSize
        let padding = blocks * chunkSize - x.dim(1)
        let x = padding > 0 ? padTime(x, left: 0, right: padding) : x
        return x.reshaped([x.dim(0), blocks, chunkSize] + Array(x.shape.dropFirst(2)))
    }

    private func extractBlockContext(_ x: MLXArray) -> MLXArray {
        let x = padTime(x, left: maxPastHorizon, right: maxFutureHorizon + chunkSize - 1)
        let blocks = (x.dim(1) - contextSize) / chunkSize + 1
        let starts = MLXArray(0 ..< blocks) * chunkSize
        let offsets = MLXArray(0 ..< contextSize)
        let indices = starts[0..., .newAxis] + offsets[.newAxis, 0...]
        return x.take(indices, axis: 1)
    }

    private func relativePositionLogits(_ queries: MLXArray, _ keys: MLXArray) -> MLXArray {
        let batch = queries.dim(0)
        let blocks = queries.dim(1)
        let positions = MLXArray(
            stride(from: maxPastHorizon, through: -maxFutureHorizon, by: -1)
        )
        .asType(.float32).reshaped(1, -1, 1)
        let span = positions.dim(1)
        let scaledTime = positions * _invTimescales
        var signal = concatenated([sin(scaledTime), cos(scaledTime)], axis: -1)
            .asType(queries.dtype)
        let dtype = (relativeKProj as? QuantizedLinear)?.scales.dtype ?? relativeKProj.weight.dtype
        signal = relativeKProj(signal.asType(dtype))
            .reshaped(span, numHeads, headDim).asType(queries.dtype)

        let q = queries.transposed(0, 3, 1, 2, 4)
        let k = keys.transposed(0, 3, 1, 4, 2)
        let termAC = matmul(q, k)
        var termBD = matmul(
            q.reshaped(batch, numHeads, blocks * chunkSize, headDim),
            signal.transposed(1, 2, 0)
        )
        .reshaped(batch, numHeads, blocks, chunkSize, span)

        // Match the pinned relative shift: pad each row, flatten, truncate, reshape.
        termBD = padded(termBD, widths: [0, 0, 0, 0, IntOrPair((0, contextSize + 1 - span))])
            .reshaped(batch, numHeads, blocks, chunkSize * (contextSize + 1))
        termBD = termBD[0..., 0..., 0..., ..<(chunkSize * contextSize)]
            .reshaped(batch, numHeads, blocks, chunkSize, contextSize)
        return termAC + termBD
    }

    func callAsFunction(
        _ hiddenStates: MLXArray, validMask: MLXArray, causalValidMask: MLXArray
    ) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let time = hiddenStates.dim(1)
        let shape = [batch, time, numHeads, headDim]
        let q =
            qProj(hiddenStates).asType(.float32).reshaped(shape)
            * (qScale * softplus(perDimScale))
        let k = kProj(hiddenStates).asType(.float32).reshaped(shape) * kScale
        let v = vProj(hiddenStates).asType(.float32).reshaped(shape)
        let queries = convertToBlocks(q)
        let keys = extractBlockContext(k)
        let values = extractBlockContext(v)
        let extractedValid = extractBlockContext(validMask)
        let condition = logicalAnd(
            extractedValid[0..., .newAxis, 0..., .newAxis, 0...],
            causalValidMask[.newAxis, .newAxis, .newAxis, 0..., 0...])

        var logits = relativePositionLogits(queries, keys)
        logits = tanh(logits / softcap) * softcap
        logits = MLX.where(condition, logits, MLXArray(invalidLogitsValue))
        let probabilities = softmax(logits, axis: -1)
        let context = einsum("bnuwc,bucnh->buwnh", probabilities, values)
            .reshaped(batch, queries.dim(1) * chunkSize, numHeads, headDim)
        return post(context[0..., ..<time].reshaped(batch, time, numHeads * headDim))
    }
}

private final class AudioDepthwiseConv1d: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray

    init(dimensions: Int, kernelSize: Int) {
        _weight.wrappedValue = MLXArray.zeros([dimensions, kernelSize, 1])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convGeneral(x, weight, strides: 1, padding: 0, groups: x.dim(-1))
    }
}

private final class AudioConformerLightConv1d: Module, UnaryLayer {
    let gradientClipping: Float
    let causalPadding: Int
    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: RMSNorm
    @ModuleInfo(key: "linear_start") var linearStart: AudioClippableLinear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv1d: AudioDepthwiseConv1d
    @ModuleInfo(key: "conv_norm") var convNorm: RMSNorm
    @ModuleInfo(key: "linear_end") var linearEnd: AudioClippableLinear

    init(_ config: EmbeddingGemma2AudioConfiguration) {
        gradientClipping = config.gradientClipping
        causalPadding = config.convKernelSize - 1
        _preLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _linearStart.wrappedValue = AudioClippableLinear(config.hiddenSize, config.hiddenSize * 2)
        _depthwiseConv1d.wrappedValue = AudioDepthwiseConv1d(
            dimensions: config.hiddenSize, kernelSize: config.convKernelSize)
        _convNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _linearEnd.wrappedValue = AudioClippableLinear(config.hiddenSize, config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(linearStart(preLayerNorm(x)), parts: 2, axis: -1)
        var h = parts[0] * sigmoid(parts[1])
        h = padded(h, widths: [0, IntOrPair((causalPadding, 0)), 0])
        h = depthwiseConv1d(h)
        h = clip(h, min: -gradientClipping, max: gradientClipping)
        return x + linearEnd(silu(convNorm(h)))
    }
}

private final class AudioConformerBlock: Module {
    let gradientClipping: Float
    @ModuleInfo(key: "feed_forward1") var feedForward1: AudioConformerFeedForward
    @ModuleInfo(key: "self_attn") var selfAttn: AudioChunkAttention
    @ModuleInfo(key: "lconv1d") var lconv1d: AudioConformerLightConv1d
    @ModuleInfo(key: "feed_forward2") var feedForward2: AudioConformerFeedForward
    @ModuleInfo(key: "norm_pre_attn") var normPreAttn: RMSNorm
    @ModuleInfo(key: "norm_post_attn") var normPostAttn: RMSNorm
    @ModuleInfo(key: "norm_out") var normOut: RMSNorm

    init(_ config: EmbeddingGemma2AudioConfiguration) {
        gradientClipping = config.gradientClipping
        _feedForward1.wrappedValue = AudioConformerFeedForward(config)
        _selfAttn.wrappedValue = AudioChunkAttention(config)
        _lconv1d.wrappedValue = AudioConformerLightConv1d(config)
        _feedForward2.wrappedValue = AudioConformerFeedForward(config)
        _normPreAttn.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
        _normPostAttn.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
        _normOut.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, validMask: MLXArray, causalValidMask: MLXArray
    ) -> MLXArray {
        let residual = feedForward1(x)
        var h = clip(residual, min: -gradientClipping, max: gradientClipping)
        h = selfAttn(normPreAttn(h), validMask: validMask, causalValidMask: causalValidMask)
        h = clip(h, min: -gradientClipping, max: gradientClipping)
        h = residual + normPostAttn(h)
        h = h * validMask.expandedDimensions(axis: -1).asType(h.dtype)
        h = feedForward2(lconv1d(h))
        return normOut(clip(h, min: -gradientClipping, max: gradientClipping))
    }
}

final class EmbeddingGemma2AudioEncoder: Module {
    let config: EmbeddingGemma2AudioConfiguration
    @ModuleInfo(key: "subsample_conv_projection") private var subsampleConvProjection:
        AudioSubsampleConvProjection
    @ModuleInfo private var layers: [AudioConformerBlock]
    @ModuleInfo(key: "output_proj") private var outputProj: Linear?

    init(_ config: EmbeddingGemma2AudioConfiguration) {
        self.config = config
        precondition(config.subsamplingConvChannels.count == 2)
        precondition(config.hiddenSize > 0 && config.numAttentionHeads > 0)
        precondition(config.hiddenSize % config.numAttentionHeads == 0)
        precondition(config.attentionChunkSize > 0 && config.attentionContextRight >= 0)
        _subsampleConvProjection.wrappedValue = AudioSubsampleConvProjection(config)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in AudioConformerBlock(config)
        }
        if let dimensions = config.outputProjDims {
            _outputProj.wrappedValue = Linear(config.hiddenSize, dimensions, bias: true)
        }
        super.init()
    }

    private func buildCausalValidMask() -> MLXArray {
        let past = max(0, config.attentionContextLeft - 1)
        let future = config.attentionContextRight
        let contextSize = config.attentionChunkSize + past + future
        let query = MLXArray(0 ..< config.attentionChunkSize)[0..., .newAxis]
        let context = MLXArray(0 ..< contextSize)[.newAxis, 0...]
        let distance = query + past - context
        let pastValid = logicalAnd(distance .>= 0, distance .< past)
        let futureValid = logicalAnd(distance .< 0, -distance .< future)
        return logicalOr(pastValid, futureValid)
    }

    /// FP32 mel features [B, T, 128] and a true-valid mask [B, T].
    func callAsFunction(
        _ features: MLXArray, validMask: MLXArray
    ) -> (features: MLXArray, validMask: MLXArray) {
        precondition(features.ndim == 3 && features.dim(2) == 128 && features.dtype == .float32)
        precondition(validMask.shape == [features.dim(0), features.dim(1)])
        precondition(validMask.dtype == .bool)
        var (encodings, currentMask) = subsampleConvProjection(features, validMask: validMask)
        let causalValidMask = buildCausalValidMask()
        for layer in layers {
            encodings = layer(encodings, validMask: currentMask, causalValidMask: causalValidMask)
        }
        if let outputProj {
            encodings = outputProj(encodings)
        }
        if currentMask.dim(1) != encodings.dim(1) {
            currentMask = currentMask[0..., ..<encodings.dim(1)]
        }
        encodings = MLX.where(
            currentMask.expandedDimensions(axis: -1), encodings, MLXArray(0, dtype: encodings.dtype)
        )
        return (encodings, currentMask)
    }
}
