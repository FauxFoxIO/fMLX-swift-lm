// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum DFlash2ConfigurationError: Error, Equatable, LocalizedError {
    case invalid(String)

    public var errorDescription: String? {
        switch self {
        case .invalid(let reason): "Invalid DFlash2 configuration: \(reason)"
        }
    }
}

public struct DFlash2Configuration: Decodable, Sendable, ModelConfigurationValidating {
    public struct Draft: Decodable, Sendable {
        public let blockSize: Int
        public let convolutionGroupSize: Int
        public let convolutionKernelSize: Int
        public let maskTokenID: Int
        public let selectorRank: Int
        public let selectorTopK: Int
        public let targetLayerIDs: [Int]

        enum CodingKeys: String, CodingKey {
            case blockSize = "block_size"
            case convolutionGroupSize = "conv_group_size"
            case convolutionKernelSize = "conv_kernel_size"
            case maskTokenID = "mask_token_id"
            case selectorRank = "selector_rank"
            case selectorTopK = "selector_top_k"
            case targetLayerIDs = "target_layer_ids"
        }
    }

    public let architectures: [String]
    public let modelType: String
    public let isCausal: Bool
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    public let targetLayers: Int
    public let layerTypes: [String]
    public let slidingWindow: Int
    public let draft: Draft

    enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case isCausal = "is_causal"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case targetLayers = "num_target_layers"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case draft = "dflash_config"
    }

    enum RopeKeys: String, CodingKey { case ropeTheta = "rope_theta" }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        architectures = try values.decode([String].self, forKey: .architectures)
        modelType = try values.decode(String.self, forKey: .modelType)
        isCausal = try values.decode(Bool.self, forKey: .isCausal)
        hiddenSize = try values.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try values.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try values.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try values.decode(Int.self, forKey: .kvHeads)
        headDim = try values.decode(Int.self, forKey: .headDim)
        intermediateSize = try values.decode(Int.self, forKey: .intermediateSize)
        vocabularySize = try values.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try values.decode(Float.self, forKey: .rmsNormEps)
        maxPositionEmbeddings = try values.decode(Int.self, forKey: .maxPositionEmbeddings)
        targetLayers = try values.decode(Int.self, forKey: .targetLayers)
        layerTypes = try values.decode([String].self, forKey: .layerTypes)
        slidingWindow = try values.decode(Int.self, forKey: .slidingWindow)
        draft = try values.decode(Draft.self, forKey: .draft)
        if let direct = try values.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            ropeTheta = direct
        } else {
            let rope = try values.nestedContainer(keyedBy: RopeKeys.self, forKey: .ropeParameters)
            ropeTheta = try rope.decode(Float.self, forKey: .ropeTheta)
        }
    }

    public func validateModelConfiguration() throws {
        guard architectures == ["DFlash2DraftModel"], modelType == "qwen3", !isCausal else {
            throw DFlash2ConfigurationError.invalid("unsupported architecture")
        }
        guard hiddenSize == 5_120, hiddenLayers == 5, attentionHeads == 32,
            kvHeads == 8, headDim == 128, intermediateSize == 17_408,
            vocabularySize == 248_320, targetLayers == 64
        else { throw DFlash2ConfigurationError.invalid("unsupported trained dimensions") }
        guard draft.blockSize == 8, draft.convolutionGroupSize == 16,
            draft.convolutionKernelSize == 2, draft.maskTokenID == 248_070,
            draft.selectorRank == 256, draft.selectorTopK == 16,
            draft.targetLayerIDs == [5, 19, 33, 47, 61]
        else { throw DFlash2ConfigurationError.invalid("unsupported trained draft contract") }
        guard slidingWindow == 2_048,
            layerTypes == Array(repeating: "sliding_attention", count: hiddenLayers)
        else { throw DFlash2ConfigurationError.invalid("a 2048-token sliding context is required") }
        guard hiddenSize.isMultiple(of: draft.convolutionGroupSize) else {
            throw DFlash2ConfigurationError.invalid("convolution groups do not divide hidden size")
        }
    }
}

final class DFlash2Attention: Module {
    let configuration: DFlash2Configuration
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProjection: Linear
    @ModuleInfo(key: "k_proj") var kProjection: Linear
    @ModuleInfo(key: "v_proj") var vProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    private let fusedQueryKeyValueProjection = FusedQuantizedLinearProjectionCache()
    private var fusedKeyValueProjection: QuantizedLinear?

    private let compiledFourTokenPreRope = CompiledTrace<DFlash2Attention>(
        state: { attention in
            let projections: [Module]
            if let fused = attention.fusedQueryKeyValueProjection.fused {
                projections = [fused]
            } else {
                projections = [
                    attention.qProjection, attention.kProjection, attention.vProjection,
                ]
            }
            return projections + [attention.queryNorm, attention.keyNorm]
        },
        body: { attention, arguments in attention.preRopeProjection(arguments[0]) })

    init(_ configuration: DFlash2Configuration) {
        self.configuration = configuration
        scale = pow(Float(configuration.headDim), -0.5)
        rope = initializeRope(
            dims: configuration.headDim, base: configuration.ropeTheta,
            traditional: false, scalingConfig: nil,
            maxPositionEmbeddings: configuration.maxPositionEmbeddings)
        _qProjection.wrappedValue = Linear(
            configuration.hiddenSize, configuration.attentionHeads * configuration.headDim,
            bias: false)
        _kProjection.wrappedValue = Linear(
            configuration.hiddenSize, configuration.kvHeads * configuration.headDim, bias: false)
        _vProjection.wrappedValue = Linear(
            configuration.hiddenSize, configuration.kvHeads * configuration.headDim, bias: false)
        _outputProjection.wrappedValue = Linear(
            configuration.attentionHeads * configuration.headDim, configuration.hiddenSize,
            bias: false)
        _queryNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDim, eps: configuration.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDim, eps: configuration.rmsNormEps)
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        let replacesQueryKeyOrValue = parameters.flattened().contains { key, _ in
            key.hasPrefix("q_proj.") || key.hasPrefix("k_proj.")
                || key.hasPrefix("v_proj.")
        }
        defer {
            if replacesQueryKeyOrValue {
                fusedQueryKeyValueProjection.invalidate()
                fusedKeyValueProjection = nil
            }
        }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func updateModule(key: String, _ value: Any) throws {
        let replacesQueryKeyOrValue = key == "q_proj" || key == "k_proj" || key == "v_proj"
        defer {
            if replacesQueryKeyOrValue {
                fusedQueryKeyValueProjection.invalidate()
                fusedKeyValueProjection = nil
            }
        }
        try super.updateModule(key: key, value)
    }

    @discardableResult
    func prepareFusedQueryKeyValueProjection() throws -> Bool {
        let prepared = try fusedQueryKeyValueProjection.prepare(
            enabled: true,
            linears: [qProjection, kProjection, vProjection]
        ) { sourceViews in
            try update(
                modules: ModuleChildren(values: [
                    "q_proj": .value(sourceViews[0]),
                    "k_proj": .value(sourceViews[1]),
                    "v_proj": .value(sourceViews[2]),
                ]), verify: [])
        }
        guard prepared, let fused = fusedQueryKeyValueProjection.fused,
            ObjectIdentifier(type(of: fused)) == ObjectIdentifier(QuantizedLinear.self)
        else { return false }

        let start = qProjection.shape.0
        let rows = start ..< fused.shape.0
        let keyValue = QuantizedLinear(
            weight: fused.weight[rows], bias: nil, scales: fused.scales[rows],
            biases: fused.biases.map { $0[rows] }, groupSize: fused.groupSize,
            bits: fused.bits, mode: fused.mode)
        keyValue.freeze()
        fusedKeyValueProjection = keyValue
        return true
    }

    private func keysAndValues(
        _ hidden: MLXArray, offset: RoPEOffset
    ) -> (MLXArray, MLXArray) {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let projectedKeys: MLXArray
        let projectedValues: MLXArray
        if let fused = fusedKeyValueProjection {
            let projected = fused(hidden)
            let keyEnd = kProjection.shape.0
            projectedKeys = projected[.ellipsis, ..<keyEnd]
            projectedValues = projected[.ellipsis, keyEnd...]
        } else {
            projectedKeys = kProjection(hidden)
            projectedValues = vProjection(hidden)
        }
        var keys = keyNorm(
            projectedKeys.reshaped(batch, length, configuration.kvHeads, -1)
        ).transposed(0, 2, 1, 3)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)
        let values = projectedValues.reshaped(
            batch, length, configuration.kvHeads, -1
        ).transposed(0, 2, 1, 3)
        return (keys, values)
    }

    func appendContext(_ hidden: MLXArray, cache: KVCache) {
        let (keys, values) = keysAndValues(hidden, offset: cache.ropeOffset)
        _ = cache.update(keys: keys, values: values)
    }

    private func preRopeProjection(_ hidden: MLXArray) -> [MLXArray] {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let projectedQueries: MLXArray
        let projectedKeys: MLXArray
        let projectedValues: MLXArray
        if let fused = fusedQueryKeyValueProjection.fused {
            let projected = fused(hidden)
            let queryEnd = qProjection.shape.0
            let keyEnd = queryEnd + kProjection.shape.0
            projectedQueries = projected[.ellipsis, ..<queryEnd]
            projectedKeys = projected[.ellipsis, queryEnd ..< keyEnd]
            projectedValues = projected[.ellipsis, keyEnd...]
        } else {
            projectedQueries = qProjection(hidden)
            projectedKeys = kProjection(hidden)
            projectedValues = vProjection(hidden)
        }
        let queries = queryNorm(
            projectedQueries.reshaped(batch, length, configuration.attentionHeads, -1)
        ).transposed(0, 2, 1, 3)
        let keys = keyNorm(
            projectedKeys.reshaped(batch, length, configuration.kvHeads, -1)
        ).transposed(0, 2, 1, 3)
        let values = projectedValues.reshaped(
            batch, length, configuration.kvHeads, -1
        ).transposed(0, 2, 1, 3)
        return [queries, keys, values]
    }

    func attendedHeads(_ hidden: MLXArray, cache: RotatingKVCache) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let proposalOffset = cache.offset
        let projected =
            length == 4
            ? compiledFourTokenPreRope(self, [hidden]) : preRopeProjection(hidden)
        var queries = projected[0]
        queries = applyRotaryPosition(rope, to: queries, offset: .scalar(proposalOffset))
        let rotatedProposalKeys = applyRotaryPosition(
            rope, to: projected[1], offset: .scalar(proposalOffset))
        let proposalValues = projected[2]

        let allKeys: MLXArray
        let allValues: MLXArray
        let contextLength: Int
        if let (contextKeys, contextValues) = cache.logicalView(
            tail: configuration.slidingWindow - 1)
        {
            contextLength = contextKeys.dim(2)
            allKeys = concatenated([contextKeys, rotatedProposalKeys], axis: 2)
            allValues = concatenated([contextValues, proposalValues], axis: 2)
        } else {
            contextLength = 0
            allKeys = rotatedProposalKeys
            allValues = proposalValues
        }
        let queryPositions = MLXArray(contextLength ..< (contextLength + length))[
            0..., .newAxis]
        let keyPositions = MLXArray(0 ..< (contextLength + length))[.newAxis, 0...]
        let contextMask = logicalAnd(
            keyPositions .< contextLength,
            queryPositions - keyPositions .< configuration.slidingWindow)
        let proposalMask = keyPositions .>= contextLength
        let mask = logicalOr(contextMask, proposalMask)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: allKeys, values: allValues, scale: scale,
            mask: .array(mask)
        ).transposed(0, 2, 1, 3).reshaped(batch, length, -1)
    }

    func callAsFunction(_ hidden: MLXArray, cache: RotatingKVCache) -> MLXArray {
        outputProjection(attendedHeads(hidden, cache: cache))
    }
}

private func dflashDynamicConvolution(
    hidden: MLXArray, dynamic: MLXArray, base: MLXArray, groupSize: Int
) -> MLXArray {
    let batch = hidden.dim(0)
    let length = hidden.dim(1)
    let width = hidden.dim(2)
    let groups = width / groupSize
    let blocks = hidden.reshaped(batch, length, groups, groupSize)
    let kernels = dynamic.reshaped(batch, length, base.dim(0), groups, 1)
    var output = MLX.zeros(like: blocks)
    for offset in 0 ..< base.dim(0) {
        let values: MLXArray
        if offset == 0 {
            values = blocks
        } else {
            values = concatenated(
                [
                    MLX.zeros(like: blocks[0..., ..<offset, 0..., 0...]),
                    blocks[0..., ..<(length - offset), 0..., 0...],
                ], axis: 1)
        }
        let fixed = base[offset].reshaped(1, 1, groups, groupSize).asType(hidden.dtype)
        output = output + fixed * values + kernels[0..., 0..., offset, 0..., 0...] * values
    }
    return output.reshaped(hidden.shape)
}

final class DFlash2DynamicConvolution: Module {
    let kernelSize: Int
    let groupSize: Int

    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        self.kernelSize = kernelSize
        self.groupSize = groupSize
        let groups = hiddenSize / groupSize
        _baseKernel.wrappedValue = MLX.zeros([2, kernelSize, hiddenSize])
        _kernelProjection.wrappedValue = Linear(
            hiddenSize, 2 * kernelSize * groups, bias: false)
    }

    func prepare(_ hidden: MLXArray) -> (MLXArray, MLXArray) {
        let groups = hidden.dim(-1) / groupSize
        let dynamic = kernelProjection(hidden).reshaped(
            hidden.dim(0), hidden.dim(1), 2, kernelSize, groups)
        return (
            dflashDynamicConvolution(
                hidden: hidden, dynamic: dynamic[0..., 0..., 0, 0..., 0...],
                base: baseKernel[0], groupSize: groupSize),
            dynamic[0..., 0..., 1, 0..., 0...]
        )
    }

    func finish(_ hidden: MLXArray, dynamic: MLXArray) -> MLXArray {
        dflashDynamicConvolution(
            hidden: hidden, dynamic: dynamic, base: baseKernel[1], groupSize: groupSize)
    }
}

final class DFlash2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DFlash2Attention
    @ModuleInfo var mlp: Qwen3MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConvolution: DFlash2DynamicConvolution
    @ModuleInfo(key: "mlp_conv") var mlpConvolution: DFlash2DynamicConvolution

    init(_ configuration: DFlash2Configuration) {
        _attention.wrappedValue = DFlash2Attention(configuration)
        _mlp.wrappedValue = Qwen3MLP(
            dimensions: configuration.hiddenSize,
            hiddenDimensions: configuration.intermediateSize)
        _inputNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _attentionConvolution.wrappedValue = DFlash2DynamicConvolution(
            hiddenSize: configuration.hiddenSize,
            kernelSize: configuration.draft.convolutionKernelSize,
            groupSize: configuration.draft.convolutionGroupSize)
        _mlpConvolution.wrappedValue = DFlash2DynamicConvolution(
            hiddenSize: configuration.hiddenSize,
            kernelSize: configuration.draft.convolutionKernelSize,
            groupSize: configuration.draft.convolutionGroupSize)
    }

    private let compiledPrepare = CompiledTrace<DFlash2DecoderLayer> { layer, arguments in
        let (input, kernel) = layer.attentionConvolution.prepare(
            layer.inputNorm(arguments[0]))
        return [input, kernel]
    }

    private let compiledFinish = CompiledTrace<DFlash2DecoderLayer> { layer, arguments in
        let projectedAttention = layer.attention.outputProjection(arguments[1])
        let attentionOutput =
            arguments[0]
            + layer.attentionConvolution.finish(projectedAttention, dynamic: arguments[2])
        let (mlpInput, mlpKernel) = layer.mlpConvolution.prepare(
            layer.postAttentionNorm(attentionOutput))
        return [
            attentionOutput
                + layer.mlpConvolution.finish(layer.mlp(mlpInput), dynamic: mlpKernel)
        ]
    }

    func appendContext(_ hidden: MLXArray, cache: KVCache) {
        attention.appendContext(hidden, cache: cache)
    }

    func prepareFusedProjections() throws {
        _ = try attention.prepareFusedQueryKeyValueProjection()
        _ = try mlp.prepareFusedGateUpProjection()
    }

    func callAsFunction(_ hidden: MLXArray, cache: RotatingKVCache) -> MLXArray {
        let prepared = compiledPrepare(self, [hidden])
        let attended = attention.attendedHeads(prepared[0], cache: cache)
        return compiledFinish(self, [hidden, attended, prepared[1]])[0]
    }
}

final class DFlash2CandidateSelector: Module {
    let vocabularySize: Int
    let topK: Int

    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    init(_ configuration: DFlash2Configuration) {
        vocabularySize = configuration.vocabularySize
        topK = configuration.draft.selectorTopK
        _predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: vocabularySize, dimensions: configuration.draft.selectorRank)
        _successorCodebook.wrappedValue = Embedding(
            embeddingCount: vocabularySize, dimensions: configuration.draft.selectorRank)
        _hiddenProjection.wrappedValue = Linear(
            configuration.hiddenSize, configuration.draft.selectorRank, bias: false)
    }

    func select(
        hidden: MLXArray, logits: MLXArray, anchor: MLXArray, sampler: any LogitSampler
    ) -> MTPDraft {
        let candidates = MLX.argPartition(-logits, kth: topK - 1, axis: -1)[
            .ellipsis, ..<topK]
        let unary = MLX.takeAlong(logits, candidates, axis: -1)
        let projected = hiddenProjection(hidden)
        var predecessor = anchor
        var tokens = [MLXArray]()
        var denseScores = [MLXArray]()
        var proposalLogProbabilities = [MLXArray]()
        for position in 0 ..< hidden.dim(1) {
            let candidateRow = candidates[0..., position, 0...]
            let edges = MLX.sum(
                predecessorCodebook(predecessor).expandedDimensions(axis: 1)
                    * projected[0..., position, 0...].expandedDimensions(axis: 1)
                    * successorCodebook(candidateRow), axis: -1)
            let scores = unary[0..., position, 0...] + edges
            let empty = MLXArray.full(
                [hidden.dim(0), vocabularySize], values: MLXArray(-Float.infinity),
                dtype: scores.dtype)
            let dense = MLX.putAlong(empty, candidateRow, values: scores, axis: -1)
            if sampler is ArgMaxSampler {
                let selected = argMax(scores, axis: -1).expandedDimensions(axis: -1)
                predecessor = MLX.takeAlong(candidateRow, selected, axis: -1).squeezed(axis: -1)
            } else if let draftSampler = sampler as? any SpeculativeDraftSampler {
                let logProbabilities = draftSampler.draftLogProbabilities(logits: dense)
                predecessor = draftSampler.sampleDraft(logProbabilities: logProbabilities)
                proposalLogProbabilities.append(logProbabilities)
            } else {
                predecessor = sampler.sample(logits: dense)
            }
            denseScores.append(dense)
            tokens.append(predecessor)
        }
        return MTPDraft(
            tokens: MLX.stacked(tokens, axis: 1), logits: MLX.stacked(denseScores, axis: 1),
            logProbabilities: proposalLogProbabilities.isEmpty
                ? nil : MLX.stacked(proposalLogProbabilities, axis: 1))
    }
}

public final class DFlash2DraftModel: Module, IncrementalMTPDrafterModel,
    ScheduledMTPPrefixCachingDrafter, LoadedWeightsPreparing
{
    public let configuration: DFlash2Configuration
    public var targetArchitectureID: String { "qwen3_5:\(configuration.hiddenSize)" }
    public var targetHiddenLayerIDs: [Int]? { configuration.draft.targetLayerIDs }
    // Width eight changes a greedy target decision on the qualified 27B fixture even when
    // compiled verification is disabled. Keep the largest shape with ordinary-output parity.
    public var maximumBlockSize: Int? { min(configuration.draft.blockSize, 4) }
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public var cacheBytesPerToken: Int {
        configuration.hiddenLayers * configuration.kvHeads * configuration.headDim * 4
    }

    @ModuleInfo var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo var layers: [DFlash2DecoderLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var candidateSelector: DFlash2CandidateSelector

    private static func appendProjectionTrace() -> CompiledTrace<DFlash2DraftModel> {
        CompiledTrace(
            state: { [$0.fc, $0.hiddenNorm] },
            body: { model, arguments in [model.hiddenNorm(model.fc(arguments[0]))] })
    }

    private let appendProjectionOne = appendProjectionTrace()
    private let appendProjectionTwo = appendProjectionTrace()
    private let appendProjectionThree = appendProjectionTrace()
    private let appendProjectionFour = appendProjectionTrace()

    public init(_ configuration: DFlash2Configuration) {
        self.configuration = configuration
        _fc.wrappedValue = Linear(
            configuration.draft.targetLayerIDs.count * configuration.hiddenSize,
            configuration.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map { _ in
            DFlash2DecoderLayer(configuration)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _candidateSelector.wrappedValue = DFlash2CandidateSelector(configuration)
        super.init()
    }

    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(
            cache: layers.map { _ in
                RotatingKVCache(maxSize: configuration.slidingWindow - 1)
            })
    }

    public func prepareLoadedWeights() throws {
        for layer in layers {
            try layer.prepareFusedProjections()
        }
    }

    private func append(_ targetHidden: MLXArray, to state: inout MTPDrafterState) {
        guard targetHidden.dim(1) > 0 else { return }
        let context: MLXArray
        switch targetHidden.dim(1) {
        case 1:
            context = appendProjectionOne(self, targetHidden)
        case 2:
            context = appendProjectionTwo(self, targetHidden)
        case 3:
            context = appendProjectionThree(self, targetHidden)
        case 4:
            context = appendProjectionFour(self, targetHidden)
        default:
            context = hiddenNorm(fc(targetHidden))
        }
        for (layer, cache) in zip(layers, state.cache) {
            layer.appendContext(context, cache: cache)
        }
        state.nextPosition += targetHidden.dim(1)
    }

    public func prepareDrafterChunk(
        target _: any LanguageModel, shiftedTokens _: MLXArray, targetHidden: MLXArray,
        isFinal _: Bool, state: inout MTPDrafterState, sampler _: any LogitSampler
    ) {
        append(targetHidden, to: &state)
    }

    public func prepareDrafterState(
        target _: any LanguageModel, promptTokens _: MLXArray, targetHidden: MLXArray,
        firstBonus _: MLXArray, positionDeltas _: MLXArray?, state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        append(targetHidden, to: &state)
    }

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?,
        queryOffset: Int, blockSize: Int, sampler: any LogitSampler
    ) -> MTPDraft {
        var state = makeState(parameters: nil)
        append(lastHidden, to: &state)
        return draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
            positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize,
            state: &state, sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)], positionDeltas _: MLXArray?,
        queryOffset _: Int, blockSize: Int, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MTPDraft {
        precondition(blockSize <= configuration.draft.blockSize)
        let (embedding, head) = targetEmbeddingAndHead(target)
        let bonus = lastToken.ndim == 1 ? lastToken.expandedDimensions(axis: 0) : lastToken
        let masks = MLXArray(
            Array(repeating: configuration.draft.maskTokenID, count: blockSize - 1)
        ).expandedDimensions(axis: 0)
        let inputs = concatenated([bonus, masks], axis: 1)
        var hidden = embedding(inputs)
        for (layer, rawCache) in zip(layers, state.cache) {
            guard let cache = rawCache as? RotatingKVCache else {
                preconditionFailure("DFlash2 requires rotating context caches")
            }
            hidden = layer(hidden, cache: cache)
        }
        hidden = norm(hidden[0..., 1..., 0...])
        let logits = head.map { $0(hidden) } ?? embedding.asLinear(hidden)
        return candidateSelector.select(
            hidden: hidden, logits: logits, anchor: inputs[0..., 0], sampler: sampler)
    }

    public func commitDrafterState(
        target _: any LanguageModel, targetHidden: MLXArray, draftTokens _: MLXArray,
        acceptedCount: Int, finalToken _: MLXArray, positionDeltas _: MLXArray?,
        state: inout MTPDrafterState, sampler _: any LogitSampler
    ) {
        append(targetHidden[0..., ..<(acceptedCount + 1), 0...], to: &state)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        for name in ["predecessor_codebook", "successor_codebook"] {
            let source = "candidate_selector.\(name)"
            if let value = weights.removeValue(forKey: source) {
                weights["\(source).weight"] = value
            }
        }
        return weights
    }

    private func targetEmbeddingAndHead(_ target: any LanguageModel) -> (Embedding, Linear?) {
        if let model = target as? Qwen35Model {
            return (model.languageModel.model.embedTokens, model.languageModel.lmHead)
        }
        if let model = target as? Qwen35TextModel {
            return (model.model.embedTokens, model.lmHead)
        }
        fatalError("DFlash2 requires a Qwen3.5 target, got \(type(of: target))")
    }
}
