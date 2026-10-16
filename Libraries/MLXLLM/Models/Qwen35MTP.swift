// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Qwen35MTPPredictor: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen35DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        var mtpArgs = args
        mtpArgs.hiddenLayers = max(args.mtpNumHiddenLayers, 1)
        mtpArgs.fullAttentionInterval = 1
        let layerTypes = Array(
            repeating: HybridAttentionSchedule.fullAttention, count: mtpArgs.hiddenLayers)
        mtpArgs.layerTypes = layerTypes
        mtpArgs.resolvedLayerTypes = layerTypes

        if args.mtpUseDedicatedEmbeddings {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize,
                dimensions: args.hiddenSize
            )
        }
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        _layers.wrappedValue = (0 ..< mtpArgs.hiddenLayers).map {
            Qwen35DecoderLayer(mtpArgs, layerIdx: $0, forceFullAttention: true)
        }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFCNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFCNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func newCache() -> [KVCache] {
        layers.map { _ in KVCacheSimple() }
    }

    func callAsFunction(
        inputsEmbeds: MLXArray,
        hiddenStates previousHidden: MLXArray,
        cache: [KVCache],
        positionOffset: Int
    ) -> MLXArray {
        var hiddenStates = concatenated(
            [preFCNormEmbedding(inputsEmbeds), preFCNormHidden(previousHidden)], axis: -1)
        hiddenStates = fc(hiddenStates)

        precondition(cache.count == layers.count, "Qwen MTP cache/layer count mismatch")
        for (layer, layerCache) in zip(layers, cache) {
            let faMask = createAttentionMask(h: hiddenStates, cache: layerCache)
            hiddenStates = layer(
                hiddenStates,
                attentionMask: faMask,
                ssmMask: nil,
                cache: layerCache,
                positionOffset: positionOffset)
        }

        return norm(hiddenStates)
    }
}

public final class Qwen35MTPDraftModel: Module, IncrementalMTPDrafterModel,
    ScheduledMTPPrefixCachingDrafter
{
    public var targetArchitectureID: String { "qwen3_5:\(configuration.hiddenSize)" }
    public var cacheBytesPerToken: Int {
        max(configuration.mtpNumHiddenLayers, 1) * configuration.kvHeads
            * (configuration.headDim ?? configuration.hiddenSize / configuration.attentionHeads) * 8
    }
    public let configuration: Qwen35TextConfiguration
    /// One Qwen MTP layer is reused autoregressively; the target retains
    /// recurrent checkpoints for at most three drafted positions plus bonus.
    public let maximumBlockSize: Int? = 4
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    private let checkpointPolicy: Qwen35CheckpointPolicy

    @ModuleInfo(key: "mtp") var mtp: Qwen35MTPPredictor

    public convenience init(
        _ configuration: Qwen35TextConfiguration,
        preconvertedNorms: Bool = false,
        mixedPreservedNorms: Bool = false
    ) {
        self.init(
            configuration,
            checkpointPolicy: .init(
                modelType: configuration.modelType,
                preconvertedNorms: preconvertedNorms,
                mixedPreservedNorms: mixedPreservedNorms))
    }

    private init(
        _ configuration: Qwen35TextConfiguration, checkpointPolicy: Qwen35CheckpointPolicy
    ) {
        self.configuration = configuration
        self.checkpointPolicy = checkpointPolicy
        _mtp.wrappedValue = Qwen35MTPPredictor(configuration)
        super.init()
    }

    public convenience init(
        _ configuration: Qwen35Configuration,
        preconvertedNorms: Bool = false
    ) {
        self.init(
            configuration.textConfig,
            checkpointPolicy: .init(
                modelType: configuration.modelType,
                preconvertedNorms: preconvertedNorms))
    }

    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: mtp.newCache())
    }

    public func prepareDrafterChunk(
        target: any LanguageModel, shiftedTokens: MLXArray, targetHidden: MLXArray,
        isFinal: Bool, state: inout MTPDrafterState, sampler: any LogitSampler
    ) {
        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let hidden = mtp(
            inputsEmbeds: inputEmbedding(shiftedTokens), hiddenStates: targetHidden,
            cache: state.cache, positionOffset: state.nextPosition)
        state.nextPosition += shiftedTokens.dim(1)
        if isFinal {
            state.seedHidden = hidden[0..., (-1)..., 0...]
            let seed = makeMTPSeed(
                hidden: state.seedHidden!,
                targetEmbedTokens: targetEmbedTokens, lmHead: lmHead, sampler: sampler)
            state.seedToken = seed.token
            state.seedLogits = seed.logits
        }
        // The iterator evaluates retained cache state and final seed arrays.
    }

    public func isCompatible(with target: any LanguageModel) -> Bool {
        target is Qwen35Model || target is Qwen35TextModel
    }

    public func prepareDrafterState(
        target: any LanguageModel,
        promptTokens: MLXArray,
        targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let prompt = normalizedMTPTokenBatch(promptTokens)
        let bonus = normalizedMTPColumn(firstBonus)
        guard prompt.dim(-1) > 0 else { return }

        let shifted = concatenated([prompt[0..., 1...], bonus], axis: 1)
        let hidden = targetHidden[0..., ..<shifted.dim(1), 0...]
        state.nextPosition = 0
        let mtpHidden = mtp(
            inputsEmbeds: inputEmbedding(shifted),
            hiddenStates: hidden,
            cache: state.cache,
            positionOffset: 0)
        state.nextPosition = shifted.dim(1)
        state.seedHidden = mtpHidden[0..., (-1)..., 0...]
        let seed = makeMTPSeed(
            hidden: state.seedHidden!, targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead, sampler: sampler)
        state.seedToken = seed.token
        state.seedLogits = seed.logits
        state.proposalAppended = 0
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MTPDraft {
        var state = makeState(parameters: nil)
        return draftBlock(
            target: target,
            lastToken: lastToken,
            lastHidden: lastHidden,
            sharedKV: sharedKV,
            positionDeltas: positionDeltas,
            queryOffset: queryOffset,
            blockSize: blockSize,
            state: &state,
            sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MTPDraft {
        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens

        if let seed = state.seedToken, let seedHidden = state.seedHidden,
            let seedLogits = state.seedLogits
        {
            state.seedToken = nil
            state.seedHidden = nil
            state.seedLogits = nil
            guard blockSize > 2 else {
                state.proposalAppended = 0
                return MTPDraft(tokens: seed, logits: seedLogits)
            }

            // Prompt prefill already advanced the MTP cache through the bonus
            // and produced `seed`. Continue from that state instead of feeding
            // the bonus a second time. The final continuation token is only
            // predicted, so exactly `blockSize - 2` entries are tentative.
            let continuation = draftMTPTokenBlock(
                targetEmbedTokens: targetEmbedTokens,
                lmHead: lmHead,
                inputEmbedding: inputEmbedding,
                lastToken: seed,
                lastHidden: seedHidden,
                queryOffset: state.nextPosition,
                blockSize: blockSize - 1,
                sampler: sampler,
                cache: state.cache
            ) { inputsEmbeds, hiddenStates, cache, positionOffset in
                mtp(
                    inputsEmbeds: inputsEmbeds,
                    hiddenStates: hiddenStates,
                    cache: cache,
                    positionOffset: positionOffset)
            }
            state.proposalAppended = blockSize - 2
            state.nextPosition += state.proposalAppended
            return MTPDraft(
                tokens: concatenated([normalizedMTPColumn(seed), continuation.tokens], axis: 1),
                logits: concatenated([seedLogits, continuation.logits], axis: 1))
        }

        state.proposalAppended = blockSize - 1
        let proposed = draftMTPTokenBlock(
            targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead,
            inputEmbedding: inputEmbedding,
            lastToken: lastToken,
            lastHidden: lastHidden,
            queryOffset: queryOffset,
            blockSize: blockSize,
            sampler: sampler,
            cache: state.cache
        ) { inputsEmbeds, hiddenStates, cache, positionOffset in
            mtp(
                inputsEmbeds: inputsEmbeds,
                hiddenStates: hiddenStates,
                cache: cache,
                positionOffset: positionOffset)
        }
        state.nextPosition += state.proposalAppended
        return proposed
    }

    public func commitDrafterState(
        target: any LanguageModel,
        targetHidden: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let keepAppended = Swift.min(acceptedCount, state.proposalAppended)
        let trim = state.proposalAppended - keepAppended
        if trim > 0 {
            trimPromptCache(state.cache, numTokens: trim)
            state.nextPosition -= trim
        }

        var tokens = [MLXArray]()
        var hiddens = [MLXArray]()
        for index in keepAppended ..< acceptedCount {
            tokens.append(draftTokens[0..., index ..< (index + 1)])
            hiddens.append(targetHidden[0..., index ..< (index + 1), 0...])
        }
        tokens.append(normalizedMTPColumn(finalToken))
        hiddens.append(targetHidden[0..., acceptedCount ..< (acceptedCount + 1), 0...])

        let (targetEmbedTokens, lmHead) = targetEmbeddingAndHead(target)
        let inputEmbedding = mtp.embedTokens ?? targetEmbedTokens
        let committedTokens = concatenated(tokens, axis: 1)
        let committedHidden = concatenated(hiddens, axis: 1)
        let mtpHidden = mtp(
            inputsEmbeds: inputEmbedding(committedTokens),
            hiddenStates: committedHidden,
            cache: state.cache,
            positionOffset: state.nextPosition)
        state.nextPosition += committedTokens.dim(1)
        state.seedHidden = mtpHidden[0..., (-1)..., 0...]
        let seed = makeMTPSeed(
            hidden: state.seedHidden!, targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead, sampler: sampler)
        state.seedToken = seed.token
        state.seedLogits = seed.logits
        state.proposalAppended = 0
    }

    public var excludedCheckpointNamespaces: [String] {
        checkpointPolicy.excludedCheckpointNamespaces
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        try checkpointPolicy.prepare(
            checkpoint, mtpNumHiddenLayers: configuration.mtpNumHiddenLayers,
            numExperts: configuration.numExperts)
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try sanitize(weights: weights, metadata: [:])
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) throws -> [String:
        MLXArray]
    {
        try prepareCheckpoint(.init(weights: weights, metadata: metadata)).weights
    }

    private func targetEmbeddingAndHead(_ target: any LanguageModel) -> (Embedding, Linear?) {
        guard let target = target as? any MTPTargetModel else {
            fatalError("Qwen35MTPDraftModel requires a Qwen35 target, got \(type(of: target))")
        }
        return (target.mtpEmbedding, target.mtpOutputHead)
    }
}

extension Qwen35TextModel: MTPTargetModel {
    public var mtpTargetArchitectureID: String { "qwen3_5:\(configuration.hiddenSize)" }
    public var mtpEmbedding: Embedding { model.embedTokens }
    public var mtpOutputHead: Linear? { lmHead }
}

extension Qwen35Model: MTPTargetModel {
    public var mtpTargetArchitectureID: String { languageModel.mtpTargetArchitectureID }
    public var mtpEmbedding: Embedding { languageModel.mtpEmbedding }
    public var mtpOutputHead: Linear? { languageModel.mtpOutputHead }
}

extension Qwen35MTPDraftModel: ModelConversionMetadataProvider {
    public var modelConversionMetadata: [String: String] {
        [Qwen35CheckpointPolicy.normMetadataKey: "scale"]
    }
}
