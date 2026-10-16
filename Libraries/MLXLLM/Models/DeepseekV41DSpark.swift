//
//  DeepseekV41DSpark.swift
//  LLM
//
//  DeepSeek-V4.1's DSpark block drafter, stored as `mtp.*` in the target checkpoint.
//  Follows `DSparkBlock`, `DSparkAttention` and `DSparkMarkovHead` in DeepSeek's
//  `inference/model.py`.
//
//  `main_proj` of the target's stream means at `dspark_target_layer_ids` gives each committed
//  position a KV row in every stage's sliding window. A draft runs the bonus token plus noise
//  tokens through the stages as one block, applies the shared head, and adds a Markov bigram
//  bias from the previous token before sampling each proposal in turn.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Bigram logit bias from the previous token.
final class DeepseekV41MarkovHead: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "head") var head: Linear

    init(vocabularySize: Int, rank: Int) {
        _embed.wrappedValue = Embedding(embeddingCount: vocabularySize, dimensions: rank)
        _head.wrappedValue = Linear(rank, vocabularySize, bias: false)
        super.init()
    }

    /// Float32 bias `[B, vocab]` for previous tokens `[B]`.
    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let x = embed(tokens)
        if head is QuantizedLinear {
            return head(x).asType(.float32)
        }
        // The reference keeps this head in float32.
        return matmul(x.asType(.float32), head.weight.asType(.float32).T)
    }
}

public final class DeepseekV41DSparkDraftModel: Module, IncrementalMTPDrafterModel {
    let configuration: DeepseekV41Configuration
    let routedExperts: Int

    @ModuleInfo(key: "main_proj") var mainProj: Linear
    @ModuleInfo(key: "main_norm") var mainNorm: RMSNorm
    @ModuleInfo(key: "stages") var stages: [DeepseekV41DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "markov_head") var markovHead: DeepseekV41MarkovHead

    public init(_ configuration: DeepseekV41Configuration) {
        self.configuration = configuration
        var stage = configuration
        if configuration.dsparkRoutedExperts > 0 {
            stage.nRoutedExperts = configuration.dsparkRoutedExperts
        }
        if configuration.dsparkExpertsPerTok > 0 {
            stage.numExpertsPerTok = configuration.dsparkExpertsPerTok
        }
        routedExperts = stage.nRoutedExperts
        let d = configuration.hiddenSize
        _mainProj.wrappedValue = Linear(
            d * configuration.dsparkTargetLayerIDs.count, d, bias: false)
        _mainNorm.wrappedValue = RMSNorm(dimensions: d, eps: configuration.rmsNormEps)
        _stages.wrappedValue = (0 ..< configuration.numNextnPredictLayers).map {
            DeepseekV41DecoderLayer(stage, layer: configuration.hiddenLayers + $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: d, eps: configuration.rmsNormEps)
        _markovHead.wrappedValue = DeepseekV41MarkovHead(
            vocabularySize: configuration.vocabularySize, rank: configuration.dsparkMarkovRank)
        super.init()
    }

    public var targetArchitectureID: String {
        "deepseek_v41:\(configuration.hiddenSize):\(configuration.vocabularySize)"
    }
    public var targetHiddenLayerIDs: [Int]? { configuration.dsparkTargetLayerIDs }
    /// The trained block holds the bonus and `dspark_block_size - 1` noise tokens and proposes
    /// one token per row, so a round verifies at most `dspark_block_size + 1` tokens.
    public var maximumBlockSize: Int? { configuration.dsparkBlockSize + 1 }
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    /// The stage windows are bounded, so the cache does not grow with the sequence.
    public var cacheBytesPerToken: Int { 0 }

    public func isCompatible(with target: any LanguageModel) -> Bool {
        (target as? any MTPTargetModel)?.mtpTargetArchitectureID == targetArchitectureID
    }

    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(
            cache: stages.map { _ in
                DeepseekV41Cache(windowSize: configuration.slidingWindow, ratio: 0, rewindLimit: 0)
            })
    }

    private func windows(_ state: MTPDrafterState) -> [DeepseekV41Cache] {
        state.cache.map {
            guard let cache = $0 as? DeepseekV41Cache else {
                fatalError("DSpark needs DeepseekV41Cache windows, got \(type(of: $0))")
            }
            return cache
        }
    }

    /// Adds the window KV of committed target positions `[B, n, targets * D]`.
    private func append(_ targetHidden: MLXArray, to state: inout MTPDrafterState) {
        guard targetHidden.dim(1) > 0 else { return }
        let mainX = mainNorm(mainProj(targetHidden))
        for (stage, cache) in zip(stages, windows(state)) {
            _ = cache.updateWindow(stage.attention.windowKV(mainX, start: state.nextPosition))
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

    public func commitDrafterState(
        target _: any LanguageModel, targetHidden: MLXArray, draftTokens _: MLXArray,
        acceptedCount: Int, finalToken _: MLXArray, positionDeltas _: MLXArray?,
        state: inout MTPDrafterState, sampler _: any LogitSampler
    ) {
        append(targetHidden[0..., ..<(acceptedCount + 1), 0...], to: &state)
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

    /// Proposes `blockSize - 1` tokens after the bonus token at `state.nextPosition`.
    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)], positionDeltas _: MLXArray?,
        queryOffset _: Int, blockSize: Int, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MTPDraft {
        let length = configuration.dsparkBlockSize
        precondition(blockSize >= 2 && blockSize <= length + 1, "DSpark proposes 1...\(length)")
        let (embedding, head) = shared(target)
        let bonus = lastToken.ndim == 1 ? lastToken.expandedDimensions(axis: -1) : lastToken
        let b = bonus.dim(0)
        let noise = broadcast(
            MLXArray(Array(repeating: Int32(configuration.dsparkNoiseTokenID), count: length - 1))
                .expandedDimensions(axis: 0), to: [b, length - 1])
        // The trained block always has `length` rows; shorter rounds keep its first proposals.
        let inputs = concatenated([bonus.asType(.int32), noise], axis: 1)
        let h = embedding(inputs)
        let (hc, d) = (configuration.hcMult, configuration.hiddenSize)
        var streams = broadcast(h.expandedDimensions(axis: 2), to: [b, length, hc, d])
        var oneHot = [Float](repeating: 0, count: hc)
        oneHot[0] = 1
        var preMix = broadcast(MLXArray(oneHot), to: [b, length, hc])
        for (stage, cache) in zip(stages, windows(state)) {
            (streams, preMix) = stage.dspark(
                streams, preMix: preMix, start: state.nextPosition, window: cache.window)
        }
        let hidden = norm(deepseekV41Collapse(streams, preMix))
        let logits = (head.map { $0(hidden) } ?? embedding.asLinear(hidden)).asType(.float32)

        var previous = bonus[0..., 0]
        var tokens = [MLXArray]()
        var rows = [MLXArray]()
        var logProbabilities = [MLXArray]()
        for i in 0 ..< (blockSize - 1) {
            let row = logits[0..., i] + markovHead(previous)
            if sampler is ArgMaxSampler {
                previous = argMax(row, axis: -1)
            } else if let draftSampler = sampler as? any SpeculativeDraftSampler {
                let lp = draftSampler.draftLogProbabilities(logits: row)
                previous = draftSampler.sampleDraft(logProbabilities: lp)
                logProbabilities.append(lp)
            } else {
                previous = sampler.sample(logits: row)
            }
            previous = previous.asType(lastToken.dtype)
            rows.append(row)
            tokens.append(previous)
        }
        return MTPDraft(
            tokens: stacked(tokens, axis: 1), logits: stacked(rows, axis: 1),
            logProbabilities: logProbabilities.isEmpty ? nil : stacked(logProbabilities, axis: 1))
    }

    private func shared(_ target: any LanguageModel) -> (Embedding, Linear?) {
        guard let target = target as? any MTPTargetModel else {
            fatalError("DSpark requires a deepseek_v41 target, got \(type(of: target))")
        }
        return (target.mtpEmbedding, target.mtpOutputHead)
    }

    // MARK: Checkpoint

    public var excludedCheckpointNamespaces: [String] {
        [
            "layers", "embed", "head", "norm", "vision", "aligner", "image_start", "image_end",
            "image_newline", "model", "lm_head",
        ]
    }

    /// Maps `mtp.<stage>.*` to the stages and the drafter heads; the confidence head, which
    /// fMLX's acceptance does not use, is dropped.
    static func rename(_ name: String) -> String? {
        let parts = name.split(separator: ".", maxSplits: 2).map(String.init)
        guard parts.count == 3, parts[0] == "mtp", Int(parts[1]) != nil else { return nil }
        let rest = parts[2]
        let head = String(rest.prefix { $0 != "." })
        switch head {
        case "main_proj", "main_norm", "norm", "markov_head": return rest
        case "confidence_head": return nil
        default:
            guard
                let mapped = DeepseekV41Checkpoint.rename("layers.\(parts[1]).\(rest)"),
                mapped.hasPrefix("model.layers.")
            else { return nil }
            return "stages." + mapped.dropFirst("model.layers.".count)
        }
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        try DeepseekV41Checkpoint.prepare(
            checkpoint, rename: Self.rename,
            layerPrefixes: stages.indices.map { "stages.\($0)" },
            routedExperts: routedExperts,
            oGroups: configuration.oGroups)
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(ModelCheckpoint(weights: weights)).weights
    }

    /// Loads the drafter from a `deepseek_v41` checkpoint directory, reading only `mtp.*`, or
    /// returns `nil` when the checkpoint has no DSpark stages.
    public static func load(directory: URL) async throws -> DeepseekV41DSparkDraftModel? {
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        guard base.modelType == "deepseek_v41" else { return nil }
        let configuration = try JSONDecoder.json5().decode(
            DeepseekV41Configuration.self, from: data)
        guard configuration.numNextnPredictLayers > 0, configuration.dsparkBlockSize > 0,
            !configuration.dsparkTargetLayerIDs.isEmpty
        else { return nil }
        let selection = WeightTensorNameSelection.prefixed("mtp.")
        guard try containsWeightTensor(in: directory, tensorNameSelection: selection) else {
            return nil
        }
        let drafter = DeepseekV41DSparkDraftModel(configuration)
        try await loadWeights(
            modelDirectory: directory, model: drafter, tensorNameSelection: selection,
            perLayerQuantization: base.perLayerQuantization)
        return drafter
    }
}

/// Registers the DSpark drafter with ``MTPDrafterTypeRegistry/shared``, so
/// `MTPDrafterModelFactory` can load it from the target's own checkpoint directory.
public enum DeepseekV41DSparkRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType("deepseek_v41") { data in
            DeepseekV41DSparkDraftModel(
                try JSONDecoder.json5().decode(DeepseekV41Configuration.self, from: data))
        }
    }
}
