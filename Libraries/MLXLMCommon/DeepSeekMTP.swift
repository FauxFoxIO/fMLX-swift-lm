import Foundation
import MLX
import MLXNN

/// A target model whose MTP head shares its token embedding and output projection.
///
/// The target publishes the hidden states its MTP head consumes through
/// ``mtpLastHiddenStatesKey`` when ``mtpEmitFlagKey`` is set. DeepSeek-style heads and Qwen
/// heads both consume the target's final, post-norm hidden states.
public protocol MTPTargetModel: LanguageModel {
    /// Architecture and width a drafter must have been trained against.
    var mtpTargetArchitectureID: String { get }

    /// Token embedding shared with the MTP head.
    var mtpEmbedding: Embedding { get }

    /// Output projection shared with the MTP head, or `nil` when tied to ``mtpEmbedding``.
    var mtpOutputHead: Linear? { get }
}

/// The decoder layer inside a DeepSeek-style MTP head. It attends causally over its own
/// cache and derives positions from that cache.
public protocol MTPDecoderBlock: Module {
    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray
    func makeCache() -> KVCache
}

/// The MTP head's `shared_head`: the final norm. The projection is the target's output head.
public final class MTPSharedHead: Module {
    @ModuleInfo(key: "norm") public var norm: RMSNorm

    public init(dimensions: Int, eps: Float) {
        _norm.wrappedValue = RMSNorm(dimensions: dimensions, eps: eps)
        super.init()
    }
}

/// DeepSeek-V3-style multi-token prediction drafter.
///
/// One step computes `eh_proj(cat(enorm(embed(token)), hnorm(hidden)))`, runs one decoder
/// layer of the target's type, and projects `shared_head.norm` of the result with the
/// target's output head. Token `i + 1` pairs with the target hidden state at position `i`.
/// The layer is reused for every drafted position and keeps a private KV cache over the
/// shifted sequence, which the iterator trims after each verification.
open class DeepSeekMTPDraftModel: Module, IncrementalMTPDrafterModel {
    @ModuleInfo(key: "enorm") var enorm: RMSNorm
    @ModuleInfo(key: "hnorm") var hnorm: RMSNorm
    @ModuleInfo(key: "eh_proj") var ehProj: Linear
    @ModuleInfo(key: "shared_head") var sharedHead: MTPSharedHead
    @ModuleInfo(key: "mtp_block") var block: Module

    public let targetArchitectureID: String
    public let cacheBytesPerToken: Int
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true

    public init(
        hiddenSize: Int, rmsNormEps: Float, block: some MTPDecoderBlock,
        targetArchitectureID: String, cacheBytesPerToken: Int
    ) {
        self.targetArchitectureID = targetArchitectureID
        self.cacheBytesPerToken = cacheBytesPerToken
        _enorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        _hnorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        _ehProj.wrappedValue = Linear(hiddenSize * 2, hiddenSize, bias: false)
        _sharedHead.wrappedValue = MTPSharedHead(dimensions: hiddenSize, eps: rmsNormEps)
        _block.wrappedValue = block
        super.init()
    }

    /// The decoder layer, for subclasses that shard it.
    public var mtpBlock: Module { block }

    private var decoderBlock: any MTPDecoderBlock {
        block as! any MTPDecoderBlock
    }

    /// Normalized MTP hidden states for `inputEmbeddings` paired with target `hidden` rows.
    public func callAsFunction(
        inputEmbeddings: MLXArray, hidden: MLXArray, cache: KVCache
    ) -> MLXArray {
        let x = ehProj(concatenated([enorm(inputEmbeddings), hnorm(hidden)], axis: -1))
        return sharedHead.norm(decoderBlock(x, cache: cache))
    }

    open var excludedCheckpointNamespaces: [String] { [] }

    open func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        checkpoint
    }

    public func isCompatible(with target: any LanguageModel) -> Bool {
        (target as? any MTPTargetModel)?.mtpTargetArchitectureID == targetArchitectureID
    }

    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: [decoderBlock.makeCache()])
    }

    private func shared(_ target: any LanguageModel) -> (Embedding, Linear?) {
        guard let target = target as? any MTPTargetModel else {
            fatalError("\(Self.self) requires an MTPTargetModel, got \(type(of: target))")
        }
        return (target.mtpEmbedding, target.mtpOutputHead)
    }

    /// Advances the drafter cache over `tokens` and returns the last row's hidden state.
    private func advance(
        _ tokens: MLXArray, hidden: MLXArray, embedding: Embedding, state: inout MTPDrafterState
    ) -> MLXArray {
        let out = self(inputEmbeddings: embedding(tokens), hidden: hidden, cache: state.cache[0])
        state.nextPosition += tokens.dim(1)
        return out[0..., (-1)..., 0...]
    }

    private func seed(
        _ hidden: MLXArray, target: any LanguageModel, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let (embedding, head) = shared(target)
        let seed = makeMTPSeed(
            hidden: hidden, targetEmbedTokens: embedding, lmHead: head, sampler: sampler)
        state.seedHidden = hidden
        state.seedToken = seed.token
        state.seedLogits = seed.logits
        state.proposalAppended = 0
    }

    public func prepareDrafterChunk(
        target: any LanguageModel, shiftedTokens: MLXArray, targetHidden: MLXArray,
        isFinal: Bool, state: inout MTPDrafterState, sampler: any LogitSampler
    ) {
        let hidden = advance(
            shiftedTokens, hidden: targetHidden, embedding: shared(target).0, state: &state)
        if isFinal {
            seed(hidden, target: target, state: &state, sampler: sampler)
        }
    }

    public func prepareDrafterState(
        target: any LanguageModel, promptTokens: MLXArray, targetHidden: MLXArray,
        firstBonus: MLXArray, positionDeltas _: MLXArray?, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let prompt = normalizedMTPTokenBatch(promptTokens)
        guard prompt.dim(-1) > 0 else { return }
        let shifted = concatenated([prompt[0..., 1...], normalizedMTPColumn(firstBonus)], axis: 1)
        let hidden = advance(
            shifted, hidden: targetHidden[0..., ..<shifted.dim(1), 0...],
            embedding: shared(target).0, state: &state)
        seed(hidden, target: target, state: &state, sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?,
        queryOffset: Int, blockSize: Int, sampler: any LogitSampler
    ) -> MTPDraft {
        var state = makeState(parameters: nil)
        return draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
            positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize,
            state: &state, sampler: sampler)
    }

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)], positionDeltas _: MLXArray?,
        queryOffset _: Int, blockSize: Int, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MTPDraft {
        let (embedding, head) = shared(target)
        let cache = state.cache
        let step: (MLXArray, MLXArray, [KVCache], Int) -> MLXArray = { embeds, hidden, cache, _ in
            self(inputEmbeddings: embeds, hidden: hidden, cache: cache[0])
        }

        // The previous commit already fed the bonus token and sampled the first proposal.
        if let seedToken = state.seedToken, let seedHidden = state.seedHidden,
            let seedLogits = state.seedLogits
        {
            state.seedToken = nil
            state.seedHidden = nil
            state.seedLogits = nil
            guard blockSize > 2 else {
                state.proposalAppended = 0
                return MTPDraft(tokens: seedToken, logits: seedLogits)
            }
            let continuation = draftMTPTokenBlock(
                targetEmbedTokens: embedding, lmHead: head, inputEmbedding: embedding,
                lastToken: seedToken, lastHidden: seedHidden, queryOffset: state.nextPosition,
                blockSize: blockSize - 1, sampler: sampler, cache: cache, forward: step)
            state.proposalAppended = blockSize - 2
            state.nextPosition += state.proposalAppended
            return MTPDraft(
                tokens: concatenated(
                    [normalizedMTPColumn(seedToken), continuation.tokens], axis: 1),
                logits: concatenated([seedLogits, continuation.logits], axis: 1))
        }

        let draft = draftMTPTokenBlock(
            targetEmbedTokens: embedding, lmHead: head, inputEmbedding: embedding,
            lastToken: lastToken, lastHidden: lastHidden, queryOffset: state.nextPosition,
            blockSize: blockSize, sampler: sampler, cache: cache, forward: step)
        state.proposalAppended = blockSize - 1
        state.nextPosition += state.proposalAppended
        return draft
    }

    /// Drops this round's proposal rows, then feeds the accepted tokens and the final target
    /// token paired with the verified target hidden states. The last row seeds the next round.
    public func commitDrafterState(
        target: any LanguageModel, targetHidden: MLXArray, draftTokens: MLXArray,
        acceptedCount: Int, finalToken: MLXArray, positionDeltas _: MLXArray?,
        state: inout MTPDrafterState, sampler: any LogitSampler
    ) {
        // Proposal rows were conditioned on drafted hidden states; recompute them from the target.
        if state.proposalAppended > 0 {
            trimPromptCache(state.cache, numTokens: state.proposalAppended)
            state.nextPosition -= state.proposalAppended
            state.proposalAppended = 0
        }
        var tokens = normalizedMTPColumn(finalToken)
        if acceptedCount > 0 {
            let accepted = normalizedMTPTokenBatch(draftTokens)[0..., ..<acceptedCount]
            tokens = concatenated([accepted.asType(tokens.dtype), tokens], axis: 1)
        }
        let hidden = targetHidden[0..., ..<(acceptedCount + 1), 0...]
        let last = advance(tokens, hidden: hidden, embedding: shared(target).0, state: &state)
        seed(last, target: target, state: &state, sampler: sampler)
    }
}
