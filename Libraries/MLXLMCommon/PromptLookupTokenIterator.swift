// Copyright © 2026 Faux Fox.

import Foundation
import MLX

/// Errors from configuring prompt-lookup decoding.
public enum PromptLookupError: Error, Equatable, Sendable {
    case invalidConfiguration
    case requiresGreedySampling
    case unsupportedInput
    case unsupportedCache
}

/// Finds continuations of recent output in the original prompt.
struct PromptLookupCandidates {
    private struct Pair: Hashable {
        let first: Int
        let second: Int
    }

    private let prompt: [Int]
    private let maximumNgramSize: Int
    private let endings: [Pair: [Int]]

    init(prompt: [Int], maximumNgramSize: Int) {
        self.prompt = prompt
        self.maximumNgramSize = maximumNgramSize
        var endings: [Pair: [Int]] = [:]
        if prompt.count >= 3 {
            for end in 1 ..< prompt.count - 1 {
                endings[Pair(first: prompt[end - 1], second: prompt[end]), default: []]
                    .append(end)
            }
        }
        self.endings = endings
    }

    func continuation(after recent: [Int], limit: Int) -> [Int] {
        guard recent.count >= 2, limit > 0,
            let positions = endings[
                Pair(first: recent[recent.count - 2], second: recent[recent.count - 1])]
        else { return [] }

        var bestEnd: Int?
        var bestMatch = 0
        var bestContinuation = 0
        let maximumMatch = Swift.min(maximumNgramSize, recent.count)
        for end in positions.reversed() {
            let available = Swift.min(limit, prompt.count - end - 1)
            var matched = 2
            let maximum = Swift.min(maximumNgramSize, Swift.min(recent.count, end + 1))
            while matched < maximum,
                prompt[end - matched] == recent[recent.count - matched - 1]
            {
                matched += 1
            }
            if matched > bestMatch || matched == bestMatch && available > bestContinuation {
                bestEnd = end
                bestMatch = matched
                bestContinuation = available
                // Older positions cannot improve either score once both limits are met.
                if bestMatch == maximumMatch && bestContinuation == limit { break }
            }
        }
        guard let bestEnd else { return [] }
        return Array(prompt[(bestEnd + 1) ..< (bestEnd + 1 + bestContinuation)])
    }
}

/// Greedy prompt-lookup speculation without a draft model.
///
/// Recent generated tokens select a continuation from the original prompt. The target model
/// verifies that continuation in one forward pass and keeps only its matching prefix. A cache
/// that cannot stage the verification falls back to ordinary one-token decoding. Model kernels
/// can change near-tied argmax results across verification shapes, so qualify exact output on
/// each target model before using this iterator in production. Hybrid recurrent targets use
/// ``ConcurrentTextRuntime``'s checkpoint-capable lookup path instead.
public struct PromptLookupTokenIterator: TokenIteratorProtocol {
    private var base: TokenIterator
    private let lookup: PromptLookupCandidates
    private let maximumNgramSize: Int
    private let numDraftTokens: Int
    private var recentTokens: [Int] = []
    private var pendingTokens: [Int]
    private var pendingIndex = 0
    private var committedPendingTokenCount = 0
    private var pendingIsSpeculative = false
    private var lastEmittedWasSpeculative = false
    private var lookupEnabled = true
    private var telemetry = SpeculativeDecodingTelemetry()

    public private(set) var tokenCount = 0
    public var maxTokens: Int? { base.maxTokens }
    public let promptPrefillTime: TimeInterval
    public var state: LMOutput.State? { base.state }
    public var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        telemetry.roundCount > 0 ? telemetry : nil
    }
    public private(set) var fallbackReason: String?

    /// Creates a prompt-lookup iterator for one unbatched, text-only input.
    /// `numDraftTokens` must be chosen after checking output parity at that
    /// verification width on the target model.
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        state: LMOutput.State? = nil, parameters: GenerateParameters,
        numDraftTokens: Int, maximumNgramSize: Int = 4,
        components: GenerationComponents = .init()
    ) throws {
        guard (1 ... 32).contains(numDraftTokens), (2 ... 16).contains(maximumNgramSize) else {
            throw PromptLookupError.invalidConfiguration
        }
        guard parameters.temperature == 0 else { throw PromptLookupError.requiresGreedySampling }
        guard input.text.tokens.ndim == 1, input.text.mask == nil,
            input.image == nil, input.video == nil, input.audio == nil
        else { throw PromptLookupError.unsupportedInput }

        let started = Date.timeIntervalSinceReferenceDate
        let lookup = PromptLookupCandidates(
            prompt: input.text.tokens.asArray(Int.self), maximumNgramSize: maximumNgramSize)
        let lookupTime = Date.timeIntervalSinceReferenceDate - started
        let base = try TokenIterator(
            input: input, model: model, cache: cache, state: state,
            parameters: parameters, components: components)
        guard !base.cache.isEmpty,
            let initialRound = base.cacheStorage.beginRound(maximumPositions: 2)
        else { throw PromptLookupError.unsupportedCache }
        base.cacheStorage.rollback(initialRound)
        self.pendingTokens = [base.y.tokens.item(Int.self)]
        self.base = base
        self.lookup = lookup
        self.maximumNgramSize = maximumNgramSize
        self.numDraftTokens = numDraftTokens
        self.promptPrefillTime = lookupTime + base.promptPrefillTime
    }

    public mutating func discardGeneratedToken() {
        if lastEmittedWasSpeculative { telemetry.discardGeneratedToken() }
    }

    public mutating func next() -> Int? {
        guard maxTokens.map({ tokenCount < $0 }) ?? true else { return nil }
        return autoreleasepool {
            if pendingIndex < pendingTokens.count { return drainPending() }

            pendingTokens.removeAll(keepingCapacity: true)
            pendingIndex = 0
            committedPendingTokenCount = 0
            pendingIsSpeculative = false

            let remaining = maxTokens.map { $0 - tokenCount } ?? numDraftTokens + 1
            if lookupEnabled {
                let proposals = lookup.continuation(
                    after: recentTokens,
                    limit: Swift.min(numDraftTokens, remaining - 1))
                if !proposals.isEmpty, verify(proposals) {
                    return drainPending()
                }
            }
            let token = base.step(previous: base.y)
            base.y = .init(tokens: token)
            asyncEval([token] + base.cache.flatMap { $0.state })
            return emit(token.item(Int.self), speculative: false)
        }
    }

    private mutating func drainPending() -> Int {
        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        return emit(token, speculative: pendingIsSpeculative)
    }

    private mutating func emit(_ token: Int, speculative: Bool) -> Int {
        tokenCount += 1
        recentTokens.append(token)
        if recentTokens.count > maximumNgramSize { recentTokens.removeFirst() }
        lastEmittedWasSpeculative = speculative
        if speculative { telemetry.recordGeneratedToken() }
        return token
    }

    private mutating func verify(_ proposals: [Int]) -> Bool {
        let count = proposals.count + 1
        guard let round = base.cacheStorage.beginRound(maximumPositions: count) else {
            lookupEnabled = false
            fallbackReason = "Target cache cannot stage prompt lookup"
            return false
        }
        let input = LMInput.Text(tokens: concatenated([base.y.tokens, MLXArray(proposals)]))
        let result = withPreparedCache(round.caches, lengths: input.sequenceLengths) {
            base.model(input[text: .newAxis], cache: round.caches, state: base.state)
        }
        eval([result.logits] + round.caches.flatMap { $0.state })
        guard round.writtenPositions == count, result.logits.ndim == 3,
            result.logits.dim(0) == 1, result.logits.dim(1) >= count
        else {
            base.cacheStorage.rollback(round)
            lookupEnabled = false
            fallbackReason = "Target did not return every verification row"
            return false
        }

        var accepted = 0
        for (index, proposal) in proposals.enumerated() {
            var logits = result.logits[0..., index, 0...]
            logits = base.processor?.process(logits: logits) ?? logits
            let target = base.sampler.sample(logits: logits)
            eval(target)
            let value = target.item(Int.self)
            base.processor?.didSample(token: target)
            pendingTokens.append(value)
            guard value == proposal else { break }
            accepted += 1
        }
        if accepted == proposals.count {
            var logits = result.logits[0..., accepted, 0...]
            logits = base.processor?.process(logits: logits) ?? logits
            let bonus = base.sampler.sample(logits: logits)
            eval(bonus)
            base.processor?.didSample(token: bonus)
            pendingTokens.append(bonus.item(Int.self))
        }

        base.cacheStorage.commit(round, retaining: accepted + 1)
        base.kvCachePlan.apply(to: base.cacheStorage)
        base.state = result.state
        guard let finalToken = pendingTokens.last else {
            preconditionFailure("Target verification produced no token")
        }
        base.y = .init(tokens: MLXArray([finalToken]))
        committedPendingTokenCount = accepted
        pendingIsSpeculative = true
        telemetry.recordRound(
            drafted: proposals.count, accepted: accepted, targetVerified: count,
            draftModelCalls: 0)
        return true
    }
}

extension PromptLookupTokenIterator: GenerationFinalizingTokenIterator {
    mutating func finalizeGeneration() {
        let consumed = Swift.min(pendingIndex, committedPendingTokenCount)
        let lookahead = committedPendingTokenCount - consumed
        if lookahead > 0 { base.cacheStorage.rewindLastRound(lookahead) }
    }
}
