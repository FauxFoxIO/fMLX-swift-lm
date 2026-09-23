// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private final class PromptLookupTransitionModel: Module, LanguageModel {
    let vocabularySize: Int
    let finalRowOnly: Bool
    var forwardCallCount = 0

    init(vocabularySize: Int = 10, finalRowOnly: Bool = false) {
        self.vocabularySize = vocabularySize
        self.finalRowOnly = finalRowOnly
        super.init()
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [KVCacheSimple()] }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forwardCallCount += 1
        let tokens = inputs.asArray(Int.self)
        let entry = MLXArray.zeros([1, 1, tokens.count, 1])
        _ = cache![0].update(keys: entry, values: entry)
        var logits = [Float](repeating: -100, count: tokens.count * vocabularySize)
        for (index, token) in tokens.enumerated() {
            logits[index * vocabularySize + (token + 1) % vocabularySize] = 100
        }
        let result = MLXArray(logits, [1, tokens.count, vocabularySize])
        return finalRowOnly ? result[0..., (-1)..., 0...] : result
    }
}

@Suite("Prompt lookup decoding")
struct PromptLookupTokenIteratorTests {
    @Test(
        "Repeated-prompt candidate search timing",
        .enabled(if: ProcessInfo.processInfo.environment["FMLX_BENCHMARK_LOOKUP_CANDIDATES"] == "1")
    )
    func repeatedPromptCandidateSearchTiming() {
        let candidates = PromptLookupCandidates(
            prompt: Array(repeating: 7, count: 32_768), maximumNgramSize: 4)
        let iterations = 1_000
        let start = ContinuousClock.now
        var proposed = 0
        for iteration in 0 ..< iterations {
            let limit = 4 + (iteration & 7)
            proposed += candidates.continuation(after: [7, 7, 7, 7], limit: limit).count
        }
        let elapsed = start.duration(to: .now)
        print("[LookupCandidates] iterations=\(iterations) elapsed=\(elapsed)")
        #expect(proposed == iterations * 7 + iterations / 2)
    }

    @Test func candidateSelectionPrefersLongerMatchesAndContinuation() {
        let candidates = PromptLookupCandidates(
            prompt: [7, 2, 3, 4, 8, 9, 1, 2, 3, 5, 6], maximumNgramSize: 3)
        #expect(candidates.continuation(after: [1, 2, 3], limit: 3) == [5, 6])
        #expect(candidates.continuation(after: [7, 2, 3], limit: 3) == [4, 8, 9])
        #expect(candidates.continuation(after: [9, 9], limit: 3).isEmpty)
    }

    @Test(arguments: [
        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
        [2, 3, 4, 5, 9, 0, 1],
    ])
    func verifiesPromptCandidatesAgainstGreedyDecode(prompt: [Int]) throws {
        let parameters = GenerateParameters(maxTokens: 12, temperature: 0)
        let input = LMInput(tokens: MLXArray(prompt))
        let ordinaryModel = PromptLookupTransitionModel()
        var ordinary = try TokenIterator(input: input, model: ordinaryModel, parameters: parameters)
        var expected: [Int] = []
        while let token = ordinary.next() { expected.append(token) }

        let lookupModel = PromptLookupTransitionModel()
        var iterator = try PromptLookupTokenIterator(
            input: input, model: lookupModel, parameters: parameters, numDraftTokens: 4)
        var actual: [Int] = []
        while let token = iterator.next() { actual.append(token) }
        iterator.finalizeGeneration()

        #expect(actual == expected)
        let telemetry = try #require(iterator.speculativeDecodingTelemetry)
        #expect(telemetry.acceptedDraftTokenCount > 0)
        #expect(telemetry.draftModelCallCount == 0)
        #expect(telemetry.targetModelCallCount == telemetry.roundCount)
        #expect(lookupModel.forwardCallCount < ordinaryModel.forwardCallCount)
    }

    @Test func earlyStopRewindsUnemittedAcceptedTokens() throws {
        let prompt = [2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
        let input = LMInput(tokens: MLXArray(prompt))
        let cache: [KVCache] = [KVCacheSimple()]
        var iterator = try PromptLookupTokenIterator(
            input: input, model: PromptLookupTransitionModel(), cache: cache,
            parameters: GenerateParameters(maxTokens: 12, temperature: 0), numDraftTokens: 4)
        #expect(iterator.next() == 2)
        #expect(iterator.next() == 3)
        #expect(iterator.next() == 4)
        #expect(cache[0].offset > prompt.count + 3)
        iterator.finalizeGeneration()
        #expect(cache[0].offset == prompt.count + 3)
    }

    @Test func unmatchedOutputUsesOrdinaryDecode() throws {
        let input = LMInput(tokens: MLXArray([1, 3, 5]))
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
        let ordinaryModel = PromptLookupTransitionModel()
        var ordinary = try TokenIterator(input: input, model: ordinaryModel, parameters: parameters)
        var expected: [Int] = []
        while let token = ordinary.next() { expected.append(token) }

        let lookupModel = PromptLookupTransitionModel()
        var iterator = try PromptLookupTokenIterator(
            input: input, model: lookupModel, parameters: parameters, numDraftTokens: 3)
        var actual: [Int] = []
        while let token = iterator.next() { actual.append(token) }
        #expect(actual == expected)
        #expect(iterator.speculativeDecodingTelemetry == nil)
        #expect(lookupModel.forwardCallCount == ordinaryModel.forwardCallCount - 1)
    }

    @Test func finalRowOnlyModelFallsBackWithoutChangingOutput() throws {
        let input = LMInput(tokens: MLXArray([2, 3, 4, 5, 6, 7, 8, 9, 0, 1]))
        let parameters = GenerateParameters(maxTokens: 8, temperature: 0)
        var ordinary = try TokenIterator(
            input: input, model: PromptLookupTransitionModel(finalRowOnly: true),
            parameters: parameters)
        var expected: [Int] = []
        while let token = ordinary.next() { expected.append(token) }

        var iterator = try PromptLookupTokenIterator(
            input: input, model: PromptLookupTransitionModel(finalRowOnly: true),
            parameters: parameters, numDraftTokens: 4)
        var actual: [Int] = []
        while let token = iterator.next() { actual.append(token) }
        #expect(actual == expected)
        #expect(iterator.fallbackReason == "Target did not return every verification row")
        #expect(iterator.speculativeDecodingTelemetry == nil)
    }

    @Test func stochasticRequestsAreRejectedBeforePrefill() throws {
        #expect(throws: PromptLookupError.requiresGreedySampling) {
            try PromptLookupTokenIterator(
                input: LMInput(tokens: MLXArray([1, 2, 3])),
                model: PromptLookupTransitionModel(),
                parameters: GenerateParameters(temperature: 0.7), numDraftTokens: 2)
        }
    }
}
