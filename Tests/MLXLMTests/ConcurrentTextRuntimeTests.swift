// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXLLM
import MLXNN
import XCTest

@testable import MLXLMCommon

private typealias Runtime = ConcurrentTextRuntime

private final class ScheduledForwardProbe: @unchecked Sendable {
    var finalPrefillCallCount = 0
    var promptTokenCounts = [Int]()
}

private actor ScheduledForwardGate {
    private var started = false
    private var released = false
    private var startWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    func recordStart() {
        started = true
        let waiters = startWaiters
        startWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }

    func waitUntilStarted() async {
        guard !started else { return }
        await withCheckedContinuation { startWaiters.append($0) }
    }

    func waitForRelease() async {
        guard !released else { return }
        await withCheckedContinuation { releaseWaiters.append($0) }
    }

    func release() {
        released = true
        let waiters = releaseWaiters
        releaseWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }
}

private final class ScheduledForwardLifecycleProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var forwardReturned = false
    private var endCount = 0
    private var endBeforeForwardReturn = false

    func recordForwardReturn() {
        lock.withLock { forwardReturned = true }
    }

    func recordEnd() {
        lock.withLock {
            endCount += 1
            endBeforeForwardReturn = endBeforeForwardReturn || !forwardReturned
        }
    }

    func snapshot() -> (endCount: Int, endBeforeForwardReturn: Bool) {
        lock.withLock { (endCount, endBeforeForwardReturn) }
    }
}

private final class LatchedScheduledModel: Module, ScheduledTextModel {
    let vocabularySize = 100
    let scheduledCacheBytesPerToken = 8
    let gate: ScheduledForwardGate
    let probe: ScheduledForwardLifecycleProbe

    init(gate: ScheduledForwardGate, probe: ScheduledForwardLifecycleProbe) {
        self.gate = gate
        self.probe = probe
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [KVCacheSimple()] }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let token = inputs.asArray(Int32.self).reduce(0, +) % Int32(vocabularySize)
        return (MLXArray(0 ..< vocabularySize) .== token).asType(.float32).reshaped(1, 1, -1)
    }

    func scheduledBeginRequest() throws {}

    func scheduledEndRequest() {
        probe.recordEnd()
    }

    func scheduledForward(_ tokens: MLXArray, cache: [KVCache]) async throws -> MLXArray {
        await gate.recordStart()
        await gate.waitForRelease()
        probe.recordForwardReturn()
        return callAsFunction(tokens, cache: cache)
    }
}

/// Uses real MLX cache writes and context-dependent logits without downloaded weights.
private final class ScheduledChecksumModel: Module, ScheduledTextModel {
    let vocabularySize = 100
    let scheduledCacheBytesPerToken = 8
    let failOnToken: Int?
    let streamed: Bool
    let streamFailure: Error?
    let probe: ScheduledForwardProbe?

    init(
        failOnToken: Int? = nil, streamed: Bool = false, streamFailure: Error? = nil,
        probe: ScheduledForwardProbe? = nil
    ) {
        self.failOnToken = failOnToken
        self.streamed = streamed
        self.streamFailure = streamFailure
        self.probe = probe
    }

    var scheduledSupportsBatchDecode: Bool { false }
    var scheduledRequiresExclusiveExecution: Bool { streamed }
    var scheduledMaximumForwardTokens: Int? { streamed ? 2 : nil }

    func scheduledBeginRequest(promptTokenCount: Int) throws {
        probe?.promptTokenCounts.append(promptTokenCount)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [KVCacheSimple()] }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let values = inputs.asType(.float32).reshaped(1, 1, -1, 1)
        let (keys, _) = cache![0].update(keys: values, values: values)
        let token = sum(keys).asType(.int32) % vocabularySize
        return (MLXArray(0 ..< vocabularySize) .== token).asType(.float32).reshaped(1, 1, -1)
    }

    func scheduledForward(_ tokens: MLXArray, cache: [KVCache]) async throws -> MLXArray {
        let logits = callAsFunction(tokens, cache: cache)
        if let failOnToken, tokens.asArray(Int32.self).contains(Int32(failOnToken)) {
            if let streamFailure { throw streamFailure }
            throw ConcurrentTextRuntimeError.invalidRequest
        }
        return logits
    }

    func scheduledFinalPrefillForward(
        _ tokens: MLXArray, cache: [KVCache]
    ) async throws -> MLXArray {
        probe?.finalPrefillCallCount += 1
        return try await scheduledForward(tokens, cache: cache)
    }
}

private final class ScheduledSamplingModel: Module, ScheduledTextModel {
    let vocabularySize = 4
    let scheduledCacheBytesPerToken = 8
    let greedyFirstToken: Bool

    init(greedyFirstToken: Bool) {
        self.greedyFirstToken = greedyFirstToken
    }

    var scheduledSupportsBatchDecode: Bool { false }
    var scheduledFirstTokenGreedy: Bool { greedyFirstToken }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [KVCacheSimple()] }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let values = inputs.asType(.float32).reshaped(1, 1, -1, 1)
        _ = cache![0].update(keys: values, values: values)
        return MLXArray.zeros([1, 1, vocabularySize])
    }

    func scheduledForward(_ tokens: MLXArray, cache: [KVCache]) async throws -> MLXArray {
        callAsFunction(tokens, cache: cache)
    }
}

private final class ScheduledBlockLimitDrafter: Module, MTPDrafterModel {
    let maximumBlockSize: Int?

    init(maximumBlockSize: Int?) {
        self.maximumBlockSize = maximumBlockSize
        super.init()
    }

    func draftBlock(
        target _: any LanguageModel, lastToken _: MLXArray, lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)], positionDeltas _: MLXArray?,
        queryOffset _: Int, blockSize _: Int, sampler _: any LogitSampler
    ) -> MTPDraft {
        fatalError("Block-limit probe never drafts")
    }
}

private enum SyntheticStreamFailure: Error, Equatable {
    case unreadableExpertRange
}

private func collect(_ generation: Runtime.Generation) async throws -> (
    tokens: [Int], reused: Int, speculativeRounds: Int
) {
    var tokens: [Int] = []
    var reused = 0
    var rounds = 0
    for try await event in generation.events {
        switch event {
        case .token(let token): tokens.append(token)
        case .admitted(let count): reused = count
        case .speculation(let telemetry): rounds += telemetry.roundCount
        default: break
        }
    }
    return (tokens, reused, rounds)
}

final class ConcurrentTextRuntimeTests: XCTestCase {
    private func identity(_ revision: String = "weights-1") -> PrefixCacheIdentity {
        .init(
            modelRevision: revision, tokenizerRevision: "tokenizer-1",
            chatTemplateRevision: "template-1", adapterRevision: "none",
            cacheLayoutRevision: "simple-f32-1")
    }

    private func runtime(
        failOnToken: Int? = nil, active: Int = 2, buffer: Int = 2048,
        memory: Int = 1_000_000, prefixBytes: Int = 100_000
    ) throws -> Runtime {
        try Runtime(
            model: ScheduledChecksumModel(failOnToken: failOnToken), identity: identity(),
            configuration: .init(
                memoryBudgetBytes: memory, prefixCacheBytes: prefixBytes, workingMemoryBytes: 4096,
                maxActiveRequests: active, maxPromptTokens: 4096, maxOutputTokens: 128,
                prefillChunkSize: 1, streamBufferSize: buffer))
    }

    func testScheduledMTPAdmissionUsesIteratorBlockLimits() {
        XCTAssertEqual(
            scheduledMTPVerificationBlockSize(
                requestedBlockSize: 4,
                drafter: ScheduledBlockLimitDrafter(maximumBlockSize: 2),
                cache: [KVCacheSimple()]), 2)
        XCTAssertEqual(
            scheduledMTPVerificationBlockSize(
                requestedBlockSize: 4,
                drafter: ScheduledBlockLimitDrafter(maximumBlockSize: 4),
                cache: [RotatingKVCache(maxSize: 2)]), 3)
    }

    func testSpeculativeBatchCommitSplitsRecurrentRollbackRows() throws {
        let first = MambaCache()
        let second = MambaCache()
        for (cache, value) in [(first, Float(1)), (second, Float(2))] {
            cache[0] = MLXArray([value]).reshaped(1, 1)
            cache[1] = MLXArray([value + 10]).reshaped(1, 1)
            cache.advance(5)
        }
        let batch = try RuntimeBatchCache(rows: [[first], [second]])
        let merged = batch.cache[0] as! MambaCache
        merged[0] = MLXArray([13, 23] as [Float]).reshaped(2, 1)
        merged[1] = MLXArray([113, 123] as [Float]).reshaped(2, 1)
        merged.saveSpeculativeCheckpoint(
            convState: MLXArray([11, 21] as [Float]).reshaped(2, 1),
            recurrentState: MLXArray([111, 121] as [Float]).reshaped(2, 1),
            advancedBy: 1, rewinding: 2)
        merged.saveSpeculativeCheckpoint(
            convState: MLXArray([12, 22] as [Float]).reshaped(2, 1),
            recurrentState: MLXArray([112, 122] as [Float]).reshaped(2, 1),
            advancedBy: 2, rewinding: 1)

        batch.commit(advancedBy: 3)
        XCTAssertTrue(first.restoreSpeculativeCheckpoint(rewinding: 2))
        XCTAssertTrue(second.restoreSpeculativeCheckpoint(rewinding: 1))
        eval(first.innerState() + second.innerState())
        XCTAssertEqual(first[0]!.item(Float.self), 11)
        XCTAssertEqual(second[0]!.item(Float.self), 22)
    }

    func testStreamedScheduledForwardCapsPrefillAndDisablesBatchDecode() async throws {
        let model = ScheduledChecksumModel(streamed: true)
        let runtime = try Runtime(
            model: model, identity: identity(),
            configuration: .init(
                memoryBudgetBytes: 1_000_000, prefixCacheBytes: 100_000,
                workingMemoryBytes: 4096, maxActiveRequests: 2, maxPromptTokens: 4096,
                maxOutputTokens: 128, prefillChunkSize: 8, streamBufferSize: 2048,
                batchDecode: true))
        let first = try await runtime.generate(.init(tokens: [1, 2, 3], maxTokens: 2))
        let second = try await runtime.generate(.init(tokens: [4, 5, 6], maxTokens: 2))
        async let firstTokens = collect(first)
        async let secondTokens = collect(second)
        let results = try await (firstTokens, secondTokens)
        XCTAssertEqual(results.0.tokens, [6, 12])
        XCTAssertEqual(results.1.tokens, [15, 30])
        let status = await runtime.status()
        XCTAssertEqual(status.batchedForwardCount, 0)
        XCTAssertEqual(status.maximumBatchSize, 1)
        XCTAssertFalse(runtime.capabilities.fusedBatching)
    }

    func testStreamedScheduledForwardPropagatesReadFailure() async throws {
        let runtime = try Runtime(
            model: ScheduledChecksumModel(
                failOnToken: 3, streamed: true,
                streamFailure: SyntheticStreamFailure.unreadableExpertRange),
            identity: identity(),
            configuration: .init(
                memoryBudgetBytes: 1_000_000, prefixCacheBytes: 100_000,
                workingMemoryBytes: 4096, maxPromptTokens: 4096, maxOutputTokens: 128,
                prefillChunkSize: 8, streamBufferSize: 2048))
        let generation = try await runtime.generate(.init(tokens: [1, 2, 3], maxTokens: 2))
        do {
            _ = try await collect(generation)
            XCTFail("Expected streamed read failure")
        } catch let error as SyntheticStreamFailure {
            XCTAssertEqual(error, .unreadableExpertRange)
        }
    }

    func testEveryOrdinaryPrefillChunkUsesFinalLogitForward() async throws {
        let probe = ScheduledForwardProbe()
        let runtime = try Runtime(
            model: ScheduledChecksumModel(probe: probe), identity: identity(),
            configuration: .init(
                memoryBudgetBytes: 1_000_000, prefixCacheBytes: 100_000,
                workingMemoryBytes: 4096, maxPromptTokens: 4096, maxOutputTokens: 128,
                prefillChunkSize: 2, streamBufferSize: 2048))
        let generation = try await runtime.generate(.init(tokens: [1, 2, 3], maxTokens: 2))
        let result = try await collect(generation)
        XCTAssertEqual(result.tokens, [6, 12])
        XCTAssertEqual(probe.finalPrefillCallCount, 2)
        XCTAssertEqual(probe.promptTokenCounts, [3])
    }

    func testScheduledGreedyFirstTokenLeavesTheSamplerAtItsInitialSeededState() async throws {
        func generatedTokens(greedyFirstToken: Bool, seed: UInt64) async throws -> [Int] {
            let runtime = try Runtime(
                model: ScheduledSamplingModel(greedyFirstToken: greedyFirstToken),
                identity: identity(),
                configuration: .init(
                    memoryBudgetBytes: 1_000_000, prefixCacheBytes: 100_000,
                    workingMemoryBytes: 4096, maxPromptTokens: 64, maxOutputTokens: 2,
                    prefillChunkSize: 1, streamBufferSize: 32))
            let generation = try await runtime.generate(
                .init(
                    tokens: [1], maxTokens: 2, temperature: 1, seed: seed,
                    speculative: false))
            return try await collect(generation).tokens
        }

        let seeds: [UInt64] = [1, 2, 3, 4, 5]
        var greedy = [[Int]]()
        var sampled = [[Int]]()
        for seed in seeds {
            greedy.append(try await generatedTokens(greedyFirstToken: true, seed: seed))
            sampled.append(try await generatedTokens(greedyFirstToken: false, seed: seed))
        }

        XCTAssertTrue(greedy.allSatisfy { $0.first == 0 })
        XCTAssertEqual(greedy.map { $0[1] }, sampled.map { $0[0] })
        XCTAssertGreaterThan(Set(sampled.map { $0[0] }).count, 1)
    }

    func testCancellationSettlesForwardBeforeEndingTheModelRequest() async throws {
        let gate = ScheduledForwardGate()
        let probe = ScheduledForwardLifecycleProbe()
        let runtime = try Runtime(
            model: LatchedScheduledModel(gate: gate, probe: probe), identity: identity(),
            configuration: .init(
                memoryBudgetBytes: 1_000_000, prefixCacheBytes: 100_000,
                workingMemoryBytes: 4096, maxPromptTokens: 64, maxOutputTokens: 2,
                prefillChunkSize: 1, streamBufferSize: 32))
        let generation = try await runtime.generate(.init(tokens: [1], maxTokens: 2))
        await gate.waitUntilStarted()

        let cancellation = Task { await runtime.cancel(generation.id) }
        for _ in 0 ..< 3 { await Task.yield() }
        XCTAssertEqual(probe.snapshot().endCount, 0)

        await gate.release()
        await cancellation.value

        let lifecycle = probe.snapshot()
        XCTAssertEqual(lifecycle.endCount, 1)
        XCTAssertFalse(lifecycle.endBeforeForwardReturn)
        let result = try await collect(generation)
        XCTAssertTrue(result.tokens.isEmpty)
        let status = await runtime.status()
        XCTAssertEqual(status.activeRequests, 0)
    }

    func testShutdownWaitsForForwardBeforeReleasingModelOwnership() async throws {
        let gate = ScheduledForwardGate()
        let probe = ScheduledForwardLifecycleProbe()
        let runtime = try Runtime(
            model: LatchedScheduledModel(gate: gate, probe: probe), identity: identity(),
            configuration: .init(
                memoryBudgetBytes: 1_000_000, prefixCacheBytes: 100_000,
                workingMemoryBytes: 4096, maxPromptTokens: 64, maxOutputTokens: 2,
                prefillChunkSize: 1, streamBufferSize: 32))
        let generation = try await runtime.generate(.init(tokens: [1], maxTokens: 2))
        await gate.waitUntilStarted()

        let shutdown = Task { await runtime.shutdown() }
        for _ in 0 ..< 3 { await Task.yield() }
        XCTAssertEqual(probe.snapshot().endCount, 0)

        await gate.release()
        await shutdown.value

        let lifecycle = probe.snapshot()
        XCTAssertEqual(lifecycle.endCount, 1)
        XCTAssertFalse(lifecycle.endBeforeForwardReturn)
        let result = try await collect(generation)
        XCTAssertTrue(result.tokens.isEmpty)
        let status = await runtime.status()
        XCTAssertEqual(status.activeRequests, 0)
        do {
            _ = try await runtime.generate(.init(tokens: [1], maxTokens: 1))
            XCTFail("Expected shutdown rejection")
        } catch { XCTAssertEqual(error as? ConcurrentTextRuntimeError, .shutDown) }
    }

    func testDistinctSimultaneousPromptsAndSlotReuse() async throws {
        let runtime = try runtime()
        let a = try await runtime.generate(.init(tokens: [1, 2], maxTokens: 3))
        let b = try await runtime.generate(.init(tokens: [4, 5], maxTokens: 3))
        async let first = collect(a)
        async let second = collect(b)
        let results = try await (first, second)
        XCTAssertEqual(results.0.tokens, [3, 6, 12])
        XCTAssertEqual(results.1.tokens, [9, 18, 36])
        let c = try await runtime.generate(.init(tokens: [7, 8], maxTokens: 3))
        let third = try await collect(c)
        XCTAssertEqual(third.tokens, [15, 30, 60])
        let status = await runtime.status()
        XCTAssertEqual(status.activeRequests, 0)
        XCTAssertEqual(status.queuedRequests, 0)
    }

    func testSharedPrefixBranchesAndIdentityMismatch() async throws {
        let runtime = try runtime()
        _ = try await collect(
            runtime.generate(
                .init(
                    tokens: [1, 2, 3], maxTokens: 2, prefixTokenCount: 2, cacheIdentity: identity())
            ))
        let a = try await runtime.generate(
            .init(
                tokens: [1, 2, 4], maxTokens: 3, prefixTokenCount: 2, cacheIdentity: identity()))
        let b = try await runtime.generate(
            .init(
                tokens: [1, 2, 8], maxTokens: 3, prefixTokenCount: 2, cacheIdentity: identity()))
        async let first = collect(a)
        async let second = collect(b)
        let results = try await (first, second)
        XCTAssertEqual(results.0.tokens, [7, 14, 28])
        XCTAssertEqual(results.1.tokens, [11, 22, 44])
        XCTAssertEqual(results.0.reused, 2)
        XCTAssertEqual(results.1.reused, 2)

        let mismatches = [
            identity("weights-2"),
            PrefixCacheIdentity(
                modelRevision: "weights-1", tokenizerRevision: "other",
                chatTemplateRevision: "template-1", adapterRevision: "none",
                cacheLayoutRevision: "simple-f32-1"),
            PrefixCacheIdentity(
                modelRevision: "weights-1", tokenizerRevision: "tokenizer-1",
                chatTemplateRevision: "other", adapterRevision: "none",
                cacheLayoutRevision: "simple-f32-1"),
            PrefixCacheIdentity(
                modelRevision: "weights-1", tokenizerRevision: "tokenizer-1",
                chatTemplateRevision: "template-1", adapterRevision: "other",
                cacheLayoutRevision: "simple-f32-1"),
            PrefixCacheIdentity(
                modelRevision: "weights-1", tokenizerRevision: "tokenizer-1",
                chatTemplateRevision: "template-1", adapterRevision: "none",
                cacheLayoutRevision: "other"),
        ]
        for mismatch in mismatches {
            let cold = try await collect(
                runtime.generate(
                    .init(
                        tokens: [1, 2, 4], maxTokens: 3, prefixTokenCount: 2,
                        cacheIdentity: mismatch)))
            XCTAssertEqual(cold.reused, 0)
            XCTAssertEqual(cold.tokens, results.0.tokens)
        }
        let warm = try await collect(
            runtime.generate(
                .init(
                    tokens: [1, 2, 4], maxTokens: 3, prefixTokenCount: 2, cacheIdentity: identity())
            ))
        XCTAssertEqual(warm.reused, 2)
        XCTAssertEqual(warm.tokens, results.0.tokens)
        await runtime.clearPrefixCache()
        let status = await runtime.status()
        XCTAssertEqual(status.cachedPrefixes, 0)
        let cleared = try await collect(
            runtime.generate(
                .init(
                    tokens: [1, 2, 4], maxTokens: 3, prefixTokenCount: 2, cacheIdentity: identity())
            ))
        XCTAssertEqual(cleared.reused, 0)
        XCTAssertEqual(cleared.tokens, results.0.tokens)
    }

    func testInteractiveCompletesDuringLongPrefillAndCancelOne() async throws {
        let runtime = try runtime()
        let long = try await runtime.generate(
            .init(
                tokens: Array(repeating: 1, count: 4096), maxTokens: 100, priority: .background))
        var longEvents = long.events.makeAsyncIterator()
        while let event = try await longEvents.next() {
            if case .prefill = event { break }
        }
        let short = try await runtime.generate(.init(tokens: [2, 3], maxTokens: 2))
        let result = try await collect(short)
        XCTAssertEqual(result.tokens, [5, 10])
        let during = await runtime.status()
        XCTAssertEqual(during.activeRequests, 1)
        let survivor = try await runtime.generate(.init(tokens: [3, 4], maxTokens: 3))
        await runtime.cancel(long.id)
        let surviving = try await collect(survivor)
        XCTAssertEqual(surviving.tokens, [7, 14, 28])
        var cancelled = false
        var progressed = false
        while let event = try await longEvents.next() {
            if case .finished(.cancelled) = event { cancelled = true }
            if case .prefill(let processed, _) = event, processed > 1 { progressed = true }
        }
        XCTAssertTrue(cancelled)
        XCTAssertTrue(progressed, "Background work must progress while interactive work runs")
        let status = await runtime.status()
        XCTAssertEqual(status.activeRequests, 0)
    }

    func testFailureAfterCacheWriteReleasesAdmissionAndPreservesPeer() async throws {
        let runtime = try runtime(failOnToken: 99)
        let bad = try await runtime.generate(.init(tokens: [1, 99], maxTokens: 3))
        let good = try await runtime.generate(.init(tokens: [3, 4], maxTokens: 3))
        do {
            _ = try await collect(bad)
            XCTFail("Expected injected failure")
        } catch { XCTAssertEqual(error as? ConcurrentTextRuntimeError, .invalidRequest) }
        let result = try await collect(good)
        XCTAssertEqual(result.tokens, [7, 14, 28])
        let next = try await collect(runtime.generate(.init(tokens: [2, 3], maxTokens: 2)))
        XCTAssertEqual(next.tokens, [5, 10])
        let status = await runtime.status()
        XCTAssertEqual(status.activeRequests, 0)
        XCTAssertEqual(status.queuedRequests, 0)
    }

    func testQueuedCancellationAndMemoryAdmission() async throws {
        let runtime = try runtime(memory: 12_000, prefixBytes: 0)
        let first = try await runtime.generate(
            .init(
                tokens: Array(repeating: 1, count: 256), maxTokens: 100, priority: .background))
        var events = first.events.makeAsyncIterator()
        _ = try await events.next()
        let queued = try await runtime.generate(.init(tokens: [2, 3], maxTokens: 2))
        let status = await runtime.status()
        XCTAssertEqual(status.activeRequests, 1)
        XCTAssertEqual(status.queuedRequests, 1)
        XCTAssertLessThanOrEqual(status.reservedBytes, 12_000)
        await runtime.cancel(queued.id)
        let cancelled = try await collect(queued)
        XCTAssertTrue(cancelled.tokens.isEmpty)
        await runtime.cancel(first.id)
        let next = try await collect(runtime.generate(.init(tokens: [4, 5], maxTokens: 2)))
        XCTAssertEqual(next.tokens, [9, 18])
    }

    func testPrefixEvictionFallsBackColdWithinBudget() async throws {
        let runtime = try runtime(prefixBytes: 2100)
        for prefix in [[1, 2], [3, 4], [1, 2]] {
            let result = try await collect(
                runtime.generate(
                    .init(
                        tokens: prefix + [5], maxTokens: 1, prefixTokenCount: 2,
                        cacheIdentity: identity())))
            XCTAssertEqual(result.reused, 0)
            XCTAssertEqual(result.tokens, [prefix.reduce(5, +)])
            let status = await runtime.status()
            XCTAssertLessThanOrEqual(status.cachedPrefixBytes, 2100)
            XCTAssertEqual(status.cachedPrefixes, 1)
        }
    }

    func testCancelDuringDecodePreservesSharedCheckpointAndPeer() async throws {
        let runtime = try runtime()
        let branch = try await runtime.generate(
            .init(
                tokens: [1, 2, 3], maxTokens: 100, priority: .background,
                prefixTokenCount: 2, cacheIdentity: identity()))
        var events = branch.events.makeAsyncIterator()
        while let event = try await events.next() {
            if case .token = event { break }
        }
        let peer = try await runtime.generate(
            .init(
                tokens: [1, 2, 4], maxTokens: 3, prefixTokenCount: 2, cacheIdentity: identity()))
        await runtime.cancel(branch.id)
        let result = try await collect(peer)
        XCTAssertEqual(result.reused, 2)
        XCTAssertEqual(result.tokens, [7, 14, 28])
    }

    func testBudgetRejectsOversizedRequestAndSlowConsumerReleasesSlot() async throws {
        let limited = try runtime(memory: 10_000, prefixBytes: 0)
        do {
            _ = try await limited.generate(
                .init(tokens: Array(repeating: 1, count: 1000), maxTokens: 2))
            XCTFail("Expected admission rejection")
        } catch { XCTAssertEqual(error as? ConcurrentTextRuntimeError, .memoryBudgetExceeded) }
        let runtime = try runtime(buffer: 1)
        let stalled = try await runtime.generate(.init(tokens: [1, 2], maxTokens: 3))
        // Waiting for a second request to settle also lets the unconsumed stream overflow.
        let peer = try await runtime.generate(.init(tokens: [3, 4], maxTokens: 2))
        _ = try? await collect(peer)
        do {
            _ = try await collect(stalled)
            XCTFail("Expected bounded-buffer failure")
        } catch { XCTAssertEqual(error as? ConcurrentTextRuntimeError, .consumerTooSlow) }
        await runtime.shutdown()
        let status = await runtime.status()
        XCTAssertEqual(status.activeRequests, 0)
        do {
            _ = try await runtime.generate(.init(tokens: [1], maxTokens: 1))
            XCTFail("Expected shutdown rejection")
        } catch { XCTAssertEqual(error as? ConcurrentTextRuntimeError, .shutDown) }
    }

    func testTinyLlamaWarmBranchMatchesColdGeneration() async throws {
        let model = LlamaModel(
            .init(
                hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64,
                attentionHeads: 4, rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 2))
        let runtime = try Runtime(
            model: model, identity: identity(),
            configuration: .init(
                memoryBudgetBytes: 16_000_000, prefixCacheBytes: 1_000_000,
                workingMemoryBytes: 1_000_000, prefillChunkSize: 2))
        _ = try await collect(
            runtime.generate(
                .init(
                    tokens: [1, 2, 3, 4, 5], maxTokens: 2, prefixTokenCount: 4,
                    cacheIdentity: identity())))
        let warm = try await runtime.generate(
            .init(
                tokens: [1, 2, 3, 4, 6], maxTokens: 4, prefixTokenCount: 4,
                cacheIdentity: identity()))
        let cold = try await runtime.generate(.init(tokens: [1, 2, 3, 4, 6], maxTokens: 4))
        async let a = collect(warm)
        async let b = collect(cold)
        let result = try await (a, b)
        XCTAssertEqual(result.0.reused, 4)
        XCTAssertEqual(result.0.tokens, result.1.tokens)
    }
    func testUnequalLengthBatchMatchesIndependentLlamaWithQuantizedKV() async throws {
        for bits in [0, 4, 8] {
            let model = LlamaModel(
                .init(
                    hiddenSize: 128, hiddenLayers: 2, intermediateSize: 128,
                    attentionHeads: 2, rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 1))
            let runtime = try Runtime(
                model: model, identity: identity(),
                configuration: .init(
                    memoryBudgetBytes: 32_000_000, prefixCacheBytes: 0,
                    workingMemoryBytes: 1_000_000, prefillChunkSize: 2,
                    cacheQuantization: bits == 0 ? nil : .init(bits: bits)))
            let short = Runtime.Request(tokens: [1, 2, 3], maxTokens: 16)
            let long = Runtime.Request(tokens: [6, 5, 4, 3, 2, 1], maxTokens: 16)
            let expectedA = try await collect(runtime.generate(short)).tokens
            let expectedB = try await collect(runtime.generate(long)).tokens
            let a = try await runtime.generate(short)
            let b = try await runtime.generate(long)
            async let first = collect(a)
            async let second = collect(b)
            let result = try await (first, second)
            XCTAssertEqual(result.0.tokens, expectedA, "KV bits: \(bits)")
            XCTAssertEqual(result.1.tokens, expectedB, "KV bits: \(bits)")
            let status = await runtime.status()
            XCTAssertGreaterThan(status.batchedForwardCount, 0)
            XCTAssertEqual(status.maximumBatchSize, 2)
            await runtime.shutdown()
        }
    }

    func testHybridBatchAndScheduledMTPPreserveIndependentOutputs() async throws {
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                """
                {"model_type":"qwen3_5_text","hidden_size":64,"num_hidden_layers":2,
                "intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,
                "head_dim":32,"linear_num_value_heads":2,"linear_num_key_heads":1,
                "linear_key_head_dim":32,"linear_value_head_dim":32,"linear_conv_kernel_dim":4,
                "vocab_size":100,"full_attention_interval":2,"mtp_num_hidden_layers":1,
                "tie_word_embeddings":true,"rope_theta":10000000.0,"partial_rotary_factor":0.25}
                """.utf8))
        for bits in [0, 4, 8] {
            let model = withRandomState(MLXRandom.RandomState(seed: 112)) {
                Qwen35TextModel(config)
            }
            let drafter = withRandomState(MLXRandom.RandomState(seed: 113)) {
                Qwen35MTPDraftModel(config)
            }
            let runtime = try Runtime(
                model: model, identity: identity(),
                configuration: .init(
                    memoryBudgetBytes: 32_000_000, prefixCacheBytes: 1_000_000,
                    workingMemoryBytes: 1_000_000, prefillChunkSize: 2,
                    cacheQuantization: bits == 0 ? nil : .init(bits: bits, groupSize: 32)),
                drafter: drafter)
            let short = [1, 2, 3]
            let long = [6, 5, 4, 3, 2, 1]
            let expectedA = try await collect(
                runtime.generate(
                    .init(
                        tokens: short,
                        maxTokens: 16, speculative: false))
            ).tokens
            let expectedB = try await collect(
                runtime.generate(
                    .init(
                        tokens: long,
                        maxTokens: 16, speculative: false))
            ).tokens
            for speculative in [false, true] {
                let batchedBefore = await runtime.status().batchedForwardCount
                let a = try await runtime.generate(
                    .init(tokens: short, maxTokens: 16, speculative: speculative))
                let b = try await runtime.generate(
                    .init(tokens: long, maxTokens: 16, speculative: speculative))
                async let first = collect(a)
                async let second = collect(b)
                let result = try await (first, second)
                XCTAssertEqual(result.0.tokens, expectedA, "KV bits \(bits), MTP \(speculative)")
                XCTAssertEqual(result.1.tokens, expectedB, "KV bits \(bits), MTP \(speculative)")
                if speculative {
                    XCTAssertGreaterThan(result.0.speculativeRounds, 0)
                    XCTAssertGreaterThan(result.1.speculativeRounds, 0)
                    if bits == 0 {
                        let batchedAfter = await runtime.status().batchedForwardCount
                        XCTAssertGreaterThan(batchedAfter, batchedBefore)
                    }
                }
            }
            let status = await runtime.status()
            XCTAssertGreaterThan(status.batchedForwardCount, 0)
            let warmRequest = Runtime.Request(
                tokens: long, maxTokens: 4, prefixTokenCount: 4,
                cacheIdentity: identity(), speculative: false)
            _ = try await collect(runtime.generate(warmRequest))
            let restored = try await collect(runtime.generate(warmRequest))
            XCTAssertEqual(restored.reused, 4)
            XCTAssertEqual(restored.tokens, Array(expectedB.prefix(4)))
            if bits == 0 {
                let sampled = try await collect(
                    runtime.generate(
                        .init(
                            tokens: short, maxTokens: 8, temperature: 0.7,
                            topP: 0.95, topK: 20, seed: 7)))
                XCTAssertEqual(sampled.tokens.count, 8)
                XCTAssertGreaterThan(sampled.speculativeRounds, 0)

                let stopToken = expectedA[3]
                let stopIndex = expectedA.firstIndex(of: stopToken)!
                let stopped = try await collect(
                    runtime.generate(
                        .init(
                            tokens: short, maxTokens: 16,
                            stopTokenIDs: [stopToken])))
                XCTAssertEqual(stopped.tokens, Array(expectedA.prefix(stopIndex)))
                let cancelled = try await runtime.generate(.init(tokens: long, maxTokens: 16))
                var events = cancelled.events.makeAsyncIterator()
                while let event = try await events.next() { if case .token = event { break } }
                let peer = try await runtime.generate(.init(tokens: short, maxTokens: 16))
                await runtime.cancel(cancelled.id)
                let result = try await collect(peer)
                XCTAssertEqual(result.tokens, expectedA)
            }
            await runtime.shutdown()
        }
    }

    func testScheduledMTPClipsPrefillAtExactPrefixFrontier() async throws {
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                """
                {"model_type":"qwen3_5_text","hidden_size":64,"num_hidden_layers":2,
                "intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,
                "head_dim":32,"linear_num_value_heads":2,"linear_num_key_heads":1,
                "linear_key_head_dim":32,"linear_value_head_dim":32,"linear_conv_kernel_dim":4,
                "vocab_size":100,"full_attention_interval":2,"mtp_num_hidden_layers":1,
                "tie_word_embeddings":true,"rope_theta":10000000.0,"partial_rotary_factor":0.25}
                """.utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: 114)) {
            Qwen35TextModel(config)
        }
        let drafter = withRandomState(MLXRandom.RandomState(seed: 115)) {
            Qwen35MTPDraftModel(config)
        }
        let runtime = try Runtime(
            model: model, identity: identity(),
            configuration: .init(
                memoryBudgetBytes: 32_000_000, prefixCacheBytes: 1_000_000,
                workingMemoryBytes: 1_000_000, prefillChunkSize: 256),
            drafter: drafter)
        let shared = (0 ..< 129).map { $0 % 99 + 1 }
        let seed = shared + [1]
        let branch = shared + [2]

        func collectDetails(_ request: Runtime.Request) async throws -> (
            tokens: [Int], reused: Int, prefill: [Int], rounds: Int
        ) {
            let generation = try await runtime.generate(request)
            var tokens = [Int]()
            var reused = 0
            var prefill = [Int]()
            var rounds = 0
            for try await event in generation.events {
                switch event {
                case .token(let token): tokens.append(token)
                case .admitted(let count): reused = count
                case .prefill(let processed, _): prefill.append(processed)
                case .speculation(let telemetry): rounds += telemetry.roundCount
                default: break
                }
            }
            return (tokens, reused, prefill, rounds)
        }

        do {
            let cold = try await collectDetails(.init(tokens: branch, maxTokens: 4))
            XCTAssertEqual(cold.prefill, [130])
            let published = try await collectDetails(
                .init(
                    tokens: seed, maxTokens: 4, prefixTokenCount: 129,
                    cacheIdentity: identity()))
            XCTAssertEqual(published.prefill, [128, 130])

            let warm = try await collectDetails(
                .init(
                    tokens: branch, maxTokens: 4, prefixTokenCount: 129,
                    cacheIdentity: identity()))
            XCTAssertEqual(warm.reused, 128)
            XCTAssertEqual(warm.prefill, [130])
            XCTAssertEqual(warm.tokens, cold.tokens)
            XCTAssertGreaterThan(warm.rounds, 0)

            await runtime.clearPrefixCache()
            let alignedShared = (0 ..< 257).map { $0 % 99 + 1 }
            let alignedSeed = alignedShared + [1]
            let alignedBranch = alignedShared + [2]
            let alignedCold = try await collectDetails(
                .init(tokens: alignedBranch, maxTokens: 4))
            let alignedPublished = try await collectDetails(
                .init(
                    tokens: alignedSeed, maxTokens: 4, prefixTokenCount: 257,
                    cacheIdentity: identity()))
            XCTAssertEqual(alignedPublished.prefill, [256, 258])
            let alignedWarm = try await collectDetails(
                .init(
                    tokens: alignedBranch, maxTokens: 4, prefixTokenCount: 257,
                    cacheIdentity: identity()))
            XCTAssertEqual(alignedWarm.reused, 256)
            XCTAssertEqual(alignedWarm.prefill, [258])
            XCTAssertEqual(alignedWarm.tokens, alignedCold.tokens)
        } catch {
            await runtime.shutdown()
            throw error
        }
        await runtime.shutdown()
    }

    func testSharedBudgetReservesInteractiveHeadroomAndReleasesCancelledDemand() async throws {
        let budget = try InferenceResourceBudget(
            capacityBytes: 20_000, interactiveHeadroomBytes: 4_000)
        let model = UUID()
        try await budget.register(model, bytes: 6_000)
        let background = UUID()
        let interactive = UUID()
        let waiting = UUID()
        let admitted = try await budget.acquire(background, bytes: 10_000, background: true)
        XCTAssertTrue(admitted)
        let blocked = try await budget.acquire(waiting, bytes: 8_000, background: true)
        XCTAssertFalse(blocked)
        let interactiveAdmitted = try await budget.acquire(
            interactive, bytes: 4_000, background: false)
        XCTAssertTrue(interactiveAdmitted)
        let full = await budget.status()
        XCTAssertEqual(full.residentBytes + full.requestBytes, 20_000)
        await budget.release(waiting)
        await budget.release(background)
        await budget.release(interactive)
        let next = try await budget.acquire(UUID(), bytes: 10_000, background: true)
        XCTAssertTrue(next, "Cancelled demand must not block later work")
    }

    func testNativeServiceRoutesModelsAndUnloadsResidentReservations() async throws {
        let service = try NativeInferenceRuntime(
            memoryBudgetBytes: 1_000_000,
            interactiveHeadroomBytes: 100_000)
        let cacheIdentity = identity()
        for id in ["first", "second"] {
            try await service.load(id: id, estimatedResidentBytes: 100_000) {
                try Runtime(
                    model: ScheduledChecksumModel(), identity: cacheIdentity,
                    configuration: .init(
                        memoryBudgetBytes: 500_000, prefixCacheBytes: 50_000,
                        workingMemoryBytes: 4096, prefillChunkSize: 1))
            }
        }
        let a = try await service.generate(
            modelID: "first", request: .init(tokens: [1, 2], maxTokens: 3))
        let b = try await service.generate(
            modelID: "second", request: .init(tokens: [4, 5], maxTokens: 3))
        async let first = collect(a)
        async let second = collect(b)
        let results = try await (first, second)
        XCTAssertEqual(results.0.tokens, [3, 6, 12])
        XCTAssertEqual(results.1.tokens, [9, 18, 36])
        await service.unload(modelID: "first")
        let remaining = await service.modelIDs()
        XCTAssertEqual(remaining, ["second"])
        await service.unload(modelID: "second")
        let status = await service.resources.status()
        XCTAssertEqual(status.residentBytes, 0)
        XCTAssertEqual(status.requestBytes, 0)
    }

    func testCancelledModelLoadReleasesModelAndBudget() async throws {
        let service = try NativeInferenceRuntime(
            memoryBudgetBytes: 1_000_000,
            interactiveHeadroomBytes: 100_000)
        let (ready, announceReady) = AsyncStream<Void>.makeStream()
        let (release, allowCompletion) = AsyncStream<Void>.makeStream()
        let cacheIdentity = identity()
        let loading = Task {
            try await service.load(id: "cancelled", estimatedResidentBytes: 100_000) {
                announceReady.yield(())
                var gate = release.makeAsyncIterator()
                _ = await gate.next()
                return try Runtime(
                    model: ScheduledChecksumModel(), identity: cacheIdentity,
                    configuration: .init(
                        memoryBudgetBytes: 500_000, prefixCacheBytes: 50_000,
                        workingMemoryBytes: 4096))
            }
        }
        var start = ready.makeAsyncIterator()
        _ = await start.next()
        loading.cancel()
        allowCompletion.yield(())
        do {
            try await loading.value
            XCTFail("Cancelled load must not publish")
        } catch { XCTAssertTrue(error is CancellationError) }
        let ids = await service.modelIDs()
        let status = await service.resources.status()
        XCTAssertTrue(ids.isEmpty)
        XCTAssertEqual(status.residentBytes, 0)
        announceReady.finish()
        allowCompletion.finish()
    }

    func testCrossModelExecutionTurnCancellationPreservesWaitingPeer() async throws {
        let budget = try InferenceResourceBudget(
            capacityBytes: 10_000, interactiveHeadroomBytes: 1000)
        let owner = UUID()
        let cancelled = UUID()
        let peer = UUID()
        for id in [owner, cancelled, peer] {
            let granted = try await budget.acquire(id, bytes: 1000, background: false)
            XCTAssertTrue(granted)
        }
        let first = await budget.beginTurn(owner, background: true)
        XCTAssertTrue(first)
        let cancelledTurn = Task { await budget.beginTurn(cancelled, background: false) }
        let peerTurn = Task { await budget.beginTurn(peer, background: false) }
        while true {
            let version = await budget.version()
            if await budget.status().pendingExecutionTurns == 2 { break }
            await budget.waitForChange(after: version)
        }
        await budget.release(cancelled)
        let wasCancelled = await cancelledTurn.value
        XCTAssertFalse(wasCancelled)
        let waiting = await budget.status()
        XCTAssertEqual(waiting.pendingExecutionTurns, 1)
        await budget.endTurn(owner)
        let peerGranted = await peerTurn.value
        XCTAssertTrue(peerGranted)
        await budget.release(peer)
        await budget.release(owner)
        let final = await budget.status()
        XCTAssertEqual(final.pendingExecutionTurns, 0)
        XCTAssertEqual(final.requestBytes, 0)
    }

    func testSlowBatchedConsumerDoesNotFailPeer() async throws {
        let model = LlamaModel(
            .init(
                hiddenSize: 128, hiddenLayers: 2, intermediateSize: 128,
                attentionHeads: 2, rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 1))
        let runtime = try Runtime(
            model: model, identity: identity(),
            configuration: .init(
                memoryBudgetBytes: 32_000_000, prefixCacheBytes: 0, workingMemoryBytes: 1_000_000,
                prefillChunkSize: 2, streamBufferSize: 8))
        let request = Runtime.Request(tokens: [1, 2, 3, 4, 5, 6], maxTokens: 16)
        let expected = try await collect(runtime.generate(request)).tokens
        let stalled = try await runtime.generate(.init(tokens: [7, 8, 9], maxTokens: 64))
        let peer = try await collect(runtime.generate(request))
        XCTAssertEqual(peer.tokens, expected)
        do {
            _ = try await collect(stalled)
            XCTFail("Expected slow-consumer failure")
        } catch { XCTAssertEqual(error as? ConcurrentTextRuntimeError, .consumerTooSlow) }
        let status = await runtime.status()
        XCTAssertGreaterThan(status.batchedForwardCount, 0)
        XCTAssertEqual(status.activeRequests, 0)
        await runtime.shutdown()
    }

}
