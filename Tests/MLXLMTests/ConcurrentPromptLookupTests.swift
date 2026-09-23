// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private final class ScheduledLookupProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    func record() { lock.withLock { count += 1 } }
    var forwardCount: Int { lock.withLock { count } }
}

private final class ScheduledLookupModel: Module, ScheduledTextModel {
    let vocabularySize = 10
    let scheduledCacheBytesPerToken = 8
    let probe: ScheduledLookupProbe

    init(probe: ScheduledLookupProbe) {
        self.probe = probe
        super.init()
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [KVCacheSimple()] }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        probe.record()
        let tokens = inputs.asArray(Int.self)
        let entry = MLXArray.zeros([1, 1, tokens.count, 1])
        _ = cache![0].update(keys: entry, values: entry)
        var logits = [Float](repeating: -100, count: tokens.count * vocabularySize)
        for (index, token) in tokens.enumerated() {
            logits[index * vocabularySize + (token + 1) % vocabularySize] = 100
        }
        return MLXArray(logits, [1, tokens.count, vocabularySize])
    }

    func scheduledForward(_ tokens: MLXArray, cache: [KVCache]) async throws -> MLXArray {
        callAsFunction(tokens, cache: cache)
    }
}

private final class HybridLookupProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var checkpointOnlyCalls = 0

    func recordCheckpointOnly() { lock.withLock { checkpointOnlyCalls += 1 } }
    var checkpointCount: Int { lock.withLock { checkpointOnlyCalls } }
}

private actor LookupForwardGate {
    private var started = false
    private var released = false
    private var startWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    func pauseOnce() async {
        guard !started else { return }
        started = true
        let waiters = startWaiters
        startWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
        if !released {
            await withCheckedContinuation { releaseWaiters.append($0) }
        }
    }

    func waitUntilStarted() async {
        guard !started else { return }
        await withCheckedContinuation { startWaiters.append($0) }
    }

    func release() {
        released = true
        let waiters = releaseWaiters
        releaseWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }
}

private final class ScheduledHybridLookupModel: Module, ScheduledTextModel,
    PromptLookupHybridModel
{
    let vocabularySize = 10
    let scheduledCacheBytesPerToken = 8
    let maximumNativeTargetCacheRewind = 3
    let probe: HybridLookupProbe
    let gate: LookupForwardGate?
    var scheduledSupportsBatchDecode: Bool { true }

    init(probe: HybridLookupProbe, gate: LookupForwardGate? = nil) {
        self.probe = probe
        self.gate = gate
        super.init()
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        [MambaCache(), KVCacheSimple()]
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(inputs, cache: cache, checkpoints: [])
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let checkpointOnly = state?[speculativeCheckpointOnlyKey] ?? false
        if checkpointOnly { probe.recordCheckpointOnly() }
        let checkpoints =
            state?[mtpCacheCheckpointIndicesKey]
            ?? state?[mtpCacheCheckpointIndexKey].map { [$0] } ?? []
        return LMOutput(logits: forward(input.tokens, cache: cache, checkpoints: checkpoints))
    }

    func scheduledForward(_ tokens: MLXArray, cache: [KVCache]) async throws -> MLXArray {
        await gate?.pauseOnce()
        return callAsFunction(tokens, cache: cache)
    }

    private func forward(
        _ input: MLXArray, cache: [KVCache]?, checkpoints: [Int]
    ) -> MLXArray {
        let tokens = input.asArray(Int.self)
        let batch = input.dim(0)
        let length = input.dim(1)
        let mamba = cache![0] as! MambaCache
        let attention = cache![1]
        let entry = MLXArray.zeros([batch, 1, length, 1])
        if let batched = attention as? RuntimeBatchKVCache {
            for row in 0 ..< batch {
                let rowEntry = entry[row ..< row + 1]
                _ = batched.slots[row].update(keys: rowEntry, values: rowEntry)
            }
        } else {
            _ = attention.update(keys: entry, values: entry)
        }
        var logits = [Float](repeating: -100, count: tokens.count * vocabularySize)
        if batch > 1 {
            let conv = MLXArray(tokens.map { Float($0) }).reshaped([batch, length])
            mamba[0] = conv
            mamba[1] = conv + 1
            mamba.advance(length)
            for (index, token) in tokens.enumerated() {
                logits[index * vocabularySize + (token + 1) % vocabularySize] = 100
            }
            return MLXArray(logits, [batch, length, vocabularySize])
        }
        for (index, token) in tokens.enumerated() {
            let conv = MLXArray([Float(token)]).reshaped([1, 1])
            let recurrent = conv + 1
            mamba[0] = conv
            mamba[1] = recurrent
            if checkpoints.contains(index + 1), index + 1 < tokens.count {
                mamba.saveSpeculativeCheckpoint(
                    convState: conv, recurrentState: recurrent,
                    advancedBy: index + 1, rewinding: tokens.count - index - 1)
            }
            logits[index * vocabularySize + (token + 1) % vocabularySize] = 100
        }
        mamba.advance(tokens.count)
        return MLXArray(logits, [batch, length, vocabularySize])
    }
}

@Suite("Concurrent prompt lookup")
struct ConcurrentPromptLookupTests {
    private func runtime(probe: ScheduledLookupProbe) throws -> ConcurrentTextRuntime {
        try ConcurrentTextRuntime(
            model: ScheduledLookupModel(probe: probe),
            identity: .init(
                modelRevision: "transition", tokenizerRevision: "tokens",
                chatTemplateRevision: "template", adapterRevision: "none",
                cacheLayoutRevision: "simple"),
            configuration: .init(
                memoryBudgetBytes: 1_000_000, prefixCacheBytes: 0,
                workingMemoryBytes: 4096, maxOutputTokens: 32,
                prefillChunkSize: 16, promptLookupDraftTokens: 4))
    }

    private func collect(
        _ runtime: ConcurrentTextRuntime, promptLookup: Bool,
        stopTokenIDs: Set<Int> = [], temperature: Float = 0,
        tokens: [Int] = [2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
    ) async throws -> (
        tokens: [Int], mode: ConcurrentTextRuntime.ExecutionMode?,
        telemetry: SpeculativeDecodingTelemetry?, fallback: String?
    ) {
        let generation = try await runtime.generate(
            .init(
                tokens: tokens, maxTokens: 12,
                temperature: temperature, stopTokenIDs: stopTokenIDs,
                promptLookup: promptLookup))
        var tokens: [Int] = []
        var mode: ConcurrentTextRuntime.ExecutionMode?
        var telemetry: SpeculativeDecodingTelemetry?
        var fallback: String?
        for try await event in generation.events {
            switch event {
            case .token(let token): tokens.append(token)
            case .execution(let execution): mode = execution
            case .speculation(let result): telemetry = result
            case .fallback(let reason): fallback = reason
            default: break
            }
        }
        return (tokens, mode, telemetry, fallback)
    }

    @Test func lookupMatchesOrdinaryOutputWithFewerTargetForwards() async throws {
        let ordinaryProbe = ScheduledLookupProbe()
        let ordinaryRuntime = try runtime(probe: ordinaryProbe)
        let ordinary = try await collect(ordinaryRuntime, promptLookup: false)
        await ordinaryRuntime.shutdown()

        let lookupProbe = ScheduledLookupProbe()
        let lookupRuntime = try runtime(probe: lookupProbe)
        #expect(lookupRuntime.capabilities.promptLookupDecoding)
        let lookup = try await collect(lookupRuntime, promptLookup: true)
        await lookupRuntime.shutdown()

        #expect(lookup.tokens == ordinary.tokens)
        #expect(lookup.mode == .promptLookup)
        let telemetry = try #require(lookup.telemetry)
        #expect(telemetry.acceptedDraftTokenCount > 0)
        #expect(telemetry.draftModelCallCount == 0)
        #expect(lookupProbe.forwardCount < ordinaryProbe.forwardCount)
    }

    @Test func stopInsideAcceptedBlockSettlesTheRequest() async throws {
        let runtime = try runtime(probe: ScheduledLookupProbe())
        let stopped = try await collect(runtime, promptLookup: true, stopTokenIDs: [4])
        #expect(stopped.tokens == [2, 3])
        #expect(stopped.telemetry?.roundCount ?? 0 > 0)
        #expect(await runtime.status().activeRequests == 0)
        let next = try await collect(runtime, promptLookup: true)
        #expect(next.tokens.count == 12)
        await runtime.shutdown()
    }

    @Test func samplingRequestReportsLookupFallback() async throws {
        let runtime = try runtime(probe: ScheduledLookupProbe())
        let result = try await collect(runtime, promptLookup: true, temperature: 0.5)
        #expect(result.mode == .interleaved)
        #expect(result.fallback == "Prompt lookup requires greedy sampling")
        #expect(result.telemetry == nil)
        await runtime.shutdown()
    }

    @Test func hybridRecurrentCacheVerifiesAndRewindsWithoutADrafter() async throws {
        let probe = HybridLookupProbe()
        let runtime = try ConcurrentTextRuntime(
            model: ScheduledHybridLookupModel(probe: probe),
            identity: .init(
                modelRevision: "hybrid", tokenizerRevision: "tokens",
                chatTemplateRevision: "template", adapterRevision: "none",
                cacheLayoutRevision: "simple-mamba"),
            configuration: .init(
                memoryBudgetBytes: 1_000_000, prefixCacheBytes: 0,
                workingMemoryBytes: 4096, maxOutputTokens: 32,
                prefillChunkSize: 16, promptLookupDraftTokens: 3))
        #expect(runtime.capabilities.recurrentState)
        #expect(runtime.capabilities.promptLookupDecoding)
        let prompt = [2, 3, 4, 5, 9, 0, 1]
        let ordinary = try await collect(runtime, promptLookup: false, tokens: prompt)
        let lookup = try await collect(runtime, promptLookup: true, tokens: prompt)
        #expect(lookup.tokens == ordinary.tokens)
        #expect(lookup.mode == .promptLookup)
        #expect(lookup.telemetry?.acceptedDraftTokenCount ?? 0 > 0)
        #expect(lookup.telemetry?.rejectedDraftTokenCount ?? 0 > 0)
        #expect(probe.checkpointCount > 0)
        let stopped = try await collect(
            runtime, promptLookup: true, stopTokenIDs: [4], tokens: prompt)
        #expect(stopped.tokens == [2, 3])
        #expect(await runtime.status().activeRequests == 0)
        await runtime.shutdown()
    }

    @Test func hybridLookupInterleavesWithAnUnmatchedBatchedPeer() async throws {
        let probe = HybridLookupProbe()
        let gate = LookupForwardGate()
        let runtime = try ConcurrentTextRuntime(
            model: ScheduledHybridLookupModel(probe: probe, gate: gate),
            identity: .init(
                modelRevision: "hybrid-batch", tokenizerRevision: "tokens",
                chatTemplateRevision: "template", adapterRevision: "none",
                cacheLayoutRevision: "simple-mamba"),
            configuration: .init(
                memoryBudgetBytes: 1_000_000, prefixCacheBytes: 0,
                workingMemoryBytes: 4096, maxActiveRequests: 2,
                maxOutputTokens: 32, prefillChunkSize: 16,
                batchDecode: true, promptLookupDraftTokens: 3))
        #expect(runtime.capabilities.fusedBatching)
        let first = try await runtime.generate(
            .init(tokens: [8, 9, 0, 1, 2, 3, 5], maxTokens: 32, promptLookup: true))
        await gate.waitUntilStarted()
        let second = try await runtime.generate(
            .init(tokens: [1, 3, 4], maxTokens: 32, promptLookup: true))
        await gate.release()

        func collect(_ generation: ConcurrentTextRuntime.Generation) async throws
            -> (tokens: [Int], telemetry: SpeculativeDecodingTelemetry?)
        {
            var tokens: [Int] = []
            var telemetry: SpeculativeDecodingTelemetry?
            for try await event in generation.events {
                switch event {
                case .token(let token): tokens.append(token)
                case .speculation(let result): telemetry = result
                default: break
                }
            }
            return (tokens, telemetry)
        }
        async let firstResult = collect(first)
        async let secondResult = collect(second)
        let results = try await (firstResult, secondResult)
        #expect(results.0.tokens == (1 ... 32).map { (5 + $0) % 10 })
        #expect(results.1.tokens == (1 ... 32).map { (4 + $0) % 10 })
        #expect(results.0.telemetry?.acceptedDraftTokenCount ?? 0 > 0)
        #expect(results.1.telemetry == nil)
        #expect(probe.checkpointCount > 0)
        let status = await runtime.status()
        #expect(status.batchedForwardCount > 0)
        #expect(status.maximumBatchSize == 2)
        #expect(status.activeRequests == 0)
        await runtime.shutdown()
    }
}
