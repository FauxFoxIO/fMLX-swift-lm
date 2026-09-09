// Copyright © 2026 Faux Fox.

import MLX
import MLXLLM
import MLXNN
import XCTest

@testable import MLXLMCommon

private typealias Runtime = ConcurrentTextRuntime

/// Uses real MLX cache writes and context-dependent logits without downloaded weights.
private final class ScheduledChecksumModel: Module, ScheduledTextModel {
    let vocabularySize = 100
    let scheduledCacheBytesPerToken = 8
    let failOnToken: Int?

    init(failOnToken: Int? = nil) { self.failOnToken = failOnToken }

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

    func scheduledForward(_ tokens: MLXArray, cache: [KVCache]) throws -> MLXArray {
        let logits = callAsFunction(tokens, cache: cache)
        if let failOnToken, tokens.asArray(Int32.self).contains(Int32(failOnToken)) {
            throw ConcurrentTextRuntimeError.invalidRequest
        }
        return logits
    }
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
            for speculative in bits == 0 ? [false, true] : [false] {
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
