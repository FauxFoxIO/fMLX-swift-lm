// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

private typealias PrefixRuntime = ConcurrentTextRuntime

private struct PrefixResult {
    var tokens = [Int]()
    var reused = 0
    var prefill = [Int]()
    var fallbacks = [String]()
    var stop: GenerateStopReason?
}

private func collectPrefix(_ generation: PrefixRuntime.Generation) async throws -> PrefixResult {
    var result = PrefixResult()
    for try await event in generation.events {
        switch event {
        case .token(let token): result.tokens.append(token)
        case .admitted(let reused): result.reused = reused
        case .prefill(let processed, _): result.prefill.append(processed)
        case .fallback(let reason): result.fallbacks.append(reason)
        case .finished(let reason): result.stop = reason
        default: break
        }
    }
    return result
}

final class Qwen35MTPPrefixCacheTests: XCTestCase {
    private func identity(_ changed: Int? = nil) -> PrefixCacheIdentity {
        let values = (0 ..< 5).map { (index: Int) -> String in
            index == changed ? "changed" : "revision-\(index)"
        }
        return .init(
            modelRevision: values[0], tokenizerRevision: values[1],
            chatTemplateRevision: values[2], adapterRevision: values[3],
            cacheLayoutRevision: values[4])
    }

    private func modelConfiguration() throws -> Qwen35TextConfiguration {
        try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                """
                {"model_type":"qwen3_5_text","hidden_size":64,"num_hidden_layers":2,
                "intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,
                "head_dim":32,"linear_num_value_heads":2,"linear_num_key_heads":1,
                "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
                "vocab_size":100,"full_attention_interval":2,"mtp_num_hidden_layers":1,
                "tie_word_embeddings":true,"rope_theta":10000000.0,"partial_rotary_factor":0.25}
                """.utf8))
    }

    private func models() throws -> (Qwen35TextModel, Qwen35MTPDraftModel) {
        let config = try modelConfiguration()
        return (
            withRandomState(MLXRandom.RandomState(seed: 112)) { Qwen35TextModel(config) },
            withRandomState(MLXRandom.RandomState(seed: 113)) { Qwen35MTPDraftModel(config) }
        )
    }

    private func runtime(prefixBytes: Int = 2_000_000) throws -> PrefixRuntime {
        let config = try modelConfiguration()
        let model = withRandomState(MLXRandom.RandomState(seed: 112)) { Qwen35TextModel(config) }
        let head = withRandomState(MLXRandom.RandomState(seed: 113)) { Qwen35MTPDraftModel(config) }
        return try PrefixRuntime(
            model: model, identity: identity(),
            configuration: .init(
                memoryBudgetBytes: 64_000_000, prefixCacheBytes: prefixBytes,
                workingMemoryBytes: 2_000_000, maxPromptTokens: 512, maxOutputTokens: 256,
                prefillChunkSize: 4, streamBufferSize: 2048), drafter: head)
    }

    private func request(
        _ tokens: [Int], prefix: Int = 13, speculative: Bool = true, maxTokens: Int = 16,
        stop: Set<Int> = [], identity: PrefixCacheIdentity? = nil
    ) -> PrefixRuntime.Request {
        .init(
            tokens: tokens, maxTokens: maxTokens, stopTokenIDs: stop,
            prefixTokenCount: prefix, cacheIdentity: identity ?? self.identity(),
            speculative: speculative)
    }

    func testPairedPrefixesPreserveBranchesStopsAndExecutionKinds() async throws {
        let runtime = try runtime()
        let a = Array(1 ... 17)
        let b = Array(1 ... 13) + [23, 25, 27, 29]
        let coldA = try await collectPrefix(runtime.generate(request(a, prefix: 0)))
        let coldB = try await collectPrefix(runtime.generate(request(b, prefix: 0)))
        let seed = try await collectPrefix(runtime.generate(request(a)))
        XCTAssertEqual(seed.reused, 0)
        XCTAssertEqual(seed.tokens, coldA.tokens)
        let first = try await runtime.generate(request(a))
        let second = try await runtime.generate(request(b))
        async let warmA = collectPrefix(first)
        async let warmB = collectPrefix(second)
        let results = try await (warmA, warmB)
        XCTAssertEqual(results.0.reused, 12)
        XCTAssertEqual(results.1.reused, 12)
        XCTAssertEqual(results.0.prefill, [16, 17])
        XCTAssertEqual(results.1.prefill, [16, 17])
        XCTAssertEqual(results.0.tokens, coldA.tokens)
        XCTAssertEqual(results.1.tokens, coldB.tokens)
        XCTAssertTrue(results.0.fallbacks.isEmpty)
        XCTAssertTrue(results.1.fallbacks.isEmpty)

        let ordinaryCold = try await collectPrefix(
            runtime.generate(request(a, speculative: false)))
        let ordinaryWarm = try await collectPrefix(
            runtime.generate(request(a, speculative: false)))
        XCTAssertEqual(ordinaryCold.reused, 0)
        XCTAssertEqual(ordinaryWarm.reused, 13)
        XCTAssertEqual(ordinaryWarm.tokens, ordinaryCold.tokens)
        let afterOrdinary = try await collectPrefix(runtime.generate(request(a)))
        XCTAssertEqual(afterOrdinary.reused, 12)
        XCTAssertEqual(afterOrdinary.tokens, coldA.tokens)
        let status = await runtime.status()
        XCTAssertEqual(status.cachedPrefixes, 2)
        XCTAssertGreaterThan(status.cachedPrefixBytes, 0)

        let stop = try XCTUnwrap(coldA.tokens.last)
        let stopIndex = try XCTUnwrap(coldA.tokens.firstIndex(of: stop))
        let stopped = try await collectPrefix(runtime.generate(request(a, stop: [stop])))
        XCTAssertEqual(stopped.reused, 12)
        XCTAssertEqual(stopped.stop, .stop)
        XCTAssertEqual(stopped.tokens, Array(coldA.tokens.prefix(stopIndex)))
        let afterStop = try await collectPrefix(runtime.generate(request(a)))
        XCTAssertEqual(afterStop.tokens, coldA.tokens)
        await runtime.shutdown()
    }

    func testChunkBoundariesLookaheadAndEveryIdentityField() async throws {
        let runtime = try runtime()
        let tokens = Array(1 ... 17)
        let expected = try await collectPrefix(runtime.generate(request(tokens, prefix: 0)))
        for limit in [0, 1, 4, 5, 8, 9, 12, 13] {
            await runtime.clearPrefixCache()
            let seed = try await collectPrefix(runtime.generate(request(tokens, prefix: limit)))
            let hit = try await collectPrefix(runtime.generate(request(tokens, prefix: limit)))
            let reused = max(0, (limit - 1) / 4) * 4
            XCTAssertEqual(seed.reused, 0, "prefix \(limit)")
            XCTAssertEqual(hit.reused, reused, "prefix \(limit)")
            XCTAssertEqual(seed.tokens, expected.tokens, "prefix \(limit)")
            XCTAssertEqual(hit.tokens, expected.tokens, "prefix \(limit)")
        }
        var differentLookahead = tokens
        differentLookahead[12] = 71
        let changed = try await collectPrefix(runtime.generate(request(differentLookahead)))
        let changedCold = try await collectPrefix(
            runtime.generate(request(differentLookahead, prefix: 0)))
        XCTAssertEqual(changed.reused, 0)
        XCTAssertEqual(changed.tokens, changedCold.tokens)
        let smaller = try await collectPrefix(runtime.generate(request(tokens, prefix: 12)))
        XCTAssertEqual(smaller.reused, 0)
        XCTAssertEqual(smaller.tokens, expected.tokens)
        let before = await runtime.status()
        for field in 0 ..< 5 {
            let mismatch = try await collectPrefix(
                runtime.generate(request(tokens, identity: identity(field))))
            XCTAssertEqual(mismatch.reused, 0, "identity field \(field)")
            XCTAssertEqual(mismatch.tokens, expected.tokens)
        }
        let after = await runtime.status()
        XCTAssertEqual(after.cachedPrefixes, before.cachedPrefixes)
        XCTAssertEqual(after.cachedPrefixBytes, before.cachedPrefixBytes)
        await runtime.shutdown()
    }

    func testDisabledBudgetAndSharedLRUEviction() async throws {
        let tokens = Array(1 ... 17)
        for budget in [0, 1] {
            let runtime = try runtime(prefixBytes: budget)
            let first = try await collectPrefix(runtime.generate(request(tokens)))
            let second = try await collectPrefix(runtime.generate(request(tokens)))
            XCTAssertEqual(second.reused, 0)
            XCTAssertEqual(first.tokens, second.tokens)
            let status = await runtime.status()
            XCTAssertEqual(status.cachedPrefixes, 0)
            await runtime.shutdown()
        }
        let sizing = try runtime()
        _ = try await collectPrefix(sizing.generate(request(tokens)))
        let size = await sizing.status().cachedPrefixBytes
        XCTAssertGreaterThan(size, 0)
        await sizing.shutdown()
        let runtime = try runtime(prefixBytes: size)
        _ = try await collectPrefix(runtime.generate(request(tokens)))
        let firstHit = try await collectPrefix(runtime.generate(request(tokens)))
        XCTAssertEqual(firstHit.reused, 12)
        let ordinary = try await collectPrefix(
            runtime.generate(request(tokens, speculative: false)))
        XCTAssertEqual(ordinary.reused, 0)
        let status = await runtime.status()
        XCTAssertEqual(status.cachedPrefixes, 1)
        XCTAssertLessThanOrEqual(status.cachedPrefixBytes, size)
        let evicted = try await collectPrefix(runtime.generate(request(tokens)))
        XCTAssertEqual(evicted.reused, 0)
        XCTAssertEqual(evicted.tokens, firstHit.tokens)
        let ordinaryEvicted = try await collectPrefix(
            runtime.generate(request(tokens, speculative: false)))
        XCTAssertEqual(ordinaryEvicted.reused, 0)
        XCTAssertEqual(ordinaryEvicted.tokens, ordinary.tokens)
        await runtime.shutdown()
    }

    func testCancellationClearAndShutdownPreserveOtherBranches() async throws {
        let runtime = try runtime()
        let tokens = Array(1 ... 17)
        let expected = try await collectPrefix(runtime.generate(request(tokens)))
        let cancelled = try await runtime.generate(request(tokens, maxTokens: 256))
        var events = cancelled.events.makeAsyncIterator()
        var admitted = false
        while let event = try await events.next() {
            if case .admitted(let reused) = event {
                XCTAssertEqual(reused, 12)
                admitted = true
                break
            }
        }
        XCTAssertTrue(admitted)
        let peer = try await runtime.generate(request(tokens))
        await runtime.cancel(cancelled.id)
        await runtime.clearPrefixCache()
        let survivor = try await collectPrefix(peer)
        XCTAssertEqual(survivor.tokens, expected.tokens)
        var cancelledStop: GenerateStopReason?
        while let event = try await events.next() {
            if case .finished(let reason) = event { cancelledStop = reason }
        }
        XCTAssertEqual(cancelledStop, .cancelled)
        await runtime.clearPrefixCache()
        let cleared = try await collectPrefix(runtime.generate(request(tokens)))
        XCTAssertEqual(cleared.reused, 0)
        XCTAssertEqual(cleared.tokens, expected.tokens)
        let interrupted = try await runtime.generate(request(tokens, maxTokens: 256))
        var interruptedEvents = interrupted.events.makeAsyncIterator()
        while let event = try await interruptedEvents.next() {
            if case .admitted(let reused) = event {
                XCTAssertEqual(reused, 12)
                break
            }
        }
        await runtime.shutdown()
        let after = await runtime.status()
        XCTAssertEqual(after.activeRequests, 0)
        XCTAssertEqual(after.cachedPrefixes, 0)
        XCTAssertEqual(after.reservedBytes, 0)
        var interruptedStop: GenerateStopReason?
        while let event = try await interruptedEvents.next() {
            if case .finished(let reason) = event { interruptedStop = reason }
        }
        XCTAssertEqual(interruptedStop, .cancelled)
    }

    func testIteratorSnapshotsOwnBothCachesAndLogicalPosition() throws {
        let (model, head) = try models()
        try model.prepare()
        let tokens = Array(1 ... 17)
        let parameters = GenerateParameters(maxTokens: 16, temperature: 0)
        var cold = try MTPSpeculativeTokenIterator(
            scheduledPrompt: tokens, mainModel: model, drafter: head,
            mainCache: model.newCache(parameters: nil), parameters: parameters, blockSize: 2)
        XCTAssertNil(cold.snapshotScheduledPrefix())
        try cold.prepareScheduledChunk(Array(tokens[0 ..< 4]), nextPromptToken: tokens[4])
        let snapshot = try XCTUnwrap(cold.snapshotScheduledPrefix())
        XCTAssertEqual(snapshot.processedTokenCount, 4)
        func restored() throws -> MTPSpeculativeTokenIterator {
            try MTPSpeculativeTokenIterator(
                scheduledPrompt: tokens, mainModel: model, drafter: head,
                mainCache: model.newCache(parameters: nil), parameters: parameters, blockSize: 2,
                prefix: snapshot)
        }
        var warm = try restored()
        XCTAssertEqual(warm.mainCacheStorage.processedTokenCount, 4)
        XCTAssertEqual(warm.drafterState?.nextPosition, 4)
        let coldCaches = cold.mainCache + (try XCTUnwrap(cold.drafterState)).cache
        let warmCaches = warm.mainCache + (try XCTUnwrap(warm.drafterState)).cache
        for (a, b) in zip(coldCaches, warmCaches) {
            let original = try XCTUnwrap(a as? BaseKVCache)
            let restored = try XCTUnwrap(b as? BaseKVCache)
            XCTAssertFalse(original === restored)
            XCTAssertEqual(a.metaState, b.metaState)
            for (x, y) in zip(a.state, b.state) {
                XCTAssertEqual(x.shape, y.shape)
                XCTAssertTrue(arrayEqual(x, y).item(Bool.self))
            }
        }
        func finish(_ iterator: inout MTPSpeculativeTokenIterator) throws -> [Int] {
            for start in stride(from: 4, to: tokens.count, by: 4) {
                let end = min(start + 4, tokens.count)
                try iterator.prepareScheduledChunk(
                    Array(tokens[start ..< end]),
                    nextPromptToken: end < tokens.count ? tokens[end] : nil)
            }
            XCTAssertNil(iterator.snapshotScheduledPrefix())
            var output = [Int]()
            while let token = iterator.next() { output.append(token) }
            iterator.finalizeGeneration()
            return output
        }
        let expected = try finish(&cold)
        XCTAssertEqual(try finish(&warm), expected)
        var secondRestore = try restored()
        XCTAssertEqual(try finish(&secondRestore), expected)
    }
}
