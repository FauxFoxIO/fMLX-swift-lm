// Copyright © 2026 Faux Fox.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

private struct MTPPrefixLifecycleFixture: Decodable, Sendable {
    struct Prompt: Decodable, Sendable {
        let tokens: [Int]
        let prefixTokenCount: Int
        let seedTokens: [Int]?
    }
    let modelDirectory: String
    let mtpDirectory: String
    let identity: PrefixCacheIdentity
    let prompts: [Prompt]
    let stopTokenIDs: Set<Int>
}

final class Qwen35MTPPrefixLifecycleTests: XCTestCase {
    private enum LifecycleError: Error {
        case peerNotAdmitted, shutdownTargetNotAdmitted
    }

    func testLocalTrainedBranchCancellationAndClear() async throws {
        guard let path = ProcessInfo.processInfo.environment["FMLX_MTP_PREFIX_FIXTURE"] else {
            throw XCTSkip("Set FMLX_MTP_PREFIX_FIXTURE to a branched local fixture")
        }
        let fixture = try JSONDecoder().decode(
            MTPPrefixLifecycleFixture.self, from: Data(contentsOf: URL(filePath: path)))
        let branch = try XCTUnwrap(fixture.prompts.first { $0.seedTokens != nil })
        let seed = try XCTUnwrap(branch.seedTokens)
        let reuse = ((branch.prefixTokenCount - 1) / 128) * 128
        XCTAssertGreaterThan(reuse, 0)
        let runtime = try await ConcurrentTextRuntime(
            model: NativeTextModelLoader.load(directory: URL(filePath: fixture.modelDirectory)),
            identity: fixture.identity,
            configuration: .init(
                memoryBudgetBytes: 64 * 1024 * 1024 * 1024,
                prefixCacheBytes: 2 * 1024 * 1024 * 1024,
                workingMemoryBytes: 2 * 1024 * 1024 * 1024, maxActiveRequests: 2,
                prefillChunkSize: 128, streamBufferSize: 2048),
            drafter: NativeTextModelLoader.loadMTP(directory: URL(filePath: fixture.mtpDirectory)))
        func request(_ tokens: [Int], prefix: Int, uninterrupted: Bool = false)
            -> ConcurrentTextRuntime.Request
        {
            .init(
                tokens: tokens, maxTokens: uninterrupted ? 256 : 64,
                stopTokenIDs: uninterrupted ? [] : fixture.stopTokenIDs,
                prefixTokenCount: prefix, cacheIdentity: fixture.identity)
        }
        func collect(_ request: ConcurrentTextRuntime.Request, reused: Int) async throws -> [Int] {
            let generation = try await runtime.generate(request)
            var tokens = [Int]()
            for try await event in generation.events {
                switch event {
                case .admitted(let count): XCTAssertEqual(count, reused)
                case .token(let token): tokens.append(token)
                case .fallback(let reason): XCTFail(reason)
                default: break
                }
            }
            return tokens
        }
        do {
            let expected = try await collect(request(branch.tokens, prefix: 0), reused: 0)
            _ = try await collect(request(seed, prefix: branch.prefixTokenCount), reused: 0)
            let warm = try await collect(
                request(branch.tokens, prefix: branch.prefixTokenCount), reused: reuse)
            XCTAssertEqual(warm, expected)

            let cancelled = try await runtime.generate(
                request(seed, prefix: branch.prefixTokenCount, uninterrupted: true))
            var cancelledEvents = cancelled.events.makeAsyncIterator()
            var reachedToken = false
            while let event = try await cancelledEvents.next() {
                if case .admitted(let count) = event { XCTAssertEqual(count, reuse) }
                if case .token = event {
                    reachedToken = true
                    break
                }
            }
            XCTAssertTrue(reachedToken)
            let peer = try await runtime.generate(
                request(branch.tokens, prefix: branch.prefixTokenCount))
            var peerEvents = peer.events.makeAsyncIterator()
            guard case .admitted(let reused) = try await peerEvents.next() else {
                throw LifecycleError.peerNotAdmitted
            }
            XCTAssertEqual(reused, reuse)
            let overlapping = await runtime.status()
            XCTAssertEqual(overlapping.activeRequests, 2)
            await runtime.cancel(cancelled.id)
            await runtime.clearPrefixCache()
            let cleared = await runtime.status()
            XCTAssertEqual(cleared.cachedPrefixes, 0)
            XCTAssertEqual(cleared.activeRequests, 1)
            var peerTokens = [Int]()
            while let event = try await peerEvents.next() {
                if case .token(let token) = event { peerTokens.append(token) }
                if case .fallback(let reason) = event { XCTFail(reason) }
            }
            XCTAssertEqual(peerTokens, expected)
            var cancelledStop: GenerateStopReason?
            while let event = try await cancelledEvents.next() {
                if case .finished(let reason) = event { cancelledStop = reason }
            }
            XCTAssertEqual(cancelledStop, .cancelled)
            let coldAfterClear = try await collect(request(branch.tokens, prefix: 0), reused: 0)
            XCTAssertEqual(coldAfterClear, expected)
            _ = try await collect(request(seed, prefix: branch.prefixTokenCount), reused: 0)

            let interrupted = try await runtime.generate(
                request(branch.tokens, prefix: branch.prefixTokenCount, uninterrupted: true))
            var interruptedEvents = interrupted.events.makeAsyncIterator()
            guard case .admitted(let reused) = try await interruptedEvents.next() else {
                throw LifecycleError.shutdownTargetNotAdmitted
            }
            XCTAssertEqual(reused, reuse)
            let beforeShutdown = await runtime.status()
            XCTAssertEqual(beforeShutdown.activeRequests, 1)
            await runtime.shutdown()
            var interruptedStop: GenerateStopReason?
            while let event = try await interruptedEvents.next() {
                if case .finished(let reason) = event { interruptedStop = reason }
            }
            XCTAssertEqual(interruptedStop, .cancelled)
            let status = await runtime.status()
            XCTAssertEqual(status.activeRequests, 0)
            XCTAssertEqual(status.queuedRequests, 0)
            XCTAssertEqual(status.cachedPrefixes, 0)
            XCTAssertEqual(status.reservedBytes, 0)
            print(
                "FMLX_MTP_PREFIX_LIFECYCLE reused=\(reuse) branchTokens=\(expected.count) complete=true"
            )
        } catch {
            await runtime.shutdown()
            throw error
        }
    }
}
