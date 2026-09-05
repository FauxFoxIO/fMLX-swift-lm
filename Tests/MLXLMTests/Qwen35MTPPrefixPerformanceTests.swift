// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

private struct MTPPrefixFixture: Decodable, Sendable {
    struct Prompt: Decodable, Sendable {
        let name: String
        let tokens: [Int]
        let prefixTokenCount: Int
        let seedTokens: [Int]?
    }
    let modelDirectory: String
    let mtpDirectory: String
    let identity: PrefixCacheIdentity
    let prompts: [Prompt]
    let outputTokens: Int
    let stopTokenIDs: Set<Int>
    let trials: Int
}

private struct MTPPrefixMeasurement: Codable, Sendable {
    let name: String
    let trial: Int
    let mode: String
    let promptTokens: Int
    let prefixTokenCount: Int
    var ttftSeconds: Double?
    var finishedSeconds: Double?
    var tokens = [Int]()
    var reusedPrefixTokens = 0
    var prefillPositions = [Int]()
    var proposed = 0
    var accepted = 0
    var fallback = [String]()
    var stopReason: String?
    var peakMLXBytes = 0
    var activeMLXBytes = 0
    var cachedPrefixBytes = 0
    var thermalState = 0
}

final class Qwen35MTPPrefixPerformanceTests: XCTestCase {
    func testLocalMatchedPrefixReuse() async throws {
        guard let path = ProcessInfo.processInfo.environment["FMLX_MTP_PREFIX_FIXTURE"] else {
            throw XCTSkip("Set FMLX_MTP_PREFIX_FIXTURE to a local matched fixture")
        }
        let fixture = try JSONDecoder().decode(
            MTPPrefixFixture.self, from: Data(contentsOf: URL(filePath: path)))
        guard !fixture.prompts.isEmpty, (2 ... 21).contains(fixture.trials),
            (1 ... 256).contains(fixture.outputTokens), !fixture.stopTokenIDs.isEmpty,
            fixture.prompts.allSatisfy({
                $0.prefixTokenCount > 128 && $0.prefixTokenCount < $0.tokens.count
                    && $0.tokens.count <= 16_384
                    && ($0.seedTokens?.count ?? $0.tokens.count) == $0.tokens.count
                    && ($0.seedTokens ?? $0.tokens).prefix($0.prefixTokenCount)
                        .elementsEqual($0.tokens.prefix($0.prefixTokenCount))
            })
        else { return XCTFail("Invalid MTP prefix fixture") }
        let runtime = try await ConcurrentTextRuntime(
            model: NativeTextModelLoader.load(directory: URL(filePath: fixture.modelDirectory)),
            identity: fixture.identity,
            configuration: .init(
                memoryBudgetBytes: 64 * 1024 * 1024 * 1024,
                prefixCacheBytes: 2 * 1024 * 1024 * 1024,
                workingMemoryBytes: 2 * 1024 * 1024 * 1024, maxActiveRequests: 2,
                prefillChunkSize: 128, streamBufferSize: 2048),
            drafter: NativeTextModelLoader.loadMTP(directory: URL(filePath: fixture.mtpDirectory)))
        do {
            for prompt in fixture.prompts {
                await runtime.clearPrefixCache()
                let reuseCount = ((prompt.prefixTokenCount - 1) / 128) * 128
                func measure(_ mode: String, trial: Int) async throws -> MTPPrefixMeasurement {
                    Memory.peakMemory = 0
                    var result = MTPPrefixMeasurement(
                        name: prompt.name, trial: trial, mode: mode,
                        promptTokens: prompt.tokens.count, prefixTokenCount: prompt.prefixTokenCount
                    )
                    let start = ProcessInfo.processInfo.systemUptime
                    let generation = try await runtime.generate(
                        .init(
                            tokens: mode == "publish"
                                ? prompt.seedTokens ?? prompt.tokens : prompt.tokens,
                            maxTokens: fixture.outputTokens,
                            stopTokenIDs: fixture.stopTokenIDs,
                            prefixTokenCount: mode == "cold" ? 0 : prompt.prefixTokenCount,
                            cacheIdentity: fixture.identity))
                    for try await event in generation.events {
                        let elapsed = ProcessInfo.processInfo.systemUptime - start
                        switch event {
                        case .token(let token):
                            result.ttftSeconds = result.ttftSeconds ?? elapsed
                            result.tokens.append(token)
                        case .admitted(let reused): result.reusedPrefixTokens = reused
                        case .prefill(let processed, _): result.prefillPositions.append(processed)
                        case .finished(let reason):
                            result.finishedSeconds = elapsed
                            result.stopReason = String(describing: reason)
                        case .speculation(let telemetry):
                            result.proposed = telemetry.draftTokenCount
                            result.accepted = telemetry.acceptedDraftTokenCount
                        case .fallback(let reason): result.fallback.append(reason)
                        default: break
                        }
                    }
                    result.peakMLXBytes = Memory.peakMemory
                    result.activeMLXBytes = Memory.activeMemory
                    result.cachedPrefixBytes = await runtime.status().cachedPrefixBytes
                    result.thermalState = ProcessInfo.processInfo.thermalState.rawValue
                    print(
                        "FMLX_MTP_PREFIX "
                            + String(decoding: try JSONEncoder().encode(result), as: UTF8.self))
                    XCTAssertTrue(result.fallback.isEmpty)
                    XCTAssertNotNil(result.ttftSeconds)
                    XCTAssertNotNil(result.finishedSeconds)
                    XCTAssertEqual(result.reusedPrefixTokens, mode == "warm" ? reuseCount : 0)
                    return result
                }
                let seed = try await measure("publish", trial: -1)
                var expected: MTPPrefixMeasurement?
                for trial in 0 ..< fixture.trials {
                    var measurements = [MTPPrefixMeasurement]()
                    for mode in trial.isMultiple(of: 2) ? ["cold", "warm"] : ["warm", "cold"] {
                        measurements.append(try await measure(mode, trial: trial))
                    }
                    expected = expected ?? measurements[0]
                    for result in measurements {
                        XCTAssertEqual(result.tokens, expected!.tokens)
                        XCTAssertEqual(result.stopReason, expected!.stopReason)
                        XCTAssertEqual(result.proposed, expected!.proposed)
                        XCTAssertEqual(result.accepted, expected!.accepted)
                        XCTAssertGreaterThan(result.cachedPrefixBytes, 0)
                        if prompt.seedTokens == nil { XCTAssertEqual(result.tokens, seed.tokens) }
                    }
                }
            }
            await runtime.shutdown()
        } catch {
            await runtime.shutdown()
            throw error
        }
    }
}
