// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXLMCommon
import XCTest
import os

@testable import MLXLLM

private struct NativePerformanceFixture: Decodable {
    struct Prompt: Decodable {
        let name: String
        let tokens: [Int]
    }
    let modelDirectory: String
    let mtpDirectory: String
    let identity: PrefixCacheIdentity
    let prompts: [Prompt]
    let outputTokens: Int
    let trials: Int
    let mode: String
    let candidateChunkSize: Int?
}

/// Local-only fixed-length measurements; trial zero is shape warmup, not a measured sample.
final class NativeInferencePerformanceTests: XCTestCase {
    func testLocalInference() async throws {
        guard let path = ProcessInfo.processInfo.environment["FMLX_PERFORMANCE_FIXTURE"] else {
            throw XCTSkip("Set FMLX_PERFORMANCE_FIXTURE to a local performance fixture")
        }
        let fixture = try JSONDecoder().decode(
            NativePerformanceFixture.self, from: Data(contentsOf: URL(filePath: path)))
        guard
            [
                "direct", "scheduled", "mtp", "phases", "mtp-verification-pairs",
                "mtp-logits-pairs", "mtp-checkpoint-pairs", "mtp-chunk-pairs",
                "mtp-cumulative-pairs",
                "mtp-retained-total-pairs",
                "mtp-decode-pairs",
                "mtp-greedy-pairs",
            ]
            .contains(
                fixture.mode), (fixture.candidateChunkSize ?? 512) > 0,
            fixture.outputTokens > 1, fixture.trials > 0,
            fixture.prompts.allSatisfy({ !$0.tokens.isEmpty })
        else {
            XCTFail("Invalid performance fixture")
            return
        }
        let model = try await NativeTextModelLoader.load(
            directory: URL(filePath: fixture.modelDirectory))
        model.train(false)
        eval(model)
        // Keep the companion resident even when measuring ordinary generation.
        let head = try await NativeTextModelLoader.loadMTP(
            directory: URL(filePath: fixture.mtpDirectory))
        head.train(false)
        eval(head)
        if fixture.mode.hasSuffix("-pairs") {
            try verificationPairs(fixture, model: model, head: head)
            return
        }
        let log = OSLog(subsystem: "fMLX.performance", category: .pointsOfInterest)
        if fixture.mode == "direct" || fixture.mode == "phases" {
            for trial in 0 ..< fixture.trials {
                for prompt in fixture.prompts {
                    Memory.peakMemory = 0
                    let start = ProcessInfo.processInfo.systemUptime
                    os_signpost(.begin, log: log, name: "Request")
                    var tokens: [Int] = []
                    var times: [Double] = []
                    var phases: [[String: Double]] = []
                    if fixture.mode == "direct" {
                        var iterator = try TokenIterator(
                            input: LMInput(tokens: MLXArray(prompt.tokens)), model: model,
                            parameters: .init(
                                maxTokens: fixture.outputTokens, temperature: 0,
                                prefill: .init(stepSize: 128, chunking: .remainder)))
                        while let token = iterator.next() {
                            tokens.append(token)
                            times.append(ProcessInfo.processInfo.systemUptime - start)
                        }
                        // TokenIterator pipelines a future step; include its drain in completion.
                        Stream().synchronize()
                    } else {
                        let cache = try model.newCache(parameters: nil)
                        var position = 0
                        while tokens.count < fixture.outputTokens {
                            try autoreleasepool {
                                let prefill = position < prompt.tokens.count
                                let input =
                                    prefill
                                    ? Array(
                                        prompt.tokens[
                                            position ..< min(position + 128, prompt.tokens.count)])
                                    : [tokens.last!]
                                let a = ProcessInfo.processInfo.systemUptime
                                let logits = try model.scheduledForward(
                                    MLXArray(input).expandedDimensions(axis: 0), cache: cache)
                                let b = ProcessInfo.processInfo.systemUptime
                                eval([logits] + cache.flatMap { $0.innerState() })
                                let c = ProcessInfo.processInfo.systemUptime
                                if prefill { position += input.count }
                                if position == prompt.tokens.count {
                                    tokens.append(
                                        argMax(logits[0..., -1, 0...], axis: -1).item(Int.self))
                                    times.append(ProcessInfo.processInfo.systemUptime - start)
                                }
                                let d = ProcessInfo.processInfo.systemUptime
                                phases.append([
                                    "prefill": prefill ? 1 : 0, "inputTokens": Double(input.count),
                                    "graphSeconds": b - a, "evalSeconds": c - b,
                                    "sampleSeconds": d - c,
                                ])
                            }
                        }
                        Stream().synchronize()
                    }
                    os_signpost(.end, log: log, name: "Request")
                    try report(
                        fixture, prompt: prompt, trial: trial, start: start,
                        tokens: tokens, times: times, phases: phases)
                }
            }
            withExtendedLifetime(head) {}
        } else {
            let runtime = try ConcurrentTextRuntime(
                model: model, identity: fixture.identity,
                configuration: .init(
                    memoryBudgetBytes: 48 * 1024 * 1024 * 1024,
                    prefixCacheBytes: 0, workingMemoryBytes: 512 * 1024 * 1024,
                    prefillChunkSize: 128, streamBufferSize: fixture.outputTokens + 1024),
                drafter: head)
            do {
                for trial in 0 ..< fixture.trials {
                    for prompt in fixture.prompts {
                        Memory.peakMemory = 0
                        let start = ProcessInfo.processInfo.systemUptime
                        os_signpost(.begin, log: log, name: "Request")
                        let generation = try await runtime.generate(
                            .init(
                                tokens: prompt.tokens, maxTokens: fixture.outputTokens,
                                speculative: fixture.mode == "mtp"))
                        var tokens: [Int] = []
                        var times: [Double] = []
                        var phases: [[String: Double]] = []
                        for try await event in generation.events {
                            switch event {
                            case .token(let token):
                                tokens.append(token)
                                times.append(ProcessInfo.processInfo.systemUptime - start)
                            case .prefill(let processed, _):
                                phases.append([
                                    "processedTokens": Double(processed),
                                    "completedSeconds": ProcessInfo.processInfo.systemUptime
                                        - start,
                                ])
                            case .speculation(let telemetry):
                                phases.append([
                                    "proposed": Double(telemetry.draftTokenCount),
                                    "accepted": Double(telemetry.acceptedDraftTokenCount),
                                ])
                            case .fallback(let reason): XCTFail(reason)
                            default: break
                            }
                        }
                        os_signpost(.end, log: log, name: "Request")
                        try report(
                            fixture, prompt: prompt, trial: trial, start: start,
                            tokens: tokens, times: times, phases: phases)
                    }
                }
                await runtime.shutdown()
            } catch {
                await runtime.shutdown()
                throw error
            }
        }
    }

    private func verificationPairs(
        _ fixture: NativePerformanceFixture, model: any ScheduledTextModel,
        head: Qwen35MTPDraftModel
    ) throws {
        let layers = model.modules().compactMap { $0 as? Qwen35DecoderLayer }
        XCTAssertFalse(layers.isEmpty)
        let candidateMode =
            switch fixture.mode {
            case "mtp-logits-pairs": "mtp-e3"
            case "mtp-checkpoint-pairs": "mtp-e4"
            case "mtp-chunk-pairs": "mtp-e6"
            case "mtp-cumulative-pairs": "mtp-cumulative"
            case "mtp-retained-total-pairs": "mtp-retained-total"
            case "mtp-decode-pairs": "mtp-decode-total"
            case "mtp-greedy-pairs": "mtp-e7"
            default: "mtp-e1"
            }
        for trial in 0 ..< fixture.trials {
            for prompt in fixture.prompts {
                for enabled in trial % 2 == 0 ? [false, true] : [true, false] {
                    for layer in layers {
                        layer.compiledVerificationEnabled =
                            ![
                                "mtp-verification-pairs", "mtp-cumulative-pairs",
                                "mtp-retained-total-pairs", "mtp-decode-pairs",
                            ]
                            .contains(
                                fixture.mode)
                            || enabled
                        layer.fusedVerificationCheckpointEnabled =
                            fixture.mode == "mtp-greedy-pairs"
                            || ([
                                "mtp-checkpoint-pairs", "mtp-cumulative-pairs",
                                "mtp-retained-total-pairs", "mtp-decode-pairs",
                            ]
                            .contains(fixture.mode)
                                && enabled)
                    }
                    Memory.peakMemory = 0
                    let start = ProcessInfo.processInfo.systemUptime
                    var iterator = try MTPSpeculativeTokenIterator(
                        scheduledPrompt: prompt.tokens, mainModel: model, drafter: head,
                        mainCache: model.newCache(parameters: nil),
                        parameters: .init(maxTokens: fixture.outputTokens, temperature: 0),
                        blockSize: 2)
                    iterator.jointGreedyVerificationEnabled =
                        (fixture.mode == "mtp-greedy-pairs" && enabled)
                        || (fixture.mode == "mtp-retained-total-pairs" && enabled)
                    let chunkSize =
                        fixture.mode == "mtp-chunk-pairs" && enabled
                        ? (fixture.candidateChunkSize ?? 512) : 128
                    let evaluateIntermediateLogits =
                        [
                            "mtp-logits-pairs", "mtp-cumulative-pairs",
                            "mtp-retained-total-pairs",
                        ].contains(fixture.mode)
                        ? !enabled
                        : ![
                            "mtp-chunk-pairs", "mtp-decode-pairs", "mtp-greedy-pairs",
                        ].contains(fixture.mode)
                    for position in stride(from: 0, to: prompt.tokens.count, by: chunkSize) {
                        let end = min(position + chunkSize, prompt.tokens.count)
                        try iterator.prepareScheduledChunk(
                            Array(prompt.tokens[position ..< end]),
                            nextPromptToken: end < prompt.tokens.count ? prompt.tokens[end] : nil,
                            evaluateIntermediateLogits: evaluateIntermediateLogits)
                    }
                    var tokens: [Int] = []
                    var times: [Double] = []
                    while let token = iterator.next() {
                        tokens.append(token)
                        times.append(ProcessInfo.processInfo.systemUptime - start)
                    }
                    Stream().synchronize()
                    try report(
                        fixture, prompt: prompt, trial: trial, start: start, tokens: tokens,
                        times: times,
                        phases: [
                            [
                                "proposed": Double(iterator.proposedCount),
                                "accepted": Double(iterator.acceptedCount),
                                "jointGreedyRounds": Double(iterator.jointGreedyVerificationCount),
                            ]
                        ],
                        mode: enabled ? candidateMode : "mtp-reference")
                }
            }
        }
    }

    private func report(
        _ fixture: NativePerformanceFixture, prompt: NativePerformanceFixture.Prompt,
        trial: Int, start: Double, tokens: [Int], times: [Double], phases: [[String: Double]],
        mode: String? = nil
    ) throws {
        let finished = ProcessInfo.processInfo.systemUptime - start
        XCTAssertEqual(tokens.count, fixture.outputTokens)
        let row: [String: Any] = [
            "mode": mode ?? fixture.mode, "prompt": prompt.name,
            "promptTokens": prompt.tokens.count,
            "trial": trial, "tokens": tokens, "tokenSeconds": times, "phases": phases,
            "finishedSeconds": finished, "peakMLXBytes": Memory.peakMemory,
            "activeMLXBytes": Memory.activeMemory, "cacheMLXBytes": Memory.cacheMemory,
            "thermalState": ProcessInfo.processInfo.thermalState.rawValue,
        ]
        print(
            "FMLX_NATIVE_PERFORMANCE "
                + String(decoding: try JSONSerialization.data(withJSONObject: row), as: UTF8.self))
    }
}
