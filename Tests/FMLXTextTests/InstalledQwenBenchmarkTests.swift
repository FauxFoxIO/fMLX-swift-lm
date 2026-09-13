// Copyright © 2026 Faux Fox.

import FMLXText
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

private enum InstalledQwenBenchmarkMode: String, Decodable, Sendable {
    case ordinary
    case mtp
}

private enum InstalledQwenPrefixWorkload: String, Decodable, Sendable {
    case uncached
    case warmPrefix
}

private enum InstalledQwenDrafterKind: String, Decodable, Sendable {
    case embedded
    case standalone
}

private struct InstalledQwenBenchmarkConfiguration: Decodable, Sendable {
    struct Model: Decodable, Sendable {
        struct Drafter: Decodable, Sendable {
            let kind: InstalledQwenDrafterKind
            let path: String?
            let modelID: String?
        }

        let name: String
        let targetPath: String?
        let targetModelID: String?
        let drafter: Drafter?
    }

    let modelRoot: String?
    let models: [Model]
    let promptLengths: [Int]
    let prefillChunkSize: Int
    let outputTokens: Int
    let trials: Int
    let warmupTrials: Int
    let modes: [InstalledQwenBenchmarkMode]
    let prefixWorkloads: [InstalledQwenPrefixWorkload]
    let prompt: String
    let systemPrompt: String
    let memoryBudgetBytes: Int?
    let prefixCacheBytes: Int?
    let workingMemoryBytes: Int?
    let speculativeBlockSize: Int
    let speculativeAdaptationEnabled: Bool

    enum CodingKeys: String, CodingKey {
        case modelRoot
        case models
        case promptLengths
        case prefillChunkSize
        case outputTokens
        case trials
        case warmupTrials
        case modes
        case prefixWorkloads
        case prompt
        case systemPrompt
        case memoryBudgetBytes
        case prefixCacheBytes
        case workingMemoryBytes
        case speculativeBlockSize
        case speculativeAdaptationEnabled
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        modelRoot = try values.decodeIfPresent(String.self, forKey: .modelRoot)
        models = try values.decode([Model].self, forKey: .models)
        promptLengths =
            try values.decodeIfPresent([Int].self, forKey: .promptLengths)
            ?? [128, 512, 2_048, 8_192]
        prefillChunkSize = try values.decodeIfPresent(Int.self, forKey: .prefillChunkSize) ?? 128
        outputTokens = try values.decodeIfPresent(Int.self, forKey: .outputTokens) ?? 256
        trials = try values.decodeIfPresent(Int.self, forKey: .trials) ?? 4
        warmupTrials = try values.decodeIfPresent(Int.self, forKey: .warmupTrials) ?? 1
        modes =
            try values.decodeIfPresent([InstalledQwenBenchmarkMode].self, forKey: .modes)
            ?? [.ordinary, .mtp]
        prefixWorkloads =
            try values.decodeIfPresent(
                [InstalledQwenPrefixWorkload].self, forKey: .prefixWorkloads)
            ?? [.uncached, .warmPrefix]
        prompt =
            try values.decodeIfPresent(String.self, forKey: .prompt)
                ?? """
                Assess this design proposal for a local text-generation runtime. Explain the trade-offs
                among prefix caching, bounded memory, fair request admission, and deterministic token
                generation. Give concrete guidance that another engineer can apply without a UI.
                """
        systemPrompt =
            try values.decodeIfPresent(String.self, forKey: .systemPrompt)
            ?? "You are a careful systems engineer. Give a concise, accurate technical answer."
        memoryBudgetBytes = try values.decodeIfPresent(Int.self, forKey: .memoryBudgetBytes)
        prefixCacheBytes = try values.decodeIfPresent(Int.self, forKey: .prefixCacheBytes)
        workingMemoryBytes = try values.decodeIfPresent(Int.self, forKey: .workingMemoryBytes)
        speculativeBlockSize =
            try values.decodeIfPresent(Int.self, forKey: .speculativeBlockSize) ?? 4
        speculativeAdaptationEnabled =
            try values.decodeIfPresent(Bool.self, forKey: .speculativeAdaptationEnabled) ?? true
    }

    func validate() throws {
        guard !models.isEmpty, !prompt.isEmpty, !systemPrompt.isEmpty,
            !promptLengths.isEmpty, promptLengths.allSatisfy({ $0 > 1 }),
            prefillChunkSize > 0, outputTokens > 1, trials >= 2,
            (0 ..< trials).contains(warmupTrials),
            !modes.isEmpty, Set(modes).count == modes.count,
            !prefixWorkloads.isEmpty, Set(prefixWorkloads).count == prefixWorkloads.count,
            memoryBudgetBytes.map({ $0 > 0 }) ?? true,
            prefixCacheBytes.map({ $0 >= 0 }) ?? true,
            workingMemoryBytes.map({ $0 > 0 }) ?? true,
            speculativeBlockSize >= 2
        else {
            throw CheckpointTextError.invalidConfiguration(
                "Invalid installed Qwen benchmark configuration")
        }
        for model in models {
            guard !model.name.isEmpty, model.targetPath != nil || model.targetModelID != nil else {
                throw CheckpointTextError.invalidConfiguration(
                    "Each installed Qwen benchmark model needs targetPath or targetModelID")
            }
            if let drafter = model.drafter, drafter.kind == .standalone,
                drafter.path == nil && drafter.modelID == nil
            {
                throw CheckpointTextError.invalidConfiguration(
                    "A standalone MTP drafter needs path or modelID")
            }
        }
    }
}

private struct InstalledQwenLoadMeasurement: Encodable {
    let schemaVersion = 1
    let model: String
    let drafterKind: String?
    let textProcessorLoadSeconds: Double
    let targetModelLoadSeconds: Double
    let drafterLoadSeconds: Double?
}

private struct InstalledQwenMeasurement: Encodable {
    let schemaVersion = 1
    let model: String
    let mode: String
    let prefixWorkload: String
    let trial: Int
    let order: String
    let requestedPromptTokens: Int
    let promptTokens: Int
    let prefixTokenCount: Int
    let reusedPrefixTokens: Int
    let outputTokens: Int
    let tokenHash: String
    let parityWithPairedMode: Bool
    let submitToFirstMilliseconds: Double?
    let admittedToFirstMilliseconds: Double?
    let token1ToToken2Milliseconds: Double?
    let prefillTokens: Int
    let prefillTokensPerSecond: Double?
    let decodeTokensPerSecond: Double?
    let interTokenP50Milliseconds: Double?
    let interTokenP95Milliseconds: Double?
    let peakMLXBytes: Int
    let cachedMLXBytes: Int
    let runtimeCachedPrefixBytes: Int
    let mtpProposedTokens: Int
    let mtpAcceptedTokens: Int
    let mtpRounds: Int
    let speculativeBlockSize: Int
    let mtpFallbacks: [String]
    let stopReason: String?
}

private struct InstalledQwenSample {
    let requestedPromptTokens: Int
    let promptTokens: Int
    let prefixTokenCount: Int
    let reusedPrefixTokens: Int
    let tokens: [Int]
    let submittedAt: Double
    let admittedAt: Double?
    let firstTokenAt: Double?
    let finishedAt: Double?
    let prefillStartedAt: Double?
    let prefillFinishedAt: Double?
    let prefillTokens: Int
    let peakMLXBytes: Int
    let cachedMLXBytes: Int
    let runtimeCachedPrefixBytes: Int
    let mtpProposedTokens: Int
    let mtpAcceptedTokens: Int
    let mtpRounds: Int
    let mtpFallbacks: [String]
    let stopReason: String?
    let tokenTimes: [Double]
}

/// Local-only benchmark. It is disabled unless `FMLX_INSTALLED_QWEN_BENCHMARK_CONFIG`
/// points to JSON. `targetModelID` and standalone drafter `modelID` resolve below the
/// existing Flow application-support model root; paths and `modelRoot` allow other local layouts.
@Suite("Installed Qwen benchmark", .serialized)
struct InstalledQwenBenchmarkTests {
    private static let configurationPath = ProcessInfo.processInfo.environment[
        "FMLX_INSTALLED_QWEN_BENCHMARK_CONFIG"
    ]

    @Test(
        "Local Qwen ordinary and MTP benchmark",
        .enabled(if: configurationPath != nil)
    )
    func installedQwenBenchmark() async throws {
        let configuration = try Self.loadConfiguration()
        let defaultRoot = try Self.defaultModelRoot()

        for model in configuration.models {
            let targetDirectory = try Self.resolveDirectory(
                path: model.targetPath, modelID: model.targetModelID,
                root: configuration.modelRoot.map {
                    URL(filePath: $0, directoryHint: .isDirectory)
                } ?? defaultRoot)
            guard FileManager.default.fileExists(atPath: targetDirectory.path) else {
                throw CheckpointTextError.invalidConfiguration(
                    "Installed target is unavailable: \(model.name)")
            }

            let textStartedAt = Self.now()
            let text = try await CheckpointTextProcessor.load(directory: targetDirectory)
            let textProcessorLoadSeconds = Self.now() - textStartedAt

            let targetStartedAt = Self.now()
            let target = try await NativeTextModelLoader.load(directory: targetDirectory)
            let targetModelLoadSeconds = Self.now() - targetStartedAt
            guard target.vocabularySize == text.vocabularySize else {
                throw CheckpointTextError.invalidConfiguration(
                    "Target and tokenizer vocabulary sizes differ for \(model.name)")
            }

            let identity = try text.cacheIdentity(
                modelRevision: "installed-qwen-benchmark/\(model.name)",
                cacheLayoutRevision: "fmlx-text-v1/native")
            let drafterStartedAt = Self.now()
            let runtime: ConcurrentTextRuntime
            let drafterKind = model.drafter?.kind
            switch drafterKind {
            case .embedded:
                runtime = try ConcurrentTextRuntime(
                    model: target, identity: identity,
                    configuration: Self.runtimeConfiguration(configuration),
                    drafter: try await Self.loadEmbeddedDrafter(directory: targetDirectory))
            case .standalone:
                let spec = try #require(model.drafter)
                let directory = try Self.resolveDirectory(
                    path: spec.path, modelID: spec.modelID,
                    root: configuration.modelRoot.map {
                        URL(filePath: $0, directoryHint: .isDirectory)
                    } ?? defaultRoot)
                guard FileManager.default.fileExists(atPath: directory.path) else {
                    throw CheckpointTextError.invalidConfiguration(
                        "Installed MTP drafter is unavailable: \(model.name)")
                }
                runtime = try ConcurrentTextRuntime(
                    model: target, identity: identity,
                    configuration: Self.runtimeConfiguration(configuration),
                    drafter: try await NativeTextModelLoader.loadMTP(directory: directory))
            case nil:
                guard !configuration.modes.contains(.mtp) else {
                    throw CheckpointTextError.invalidConfiguration(
                        "MTP mode requires a drafter for \(model.name)")
                }
                runtime = try ConcurrentTextRuntime(
                    model: target, identity: identity,
                    configuration: Self.runtimeConfiguration(configuration))
            }
            let drafterLoadSeconds = drafterKind == nil ? nil : Self.now() - drafterStartedAt
            Self.emit(
                InstalledQwenLoadMeasurement(
                    model: model.name, drafterKind: drafterKind?.rawValue,
                    textProcessorLoadSeconds: textProcessorLoadSeconds,
                    targetModelLoadSeconds: targetModelLoadSeconds,
                    drafterLoadSeconds: drafterLoadSeconds))
            do {
                try await Self.measure(
                    model: model.name, text: text, runtime: runtime, configuration: configuration)
                await runtime.shutdown()
            } catch {
                await runtime.shutdown()
                throw error
            }
        }
    }

    private static func measure(
        model: String, text: CheckpointTextProcessor, runtime: ConcurrentTextRuntime,
        configuration: InstalledQwenBenchmarkConfiguration
    ) async throws {
        var stableTokens: [String: [Int]] = [:]
        for requestedLength in configuration.promptLengths {
            let tokens = try promptTokens(
                text: text, prompt: configuration.prompt, systemPrompt: configuration.systemPrompt,
                requestedLength: requestedLength)
            for workload in configuration.prefixWorkloads {
                for trial in 0 ..< configuration.trials {
                    let order: [InstalledQwenBenchmarkMode] =
                        trial.isMultiple(of: 2)
                        ? configuration.modes : Array(configuration.modes.reversed())
                    var paired: [InstalledQwenBenchmarkMode: InstalledQwenSample] = [:]
                    for mode in order {
                        let sample = try await run(
                            text: text, runtime: runtime, tokens: tokens,
                            requestedLength: requestedLength, mode: mode, workload: workload,
                            outputTokens: configuration.outputTokens)
                        paired[mode] = sample
                    }
                    let ordinary = paired[.ordinary]?.tokens
                    let mtp = paired[.mtp]?.tokens
                    let parity: Bool
                    if let ordinary, let mtp {
                        parity = ordinary == mtp
                        if !parity {
                            let mismatch = zip(ordinary, mtp).enumerated().first {
                                $0.element.0 != $0.element.1
                            }?.offset
                            let mismatchDescription = mismatch.map(String.init) ?? "length"
                            let ordinaryToken = mismatch.map { String(ordinary[$0]) } ?? "missing"
                            let mtpToken = mismatch.map { String(mtp[$0]) } ?? "missing"
                            print(
                                "FMLX_INSTALLED_QWEN_PARITY_MISMATCH index=\(mismatchDescription) ordinary=\(ordinaryToken) mtp=\(mtpToken)"
                            )
                        }
                        #expect(parity)
                    } else {
                        parity = true
                    }

                    guard trial >= configuration.warmupTrials else { continue }
                    for mode in order {
                        let sample = try #require(paired[mode])
                        let key = "\(requestedLength)/\(workload.rawValue)/\(mode.rawValue)"
                        if let expected = stableTokens[key] {
                            #expect(sample.tokens == expected)
                        } else {
                            stableTokens[key] = sample.tokens
                        }
                        let measurement = measurement(
                            model: model, mode: mode, workload: workload,
                            trial: trial - configuration.warmupTrials,
                            order: order.map(\.rawValue).joined(separator: "/"), sample: sample,
                            speculativeBlockSize: configuration.speculativeBlockSize,
                            parity: parity)
                        emit(measurement)
                    }
                }
            }
        }
    }

    private static func run(
        text: CheckpointTextProcessor, runtime: ConcurrentTextRuntime, tokens: [Int],
        requestedLength: Int, mode: InstalledQwenBenchmarkMode,
        workload: InstalledQwenPrefixWorkload, outputTokens: Int
    ) async throws -> InstalledQwenSample {
        await runtime.clearPrefixCache()
        let prefixTokenCount = workload == .uncached ? 0 : tokens.count - 1
        if workload == .warmPrefix {
            _ = try await collect(
                runtime: runtime,
                request: try request(
                    text: text, tokens: tokens, maxTokens: 1, mode: mode,
                    prefixTokenCount: prefixTokenCount, identity: runtime.identity),
                requestedLength: requestedLength)
        }
        Memory.peakMemory = 0
        return try await collect(
            runtime: runtime,
            request: try request(
                text: text, tokens: tokens, maxTokens: outputTokens, mode: mode,
                prefixTokenCount: prefixTokenCount, identity: runtime.identity),
            requestedLength: requestedLength)
    }

    private static func request(
        text: CheckpointTextProcessor, tokens: [Int], maxTokens: Int,
        mode: InstalledQwenBenchmarkMode, prefixTokenCount: Int, identity: PrefixCacheIdentity
    ) throws -> ConcurrentTextRuntime.Request {
        try text.validateContext(promptTokenCount: tokens.count, maximumOutputTokens: maxTokens)
        return ConcurrentTextRuntime.Request(
            tokens: tokens, maxTokens: maxTokens, temperature: 0, stopTokenIDs: text.stopTokenIDs,
            prefixTokenCount: prefixTokenCount, cacheIdentity: identity,
            speculative: mode == .mtp)
    }

    private static func collect(
        runtime: ConcurrentTextRuntime, request: ConcurrentTextRuntime.Request, requestedLength: Int
    ) async throws -> InstalledQwenSample {
        let submittedAt = now()
        let generation = try await runtime.generate(request)
        var admittedAt: Double?
        var firstTokenAt: Double?
        var finishedAt: Double?
        var prefillStartedAt: Double?
        var prefillFinishedAt: Double?
        var prefillTokens = 0
        var reusedPrefixTokens = 0
        var tokens: [Int] = []
        var tokenTimes: [Double] = []
        var proposed = 0
        var accepted = 0
        var rounds = 0
        var fallbacks: [String] = []
        var stopReason: String?
        for try await event in generation.events {
            let eventAt = now()
            switch event {
            case .admitted(let reused):
                admittedAt = eventAt
                reusedPrefixTokens = reused
                prefillStartedAt = eventAt
            case .prefill(let processed, _):
                prefillTokens = processed
                prefillFinishedAt = eventAt
            case .token(let token):
                firstTokenAt = firstTokenAt ?? eventAt
                tokens.append(token)
                tokenTimes.append(eventAt)
            case .finished(let reason):
                finishedAt = eventAt
                stopReason = String(describing: reason)
            case .speculation(let telemetry):
                proposed += telemetry.draftTokenCount
                accepted += telemetry.acceptedDraftTokenCount
                rounds += telemetry.roundCount
            case .fallback(let reason):
                fallbacks.append(reason)
            default:
                break
            }
        }
        let status = await runtime.status()
        return InstalledQwenSample(
            requestedPromptTokens: requestedLength, promptTokens: request.tokens.count,
            prefixTokenCount: request.prefixTokenCount, reusedPrefixTokens: reusedPrefixTokens,
            tokens: tokens, submittedAt: submittedAt, admittedAt: admittedAt,
            firstTokenAt: firstTokenAt, finishedAt: finishedAt,
            prefillStartedAt: prefillStartedAt, prefillFinishedAt: prefillFinishedAt,
            prefillTokens: prefillTokens, peakMLXBytes: Memory.peakMemory,
            cachedMLXBytes: Memory.cacheMemory, runtimeCachedPrefixBytes: status.cachedPrefixBytes,
            mtpProposedTokens: proposed, mtpAcceptedTokens: accepted, mtpRounds: rounds,
            mtpFallbacks: fallbacks, stopReason: stopReason, tokenTimes: tokenTimes)
    }

    private static func measurement(
        model: String, mode: InstalledQwenBenchmarkMode, workload: InstalledQwenPrefixWorkload,
        trial: Int, order: String, sample: InstalledQwenSample, speculativeBlockSize: Int,
        parity: Bool
    ) -> InstalledQwenMeasurement {
        let interTokenMilliseconds = zip(sample.tokenTimes, sample.tokenTimes.dropFirst()).map {
            ($1 - $0) * 1_000
        }
        let prefillSeconds: Double?
        if let startedAt = sample.prefillStartedAt, let finishedAt = sample.prefillFinishedAt {
            prefillSeconds = finishedAt - startedAt
        } else {
            prefillSeconds = nil
        }
        let decodeSeconds: Double?
        if let firstTokenAt = sample.firstTokenAt, let finishedAt = sample.finishedAt {
            decodeSeconds = finishedAt - firstTokenAt
        } else {
            decodeSeconds = nil
        }
        let evaluatedPrefillTokens = max(0, sample.prefillTokens - sample.reusedPrefixTokens)
        return InstalledQwenMeasurement(
            model: model, mode: mode.rawValue, prefixWorkload: workload.rawValue, trial: trial,
            order: order, requestedPromptTokens: sample.requestedPromptTokens,
            promptTokens: sample.promptTokens, prefixTokenCount: sample.prefixTokenCount,
            reusedPrefixTokens: sample.reusedPrefixTokens, outputTokens: sample.tokens.count,
            tokenHash: tokenHash(sample.tokens), parityWithPairedMode: parity,
            submitToFirstMilliseconds: duration(sample.submittedAt, sample.firstTokenAt),
            admittedToFirstMilliseconds: duration(sample.admittedAt, sample.firstTokenAt),
            token1ToToken2Milliseconds: interTokenMilliseconds.first,
            prefillTokens: evaluatedPrefillTokens,
            prefillTokensPerSecond: rate(Double(evaluatedPrefillTokens), over: prefillSeconds),
            decodeTokensPerSecond: rate(
                Double(max(0, sample.tokens.count - 1)), over: decodeSeconds),
            interTokenP50Milliseconds: percentile(interTokenMilliseconds, percentile: 0.50),
            interTokenP95Milliseconds: percentile(interTokenMilliseconds, percentile: 0.95),
            peakMLXBytes: sample.peakMLXBytes, cachedMLXBytes: sample.cachedMLXBytes,
            runtimeCachedPrefixBytes: sample.runtimeCachedPrefixBytes,
            mtpProposedTokens: sample.mtpProposedTokens,
            mtpAcceptedTokens: sample.mtpAcceptedTokens, mtpRounds: sample.mtpRounds,
            speculativeBlockSize: speculativeBlockSize,
            mtpFallbacks: sample.mtpFallbacks, stopReason: sample.stopReason)
    }

    private static func promptTokens(
        text: CheckpointTextProcessor, prompt: String, systemPrompt: String, requestedLength: Int
    ) throws -> [Int] {
        var content = prompt
        var tokens = try text.prepareChat(messages: [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": content],
        ])
        while tokens.count < requestedLength {
            content += "\n\n" + prompt
            tokens = try text.prepareChat(messages: [
                ["role": "system", "content": systemPrompt],
                ["role": "user", "content": content],
            ])
        }
        // Preserve the template's assistant-generation tail. Arbitrary prefix truncation can
        // leave an end-of-turn token last and make a valid benchmark stop before token one.
        let suffixCount = min(32, max(1, requestedLength / 4))
        return Array(tokens.prefix(requestedLength - suffixCount))
            + Array(tokens.suffix(suffixCount))
    }

    private static func runtimeConfiguration(
        _ configuration: InstalledQwenBenchmarkConfiguration
    ) -> ConcurrentTextRuntime.Configuration {
        let physicalMemory = Int(ProcessInfo.processInfo.physicalMemory)
        let workingMemoryBytes =
            configuration.workingMemoryBytes
            ?? min(2 * gibibyte, max(512 * mebibyte, physicalMemory / 16))
        return .init(
            memoryBudgetBytes: configuration.memoryBudgetBytes
                ?? max(workingMemoryBytes, physicalMemory - 4 * gibibyte),
            prefixCacheBytes: configuration.prefixCacheBytes
                ?? min(2 * gibibyte, max(512 * mebibyte, physicalMemory / 12)),
            workingMemoryBytes: workingMemoryBytes, maxActiveRequests: 1, maxQueuedRequests: 1,
            maxPromptTokens: configuration.promptLengths.max()! + configuration.outputTokens,
            maxOutputTokens: configuration.outputTokens,
            prefillChunkSize: configuration.prefillChunkSize,
            streamBufferSize: configuration.outputTokens
                + configuration.promptLengths.max()! / configuration.prefillChunkSize + 16,
            batchDecode: false,
            speculativeAdaptation: configuration.speculativeAdaptationEnabled ? .init() : nil,
            speculativeBlockSize: configuration.speculativeBlockSize)
    }

    private static func loadConfiguration() throws -> InstalledQwenBenchmarkConfiguration {
        let path = try #require(configurationPath)
        let configuration = try JSONDecoder().decode(
            InstalledQwenBenchmarkConfiguration.self, from: Data(contentsOf: URL(filePath: path)))
        try configuration.validate()
        return configuration
    }

    private static func loadEmbeddedDrafter(directory: URL) async throws
        -> sending Qwen35MTPDraftModel
    {
        guard let drafter = try await NativeTextModelLoader.loadEmbeddedMTP(directory: directory)
        else {
            throw CheckpointTextError.invalidConfiguration(
                "The installed checkpoint does not contain an embedded MTP head")
        }
        return drafter
    }

    private static func defaultModelRoot() throws -> URL {
        guard
            let root = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        else {
            throw CheckpointTextError.invalidConfiguration("Application Support is unavailable")
        }
        return root.appending(path: "Flow/Models", directoryHint: .isDirectory)
    }

    private static func resolveDirectory(path: String?, modelID: String?, root: URL) throws -> URL {
        if let path { return URL(filePath: path, directoryHint: .isDirectory) }
        guard let modelID, !modelID.isEmpty else {
            throw CheckpointTextError.invalidConfiguration(
                "A local model path or model ID is required")
        }
        return root.appending(path: modelID, directoryHint: .isDirectory)
    }

    private static func emit<T: Encodable>(_ value: T) {
        guard let data = try? JSONEncoder().encode(value) else { return }
        print("FMLX_INSTALLED_QWEN_BENCHMARK \(String(decoding: data, as: UTF8.self))")
    }

    private static func tokenHash(_ tokens: [Int]) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for token in tokens {
            var value = UInt64(bitPattern: Int64(token)).littleEndian
            for _ in 0 ..< MemoryLayout<UInt64>.size {
                hash ^= value & 0xff
                hash &*= 0x0000_0100_0000_01b3
                value >>= 8
            }
        }
        return String(hash, radix: 16)
    }

    private static func duration(_ start: Double?, _ end: Double?) -> Double? {
        guard let start, let end else { return nil }
        return (end - start) * 1_000
    }

    private static func rate(_ count: Double, over seconds: Double?) -> Double? {
        guard let seconds, seconds > 0 else { return nil }
        return count / seconds
    }

    private static func percentile(_ values: [Double], percentile: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        let index = Int((Double(values.count) * percentile).rounded(.up)) - 1
        return values.sorted()[max(0, index)]
    }

    private static func now() -> Double { ProcessInfo.processInfo.systemUptime }

    private static let mebibyte = 1_024 * 1_024
    private static let gibibyte = 1_024 * mebibyte
}
