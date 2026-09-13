// Copyright © 2026 Faux Fox.

import FMLXText
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

private struct ExpertStreamingRangeBenchmarkConfiguration: Decodable, Sendable {
    enum CodingKeys: String, CodingKey {
        case checkpointPath
        case expertTensorPrefixes
        case maximumRanges
        case bytesPerRead
        case rangeTrials
        case outputTokens
        case prompt
        case memoryBudgetBytes
        case prefixCacheBytes
        case workingMemoryBytes
        case streamedExpertBytesPerLayer
        case iPhoneDecodeTokensPerSecond = "iphoneDecodeTokensPerSecond"
        case iPhoneFlashBytesPerSecond = "iphoneFlashBytesPerSecond"
        case routeTracePath
    }

    let checkpointPath: String?
    let expertTensorPrefixes: [String]
    let maximumRanges: Int
    let bytesPerRead: Int
    let rangeTrials: Int
    let outputTokens: Int
    let prompt: String
    let memoryBudgetBytes: Int?
    let prefixCacheBytes: Int?
    let workingMemoryBytes: Int?
    let streamedExpertBytesPerLayer: Int?
    let iPhoneDecodeTokensPerSecond: Double
    let iPhoneFlashBytesPerSecond: Double?
    let routeTracePath: String?

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        checkpointPath = try values.decodeIfPresent(String.self, forKey: .checkpointPath)
        expertTensorPrefixes =
            try values.decodeIfPresent([String].self, forKey: .expertTensorPrefixes)
            ?? [".experts."]
        maximumRanges = try values.decodeIfPresent(Int.self, forKey: .maximumRanges) ?? 24
        bytesPerRead = try values.decodeIfPresent(Int.self, forKey: .bytesPerRead) ?? 256 * 1_024
        rangeTrials = try values.decodeIfPresent(Int.self, forKey: .rangeTrials) ?? 3
        outputTokens = try values.decodeIfPresent(Int.self, forKey: .outputTokens) ?? 64
        prompt =
            try values.decodeIfPresent(String.self, forKey: .prompt)
            ?? "Explain why a streaming model runtime must keep output ordering deterministic."
        memoryBudgetBytes = try values.decodeIfPresent(Int.self, forKey: .memoryBudgetBytes)
        prefixCacheBytes = try values.decodeIfPresent(Int.self, forKey: .prefixCacheBytes)
        workingMemoryBytes = try values.decodeIfPresent(Int.self, forKey: .workingMemoryBytes)
        streamedExpertBytesPerLayer = try values.decodeIfPresent(
            Int.self, forKey: .streamedExpertBytesPerLayer)
        iPhoneDecodeTokensPerSecond =
            try values.decodeIfPresent(Double.self, forKey: .iPhoneDecodeTokensPerSecond) ?? 20
        iPhoneFlashBytesPerSecond = try values.decodeIfPresent(
            Double.self, forKey: .iPhoneFlashBytesPerSecond)
        routeTracePath = try values.decodeIfPresent(String.self, forKey: .routeTracePath)
    }

    func validate() throws {
        guard !expertTensorPrefixes.isEmpty, expertTensorPrefixes.allSatisfy({ !$0.isEmpty }),
            maximumRanges > 0, bytesPerRead > 0, rangeTrials >= 2, outputTokens > 1,
            !prompt.isEmpty, iPhoneDecodeTokensPerSecond.isFinite,
            iPhoneDecodeTokensPerSecond > 0,
            iPhoneFlashBytesPerSecond.map({ $0.isFinite && $0 > 0 }) ?? true,
            memoryBudgetBytes.map({ $0 > 0 }) ?? true,
            prefixCacheBytes.map({ $0 >= 0 }) ?? true,
            workingMemoryBytes.map({ $0 > 0 }) ?? true,
            streamedExpertBytesPerLayer.map({ $0 >= 0 }) ?? true
        else {
            throw CheckpointTextError.invalidConfiguration(
                "Invalid expert streaming range benchmark configuration")
        }
    }
}

private struct ExpertStreamingRange: Sendable {
    let reader: SafetensorRangeReader
    let tensor: String
    let leadingRange: Range<UInt64>
}

private struct ExpertStreamingRangeStats: Encodable {
    let cache: String
    let selectedRanges: Int
    let reads: Int
    let bytesRead: Int
    let bytesPerRead: Double
    let p50Milliseconds: Double?
    let p95Milliseconds: Double?
}

private struct ExpertStreamingGenerationSample {
    let tokens: [Int]
    let tokenTimes: [Double]
    let reusedPrefixTokens: Int
    let timeToFirstTokenSeconds: Double?
}

private struct ExpertRouteTrace: Decodable {
    struct Token: Decodable {
        let layers: [[Int]]
    }

    let expertBytes: Int?
    let tokens: [Token]
}

private struct ExpertRouteTraceLRUHitRate: Encodable {
    let layer: Int
    let hotExpertCount: Int
    let accesses: Int
    let hitRate: Double
}

private struct ExpertRouteTraceAnalysis: Encodable {
    let tokens: Int
    let layers: Int
    let lruHitRates: [ExpertRouteTraceLRUHitRate]
    let previousTokenExpertReuseRate: Double?
    let mtpTwoRowUnionExpertsPerPair: Double?
    let requiredFlashBytesPerToken: Double?
    let mtpTwoRowFlashBytesPerToken: Double?
}

private struct ExpertStreamingBenchmarkMeasurement: Encodable {
    let schemaVersion = 1
    let checkpoint: String
    let loadPolicy: String
    let maximumResidentWeightBytes: Int
    let routedExpertWeightBytes: Int
    let selectedTensorPrefixes: [String]
    let rangeSelection: String
    let coldReads: ExpertStreamingRangeStats
    let warmReads: ExpertStreamingRangeStats
    let outputTokens: Int
    let outputParity: Bool
    let reusedPrefixTokens: Int
    let decodeTokensPerSecond: Double?
    let interTokenP50Milliseconds: Double?
    let interTokenP95Milliseconds: Double?
    let peakMLXBytes: Int
    let activeMLXBytes: Int
    let cachedMLXBytes: Int
    let runtimeCachedPrefixBytes: Int
    let iPhoneTargetDecodeTokensPerSecond: Double
    let iPhoneRequiredFlashBytesPerSecond: Double?
    let iPhoneProjectedDecodeTokensPerSecond: Double?
    let iPhoneProjectedDecodeSeconds: Double?
    let hostTimeToFirstTokenSeconds: Double?
    let routeTrace: ExpertRouteTraceAnalysis?
}

/// A local-only microbenchmark for the streamed expert-weight path. It is disabled unless
/// `FMLX_EXPERT_STREAMING_BENCHMARK_CONFIG` names a JSON configuration file.
@Suite("Expert streaming range benchmark", .serialized)
struct ExpertStreamingRangeBenchmarkTests {
    private static let configurationPath = ProcessInfo.processInfo.environment[
        "FMLX_EXPERT_STREAMING_BENCHMARK_CONFIG"
    ]
    private static let offlineRouteTracePath = ProcessInfo.processInfo.environment[
        "FMLX_EXPERT_ROUTE_TRACE"
    ]

    @Test("Offline expert route-trace analysis", .enabled(if: offlineRouteTracePath != nil))
    func offlineRouteTraceAnalysis() throws {
        let path = try #require(Self.offlineRouteTracePath)
        let analysis = try Self.analyzeRouteTrace(at: URL(filePath: path))
        guard let data = try? JSONEncoder().encode(analysis) else { return }
        print("FMLX_EXPERT_ROUTE_TRACE " + String(decoding: data, as: UTF8.self))
    }

    @Test(
        "Selected expert reads, warm cache, and deterministic streaming",
        .enabled(if: configurationPath != nil)
    )
    func selectedExpertReadsAndStreamingParity() async throws {
        let configuration = try Self.loadConfiguration()
        let checkpoint = try Self.checkpointDirectory(configuration)
        let selection = try Self.selectedRanges(in: checkpoint, configuration: configuration)
        let routeTrace = try configuration.routeTracePath.map {
            try Self.analyzeRouteTrace(at: URL(filePath: $0))
        }

        // Foundation cannot evict the system page cache, so this is the first pass made by
        // this harness; the following passes are the warmed comparison.
        let coldReads = try await Self.measureRanges(
            selection.ranges, trials: 1, cache: "cold-first-pass")
        let warmReads = try await Self.measureRanges(
            selection.ranges, trials: configuration.rangeTrials, cache: "warm-page-cache")

        let text = try await CheckpointTextProcessor.load(directory: checkpoint)
        guard text.stopStrings.isEmpty else {
            throw CheckpointTextError.stringStopsRequireTextGeneration
        }
        let prompt = try text.prepareChat(
            messages: [["role": "user", "content": configuration.prompt]])
        try text.validateContext(
            promptTokenCount: prompt.count, maximumOutputTokens: configuration.outputTokens)

        let loadPolicy: NativeTextModelLoadPolicy =
            if let bytes = configuration.streamedExpertBytesPerLayer {
                .streamedExperts(.init(maximumResidentBytesPerLayer: bytes))
            } else {
                .resident
            }
        let loadPlan = try NativeTextModelLoader.loadPlan(
            directory: checkpoint, policy: loadPolicy)
        let model = try await NativeTextModelLoader.load(
            directory: checkpoint, policy: loadPolicy)
        let runtime = try ConcurrentTextRuntime(
            model: model,
            identity: try text.cacheIdentity(
                modelRevision: "expert-streaming/\(checkpoint.lastPathComponent)",
                cacheLayoutRevision: "fmlx-text-v1/native"),
            configuration: Self.runtimeConfiguration(configuration))
        defer { Task { await runtime.shutdown() } }

        Memory.peakMemory = 0
        let cold = try await Self.collect(
            runtime: runtime, text: text, tokens: prompt, outputTokens: configuration.outputTokens,
            cachePrefix: false)
        _ = try await Self.collect(
            runtime: runtime, text: text, tokens: prompt, outputTokens: 1, cachePrefix: true)
        let warm = try await Self.collect(
            runtime: runtime, text: text, tokens: prompt, outputTokens: configuration.outputTokens,
            cachePrefix: true)
        let status = await runtime.status()

        let parity = cold.tokens == warm.tokens
        #expect(!cold.tokens.isEmpty)
        #expect(parity)
        #expect(warm.reusedPrefixTokens > 0)

        let interTokenMilliseconds = zip(warm.tokenTimes, warm.tokenTimes.dropFirst()).map {
            ($1 - $0) * 1_000
        }
        let decodeSeconds = warm.tokenTimes.first.flatMap { first in
            warm.tokenTimes.last.map { $0 - first }
        }
        let decodeTokensPerSecond = decodeSeconds.flatMap { duration in
            duration > 0 ? Double(max(warm.tokens.count - 1, 0)) / duration : nil
        }
        let flashBytesPerToken =
            routeTrace?.mtpTwoRowFlashBytesPerToken
            ?? routeTrace?.requiredFlashBytesPerToken
        let requiredFlashBytesPerSecond = flashBytesPerToken.map {
            $0 * configuration.iPhoneDecodeTokensPerSecond
        }
        let iPhoneTokensPerSecond = configuration.iPhoneFlashBytesPerSecond.flatMap { bandwidth in
            flashBytesPerToken.map {
                min(configuration.iPhoneDecodeTokensPerSecond, bandwidth / $0)
            }
        }
        Self.emit(
            ExpertStreamingBenchmarkMeasurement(
                checkpoint: checkpoint.path,
                loadPolicy: configuration.streamedExpertBytesPerLayer == nil
                    ? "resident" : "streamedExperts",
                maximumResidentWeightBytes: loadPlan.maximumResidentWeightBytes,
                routedExpertWeightBytes: loadPlan.routedExpertWeightBytes,
                selectedTensorPrefixes: configuration.expertTensorPrefixes,
                rangeSelection: selection.description, coldReads: coldReads, warmReads: warmReads,
                outputTokens: warm.tokens.count, outputParity: parity,
                reusedPrefixTokens: warm.reusedPrefixTokens,
                decodeTokensPerSecond: decodeTokensPerSecond,
                interTokenP50Milliseconds: Self.percentile(interTokenMilliseconds, at: 0.50),
                interTokenP95Milliseconds: Self.percentile(interTokenMilliseconds, at: 0.95),
                peakMLXBytes: Memory.peakMemory, activeMLXBytes: Memory.activeMemory,
                cachedMLXBytes: Memory.cacheMemory,
                runtimeCachedPrefixBytes: status.cachedPrefixBytes,
                iPhoneTargetDecodeTokensPerSecond: configuration.iPhoneDecodeTokensPerSecond,
                iPhoneRequiredFlashBytesPerSecond: requiredFlashBytesPerSecond,
                iPhoneProjectedDecodeTokensPerSecond: iPhoneTokensPerSecond,
                iPhoneProjectedDecodeSeconds: iPhoneTokensPerSecond.map {
                    Double(warm.tokens.count) / $0
                },
                hostTimeToFirstTokenSeconds: warm.timeToFirstTokenSeconds,
                routeTrace: routeTrace))
    }

    private static func loadConfiguration() throws -> ExpertStreamingRangeBenchmarkConfiguration {
        let path = try #require(configurationPath)
        let configuration = try JSONDecoder().decode(
            ExpertStreamingRangeBenchmarkConfiguration.self,
            from: Data(contentsOf: URL(filePath: path)))
        try configuration.validate()
        return configuration
    }

    private static func checkpointDirectory(
        _ configuration: ExpertStreamingRangeBenchmarkConfiguration
    ) throws -> URL {
        if let path = configuration.checkpointPath {
            return URL(filePath: path, directoryHint: .isDirectory)
        }
        guard
            let support = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        else {
            throw CheckpointTextError.invalidConfiguration("Application Support is unavailable")
        }
        return support.appending(
            path: "Flow/Models/OsaurusAI/Qwen3.6-35B-A3B-MXFP4-MTP",
            directoryHint: .isDirectory)
    }

    private static func selectedRanges(
        in checkpoint: URL, configuration: ExpertStreamingRangeBenchmarkConfiguration
    ) throws -> (ranges: [ExpertStreamingRange], description: String) {
        let files = try FileManager.default.contentsOfDirectory(
            at: checkpoint, includingPropertiesForKeys: [.isRegularFileKey]
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var matching: [ExpertStreamingRange] = []
        var fallback: [ExpertStreamingRange] = []
        for file in files {
            let reader = try SafetensorRangeReader(url: file)
            for tensor in reader.tensors {
                guard let leadingDimension = tensor.shape.first, leadingDimension > 0 else {
                    continue
                }
                let rowBytes = tensor.byteCount / leadingDimension
                let requestedRows = max(
                    UInt64(1), UInt64(configuration.bytesPerRead) / max(rowBytes, 1))
                let leadingRange = 0 ..< min(leadingDimension, requestedRows)
                let span = ExpertStreamingRange(
                    reader: reader, tensor: tensor.name, leadingRange: leadingRange)
                fallback.append(span)
                if configuration.expertTensorPrefixes.contains(where: { span.tensor.contains($0) })
                {
                    matching.append(span)
                }
            }
        }
        let ranges = Array(
            (matching.isEmpty ? fallback : matching).prefix(configuration.maximumRanges))
        guard !ranges.isEmpty else {
            throw CheckpointTextError.invalidConfiguration("No safetensor tensor ranges were found")
        }
        return (ranges, matching.isEmpty ? "fallback-first-tensors" : "expert-tensor-prefixes")
    }

    private static func measureRanges(
        _ ranges: [ExpertStreamingRange], trials: Int, cache: String
    ) async throws -> ExpertStreamingRangeStats {
        var times: [Double] = []
        var bytesRead = 0
        for _ in 0 ..< trials {
            for range in ranges {
                let startedAt = ProcessInfo.processInfo.systemUptime
                let data = try await range.reader.readSlice(
                    named: range.tensor, leadingRange: range.leadingRange)
                bytesRead += data.count
                times.append((ProcessInfo.processInfo.systemUptime - startedAt) * 1_000)
            }
        }
        return ExpertStreamingRangeStats(
            cache: cache, selectedRanges: ranges.count,
            reads: times.count, bytesRead: bytesRead,
            bytesPerRead: Double(bytesRead) / Double(max(times.count, 1)),
            p50Milliseconds: Self.percentile(times, at: 0.50),
            p95Milliseconds: Self.percentile(times, at: 0.95))
    }

    private static func runtimeConfiguration(
        _ configuration: ExpertStreamingRangeBenchmarkConfiguration
    ) -> ConcurrentTextRuntime.Configuration {
        let physicalMemory = Int(ProcessInfo.processInfo.physicalMemory)
        let memoryBudget = configuration.memoryBudgetBytes ?? physicalMemory * 7 / 8
        let prefixCache = configuration.prefixCacheBytes ?? 128 * 1_024 * 1_024
        let workingMemory = configuration.workingMemoryBytes ?? 512 * 1_024 * 1_024
        return .init(
            memoryBudgetBytes: memoryBudget, prefixCacheBytes: prefixCache,
            workingMemoryBytes: workingMemory, maxActiveRequests: 1, maxQueuedRequests: 1,
            maxPromptTokens: 8_192, maxOutputTokens: configuration.outputTokens,
            prefillChunkSize: 128, streamBufferSize: configuration.outputTokens + 16,
            batchDecode: true)
    }

    /// The optional JSON trace is `{ "expertBytes": Int?, "tokens": [{"layers": [[Int]]}] }`.
    /// Each token lists selected expert IDs for every layer; the exact MTP two-row metric uses
    /// adjacent trace tokens, preserving layer identity when it forms each union.
    private static func analyzeRouteTrace(at url: URL) throws -> ExpertRouteTraceAnalysis {
        let trace = try JSONDecoder().decode(ExpertRouteTrace.self, from: Data(contentsOf: url))
        guard !trace.tokens.isEmpty, let layerCount = trace.tokens.first?.layers.count,
            layerCount > 0, trace.tokens.allSatisfy({ $0.layers.count == layerCount }),
            trace.tokens.allSatisfy({
                $0.layers.allSatisfy {
                    $0.allSatisfy({ $0 >= 0 }) && Set($0).count == $0.count
                }
            }),
            trace.expertBytes.map({ $0 > 0 }) ?? true
        else {
            throw CheckpointTextError.invalidConfiguration("Invalid expert route trace")
        }

        let capacities = [8, 16, 32, 64, 128]
        var lruHitRates: [ExpertRouteTraceLRUHitRate] = []
        for capacity in capacities {
            var recency = Array(repeating: [Int](), count: layerCount)
            var hits = Array(repeating: 0, count: layerCount)
            var accesses = Array(repeating: 0, count: layerCount)
            for token in trace.tokens {
                for (layer, experts) in token.layers.enumerated() {
                    for expert in experts {
                        accesses[layer] += 1
                        if let index = recency[layer].firstIndex(of: expert) {
                            hits[layer] += 1
                            recency[layer].remove(at: index)
                        }
                        recency[layer].insert(expert, at: 0)
                        if recency[layer].count > capacity { recency[layer].removeLast() }
                    }
                }
            }
            for layer in 0 ..< layerCount {
                lruHitRates.append(
                    ExpertRouteTraceLRUHitRate(
                        layer: layer, hotExpertCount: capacity, accesses: accesses[layer],
                        hitRate: accesses[layer] == 0
                            ? 0 : Double(hits[layer]) / Double(accesses[layer])))
            }
        }

        var previousTokenAccesses = 0
        var previousTokenHits = 0
        var twoRowUnionExperts = 0
        var twoRowPairs = 0
        var flashBytes = 0
        for (tokenIndex, token) in trace.tokens.enumerated() {
            for experts in token.layers {
                flashBytes += Set(experts).count * (trace.expertBytes ?? 0)
            }
            guard tokenIndex > 0 else { continue }
            let previous = trace.tokens[tokenIndex - 1]
            for layer in 0 ..< layerCount {
                let priorExperts = Set(previous.layers[layer])
                previousTokenAccesses += token.layers[layer].count
                previousTokenHits += token.layers[layer].filter { priorExperts.contains($0) }.count
                twoRowUnionExperts += Set(previous.layers[layer]).union(token.layers[layer]).count
            }
            twoRowPairs += 1
        }
        let flashBytesPerToken = trace.expertBytes.map { _ in
            Double(flashBytes) / Double(trace.tokens.count)
        }
        let twoRowFlashBytesPerToken = trace.expertBytes.flatMap { (bytes: Int) -> Double? in
            guard twoRowPairs > 0 else { return nil }
            return Double(twoRowUnionExperts * bytes) / Double(twoRowPairs * 2)
        }
        return ExpertRouteTraceAnalysis(
            tokens: trace.tokens.count, layers: layerCount, lruHitRates: lruHitRates,
            previousTokenExpertReuseRate: previousTokenAccesses == 0
                ? nil : Double(previousTokenHits) / Double(previousTokenAccesses),
            mtpTwoRowUnionExpertsPerPair: twoRowPairs == 0
                ? nil : Double(twoRowUnionExperts) / Double(twoRowPairs),
            requiredFlashBytesPerToken: flashBytesPerToken,
            mtpTwoRowFlashBytesPerToken: twoRowFlashBytesPerToken)
    }

    private static func collect(
        runtime: ConcurrentTextRuntime, text: CheckpointTextProcessor, tokens: [Int],
        outputTokens: Int, cachePrefix: Bool
    ) async throws -> ExpertStreamingGenerationSample {
        let startedAt = ProcessInfo.processInfo.systemUptime
        let generation = try await runtime.generate(
            .init(
                tokens: tokens, maxTokens: outputTokens, temperature: 0,
                stopTokenIDs: text.stopTokenIDs,
                prefixTokenCount: cachePrefix ? tokens.count - 1 : 0,
                cacheIdentity: runtime.identity, speculative: false))
        var output: [Int] = []
        var tokenTimes: [Double] = []
        var reusedPrefixTokens = 0
        for try await event in generation.events {
            switch event {
            case .admitted(let reused): reusedPrefixTokens = reused
            case .token(let token):
                output.append(token)
                tokenTimes.append(ProcessInfo.processInfo.systemUptime)
            default: break
            }
        }
        return ExpertStreamingGenerationSample(
            tokens: output, tokenTimes: tokenTimes, reusedPrefixTokens: reusedPrefixTokens,
            timeToFirstTokenSeconds: tokenTimes.first.map { $0 - startedAt })
    }

    private static func percentile(_ values: [Double], at percentile: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let index = Int((Double(sorted.count - 1) * percentile).rounded())
        return sorted[index]
    }

    private static func emit(_ measurement: ExpertStreamingBenchmarkMeasurement) {
        guard let data = try? JSONEncoder().encode(measurement) else { return }
        print("FMLX_EXPERT_STREAMING_BENCHMARK " + String(decoding: data, as: UTF8.self))
    }
}
