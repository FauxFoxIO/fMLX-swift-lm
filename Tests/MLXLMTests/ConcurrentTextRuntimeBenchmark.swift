// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

private struct MixedWorkloadFixture: Decodable, Sendable {
    let modelDirectory: String
    let identity: PrefixCacheIdentity
    let longPrompt: [Int]
    let conversationPrompt: [Int]
    let shortPrompt: [Int]
    let prefixTokenCount: Int
    let memoryBudgetBytes: Int
    let prefixCacheBytes: Int
    let workingMemoryBytes: Int
    let trials: Int
    let longOutputTokens: Int?
    let conversationOutputTokens: Int?
    let persistentCacheDirectory: String?
    let mtpDirectory: String?
    let secondaryFixture: String?
    let batchDecode: Bool?
    let activeLimits: [Int]?
    let warmCases: [Bool]?
}

private struct WorkloadMeasurement: Codable, Sendable {
    let workload: String
    let submitted: Double
    var firstToken: Double?
    var finished: Double?
    var tokens = 0
    var reusedPrefixTokens = 0

    mutating func observe(_ event: ConcurrentTextRuntime.Event) {
        let now = ProcessInfo.processInfo.systemUptime
        switch event {
        case .admitted(let reused): reusedPrefixTokens = reused
        case .token:
            firstToken = firstToken ?? now
            tokens += 1
        case .finished: finished = now
        default: break
        }
    }
}

private struct WorkloadRound: Codable {
    let activeLimit: Int
    let backend: String
    let batchDecode: Bool
    let warm: Bool
    let duringDecode: Bool
    let trial: Int
    let measurements: [WorkloadMeasurement]
    let peakMLXBytes: Int
    let cachedMLXBytes: Int
}

private func measureRequest(
    runtime: ConcurrentTextRuntime, request: ConcurrentTextRuntime.Request, name: String
) async throws -> WorkloadMeasurement {
    var measurement = WorkloadMeasurement(
        workload: name, submitted: ProcessInfo.processInfo.systemUptime)
    let generation = try await runtime.generate(request)
    for try await event in generation.events { measurement.observe(event) }
    return measurement
}

private final class BenchmarkArrival: @unchecked Sendable {
    private let lock = NSLock()
    private var time: Double?
    func record() {
        lock.withLock { if time == nil { time = ProcessInfo.processInfo.systemUptime } }
    }
    var value: Double? { lock.withLock { time } }
}

/// Opt-in, local-files-only benchmark. See docs/concurrent-text-runtime.md.
final class ConcurrentTextRuntimeBenchmark: XCTestCase {
    func testTrainedModelParity() async throws {
        guard let path = ProcessInfo.processInfo.environment["FMLX_BENCHMARK_FIXTURE"] else {
            throw XCTSkip("Set FMLX_BENCHMARK_FIXTURE to a local benchmark fixture")
        }
        let fixture = try JSONDecoder().decode(
            MixedWorkloadFixture.self,
            from: Data(contentsOf: URL(filePath: path)))
        let model = try await NativeTextModelLoader.load(
            directory: URL(filePath: fixture.modelDirectory))
        let prompts = [fixture.shortPrompt, fixture.conversationPrompt]
        var expected: [[Int]] = []
        for prompt in prompts {
            let iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(prompt)), model: model,
                parameters: GenerateParameters(maxTokens: 32, temperature: 0))
            expected.append(Array(iterator))
        }
        let runtime = try ConcurrentTextRuntime(
            model: model, identity: fixture.identity,
            configuration: .init(
                memoryBudgetBytes: fixture.memoryBudgetBytes,
                prefixCacheBytes: fixture.prefixCacheBytes,
                workingMemoryBytes: fixture.workingMemoryBytes,
                prefillChunkSize: 32, streamBufferSize: 2048))
        var generations: [ConcurrentTextRuntime.Generation] = []
        for prompt in prompts {
            generations.append(try await runtime.generate(.init(tokens: prompt, maxTokens: 32)))
        }
        for (index, generation) in generations.enumerated() {
            var tokens: [Int] = []
            for try await event in generation.events {
                if case .token(let token) = event { tokens.append(token) }
            }
            XCTAssertEqual(tokens, expected[index])
            let data = try JSONSerialization.data(withJSONObject: [
                "prompt": prompts[index], "tokens": tokens,
            ])
            print("FMLX_PARITY " + String(decoding: data, as: UTF8.self))
        }
        let status = await runtime.status()
        XCTAssertGreaterThan(status.batchedForwardCount, 0)
        print("FMLX_PARITY_BATCHES \(status.batchedForwardCount)")
        await runtime.shutdown()
    }

    func testTrainedPersistentAndQuantizedPrefixes() async throws {
        guard let path = ProcessInfo.processInfo.environment["FMLX_BENCHMARK_FIXTURE"] else {
            throw XCTSkip("Set FMLX_BENCHMARK_FIXTURE to a local benchmark fixture")
        }
        let fixture = try JSONDecoder().decode(
            MixedWorkloadFixture.self,
            from: Data(contentsOf: URL(filePath: path)))
        guard let cacheDirectory = fixture.persistentCacheDirectory else {
            throw XCTSkip("Fixture needs an external persistentCacheDirectory")
        }
        func collect(_ generation: ConcurrentTextRuntime.Generation) async throws -> ([Int], Int) {
            var tokens: [Int] = []
            var reused = 0
            for try await event in generation.events {
                if case .token(let token) = event { tokens.append(token) }
                if case .admitted(let count) = event { reused = count }
            }
            return (tokens, reused)
        }
        for bits in [0, 4, 8] {
            let configuration = ConcurrentTextRuntime.Configuration(
                memoryBudgetBytes: fixture.memoryBudgetBytes,
                prefixCacheBytes: fixture.prefixCacheBytes,
                workingMemoryBytes: fixture.workingMemoryBytes, prefillChunkSize: 128,
                streamBufferSize: 2048, cacheQuantization: bits == 0 ? nil : .init(bits: bits),
                persistentCache: .init(
                    directory: URL(filePath: cacheDirectory), maximumBytes: 1_073_741_824))
            let runtime = try await ConcurrentTextRuntime(
                model: NativeTextModelLoader.load(directory: URL(filePath: fixture.modelDirectory)),
                identity: fixture.identity, configuration: configuration)
            try await runtime.clearCaches(persistent: true)
            let expected = try await collect(
                runtime.generate(.init(tokens: fixture.longPrompt, maxTokens: 8))
            ).0
            let request = ConcurrentTextRuntime.Request(
                tokens: fixture.longPrompt, maxTokens: 8,
                prefixTokenCount: fixture.prefixTokenCount, cacheIdentity: fixture.identity)
            let published = try await collect(runtime.generate(request))
            XCTAssertEqual(published.0, expected)
            let warm = try await collect(runtime.generate(request))
            XCTAssertEqual(warm.0, expected)
            XCTAssertEqual(warm.1, fixture.prefixTokenCount)
            let before = await runtime.status()
            XCTAssertEqual(before.persistentCacheFailures, 0)
            await runtime.shutdown()
            let restoredRuntime = try await ConcurrentTextRuntime(
                model: NativeTextModelLoader.load(directory: URL(filePath: fixture.modelDirectory)),
                identity: fixture.identity, configuration: configuration)
            let restored = try await collect(restoredRuntime.generate(request))
            XCTAssertEqual(restored.0, expected)
            XCTAssertEqual(restored.1, fixture.prefixTokenCount)
            let after = await restoredRuntime.status()
            XCTAssertEqual(after.persistentCacheFailures, 0)
            print("FMLX_PERSISTENT_PARITY bits=\(bits) reused=\(restored.1) tokens=\(restored.0)")
            await restoredRuntime.shutdown()
        }
    }

    func testTrainedMTPParity() async throws {
        guard let path = ProcessInfo.processInfo.environment["FMLX_BENCHMARK_FIXTURE"] else {
            throw XCTSkip("Set FMLX_BENCHMARK_FIXTURE to a local benchmark fixture")
        }
        let fixture = try JSONDecoder().decode(
            MixedWorkloadFixture.self,
            from: Data(contentsOf: URL(filePath: path)))
        guard let headPath = fixture.mtpDirectory,
            FileManager.default.fileExists(atPath: headPath + "/model.safetensors")
        else {
            throw XCTSkip("The trained MTP companion has not been downloaded")
        }
        let model = try await NativeTextModelLoader.load(
            directory: URL(filePath: fixture.modelDirectory))
        let drafter = try await NativeTextModelLoader.loadMTP(directory: URL(filePath: headPath))
        let prompts = [fixture.shortPrompt, fixture.conversationPrompt]
        var expected: [[Int]] = []
        for prompt in prompts {
            expected.append(
                Array(
                    try TokenIterator(
                        input: LMInput(tokens: MLXArray(prompt)),
                        model: model, parameters: GenerateParameters(maxTokens: 32, temperature: 0))
                ))
        }
        let runtime = try ConcurrentTextRuntime(
            model: model, identity: fixture.identity,
            configuration: .init(
                memoryBudgetBytes: fixture.memoryBudgetBytes,
                prefixCacheBytes: fixture.prefixCacheBytes,
                workingMemoryBytes: fixture.workingMemoryBytes,
                prefillChunkSize: 16, streamBufferSize: 2048), drafter: drafter)
        var requests: [ConcurrentTextRuntime.Generation] = []
        for prompt in prompts {
            requests.append(try await runtime.generate(.init(tokens: prompt, maxTokens: 32)))
        }
        for (index, generation) in requests.enumerated() {
            var tokens: [Int] = []
            var rounds = 0
            for try await event in generation.events {
                switch event {
                case .token(let token): tokens.append(token)
                case .speculation(let telemetry):
                    rounds += telemetry.roundCount
                    print(
                        "FMLX_MTP rounds=\(telemetry.roundCount) proposed=\(telemetry.draftTokenCount) accepted=\(telemetry.acceptedDraftTokenCount)"
                    )
                case .fallback(let reason): XCTFail(reason)
                default: break
                }
            }
            XCTAssertEqual(tokens, expected[index])
            XCTAssertGreaterThan(rounds, 0)
        }
        let cancelled = try await runtime.generate(.init(tokens: fixture.longPrompt, maxTokens: 32))
        var events = cancelled.events.makeAsyncIterator()
        while let event = try await events.next() { if case .prefill = event { break } }
        await runtime.cancel(cancelled.id)
        let peer = try await runtime.generate(.init(tokens: prompts[0], maxTokens: 32))
        var tokens: [Int] = []
        for try await event in peer.events {
            if case .token(let token) = event { tokens.append(token) }
        }
        XCTAssertEqual(tokens, expected[0])
        await runtime.shutdown()
    }

    func testTrainedMTPPerformance() async throws {
        guard let path = ProcessInfo.processInfo.environment["FMLX_BENCHMARK_FIXTURE"] else {
            throw XCTSkip("Set FMLX_BENCHMARK_FIXTURE to a local benchmark fixture")
        }
        let fixture = try JSONDecoder().decode(
            MixedWorkloadFixture.self,
            from: Data(contentsOf: URL(filePath: path)))
        guard let headPath = fixture.mtpDirectory else {
            throw XCTSkip("Fixture needs mtpDirectory")
        }
        let runtime = try await ConcurrentTextRuntime(
            model: NativeTextModelLoader.load(directory: URL(filePath: fixture.modelDirectory)),
            identity: fixture.identity,
            configuration: .init(
                memoryBudgetBytes: fixture.memoryBudgetBytes,
                prefixCacheBytes: fixture.prefixCacheBytes,
                workingMemoryBytes: fixture.workingMemoryBytes,
                prefillChunkSize: 128, streamBufferSize: 2048),
            drafter: NativeTextModelLoader.loadMTP(directory: URL(filePath: headPath)))
        do {
            for trial in 0 ..< 11 {
                // Alternate order to reduce drift between ordinary and speculative samples.
                for speculative in trial.isMultiple(of: 2) ? [false, true] : [true, false] {
                    Memory.peakMemory = 0
                    var measurement = WorkloadMeasurement(
                        workload: "conversation",
                        submitted: ProcessInfo.processInfo.systemUptime)
                    let generation = try await runtime.generate(
                        .init(
                            tokens: fixture.conversationPrompt,
                            maxTokens: 64, speculative: speculative))
                    var proposed = 0
                    var accepted = 0
                    for try await event in generation.events {
                        measurement.observe(event)
                        if case .speculation(let telemetry) = event {
                            proposed = telemetry.draftTokenCount
                            accepted = telemetry.acceptedDraftTokenCount
                        }
                        if case .fallback(let reason) = event { XCTFail(reason) }
                    }
                    XCTAssertEqual(measurement.tokens, 64)
                    if speculative { XCTAssertGreaterThan(proposed, 0) }
                    let row: [String: Any] = [
                        "trial": trial, "speculative": speculative,
                        "ttftMs": (try XCTUnwrap(measurement.firstToken) - measurement.submitted)
                            * 1000,
                        "tokensPerSecond": 64
                            / (try XCTUnwrap(measurement.finished) - measurement.submitted),
                        "proposed": proposed, "accepted": accepted,
                        "peakMLXBytes": Memory.peakMemory,
                    ]
                    print(
                        "FMLX_MTP_PERFORMANCE "
                            + String(
                                decoding: try JSONSerialization.data(withJSONObject: row),
                                as: UTF8.self))
                }
            }
            await runtime.shutdown()
        } catch {
            await runtime.shutdown()
            throw error
        }
    }

    func testTrainedMultiModelQualification() async throws {
        guard let path = ProcessInfo.processInfo.environment["FMLX_BENCHMARK_FIXTURE"] else {
            throw XCTSkip("Set FMLX_BENCHMARK_FIXTURE to a local benchmark fixture")
        }
        let fixture = try JSONDecoder().decode(
            MixedWorkloadFixture.self, from: Data(contentsOf: URL(filePath: path)))
        guard let secondaryPath = fixture.secondaryFixture else {
            throw XCTSkip("Fixture needs a secondaryFixture for a second trained model")
        }
        let secondary = try JSONDecoder().decode(
            MixedWorkloadFixture.self, from: Data(contentsOf: URL(filePath: secondaryPath)))
        func reference(_ fixture: MixedWorkloadFixture, prompt: [Int], count: Int) async throws
            -> [Int]
        {
            let model = try await NativeTextModelLoader.load(
                directory: URL(filePath: fixture.modelDirectory))
            return Array(
                try TokenIterator(
                    input: LMInput(tokens: MLXArray(prompt)), model: model,
                    parameters: GenerateParameters(maxTokens: count, temperature: 0)))
        }
        let expectedLarge = try await reference(fixture, prompt: fixture.longPrompt, count: 32)
        let expectedSmall = try await reference(secondary, prompt: secondary.shortPrompt, count: 8)
        let service = try NativeInferenceRuntime(
            memoryBudgetBytes: 34_359_738_368,
            interactiveHeadroomBytes: 2_147_483_648)
        for (id, current, reservation) in [
            ("large", fixture, 25_769_803_776),
            ("small", secondary, 2_147_483_648),
        ] {
            try await service.load(id: id, estimatedResidentBytes: reservation) {
                try await ConcurrentTextRuntime(
                    model: NativeTextModelLoader.load(
                        directory: URL(filePath: current.modelDirectory)),
                    identity: current.identity,
                    configuration: .init(
                        memoryBudgetBytes: current.memoryBudgetBytes,
                        prefixCacheBytes: current.prefixCacheBytes,
                        workingMemoryBytes: current.workingMemoryBytes,
                        prefillChunkSize: 32, streamBufferSize: 2048))
            }
        }
        do {
            for duringDecode in [false, true] {
                for cancel in [false, true] {
                    Memory.peakMemory = 0
                    var large = WorkloadMeasurement(
                        workload: "background",
                        submitted: ProcessInfo.processInfo.systemUptime)
                    let generation = try await service.generate(
                        modelID: "large",
                        request: .init(
                            tokens: fixture.longPrompt, maxTokens: 32, priority: .background))
                    var peer: Task<([Int], WorkloadMeasurement), Error>?
                    var largeTokens: [Int] = []
                    for try await event in generation.events {
                        large.observe(event)
                        if case .token(let token) = event { largeTokens.append(token) }
                        let trigger: Bool
                        switch event {
                        case .token: trigger = duringDecode
                        case .prefill(let processed, let total):
                            trigger = !duringDecode && processed < total
                        default: trigger = false
                        }
                        if trigger && peer == nil {
                            peer = Task {
                                var measurement = WorkloadMeasurement(
                                    workload: "short",
                                    submitted: ProcessInfo.processInfo.systemUptime)
                                let small = try await service.generate(
                                    modelID: "small",
                                    request: .init(tokens: secondary.shortPrompt, maxTokens: 8))
                                var tokens: [Int] = []
                                for try await event in small.events {
                                    measurement.observe(event)
                                    if case .token(let token) = event { tokens.append(token) }
                                }
                                return (tokens, measurement)
                            }
                            if cancel {
                                await service.cancel(modelID: "large", requestID: generation.id)
                            }
                        }
                    }
                    let (tokens, small) = try await XCTUnwrap(peer).value
                    XCTAssertEqual(tokens, expectedSmall)
                    if !cancel {
                        XCTAssertEqual(largeTokens, expectedLarge)
                        XCTAssertLessThan(
                            try XCTUnwrap(small.firstToken), try XCTUnwrap(large.finished))
                    } else {
                        XCTAssertLessThan(largeTokens.count, 32)
                    }
                    let round = WorkloadRound(
                        activeLimit: 2, backend: cancel ? "multi-model-cancel" : "multi-model",
                        batchDecode: false, warm: false, duringDecode: duringDecode, trial: 1,
                        measurements: [large, small], peakMLXBytes: Memory.peakMemory,
                        cachedMLXBytes: Memory.cacheMemory)
                    print(
                        "FMLX_BENCHMARK "
                            + String(decoding: try JSONEncoder().encode(round), as: UTF8.self))
                }
            }
            let unloading = try await service.generate(
                modelID: "large",
                request: .init(tokens: fixture.longPrompt, maxTokens: 32, priority: .background))
            var events = unloading.events.makeAsyncIterator()
            while let event = try await events.next() { if case .prefill = event { break } }
            await service.unload(modelID: "large")
            let ids = await service.modelIDs()
            XCTAssertEqual(ids, ["small"])
            let survivor = try await service.generate(
                modelID: "small",
                request: .init(tokens: secondary.shortPrompt, maxTokens: 8))
            var tokens: [Int] = []
            for try await event in survivor.events {
                if case .token(let token) = event { tokens.append(token) }
            }
            XCTAssertEqual(tokens, expectedSmall)
            await service.unload(modelID: "small")
            let status = await service.resources.status()
            XCTAssertEqual(status.residentBytes, 0)
            XCTAssertEqual(status.requestBytes, 0)
        } catch {
            await service.unload(modelID: "large")
            await service.unload(modelID: "small")
            throw error
        }
    }

    func testExistingIteratorSerializedBaseline() async throws {
        guard let path = ProcessInfo.processInfo.environment["FMLX_BENCHMARK_FIXTURE"] else {
            throw XCTSkip("Set FMLX_BENCHMARK_FIXTURE to a local benchmark fixture")
        }
        let fixture = try JSONDecoder().decode(
            MixedWorkloadFixture.self,
            from: Data(contentsOf: URL(filePath: path)))
        let model = try await NativeTextModelLoader.load(
            directory: URL(filePath: fixture.modelDirectory))
        for duringDecode in [false, true] {
            for trial in 0 ..< fixture.trials {
                Memory.peakMemory = 0
                let arrival = BenchmarkArrival()
                func run(_ tokens: [Int], limit: Int, name: String, submitted: Double) throws
                    -> WorkloadMeasurement
                {
                    var measurement = WorkloadMeasurement(workload: name, submitted: submitted)
                    let prefill = PrefillParameters(
                        stepSize: 128,
                        progress: { processed, total in
                            if name == "background", !duringDecode, processed > 0, processed < total
                            {
                                arrival.record()
                            }
                        })
                    let iterator = try TokenIterator(
                        input: LMInput(tokens: MLXArray(tokens)), model: model,
                        parameters: GenerateParameters(
                            maxTokens: limit, temperature: 0, prefill: prefill))
                    for _ in iterator {
                        measurement.firstToken =
                            measurement.firstToken ?? ProcessInfo.processInfo.systemUptime
                        measurement.tokens += 1
                        if name == "background", duringDecode { arrival.record() }
                    }
                    measurement.finished = ProcessInfo.processInfo.systemUptime
                    return measurement
                }
                let background = try run(
                    fixture.longPrompt, limit: fixture.longOutputTokens ?? 512,
                    name: "background", submitted: ProcessInfo.processInfo.systemUptime)
                let submitted = try XCTUnwrap(arrival.value)
                let short = try run(
                    fixture.shortPrompt, limit: 8, name: "short", submitted: submitted)
                let conversation = try run(
                    fixture.conversationPrompt,
                    limit: fixture.conversationOutputTokens ?? 128, name: "conversation",
                    submitted: submitted)
                let round = WorkloadRound(
                    activeLimit: 1, backend: "TokenIterator", batchDecode: false,
                    warm: false, duringDecode: duringDecode, trial: trial,
                    measurements: [background, short, conversation],
                    peakMLXBytes: Memory.peakMemory,
                    cachedMLXBytes: Memory.cacheMemory)
                let json = try JSONEncoder().encode(round)
                print("FMLX_BENCHMARK " + String(decoding: json, as: UTF8.self))
            }
        }
    }

    func testMixedWorkload() async throws {
        guard let path = ProcessInfo.processInfo.environment["FMLX_BENCHMARK_FIXTURE"] else {
            throw XCTSkip("Set FMLX_BENCHMARK_FIXTURE to a local benchmark fixture")
        }
        let fixture = try JSONDecoder().decode(
            MixedWorkloadFixture.self, from: Data(contentsOf: URL(filePath: path)))
        guard fixture.trials >= 1, fixture.trials <= 1000,
            fixture.longPrompt.count > 128, fixture.prefixTokenCount > 0,
            fixture.prefixTokenCount < fixture.longPrompt.count
        else { throw ConcurrentTextRuntimeError.invalidConfiguration }
        let directory = URL(filePath: fixture.modelDirectory)
        for activeLimit in fixture.activeLimits ?? [1, 4] {
            let model = try await NativeTextModelLoader.load(directory: directory)
            let runtime = try ConcurrentTextRuntime(
                model: model, identity: fixture.identity,
                configuration: .init(
                    memoryBudgetBytes: fixture.memoryBudgetBytes,
                    prefixCacheBytes: fixture.prefixCacheBytes,
                    workingMemoryBytes: fixture.workingMemoryBytes,
                    maxActiveRequests: activeLimit, maxPromptTokens: 32_768,
                    maxOutputTokens: 512, prefillChunkSize: 128, streamBufferSize: 2048,
                    batchDecode: fixture.batchDecode ?? true))
            do {
                for duringDecode in [false, true] {
                    for warm in fixture.warmCases ?? [false, true] {
                        for trial in 0 ..< fixture.trials {
                            await runtime.clearPrefixCache()
                            let long = ConcurrentTextRuntime.Request(
                                tokens: fixture.longPrompt,
                                maxTokens: fixture.longOutputTokens ?? 512, priority: .background,
                                prefixTokenCount: fixture.prefixTokenCount,
                                cacheIdentity: fixture.identity)
                            if warm {
                                _ = try await measureRequest(
                                    runtime: runtime,
                                    request: .init(
                                        tokens: fixture.longPrompt, maxTokens: 1,
                                        prefixTokenCount: fixture.prefixTokenCount,
                                        cacheIdentity: fixture.identity), name: "warmup")
                            }
                            Memory.peakMemory = 0
                            var background = WorkloadMeasurement(
                                workload: "background",
                                submitted: ProcessInfo.processInfo.systemUptime)
                            let generation = try await runtime.generate(long)
                            var peers: [Task<WorkloadMeasurement, Error>] = []
                            for try await event in generation.events {
                                background.observe(event)
                                let trigger: Bool
                                switch event {
                                case .token: trigger = duringDecode
                                case .prefill(let processed, let total):
                                    trigger = !duringDecode && processed < total
                                default: trigger = false
                                }
                                if trigger && peers.isEmpty {
                                    peers.append(
                                        Task {
                                            try await measureRequest(
                                                runtime: runtime,
                                                request: .init(
                                                    tokens: fixture.shortPrompt, maxTokens: 8),
                                                name: "short")
                                        })
                                    peers.append(
                                        Task {
                                            try await measureRequest(
                                                runtime: runtime,
                                                request: .init(
                                                    tokens: fixture.conversationPrompt,
                                                    maxTokens: fixture.conversationOutputTokens
                                                        ?? 128), name: "conversation")
                                        })
                                }
                            }
                            XCTAssertEqual(
                                peers.count, 2,
                                "Fixture must leave a prefill chunk after prefix restore")
                            var measurements = [background]
                            for peer in peers { measurements.append(try await peer.value) }
                            let round = WorkloadRound(
                                activeLimit: activeLimit, backend: "scheduler",
                                batchDecode: fixture.batchDecode ?? true,
                                warm: warm, duringDecode: duringDecode,
                                trial: trial, measurements: measurements,
                                peakMLXBytes: Memory.peakMemory, cachedMLXBytes: Memory.cacheMemory)
                            let json = try JSONEncoder().encode(round)
                            print("FMLX_BENCHMARK \(String(decoding: json, as: UTF8.self))")
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
}
