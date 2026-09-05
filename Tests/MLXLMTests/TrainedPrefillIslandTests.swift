#if os(macOS)
import CoreML
import Darwin
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Metal
import XCTest

@testable import MLXLLM

private struct ModelIslandFixture: Decodable {
    struct Prompt: Decodable {
        let name: String
        let tokens: [Int]
        let outputTokens: Int?
        let resultKey: String?
    }
    let modelDirectory: String
    let mtpDirectory: String
    let islandDirectory: String
    let prompts: [Prompt]
    let outputTokens: Int
    let trials: Int
    let bridgeVariants: [String]?
    let qualityEvaluation: Bool?
    let stopTokenIDs: [Int]?
    let evaluationID: String?
    let resultDirectory: String?
    let cpuGateIslandDirectory: String?
    let cpuGateRows: Int?
    let memoryProbePauseSeconds: Int?
}

private struct IslandGenerationLimit {
    let maxTokens: Int
    let stopTokenIDs: Set<Int>

    func shouldContinue(_ tokens: [Int]) -> Bool {
        tokens.count < maxTokens && !(tokens.last.map(stopTokenIDs.contains) ?? false)
    }

    func reason(_ tokens: [Int]) -> String {
        (tokens.last.map(stopTokenIDs.contains) ?? false) ? "stop" : "length"
    }
}

private struct GateSplitFixture: Decodable {
    struct Export: Decodable {
        let name: String
        let rows: Int
    }
    let sourceDirectory: String
    let tokens: Int
    let exports: [Export]
}

// The owner keeps the backing alive until prediction completes; Core ML only reads it.
private struct BorrowedIslandInput {
    let provider: MLDictionaryFeatureProvider
    let backing: MLXArray
    let buffer: any MTLBuffer
}

// Inputs and results are immutable. No MLX state crosses the Core ML tasks.
private struct IslandFeatures: @unchecked Sendable {
    let value: any MLFeatureProvider
    let completedSeconds: Double
}

// Each branch owns a distinct model, invoked only once per joined trial.
private final class IslandBranch: @unchecked Sendable {
    let model: MLModel

    init(url: URL, units: MLComputeUnits) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        model = try MLModel(contentsOf: url, configuration: configuration)
    }

    @available(macOS 14.0, *)
    func predict(_ input: IslandFeatures) async throws -> IslandFeatures {
        let result = try await model.prediction(from: input.value)
        return IslandFeatures(value: result, completedSeconds: ProcessInfo.processInfo.systemUptime)
    }
}

/// Local experiment: trained projection weights, synthetic activations, explicit precision changes.
final class TrainedPrefillIslandTests: XCTestCase {
    private var reportedLayouts = Set<[Int]>()
    private func emit(_ values: [String: Any]) throws {
        let data = try JSONSerialization.data(withJSONObject: values, options: [.sortedKeys])
        print("FMLX_PREFILL_ISLAND " + String(decoding: data, as: UTF8.self))
    }

    private func now() -> Double { ProcessInfo.processInfo.systemUptime }

    private func footprint() throws -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard status == KERN_SUCCESS else {
            throw NSError(domain: NSMachErrorDomain, code: Int(status))
        }
        return info.phys_footprint
    }

    func testLocalTrainedModelPrefill() async throws {
        guard #available(macOS 14.0, *),
            let path = ProcessInfo.processInfo.environment["FMLX_TRAINED_MODEL_FIXTURE"]
        else { throw XCTSkip("Set FMLX_TRAINED_MODEL_FIXTURE on macOS 14+") }
        let fixture = try JSONDecoder().decode(
            ModelIslandFixture.self, from: Data(contentsOf: URL(filePath: path)))
        let quality = fixture.qualityEvaluation ?? false
        XCTAssertGreaterThan(fixture.trials, quality ? 0 : 1)
        XCTAssertGreaterThan(fixture.outputTokens, 1)
        let bridges = fixture.bridgeVariants ?? ["ane", "ane-copy-once"]
        guard
            quality
                ? bridges == ["ane-copy-once"]
                : ["ane", "ane-copy-once"].contains(bridges.first ?? ""),
            Set(bridges).count == bridges.count,
            bridges.allSatisfy({
                ["ane", "ane-borrowed", "ane-copy-once", "ane-cpu-gate"].contains($0)
            })
        else { return XCTFail("Invalid bridge variants") }
        let stopIDs = Set(fixture.stopTokenIDs ?? [])
        let resultDirectory = fixture.resultDirectory.map { URL(filePath: $0).standardizedFileURL }
        if quality {
            guard fixture.trials == 1, stopIDs == [248044, 248046],
                let evaluationID = fixture.evaluationID, !evaluationID.isEmpty,
                let resultDirectory,
                resultDirectory.path.hasPrefix(
                    "/Users/ethan/Documents/fMLX-performance-cdad/quality/")
            else {
                return XCTFail("Quality mode requires explicit EOS, identity and external output")
            }
            try FileManager.default.createDirectory(
                at: resultDirectory, withIntermediateDirectories: true)
        }
        defer { Stream().synchronize() }
        let loaded = try await NativeTextModelLoader.load(
            directory: URL(filePath: fixture.modelDirectory))
        let wrapper = try XCTUnwrap(loaded as? Qwen35Model)
        let model = wrapper.languageModel
        wrapper.train(false)
        eval(wrapper)
        let head = try await NativeTextModelLoader.loadMTP(
            directory: URL(filePath: fixture.mtpDirectory))
        head.train(false)
        eval(head)
        let layers = try model.loraLayers.map { try XCTUnwrap($0 as? Qwen35DecoderLayer) }
        XCTAssertEqual(layers.count, 64)
        let baseFootprint = try footprint()
        if let pause = fixture.memoryProbePauseSeconds {
            guard !quality, (1 ... 30).contains(pause)
            else { return XCTFail("Memory probe pause must be between1 and30 seconds") }
            try emit([
                "variant": "memory-probe", "phase": "native-resident",
                "footprintBytes": baseFootprint, "pauseSeconds": pause,
            ])
            try await Task.sleep(for: .seconds(pause))
        }
        let loadStart = now()
        var islands: [IslandBranch] = []
        for index in layers.indices {
            islands.append(
                try IslandBranch(
                    url: URL(filePath: fixture.islandDirectory)
                        .appendingPathComponent("up-\(index).mlmodelc"),
                    units: .cpuAndNeuralEngine))
        }
        var cpuGateIslands: [IslandBranch] = []
        var gatePrefixes: [[MLXArray]] = []
        let cpuGateRows = fixture.cpuGateRows ?? 0
        if bridges.contains("ane-cpu-gate") {
            guard !quality, let directory = fixture.cpuGateIslandDirectory,
                [1024, 2048].contains(cpuGateRows)
            else { return XCTFail("CPU gate experiment needs matching row-specific sidecars") }
            let gpuGateRows = 17408 - cpuGateRows
            for (index, layer) in layers.enumerated() {
                let mlp = try XCTUnwrap(layer.mlp as? Qwen3NextMLP)
                let gate = try XCTUnwrap(mlp.gateProj as? QuantizedLinear)
                guard gate.bits == 4, gate.groupSize == 64, gate.mode == .affine,
                    gate.bias == nil, gate.weight.shape == [17408, 640]
                else { return XCTFail("CPU gate experiment requires the original affine4 gate") }
                gatePrefixes.append([
                    gate.weight[..<gpuGateRows], gate.scales[..<gpuGateRows],
                    try XCTUnwrap(gate.biases)[..<gpuGateRows],
                ])
                cpuGateIslands.append(
                    try IslandBranch(
                        url: URL(filePath: directory).appendingPathComponent(
                            "gate-\(index).mlmodelc"),
                        units: .cpuOnly))
            }
            eval(gatePrefixes.flatMap { $0 })
        }
        let loadedFootprint = try footprint()
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        try emit([
            "variant": "model-load", "modelCount": islands.count,
            "cpuGateModelCount": cpuGateIslands.count,
            "cpuGateRows": cpuGateRows,
            "seconds": now() - loadStart, "baseFootprintBytes": baseFootprint,
            "loadedFootprintBytes": loadedFootprint,
        ])
        if let pause = fixture.memoryProbePauseSeconds {
            try emit([
                "variant": "memory-probe", "phase": "ane-resident",
                "footprintBytes": loadedFootprint, "pauseSeconds": pause,
            ])
            try await Task.sleep(for: .seconds(pause))
        }

        for (promptIndex, prompt) in fixture.prompts.enumerated() {
            XCTAssertFalse(prompt.tokens.isEmpty)
            let limit = IslandGenerationLimit(
                maxTokens: prompt.outputTokens ?? fixture.outputTokens, stopTokenIDs: stopIDs)
            XCTAssertGreaterThan(limit.maxTokens, 0)
            var referenceState: [any KVCache] = []
            var aneState: [any KVCache] = []
            var referenceTokens: [Int] = []
            var aneTokens: [Int] = []
            var cpuGateTokens: [Int] = []
            for trial in 0 ..< fixture.trials {
                let variants: [String] =
                    quality
                    ? (promptIndex % 2 == 0
                        ? ["native", "ane-copy-once"] : ["ane-copy-once", "native"])
                    : trial == 0
                        ? ["native", "manual-gpu"] + bridges
                        : trial % 2 == 0
                            ? ["native"] + bridges
                            : Array((["native"] + bridges).reversed())
                for variant in variants {
                    try Task.checkCancellation()
                    var resultURL: URL?
                    if quality {
                        let key = try XCTUnwrap(prompt.resultKey)
                        guard key.count == 64,
                            key.allSatisfy({ "0123456789abcdef".contains($0) })
                        else { return XCTFail("Invalid quality result key") }
                        let url = try XCTUnwrap(resultDirectory)
                            .appendingPathComponent("\(key)-\(variant).json")
                        resultURL = url
                        if FileManager.default.fileExists(atPath: url.path) {
                            var saved = try XCTUnwrap(
                                JSONSerialization.jsonObject(with: Data(contentsOf: url))
                                    as? [String: Any])
                            guard saved["evaluationID"] as? String == fixture.evaluationID,
                                saved["prompt"] as? String == prompt.name,
                                saved["variant"] as? String == "model-" + variant,
                                saved["outputLimit"] as? Int == limit.maxTokens,
                                saved["complete"] as? Bool == true,
                                saved["tokens"] is [Int]
                            else {
                                return XCTFail("Existing quality result does not match fixture")
                            }
                            saved["reusedResult"] = true
                            try emit(saved)
                            continue
                        }
                    }
                    let cache = try model.newCache(parameters: nil)
                    Memory.peakMemory = 0
                    let start = now()
                    var firstLogits: MLXArray?
                    var phases: [String: Double] = [:]
                    var aneProjectionCount = 0
                    var cpuGateProjectionCount = 0
                    var observedFootprint = loadedFootprint
                    for position in stride(from: 0, to: prompt.tokens.count, by: 128) {
                        try Task.checkCancellation()
                        let end = min(position + 128, prompt.tokens.count)
                        let input = MLXArray(Array(prompt.tokens[position ..< end]))
                            .expandedDimensions(axis: 0)
                        let logits: MLXArray
                        if variant == "native" || input.dim(1) != 128 {
                            logits = model(input, cache: cache)
                        } else {
                            var hidden = model.model.embedTokens(input)
                            let attentionMask = createAttentionMask(
                                h: hidden, cache: cache[model.model.faIdx])
                            let ssmMask = createSSMMask(
                                h: hidden, cache: cache[model.model.ssmIdx] as? MambaCache)
                            for (index, layer) in layers.enumerated() {
                                let normalized = layer.inputLayerNorm(hidden)
                                let attention: MLXArray
                                if let gdn = layer.linearAttn {
                                    attention = gdn(
                                        normalized, mask: ssmMask,
                                        cache: cache[index] as? MambaCache)
                                } else {
                                    attention = try XCTUnwrap(layer.selfAttn)(
                                        normalized, mask: attentionMask, cache: cache[index])
                                }
                                let residual = hidden + attention
                                let mlp = try XCTUnwrap(layer.mlp as? Qwen3NextMLP)
                                let mlpInput = layer.postAttentionLayerNorm(residual)
                                if variant == "manual-gpu" {
                                    hidden = residual + mlp(mlpInput)
                                } else {
                                    let splitGate = variant == "ane-cpu-gate"
                                    let copyOnce = variant == "ane-copy-once" || splitGate
                                    let a = now()
                                    let borrowed =
                                        variant == "ane-borrowed"
                                        ? try borrowedProvider(mlpInput, device: device) : nil
                                    let features =
                                        try borrowed?.provider
                                        ?? provider(
                                            mlpInput, gpuLayout: true,
                                            singleCopy: copyOnce)
                                    let b = now()
                                    let gateGPU: MLXArray
                                    if splitGate {
                                        let weights = gatePrefixes[index]
                                        gateGPU = quantizedMM(
                                            mlpInput, weights[0], scales: weights[1],
                                            biases: weights[2],
                                            transpose: true, groupSize: 64, bits: 4)
                                    } else {
                                        gateGPU = mlp.gateProj(mlpInput)
                                    }
                                    asyncEval(gateGPU)
                                    let prediction: any MLFeatureProvider
                                    let cpuPrediction: (any MLFeatureProvider)?
                                    if splitGate {
                                        let inputFeatures = IslandFeatures(
                                            value: features, completedSeconds: now())
                                        let ane = islands[index]
                                        let cpu = cpuGateIslands[index]
                                        async let upResult = ane.predict(inputFeatures)
                                        async let gateResult = cpu.predict(inputFeatures)
                                        let results = try await (upResult, gateResult)
                                        prediction = results.0.value
                                        cpuPrediction = results.1.value
                                        phases["anePredictionCompletionMs", default: 0] +=
                                            (results.0.completedSeconds - b) * 1000
                                        phases["cpuPredictionCompletionMs", default: 0] +=
                                            (results.1.completedSeconds - b) * 1000
                                        phases["cpuCompletionAfterANEMs", default: 0] +=
                                            max(
                                                0,
                                                results.1.completedSeconds
                                                    - results.0.completedSeconds) * 1000
                                        cpuGateProjectionCount += 1
                                    } else {
                                        prediction = try await islands[index].model.prediction(
                                            from: features)
                                        cpuPrediction = nil
                                    }
                                    aneProjectionCount += 1
                                    let c = now()
                                    let up = try output(
                                        prediction, length: 128, singleCopy: borrowed != nil,
                                        stridedCopy: copyOnce)
                                    let gate: MLXArray
                                    if let cpuPrediction {
                                        gate = concatenated(
                                            [
                                                gateGPU,
                                                try output(
                                                    cpuPrediction, length: 128,
                                                    channels: cpuGateRows,
                                                    stridedCopy: true),
                                            ], axis: -1)
                                    } else {
                                        gate = gateGPU
                                    }
                                    withExtendedLifetime(borrowed) {}
                                    hidden = residual + mlp.downProj(silu(gate) * up)
                                    let d = now()
                                    phases["inputBridgeAndUpstreamMs", default: 0] += (b - a) * 1000
                                    phases["gateDispatchAndPredictionWaitMs", default: 0] +=
                                        (c - b) * 1000
                                    phases["outputBridgeAndGraphMs", default: 0] += (d - c) * 1000
                                }
                            }
                            hidden = model.model.norm(hidden)
                            logits =
                                model.lmHead?(hidden) ?? model.model.embedTokens.asLinear(hidden)
                        }
                        if end == prompt.tokens.count {
                            firstLogits = logits[0..., -1, 0...]
                            eval(try XCTUnwrap(firstLogits))
                        }
                        eval(cache.flatMap { $0.innerState() })
                    }
                    let firstToken = argMax(try XCTUnwrap(firstLogits), axis: -1).item(Int.self)
                    let ttft = now() - start
                    observedFootprint = max(observedFootprint, try footprint())
                    if !quality && trial == 0 && variant == "native" {
                        referenceState = cache.map { $0.copy() }
                    } else if !quality && trial == 0 && variant == "manual-gpu" {
                        for (actual, expected) in zip(cache, referenceState) {
                            XCTAssertEqual(actual.metaState, expected.metaState)
                            XCTAssertEqual(actual.state.count, expected.state.count)
                            for (a, b) in zip(actual.state, expected.state) {
                                XCTAssertTrue(arrayEqual(a, b).item(Bool.self))
                            }
                        }
                        referenceState.removeAll()
                    } else if !quality && trial == 0 && variant == bridges.first {
                        aneState = cache.map { $0.copy() }
                    } else if !quality && trial == 0 && variant.hasPrefix("ane-")
                        && variant != "ane-cpu-gate"
                    {
                        for (actual, expected) in zip(cache, aneState) {
                            XCTAssertEqual(actual.metaState, expected.metaState)
                            XCTAssertEqual(actual.state.count, expected.state.count)
                            for (a, b) in zip(actual.state, expected.state) {
                                XCTAssertTrue(arrayEqual(a, b).item(Bool.self))
                            }
                        }
                    }
                    var tokens = [firstToken]
                    while limit.shouldContinue(tokens) {
                        try Task.checkCancellation()
                        let logits = model(MLXArray([tokens.last!]).reshaped(1, 1), cache: cache)
                        tokens.append(argMax(logits[0..., -1, 0...], axis: -1).item(Int.self))
                    }
                    Stream().synchronize()
                    let elapsed = now() - start
                    observedFootprint = max(observedFootprint, try footprint())
                    if !quality {
                        if trial == 0 && variant == "native" { referenceTokens = tokens }
                        if trial == 0 && variant == bridges.first { aneTokens = tokens }
                        if trial == 0 && variant == "ane-cpu-gate" { cpuGateTokens = tokens }
                        XCTAssertEqual(
                            tokens,
                            variant == "ane-cpu-gate"
                                ? cpuGateTokens
                                : variant.hasPrefix("ane") ? aneTokens : referenceTokens)
                    }
                    var row: [String: Any] = [
                        "variant": "model-" + variant, "prompt": prompt.name,
                        "promptTokens": prompt.tokens.count, "trial": trial,
                        "ttftSeconds": ttft, "finishedSeconds": elapsed,
                        "tokens": tokens, "tokensMatchNativeWarmup": tokens == referenceTokens,
                        "phases": phases, "peakMLXBytes": Memory.peakMemory,
                        "maxObservedFootprintBytes": observedFootprint,
                        "thermalState": ProcessInfo.processInfo.thermalState.rawValue,
                        "outputLimit": limit.maxTokens, "stopReason": limit.reason(tokens),
                        "aneProjectionCount": aneProjectionCount,
                    ]
                    if !quality {
                        let expected = (prompt.tokens.count / 128) * layers.count
                        XCTAssertEqual(
                            cpuGateProjectionCount, variant == "ane-cpu-gate" ? expected : 0)
                        XCTAssertEqual(aneProjectionCount, variant.hasPrefix("ane") ? expected : 0)
                        row["cpuGateProjectionCount"] = cpuGateProjectionCount
                        row["tokensMatchTwoWayWarmup"] = tokens == aneTokens
                    }
                    if quality {
                        let expectedCalls =
                            variant == "native" ? 0 : (prompt.tokens.count / 128) * layers.count
                        guard aneProjectionCount == expectedCalls else {
                            return XCTFail(
                                "Quality result did not execute the expected projection route")
                        }
                        row.removeValue(forKey: "tokensMatchNativeWarmup")
                        row["evaluationID"] = try XCTUnwrap(fixture.evaluationID)
                        row["complete"] = true
                        let data = try JSONSerialization.data(
                            withJSONObject: row, options: [.sortedKeys])
                        try data.write(to: XCTUnwrap(resultURL), options: .atomic)
                    }
                    try emit(row)
                }
            }
        }
        withExtendedLifetime((wrapper, head, islands, cpuGateIslands, gatePrefixes)) {}
    }

    func testQualityGenerationStopsAtEOSAndLimit() {
        let eos = IslandGenerationLimit(maxTokens: 4, stopTokenIDs: [248044, 248046])
        XCTAssertTrue(eos.shouldContinue([1]))
        XCTAssertFalse(eos.shouldContinue([248044]))
        XCTAssertFalse(eos.shouldContinue([1, 248046]))
        XCTAssertEqual(eos.reason([1, 248046]), "stop")
        XCTAssertFalse(eos.shouldContinue([1, 2, 3, 4]))
        XCTAssertEqual(eos.reason([1, 2, 3, 4]), "length")
        let fixed = IslandGenerationLimit(maxTokens: 3, stopTokenIDs: [])
        XCTAssertTrue(fixed.shouldContinue([1, 248046]))
        XCTAssertFalse(fixed.shouldContinue([1, 248046, 2]))
    }

    func testLocalGateUpFusion() throws {
        guard let path = ProcessInfo.processInfo.environment["FMLX_TRAINED_ISLAND_FIXTURE"] else {
            throw XCTSkip("Set FMLX_TRAINED_ISLAND_FIXTURE")
        }
        let tensors = try loadArrays(
            url: URL(filePath: path).appendingPathComponent("projection.safetensors"))
        let seedInput = try XCTUnwrap(tensors["input"]).asType(.bfloat16)
        let weights = try ["gate", "up"].map { try XCTUnwrap(tensors[$0 + ".weight"]) }
        let scales = try ["gate", "up"].map { try XCTUnwrap(tensors[$0 + ".scales"]) }
        let biases = try ["gate", "up"].map { try XCTUnwrap(tensors[$0 + ".biases"]) }
        let fusedWeight = concatenated(weights, axis: 0)
        let fusedScales = concatenated(scales, axis: 0)
        let fusedBiases = concatenated(biases, axis: 0)
        eval(weights + scales + biases + [seedInput, fusedWeight, fusedScales, fusedBiases])
        let fusedProjection = MLX.compile { args in
            let pair = quantizedMM(
                args[0], args[1], scales: args[2], biases: args[3],
                transpose: true, groupSize: 64, bits: 4)
            return [silu(pair[0..., 0..., ..<17408]) * pair[0..., 0..., 17408...]]
        }
        let splitProjection = MLX.compile { args in
            let gate = quantizedMM(
                args[0], args[1], scales: args[2], biases: args[3],
                transpose: true, groupSize: 64, bits: 4)
            let up = quantizedMM(
                args[0], args[4], scales: args[5], biases: args[6],
                transpose: true, groupSize: 64, bits: 4)
            return [silu(gate) * up]
        }
        for length in [1, 2, 128, 512] {
            let repeated = tiled(
                seedInput, repetitions: [1, (length + seedInput.dim(1) - 1) / seedInput.dim(1), 1])
            let input = repeated[0..., ..<length]
            eval(input)
            func run(fused: Bool) -> MLXArray {
                if fused {
                    return fusedProjection([input, fusedWeight, fusedScales, fusedBiases])[0]
                }
                return splitProjection([
                    input, weights[0], scales[0], biases[0], weights[1], scales[1], biases[1],
                ])[0]
            }
            let reference = run(fused: false)
            eval(reference)
            for trial in 0 ..< 20 {
                for fused in trial % 2 == 0 ? [false, true] : [true, false] {
                    let start = now()
                    var result = reference
                    for _ in 0 ..< 8 {
                        result = run(fused: fused)
                        eval(result)
                    }
                    let milliseconds = (now() - start) * 1000 / 8
                    let same = arrayEqual(result, reference).item(Bool.self)
                    XCTAssertTrue(same, "S=\(length), fused=\(fused)")
                    let row: [String: Any] = [
                        "tokens": length, "fused": fused, "trial": trial,
                        "ms": milliseconds, "bitIdentical": same, "compiled": true,
                    ]
                    let data = try JSONSerialization.data(
                        withJSONObject: row, options: [.sortedKeys])
                    print("FMLX_GATE_UP " + String(decoding: data, as: UTF8.self))
                }
            }
        }
    }

    private func provider(_ input: MLXArray, gpuLayout: Bool, singleCopy: Bool = false) throws
        -> MLDictionaryFeatureProvider
    {
        let length = input.dim(1)
        let transposed = input.asType(.float16).transposed(0, 2, 1)
        let layout = gpuLayout ? contiguous(transposed) : transposed
        let array = try MLMultiArray(
            shape: [1, 5120, 1, NSNumber(value: length)], dataType: .float16)
        if singleCopy {
            layout.asData(access: .noCopyIfContiguous).data.withUnsafeBytes { source in
                array.dataPointer.copyMemory(from: source.baseAddress!, byteCount: source.count)
            }
            withExtendedLifetime(layout) {}
        } else {
            let host = layout.asArray(Float16.self)
            host.withUnsafeBufferPointer { source in
                array.dataPointer.assumingMemoryBound(to: Float16.self)
                    .update(from: source.baseAddress!, count: host.count)
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: array)])
    }

    private func borrowedProvider(_ input: MLXArray, device: any MTLDevice) throws
        -> BorrowedIslandInput
    {
        let length = input.dim(1)
        let backing = contiguous(input.asType(.float16).transposed(0, 2, 1))
        let buffer = try XCTUnwrap(backing.asMTLBuffer(device: device, noCopy: true))
        let array = try MLMultiArray(
            dataPointer: buffer.contents(), shape: [1, 5120, 1, NSNumber(value: length)],
            dataType: .float16,
            strides: [
                NSNumber(value: 5120 * length), NSNumber(value: length),
                NSNumber(value: length), 1,
            ], deallocator: nil)
        let provider = try MLDictionaryFeatureProvider(
            dictionary: ["x": MLFeatureValue(multiArray: array)])
        return BorrowedIslandInput(provider: provider, backing: backing, buffer: buffer)
    }

    private func output(
        _ prediction: any MLFeatureProvider, length: Int, channels: Int = 17408,
        singleCopy: Bool = false, stridedCopy: Bool = false
    ) throws -> MLXArray {
        let array = try XCTUnwrap(prediction.featureValue(for: "y")?.multiArrayValue)
        XCTAssertEqual(array.dataType, .float16)
        XCTAssertEqual(array.shape.map(\.intValue), [1, channels, 1, length])
        let strides = array.strides.map(\.intValue)
        if reportedLayouts.insert(strides).inserted {
            try emit([
                "variant": "output-layout", "shape": array.shape.map(\.intValue),
                "strides": strides,
            ])
        }
        if stridedCopy {
            return try array.withUnsafeBytes { bytes in
                let liveStrides = array.strides.map(\.intValue)
                let physicalCount =
                    (channels - 1) * liveStrides[1] + (length - 1) * liveStrides[3] + 1
                guard liveStrides.allSatisfy({ $0 > 0 }),
                    physicalCount > 0, physicalCount <= bytes.count / MemoryLayout<Float16>.size
                else { throw NSError(domain: "FMLXIslandLayout", code: 1) }
                let copied = MLXArray(
                    UnsafeBufferPointer(
                        start: bytes.baseAddress!.assumingMemoryBound(to: Float16.self),
                        count: physicalCount))
                return asStrided(
                    copied, [1, channels, length],
                    strides: [physicalCount, liveStrides[1], liveStrides[3]]
                ).transposed(0, 2, 1).asType(.bfloat16)
            }
        }
        if singleCopy && strides[1] == length && strides[3] == 1 {
            return withExtendedLifetime(array) {
                let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
                return MLXArray(
                    UnsafeBufferPointer(start: pointer, count: channels * length),
                    [1, channels, length]
                ).transposed(0, 2, 1).asType(.bfloat16)
            }
        }
        let values: [Float16] = withExtendedLifetime(array) {
            let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
            if strides[1] == length && strides[3] == 1 {
                return Array(UnsafeBufferPointer(start: pointer, count: channels * length))
            }
            return (0 ..< channels).flatMap { channel in
                (0 ..< length).map { token in pointer[channel * strides[1] + token * strides[3]] }
            }
        }
        return MLXArray(values, [1, channels, length]).transposed(0, 2, 1).asType(.bfloat16)
    }

    @available(macOS 14.4, *)
    private func placement(_ url: URL, configuration: MLModelConfiguration) async throws
        -> [[String: Any]]
    {
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
        guard case .program(let program) = plan.modelStructure else { return [] }
        func name(_ device: MLComputeDevice) -> String {
            switch device {
            case .cpu: "cpu"
            case .gpu: "gpu"
            case .neuralEngine: "neuralEngine"
            @unknown default: "unknown"
            }
        }
        return program.functions.values.flatMap { $0.block.operations }.map { operation in
            let usage = plan.deviceUsage(for: operation)
            return [
                "operation": operation.operatorName,
                "preferred": usage.map { name($0.preferred) } ?? "unknown",
                "supported": usage?.supported.map(name) ?? [],
            ]
        }
    }

    func testLocalGateSplitOverlap() async throws {
        guard #available(macOS 14.4, *),
            let path = ProcessInfo.processInfo.environment["FMLX_GATE_SPLIT_FIXTURE"]
        else { throw XCTSkip("Set FMLX_GATE_SPLIT_FIXTURE on macOS 14.4+") }
        let fixtureURL = URL(filePath: path)
        let fixture = try JSONDecoder().decode(
            GateSplitFixture.self, from: Data(contentsOf: fixtureURL))
        guard fixture.tokens == 128, fixture.exports.map(\.rows) == [1024, 2048, 4096],
            fixture.exports.allSatisfy({ $0.name == "gate-cpu-tail-\($0.rows)" })
        else { return XCTFail("Invalid gate split fixture") }
        defer { Stream().synchronize() }
        let source = URL(filePath: fixture.sourceDirectory)
        let tensors = try loadArrays(url: source.appendingPathComponent("projection.safetensors"))
        let input = try XCTUnwrap(tensors["input"]).asType(.bfloat16)
        XCTAssertEqual(input.shape, [1, 128, 5120])
        let gateWeight = try XCTUnwrap(tensors["gate.weight"])
        let gateScales = try XCTUnwrap(tensors["gate.scales"])
        let gateBiases = try XCTUnwrap(tensors["gate.biases"])
        let upWeight = try XCTUnwrap(tensors["up.weight"])
        let upScales = try XCTUnwrap(tensors["up.scales"])
        let upBiases = try XCTUnwrap(tensors["up.biases"])
        var prefixes: [Int: [MLXArray]] = [:]
        var cpuBranches: [Int: IslandBranch] = [:]
        let wholeURL = source.appendingPathComponent("up-fp16.mlmodelc")
        let whole = try IslandBranch(url: wholeURL, units: .cpuAndNeuralEngine)
        try emit([
            "variant": "gate-split-plan", "rows": 17408, "projection": "up",
            "operations": try await placement(wholeURL, configuration: whole.model.configuration),
        ])
        for item in fixture.exports {
            let end = 17408 - item.rows
            prefixes[item.rows] = [gateWeight[..<end], gateScales[..<end], gateBiases[..<end]]
            let url = fixtureURL.deletingLastPathComponent().appendingPathComponent(
                item.name + ".mlmodelc")
            let branch = try IslandBranch(url: url, units: .cpuOnly)
            cpuBranches[item.rows] = branch
            try emit([
                "variant": "gate-split-plan", "rows": item.rows, "projection": "gate",
                "operations": try await placement(url, configuration: branch.model.configuration),
            ])
        }
        eval(Array(tensors.values) + [input] + prefixes.values.flatMap { $0 })
        func gate(rows: Int = 0) -> MLXArray {
            let weights = prefixes[rows] ?? [gateWeight, gateScales, gateBiases]
            return quantizedMM(
                input, weights[0], scales: weights[1], biases: weights[2],
                transpose: true, groupSize: 64, bits: 4)
        }
        func up() -> MLXArray {
            quantizedMM(
                input, upWeight, scales: upScales, biases: upBiases,
                transpose: true, groupSize: 64, bits: 4)
        }
        let nativeGate = gate()
        let native = silu(nativeGate) * up()
        eval(nativeGate, native)
        for rows in fixture.exports.map(\.rows) {
            XCTAssertTrue(
                arrayEqual(gate(rows: rows), nativeGate[0..., 0..., ..<(17408 - rows)]).item(
                    Bool.self))
        }
        let native32 = native.asType(.float32)
        let initial = try await whole.predict(
            IslandFeatures(
                value: provider(input, gpuLayout: true, singleCopy: true), completedSeconds: now()))
        let twoWay = silu(gate()) * (try output(initial.value, length: 128, stridedCopy: true))
        let twoWay32 = twoWay.asType(.float32)
        eval(native32, twoWay32)
        var references: [String: MLXArray] = [:]
        let variants = [
            "native", "two-way", "gate-1024", "gate-2048", "gate-4096", "gate-4096-serial",
        ]
        for trial in 0 ..< 21 {
            for variant in trial.isMultiple(of: 2) ? variants : Array(variants.reversed()) {
                try Task.checkCancellation()
                let rows = variant.hasPrefix("gate-") ? Int(variant.split(separator: "-")[1])! : 0
                let serial = variant.hasSuffix("serial")
                Memory.peakMemory = 0
                let start = now()
                let result: MLXArray
                var phases: [String: Double] = [:]
                if variant == "native" {
                    result = silu(gate()) * up()
                } else {
                    let features = IslandFeatures(
                        value: try provider(input, gpuLayout: true, singleCopy: true),
                        completedSeconds: now())
                    let inputReady = now()
                    let gateGPU = gate(rows: rows)
                    if serial { eval(gateGPU) } else { asyncEval(gateGPU) }
                    let submitted = now()
                    let upResult: MLXArray
                    let combinedGate: MLXArray
                    let predicted: Double
                    if let cpu = cpuBranches[rows] {
                        let predictions: (IslandFeatures, IslandFeatures)
                        if serial {
                            predictions = (
                                try await whole.predict(features), try await cpu.predict(features)
                            )
                        } else {
                            async let aneResult = whole.predict(features)
                            async let cpuResult = cpu.predict(features)
                            predictions = try await (aneResult, cpuResult)
                        }
                        predicted = now()
                        phases["aneCompleteMs"] =
                            (predictions.0.completedSeconds - submitted) * 1000
                        phases["cpuCompleteMs"] =
                            (predictions.1.completedSeconds - submitted) * 1000
                        upResult = try output(predictions.0.value, length: 128, stridedCopy: true)
                        combinedGate = concatenated(
                            [
                                gateGPU,
                                try output(
                                    predictions.1.value, length: 128, channels: rows,
                                    stridedCopy: true),
                            ], axis: -1)
                    } else {
                        let prediction = try await whole.predict(features)
                        predicted = now()
                        phases["aneCompleteMs"] = (prediction.completedSeconds - submitted) * 1000
                        upResult = try output(prediction.value, length: 128, stridedCopy: true)
                        combinedGate = gateGPU
                    }
                    let copied = now()
                    result = silu(combinedGate) * upResult
                    eval(result)
                    phases.merge([
                        "inputMs": (inputReady - start) * 1000,
                        "gateSubmitOrWaitMs": (submitted - inputReady) * 1000,
                        "predictionMs": (predicted - submitted) * 1000,
                        "outputCopyMs": (copied - predicted) * 1000,
                        "joinMs": (now() - copied) * 1000,
                    ]) { _, new in new }
                }
                eval(result)
                let elapsed = (now() - start) * 1000
                let peak = Memory.peakMemory
                let value = result.asType(.float32)
                func error(_ reference: MLXArray) -> Float {
                    sqrt(mean(square(value - reference)) / maximum(mean(square(reference)), 1e-20))
                        .item(Float.self)
                }
                let nativeError = error(native32)
                let twoWayError = error(twoWay32)
                XCTAssertTrue(nativeError.isFinite && twoWayError.isFinite)
                let reference = references[variant] ?? result
                let stable = arrayEqual(result, reference).item(Bool.self)
                XCTAssertTrue(stable)
                references[variant] = reference
                try emit([
                    "variant": "gate-split-" + variant, "trial": trial, "tokens": 128,
                    "cpuGateRows": rows, "aneUpRows": variant == "native" ? 0 : 17408,
                    "totalMs": elapsed, "phases": phases, "peakMLXBytes": peak,
                    "relativeRMSE": nativeError, "relativeRMSEVsTwoWay": twoWayError,
                    "repeatIdentical": stable,
                    "thermalState": ProcessInfo.processInfo.thermalState.rawValue,
                ])
            }
        }
    }

    func testLocalTrainedProjectionOverlap() async throws {
        guard #available(macOS 14.4, *),
            let path = ProcessInfo.processInfo.environment["FMLX_TRAINED_ISLAND_FIXTURE"]
        else { throw XCTSkip("Set FMLX_TRAINED_ISLAND_FIXTURE on macOS 14.4+") }
        let root = URL(filePath: path)
        let tensors = try loadArrays(url: root.appendingPathComponent("projection.safetensors"))
        let input = try XCTUnwrap(tensors["input"]).asType(.bfloat16)
        let halfInput = input.asType(.float16)
        let gateWeight = try XCTUnwrap(tensors["gate.weight"])
        let gateScales = try XCTUnwrap(tensors["gate.scales"])
        let gateBiases = try XCTUnwrap(tensors["gate.biases"])
        let upWeight = try XCTUnwrap(tensors["up.weight"])
        let upScales = try XCTUnwrap(tensors["up.scales"])
        let upBiases = try XCTUnwrap(tensors["up.biases"])
        let upHalf = try XCTUnwrap(tensors["up.fp16"])
        let gateHalf = try XCTUnwrap(tensors["gate.fp16"])
        eval(Array(tensors.values) + [input, halfInput])
        func gate() -> MLXArray {
            quantizedMM(
                input, gateWeight, scales: gateScales, biases: gateBiases,
                transpose: true, groupSize: 64, bits: 4)
        }
        func up() -> MLXArray {
            quantizedMM(
                input, upWeight, scales: upScales, biases: upBiases,
                transpose: true, groupSize: 64, bits: 4)
        }
        let reference = silu(gate()) * up()
        eval(reference)
        let reference32 = reference.asType(.float32)
        eval(reference32)

        func report(
            _ result: MLXArray, variant: String, unit: String, trial: Int,
            start: Double, phases: [String: Double]
        ) throws {
            let totalMS = (now() - start) * 1000
            let diff = result.asType(.float32) - reference32
            let relativeRMSE = sqrt(mean(square(diff)) / maximum(mean(square(reference32)), 1e-20))
                .item(Float.self)
            let mismatches = sum(result .!= reference).item(Int.self)
            try emit([
                "variant": variant, "units": unit, "trial": trial,
                "tokens": input.dim(1), "totalMs": totalMS, "phases": phases,
                "relativeRMSE": relativeRMSE, "mismatchedElements": mismatches,
                "elements": result.size, "peakMLXBytes": Memory.peakMemory,
                "thermalState": ProcessInfo.processInfo.thermalState.rawValue,
            ])
        }

        for trial in 0 ..< 10 {
            let variants = ["quantized-gpu", "dense-up-gpu", "dense-both-gpu"]
            for variant in trial % 2 == 0 ? variants : Array(variants.reversed()) {
                Memory.peakMemory = 0
                let start = now()
                let result: MLXArray
                switch variant {
                case "dense-up-gpu":
                    result = silu(gate()) * matmul(halfInput, upHalf.T).asType(.bfloat16)
                case "dense-both-gpu":
                    result =
                        silu(matmul(halfInput, gateHalf.T).asType(.bfloat16))
                        * matmul(halfInput, upHalf.T).asType(.bfloat16)
                default: result = silu(gate()) * up()
                }
                eval(result)
                try report(
                    result, variant: variant, unit: "gpu", trial: trial, start: start, phases: [:])
            }
        }

        let url = root.appendingPathComponent("up-fp16.mlmodelc")
        for (unit, units) in [
            ("cpuAndNeuralEngine", MLComputeUnits.cpuAndNeuralEngine), ("cpuOnly", .cpuOnly),
        ] {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = units
            configuration.allowLowPrecisionAccumulationOnGPU = false
            let loadStart = now()
            let coreModel = try MLModel(contentsOf: url, configuration: configuration)
            try emit([
                "variant": "compute-plan", "units": unit,
                "loadMs": (now() - loadStart) * 1000,
                "operations": try await placement(url, configuration: configuration),
            ])
            for trial in 0 ..< 20 {
                let variants = [
                    "quantized-gpu", "split-serial", "split-parallel", "gpu-layout-serial",
                    "gpu-layout-parallel",
                ]
                for variant in trial % 2 == 0 ? variants : Array(variants.reversed()) {
                    let parallel = variant.hasSuffix("parallel")
                    Memory.peakMemory = 0
                    let start = now()
                    if variant == "quantized-gpu" {
                        let result = silu(gate()) * up()
                        eval(result)
                        try report(
                            result, variant: variant, unit: unit, trial: trial,
                            start: start, phases: [:])
                        continue
                    }
                    let features = try provider(input, gpuLayout: variant.hasPrefix("gpu-layout"))
                    let inputReady = now()
                    let gateResult = gate()
                    if parallel { asyncEval(gateResult) } else { eval(gateResult) }
                    let submitted = now()
                    // This task retains sole ownership of its MLX arrays while both devices run.
                    let prediction = try await coreModel.prediction(from: features)
                    let predicted = now()
                    let upResult = try output(prediction, length: input.dim(1))
                    let copied = now()
                    let result = silu(gateResult) * upResult
                    eval(result)
                    try report(
                        result, variant: variant,
                        unit: unit, trial: trial, start: start,
                        phases: [
                            "inputMs": (inputReady - start) * 1000,
                            "gateSubmitOrWaitMs": (submitted - inputReady) * 1000,
                            "predictionMs": (predicted - submitted) * 1000,
                            "outputCopyMs": (copied - predicted) * 1000,
                            "joinMs": (now() - copied) * 1000,
                        ])
                }
            }
        }
        if ProcessInfo.processInfo.environment["FMLX_ISLAND_THREEWAY"] == "1" {
            let whole = try IslandBranch(url: url, units: .cpuAndNeuralEngine)
            let aneURL = root.appendingPathComponent("up-ane-fp16.mlmodelc")
            let cpuURL = root.appendingPathComponent("up-cpu-fp16.mlmodelc")
            let ane = try IslandBranch(url: aneURL, units: .cpuAndNeuralEngine)
            let cpu = try IslandBranch(url: cpuURL, units: .cpuOnly)
            for (name, branchURL, branch) in [("ane-rows", aneURL, ane), ("cpu-rows", cpuURL, cpu)]
            {
                try emit([
                    "variant": "compute-plan", "units": name,
                    "operations": try await placement(
                        branchURL, configuration: branch.model.configuration),
                ])
            }
            for trial in 0 ..< 20 {
                let variants = ["quantized-gpu", "two-way", "three-way", "three-way-serial"]
                for variant in trial % 2 == 0 ? variants : Array(variants.reversed()) {
                    Memory.peakMemory = 0
                    let start = now()
                    if variant == "quantized-gpu" {
                        let result = silu(gate()) * up()
                        eval(result)
                        try report(
                            result, variant: variant, unit: "cpu-gpu-ane", trial: trial,
                            start: start, phases: [:])
                        continue
                    }
                    let features = IslandFeatures(
                        value: try provider(input, gpuLayout: true), completedSeconds: now())
                    let inputReady = now()
                    let gateResult = gate()
                    if variant == "three-way-serial" {
                        eval(gateResult)
                    } else {
                        asyncEval(gateResult)
                    }
                    let submitted = now()
                    let upResult: MLXArray
                    let predicted: Double
                    var branchTimes: [String: Double] = [:]
                    if variant == "two-way" {
                        let result = try await whole.predict(features)
                        predicted = now()
                        upResult = try output(result.value, length: input.dim(1))
                    } else {
                        let results: (IslandFeatures, IslandFeatures)
                        if variant == "three-way-serial" {
                            results = (
                                try await ane.predict(features), try await cpu.predict(features)
                            )
                        } else {
                            async let aneResult = ane.predict(features)
                            async let cpuResult = cpu.predict(features)
                            results = try await (aneResult, cpuResult)
                        }
                        predicted = now()
                        branchTimes["aneCompleteMs"] =
                            (results.0.completedSeconds - submitted) * 1000
                        branchTimes["cpuCompleteMs"] =
                            (results.1.completedSeconds - submitted) * 1000
                        upResult = concatenated(
                            [
                                try output(results.0.value, length: input.dim(1), channels: 16384),
                                try output(results.1.value, length: input.dim(1), channels: 1024),
                            ], axis: -1)
                    }
                    let copied = now()
                    let result = silu(gateResult) * upResult
                    eval(result)
                    branchTimes.merge([
                        "inputMs": (inputReady - start) * 1000,
                        "gateSubmitOrWaitMs": (submitted - inputReady) * 1000,
                        "predictionMs": (predicted - submitted) * 1000,
                        "outputCopyMs": (copied - predicted) * 1000,
                        "joinMs": (now() - copied) * 1000,
                    ]) { _, new in new }
                    try report(
                        result, variant: variant, unit: "cpu-gpu-ane", trial: trial, start: start,
                        phases: branchTimes)
                }
            }
        }
        Stream().synchronize()
    }
}
#endif
