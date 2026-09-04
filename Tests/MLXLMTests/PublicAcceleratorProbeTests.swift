// Copyright © 2026 Apple Inc.

#if os(macOS)
import CoreML
import Foundation

#if !PUBLIC_ACCELERATOR_STANDALONE
import MLX
import XCTest
#endif

/// Untrained graph measurements. Generate external fixtures with public-accelerator-fixtures.py.
private enum PublicAcceleratorProbe {
    static let units: [(String, MLComputeUnits)] = [
        ("cpuOnly", .cpuOnly), ("cpuAndGPU", .cpuAndGPU),
        ("cpuAndNeuralEngine", .cpuAndNeuralEngine), ("all", .all),
    ]

    struct Failure: Error { let message: String }

    static func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw Failure(message: message) }
    }

    static func milliseconds(since start: ContinuousClock.Instant) -> Double {
        let duration = ContinuousClock.now - start
        return Double(duration.components.seconds) * 1000
            + Double(duration.components.attoseconds) / 1e15
    }

    static func median(_ samples: [Double]) -> Double {
        samples.sorted()[samples.count / 2]
    }

    static func emit(_ values: [String: Any]) throws {
        let data = try JSONSerialization.data(withJSONObject: values, options: [.sortedKeys])
        print("[PUBLIC_ACCELERATOR] " + String(decoding: data, as: UTF8.self))
    }

    static func read<T>(_ name: String, root: URL, as type: T.Type) throws -> [T] {
        let data = try Data(contentsOf: root.appendingPathComponent(name + ".bin"))
        try require(data.count.isMultiple(of: MemoryLayout<T>.stride), "Invalid fixture size")
        return data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: MemoryLayout<T>.stride).map {
                bytes.loadUnaligned(fromByteOffset: $0, as: T.self)
            }
        }
    }

    static func makeArray(_ values: [Float16], shape: [Int]) throws -> MLMultiArray {
        try require(values.count == shape.reduce(1, *), "Invalid input shape")
        let result = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float16)
        try write(values, to: result)
        return result
    }

    static func offsets(_ array: MLMultiArray) -> [Int]? {
        let dimensions = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        var expected = 1
        var contiguous = true
        for i in dimensions.indices.reversed() {
            if dimensions[i] > 1 && strides[i] != expected { contiguous = false }
            expected *= dimensions[i]
        }
        guard !contiguous else { return nil }
        return (0 ..< array.count).map { index in
            var remainder = index
            var offset = 0
            for i in dimensions.indices.reversed() {
                offset += (remainder % dimensions[i]) * strides[i]
                remainder /= dimensions[i]
            }
            return offset
        }
    }

    static func write(_ values: [Float16], to array: MLMultiArray) throws {
        try require(array.dataType == .float16 && array.count == values.count, "Invalid buffer")
        let layout = offsets(array)
        withExtendedLifetime(array) {
            let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
            if let layout {
                for i in values.indices { pointer[layout[i]] = values[i] }
            } else {
                values.withUnsafeBufferPointer { source in
                    pointer.update(from: source.baseAddress!, count: source.count)
                }
            }
        }
    }

    static func read(_ array: MLMultiArray) throws -> [Float16] {
        try require(array.dataType == .float16, "Expected FP16 output")
        let layout = offsets(array)
        return withExtendedLifetime(array) {
            let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
            if let layout { return layout.map { pointer[$0] } }
            return Array(UnsafeBufferPointer(start: pointer, count: array.count))
        }
    }

    static func error(_ values: [Float16], reference: [Float]) throws -> Double {
        try require(values.count == reference.count, "Output size mismatch")
        var squaredError = 0.0
        var squaredReference = 0.0
        for (value, reference) in zip(values, reference) {
            try require(value.isFinite, "Nonfinite output")
            squaredError += pow(Double(value) - Double(reference), 2)
            squaredReference += pow(Double(reference), 2)
        }
        return sqrt(squaredError / max(squaredReference, 1e-20))
    }

    @available(macOS 14.4, *)
    static func plan(url: URL, configuration: MLModelConfiguration) async throws -> [[String: Any]]
    {
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
        guard case .program(let program) = plan.modelStructure else { return [] }
        func deviceName(_ device: MLComputeDevice) -> String {
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
                "preferred": usage.map { deviceName($0.preferred) } ?? "unknown",
                "supported": usage?.supported.map(deviceName) ?? [],
            ]
        }
    }

    static func run(root: URL) async throws {
        struct Fixture: Decodable { let shape: [Int] }
        let fixture = try JSONDecoder().decode(
            Fixture.self, from: Data(contentsOf: root.appendingPathComponent("fixture.json")))
        let shape = fixture.shape
        try require(
            shape.count == 4 && shape[0] == 1 && shape[1] == 512 && shape[2] == 1
                && [128, 2048].contains(shape[3]), "Unexpected fixture dimensions")
        let input = try read("input-f32", root: root, as: Float.self)
        let reference = try read("reference-f32", root: root, as: Float.self)
        try require(input.count == shape.reduce(1, *), "Unexpected fixture shape")
        #if !PUBLIC_ACCELERATOR_STANDALONE
        let mlxInput = MLXArray(input, shape)
        eval(mlxInput)
        let inputBoundary = "evaluatedMLXFloat32-to-CoreMLFloat16"
        #else
        let inputBoundary = "hostFloat32-to-CoreMLFloat16"
        #endif

        var variants = ["projection-fp16"]
        if #available(macOS 15, *) { variants.append("projection-int4") }
        for variant in variants {
            for (unitName, unit) in units {
                let url = root.appendingPathComponent("compiled/\(variant).mlmodelc")
                let configuration = MLModelConfiguration()
                configuration.computeUnits = unit
                configuration.allowLowPrecisionAccumulationOnGPU = false
                let loadStart = ContinuousClock.now
                let model = try MLModel(contentsOf: url, configuration: configuration)
                let loadMS = milliseconds(since: loadStart)
                var inputs: [Double] = []
                var predictions: [Double] = []
                var outputs: [Double] = []
                var totals: [Double] = []
                var final: [Float16] = []
                var firstTotalMS = 0.0
                for iteration in 0 ..< 25 {
                    let sample = try autoreleasepool {
                        () throws -> (Double, Double, Double, Double) in
                        let totalStart = ContinuousClock.now
                        #if !PUBLIC_ACCELERATOR_STANDALONE
                        let values = mlxInput.asType(.float16).asArray(Float16.self)
                        #else
                        let values = input.map(Float16.init)
                        #endif
                        let array = try makeArray(values, shape: shape)
                        let features = try MLDictionaryFeatureProvider(dictionary: ["x": array])
                        let inputMS = milliseconds(since: totalStart)
                        let predictionStart = ContinuousClock.now
                        let prediction = try model.prediction(from: features)
                        guard let output = prediction.featureValue(for: "y")?.multiArrayValue else {
                            throw Failure(message: "Missing projection output")
                        }
                        let predictionMS = milliseconds(since: predictionStart)
                        let outputStart = ContinuousClock.now
                        final = try read(output)
                        #if !PUBLIC_ACCELERATOR_STANDALONE
                        let restored = MLXArray(final, shape)
                        eval(restored)
                        #endif
                        return (
                            inputMS, predictionMS, milliseconds(since: outputStart),
                            milliseconds(since: totalStart)
                        )
                    }
                    if iteration == 0 { firstTotalMS = sample.3 }
                    if iteration >= 5 {
                        inputs.append(sample.0)
                        predictions.append(sample.1)
                        outputs.append(sample.2)
                        totals.append(sample.3)
                    }
                }
                let relativeRMSE = try error(final, reference: reference)
                try require(
                    relativeRMSE < (variant == "projection-int4" ? 0.25 : 0.02),
                    "Precision regression")
                var report: [String: Any] = [
                    "graph": variant, "computeUnits": unitName, "input_boundary": inputBoundary,
                    "shape": shape,
                    "warmup": 5, "samples": 20, "load_ms": loadMS,
                    "first_total_ms": firstTotalMS, "input_median_ms": median(inputs),
                    "prediction_median_ms": median(predictions),
                    "output_median_ms": median(outputs),
                    "total_median_ms": median(totals), "relative_rmse": relativeRMSE,
                ]
                if #available(macOS 14.4, *) {
                    report["compute_plan"] = try await plan(url: url, configuration: configuration)
                }
                try emit(report)
            }
        }
        #if !PUBLIC_ACCELERATOR_STANDALONE
        let w1 = MLXArray(try read("weight1-f16", root: root, as: Float16.self), [512, 512])
        let w2 = MLXArray(try read("weight2-f16", root: root, as: Float16.self), [512, 512])
        let x = mlxInput.asType(.float16).reshaped([512, shape[3]])
        eval(w1, w2, x)
        var samples: [Double] = []
        var final: [Float16] = []
        for iteration in 0 ..< 25 {
            let start = ContinuousClock.now
            let y = matmul(w2, maximum(matmul(w1, x), 0))
            eval(y)
            if iteration >= 5 { samples.append(milliseconds(since: start)) }
            final = y.asArray(Float16.self)
        }
        let relativeRMSE = try error(final, reference: reference)
        try require(relativeRMSE < 0.02, "MLX precision regression")
        try emit([
            "graph": "projection-fp16", "computeUnits": "MLX-default-device",
            "shape": shape,
            "prediction_median_ms": median(samples), "relative_rmse": relativeRMSE,
            "boundary": "resident-evaluated-MLX-arrays",
        ])
        #endif
        if #available(macOS 15, *) { try stateRoundTrip(root: root) }
    }

    @available(macOS 15, *)
    static func stateRoundTrip(root: URL) throws {
        let url = root.appendingPathComponent("compiled/accumulator.mlmodelc")
        let stateShape = [1, 4, 32, 128]
        let count = stateShape.reduce(1, *)
        for (unitName, unit) in units {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = unit
            let model = try MLModel(contentsOf: url, configuration: configuration)
            let state = model.makeState()
            let array = try makeArray(Array(repeating: Float16(1), count: count), shape: stateShape)
            let features = try MLDictionaryFeatureProvider(dictionary: ["x": array])
            _ = try model.prediction(from: features, using: state)
            let restored = model.makeState()
            let start = ContinuousClock.now
            let exported = try state.withMultiArray(for: "cache") { try read($0) }
            #if !PUBLIC_ACCELERATOR_STANDALONE
            let mlxState = MLXArray(exported, stateShape)
            let transported = mlxState.asArray(Float16.self)
            #else
            let transported = exported
            #endif
            try restored.withMultiArray(for: "cache") { try write(transported, to: $0) }
            let transferMS = milliseconds(since: start)
            try require(exported.allSatisfy { $0 == 1 }, "State update failed")
            _ = try model.prediction(from: features, using: restored)
            let continued = try restored.withMultiArray(for: "cache") { try read($0) }
            try require(continued.allSatisfy { $0 == 2 }, "State continuation failed")
            let original = try state.withMultiArray(for: "cache") { try read($0) }
            try require(original == exported, "Independent state was mutated")
            try emit([
                "graph": "accumulator", "computeUnits": unitName,
                "state_bytes": count * 2, "state_export_import_ms": transferMS,
                "continuation_equal": true,
            ])
        }
    }
}

#if PUBLIC_ACCELERATOR_STANDALONE
@main
private struct PublicAcceleratorMain {
    static func main() async throws {
        guard CommandLine.arguments.count == 2 else {
            throw PublicAcceleratorProbe.Failure(message: "Pass the external fixture directory")
        }
        try await PublicAcceleratorProbe.run(root: URL(fileURLWithPath: CommandLine.arguments[1]))
    }
}
#else
final class PublicAcceleratorProbeTests: XCTestCase {
    func testPublicCoreMLComputeAndStateTransfer() async throws {
        guard let path = ProcessInfo.processInfo.environment["MLX_PUBLIC_ACCELERATOR_FIXTURES"]
        else {
            throw XCTSkip(
                "Set MLX_PUBLIC_ACCELERATOR_FIXTURES to opt into synthetic Core ML measurements")
        }
        try await PublicAcceleratorProbe.run(root: URL(fileURLWithPath: path))
    }
}
#endif
#endif
