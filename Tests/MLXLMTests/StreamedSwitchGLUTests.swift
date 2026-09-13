// Copyright © 2026 Faux Fox.

import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

final class StreamedSwitchGLUTests: XCTestCase {
    func testAffineCompactDecodeMatchesResidentWithPermutedSlots() throws {
        let resident = makeResidentLayer()
        let payloads = try makePayloads(from: resident)
        let streamed = StreamedSwitchGLU(inputDims: 64, hiddenDims: 64)

        for tokenCount in 1 ... 2 {
            let input = MLXRandom.normal([tokenCount, 64]).asType(.bfloat16)
            let originalIDs: [UInt32] = tokenCount == 1 ? [3, 1, 3, 0] : [3, 1, 3, 0, 2, 1, 4, 0]
            let topK = 4
            let scores = softmax(
                MLXRandom.normal([tokenCount, topK]), axis: -1, precise: true
            ).asType(.bfloat16)
            let expected = resident.callAndWeightedReduce(
                input,
                MLXArray(originalIDs).reshaped(tokenCount, topK),
                weights: scores,
                fuseSortedReduction: false)

            let compactIDs = firstOccurrenceSlots(originalIDs)
            let compactExperts = uniqueIDs(originalIDs).map { payloads[Int($0)] }
            let actual = try streamed.callAndWeightedReduce(
                input,
                compactRouterIDs: compactIDs,
                topK: topK,
                weights: scores,
                experts: compactExperts)
            eval(expected, actual)

            assertBitIdentical(actual, expected, "S=\(tokenCount)")
        }
    }

    func testForcedEvictionStillExecutesWithTheSelectedCompactExperts() throws {
        let resident = makeResidentLayer()
        let payloads = try makePayloads(from: resident)
        let streamed = StreamedSwitchGLU(inputDims: 64, hiddenDims: 64)
        var cache = try ExpertWeightLRUCache<Int, StreamedQuantizedExpertWeights>(
            capacityBytes: 200)
        _ = try cache.insert(payloads[3], for: 3, byteCount: 100)
        _ = try cache.insert(payloads[1], for: 1, byteCount: 100)
        _ = cache.value(for: 3)
        XCTAssertEqual(
            try cache.insert(payloads[2], for: 2, byteCount: 100),
            .stored([.init(key: 1, byteCount: 100)]))

        let compactExperts = [
            try XCTUnwrap(cache.value(for: 3)), try XCTUnwrap(cache.value(for: 2)),
        ]
        let input = MLXRandom.normal([1, 64]).asType(.bfloat16)
        let originalIDs: [UInt32] = [3, 2]
        let scores = MLXArray([Float(0.75), 0.25]).asType(.bfloat16).reshaped(1, 2)
        let expected = resident.callAndWeightedReduce(
            input, MLXArray(originalIDs).reshaped(1, 2), weights: scores,
            fuseSortedReduction: false)
        let actual = try streamed.callAndWeightedReduce(
            input, compactRouterIDs: [0, 1], topK: 2, weights: scores, experts: compactExperts)
        eval(expected, actual)

        assertBitIdentical(actual, expected, "evicted compact slots")
    }

    func testGroup64BFloat16ParametersMatchFullExpertAxis() throws {
        MLXRandom.seed(59)
        let experts = 5
        let inputDims = 64
        let hiddenDims = 64
        let (gateWeight, gateScales, gateBiases) = try quantizedGroup64(
            MLXRandom.normal([experts, hiddenDims, inputDims]))
        let (upWeight, upScales, upBiases) = try quantizedGroup64(
            MLXRandom.normal([experts, hiddenDims, inputDims]))
        let (downWeight, downScales, downBiases) = try quantizedGroup64(
            MLXRandom.normal([experts, inputDims, hiddenDims]))
        let payloads = try (0 ..< experts).map { expertID in
            try StreamedQuantizedExpertWeights(
                gateWeight: gateWeight[expertID], gateScales: gateScales[expertID],
                gateBiases: gateBiases[expertID], upWeight: upWeight[expertID],
                upScales: upScales[expertID], upBiases: upBiases[expertID],
                downWeight: downWeight[expertID], downScales: downScales[expertID],
                downBiases: downBiases[expertID], byteCount: 100)
        }
        let input = MLXRandom.normal([2, inputDims]).asType(.bfloat16)
        let originalIDs: [UInt32] = [4, 1, 3, 1, 0, 4, 2, 0]
        let topK = 4
        let scores = softmax(MLXRandom.normal([2, topK]), axis: -1, precise: true).asType(.bfloat16)
        let originalIndices = MLXArray(originalIDs).reshaped(2, topK)
        let expected = fullExpertAxisOutput(
            input: input, indices: originalIndices, scores: scores,
            gateWeight: gateWeight, gateScales: gateScales, gateBiases: gateBiases,
            upWeight: upWeight, upScales: upScales, upBiases: upBiases,
            downWeight: downWeight, downScales: downScales, downBiases: downBiases)
        let compactExpertIDs = uniqueIDs(originalIDs)
        let actual = try StreamedSwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, groupSize: 64
        ).callAndWeightedReduce(
            input, compactRouterIDs: firstOccurrenceSlots(originalIDs), topK: topK,
            weights: scores, experts: compactExpertIDs.map { payloads[Int($0)] })
        eval(expected, actual)

        assertBitIdentical(actual, expected, "group64 BF16")
    }

    func testRejectsPrefillBeforeConstructingCompactExpertTensors() throws {
        let resident = makeResidentLayer()
        let payloads = try makePayloads(from: resident)
        let streamed = StreamedSwitchGLU(inputDims: 64, hiddenDims: 64)
        let input = MLXRandom.normal([3, 64]).asType(.bfloat16)
        let scores = MLXArray.ones([3, 1]).asType(.bfloat16)

        XCTAssertThrowsError(
            try streamed.callAndWeightedReduce(
                input, compactRouterIDs: [0, 0, 0], topK: 1, weights: scores,
                experts: [payloads[0]])
        ) {
            XCTAssertEqual($0 as? StreamedSwitchGLUError, .unsupportedSequenceLength(3))
        }
    }

    private func makeResidentLayer() -> SwitchGLU {
        MLXRandom.seed(53)
        let layer = SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 5)
        layer.apply { $0.asType(.bfloat16) }
        quantize(model: layer, groupSize: 32, bits: 4)
        return layer
    }

    private func makePayloads(from layer: SwitchGLU) throws -> [StreamedQuantizedExpertWeights] {
        let gate = try XCTUnwrap(layer.gateProj as? QuantizedSwitchLinear)
        let up = try XCTUnwrap(layer.upProj as? QuantizedSwitchLinear)
        let down = try XCTUnwrap(layer.downProj as? QuantizedSwitchLinear)
        let gateBiases = try XCTUnwrap(gate.biases)
        let upBiases = try XCTUnwrap(up.biases)
        let downBiases = try XCTUnwrap(down.biases)
        return try (0 ..< 5).map { expertID in
            try StreamedQuantizedExpertWeights(
                gateWeight: gate.weight[expertID],
                gateScales: gate.scales[expertID],
                gateBiases: gateBiases[expertID],
                upWeight: up.weight[expertID],
                upScales: up.scales[expertID],
                upBiases: upBiases[expertID],
                downWeight: down.weight[expertID],
                downScales: down.scales[expertID],
                downBiases: downBiases[expertID],
                byteCount: 100)
        }
    }

    private func firstOccurrenceSlots(_ expertIDs: [UInt32]) -> [UInt32] {
        var slots: [UInt32: UInt32] = [:]
        return expertIDs.map { expertID in
            if let slot = slots[expertID] {
                return slot
            }
            let slot = UInt32(slots.count)
            slots[expertID] = slot
            return slot
        }
    }

    private func uniqueIDs(_ expertIDs: [UInt32]) -> [UInt32] {
        var seen = Set<UInt32>()
        return expertIDs.filter { seen.insert($0).inserted }
    }

    private func quantizedGroup64(_ weight: MLXArray) throws -> (MLXArray, MLXArray, MLXArray) {
        let (quantizedWeight, scales, biases) = MLX.quantized(
            weight, groupSize: 64, bits: 4, mode: .affine)
        return (
            quantizedWeight,
            scales.asType(.bfloat16),
            try XCTUnwrap(biases).asType(.bfloat16)
        )
    }

    private func fullExpertAxisOutput(
        input: MLXArray,
        indices: MLXArray,
        scores: MLXArray,
        gateWeight: MLXArray,
        gateScales: MLXArray,
        gateBiases: MLXArray,
        upWeight: MLXArray,
        upScales: MLXArray,
        upBiases: MLXArray,
        downWeight: MLXArray,
        downScales: MLXArray,
        downBiases: MLXArray
    ) -> MLXArray {
        var expanded = MLX.expandedDimensions(input, axes: [-2, -3])
        let up = MLX.gatherQuantizedMM(
            expanded, upWeight, scales: upScales, biases: upBiases, rhsIndices: indices,
            transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: false)
        let gate = MLX.gatherQuantizedMM(
            expanded, gateWeight, scales: gateScales, biases: gateBiases, rhsIndices: indices,
            transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: false)
        expanded = MLX.gatherQuantizedMM(
            compiledSiluProduct(gate, up), downWeight, scales: downScales, biases: downBiases,
            rhsIndices: indices, transpose: true, groupSize: 64, bits: 4, mode: .affine,
            sortedIndices: false)
        return weightedExpertSum(MLX.squeezed(expanded, axis: -2), scores)
    }

    private func assertBitIdentical(
        _ actual: MLXArray,
        _ expected: MLXArray,
        _ label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.dtype, expected.dtype, "\(label): dtype", file: file, line: line)
        XCTAssertEqual(actual.shape, expected.shape, "\(label): shape", file: file, line: line)
        let actualValues = actual.asType(.float32).asArray(Float.self)
        let expectedValues = expected.asType(.float32).asArray(Float.self)
        XCTAssertEqual(
            actualValues.map(\.bitPattern), expectedValues.map(\.bitPattern),
            "\(label): values", file: file, line: line)
    }
}
