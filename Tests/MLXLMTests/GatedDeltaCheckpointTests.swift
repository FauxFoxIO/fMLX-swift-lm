import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class GatedDeltaCheckpointTests: XCTestCase {
    private func inputs(
        dtype: DType, dimension: Int, fullWidth: Bool, length: Int = 2
    ) -> [MLXArray] {
        withRandomState(MLXRandom.RandomState(seed: 184)) {
            let batch = fullWidth ? 1 : 2
            let keyHeads = fullWidth ? 16 : 2
            let valueHeads = fullWidth ? 48 : 4
            let valueDimension = fullWidth ? 128 : 8
            return [
                (MLXRandom.normal([batch, length, keyHeads, dimension]) * 0.05).asType(dtype),
                (MLXRandom.normal([batch, length, keyHeads, dimension]) * 0.05).asType(dtype),
                MLXRandom.normal([batch, length, valueHeads, valueDimension]).asType(dtype),
                sigmoid(MLXRandom.normal([batch, length, valueHeads])),
                sigmoid(MLXRandom.normal([batch, length, valueHeads])),
                MLXRandom.normal([batch, valueHeads, valueDimension, dimension]),
            ]
        }
    }

    private func run(_ inputs: [MLXArray], fused: Bool) -> [MLXArray] {
        if fused {
            let result = gatedDeltaCheckpointKernel(
                q: inputs[0], k: inputs[1], v: inputs[2], g: inputs[3], beta: inputs[4],
                state: inputs[5])
            return [result.0, result.1, result.2]
        }
        let prefix = inputs.prefix(5).map { $0[0..., ..<1] }
        let suffix = inputs.prefix(5).map { $0[0..., 1...] }
        let first = gatedDeltaKernel(
            q: prefix[0], k: prefix[1], v: prefix[2], g: prefix[3], beta: prefix[4],
            state: inputs[5])
        let second = gatedDeltaKernel(
            q: suffix[0], k: suffix[1], v: suffix[2], g: suffix[3], beta: suffix[4], state: first.1)
        return [concatenated([first.0, second.0], axis: 1), second.1, first.1]
    }

    func testCheckpointMatchesSplitRecurrenceBitwise() {
        for dtype in [DType.float32, .float16, .bfloat16] {
            for dimension in [32, 128, 192] {
                let input = inputs(dtype: dtype, dimension: dimension, fullWidth: dimension == 128)
                let reference = run(input, fused: false)
                let candidate = run(input, fused: true)
                for (a, b) in zip(reference, candidate) {
                    let left = a.asType(.float32).asArray(Float.self).map(\.bitPattern)
                    let right = b.asType(.float32).asArray(Float.self).map(\.bitPattern)
                    XCTAssertEqual(left, right, "dtype=\(dtype) Dk=\(dimension) shape=\(a.shape)")
                }
            }
        }
    }

    func testFourCheckpointsMatchOnePassAndEveryPrefixBitwise() {
        for dtype in [DType.float32, .float16, .bfloat16] {
            for dimension in [32, 128, 192] {
                let input = inputs(
                    dtype: dtype, dimension: dimension, fullWidth: dimension == 128, length: 4)
                let candidate = gatedDeltaFourCheckpointKernel(
                    q: input[0], k: input[1], v: input[2], g: input[3], beta: input[4],
                    state: input[5])
                let reference = gatedDeltaKernel(
                    q: input[0], k: input[1], v: input[2], g: input[3], beta: input[4],
                    state: input[5])

                var chainedState = input[5]
                var chainedOutputs = [MLXArray]()
                var prefixStates = [MLXArray]()
                for index in 0 ..< 4 {
                    let step = gatedDeltaKernel(
                        q: input[0][0..., index ..< (index + 1), 0..., 0...],
                        k: input[1][0..., index ..< (index + 1), 0..., 0...],
                        v: input[2][0..., index ..< (index + 1), 0..., 0...],
                        g: input[3][0..., index ..< (index + 1), 0...],
                        beta: input[4][0..., index ..< (index + 1), 0...], state: chainedState)
                    chainedOutputs.append(step.0)
                    chainedState = step.1
                    if index < 3 { prefixStates.append(chainedState) }
                }

                for (actual, expected) in zip(
                    [candidate.0, candidate.1] + candidate.2,
                    [reference.0, reference.1] + prefixStates
                ) {
                    eval(actual, expected)
                    XCTAssertEqual(
                        actual.asType(.float32).asArray(Float.self).map(\.bitPattern),
                        expected.asType(.float32).asArray(Float.self).map(\.bitPattern),
                        "dtype=\(dtype) Dk=\(dimension) shape=\(actual.shape)")
                }
                let chained = concatenated(chainedOutputs, axis: 1)
                eval(candidate.0, chained, candidate.1, chainedState)
                XCTAssertEqual(
                    candidate.0.asType(.float32).asArray(Float.self).map(\.bitPattern),
                    chained.asType(.float32).asArray(Float.self).map(\.bitPattern))
                XCTAssertEqual(
                    candidate.1.asArray(Float.self).map(\.bitPattern),
                    chainedState.asArray(Float.self).map(\.bitPattern))
            }
        }
    }

    func testFourCheckpointFallbackHandlesNonKernelDimensions() {
        let input = inputs(dtype: .bfloat16, dimension: 48, fullWidth: false, length: 4)
        let candidate = gatedDeltaUpdateFourCheckpoints(
            q: input[0], k: input[1], v: input[2], a: input[3], b: input[4],
            aLog: MLXArray.zeros([input[2].dim(2)]), dtBias: MLXArray.zeros([input[2].dim(2)]),
            state: input[5])
        eval(candidate.0, candidate.1, candidate.2)
        XCTAssertEqual(candidate.0.shape[1], 4)
        XCTAssertEqual(candidate.2.count, 3)
    }

    func testLocalCheckpointPerformance() throws {
        guard ProcessInfo.processInfo.environment["FMLX_GDN_BENCHMARK"] == "1" else {
            throw XCTSkip("Set FMLX_GDN_BENCHMARK=1 for local kernel measurements")
        }
        let input = inputs(dtype: .bfloat16, dimension: 128, fullWidth: true)
        eval(input)
        for trial in 0 ..< 12 {
            for fused in trial % 2 == 0 ? [false, true] : [true, false] {
                var current = input
                var result: [MLXArray] = []
                let start = ProcessInfo.processInfo.systemUptime
                for _ in 0 ..< 16 {
                    result = run(current, fused: fused)
                    current[5] = result[1]
                }
                eval(result)
                let milliseconds = (ProcessInfo.processInfo.systemUptime - start) * 1000 / 16
                print("FMLX_GDN_CHECKPOINT fused=\(fused) trial=\(trial) ms=\(milliseconds)")
            }
        }
    }
}
