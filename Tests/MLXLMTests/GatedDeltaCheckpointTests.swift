import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class GatedDeltaCheckpointTests: XCTestCase {
    private func inputs(dtype: DType, dimension: Int, fullWidth: Bool) -> [MLXArray] {
        withRandomState(MLXRandom.RandomState(seed: 184)) {
            let batch = fullWidth ? 1 : 2
            let keyHeads = fullWidth ? 16 : 2
            let valueHeads = fullWidth ? 48 : 4
            let valueDimension = fullWidth ? 128 : 8
            return [
                (MLXRandom.normal([batch, 2, keyHeads, dimension]) * 0.05).asType(dtype),
                (MLXRandom.normal([batch, 2, keyHeads, dimension]) * 0.05).asType(dtype),
                MLXRandom.normal([batch, 2, valueHeads, valueDimension]).asType(dtype),
                sigmoid(MLXRandom.normal([batch, 2, valueHeads])),
                sigmoid(MLXRandom.normal([batch, 2, valueHeads])),
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
