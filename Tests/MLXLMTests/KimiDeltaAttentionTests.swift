import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// The fused KDA kernel against the ops recurrence it replaces.
struct KimiDeltaAttentionTests {
    private struct Inputs {
        let q, k, v, f, b, aLog, dtBias, state: MLXArray
    }

    private func makeInputs(
        batch: Int = 2, length: Int = 9, heads: Int = 3, keyDim: Int = 64, valueDim: Int = 48,
        dtype: DType = .float32, seed: UInt64 = 7
    ) -> Inputs {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            Inputs(
                q: MLXRandom.normal([batch, length, heads, keyDim]).asType(dtype),
                k: MLXRandom.normal([batch, length, heads, keyDim]).asType(dtype),
                v: MLXRandom.normal([batch, length, heads, valueDim]).asType(dtype),
                f: MLXRandom.normal([batch, length, heads, keyDim]).asType(dtype),
                b: MLXRandom.normal([batch, length, heads]).asType(dtype),
                aLog: MLXRandom.normal([heads]) * 0.5,
                dtBias: MLXRandom.normal([heads * keyDim]),
                state: MLXRandom.normal([batch, heads, valueDim, keyDim]) * 0.1)
        }
    }

    private func decay(_ inputs: Inputs, lowerBound: Float? = -5) -> MLXArray {
        let heads = inputs.aLog.dim(0)
        return kimiDeltaDecay(
            f: inputs.f, aLog: inputs.aLog.reshaped(heads, 1),
            dtBias: inputs.dtBias.reshaped(heads, -1), lowerBound: lowerBound)
    }

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    @Test("Safe-gate decay matches exp(lower_bound * sigmoid(exp(A_log) * (f + dt_bias)))")
    func decayMatchesFormula() {
        let inputs = makeInputs()
        let heads = inputs.aLog.dim(0)
        let expected = exp(
            -5
                * sigmoid(
                    exp(inputs.aLog).reshaped(1, 1, heads, 1)
                        * (inputs.f + inputs.dtBias.reshaped(heads, -1))))
        #expect(maxDifference(decay(inputs), expected) < 1e-6)

        let softplusExpected = exp(
            -exp(inputs.aLog).reshaped(1, 1, heads, 1)
                * softplus(inputs.f + inputs.dtBias.reshaped(heads, -1)))
        #expect(maxDifference(decay(inputs, lowerBound: nil), softplusExpected) < 1e-6)
    }

    @Test("Kernel matches the ops recurrence", arguments: [true, false])
    func kernelMatchesOps(normalizeQK: Bool) {
        let inputs = makeInputs()
        let g = decay(inputs)
        let beta = sigmoid(inputs.b)
        let (y, state) = kimiDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v, decay: g, beta: beta, state: inputs.state,
            normalizeQK: normalizeQK)
        let reference = kimiDeltaOps(
            q: inputs.q, k: inputs.k, v: inputs.v, decay: g, beta: beta, state: inputs.state,
            normalizeQK: normalizeQK)
        #expect(state.dtype == .float32)
        #expect(maxDifference(y, reference.output) < 1e-4)
        #expect(maxDifference(state, reference.state) < 1e-4)
    }

    @Test("Masked positions keep the state and output zeros")
    func maskedKernelMatchesOps() {
        let inputs = makeInputs()
        let g = decay(inputs)
        let beta = sigmoid(inputs.b)
        let mask = MLXArray(
            [
                true, true, false, true, false, true, true, true, false,
                false, false, true, true, true, true, false, true, true,
            ],
            [2, 9])
        let (y, state) = kimiDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v, decay: g, beta: beta, state: inputs.state,
            mask: mask)
        let reference = kimiDeltaOps(
            q: inputs.q, k: inputs.k, v: inputs.v, decay: g, beta: beta, state: inputs.state,
            mask: mask)
        #expect(maxDifference(y, reference.output) < 1e-4)
        #expect(maxDifference(state, reference.state) < 1e-4)
        #expect(abs(y[0, 2]).max().item(Float.self) == 0)
    }

    @Test("Checkpoints equal the state after each prefix")
    func checkpointsMatchPrefixes() {
        let inputs = makeInputs(length: 5)
        let g = decay(inputs)
        let beta = sigmoid(inputs.b)
        let result = kimiDeltaUpdateWithCheckpoints(
            q: inputs.q, k: inputs.k, v: inputs.v, decay: g, beta: beta, state: inputs.state)
        let (yFull, stateFull) = kimiDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v, decay: g, beta: beta, state: inputs.state)
        #expect(result.checkpoints.count == 4)
        #expect(maxDifference(result.output, yFull) < 1e-5)
        #expect(maxDifference(result.state, stateFull) < 1e-5)
        for split in 1 ..< 5 {
            let (_, prefixState) = kimiDeltaUpdate(
                q: inputs.q[0..., ..<split], k: inputs.k[0..., ..<split],
                v: inputs.v[0..., ..<split], decay: g[0..., ..<split],
                beta: beta[0..., ..<split], state: inputs.state)
            #expect(maxDifference(result.checkpoints[split - 1], prefixState) < 1e-5)
        }
        let ops = kimiDeltaUpdateWithCheckpoints(
            q: inputs.q, k: inputs.k, v: inputs.v, decay: g, beta: beta, state: inputs.state,
            useKernel: false)
        for (kernel, reference) in zip(result.checkpoints, ops.checkpoints) {
            #expect(maxDifference(kernel, reference) < 1e-4)
        }
    }

    @Test("Chunked recurrence carries float32 state across chunks")
    func chunkedMatchesSingle() {
        let inputs = makeInputs(length: 12, dtype: .bfloat16)
        let g = decay(inputs)
        let beta = sigmoid(inputs.b)
        let (ySingle, stateSingle) = kimiDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v, decay: g, beta: beta)
        var state: MLXArray? = nil
        var ys = [MLXArray]()
        for (start, end) in [(0, 5), (5, 6), (6, 12)] {
            let (y, next) = kimiDeltaUpdate(
                q: inputs.q[0..., start ..< end], k: inputs.k[0..., start ..< end],
                v: inputs.v[0..., start ..< end], decay: g[0..., start ..< end],
                beta: beta[0..., start ..< end], state: state)
            ys.append(y)
            state = next
        }
        #expect(ySingle.dtype == .bfloat16)
        #expect(maxDifference(concatenated(ys, axis: 1), ySingle) < 1e-2)
        #expect(maxDifference(state!, stateSingle) < 1e-4)
    }

    @Test("A key head dim that is not a multiple of 32 uses the ops path")
    func oddKeyDimFallsBack() {
        let inputs = makeInputs(length: 4, keyDim: 24, valueDim: 8)
        let g = decay(inputs)
        let beta = sigmoid(inputs.b)
        let (y, state) = kimiDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v, decay: g, beta: beta, state: inputs.state)
        let reference = kimiDeltaOps(
            q: inputs.q, k: inputs.k, v: inputs.v, decay: g, beta: beta, state: inputs.state)
        #expect(maxDifference(y, reference.output) < 1e-6)
        #expect(maxDifference(state, reference.state) < 1e-6)
    }
}
