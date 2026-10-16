import MLX
import Testing

@testable import MLXLMCommon

/// The fused sigmoid + bias router against the unfused MLX ops it replaces.
struct MoESigmoidRouterTopKTests {
    @Test(
        "Fused sigmoid router selects the same experts with the same weights",
        arguments: [(1, 256, 8), (2, 256, 8), (17, 256, 8), (3, 4, 2), (5, 64, 1)])
    func matchesUnfusedRouter(rows: Int, experts: Int, k: Int) {
        for normalize in [true, false] {
            let logits = MLXRandom.normal([1, rows, experts], key: MLXRandom.key(UInt64(rows)))
            let bias = 0.1 * MLXRandom.normal([experts], key: MLXRandom.key(7))
            #expect(supportsFusedSigmoidRouterTopK(logits, k: k))
            let fused = fusedSigmoidRouterTopK(
                logits: logits, bias: bias, k: k, normalize: normalize, scalingFactor: 2.5)
            let chain = chainSigmoidRouterTopK(
                logits: logits, bias: bias, k: k, normalize: normalize, scalingFactor: 2.5)
            #expect(fused.indices.shape == [1, rows, k])
            #expect(fused.indices.dtype == .uint32)
            #expect(fused.scores.dtype == .float32)

            // The unfused order is unspecified, so compare after sorting by expert index.
            let fusedOrder = argSort(fused.indices, axis: -1)
            let chainOrder = argSort(chain.indices, axis: -1)
            let fusedIndices = takeAlong(fused.indices, fusedOrder, axis: -1)
            let chainIndices = takeAlong(chain.indices, chainOrder, axis: -1)
            #expect((fusedIndices .== chainIndices).all().item(Bool.self))
            let fusedScores = takeAlong(fused.scores, fusedOrder, axis: -1)
            let chainScores = takeAlong(chain.scores, chainOrder, axis: -1)
            #expect(abs(fusedScores - chainScores).max().item(Float.self) < 1e-5)

            // Winners come out in descending order of their bias-corrected score.
            let selection = takeAlong(
                sigmoid(logits) + bias, fused.indices.asType(.int32), axis: -1)
            if k > 1 {
                let steps = selection[.ellipsis, ..<(k - 1)] - selection[.ellipsis, 1...]
                #expect(steps.min().item(Float.self) >= 0)
            }
        }
    }

    @Test("Normalized weights sum to the routed scaling factor")
    func normalizedWeightsSumToScale() {
        let logits = MLXRandom.normal([4, 256], key: MLXRandom.key(3))
        let bias = MLXArray.zeros([256])
        let (_, scores) = sigmoidRouterTopK(
            logits: logits, bias: bias, k: 8, normalize: true, scalingFactor: 2.5)
        #expect(abs(scores.sum(axis: -1) - 2.5).max().item(Float.self) < 1e-5)
    }
}
