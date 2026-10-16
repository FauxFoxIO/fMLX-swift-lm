import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// GLM-MoE-DSA against transformers on tiny fixtures written by `scripts/parity/glm_moe_dsa.py`.
/// The 14-token sequence exceeds `index_topk` (5), and layers 1 and 3 share the indexer.
struct GLMMoEDSAParityTests {
    @Test("Prefill logits match transformers")
    func prefillMatchesReference() async throws {
        let fixture = try ParityFixture("glm_moe_dsa")
        let model = try await fixture.loadLanguageModel()
        let logits = model(fixture.inputIDs, cache: nil)
        #expect(fixture.maxAbsDifference(logits) < 1e-4)
    }

    @Test("Chunked prefill and cached decode match full-sequence logits")
    func cachedDecodeMatchesReference() async throws {
        let fixture = try ParityFixture("glm_moe_dsa")
        let model = try await fixture.loadLanguageModel()
        let cache = try model.newCache(parameters: nil)
        let length = fixture.inputIDs.dim(1)

        // A dense first chunk, a sparse chunk at an offset, then single-token decode steps.
        var bounds = [0, 5, 9]
        bounds += Array(10 ... length)
        var rows = [MLXArray]()
        for (start, end) in zip(bounds, bounds.dropFirst()) {
            rows.append(model(fixture.inputIDs[0..., start ..< end], cache: cache))
        }
        let logits = concatenated(rows, axis: 1)
        #expect(cache[0].offset == length)
        #expect(fixture.maxAbsDifference(logits) < 1e-4)
    }

    @Test("Mixed 4/5-bit mlx-community checkpoint matches transformers on dequantized weights")
    func mixedQuantizationMatchesReference() async throws {
        let fixture = try ParityFixture("glm_moe_dsa_mixed")
        let model = try await fixture.loadLanguageModel()
        let logits = model(fixture.inputIDs, cache: nil)
        #expect(fixture.maxAbsDifference(logits) < 1e-3)
    }

    @Test("FP8 block dequantization applies one scale per 128x128 block")
    func fp8BlockDequantization() {
        let (m, n) = (130, 260)
        let weight = MLXRandom.normal([m, n], key: MLXRandom.key(0))
        let scaleInv = MLXArray((1 ... 6).map { Float($0) * 0.5 }, [2, 3])
        let blockScale = repeated(repeated(scaleInv, count: 128, axis: 0), count: 128, axis: 1)[
            ..<m, ..<n]
        let fp8 = toFP8(weight / blockScale)
        let expected = fromFP8(fp8, dtype: .float32) * blockScale
        let actual = GLMMoEDSACheckpoint.dequantizeFP8(fp8, scaleInv: scaleInv)
        #expect(actual.dtype == .bfloat16)
        #expect(abs(actual.asType(.float32) - expected).max().item(Float.self) < 0.05)
        #expect(abs(actual.asType(.float32) - weight).max().item(Float.self) < 0.5)
    }
}
