import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// GLM-5.3-Flash (`glm5_next`) against transformers on tiny fixtures written by
/// `scripts/parity/glm5_next.py`. Layers 0 and 3 are KDA, layers 1 and 2 are sparse attention
/// (layer 2 shares layer 1's indexer). The 20-token sequence has more complete key pools than
/// the indexer keeps, so later queries drop pools.
struct GLM5NextParityTests {
    @Test("Prefill logits match transformers")
    func prefillMatchesReference() async throws {
        let fixture = try ParityFixture("glm5_next")
        let model = try await fixture.loadLanguageModel()
        #expect(model is GLM5NextModel)
        let logits = model(fixture.inputIDs, cache: nil)
        #expect(fixture.maxAbsDifference(logits) < 1e-4)
    }

    @Test("Chunked prefill and cached decode match full-sequence logits")
    func cachedDecodeMatchesReference() async throws {
        let fixture = try ParityFixture("glm5_next")
        let model = try await fixture.loadLanguageModel()
        let cache = try model.newCache(parameters: nil)
        #expect(cache[0] is MambaCache && cache[1] is KVCacheSimple)
        let length = fixture.inputIDs.dim(1)

        // A dense first chunk, a chunk that crosses into sparse selection, then decode steps.
        var bounds = [0, 6, 13]
        bounds += Array(14 ... length)
        var rows = [MLXArray]()
        for (start, end) in zip(bounds, bounds.dropFirst()) {
            rows.append(model(fixture.inputIDs[0..., start ..< end], cache: cache))
        }
        let logits = concatenated(rows, axis: 1)
        #expect(cache[1].offset == length)
        #expect(fixture.maxAbsDifference(logits) < 1e-4)
    }

    @Test("Mixed-precision mlx-community checkpoint matches transformers on dequantized weights")
    func mixedQuantizationMatchesReference() async throws {
        let fixture = try ParityFixture("glm5_next_mixed")
        let model = try #require(try await fixture.loadLanguageModel() as? GLM5NextModel)
        let moe = try #require(model.model.layers[1].mlp as? GLM5NextMoE)
        let down = try #require(moe.switchMLP.downProj as? QuantizedSwitchLinear)
        #expect(down.bits == 5)
        let shared = try #require(moe.sharedExperts?.upProj as? QuantizedLinear)
        #expect(shared.bits == 6)
        let kda = try #require(model.model.layers[0].attention as? GLM5NextLinearAttention)
        #expect(!(kda.qProj is QuantizedLinear))
        let logits = model(fixture.inputIDs, cache: nil)
        #expect(fixture.maxAbsDifference(logits) < 1e-3)
    }

    @Test("The configuration reads text_config, linear_attn_config, and layer types")
    func configurationDecoding() throws {
        let fixture = try ParityFixture("glm5_next")
        let data = try Data(contentsOf: fixture.directory.appending(component: "config.json"))
        let configuration = try JSONDecoder.json5().decode(GLM5NextConfiguration.self, from: data)
        #expect(configuration.linearLayers == [true, false, false, true])
        #expect(configuration.linearHeads == 2 && configuration.linearHeadDim == 32)
        #expect(configuration.linearLowerBound == -5)
        #expect(configuration.indexKPool == 4 && configuration.indexTopK == 8)
        #expect(configuration.swigluLimit == 0.75)
    }
}
