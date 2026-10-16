import MLX
import MLXLMCommon
import Testing

struct ParityHarnessTests {
    @Test("Qwen3 matches mlx-lm on a tiny random checkpoint")
    func qwen3MatchesReference() async throws {
        let fixture = try ParityFixture("qwen3")
        let model = try await fixture.loadLanguageModel()
        let logits = model(fixture.inputIDs, cache: nil)
        #expect(fixture.maxAbsDifference(logits) < 1e-3)
    }
}
