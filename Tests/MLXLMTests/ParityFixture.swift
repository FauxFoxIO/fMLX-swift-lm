import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// A tiny random-weight checkpoint with reference logits from a Python implementation.
///
/// Fixtures live in `Resources/Parity/<name>/` beside this file and are written by
/// `scripts/parity/<name>.py`. Tests read them from the source tree, not the bundle.
/// Each has `config.json`, `model.safetensors`, and `expected.safetensors` holding
/// `input_ids` [1, L] and `logits` [1, L, vocab] in float32.
struct ParityFixture {
    let directory: URL
    let inputIDs: MLXArray
    let logits: MLXArray

    init(_ name: String) throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appending(components: "Resources", "Parity", name)
        guard FileManager.default.fileExists(atPath: root.path) else {
            throw ParityFixtureError.missing(name)
        }
        directory = root
        let expected = try loadArrays(url: root.appending(component: "expected.safetensors"))
        guard let ids = expected["input_ids"], let logits = expected["logits"] else {
            throw ParityFixtureError.missing("\(name)/expected.safetensors")
        }
        inputIDs = ids.asType(.int32)
        self.logits = logits
    }

    /// Builds the model from `config.json` through the LLM type registry and loads weights.
    func loadLanguageModel() async throws -> LanguageModel {
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: base.modelType)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        return model
    }

    /// Largest absolute difference between `actual` and the reference logits.
    func maxAbsDifference(_ actual: MLXArray) -> Float {
        abs(actual.asType(.float32) - logits).max().item(Float.self)
    }
}

enum ParityFixtureError: Error {
    case missing(String)
}
